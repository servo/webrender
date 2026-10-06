/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Rendering logic related to the vertex shaders and their states, uncluding
//!  - Vertex Array Objects
//!  - vertex layout descriptors
//!  - textures bound at vertex stage

use std::{marker::PhantomData, mem, num::NonZeroUsize, ops};
use api::units::*;
use crate::{
    device::{
        Buffer, BufferKind, Device, Texture, TextureFilter, TextureUploader, UploadBufferPool,
        VertexArray, VertexDescriptor, VertexUsageHint,
    },
    frame_builder::Frame,
    gpu_types::{PrimitiveHeaderI, PrimitiveHeaderF},
    internal_types::Swizzle,
    render_task::RenderTaskData,
    transform::TransformData,
    util::round_up_to_multiple,
};

use crate::internal_types::FrameVec;

pub const VERTEX_TEXTURE_EXTRA_ROWS: i32 = 10;

pub const MAX_VERTEX_TEXTURE_WIDTH: usize = webrender_build::MAX_VERTEX_TEXTURE_WIDTH;

/// Upper bound on the size of the CPU-side copy retained to detect unchanged
/// vertex data. Buffers larger than this are always treated as dirty, which
/// bounds both the retained memory and the cost of the comparison. One texture
/// row is ample for the frames this is meant to catch, where nothing is drawn
/// and only a handful of entries are present.
const MAX_TRACKED_UPLOAD_BYTES: usize = MAX_VERTEX_TEXTURE_WIDTH * 16;

pub mod desc {
    use crate::device::{VertexAttribute, VertexAttributeKind, VertexDescriptor};

    pub const PRIM_INSTANCES: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[VertexAttribute::quad_instance_vertex()],
        instance_attributes: &[VertexAttribute {
            name: "aData",
            count: 4,
            kind: VertexAttributeKind::I32,
        }],
    };

    pub const BLUR: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[VertexAttribute::quad_instance_vertex()],
        instance_attributes: &[
            VertexAttribute::gpu_buffer_address("aBlurRenderTaskAddress"),
            VertexAttribute::gpu_buffer_address("aBlurSourceTaskAddress"),
            VertexAttribute::i32("aBlurDirection"),
            VertexAttribute::i32("aBlurEdgeMode"),
            VertexAttribute::f32x3("aBlurParams"),
        ],
    };

    pub const LINE: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[VertexAttribute::quad_instance_vertex()],
        instance_attributes: &[
            VertexAttribute::f32x4("aTaskRect"),
            VertexAttribute::f32x2("aLocalSize"),
            VertexAttribute::f32("aWavyLineThickness"),
            VertexAttribute::i32("aStyle"),
            VertexAttribute::f32("aAxisSelect"),
        ],
    };


    pub const BORDER: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[VertexAttribute::quad_instance_vertex()],
        instance_attributes: &[
            VertexAttribute::f32x2("aTaskOrigin"),
            VertexAttribute::i32("aFlags"),
            VertexAttribute::gpu_buffer_address("aGpuDataAddress"),
            VertexAttribute::f32x4("aClipParams1"),
            VertexAttribute::f32x4("aClipParams2"),
        ],
    };

    pub const SCALE: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[VertexAttribute::quad_instance_vertex()],
        instance_attributes: &[
            VertexAttribute::f32x4("aScaleTargetRect"),
            VertexAttribute::f32x4("aScaleSourceRect"),
            VertexAttribute::f32("aSourceRectType"),
        ],
    };


    pub const SVG_FILTER_NODE: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[VertexAttribute::quad_instance_vertex()],
        instance_attributes: &[
            VertexAttribute::f32x4("aFilterTargetRect"),
            VertexAttribute::f32x4("aFilterInput1ContentScaleAndOffset"),
            VertexAttribute::f32x4("aFilterInput2ContentScaleAndOffset"),
            VertexAttribute::gpu_buffer_address("aFilterInput1TaskAddress"),
            VertexAttribute::gpu_buffer_address("aFilterInput2TaskAddress"),
            VertexAttribute::u16x2("aFilterKindAndInputCount"),
            VertexAttribute::gpu_buffer_address("aFilterExtraDataAddress"),
        ],
    };

    pub const MASK: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[VertexAttribute::quad_instance_vertex()],
        instance_attributes: &[
            VertexAttribute::i32x4("aData"),
            VertexAttribute::i32x4("aClipData"),
        ],
    };

    pub const COMPOSITE: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[VertexAttribute::quad_instance_vertex()],
        instance_attributes: &[
            VertexAttribute::f32x4("aDeviceRect"),
            VertexAttribute::f32x4("aDeviceClipRect"),
            VertexAttribute::f32x4("aColor"),
            VertexAttribute::f32x4("aParams"),
            VertexAttribute::f32x4("aUvRect0"),
            VertexAttribute::f32x4("aUvRect1"),
            VertexAttribute::f32x4("aUvRect2"),
            VertexAttribute::f32x2("aFlip"),
            VertexAttribute::f32x4("aDeviceRoundedClipRect"),
            VertexAttribute::f32x4("aDeviceRoundedClipRadii"),
        ],
    };

    pub const CLEAR: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[VertexAttribute::quad_instance_vertex()],
        instance_attributes: &[
            VertexAttribute::f32x4("aRect"),
            VertexAttribute::f32x4("aColor"),
        ],
    };

    pub const COPY: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[VertexAttribute::quad_instance_vertex()],
        instance_attributes: &[
            VertexAttribute::f32x4("a_src_rect"),
            VertexAttribute::f32x4("a_dst_rect"),
            VertexAttribute::f32x2("a_dst_texture_size"),
        ],
    };
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum VertexArrayKind {
    Primitive,
    Blur,
    Border,
    Scale,
    LineDecoration,
    SvgFilterNode,
    Composite,
    Clear,
    Copy,
    Mask,
}

pub struct VertexDataTexture<T> {
    texture: Option<Texture>,
    format: api::ImageFormat,
    _marker: PhantomData<T>,
}

impl<T> VertexDataTexture<T> {
    pub fn new(format: api::ImageFormat) -> Self {
        Self {
            texture: None,
            format,
            _marker: PhantomData,
        }
    }

    /// Returns a borrow of the GPU texture. Panics if it hasn't been initialized.
    pub fn texture(&self) -> &Texture {
        self.texture.as_ref().unwrap()
    }

    /// Returns an estimate of the GPU memory consumed by this VertexDataTexture.
    pub fn size_in_bytes(&self) -> usize {
        self.texture.as_ref().map_or(0, |t| t.size_in_bytes())
    }

    pub fn update<'a>(
        &'a mut self,
        device: &mut Device,
        texture_uploader: &mut TextureUploader<'a>,
        data: &mut FrameVec<T>,
    ) {
        debug_assert!(mem::size_of::<T>() % 16 == 0);
        let texels_per_item = mem::size_of::<T>() / 16;
        let items_per_row = MAX_VERTEX_TEXTURE_WIDTH / texels_per_item;
        debug_assert_ne!(items_per_row, 0);

        // Ensure we always end up with a texture when leaving this method.
        let mut len = data.len();
        if len == 0 {
            if self.texture.is_some() {
                return;
            }
            data.reserve(items_per_row);
            len = items_per_row;
        } else {
            // Extend the data array to have enough capacity to upload at least
            // a multiple of the row size.  This ensures memory safety when the
            // array is passed to OpenGL to upload to the GPU.
            let extra = len % items_per_row;
            if extra != 0 {
                let padding = items_per_row - extra;
                data.reserve(padding);
                len += padding;
            }
        }

        let needed_height = (len / items_per_row) as i32;
        let existing_height = self
            .texture
            .as_ref()
            .map_or(0, |t| t.get_dimensions().height);

        // Create a new texture if needed.
        //
        // These textures are generally very small, which is why we don't bother
        // with incremental updates and just re-upload every frame. For most pages
        // they're one row each, and on stress tests like css-francine they end up
        // in the 6-14 range. So we size the texture tightly to what we need (usually
        // 1), and shrink it if the waste would be more than `VERTEX_TEXTURE_EXTRA_ROWS`
        // rows. This helps with memory overhead, especially because there are several
        // instances of these textures per Renderer.
        if needed_height > existing_height
            || needed_height + VERTEX_TEXTURE_EXTRA_ROWS < existing_height
        {
            // Drop the existing texture, if any.
            if let Some(t) = self.texture.take() {
                device.delete_texture(t);
            }

            let texture = device.create_texture(
                api::ImageBufferKind::Texture2D,
                self.format,
                MAX_VERTEX_TEXTURE_WIDTH as i32,
                // Ensure height is at least two to work around
                // https://bugs.chromium.org/p/angleproject/issues/detail?id=3039
                needed_height.max(2),
                TextureFilter::Nearest,
                None,
            );
            self.texture = Some(texture);
        }

        // Note: the actual width can be larger than the logical one, with a few texels
        // of each row unused at the tail. This is needed because there is still hardware
        // (like Intel iGPUs) that prefers power-of-two sizes of textures ([1]).
        //
        // [1] https://software.intel.com/en-us/articles/opengl-performance-tips-power-of-two-textures-have-better-performance
        let logical_width = if needed_height == 1 {
            data.len() * texels_per_item
        } else {
            MAX_VERTEX_TEXTURE_WIDTH - (MAX_VERTEX_TEXTURE_WIDTH % texels_per_item)
        };

        let rect = DeviceIntRect::from_size(
            DeviceIntSize::new(logical_width as i32, needed_height),
        );

        debug_assert!(len <= data.capacity(), "CPU copy will read out of bounds");
        texture_uploader.upload(
            device,
            self.texture(),
            rect,
            None,
            None,
            data.as_ptr(),
            len,
        );
    }

    pub fn deinit(mut self, device: &mut Device) {
        if let Some(t) = self.texture.take() {
            device.delete_texture(t);
        }
    }
}

pub struct VertexDataTextures {
    prim_header_f_texture: VertexDataTexture<PrimitiveHeaderF>,
    prim_header_i_texture: VertexDataTexture<PrimitiveHeaderI>,
    transforms_texture: VertexDataTexture<TransformData>,
    render_task_texture: VertexDataTexture<RenderTaskData>,
}

impl VertexDataTextures {
    pub fn new() -> Self {
        VertexDataTextures {
            prim_header_f_texture: VertexDataTexture::new(api::ImageFormat::RGBAF32),
            prim_header_i_texture: VertexDataTexture::new(api::ImageFormat::RGBAI32),
            transforms_texture: VertexDataTexture::new(api::ImageFormat::RGBAF32),
            render_task_texture: VertexDataTexture::new(api::ImageFormat::RGBAF32),
        }
    }

    fn upload(&mut self, device: &mut Device, pbo_pool: &mut UploadBufferPool, frame: &mut Frame) {
        let mut texture_uploader = device.upload_texture(pbo_pool);
        self.prim_header_f_texture.update(
            device,
            &mut texture_uploader,
            &mut frame.prim_headers.headers_float,
        );
        self.prim_header_i_texture.update(
            device,
            &mut texture_uploader,
            &mut frame.prim_headers.headers_int,
        );
        self.transforms_texture
            .update(device, &mut texture_uploader, &mut frame.transform_palette);
        self.render_task_texture.update(
            device,
            &mut texture_uploader,
            &mut frame.render_tasks.task_data,
        );

        // Flush and drop the texture uploader now, so that
        // we can borrow the textures to bind them.
        texture_uploader.flush(device);
    }

    fn bind(&self, device: &mut Device) {
        device.bind_texture(
            super::TextureSampler::PrimitiveHeadersF,
            self.prim_header_f_texture.texture(),
            Swizzle::default(),
        );
        device.bind_texture(
            super::TextureSampler::PrimitiveHeadersI,
            self.prim_header_i_texture.texture(),
            Swizzle::default(),
        );
        device.bind_texture(
            super::TextureSampler::TransformPalette,
            self.transforms_texture.texture(),
            Swizzle::default(),
        );
        device.bind_texture(
            super::TextureSampler::RenderTasks,
            self.render_task_texture.texture(),
            Swizzle::default(),
        );
    }

    pub fn size_in_bytes(&self) -> usize {
        self.prim_header_f_texture.size_in_bytes()
            + self.prim_header_i_texture.size_in_bytes()
            + self.transforms_texture.size_in_bytes()
            + self.render_task_texture.size_in_bytes()
    }

    pub fn deinit(self, device: &mut Device) {
        self.transforms_texture.deinit(device);
        self.prim_header_f_texture.deinit(device);
        self.prim_header_i_texture.deinit(device);
        self.render_task_texture.deinit(device);
    }
}

/// A copy of the bytes last uploaded to one vertex data texture, used to detect
/// frames that produce byte-identical data so that the upload can be skipped.
#[derive(Default)]
struct LastUpload {
    bytes: Vec<u8>,
    /// False if `bytes` does not describe the texture contents, either because
    /// nothing has been uploaded yet or because the data exceeded
    /// `MAX_TRACKED_UPLOAD_BYTES` and was not retained.
    valid: bool,
}

impl LastUpload {
    /// Compares `data` against the previous upload and records it as the new
    /// contents. Returns true if the data differs and must be re-uploaded.
    fn record_if_changed<T>(&mut self, data: &[T]) -> bool {
        // Empty data leaves the existing texture contents alone (see
        // `VertexDataTexture::update`), so it is neither dirty nor worth
        // recording: the texture still holds what was uploaded last.
        if data.is_empty() {
            return false;
        }

        // SAFETY: `T` is a `#[repr(C)]` type built only from 4-byte fields, so
        // it contains no padding and every byte of the slice is initialized.
        // This mirrors `texels_to_u8_slice` in `device::gl`.
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, mem::size_of_val(data))
        };

        // Content this large is being rebuilt every frame anyway, so retaining
        // and comparing it would cost more than the upload it might save.
        if bytes.len() > MAX_TRACKED_UPLOAD_BYTES {
            self.bytes = Vec::new();
            self.valid = false;
            return true;
        }

        if self.valid && self.bytes == bytes {
            return false;
        }

        self.bytes.clear();
        self.bytes.extend_from_slice(bytes);
        self.valid = true;
        true
    }

    fn size_in_bytes(&self) -> usize {
        self.bytes.capacity()
    }
}

/// The contents of the vertex data textures at the time of the last upload.
#[derive(Default)]
struct VertexDataContents {
    prim_header_f: LastUpload,
    prim_header_i: LastUpload,
    transforms: LastUpload,
    render_tasks: LastUpload,
    /// False until the first upload, so that the initial frame always creates
    /// the textures even if every buffer is empty.
    initialized: bool,
}

impl VertexDataContents {
    /// Compares the frame's vertex data against the last upload and records it.
    /// Returns true if anything differs and the textures must be re-uploaded.
    fn record_if_changed(&mut self, frame: &Frame) -> bool {
        // Collected into an array rather than chained with `||` so that every
        // buffer is compared and records its contents, not just those up to
        // the first one that changed.
        let changed = [
            self.prim_header_f.record_if_changed(&frame.prim_headers.headers_float),
            self.prim_header_i.record_if_changed(&frame.prim_headers.headers_int),
            self.transforms.record_if_changed(&frame.transform_palette),
            self.render_tasks.record_if_changed(&frame.render_tasks.task_data),
        ].contains(&true);

        let dirty = changed || !self.initialized;
        self.initialized = true;
        dirty
    }

    fn size_in_bytes(&self) -> usize {
        self.prim_header_f.size_in_bytes()
            + self.prim_header_i.size_in_bytes()
            + self.transforms.size_in_bytes()
            + self.render_tasks.size_in_bytes()
    }
}

/// A ring of vertex data texture sets, of which one is bound at a time. The
/// ring rotates on upload so that we avoid writing to a texture the GPU may
/// still be sampling from an earlier frame (see `VERTEX_DATA_TEXTURE_COUNT`).
///
/// A copy of the data last uploaded is retained so that a frame producing
/// byte-identical vertex data can skip the upload entirely. This matters
/// because creating a `TextureUploader` probes the upload PBO pool's fences,
/// which flushes the GPU command stream on some drivers. It happens whenever a
/// frame draws nothing but is still presented, for instance when a promoted
/// video surface updates during direct scanout.
pub struct VertexDataRing {
    sets: Vec<VertexDataTextures>,
    /// Index of the set holding the data currently on the GPU.
    current: usize,
    last_upload: VertexDataContents,
}

impl VertexDataRing {
    pub fn new() -> Self {
        let count = super::VERTEX_DATA_TEXTURE_COUNT;
        let mut sets = Vec::with_capacity(count);
        for _ in 0 .. count {
            sets.push(VertexDataTextures::new());
        }

        VertexDataRing {
            sets,
            // Start at the end so that the first upload rotates onto set 0.
            current: count - 1,
            last_upload: VertexDataContents::default(),
        }
    }

    /// Uploads the frame's vertex data if it differs from the last upload, and
    /// binds the textures holding it. Returns true if an upload was performed.
    pub fn update(
        &mut self,
        device: &mut Device,
        pbo_pool: &mut UploadBufferPool,
        frame: &mut Frame,
    ) -> bool {
        let uploaded = self.last_upload.record_if_changed(frame);

        if uploaded {
            // Only rotate when we are actually going to write. The rotation
            // exists to avoid stalling on a texture the GPU may still be
            // sampling; if we don't write there is no such hazard, and the set
            // written last already holds the correct data.
            self.current = (self.current + 1) % super::VERTEX_DATA_TEXTURE_COUNT;
            self.sets[self.current].upload(device, pbo_pool, frame);
        }

        self.sets[self.current].bind(device);

        uploaded
    }

    /// GPU memory consumed by the textures in the ring.
    pub fn gpu_size_in_bytes(&self) -> usize {
        self.sets.iter().map(|set| set.size_in_bytes()).sum()
    }

    /// CPU memory consumed by the retained copy of the last upload.
    pub fn cpu_size_in_bytes(&self) -> usize {
        self.last_upload.size_in_bytes()
    }

    pub fn deinit(&mut self, device: &mut Device) {
        for set in self.sets.drain(..) {
            set.deinit(device);
        }
    }
}

/// The size of the shared instance buffer. Callers must chunk their draws so
/// that no single upload exceeds this.
pub(crate) const SHARED_INSTANCE_BUFFER_SIZE: usize = 1024 * 1024;

/// An instance data buffer shared between all vertex arrays. Rather than
/// reallocating a per-array instance buffer on every draw, each draw uploads
/// its instance data to the next free offset within this buffer via an
/// unsynchronized mapping and draws from that offset. The buffer is
/// reallocated and used count reset to zero whenever a draw would not fit.
pub struct SharedInstanceBuffer {
    buffer: Buffer,
    /// Number of bytes currently used.
    used: usize,
}

impl SharedInstanceBuffer {
    fn new(device: &mut Device) -> Self {
        let mut buffer = device.create_buffer(BufferKind::Vertex);
        device.reallocate_buffer(&mut buffer, SHARED_INSTANCE_BUFFER_SIZE);
        SharedInstanceBuffer { buffer, used: 0 }
    }

    /// Uploads a chunk of instance data to the shared buffer and returns the
    /// byte offset at which it was written. The offset will be aligned to the
    /// instance stride. The caller must ensure the data fits within
    /// `SHARED_INSTANCE_BUFFER_SIZE`.
    pub fn push_instances<V>(&mut self, device: &mut Device, instances: &[V]) -> usize {
        let stride = mem::size_of::<V>();
        let needed = instances.len() * stride;
        assert!(needed <= SHARED_INSTANCE_BUFFER_SIZE);

        // The buffer may previously have been used for a different vertex array
        // with a different stride, so we must round up the current used offset to
        // the next multiple of the stride to ensure our data is correctly aligned.
        let mut offset = round_up_to_multiple(self.used, NonZeroUsize::new(stride).unwrap());

        if offset + needed > SHARED_INSTANCE_BUFFER_SIZE {
            device.reallocate_buffer(&mut self.buffer, SHARED_INSTANCE_BUFFER_SIZE);
            offset = 0;
        }

        device.write_buffer_unsynchronized(&self.buffer, offset, instances);
        self.used = offset + needed;

        offset
    }
}

/// Every kind, in the order `RendererVAOs::instance_buffers` is indexed.
const VERTEX_ARRAY_KINDS: [VertexArrayKind; 10] = [
    VertexArrayKind::Primitive,
    VertexArrayKind::Blur,
    VertexArrayKind::Border,
    VertexArrayKind::Scale,
    VertexArrayKind::LineDecoration,
    VertexArrayKind::SvgFilterNode,
    VertexArrayKind::Composite,
    VertexArrayKind::Clear,
    VertexArrayKind::Copy,
    VertexArrayKind::Mask,
];

pub struct RendererVAOs {
    /// The unit quad's indices and vertices, read by every vertex array.
    quad_indices: Buffer,
    quad_vertices: Buffer,
    /// One per kind, in `VERTEX_ARRAY_KINDS` order, unless the shared
    /// instance buffer is in use.
    instance_buffers: Vec<Buffer>,
    prim_vao: VertexArray,
    blur_vao: VertexArray,
    border_vao: VertexArray,
    line_vao: VertexArray,
    scale_vao: VertexArray,
    svg_filter_node_vao: VertexArray,
    composite_vao: VertexArray,
    clear_vao: VertexArray,
    copy_vao: VertexArray,
    mask_vao: VertexArray,
    pub shared_instance_buffer: Option<SharedInstanceBuffer>,
}

impl RendererVAOs {
    pub fn new(
        device: &mut Device,
        indexed_quads: Option<NonZeroUsize>,
        use_shared_instance_buffer: bool,
    ) -> Self {
        const QUAD_INDICES: [u16; 6] = [0, 1, 2, 2, 1, 3];
        const QUAD_VERTICES: [[u8; 2]; 4] = [[0, 0], [0xFF, 0], [0, 0xFF], [0xFF, 0xFF]];

        let instance_divisor = if indexed_quads.is_some() { 0 } else { 1 };

        let mut quad_indices = device.create_buffer(BufferKind::Index);
        let mut quad_vertices = device.create_buffer(BufferKind::Vertex);

        // In shared instance buffer mode every vertex array reads its instances
        // from the one shared buffer, otherwise each gets its own.
        let shared_instance_buffer = use_shared_instance_buffer.then(|| SharedInstanceBuffer::new(device));
        let instance_buffers: Vec<Buffer> = if use_shared_instance_buffer {
            Vec::new()
        } else {
            VERTEX_ARRAY_KINDS.iter().map(|_| device.create_buffer(BufferKind::Vertex)).collect()
        };
        let instances_of = |kind: VertexArrayKind| -> &Buffer {
            match &shared_instance_buffer {
                Some(shared) => &shared.buffer,
                None => &instance_buffers[kind as usize],
            }
        };

        let prim_vao = device.create_vertex_array(
            &desc::PRIM_INSTANCES,
            &quad_vertices,
            Some(instances_of(VertexArrayKind::Primitive)),
            Some(&quad_indices),
            instance_divisor,
        );

        device.bind_vertex_array(&prim_vao);
        match indexed_quads {
            Some(count) => {
                assert!(count.get() < u16::MAX as usize);
                let indices = (0 .. count.get() as u16)
                    .flat_map(|instance| QUAD_INDICES.iter().map(move |&index| instance * 4 + index))
                    .collect::<Vec<_>>();
                device.write_buffer(&mut quad_indices, &indices, VertexUsageHint::Static);
                let vertices = (0 .. count.get() as u16)
                    .flat_map(|_| QUAD_VERTICES.iter().cloned())
                    .collect::<Vec<_>>();
                device.write_buffer(&mut quad_vertices, &vertices, VertexUsageHint::Static);
            }
            None => {
                device.write_buffer(&mut quad_indices, &QUAD_INDICES, VertexUsageHint::Static);
                device.write_buffer(&mut quad_vertices, &QUAD_VERTICES, VertexUsageHint::Static);
            }
        }

        let make_vao = |device: &mut Device, layout: &VertexDescriptor, kind: VertexArrayKind| {
            device.create_vertex_array(
                layout,
                &quad_vertices,
                Some(instances_of(kind)),
                Some(&quad_indices),
                instance_divisor,
            )
        };
        let blur_vao = make_vao(device, &desc::BLUR, VertexArrayKind::Blur);
        let border_vao = make_vao(device, &desc::BORDER, VertexArrayKind::Border);
        let scale_vao = make_vao(device, &desc::SCALE, VertexArrayKind::Scale);
        let line_vao = make_vao(device, &desc::LINE, VertexArrayKind::LineDecoration);
        let svg_filter_node_vao = make_vao(device, &desc::SVG_FILTER_NODE, VertexArrayKind::SvgFilterNode);
        let composite_vao = make_vao(device, &desc::COMPOSITE, VertexArrayKind::Composite);
        let clear_vao = make_vao(device, &desc::CLEAR, VertexArrayKind::Clear);
        let copy_vao = make_vao(device, &desc::COPY, VertexArrayKind::Copy);
        let mask_vao = make_vao(device, &desc::MASK, VertexArrayKind::Mask);

        RendererVAOs {
            quad_indices,
            quad_vertices,
            instance_buffers,
            prim_vao,
            blur_vao,
            border_vao,
            line_vao,
            scale_vao,
            svg_filter_node_vao,
            composite_vao,
            clear_vao,
            copy_vao,
            mask_vao,
            shared_instance_buffer,
        }
    }

    /// The instance buffer of `kind`, when the shared buffer is not in use.
    pub fn instance_buffer_mut(&mut self, kind: VertexArrayKind) -> &mut Buffer {
        &mut self.instance_buffers[kind as usize]
    }

    pub fn deinit(self, device: &mut Device) {
        device.delete_vertex_array(self.prim_vao);
        device.delete_vertex_array(self.blur_vao);
        device.delete_vertex_array(self.line_vao);
        device.delete_vertex_array(self.border_vao);
        device.delete_vertex_array(self.scale_vao);
        device.delete_vertex_array(self.svg_filter_node_vao);
        device.delete_vertex_array(self.composite_vao);
        device.delete_vertex_array(self.clear_vao);
        device.delete_vertex_array(self.copy_vao);
        device.delete_vertex_array(self.mask_vao);
        device.delete_buffer(self.quad_indices);
        device.delete_buffer(self.quad_vertices);
        for buffer in self.instance_buffers {
            device.delete_buffer(buffer);
        }
        if let Some(shared) = self.shared_instance_buffer {
            device.delete_buffer(shared.buffer);
        }
    }
}

impl ops::Index<VertexArrayKind> for RendererVAOs {
    type Output = VertexArray;
    fn index(&self, kind: VertexArrayKind) -> &VertexArray {
        match kind {
            VertexArrayKind::Primitive => &self.prim_vao,
            VertexArrayKind::Blur => &self.blur_vao,
            VertexArrayKind::Border => &self.border_vao,
            VertexArrayKind::Scale => &self.scale_vao,
            VertexArrayKind::LineDecoration => &self.line_vao,
            VertexArrayKind::SvgFilterNode => &self.svg_filter_node_vao,
            VertexArrayKind::Composite => &self.composite_vao,
            VertexArrayKind::Clear => &self.clear_vao,
            VertexArrayKind::Copy => &self.copy_vao,
            VertexArrayKind::Mask => &self.mask_vao,
        }
    }
}
