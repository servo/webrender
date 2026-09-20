/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! The device layer is the GPU-facing API the renderer draws through.
//!
//! [`GpuBackend`] is what a graphics API backend implements. [`Device`] wraps
//! the backend in use and adds the typed conveniences the renderer calls;
//! everything else on it is reached through `Deref` to the backend.

mod gl;
pub mod query;
mod types;
mod upload;

use api::{ExternalTextureHandle, ImageBufferKind, ImageDescriptor, ImageFormat, Parameter};
use api::units::*;
use euclid::default::Transform3D;
use malloc_size_of::MallocSizeOfOps;
use std::borrow::Cow;
use std::mem;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::os::raw::c_void;
use std::ptr;
use std::rc::Rc;
use std::slice;
use webrender_build::shader::ShaderFeatureFlags;
#[cfg(feature = "debugger")]
use crate::internal_types::FastHashSet;
use crate::internal_types::{RenderTargetInfo, Swizzle, SwizzleSettings};
use crate::render_api::MemoryReport;
use self::query::GpuProfiler;

pub use self::gl::*;
pub use self::types::*;
pub use self::upload::*;

/// The graphics API a device runs on, together with what the backend needs
/// from the embedder to drive it.
pub enum GpuBackendConfig {
    /// OpenGL or OpenGL ES through the given context, which must be current
    /// on the render thread.
    Gl(Rc<dyn gleam::gl::Gl>),
}

/// A graphics API backend. Resources are created and destroyed through it,
/// and all drawing happens inside a render pass with a pipeline bound.
///
/// Vertex, instance and texel data are passed as bytes; `Device` offers the
/// typed versions.
pub trait GpuBackend {
    /// Number of textures created since the last `begin_frame`.
    fn textures_created(&self) -> u32;

    /// Number of textures deleted since the last `begin_frame`.
    fn textures_deleted(&self) -> u32;

    /// If enabled, initialize the pixels of newly created color render targets
    /// with an opaque pink color for debugging purposes.
    fn set_initialize_color_targets_with_pink(&mut self, enabled: bool);

    /// Selects the best available means of annotating the command stream when
    /// `enable_markers` is set.
    fn create_gpu_profiler(&self, enable_markers: bool) -> GpuProfiler;

    fn set_parameter(&mut self, param: &Parameter);

    /// Returns the limit on texture dimensions (width or height).
    fn max_texture_size(&self) -> i32;

    fn surface_origin_is_top_left(&self) -> bool;

    fn get_capabilities(&self) -> &Capabilities;

    fn api_info(&self) -> GraphicsApiInfo;

    /// Consumes any pending device error and reports whether it was an
    /// out-of-memory condition.
    fn take_out_of_memory_error(&self) -> bool;

    /// Orders reads of the framebuffer by subsequent advanced blend draws
    /// after preceding writes to the same pixels.
    fn blend_barrier(&self);

    fn shader_feature_flags(&self) -> ShaderFeatureFlags;

    fn preferred_color_formats(&self) -> TextureFormatPair<ImageFormat>;

    fn swizzle_settings(&self) -> Option<SwizzleSettings>;

    fn max_depth_ids(&self) -> i32;

    fn ortho_near_plane(&self) -> f32;

    fn ortho_far_plane(&self) -> f32;

    fn required_transfer_stride(&self) -> StrideAlignment;

    fn upload_method(&self) -> &UploadMethod;

    fn use_batched_texture_uploads(&self) -> bool;

    fn use_draw_calls_for_texture_copy(&self) -> bool;

    fn batched_upload_threshold(&self) -> i32;

    fn reset_state(&mut self);

    fn begin_frame(&mut self) -> GpuFrameId;

    fn bind_texture(&mut self, slot: TextureSlot, texture: &Texture, swizzle: Swizzle);

    fn bind_external_texture(&mut self, slot: TextureSlot, external_texture: &ExternalTexture);

    fn reset_read_target(&mut self);

    /// Begins rendering to the target described by `desc`. Draws, clears and
    /// blits into the target must happen before the matching `end_render_pass`.
    /// Passes may not nest.
    fn begin_render_pass(&mut self, desc: &RenderPassDescriptor);

    /// Ends the current render pass. `depth_store` says whether the depth
    /// attachment's contents are needed afterwards; the color attachment is
    /// always stored.
    fn end_render_pass(&mut self, depth_store: StoreOp);

    /// Link a program, attaching the supplied vertex format.
    ///
    /// If `create_program()` finds a binary shader on disk, it will kick
    /// off linking immediately, which some drivers (notably ANGLE) run
    /// in parallel on background threads. As such, this function should
    /// ideally be run sometime later, to give the driver time to do that
    /// before blocking due to an API call accessing the shader.
    ///
    /// This generally means that the first run of the application will have
    /// to do a bunch of blocking work to compile the shader from source, but
    /// subsequent runs should load quickly.
    fn link_program(
        &mut self,
        program: &mut Program,
        descriptor: &VertexDescriptor,
    ) -> Result<(), ShaderError>;

    /// Makes `program` and `state` current for subsequent draws.
    fn bind_pipeline(&mut self, program: &Program, state: &RenderState) -> bool;

    fn create_texture(
        &mut self,
        target: ImageBufferKind,
        format: ImageFormat,
        width: i32,
        height: i32,
        filter: TextureFilter,
        render_target: Option<RenderTargetInfo>,
    ) -> Texture;

    /// Copies the specified subregion from src_texture to dest_texture.
    fn copy_texture_sub_region(
        &mut self,
        src_texture: &Texture,
        src_x: usize,
        src_y: usize,
        dest_texture: &Texture,
        dest_x: usize,
        dest_y: usize,
        width: usize,
        height: usize,
    );

    /// Notifies the device that the contents of a render target are no longer
    /// needed.
    fn invalidate_render_target(&mut self, texture: &Texture);

    /// Notifies the device that a render target is about to be reused.
    ///
    /// This method adds or removes a depth target as necessary.
    fn reuse_render_target(
        &mut self,
        texture: &mut Texture,
        rt_info: RenderTargetInfo,
    );

    /// Perform a blit between src_target and dest_target.
    /// This will overwrite self.bound_read_fbo and self.bound_draw_fbo.
    fn blit_render_target(
        &mut self,
        src_target: ReadTarget,
        src_rect: FramebufferIntRect,
        dest_target: DrawTarget,
        dest_rect: FramebufferIntRect,
        filter: TextureFilter,
    );

    fn delete_texture(&mut self, texture: Texture);

    #[cfg(feature = "replay")]
    fn delete_external_texture(&mut self, external: ExternalTexture);

    fn delete_program(&mut self, program: Program);

    /// Create a shader program. This does minimal amount of work to start
    /// loading a binary shader. If a binary shader is found, we invoke
    /// glProgramBinary, which, at least on ANGLE, will load and link the
    /// binary on a background thread. This can speed things up later when
    /// we invoke `link_program()`.
    fn create_program(
        &mut self,
        base_filename: &'static str,
        features: &[&'static str],
    ) -> Result<Program, ShaderError>;

    /// Whether shader sources can be replaced at runtime.
    #[cfg(feature = "debugger")]
    fn supports_shader_source_override(&self) -> bool;

    /// Names of every `.glsl` file built into this binary, sorted.
    #[cfg(feature = "debugger")]
    fn shader_file_names(&self) -> Vec<&'static str>;

    /// The source built into the binary for `name`, ignoring any override.
    #[cfg(feature = "debugger")]
    fn builtin_shader_source(&self, name: &str) -> Option<&'static str>;

    /// The source currently in effect for `name`: the override if one is
    /// installed, otherwise the source built into the binary.
    fn get_shader_source(&self, name: &str) -> Cow<'static, str>;

    /// The override installed for `name`, if any.
    #[cfg(feature = "debugger")]
    fn shader_source_override(&self, name: &str) -> Option<&str>;

    /// Whether any override is installed.
    #[cfg(feature = "debugger")]
    fn has_shader_source_overrides(&self) -> bool;

    /// Install `source` as the override for `name`.
    #[cfg(feature = "debugger")]
    fn set_shader_source_override(&mut self, name: &str, source: String);

    /// Drop the override for `name`, returning whether there was one.
    #[cfg(feature = "debugger")]
    fn clear_shader_source_override(&mut self, name: &str) -> bool;

    /// The set of `.glsl` files `base_filename` pulls in, including itself.
    #[cfg(feature = "debugger")]
    fn shader_include_closure(&self, base_filename: &str) -> FastHashSet<String>;

    /// The preprocessed vertex and fragment source handed to the driver for
    /// one variant, built from the sources currently in effect.
    #[cfg(feature = "debugger")]
    fn expanded_shader_source(
        &self,
        base_filename: &str,
        features: &[&'static str],
    ) -> (String, String);

    fn bind_shader_samplers(&mut self, program: &Program, bindings: &[(&'static str, TextureSlot)]);

    fn set_uniforms(
        &self,
        program: &Program,
        transform: &Transform3D<f32>,
    );

    /// Sets the uTextureSize uniform. Most shaders do not require this to be called
    /// as they use the textureSize GLSL function instead.
    fn set_shader_texture_size(
        &self,
        program: &Program,
        texture_size: DeviceSize,
    );

    fn create_transfer_buffer_with_size(&mut self, size: usize) -> TransferBuffer;

    fn read_pixels_into_transfer_buffer(
        &mut self,
        read_target: ReadTarget,
        rect: DeviceIntRect,
        format: ImageFormat,
        pbo: &TransferBuffer,
    );

    fn map_transfer_buffer<'a>(&'a mut self, pbo: &'a TransferBuffer) -> Option<MappedTransferBuffer<'a>>;

    fn unmap_transfer_buffer(&mut self);

    fn delete_transfer_buffer(&mut self, pbo: TransferBuffer);

    /// Creates a transfer buffer with no storage; the upload buffer methods
    /// allocate it.
    fn create_transfer_buffer(&mut self) -> TransferBuffer;

    /// Returns the size and stride in bytes required to upload an area of pixels
    /// of the specified size, to a texture of the specified format.
    fn required_upload_size_and_stride(&self, size: DeviceIntSize, format: ImageFormat) -> (usize, usize);

    /// Allocates `size` bytes of storage for an upload buffer and maps it for
    /// writing. A `persistent` mapping stays valid across flushes, and needs
    /// `Capabilities::supports_buffer_storage`.
    fn allocate_upload_buffer(
        &mut self,
        buffer: &mut TransferBuffer,
        size: usize,
        usage_hint: VertexUsageHint,
        persistent: bool,
    ) -> Result<UploadBufferMapping, String>;

    /// Maps an allocated, unmapped upload buffer for writing until the next
    /// `flush_upload_buffer`. The caller guarantees no GPU commands still read
    /// from it.
    fn map_upload_buffer(
        &mut self,
        buffer: &TransferBuffer,
    ) -> Result<ptr::NonNull<mem::MaybeUninit<u8>>, String>;

    /// Makes the first `size_used` bytes written through `mapping` visible to
    /// the GPU, unmapping a transient mapping, then copies `chunks` from the
    /// buffer into their textures.
    fn flush_upload_buffer(
        &mut self,
        buffer: &TransferBuffer,
        mapping: &UploadBufferMapping,
        size_used: usize,
        chunks: &[UploadChunk],
    );

    /// Releases the storage of an upload buffer while keeping its handle, so
    /// that it can be allocated again later.
    fn orphan_upload_buffer(&mut self, buffer: &mut TransferBuffer);

    /// Uploads `data` from CPU memory into `rect` of `texture`.
    fn upload_texture_region(
        &mut self,
        texture: &Texture,
        rect: DeviceIntRect,
        stride: Option<i32>,
        format_override: Option<ImageFormat>,
        data: &[u8],
    );

    /// Creates a fence that is signaled once all commands issued so far have
    /// completed, or `None` if the device could not create one.
    fn create_fence(&mut self) -> Option<Fence>;

    fn poll_fence(&self, fence: &Fence) -> FenceStatus;

    fn delete_fence(&mut self, fence: Fence);

    /// Performs an immediate (non-PBO) upload of the whole texture.
    fn upload_texture_immediate(&mut self, texture: &Texture, pixels: &[u8]);

    fn read_pixels(&mut self, img_desc: &ImageDescriptor) -> Vec<u8>;

    /// Read rectangle of pixels into the specified output slice.
    ///
    /// Reading back `BGRA8` requires `Capabilities::supports_bgra_read`. When
    /// that is false the caller must instead read `RGBA8` and swap the red and
    /// blue channels itself.
    fn read_pixels_into(
        &mut self,
        rect: FramebufferIntRect,
        format: ImageFormat,
        output: &mut [u8],
    );

    /// Makes an application-owned texture the current read target.
    fn attach_read_texture_external(
        &mut self, handle: ExternalTextureHandle, target: ImageBufferKind
    );

    fn attach_read_texture(&mut self, texture: &Texture);

    fn bind_vao(&mut self, vao: &VAO);

    fn create_vao(&mut self, descriptor: &VertexDescriptor, instance_divisor: u32) -> VAO;

    fn delete_vao(&mut self, vao: VAO);

    fn create_vao_with_new_instances(
        &mut self,
        descriptor: &VertexDescriptor,
        base_vao: &VAO,
    ) -> VAO;

    fn create_vao_with_shared_instances(
        &mut self,
        descriptor: &VertexDescriptor,
        base_vao: &VAO,
    ) -> VAO;

    fn update_vao_main_vertices(
        &mut self,
        vao: &VAO,
        vertices: &[u8],
        usage_hint: VertexUsageHint,
    );

    fn update_vao_instances(
        &mut self,
        vao: &VAO,
        instances: &[u8],
        instance_stride: usize,
        usage_hint: VertexUsageHint,
        repeat: Option<NonZeroUsize>,
    );

    fn update_vao_indices(&mut self, vao: &VAO, indices: &[u8], usage_hint: VertexUsageHint);

    /// (Re)allocates the storage of a VBO to `size` bytes, leaving the contents uninitialized.
    fn reallocate_vbo(&mut self, vbo: VBOId, size: usize);

    /// Writes `data` into a VBO at the given byte offset using an unsynchronized mapping, i.e.
    /// without waiting for in-flight draws to complete. The caller must guarantee the written range
    /// does not overlap data still being read by those draws.
    fn update_vbo_data_unsynchronized(&mut self, vbo: VBOId, data: &[u8], offset: usize);

    fn draw_triangles_u32(&mut self, first_vertex: i32, index_count: i32);

    fn draw_nonindexed_lines(&mut self, first_vertex: i32, vertex_count: i32);

    fn draw_indexed_triangles(&mut self, index_count: i32);

    fn draw_indexed_triangles_instanced_u16(&mut self, index_count: i32, instance_count: i32);

    fn draw_indexed_triangles_instanced_base_instance_u16(
        &mut self,
        index_count: i32,
        instance_count: i32,
        base_instance: u32,
    );

    /// Releases resources the device created for its own use. Must be called
    /// inside a frame, before the device is dropped.
    fn deinit(&mut self);

    fn end_frame(&mut self);

    /// Clears the current render pass target. Clears are independent of the
    /// bound pipeline's write masks.
    fn clear_target(
        &mut self,
        color: Option<[f32; 4]>,
        depth: Option<f32>,
        rect: Option<FramebufferIntRect>,
    );

    fn set_scissor_rect(&self, rect: FramebufferIntRect);

    fn enable_scissor(&self);

    fn disable_scissor(&self);

    fn echo_driver_messages(&self);

    /// Generates a memory report for the resources managed by the device layer.
    fn report_memory(&self, size_op_funs: &MallocSizeOfOps, swgl: *mut c_void) -> MemoryReport;

    fn depth_targets_memory(&self) -> usize;
}

/// The device the renderer draws through: the active backend plus the render
/// state requested for the next pipeline bind.
pub struct Device {
    backend: Box<dyn GpuBackend>,
    /// Render state requested through the `set_*` methods, applied together
    /// with the program on the next `bind_program`.
    pending_state: RenderState,
    /// Whether a pipeline has been bound since the last render state change,
    /// which draws require so that a backend always sees the state and the
    /// program together.
    #[cfg(debug_assertions)]
    pipeline_bound: bool,
}

impl Deref for Device {
    type Target = dyn GpuBackend;

    fn deref(&self) -> &Self::Target {
        &*self.backend
    }
}

impl DerefMut for Device {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.backend
    }
}

impl Device {
    pub fn new(config: GpuBackendConfig, options: DeviceOptions) -> Device {
        let backend: Box<dyn GpuBackend> = match config {
            GpuBackendConfig::Gl(gl) => Box::new(GlDevice::new(gl, options)),
        };
        Device {
            backend,
            pending_state: RenderState::default(),
            #[cfg(debug_assertions)]
            pipeline_bound: false,
        }
    }

    pub fn begin_frame(&mut self) -> GpuFrameId {
        #[cfg(debug_assertions)]
        {
            self.pipeline_bound = false;
        }
        self.backend.begin_frame()
    }

    fn state_changed(&mut self) {
        #[cfg(debug_assertions)]
        {
            self.pipeline_bound = false;
        }
    }

    pub fn set_blend_mode(&mut self, mode: BlendMode) {
        self.pending_state.blend_mode = mode;
        self.state_changed();
    }

    pub fn set_depth_test(&mut self, depth_func: Option<DepthFunction>) {
        self.pending_state.depth_test = depth_func;
        self.state_changed();
    }

    pub fn set_depth_write(&mut self, enable: bool) {
        self.pending_state.depth_write = enable;
        self.state_changed();
    }

    pub fn set_color_write(&mut self, enable: bool) {
        self.pending_state.color_write = enable;
        self.state_changed();
    }

    /// Binds `program` together with the render state requested through the
    /// `set_*` methods.
    pub fn bind_program(&mut self, program: &Program) -> bool {
        #[cfg(debug_assertions)]
        {
            self.pipeline_bound = true;
        }
        self.backend.bind_pipeline(program, &self.pending_state)
    }

    fn check_pipeline(&self) {
        #[cfg(debug_assertions)]
        debug_assert!(
            self.pipeline_bound,
            "draw with render state changed since the program was bound"
        );
    }

    pub fn draw_triangles_u32(&mut self, first_vertex: i32, index_count: i32) {
        self.check_pipeline();
        self.backend.draw_triangles_u32(first_vertex, index_count)
    }

    pub fn draw_nonindexed_lines(&mut self, first_vertex: i32, vertex_count: i32) {
        self.check_pipeline();
        self.backend.draw_nonindexed_lines(first_vertex, vertex_count)
    }

    pub fn draw_indexed_triangles(&mut self, index_count: i32) {
        self.check_pipeline();
        self.backend.draw_indexed_triangles(index_count)
    }

    pub fn draw_indexed_triangles_instanced_u16(&mut self, index_count: i32, instance_count: i32) {
        self.check_pipeline();
        self.backend.draw_indexed_triangles_instanced_u16(index_count, instance_count)
    }

    pub fn draw_indexed_triangles_instanced_base_instance_u16(
        &mut self,
        index_count: i32,
        instance_count: i32,
        base_instance: u32,
    ) {
        self.check_pipeline();
        self.backend.draw_indexed_triangles_instanced_base_instance_u16(index_count, instance_count, base_instance)
    }

    /// Create a shader program and link it immediately.
    pub fn create_program_linked(
        &mut self,
        base_filename: &'static str,
        features: &[&'static str],
        descriptor: &VertexDescriptor,
    ) -> Result<Program, ShaderError> {
        let mut program = self.create_program(base_filename, features)?;
        self.link_program(&mut program, descriptor)?;
        Ok(program)
    }

    /// Performs a blit while flipping vertically. Useful for blitting textures
    /// (which use origin-bottom-left) to the main framebuffer (which uses
    /// origin-top-left).
    pub fn blit_render_target_invert_y(
        &mut self,
        src_target: ReadTarget,
        src_rect: FramebufferIntRect,
        dest_target: DrawTarget,
        dest_rect: FramebufferIntRect,
    ) {
        let mut inverted_dest_rect = dest_rect;
        inverted_dest_rect.min.y = dest_rect.max.y;
        inverted_dest_rect.max.y = dest_rect.min.y;

        self.blit_render_target(
            src_target,
            src_rect,
            dest_target,
            inverted_dest_rect,
            TextureFilter::Linear,
        );
    }

    pub fn bind_texture<S>(&mut self, slot: S, texture: &Texture, swizzle: Swizzle)
    where
        S: Into<TextureSlot>,
    {
        self.backend.bind_texture(slot.into(), texture, swizzle)
    }

    pub fn bind_external_texture<S>(&mut self, slot: S, external_texture: &ExternalTexture)
    where
        S: Into<TextureSlot>,
    {
        self.backend.bind_external_texture(slot.into(), external_texture)
    }

    pub fn bind_shader_samplers<S>(&mut self, program: &Program, bindings: &[(&'static str, S)])
    where
        S: Into<TextureSlot> + Copy,
    {
        let bindings: Vec<(&'static str, TextureSlot)> = bindings
            .iter()
            .map(|&(name, slot)| (name, slot.into()))
            .collect();
        self.backend.bind_shader_samplers(program, &bindings)
    }

    pub fn update_vao_main_vertices<V>(
        &mut self,
        vao: &VAO,
        vertices: &[V],
        usage_hint: VertexUsageHint,
    ) {
        self.backend.update_vao_main_vertices(vao, as_bytes(vertices), usage_hint)
    }

    /// If `repeat` is `Some(count)`, each instance is repeated `count` times.
    pub fn update_vao_instances<V>(
        &mut self,
        vao: &VAO,
        instances: &[V],
        usage_hint: VertexUsageHint,
        repeat: Option<NonZeroUsize>,
    ) {
        self.backend.update_vao_instances(vao, as_bytes(instances), mem::size_of::<V>(), usage_hint, repeat)
    }

    pub fn update_vao_indices<I>(&mut self, vao: &VAO, indices: &[I], usage_hint: VertexUsageHint) {
        self.backend.update_vao_indices(vao, as_bytes(indices), usage_hint)
    }

    /// Writes `data` into a VBO at the given byte offset using an unsynchronized mapping, i.e.
    /// without waiting for in-flight draws to complete. The caller must guarantee the written range
    /// does not overlap data still being read by those draws.
    pub fn update_vbo_data_unsynchronized<V>(&mut self, vbo: VBOId, data: &[V], offset: usize) {
        self.backend.update_vbo_data_unsynchronized(vbo, as_bytes(data), offset)
    }

    /// Performs an immediate (non-PBO) upload of the whole texture.
    pub fn upload_texture_immediate<T: Texel>(&mut self, texture: &Texture, pixels: &[T]) {
        self.backend.upload_texture_immediate(texture, as_bytes(pixels))
    }

    /// Returns a `TextureUploader` which can be used to upload texture data.
    /// Once uploads have been performed the uploader must be flushed with `TextureUploader::flush()`.
    pub fn upload_texture<'a>(&mut self, pbo_pool: &'a mut UploadBufferPool) -> TextureUploader<'a> {
        TextureUploader::new(self, pbo_pool)
    }
}

/// Views an array of plain data as the bytes that are uploaded to the GPU.
fn as_bytes<T>(data: &[T]) -> &[u8] {
    unsafe { slice::from_raw_parts(data.as_ptr() as *const u8, mem::size_of_val(data)) }
}
