/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Types shared by every device backend: resource handles with their
//! backend-neutral metadata, state and descriptor structs, and the program
//! binary cache. Fields the backends fill in are `pub(super)`.

use api::{CrashAnnotator, ExternalTextureHandle, ImageBufferKind, ImageFormat, ImageRendering, MixBlendMode, VoidPtrToSizeFn};
use api::units::*;
use crate::composite::NativeSurfaceHandle;
use crate::internal_types::{FastHashMap, Swizzle};
use std::{
    cell::{Cell, RefCell},
    mem,
    num::NonZeroUsize,
    ops::Add,
    os::raw::c_void,
    path::PathBuf,
    ptr,
    rc::Rc,
    sync::Arc,
    thread,
};
use webrender_build::shader::{ProgramSourceDigest, ShaderLogLine, ShaderVersion};
use super::GpuBackend;

/// Sequence number for frames, as tracked by the device layer.
#[derive(Debug, Copy, Clone, PartialEq, Ord, Eq, PartialOrd)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct GpuFrameId(pub(super) usize);

impl GpuFrameId {
    pub fn new(value: usize) -> Self {
        GpuFrameId(value)
    }
}

impl Add<usize> for GpuFrameId {
    type Output = GpuFrameId;

    fn add(self, other: usize) -> GpuFrameId {
        GpuFrameId(self.0 + other)
    }
}

pub struct TextureSlot(pub usize);

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DepthFunction {
    Always,
    Less,
    LessEqual,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum TextureFilter {
    Nearest,
    Linear,
    Trilinear,
}

/// A structure defining a particular workflow of texture transfers.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct TextureFormatPair<T> {
    /// Format the GPU natively stores texels in.
    pub internal: T,
    /// Format we expect the users to provide the texels in.
    pub external: T,
}

impl<T: Copy> From<T> for TextureFormatPair<T> {
    fn from(value: T) -> Self {
        TextureFormatPair {
            internal: value,
            external: value,
        }
    }
}

#[derive(Debug)]
pub enum VertexAttributeKind {
    F32,
    U8Norm,
    U16Norm,
    I32,
    U16,
}

#[derive(Debug)]
pub struct VertexAttribute {
    pub name: &'static str,
    pub count: u32,
    pub kind: VertexAttributeKind,
}

impl VertexAttribute {
    pub const fn quad_instance_vertex() -> Self {
        VertexAttribute {
            name: "aPosition",
            count: 2,
            kind: VertexAttributeKind::U8Norm,
        }
    }

    pub const fn gpu_buffer_address(name: &'static str) -> Self {
        VertexAttribute {
            name,
            count: 1,
            kind: VertexAttributeKind::I32,
        }
    }

    pub const fn f32x4(name: &'static str) -> Self {
        VertexAttribute {
            name,
            count: 4,
            kind: VertexAttributeKind::F32,
        }
    }

    pub const fn f32x3(name: &'static str) -> Self {
        VertexAttribute {
            name,
            count: 3,
            kind: VertexAttributeKind::F32,
        }
    }

    pub const fn f32x2(name: &'static str) -> Self {
        VertexAttribute {
            name,
            count: 2,
            kind: VertexAttributeKind::F32,
        }
    }

    pub const fn f32(name: &'static str) -> Self {
        VertexAttribute {
            name,
            count: 1,
            kind: VertexAttributeKind::F32,
        }
    }

    pub const fn i32x4(name: &'static str) -> Self {
        VertexAttribute {
            name,
            count: 4,
            kind: VertexAttributeKind::I32,
        }
    }

    pub const fn i32x2(name: &'static str) -> Self {
        VertexAttribute {
            name,
            count: 2,
            kind: VertexAttributeKind::I32,
        }
    }

    pub const fn i32(name: &'static str) -> Self {
        VertexAttribute {
            name,
            count: 1,
            kind: VertexAttributeKind::I32,
        }
    }

    pub const fn u16x2(name: &'static str) -> Self {
        VertexAttribute {
            name,
            count: 2,
            kind: VertexAttributeKind::U16,
        }
    }
}

#[derive(Debug)]
pub struct VertexDescriptor {
    pub vertex_attributes: &'static [VertexAttribute],
    pub instance_attributes: &'static [VertexAttribute],
}

/// Method of uploading texel data from CPU to GPU.
#[derive(Debug, Clone)]
pub enum UploadMethod {
    /// Just call `glTexSubImage` directly with the CPU data pointer
    Immediate,
    /// Accumulate the changes in PBO first before transferring to a texture.
    PixelBuffer(VertexUsageHint),
}

/// Plain old data that can be used to initialize a texture.
pub unsafe trait Texel: Copy + Default {
    fn image_format() -> ImageFormat;
}

unsafe impl Texel for u8 {
    fn image_format() -> ImageFormat { ImageFormat::R8 }
}

impl VertexAttributeKind {
    pub(super) fn size_in_bytes(&self) -> u32 {
        match *self {
            VertexAttributeKind::F32 => 4,
            VertexAttributeKind::U8Norm => 1,
            VertexAttributeKind::U16Norm => 2,
            VertexAttributeKind::I32 => 4,
            VertexAttributeKind::U16 => 2,
        }
    }
}

#[cfg_attr(feature = "replay", derive(Clone))]
#[derive(Debug)]
pub struct ExternalTexture {
    /// Backend-defined identifier of the application-owned texture.
    pub(super) id: u32,
    pub(super) target: ImageBufferKind,
    uv_rect: TexelRect,
    pub(super) image_rendering: ImageRendering,
}

impl ExternalTexture {
    pub fn new(
        handle: ExternalTextureHandle,
        target: ImageBufferKind,
        uv_rect: TexelRect,
        image_rendering: ImageRendering,
    ) -> Self {
        ExternalTexture {
            id: handle.0 as u32,
            target,
            uv_rect,
            image_rendering,
        }
    }

    #[cfg(feature = "replay")]
    pub fn handle(&self) -> ExternalTextureHandle {
        ExternalTextureHandle(self.id as u64)
    }

    pub fn get_uv_rect(&self) -> TexelRect {
        self.uv_rect
    }
}

bitflags! {
    #[derive(Default, Debug, Copy, PartialEq, Eq, Clone, PartialOrd, Ord, Hash)]
    pub struct TextureFlags: u32 {
        /// This texture corresponds to one of the shared texture caches.
        const IS_SHARED_TEXTURE_CACHE = 1 << 0;
    }
}

/// WebRender interface to a GPU texture.
///
/// Because freeing a texture requires various device handles that are not
/// reachable from this struct, manual destruction via `Device` is required.
/// Our `Drop` implementation asserts that this has happened.
#[derive(Debug)]
pub struct Texture {
    /// Backend-defined identifier of the texture.
    pub(super) id: u32,
    pub(super) target: ImageBufferKind,
    pub(super) format: ImageFormat,
    pub(super) size: DeviceIntSize,
    pub(super) filter: TextureFilter,
    pub(super) flags: TextureFlags,
    /// An internally mutable swizzling state that may change between batches.
    pub(super) active_swizzle: Cell<Swizzle>,
    /// Backend-defined handle for rendering to this texture.
    ///
    /// Empty if this texture is not used as a render target or if a depth buffer is needed.
    pub(super) fbo: Option<FBOId>,
    /// Same as the above, but with a depth buffer attached.
    ///
    /// FBOs are cheap to create but expensive to reconfigure (since doing so
    /// invalidates framebuffer completeness caching). Moreover, rendering with
    /// a depth buffer attached but the depth write+test disabled relies on the
    /// driver to optimize it out of the rendering pass, which most drivers
    /// probably do but, according to jgilbert, is best not to rely on.
    ///
    /// So we lazily generate a second list of FBOs with depth. This list is
    /// empty if this texture is not used as a render target _or_ if it is, but
    /// the depth buffer has never been requested.
    ///
    /// Note that we always fill fbo, and then lazily create fbo_with_depth
    /// when needed. We could make both lazy (i.e. render targets would have one
    /// or the other, but not both, unless they were actually used in both
    /// configurations). But that would complicate a lot of logic in this module,
    /// and FBOs are cheap enough to create.
    pub(super) fbo_with_depth: Option<FBOId>,
    pub(super) last_frame_used: GpuFrameId,
}

impl Texture {
    pub fn get_dimensions(&self) -> DeviceIntSize {
        self.size
    }

    pub fn get_format(&self) -> ImageFormat {
        self.format
    }

    pub fn get_filter(&self) -> TextureFilter {
        self.filter
    }

    pub fn get_target(&self) -> ImageBufferKind {
        self.target
    }

    pub fn supports_depth(&self) -> bool {
        self.fbo_with_depth.is_some()
    }

    pub fn last_frame_used(&self) -> GpuFrameId {
        self.last_frame_used
    }

    /// Returns true if this texture was used within `threshold` frames of
    /// the current frame.
    pub fn used_recently(&self, current_frame_id: GpuFrameId, threshold: usize) -> bool {
        self.last_frame_used + threshold >= current_frame_id
    }

    /// Returns the flags for this texture.
    pub fn flags(&self) -> &TextureFlags {
        &self.flags
    }

    /// Returns a mutable borrow of the flags for this texture.
    pub fn flags_mut(&mut self) -> &mut TextureFlags {
        &mut self.flags
    }

    /// Returns the number of bytes (generally in GPU memory) that this texture
    /// consumes.
    pub fn size_in_bytes(&self) -> usize {
        let bpp = self.format.bytes_per_pixel() as usize;
        let w = self.size.width as usize;
        let h = self.size.height as usize;
        bpp * w * h
    }

    #[cfg(feature = "replay")]
    pub fn into_external(mut self) -> ExternalTexture {
        let ext = ExternalTexture {
            id: self.id,
            target: self.target,
            // TODO(gw): Support custom UV rect for external textures during captures
            uv_rect: TexelRect::new(
                0.0,
                0.0,
                self.size.width as f32,
                self.size.height as f32,
            ),
            image_rendering: ImageRendering::Auto,
        };
        self.id = 0; // don't complain, moved out
        ext
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        debug_assert!(thread::panicking() || self.id == 0);
    }
}

pub struct Program {
    /// Backend-defined identifier of the program.
    pub(super) id: u32,
    /// Backend-defined locations of the uTransform and uTextureSize uniforms,
    /// valid once the program is linked.
    pub(super) u_transform: i32,
    pub(super) u_texture_size: i32,
    pub(super) source_info: ProgramSourceInfo,
    pub(super) is_initialized: bool,
}

impl Program {
    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        debug_assert!(
            thread::panicking() || self.id == 0,
            "renderer::deinit not called"
        );
    }
}

pub struct VAO {
    /// Backend-defined identifier of the vertex array.
    pub(super) id: u32,
    pub(super) ibo_id: IBOId,
    pub(super) main_vbo_id: VBOId,
    pub(super) instance_vbo_id: VBOId,
    pub(super) instance_stride: usize,
    pub(super) instance_divisor: u32,
    pub(super) owns_vertices_and_indices: bool,
    pub(super) owns_instances: bool,
}

impl VAO {
    pub fn instance_stride(&self) -> usize {
        self.instance_stride
    }

    pub fn instance_vbo_id(&self) -> VBOId {
        self.instance_vbo_id
    }
}

impl Drop for VAO {
    fn drop(&mut self) {
        debug_assert!(
            thread::panicking() || self.id == 0,
            "renderer::deinit not called"
        );
    }
}

#[derive(Debug)]
pub struct TransferBuffer {
    /// Backend-defined identifier of the buffer.
    pub(super) id: u32,
    pub(super) reserved_size: usize,
}

impl TransferBuffer {
    pub fn get_reserved_size(&self) -> usize {
        self.reserved_size
    }
}

impl Drop for TransferBuffer {
    fn drop(&mut self) {
        debug_assert!(
            thread::panicking() || self.id == 0,
            "renderer::deinit not called or TransferBuffer not returned to pool"
        );
    }
}

pub struct MappedTransferBuffer<'a> {
    pub(super) device: &'a mut dyn GpuBackend,
    pub data: &'a [u8]
}

/// Backend-defined handle of a GPU-side completion marker, created by
/// `Device::create_fence` after a batch of commands. Once it is signaled,
/// resources those commands read from may be reused.
#[derive(Debug)]
pub struct Fence(pub(super) usize);

#[derive(Debug, PartialEq)]
pub enum FenceStatus {
    Signaled,
    Pending,
    /// The fence could not be queried; treat any resources it guards as lost.
    Error,
}

/// How a transfer buffer used for uploads is currently mapped into CPU
/// memory. The pointer is valid for the buffer's reserved size.
#[derive(Debug)]
pub enum UploadBufferMapping {
    Unmapped,
    /// Mapped only until the next `Device::flush_upload_buffer`.
    Transient(ptr::NonNull<mem::MaybeUninit<u8>>),
    /// Mapped for the buffer's lifetime; writes become visible to the GPU on
    /// `Device::flush_upload_buffer`.
    Persistent(ptr::NonNull<mem::MaybeUninit<u8>>),
}

/// One texture update sourced from an upload buffer.
#[derive(Debug)]
pub struct UploadChunk<'a> {
    pub rect: DeviceIntRect,
    /// Row stride of the data in bytes; the texture width if `None`.
    pub stride: Option<i32>,
    /// Byte offset of the data within the upload buffer.
    pub offset: usize,
    pub format_override: Option<ImageFormat>,
    pub texture: &'a Texture,
}

impl<'a> Drop for MappedTransferBuffer<'a> {
    fn drop(&mut self) {
        self.device.unmap_transfer_buffer();
    }
}

/// Backend-defined identifier of a framebuffer, i.e. a set of attachments
/// that can be drawn to or read from.
#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct FBOId(pub(super) u32);

/// Backend-defined identifier of a vertex buffer.
#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct VBOId(pub(super) u32);

/// Backend-defined identifier of an index buffer.
#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct IBOId(pub(super) u32);

#[derive(Clone, Debug)]
pub(super) enum ProgramSourceType {
    Unoptimized,
    Optimized(ShaderVersion),
}

#[derive(Clone, Debug)]
pub struct ProgramSourceInfo {
    pub(super) base_filename: &'static str,
    pub(super) features: Vec<&'static str>,
    pub(super) full_name_cstr: Rc<std::ffi::CString>,
    pub(super) source_type: ProgramSourceType,
    /// Set when an in-memory source override contributed to this program. Such
    /// a program must not be written to the binary program cache, so that a
    /// throwaway edit cannot outlive the session it was made in.
    #[cfg(feature = "debugger")]
    pub(super) from_source_override: bool,
    pub(super) digest: ProgramSourceDigest,
}

#[cfg_attr(feature = "serialize_program", derive(Deserialize, Serialize))]
pub struct ProgramBinary {
    pub(super) bytes: Vec<u8>,
    /// Backend-defined format tag for `bytes`. For OpenGL this is the binary
    /// format returned by glGetProgramBinary.
    pub(super) format: u32,
    source_digest: ProgramSourceDigest,
}

impl ProgramBinary {
    pub(super) fn new(bytes: Vec<u8>,
           format: u32,
           source_digest: ProgramSourceDigest) -> Self {
        ProgramBinary {
            bytes,
            format,
            source_digest,
        }
    }

    /// Returns a reference to the source digest hash.
    pub fn source_digest(&self) -> &ProgramSourceDigest {
        &self.source_digest
    }
}

/// The interfaces that an application can implement to handle ProgramCache update
pub trait ProgramCacheObserver {
    fn save_shaders_to_disk(&self, entries: Vec<Arc<ProgramBinary>>);
    fn set_startup_shaders(&self, entries: Vec<Arc<ProgramBinary>>);
    fn try_load_shader_from_disk(&self, digest: &ProgramSourceDigest, program_cache: &Rc<ProgramCache>);
    fn notify_program_binary_failed(&self, program_binary: &Arc<ProgramBinary>);
}

pub(super) struct ProgramCacheEntry {
    /// The binary.
    pub(super) binary: Arc<ProgramBinary>,
    /// True if the binary has been linked, i.e. used for rendering.
    pub(super) linked: bool,
}

pub struct ProgramCache {
    pub(super) entries: RefCell<FastHashMap<ProgramSourceDigest, ProgramCacheEntry>>,

    /// Optional trait object that allows the client
    /// application to handle ProgramCache updating
    pub(super) program_cache_handler: Option<Box<dyn ProgramCacheObserver>>,

    /// Programs that have not yet been cached to disk (by program_cache_handler)
    pending_entries: RefCell<Vec<Arc<ProgramBinary>>>,
}

impl ProgramCache {
    pub fn new(program_cache_observer: Option<Box<dyn ProgramCacheObserver>>) -> Rc<Self> {
        Rc::new(
            ProgramCache {
                entries: RefCell::new(FastHashMap::default()),
                program_cache_handler: program_cache_observer,
                pending_entries: RefCell::new(Vec::default()),
            }
        )
    }

    /// Save any new program binaries to the disk cache, and if startup has
    /// just completed then write the list of shaders to load on next startup.
    pub(super) fn update_disk_cache(&self, startup_complete: bool) {
        if let Some(ref handler) = self.program_cache_handler {
            if !self.pending_entries.borrow().is_empty() {
                let pending_entries = self.pending_entries.replace(Vec::default());
                handler.save_shaders_to_disk(pending_entries);
            }

            if startup_complete {
                let startup_shaders = self.entries.borrow().values()
                    .filter(|e| e.linked).map(|e| e.binary.clone())
                    .collect::<Vec<_>>();
                handler.set_startup_shaders(startup_shaders);
            }
        }
    }

    /// Add a new ProgramBinary to the cache.
    /// This function is typically used after compiling and linking a new program.
    /// The binary will be saved to disk the next time update_disk_cache() is called.
    pub(super) fn add_new_program_binary(&self, program_binary: Arc<ProgramBinary>) {
        self.pending_entries.borrow_mut().push(program_binary.clone());

        let digest = program_binary.source_digest.clone();
        let entry = ProgramCacheEntry {
            binary: program_binary,
            linked: true,
        };
        self.entries.borrow_mut().insert(digest, entry);
    }

    /// Load ProgramBinary to ProgramCache.
    /// The function is typically used to load ProgramBinary from disk.
    #[cfg(feature = "serialize_program")]
    pub fn load_program_binary(&self, program_binary: Arc<ProgramBinary>) {
        let digest = program_binary.source_digest.clone();
        let entry = ProgramCacheEntry {
            binary: program_binary,
            linked: false,
        };
        self.entries.borrow_mut().insert(digest, entry);
    }

    /// Returns the number of bytes allocated for shaders in the cache.
    pub fn report_memory(&self, op: VoidPtrToSizeFn) -> usize {
        self.entries.borrow().values()
            .map(|e| unsafe { op(e.binary.bytes.as_ptr() as *const c_void ) })
            .sum()
    }
}

#[derive(Debug, Copy, Clone)]
pub enum VertexUsageHint {
    Static,
    Dynamic,
    Stream,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GraphicsApi {
    OpenGL,
}

/// How a draw is blended with the contents of the bound draw target.
#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum BlendMode {
    None,
    Alpha,
    PremultipliedAlpha,
    PremultipliedDestOut,
    /// Destination scaled by source, used to intersect clip masks.
    Multiply,
    SubpixelDualSource,
    Advanced(MixBlendMode),
    Screen,
    Exclusion,
    PlusLighter,
    /// Debug visualisation that accumulates overdraw.
    ShowOverdraw,
}

/// How the existing contents of a color attachment are treated when a
/// render pass begins.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LoadOp {
    Load,
    /// The pass overwrites everything it later reads, so tiled GPUs need not
    /// load the previous contents.
    DontCare,
}

/// What happens to an attachment's contents when a render pass ends.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum StoreOp {
    Store,
    /// The contents are not needed after the pass, so tiled GPUs need not
    /// write them back to memory.
    Discard,
}

/// Fixed-function state that, together with a program and the render pass
/// target, makes up a pipeline. Requested through the `Device::set_*` methods
/// and applied when a program is bound.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct RenderState {
    pub blend_mode: BlendMode,
    pub depth_test: Option<DepthFunction>,
    pub depth_write: bool,
    pub color_write: bool,
}

impl Default for RenderState {
    fn default() -> Self {
        RenderState {
            blend_mode: BlendMode::None,
            depth_test: None,
            depth_write: false,
            color_write: true,
        }
    }
}

/// Parameters of a render pass. All draws and clears to a target must happen
/// between `Device::begin_render_pass` and `Device::end_render_pass`.
#[derive(Debug, Copy, Clone)]
pub struct RenderPassDescriptor {
    pub target: DrawTarget,
    /// The region of the target this pass writes to, if known. Tiled GPUs
    /// only need to load and store this region.
    pub render_area: Option<DeviceIntRect>,
    pub color_load: LoadOp,
}

/// Describes the graphics API and driver a device is running on.
#[derive(Clone, Debug)]
pub struct GraphicsApiInfo {
    pub kind: GraphicsApi,
    pub renderer: String,
    pub version: String,
}

/// Configuration for creating a `Device`.
pub struct DeviceOptions {
    pub crash_annotator: Option<Box<dyn CrashAnnotator>>,
    pub resource_override_path: Option<PathBuf>,
    pub use_optimized_shaders: bool,
    pub upload_method: UploadMethod,
    pub batched_upload_threshold: i32,
    pub cached_programs: Option<Rc<ProgramCache>>,
    pub allow_texture_storage_support: bool,
    pub allow_texture_swizzling: bool,
    pub dump_shader_source: Option<String>,
    pub surface_origin_is_top_left: bool,
    pub panic_on_gl_error: bool,
}

#[derive(Debug)]
pub struct Capabilities {
    /// Whether multisampled render targets are supported.
    pub supports_multisampling: bool,
    /// Whether the function `glCopyImageSubData` is available.
    pub supports_copy_image_sub_data: bool,
    /// Whether the device supports persistently mapped buffers, via glBufferStorage.
    pub supports_buffer_storage: bool,
    /// Whether advanced blend equations are supported.
    pub supports_advanced_blend_equation: bool,
    /// Whether advanced blend equations are coherent, meaning no barrier is
    /// required between overlapping draws.
    pub supports_advanced_blend_equation_coherent: bool,
    /// Whether dual-source blending is supported.
    pub supports_dual_source_blending: bool,
    /// Whether KHR_debug is supported for getting debug messages from
    /// the driver.
    pub supports_khr_debug: bool,
    /// Whether we can configure texture units to do swizzling on sampling.
    pub supports_texture_swizzle: bool,
    /// Whether the driver supports uploading to textures from a non-zero
    /// offset within a PBO.
    pub supports_nonzero_pbo_offsets: bool,
    /// Whether the driver supports specifying the texture usage up front.
    pub supports_texture_usage: bool,
    /// Whether offscreen render targets can be partially updated.
    pub supports_render_target_partial_update: bool,
    /// Whether we can use SSBOs.
    pub supports_shader_storage_object: bool,
    /// Whether to enforce that texture uploads be batched regardless of what
    /// the pref says.
    pub requires_batched_texture_uploads: Option<bool>,
    /// Whether we are able to ue glClear to clear regions of an alpha render target.
    /// If false, we must use a shader to clear instead.
    pub supports_alpha_target_clears: bool,
    /// Whether we must perform a full unscissored glClear on alpha targets
    /// prior to rendering.
    pub requires_alpha_target_full_clear: bool,
    /// Whether clearing a render target (immediately after binding it) is faster using a scissor
    /// rect to clear just the required area, or clearing the entire target without a scissor rect.
    pub prefers_clear_scissor: bool,
    /// Whether the driver can correctly invalidate render targets. This can be
    /// a worthwhile optimization, but is buggy on some devices.
    pub supports_render_target_invalidate: bool,
    /// Whether the driver can reliably upload data to R8 format textures.
    pub supports_r8_texture_upload: bool,
    /// Whether the extension QCOM_tiled_rendering is supported.
    pub supports_qcom_tiled_rendering: bool,
    /// Whether clip-masking is supported natively by the GL implementation
    /// rather than emulated in shaders.
    pub uses_native_clip_mask: bool,
    /// Whether anti-aliasing is supported natively by the GL implementation
    /// rather than emulated in shaders.
    pub uses_native_antialiasing: bool,
    /// Whether the extension GL_OES_EGL_image_external_essl3 is supported. If true, external
    /// textures can be used as normal. If false, external textures can only be rendered with
    /// certain shaders, and must first be copied in to regular textures for others.
    pub supports_image_external_essl3: bool,
    /// Whether rectangle textures (GL_TEXTURE_RECTANGLE) can be sampled.
    pub supports_texture_rect: bool,
    /// Whether external textures (GL_TEXTURE_EXTERNAL_OES) can be sampled.
    pub supports_texture_external: bool,
    /// Whether external textures can be sampled as BT.709 YUV, via GL_EXT_YUV_target.
    pub supports_texture_external_bt709: bool,
    /// Whether pixels read back from the default framebuffer arrive with the
    /// top row first.
    pub readback_rows_top_down: bool,
    /// Whether the VAO must be rebound after an attached VBO has been orphaned.
    pub requires_vao_rebind_after_orphaning: bool,
    /// Whether glReadPixels can read back BGRA directly (e.g. on GLES this
    /// requires GL_EXT_read_format_bgra). If false, callers must read RGBA
    /// instead and swap the red and blue channels themselves.
    pub supports_bgra_read: bool,
    /// Whether glDrawElementsInstancedBaseInstance and friends are supported,
    /// via ARB_base_instance (or GL 4.2) on desktop or EXT_base_instance on GLES.
    pub supports_base_instance: bool,
    /// The name of the renderer, as reported by GL
    pub renderer_name: String,
}

#[derive(Clone, Debug)]
pub enum ShaderError {
    /// Variant name, the driver's raw log, and the log parsed into per-line
    /// diagnostics with locations resolved back to the `.glsl` sources.
    Compilation(String, String, Vec<ShaderLogLine>),
    /// Variant name, the driver's raw log, and its parsed diagnostics. Link
    /// logs rarely carry locations, so the diagnostics are usually unmapped.
    Link(String, String, Vec<ShaderLogLine>),
}

impl ShaderError {
    pub fn name(&self) -> &str {
        match self {
            ShaderError::Compilation(name, ..) | ShaderError::Link(name, ..) => name,
        }
    }

    pub fn log(&self) -> &str {
        match self {
            ShaderError::Compilation(_, log, _) | ShaderError::Link(_, log, _) => log,
        }
    }

    pub fn diagnostics(&self) -> &[ShaderLogLine] {
        match self {
            ShaderError::Compilation(.., diagnostics) | ShaderError::Link(.., diagnostics) => {
                diagnostics
            }
        }
    }
}

/// Describes a required alignment for a stride,
/// which can either be represented in bytes or pixels.
#[derive(Copy, Clone, Debug)]
pub enum StrideAlignment {
    Bytes(NonZeroUsize),
    Pixels(NonZeroUsize),
}

impl StrideAlignment {
    pub fn num_bytes(&self, format: ImageFormat) -> NonZeroUsize {
        match *self {
            Self::Bytes(bytes) => bytes,
            Self::Pixels(pixels) => {
                assert!(format.bytes_per_pixel() > 0);
                NonZeroUsize::new(pixels.get() * format.bytes_per_pixel() as usize).unwrap()
            }
        }
    }
}

/// Contains the parameters necessary to bind a draw target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DrawTarget {
    /// Use the device's default draw target, with the provided dimensions,
    /// which are used to set the viewport.
    Default {
        /// Target rectangle to draw.
        rect: FramebufferIntRect,
        /// Total size of the target.
        total_size: FramebufferIntSize,
        surface_origin_is_top_left: bool,
    },
    /// Use the provided texture.
    Texture {
        /// Size of the texture in pixels
        dimensions: DeviceIntSize,
        /// Whether to draw with the texture's associated depth target
        with_depth: bool,
        /// FBO that corresponds to the selected layer / depth mode
        fbo_id: FBOId,
    },
    /// An OS compositor surface
    NativeSurface {
        offset: DeviceIntPoint,
        handle: NativeSurfaceHandle,
        dimensions: DeviceIntSize,
    },
}

impl DrawTarget {
    pub fn new_default(size: DeviceIntSize, surface_origin_is_top_left: bool) -> Self {
        let total_size = device_size_as_framebuffer_size(size);
        DrawTarget::Default {
            rect: total_size.into(),
            total_size,
            surface_origin_is_top_left,
        }
    }

    /// Returns true if this draw target corresponds to the default framebuffer.
    pub fn is_default(&self) -> bool {
        match *self {
            DrawTarget::Default {..} => true,
            _ => false,
        }
    }

    pub fn from_texture(
        texture: &Texture,
        with_depth: bool,
    ) -> Self {
        let fbo_id = if with_depth {
            texture.fbo_with_depth.unwrap()
        } else {
            texture.fbo.unwrap()
        };

        DrawTarget::Texture {
            dimensions: texture.get_dimensions(),
            fbo_id,
            with_depth,
        }
    }

    /// Returns the dimensions of this draw-target.
    pub fn dimensions(&self) -> DeviceIntSize {
        match *self {
            DrawTarget::Default { total_size, .. } => total_size.cast_unit(),
            DrawTarget::Texture { dimensions, .. } => dimensions,
            DrawTarget::NativeSurface { dimensions, .. } => dimensions,
        }
    }

    pub fn offset(&self) -> DeviceIntPoint {
        match *self {
            DrawTarget::Default { .. } |
            DrawTarget::Texture { .. } => {
                DeviceIntPoint::zero()
            }
            DrawTarget::NativeSurface { offset, .. } => offset,
        }
    }

    pub fn to_framebuffer_rect(&self, device_rect: DeviceIntRect) -> FramebufferIntRect {
        let mut fb_rect = device_rect_as_framebuffer_rect(&device_rect);
        match *self {
            DrawTarget::Default { ref rect, surface_origin_is_top_left, .. } => {
                // perform a Y-flip here
                if !surface_origin_is_top_left {
                    let w = fb_rect.width();
                    let h = fb_rect.height();
                    fb_rect.min.x = fb_rect.min.x + rect.min.x;
                    fb_rect.min.y = rect.max.y - fb_rect.max.y;
                    fb_rect.max.x = fb_rect.min.x + w;
                    fb_rect.max.y = fb_rect.min.y + h;
                }
            }
            DrawTarget::Texture { .. } | DrawTarget::NativeSurface { .. } => (),
        }
        fb_rect
    }

    pub fn surface_origin_is_top_left(&self) -> bool {
        match *self {
            DrawTarget::Default { surface_origin_is_top_left, .. } => surface_origin_is_top_left,
            DrawTarget::Texture { .. } | DrawTarget::NativeSurface { .. } => true,
        }
    }

    /// Given a scissor rect, convert it to the right coordinate space
    /// depending on the draw target kind. If no scissor rect was supplied,
    /// returns a scissor rect that encloses the entire render target.
    pub fn build_scissor_rect(
        &self,
        scissor_rect: Option<DeviceIntRect>,
    ) -> FramebufferIntRect {
        let dimensions = self.dimensions();

        match scissor_rect {
            Some(scissor_rect) => match *self {
                DrawTarget::Default { ref rect, .. } => {
                    self.to_framebuffer_rect(scissor_rect)
                        .intersection(rect)
                        .unwrap_or_else(FramebufferIntRect::zero)
                }
                DrawTarget::NativeSurface { offset, .. } => {
                    device_rect_as_framebuffer_rect(&scissor_rect.translate(offset.to_vector()))
                }
                DrawTarget::Texture { .. } => {
                    device_rect_as_framebuffer_rect(&scissor_rect)
                }
            }
            None => {
                FramebufferIntRect::from_size(
                    device_size_as_framebuffer_size(dimensions),
                )
            }
        }
    }
}

/// Contains the parameters necessary to bind a texture-backed read target.
#[derive(Clone, Copy, Debug)]
pub enum ReadTarget {
    /// Use the device's default draw target.
    Default,
    /// Use the provided texture,
    Texture {
        /// ID of the FBO to read from.
        fbo_id: FBOId,
    },
    /// An FBO bound to a native (OS compositor) surface
    NativeSurface {
        fbo_id: FBOId,
        offset: DeviceIntPoint,
    },
}

impl ReadTarget {
    pub fn from_texture(
        texture: &Texture,
    ) -> Self {
        ReadTarget::Texture {
            fbo_id: texture.fbo.unwrap(),
        }
    }

    pub(super) fn offset(&self) -> DeviceIntPoint {
        match *self {
            ReadTarget::Default |
            ReadTarget::Texture { .. } => {
                DeviceIntPoint::zero()
            }

            ReadTarget::NativeSurface { offset, .. } => {
                offset
            }
        }
    }
}

impl From<DrawTarget> for ReadTarget {
    fn from(t: DrawTarget) -> Self {
        match t {
            DrawTarget::Default { .. } => {
                ReadTarget::Default
            }
            DrawTarget::NativeSurface { handle, offset, .. } => {
                ReadTarget::NativeSurface {
                    fbo_id: FBOId(handle.0 as u32),
                    offset,
                }
            }
            DrawTarget::Texture { fbo_id, .. } => {
                ReadTarget::Texture { fbo_id }
            }
        }
    }
}
