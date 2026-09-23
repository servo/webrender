/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::super::shader_source::{OPTIMIZED_SHADERS, UNOPTIMIZED_SHADERS};
use super::query::{GpuProfiler, GpuQueryBackend, GpuQueryId, GpuQueryKind};
use super::types::*;
use super::GpuBackend;
use api::{ImageDescriptor, ImageFormat, Parameter, BoolParameter, IntParameter, ImageRendering};
use api::{ExternalTextureHandle, MixBlendMode, ImageBufferKind};
use api::{CrashAnnotator, CrashAnnotation, CrashAnnotatorGuard};
use api::units::*;
use euclid::default::Transform3D;
use gleam::gl;
use crate::render_api::MemoryReport;
use crate::internal_types::{FastHashMap, RenderTargetInfo, Swizzle, SwizzleSettings};
#[cfg(feature = "debugger")]
use crate::internal_types::FastHashSet;
use crate::util::round_up_to_multiple;
use crate::profiler;
use log::Level;
#[cfg(feature = "debugger")]
use std::cell::RefCell;
use std::{
    borrow::Cow,
    cell::Cell,
    cmp,
    collections::hash_map::Entry,
    mem,
    num::NonZeroUsize,
    os::raw::c_void,
    path::PathBuf,
    ptr,
    rc::Rc,
    slice,
    sync::Arc,
    time::Duration,
};
use webrender_build::shader::{
    ProgramSourceDigest, ShaderFeatureFlags, ShaderKind, ShaderSourceMap,
    ShaderVersion,
    build_shader_main_string, build_shader_prefix_string, do_build_shader_string,
    shader_source_from_file,
};
use malloc_size_of::MallocSizeOfOps;

// In some places we need to temporarily bind a texture to any slot.
const DEFAULT_TEXTURE: TextureSlot = TextureSlot(0);

impl DepthFunction {
    fn to_gl(self) -> gl::GLenum {
        match self {
            DepthFunction::Always => gl::ALWAYS,
            DepthFunction::Less => gl::LESS,
            DepthFunction::LessEqual => gl::LEQUAL,
        }
    }
}

enum FBOTarget {
    Read,
    Draw,
}

/// Returns the size in bytes of a depth target with the given dimensions.
fn depth_target_size_in_bytes(dimensions: &DeviceIntSize) -> usize {
    // DEPTH24 textures generally reserve 3 bytes for depth and 1 byte
    // for stencil, so we measure them as 32 bits.
    let pixels = dimensions.width * dimensions.height;
    (pixels as usize) * 4
}

fn get_gl_target(target: ImageBufferKind) -> gl::GLuint {
    match target {
        ImageBufferKind::Texture2D => gl::TEXTURE_2D,
        ImageBufferKind::TextureRect => gl::TEXTURE_RECTANGLE,
        ImageBufferKind::TextureExternal => gl::TEXTURE_EXTERNAL_OES,
        ImageBufferKind::TextureExternalBT709 => gl::TEXTURE_EXTERNAL_OES,
    }
}

fn supports_extension(extensions: &[String], extension: &str) -> bool {
    extensions.iter().any(|s| s == extension)
}

/// Which GL extension, if any, annotates the command stream for the GPU
/// profiler.
#[derive(Copy, Clone, Debug)]
enum GpuDebugMethod {
    None,
    MarkerEXT,
    KHR,
}

/// GPU queries and markers on a GL context.
struct GlQueries {
    gl: Rc<dyn gl::Gl>,
    debug_method: GpuDebugMethod,
}

impl GpuQueryBackend for GlQueries {
    fn create_queries(&self, count: usize) -> Vec<GpuQueryId> {
        self.gl.gen_queries(count as gl::GLsizei).into_iter().map(GpuQueryId).collect()
    }

    fn delete_queries(&self, queries: &[GpuQueryId]) {
        let ids: Vec<gl::GLuint> = queries.iter().map(|q| q.0).collect();
        self.gl.delete_queries(&ids);
    }

    fn begin_query(&self, kind: GpuQueryKind, query: GpuQueryId) {
        self.gl.begin_query(gl_query_target(kind), query.0);
    }

    fn end_query(&self, kind: GpuQueryKind) {
        self.gl.end_query(gl_query_target(kind));
    }

    fn query_result(&self, query: GpuQueryId) -> u64 {
        self.gl.get_query_object_ui64v(query.0, gl::QUERY_RESULT)
    }

    fn supports_markers(&self) -> bool {
        !matches!(self.debug_method, GpuDebugMethod::None)
    }

    fn push_marker_group(&self, label: &str) {
        match self.debug_method {
            GpuDebugMethod::KHR => self.gl.push_debug_group_khr(gl::DEBUG_SOURCE_APPLICATION, 0, label),
            GpuDebugMethod::MarkerEXT => self.gl.push_group_marker_ext(label),
            GpuDebugMethod::None => {}
        }
    }

    fn pop_marker_group(&self) {
        match self.debug_method {
            GpuDebugMethod::KHR => self.gl.pop_debug_group_khr(),
            GpuDebugMethod::MarkerEXT => self.gl.pop_group_marker_ext(),
            GpuDebugMethod::None => {}
        }
    }

    fn insert_marker(&self, label: &str) {
        match self.debug_method {
            GpuDebugMethod::KHR => self.gl.debug_message_insert_khr(gl::DEBUG_SOURCE_APPLICATION, gl::DEBUG_TYPE_MARKER, 0, gl::DEBUG_SEVERITY_NOTIFICATION, label),
            GpuDebugMethod::MarkerEXT => self.gl.insert_event_marker_ext(label),
            GpuDebugMethod::None => {}
        }
    }
}

fn gl_query_target(kind: GpuQueryKind) -> gl::GLenum {
    match kind {
        GpuQueryKind::TimeElapsed => gl::TIME_ELAPSED,
        GpuQueryKind::SamplesPassed => gl::SAMPLES_PASSED,
    }
}

fn get_shader_version(gl: &dyn gl::Gl) -> ShaderVersion {
    match gl.get_type() {
        gl::GlType::Gl => ShaderVersion::Gl,
        gl::GlType::Gles => ShaderVersion::Gles,
    }
}

// Get an unoptimized shader string by name, from the built in resources or
// an override path, if supplied.
pub fn get_unoptimized_shader_source(shader_name: &str, base_path: Option<&PathBuf>) -> Cow<'static, str> {
    if let Some(ref base) = base_path {
        let shader_path = base.join(&format!("{}.glsl", shader_name));
        Cow::Owned(shader_source_from_file(&shader_path))
    } else {
        Cow::Borrowed(
            UNOPTIMIZED_SHADERS
            .get(shader_name)
            .expect("Shader not found")
            .source
        )
    }
}

impl VertexAttribute {
    fn size_in_bytes(&self) -> u32 {
        self.count * self.kind.size_in_bytes()
    }

    fn bind_to_vao(
        &self,
        attr_index: gl::GLuint,
        divisor: gl::GLuint,
        stride: gl::GLint,
        offset: gl::GLuint,
        gl: &dyn gl::Gl,
    ) {
        gl.enable_vertex_attrib_array(attr_index);
        gl.vertex_attrib_divisor(attr_index, divisor);

        match self.kind {
            VertexAttributeKind::F32 => {
                gl.vertex_attrib_pointer(
                    attr_index,
                    self.count as gl::GLint,
                    gl::FLOAT,
                    false,
                    stride,
                    offset,
                );
            }
            VertexAttributeKind::U8Norm => {
                gl.vertex_attrib_pointer(
                    attr_index,
                    self.count as gl::GLint,
                    gl::UNSIGNED_BYTE,
                    true,
                    stride,
                    offset,
                );
            }
            VertexAttributeKind::U16Norm => {
                gl.vertex_attrib_pointer(
                    attr_index,
                    self.count as gl::GLint,
                    gl::UNSIGNED_SHORT,
                    true,
                    stride,
                    offset,
                );
            }
            VertexAttributeKind::I32 => {
                gl.vertex_attrib_i_pointer(
                    attr_index,
                    self.count as gl::GLint,
                    gl::INT,
                    stride,
                    offset,
                );
            }
            VertexAttributeKind::U16 => {
                gl.vertex_attrib_i_pointer(
                    attr_index,
                    self.count as gl::GLint,
                    gl::UNSIGNED_SHORT,
                    stride,
                    offset,
                );
            }
        }
    }
}

impl VertexDescriptor {
    fn instance_stride(&self) -> u32 {
        self.instance_attributes
            .iter()
            .map(|attr| attr.size_in_bytes())
            .sum()
    }

    fn bind_attributes(
        attributes: &[VertexAttribute],
        start_index: usize,
        divisor: u32,
        gl: &dyn gl::Gl,
        vbo: VBOId,
    ) {
        vbo.bind(gl);

        let stride: u32 = attributes
            .iter()
            .map(|attr| attr.size_in_bytes())
            .sum();

        let mut offset = 0;
        for (i, attr) in attributes.iter().enumerate() {
            let attr_index = (start_index + i) as gl::GLuint;
            attr.bind_to_vao(attr_index, divisor, stride as _, offset, gl);
            offset += attr.size_in_bytes();
        }
    }

    fn bind(&self, gl: &dyn gl::Gl, main: VBOId, instance: VBOId, instance_divisor: u32) {
        Self::bind_attributes(self.vertex_attributes, 0, 0, gl, main);

        if !self.instance_attributes.is_empty() {
            Self::bind_attributes(
                self.instance_attributes,
                self.vertex_attributes.len(),
                instance_divisor,
                gl,
                instance,
            );
        }
    }
}

impl VBOId {
    fn bind(&self, gl: &dyn gl::Gl) {
        gl.bind_buffer(gl::ARRAY_BUFFER, self.0);
    }
}

impl IBOId {
    fn bind(&self, gl: &dyn gl::Gl) {
        gl.bind_buffer(gl::ELEMENT_ARRAY_BUFFER, self.0);
    }
}

impl FBOId {
    fn bind(&self, gl: &dyn gl::Gl, target: FBOTarget) {
        let target = match target {
            FBOTarget::Read => gl::READ_FRAMEBUFFER,
            FBOTarget::Draw => gl::DRAW_FRAMEBUFFER,
        };
        gl.bind_framebuffer(target, self.0);
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct RBOId(gl::GLuint);

impl ProgramSourceInfo {
    fn new(
        device: &GlDevice,
        name: &'static str,
        features: &[&'static str],
    ) -> Self {

        // Compute the digest. Assuming the device has a `ProgramCache`, this
        // will always be needed, whereas the source is rarely needed.

        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        // Setup.
        let mut hasher = DefaultHasher::new();
        // Cached binaries are only valid for the graphics API that produced
        // them, so keep digests from different backends distinct.
        hasher.write(b"opengl");
        let gl_version = get_shader_version(&*device.gl());

        // Hash the renderer name.
        hasher.write(device.capabilities.renderer_name.as_bytes());

        let full_name = Self::make_full_name(name, features);

        // An overridden source only exists as `.glsl`, so the build-time
        // optimized variant no longer describes this program. Without this the
        // edit would be silently ignored wherever optimized shaders are in use.
        let has_source_override = device.has_shader_source_override_for(name);

        let optimized_source = if device.use_optimized_shaders && !has_source_override {
            OPTIMIZED_SHADERS.get(&(gl_version, &full_name)).or_else(|| {
                warn!("Missing optimized shader source for {}", &full_name);
                None
            })
        } else {
            None
        };

        let source_type = match optimized_source {
            Some(source_and_digest) => {
                // Optimized shader sources are used as-is, without any run-time processing.
                // The vertex and fragment shaders are different, so must both be hashed.
                // We use the hashes that were computed at build time, and verify it in debug builds.
                if cfg!(debug_assertions) {
                    let mut h = DefaultHasher::new();
                    h.write(source_and_digest.vert_source.as_bytes());
                    h.write(source_and_digest.frag_source.as_bytes());
                    let d: ProgramSourceDigest = h.into();
                    let digest = d.to_string();
                    debug_assert_eq!(digest, source_and_digest.digest);
                    hasher.write(digest.as_bytes());
                } else {
                    hasher.write(source_and_digest.digest.as_bytes());
                }

                ProgramSourceType::Optimized(gl_version)
            }
            None => {
                // For non-optimized sources we compute the hash by walking the static strings
                // in the same order as we would when concatenating the source, to avoid
                // heap-allocating in the common case.
                //
                // Note that we cheat a bit to make the hashing more efficient. First, the only
                // difference between the vertex and fragment shader is a single deterministic
                // define, so we don't need to hash both. Second, we precompute the digest of the
                // expanded source file at build time, and then just hash that digest here.
                let override_path = device.resource_override_path.as_ref();
                let overridden = override_path.is_some() || has_source_override;
                let source_and_digest = UNOPTIMIZED_SHADERS.get(&name).expect("Shader not found");

                let mut source_map = ShaderSourceMap::new();

                // Hash the prefix string.
                build_shader_prefix_string(
                    gl_version,
                    &features,
                    ShaderKind::Vertex,
                    &name,
                    &mut source_map,
                    &mut |s| hasher.write(s.as_bytes()),
                );

                // Hash the shader file contents. We use a precomputed digest, and
                // verify it in debug builds.
                if overridden || cfg!(debug_assertions) {
                    let mut h = DefaultHasher::new();
                    build_shader_main_string(
                        &name,
                        &|f| device.get_shader_source(f),
                        &mut source_map,
                        &mut |s| h.write(s.as_bytes())
                    );
                    let d: ProgramSourceDigest = h.into();
                    let digest = format!("{}", d);
                    debug_assert!(overridden || digest == source_and_digest.digest);
                    hasher.write(digest.as_bytes());
                } else {
                    hasher.write(source_and_digest.digest.as_bytes());
                }

                ProgramSourceType::Unoptimized
            }
        };

        // Finish.
        ProgramSourceInfo {
            base_filename: name,
            features: features.to_vec(),
            full_name_cstr: Rc::new(std::ffi::CString::new(full_name).unwrap()),
            source_type,
            #[cfg(feature = "debugger")]
            from_source_override: has_source_override,
            digest: hasher.into(),
        }
    }

    /// Build the source to hand to the driver, along with the map needed to
    /// resolve the driver's log back to the `.glsl` sources. Optimized sources
    /// are preprocessed at build time and have no map.
    fn compute_source(
        &self,
        device: &GlDevice,
        kind: ShaderKind,
    ) -> (String, Option<ShaderSourceMap>) {
        let full_name = self.full_name();
        match self.source_type {
            ProgramSourceType::Optimized(gl_version) => {
                let shader = OPTIMIZED_SHADERS
                    .get(&(gl_version, &full_name))
                    .unwrap_or_else(|| panic!("Missing optimized shader source for {}", full_name));

                let source = match kind {
                    ShaderKind::Vertex => shader.vert_source.to_string(),
                    ShaderKind::Fragment => shader.frag_source.to_string(),
                };
                (source, None)
            },
            ProgramSourceType::Unoptimized => {
                let mut src = String::new();
                let source_map = device.build_shader_string(
                    &self.features,
                    kind,
                    self.base_filename,
                    |s| src.push_str(s),
                );
                (src, Some(source_map))
            }
        }
    }

    fn make_full_name(base_filename: &'static str, features: &[&'static str]) -> String {
        if features.is_empty() {
            base_filename.to_string()
        } else {
            format!("{}_{}", base_filename, features.join("_"))
        }
    }

    fn full_name(&self) -> String {
        Self::make_full_name(self.base_filename, &self.features)
    }

    /// Whether a runtime source override contributed to this program, and so
    /// its binary must be kept out of the program cache. Always false when the
    /// debugger is not built in, since nothing can install an override.
    #[cfg(feature = "debugger")]
    fn from_source_override(&self) -> bool {
        self.from_source_override
    }

    #[cfg(not(feature = "debugger"))]
    fn from_source_override(&self) -> bool {
        false
    }
}

impl VertexUsageHint {
    fn to_gl(&self) -> gl::GLuint {
        match *self {
            VertexUsageHint::Static => gl::STATIC_DRAW,
            VertexUsageHint::Dynamic => gl::DYNAMIC_DRAW,
            VertexUsageHint::Stream => gl::STREAM_DRAW,
        }
    }
}

/// The render state the GL context is known to hold. A field is `None` when
/// it is unknown, e.g. after code outside the device may have used the
/// context, and is then applied unconditionally on the next use.
struct GlRenderStateCache {
    blend_mode: Option<BlendMode>,
    depth_test: Option<Option<DepthFunction>>,
    depth_write: Option<bool>,
    color_write: Option<bool>,
}

impl Default for GlRenderStateCache {
    fn default() -> Self {
        GlRenderStateCache {
            blend_mode: None,
            depth_test: None,
            depth_write: None,
            // The color mask is assumed to be left alone by code outside the
            // device, as it always has been; SWGL does not implement
            // glColorMask, so it must not be set unless the renderer asks.
            color_write: Some(true),
        }
    }
}

/// A refcounted depth target, which may be shared by multiple textures across
/// the device.
struct SharedDepthTarget {
    /// The Render Buffer Object representing the depth target.
    rbo_id: RBOId,
    /// Reference count. When this drops to zero, the RBO is deleted.
    refcount: usize,
}

#[cfg(debug_assertions)]
impl Drop for SharedDepthTarget {
    fn drop(&mut self) {
        debug_assert!(std::thread::panicking() || self.refcount == 0);
    }
}

/// Describes for which texture formats to use the glTexStorage*
/// family of functions.
#[derive(PartialEq, Debug)]
enum TexStorageUsage {
    Never,
    NonBGRA8,
    Always,
}

// We get 24 bits of Z value - use up 22 bits of it to give us
// 4 bits to account for GPU issues. This seems to manifest on
// some GPUs under certain perspectives due to z interpolation
// precision problems.
const RESERVE_DEPTH_BITS: i32 = 2;

pub struct GlDevice {
    gl: Rc<dyn gl::Gl>,

    /// If non-None, |gl| points to a profiling wrapper, and this points to the
    /// underling Gl instance.
    base_gl: Option<Rc<dyn gl::Gl>>,

    // device state
    bound_textures: [gl::GLuint; 16],
    bound_program: gl::GLuint,
    bound_program_name: Rc<std::ffi::CString>,
    bound_vao: gl::GLuint,
    bound_read_fbo: (FBOId, DeviceIntPoint),
    bound_draw_fbo: FBOId,
    current_render_pass: Option<RenderPassDescriptor>,
    /// Framebuffer used to read back textures, created on first use.
    scratch_read_fbo: Option<FBOId>,
    default_read_fbo: FBOId,
    default_draw_fbo: FBOId,

    /// Track depth state for assertions. Note that the default FBO has depth,
    /// so this defaults to true.
    depth_available: bool,

    upload_method: UploadMethod,
    use_batched_texture_uploads: bool,
    /// Whether to use draw calls instead of regular blitting commands.
    ///
    /// Note: this currently only applies to the batched texture uploads
    /// path.
    use_draw_calls_for_texture_copy: bool,
    /// Number of pixels below which we prefer batched uploads.
    batched_upload_threshold: i32,

    // HW or API capabilities
    capabilities: Capabilities,

    color_formats: TextureFormatPair<ImageFormat>,
    bgra_formats: TextureFormatPair<gl::GLuint>,
    bgra_pixel_type: gl::GLuint,
    swizzle_settings: SwizzleSettings,
    depth_format: gl::GLuint,

    /// Map from texture dimensions to shared depth buffers for render targets.
    ///
    /// Render targets often have the same width/height, so we can save memory
    /// by sharing these across targets.
    depth_targets: FastHashMap<DeviceIntSize, SharedDepthTarget>,

    // debug
    inside_frame: bool,
    crash_annotator: Option<Box<dyn CrashAnnotator>>,
    annotate_draw_call_crashes: bool,

    // resources
    resource_override_path: Option<PathBuf>,

    /// Whether to use shaders that have been optimized at build time.
    use_optimized_shaders: bool,

    max_texture_size: i32,
    cached_programs: Option<Rc<ProgramCache>>,

    // Frame counter. This is used to map between CPU
    // frames and GPU frames.
    frame_id: GpuFrameId,

    /// When to use glTexStorage*. We prefer this over glTexImage* because it
    /// guarantees that mipmaps won't be generated (which they otherwise are on
    /// some drivers, particularly ANGLE). However, it is not always supported
    /// at all, or for BGRA8 format. If it's not supported for the required
    /// format, we fall back to glTexImage*.
    texture_storage_usage: TexStorageUsage,

    /// Required stride alignment for pixel transfers. This may be required for
    /// correctness reasons due to driver bugs, or for performance reasons to
    /// ensure we remain on the fast-path for transfers.
    required_transfer_stride: StrideAlignment,

    /// Whether we must ensure the source strings passed to glShaderSource()
    /// are null-terminated, to work around driver bugs.
    requires_null_terminated_shader_source: bool,

    /// Whether we must unbind any texture from GL_TEXTURE_EXTERNAL_OES before
    /// binding to GL_TEXTURE_2D, to work around an android emulator bug.
    requires_texture_external_unbind: bool,

    ///
    is_software_webrender: bool,

    // GL extensions
    extensions: Vec<String>,

    /// Dumps the source of the shader with the given name
    dump_shader_source: Option<String>,

    /// Shader sources pushed at runtime by the remote debugger, keyed by
    /// `.glsl` file stem. Takes precedence over `resource_override_path` and
    /// over the sources built into the binary.
    #[cfg(feature = "debugger")]
    shader_source_overrides: FastHashMap<String, String>,

    /// `#include` closure of each shader, keyed by base filename. Only
    /// populated while overrides are installed, and dropped whenever the
    /// override set changes, since an edit can add or remove an `#include`.
    #[cfg(feature = "debugger")]
    shader_include_closures: RefCell<FastHashMap<String, FastHashSet<String>>>,

    surface_origin_is_top_left: bool,

    gl_state: GlRenderStateCache,

    // count created/deleted textures to report in the profiler.
    textures_created: u32,
    textures_deleted: u32,

    /// When true, the pixels of newly created color render targets are
    /// initialized with an opaque pink color for debugging purposes.
    /// Controlled by the `DebugFlags::COLOR_TARGET_INIT` debug flag.
    initialize_color_targets_with_pink: bool,
}

/// Parses the major, release, and patch versions from a GL_VERSION string on
/// Mali devices. For example, for the version string
/// "OpenGL ES 3.2 v1.r36p0-01eac0.28ab3a577f105e026887e2b4c93552fb" this
/// returns Some((1, 36, 0)). Returns None if the version cannot be parsed.
fn parse_mali_version(version_string: &str) -> Option<(u32, u32, u32)> {
    let (_prefix, version_string) = version_string.split_once("v")?;
    let (v_str, version_string) = version_string.split_once(".r")?;
    let v = v_str.parse().ok()?;

    let (r_str, version_string) = version_string.split_once("p")?;
    let r = r_str.parse().ok()?;

    // Not all devices have the trailing string following the "p" number.
    let (p_str, _) = version_string.split_once("-").unwrap_or((version_string, ""));
    let p = p_str.parse().ok()?;

    Some((v, r, p))
}

/// Returns whether this GPU belongs to the Mali Midgard family
fn is_mali_midgard(renderer_name: &str) -> bool {
    renderer_name.starts_with("Mali-T")
}

/// Returns whether this GPU belongs to the Mali Bifrost family
fn is_mali_bifrost(renderer_name: &str) -> bool {
    renderer_name == "Mali-G31"
        || renderer_name == "Mali-G51"
        || renderer_name == "Mali-G71"
        || renderer_name == "Mali-G52"
        || renderer_name == "Mali-G72"
        || renderer_name == "Mali-G76"
}

/// Returns whether this GPU belongs to the Mali Valhall family
fn is_mali_valhall(renderer_name: &str) -> bool {
    // As new Valhall GPUs may be released in the future we match all Mali-G models, apart from
    // Bifrost models (of which we don't expect any new ones to be released)
    renderer_name.starts_with("Mali-G") && !is_mali_bifrost(renderer_name)
}
#[inline(never)]
fn gl_error_string(code: u32) -> &'static str {
    match code {
        gl::INVALID_ENUM => "GL_INVALID_ENUM",
        gl::INVALID_VALUE => "GL_INVALID_VALUE",
        gl::INVALID_OPERATION => "GL_INVALID_OPERATION",
        gl::STACK_OVERFLOW => "GL_STACK_OVERFLOW",
        gl::STACK_UNDERFLOW => "GL_STACK_UNDERFLOW",
        gl::OUT_OF_MEMORY => "GL_OUT_OF_MEMORY",
        gl::INVALID_FRAMEBUFFER_OPERATION => "GL_INVALID_FRAMEBUFFER_OPERATION",
        0x507 => "GL_CONTEXT_LOST",
        _ => "(unknown error code)",
    }
}

impl GlDevice {
    pub fn new(
        mut gl: Rc<dyn gl::Gl>,
        options: DeviceOptions,
    ) -> GlDevice {
        let DeviceOptions {
            crash_annotator,
            resource_override_path,
            use_optimized_shaders,
            upload_method,
            batched_upload_threshold,
            cached_programs,
            allow_texture_storage_support,
            allow_texture_swizzling,
            dump_shader_source,
            surface_origin_is_top_left,
            panic_on_gl_error,
        } = options;
        let mut max_texture_size = [0];
        unsafe {
            gl.get_integer_v(gl::MAX_TEXTURE_SIZE, &mut max_texture_size);
        }

        // We cap the max texture size at 16384. Some hardware report higher
        // capabilities but get very unstable with very large textures.
        // Bug 1702494 tracks re-evaluating this cap.
        let max_texture_size = max_texture_size[0].min(16384);

        let renderer_name = gl.get_string(gl::RENDERER);
        info!("Renderer: {}", renderer_name);
        let version_string = gl.get_string(gl::VERSION);
        info!("Version: {}", version_string);
        info!("Max texture size: {}", max_texture_size);

        let mut extension_count = [0];
        unsafe {
            gl.get_integer_v(gl::NUM_EXTENSIONS, &mut extension_count);
        }
        let extension_count = extension_count[0] as gl::GLuint;
        let mut extensions = Vec::new();
        for i in 0 .. extension_count {
            extensions.push(gl.get_string_i(gl::EXTENSIONS, i));
        }

        // We block this on Mali Valhall GPUs as the extension's functions always return
        // GL_OUT_OF_MEMORY, causing us to panic in debug builds.
        let supports_khr_debug = supports_extension(&extensions, "GL_KHR_debug")
            && !is_mali_valhall(&renderer_name);

        // On debug builds, assert that each GL call is error-free. We don't do
        // this on release builds because the synchronous call can stall the
        // pipeline.
        if panic_on_gl_error || cfg!(debug_assertions) {
            gl = gl::ErrorReactingGl::wrap(gl, move |gl, name, code| {
                if supports_khr_debug {
                    Self::log_driver_messages(gl);
                }
                let err_name = gl_error_string(code);
                error!("Caught GL error 0x{:x} {} at {}", code, err_name, name);
                panic!("Caught GL error 0x{:x} {} at {}", code, err_name, name);
            });
        }

        if supports_extension(&extensions, "GL_ANGLE_provoking_vertex") {
            gl.provoking_vertex_angle(gl::FIRST_VERTEX_CONVENTION);
        }

        let supports_texture_usage = supports_extension(&extensions, "GL_ANGLE_texture_usage");

        // Our common-case image data in Firefox is BGRA, so we make an effort
        // to use BGRA as the internal texture storage format to avoid the need
        // to swizzle during upload. Currently we only do this on GLES (and thus
        // for Windows, via ANGLE).
        //
        // On Mac, Apple docs [1] claim that BGRA is a more efficient internal
        // format, but they don't support it with glTextureStorage. As a workaround,
        // we pretend that it's RGBA8 for the purposes of texture transfers,
        // but swizzle R with B for the texture sampling.
        //
        // We also need our internal format types to be sized, since glTexStorage*
        // will reject non-sized internal format types.
        //
        // Unfortunately, with GL_EXT_texture_format_BGRA8888, BGRA8 is not a
        // valid internal format (for glTexImage* or glTexStorage*) unless
        // GL_EXT_texture_storage is also available [2][3], which is usually
        // not the case on GLES 3 as the latter's functionality has been
        // included by default but the former has not been updated.
        // The extension is available on ANGLE, but on Android this usually
        // means we must fall back to using unsized BGRA and glTexImage*.
        //
        // Overall, we have the following factors in play when choosing the formats:
        //   - with glTexStorage, the internal format needs to match the external format,
        //     or the driver would have to do the conversion, which is slow
        //   - on desktop GL, there is no BGRA internal format. However, initializing
        //     the textures with glTexImage as RGBA appears to use BGRA internally,
        //     preferring BGRA external data [4].
        //   - when glTexStorage + BGRA internal format is not supported,
        //     and the external data is BGRA, we have the following options:
        //       1. use glTexImage with RGBA internal format, this costs us VRAM for mipmaps
        //       2. use glTexStorage with RGBA internal format, this costs us the conversion by the driver
        //       3. pretend we are uploading RGBA and set up the swizzling of the texture unit - this costs us batch breaks
        //
        // [1] https://developer.apple.com/library/archive/documentation/
        //     GraphicsImaging/Conceptual/OpenGL-MacProgGuide/opengl_texturedata/
        //     opengl_texturedata.html#//apple_ref/doc/uid/TP40001987-CH407-SW22
        // [2] https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_format_BGRA8888.txt
        // [3] https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_storage.txt
        // [4] http://http.download.nvidia.com/developer/Papers/2005/Fast_Texture_Transfers/Fast_Texture_Transfers.pdf

        // On the android emulator glTexImage fails to create textures larger than 3379.
        // So we must use glTexStorage instead. See bug 1591436.
        let is_emulator = renderer_name.starts_with("Android Emulator");
        let avoid_tex_image = is_emulator;
        let mut gl_version = [0; 2];
        unsafe {
            gl.get_integer_v(gl::MAJOR_VERSION, &mut gl_version[0..1]);
            gl.get_integer_v(gl::MINOR_VERSION, &mut gl_version[1..2]);
        }
        info!("GL context {:?} {}.{}", gl.get_type(), gl_version[0], gl_version[1]);

        let is_macos_native_gl = cfg!(target_os = "macos") &&
            !renderer_name.starts_with("ANGLE");

        // We block texture storage on mac with native GL because it doesn't support BGRA
        let supports_texture_storage = allow_texture_storage_support && !is_macos_native_gl &&
            match gl.get_type() {
                gl::GlType::Gl => supports_extension(&extensions, "GL_ARB_texture_storage"),
                gl::GlType::Gles => true,
            };

        // The GL_EXT_texture_format_BGRA8888 extension allows us to use BGRA as an internal format
        // with glTexImage on GLES. However, we can only use BGRA8 as an internal format for
        // glTexStorage when GL_EXT_texture_storage is also explicitly supported. This is because
        // glTexStorage was added in GLES 3, but GL_EXT_texture_format_BGRA8888 was written against
        // GLES 2 and GL_EXT_texture_storage.
        // To complicate things even further, some Intel devices claim to support both extensions
        // but in practice do not allow BGRA to be used with glTexStorage.
        let supports_gles_bgra = supports_extension(&extensions, "GL_EXT_texture_format_BGRA8888");
        let supports_texture_storage_with_gles_bgra = supports_gles_bgra
            && supports_extension(&extensions, "GL_EXT_texture_storage")
            && !renderer_name.starts_with("Intel(R) HD Graphics for BayTrail")
            && !renderer_name.starts_with("Intel(R) HD Graphics for Atom(TM) x5/x7");

        let supports_texture_swizzle = allow_texture_swizzling &&
            match gl.get_type() {
                // see https://www.g-truc.net/post-0734.html
                gl::GlType::Gl => gl_version >= [3, 3] ||
                    supports_extension(&extensions, "GL_ARB_texture_swizzle"),
                gl::GlType::Gles => true,
            };

        // Reading pixels back as BGRA with glReadPixels is always supported in
        // desktop GL, but on GLES it requires the GL_EXT_read_format_bgra
        // extension. When it is missing we read as RGBA and swap the red and
        // blue channels on the CPU instead.
        let supports_bgra_read = match gl.get_type() {
            gl::GlType::Gl => true,
            gl::GlType::Gles => supports_extension(&extensions, "GL_EXT_read_format_bgra"),
        };

        let (color_formats, bgra_formats, bgra_pixel_type, bgra8_sampling_swizzle, texture_storage_usage) = match gl.get_type() {
            // There is `glTexStorage`, use it and expect RGBA on the input.
            gl::GlType::Gl if supports_texture_storage && supports_texture_swizzle => (
                TextureFormatPair::from(ImageFormat::RGBA8),
                TextureFormatPair { internal: gl::RGBA8, external: gl::RGBA },
                gl::UNSIGNED_BYTE,
                Swizzle::Bgra, // pretend it's RGBA, rely on swizzling
                TexStorageUsage::Always
            ),
            // There is no `glTexStorage`, upload as `glTexImage` with BGRA input.
            gl::GlType::Gl => (
                TextureFormatPair { internal: ImageFormat::BGRA8, external: ImageFormat::BGRA8 },
                TextureFormatPair { internal: gl::RGBA, external: gl::BGRA },
                gl::UNSIGNED_INT_8_8_8_8_REV,
                Swizzle::Rgba, // converted on uploads by the driver, no swizzling needed
                TexStorageUsage::Never
            ),
            // glTexStorage is always supported in GLES 3, but because the GL_EXT_texture_storage
            // extension is supported we can use glTexStorage with BGRA8 as the internal format.
            // Prefer BGRA textures over RGBA.
            gl::GlType::Gles if supports_texture_storage_with_gles_bgra => (
                TextureFormatPair::from(ImageFormat::BGRA8),
                TextureFormatPair { internal: gl::BGRA8_EXT, external: gl::BGRA_EXT },
                gl::UNSIGNED_BYTE,
                Swizzle::Rgba, // no conversion needed
                TexStorageUsage::Always,
            ),
            // BGRA is not supported as an internal format with glTexStorage, therefore we will
            // use RGBA textures instead and pretend BGRA data is RGBA when uploading.
            // The swizzling will happen at the texture unit.
            gl::GlType::Gles if supports_texture_swizzle => (
                TextureFormatPair::from(ImageFormat::RGBA8),
                TextureFormatPair { internal: gl::RGBA8, external: gl::RGBA },
                gl::UNSIGNED_BYTE,
                Swizzle::Bgra, // pretend it's RGBA, rely on swizzling
                TexStorageUsage::Always,
            ),
            // BGRA is not supported as an internal format with glTexStorage, and we cannot use
            // swizzling either. Therefore prefer BGRA textures over RGBA, but use glTexImage
            // to initialize BGRA textures. glTexStorage can still be used for other formats.
            gl::GlType::Gles if supports_gles_bgra && !avoid_tex_image => (
                TextureFormatPair::from(ImageFormat::BGRA8),
                TextureFormatPair::from(gl::BGRA_EXT),
                gl::UNSIGNED_BYTE,
                Swizzle::Rgba, // no conversion needed
                TexStorageUsage::NonBGRA8,
            ),
            // Neither BGRA or swizzling are supported. GLES does not allow format conversion
            // during upload so we must use RGBA textures and pretend BGRA data is RGBA when
            // uploading. Images may be rendered incorrectly as a result.
            gl::GlType::Gles => {
                warn!("Neither BGRA or texture swizzling are supported. Images may be rendered incorrectly.");
                (
                    TextureFormatPair::from(ImageFormat::RGBA8),
                    TextureFormatPair { internal: gl::RGBA8, external: gl::RGBA },
                    gl::UNSIGNED_BYTE,
                    Swizzle::Rgba,
                    TexStorageUsage::Always,
                )
            }
        };

        let is_software_webrender = renderer_name.starts_with("Software WebRender");
        let upload_method = if is_software_webrender {
            // Uploads in SWGL generally reduce to simple memory copies.
            UploadMethod::Immediate
        } else {
            upload_method
        };
        // Prefer 24-bit depth format. While 16-bit depth also works, it may exhaust depth ids easily.
        let depth_format = gl::DEPTH_COMPONENT24;

        info!("GL texture cache {:?}, bgra {:?} swizzle {:?}, texture storage {:?}, depth {:?}",
            color_formats, bgra_formats, bgra8_sampling_swizzle, texture_storage_usage, depth_format);

        // On Mali-T devices glCopyImageSubData appears to stall the pipeline until any pending
        // renders to the source texture have completed. On Mali-G, it has been observed to
        // indefinitely hang in some circumstances. Using an alternative such as glBlitFramebuffer
        // is preferable on such devices, so pretend we don't support glCopyImageSubData.
        // See bugs 1669494 and 1677757.
        let supports_copy_image_sub_data = if renderer_name.starts_with("Mali") {
            false
        } else {
            supports_extension(&extensions, "GL_EXT_copy_image") ||
            supports_extension(&extensions, "GL_ARB_copy_image")
        };

        let is_adreno = renderer_name.starts_with("Adreno");

        // There appears to be a driver bug on older versions of the Adreno
        // driver which prevents usage of persistenly mapped buffers.
        // See bugs 1678585 and 1683936.
        // TODO: only disable feature for affected driver versions.
        let supports_buffer_storage = if is_adreno {
            false
        } else {
            supports_extension(&extensions, "GL_EXT_buffer_storage") ||
            supports_extension(&extensions, "GL_ARB_buffer_storage")
        };

        // KHR_blend_equation_advanced renders incorrectly on Adreno
        // devices. This has only been confirmed up to Adreno 5xx, and has been
        // fixed for Android 9, so this condition could be made more specific.
        let supports_advanced_blend_equation =
            supports_extension(&extensions, "GL_KHR_blend_equation_advanced") &&
            !is_adreno;
        let supports_advanced_blend_equation_coherent =
            supports_extension(&extensions, "GL_KHR_blend_equation_advanced_coherent");

        let supports_dual_source_blending = match gl.get_type() {
            gl::GlType::Gl => supports_extension(&extensions,"GL_ARB_blend_func_extended") &&
                supports_extension(&extensions,"GL_ARB_explicit_attrib_location"),
            gl::GlType::Gles => supports_extension(&extensions,"GL_EXT_blend_func_extended"),
        };

        // Software webrender relies on the unoptimized shader source.
        let use_optimized_shaders = use_optimized_shaders && !is_software_webrender;

        // On the android emulator, and possibly some Mali devices, glShaderSource
        // can crash if the source strings are not null-terminated.
        // See bug 1591945 and bug 1799722.
        // Likewise on Lenovo devices with Adreno 750 GPUs we have seen glCompileShader
        // failures and subsequent crashes due to glGetShaderInfoLog returning invalid
        // UTF-8. See bug 2014925.
        let requires_null_terminated_shader_source = is_emulator || renderer_name == "Mali-T628"
            || renderer_name == "Mali-T720" || renderer_name == "Mali-T760"
            || renderer_name == "Mali-G57" || renderer_name == "Adreno (TM) 750";

        // The android emulator gets confused if you don't explicitly unbind any texture
        // from GL_TEXTURE_EXTERNAL_OES before binding another to GL_TEXTURE_2D. See bug 1636085.
        let requires_texture_external_unbind = is_emulator;

        let is_windows_angle = cfg!(target_os = "windows")
            && renderer_name.starts_with("ANGLE");
        let is_adreno_3xx = renderer_name.starts_with("Adreno (TM) 3");

        // Some GPUs require the stride of the data during texture uploads to be
        // aligned to certain requirements, either for correctness or performance
        // reasons.
        let required_transfer_stride = if is_adreno_3xx {
            // On Adreno 3xx, alignments of < 128 bytes can result in corrupted
            // glyphs. See bug 1696039.
            StrideAlignment::Bytes(NonZeroUsize::new(128).unwrap())
        } else if is_adreno {
            // On later Adreno devices it must be a multiple of 64 *pixels* to
            // hit the fast path, meaning value in bytes varies with the texture
            // format. This is purely an optimization.
            StrideAlignment::Pixels(NonZeroUsize::new(64).unwrap())
        } else if is_macos_native_gl {
            // On AMD Mac, it must always be a multiple of 256 bytes. We apply
            // this restriction to all GPUs when using native GL to handle
            // switching.
            StrideAlignment::Bytes(NonZeroUsize::new(256).unwrap())
        } else if is_windows_angle {
            // On ANGLE-on-D3D, PBO texture uploads get incorrectly truncated
            // if the stride is greater than the width * bpp.
            StrideAlignment::Bytes(NonZeroUsize::new(1).unwrap())
        } else {
            // Other platforms may have similar requirements and should be added
            // here. The default value should be 4 bytes.
            StrideAlignment::Bytes(NonZeroUsize::new(4).unwrap())
        };

        // On AMD Macs there is a driver bug which causes some texture uploads
        // from a non-zero offset within a PBO to fail. See bug 1603783. We
        // apply this restriction to all GPUs when using native GL to handle
        // switching.
        let supports_nonzero_pbo_offsets = !is_macos_native_gl;

        // We have encountered several issues when only partially updating render targets on a
        // variety of Mali GPUs. As a precaution avoid doing so on all Midgard and Bifrost GPUs.
        // Valhall (eg Mali-Gx7 onwards) appears to be unaffected. See bug 1691955, bug 1558374,
        // and bug 1663355.
        // We have Additionally encountered issues on PowerVR D-Series. See bug 2005312.
        let supports_render_target_partial_update = !is_mali_midgard(&renderer_name)
            && !is_mali_bifrost(&renderer_name)
            && !renderer_name.starts_with("PowerVR D-Series");

        let supports_shader_storage_object = match gl.get_type() {
            // see https://www.g-truc.net/post-0734.html
            gl::GlType::Gl => supports_extension(&extensions, "GL_ARB_shader_storage_buffer_object"),
            gl::GlType::Gles => gl_version >= [3, 1],
        };

        // SWGL uses swgl_clipMask() instead of implementing clip-masking in shaders.
        // This allows certain shaders to potentially bypass the more expensive alpha-
        // pass variants if they know the alpha-pass was only required to deal with
        // clip-masking.
        let uses_native_clip_mask = is_software_webrender;

        // SWGL uses swgl_antiAlias() instead of implementing anti-aliasing in shaders.
        // As above, this allows bypassing certain alpha-pass variants.
        let uses_native_antialiasing = is_software_webrender;

        // If running on android with a mesa driver (eg intel chromebooks), parse the mesa version.
        let mut android_mesa_version = None;
        if cfg!(target_os = "android") && renderer_name.starts_with("Mesa") {
            if let Some((_, mesa_version)) = version_string.split_once("Mesa ") {
                if let Some((major_str, _)) = mesa_version.split_once(".") {
                    if let Ok(major) = major_str.parse::<i32>() {
                        android_mesa_version = Some(major);
                    }
                }
            }
        }

        // If the device supports OES_EGL_image_external_essl3 we can use it to render
        // external images. If not, we must use the ESSL 1.0 OES_EGL_image_external
        // extension instead.
        // Mesa versions prior to 20.0 do not implement textureSize(samplerExternalOES),
        // so we must use the fallback path.
        let supports_image_external_essl3 = match android_mesa_version {
            Some(major) if major < 20 => false,
            _ => supports_extension(&extensions, "GL_OES_EGL_image_external_essl3"),
        };

        let (supports_texture_rect, supports_texture_external) = match gl.get_type() {
            gl::GlType::Gl => (true, false),
            gl::GlType::Gles => (false, true),
        };
        let supports_texture_external_bt709 =
            supports_texture_external && supports_extension(&extensions, "GL_EXT_YUV_target");

        // On Windows a GLES context is an ANGLE context, whose default framebuffer
        // is a D3D surface with a top-left origin.
        let readback_rows_top_down = cfg!(windows) && gl.get_type() == gl::GlType::Gles;

        let mut requires_batched_texture_uploads = None;
        if is_software_webrender {
            // No benefit to batching texture uploads with swgl.
            requires_batched_texture_uploads = Some(false);
        } else if renderer_name.starts_with("Mali-G") {
            // On Mali-Gxx the driver really struggles with many small texture uploads,
            // and handles fewer, larger uploads better.
            requires_batched_texture_uploads = Some(true);
        }

        // On Mali-Txxx devices we have observed crashes during draw calls when rendering
        // to an alpha target immediately after using glClear to clear regions of it.
        // Using a shader to clear the regions avoids the crash. See bug 1638593.
        // On Adreno 510 devices we have seen garbage being used as masks when clearing
        // alpha targets with glClear. Using quads to clear avoids this. See bug 1941154.
        let is_adreno_510 = renderer_name.starts_with("Adreno (TM) 510");
        let supports_alpha_target_clears = !is_mali_midgard(&renderer_name) && !is_adreno_510;

        // On Adreno 4xx devices with older drivers we have seen render tasks to alpha targets have
        // no effect unless the target is fully cleared prior to rendering. See bug 1714227.
        let is_adreno_4xx = renderer_name.starts_with("Adreno (TM) 4");
        let requires_alpha_target_full_clear = is_adreno_4xx;

        // Testing on Intel and nVidia GPUs, as well as software webrender, showed large performance
        // wins applying a scissor rect when clearing render targets. Assume this is the best
        // default. On mobile GPUs, however, it can be much more efficient to clear the entire
        // render target. For now, enable the scissor everywhere except Android hardware
        // webrender. We can tweak this further if needs be.
        let prefers_clear_scissor = !cfg!(target_os = "android") || is_software_webrender;

        let mut supports_render_target_invalidate = true;

        // On PowerVR Rogue devices we have seen that invalidating render targets after we are done
        // with them can incorrectly cause pending renders to be written to different targets
        // instead. See bug 1719345.
        let is_powervr_rogue = renderer_name.starts_with("PowerVR Rogue");
        if is_powervr_rogue {
            supports_render_target_invalidate = false;
        }

        // On Mali Valhall devices with a driver version v1.r36p0 we have seen that invalidating
        // render targets can result in image corruption, perhaps due to subsequent reuses of the
        // render target not correctly reinitializing them to a valid state. See bug 1787520.
        if is_mali_valhall(&renderer_name) {
            match parse_mali_version(&version_string) {
                Some(version) if version >= (1, 36, 0) => supports_render_target_invalidate = false,
                _ => {}
            }
        }

        // On Linux we we have seen uploads to R8 format textures result in
        // corruption on some AMD cards.
        // See https://bugzilla.mozilla.org/show_bug.cgi?id=1687554#c13
        let supports_r8_texture_upload = if cfg!(target_os = "linux")
            && renderer_name.starts_with("AMD Radeon RX")
        {
            false
        } else {
            true
        };

        let supports_qcom_tiled_rendering = if is_adreno && version_string.contains("V@0490") {
            // We have encountered rendering errors on a variety of Adreno GPUs specifically on
            // driver version V@0490, so block this extension on that driver version. See bug 1828248.
            false
        } else if renderer_name == "Adreno (TM) 308" {
            // And specifically on Areno 308 GPUs we have encountered rendering errors on driver
            // versions V@331, V@415, and V@0502. We presume this therefore affects all driver
            // versions. See bug 1843749 and bug 1847319.
            false
        } else {
            supports_extension(&extensions, "GL_QCOM_tiled_rendering")
        };

        // On some Adreno 3xx devices the vertex array object must be unbound and rebound after
        // an attached buffer has been orphaned.
        let requires_vao_rebind_after_orphaning = is_adreno_3xx;

        let supports_base_instance = !is_software_webrender && match gl.get_type() {
            gl::GlType::Gl => {
                gl_version >= [4, 2] || supports_extension(&extensions, "GL_ARB_base_instance")
            }
            gl::GlType::Gles => supports_extension(&extensions, "GL_EXT_base_instance"),
        };

        GlDevice {
            gl,
            base_gl: None,
            crash_annotator,
            annotate_draw_call_crashes: false,
            resource_override_path,
            use_optimized_shaders,
            upload_method,
            use_batched_texture_uploads: requires_batched_texture_uploads.unwrap_or(false),
            use_draw_calls_for_texture_copy: false,
            batched_upload_threshold,

            inside_frame: false,

            capabilities: Capabilities {
                supports_multisampling: false, //TODO
                supports_copy_image_sub_data,
                supports_buffer_storage,
                supports_advanced_blend_equation,
                supports_advanced_blend_equation_coherent,
                supports_dual_source_blending,
                supports_khr_debug,
                supports_texture_swizzle,
                supports_nonzero_pbo_offsets,
                supports_texture_usage,
                supports_render_target_partial_update,
                supports_shader_storage_object,
                requires_batched_texture_uploads,
                supports_alpha_target_clears,
                requires_alpha_target_full_clear,
                prefers_clear_scissor,
                supports_render_target_invalidate,
                supports_r8_texture_upload,
                supports_qcom_tiled_rendering,
                uses_native_clip_mask,
                uses_native_antialiasing,
                supports_image_external_essl3,
                supports_texture_rect,
                supports_texture_external,
                supports_texture_external_bt709,
                readback_rows_top_down,
                requires_vao_rebind_after_orphaning,
                supports_bgra_read,
                supports_base_instance,
                renderer_name,
            },

            color_formats,
            bgra_formats,
            bgra_pixel_type,
            swizzle_settings: SwizzleSettings {
                bgra8_sampling_swizzle,
            },
            depth_format,

            depth_targets: FastHashMap::default(),

            bound_textures: [0; 16],
            bound_program: 0,
            bound_program_name: Rc::new(std::ffi::CString::new("").unwrap()),
            bound_vao: 0,
            bound_read_fbo: (FBOId(0), DeviceIntPoint::zero()),
            current_render_pass: None,
            scratch_read_fbo: None,
            bound_draw_fbo: FBOId(0),
            default_read_fbo: FBOId(0),
            default_draw_fbo: FBOId(0),

            depth_available: true,

            max_texture_size,
            cached_programs,
            frame_id: GpuFrameId(0),
            extensions,
            texture_storage_usage,
            requires_null_terminated_shader_source,
            requires_texture_external_unbind,
            is_software_webrender,
            required_transfer_stride,
            dump_shader_source,
            #[cfg(feature = "debugger")]
            shader_source_overrides: FastHashMap::default(),
            #[cfg(feature = "debugger")]
            shader_include_closures: RefCell::new(FastHashMap::default()),
            surface_origin_is_top_left,

            gl_state: GlRenderStateCache::default(),

            textures_created: 0,
            textures_deleted: 0,

            initialize_color_targets_with_pink: false,
        }
    }

    fn gl(&self) -> &dyn gl::Gl {
        &*self.gl
    }

    fn depth_bits(&self) -> i32 {
        match self.depth_format {
            gl::DEPTH_COMPONENT16 => 16,
            gl::DEPTH_COMPONENT24 => 24,
            _ => panic!("Unknown depth format {:?}", self.depth_format),
        }
    }

    fn compile_shader(
        &self,
        name: &str,
        shader_type: gl::GLenum,
        source: &String,
        source_map: Option<&ShaderSourceMap>,
    ) -> Result<gl::GLuint, ShaderError> {
        debug!("compile {}", name);
        let id = self.gl.create_shader(shader_type);

        let mut new_source = Cow::from(source.as_str());
        // Ensure the source strings we pass to glShaderSource are
        // null-terminated on buggy platforms.
        if self.requires_null_terminated_shader_source {
            new_source.to_mut().push('\0');
        }

        self.gl.shader_source(id, &[new_source.as_bytes()]);
        self.gl.compile_shader(id);
        let log = self.gl.get_shader_info_log(id);
        let mut status = [0];
        unsafe {
            self.gl.get_shader_iv(id, gl::COMPILE_STATUS, &mut status);
        }
        if status[0] == 0 {
            let type_str = match shader_type {
                gl::VERTEX_SHADER => "vertex",
                gl::FRAGMENT_SHADER => "fragment",
                _ => panic!("Unexpected shader type {:x}", shader_type),
            };
            let diagnostics = match source_map {
                Some(source_map) => source_map.map_log(&log),
                None => Vec::new(),
            };
            error!("Failed to compile {} shader: {}", type_str, name);
            if diagnostics.is_empty() {
                error!("{}", log);
            } else {
                for diagnostic in &diagnostics {
                    error!("{}", diagnostic);
                }
            }
            Err(ShaderError::Compilation(name.to_string(), log, diagnostics))
        } else {
            if !log.is_empty() {
                warn!("Warnings detected on shader: {}\n{}", name, log);
            }
            Ok(id)
        }
    }

    fn bind_texture_impl(
        &mut self,
        slot: TextureSlot,
        id: gl::GLuint,
        target: gl::GLenum,
        set_swizzle: Option<Swizzle>,
        image_rendering: Option<ImageRendering>,
    ) {
        debug_assert!(self.inside_frame);

        if self.bound_textures[slot.0] != id || set_swizzle.is_some() || image_rendering.is_some() {
            self.gl.active_texture(gl::TEXTURE0 + slot.0 as gl::GLuint);
            // The android emulator gets confused if you don't explicitly unbind any texture
            // from GL_TEXTURE_EXTERNAL_OES before binding to GL_TEXTURE_2D. See bug 1636085.
            if target == gl::TEXTURE_2D && self.requires_texture_external_unbind {
                self.gl.bind_texture(gl::TEXTURE_EXTERNAL_OES, 0);
            }
            self.gl.bind_texture(target, id);
            if let Some(swizzle) = set_swizzle {
                if self.capabilities.supports_texture_swizzle {
                    let components = match swizzle {
                        Swizzle::Rgba => [gl::RED, gl::GREEN, gl::BLUE, gl::ALPHA],
                        Swizzle::Bgra => [gl::BLUE, gl::GREEN, gl::RED, gl::ALPHA],
                    };
                    self.gl.tex_parameter_i(target, gl::TEXTURE_SWIZZLE_R, components[0] as i32);
                    self.gl.tex_parameter_i(target, gl::TEXTURE_SWIZZLE_G, components[1] as i32);
                    self.gl.tex_parameter_i(target, gl::TEXTURE_SWIZZLE_B, components[2] as i32);
                    self.gl.tex_parameter_i(target, gl::TEXTURE_SWIZZLE_A, components[3] as i32);
                } else {
                    debug_assert_eq!(swizzle, Swizzle::default());
                }
            }
            if let Some(image_rendering) = image_rendering {
                let filter = match image_rendering {
                    ImageRendering::Auto | ImageRendering::CrispEdges => gl::LINEAR,
                    ImageRendering::Pixelated => gl::NEAREST,
                };
                self.gl.tex_parameter_i(target, gl::TEXTURE_MIN_FILTER, filter as i32);
                self.gl.tex_parameter_i(target, gl::TEXTURE_MAG_FILTER, filter as i32);
            }
            self.gl.active_texture(gl::TEXTURE0);
            self.bound_textures[slot.0] = id;
        }
    }

    fn bind_read_target_impl(
        &mut self,
        fbo_id: FBOId,
        offset: DeviceIntPoint,
    ) {
        debug_assert!(self.inside_frame);

        if self.bound_read_fbo != (fbo_id, offset) {
            fbo_id.bind(self.gl(), FBOTarget::Read);
        }

        self.bound_read_fbo = (fbo_id, offset);
    }

    fn bind_read_target(&mut self, target: ReadTarget) {
        let fbo_id = match target {
            ReadTarget::Default => self.default_read_fbo,
            ReadTarget::Texture { fbo_id } => fbo_id,
            ReadTarget::NativeSurface { fbo_id, .. } => fbo_id,
        };

        self.bind_read_target_impl(fbo_id, target.offset())
    }

    fn bind_draw_target_impl(&mut self, fbo_id: FBOId) {
        debug_assert!(self.inside_frame);

        if self.bound_draw_fbo != fbo_id {
            self.bound_draw_fbo = fbo_id;
            fbo_id.bind(self.gl(), FBOTarget::Draw);
        }
    }

    fn reset_draw_target(&mut self) {
        let fbo = self.default_draw_fbo;
        self.bind_draw_target_impl(fbo);
        self.depth_available = true;
    }

    fn bind_draw_target(
        &mut self,
        target: DrawTarget,
    ) {
        let (fbo_id, rect, depth_available) = match target {
            DrawTarget::Default { rect, .. } => {
                (self.default_draw_fbo, rect, false)
            }
            DrawTarget::Texture { dimensions, fbo_id, with_depth, .. } => {
                let rect = FramebufferIntRect::from_size(
                    device_size_as_framebuffer_size(dimensions),
                );
                (fbo_id, rect, with_depth)
            },
            DrawTarget::NativeSurface { handle, offset, dimensions, .. } => {
                (
                    FBOId(handle.0 as u32),
                    device_rect_as_framebuffer_rect(&DeviceIntRect::from_origin_and_size(offset, dimensions)),
                    true
                )
            }
        };

        self.depth_available = depth_available;
        self.bind_draw_target_impl(fbo_id);
        self.gl.viewport(
            rect.min.x,
            rect.min.y,
            rect.width(),
            rect.height(),
        );
    }

    /// Creates an unbound FBO object. Additional attachment API calls are
    /// required to make it complete.
    fn create_fbo(&mut self) -> FBOId {
        FBOId(self.gl.gen_framebuffers(1)[0])
    }

    fn delete_fbo(&mut self, fbo: FBOId) {
        self.gl.delete_framebuffers(&[fbo.0]);
    }

    fn bind_external_draw_target(&mut self, fbo_id: FBOId) {
        debug_assert!(self.inside_frame);

        if self.bound_draw_fbo != fbo_id {
            self.bound_draw_fbo = fbo_id;
            fbo_id.bind(self.gl(), FBOTarget::Draw);
        }
    }

    fn set_texture_parameters(&mut self, target: gl::GLuint, filter: TextureFilter) {
        let mag_filter = match filter {
            TextureFilter::Nearest => gl::NEAREST,
            TextureFilter::Linear | TextureFilter::Trilinear => gl::LINEAR,
        };

        let min_filter = match filter {
            TextureFilter::Nearest => gl::NEAREST,
            TextureFilter::Linear => gl::LINEAR,
            TextureFilter::Trilinear => gl::LINEAR_MIPMAP_LINEAR,
        };

        self.gl
            .tex_parameter_i(target, gl::TEXTURE_MAG_FILTER, mag_filter as gl::GLint);
        self.gl
            .tex_parameter_i(target, gl::TEXTURE_MIN_FILTER, min_filter as gl::GLint);

        self.gl
            .tex_parameter_i(target, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as gl::GLint);
        self.gl
            .tex_parameter_i(target, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as gl::GLint);
    }

    /// Notifies the device that the contents of the current framebuffer's depth
    /// attachment is no longer needed. Unlike invalidate_render_target, this can
    /// be called even when the contents of the colour attachment is still required.
    /// This should be called before unbinding the framebuffer at the end of a pass,
    /// to allow tiled GPUs to avoid writing the contents back to memory.
    fn invalidate_depth_target(&mut self) {
        assert!(self.depth_available);
        let attachments = if self.bound_draw_fbo == self.default_draw_fbo {
            &[gl::DEPTH] as &[gl::GLenum]
        } else {
            &[gl::DEPTH_ATTACHMENT] as &[gl::GLenum]
        };
        self.gl.invalidate_framebuffer(gl::DRAW_FRAMEBUFFER, attachments);
    }

    fn init_fbos(&mut self, texture: &mut Texture, with_depth: bool) {
        let (fbo, depth_rb) = if with_depth {
            let depth_target = self.acquire_depth_target(texture.get_dimensions());
            (&mut texture.fbo_with_depth, Some(depth_target))
        } else {
            (&mut texture.fbo, None)
        };

        // Generate the FBOs.
        assert!(fbo.is_none());
        let fbo_id = FBOId(*self.gl.gen_framebuffers(1).first().unwrap());
        *fbo = Some(fbo_id);

        // Bind the FBOs.
        let original_bound_fbo = self.bound_draw_fbo;

        self.bind_external_draw_target(fbo_id);

        self.gl.framebuffer_texture_2d(
            gl::DRAW_FRAMEBUFFER,
            gl::COLOR_ATTACHMENT0,
            get_gl_target(texture.target),
            texture.id,
            0,
        );

        if let Some(depth_rb) = depth_rb {
            self.gl.framebuffer_renderbuffer(
                gl::DRAW_FRAMEBUFFER,
                gl::DEPTH_ATTACHMENT,
                gl::RENDERBUFFER,
                depth_rb.0,
            );
        }

        debug_assert_eq!(
            self.gl.check_frame_buffer_status(gl::DRAW_FRAMEBUFFER),
            gl::FRAMEBUFFER_COMPLETE,
            "Incomplete framebuffer",
        );

        self.bind_external_draw_target(original_bound_fbo);
    }

    fn acquire_depth_target(&mut self, dimensions: DeviceIntSize) -> RBOId {
        let gl = &self.gl;
        let depth_format = self.depth_format;
        let target = self.depth_targets.entry(dimensions).or_insert_with(|| {
            let renderbuffer_ids = gl.gen_renderbuffers(1);
            let depth_rb = renderbuffer_ids[0];
            gl.bind_renderbuffer(gl::RENDERBUFFER, depth_rb);
            gl.renderbuffer_storage(
                gl::RENDERBUFFER,
                depth_format,
                dimensions.width as _,
                dimensions.height as _,
            );
            SharedDepthTarget {
                rbo_id: RBOId(depth_rb),
                refcount: 0,
            }
        });
        target.refcount += 1;
        target.rbo_id
    }

    fn release_depth_target(&mut self, dimensions: DeviceIntSize) {
        let mut entry = match self.depth_targets.entry(dimensions) {
            Entry::Occupied(x) => x,
            Entry::Vacant(..) => panic!("Releasing unknown depth target"),
        };
        debug_assert!(entry.get().refcount != 0);
        entry.get_mut().refcount -= 1;
        if entry.get().refcount == 0 {
            let (_, target) = entry.remove_entry();
            self.gl.delete_renderbuffers(&[target.rbo_id.0]);
        }
    }

    /// Perform a blit between self.bound_read_fbo and self.bound_draw_fbo.
    fn blit_render_target_impl(
        &mut self,
        src_rect: FramebufferIntRect,
        dest_rect: FramebufferIntRect,
        filter: TextureFilter,
    ) {
        debug_assert!(self.inside_frame);

        let filter = match filter {
            TextureFilter::Nearest => gl::NEAREST,
            TextureFilter::Linear | TextureFilter::Trilinear => gl::LINEAR,
        };

        let src_x0 = src_rect.min.x + self.bound_read_fbo.1.x;
        let src_y0 = src_rect.min.y + self.bound_read_fbo.1.y;

        self.gl.blit_framebuffer(
            src_x0,
            src_y0,
            src_x0 + src_rect.width(),
            src_y0 + src_rect.height(),
            dest_rect.min.x,
            dest_rect.min.y,
            dest_rect.max.x,
            dest_rect.max.y,
            gl::COLOR_BUFFER_BIT,
            filter,
        );
    }

    /// Whether any file `base_filename` pulls in, including itself, has an
    /// override installed.
    #[cfg(feature = "debugger")]
    fn has_shader_source_override_for(&self, base_filename: &str) -> bool {
        // The common case is no overrides at all, in which case there is no
        // need to walk the include graph.
        if self.shader_source_overrides.is_empty() {
            return false;
        }

        if self.shader_source_overrides.contains_key(base_filename) {
            return true;
        }

        self.shader_include_closure(base_filename)
            .iter()
            .any(|file| self.shader_source_overrides.contains_key(file))
    }

    /// Nothing can install an override without the debugger, so no shader is
    /// ever built from one.
    #[cfg(not(feature = "debugger"))]
    fn has_shader_source_override_for(&self, _base_filename: &str) -> bool {
        false
    }

    fn build_shader_string<F: FnMut(&str)>(
        &self,
        features: &[&'static str],
        kind: ShaderKind,
        base_filename: &str,
        output: F,
    ) -> ShaderSourceMap {
        let mut source_map = ShaderSourceMap::new();
        do_build_shader_string(
            get_shader_version(&*self.gl),
            features,
            kind,
            base_filename,
            &mut source_map,
            &|f| self.get_shader_source(f),
            output,
        );
        source_map
    }

    /// Issues one texture update. `source` is an offset into the bound pixel
    /// unpack buffer, or a client memory address when none is bound.
    fn upload_chunk(
        &mut self,
        texture: &Texture,
        rect: DeviceIntRect,
        stride: Option<i32>,
        format_override: Option<ImageFormat>,
        source: usize,
    ) {
        self.bind_texture(DEFAULT_TEXTURE, texture, Swizzle::default());

        let format = format_override.unwrap_or(texture.format);
        let (gl_format, bpp, data_type) = match format {
            ImageFormat::R8 => (gl::RED, 1, gl::UNSIGNED_BYTE),
            ImageFormat::R16 => (gl::RED, 2, gl::UNSIGNED_SHORT),
            ImageFormat::BGRA8 => (self.bgra_formats.external, 4, self.bgra_pixel_type),
            ImageFormat::RGBA8 => (gl::RGBA, 4, gl::UNSIGNED_BYTE),
            ImageFormat::RG8 => (gl::RG, 2, gl::UNSIGNED_BYTE),
            ImageFormat::RG16 => (gl::RG, 4, gl::UNSIGNED_SHORT),
            ImageFormat::RGBAF32 => (gl::RGBA, 16, gl::FLOAT),
            ImageFormat::RGBAI32 => (gl::RGBA_INTEGER, 16, gl::INT),
        };

        let row_length = match stride {
            Some(value) => value / bpp,
            None => texture.size.width,
        };

        if stride.is_some() {
            self.gl.pixel_store_i(
                gl::UNPACK_ROW_LENGTH,
                row_length as _,
            );
        }

        let pos = rect.min;
        let size = rect.size();
        let gl_target = get_gl_target(texture.target);

        self.gl.tex_sub_image_2d_pbo(
            gl_target,
            0,
            pos.x as _,
            pos.y as _,
            size.width as _,
            size.height as _,
            gl_format,
            data_type,
            source,
        );

        // If using tri-linear filtering, build the mip-map chain for this texture.
        if texture.filter == TextureFilter::Trilinear {
            self.gl.generate_mipmap(gl_target);
        }

        // Reset row length to 0, otherwise the stride would apply to all texture uploads.
        if stride.is_some() {
            self.gl.pixel_store_i(gl::UNPACK_ROW_LENGTH, 0 as _);
        }
    }

    /// Attaches the provided texture to the current Read FBO binding.
    fn attach_read_texture_raw(&mut self, texture_id: gl::GLuint, target: gl::GLuint) {
        self.gl.framebuffer_texture_2d(
            gl::READ_FRAMEBUFFER,
            gl::COLOR_ATTACHMENT0,
            target,
            texture_id,
            0,
        )
    }

    /// Binds the device-owned scratch read framebuffer, so that a texture can
    /// be attached to it and read back.
    fn bind_scratch_read_target(&mut self) {
        let fbo = match self.scratch_read_fbo {
            Some(fbo) => fbo,
            None => {
                let fbo = self.create_fbo();
                self.scratch_read_fbo = Some(fbo);
                fbo
            }
        };
        self.bind_read_target_impl(fbo, DeviceIntPoint::zero());
    }

    fn bind_vao_impl(&mut self, id: gl::GLuint) {
        debug_assert!(self.inside_frame);

        if self.bound_vao != id {
            self.bound_vao = id;
            self.gl.bind_vertex_array(id);
        }
    }

    fn create_vao_with_vbos(
        &mut self,
        descriptor: &VertexDescriptor,
        main_vbo_id: VBOId,
        instance_vbo_id: VBOId,
        instance_divisor: u32,
        ibo_id: IBOId,
        owns_vertices_and_indices: bool,
        owns_instances: bool,
    ) -> VAO {
        let instance_stride = descriptor.instance_stride() as usize;
        let vao_id = self.gl.gen_vertex_arrays(1)[0];

        self.bind_vao_impl(vao_id);

        descriptor.bind(self.gl(), main_vbo_id, instance_vbo_id, instance_divisor);
        ibo_id.bind(self.gl()); // force it to be a part of VAO

        VAO {
            id: vao_id,
            ibo_id,
            main_vbo_id,
            instance_vbo_id,
            instance_stride,
            instance_divisor,
            owns_vertices_and_indices,
            owns_instances,
        }
    }

    fn update_vbo_data(
        &mut self,
        vbo: VBOId,
        data: &[u8],
        usage_hint: VertexUsageHint,
    ) {
        debug_assert!(self.inside_frame);

        vbo.bind(self.gl());
        gl::buffer_data(self.gl(), gl::ARRAY_BUFFER, data, usage_hint.to_gl());
    }

    fn clear_target_impl(
        &mut self,
        color: Option<[f32; 4]>,
        depth: Option<f32>,
        rect: Option<FramebufferIntRect>,
    ) {
        let mut clear_bits = 0;

        if let Some(color) = color {
            if self.gl_state.color_write != Some(true) {
                self.gl.color_mask(true, true, true, true);
                self.gl_state.color_write = Some(true);
            }
            self.gl.clear_color(color[0], color[1], color[2], color[3]);
            clear_bits |= gl::COLOR_BUFFER_BIT;
        }

        if let Some(depth) = depth {
            if self.gl_state.depth_write != Some(true) {
                self.gl.depth_mask(true);
                self.gl_state.depth_write = Some(true);
            }
            self.gl.clear_depth(depth as f64);
            clear_bits |= gl::DEPTH_BUFFER_BIT;
        }

        if clear_bits != 0 {
            match rect {
                Some(rect) => {
                    self.gl.enable(gl::SCISSOR_TEST);
                    self.gl.scissor(
                        rect.min.x,
                        rect.min.y,
                        rect.width(),
                        rect.height(),
                    );
                    self.gl.clear(clear_bits);
                    self.gl.disable(gl::SCISSOR_TEST);
                }
                None => {
                    self.gl.clear(clear_bits);
                }
            }
        }
    }

    /// Issues the GL calls that bring the context to `state`, skipping the
    /// parts it is known to hold already.
    fn apply_render_state(&mut self, state: &RenderState) {
        if self.gl_state.blend_mode != Some(state.blend_mode) {
            self.apply_blend_mode(state.blend_mode);
            self.gl_state.blend_mode = Some(state.blend_mode);
        }

        if self.gl_state.depth_test != Some(state.depth_test) {
            match state.depth_test {
                Some(depth_func) => {
                    assert!(self.depth_available, "Enabling depth test without depth target");
                    self.gl.enable(gl::DEPTH_TEST);
                    self.gl.depth_func(depth_func.to_gl());
                }
                None => {
                    self.gl.disable(gl::DEPTH_TEST);
                }
            }
            self.gl_state.depth_test = Some(state.depth_test);
        }

        if self.gl_state.depth_write != Some(state.depth_write) {
            if state.depth_write {
                assert!(self.depth_available, "Enabling depth write without depth target");
            }
            self.gl.depth_mask(state.depth_write);
            self.gl_state.depth_write = Some(state.depth_write);
        }

        if self.gl_state.color_write != Some(state.color_write) {
            let enable = state.color_write;
            self.gl.color_mask(enable, enable, enable, enable);
            self.gl_state.color_write = Some(enable);
        }
    }

    fn set_blend(&mut self, enable: bool) {
        if enable {
            self.gl.enable(gl::BLEND);
        } else {
            self.gl.disable(gl::BLEND);
        }
    }

    fn apply_blend_mode(&mut self, mode: BlendMode) {
        if mode == BlendMode::None {
            self.set_blend(false);
            return;
        }
        self.set_blend(true);
        match mode {
            BlendMode::None => unreachable!(),
            BlendMode::Alpha => self.set_blend_mode_alpha(),
            BlendMode::PremultipliedAlpha => self.set_blend_mode_premultiplied_alpha(),
            BlendMode::PremultipliedDestOut => self.set_blend_mode_premultiplied_dest_out(),
            BlendMode::Multiply => self.set_blend_mode_multiply(),
            BlendMode::SubpixelDualSource => self.set_blend_mode_subpixel_dual_source(),
            BlendMode::Advanced(mix_mode) => self.set_blend_mode_advanced(mix_mode),
            BlendMode::Screen => self.set_blend_mode_screen(),
            BlendMode::Exclusion => self.set_blend_mode_exclusion(),
            BlendMode::PlusLighter => self.set_blend_mode_plus_lighter(),
            BlendMode::ShowOverdraw => self.set_blend_mode_show_overdraw(),
        }
    }

    fn set_blend_factors(
        &mut self,
        color: (gl::GLenum, gl::GLenum),
        alpha: (gl::GLenum, gl::GLenum),
    ) {
        self.gl.blend_equation(gl::FUNC_ADD);
        if color == alpha {
            self.gl.blend_func(color.0, color.1);
        } else {
            self.gl.blend_func_separate(color.0, color.1, alpha.0, alpha.1);
        }
    }

    fn set_blend_mode_alpha(&mut self) {
        self.set_blend_factors(
            (gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA),
            (gl::ONE, gl::ONE_MINUS_SRC_ALPHA),
        );
    }

    fn set_blend_mode_premultiplied_alpha(&mut self) {
        self.set_blend_factors(
            (gl::ONE, gl::ONE_MINUS_SRC_ALPHA),
            (gl::ONE, gl::ONE_MINUS_SRC_ALPHA),
        );
    }

    fn set_blend_mode_premultiplied_dest_out(&mut self) {
        self.set_blend_factors(
            (gl::ZERO, gl::ONE_MINUS_SRC_ALPHA),
            (gl::ZERO, gl::ONE_MINUS_SRC_ALPHA),
        );
    }

    fn set_blend_mode_multiply(&mut self) {
        self.set_blend_factors(
            (gl::ZERO, gl::SRC_COLOR),
            (gl::ZERO, gl::SRC_ALPHA),
        );
    }
    fn set_blend_mode_subpixel_dual_source(&mut self) {
        self.set_blend_factors(
            (gl::ONE, gl::ONE_MINUS_SRC1_COLOR),
            (gl::ONE, gl::ONE_MINUS_SRC1_ALPHA),
        );
    }
    fn set_blend_mode_screen(&mut self) {
        self.set_blend_factors(
            (gl::ONE, gl::ONE_MINUS_SRC_COLOR),
            (gl::ONE, gl::ONE_MINUS_SRC_ALPHA),
        );
    }
    fn set_blend_mode_plus_lighter(&mut self) {
        self.set_blend_factors(
            (gl::ONE, gl::ONE),
            (gl::ONE, gl::ONE),
        );
    }
    fn set_blend_mode_exclusion(&mut self) {
        self.set_blend_factors(
            (gl::ONE_MINUS_DST_COLOR, gl::ONE_MINUS_SRC_COLOR),
            (gl::ONE, gl::ONE_MINUS_SRC_ALPHA),
        );
    }
    fn set_blend_mode_show_overdraw(&mut self) {
        self.set_blend_factors(
            (gl::ONE, gl::ONE_MINUS_SRC_ALPHA),
            (gl::ONE, gl::ONE_MINUS_SRC_ALPHA),
        );
    }

    fn set_blend_mode_advanced(&mut self, mode: MixBlendMode) {
        self.gl.blend_equation(match mode {
            MixBlendMode::Normal => {
                // blend factor only make sense for the normal mode
                self.gl.blend_func_separate(gl::ZERO, gl::SRC_COLOR, gl::ZERO, gl::SRC_ALPHA);
                gl::FUNC_ADD
            },
            MixBlendMode::PlusLighter => {
                return self.set_blend_mode_plus_lighter();
            },
            MixBlendMode::Multiply => gl::MULTIPLY_KHR,
            MixBlendMode::Screen => gl::SCREEN_KHR,
            MixBlendMode::Overlay => gl::OVERLAY_KHR,
            MixBlendMode::Darken => gl::DARKEN_KHR,
            MixBlendMode::Lighten => gl::LIGHTEN_KHR,
            MixBlendMode::ColorDodge => gl::COLORDODGE_KHR,
            MixBlendMode::ColorBurn => gl::COLORBURN_KHR,
            MixBlendMode::HardLight => gl::HARDLIGHT_KHR,
            MixBlendMode::SoftLight => gl::SOFTLIGHT_KHR,
            MixBlendMode::Difference => gl::DIFFERENCE_KHR,
            MixBlendMode::Exclusion => gl::EXCLUSION_KHR,
            MixBlendMode::Hue => gl::HSL_HUE_KHR,
            MixBlendMode::Saturation => gl::HSL_SATURATION_KHR,
            MixBlendMode::Color => gl::HSL_COLOR_KHR,
            MixBlendMode::Luminosity => gl::HSL_LUMINOSITY_KHR,
        });
    }

    fn supports_extension(&self, extension: &str) -> bool {
        supports_extension(&self.extensions, extension)
    }

    fn log_driver_messages(gl: &dyn gl::Gl) {
        for msg in gl.get_debug_messages() {
            let level = match msg.severity {
                gl::DEBUG_SEVERITY_HIGH => Level::Error,
                gl::DEBUG_SEVERITY_MEDIUM => Level::Warn,
                gl::DEBUG_SEVERITY_LOW => Level::Info,
                gl::DEBUG_SEVERITY_NOTIFICATION => Level::Debug,
                _ => Level::Trace,
            };
            let ty = match msg.ty {
                gl::DEBUG_TYPE_ERROR => "error",
                gl::DEBUG_TYPE_DEPRECATED_BEHAVIOR => "deprecated",
                gl::DEBUG_TYPE_UNDEFINED_BEHAVIOR => "undefined",
                gl::DEBUG_TYPE_PORTABILITY => "portability",
                gl::DEBUG_TYPE_PERFORMANCE => "perf",
                gl::DEBUG_TYPE_MARKER => "marker",
                gl::DEBUG_TYPE_PUSH_GROUP => "group push",
                gl::DEBUG_TYPE_POP_GROUP => "group pop",
                gl::DEBUG_TYPE_OTHER => "other",
                _ => "?",
            };
            log!(level, "({}) {}", ty, msg.message);
        }
    }

    fn gl_describe_format(&self, format: ImageFormat) -> FormatDesc {
        match format {
            ImageFormat::R8 => FormatDesc {
                internal: gl::R8,
                external: gl::RED,
                read: gl::RED,
                pixel_type: gl::UNSIGNED_BYTE,
            },
            ImageFormat::R16 => FormatDesc {
                internal: gl::R16,
                external: gl::RED,
                read: gl::RED,
                pixel_type: gl::UNSIGNED_SHORT,
            },
            ImageFormat::BGRA8 => {
                FormatDesc {
                    internal: self.bgra_formats.internal,
                    external: self.bgra_formats.external,
                    read: gl::BGRA,
                    pixel_type: self.bgra_pixel_type,
                }
            },
            ImageFormat::RGBA8 => {
                FormatDesc {
                    internal: gl::RGBA8,
                    external: gl::RGBA,
                    read: gl::RGBA,
                    pixel_type: gl::UNSIGNED_BYTE,
                }
            },
            ImageFormat::RGBAF32 => FormatDesc {
                internal: gl::RGBA32F,
                external: gl::RGBA,
                read: gl::RGBA,
                pixel_type: gl::FLOAT,
            },
            ImageFormat::RGBAI32 => FormatDesc {
                internal: gl::RGBA32I,
                external: gl::RGBA_INTEGER,
                read: gl::RGBA_INTEGER,
                pixel_type: gl::INT,
            },
            ImageFormat::RG8 => FormatDesc {
                internal: gl::RG8,
                external: gl::RG,
                read: gl::RG,
                pixel_type: gl::UNSIGNED_BYTE,
            },
            ImageFormat::RG16 => FormatDesc {
                internal: gl::RG16,
                external: gl::RG,
                read: gl::RG,
                pixel_type: gl::UNSIGNED_SHORT,
            },
        }
    }
}

impl GpuBackend for GlDevice {
    fn textures_created(&self) -> u32 {
        self.textures_created
    }

    fn textures_deleted(&self) -> u32 {
        self.textures_deleted
    }

    fn set_initialize_color_targets_with_pink(&mut self, enabled: bool) {
        self.initialize_color_targets_with_pink = enabled;
    }

    fn create_gpu_profiler(&self, enable_markers: bool) -> GpuProfiler {
        let debug_method = if !enable_markers {
            GpuDebugMethod::None
        } else if self.capabilities.supports_khr_debug {
            GpuDebugMethod::KHR
        } else if self.supports_extension("GL_EXT_debug_marker") {
            GpuDebugMethod::MarkerEXT
        } else {
            warn!("asking to enable_gpu_markers but no supporting extension was found");
            GpuDebugMethod::None
        };

        info!("using {:?}", debug_method);

        GpuProfiler::new(Rc::new(GlQueries { gl: Rc::clone(&self.gl), debug_method }))
    }

    fn set_parameter(&mut self, param: &Parameter) {
        match param {
            Parameter::Bool(BoolParameter::PboUploads, enabled) => {
                if !self.is_software_webrender {
                    self.upload_method = if *enabled {
                        UploadMethod::PixelBuffer(crate::ONE_TIME_USAGE_HINT)
                    } else {
                        UploadMethod::Immediate
                    };
                }
            }
            Parameter::Bool(BoolParameter::BatchedUploads, enabled) => {
                if self.capabilities.requires_batched_texture_uploads.is_none() {
                    self.use_batched_texture_uploads = *enabled;
                }
            }
            Parameter::Bool(BoolParameter::DrawCallsForTextureCopy, enabled) => {
                self.use_draw_calls_for_texture_copy = *enabled;
            }
            Parameter::Int(IntParameter::BatchedUploadThreshold, threshold) => {
                self.batched_upload_threshold = *threshold;
            }
            _ => {}
        }
    }

    fn max_texture_size(&self) -> i32 {
        self.max_texture_size
    }

    fn surface_origin_is_top_left(&self) -> bool {
        self.surface_origin_is_top_left
    }

    fn get_capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn api_info(&self) -> GraphicsApiInfo {
        GraphicsApiInfo {
            kind: GraphicsApi::OpenGL,
            version: self.gl.get_string(gl::VERSION),
            renderer: self.gl.get_string(gl::RENDERER),
        }
    }

    fn take_out_of_memory_error(&self) -> bool {
        // Probably should check for other errors?
        self.gl.get_error() == gl::OUT_OF_MEMORY
    }

    fn blend_barrier(&self) {
        self.gl.blend_barrier_khr();
    }

    fn shader_feature_flags(&self) -> ShaderFeatureFlags {
        match self.gl.get_type() {
            gl::GlType::Gl => ShaderFeatureFlags::GL,
            gl::GlType::Gles => {
                let mut flags = ShaderFeatureFlags::GLES;
                flags |= if self.capabilities.supports_image_external_essl3 {
                    ShaderFeatureFlags::TEXTURE_EXTERNAL
                } else {
                    ShaderFeatureFlags::TEXTURE_EXTERNAL_ESSL1
                };
                if self.capabilities.supports_texture_external_bt709 {
                    flags |= ShaderFeatureFlags::TEXTURE_EXTERNAL_BT709;
                }
                flags
            }
        }
    }

    fn preferred_color_formats(&self) -> TextureFormatPair<ImageFormat> {
        self.color_formats.clone()
    }

    fn swizzle_settings(&self) -> Option<SwizzleSettings> {
        if self.capabilities.supports_texture_swizzle {
            Some(self.swizzle_settings)
        } else {
            None
        }
    }

    fn max_depth_ids(&self) -> i32 {
        return 1 << (self.depth_bits() - RESERVE_DEPTH_BITS);
    }

    fn ortho_near_plane(&self) -> f32 {
        return -self.max_depth_ids() as f32;
    }

    fn ortho_far_plane(&self) -> f32 {
        return (self.max_depth_ids() - 1) as f32;
    }

    fn required_transfer_stride(&self) -> StrideAlignment {
        self.required_transfer_stride
    }

    fn upload_method(&self) -> &UploadMethod {
        &self.upload_method
    }

    fn use_batched_texture_uploads(&self) -> bool {
        self.use_batched_texture_uploads
    }

    fn use_draw_calls_for_texture_copy(&self) -> bool {
        self.use_draw_calls_for_texture_copy
    }

    fn batched_upload_threshold(&self) -> i32 {
        self.batched_upload_threshold
    }

    fn reset_state(&mut self) {
        for i in 0 .. self.bound_textures.len() {
            self.bound_textures[i] = 0;
            self.gl.active_texture(gl::TEXTURE0 + i as gl::GLuint);
            self.gl.bind_texture(gl::TEXTURE_2D, 0);
        }

        self.bound_vao = 0;
        self.gl.bind_vertex_array(0);

        self.bound_read_fbo = (self.default_read_fbo, DeviceIntPoint::zero());
        self.gl.bind_framebuffer(gl::READ_FRAMEBUFFER, self.default_read_fbo.0);

        self.bound_draw_fbo = self.default_draw_fbo;
        self.gl.bind_framebuffer(gl::DRAW_FRAMEBUFFER, self.bound_draw_fbo.0);

        self.gl_state = GlRenderStateCache::default();
    }

    fn begin_frame(&mut self) -> GpuFrameId {
        debug_assert!(!self.inside_frame);
        self.inside_frame = true;

        self.textures_created = 0;
        self.textures_deleted = 0;

        // If our profiler state has changed, apply or remove the profiling
        // wrapper from our GL context.
        let being_profiled = profiler::thread_is_being_profiled();
        let using_wrapper = self.base_gl.is_some();

        // We can usually unwind driver stacks on OSes other than Android, so we don't need to
        // manually instrument gl calls there. Timestamps can be pretty expensive on Windows (2us
        // each and perhaps an opportunity to be descheduled?) which makes the profiles gathered
        // with this turned on less useful so only profile on ARM Android.
        if cfg!(any(target_arch = "arm", target_arch = "aarch64"))
            && cfg!(target_os = "android")
            && being_profiled
            && !using_wrapper
        {
            fn note(name: &str, duration: Duration) {
                profiler::add_text_marker("OpenGL Calls", name, duration);
            }
            let threshold = Duration::from_millis(1);
            let wrapped = gl::ProfilingGl::wrap(self.gl.clone(), threshold, note);
            let base = mem::replace(&mut self.gl, wrapped);
            self.base_gl = Some(base);
        } else if !being_profiled && using_wrapper {
            self.gl = self.base_gl.take().unwrap();
        }

        // Retrieve the currently set FBO.
        let mut default_read_fbo = [0];
        unsafe {
            self.gl.get_integer_v(gl::READ_FRAMEBUFFER_BINDING, &mut default_read_fbo);
        }
        self.default_read_fbo = FBOId(default_read_fbo[0] as gl::GLuint);
        let mut default_draw_fbo = [0];
        unsafe {
            self.gl.get_integer_v(gl::DRAW_FRAMEBUFFER_BINDING, &mut default_draw_fbo);
        }
        self.default_draw_fbo = FBOId(default_draw_fbo[0] as gl::GLuint);

        // Shader state
        self.bound_program = 0;
        self.gl.use_program(0);

        // Reset common state
        self.reset_state();
        self.gl.disable(gl::STENCIL_TEST);

        // Pixel op state
        self.gl.pixel_store_i(gl::UNPACK_ALIGNMENT, 1);
        self.gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, 0);

        // Default is sampler 0, always
        self.gl.active_texture(gl::TEXTURE0);

        self.frame_id
    }

    fn bind_texture(&mut self, slot: TextureSlot, texture: &Texture, swizzle: Swizzle) {
        let old_swizzle = texture.active_swizzle.replace(swizzle);
        let set_swizzle = if old_swizzle != swizzle {
            Some(swizzle)
        } else {
            None
        };
        self.bind_texture_impl(slot, texture.id, get_gl_target(texture.target), set_swizzle, None);
    }

    fn bind_external_texture(&mut self, slot: TextureSlot, external_texture: &ExternalTexture) {
        self.bind_texture_impl(
            slot,
            external_texture.id,
            get_gl_target(external_texture.target),
            None,
            Some(external_texture.image_rendering),
        );
    }

    fn reset_read_target(&mut self) {
        let fbo = self.default_read_fbo;
        self.bind_read_target_impl(fbo, DeviceIntPoint::zero());
    }

    fn begin_render_pass(&mut self, desc: &RenderPassDescriptor) {
        debug_assert!(self.inside_frame);
        debug_assert!(self.current_render_pass.is_none(), "render pass already in progress");

        self.bind_draw_target(desc.target);

        if self.capabilities.supports_qcom_tiled_rendering {
            if let Some(area) = desc.render_area {
                let preserve_mask = match desc.color_load {
                    LoadOp::Load => gl::COLOR_BUFFER_BIT0_QCOM,
                    LoadOp::DontCare => 0,
                };
                self.gl.start_tiling_qcom(
                    area.min.x.max(0) as _,
                    area.min.y.max(0) as _,
                    area.width() as _,
                    area.height() as _,
                    preserve_mask,
                );
            }
        }

        self.current_render_pass = Some(*desc);
    }

    fn end_render_pass(&mut self, depth_store: StoreOp) {
        debug_assert!(self.inside_frame);
        let desc = self.current_render_pass.take().expect("no render pass in progress");

        if depth_store == StoreOp::Discard {
            self.invalidate_depth_target();
        }

        if self.capabilities.supports_qcom_tiled_rendering && desc.render_area.is_some() {
            self.gl.end_tiling_qcom(gl::COLOR_BUFFER_BIT0_QCOM);
        }
    }

    fn link_program(
        &mut self,
        program: &mut Program,
        descriptor: &VertexDescriptor,
    ) -> Result<(), ShaderError> {
        profile_marker!("compile shader", program.source_info.base_filename);

        let _guard = CrashAnnotatorGuard::new(
            &self.crash_annotator,
            CrashAnnotation::CompileShader,
            &program.source_info.full_name_cstr
        );

        assert!(!program.is_initialized());
        let mut build_program = true;
        let info = &program.source_info;

        // See if we hit the binary shader cache
        if let Some(ref cached_programs) = self.cached_programs {
            // If the shader is not in the cache, attempt to load it from disk
            if cached_programs.entries.borrow().get(&program.source_info.digest).is_none() {
                if let Some(ref handler) = cached_programs.program_cache_handler {
                    handler.try_load_shader_from_disk(&program.source_info.digest, cached_programs);
                    if let Some(entry) = cached_programs.entries.borrow().get(&program.source_info.digest) {
                        self.gl.program_binary(program.id, entry.binary.format, &entry.binary.bytes);
                    }
                }
            }

            if let Some(entry) = cached_programs.entries.borrow_mut().get_mut(&info.digest) {
                let mut link_status = [0];
                unsafe {
                    self.gl.get_program_iv(program.id, gl::LINK_STATUS, &mut link_status);
                }
                if link_status[0] == 0 {
                    let error_log = self.gl.get_program_info_log(program.id);
                    error!(
                      "Failed to load a program object with a program binary: {} renderer {}\n{}",
                      &info.base_filename,
                      self.capabilities.renderer_name,
                      error_log
                    );
                    if let Some(ref program_cache_handler) = cached_programs.program_cache_handler {
                        program_cache_handler.notify_program_binary_failed(&entry.binary);
                    }
                } else {
                    entry.linked = true;
                    build_program = false;
                }
            }
        }

        // If not, we need to do a normal compile + link pass.
        if build_program {
            // Compile the vertex shader
            let (vs_source, vs_source_map) = info.compute_source(self, ShaderKind::Vertex);
            let vs_id = match self.compile_shader(
                &info.full_name(),
                gl::VERTEX_SHADER,
                &vs_source,
                vs_source_map.as_ref(),
            ) {
                    Ok(vs_id) => vs_id,
                    Err(err) => return Err(err),
                };

            // Compile the fragment shader
            let (fs_source, fs_source_map) = info.compute_source(self, ShaderKind::Fragment);
            let fs_id =
                match self.compile_shader(
                    &info.full_name(),
                    gl::FRAGMENT_SHADER,
                    &fs_source,
                    fs_source_map.as_ref(),
                ) {
                    Ok(fs_id) => fs_id,
                    Err(err) => {
                        self.gl.delete_shader(vs_id);
                        return Err(err);
                    }
                };

            // Check if shader source should be dumped
            if Some(info.base_filename) == self.dump_shader_source.as_ref().map(String::as_ref) {
                let path = std::path::Path::new(info.base_filename);
                std::fs::write(path.with_extension("vert"), vs_source).unwrap();
                std::fs::write(path.with_extension("frag"), fs_source).unwrap();
            }

            // Attach shaders
            self.gl.attach_shader(program.id, vs_id);
            self.gl.attach_shader(program.id, fs_id);

            // Bind vertex attributes
            for (i, attr) in descriptor
                .vertex_attributes
                .iter()
                .chain(descriptor.instance_attributes.iter())
                .enumerate()
            {
                self.gl
                    .bind_attrib_location(program.id, i as gl::GLuint, attr.name);
            }

            if self.cached_programs.is_some() {
                self.gl.program_parameter_i(program.id, gl::PROGRAM_BINARY_RETRIEVABLE_HINT, gl::TRUE as gl::GLint);
            }

            // Link!
            self.gl.link_program(program.id);

            // GL recommends detaching and deleting shaders once the link
            // is complete (whether successful or not). This allows the driver
            // to free any memory associated with the parsing and compilation.
            self.gl.detach_shader(program.id, vs_id);
            self.gl.detach_shader(program.id, fs_id);
            self.gl.delete_shader(vs_id);
            self.gl.delete_shader(fs_id);

            let mut link_status = [0];
            unsafe {
                self.gl.get_program_iv(program.id, gl::LINK_STATUS, &mut link_status);
            }
            if link_status[0] == 0 {
                let error_log = self.gl.get_program_info_log(program.id);
                error!(
                    "Failed to link shader program: {}\n{}",
                    &info.base_filename,
                    error_log
                );
                // The program object is gone, so clear the id rather than
                // leaving the caller holding a dangling GL name that a later
                // link or delete would operate on.
                self.gl.delete_program(program.id);
                if self.bound_program == program.id {
                    self.gl.use_program(0);
                    self.bound_program = 0;
                }
                program.id = 0;
                let diagnostics = ShaderSourceMap::new().map_log(&error_log);
                return Err(ShaderError::Link(
                    info.base_filename.to_owned(),
                    error_log,
                    diagnostics,
                ));
            }

            if let Some(ref cached_programs) = self.cached_programs {
                if !info.from_source_override()
                    && !cached_programs.entries.borrow().contains_key(&info.digest)
                {
                    let (buffer, format) = self.gl.get_program_binary(program.id);
                    if buffer.len() > 0 {
                        let binary = Arc::new(ProgramBinary::new(buffer, format, info.digest.clone()));
                        cached_programs.add_new_program_binary(binary);
                    }
                }
            }
        }

        // If we get here, the link succeeded, so get the uniforms.
        program.is_initialized = true;
        program.u_transform = self.gl.get_uniform_location(program.id, "uTransform");
        program.u_texture_size = self.gl.get_uniform_location(program.id, "uTextureSize");

        Ok(())
    }

    fn bind_pipeline(&mut self, program: &Program, state: &RenderState) -> bool {
        debug_assert!(self.inside_frame);
        debug_assert!(program.is_initialized());
        if !program.is_initialized() {
            return false;
        }

        self.apply_render_state(state);

        if self.bound_program != program.id {
            self.gl.use_program(program.id);
            self.bound_program = program.id;
            self.bound_program_name = program.source_info.full_name_cstr.clone();
        }
        true
    }

    fn create_texture(
        &mut self,
        target: ImageBufferKind,
        format: ImageFormat,
        mut width: i32,
        mut height: i32,
        filter: TextureFilter,
        render_target: Option<RenderTargetInfo>,
    ) -> Texture {
        debug_assert!(self.inside_frame);

        if width > self.max_texture_size || height > self.max_texture_size {
            error!("Attempting to allocate a texture of size {}x{} above the limit, trimming", width, height);
            width = width.min(self.max_texture_size);
            height = height.min(self.max_texture_size);
        }

        // Set up the texture book-keeping.
        let gl_target = get_gl_target(target);
        let mut texture = Texture {
            id: self.gl.gen_textures(1)[0],
            target,
            size: DeviceIntSize::new(width, height),
            format,
            filter,
            active_swizzle: Cell::default(),
            fbo: None,
            fbo_with_depth: None,
            last_frame_used: self.frame_id,
            flags: TextureFlags::default(),
        };
        self.bind_texture(DEFAULT_TEXTURE, &texture, Swizzle::default());
        self.set_texture_parameters(gl_target, filter);

        if self.capabilities.supports_texture_usage && render_target.is_some() {
            self.gl.tex_parameter_i(gl_target, gl::TEXTURE_USAGE_ANGLE, gl::FRAMEBUFFER_ATTACHMENT_ANGLE as gl::GLint);
        }

        // Allocate storage.
        let desc = self.gl_describe_format(texture.format);

        // Firefox doesn't use mipmaps, but Servo uses them for standalone image
        // textures images larger than 512 pixels. This is the only case where
        // we set the filter to trilinear.
        let mipmap_levels =  if texture.filter == TextureFilter::Trilinear {
            let max_dimension = cmp::max(width, height);
            ((max_dimension) as f64).log2() as gl::GLint + 1
        } else {
            1
        };

        // We never want to upload texture data at the same time as allocating the texture.
        self.gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, 0);

        // Use glTexStorage where available, since it avoids allocating
        // unnecessary mipmap storage and generally improves performance with
        // stronger invariants.
        let use_texture_storage = match self.texture_storage_usage {
            TexStorageUsage::Always => true,
            TexStorageUsage::NonBGRA8 => texture.format != ImageFormat::BGRA8,
            TexStorageUsage::Never => false,
        };
        if use_texture_storage {
            self.gl.tex_storage_2d(
                gl_target,
                mipmap_levels,
                desc.internal,
                texture.size.width as gl::GLint,
                texture.size.height as gl::GLint,
            );
        } else {
            self.gl.tex_image_2d(
                gl_target,
                0,
                desc.internal as gl::GLint,
                texture.size.width as gl::GLint,
                texture.size.height as gl::GLint,
                0,
                desc.external,
                desc.pixel_type,
                None,
            );
        }

        // Set up FBOs, if required.
        if let Some(rt_info) = render_target {
            self.init_fbos(&mut texture, false);
            if rt_info.has_depth {
                self.init_fbos(&mut texture, true);
            }
        }

        self.textures_created += 1;

        if self.initialize_color_targets_with_pink
            && format == ImageFormat::BGRA8
            && render_target.is_some()
        {
            self.bind_draw_target(DrawTarget::from_texture(
                &texture,
                false,
            ));
            self.clear_target_impl(Some([1.0, 0.0, 1.0, 1.0]), None, None);
            if let Some(pass) = self.current_render_pass {
                self.bind_draw_target(pass.target);
            }
        }

        texture
    }

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
    ) {
        if self.capabilities.supports_copy_image_sub_data {
            assert_ne!(
                src_texture.id, dest_texture.id,
                "glCopyImageSubData's behaviour is undefined if src and dst images are identical and the rectangles overlap."
            );
            unsafe {
                self.gl.copy_image_sub_data(
                    src_texture.id,
                    get_gl_target(src_texture.target),
                    0,
                    src_x as _,
                    src_y as _,
                    0,
                    dest_texture.id,
                    get_gl_target(dest_texture.target),
                    0,
                    dest_x as _,
                    dest_y as _,
                    0,
                    width as _,
                    height as _,
                    1,
                );
            }
        } else {
            let src_offset = FramebufferIntPoint::new(src_x as i32, src_y as i32);
            let dest_offset = FramebufferIntPoint::new(dest_x as i32, dest_y as i32);
            let size = FramebufferIntSize::new(width as i32, height as i32);

            self.blit_render_target(
                ReadTarget::from_texture(src_texture),
                FramebufferIntRect::from_origin_and_size(src_offset, size),
                DrawTarget::from_texture(dest_texture, false),
                FramebufferIntRect::from_origin_and_size(dest_offset, size),
                // In most cases the filter shouldn't matter, as there is no scaling involved
                // in the blit. We were previously using Linear, but this caused issues when
                // blitting RGBAF32 textures on Mali, so use Nearest to be safe.
                TextureFilter::Nearest,
            );
        }
    }

    fn invalidate_render_target(&mut self, texture: &Texture) {
        if self.capabilities.supports_render_target_invalidate {
            let (fbo, attachments) = if texture.supports_depth() {
                (&texture.fbo_with_depth,
                 &[gl::COLOR_ATTACHMENT0, gl::DEPTH_ATTACHMENT] as &[gl::GLenum])
            } else {
                (&texture.fbo, &[gl::COLOR_ATTACHMENT0] as &[gl::GLenum])
            };

            if let Some(fbo_id) = fbo {
                let original_bound_fbo = self.bound_draw_fbo;
                // Note: The invalidate extension may not be supported, in which
                // case this is a no-op. That's ok though, because it's just a
                // hint.
                self.bind_external_draw_target(*fbo_id);
                self.gl.invalidate_framebuffer(gl::FRAMEBUFFER, attachments);
                self.bind_external_draw_target(original_bound_fbo);
            }
        }
    }

    fn reuse_render_target(
        &mut self,
        texture: &mut Texture,
        rt_info: RenderTargetInfo,
    ) {
        texture.last_frame_used = self.frame_id;

        // Add depth support if needed.
        if rt_info.has_depth && !texture.supports_depth() {
            self.init_fbos(texture, true);
        }
    }

    fn blit_render_target(
        &mut self,
        src_target: ReadTarget,
        src_rect: FramebufferIntRect,
        dest_target: DrawTarget,
        dest_rect: FramebufferIntRect,
        filter: TextureFilter,
    ) {
        debug_assert!(self.inside_frame);

        self.bind_read_target(src_target);

        self.bind_draw_target(dest_target);

        self.blit_render_target_impl(src_rect, dest_rect, filter);

        // A blit into another target from inside a render pass leaves the
        // pass's own target unbound, so restore it.
        if let Some(pass) = self.current_render_pass {
            if pass.target != dest_target {
                self.bind_draw_target(pass.target);
                self.reset_read_target();
            }
        }
    }

    fn delete_texture(&mut self, mut texture: Texture) {
        debug_assert!(self.inside_frame);
        let had_depth = texture.supports_depth();
        if let Some(fbo) = texture.fbo {
            self.gl.delete_framebuffers(&[fbo.0]);
            texture.fbo = None;
        }
        if let Some(fbo) = texture.fbo_with_depth {
            self.gl.delete_framebuffers(&[fbo.0]);
            texture.fbo_with_depth = None;
        }

        if had_depth {
            self.release_depth_target(texture.get_dimensions());
        }

        self.gl.delete_textures(&[texture.id]);

        for bound_texture in &mut self.bound_textures {
            if *bound_texture == texture.id {
                *bound_texture = 0;
            }
        }

        self.textures_deleted += 1;

        // Disarm the assert in Texture::drop().
        texture.id = 0;
    }

    #[cfg(feature = "replay")]
    fn delete_external_texture(&mut self, external: ExternalTexture) {
        self.gl.delete_textures(&[external.id]);
    }

    fn delete_program(&mut self, mut program: Program) {
        if program.id == 0 {
            return;
        }
        // GL recycles names, so a program created after this one is deleted can
        // be handed the same id. Drop the binding cache entry, otherwise
        // `bind_program` would skip the `use_program` call for the new program.
        if self.bound_program == program.id {
            self.gl.use_program(0);
            self.bound_program = 0;
        }
        self.gl.delete_program(program.id);
        program.id = 0;
    }

    fn create_program(
        &mut self,
        base_filename: &'static str,
        features: &[&'static str],
    ) -> Result<Program, ShaderError> {
        debug_assert!(self.inside_frame);

        let source_info = ProgramSourceInfo::new(self, base_filename, features);

        // Create program
        let pid = self.gl.create_program();

        // Attempt to load a cached binary if possible.
        if let Some(ref cached_programs) = self.cached_programs {
            if let Some(entry) = cached_programs.entries.borrow().get(&source_info.digest) {
                self.gl.program_binary(pid, entry.binary.format, &entry.binary.bytes);
            }
        }

        // Use 0 for the uniforms as they are initialized by link_program.
        let program = Program {
            id: pid,
            u_transform: 0,
            u_texture_size: 0,
            source_info,
            is_initialized: false,
        };

        Ok(program)
    }

    /// Whether shader sources can be replaced at runtime.
    ///
    /// SWGL discards the GLSL it is handed and dispatches to a program
    /// transpiled to C++ at build time (see `swgl::Context::shader_source`),
    /// so there is nothing for an override to recompile.
    #[cfg(feature = "debugger")]
    fn supports_shader_source_override(&self) -> bool {
        !self.is_software_webrender
    }

    /// Names of every `.glsl` file built into this binary, sorted.
    #[cfg(feature = "debugger")]
    fn shader_file_names(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = UNOPTIMIZED_SHADERS.keys().cloned().collect();
        names.sort_unstable();
        names
    }

    /// The source built into the binary for `name`, ignoring any override.
    #[cfg(feature = "debugger")]
    fn builtin_shader_source(&self, name: &str) -> Option<&'static str> {
        UNOPTIMIZED_SHADERS.get(name).map(|entry| entry.source)
    }

    /// The source currently in effect for `name`: the override if one is
    /// installed, otherwise whatever `get_unoptimized_shader_source` resolves.
    #[cfg(feature = "debugger")]
    fn get_shader_source(&self, name: &str) -> Cow<'static, str> {
        match self.shader_source_overrides.get(name) {
            Some(source) => Cow::Owned(source.clone()),
            None => get_unoptimized_shader_source(name, self.resource_override_path.as_ref()),
        }
    }

    /// The source in effect for `name`. Without the debugger there are no
    /// runtime overrides, so this is whatever `get_unoptimized_shader_source`
    /// resolves.
    #[cfg(not(feature = "debugger"))]
    fn get_shader_source(&self, name: &str) -> Cow<'static, str> {
        get_unoptimized_shader_source(name, self.resource_override_path.as_ref())
    }

    #[cfg(feature = "debugger")]
    fn shader_source_override(&self, name: &str) -> Option<&str> {
        self.shader_source_overrides.get(name).map(String::as_str)
    }

    #[cfg(feature = "debugger")]
    fn has_shader_source_overrides(&self) -> bool {
        !self.shader_source_overrides.is_empty()
    }

    #[cfg(feature = "debugger")]
    fn set_shader_source_override(&mut self, name: &str, source: String) {
        self.shader_source_overrides.insert(name.to_string(), source);
        self.shader_include_closures.borrow_mut().clear();
    }

    /// Drop the override for `name`, returning whether there was one.
    #[cfg(feature = "debugger")]
    fn clear_shader_source_override(&mut self, name: &str) -> bool {
        let had_override = self.shader_source_overrides.remove(name).is_some();
        if had_override {
            self.shader_include_closures.borrow_mut().clear();
        }
        had_override
    }

    /// The set of `.glsl` files `base_filename` pulls in, including itself.
    #[cfg(feature = "debugger")]
    fn shader_include_closure(&self, base_filename: &str) -> FastHashSet<String> {
        if let Some(closure) = self.shader_include_closures.borrow().get(base_filename) {
            return closure.clone();
        }

        let closure: FastHashSet<String> =
            webrender_build::shader::shader_include_closure(
                base_filename,
                &|f| self.get_shader_source(f),
            )
                .into_iter()
                .collect();
        self.shader_include_closures
            .borrow_mut()
            .insert(base_filename.to_string(), closure.clone());

        closure
    }

    /// The preprocessed vertex and fragment source handed to the driver for
    /// one variant, built from the sources currently in effect.
    ///
    /// This is the text a driver log's line numbers refer to when no known
    /// driver pattern matched it and the location could not be resolved.
    #[cfg(feature = "debugger")]
    fn expanded_shader_source(
        &self,
        base_filename: &str,
        features: &[&'static str],
    ) -> (String, String) {
        let mut vertex = String::new();
        self.build_shader_string(features, ShaderKind::Vertex, base_filename, |s| {
            vertex.push_str(s)
        });

        let mut fragment = String::new();
        self.build_shader_string(features, ShaderKind::Fragment, base_filename, |s| {
            fragment.push_str(s)
        });

        (vertex, fragment)
    }

    fn bind_shader_samplers(&mut self, program: &Program, bindings: &[(&'static str, TextureSlot)]) {
        // The program must be bound before calling bind_shader_samplers
        assert_eq!(self.bound_program, program.id);

        for binding in bindings {
            let u_location = self.gl.get_uniform_location(program.id, binding.0);
            if u_location != -1 {
                self.gl
                    .uniform_1i(u_location, (binding.1).0 as gl::GLint);
            }
        }
    }

    fn set_uniforms(
        &self,
        program: &Program,
        transform: &Transform3D<f32>,
    ) {
        debug_assert!(self.inside_frame);
        debug_assert_eq!(self.bound_program, program.id);

        self.gl
            .uniform_matrix_4fv(program.u_transform, false, &transform.to_array());
    }

    fn set_shader_texture_size(
        &self,
        program: &Program,
        texture_size: DeviceSize,
    ) {
        debug_assert!(self.inside_frame);
        debug_assert_eq!(self.bound_program, program.id);

        if program.u_texture_size != -1 {
            self.gl.uniform_2f(program.u_texture_size, texture_size.width, texture_size.height);
        }
    }

    fn create_transfer_buffer_with_size(&mut self, size: usize) -> TransferBuffer {
        let mut pbo = self.create_transfer_buffer();

        self.gl.bind_buffer(gl::PIXEL_PACK_BUFFER, pbo.id);
        self.gl.pixel_store_i(gl::PACK_ALIGNMENT, 1);
        self.gl.buffer_data_untyped(
            gl::PIXEL_PACK_BUFFER,
            size as _,
            ptr::null(),
            gl::STREAM_READ,
        );
        self.gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, 0);

        pbo.reserved_size = size;
        pbo
    }

    fn read_pixels_into_transfer_buffer(
        &mut self,
        read_target: ReadTarget,
        rect: DeviceIntRect,
        format: ImageFormat,
        pbo: &TransferBuffer,
    ) {
        let byte_size = rect.area() as usize * format.bytes_per_pixel() as usize;

        assert!(byte_size <= pbo.reserved_size);

        self.bind_read_target(read_target);

        self.gl.bind_buffer(gl::PIXEL_PACK_BUFFER, pbo.id);
        self.gl.pixel_store_i(gl::PACK_ALIGNMENT, 1);

        let gl_format = self.gl_describe_format(format);

        unsafe {
            self.gl.read_pixels_into_pbo(
                rect.min.x as _,
                rect.min.y as _,
                rect.width() as _,
                rect.height() as _,
                gl_format.read,
                gl_format.pixel_type,
            );
        }

        self.gl.bind_buffer(gl::PIXEL_PACK_BUFFER, 0);
    }

    fn map_transfer_buffer<'a>(&'a mut self, pbo: &'a TransferBuffer) -> Option<MappedTransferBuffer<'a>> {
        self.gl.bind_buffer(gl::PIXEL_PACK_BUFFER, pbo.id);

        let buf_ptr = match self.gl.get_type() {
            gl::GlType::Gl => {
                self.gl.map_buffer(gl::PIXEL_PACK_BUFFER, gl::READ_ONLY)
            }

            gl::GlType::Gles => {
                self.gl.map_buffer_range(
                    gl::PIXEL_PACK_BUFFER,
                    0,
                    pbo.reserved_size as _,
                    gl::MAP_READ_BIT)
            }
        };

        if buf_ptr.is_null() {
            return None;
        }

        let buffer = unsafe { slice::from_raw_parts(buf_ptr as *const u8, pbo.reserved_size) };

        Some(MappedTransferBuffer {
            device: self,
            data: buffer,
        })
    }

    fn unmap_transfer_buffer(&mut self) {
        self.gl.unmap_buffer(gl::PIXEL_PACK_BUFFER);
        self.gl.bind_buffer(gl::PIXEL_PACK_BUFFER, 0);
    }

    fn delete_transfer_buffer(&mut self, mut pbo: TransferBuffer) {
        self.gl.delete_buffers(&[pbo.id]);
        pbo.id = 0;
        pbo.reserved_size = 0
    }

    fn allocate_upload_buffer(
        &mut self,
        buffer: &mut TransferBuffer,
        size: usize,
        usage_hint: VertexUsageHint,
        persistent: bool,
    ) -> Result<UploadBufferMapping, String> {
        assert_eq!(buffer.reserved_size, 0);
        buffer.reserved_size = size;

        self.gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, buffer.id);
        if persistent {
            assert!(self.capabilities.supports_buffer_storage);
            self.gl.buffer_storage(
                gl::PIXEL_UNPACK_BUFFER,
                size as _,
                ptr::null(),
                gl::MAP_WRITE_BIT | gl::MAP_PERSISTENT_BIT,
            );
            let ptr = self.gl.map_buffer_range(
                gl::PIXEL_UNPACK_BUFFER,
                0,
                size as _,
                // GL_MAP_COHERENT_BIT doesn't seem to work on Adreno, so use glFlushMappedBufferRange.
                // kvark notes that coherent memory can be faster on some platforms, such as nvidia,
                // so in the future we could choose which to use at run time.
                gl::MAP_WRITE_BIT | gl::MAP_PERSISTENT_BIT | gl::MAP_FLUSH_EXPLICIT_BIT,
            ) as *mut _;

            let ptr = ptr::NonNull::new(ptr).ok_or_else(
                || format!("Failed to persistently map TransferBuffer of size {} bytes", size)
            )?;

            Ok(UploadBufferMapping::Persistent(ptr))
        } else {
            self.gl.buffer_data_untyped(
                gl::PIXEL_UNPACK_BUFFER,
                size as _,
                ptr::null(),
                usage_hint.to_gl(),
            );
            let ptr = self.gl.map_buffer_range(
                gl::PIXEL_UNPACK_BUFFER,
                0,
                size as _,
                // Unlike map_upload_buffer, where we are re-mapping a buffer that has previously been unmapped,
                // this buffer has just been created there is no need for GL_MAP_UNSYNCHRONIZED_BIT.
                gl::MAP_WRITE_BIT,
            ) as *mut _;

            let ptr = ptr::NonNull::new(ptr).ok_or_else(
                || format!("Failed to transiently map TransferBuffer of size {} bytes", size)
            )?;

            Ok(UploadBufferMapping::Transient(ptr))
        }
    }

    fn map_upload_buffer(
        &mut self,
        buffer: &TransferBuffer,
    ) -> Result<ptr::NonNull<mem::MaybeUninit<u8>>, String> {
        self.gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, buffer.id);
        let ptr = self.gl.map_buffer_range(
            gl::PIXEL_UNPACK_BUFFER,
            0,
            buffer.reserved_size as _,
            gl::MAP_WRITE_BIT | gl::MAP_UNSYNCHRONIZED_BIT,
        ) as *mut _;

        ptr::NonNull::new(ptr).ok_or_else(
            || format!("Failed to transiently map TransferBuffer of size {} bytes", buffer.reserved_size)
        )
    }

    fn flush_upload_buffer(
        &mut self,
        buffer: &TransferBuffer,
        mapping: &UploadBufferMapping,
        size_used: usize,
        chunks: &[UploadChunk],
    ) {
        self.gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, buffer.id);
        match mapping {
            UploadBufferMapping::Unmapped => unreachable!("upload buffer should be mapped at this stage."),
            UploadBufferMapping::Transient(_) => {
                self.gl.unmap_buffer(gl::PIXEL_UNPACK_BUFFER);
            }
            UploadBufferMapping::Persistent(_) => {
                self.gl.flush_mapped_buffer_range(gl::PIXEL_UNPACK_BUFFER, 0, size_used as _);
            }
        }
        for chunk in chunks {
            self.upload_chunk(chunk.texture, chunk.rect, chunk.stride, chunk.format_override, chunk.offset);
        }
        self.gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, 0);
    }

    fn orphan_upload_buffer(&mut self, buffer: &mut TransferBuffer) {
        self.gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, buffer.id);
        self.gl.buffer_data_untyped(
            gl::PIXEL_UNPACK_BUFFER,
            0,
            ptr::null(),
            gl::STREAM_DRAW,
        );
        self.gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, 0);
        buffer.reserved_size = 0;
    }

    fn upload_texture_region(
        &mut self,
        texture: &Texture,
        rect: DeviceIntRect,
        stride: Option<i32>,
        format_override: Option<ImageFormat>,
        data: &[u8],
    ) {
        if cfg!(debug_assertions) {
            let mut bound_buffer = [0];
            unsafe {
                self.gl.get_integer_v(gl::PIXEL_UNPACK_BUFFER_BINDING, &mut bound_buffer);
            }
            assert_eq!(bound_buffer[0], 0, "GL_PIXEL_UNPACK_BUFFER must not be bound for immediate uploads.");
        }
        self.upload_chunk(texture, rect, stride, format_override, data.as_ptr() as usize);
    }

    fn create_fence(&mut self) -> Option<Fence> {
        let sync = self.gl.fence_sync(gl::SYNC_GPU_COMMANDS_COMPLETE, 0);
        if sync.is_null() {
            None
        } else {
            Some(Fence(sync as usize))
        }
    }

    fn poll_fence(&self, fence: &Fence) -> FenceStatus {
        match self.gl.client_wait_sync(fence.0 as gl::GLsync, 0, 0) {
            gl::TIMEOUT_EXPIRED => FenceStatus::Pending,
            gl::ALREADY_SIGNALED | gl::CONDITION_SATISFIED => FenceStatus::Signaled,
            gl::WAIT_FAILED | _ => FenceStatus::Error,
        }
    }

    fn delete_fence(&mut self, fence: Fence) {
        self.gl.delete_sync(fence.0 as gl::GLsync);
    }

    fn upload_texture_immediate(&mut self, texture: &Texture, pixels: &[u8]) {
        self.bind_texture(DEFAULT_TEXTURE, texture, Swizzle::default());
        let desc = self.gl_describe_format(texture.format);
        self.gl.tex_sub_image_2d(
            get_gl_target(texture.target),
            0,
            0,
            0,
            texture.size.width as gl::GLint,
            texture.size.height as gl::GLint,
            desc.external,
            desc.pixel_type,
            pixels,
        );
    }

    fn read_pixels(&mut self, img_desc: &ImageDescriptor) -> Vec<u8> {
        let desc = self.gl_describe_format(img_desc.format);
        self.gl.read_pixels(
            0, 0,
            img_desc.size.width as i32,
            img_desc.size.height as i32,
            desc.read,
            desc.pixel_type,
        )
    }

    fn read_pixels_into(
        &mut self,
        rect: FramebufferIntRect,
        format: ImageFormat,
        output: &mut [u8],
    ) {
        let bytes_per_pixel = format.bytes_per_pixel();
        let desc = self.gl_describe_format(format);
        let size_in_bytes = (bytes_per_pixel * rect.area()) as usize;
        assert_eq!(output.len(), size_in_bytes);

        self.gl.flush();
        self.gl.read_pixels_into_buffer(
            rect.min.x as _,
            rect.min.y as _,
            rect.width() as _,
            rect.height() as _,
            desc.read,
            desc.pixel_type,
            output,
        );
    }

    fn attach_read_texture_external(
        &mut self, handle: ExternalTextureHandle, target: ImageBufferKind
    ) {
        self.bind_scratch_read_target();
        self.attach_read_texture_raw(handle.0 as gl::GLuint, get_gl_target(target))
    }

    fn attach_read_texture(&mut self, texture: &Texture) {
        self.bind_scratch_read_target();
        self.attach_read_texture_raw(texture.id, get_gl_target(texture.target))
    }

    fn bind_vao(&mut self, vao: &VAO) {
        self.bind_vao_impl(vao.id)
    }

    fn create_vao(&mut self, descriptor: &VertexDescriptor, instance_divisor: u32) -> VAO {
        debug_assert!(self.inside_frame);

        let buffer_ids = self.gl.gen_buffers(3);
        let ibo_id = IBOId(buffer_ids[0]);
        let main_vbo_id = VBOId(buffer_ids[1]);
        let instance_vbo_id = VBOId(buffer_ids[2]);

        self.create_vao_with_vbos(
            descriptor,
            main_vbo_id,
            instance_vbo_id,
            instance_divisor,ibo_id,
            /* owns_vertices_and_indices */ true,
            /* owns_instances */ true
        )
    }

    fn delete_vao(&mut self, mut vao: VAO) {
        self.gl.delete_vertex_arrays(&[vao.id]);
        vao.id = 0;

        if vao.owns_vertices_and_indices {
            self.gl.delete_buffers(&[vao.ibo_id.0]);
            self.gl.delete_buffers(&[vao.main_vbo_id.0]);
        }

        if vao.owns_instances {
            self.gl.delete_buffers(&[vao.instance_vbo_id.0]);
        }
    }

    fn create_vao_with_new_instances(
        &mut self,
        descriptor: &VertexDescriptor,
        base_vao: &VAO,
    ) -> VAO {
        debug_assert!(self.inside_frame);

        let buffer_ids = self.gl.gen_buffers(1);
        let instance_vbo_id = VBOId(buffer_ids[0]);

        self.create_vao_with_vbos(
            descriptor,
            base_vao.main_vbo_id,
            instance_vbo_id,
            base_vao.instance_divisor,
            base_vao.ibo_id,
            /* owns_vertices_and_indices */ false,
            /* owns_instances */ true,
        )
    }

    fn create_vao_with_shared_instances(
        &mut self,
        descriptor: &VertexDescriptor,
        base_vao: &VAO,
    ) -> VAO {
        debug_assert!(self.inside_frame);

        self.create_vao_with_vbos(
            descriptor,
            base_vao.main_vbo_id,
            base_vao.instance_vbo_id,
            base_vao.instance_divisor,
            base_vao.ibo_id,
            /* owns_vertices_and_indices */ false,
            /* owns_instances */ false,
        )
    }

    fn update_vao_main_vertices(
        &mut self,
        vao: &VAO,
        vertices: &[u8],
        usage_hint: VertexUsageHint,
    ) {
        debug_assert_eq!(self.bound_vao, vao.id);
        self.update_vbo_data(vao.main_vbo_id, vertices, usage_hint)
    }

    fn update_vao_instances(
        &mut self,
        vao: &VAO,
        instances: &[u8],
        instance_stride: usize,
        usage_hint: VertexUsageHint,
        repeat: Option<NonZeroUsize>,
    ) {
        debug_assert_eq!(self.bound_vao, vao.id);
        debug_assert_eq!(vao.instance_stride, instance_stride);

        match repeat {
            Some(count) => {
                let count = count.get();
                let target = gl::ARRAY_BUFFER;
                self.gl.bind_buffer(target, vao.instance_vbo_id.0);
                let size = instances.len() * count;
                self.gl.buffer_data_untyped(
                    target,
                    size as _,
                    ptr::null(),
                    usage_hint.to_gl(),
                );

                let ptr = match self.gl.get_type() {
                    gl::GlType::Gl => {
                        self.gl.map_buffer(target, gl::WRITE_ONLY)
                    }
                    gl::GlType::Gles => {
                        self.gl.map_buffer_range(target, 0, size as _, gl::MAP_WRITE_BIT)
                    }
                };
                assert!(!ptr.is_null());

                let buffer_slice = unsafe {
                    slice::from_raw_parts_mut(ptr as *mut u8, size)
                };
                let repeated_stride = instance_stride * count;
                for (dst, instance) in buffer_slice.chunks_mut(repeated_stride).zip(instances.chunks(instance_stride)) {
                    for copy in dst.chunks_mut(instance_stride) {
                        copy.copy_from_slice(instance);
                    }
                }
                self.gl.unmap_buffer(target);
            }
            None => {
                self.update_vbo_data(vao.instance_vbo_id, instances, usage_hint);
            }
        }

        // On some devices the VAO must be manually unbound and rebound after an attached buffer has
        // been orphaned. Failure to do so appeared to result in the orphaned buffer's contents
        // being used for the subsequent draw call, rather than the new buffer's contents.
        if self.capabilities.requires_vao_rebind_after_orphaning {
            self.bind_vao_impl(0);
            self.bind_vao_impl(vao.id);
        }
    }

    fn update_vao_indices(&mut self, vao: &VAO, indices: &[u8], usage_hint: VertexUsageHint) {
        debug_assert!(self.inside_frame);
        debug_assert_eq!(self.bound_vao, vao.id);

        vao.ibo_id.bind(self.gl());
        gl::buffer_data(
            self.gl(),
            gl::ELEMENT_ARRAY_BUFFER,
            indices,
            usage_hint.to_gl(),
        );
    }

    fn reallocate_vbo(&mut self, vbo: VBOId, size: usize) {
        debug_assert!(self.inside_frame);

        vbo.bind(self.gl());
        self.gl.buffer_data_untyped(
            gl::ARRAY_BUFFER,
            size as _,
            ptr::null(),
            VertexUsageHint::Stream.to_gl(),
        );
    }

    fn update_vbo_data_unsynchronized(&mut self, vbo: VBOId, data: &[u8], offset: usize) {
        debug_assert!(self.inside_frame);

        let size = data.len();
        vbo.bind(self.gl());
        let ptr = self.gl.map_buffer_range(
            gl::ARRAY_BUFFER,
            offset as _,
            size as _,
            gl::MAP_WRITE_BIT | gl::MAP_UNSYNCHRONIZED_BIT,
        );
        assert!(!ptr.is_null());

        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, size);
        }

        self.gl.unmap_buffer(gl::ARRAY_BUFFER);
    }

    fn draw_triangles_u32(&mut self, first_vertex: i32, index_count: i32) {
        debug_assert!(self.inside_frame);
        debug_assert!(self.current_render_pass.is_some(), "draw outside of a render pass");
        debug_assert!(self.bound_program != 0, "draw without a bound pipeline");

        let _guard = if self.annotate_draw_call_crashes {
            Some(CrashAnnotatorGuard::new(
                &self.crash_annotator,
                CrashAnnotation::DrawShader,
                &self.bound_program_name,
            ))
        } else {
            None
        };

        self.gl.draw_elements(
            gl::TRIANGLES,
            index_count,
            gl::UNSIGNED_INT,
            first_vertex as u32 * 4,
        );
    }

    fn draw_nonindexed_lines(&mut self, first_vertex: i32, vertex_count: i32) {
        debug_assert!(self.inside_frame);
        debug_assert!(self.current_render_pass.is_some(), "draw outside of a render pass");
        debug_assert!(self.bound_program != 0, "draw without a bound pipeline");

        let _guard = if self.annotate_draw_call_crashes {
            Some(CrashAnnotatorGuard::new(
                &self.crash_annotator,
                CrashAnnotation::DrawShader,
                &self.bound_program_name,
            ))
        } else {
            None
        };

        self.gl.draw_arrays(gl::LINES, first_vertex, vertex_count);
    }

    fn draw_indexed_triangles(&mut self, index_count: i32) {
        debug_assert!(self.inside_frame);
        debug_assert!(self.current_render_pass.is_some(), "draw outside of a render pass");
        debug_assert!(self.bound_program != 0, "draw without a bound pipeline");

        let _guard = if self.annotate_draw_call_crashes {
            Some(CrashAnnotatorGuard::new(
                &self.crash_annotator,
                CrashAnnotation::DrawShader,
                &self.bound_program_name,
            ))
        } else {
            None
        };

        self.gl.draw_elements(
            gl::TRIANGLES,
            index_count,
            gl::UNSIGNED_SHORT,
            0,
        );
    }

    fn draw_indexed_triangles_instanced_u16(&mut self, index_count: i32, instance_count: i32) {
        debug_assert!(self.inside_frame);
        debug_assert!(self.current_render_pass.is_some(), "draw outside of a render pass");
        debug_assert!(self.bound_program != 0, "draw without a bound pipeline");

        let _guard = if self.annotate_draw_call_crashes {
            Some(CrashAnnotatorGuard::new(
                &self.crash_annotator,
                CrashAnnotation::DrawShader,
                &self.bound_program_name,
            ))
        } else {
            None
        };

        self.gl.draw_elements_instanced(
            gl::TRIANGLES,
            index_count,
            gl::UNSIGNED_SHORT,
            0,
            instance_count,
        );
    }

    fn draw_indexed_triangles_instanced_base_instance_u16(
        &mut self,
        index_count: i32,
        instance_count: i32,
        base_instance: u32,
    ) {
        debug_assert!(self.inside_frame);
        debug_assert!(self.current_render_pass.is_some(), "draw outside of a render pass");
        debug_assert!(self.bound_program != 0, "draw without a bound pipeline");

        let _guard = if self.annotate_draw_call_crashes {
            Some(CrashAnnotatorGuard::new(
                &self.crash_annotator,
                CrashAnnotation::DrawShader,
                &self.bound_program_name,
            ))
        } else {
            None
        };

        self.gl.draw_elements_instanced_base_instance(
            gl::TRIANGLES,
            index_count,
            gl::UNSIGNED_SHORT,
            0,
            instance_count,
            base_instance,
        );
    }

    fn deinit(&mut self) {
        debug_assert!(self.inside_frame);
        if let Some(fbo) = self.scratch_read_fbo.take() {
            self.delete_fbo(fbo);
        }
    }

    fn end_frame(&mut self) {
        self.reset_draw_target();
        self.reset_read_target();

        debug_assert!(self.inside_frame);
        debug_assert!(self.current_render_pass.is_none(), "render pass still in progress");
        self.inside_frame = false;

        self.gl.bind_texture(gl::TEXTURE_2D, 0);
        self.gl.use_program(0);

        for i in 0 .. self.bound_textures.len() {
            self.gl.active_texture(gl::TEXTURE0 + i as gl::GLuint);
            self.gl.bind_texture(gl::TEXTURE_2D, 0);
        }

        self.gl.active_texture(gl::TEXTURE0);

        self.frame_id.0 += 1;

        // Save any shaders compiled this frame to disk.
        // If this is the tenth frame then treat startup as complete, meaning the
        // current set of in-use shaders are the ones to load on the next startup.
        if let Some(ref cache) = self.cached_programs {
            cache.update_disk_cache(self.frame_id.0 == 10);
        }
    }

    fn clear_target(
        &mut self,
        color: Option<[f32; 4]>,
        depth: Option<f32>,
        rect: Option<FramebufferIntRect>,
    ) {
        debug_assert!(self.current_render_pass.is_some(), "clear outside of a render pass");
        self.clear_target_impl(color, depth, rect);
    }

    fn set_scissor_rect(&self, rect: FramebufferIntRect) {
        self.gl.scissor(
            rect.min.x,
            rect.min.y,
            rect.width(),
            rect.height(),
        );
    }

    fn enable_scissor(&self) {
        self.gl.enable(gl::SCISSOR_TEST);
    }

    fn disable_scissor(&self) {
        self.gl.disable(gl::SCISSOR_TEST);
    }

    fn echo_driver_messages(&self) {
        if self.capabilities.supports_khr_debug {
            GlDevice::log_driver_messages(self.gl());
        }
    }

    fn report_memory(&self, size_op_funs: &MallocSizeOfOps, swgl: *mut c_void) -> MemoryReport {
        let mut report = MemoryReport::default();
        report.depth_target_textures += self.depth_targets_memory();

        #[cfg(feature = "sw_compositor")]
        if !swgl.is_null() {
            report.swgl += swgl::Context::from(swgl).report_memory(size_op_funs.size_of_op);
        }
        // unconditionally use swgl stuff
        let _ = size_op_funs;
        let _ = swgl;
        report
    }

    fn depth_targets_memory(&self) -> usize {
        let mut total = 0;
        for dim in self.depth_targets.keys() {
            total += depth_target_size_in_bytes(dim);
        }

        total
    }

    fn create_transfer_buffer(&mut self) -> TransferBuffer {
        let id = self.gl.gen_buffers(1)[0];
        TransferBuffer {
            id,
            reserved_size: 0,
        }
    }

    /// Returns the size and stride in bytes required to upload an area of pixels
    /// of the specified size, to a texture of the specified format.
    fn required_upload_size_and_stride(&self, size: DeviceIntSize, format: ImageFormat) -> (usize, usize) {
        assert!(size.width >= 0);
        assert!(size.height >= 0);

        let bytes_pp = format.bytes_per_pixel() as usize;
        let width_bytes = size.width as usize * bytes_pp;

        let dst_stride = round_up_to_multiple(width_bytes, self.required_transfer_stride.num_bytes(format));

        // The size of the chunk should only need to be (height - 1) * dst_stride + width_bytes,
        // however, the android emulator will error unless it is height * dst_stride.
        // See bug 1587047 for details.
        // Using the full final row also ensures that the offset of the next chunk is
        // optimally aligned.
        let dst_size = dst_stride * size.height as usize;

        (dst_size, dst_stride)
    }
}

pub struct FormatDesc {
    /// Format the texel data is internally stored in within a texture.
    pub internal: gl::GLenum,
    /// Format that we expect the data to be provided when filling the texture.
    pub external: gl::GLuint,
    /// Format to read the texels as, so that they can be uploaded as `external`
    /// later on.
    pub read: gl::GLuint,
    /// Associated pixel type.
    pub pixel_type: gl::GLuint,
}

