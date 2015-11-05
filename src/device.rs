use euclid::Matrix4;
use fnv::FnvHasher;
use gleam::gl;
use internal_types::{PackedVertex, PackedVertexForTextureCacheUpdate, RenderTargetMode};
use internal_types::{TextureSampler, TextureTarget, VertexAttribute};
use std::collections::HashMap;
use std::collections::hash_state::DefaultState;
use std::fs::File;
use std::path::PathBuf;
use std::mem;
use std::io::Read;
use types::ImageFormat;

#[cfg(not(any(target_os = "android", target_os = "gonk")))]
const GL_FORMAT_BGRA: gl::GLuint = gl::BGRA;

#[cfg(not(any(target_os = "android", target_os = "gonk")))]
const GL_FORMAT_A: gl::GLuint = gl::RED;

#[cfg(any(target_os = "android", target_os = "gonk"))]
const GL_FORMAT_BGRA: gl::GLuint = gl::BGRA_EXT;

#[cfg(any(target_os = "android", target_os = "gonk"))]
const GL_FORMAT_A: gl::GLuint = gl::ALPHA;

#[cfg(any(target_os = "android", target_os = "gonk"))]
static FRAGMENT_SHADER_PREAMBLE: &'static str = "es2_common.fs.glsl";

#[cfg(not(any(target_os = "android", target_os = "gonk")))]
static FRAGMENT_SHADER_PREAMBLE: &'static str = "gl3_common.fs.glsl";

#[cfg(any(target_os = "android", target_os = "gonk"))]
static VERTEX_SHADER_PREAMBLE: &'static str = "es2_common.vs.glsl";

#[cfg(not(any(target_os = "android", target_os = "gonk")))]
static VERTEX_SHADER_PREAMBLE: &'static str = "gl3_common.vs.glsl";

impl TextureId {
    fn bind(&self, target: TextureTarget) {
        let TextureId(id) = *self;
        gl::bind_texture(target.to_gl(), id);
    }

    pub fn invalid() -> TextureId {
        TextureId(0)
    }
}

impl ProgramId {
    fn bind(&self) {
        let ProgramId(id) = *self;
        gl::use_program(id);
    }
}

impl VBOId {
    fn bind(&self) {
        let VBOId(id) = *self;
        gl::bind_buffer(gl::ARRAY_BUFFER, id);
    }
}

impl IBOId {
    fn bind(&self) {
        let IBOId(id) = *self;
        gl::bind_buffer(gl::ELEMENT_ARRAY_BUFFER, id);
    }
}

impl FBOId {
    fn bind(&self) {
        let FBOId(id) = *self;
        gl::bind_framebuffer(gl::FRAMEBUFFER, id);
        gl::bind_framebuffer(gl::READ_FRAMEBUFFER, id);
    }
}

struct Texture {
    id: gl::GLuint,
    format: ImageFormat,
    width: u32,
    height: u32,
    fbo_ids: Vec<FBOId>,
}

impl Drop for Texture {
    fn drop(&mut self) {
        if !self.fbo_ids.is_empty() {
            let fbo_ids: Vec<_> = self.fbo_ids.iter().map(|&FBOId(fbo_id)| fbo_id).collect();
            gl::delete_framebuffers(&fbo_ids[..]);
        }
        gl::delete_textures(&[self.id]);
    }
}

struct Program {
    id: gl::GLuint,
    u_transform: gl::GLint,
}

impl Drop for Program {
    fn drop(&mut self) {
        gl::delete_program(self.id);
    }
}

struct VAO {
    id: gl::GLuint,
    vbo_id: VBOId,
    ibo_id: IBOId,
}

#[cfg(any(target_os = "android", target_os = "gonk"))]
impl Drop for VAO {
    fn drop(&mut self) {
        // todo(gw): maybe make these there own type with hashmap?
        let VBOId(vbo_id) = self.vbo_id;
        let IBOId(ibo_id) = self.ibo_id;
        gl::delete_buffers(&[vbo_id]);
        gl::delete_buffers(&[ibo_id]);
    }
}

#[cfg(not(any(target_os = "android", target_os = "gonk")))]
impl Drop for VAO {
    fn drop(&mut self) {
        gl::delete_vertex_arrays(&[self.id]);

        // todo(gw): maybe make these there own type with hashmap?
        let VBOId(vbo_id) = self.vbo_id;
        let IBOId(ibo_id) = self.ibo_id;
        gl::delete_buffers(&[vbo_id]);
        gl::delete_buffers(&[ibo_id]);
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct TextureId(pub gl::GLuint);       // TODO: HACK: Should not be public!

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct TextureIndex(pub u8);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct ProgramId(gl::GLuint);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct VAOId(gl::GLuint);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct FBOId(gl::GLuint);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
struct VBOId(gl::GLuint);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
struct IBOId(gl::GLuint);

#[derive(Debug, Copy, Clone)]
pub enum VertexUsageHint {
    Static,
    Dynamic,
}

impl VertexUsageHint {
    fn to_gl(&self) -> gl::GLuint {
        match *self {
            VertexUsageHint::Static => gl::STATIC_DRAW,
            VertexUsageHint::Dynamic => gl::DYNAMIC_DRAW,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct UniformLocation(gl::GLint);

pub struct Device {
    // device state
    bound_color_texture: TextureId,
    bound_mask_texture: TextureId,
    bound_program: ProgramId,
    bound_vao: VAOId,
    bound_fbo: FBOId,
    default_fbo: gl::GLuint,
    device_pixel_ratio: f32,

    // debug
    inside_frame: bool,

    // resources
    resource_path: PathBuf,
    textures: HashMap<TextureId, Texture, DefaultState<FnvHasher>>,
    programs: HashMap<ProgramId, Program, DefaultState<FnvHasher>>,
    vaos: HashMap<VAOId, VAO, DefaultState<FnvHasher>>,

    // misc.
    vertex_shader_preamble: String,
    fragment_shader_preamble: String,

    // Used on android only
    #[allow(dead_code)]
    next_vao_id: gl::GLuint,
}

impl Device {
    pub fn new(resource_path: PathBuf, device_pixel_ratio: f32) -> Device {
        let mut path = resource_path.clone();
        path.push(VERTEX_SHADER_PREAMBLE);
        let mut f = File::open(&path).unwrap();
        let mut vertex_shader_preamble = String::new();
        f.read_to_string(&mut vertex_shader_preamble).unwrap();

        let mut path = resource_path.clone();
        path.push(FRAGMENT_SHADER_PREAMBLE);
        let mut f = File::open(&path).unwrap();
        let mut fragment_shader_preamble = String::new();
        f.read_to_string(&mut fragment_shader_preamble).unwrap();

        Device {
            resource_path: resource_path,
            device_pixel_ratio: device_pixel_ratio,
            inside_frame: false,

            bound_color_texture: TextureId(0),
            bound_mask_texture: TextureId(0),
            bound_program: ProgramId(0),
            bound_vao: VAOId(0),
            bound_fbo: FBOId(0),
            default_fbo: 0,

            textures: HashMap::with_hash_state(Default::default()),
            programs: HashMap::with_hash_state(Default::default()),
            vaos: HashMap::with_hash_state(Default::default()),

            vertex_shader_preamble: vertex_shader_preamble,
            fragment_shader_preamble: fragment_shader_preamble,

            next_vao_id: 1,
        }
    }

    pub fn compile_shader(filename: &str,
                          shader_type: gl::GLenum,
                          resource_path: &PathBuf,
                          shader_preamble: &str)
                          -> gl::GLuint {
        let mut path = resource_path.clone();
        path.push(filename);

        println!("compile {:?}", path);

        let mut f = File::open(&path).unwrap();
        let mut s = shader_preamble.to_owned();
        f.read_to_string(&mut s).unwrap();

        let id = gl::create_shader(shader_type);
        let mut source = Vec::new();
        source.push_all(s.as_bytes());
        gl::shader_source(id, &[&source[..]]);
        gl::compile_shader(id);
        if gl::get_shader_iv(id, gl::COMPILE_STATUS) == (0 as gl::GLint) {
            panic!("Failed to compile shader: {}", gl::get_shader_info_log(id));
        }

        id
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    fn unbind_2d_texture_array(&mut self) {}

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    fn unbind_2d_texture_array(&mut self) {
        gl::bind_texture(gl::TEXTURE_2D_ARRAY, 0);
    }

    pub fn begin_frame(&mut self) {
        debug_assert!(!self.inside_frame);
        self.inside_frame = true;

        // Retrive the currently set FBO.
        let mut default_fbo = 0;
        gl::get_integer_v(gl::FRAMEBUFFER_BINDING, &mut default_fbo);
        self.default_fbo = default_fbo as gl::GLuint;

        // Texture state
        self.bound_color_texture = TextureId(0);
        gl::active_texture(gl::TEXTURE0);
        gl::bind_texture(gl::TEXTURE_2D, 0);
        self.unbind_2d_texture_array();

        self.bound_mask_texture = TextureId(0);
        gl::active_texture(gl::TEXTURE1);
        gl::bind_texture(gl::TEXTURE_2D, 0);
        self.unbind_2d_texture_array();

        // Shader state
        self.bound_program = ProgramId(0);
        gl::use_program(0);

        // Vertex state
        self.bound_vao = VAOId(0);
        self.clear_vertex_array();

        // FBO state
        self.bound_fbo = FBOId(self.default_fbo);

        // Pixel op state
        gl::pixel_store_i(gl::UNPACK_ALIGNMENT, 1);

        // Default is sampler 0, always
        gl::active_texture(gl::TEXTURE0);
    }

    pub fn bind_color_texture(&mut self, target: TextureTarget, texture_id: TextureId) {
        debug_assert!(self.inside_frame);

        if self.bound_color_texture != texture_id {
            self.bound_color_texture = texture_id;
            texture_id.bind(target);
        }
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    pub fn bind_color_texture_for_noncomposite_operation(&mut self, texture_id: TextureId) {
        self.bind_color_texture(TextureTarget::Texture2D, texture_id);
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    pub fn bind_color_texture_for_noncomposite_operation(&mut self, texture_id: TextureId) {
        self.bind_color_texture(TextureTarget::TextureArray, texture_id);
    }

    pub fn bind_mask_texture(&mut self, target: TextureTarget, texture_id: TextureId) {
        debug_assert!(self.inside_frame);

        if self.bound_mask_texture != texture_id {
            self.bound_mask_texture = texture_id;
            gl::active_texture(gl::TEXTURE1);
            texture_id.bind(target);
            gl::active_texture(gl::TEXTURE0);
        }
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    pub fn bind_mask_texture_for_noncomposite_operation(&mut self, texture_id: TextureId) {
        self.bind_mask_texture(TextureTarget::Texture2D, texture_id);
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    pub fn bind_mask_texture_for_noncomposite_operation(&mut self, texture_id: TextureId) {
        self.bind_mask_texture(TextureTarget::TextureArray, texture_id);
    }

    pub fn bind_render_target(&mut self, texture_id_and_index: Option<(TextureId, TextureIndex)>) {
        debug_assert!(self.inside_frame);

        let fbo_id = texture_id_and_index.map_or(FBOId(self.default_fbo), |texture_id_and_index| {
            self.textures
                .get(&texture_id_and_index.0)
                .unwrap()
                .fbo_ids[(texture_id_and_index.1).0 as usize]
        });

        if self.bound_fbo != fbo_id {
            self.bound_fbo = fbo_id;
            fbo_id.bind();
        }
    }

    pub fn bind_program(&mut self,
                        program_id: ProgramId,
                        projection: &Matrix4) {
        debug_assert!(self.inside_frame);

        if self.bound_program != program_id {
            self.bound_program = program_id;
            program_id.bind();
        }

        let program = self.programs.get(&program_id).unwrap();
        self.set_uniforms(program, projection);
    }

    pub fn create_texture_ids(&mut self, count: i32) -> Vec<TextureId> {
        let id_list = gl::gen_textures(count);
        let mut texture_ids = Vec::new();

        for id in id_list {
            let texture_id = TextureId(id);

            let texture = Texture {
                id: id,
                width: 0,
                height: 0,
                format: ImageFormat::Invalid,
                fbo_ids: vec![],
            };

            debug_assert!(self.textures.contains_key(&texture_id) == false);
            self.textures.insert(texture_id, texture);

            texture_ids.push(texture_id);
        }

        texture_ids
    }

    pub fn get_texture_dimensions(&self, texture_id: TextureId) -> (u32, u32) {
        let texture = self.textures.get(&texture_id).unwrap();
        (texture.width, texture.height)
    }

    fn set_texture_parameters(&mut self, target: TextureTarget) {
        gl::tex_parameter_i(target.to_gl(), gl::TEXTURE_MAG_FILTER, gl::NEAREST as gl::GLint);
        gl::tex_parameter_i(target.to_gl(), gl::TEXTURE_MIN_FILTER, gl::NEAREST as gl::GLint);
    }

    fn upload_2d_texture_image(&mut self,
                               width: u32,
                               height: u32,
                               internal_format: u32,
                               format: u32,
                               pixels: Option<&[u8]>) {
        gl::tex_image_2d(gl::TEXTURE_2D,
                         0,
                         internal_format as gl::GLint,
                         width as gl::GLint, height as gl::GLint,
                         0,
                         format,
                         gl::UNSIGNED_BYTE,
                         pixels);
    }

    fn upload_texture_array_image(&mut self,
                                  width: u32,
                                  height: u32,
                                  levels: u32,
                                  internal_format: u32,
                                  format: u32,
                                  pixels: Option<&[u8]>) {
        gl::tex_image_3d(gl::TEXTURE_2D_ARRAY,
                         0,
                         internal_format as gl::GLint,
                         width as gl::GLint, height as gl::GLint, levels as gl::GLint,
                         0,
                         format,
                         gl::UNSIGNED_BYTE,
                         pixels);
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    fn upload_texture_image(&mut self,
                            target: TextureTarget,
                            width: u32, height: u32, levels: u32,
                            internal_format: u32,
                            format: u32,
                            pixels: Option<&[u8]>) {
        debug_assert!(target == TextureTarget::Texture2D);
        debug_assert!(levels == 1);
        self.upload_2d_texture_image(width, height, internal_format, format, pixels)
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    fn upload_texture_image(&mut self,
                            target: TextureTarget,
                            width: u32, height: u32, levels: u32,
                            internal_format: u32,
                            format: u32,
                            pixels: Option<&[u8]>) {
        match target {
            TextureTarget::Texture2D => {
                debug_assert!(levels == 1);
                self.upload_2d_texture_image(width, height, internal_format, format, pixels)
            }
            TextureTarget::TextureArray => {
                self.upload_texture_array_image(width, height, levels,
                                                internal_format,
                                                format,
                                                pixels)
            }
        }
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    fn deinit_texture_image(&mut self, _: TextureTarget) {
        gl::tex_image_2d(gl::TEXTURE_2D,
                         0,
                         gl::RGB as gl::GLint,
                         0,
                         0,
                         0,
                         gl::RGB,
                         gl::UNSIGNED_BYTE,
                         None);
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    fn deinit_texture_image(&mut self, target: TextureTarget) {
        gl::tex_image_3d(target.to_gl(),
                         0,
                         gl::RGB as gl::GLint,
                         0,
                         0,
                         0,
                         0,
                         gl::RGB,
                         gl::UNSIGNED_BYTE,
                         None);
    }

    pub fn init_texture(&mut self,
                        target: TextureTarget,
                        texture_id: TextureId,
                        width: u32,
                        height: u32,
                        levels: u32,
                        format: ImageFormat,
                        mode: RenderTargetMode,
                        pixels: Option<&[u8]>) {
        debug_assert!(self.inside_frame);

        self.textures.get_mut(&texture_id).unwrap().format = format;

        let (internal_format, gl_format) = match format {
            ImageFormat::A8 => (GL_FORMAT_A, GL_FORMAT_A),
            ImageFormat::RGB8 => (gl::RGB, gl::RGB),
            ImageFormat::RGBA8 => (gl::RGBA, GL_FORMAT_BGRA),
            ImageFormat::Invalid => unreachable!(),
        };

        match mode {
            RenderTargetMode::RenderTarget => {
                self.bind_color_texture(target, texture_id);
                self.set_texture_parameters(target);

                match target {
                    TextureTarget::Texture2D => {
                        debug_assert!(levels == 1);
                        self.upload_2d_texture_image(width,
                                                     height,
                                                     internal_format,
                                                     gl_format,
                                                     None)
                    }
                    TextureTarget::TextureArray => {
                        self.upload_texture_array_image(width,
                                                        height,
                                                        levels,
                                                        internal_format,
                                                        gl_format,
                                                        None)
                    }
                }

                let fbo_ids: Vec<_> =
                    gl::gen_framebuffers(levels as i32).into_iter()
                                                       .map(|fbo_id| FBOId(fbo_id))
                                                       .collect();
                for (i, fbo_id) in fbo_ids.iter().enumerate() {
                    gl::bind_framebuffer(gl::FRAMEBUFFER, fbo_id.0);

                    match target {
                        TextureTarget::Texture2D => {
                            gl::framebuffer_texture_2d(gl::FRAMEBUFFER,
                                                       gl::COLOR_ATTACHMENT0,
                                                       gl::TEXTURE_2D,
                                                       texture_id.0,
                                                       0);
                        }
                        TextureTarget::TextureArray => {
                            gl::framebuffer_texture_layer(gl::FRAMEBUFFER,
                                                          gl::COLOR_ATTACHMENT0,
                                                          texture_id.0,
                                                          0,
                                                          i as gl::GLint);
                        }
                    }
                }

                gl::bind_framebuffer(gl::FRAMEBUFFER, self.default_fbo);

                // TODO: ugh, messy!
                self.textures.get_mut(&texture_id).unwrap().width = width;
                self.textures.get_mut(&texture_id).unwrap().height = height;
                self.textures.get_mut(&texture_id).unwrap().fbo_ids = fbo_ids;
            }
            RenderTargetMode::None => {
                texture_id.bind(target);
                self.set_texture_parameters(target);

                self.upload_texture_image(target,
                                          width, height, levels,
                                          internal_format,
                                          gl_format,
                                          pixels);
            }
        }
    }

    pub fn deinit_texture(&mut self, target: TextureTarget, texture_id: TextureId) {
        debug_assert!(self.inside_frame);

        self.bind_color_texture(target, texture_id);
        self.deinit_texture_image(target);

        let texture = self.textures.get_mut(&texture_id).unwrap();
        if !texture.fbo_ids.is_empty() {
            let fbo_ids: Vec<_> = texture.fbo_ids.iter().map(|&FBOId(fbo_id)| fbo_id).collect();
            gl::delete_framebuffers(&fbo_ids[..]);
        }

        texture.format = ImageFormat::Invalid;
        texture.width = 0;
        texture.height = 0;
        texture.fbo_ids.clear();
    }

    pub fn create_program(&mut self,
                          vs_filename: &str,
                          fs_filename: &str) -> ProgramId {
        debug_assert!(self.inside_frame);

        let pid = gl::create_program();

        // todo(gw): store shader ids so they can be freed!
        let vs_id = Device::compile_shader(vs_filename,
                                           gl::VERTEX_SHADER,
                                           &self.resource_path,
                                           &*self.vertex_shader_preamble);
        let fs_id = Device::compile_shader(fs_filename,
                                           gl::FRAGMENT_SHADER,
                                           &self.resource_path,
                                           &*self.fragment_shader_preamble);

        gl::attach_shader(pid, vs_id);
        gl::attach_shader(pid, fs_id);

        gl::bind_attrib_location(pid, VertexAttribute::Position as gl::GLuint, "aPosition");
        gl::bind_attrib_location(pid, VertexAttribute::Color as gl::GLuint, "aColor");
        gl::bind_attrib_location(pid,
                                 VertexAttribute::ColorTexCoord as gl::GLuint,
                                 "aColorTexCoord");
        gl::bind_attrib_location(pid,
                                 VertexAttribute::MaskTexCoord as gl::GLuint,
                                 "aMaskTexCoord");
        gl::bind_attrib_location(pid, VertexAttribute::BorderRadii as gl::GLuint, "aBorderRadii");
        gl::bind_attrib_location(pid,
                                 VertexAttribute::BorderPosition as gl::GLuint,
                                 "aBorderPosition");
        gl::bind_attrib_location(pid, VertexAttribute::BlurRadius as gl::GLuint, "aBlurRadius");
        gl::bind_attrib_location(pid,
                                 VertexAttribute::DestTextureSize as gl::GLuint,
                                 "aDestTextureSize");
        gl::bind_attrib_location(pid,
                                 VertexAttribute::SourceTextureSize as gl::GLuint,
                                 "aSourceTextureSize");
        gl::bind_attrib_location(pid, VertexAttribute::Misc as gl::GLuint, "aMisc");

        gl::link_program(pid);
        if gl::get_program_iv(pid, gl::LINK_STATUS) == (0 as gl::GLint) {
            panic!("Failed to compile shader program: {}", gl::get_program_info_log(pid));
        }

        let u_transform = gl::get_uniform_location(pid, "uTransform");

        let program_id = ProgramId(pid);

        let program = Program {
            id: pid,
            u_transform: u_transform,
        };

        debug_assert!(self.programs.contains_key(&program_id) == false);
        self.programs.insert(program_id, program);

        program_id.bind();
        let u_diffuse = gl::get_uniform_location(pid, "sDiffuse");
        if u_diffuse != -1 {
            gl::uniform_1i(u_diffuse, TextureSampler::Color as i32);
        }
        let u_mask = gl::get_uniform_location(pid, "sMask");
        if u_mask != -1 {
            gl::uniform_1i(u_mask, TextureSampler::Mask as i32);
        }
        let u_diffuse2d = gl::get_uniform_location(pid, "sDiffuse2D");
        if u_diffuse2d != -1 {
            gl::uniform_1i(u_diffuse2d, TextureSampler::Color as i32);
        }
        let u_mask2d = gl::get_uniform_location(pid, "sMask2D");
        if u_mask2d != -1 {
            gl::uniform_1i(u_mask2d, TextureSampler::Mask as i32);
        }
        let u_device_pixel_ratio = gl::get_uniform_location(pid, "uDevicePixelRatio");
        if u_device_pixel_ratio != -1 {
            gl::uniform_1f(u_device_pixel_ratio, self.device_pixel_ratio);
        }

        program_id
    }

    pub fn get_uniform_location(&self, program_id: ProgramId, name: &str) -> UniformLocation {
        debug_assert!(self.inside_frame);
        let ProgramId(program_id) = program_id;
        UniformLocation(gl::get_uniform_location(program_id, name))
    }

    pub fn set_uniform_2f(&self, uniform: UniformLocation, x: f32, y: f32) {
        debug_assert!(self.inside_frame);
        let UniformLocation(location) = uniform;
        gl::uniform_2f(location, x, y);
    }

    pub fn set_uniform_4f(&self,
                          uniform: UniformLocation,
                          x: f32,
                          y: f32,
                          z: f32,
                          w: f32) {
        debug_assert!(self.inside_frame);
        let UniformLocation(location) = uniform;
        gl::uniform_4f(location, x, y, z, w);
    }

    pub fn set_uniform_mat4_array(&self,
                                  uniform: UniformLocation,
                                  matrices: &[Matrix4]) {
        debug_assert!(self.inside_frame);
        let UniformLocation(location) = uniform;

        // TODO(gw): Avoid alloc here by storing as 3x3 matrices at a higher level...
        let mut floats = Vec::new();
        for matrix in matrices {
            floats.push(matrix.m11);
            floats.push(matrix.m12);
            floats.push(matrix.m13);
            floats.push(matrix.m14);
            floats.push(matrix.m21);
            floats.push(matrix.m22);
            floats.push(matrix.m23);
            floats.push(matrix.m24);
            floats.push(matrix.m31);
            floats.push(matrix.m32);
            floats.push(matrix.m33);
            floats.push(matrix.m34);
            floats.push(matrix.m41);
            floats.push(matrix.m42);
            floats.push(matrix.m43);
            floats.push(matrix.m44);
        }

        gl::uniform_matrix_4fv(location, false, &floats);
    }

    pub fn set_uniforms(&self, program: &Program, transform: &Matrix4) {
        debug_assert!(self.inside_frame);
        gl::uniform_matrix_4fv(program.u_transform, false, &transform.to_array());
    }

    fn update_image_for_2d_texture(&mut self,
                                   x0: gl::GLint,
                                   y0: gl::GLint,
                                   width: gl::GLint,
                                   height: gl::GLint,
                                   format: gl::GLuint,
                                   data: &[u8]) {
        gl::tex_sub_image_2d(gl::TEXTURE_2D,
                             0,
                             x0, y0,
                             width, height,
                             format,
                             gl::UNSIGNED_BYTE,
                             data);
    }

    fn update_image_for_texture_array(&mut self,
                                      x0: gl::GLint,
                                      y0: gl::GLint,
                                      level: gl::GLint,
                                      width: gl::GLint,
                                      height: gl::GLint,
                                      format: gl::GLuint,
                                      data: &[u8]) {
        gl::tex_sub_image_3d(gl::TEXTURE_2D_ARRAY,
                             0,
                             x0, y0, level,
                             width, height, 1,
                             format,
                             gl::UNSIGNED_BYTE,
                             data);
    }

    pub fn update_texture(&mut self,
                          target: TextureTarget,
                          texture_id: TextureId,
                          texture_index: TextureIndex,
                          x0: u32,
                          y0: u32,
                          width: u32,
                          height: u32,
                          data: &[u8]) {
        debug_assert!(self.inside_frame);

        let (gl_format, bpp) = match self.textures.get(&texture_id).unwrap().format {
            ImageFormat::A8 => (GL_FORMAT_A, 1),
            ImageFormat::RGB8 => (gl::RGB, 3),
            ImageFormat::RGBA8 => (GL_FORMAT_BGRA, 4),
            ImageFormat::Invalid => unreachable!(),
        };

        debug_assert!(data.len() as u32 == bpp * width * height);

        self.bind_color_texture(target, texture_id);

        match target {
            TextureTarget::TextureArray => {
                self.update_image_for_texture_array(x0 as gl::GLint,
                                                    y0 as gl::GLint,
                                                    texture_index.0 as gl::GLint,
                                                    width as gl::GLint,
                                                    height as gl::GLint,
                                                    gl_format,
                                                    data);
            }
            TextureTarget::Texture2D => {
                debug_assert!(texture_index == TextureIndex(0));
                self.update_image_for_2d_texture(x0 as gl::GLint,
                                                 y0 as gl::GLint,
                                                 width as gl::GLint,
                                                 height as gl::GLint,
                                                 gl_format,
                                                 data);
            }
        }
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    pub fn update_texture_for_noncomposite_operation(&mut self,
                                                     texture_id: TextureId,
                                                     texture_index: TextureIndex,
                                                     x0: u32,
                                                     y0: u32,
                                                     width: u32,
                                                     height: u32,
                                                     data: &[u8]) {
        self.update_texture(TextureTarget::Texture2D,
                            texture_id,
                            texture_index,
                            x0, y0,
                            width, height,
                            data)
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    pub fn update_texture_for_noncomposite_operation(&mut self,
                                                     texture_id: TextureId,
                                                     texture_index: TextureIndex,
                                                     x0: u32,
                                                     y0: u32,
                                                     width: u32,
                                                     height: u32,
                                                     data: &[u8]) {
        self.update_texture(TextureTarget::TextureArray,
                            texture_id,
                            texture_index,
                            x0, y0,
                            width, height,
                            data)
    }

    fn read_framebuffer_rect_for_2d_texture(&mut self,
                                            texture_id: TextureId,
                                            x: u32, y: u32,
                                            width: u32, height: u32) {
        self.bind_color_texture(TextureTarget::Texture2D, texture_id);
        gl::copy_tex_sub_image_2d(gl::TEXTURE_2D,
                                  0,
                                  0,
                                  0,
                                  x as gl::GLint, y as gl::GLint,
                                  width as gl::GLint, height as gl::GLint);
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    pub fn read_framebuffer_rect(&mut self,
                                 texture_target: TextureTarget,
                                 texture_id: TextureId,
                                 texture_index: TextureIndex,
                                 x: u32,
                                 y: u32,
                                 width: u32,
                                 height: u32) {
        debug_assert!(texture_target == TextureTarget::Texture2D);
        self.read_framebuffer_rect_for_2d_texture(texture_id, x, y, width, height)
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    pub fn read_framebuffer_rect(&mut self,
                                 texture_target: TextureTarget,
                                 texture_id: TextureId,
                                 texture_index: TextureIndex,
                                 x: u32,
                                 y: u32,
                                 width: u32,
                                 height: u32) {
        match texture_target {
            TextureTarget::Texture2D => {
                self.read_framebuffer_rect_for_2d_texture(texture_id, x, y, width, height)
            }
            TextureTarget::TextureArray => {
                self.bind_color_texture(TextureTarget::TextureArray, texture_id);
                gl::copy_tex_sub_image_3d(gl::TEXTURE_2D_ARRAY,
                                          0,
                                          0,
                                          0,
                                          x as gl::GLint,
                                          y as gl::GLint,
                                          texture_index.0 as gl::GLint,
                                          width as gl::GLint,
                                          height as gl::GLint)
            }
        }
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    fn clear_vertex_array(&mut self) {
        debug_assert!(self.inside_frame);

        gl::disable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::Color as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::ColorTexCoord as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::MaskTexCoord as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::BorderRadii as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::BorderPosition as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::BlurRadius as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::DestTextureSize as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::SourceTextureSize as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::Misc as gl::GLuint);
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    pub fn bind_vao(&mut self, vao_id: VAOId) {
        debug_assert!(self.inside_frame);

        if self.bound_vao != vao_id {
            self.bound_vao = vao_id;

            let vao = self.vaos.get(&vao_id).unwrap();
            vao.vbo_id.bind();
            vao.ibo_id.bind();

            let vertex_stride = mem::size_of::<PackedVertex>() as gl::GLint;

            gl::enable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
            gl::enable_vertex_attrib_array(VertexAttribute::Color as gl::GLuint);
            gl::enable_vertex_attrib_array(VertexAttribute::ColorTexCoord as gl::GLuint);
            gl::enable_vertex_attrib_array(VertexAttribute::MaskTexCoord as gl::GLuint);
            gl::enable_vertex_attrib_array(VertexAttribute::Misc as gl::GLuint);

            gl::vertex_attrib_pointer(VertexAttribute::Position as gl::GLuint,
                                      2,
                                      gl::FLOAT,
                                      false,
                                      vertex_stride,
                                      0);
            gl::vertex_attrib_pointer(VertexAttribute::Color as gl::GLuint,
                                      4,
                                      gl::UNSIGNED_BYTE,
                                      false,
                                      vertex_stride,
                                      8);
            gl::vertex_attrib_pointer(VertexAttribute::ColorTexCoord as gl::GLuint,
                                      2,
                                      gl::UNSIGNED_SHORT,
                                      false,
                                      vertex_stride,
                                      12);
            gl::vertex_attrib_pointer(VertexAttribute::MaskTexCoord as gl::GLuint,
                                      2,
                                      gl::UNSIGNED_SHORT,
                                      false,
                                      vertex_stride,
                                      16);
            gl::vertex_attrib_pointer(VertexAttribute::Misc as gl::GLuint,
                                      4,
                                      gl::UNSIGNED_BYTE,
                                      false,
                                      vertex_stride,
                                      20);
        }
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    pub fn bind_vao_for_texture_cache_update(&mut self, vao_id: VAOId) {
        debug_assert!(self.inside_frame);

        if self.bound_vao != vao_id {
            self.bound_vao = vao_id;

            let VAOId(id) = vao_id;
            gl::bind_vertex_array(id);
        }
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    pub fn bind_vao_for_texture_cache_update(&mut self, vao_id: VAOId) {
        debug_assert!(self.inside_frame);

        if self.bound_vao == vao_id {
            return
        }

        self.bound_vao = vao_id;

        let vao = self.vaos.get(&vao_id).unwrap();
        vao.vbo_id.bind();
        vao.ibo_id.bind();

        let vertex_stride = mem::size_of::<PackedVertexForTextureCacheUpdate>() as gl::GLint;

        gl::enable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::Color as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::ColorTexCoord as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::BorderRadii as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::BorderPosition as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::BlurRadius as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::DestTextureSize as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::SourceTextureSize as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::Misc as gl::GLuint);

        gl::vertex_attrib_pointer(VertexAttribute::Position as gl::GLuint,
                                  2,
                                  gl::FLOAT,
                                  false,
                                  vertex_stride,
                                  0);
        gl::vertex_attrib_pointer(VertexAttribute::Color as gl::GLuint,
                                  4,
                                  gl::UNSIGNED_BYTE,
                                  true,
                                  vertex_stride,
                                  8);
        gl::vertex_attrib_pointer(VertexAttribute::ColorTexCoord as gl::GLuint,
                                  2,
                                  gl::UNSIGNED_SHORT,
                                  true,
                                  vertex_stride,
                                  12);
        gl::vertex_attrib_pointer(VertexAttribute::BorderRadii as gl::GLuint,
                                  4,
                                  gl::FLOAT,
                                  false,
                                  vertex_stride,
                                  16);
        gl::vertex_attrib_pointer(VertexAttribute::BorderPosition as gl::GLuint,
                                  4,
                                  gl::FLOAT,
                                  false,
                                  vertex_stride,
                                  32);
        gl::vertex_attrib_pointer(VertexAttribute::DestTextureSize as gl::GLuint,
                                  2,
                                  gl::FLOAT,
                                  false,
                                  vertex_stride,
                                  48);
        gl::vertex_attrib_pointer(VertexAttribute::SourceTextureSize as gl::GLuint,
                                  2,
                                  gl::FLOAT,
                                  false,
                                  vertex_stride,
                                  56);
        gl::vertex_attrib_pointer(VertexAttribute::BlurRadius as gl::GLuint,
                                  1,
                                  gl::FLOAT,
                                  false,
                                  vertex_stride,
                                  64);
        gl::vertex_attrib_pointer(VertexAttribute::Misc as gl::GLuint,
                                  4,
                                  gl::UNSIGNED_BYTE,
                                  false,
                                  vertex_stride,
                                  68);
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    pub fn create_vao(&mut self) -> VAOId {
        debug_assert!(self.inside_frame);

        let vao_id = self.next_vao_id;
        self.next_vao_id += 1;
        let buffer_ids = gl::gen_buffers(2);

        let vbo_id = buffer_ids[0];
        let ibo_id = buffer_ids[1];

        let vbo_id = VBOId(vbo_id);
        let ibo_id = IBOId(ibo_id);

        let vao = VAO {
            //id: vao_id,
            vbo_id: vbo_id,
            ibo_id: ibo_id,
        };

        let vao_id = VAOId(vao_id);

        debug_assert!(self.vaos.contains_key(&vao_id) == false);
        self.vaos.insert(vao_id, vao);

        vao_id
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    fn clear_vertex_array(&mut self) {
        debug_assert!(self.inside_frame);
        gl::bind_vertex_array(0);
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    pub fn bind_vao(&mut self, vao_id: VAOId) {
        debug_assert!(self.inside_frame);

        if self.bound_vao != vao_id {
            self.bound_vao = vao_id;

            let VAOId(id) = vao_id;
            gl::bind_vertex_array(id);
        }
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    pub fn create_vao(&mut self) -> VAOId {
        debug_assert!(self.inside_frame);

        let buffer_ids = gl::gen_buffers(2);
        let vao_ids = gl::gen_vertex_arrays(1);

        let vbo_id = buffer_ids[0];
        let ibo_id = buffer_ids[1];
        let vao_id = vao_ids[0];

        gl::bind_vertex_array(vao_id);
        gl::bind_buffer(gl::ARRAY_BUFFER, vbo_id);
        gl::bind_buffer(gl::ELEMENT_ARRAY_BUFFER, ibo_id);

        let vertex_stride = mem::size_of::<PackedVertex>() as gl::GLint;

        gl::enable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::Color as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::ColorTexCoord as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::MaskTexCoord as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::Misc as gl::GLuint);

        gl::vertex_attrib_pointer(VertexAttribute::Position as gl::GLuint,
                                  2,
                                  gl::FLOAT,
                                  false,
                                  vertex_stride,
                                  0);
        gl::vertex_attrib_pointer(VertexAttribute::Color as gl::GLuint,
                                  4,
                                  gl::UNSIGNED_BYTE,
                                  false,
                                  vertex_stride,
                                  8);
        gl::vertex_attrib_pointer(VertexAttribute::ColorTexCoord as gl::GLuint,
                                  2,
                                  gl::UNSIGNED_SHORT,
                                  false,
                                  vertex_stride,
                                  12);
        gl::vertex_attrib_pointer(VertexAttribute::MaskTexCoord as gl::GLuint,
                                  2,
                                  gl::UNSIGNED_SHORT,
                                  false,
                                  vertex_stride,
                                  16);
        gl::vertex_attrib_pointer(VertexAttribute::Misc as gl::GLuint,
                                  4,
                                  gl::UNSIGNED_BYTE,
                                  false,
                                  vertex_stride,
                                  20);

        gl::bind_vertex_array(0);

        let vbo_id = VBOId(vbo_id);
        let ibo_id = IBOId(ibo_id);

        let vao = VAO {
            id: vao_id,
            vbo_id: vbo_id,
            ibo_id: ibo_id,
        };

        let vao_id = VAOId(vao_id);

        debug_assert!(self.vaos.contains_key(&vao_id) == false);
        self.vaos.insert(vao_id, vao);

        vao_id
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    pub fn create_vao_for_texture_cache_update(&mut self) -> VAOId {
        debug_assert!(self.inside_frame);

        let buffer_ids = gl::gen_buffers(2);
        let vao_ids = gl::gen_vertex_arrays(1);

        let vbo_id = buffer_ids[0];
        let ibo_id = buffer_ids[1];
        let vao_id = vao_ids[0];

        gl::bind_vertex_array(vao_id);
        gl::bind_buffer(gl::ARRAY_BUFFER, vbo_id);
        gl::bind_buffer(gl::ELEMENT_ARRAY_BUFFER, ibo_id);

        let vertex_stride = mem::size_of::<PackedVertexForTextureCacheUpdate>() as gl::GLint;

        gl::enable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::Color as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::ColorTexCoord as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::BorderRadii as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::BorderPosition as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::BlurRadius as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::DestTextureSize as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::SourceTextureSize as gl::GLuint);
        gl::enable_vertex_attrib_array(VertexAttribute::Misc as gl::GLuint);

        gl::vertex_attrib_pointer(VertexAttribute::Position as gl::GLuint,
                                  2,
                                  gl::FLOAT,
                                  false,
                                  vertex_stride,
                                  0);
        gl::vertex_attrib_pointer(VertexAttribute::Color as gl::GLuint,
                                  4,
                                  gl::UNSIGNED_BYTE,
                                  true,
                                  vertex_stride,
                                  8);
        gl::vertex_attrib_pointer(VertexAttribute::ColorTexCoord as gl::GLuint,
                                  2,
                                  gl::UNSIGNED_SHORT,
                                  true,
                                  vertex_stride,
                                  12);
        gl::vertex_attrib_pointer(VertexAttribute::BorderRadii as gl::GLuint,
                                  4,
                                  gl::FLOAT,
                                  false,
                                  vertex_stride,
                                  16);
        gl::vertex_attrib_pointer(VertexAttribute::BorderPosition as gl::GLuint,
                                  4,
                                  gl::FLOAT,
                                  false,
                                  vertex_stride,
                                  32);
        gl::vertex_attrib_pointer(VertexAttribute::DestTextureSize as gl::GLuint,
                                  2,
                                  gl::FLOAT,
                                  false,
                                  vertex_stride,
                                  48);
        gl::vertex_attrib_pointer(VertexAttribute::SourceTextureSize as gl::GLuint,
                                  2,
                                  gl::FLOAT,
                                  false,
                                  vertex_stride,
                                  56);
        gl::vertex_attrib_pointer(VertexAttribute::BlurRadius as gl::GLuint,
                                  1,
                                  gl::FLOAT,
                                  false,
                                  vertex_stride,
                                  64);
        gl::vertex_attrib_pointer(VertexAttribute::Misc as gl::GLuint,
                                  4,
                                  gl::UNSIGNED_BYTE,
                                  false,
                                  vertex_stride,
                                  68);

        gl::bind_vertex_array(0);

        let vbo_id = VBOId(vbo_id);
        let ibo_id = IBOId(ibo_id);

        let vao = VAO {
            id: vao_id,
            vbo_id: vbo_id,
            ibo_id: ibo_id,
        };

        let vao_id = VAOId(vao_id);

        debug_assert!(self.vaos.contains_key(&vao_id) == false);
        self.vaos.insert(vao_id, vao);

        vao_id
    }

    pub fn update_vao_vertices<V>(&mut self,
                                  vao_id: VAOId,
                                  vertices: &[V],
                                  usage_hint: VertexUsageHint) {
        debug_assert!(self.inside_frame);

        let vao = self.vaos.get(&vao_id).unwrap();
        debug_assert!(self.bound_vao == vao_id);

        vao.vbo_id.bind();
        gl::buffer_data(gl::ARRAY_BUFFER, &vertices, usage_hint.to_gl());
    }

    pub fn update_vao_indices<I>(&mut self,
                                 vao_id: VAOId,
                                 indices: &[I],
                                 usage_hint: VertexUsageHint) {
        debug_assert!(self.inside_frame);

        let vao = self.vaos.get(&vao_id).unwrap();
        debug_assert!(self.bound_vao == vao_id);

        vao.ibo_id.bind();
        gl::buffer_data(gl::ELEMENT_ARRAY_BUFFER, &indices, usage_hint.to_gl());
    }

    pub fn draw_triangles_u16(&mut self, index_count: i32) {
        debug_assert!(self.inside_frame);
        gl::draw_elements(gl::TRIANGLES, index_count, gl::UNSIGNED_SHORT, 0);
    }

    pub fn delete_vao(&mut self, vao_id: VAOId) {
        self.vaos.remove(&vao_id).expect(&format!("unable to remove vao {:?}", vao_id));
        if self.bound_vao == vao_id {
            self.bound_vao = VAOId(0);
        }
    }

    pub fn end_frame(&mut self) {
        debug_assert!(self.inside_frame);
        self.inside_frame = false;

        self.unbind_2d_texture_array();
        gl::bind_texture(gl::TEXTURE_2D, 0);
        gl::use_program(0);
    }
}
