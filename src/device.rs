use euclid::Matrix4;
use gleam::gl;
use internal_types::{TextureSampler, VertexAttribute, VertexFormat, PackedVertex, RenderTargetMode};
use std::collections::HashMap;
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

impl TextureId {
    fn bind(&self) {
        let TextureId(id) = *self;
        gl::bind_texture(gl::TEXTURE_2D, id);
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
    }
}

struct Texture {
    id: gl::GLuint,
    format: ImageFormat,
    width: u32,
    height: u32,
    fbo_id: Option<FBOId>,
}

impl Drop for Texture {
    fn drop(&mut self) {
        if let Some(fbo_id) = self.fbo_id {
            let FBOId(fbo_id) = fbo_id;
            gl::delete_framebuffers(&[fbo_id]);
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

/*
struct FBO {
    id: gl::GLuint,
    texture_id: gl::GLuint,
}

impl Drop for FBO {
    fn drop(&mut self) {
        println!("TODO: FBO::Drop");
    }
}*/

struct VAO {
    id: gl::GLuint,
    #[cfg(any(target_os = "android", target_os = "gonk", target_os = "macos"))]
    vertex_format: VertexFormat,
    vbo_id: VBOId,
    ibo_id: IBOId,
}

#[cfg(any(target_os = "android", target_os = "gonk", target_os = "macos"))]
impl Drop for VAO {
    fn drop(&mut self) {
        // todo(gw): maybe make these there own type with hashmap?
        let VBOId(vbo_id) = self.vbo_id;
        let IBOId(ibo_id) = self.ibo_id;
        gl::delete_buffers(&[vbo_id]);
        gl::delete_buffers(&[ibo_id]);
    }
}

#[cfg(not(any(target_os = "android", target_os = "gonk", target_os = "macos")))]
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
pub struct ProgramId(gl::GLuint);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct VAOId(gl::GLuint);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct FBOId(gl::GLuint);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
struct VBOId(gl::GLuint);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
struct IBOId(gl::GLuint);

#[derive(Copy, Clone, Debug)]
pub struct UniformLocation(gl::GLint);

pub struct Device {
    // device state
    bound_color_texture: TextureId,
    bound_mask_texture: TextureId,
    bound_program: ProgramId,
    bound_vao: VAOId,
    bound_fbo: FBOId,

    // debug
    inside_frame: bool,

    // resources
    resource_path: PathBuf,
    textures: HashMap<TextureId, Texture>,
    programs: HashMap<ProgramId, Program>,
    vaos: HashMap<VAOId, VAO>,
    //fbos: HashMap<FBOId, FBO>,

    // Used on android only
    #[allow(dead_code)]
    next_vao_id: gl::GLuint,
}

#[cfg(not(any(target_os = "android", target_os = "gonk")))]
fn shader_preamble() -> &'static str {
    ""
}

#[cfg(any(target_os = "android", target_os = "gonk"))]
fn shader_preamble() -> &'static str {
    "#define PLATFORM_ANDROID\n"
}

impl Device {
    pub fn new(resource_path: PathBuf) -> Device {
        Device {
            resource_path: resource_path,
            inside_frame: false,

            bound_color_texture: TextureId(0),
            bound_mask_texture: TextureId(0),
            bound_program: ProgramId(0),
            bound_vao: VAOId(0),
            bound_fbo: FBOId(0),

            textures: HashMap::new(),
            programs: HashMap::new(),
            vaos: HashMap::new(),
            //fbos: HashMap::new(),

            next_vao_id: 0,
        }
    }

    pub fn compile_shader(filename: &str,
                          shader_type: gl::GLenum,
                          resource_path: &PathBuf) -> gl::GLuint {
        let mut path = resource_path.clone();
        path.push(filename);

        println!("compile {:?}", path);

        let mut f = File::open(&path).unwrap();
        let mut s = String::new();
        f.read_to_string(&mut s).unwrap();

        let id = gl::create_shader(shader_type);
        gl::shader_source(id, &[ shader_preamble().as_bytes(), s.as_bytes() ]);
        gl::compile_shader(id);
        if gl::get_shader_iv(id, gl::COMPILE_STATUS) == (0 as gl::GLint) {
            panic!("Failed to compile shader: {}", gl::get_shader_info_log(id));
        }

        id
    }

    pub fn begin_frame(&mut self) {
        debug_assert!(!self.inside_frame);
        self.inside_frame = true;

        // Texture state
        self.bound_color_texture = TextureId(0);
        gl::active_texture(gl::TEXTURE0);
        gl::bind_texture(gl::TEXTURE_2D, 0);

        self.bound_mask_texture = TextureId(0);
        gl::active_texture(gl::TEXTURE1);
        gl::bind_texture(gl::TEXTURE_2D, 0);

        // Shader state
        self.bound_program = ProgramId(0);
        gl::use_program(0);

        // Vertex state
        self.bound_vao = VAOId(0);
        self.clear_vertex_array();

        // FBO state
        self.bound_fbo = FBOId(0);
        gl::bind_framebuffer(gl::FRAMEBUFFER, 0);

        // Pixel op state
        gl::pixel_store_i(gl::UNPACK_ALIGNMENT, 1);

        // Default is sampler 0, always
        gl::active_texture(gl::TEXTURE0);
    }

    pub fn bind_color_texture(&mut self, texture_id: TextureId) {
        debug_assert!(self.inside_frame);

        if self.bound_color_texture != texture_id {
            self.bound_color_texture = texture_id;
            texture_id.bind();
        }
    }

    pub fn bind_mask_texture(&mut self, texture_id: TextureId) {
        debug_assert!(self.inside_frame);

        if self.bound_mask_texture != texture_id {
            self.bound_mask_texture = texture_id;
            gl::active_texture(gl::TEXTURE1);
            texture_id.bind();
            gl::active_texture(gl::TEXTURE0);
        }
    }

    pub fn bind_render_target(&mut self, texture_id: Option<TextureId>) {
        debug_assert!(self.inside_frame);

        let fbo_id = texture_id.map_or(FBOId(0), |texture_id| {
            self.textures.get(&texture_id).unwrap().fbo_id.expect("Binding normal texture as render target!")
        });

        if self.bound_fbo != fbo_id {
            self.bound_fbo = fbo_id;
            fbo_id.bind();
        }
    }

/*
    pub fn bind_fbo(&mut self, fbo_id: FBOId) {
        debug_assert!(self.inside_frame);

        if self.bound_fbo != fbo_id {
            self.bound_fbo = fbo_id;
            fbo_id.bind();
        }
    }
*/

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
                fbo_id: None,
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

    pub fn init_texture(&mut self,
                        texture_id: TextureId,
                        width: u32,
                        height: u32,
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

        self.bind_color_texture(texture_id);

        gl::tex_parameter_i(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as gl::GLint);
        gl::tex_parameter_i(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as gl::GLint);

        match mode {
            RenderTargetMode::RenderTarget => {
                gl::tex_image_2d(gl::TEXTURE_2D, 0, internal_format as gl::GLint, width as gl::GLsizei,
                                 height as gl::GLsizei, 0, gl_format, gl::UNSIGNED_BYTE, None);

                let fbo_id = gl::gen_framebuffers(1)[0];
                gl::bind_framebuffer(gl::FRAMEBUFFER, fbo_id);

                let TextureId(gl_texture_id) = texture_id;
                gl::framebuffer_texture_2d(gl::FRAMEBUFFER,
                                           gl::COLOR_ATTACHMENT0,
                                           gl::TEXTURE_2D,
                                           gl_texture_id,
                                           0);

                gl::bind_framebuffer(gl::FRAMEBUFFER, 0);

                let fbo_id = FBOId(fbo_id);

                // TODO: ugh, messy!
                self.textures.get_mut(&texture_id).unwrap().width = width;
                self.textures.get_mut(&texture_id).unwrap().height = height;
                self.textures.get_mut(&texture_id).unwrap().fbo_id = Some(fbo_id);
            }
            RenderTargetMode::None => {
                gl::tex_image_2d(gl::TEXTURE_2D, 0, internal_format as gl::GLint, width as gl::GLint,
                                 height as gl::GLint, 0, gl_format, gl::UNSIGNED_BYTE, pixels);
            }
        }
    }

/*
    pub fn free_texture(&mut self, _texture_id: TextureId) {
        debug_assert!(self.inside_frame);
        // TODO: Should only really clear the data in the FBO, not
        // remove the texture - since the texture handle is managed
        // by the backend...
        //self.textures.remove(&texture_id).unwrap();
    }
*/

    pub fn create_program(&mut self,
                          vs_filename: &str,
                          fs_filename: &str) -> ProgramId {
        debug_assert!(self.inside_frame);

        let pid = gl::create_program();

        // todo(gw): store shader ids so they can be freed!
        let vs_id = Device::compile_shader(vs_filename, gl::VERTEX_SHADER, &self.resource_path);
        let fs_id = Device::compile_shader(fs_filename, gl::FRAGMENT_SHADER, &self.resource_path);

        gl::attach_shader(pid, vs_id);
        gl::attach_shader(pid, fs_id);

        gl::bind_attrib_location(pid, VertexAttribute::Position as gl::GLuint, "aPosition");
        gl::bind_attrib_location(pid, VertexAttribute::Color as gl::GLuint, "aColor");
        gl::bind_attrib_location(pid, VertexAttribute::ColorTexCoord as gl::GLuint, "aColorTexCoord");
        gl::bind_attrib_location(pid, VertexAttribute::MaskTexCoord as gl::GLuint, "aMaskTexCoord");

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

        program_id
    }

    pub fn get_uniform_location(&self, program_id: ProgramId, name: &str) -> UniformLocation {
        debug_assert!(self.inside_frame);
        let ProgramId(program_id) = program_id;
        UniformLocation(gl::get_uniform_location(program_id, name))
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

    pub fn set_uniforms(&self, program: &Program, transform: &Matrix4) {
        debug_assert!(self.inside_frame);
        gl::uniform_matrix_4fv(program.u_transform, false, &transform.to_array());
    }

    pub fn update_texture(&mut self,
                          texture_id: TextureId,
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

        self.bind_color_texture(texture_id);

        gl::tex_sub_image_2d(gl::TEXTURE_2D,
                             0,
                             x0 as gl::GLint,
                             y0 as gl::GLint,
                             width as gl::GLint,
                             height as gl::GLint,
                             gl_format,
                             gl::UNSIGNED_BYTE,
                             data);
    }

    pub fn read_framebuffer_rect(&mut self,
                                 texture_id: TextureId,
                                 x: u32,
                                 y: u32,
                                 width: u32,
                                 height: u32) {
        self.bind_color_texture(texture_id);
        gl::copy_tex_sub_image_2d(gl::TEXTURE_2D, 0, 0, 0, x as gl::GLint, y as gl::GLint, width as gl::GLint, height as gl::GLint);
    }

    #[cfg(any(target_os = "android", target_os = "gonk", target_os = "macos"))]
    fn clear_vertex_array(&mut self) {
        debug_assert!(self.inside_frame);

        gl::disable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::Color as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::ColorTexCoord as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::MaskTexCoord as gl::GLuint);
    }

    #[cfg(any(target_os = "android", target_os = "gonk", target_os = "macos"))]
    pub fn bind_vao(&mut self, vao_id: VAOId) {
        debug_assert!(self.inside_frame);

        if self.bound_vao != vao_id {
            self.bound_vao = vao_id;

            let vao = self.vaos.get(&vao_id).unwrap();
            vao.vbo_id.bind();
            vao.ibo_id.bind();

            match vao.vertex_format {
                VertexFormat::Default => {
                    let vertex_stride = mem::size_of::<PackedVertex>() as gl::GLint;

                    gl::enable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
                    gl::enable_vertex_attrib_array(VertexAttribute::Color as gl::GLuint);
                    gl::enable_vertex_attrib_array(VertexAttribute::ColorTexCoord as gl::GLuint);
                    gl::enable_vertex_attrib_array(VertexAttribute::MaskTexCoord as gl::GLuint);

                    gl::vertex_attrib_pointer(VertexAttribute::Position as gl::GLuint, 3, gl::FLOAT, false, vertex_stride, 0);
                    gl::vertex_attrib_pointer(VertexAttribute::Color as gl::GLuint, 4, gl::UNSIGNED_BYTE, true, vertex_stride, 12);
                    gl::vertex_attrib_pointer(VertexAttribute::ColorTexCoord as gl::GLuint, 2, gl::UNSIGNED_SHORT, true, vertex_stride, 16);
                    gl::vertex_attrib_pointer(VertexAttribute::MaskTexCoord as gl::GLuint, 2, gl::UNSIGNED_SHORT, true, vertex_stride, 20);
                }
    /*            VertexFormat::Debug => {
                    let vertex_stride = mem::size_of::<DebugVertex>() as gl::GLint;

                    gl::enable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
                    gl::enable_vertex_attrib_array(VertexAttribute::Color as gl::GLuint);

                    gl::vertex_attrib_pointer(VertexAttribute::Position as gl::GLuint, 2, gl::FLOAT, false, vertex_stride, 0);
                    gl::vertex_attrib_pointer(VertexAttribute::Color as gl::GLuint, 4, gl::UNSIGNED_BYTE, true, vertex_stride, 8);
                }*/
            }
        }
    }

    #[cfg(any(target_os = "android", target_os = "gonk", target_os = "macos"))]
    pub fn create_vao(&mut self, vertex_format: VertexFormat) -> VAOId {
        debug_assert!(self.inside_frame);

        let vao_id = self.next_vao_id;
        self.next_vao_id += 1;
        let buffer_ids = gl::gen_buffers(2);

        let vbo_id = buffer_ids[0];
        let ibo_id = buffer_ids[1];

        let vbo_id = VBOId(vbo_id);
        let ibo_id = IBOId(ibo_id);

        let vao = VAO {
            id: vao_id,
            vertex_format: vertex_format,
            vbo_id: vbo_id,
            ibo_id: ibo_id,
        };

        let vao_id = VAOId(vao_id);

        debug_assert!(self.vaos.contains_key(&vao_id) == false);
        self.vaos.insert(vao_id, vao);

        vao_id
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk", target_os = "macos")))]
    fn clear_vertex_array(&mut self) {
        debug_assert!(self.inside_frame);
        gl::bind_vertex_array(0);
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk", target_os = "macos")))]
    pub fn bind_vao(&mut self, vao_id: VAOId) {
        debug_assert!(self.inside_frame);

        if self.bound_vao != vao_id {
            self.bound_vao = vao_id;

            let VAOId(id) = vao_id;
            gl::bind_vertex_array(id);
        }
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk", target_os = "macos")))]
    pub fn create_vao(&mut self, vertex_format: VertexFormat) -> VAOId {
        debug_assert!(self.inside_frame);

        let buffer_ids = gl::gen_buffers(2);
        let vao_ids = gl::gen_vertex_arrays(1);

        let vbo_id = buffer_ids[0];
        let ibo_id = buffer_ids[1];
        let vao_id = vao_ids[0];

        gl::bind_vertex_array(vao_id);
        gl::bind_buffer(gl::ARRAY_BUFFER, vbo_id);
        gl::bind_buffer(gl::ELEMENT_ARRAY_BUFFER, ibo_id);

        match vertex_format {
            VertexFormat::Default => {
                let vertex_stride = mem::size_of::<PackedVertex>() as gl::GLint;

                gl::enable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::Color as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::ColorTexCoord as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::MaskTexCoord as gl::GLuint);

                gl::vertex_attrib_pointer(VertexAttribute::Position as gl::GLuint, 3, gl::FLOAT, false, vertex_stride, 0);
                gl::vertex_attrib_pointer(VertexAttribute::Color as gl::GLuint, 4, gl::UNSIGNED_BYTE, true, vertex_stride, 12);
                gl::vertex_attrib_pointer(VertexAttribute::ColorTexCoord as gl::GLuint, 2, gl::UNSIGNED_SHORT, true, vertex_stride, 16);
                gl::vertex_attrib_pointer(VertexAttribute::MaskTexCoord as gl::GLuint, 2, gl::UNSIGNED_SHORT, true, vertex_stride, 20);
            }
/*            VertexFormat::Debug => {
                let vertex_stride = mem::size_of::<DebugVertex>() as gl::GLint;

                gl::enable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::Color as gl::GLuint);

                gl::vertex_attrib_pointer(VertexAttribute::Position as gl::GLuint, 2, gl::FLOAT, false, vertex_stride, 0);
                gl::vertex_attrib_pointer(VertexAttribute::Color as gl::GLuint, 4, gl::UNSIGNED_BYTE, true, vertex_stride, 8);
            }*/
        }

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

    pub fn update_vao_vertices<V>(&mut self, vao_id: VAOId, vertices: &[V]) {
        debug_assert!(self.inside_frame);

        let vao = self.vaos.get(&vao_id).unwrap();
        debug_assert!(self.bound_vao == vao_id);

        vao.vbo_id.bind();
        gl::buffer_data(gl::ARRAY_BUFFER, &vertices, gl::DYNAMIC_DRAW);
    }

    pub fn update_vao_indices<I>(&mut self, vao_id: VAOId, indices: &[I]) {
        debug_assert!(self.inside_frame);

        let vao = self.vaos.get(&vao_id).unwrap();
        debug_assert!(self.bound_vao == vao_id);

        vao.ibo_id.bind();
        gl::buffer_data(gl::ELEMENT_ARRAY_BUFFER, &indices, gl::DYNAMIC_DRAW);
    }

/*
    pub fn draw_lines_u32(&mut self, index_count: i32) {
        debug_assert!(self.inside_frame);
        gl::draw_elements(gl::LINES, index_count, gl::UNSIGNED_INT, 0);
    }
*/

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

/*
    pub fn create_fbo(&mut self, width: u32, height: u32) -> FBOId {
        let fbo_id = gl::gen_framebuffers(1)[0];
        gl::bind_framebuffer(gl::FRAMEBUFFER, fbo_id);

        let texture_id = gl::gen_textures(1)[0];
        gl::bind_texture(gl::TEXTURE_2D, texture_id);

        gl::tex_image_2d(gl::TEXTURE_2D, 0, gl::RGB as gl::GLint, width as gl::GLsizei,
                         height as gl::GLsizei, 0, gl::RGB, gl::UNSIGNED_BYTE, None);
        gl::tex_parameter_i(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as gl::GLint);
        gl::tex_parameter_i(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as gl::GLint);

        gl::framebuffer_texture_2d(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::TEXTURE_2D,
                                   texture_id, 0);

        gl::bind_texture(gl::TEXTURE_2D, 0);
        gl::bind_framebuffer(gl::FRAMEBUFFER, 0);

        let fbo = FBO {
            id: fbo_id,
            texture_id: texture_id,
        };

        println!("create fbo id={} tex={}", fbo_id, texture_id);

        let fbo_id = FBOId(fbo_id);

        debug_assert!(self.fbos.contains_key(&fbo_id) == false);
        self.fbos.insert(fbo_id, fbo);

        fbo_id
    }

    pub fn delete_fbo(&mut self, fbo_id: FBOId) {
        panic!("todo");
    }*/

    pub fn end_frame(&mut self) {
        debug_assert!(self.inside_frame);
        self.inside_frame = false;

        gl::bind_texture(gl::TEXTURE_2D, 0);
        gl::use_program(0);
    }
}
