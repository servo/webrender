use core::nonzero::NonZero;
use ipc_channel::ipc::IpcSender;
use gleam::gl;
use offscreen_gl_context::{GLContext, GLContextAttributes, NativeGLContextMethods};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WebGLContextId(pub usize);

#[derive(Clone, Copy, PartialEq, Deserialize, Serialize, HeapSizeOf)]
pub enum WebGLError {
    InvalidEnum,
    InvalidOperation,
    InvalidValue,
    OutOfMemory,
    ContextLost,
}

pub type WebGLResult<T> = Result<T, WebGLError>;

#[derive(Clone, Deserialize, Serialize)]
pub enum WebGLFramebufferBindingRequest {
    Explicit(u32),
    Default,
}

#[derive(Clone, Deserialize, Serialize)]
pub enum WebGLShaderParameter {
    Int(i32),
    Bool(bool),
    Invalid,
}

#[derive(Clone, Deserialize, Serialize)]
pub enum WebGLCommand {
    GetContextAttributes(IpcSender<GLContextAttributes>),
    ActiveTexture(u32),
    BlendColor(f32, f32, f32, f32),
    BlendEquation(u32),
    BlendEquationSeparate(u32, u32),
    BlendFunc(u32, u32),
    BlendFuncSeparate(u32, u32, u32, u32),
    AttachShader(u32, u32),
    BufferData(u32, Vec<u8>, u32),
    BufferSubData(u32, isize, Vec<u8>),
    Clear(u32),
    ClearColor(f32, f32, f32, f32),
    ClearDepth(f64),
    ClearStencil(i32),
    ColorMask(bool, bool, bool, bool),
    CullFace(u32),
    FrontFace(u32),
    DepthFunc(u32),
    DepthMask(bool),
    DepthRange(f64, f64),
    Enable(u32),
    Disable(u32),
    CompileShader(u32, String),
    CreateBuffer(IpcSender<Option<NonZero<u32>>>),
    CreateFramebuffer(IpcSender<Option<NonZero<u32>>>),
    CreateRenderbuffer(IpcSender<Option<NonZero<u32>>>),
    CreateTexture(IpcSender<Option<NonZero<u32>>>),
    CreateProgram(IpcSender<Option<NonZero<u32>>>),
    CreateShader(u32, IpcSender<Option<NonZero<u32>>>),
    DeleteBuffer(u32),
    DeleteFramebuffer(u32),
    DeleteRenderbuffer(u32),
    DeleteTexture(u32),
    DeleteProgram(u32),
    DeleteShader(u32),
    BindBuffer(u32, u32),
    BindFramebuffer(u32, WebGLFramebufferBindingRequest),
    BindRenderbuffer(u32, u32),
    BindTexture(u32, u32),
    DrawArrays(u32, i32, i32),
    EnableVertexAttribArray(u32),
    GetShaderParameter(u32, u32, IpcSender<WebGLShaderParameter>),
    GetAttribLocation(u32, String, IpcSender<Option<i32>>),
    GetUniformLocation(u32, String, IpcSender<Option<i32>>),
    PolygonOffset(f32, f32),
    Hint(u32, u32),
    LineWidth(f32),
    PixelStorei(u32, i32),
    LinkProgram(u32),
    Uniform4fv(i32, Vec<f32>),
    UseProgram(u32),
    VertexAttribPointer2f(u32, i32, bool, i32, u32),
    Viewport(i32, i32, i32, i32),
    TexImage2D(u32, i32, i32, i32, i32, u32, u32, Vec<u8>),
    TexParameteri(u32, u32, i32),
    TexParameterf(u32, u32, f32),
    DrawingBufferWidth(IpcSender<i32>),
    DrawingBufferHeight(IpcSender<i32>),
}

impl WebGLCommand {
    /// NOTE: This method consumes the command
    pub fn apply<Native: NativeGLContextMethods>(self, ctx: &GLContext<Native>) {
        match self {
            WebGLCommand::GetContextAttributes(sender) =>
                sender.send(*ctx.borrow_attributes()).unwrap(),
            WebGLCommand::ActiveTexture(target) =>
                gl::active_texture(target),
            WebGLCommand::BlendColor(r, g, b, a) =>
                gl::blend_color(r, g, b, a),
            WebGLCommand::BlendEquation(mode) =>
                gl::blend_equation(mode),
            WebGLCommand::BlendEquationSeparate(mode_rgb, mode_alpha) =>
                gl::blend_equation_separate(mode_rgb, mode_alpha),
            WebGLCommand::BlendFunc(src, dest) =>
                gl::blend_func(src, dest),
            WebGLCommand::BlendFuncSeparate(src_rgb, dest_rgb, src_alpha, dest_alpha) =>
                gl::blend_func_separate(src_rgb, dest_rgb, src_alpha, dest_alpha),
            WebGLCommand::AttachShader(program_id, shader_id) =>
                gl::attach_shader(program_id, shader_id),
            WebGLCommand::BufferData(buffer_type, data, usage) =>
                gl::buffer_data(buffer_type, &data, usage),
            WebGLCommand::BufferSubData(buffer_type, offset, data) =>
                gl::buffer_sub_data(buffer_type, offset, &data),
            WebGLCommand::Clear(mask) =>
                gl::clear(mask),
            WebGLCommand::ClearColor(r, g, b, a) =>
                gl::clear_color(r, g, b, a),
            WebGLCommand::ClearDepth(depth) =>
                gl::clear_depth(depth),
            WebGLCommand::ClearStencil(stencil) =>
                gl::clear_stencil(stencil),
            WebGLCommand::ColorMask(r, g, b, a) =>
                gl::color_mask(r, g, b, a),
            WebGLCommand::CullFace(mode) =>
                gl::cull_face(mode),
            WebGLCommand::DepthFunc(func) =>
                gl::depth_func(func),
            WebGLCommand::DepthMask(flag) =>
                gl::depth_mask(flag),
            WebGLCommand::DepthRange(near, far) =>
                gl::depth_range(near, far),
            WebGLCommand::Disable(cap) =>
                gl::disable(cap),
            WebGLCommand::Enable(cap) =>
                gl::enable(cap),
            WebGLCommand::FrontFace(mode) =>
                gl::front_face(mode),
            WebGLCommand::DrawArrays(mode, first, count) =>
                gl::draw_arrays(mode, first, count),
            WebGLCommand::Hint(name, val) =>
                gl::hint(name, val),
            WebGLCommand::LineWidth(width) =>
                gl::line_width(width),
            WebGLCommand::PixelStorei(name, val) =>
                gl::pixel_store_i(name, val),
            WebGLCommand::PolygonOffset(factor, units) =>
                gl::polygon_offset(factor, units),
            WebGLCommand::EnableVertexAttribArray(attrib_id) =>
                gl::enable_vertex_attrib_array(attrib_id),
            WebGLCommand::GetAttribLocation(program_id, name, chan) =>
                WebGLCommand::attrib_location(program_id, name, chan),
            WebGLCommand::GetShaderParameter(shader_id, param_id, chan) =>
                WebGLCommand::shader_parameter(shader_id, param_id, chan),
            WebGLCommand::GetUniformLocation(program_id, name, chan) =>
                WebGLCommand::uniform_location(program_id, name, chan),
            WebGLCommand::CompileShader(shader_id, source) =>
                WebGLCommand::compile_shader(shader_id, source),
            WebGLCommand::CreateBuffer(chan) =>
                WebGLCommand::create_buffer(chan),
            WebGLCommand::CreateFramebuffer(chan) =>
                WebGLCommand::create_framebuffer(chan),
            WebGLCommand::CreateRenderbuffer(chan) =>
                WebGLCommand::create_renderbuffer(chan),
            WebGLCommand::CreateTexture(chan) =>
                WebGLCommand::create_texture(chan),
            WebGLCommand::CreateProgram(chan) =>
                WebGLCommand::create_program(chan),
            WebGLCommand::CreateShader(shader_type, chan) =>
                WebGLCommand::create_shader(shader_type, chan),
            WebGLCommand::DeleteBuffer(id) =>
                gl::delete_buffers(&[id]),
            WebGLCommand::DeleteFramebuffer(id) =>
                gl::delete_framebuffers(&[id]),
            WebGLCommand::DeleteRenderbuffer(id) =>
                gl::delete_renderbuffers(&[id]),
            WebGLCommand::DeleteTexture(id) =>
                gl::delete_textures(&[id]),
            WebGLCommand::DeleteProgram(id) =>
                gl::delete_program(id),
            WebGLCommand::DeleteShader(id) =>
                gl::delete_shader(id),
            WebGLCommand::BindBuffer(target, id) =>
                gl::bind_buffer(target, id),
            WebGLCommand::BindFramebuffer(target, request) =>
                WebGLCommand::bind_framebuffer(target, request, ctx),
            WebGLCommand::BindRenderbuffer(target, id) =>
                gl::bind_renderbuffer(target, id),
            WebGLCommand::BindTexture(target, id) =>
                gl::bind_texture(target, id),
            WebGLCommand::LinkProgram(program_id) =>
                gl::link_program(program_id),
            WebGLCommand::Uniform4fv(uniform_id, data) =>
                gl::uniform_4f(uniform_id, data[0], data[1], data[2], data[3]),
            WebGLCommand::UseProgram(program_id) =>
                gl::use_program(program_id),
            WebGLCommand::VertexAttribPointer2f(attrib_id, size, normalized, stride, offset) =>
                gl::vertex_attrib_pointer_f32(attrib_id, size, normalized, stride, offset as u32),
            WebGLCommand::Viewport(x, y, width, height) =>
                gl::viewport(x, y, width, height),
            WebGLCommand::TexImage2D(target, level, internal, width, height, format, data_type, data) =>
                gl::tex_image_2d(target, level, internal, width, height, /*border*/0, format, data_type, Some(&data)),
            WebGLCommand::TexParameteri(target, name, value) =>
                gl::tex_parameter_i(target, name, value),
            WebGLCommand::TexParameterf(target, name, value) =>
                gl::tex_parameter_f(target, name, value),
            WebGLCommand::DrawingBufferWidth(sender) =>
                sender.send(ctx.borrow_draw_buffer().unwrap().size().width).unwrap(),
            WebGLCommand::DrawingBufferHeight(sender) =>
                sender.send(ctx.borrow_draw_buffer().unwrap().size().height).unwrap(),
        }

        // FIXME: Convert to `debug_assert!` once tests are run with debug assertions
        assert!(gl::get_error() == gl::NO_ERROR);
    }

    fn attrib_location(program_id: u32,
                       name: String,
                       chan: IpcSender<Option<i32>> ) {
        let attrib_location = gl::get_attrib_location(program_id, &name);

        let attrib_location = if attrib_location == -1 {
            None
        } else {
            Some(attrib_location)
        };

        chan.send(attrib_location).unwrap();
    }

    fn shader_parameter(shader_id: u32,
                        param_id: u32,
                        chan: IpcSender<WebGLShaderParameter>) {
        let result = match param_id {
            gl::SHADER_TYPE =>
                WebGLShaderParameter::Int(gl::get_shader_iv(shader_id, param_id)),
            gl::DELETE_STATUS | gl::COMPILE_STATUS =>
                WebGLShaderParameter::Bool(gl::get_shader_iv(shader_id, param_id) != 0),
            _ => panic!("Unexpected shader parameter type"),
        };

        chan.send(result).unwrap();
    }

    fn uniform_location(program_id: u32,
                        name: String,
                        chan: IpcSender<Option<i32>>) {
        let location = gl::get_uniform_location(program_id, &name);
        let location = if location == -1 {
            None
        } else {
            Some(location)
        };

        chan.send(location).unwrap();
    }

    fn create_buffer(chan: IpcSender<Option<NonZero<u32>>>) {
        let buffer = gl::gen_buffers(1)[0];
        let buffer = if buffer == 0 {
            None
        } else {
            Some(unsafe { NonZero::new(buffer) })
        };
        chan.send(buffer).unwrap();
    }

    fn create_framebuffer(chan: IpcSender<Option<NonZero<u32>>>) {
        let framebuffer = gl::gen_framebuffers(1)[0];
        let framebuffer = if framebuffer == 0 {
            None
        } else {
            Some(unsafe { NonZero::new(framebuffer) })
        };
        chan.send(framebuffer).unwrap();
    }


    fn create_renderbuffer(chan: IpcSender<Option<NonZero<u32>>>) {
        let renderbuffer = gl::gen_renderbuffers(1)[0];
        let renderbuffer = if renderbuffer == 0 {
            None
        } else {
            Some(unsafe { NonZero::new(renderbuffer) })
        };
        chan.send(renderbuffer).unwrap();
    }

    fn create_texture(chan: IpcSender<Option<NonZero<u32>>>) {
        let texture = gl::gen_textures(1)[0];
        let texture = if texture == 0 {
            None
        } else {
            Some(unsafe { NonZero::new(texture) })
        };
        chan.send(texture).unwrap();
    }


    fn create_program(chan: IpcSender<Option<NonZero<u32>>>) {
        let program = gl::create_program();
        let program = if program == 0 {
            None
        } else {
            Some(unsafe { NonZero::new(program) })
        };
        chan.send(program).unwrap();
    }


    fn create_shader(shader_type: u32, chan: IpcSender<Option<NonZero<u32>>>) {
        let shader = gl::create_shader(shader_type);
        let shader = if shader == 0 {
            None
        } else {
            Some(unsafe { NonZero::new(shader) })
        };
        chan.send(shader).unwrap();
    }

    #[inline]
    fn bind_framebuffer<Native: NativeGLContextMethods>(target: u32,
                                                        request: WebGLFramebufferBindingRequest,
                                                        ctx: &GLContext<Native>) {
        let id = match request {
            WebGLFramebufferBindingRequest::Explicit(id) => id,
            WebGLFramebufferBindingRequest::Default =>
                ctx.borrow_draw_buffer().unwrap().get_framebuffer(),
        };

        gl::bind_framebuffer(target, id);
    }


    #[inline]
    fn compile_shader(shader_id: u32, source: String) {
        gl::shader_source(shader_id, &[source.as_bytes()]);
        gl::compile_shader(shader_id);
    }
}
