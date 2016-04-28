/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use core::nonzero::NonZero;
use gleam::gl;
use ipc_channel::ipc::IpcSender;
use offscreen_gl_context::{GLContext, GLContextAttributes, NativeGLContextMethods};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WebGLContextId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum WebGLError {
    InvalidEnum,
    InvalidOperation,
    InvalidValue,
    OutOfMemory,
    ContextLost,
}

pub type WebGLResult<T> = Result<T, WebGLError>;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum WebGLFramebufferBindingRequest {
    Explicit(u32),
    Default,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum WebGLParameter {
    Int(i32),
    Bool(bool),
    String(String),
    Float(f32),
    FloatArray(Vec<f32>),
    Invalid,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
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
    DetachShader(u32, u32),
    BindAttribLocation(u32, u32, String),
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
    CopyTexImage2D(u32, i32, u32, i32, i32, i32, i32, i32),
    CopyTexSubImage2D(u32, i32, i32, i32, i32, i32, i32, i32),
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
    DrawElements(u32, i32, u32, i64),
    EnableVertexAttribArray(u32),
    GetBufferParameter(u32, u32, IpcSender<WebGLResult<WebGLParameter>>),
    GetParameter(u32, IpcSender<WebGLResult<WebGLParameter>>),
    GetProgramParameter(u32, u32, IpcSender<WebGLResult<WebGLParameter>>),
    GetShaderParameter(u32, u32, IpcSender<WebGLResult<WebGLParameter>>),
    GetActiveAttrib(u32, u32, IpcSender<WebGLResult<(i32, u32, String)>>),
    GetActiveUniform(u32, u32, IpcSender<WebGLResult<(i32, u32, String)>>),
    GetAttribLocation(u32, String, IpcSender<Option<i32>>),
    GetUniformLocation(u32, String, IpcSender<Option<i32>>),
    GetVertexAttrib(u32, u32, IpcSender<WebGLResult<WebGLParameter>>),
    PolygonOffset(f32, f32),
    ReadPixels(i32, i32, i32, i32, u32, u32, IpcSender<Vec<u8>>),
    Scissor(i32, i32, i32, i32),
    StencilFunc(u32, i32, u32),
    StencilFuncSeparate(u32, u32, i32, u32),
    StencilMask(u32),
    StencilMaskSeparate(u32, u32),
    StencilOp(u32, u32, u32),
    StencilOpSeparate(u32, u32, u32, u32),
    Hint(u32, u32),
    LineWidth(f32),
    PixelStorei(u32, i32),
    LinkProgram(u32),
    Uniform1f(i32, f32),
    Uniform1fv(i32, Vec<f32>),
    Uniform1i(i32, i32),
    Uniform1iv(i32, Vec<i32>),
    Uniform2f(i32, f32, f32),
    Uniform2fv(i32, Vec<f32>),
    Uniform2i(i32, i32, i32),
    Uniform2iv(i32, Vec<i32>),
    Uniform3f(i32, f32, f32, f32),
    Uniform3fv(i32, Vec<f32>),
    Uniform3i(i32, i32, i32, i32),
    Uniform3iv(i32, Vec<i32>),
    Uniform4f(i32, f32, f32, f32, f32),
    Uniform4fv(i32, Vec<f32>),
    Uniform4i(i32, i32, i32, i32, i32),
    Uniform4iv(i32, Vec<i32>),
    UseProgram(u32),
    VertexAttrib(u32, f32, f32, f32, f32),
    VertexAttribPointer2f(u32, i32, bool, i32, u32),
    Viewport(i32, i32, i32, i32),
    TexImage2D(u32, i32, i32, i32, i32, u32, u32, Vec<u8>),
    TexParameteri(u32, u32, i32),
    TexParameterf(u32, u32, f32),
    TexSubImage2D(u32, i32, i32, i32, i32, i32, u32, u32, Vec<u8>),
    DrawingBufferWidth(IpcSender<i32>),
    DrawingBufferHeight(IpcSender<i32>),
    Finish(IpcSender<()>),
    Flush,
    GenerateMipmap(u32),
}

impl fmt::Debug for WebGLCommand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use WebGLCommand::*;
        let name = match *self {
            GetContextAttributes(..) => "GetContextAttributes",
            ActiveTexture(..) => "ActiveTexture",
            BlendColor(..) => "BlendColor",
            BlendEquation(..) => "BlendEquation",
            BlendEquationSeparate(..) => "BlendEquationSeparate",
            BlendFunc(..) => "BlendFunc",
            BlendFuncSeparate(..) => "BlendFuncSeparate",
            AttachShader(..) => "AttachShader",
            DetachShader(..) => "DetachShader",
            BindAttribLocation(..) => "BindAttribLocation",
            BufferData(..) => "BufferData",
            BufferSubData(..) => "BufferSubData",
            Clear(..) => "Clear",
            ClearColor(..) => "ClearColor",
            ClearDepth(..) => "ClearDepth",
            ClearStencil(..) => "ClearStencil",
            ColorMask(..) => "ColorMask",
            CopyTexImage2D(..) => "CopyTexImage2D",
            CopyTexSubImage2D(..) => "CopyTexSubImage2D",
            CullFace(..) => "CullFace",
            FrontFace(..) => "FrontFace",
            DepthFunc(..) => "DepthFunc",
            DepthMask(..) => "DepthMask",
            DepthRange(..) => "DepthRange",
            Enable(..) => "Enable",
            Disable(..) => "Disable",
            CompileShader(..) => "CompileShader",
            CreateBuffer(..) => "CreateBuffer",
            CreateFramebuffer(..) => "CreateFramebuffer",
            CreateRenderbuffer(..) => "CreateRenderbuffer",
            CreateTexture(..) => "CreateTexture",
            CreateProgram(..) => "CreateProgram",
            CreateShader(..) => "CreateShader",
            DeleteBuffer(..) => "DeleteBuffer",
            DeleteFramebuffer(..) => "DeleteFramebuffer",
            DeleteRenderbuffer(..) => "DeleteRenderBuffer",
            DeleteTexture(..) => "DeleteTexture",
            DeleteProgram(..) => "DeleteProgram",
            DeleteShader(..) => "DeleteShader",
            BindBuffer(..) => "BindBuffer",
            BindFramebuffer(..) => "BindFramebuffer",
            BindRenderbuffer(..) => "BindRenderbuffer",
            BindTexture(..) => "BindTexture",
            DrawArrays(..) => "DrawArrays",
            DrawElements(..) => "DrawElements",
            EnableVertexAttribArray(..) => "EnableVertexAttribArray",
            GetBufferParameter(..) => "GetBufferParameter",
            GetParameter(..) => "GetParameter",
            GetProgramParameter(..) => "GetProgramParameter",
            GetShaderParameter(..) => "GetShaderParameter",
            GetActiveAttrib(..) => "GetActiveAttrib",
            GetActiveUniform(..) => "GetActiveUniform",
            GetAttribLocation(..) => "GetAttribLocation",
            GetUniformLocation(..) => "GetUniformLocation",
            GetVertexAttrib(..) => "GetVertexAttrib",
            PolygonOffset(..) => "PolygonOffset",
            ReadPixels(..) => "ReadPixels",
            Scissor(..) => "Scissor",
            StencilFunc(..) => "StencilFunc",
            StencilFuncSeparate(..) => "StencilFuncSeparate",
            StencilMask(..) => "StencilMask",
            StencilMaskSeparate(..) => "StencilMaskSeparate",
            StencilOp(..) => "StencilOp",
            StencilOpSeparate(..) => "StencilOpSeparate",
            Hint(..) => "Hint",
            LineWidth(..) => "LineWidth",
            PixelStorei(..) => "PixelStorei",
            LinkProgram(..) => "LinkProgram",
            Uniform1f(..) => "Uniform1f",
            Uniform1fv(..) => "Uniform1fv",
            Uniform1i(..) => "Uniform1i",
            Uniform1iv(..) => "Uniform1iv",
            Uniform2f(..) => "Uniform2f",
            Uniform2fv(..) => "Uniform2fv",
            Uniform2i(..) => "Uniform2i",
            Uniform2iv(..) => "Uniform2iv",
            Uniform3f(..) => "Uniform3f",
            Uniform3fv(..) => "Uniform3fv",
            Uniform3i(..) => "Uniform3i",
            Uniform3iv(..) => "Uniform3iv",
            Uniform4f(..) => "Uniform4f",
            Uniform4fv(..) => "Uniform4fv",
            Uniform4i(..) => "Uniform4i",
            Uniform4iv(..) => "Uniform4iv",
            UseProgram(..) => "UseProgram",
            VertexAttrib(..) => "VertexAttrib",
            VertexAttribPointer2f(..) => "VertexAttribPointer2f",
            Viewport(..) => "Viewport",
            TexImage2D(..) => "TexImage2D",
            TexParameteri(..) => "TexParameteri",
            TexParameterf(..) => "TexParameterf",
            TexSubImage2D(..) => "TexSubImage2D",
            DrawingBufferWidth(..) => "DrawingBufferWidth",
            DrawingBufferHeight(..) => "DrawingBufferHeight",
            Finish(..) => "Finish",
            Flush => "Flush",
            GenerateMipmap(..) => "GenerateMipmap",
        };

        write!(f, "CanvasWebGLMsg::{}(..)", name)
    }
}

impl WebGLCommand {
    /// NOTE: This method consumes the command
    pub fn apply<Native: NativeGLContextMethods>(self, ctx: &GLContext<Native>) {
        match self {
            WebGLCommand::GetContextAttributes(sender) =>
                sender.send(*ctx.borrow_attributes()).unwrap(),
            WebGLCommand::ActiveTexture(target) =>
                gl::active_texture(target),
            WebGLCommand::AttachShader(program_id, shader_id) =>
                gl::attach_shader(program_id, shader_id),
            WebGLCommand::DetachShader(program_id, shader_id) =>
                gl::detach_shader(program_id, shader_id),
            WebGLCommand::BindAttribLocation(program_id, index, name) =>
                gl::bind_attrib_location(program_id, index, &name),
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
            WebGLCommand::CopyTexImage2D(target, level, internal_format, x, y, width, height, border) =>
                gl::copy_tex_image_2d(target, level, internal_format, x, y, width, height, border),
            WebGLCommand::CopyTexSubImage2D(target, level, xoffset, yoffset, x, y, width, height) =>
                gl::copy_tex_sub_image_2d(target, level, xoffset, yoffset, x, y, width, height),
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
            WebGLCommand::DrawElements(mode, count, type_, offset) =>
                gl::draw_elements(mode, count, type_, offset as u32),
            WebGLCommand::EnableVertexAttribArray(attrib_id) =>
                gl::enable_vertex_attrib_array(attrib_id),
            WebGLCommand::Hint(name, val) =>
                gl::hint(name, val),
            WebGLCommand::LineWidth(width) =>
                gl::line_width(width),
            WebGLCommand::PixelStorei(name, val) =>
                gl::pixel_store_i(name, val),
            WebGLCommand::PolygonOffset(factor, units) =>
                gl::polygon_offset(factor, units),
            WebGLCommand::ReadPixels(x, y, width, height, format, pixel_type, chan) =>
                Self::read_pixels(x, y, width, height, format, pixel_type, chan),
            WebGLCommand::Scissor(x, y, width, height) =>
                gl::scissor(x, y, width, height),
            WebGLCommand::StencilFunc(func, ref_, mask) =>
                gl::stencil_func(func, ref_, mask),
            WebGLCommand::StencilFuncSeparate(face, func, ref_, mask) =>
                gl::stencil_func_separate(face, func, ref_, mask),
            WebGLCommand::StencilMask(mask) =>
                gl::stencil_mask(mask),
            WebGLCommand::StencilMaskSeparate(face, mask) =>
                gl::stencil_mask_separate(face, mask),
            WebGLCommand::StencilOp(fail, zfail, zpass) =>
                gl::stencil_op(fail, zfail, zpass),
            WebGLCommand::StencilOpSeparate(face, fail, zfail, zpass) =>
                gl::stencil_op_separate(face, fail, zfail, zpass),
            WebGLCommand::GetActiveAttrib(program_id, index, chan) =>
                Self::active_attrib(program_id, index, chan),
            WebGLCommand::GetActiveUniform(program_id, index, chan) =>
                Self::active_uniform(program_id, index, chan),
            WebGLCommand::GetAttribLocation(program_id, name, chan) =>
                Self::attrib_location(program_id, name, chan),
            WebGLCommand::GetVertexAttrib(index, pname, chan) =>
                Self::vertex_attrib(index, pname, chan),
            WebGLCommand::GetBufferParameter(target, param_id, chan) =>
                Self::buffer_parameter(target, param_id, chan),
            WebGLCommand::GetParameter(param_id, chan) =>
                Self::parameter(param_id, chan),
            WebGLCommand::GetProgramParameter(program_id, param_id, chan) =>
                Self::program_parameter(program_id, param_id, chan),
            WebGLCommand::GetShaderParameter(shader_id, param_id, chan) =>
                Self::shader_parameter(shader_id, param_id, chan),
            WebGLCommand::GetUniformLocation(program_id, name, chan) =>
                Self::uniform_location(program_id, name, chan),
            WebGLCommand::CompileShader(shader_id, source) =>
                Self::compile_shader(shader_id, source),
            WebGLCommand::CreateBuffer(chan) =>
                Self::create_buffer(chan),
            WebGLCommand::CreateFramebuffer(chan) =>
                Self::create_framebuffer(chan),
            WebGLCommand::CreateRenderbuffer(chan) =>
                Self::create_renderbuffer(chan),
            WebGLCommand::CreateTexture(chan) =>
                Self::create_texture(chan),
            WebGLCommand::CreateProgram(chan) =>
                Self::create_program(chan),
            WebGLCommand::CreateShader(shader_type, chan) =>
                Self::create_shader(shader_type, chan),
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
                Self::bind_framebuffer(target, request, ctx),
            WebGLCommand::BindRenderbuffer(target, id) =>
                gl::bind_renderbuffer(target, id),
            WebGLCommand::BindTexture(target, id) =>
                gl::bind_texture(target, id),
            WebGLCommand::LinkProgram(program_id) =>
                gl::link_program(program_id),
            WebGLCommand::Uniform1f(uniform_id, v) =>
                gl::uniform_1f(uniform_id, v),
            WebGLCommand::Uniform1fv(uniform_id, v) =>
                gl::uniform_1fv(uniform_id, &v),
            WebGLCommand::Uniform1i(uniform_id, v) =>
                gl::uniform_1i(uniform_id, v),
            WebGLCommand::Uniform1iv(uniform_id, v) =>
                gl::uniform_1iv(uniform_id, &v),
            WebGLCommand::Uniform2f(uniform_id, x, y) =>
                gl::uniform_2f(uniform_id, x, y),
            WebGLCommand::Uniform2fv(uniform_id, v) =>
                gl::uniform_2fv(uniform_id, &v),
            WebGLCommand::Uniform2i(uniform_id, x, y) =>
                gl::uniform_2i(uniform_id, x, y),
            WebGLCommand::Uniform2iv(uniform_id, v) =>
                gl::uniform_2iv(uniform_id, &v),
            WebGLCommand::Uniform3f(uniform_id, x, y, z) =>
                gl::uniform_3f(uniform_id, x, y, z),
            WebGLCommand::Uniform3fv(uniform_id, v) =>
                gl::uniform_3fv(uniform_id, &v),
            WebGLCommand::Uniform3i(uniform_id, x, y, z) =>
                gl::uniform_3i(uniform_id, x, y, z),
            WebGLCommand::Uniform3iv(uniform_id, v) =>
                gl::uniform_3iv(uniform_id, &v),
            WebGLCommand::Uniform4f(uniform_id, x, y, z, w) =>
                gl::uniform_4f(uniform_id, x, y, z, w),
            WebGLCommand::Uniform4fv(uniform_id, v) =>
                gl::uniform_4fv(uniform_id, &v),
            WebGLCommand::Uniform4i(uniform_id, x, y, z, w) =>
                gl::uniform_4i(uniform_id, x, y, z, w),
            WebGLCommand::Uniform4iv(uniform_id, v) =>
                gl::uniform_4iv(uniform_id, &v),
            WebGLCommand::UseProgram(program_id) =>
                gl::use_program(program_id),
            WebGLCommand::VertexAttrib(attrib_id, x, y, z, w) =>
                gl::vertex_attrib_4f(attrib_id, x, y, z, w),
            WebGLCommand::VertexAttribPointer2f(attrib_id, size, normalized, stride, offset) =>
                gl::vertex_attrib_pointer_f32(attrib_id, size, normalized, stride, offset),
            WebGLCommand::Viewport(x, y, width, height) =>
                gl::viewport(x, y, width, height),
            WebGLCommand::TexImage2D(target, level, internal, width, height, format, data_type, data) =>
                gl::tex_image_2d(target, level, internal, width, height, /*border*/0, format, data_type, Some(&data)),
            WebGLCommand::TexParameteri(target, name, value) =>
                gl::tex_parameter_i(target, name, value),
            WebGLCommand::TexParameterf(target, name, value) =>
                gl::tex_parameter_f(target, name, value),
            WebGLCommand::TexSubImage2D(target, level, xoffset, yoffset, x, y, width, height, data) =>
                gl::tex_sub_image_2d(target, level, xoffset, yoffset, x, y, width, height, &data),
            WebGLCommand::DrawingBufferWidth(sender) =>
                sender.send(ctx.borrow_draw_buffer().unwrap().size().width).unwrap(),
            WebGLCommand::DrawingBufferHeight(sender) =>
                sender.send(ctx.borrow_draw_buffer().unwrap().size().height).unwrap(),
            WebGLCommand::Finish(sender) =>
                Self::finish(sender),
            WebGLCommand::Flush =>
                gl::flush(),
            WebGLCommand::GenerateMipmap(target) =>
                gl::generate_mipmap(target),
        }

        // FIXME: Use debug_assertions once tests are run with them
        let error = gl::get_error();
        assert!(error == gl::NO_ERROR, "Unexpected WebGL error: 0x{:x} ({})", error, error);
    }

    fn read_pixels(x: i32, y: i32, width: i32, height: i32, format: u32, pixel_type: u32,
                   chan: IpcSender<Vec<u8>>) {
      let result = gl::read_pixels(x, y, width, height, format, pixel_type);
      chan.send(result).unwrap()
    }

    fn active_attrib(program_id: u32,
                     index: u32,
                     chan: IpcSender<WebGLResult<(i32, u32, String)>>) {
        let result = if index >= gl::get_program_iv(program_id, gl::ACTIVE_ATTRIBUTES) as u32 {
            Err(WebGLError::InvalidValue)
        } else {
            Ok(gl::get_active_attrib(program_id, index))
        };
        chan.send(result).unwrap();
    }

    fn active_uniform(program_id: u32,
                      index: u32,
                      chan: IpcSender<WebGLResult<(i32, u32, String)>>) {
        let result = if index >= gl::get_program_iv(program_id, gl::ACTIVE_UNIFORMS) as u32 {
            Err(WebGLError::InvalidValue)
        } else {
            Ok(gl::get_active_uniform(program_id, index))
        };
        chan.send(result).unwrap();
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

    fn parameter(param_id: u32,
                 chan: IpcSender<WebGLResult<WebGLParameter>>) {
        let result = match param_id {
            gl::ACTIVE_TEXTURE |
            //gl::ALPHA_BITS |
            gl::BLEND_DST_ALPHA |
            gl::BLEND_DST_RGB |
            gl::BLEND_EQUATION_ALPHA |
            gl::BLEND_EQUATION_RGB |
            gl::BLEND_SRC_ALPHA |
            gl::BLEND_SRC_RGB |
            //gl::BLUE_BITS |
            gl::CULL_FACE_MODE |
            //gl::DEPTH_BITS |
            gl::DEPTH_FUNC |
            gl::FRONT_FACE |
            //gl::GENERATE_MIPMAP_HINT |
            //gl::GREEN_BITS |
            //gl::IMPLEMENTATION_COLOR_READ_FORMAT |
            //gl::IMPLEMENTATION_COLOR_READ_TYPE |
            gl::MAX_COMBINED_TEXTURE_IMAGE_UNITS |
            gl::MAX_CUBE_MAP_TEXTURE_SIZE |
            //gl::MAX_FRAGMENT_UNIFORM_VECTORS |
            gl::MAX_RENDERBUFFER_SIZE |
            gl::MAX_TEXTURE_IMAGE_UNITS |
            gl::MAX_TEXTURE_SIZE |
            //gl::MAX_VARYING_VECTORS |
            gl::MAX_VERTEX_ATTRIBS |
            gl::MAX_VERTEX_TEXTURE_IMAGE_UNITS |
            //gl::MAX_VERTEX_UNIFORM_VECTORS |
            gl::PACK_ALIGNMENT |
            //gl::RED_BITS |
            gl::SAMPLE_BUFFERS |
            gl::SAMPLES |
            gl::STENCIL_BACK_FAIL |
            gl::STENCIL_BACK_FUNC |
            gl::STENCIL_BACK_PASS_DEPTH_FAIL |
            gl::STENCIL_BACK_PASS_DEPTH_PASS |
            gl::STENCIL_BACK_REF |
            gl::STENCIL_BACK_VALUE_MASK |
            gl::STENCIL_BACK_WRITEMASK |
            //gl::STENCIL_BITS |
            gl::STENCIL_CLEAR_VALUE |
            gl::STENCIL_FAIL |
            gl::STENCIL_FUNC |
            gl::STENCIL_PASS_DEPTH_FAIL |
            gl::STENCIL_PASS_DEPTH_PASS |
            gl::STENCIL_REF |
            gl::STENCIL_VALUE_MASK |
            gl::STENCIL_WRITEMASK |
            gl::SUBPIXEL_BITS |
            gl::UNPACK_ALIGNMENT =>
            //gl::UNPACK_COLORSPACE_CONVERSION_WEBGL =>
                Ok(WebGLParameter::Int(gl::get_integer_v(param_id))),

            gl::BLEND |
            gl::CULL_FACE |
            gl::DEPTH_TEST |
            gl::DEPTH_WRITEMASK |
            gl::DITHER |
            gl::POLYGON_OFFSET_FILL |
            gl::SAMPLE_COVERAGE_INVERT |
            gl::STENCIL_TEST =>
            //gl::UNPACK_FLIP_Y_WEBGL |
            //gl::UNPACK_PREMULTIPLY_ALPHA_WEBGL =>
                Ok(WebGLParameter::Bool(gl::get_boolean_v(param_id) != 0)),

            gl::DEPTH_CLEAR_VALUE |
            gl::LINE_WIDTH |
            gl::POLYGON_OFFSET_FACTOR |
            gl::POLYGON_OFFSET_UNITS |
            gl::SAMPLE_COVERAGE_VALUE =>
                Ok(WebGLParameter::Float(gl::get_float_v(param_id))),

            gl::VERSION => Ok(WebGLParameter::String("WebGL 1.0".to_owned())),
            gl::RENDERER |
            gl::VENDOR => Ok(WebGLParameter::String("Mozilla/Servo".to_owned())),
            gl::SHADING_LANGUAGE_VERSION => Ok(WebGLParameter::String("WebGL GLSL ES 1.0".to_owned())),

            // TODO(zbarsky, emilio): Implement support for the following valid parameters
            // Float32Array
            gl::ALIASED_LINE_WIDTH_RANGE |
            //gl::ALIASED_POINT_SIZE_RANGE |
            //gl::BLEND_COLOR |
            gl::COLOR_CLEAR_VALUE |
            gl::DEPTH_RANGE |

            // WebGLBuffer
            gl::ARRAY_BUFFER_BINDING |
            gl::ELEMENT_ARRAY_BUFFER_BINDING |

            // WebGLFrameBuffer
            gl::FRAMEBUFFER_BINDING |

            // WebGLRenderBuffer
            gl::RENDERBUFFER_BINDING |

            // WebGLProgram
            gl::CURRENT_PROGRAM |

            // WebGLTexture
            gl::TEXTURE_BINDING_2D |
            gl::TEXTURE_BINDING_CUBE_MAP |

            // sequence<GlBoolean>
            gl::COLOR_WRITEMASK |

            // Uint32Array
            gl::COMPRESSED_TEXTURE_FORMATS |

            // Int32Array
            gl::MAX_VIEWPORT_DIMS |
            gl::SCISSOR_BOX |
            gl::VIEWPORT => Err(WebGLError::InvalidEnum),

            // Invalid parameters
            _ => Err(WebGLError::InvalidEnum)
        };

        chan.send(result).unwrap();
    }

    fn finish(chan: IpcSender<()>) {
        gl::finish();
        chan.send(()).unwrap();
    }

    fn vertex_attrib(index: u32,
                     pname: u32,
                     chan: IpcSender<WebGLResult<WebGLParameter>>) {
        let result = if index >= gl::get_integer_v(gl::MAX_VERTEX_ATTRIBS) as u32 {
            Err(WebGLError::InvalidValue)
        } else {
            match pname {
                gl::VERTEX_ATTRIB_ARRAY_ENABLED |
                gl::VERTEX_ATTRIB_ARRAY_NORMALIZED =>
                    Ok(WebGLParameter::Bool(gl::get_vertex_attrib_iv(index, pname) != 0)),
                gl::VERTEX_ATTRIB_ARRAY_SIZE |
                gl::VERTEX_ATTRIB_ARRAY_STRIDE |
                gl::VERTEX_ATTRIB_ARRAY_TYPE =>
                    Ok(WebGLParameter::Int(gl::get_vertex_attrib_iv(index, pname))),
                gl::CURRENT_VERTEX_ATTRIB =>
                    Ok(WebGLParameter::FloatArray(gl::get_vertex_attrib_fv(index, pname))),
                // gl::VERTEX_ATTRIB_ARRAY_BUFFER_BINDING should return WebGLBuffer
                _ => Err(WebGLError::InvalidEnum),
            }
        };

        chan.send(result).unwrap();
    }

    fn buffer_parameter(target: u32,
                        param_id: u32,
                        chan: IpcSender<WebGLResult<WebGLParameter>>) {
        let result = match param_id {
            gl::BUFFER_SIZE |
            gl::BUFFER_USAGE =>
                Ok(WebGLParameter::Int(gl::get_buffer_parameter_iv(target, param_id))),
            _ => Err(WebGLError::InvalidEnum),
        };

        chan.send(result).unwrap();
    }

    fn program_parameter(program_id: u32,
                         param_id: u32,
                         chan: IpcSender<WebGLResult<WebGLParameter>>) {
        let result = match param_id {
            gl::DELETE_STATUS |
            gl::LINK_STATUS |
            gl::VALIDATE_STATUS =>
                Ok(WebGLParameter::Bool(gl::get_program_iv(program_id, param_id) != 0)),
            gl::ATTACHED_SHADERS |
            gl::ACTIVE_ATTRIBUTES |
            gl::ACTIVE_UNIFORMS =>
                Ok(WebGLParameter::Int(gl::get_program_iv(program_id, param_id))),
            _ => Err(WebGLError::InvalidEnum),
        };

        chan.send(result).unwrap();
    }

    fn shader_parameter(shader_id: u32,
                        param_id: u32,
                        chan: IpcSender<WebGLResult<WebGLParameter>>) {
        let result = match param_id {
            gl::SHADER_TYPE =>
                Ok(WebGLParameter::Int(gl::get_shader_iv(shader_id, param_id))),
            gl::DELETE_STATUS |
            gl::COMPILE_STATUS =>
                Ok(WebGLParameter::Bool(gl::get_shader_iv(shader_id, param_id) != 0)),
            _ => Err(WebGLError::InvalidEnum),
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
