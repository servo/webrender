/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use channel::MsgSender;
use gleam::gl;
use offscreen_gl_context::{GLContext, NativeGLContextMethods};
use std::fmt;
use {WebGLBufferId, WebGLCommand, WebGLError, WebGLFramebufferBindingRequest};
use {WebGLFramebufferId, WebGLParameter, WebGLProgramId, WebGLRenderbufferId};
use {WebGLResult, WebGLShaderId, WebGLTextureId};

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
            DisableVertexAttribArray(..) => "DisableVertexAttribArray",
            DrawArrays(..) => "DrawArrays",
            DrawElements(..) => "DrawElements",
            EnableVertexAttribArray(..) => "EnableVertexAttribArray",
            FramebufferRenderbuffer(..) => "FramebufferRenderbuffer",
            FramebufferTexture2D(..) => "FramebufferTexture2D",
            GetBufferParameter(..) => "GetBufferParameter",
            GetParameter(..) => "GetParameter",
            GetProgramParameter(..) => "GetProgramParameter",
            GetShaderParameter(..) => "GetShaderParameter",
            GetActiveAttrib(..) => "GetActiveAttrib",
            GetActiveUniform(..) => "GetActiveUniform",
            GetAttribLocation(..) => "GetAttribLocation",
            GetUniformLocation(..) => "GetUniformLocation",
            GetShaderInfoLog(..) => "GetShaderInfoLog",
            GetProgramInfoLog(..) => "GetProgramInfoLog",
            GetVertexAttrib(..) => "GetVertexAttrib",
            PolygonOffset(..) => "PolygonOffset",
            ReadPixels(..) => "ReadPixels",
            RenderbufferStorage(..) => "RenderbufferStorage",
            SampleCoverage(..) => "SampleCoverage",
            Scissor(..) => "Scissor",
            StencilFunc(..) => "StencilFunc",
            StencilFuncSeparate(..) => "StencilFuncSeparate",
            StencilMask(..) => "StencilMask",
            StencilMaskSeparate(..) => "StencilMaskSeparate",
            StencilOp(..) => "StencilOp",
            StencilOpSeparate(..) => "StencilOpSeparate",
            Hint(..) => "Hint",
            IsEnabled(..) => "IsEnabled",
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
            UniformMatrix2fv(..) => "UniformMatrix2fv",
            UniformMatrix3fv(..) => "UniformMatrix3fv",
            UniformMatrix4fv(..) => "UniformMatrix4fv",
            UseProgram(..) => "UseProgram",
            ValidateProgram(..) => "ValidateProgram",
            VertexAttrib(..) => "VertexAttrib",
            VertexAttribPointer2f(..) => "VertexAttribPointer2f",
            VertexAttribPointer(..) => "VertexAttribPointer",
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
                ctx.gl().active_texture(target),
            WebGLCommand::AttachShader(program_id, shader_id) =>
                ctx.gl().attach_shader(program_id.get(), shader_id.get()),
            WebGLCommand::DetachShader(program_id, shader_id) =>
                ctx.gl().detach_shader(program_id.get(), shader_id.get()),
            WebGLCommand::BindAttribLocation(program_id, index, name) =>
                ctx.gl().bind_attrib_location(program_id.get(), index, &name),
            WebGLCommand::BlendColor(r, g, b, a) =>
                ctx.gl().blend_color(r, g, b, a),
            WebGLCommand::BlendEquation(mode) =>
                ctx.gl().blend_equation(mode),
            WebGLCommand::BlendEquationSeparate(mode_rgb, mode_alpha) =>
                ctx.gl().blend_equation_separate(mode_rgb, mode_alpha),
            WebGLCommand::BlendFunc(src, dest) =>
                ctx.gl().blend_func(src, dest),
            WebGLCommand::BlendFuncSeparate(src_rgb, dest_rgb, src_alpha, dest_alpha) =>
                ctx.gl().blend_func_separate(src_rgb, dest_rgb, src_alpha, dest_alpha),
            WebGLCommand::BufferData(buffer_type, data, usage) =>
                gl::buffer_data(ctx.gl(), buffer_type, &data, usage),
            WebGLCommand::BufferSubData(buffer_type, offset, data) =>
                gl::buffer_sub_data(ctx.gl(), buffer_type, offset, &data),
            WebGLCommand::Clear(mask) =>
                ctx.gl().clear(mask),
            WebGLCommand::ClearColor(r, g, b, a) =>
                ctx.gl().clear_color(r, g, b, a),
            WebGLCommand::ClearDepth(depth) =>
                ctx.gl().clear_depth(depth),
            WebGLCommand::ClearStencil(stencil) =>
                ctx.gl().clear_stencil(stencil),
            WebGLCommand::ColorMask(r, g, b, a) =>
                ctx.gl().color_mask(r, g, b, a),
            WebGLCommand::CopyTexImage2D(target, level, internal_format, x, y, width, height, border) =>
                ctx.gl().copy_tex_image_2d(target, level, internal_format, x, y, width, height, border),
            WebGLCommand::CopyTexSubImage2D(target, level, xoffset, yoffset, x, y, width, height) =>
                ctx.gl().copy_tex_sub_image_2d(target, level, xoffset, yoffset, x, y, width, height),
            WebGLCommand::CullFace(mode) =>
                ctx.gl().cull_face(mode),
            WebGLCommand::DepthFunc(func) =>
                ctx.gl().depth_func(func),
            WebGLCommand::DepthMask(flag) =>
                ctx.gl().depth_mask(flag),
            WebGLCommand::DepthRange(near, far) =>
                ctx.gl().depth_range(near, far),
            WebGLCommand::Disable(cap) =>
                ctx.gl().disable(cap),
            WebGLCommand::Enable(cap) =>
                ctx.gl().enable(cap),
            WebGLCommand::FramebufferRenderbuffer(target, attachment, renderbuffertarget, rb) =>
                ctx.gl().framebuffer_renderbuffer(target, attachment, renderbuffertarget, rb.map_or(0, WebGLRenderbufferId::get)),
            WebGLCommand::FramebufferTexture2D(target, attachment, textarget, texture, level) =>
                ctx.gl().framebuffer_texture_2d(target, attachment, textarget, texture.map_or(0, WebGLTextureId::get), level),
            WebGLCommand::FrontFace(mode) =>
                ctx.gl().front_face(mode),
            WebGLCommand::DisableVertexAttribArray(attrib_id) =>
                ctx.gl().disable_vertex_attrib_array(attrib_id),
            WebGLCommand::DrawArrays(mode, first, count) =>
                ctx.gl().draw_arrays(mode, first, count),
            WebGLCommand::DrawElements(mode, count, type_, offset) =>
                ctx.gl().draw_elements(mode, count, type_, offset as u32),
            WebGLCommand::EnableVertexAttribArray(attrib_id) =>
                ctx.gl().enable_vertex_attrib_array(attrib_id),
            WebGLCommand::Hint(name, val) =>
                ctx.gl().hint(name, val),
            WebGLCommand::IsEnabled(cap, chan) =>
                chan.send(ctx.gl().is_enabled(cap) != 0).unwrap(),
            WebGLCommand::LineWidth(width) =>
                ctx.gl().line_width(width),
            WebGLCommand::PixelStorei(name, val) =>
                ctx.gl().pixel_store_i(name, val),
            WebGLCommand::PolygonOffset(factor, units) =>
                ctx.gl().polygon_offset(factor, units),
            WebGLCommand::ReadPixels(x, y, width, height, format, pixel_type, chan) =>
                Self::read_pixels(ctx.gl(), x, y, width, height, format, pixel_type, chan),
            WebGLCommand::RenderbufferStorage(target, format, width, height) =>
                ctx.gl().renderbuffer_storage(target, format, width, height),
            WebGLCommand::SampleCoverage(value, invert) =>
                ctx.gl().sample_coverage(value, invert),
            WebGLCommand::Scissor(x, y, width, height) =>
                ctx.gl().scissor(x, y, width, height),
            WebGLCommand::StencilFunc(func, ref_, mask) =>
                ctx.gl().stencil_func(func, ref_, mask),
            WebGLCommand::StencilFuncSeparate(face, func, ref_, mask) =>
                ctx.gl().stencil_func_separate(face, func, ref_, mask),
            WebGLCommand::StencilMask(mask) =>
                ctx.gl().stencil_mask(mask),
            WebGLCommand::StencilMaskSeparate(face, mask) =>
                ctx.gl().stencil_mask_separate(face, mask),
            WebGLCommand::StencilOp(fail, zfail, zpass) =>
                ctx.gl().stencil_op(fail, zfail, zpass),
            WebGLCommand::StencilOpSeparate(face, fail, zfail, zpass) =>
                ctx.gl().stencil_op_separate(face, fail, zfail, zpass),
            WebGLCommand::GetActiveAttrib(program_id, index, chan) =>
                Self::active_attrib(ctx.gl(), program_id, index, chan),
            WebGLCommand::GetActiveUniform(program_id, index, chan) =>
                Self::active_uniform(ctx.gl(), program_id, index, chan),
            WebGLCommand::GetAttribLocation(program_id, name, chan) =>
                Self::attrib_location(ctx.gl(), program_id, name, chan),
            WebGLCommand::GetVertexAttrib(index, pname, chan) =>
                Self::vertex_attrib(ctx.gl(), index, pname, chan),
            WebGLCommand::GetBufferParameter(target, param_id, chan) =>
                Self::buffer_parameter(ctx.gl(), target, param_id, chan),
            WebGLCommand::GetParameter(param_id, chan) =>
                Self::parameter(ctx.gl(), param_id, chan),
            WebGLCommand::GetProgramParameter(program_id, param_id, chan) =>
                Self::program_parameter(ctx.gl(), program_id, param_id, chan),
            WebGLCommand::GetShaderParameter(shader_id, param_id, chan) =>
                Self::shader_parameter(ctx.gl(), shader_id, param_id, chan),
            WebGLCommand::GetUniformLocation(program_id, name, chan) =>
                Self::uniform_location(ctx.gl(), program_id, name, chan),
            WebGLCommand::GetShaderInfoLog(shader_id, chan) =>
                Self::shader_info_log(ctx.gl(), shader_id, chan),
            WebGLCommand::GetProgramInfoLog(program_id, chan) =>
                Self::program_info_log(ctx.gl(), program_id, chan),
            WebGLCommand::CompileShader(shader_id, source) =>
                Self::compile_shader(ctx.gl(), shader_id, source),
            WebGLCommand::CreateBuffer(chan) =>
                Self::create_buffer(ctx.gl(), chan),
            WebGLCommand::CreateFramebuffer(chan) =>
                Self::create_framebuffer(ctx.gl(), chan),
            WebGLCommand::CreateRenderbuffer(chan) =>
                Self::create_renderbuffer(ctx.gl(), chan),
            WebGLCommand::CreateTexture(chan) =>
                Self::create_texture(ctx.gl(), chan),
            WebGLCommand::CreateProgram(chan) =>
                Self::create_program(ctx.gl(), chan),
            WebGLCommand::CreateShader(shader_type, chan) =>
                Self::create_shader(ctx.gl(), shader_type, chan),
            WebGLCommand::DeleteBuffer(id) =>
                ctx.gl().delete_buffers(&[id.get()]),
            WebGLCommand::DeleteFramebuffer(id) =>
                ctx.gl().delete_framebuffers(&[id.get()]),
            WebGLCommand::DeleteRenderbuffer(id) =>
                ctx.gl().delete_renderbuffers(&[id.get()]),
            WebGLCommand::DeleteTexture(id) =>
                ctx.gl().delete_textures(&[id.get()]),
            WebGLCommand::DeleteProgram(id) =>
                ctx.gl().delete_program(id.get()),
            WebGLCommand::DeleteShader(id) =>
                ctx.gl().delete_shader(id.get()),
            WebGLCommand::BindBuffer(target, id) =>
                ctx.gl().bind_buffer(target, id.map_or(0, WebGLBufferId::get)),
            WebGLCommand::BindFramebuffer(target, request) =>
                Self::bind_framebuffer(ctx.gl(), target, request, ctx),
            WebGLCommand::BindRenderbuffer(target, id) =>
                ctx.gl().bind_renderbuffer(target, id.map_or(0, WebGLRenderbufferId::get)),
            WebGLCommand::BindTexture(target, id) =>
                ctx.gl().bind_texture(target, id.map_or(0, WebGLTextureId::get)),
            WebGLCommand::LinkProgram(program_id) =>
                ctx.gl().link_program(program_id.get()),
            WebGLCommand::Uniform1f(uniform_id, v) =>
                ctx.gl().uniform_1f(uniform_id, v),
            WebGLCommand::Uniform1fv(uniform_id, v) =>
                ctx.gl().uniform_1fv(uniform_id, &v),
            WebGLCommand::Uniform1i(uniform_id, v) =>
                ctx.gl().uniform_1i(uniform_id, v),
            WebGLCommand::Uniform1iv(uniform_id, v) =>
                ctx.gl().uniform_1iv(uniform_id, &v),
            WebGLCommand::Uniform2f(uniform_id, x, y) =>
                ctx.gl().uniform_2f(uniform_id, x, y),
            WebGLCommand::Uniform2fv(uniform_id, v) =>
                ctx.gl().uniform_2fv(uniform_id, &v),
            WebGLCommand::Uniform2i(uniform_id, x, y) =>
                ctx.gl().uniform_2i(uniform_id, x, y),
            WebGLCommand::Uniform2iv(uniform_id, v) =>
                ctx.gl().uniform_2iv(uniform_id, &v),
            WebGLCommand::Uniform3f(uniform_id, x, y, z) =>
                ctx.gl().uniform_3f(uniform_id, x, y, z),
            WebGLCommand::Uniform3fv(uniform_id, v) =>
                ctx.gl().uniform_3fv(uniform_id, &v),
            WebGLCommand::Uniform3i(uniform_id, x, y, z) =>
                ctx.gl().uniform_3i(uniform_id, x, y, z),
            WebGLCommand::Uniform3iv(uniform_id, v) =>
                ctx.gl().uniform_3iv(uniform_id, &v),
            WebGLCommand::Uniform4f(uniform_id, x, y, z, w) =>
                ctx.gl().uniform_4f(uniform_id, x, y, z, w),
            WebGLCommand::Uniform4fv(uniform_id, v) =>
                ctx.gl().uniform_4fv(uniform_id, &v),
            WebGLCommand::Uniform4i(uniform_id, x, y, z, w) =>
                ctx.gl().uniform_4i(uniform_id, x, y, z, w),
            WebGLCommand::Uniform4iv(uniform_id, v) =>
                ctx.gl().uniform_4iv(uniform_id, &v),
            WebGLCommand::UniformMatrix2fv(uniform_id, transpose,  v) =>
                ctx.gl().uniform_matrix_2fv(uniform_id, transpose, &v),
            WebGLCommand::UniformMatrix3fv(uniform_id, transpose,  v) =>
                ctx.gl().uniform_matrix_3fv(uniform_id, transpose, &v),
            WebGLCommand::UniformMatrix4fv(uniform_id, transpose,  v) =>
                ctx.gl().uniform_matrix_4fv(uniform_id, transpose, &v),
            WebGLCommand::UseProgram(program_id) =>
                ctx.gl().use_program(program_id.get()),
            WebGLCommand::ValidateProgram(program_id) =>
                ctx.gl().validate_program(program_id.get()),
            WebGLCommand::VertexAttrib(attrib_id, x, y, z, w) =>
                ctx.gl().vertex_attrib_4f(attrib_id, x, y, z, w),
            WebGLCommand::VertexAttribPointer2f(attrib_id, size, normalized, stride, offset) =>
                ctx.gl().vertex_attrib_pointer_f32(attrib_id, size, normalized, stride, offset),
            WebGLCommand::VertexAttribPointer(attrib_id, size, data_type, normalized, stride, offset) =>
                ctx.gl().vertex_attrib_pointer(attrib_id, size, data_type, normalized, stride, offset),
            WebGLCommand::Viewport(x, y, width, height) =>
                ctx.gl().viewport(x, y, width, height),
            WebGLCommand::TexImage2D(target, level, internal, width, height, format, data_type, data) =>
                ctx.gl().tex_image_2d(target, level, internal, width, height, /*border*/0, format, data_type, Some(&data)),
            WebGLCommand::TexParameteri(target, name, value) =>
                ctx.gl().tex_parameter_i(target, name, value),
            WebGLCommand::TexParameterf(target, name, value) =>
                ctx.gl().tex_parameter_f(target, name, value),
            WebGLCommand::TexSubImage2D(target, level, xoffset, yoffset, x, y, width, height, data) =>
                ctx.gl().tex_sub_image_2d(target, level, xoffset, yoffset, x, y, width, height, &data),
            WebGLCommand::DrawingBufferWidth(sender) =>
                sender.send(ctx.borrow_draw_buffer().unwrap().size().width).unwrap(),
            WebGLCommand::DrawingBufferHeight(sender) =>
                sender.send(ctx.borrow_draw_buffer().unwrap().size().height).unwrap(),
            WebGLCommand::Finish(sender) =>
                Self::finish(ctx.gl(), sender),
            WebGLCommand::Flush =>
                ctx.gl().flush(),
            WebGLCommand::GenerateMipmap(target) =>
                ctx.gl().generate_mipmap(target),
        }

        // FIXME: Use debug_assertions once tests are run with them
        let error = ctx.gl().get_error();
        assert!(error == gl::NO_ERROR, "Unexpected WebGL error: 0x{:x} ({})", error, error);
    }

    fn read_pixels(gl_: &gl::Gl, x: i32, y: i32, width: i32, height: i32, format: u32, pixel_type: u32,
                   chan: MsgSender<Vec<u8>>) {
      let result = gl_.read_pixels(x, y, width, height, format, pixel_type);
      chan.send(result).unwrap()
    }

    fn active_attrib(gl_: &gl::Gl,
                     program_id: WebGLProgramId,
                     index: u32,
                     chan: MsgSender<WebGLResult<(i32, u32, String)>>) {
        let result = if index >= gl_.get_program_iv(program_id.get(), gl::ACTIVE_ATTRIBUTES) as u32 {
            Err(WebGLError::InvalidValue)
        } else {
            Ok(gl_.get_active_attrib(program_id.get(), index))
        };
        chan.send(result).unwrap();
    }

    fn active_uniform(gl_: &gl::Gl,
                      program_id: WebGLProgramId,
                      index: u32,
                      chan: MsgSender<WebGLResult<(i32, u32, String)>>) {
        let result = if index >= gl_.get_program_iv(program_id.get(), gl::ACTIVE_UNIFORMS) as u32 {
            Err(WebGLError::InvalidValue)
        } else {
            Ok(gl_.get_active_uniform(program_id.get(), index))
        };
        chan.send(result).unwrap();
    }

    fn attrib_location(gl_: &gl::Gl,
                       program_id: WebGLProgramId,
                       name: String,
                       chan: MsgSender<Option<i32>> ) {
        let attrib_location = gl_.get_attrib_location(program_id.get(), &name);

        let attrib_location = if attrib_location == -1 {
            None
        } else {
            Some(attrib_location)
        };

        chan.send(attrib_location).unwrap();
    }

    fn parameter(gl_: &gl::Gl,
                 param_id: u32,
                 chan: MsgSender<WebGLResult<WebGLParameter>>) {
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
                Ok(WebGLParameter::Int(gl_.get_integer_v(param_id))),

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
                Ok(WebGLParameter::Bool(gl_.get_boolean_v(param_id) != 0)),

            gl::DEPTH_CLEAR_VALUE |
            gl::LINE_WIDTH |
            gl::POLYGON_OFFSET_FACTOR |
            gl::POLYGON_OFFSET_UNITS |
            gl::SAMPLE_COVERAGE_VALUE =>
                Ok(WebGLParameter::Float(gl_.get_float_v(param_id))),

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

    fn finish(gl_: &gl::Gl, chan: MsgSender<()>) {
        gl_.finish();
        chan.send(()).unwrap();
    }

    fn vertex_attrib(gl_: &gl::Gl,
                     index: u32,
                     pname: u32,
                     chan: MsgSender<WebGLResult<WebGLParameter>>) {
        let result = if index >= gl_.get_integer_v(gl::MAX_VERTEX_ATTRIBS) as u32 {
            Err(WebGLError::InvalidValue)
        } else {
            match pname {
                gl::VERTEX_ATTRIB_ARRAY_ENABLED |
                gl::VERTEX_ATTRIB_ARRAY_NORMALIZED =>
                    Ok(WebGLParameter::Bool(gl_.get_vertex_attrib_iv(index, pname) != 0)),
                gl::VERTEX_ATTRIB_ARRAY_SIZE |
                gl::VERTEX_ATTRIB_ARRAY_STRIDE |
                gl::VERTEX_ATTRIB_ARRAY_TYPE =>
                    Ok(WebGLParameter::Int(gl_.get_vertex_attrib_iv(index, pname))),
                gl::CURRENT_VERTEX_ATTRIB =>
                    Ok(WebGLParameter::FloatArray(gl_.get_vertex_attrib_fv(index, pname))),
                // gl::VERTEX_ATTRIB_ARRAY_BUFFER_BINDING should return WebGLBuffer
                _ => Err(WebGLError::InvalidEnum),
            }
        };

        chan.send(result).unwrap();
    }

    fn buffer_parameter(gl_: &gl::Gl,
                        target: u32,
                        param_id: u32,
                        chan: MsgSender<WebGLResult<WebGLParameter>>) {
        let result = match param_id {
            gl::BUFFER_SIZE |
            gl::BUFFER_USAGE =>
                Ok(WebGLParameter::Int(gl_.get_buffer_parameter_iv(target, param_id))),
            _ => Err(WebGLError::InvalidEnum),
        };

        chan.send(result).unwrap();
    }

    fn program_parameter(gl_: &gl::Gl,
                         program_id: WebGLProgramId,
                         param_id: u32,
                         chan: MsgSender<WebGLResult<WebGLParameter>>) {
        let result = match param_id {
            gl::DELETE_STATUS |
            gl::LINK_STATUS |
            gl::VALIDATE_STATUS =>
                Ok(WebGLParameter::Bool(gl_.get_program_iv(program_id.get(), param_id) != 0)),
            gl::ATTACHED_SHADERS |
            gl::ACTIVE_ATTRIBUTES |
            gl::ACTIVE_UNIFORMS =>
                Ok(WebGLParameter::Int(gl_.get_program_iv(program_id.get(), param_id))),
            _ => Err(WebGLError::InvalidEnum),
        };

        chan.send(result).unwrap();
    }

    fn shader_parameter(gl_: &gl::Gl,
                        shader_id: WebGLShaderId,
                        param_id: u32,
                        chan: MsgSender<WebGLResult<WebGLParameter>>) {
        let result = match param_id {
            gl::SHADER_TYPE =>
                Ok(WebGLParameter::Int(gl_.get_shader_iv(shader_id.get(), param_id))),
            gl::DELETE_STATUS |
            gl::COMPILE_STATUS =>
                Ok(WebGLParameter::Bool(gl_.get_shader_iv(shader_id.get(), param_id) != 0)),
            _ => Err(WebGLError::InvalidEnum),
        };

        chan.send(result).unwrap();
    }

    fn uniform_location(gl_: &gl::Gl,
                        program_id: WebGLProgramId,
                        name: String,
                        chan: MsgSender<Option<i32>>) {
        let location = gl_.get_uniform_location(program_id.get(), &name);
        let location = if location == -1 {
            None
        } else {
            Some(location)
        };

        chan.send(location).unwrap();
    }


    fn shader_info_log(gl_: &gl::Gl, shader_id: WebGLShaderId, chan: MsgSender<String>) {
        let log = gl_.get_shader_info_log(shader_id.get());
        chan.send(log).unwrap();
    }

    fn program_info_log(gl_: &gl::Gl, program_id: WebGLProgramId, chan: MsgSender<String>) {
        let log = gl_.get_program_info_log(program_id.get());
        chan.send(log).unwrap();
    }

    fn create_buffer(gl_: &gl::Gl, chan: MsgSender<Option<WebGLBufferId>>) {
        let buffer = gl_.gen_buffers(1)[0];
        let buffer = if buffer == 0 {
            None
        } else {
            Some(unsafe { WebGLBufferId::new(buffer) })
        };
        chan.send(buffer).unwrap();
    }

    fn create_framebuffer(gl_: &gl::Gl, chan: MsgSender<Option<WebGLFramebufferId>>) {
        let framebuffer = gl_.gen_framebuffers(1)[0];
        let framebuffer = if framebuffer == 0 {
            None
        } else {
            Some(unsafe { WebGLFramebufferId::new(framebuffer) })
        };
        chan.send(framebuffer).unwrap();
    }


    fn create_renderbuffer(gl_: &gl::Gl, chan: MsgSender<Option<WebGLRenderbufferId>>) {
        let renderbuffer = gl_.gen_renderbuffers(1)[0];
        let renderbuffer = if renderbuffer == 0 {
            None
        } else {
            Some(unsafe { WebGLRenderbufferId::new(renderbuffer) })
        };
        chan.send(renderbuffer).unwrap();
    }

    fn create_texture(gl_: &gl::Gl, chan: MsgSender<Option<WebGLTextureId>>) {
        let texture = gl_.gen_textures(1)[0];
        let texture = if texture == 0 {
            None
        } else {
            Some(unsafe { WebGLTextureId::new(texture) })
        };
        chan.send(texture).unwrap();
    }


    fn create_program(gl_: &gl::Gl, chan: MsgSender<Option<WebGLProgramId>>) {
        let program = gl_.create_program();
        let program = if program == 0 {
            None
        } else {
            Some(unsafe { WebGLProgramId::new(program) })
        };
        chan.send(program).unwrap();
    }

    fn create_shader(gl_: &gl::Gl, shader_type: u32, chan: MsgSender<Option<WebGLShaderId>>) {
        let shader = gl_.create_shader(shader_type);
        let shader = if shader == 0 {
            None
        } else {
            Some(unsafe { WebGLShaderId::new(shader) })
        };
        chan.send(shader).unwrap();
    }

    #[inline]
    fn bind_framebuffer<Native: NativeGLContextMethods>(gl_: &gl::Gl,
                                                        target: u32,
                                                        request: WebGLFramebufferBindingRequest,
                                                        ctx: &GLContext<Native>) {
        let id = match request {
            WebGLFramebufferBindingRequest::Explicit(id) => id.get(),
            WebGLFramebufferBindingRequest::Default =>
                ctx.borrow_draw_buffer().unwrap().get_framebuffer(),
        };

        gl_.bind_framebuffer(target, id);
    }


    #[inline]
    fn compile_shader(gl_: &gl::Gl, shader_id: WebGLShaderId, source: String) {
        gl_.shader_source(shader_id.get(), &[source.as_bytes()]);
        gl_.compile_shader(shader_id.get());
    }
}
