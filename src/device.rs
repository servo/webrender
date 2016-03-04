/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::Matrix4;
use fnv::FnvHasher;
use gleam::gl;
use internal_types::{PackedColor, PackedVertex, PackedVertexForQuad};
use internal_types::{PackedVertexForTextureCacheUpdate, RenderTargetMode, TextureSampler};
use internal_types::{VertexAttribute, DebugFontVertex, DebugColorVertex};
//use notify::{self, Watcher};
use std::collections::HashMap;
use std::fs::File;
use std::hash::BuildHasherDefault;
use std::io::Read;
use std::path::PathBuf;
use std::mem;
//use std::sync::mpsc::{channel, Sender};
//use std::thread;
use webrender_traits::ImageFormat;

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

static QUAD_VERTICES: [PackedVertex; 6] = [
    PackedVertex {
        x: 0.0, y: 0.0,
        color: PackedColor { r: 255, g: 255, b: 255, a: 255 },
        u: 0.0, v: 0.0,
        mu: 0, mv: 0,
        matrix_index: 0,
        clip_in_rect_index: 0,
        clip_out_rect_index: 0,
        tile_params_index: 0,
    },
    PackedVertex {
        x: 1.0, y: 0.0,
        color: PackedColor { r: 255, g: 255, b: 255, a: 255 },
        u: 0.0, v: 0.0,
        mu: 0, mv: 0,
        matrix_index: 0,
        clip_in_rect_index: 0,
        clip_out_rect_index: 0,
        tile_params_index: 0,
    },
    PackedVertex {
        x: 1.0, y: 1.0,
        color: PackedColor { r: 255, g: 255, b: 255, a: 255 },
        u: 0.0, v: 0.0,
        mu: 0, mv: 0,
        matrix_index: 0,
        clip_in_rect_index: 0,
        clip_out_rect_index: 0,
        tile_params_index: 0,
    },
    PackedVertex {
        x: 0.0, y: 0.0,
        color: PackedColor { r: 255, g: 255, b: 255, a: 255 },
        u: 0.0, v: 0.0,
        mu: 0, mv: 0,
        matrix_index: 0,
        clip_in_rect_index: 0,
        clip_out_rect_index: 0,
        tile_params_index: 0,
    },
    PackedVertex {
        x: 1.0, y: 1.0,
        color: PackedColor { r: 255, g: 255, b: 255, a: 255 },
        u: 0.0, v: 0.0,
        mu: 0, mv: 0,
        matrix_index: 0,
        clip_in_rect_index: 0,
        clip_out_rect_index: 0,
        tile_params_index: 0,
    },
    PackedVertex {
        x: 0.0, y: 1.0,
        color: PackedColor { r: 255, g: 255, b: 255, a: 255 },
        u: 0.0, v: 0.0,
        mu: 0, mv: 0,
        matrix_index: 0,
        clip_in_rect_index: 0,
        clip_out_rect_index: 0,
        tile_params_index: 0,
    },
];

lazy_static! {
    pub static ref MAX_TEXTURE_SIZE: gl::GLint = {
        gl::get_integer_v(gl::MAX_TEXTURE_SIZE)
    };
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TextureFilter {
    Nearest,
    Linear,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum VertexFormat {
    Triangles,
    Rectangles,
    DebugFont,
    DebugColor,
    RasterOp,
}

pub trait FileWatcherHandler : Send {
    fn file_changed(&self, path: PathBuf);
}

impl VertexFormat {
    fn bind(&self, main: VBOId, aux: Option<VBOId>, offset: gl::GLuint) {
        main.bind();

        match *self {
            VertexFormat::DebugFont => {
                gl::enable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::ColorRectTL as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::ColorTexCoordRectTop as
                                               gl::GLuint);

                self.set_divisors(0);

                let vertex_stride = mem::size_of::<DebugFontVertex>() as gl::GLuint;

                gl::vertex_attrib_pointer(VertexAttribute::Position as gl::GLuint,
                                          2,
                                          gl::FLOAT,
                                          false,
                                          vertex_stride as gl::GLint,
                                          0 + vertex_stride * offset);
                gl::vertex_attrib_pointer(VertexAttribute::ColorRectTL as gl::GLuint,
                                          4,
                                          gl::UNSIGNED_BYTE,
                                          true,
                                          vertex_stride as gl::GLint,
                                          8 + vertex_stride * offset);
                gl::vertex_attrib_pointer(VertexAttribute::ColorTexCoordRectTop as gl::GLuint,
                                          2,
                                          gl::FLOAT,
                                          false,
                                          vertex_stride as gl::GLint,
                                          12 + vertex_stride * offset);
            }
            VertexFormat::DebugColor => {
                gl::enable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::ColorRectTL as gl::GLuint);

                self.set_divisors(0);

                let vertex_stride = mem::size_of::<DebugColorVertex>() as gl::GLuint;

                gl::vertex_attrib_pointer(VertexAttribute::Position as gl::GLuint,
                                          2,
                                          gl::FLOAT,
                                          false,
                                          vertex_stride as gl::GLint,
                                          0 + vertex_stride * offset);
                gl::vertex_attrib_pointer(VertexAttribute::ColorRectTL as gl::GLuint,
                                          4,
                                          gl::UNSIGNED_BYTE,
                                          true,
                                          vertex_stride as gl::GLint,
                                          8 + vertex_stride * offset);
            }
            VertexFormat::Rectangles => {
                gl::enable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);

                let vertex_stride = mem::size_of::<PackedVertex>() as gl::GLuint;

                gl::vertex_attrib_pointer(VertexAttribute::Position as gl::GLuint,
                                          2,
                                          gl::FLOAT,
                                          false,
                                          vertex_stride as gl::GLint,
                                          0);

                aux.as_ref().unwrap().bind();

                gl::enable_vertex_attrib_array(VertexAttribute::PositionRect as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::ColorRectTL as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::ColorRectTR as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::ColorRectBR as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::ColorRectBL as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::ColorTexCoordRectTop as
                                               gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::ColorTexCoordRectBottom as
                                               gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::MaskTexCoordRectTop as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::MaskTexCoordRectBottom as
                                               gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::Misc as gl::GLuint);

                self.set_divisors(1);

                let vertex_stride = mem::size_of::<PackedVertexForQuad>() as gl::GLuint;

                gl::vertex_attrib_pointer(VertexAttribute::PositionRect as gl::GLuint,
                                          4,
                                          gl::FLOAT,
                                          false,
                                          vertex_stride as gl::GLint,
                                          0 + vertex_stride * offset);
                gl::vertex_attrib_pointer(VertexAttribute::ColorRectTL as gl::GLuint,
                                          4,
                                          gl::UNSIGNED_BYTE,
                                          false,
                                          vertex_stride as gl::GLint,
                                          16 + vertex_stride * offset);
                gl::vertex_attrib_pointer(VertexAttribute::ColorRectTR as gl::GLuint,
                                          4,
                                          gl::UNSIGNED_BYTE,
                                          false,
                                          vertex_stride as gl::GLint,
                                          20 + vertex_stride * offset);
                gl::vertex_attrib_pointer(VertexAttribute::ColorRectBR as gl::GLuint,
                                          4,
                                          gl::UNSIGNED_BYTE,
                                          false,
                                          vertex_stride as gl::GLint,
                                          24 + vertex_stride * offset);
                gl::vertex_attrib_pointer(VertexAttribute::ColorRectBL as gl::GLuint,
                                          4,
                                          gl::UNSIGNED_BYTE,
                                          false,
                                          vertex_stride as gl::GLint,
                                          28 + vertex_stride * offset);
                gl::vertex_attrib_pointer(VertexAttribute::ColorTexCoordRectTop as gl::GLuint,
                                          4,
                                          gl::FLOAT,
                                          false,
                                          vertex_stride as gl::GLint,
                                          32 + vertex_stride * offset);
                gl::vertex_attrib_pointer(VertexAttribute::ColorTexCoordRectBottom as gl::GLuint,
                                          4,
                                          gl::FLOAT,
                                          false,
                                          vertex_stride as gl::GLint,
                                          48 + vertex_stride * offset);
                gl::vertex_attrib_pointer(VertexAttribute::MaskTexCoordRectTop as gl::GLuint,
                                          4,
                                          gl::UNSIGNED_SHORT,
                                          false,
                                          vertex_stride as gl::GLint,
                                          64 + vertex_stride * offset);
                gl::vertex_attrib_pointer(VertexAttribute::MaskTexCoordRectBottom as gl::GLuint,
                                          4,
                                          gl::UNSIGNED_SHORT,
                                          false,
                                          vertex_stride as gl::GLint,
                                          72 + vertex_stride * offset);
                gl::vertex_attrib_pointer(VertexAttribute::Misc as gl::GLuint,
                                          4,
                                          gl::UNSIGNED_BYTE,
                                          false,
                                          vertex_stride as gl::GLint,
                                          80 + vertex_stride * offset);
            }
            VertexFormat::Triangles => {
                gl::enable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::ColorRectTL as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::ColorTexCoordRectTop as
                                               gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::MaskTexCoordRectTop as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::Misc as gl::GLuint);

                self.set_divisors(0);

                let vertex_stride = mem::size_of::<PackedVertex>() as gl::GLuint;

                gl::vertex_attrib_pointer(VertexAttribute::Position as gl::GLuint,
                                          2,
                                          gl::FLOAT,
                                          false,
                                          vertex_stride as gl::GLint,
                                          0 + vertex_stride * offset);
                gl::vertex_attrib_pointer(VertexAttribute::ColorRectTL as gl::GLuint,
                                          4,
                                          gl::UNSIGNED_BYTE,
                                          false,
                                          vertex_stride as gl::GLint,
                                          8 + vertex_stride * offset);
                gl::vertex_attrib_pointer(VertexAttribute::ColorTexCoordRectTop as gl::GLuint,
                                          2,
                                          gl::FLOAT,
                                          false,
                                          vertex_stride as gl::GLint,
                                          12 + vertex_stride * offset);
                gl::vertex_attrib_pointer(VertexAttribute::MaskTexCoordRectTop as gl::GLuint,
                                          2,
                                          gl::UNSIGNED_SHORT,
                                          false,
                                          vertex_stride as gl::GLint,
                                          20 + vertex_stride * offset);
                gl::vertex_attrib_pointer(VertexAttribute::Misc as gl::GLuint,
                                          4,
                                          gl::UNSIGNED_BYTE,
                                          false,
                                          vertex_stride as gl::GLint,
                                          24 + vertex_stride * offset);
            }
            VertexFormat::RasterOp => {
                gl::enable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::ColorRectTL as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::ColorTexCoordRectTop as
                                               gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::BorderRadii as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::BorderPosition as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::BlurRadius as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::DestTextureSize as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::SourceTextureSize as gl::GLuint);
                gl::enable_vertex_attrib_array(VertexAttribute::Misc as gl::GLuint);

                self.set_divisors(0);

                let vertex_stride = mem::size_of::<PackedVertexForTextureCacheUpdate>() as
                    gl::GLuint;

                gl::vertex_attrib_pointer(VertexAttribute::Position as gl::GLuint,
                                          2,
                                          gl::FLOAT,
                                          false,
                                          vertex_stride as gl::GLint,
                                          0);
                gl::vertex_attrib_pointer(VertexAttribute::ColorRectTL as gl::GLuint,
                                          4,
                                          gl::UNSIGNED_BYTE,
                                          true,
                                          vertex_stride as gl::GLint,
                                          8);
                gl::vertex_attrib_pointer(VertexAttribute::ColorTexCoordRectTop as gl::GLuint,
                                          2,
                                          gl::UNSIGNED_SHORT,
                                          true,
                                          vertex_stride as gl::GLint,
                                          12);
                gl::vertex_attrib_pointer(VertexAttribute::BorderRadii as gl::GLuint,
                                          4,
                                          gl::FLOAT,
                                          false,
                                          vertex_stride as gl::GLint,
                                          16);
                gl::vertex_attrib_pointer(VertexAttribute::BorderPosition as gl::GLuint,
                                          4,
                                          gl::FLOAT,
                                          false,
                                          vertex_stride as gl::GLint,
                                          32);
                gl::vertex_attrib_pointer(VertexAttribute::DestTextureSize as gl::GLuint,
                                          2,
                                          gl::FLOAT,
                                          false,
                                          vertex_stride as gl::GLint,
                                          48);
                gl::vertex_attrib_pointer(VertexAttribute::SourceTextureSize as gl::GLuint,
                                          2,
                                          gl::FLOAT,
                                          false,
                                          vertex_stride as gl::GLint,
                                          56);
                gl::vertex_attrib_pointer(VertexAttribute::BlurRadius as gl::GLuint,
                                          1,
                                          gl::FLOAT,
                                          false,
                                          vertex_stride as gl::GLint,
                                          64);
                gl::vertex_attrib_pointer(VertexAttribute::Misc as gl::GLuint,
                                          4,
                                          gl::UNSIGNED_BYTE,
                                          false,
                                          vertex_stride as gl::GLint,
                                          68);
            }
        }
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    fn unbind(&self) {
        // TODO(gw): This can be made smarter by diffing the two vertex formats.
        match *self {
            VertexFormat::DebugFont => {
                gl::disable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectTL as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectTR as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectBR as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectBL as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorTexCoordRectTop as
                                                gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorTexCoordRectBottom as
                                                gl::GLuint);
            }
            VertexFormat::DebugColor => {
                gl::disable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectTL as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectTR as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectBR as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectBL as gl::GLuint);
            }
            VertexFormat::Rectangles => {
                gl::disable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectTL as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectTR as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectBR as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectBL as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorTexCoordRectTop as
                                                gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorTexCoordRectBottom as
                                                gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::MaskTexCoordRectTop as
                                                gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::MaskTexCoordRectBottom as
                                                gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::Misc as gl::GLuint);
            }
            VertexFormat::Triangles => {
                gl::disable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectTL as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectTR as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectBR as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectBL as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorTexCoordRectTop as
                                                gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorTexCoordRectBottom as
                                                gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::MaskTexCoordRectTop as
                                                gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::MaskTexCoordRectBottom as
                                                gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::Misc as gl::GLuint);
            }
            VertexFormat::RasterOp => {
                gl::disable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectTL as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectTR as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectBR as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorRectBL as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorTexCoordRectTop as
                                                gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::ColorTexCoordRectBottom as
                                                gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::BorderRadii as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::BorderPosition as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::BlurRadius as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::DestTextureSize as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::SourceTextureSize as gl::GLuint);
                gl::disable_vertex_attrib_array(VertexAttribute::Misc as gl::GLuint);
            }
        }
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    fn set_divisors(&self, _: u32) {}

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    fn set_divisors(&self, divisor: u32) {
        gl::vertex_attrib_divisor(VertexAttribute::PositionRect as gl::GLuint, divisor);
        gl::vertex_attrib_divisor(VertexAttribute::ColorRectTL as gl::GLuint, divisor);
        gl::vertex_attrib_divisor(VertexAttribute::ColorRectTR as gl::GLuint, divisor);
        gl::vertex_attrib_divisor(VertexAttribute::ColorRectBR as gl::GLuint, divisor);
        gl::vertex_attrib_divisor(VertexAttribute::ColorRectBL as gl::GLuint, divisor);
        gl::vertex_attrib_divisor(VertexAttribute::ColorTexCoordRectTop as gl::GLuint, divisor);
        gl::vertex_attrib_divisor(VertexAttribute::ColorTexCoordRectBottom as gl::GLuint, divisor);
        gl::vertex_attrib_divisor(VertexAttribute::MaskTexCoordRectTop as gl::GLuint, divisor);
        gl::vertex_attrib_divisor(VertexAttribute::MaskTexCoordRectBottom as gl::GLuint, divisor);
        gl::vertex_attrib_divisor(VertexAttribute::Misc as gl::GLuint, divisor);
    }
}

impl TextureId {
    fn bind(&self) {
        let TextureId(id) = *self;
        gl::bind_texture(gl::TEXTURE_2D, id);
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
    }
}

struct Texture {
    id: gl::GLuint,
    format: ImageFormat,
    width: u32,
    height: u32,
    filter: TextureFilter,
    mode: RenderTargetMode,
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
    vs_path: PathBuf,
    fs_path: PathBuf,
    vs_id: Option<gl::GLuint>,
    fs_id: Option<gl::GLuint>,
}

impl Program {
    fn attach_and_bind_shaders(&mut self,
                               vs_id: gl::GLuint,
                               fs_id: gl::GLuint,
                               panic_on_fail: bool) -> bool {
        gl::attach_shader(self.id, vs_id);
        gl::attach_shader(self.id, fs_id);

        gl::bind_attrib_location(self.id, VertexAttribute::Position as gl::GLuint, "aPosition");
        gl::bind_attrib_location(self.id,
                                 VertexAttribute::PositionRect as gl::GLuint,
                                 "aPositionRect");
        gl::bind_attrib_location(self.id,
                                 VertexAttribute::ColorRectTL as gl::GLuint,
                                 "aColorRectTL");
        gl::bind_attrib_location(self.id,
                                 VertexAttribute::ColorRectTR as gl::GLuint,
                                 "aColorRectTR");
        gl::bind_attrib_location(self.id,
                                 VertexAttribute::ColorRectBR as gl::GLuint,
                                 "aColorRectBR");
        gl::bind_attrib_location(self.id,
                                 VertexAttribute::ColorRectBL as gl::GLuint,
                                 "aColorRectBL");
        gl::bind_attrib_location(self.id,
                                 VertexAttribute::ColorTexCoordRectTop as gl::GLuint,
                                 "aColorTexCoordRectTop");
        gl::bind_attrib_location(self.id,
                                 VertexAttribute::MaskTexCoordRectTop as gl::GLuint,
                                 "aMaskTexCoordRectTop");
        gl::bind_attrib_location(self.id,
                                 VertexAttribute::ColorTexCoordRectBottom as gl::GLuint,
                                 "aColorTexCoordRectBottom");
        gl::bind_attrib_location(self.id,
                                 VertexAttribute::MaskTexCoordRectBottom as gl::GLuint,
                                 "aMaskTexCoordRectBottom");
        gl::bind_attrib_location(self.id,
                                 VertexAttribute::BorderRadii as gl::GLuint,
                                 "aBorderRadii");
        gl::bind_attrib_location(self.id,
                                 VertexAttribute::BorderPosition as gl::GLuint,
                                 "aBorderPosition");
        gl::bind_attrib_location(self.id,
                                 VertexAttribute::BlurRadius as gl::GLuint,
                                 "aBlurRadius");
        gl::bind_attrib_location(self.id,
                                 VertexAttribute::DestTextureSize as gl::GLuint,
                                 "aDestTextureSize");
        gl::bind_attrib_location(self.id,
                                 VertexAttribute::SourceTextureSize as gl::GLuint,
                                 "aSourceTextureSize");
        gl::bind_attrib_location(self.id, VertexAttribute::Misc as gl::GLuint, "aMisc");

        gl::link_program(self.id);
        if gl::get_program_iv(self.id, gl::LINK_STATUS) == (0 as gl::GLint) {
            println!("Failed to link shader program: {}", gl::get_program_info_log(self.id));
            gl::detach_shader(self.id, vs_id);
            gl::detach_shader(self.id, fs_id);
            if panic_on_fail {
                panic!("-- Program link failed - exiting --");
            }
            false
        } else {
            true
        }
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        gl::delete_program(self.id);
    }
}

struct VAO {
    id: gl::GLuint,
    vertex_format: VertexFormat,
    main_vbo_id: VBOId,
    aux_vbo_id: Option<VBOId>,
    ibo_id: IBOId,
    owns_vbos: bool,
}

#[cfg(any(target_os = "android", target_os = "gonk"))]
impl Drop for VAO {
    fn drop(&mut self) {
        // In the case of a rect batch, the main VBO is the shared quad VBO, so keep that around.
        if self.vertex_format != VertexFormat::Rectangles {
            gl::delete_buffers(&[self.main_vbo_id.0]);
        }
        if let Some(VBOId(aux_vbo_id)) = self.aux_vbo_id {
            gl::delete_buffers(&[aux_vbo_id]);
        }

        // todo(gw): maybe make these their own type with hashmap?
        let IBOId(ibo_id) = self.ibo_id;
        gl::delete_buffers(&[ibo_id]);
    }
}

#[cfg(not(any(target_os = "android", target_os = "gonk")))]
impl Drop for VAO {
    fn drop(&mut self) {
        gl::delete_vertex_arrays(&[self.id]);

        if self.owns_vbos {
            // In the case of a rect batch, the main VBO is the shared quad VBO, so keep that
            // around.
            if self.vertex_format != VertexFormat::Rectangles {
                gl::delete_buffers(&[self.main_vbo_id.0]);
            }
            if let Some(VBOId(aux_vbo_id)) = self.aux_vbo_id {
                gl::delete_buffers(&[aux_vbo_id]);
            }

            // todo(gw): maybe make these their own type with hashmap?
            let IBOId(ibo_id) = self.ibo_id;
            gl::delete_buffers(&[ibo_id]);
        }
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
pub struct VBOId(gl::GLuint);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
struct IBOId(gl::GLuint);

pub struct GpuProfile {
    active_query: usize,
    gl_queries: Vec<gl::GLuint>,
}

impl GpuProfile {
    pub fn new() -> GpuProfile {
        let queries = gl::gen_queries(4);

        // Kick initial queries off so that when we wait for the prev
        // query it doesn't block on the first run through.
        for qid in &queries {
            gl::query_counter(*qid, gl::TIMESTAMP);
        }

        GpuProfile {
            active_query: 0,
            gl_queries: queries,
        }
    }

    pub fn begin(&mut self) {
        let qid = self.gl_queries[self.active_query * 2 + 0];
        gl::query_counter(qid, gl::TIMESTAMP);
    }

    pub fn end(&mut self) -> u64 {
        let qid = self.gl_queries[self.active_query * 2 + 1];
        gl::query_counter(qid, gl::TIMESTAMP);

        self.active_query = 1 - self.active_query;

        // Wait until the previous results are available
        let qid_prev_start = self.gl_queries[self.active_query * 2 + 0];
        let qid_prev_end = self.gl_queries[self.active_query * 2 + 1];
        while gl::get_query_object_iv(qid_prev_end, gl::QUERY_RESULT_AVAILABLE) == 0 {}

        let start_time = gl::get_query_object_ui64v(qid_prev_start, gl::QUERY_RESULT);
        let end_time = gl::get_query_object_ui64v(qid_prev_end, gl::QUERY_RESULT);

        end_time - start_time
    }
}

impl Drop for GpuProfile {
    fn drop(&mut self) {
        gl::delete_queries(&self.gl_queries);
    }
}

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

impl UniformLocation {
    pub fn invalid() -> UniformLocation {
        UniformLocation(-1)
    }
}

// TODO(gw): Fix up notify cargo deps and re-enable this!
/*
enum FileWatcherCmd {
    AddWatch(PathBuf),
    Exit,
}

struct FileWatcherThread {
    api_tx: Sender<FileWatcherCmd>,
}

impl FileWatcherThread {
    fn new(handler: Box<FileWatcherHandler>) -> FileWatcherThread {
        let (api_tx, api_rx) = channel();

        thread::spawn(move || {

            let (watch_tx, watch_rx) = channel();

            enum Request {
                Watcher(notify::Event),
                Command(FileWatcherCmd),
            }

            let mut file_watcher: notify::RecommendedWatcher = notify::Watcher::new(watch_tx).unwrap();

            loop {
                let request = {
                    let receiver_from_api = &api_rx;
                    let receiver_from_watcher = &watch_rx;
                    select! {
                        msg = receiver_from_api.recv() => Request::Command(msg.unwrap()),
                        msg = receiver_from_watcher.recv() => Request::Watcher(msg.unwrap())
                    }
                };

                match request {
                    Request::Watcher(event) => {
                        handler.file_changed(event.path.unwrap());
                    }
                    Request::Command(cmd) => {
                        match cmd {
                            FileWatcherCmd::AddWatch(path) => {
                                file_watcher.watch(path).ok();
                            }
                            FileWatcherCmd::Exit => {
                                break;
                            }
                        }
                    }
                }
            }
        });

        FileWatcherThread {
            api_tx: api_tx,
        }
    }

    fn exit(&self) {
        self.api_tx.send(FileWatcherCmd::Exit).ok();
    }

    fn add_watch(&self, path: PathBuf) {
        self.api_tx.send(FileWatcherCmd::AddWatch(path)).ok();
    }
}
*/

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
    textures: HashMap<TextureId, Texture, BuildHasherDefault<FnvHasher>>,
    raw_textures: HashMap<TextureId, (u32, u32, u32, u32), BuildHasherDefault<FnvHasher>>,
    programs: HashMap<ProgramId, Program, BuildHasherDefault<FnvHasher>>,
    vaos: HashMap<VAOId, VAO, BuildHasherDefault<FnvHasher>>,

    // misc.
    vertex_shader_preamble: String,
    fragment_shader_preamble: String,
    //file_watcher: FileWatcherThread,

    // Used on android only
    #[allow(dead_code)]
    next_vao_id: gl::GLuint,
}

impl Device {
    pub fn new(resource_path: PathBuf,
               device_pixel_ratio: f32,
               _file_changed_handler: Box<FileWatcherHandler>) -> Device {
        //let file_watcher = FileWatcherThread::new(file_changed_handler);

        let mut path = resource_path.clone();
        path.push(VERTEX_SHADER_PREAMBLE);
        let mut f = File::open(&path).unwrap();
        let mut vertex_shader_preamble = String::new();
        f.read_to_string(&mut vertex_shader_preamble).unwrap();
        //file_watcher.add_watch(path);

        let mut path = resource_path.clone();
        path.push(FRAGMENT_SHADER_PREAMBLE);
        let mut f = File::open(&path).unwrap();
        let mut fragment_shader_preamble = String::new();
        f.read_to_string(&mut fragment_shader_preamble).unwrap();
        //file_watcher.add_watch(path);

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

            textures: HashMap::with_hasher(Default::default()),
            raw_textures: HashMap::with_hasher(Default::default()),
            programs: HashMap::with_hasher(Default::default()),
            vaos: HashMap::with_hasher(Default::default()),

            vertex_shader_preamble: vertex_shader_preamble,
            fragment_shader_preamble: fragment_shader_preamble,

            next_vao_id: 1,
            //file_watcher: file_watcher,
        }
    }

    pub fn compile_shader(path: &PathBuf,
                          shader_type: gl::GLenum,
                          shader_preamble: &str,
                          panic_on_fail: bool)
                          -> Option<gl::GLuint> {
        println!("compile {:?}", path);

        let mut f = File::open(&path).unwrap();
        let mut s = shader_preamble.to_owned();
        f.read_to_string(&mut s).unwrap();

        let id = gl::create_shader(shader_type);
        let mut source = Vec::new();
        source.extend_from_slice(s.as_bytes());
        gl::shader_source(id, &[&source[..]]);
        gl::compile_shader(id);
        if gl::get_shader_iv(id, gl::COMPILE_STATUS) == (0 as gl::GLint) {
            println!("Failed to compile shader: {}", gl::get_shader_info_log(id));
            if panic_on_fail {
                panic!("-- Shader compile failed - exiting --");
            }

            None
        } else {
            Some(id)
        }
    }

    pub fn begin_frame(&mut self) {
        debug_assert!(!self.inside_frame);
        self.inside_frame = true;

        // Retrive the currently set FBO.
        let default_fbo = gl::get_integer_v(gl::FRAMEBUFFER_BINDING);
        self.default_fbo = default_fbo as gl::GLuint;

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
        self.bound_fbo = FBOId(self.default_fbo);

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

        let fbo_id = texture_id.map_or(FBOId(self.default_fbo), |texture_id| {
            self.textures.get(&texture_id).unwrap().fbo_ids[0]
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
                filter: TextureFilter::Nearest,
                mode: RenderTargetMode::None,
                fbo_ids: vec![],
            };

            debug_assert!(self.textures.contains_key(&texture_id) == false);
            self.textures.insert(texture_id, texture);

            texture_ids.push(texture_id);
        }

        texture_ids
    }

    pub fn get_texture_dimensions(&self, texture_id: TextureId) -> (u32, u32) {
        if let Some(texture) = self.textures.get(&texture_id) {
            (texture.width, texture.height)
        } else {
            let dimensions = self.raw_textures.get(&texture_id).unwrap();
            (dimensions.2, dimensions.3)
        }
    }

    pub fn texture_has_alpha(&self, texture_id: TextureId) -> bool {
        if let Some(texture) = self.textures.get(&texture_id) {
            texture.format == ImageFormat::RGBA8
        } else {
            true
        }
    }

    pub fn update_raw_texture(&mut self,
                              texture_id: TextureId,
                              x0: u32,
                              y0: u32,
                              width: u32,
                              height: u32) {
        self.raw_textures.insert(texture_id, (x0, y0, width, height));
    }

    fn set_texture_parameters(&mut self, filter: TextureFilter) {
        let filter = match filter {
            TextureFilter::Nearest => {
                gl::NEAREST
            }
            TextureFilter::Linear => {
                gl::LINEAR
            }
        };

        gl::tex_parameter_i(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, filter as gl::GLint);
        gl::tex_parameter_i(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, filter as gl::GLint);

        gl::tex_parameter_i(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as gl::GLint);
        gl::tex_parameter_i(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as gl::GLint);
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

    fn upload_texture_image(&mut self,
                            width: u32,
                            height: u32,
                            internal_format: u32,
                            format: u32,
                            pixels: Option<&[u8]>) {
        self.upload_2d_texture_image(width, height, internal_format, format, pixels)
    }

    fn deinit_texture_image(&mut self) {
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

    pub fn init_texture(&mut self,
                        texture_id: TextureId,
                        width: u32,
                        height: u32,
                        format: ImageFormat,
                        filter: TextureFilter,
                        mode: RenderTargetMode,
                        pixels: Option<&[u8]>) {
        debug_assert!(self.inside_frame);

        {
            let texture = self.textures.get_mut(&texture_id).expect("Didn't find texture!");
            texture.format = format;
            texture.width = width;
            texture.height = height;
            texture.filter = filter;
            texture.mode = mode
        }

        let (internal_format, gl_format) = match format {
            ImageFormat::A8 => (GL_FORMAT_A, GL_FORMAT_A),
            ImageFormat::RGB8 => (gl::RGB, gl::RGB),
            ImageFormat::RGBA8 => {
                if cfg!(target_os="android") {
                    (GL_FORMAT_BGRA, GL_FORMAT_BGRA)
                } else {
                    (gl::RGBA, GL_FORMAT_BGRA)
                }
            }
            ImageFormat::Invalid => unreachable!(),
        };

        match mode {
            RenderTargetMode::RenderTarget => {
                self.bind_color_texture(texture_id);
                self.set_texture_parameters(filter);
                self.upload_2d_texture_image(width, height, internal_format, gl_format, None);
                self.create_fbo_for_texture_if_necessary(texture_id);
            }
            RenderTargetMode::None => {
                texture_id.bind();
                self.set_texture_parameters(filter);

                self.upload_texture_image(width, height,
                                          internal_format,
                                          gl_format,
                                          pixels);
            }
        }
    }

    pub fn create_fbo_for_texture_if_necessary(&mut self, texture_id: TextureId) {
        if !self.textures.get(&texture_id).unwrap().fbo_ids.is_empty() {
            return
        }

        let fbo_ids: Vec<_> =
            gl::gen_framebuffers(1).into_iter().map(|fbo_id| FBOId(fbo_id)).collect();
        for fbo_id in &fbo_ids[..] {
            gl::bind_framebuffer(gl::FRAMEBUFFER, fbo_id.0);

            gl::framebuffer_texture_2d(gl::FRAMEBUFFER,
                                       gl::COLOR_ATTACHMENT0,
                                       gl::TEXTURE_2D,
                                       texture_id.0,
                                       0);
        }

        gl::bind_framebuffer(gl::FRAMEBUFFER, self.default_fbo);

        self.textures.get_mut(&texture_id).unwrap().fbo_ids = fbo_ids;
    }

    pub fn resize_texture(&mut self,
                          texture_id: TextureId,
                          new_width: u32,
                          new_height: u32,
                          format: ImageFormat,
                          filter: TextureFilter,
                          mode: RenderTargetMode) {
        debug_assert!(self.inside_frame);

        let (old_width, old_height) = self.get_texture_dimensions(texture_id);

        let temp_texture_id = self.create_texture_ids(1)[0];
        self.init_texture(temp_texture_id, old_width, old_height, format, filter, mode, None);
        self.create_fbo_for_texture_if_necessary(temp_texture_id);

        self.bind_render_target(Some(texture_id));
        self.bind_color_texture(temp_texture_id);

        gl::copy_tex_sub_image_2d(gl::TEXTURE_2D,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  old_width as i32,
                                  old_height as i32);

        self.deinit_texture(texture_id);
        self.init_texture(texture_id, new_width, new_height, format, filter, mode, None);
        self.create_fbo_for_texture_if_necessary(texture_id);
        self.bind_render_target(Some(temp_texture_id));
        self.bind_color_texture(texture_id);

        gl::copy_tex_sub_image_2d(gl::TEXTURE_2D,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  old_width as i32,
                                  old_height as i32);

        self.bind_render_target(None);
        self.deinit_texture(temp_texture_id);
    }

    pub fn deinit_texture(&mut self, texture_id: TextureId) {
        debug_assert!(self.inside_frame);

        self.bind_color_texture(texture_id);
        self.deinit_texture_image();

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

    pub fn init_texture_if_necessary(&mut self,
                                     texture_id: TextureId,
                                     width: u32,
                                     height: u32,
                                     format: ImageFormat,
                                     filter: TextureFilter,
                                     mode: RenderTargetMode) {
        debug_assert!(self.inside_frame);
        {
            let texture = self.textures.get_mut(&texture_id).expect("Didn't find texture!");
            if texture.format == format &&
                    texture.width == width &&
                    texture.height == height &&
                    texture.filter == filter &&
                    texture.mode == mode {
                return
            }
        }
        self.init_texture(texture_id, width, height, format, filter, mode, None)
    }

    pub fn create_program(&mut self, base_filename: &str) -> ProgramId {
        debug_assert!(self.inside_frame);

        let pid = gl::create_program();

        let mut vs_path = self.resource_path.clone();
        vs_path.push(&format!("{}.vs.glsl", base_filename));
        //self.file_watcher.add_watch(vs_path.clone());

        let mut fs_path = self.resource_path.clone();
        fs_path.push(&format!("{}.fs.glsl", base_filename));
        //self.file_watcher.add_watch(fs_path.clone());

        let program = Program {
            id: pid,
            u_transform: -1,
            vs_path: vs_path,
            fs_path: fs_path,
            vs_id: None,
            fs_id: None,
        };

        let program_id = ProgramId(pid);

        debug_assert!(self.programs.contains_key(&program_id) == false);
        self.programs.insert(program_id, program);

        self.load_program(program_id, true);

        program_id
    }

    fn load_program(&mut self,
                    program_id: ProgramId,
                    panic_on_fail: bool) {
        debug_assert!(self.inside_frame);

        let program = self.programs.get_mut(&program_id).unwrap();

        // todo(gw): store shader ids so they can be freed!
        let vs_id = Device::compile_shader(&program.vs_path,
                                           gl::VERTEX_SHADER,
                                           &*self.vertex_shader_preamble,
                                           panic_on_fail);
        let fs_id = Device::compile_shader(&program.fs_path,
                                           gl::FRAGMENT_SHADER,
                                           &*self.fragment_shader_preamble,
                                           panic_on_fail);

        match (vs_id, fs_id) {
            (Some(vs_id), None) => {
                println!("FAILED to load fs - falling back to previous!");
                gl::delete_shader(vs_id);
            }
            (None, Some(fs_id)) => {
                println!("FAILED to load vs - falling back to previous!");
                gl::delete_shader(fs_id);
            }
            (None, None) => {
                println!("FAILED to load vs/fs - falling back to previous!");
            }
            (Some(vs_id), Some(fs_id)) => {
                if let Some(vs_id) = program.vs_id {
                    gl::detach_shader(program.id, vs_id);
                }

                if let Some(fs_id) = program.fs_id {
                    gl::detach_shader(program.id, fs_id);
                }

                if program.attach_and_bind_shaders(vs_id, fs_id, panic_on_fail) {
                    if let Some(vs_id) = program.vs_id {
                        gl::delete_shader(vs_id);
                    }

                    if let Some(fs_id) = program.fs_id {
                        gl::delete_shader(fs_id);
                    }

                    program.vs_id = Some(vs_id);
                    program.fs_id = Some(fs_id);
                } else {
                    let vs_id = program.vs_id.unwrap();
                    let fs_id = program.fs_id.unwrap();
                    program.attach_and_bind_shaders(vs_id, fs_id, true);
                }

                program.u_transform = gl::get_uniform_location(program.id, "uTransform");

                program_id.bind();
                let u_diffuse = gl::get_uniform_location(program.id, "sDiffuse");
                if u_diffuse != -1 {
                    gl::uniform_1i(u_diffuse, TextureSampler::Color as i32);
                }
                let u_mask = gl::get_uniform_location(program.id, "sMask");
                if u_mask != -1 {
                    gl::uniform_1i(u_mask, TextureSampler::Mask as i32);
                }
                let u_diffuse2d = gl::get_uniform_location(program.id, "sDiffuse2D");
                if u_diffuse2d != -1 {
                    gl::uniform_1i(u_diffuse2d, TextureSampler::Color as i32);
                }
                let u_mask2d = gl::get_uniform_location(program.id, "sMask2D");
                if u_mask2d != -1 {
                    gl::uniform_1i(u_mask2d, TextureSampler::Mask as i32);
                }
                let u_device_pixel_ratio = gl::get_uniform_location(program.id, "uDevicePixelRatio");
                if u_device_pixel_ratio != -1 {
                    gl::uniform_1f(u_device_pixel_ratio, self.device_pixel_ratio);
                }
            }
        }
    }

    pub fn refresh_shader(&mut self, path: PathBuf) {
        let mut vs_preamble_path = self.resource_path.clone();
        vs_preamble_path.push(VERTEX_SHADER_PREAMBLE);

        let mut fs_preamble_path = self.resource_path.clone();
        fs_preamble_path.push(FRAGMENT_SHADER_PREAMBLE);

        let mut refresh_all = false;

        if path == vs_preamble_path {
            let mut f = File::open(&vs_preamble_path).unwrap();
            self.vertex_shader_preamble = String::new();
            f.read_to_string(&mut self.vertex_shader_preamble).unwrap();
            refresh_all = true;
        }

        if path == fs_preamble_path {
            let mut f = File::open(&fs_preamble_path).unwrap();
            self.fragment_shader_preamble = String::new();
            f.read_to_string(&mut self.fragment_shader_preamble).unwrap();
            refresh_all = true;
        }

        let mut programs_to_update = Vec::new();

        for (program_id, program) in &mut self.programs {
            if refresh_all || program.vs_path == path || program.fs_path == path {
                programs_to_update.push(*program_id)
            }
        }

        for program_id in programs_to_update {
            self.load_program(program_id, false);
        }
    }

    pub fn get_uniform_location(&self, program_id: ProgramId, name: &str) -> UniformLocation {
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

    pub fn set_uniform_vec4_array(&self,
                                  uniform: UniformLocation,
                                  vectors: &[f32]) {
        debug_assert!(self.inside_frame);
        let UniformLocation(location) = uniform;
        gl::uniform_4fv(location, vectors);
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

    fn set_uniforms(&self, program: &Program, transform: &Matrix4) {
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
        self.update_image_for_2d_texture(x0 as gl::GLint,
                                         y0 as gl::GLint,
                                         width as gl::GLint,
                                         height as gl::GLint,
                                         gl_format,
                                         data);
    }

    pub fn read_framebuffer_rect(&mut self,
                                 texture_id: TextureId,
                                 dest_x: i32,
                                 dest_y: i32,
                                 src_x: i32,
                                 src_y: i32,
                                 width: i32,
                                 height: i32) {
        self.bind_color_texture(texture_id);
        gl::copy_tex_sub_image_2d(gl::TEXTURE_2D,
                                  0,
                                  dest_x,
                                  dest_y,
                                  src_x as gl::GLint,
                                  src_y as gl::GLint,
                                  width as gl::GLint,
                                  height as gl::GLint);
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    fn clear_vertex_array(&mut self) {
        debug_assert!(self.inside_frame);
        gl::bind_vertex_array(0);
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    fn clear_vertex_array(&mut self) {
        debug_assert!(self.inside_frame);
        gl::disable_vertex_attrib_array(VertexAttribute::Position as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::ColorRectTL as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::ColorRectTR as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::ColorRectBR as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::ColorRectBL as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::ColorTexCoordRectTop as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::ColorTexCoordRectBottom as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::MaskTexCoordRectTop as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::MaskTexCoordRectBottom as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::BorderRadii as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::BorderPosition as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::BlurRadius as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::DestTextureSize as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::SourceTextureSize as gl::GLuint);
        gl::disable_vertex_attrib_array(VertexAttribute::Misc as gl::GLuint);
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

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    pub fn bind_vao(&mut self, vao_id: VAOId) {
        debug_assert!(self.inside_frame);

        if self.bound_vao != vao_id {
            if let Some(prev_vao) = self.vaos.get(&self.bound_vao) {
                prev_vao.vertex_format.unbind();
            }

            let vao = self.vaos.get(&vao_id).unwrap();
            self.bound_vao = vao_id;

            vao.ibo_id.bind();
            vao.vertex_format.bind(vao.main_vbo_id, vao.aux_vbo_id, 0);
        }
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    fn create_vao_with_vbos(&mut self,
                            format: VertexFormat,
                            main_vbo_id: VBOId,
                            aux_vbo_id: Option<VBOId>,
                            ibo_id: IBOId,
                            _: u32,
                            owns_vbos: bool)
                            -> VAOId {
        debug_assert!(self.inside_frame);

        let vao_id = self.next_vao_id;
        self.next_vao_id += 1;

        format.bind(main_vbo_id, aux_vbo_id, 0);

        let vao = VAO {
            id: vao_id,
            vertex_format: format,
            main_vbo_id: main_vbo_id,
            aux_vbo_id: aux_vbo_id,
            ibo_id: ibo_id,
            owns_vbos: owns_vbos,
        };

        let vao_id = VAOId(vao_id);

        debug_assert!(!self.vaos.contains_key(&vao_id));
        self.vaos.insert(vao_id, vao);

        vao_id
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    pub fn create_vao(&mut self, format: VertexFormat, quad_vertex_buffer: Option<VBOId>)
                      -> VAOId {
        debug_assert!(self.inside_frame);

        let buffer_ids = gl::gen_buffers(2);
        let ibo_id = IBOId(buffer_ids[0]);
        let (main_vbo_id, aux_vbo_id) = if format == VertexFormat::Rectangles {
            (quad_vertex_buffer.expect("A quad vertex buffer must be supplied to `create_vao()` if
                                        we are to render rectangles!"),
             Some(VBOId(buffer_ids[1])))
        } else {
            (VBOId(buffer_ids[1]), None)
        };

        self.create_vao_with_vbos(format, main_vbo_id, aux_vbo_id, ibo_id, 0, true)
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    fn create_vao_with_vbos(&mut self,
                            format: VertexFormat,
                            main_vbo_id: VBOId,
                            aux_vbo_id: Option<VBOId>,
                            ibo_id: IBOId,
                            offset: gl::GLuint,
                            owns_vbos: bool)
                            -> VAOId {
        debug_assert!(self.inside_frame);

        let vao_ids = gl::gen_vertex_arrays(1);
        let vao_id = vao_ids[0];

        gl::bind_vertex_array(vao_id);

        format.bind(main_vbo_id, aux_vbo_id, offset);

        let vao = VAO {
            id: vao_id,
            vertex_format: format,
            main_vbo_id: main_vbo_id,
            aux_vbo_id: aux_vbo_id,
            ibo_id: ibo_id,
            owns_vbos: owns_vbos,
        };

        gl::bind_vertex_array(0);

        let vao_id = VAOId(vao_id);

        debug_assert!(!self.vaos.contains_key(&vao_id));
        self.vaos.insert(vao_id, vao);

        vao_id
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    pub fn create_vao(&mut self, format: VertexFormat, quad_vertex_buffer: Option<VBOId>)
                      -> VAOId {
        debug_assert!(self.inside_frame);

        let buffer_ids = gl::gen_buffers(2);
        let ibo_id = IBOId(buffer_ids[0]);
        let (main_vbo_id, aux_vbo_id) = if format == VertexFormat::Rectangles {
            (quad_vertex_buffer.expect("A quad vertex buffer must be supplied to `create_vao()` if
                                        we are to render rectangles!"),
             Some(VBOId(buffer_ids[1])))
        } else {
            (VBOId(buffer_ids[1]), None)
        };

        self.create_vao_with_vbos(format, main_vbo_id, aux_vbo_id, ibo_id, 0, true)
    }

    #[inline(never)]
    pub fn create_similar_vao(&mut self,
                              format: VertexFormat,
                              source_vao_id: VAOId,
                              offset: gl::GLuint)
                              -> VAOId {
        let &VAO {
            main_vbo_id,
            aux_vbo_id,
            ibo_id,
            ..
        } = self.vaos.get(&source_vao_id).expect("Bad VAO ID in `create_similar_vao()`!");
        self.create_vao_with_vbos(format, main_vbo_id, aux_vbo_id, ibo_id, offset, false)
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    pub fn create_quad_vertex_buffer(&mut self) -> VBOId {
        let buffer_id = VBOId(gl::gen_buffers(1)[0]);
        buffer_id.bind();
        let mut buffer: Vec<_> =
            (0..0x10000).flat_map(|_| QUAD_VERTICES.iter().cloned()).collect();
        gl::buffer_data(gl::ARRAY_BUFFER, &buffer[..], gl::STATIC_DRAW);
        buffer_id
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    pub fn create_quad_vertex_buffer(&mut self) -> VBOId {
        let buffer_id = VBOId(gl::gen_buffers(1)[0]);
        buffer_id.bind();
        gl::buffer_data(gl::ARRAY_BUFFER, &QUAD_VERTICES, gl::STATIC_DRAW);
        buffer_id
    }

    pub fn update_vao_main_vertices<V>(&mut self,
                                       vao_id: VAOId,
                                       vertices: &[V],
                                       usage_hint: VertexUsageHint) {
        debug_assert!(self.inside_frame);

        let vao = self.vaos.get(&vao_id).unwrap();
        debug_assert!(self.bound_vao == vao_id);

        vao.main_vbo_id.bind();
        gl::buffer_data(gl::ARRAY_BUFFER, &vertices, usage_hint.to_gl());
    }

    pub fn update_vao_aux_vertices<V>(&mut self,
                                      vao_id: VAOId,
                                      vertices: &[V],
                                      usage_hint: VertexUsageHint) {
        debug_assert!(self.inside_frame);

        let vao = self.vaos.get(&vao_id).unwrap();
        debug_assert!(self.bound_vao == vao_id);

        vao.aux_vbo_id.as_ref().unwrap().bind();
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

    pub fn draw_triangles_u16(&mut self, first_vertex: i32, index_count: i32) {
        debug_assert!(self.inside_frame);
        gl::draw_elements(gl::TRIANGLES,
                          index_count,
                          gl::UNSIGNED_SHORT,
                          first_vertex as u32 * 2);
    }

    pub fn draw_triangles_u32(&mut self, first_vertex: i32, index_count: i32) {
        debug_assert!(self.inside_frame);
        gl::draw_elements(gl::TRIANGLES,
                          index_count,
                          gl::UNSIGNED_INT,
                          first_vertex as u32 * 4);
    }

    pub fn draw_nonindexed_lines(&mut self, first_vertex: i32, vertex_count: i32) {
        debug_assert!(self.inside_frame);
        gl::draw_arrays(gl::LINES,
                          first_vertex,
                          vertex_count);
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    pub fn draw_triangles_instanced_u16(&mut self,
                                        first_vertex: i32,
                                        index_count: i32,
                                        instance_count: i32) {
        debug_assert!(self.inside_frame);
        gl::draw_arrays(gl::TRIANGLES, first_vertex * index_count, instance_count * index_count);
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    pub fn draw_triangles_instanced_u16(&mut self,
                                        first_vertex: i32,
                                        index_count: i32,
                                        instance_count: i32) {
        debug_assert!(self.inside_frame);
        gl::draw_arrays_instanced(gl::TRIANGLES, first_vertex as i32, index_count, instance_count);
    }

    pub fn delete_vao(&mut self, vao_id: VAOId) {
        self.vaos.remove(&vao_id).expect(&format!("unable to remove vao {:?}", vao_id));
        if self.bound_vao == vao_id {
            self.bound_vao = VAOId(0);
        }
    }

    pub fn end_frame(&mut self) {
        self.bind_render_target(None);

        debug_assert!(self.inside_frame);
        self.inside_frame = false;

        gl::bind_texture(gl::TEXTURE_2D, 0);
        gl::use_program(0);
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        //self.file_watcher.exit();
    }
}
