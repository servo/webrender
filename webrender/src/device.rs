/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::Matrix4D;
use fnv::FnvHasher;
use gleam::gl;
use internal_types::{PackedVertex, PackedVertexForQuad};
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
use webrender_traits::{ColorF, ImageFormat};

#[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
const GL_FORMAT_A: gl::GLuint = gl::RED;

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
const GL_FORMAT_A: gl::GLuint = gl::ALPHA;

#[cfg(any(target_os = "windows", all(unix, not(target_os = "android"))))]
const GL_FORMAT_BGRA: gl::GLuint = gl::BGRA;

#[cfg(target_os = "android")]
const GL_FORMAT_BGRA: gl::GLuint = gl::BGRA_EXT;

#[cfg(not(any(target_arch = "arm", target_arch = "aarch64")))]
const SHADER_VERSION: &'static str = "#version 150\n";

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
const SHADER_VERSION: &'static str = "#version 300 es\n";

static SHADER_PREAMBLE: &'static str = "shared.glsl";

pub type ViewportDimensions = [u32; 2];

lazy_static! {
    pub static ref MAX_TEXTURE_SIZE: gl::GLint = {
        gl::get_integer_v(gl::MAX_TEXTURE_SIZE)
    };
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TextureTarget {
    Default,
    Array,
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

                self.set_divisors(0);

                let vertex_stride = mem::size_of::<PackedVertex>() as gl::GLuint;

                gl::vertex_attrib_pointer(VertexAttribute::Position as gl::GLuint,
                                          2,
                                          gl::FLOAT,
                                          false,
                                          vertex_stride as gl::GLint,
                                          0 + vertex_stride * offset);
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
    pub fn bind(&self) {
        gl::bind_texture(self.target, self.name);
    }

    pub fn new(name: gl::GLuint) -> TextureId {
        TextureId {
            name: name,
            target: gl::TEXTURE_2D,
        }
    }

    pub fn invalid() -> TextureId {
        TextureId {
            name: 0,
            target: gl::TEXTURE_2D,
        }
    }
}

impl ProgramId {
    fn bind(&self) {
        gl::use_program(self.0);
    }
}

impl VBOId {
    fn bind(&self) {
        gl::bind_buffer(gl::ARRAY_BUFFER, self.0);
    }
}

impl IBOId {
    fn bind(&self) {
        gl::bind_buffer(gl::ELEMENT_ARRAY_BUFFER, self.0);
    }
}

impl UBOId {
    fn _bind(&self) {
        gl::bind_buffer(gl::UNIFORM_BUFFER, self.0);
    }
}

impl FBOId {
    fn bind(&self) {
        gl::bind_framebuffer(gl::FRAMEBUFFER, self.0);
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
    prefix: Option<String>,
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
            //println!("{}", gl::get_program_info_log(self.id));
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

#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Copy, Clone)]
pub struct TextureId {
    name: gl::GLuint,
    target: gl::GLuint,
}

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct ProgramId(pub gl::GLuint);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct VAOId(gl::GLuint);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct FBOId(gl::GLuint);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct VBOId(gl::GLuint);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
struct IBOId(gl::GLuint);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct UBOId(gl::GLuint);

const MAX_EVENTS_PER_FRAME: usize = 256;
const MAX_PROFILE_FRAMES: usize = 4;

#[derive(Debug, Clone)]
pub struct GpuSample<T> {
    pub tag: T,
    pub time_ns: u64,
}

pub struct GpuFrameProfile<T> {
    queries: Vec<gl::GLuint>,
    samples: Vec<GpuSample<T>>,
    next_query: usize,
    pending_query: gl::GLuint,
}

impl<T> GpuFrameProfile<T> {
    #[cfg(not(target_os = "android"))]
    fn new() -> GpuFrameProfile<T> {
        let queries = gl::gen_queries(MAX_EVENTS_PER_FRAME as gl::GLint);

        GpuFrameProfile {
            queries: queries,
            samples: Vec::new(),
            next_query: 0,
            pending_query: 0,
        }
    }

    #[cfg(target_os = "android")]
    fn new() -> GpuFrameProfile<T> {
        GpuFrameProfile {
            queries: Vec::new(),
            samples: Vec::new(),
            next_query: 0,
            pending_query: 0,
        }
    }

    fn begin_frame(&mut self) {
        self.next_query = 0;
        self.pending_query = 0;
        self.samples.clear();
    }

    #[cfg(not(target_os = "android"))]
    fn end_frame(&mut self) {
        if self.pending_query != 0 {
            gl::end_query(gl::TIME_ELAPSED);
        }
    }

    #[cfg(target_os = "android")]
    fn end_frame(&mut self) {
    }

    #[cfg(not(target_os = "android"))]
    fn add_marker(&mut self, tag: T) {
        if self.pending_query != 0 {
            gl::end_query(gl::TIME_ELAPSED);
        }

        if self.next_query < MAX_EVENTS_PER_FRAME {
            self.pending_query = self.queries[self.next_query];
            gl::begin_query(gl::TIME_ELAPSED, self.pending_query);
            self.samples.push(GpuSample {
                tag: tag,
                time_ns: 0,
            });
        } else {
            self.pending_query = 0;
        }

        self.next_query += 1;
    }

    #[cfg(target_os = "android")]
    fn add_marker(&mut self, tag: T) {
        self.samples.push(GpuSample {
            tag: tag,
            time_ns: 0,
        });
    }

    fn is_valid(&self) -> bool {
        self.next_query <= MAX_EVENTS_PER_FRAME
    }

    #[cfg(not(target_os = "android"))]
    fn build_samples(&mut self) -> Vec<GpuSample<T>> {
        for (index, sample) in self.samples.iter_mut().enumerate() {
            sample.time_ns = gl::get_query_object_ui64v(self.queries[index], gl::QUERY_RESULT)
        }

        mem::replace(&mut self.samples, Vec::new())
    }

    #[cfg(target_os = "android")]
    fn build_samples(&mut self) -> Vec<GpuSample<T>> {
        mem::replace(&mut self.samples, Vec::new())
    }
}

impl<T> Drop for GpuFrameProfile<T> {
    #[cfg(not(target_os = "android"))]
    fn drop(&mut self) {
        gl::delete_queries(&self.queries);
    }

    #[cfg(target_os = "android")]
    fn drop(&mut self) {
    }
}

pub struct GpuProfiler<T> {
    frames: [GpuFrameProfile<T>; MAX_PROFILE_FRAMES],
    next_frame: usize,
}

impl<T> GpuProfiler<T> {
    pub fn new() -> GpuProfiler<T> {
        GpuProfiler {
            next_frame: 0,
            frames: [
                      GpuFrameProfile::new(),
                      GpuFrameProfile::new(),
                      GpuFrameProfile::new(),
                      GpuFrameProfile::new(),
                    ],
        }
    }

    pub fn build_samples(&mut self) -> Option<Vec<GpuSample<T>>> {
        let frame = &mut self.frames[self.next_frame];
        if frame.is_valid() {
            Some(frame.build_samples())
        } else {
            None
        }
    }

    pub fn begin_frame(&mut self) {
        let frame = &mut self.frames[self.next_frame];
        frame.begin_frame();
    }

    pub fn end_frame(&mut self) {
        let frame = &mut self.frames[self.next_frame];
        frame.end_frame();
        self.next_frame = (self.next_frame + 1) % MAX_PROFILE_FRAMES;
    }

    pub fn add_marker(&mut self, tag: T) {
        let frame = &mut self.frames[self.next_frame];
        frame.add_marker(tag);
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

pub struct Capabilities {
    pub max_ubo_size: usize,
    pub supports_multisampling: bool,
}

pub struct Device {
    // device state
    bound_textures: [TextureId; 16],
    bound_program: ProgramId,
    bound_vao: VAOId,
    bound_fbo: FBOId,
    default_fbo: gl::GLuint,
    device_pixel_ratio: f32,

    // HW or API capabilties
    capabilities: Capabilities,

    // debug
    inside_frame: bool,

    // resources
    resource_path: PathBuf,
    textures: HashMap<TextureId, Texture, BuildHasherDefault<FnvHasher>>,
    raw_textures: HashMap<TextureId, (u32, u32, u32, u32), BuildHasherDefault<FnvHasher>>,
    programs: HashMap<ProgramId, Program, BuildHasherDefault<FnvHasher>>,
    vaos: HashMap<VAOId, VAO, BuildHasherDefault<FnvHasher>>,

    // misc.
    shader_preamble: String,
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
        path.push(SHADER_PREAMBLE);
        let mut f = File::open(&path).unwrap();
        let mut shader_preamble = String::new();
        f.read_to_string(&mut shader_preamble).unwrap();
        //file_watcher.add_watch(path);

        Device {
            resource_path: resource_path,
            device_pixel_ratio: device_pixel_ratio,
            inside_frame: false,

            capabilities: Capabilities {
                max_ubo_size: gl::get_integer_v(gl::MAX_UNIFORM_BLOCK_SIZE) as usize,
                supports_multisampling: false, //TODO
            },

            bound_textures: [ TextureId::invalid(); 16 ],
            bound_program: ProgramId(0),
            bound_vao: VAOId(0),
            bound_fbo: FBOId(0),
            default_fbo: 0,

            textures: HashMap::with_hasher(Default::default()),
            raw_textures: HashMap::with_hasher(Default::default()),
            programs: HashMap::with_hasher(Default::default()),
            vaos: HashMap::with_hasher(Default::default()),

            shader_preamble: shader_preamble,

            next_vao_id: 1,
            //file_watcher: file_watcher,
        }
    }

    pub fn get_capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    pub fn compile_shader(path: &PathBuf,
                          shader_type: gl::GLenum,
                          shader_preamble: &[String],
                          panic_on_fail: bool)
                          -> Option<gl::GLuint> {
        debug!("compile {:?}", path);

        let mut f = File::open(&path).unwrap();
        let mut s = String::new();
        s.push_str(SHADER_VERSION);
        for prefix in shader_preamble {
            s.push_str(&prefix);
        }
        f.read_to_string(&mut s).unwrap();

        let id = gl::create_shader(shader_type);
        let mut source = Vec::new();
        source.extend_from_slice(s.as_bytes());
        gl::shader_source(id, &[&source[..]]);
        gl::compile_shader(id);
        let log = gl::get_shader_info_log(id);
        if gl::get_shader_iv(id, gl::COMPILE_STATUS) == (0 as gl::GLint) {
            println!("Failed to compile shader: {:?}\n{}", path, log);
            if panic_on_fail {
                panic!("-- Shader compile failed - exiting --");
            }

            None
        } else {
            if !log.is_empty() {
                println!("Warnings detected on shader: {:?}\n{}", path, log);
            }
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
        for i in 0..self.bound_textures.len() {
            self.bound_textures[i] = TextureId::invalid();
            gl::active_texture(gl::TEXTURE0 + i as gl::GLuint);
            gl::bind_texture(gl::TEXTURE_2D, 0);
        }

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

    pub fn bind_texture(&mut self,
                        sampler: TextureSampler,
                        texture_id: TextureId) {
        debug_assert!(self.inside_frame);

        let sampler_index = sampler as usize;
        if self.bound_textures[sampler_index] != texture_id {
            self.bound_textures[sampler_index] = texture_id;
            gl::active_texture(gl::TEXTURE0 + sampler_index as gl::GLuint);
            texture_id.bind();
            gl::active_texture(gl::TEXTURE0);
        }
    }

    pub fn bind_render_target(&mut self,
                              texture_id: Option<(TextureId, i32)>,
                              dimensions: Option<ViewportDimensions>) {
        debug_assert!(self.inside_frame);

        let fbo_id = texture_id.map_or(FBOId(self.default_fbo), |texture_id| {
            self.textures.get(&texture_id.0).unwrap().fbo_ids[texture_id.1 as usize]
        });

        if self.bound_fbo != fbo_id {
            self.bound_fbo = fbo_id;
            fbo_id.bind();
        }

        if let Some(dimensions) = dimensions {
            gl::viewport(0, 0, dimensions[0] as gl::GLint, dimensions[1] as gl::GLint);
        }
    }

    pub fn bind_program(&mut self,
                        program_id: ProgramId,
                        projection: &Matrix4D<f32>) {
        debug_assert!(self.inside_frame);

        if self.bound_program != program_id {
            self.bound_program = program_id;
            program_id.bind();
        }

        let program = self.programs.get(&program_id).unwrap();
        self.set_uniforms(program, projection);
    }

    pub fn create_texture_ids(&mut self,
                              count: i32,
                              target: TextureTarget) -> Vec<TextureId> {
        let id_list = gl::gen_textures(count);
        let mut texture_ids = Vec::new();

        let target = match target {
            TextureTarget::Default => gl::TEXTURE_2D,
            TextureTarget::Array => gl::TEXTURE_2D_ARRAY,
        };

        for id in id_list {
            let texture_id = TextureId {
                name: id,
                target: target,
            };

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

    pub fn remove_raw_texture(&mut self, texture_id: TextureId) {
        self.raw_textures.remove(&texture_id);
    }

    fn set_texture_parameters(&mut self, target: gl::GLuint, filter: TextureFilter) {
        let filter = match filter {
            TextureFilter::Nearest => {
                gl::NEAREST
            }
            TextureFilter::Linear => {
                gl::LINEAR
            }
        };

        gl::tex_parameter_i(target, gl::TEXTURE_MAG_FILTER, filter as gl::GLint);
        gl::tex_parameter_i(target, gl::TEXTURE_MIN_FILTER, filter as gl::GLint);

        gl::tex_parameter_i(target, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as gl::GLint);
        gl::tex_parameter_i(target, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as gl::GLint);
    }

    fn upload_texture_image(&mut self,
                            target: gl::GLuint,
                            width: u32,
                            height: u32,
                            internal_format: u32,
                            format: u32,
                            type_: u32,
                            pixels: Option<&[u8]>) {
        gl::tex_image_2d(target,
                         0,
                         internal_format as gl::GLint,
                         width as gl::GLint, height as gl::GLint,
                         0,
                         format,
                         type_,
                         pixels);
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
            texture.mode = mode;
        }

        let (internal_format, gl_format) = gl_texture_formats_for_image_format(format);
        let type_ = gl_type_for_texture_format(format);

        match mode {
            RenderTargetMode::SimpleRenderTarget => {
                self.bind_texture(TextureSampler::Slot0, texture_id);
                self.set_texture_parameters(texture_id.target, filter);
                self.upload_texture_image(texture_id.target,
                                          width,
                                          height,
                                          internal_format as u32,
                                          gl_format,
                                          type_,
                                          None);
                self.create_fbo_for_texture_if_necessary(texture_id, None);
            }
            RenderTargetMode::LayerRenderTarget(layer_count) => {
                self.bind_texture(TextureSampler::Slot0, texture_id);
                self.set_texture_parameters(texture_id.target, filter);
                self.create_fbo_for_texture_if_necessary(texture_id, Some(layer_count));
            }
            RenderTargetMode::None => {
                self.bind_texture(TextureSampler::Slot0, texture_id);
                self.set_texture_parameters(texture_id.target, filter);
                self.upload_texture_image(texture_id.target,
                                          width,
                                          height,
                                          internal_format as u32,
                                          gl_format,
                                          type_,
                                          pixels);
            }
        }
    }

    pub fn create_fbo_for_texture_if_necessary(&mut self,
                                               texture_id: TextureId,
                                               layer_count: Option<i32>) {
        let texture = self.textures.get_mut(&texture_id).unwrap();

        match layer_count {
            Some(layer_count) => {
                debug_assert!(layer_count > 0);

                // If we have enough layers allocated already, just use them.
                // TODO(gw): Probably worth removing some after a while if
                //           there is a surplus?
                let current_layer_count = texture.fbo_ids.len() as i32;
                if current_layer_count >= layer_count {
                    return;
                }

                let (internal_format, gl_format) = gl_texture_formats_for_image_format(texture.format);
                let type_ = gl_type_for_texture_format(texture.format);

                gl::tex_image_3d(texture_id.target,
                                 0,
                                 internal_format as gl::GLint,
                                 texture.width as gl::GLint,
                                 texture.height as gl::GLint,
                                 layer_count,
                                 0,
                                 gl_format,
                                 type_,
                                 None);

                let needed_layer_count = layer_count - current_layer_count;
                let new_fbos = gl::gen_framebuffers(needed_layer_count);
                texture.fbo_ids.extend(new_fbos.into_iter().map(|id| FBOId(id)));

                for (fbo_index, fbo_id) in texture.fbo_ids.iter().enumerate() {
                    gl::bind_framebuffer(gl::FRAMEBUFFER, fbo_id.0);
                    gl::framebuffer_texture_layer(gl::FRAMEBUFFER,
                                                  gl::COLOR_ATTACHMENT0,
                                                  texture_id.name,
                                                  0,
                                                  fbo_index as gl::GLint);
                }
            }
            None => {
                debug_assert!(texture.fbo_ids.len() == 0 || texture.fbo_ids.len() == 1);
                if texture.fbo_ids.is_empty() {
                    let new_fbo = gl::gen_framebuffers(1)[0];

                    gl::bind_framebuffer(gl::FRAMEBUFFER, new_fbo);

                    gl::framebuffer_texture_2d(gl::FRAMEBUFFER,
                                               gl::COLOR_ATTACHMENT0,
                                               texture_id.target,
                                               texture_id.name,
                                               0);

                    texture.fbo_ids.push(FBOId(new_fbo));
                }
            }
        }

        gl::bind_framebuffer(gl::FRAMEBUFFER, self.default_fbo);
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

        let temp_texture_id = self.create_texture_ids(1, TextureTarget::Default)[0];
        self.init_texture(temp_texture_id, old_width, old_height, format, filter, mode, None);
        self.create_fbo_for_texture_if_necessary(temp_texture_id, None);

        self.bind_render_target(Some((texture_id, 0)), None);
        self.bind_texture(TextureSampler::Slot0, temp_texture_id);

        gl::copy_tex_sub_image_2d(temp_texture_id.target,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  old_width as i32,
                                  old_height as i32);

        self.deinit_texture(texture_id);
        self.init_texture(texture_id, new_width, new_height, format, filter, mode, None);
        self.create_fbo_for_texture_if_necessary(texture_id, None);
        self.bind_render_target(Some((temp_texture_id, 0)), None);
        self.bind_texture(TextureSampler::Slot0, texture_id);

        gl::copy_tex_sub_image_2d(texture_id.target,
                                  0,
                                  0,
                                  0,
                                  0,
                                  0,
                                  old_width as i32,
                                  old_height as i32);

        self.bind_render_target(None, None);
        self.deinit_texture(temp_texture_id);
    }

    pub fn deinit_texture(&mut self, texture_id: TextureId) {
        debug_assert!(self.inside_frame);

        self.bind_texture(TextureSampler::Slot0, texture_id);

        let texture = self.textures.get_mut(&texture_id).unwrap();
        let (internal_format, gl_format) = gl_texture_formats_for_image_format(texture.format);
        let type_ = gl_type_for_texture_format(texture.format);

        gl::tex_image_2d(texture_id.target,
                         0,
                         internal_format,
                         0,
                         0,
                         0,
                         gl_format,
                         type_,
                         None);

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
                          base_filename: &str,
                          include_filename: &str) -> ProgramId {
        self.create_program_with_prefix(base_filename, &[include_filename], None)
    }

    pub fn create_program_with_prefix(&mut self,
                                      base_filename: &str,
                                      include_filenames: &[&str],
                                      prefix: Option<String>) -> ProgramId {
        debug_assert!(self.inside_frame);

        let pid = gl::create_program();
        let base_path = self.resource_path.join(base_filename);

        let vs_path = base_path.with_extension("vs.glsl");
        //self.file_watcher.add_watch(vs_path.clone());

        let fs_path = base_path.with_extension("fs.glsl");
        //self.file_watcher.add_watch(fs_path.clone());

        let mut include = format!("// Base shader: {}\n", base_filename);
        for inc_filename in include_filenames {
            let include_path = self.resource_path.join(inc_filename).with_extension("glsl");
            File::open(&include_path).unwrap().read_to_string(&mut include).unwrap();
        }

        let shared_path = base_path.with_extension("glsl");
        if let Ok(mut f) = File::open(&shared_path) {
            f.read_to_string(&mut include).unwrap();
        }

        let program = Program {
            id: pid,
            u_transform: -1,
            vs_path: vs_path,
            fs_path: fs_path,
            prefix: prefix,
            vs_id: None,
            fs_id: None,
        };

        let program_id = ProgramId(pid);

        debug_assert!(self.programs.contains_key(&program_id) == false);
        self.programs.insert(program_id, program);

        self.load_program(program_id, include, true);

        program_id
    }

    fn load_program(&mut self,
                    program_id: ProgramId,
                    include: String,
                    panic_on_fail: bool) {
        debug_assert!(self.inside_frame);

        let program = self.programs.get_mut(&program_id).unwrap();

        let mut vs_preamble = Vec::new();
        let mut fs_preamble = Vec::new();

        vs_preamble.push("#define WR_VERTEX_SHADER\n".to_owned());
        fs_preamble.push("#define WR_FRAGMENT_SHADER\n".to_owned());

        if let Some(ref prefix) = program.prefix {
            vs_preamble.push(prefix.clone());
            fs_preamble.push(prefix.clone());
        }

        vs_preamble.push(self.shader_preamble.to_owned());
        fs_preamble.push(self.shader_preamble.to_owned());

        vs_preamble.push(include.clone());
        fs_preamble.push(include);

        // todo(gw): store shader ids so they can be freed!
        let vs_id = Device::compile_shader(&program.vs_path,
                                           gl::VERTEX_SHADER,
                                           &vs_preamble,
                                           panic_on_fail);
        let fs_id = Device::compile_shader(&program.fs_path,
                                           gl::FRAGMENT_SHADER,
                                           &fs_preamble,
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
                let u_texture_0 = gl::get_uniform_location(program.id, "sTexture0");
                if u_texture_0 != -1 {
                    gl::uniform_1i(u_texture_0, TextureSampler::Slot0 as i32);
                }
                let u_texture_1 = gl::get_uniform_location(program.id, "sTexture1");
                if u_texture_1 != -1 {
                    gl::uniform_1i(u_texture_1, TextureSampler::Slot1 as i32);
                }
                let u_device_pixel_ratio = gl::get_uniform_location(program.id, "uDevicePixelRatio");
                if u_device_pixel_ratio != -1 {
                    gl::uniform_1f(u_device_pixel_ratio, self.device_pixel_ratio);
                }

                let u_cache = gl::get_uniform_location(program.id, "sCache");
                if u_cache != -1 {
                    gl::uniform_1i(u_cache, TextureSampler::Cache as i32);
                }

                let u_layers = gl::get_uniform_location(program.id, "sLayers");
                if u_layers != -1 {
                    gl::uniform_1i(u_layers, TextureSampler::Layers as i32);
                }

                let u_tasks = gl::get_uniform_location(program.id, "sRenderTasks");
                if u_tasks != -1 {
                    gl::uniform_1i(u_tasks, TextureSampler::RenderTasks as i32);
                }

                let u_prim_geom = gl::get_uniform_location(program.id, "sPrimGeometry");
                if u_prim_geom != -1 {
                    gl::uniform_1i(u_prim_geom, TextureSampler::Geometry as i32);
                }

                let u_data16 = gl::get_uniform_location(program.id, "sData16");
                if u_data16 != -1 {
                    gl::uniform_1i(u_data16, TextureSampler::Data16 as i32);
                }

                let u_data32 = gl::get_uniform_location(program.id, "sData32");
                if u_data32 != -1 {
                    gl::uniform_1i(u_data32, TextureSampler::Data32 as i32);
                }

                let u_data64 = gl::get_uniform_location(program.id, "sData64");
                if u_data64 != -1 {
                    gl::uniform_1i(u_data64, TextureSampler::Data64 as i32);
                }

                let u_data128 = gl::get_uniform_location(program.id, "sData128");
                if u_data128 != -1 {
                    gl::uniform_1i(u_data128, TextureSampler::Data128    as i32);
                }
            }
        }
    }

/*
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
    }*/

    pub fn get_uniform_location(&self, program_id: ProgramId, name: &str) -> UniformLocation {
        let ProgramId(program_id) = program_id;
        UniformLocation(gl::get_uniform_location(program_id, name))
    }

    pub fn set_uniform_2f(&self, uniform: UniformLocation, x: f32, y: f32) {
        debug_assert!(self.inside_frame);
        let UniformLocation(location) = uniform;
        gl::uniform_2f(location, x, y);
    }

    fn set_uniforms(&self, program: &Program, transform: &Matrix4D<f32>) {
        debug_assert!(self.inside_frame);
        gl::uniform_matrix_4fv(program.u_transform,
                               false,
                               &transform.to_row_major_array());
    }

    fn update_image_for_2d_texture(&mut self,
                                   target: gl::GLuint,
                                   x0: gl::GLint,
                                   y0: gl::GLint,
                                   width: gl::GLint,
                                   height: gl::GLint,
                                   format: gl::GLuint,
                                   data: &[u8]) {
        gl::tex_sub_image_2d(target,
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
                          stride: Option<u32>,
                          data: &[u8]) {
        debug_assert!(self.inside_frame);

        let mut expanded_data = Vec::new();

        let (gl_format, bpp, data) = match self.textures.get(&texture_id).unwrap().format {
            ImageFormat::A8 => {
                if cfg!(any(target_arch="arm", target_arch="aarch64")) {
                    for byte in data {
                        expanded_data.push(*byte);
                        expanded_data.push(*byte);
                        expanded_data.push(*byte);
                        expanded_data.push(*byte);
                    }
                    (GL_FORMAT_BGRA, 4, expanded_data.as_slice())
                } else {
                    (GL_FORMAT_A, 1, data)
                }
            }
            ImageFormat::RGB8 => (gl::RGB, 3, data),
            ImageFormat::RGBA8 => (GL_FORMAT_BGRA, 4, data),
            ImageFormat::Invalid | ImageFormat::RGBAF32 => unreachable!(),
        };

        let row_length = match stride {
            Some(value) => value / bpp,
            None => width,
        };

        assert!(data.len() as u32 == bpp * row_length * height);

        if let Some(..) = stride {
            gl::pixel_store_i(gl::UNPACK_ROW_LENGTH, row_length as gl::GLint);
        }

        self.bind_texture(TextureSampler::Slot0, texture_id);
        self.update_image_for_2d_texture(texture_id.target,
                                         x0 as gl::GLint,
                                         y0 as gl::GLint,
                                         width as gl::GLint,
                                         height as gl::GLint,
                                         gl_format,
                                         data);

        // Reset row length to 0, otherwise the stride would apply to all texture uploads.
        if let Some(..) = stride {
            gl::pixel_store_i(gl::UNPACK_ROW_LENGTH, 0 as gl::GLint);
        }
    }

    pub fn read_framebuffer_rect(&mut self,
                                 texture_id: TextureId,
                                 dest_x: i32,
                                 dest_y: i32,
                                 src_x: i32,
                                 src_y: i32,
                                 width: i32,
                                 height: i32) {
        self.bind_texture(TextureSampler::Slot0, texture_id);
        gl::copy_tex_sub_image_2d(texture_id.target,
                                  0,
                                  dest_x,
                                  dest_y,
                                  src_x as gl::GLint,
                                  src_y as gl::GLint,
                                  width as gl::GLint,
                                  height as gl::GLint);
    }

    fn clear_vertex_array(&mut self) {
        debug_assert!(self.inside_frame);
        gl::bind_vertex_array(0);
    }

    pub fn bind_vao(&mut self, vao_id: VAOId) {
        debug_assert!(self.inside_frame);

        if self.bound_vao != vao_id {
            self.bound_vao = vao_id;

            let VAOId(id) = vao_id;
            gl::bind_vertex_array(id);
        }
    }

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

    pub fn draw_indexed_triangles_instanced_u16(&mut self,
                                                index_count: i32,
                                                instance_count: i32) {
        debug_assert!(self.inside_frame);
        gl::draw_elements_instanced(gl::TRIANGLES, index_count, gl::UNSIGNED_SHORT, 0, instance_count);
    }

    pub fn end_frame(&mut self) {
        self.bind_render_target(None, None);

        debug_assert!(self.inside_frame);
        self.inside_frame = false;

        gl::bind_texture(gl::TEXTURE_2D, 0);
        gl::use_program(0);

        for i in 0..self.bound_textures.len() {
            gl::active_texture(gl::TEXTURE0 + i as gl::GLuint);
            gl::bind_texture(gl::TEXTURE_2D, 0);
        }

        gl::active_texture(gl::TEXTURE0);
    }

    pub fn assign_ubo_binding(&self, program_id: ProgramId, name: &str, value: u32) -> u32 {
        let index = gl::get_uniform_block_index(program_id.0, name);
        gl::uniform_block_binding(program_id.0, index, value);
        index
    }

    pub fn create_ubo<T>(&self, data: &[T], binding: u32) -> UBOId {
        let ubo = gl::gen_buffers(1)[0];
        gl::bind_buffer(gl::UNIFORM_BUFFER, ubo);
        gl::buffer_data(gl::UNIFORM_BUFFER, data, gl::STATIC_DRAW);
        gl::bind_buffer_base(gl::UNIFORM_BUFFER, binding, ubo);
        UBOId(ubo)
    }

    pub fn reset_ubo(&self, binding: u32) {
        gl::bind_buffer(gl::UNIFORM_BUFFER, 0);
        gl::bind_buffer_base(gl::UNIFORM_BUFFER, binding, 0);
    }

    pub fn delete_buffer(&self, buffer: UBOId) {
        gl::delete_buffers(&[buffer.0]);
    }

    #[cfg(target_os = "android")]
    pub fn set_multisample(&self, enable: bool) {
    }

    #[cfg(not(target_os = "android"))]
    pub fn set_multisample(&self, enable: bool) {
        if self.capabilities.supports_multisampling {
            if enable {
                gl::enable(gl::MULTISAMPLE);
            } else {
                gl::disable(gl::MULTISAMPLE);
            }
        }
    }

    pub fn clear_color(&self, c: [f32; 4]) {
        gl::clear_color(c[0], c[1], c[2], c[3]);
        gl::clear(gl::COLOR_BUFFER_BIT);
    }

    pub fn disable_depth(&self) {
        gl::disable(gl::DEPTH_TEST);
    }

    pub fn disable_depth_write(&self) {
        gl::depth_mask(false);
    }

    pub fn disable_stencil(&self) {
        gl::disable(gl::STENCIL_TEST);
    }

    pub fn disable_scissor(&self) {
        gl::disable(gl::SCISSOR_TEST);
    }

    pub fn set_blend(&self, enable: bool) {
        if enable {
            gl::enable(gl::BLEND);
        } else {
            gl::disable(gl::BLEND);
        }
    }

    pub fn set_blend_mode_premultiplied_alpha(&self) {
        gl::blend_func(gl::SRC_ALPHA, gl::ZERO);
        gl::blend_equation(gl::FUNC_ADD);
    }

    pub fn set_blend_mode_alpha(&self) {
        //gl::blend_func(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
        gl::blend_func_separate(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA,
                                gl::ONE, gl::ONE);
        gl::blend_equation(gl::FUNC_ADD);
    }

    pub fn set_blend_mode_subpixel(&self, color: ColorF) {
        gl::blend_color(color.r, color.g, color.b, color.a);
        gl::blend_func(gl::CONSTANT_COLOR, gl::ONE_MINUS_SRC_COLOR);
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        //self.file_watcher.exit();
    }
}

fn gl_texture_formats_for_image_format(format: ImageFormat) -> (gl::GLint, gl::GLuint) {
    match format {
        ImageFormat::A8 => {
            if cfg!(any(target_arch="arm", target_arch="aarch64")) {
                (GL_FORMAT_BGRA as gl::GLint, GL_FORMAT_BGRA)
            } else {
                (GL_FORMAT_A as gl::GLint, GL_FORMAT_A)
            }
        },
        ImageFormat::RGB8 => (gl::RGB as gl::GLint, gl::RGB),
        ImageFormat::RGBA8 => {
            if cfg!(any(target_arch="arm", target_arch="aarch64")) {
                (GL_FORMAT_BGRA as gl::GLint, GL_FORMAT_BGRA)
            } else {
                (gl::RGBA as gl::GLint, GL_FORMAT_BGRA)
            }
        }
        ImageFormat::RGBAF32 => (gl::RGBA32F as gl::GLint, gl::RGBA),
        ImageFormat::Invalid => unreachable!(),
    }
}

fn gl_type_for_texture_format(format: ImageFormat) -> gl::GLuint {
    match format {
        ImageFormat::RGBAF32 => gl::FLOAT,
        _ => gl::UNSIGNED_BYTE,
    }
}

