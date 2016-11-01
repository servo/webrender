/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! The webrender API.
//!
//! The `webrender::renderer` module provides the interface to webrender, which
//! is accessible through [`Renderer`][renderer]
//!
//! [renderer]: struct.Renderer.html

use batch::RasterBatch;
use debug_render::DebugRenderer;
use device::{Device, ProgramId, TextureId, UniformLocation, VertexFormat, GpuProfiler};
use device::{TextureFilter, VAOId, VertexUsageHint, FileWatcherHandler, TextureTarget};
use euclid::{Matrix4D, Point2D, Rect, Size2D};
use fnv::FnvHasher;
use internal_types::{RendererFrame, ResultMsg, TextureUpdateOp};
use internal_types::{TextureUpdateDetails, TextureUpdateList, PackedVertex, RenderTargetMode};
use internal_types::{ORTHO_NEAR_PLANE, ORTHO_FAR_PLANE, DevicePoint};
use internal_types::{PackedVertexForTextureCacheUpdate};
use internal_types::{AxisDirection, TextureSampler, GLContextHandleWrapper};
use ipc_channel::ipc;
use profiler::{Profiler, BackendProfileCounters};
use profiler::{GpuProfileTag, RendererProfileTimers, RendererProfileCounters};
use render_backend::RenderBackend;
use resource_cache::DummyResources;
use std::cmp;
use std::collections::HashMap;
use std::f32;
use std::hash::BuildHasherDefault;
use std::mem;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use texture_cache::{BorderType, TextureCache, TextureInsertOp};
use tiling::{self, Frame, FrameBuilderConfig, PrimitiveBatchData};
use tiling::{RenderTarget, ClearTile};
use time::precise_time_ns;
use util::TransformedRectKind;
use webrender_traits::{ColorF, Epoch, PipelineId, RenderNotifier, RenderDispatcher};
use webrender_traits::{ImageFormat, RenderApiSender, RendererKind};

pub const BLUR_INFLATION_FACTOR: u32 = 3;
pub const MAX_RASTER_OP_SIZE: u32 = 2048;
pub const MAX_VERTEX_TEXTURE_WIDTH: usize = 1024;

const UBO_BIND_DATA: u32 = 1;

// Black
const GPU_TAG_CACHE_BOX_SHADOW: GpuProfileTag = GpuProfileTag { label: "C_BoxShadow", color: ColorF { r: 0.0, g: 0.0, b: 0.0, a: 1.0 } };

// White
const GPU_TAG_INIT: GpuProfileTag = GpuProfileTag { label: "Init", color: ColorF { r: 1.0, g: 1.0, b: 1.0, a: 1.0 } };

// Grey
const GPU_TAG_SETUP_TARGET: GpuProfileTag = GpuProfileTag { label: "Target", color: ColorF { r: 0.5, g: 0.5, b: 0.5, a: 1.0 } };

// Black
const GPU_TAG_CLEAR_TILES: GpuProfileTag = GpuProfileTag { label: "Clear Tiles", color: ColorF { r: 0.0, g: 0.0, b: 0.0, a: 1.0 } };

// Red / dark red
const GPU_TAG_PRIM_RECT: GpuProfileTag = GpuProfileTag { label: "Rect", color: ColorF { r: 1.0, g: 0.0, b: 0.0, a: 1.0 } };
const GPU_TAG_PRIM_RECT_CLIP: GpuProfileTag = GpuProfileTag { label: "RectClip", color: ColorF { r: 0.7, g: 0.0, b: 0.0, a: 1.0 } };

// Green / dark green
const GPU_TAG_PRIM_IMAGE: GpuProfileTag = GpuProfileTag { label: "Image", color: ColorF { r: 0.0, g: 1.0, b: 0.0, a: 1.0 } };
const GPU_TAG_PRIM_IMAGE_CLIP: GpuProfileTag = GpuProfileTag { label: "ImageClip", color: ColorF { r: 0.0, g: 0.7, b: 0.0, a: 1.0 } };

// Light blue
const GPU_TAG_PRIM_BLEND: GpuProfileTag = GpuProfileTag { label: "Blend", color: ColorF { r: 0.8, g: 1.0, b: 1.0, a: 1.0 } };

// Magenta
const GPU_TAG_PRIM_COMPOSITE: GpuProfileTag = GpuProfileTag { label: "Composite", color: ColorF { r: 1.0, g: 0.0, b: 1.0, a: 1.0 } };

// Blue
const GPU_TAG_PRIM_TEXT_RUN: GpuProfileTag = GpuProfileTag { label: "TextRun", color: ColorF { r: 0.0, g: 0.0, b: 1.0, a: 1.0 } };

// Yellow / dark yellow
const GPU_TAG_PRIM_GRADIENT: GpuProfileTag = GpuProfileTag { label: "Gradient", color: ColorF { r: 1.0, g: 1.0, b: 0.0, a: 1.0 } };
const GPU_TAG_PRIM_GRADIENT_CLIP: GpuProfileTag = GpuProfileTag { label: "GradientClip", color: ColorF { r: 1.0, g: 1.0, b: 0.0, a: 1.0 } };
const GPU_TAG_PRIM_ANGLE_GRADIENT: GpuProfileTag = GpuProfileTag { label: "AngleGradient", color: ColorF { r: 0.7, g: 0.7, b: 0.0, a: 1.0 } };

// Cyan
const GPU_TAG_PRIM_BOX_SHADOW: GpuProfileTag = GpuProfileTag { label: "BoxShadow", color: ColorF { r: 0.0, g: 1.0, b: 1.0, a: 1.0 } };

// Orange
const GPU_TAG_PRIM_BORDER: GpuProfileTag = GpuProfileTag { label: "Border", color: ColorF { r: 1.0, g: 0.5, b: 0.0, a: 1.0 } };

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BlendMode {
    None,
    Alpha,
}

struct VertexDataTexture {
    id: TextureId,
}

impl VertexDataTexture {
    fn new(device: &mut Device) -> VertexDataTexture {
        let id = device.create_texture_ids(1, TextureTarget::Default)[0];

        VertexDataTexture {
            id: id,
        }
    }

    fn init<T: Default>(&mut self,
                        device: &mut Device,
                        data: &mut Vec<T>) {
        if data.is_empty() {
            return;
        }

        let item_size = mem::size_of::<T>();
        debug_assert!(item_size % 16 == 0);
        let vecs_per_item = item_size / 16;

        let items_per_row = MAX_VERTEX_TEXTURE_WIDTH / vecs_per_item;

        // Extend the data array to be a multiple of the row size.
        // This ensures memory safety when the array is passed to
        // OpenGL to upload to the GPU.
        while data.len() % items_per_row != 0 {
            data.push(T::default());
        }

        let width = items_per_row * vecs_per_item;
        let height = data.len() / items_per_row;

        device.init_texture(self.id,
                            width as u32,
                            height as u32,
                            ImageFormat::RGBAF32,
                            TextureFilter::Nearest,
                            RenderTargetMode::None,
                            Some(unsafe { mem::transmute(data.as_slice()) } ));
    }
}

const TRANSFORM_FEATURE: &'static [&'static str] = &["TRANSFORM"];

enum ShaderKind {
    Primitive,
    Clear,
    Cache,
}

struct LazilyCompiledShader {
    id: Option<ProgramId>,
    name: &'static str,
    kind: ShaderKind,
    max_ubo_vectors: usize,
    features: &'static [&'static str],
}

impl LazilyCompiledShader {
    fn new(kind: ShaderKind,
           name: &'static str,
           max_ubo_vectors: usize,
           features: &'static [&'static str],
           device: &mut Device,
           precache: bool) -> LazilyCompiledShader {
        let mut shader = LazilyCompiledShader {
            id: None,
            name: name,
            kind: kind,
            max_ubo_vectors: max_ubo_vectors,
            features: features,
        };

        if precache {
            shader.get(device);
        }

        shader
    }

    fn get(&mut self, device: &mut Device) -> ProgramId {
        if self.id.is_none() {
            let id = match self.kind {
                ShaderKind::Clear => {
                    create_clear_shader(self.name,
                                        device,
                                        self.max_ubo_vectors)
                }
                ShaderKind::Primitive | ShaderKind::Cache => {
                    create_prim_shader(self.name,
                                       device,
                                       self.max_ubo_vectors,
                                       self.features)
                }
            };
            self.id = Some(id);
        }

        self.id.unwrap()
    }
}

struct PrimitiveShader {
    simple: LazilyCompiledShader,
    transform: LazilyCompiledShader,
    max_items: usize,
}

struct FileWatcher {
    notifier: Arc<Mutex<Option<Box<RenderNotifier>>>>,
    result_tx: Sender<ResultMsg>,
}

impl FileWatcherHandler for FileWatcher {
    fn file_changed(&self, path: PathBuf) {
        self.result_tx.send(ResultMsg::RefreshShader(path)).ok();
        let mut notifier = self.notifier.lock();
        notifier.as_mut().unwrap().as_mut().unwrap().new_frame_ready();
    }
}

fn get_ubo_max_len<T>(max_ubo_size: usize) -> usize {
    let item_size = mem::size_of::<T>();
    let max_items = max_ubo_size / item_size;

    // TODO(gw): Clamping to 1024 since some shader compilers
    //           seem to go very slow when you have high
    //           constants for array lengths. Investigate
    //           whether this clamping actually hurts performance!
    cmp::min(max_items, 1024)
}

impl PrimitiveShader {
    fn new(name: &'static str,
           max_ubo_vectors: usize,
           max_prim_items: usize,
           device: &mut Device,
           precache: bool) -> PrimitiveShader {
        let simple = LazilyCompiledShader::new(ShaderKind::Primitive,
                                               name,
                                               max_ubo_vectors,
                                               &[],
                                               device,
                                               precache);

        let transform = LazilyCompiledShader::new(ShaderKind::Primitive,
                                                  name,
                                                  max_ubo_vectors,
                                                  TRANSFORM_FEATURE,
                                                  device,
                                                  precache);

        PrimitiveShader {
            simple: simple,
            transform: transform,
            max_items: max_prim_items,
        }
    }

    fn get(&mut self,
           device: &mut Device,
           transform_kind: TransformedRectKind) -> (ProgramId, usize) {
        let shader = match transform_kind {
            TransformedRectKind::AxisAligned => self.simple.get(device),
            TransformedRectKind::Complex => self.transform.get(device),
        };

        (shader, self.max_items)
    }
}

fn create_prim_shader(name: &'static str,
                      device: &mut Device,
                      max_ubo_vectors: usize,
                      features: &[&'static str]) -> ProgramId {
    let mut prefix = format!("#define WR_MAX_UBO_VECTORS {}\n\
                              #define WR_MAX_VERTEX_TEXTURE_WIDTH {}\n",
                              max_ubo_vectors,
                              MAX_VERTEX_TEXTURE_WIDTH);

    for feature in features {
        prefix.push_str(&format!("#define WR_FEATURE_{}\n", feature));
    }

    let includes_base = ["prim_shared"];
    let includes_clip = ["prim_shared", "clip_shared"];
    let includes: &[&str] = if name.ends_with("_clip") {
        &includes_clip
    } else {
        &includes_base
    };
    let program_id = device.create_program_with_prefix(name,
                                                       includes,
                                                       Some(prefix));
    let data_index = device.assign_ubo_binding(program_id, "Data", UBO_BIND_DATA);

    debug!("PrimShader {}: data={} max={}", name, data_index, max_ubo_vectors);

    program_id
}

fn create_clear_shader(name: &'static str,
                       device: &mut Device,
                       max_ubo_vectors: usize) -> ProgramId {
    let prefix = format!("#define WR_MAX_UBO_VECTORS {}", max_ubo_vectors);

    let includes = &["shared_other"];
    let program_id = device.create_program_with_prefix(name,
                                                       includes,
                                                       Some(prefix));

    let data_index = device.assign_ubo_binding(program_id, "Data", UBO_BIND_DATA);

    debug!("ClearShader {}: data={} max={}", name, data_index, max_ubo_vectors);

    program_id
}

/// The renderer is responsible for submitting to the GPU the work prepared by the
/// RenderBackend.
pub struct Renderer {
    result_rx: Receiver<ResultMsg>,
    device: Device,
    pending_texture_updates: Vec<TextureUpdateList>,
    pending_shader_updates: Vec<PathBuf>,
    current_frame: Option<RendererFrame>,
    device_pixel_ratio: f32,
    raster_batches: Vec<RasterBatch>,
    raster_op_vao: Option<VAOId>,

    blur_program_id: ProgramId,
    u_direction: UniformLocation,

    cs_box_shadow: LazilyCompiledShader,

    ps_rectangle: PrimitiveShader,
    ps_text_run: PrimitiveShader,
    ps_image: PrimitiveShader,
    ps_border: PrimitiveShader,
    ps_gradient: PrimitiveShader,
    ps_gradient_clip: PrimitiveShader,
    ps_angle_gradient: PrimitiveShader,
    ps_box_shadow: PrimitiveShader,
    ps_rectangle_clip: PrimitiveShader,
    ps_image_clip: PrimitiveShader,

    ps_blend: LazilyCompiledShader,
    ps_composite: LazilyCompiledShader,

    tile_clear_shader: LazilyCompiledShader,

    max_clear_tiles: usize,
    max_prim_blends: usize,
    max_prim_composites: usize,
    max_cache_instances: usize,

    notifier: Arc<Mutex<Option<Box<RenderNotifier>>>>,

    enable_profiler: bool,
    debug: DebugRenderer,
    backend_profile_counters: BackendProfileCounters,
    profile_counters: RendererProfileCounters,
    profiler: Profiler,
    last_time: u64,

    max_raster_op_size: u32,
    raster_op_target_a8: TextureId,
    raster_op_target_rgba8: TextureId,
    render_targets: Vec<TextureId>,

    gpu_profile: GpuProfiler<GpuProfileTag>,
    quad_vao_id: VAOId,

    layer_texture: VertexDataTexture,
    render_task_texture: VertexDataTexture,
    prim_geom_texture: VertexDataTexture,
    data16_texture: VertexDataTexture,
    data32_texture: VertexDataTexture,
    data64_texture: VertexDataTexture,
    data128_texture: VertexDataTexture,
    pipeline_epoch_map: HashMap<PipelineId, Epoch, BuildHasherDefault<FnvHasher>>,
    /// Used to dispatch functions to the main thread's event loop.
    /// Required to allow GLContext sharing in some implementations like WGL.
    main_thread_dispatcher: Arc<Mutex<Option<Box<RenderDispatcher>>>>,
}

impl Renderer {
    /// Initializes webrender and creates a Renderer and RenderApiSender.
    ///
    /// # Examples
    /// Initializes a Renderer with some reasonable values. For more information see
    /// [RendererOptions][rendereroptions].
    /// [rendereroptions]: struct.RendererOptions.html
    ///
    /// ```rust,ignore
    /// # use webrender::renderer::Renderer;
    /// # use std::path::PathBuf;
    /// let opts = webrender::RendererOptions {
    ///    device_pixel_ratio: 1.0,
    ///    resource_path: PathBuf::from("../webrender/res"),
    ///    enable_aa: false,
    ///    enable_msaa: false,
    ///    enable_profiler: false,
    /// };
    /// let (renderer, sender) = Renderer::new(opts);
    /// ```
    pub fn new(options: RendererOptions) -> (Renderer, RenderApiSender) {
        let (api_tx, api_rx) = ipc::channel().unwrap();
        let (payload_tx, payload_rx) = ipc::bytes_channel().unwrap();
        let (result_tx, result_rx) = channel();

        let notifier = Arc::new(Mutex::new(None));

        let file_watch_handler = FileWatcher {
            result_tx: result_tx.clone(),
            notifier: notifier.clone(),
        };

        let mut device = Device::new(options.resource_path.clone(),
                                     options.device_pixel_ratio,
                                     Box::new(file_watch_handler));
        device.begin_frame();

        let blur_program_id = device.create_program("blur", "shared_other");
        let max_raster_op_size = MAX_RASTER_OP_SIZE * options.device_pixel_ratio as u32;

        let max_ubo_size = device.get_capabilities().max_ubo_size;
        let max_ubo_vectors = max_ubo_size / 16;

        let max_prim_instances = get_ubo_max_len::<tiling::PrimitiveInstance>(max_ubo_size);
        let max_cache_instances = get_ubo_max_len::<tiling::CachePrimitiveInstance>(max_ubo_size);
        let max_prim_blends = get_ubo_max_len::<tiling::PackedBlendPrimitive>(max_ubo_size);
        let max_prim_composites = get_ubo_max_len::<tiling::PackedCompositePrimitive>(max_ubo_size);

        let cs_box_shadow = LazilyCompiledShader::new(ShaderKind::Cache,
                                                      "cs_box_shadow",
                                                      max_cache_instances,
                                                      &[],
                                                      &mut device,
                                                      options.precache_shaders);

        let ps_rectangle = PrimitiveShader::new("ps_rectangle",
                                                max_ubo_vectors,
                                                max_prim_instances,
                                                &mut device,
                                                options.precache_shaders);
        let ps_text_run = PrimitiveShader::new("ps_text_run",
                                               max_ubo_vectors,
                                               max_prim_instances,
                                               &mut device,
                                               options.precache_shaders);
        let ps_image = PrimitiveShader::new("ps_image",
                                            max_ubo_vectors,
                                            max_prim_instances,
                                            &mut device,
                                            options.precache_shaders);
        let ps_border = PrimitiveShader::new("ps_border",
                                             max_ubo_vectors,
                                             max_prim_instances,
                                             &mut device,
                                             options.precache_shaders);
        let ps_rectangle_clip = PrimitiveShader::new("ps_rectangle_clip",
                                                     max_ubo_vectors,
                                                     max_prim_instances,
                                                     &mut device,
                                                     options.precache_shaders);
        let ps_image_clip = PrimitiveShader::new("ps_image_clip",
                                                 max_ubo_vectors,
                                                 max_prim_instances,
                                                 &mut device,
                                                 options.precache_shaders);

        let ps_box_shadow = PrimitiveShader::new("ps_box_shadow",
                                                 max_ubo_vectors,
                                                 max_prim_instances,
                                                 &mut device,
                                                 options.precache_shaders);

        let ps_gradient = PrimitiveShader::new("ps_gradient",
                                               max_ubo_vectors,
                                               max_prim_instances,
                                               &mut device,
                                               options.precache_shaders);
        let ps_gradient_clip = PrimitiveShader::new("ps_gradient_clip",
                                                    max_ubo_vectors,
                                                    max_prim_instances,
                                                    &mut device,
                                                    options.precache_shaders);
        let ps_angle_gradient = PrimitiveShader::new("ps_angle_gradient",
                                                     max_ubo_vectors,
                                                     max_prim_instances,
                                                     &mut device,
                                                     options.precache_shaders);

        let ps_blend = LazilyCompiledShader::new(ShaderKind::Primitive,
                                                 "ps_blend",
                                                 max_ubo_vectors,
                                                 &[],
                                                 &mut device,
                                                 options.precache_shaders);
        let ps_composite = LazilyCompiledShader::new(ShaderKind::Primitive,
                                                     "ps_composite",
                                                     max_ubo_vectors,
                                                     &[],
                                                     &mut device,
                                                     options.precache_shaders);

        let max_clear_tiles = get_ubo_max_len::<ClearTile>(max_ubo_size);
        let tile_clear_shader = LazilyCompiledShader::new(ShaderKind::Clear,
                                                          "ps_clear",
                                                           max_ubo_vectors,
                                                           &[],
                                                           &mut device,
                                                           options.precache_shaders);

        let texture_ids = device.create_texture_ids(1024, TextureTarget::Default);
        let mut texture_cache = TextureCache::new(texture_ids);

        let white_pixels: Vec<u8> = vec![
            0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff,
        ];
        let mask_pixels: Vec<u8> = vec![
            0xff, 0xff,
            0xff, 0xff,
        ];
        // TODO: Ensure that the white texture can never get evicted when the cache supports LRU eviction!
        let white_image_id = texture_cache.new_item_id();
        texture_cache.insert(white_image_id,
                             0,
                             0,
                             2,
                             2,
                             ImageFormat::RGBA8,
                             TextureFilter::Linear,
                             TextureInsertOp::Blit(white_pixels),
                             BorderType::SinglePixel);

        let dummy_mask_image_id = texture_cache.new_item_id();
        texture_cache.insert(dummy_mask_image_id,
                             0,
                             0,
                             2,
                             2,
                             ImageFormat::A8,
                             TextureFilter::Linear,
                             TextureInsertOp::Blit(mask_pixels),
                             BorderType::SinglePixel);

        let dummy_resources = DummyResources {
            white_image_id: white_image_id,
            opaque_mask_image_id: dummy_mask_image_id,
        };

        let debug_renderer = DebugRenderer::new(&mut device);

        let raster_op_target_a8 = device.create_texture_ids(1, TextureTarget::Default)[0];
        device.init_texture(raster_op_target_a8,
                            max_raster_op_size,
                            max_raster_op_size,
                            ImageFormat::A8,
                            TextureFilter::Nearest,
                            RenderTargetMode::SimpleRenderTarget,
                            None);

        let raster_op_target_rgba8 = device.create_texture_ids(1, TextureTarget::Default)[0];
        device.init_texture(raster_op_target_rgba8,
                            max_raster_op_size,
                            max_raster_op_size,
                            ImageFormat::RGBA8,
                            TextureFilter::Nearest,
                            RenderTargetMode::SimpleRenderTarget,
                            None);

        let layer_texture = VertexDataTexture::new(&mut device);
        let render_task_texture = VertexDataTexture::new(&mut device);
        let prim_geom_texture = VertexDataTexture::new(&mut device);

        let data16_texture = VertexDataTexture::new(&mut device);
        let data32_texture = VertexDataTexture::new(&mut device);
        let data64_texture = VertexDataTexture::new(&mut device);
        let data128_texture = VertexDataTexture::new(&mut device);

        let x0 = 0.0;
        let y0 = 0.0;
        let x1 = 1.0;
        let y1 = 1.0;

        // TODO(gw): Consider separate VBO for quads vs border corners if VS ever shows up in profile!
        let quad_indices: [u16; 6] = [ 0, 1, 2, 2, 1, 3 ];
        let quad_vertices = [
            PackedVertex {
                pos: [x0, y0],
            },
            PackedVertex {
                pos: [x1, y0],
            },
            PackedVertex {
                pos: [x0, y1],
            },
            PackedVertex {
                pos: [x1, y1],
            },
        ];

        let quad_vao_id = device.create_vao(VertexFormat::Triangles, None);
        device.bind_vao(quad_vao_id);
        device.update_vao_indices(quad_vao_id, &quad_indices, VertexUsageHint::Static);
        device.update_vao_main_vertices(quad_vao_id, &quad_vertices, VertexUsageHint::Static);

        device.end_frame();

        let main_thread_dispatcher = Arc::new(Mutex::new(None));
        let backend_notifier = notifier.clone();
        let backend_main_thread_dispatcher = main_thread_dispatcher.clone();

        // We need a reference to the webrender context from the render backend in order to share
        // texture ids
        let context_handle = match options.renderer_kind {
            RendererKind::Native => GLContextHandleWrapper::current_native_handle(),
            RendererKind::OSMesa => GLContextHandleWrapper::current_osmesa_handle(),
        };

        let config = FrameBuilderConfig::new(options.enable_scrollbars);

        let debug = options.debug;
        let (device_pixel_ratio, enable_aa) = (options.device_pixel_ratio, options.enable_aa);
        let payload_tx_for_backend = payload_tx.clone();
        let enable_recording = options.enable_recording;
        thread::spawn(move || {
            let mut backend = RenderBackend::new(api_rx,
                                                 payload_rx,
                                                 payload_tx_for_backend,
                                                 result_tx,
                                                 device_pixel_ratio,
                                                 texture_cache,
                                                 dummy_resources,
                                                 enable_aa,
                                                 backend_notifier,
                                                 context_handle,
                                                 config,
                                                 debug,
                                                 enable_recording,
                                                 backend_main_thread_dispatcher);
            backend.run();
        });

        let mut renderer = Renderer {
            result_rx: result_rx,
            device: device,
            current_frame: None,
            raster_batches: Vec::new(),
            raster_op_vao: None,
            pending_texture_updates: Vec::new(),
            pending_shader_updates: Vec::new(),
            device_pixel_ratio: options.device_pixel_ratio,
            blur_program_id: blur_program_id,
            tile_clear_shader: tile_clear_shader,
            cs_box_shadow: cs_box_shadow,
            ps_rectangle: ps_rectangle,
            ps_text_run: ps_text_run,
            ps_image: ps_image,
            ps_border: ps_border,
            ps_rectangle_clip: ps_rectangle_clip,
            ps_image_clip: ps_image_clip,
            ps_box_shadow: ps_box_shadow,
            ps_gradient: ps_gradient,
            ps_gradient_clip: ps_gradient_clip,
            ps_angle_gradient: ps_angle_gradient,
            ps_blend: ps_blend,
            ps_composite: ps_composite,
            max_clear_tiles: max_clear_tiles,
            max_prim_blends: max_prim_blends,
            max_prim_composites: max_prim_composites,
            max_cache_instances: max_cache_instances,
            u_direction: UniformLocation::invalid(),
            notifier: notifier,
            debug: debug_renderer,
            backend_profile_counters: BackendProfileCounters::new(),
            profile_counters: RendererProfileCounters::new(),
            profiler: Profiler::new(),
            enable_profiler: options.enable_profiler,
            last_time: 0,
            raster_op_target_a8: raster_op_target_a8,
            raster_op_target_rgba8: raster_op_target_rgba8,
            render_targets: Vec::new(),
            max_raster_op_size: max_raster_op_size,
            gpu_profile: GpuProfiler::new(),
            quad_vao_id: quad_vao_id,
            layer_texture: layer_texture,
            render_task_texture: render_task_texture,
            prim_geom_texture: prim_geom_texture,
            data16_texture: data16_texture,
            data32_texture: data32_texture,
            data64_texture: data64_texture,
            data128_texture: data128_texture,
            pipeline_epoch_map: HashMap::with_hasher(Default::default()),
            main_thread_dispatcher: main_thread_dispatcher,
        };

        renderer.update_uniform_locations();

        let sender = RenderApiSender::new(api_tx, payload_tx);
        (renderer, sender)
    }

    fn update_uniform_locations(&mut self) {
        self.u_direction = self.device.get_uniform_location(self.blur_program_id, "uDirection");
    }

    /// Sets the new RenderNotifier.
    ///
    /// The RenderNotifier will be called when processing e.g. of a (scrolling) frame is done,
    /// and therefore the screen should be updated.
    pub fn set_render_notifier(&self, notifier: Box<RenderNotifier>) {
        let mut notifier_arc = self.notifier.lock().unwrap();
        *notifier_arc = Some(notifier);
    }

    /// Sets the new MainThreadDispatcher.
    ///
    /// Allows to dispatch functions to the main thread's event loop.
    pub fn set_main_thread_dispatcher(&self, dispatcher: Box<RenderDispatcher>) {
        let mut dispatcher_arc = self.main_thread_dispatcher.lock().unwrap();
        *dispatcher_arc = Some(dispatcher);
    }

    /// Returns the Epoch of the current frame in a pipeline.
    pub fn current_epoch(&self, pipeline_id: PipelineId) -> Option<Epoch> {
        self.pipeline_epoch_map.get(&pipeline_id).map(|epoch| *epoch)
    }

    /// Processes the result queue.
    ///
    /// Should be called before `render()`, as texture cache updates are done here.
    pub fn update(&mut self) {
        // Pull any pending results and return the most recent.
        while let Ok(msg) = self.result_rx.try_recv() {
            match msg {
                ResultMsg::UpdateTextureCache(update_list) => {
                    self.pending_texture_updates.push(update_list);
                }
                ResultMsg::NewFrame(frame, profile_counters) => {
                    self.backend_profile_counters = profile_counters;

                    // Update the list of available epochs for use during reftests.
                    // This is a workaround for https://github.com/servo/servo/issues/13149.
                    for (pipeline_id, epoch) in &frame.pipeline_epoch_map {
                        self.pipeline_epoch_map.insert(*pipeline_id, *epoch);
                    }

                    self.current_frame = Some(frame);
                }
                ResultMsg::RefreshShader(path) => {
                    self.pending_shader_updates.push(path);
                }
            }
        }
    }

    /// Renders the current frame.
    ///
    /// A Frame is supplied by calling [set_root_stacking_context()][newframe].
    /// [newframe]: ../../webrender_traits/struct.RenderApi.html#method.set_root_stacking_context
    pub fn render(&mut self, framebuffer_size: Size2D<u32>) {
        if let Some(mut frame) = self.current_frame.take() {
            if let Some(ref mut frame) = frame.frame {
                let mut profile_timers = RendererProfileTimers::new();

                // Block CPU waiting for last frame's GPU profiles to arrive.
                // In general this shouldn't block unless heavily GPU limited.
                if let Some(samples) = self.gpu_profile.build_samples() {
                    profile_timers.gpu_samples = samples;
                }

                profile_timers.cpu_time.profile(|| {
                    self.device.begin_frame();
                    self.gpu_profile.begin_frame();
                    self.gpu_profile.add_marker(GPU_TAG_INIT);

                    self.device.disable_scissor();
                    self.device.disable_depth();
                    self.device.set_blend(false);

                    //self.update_shaders();
                    self.update_texture_cache();
                    self.draw_tile_frame(frame, &framebuffer_size);

                    self.device.reset_ubo(UBO_BIND_DATA);
                    self.gpu_profile.end_frame();
                });

                let current_time = precise_time_ns();
                let ns = current_time - self.last_time;
                self.profile_counters.frame_time.set(ns);

                if self.enable_profiler {
                    self.profiler.draw_profile(&frame.profile_counters,
                                               &self.backend_profile_counters,
                                               &self.profile_counters,
                                               &mut profile_timers,
                                               &mut self.debug);
                }

                self.profile_counters.reset();
                self.profile_counters.frame_counter.inc();

                let debug_size = Size2D::new(framebuffer_size.width as u32,
                                             framebuffer_size.height as u32);
                self.debug.render(&mut self.device, &debug_size);
                self.device.end_frame();
                self.last_time = current_time;
            }

            // Restore frame - avoid borrow checker!
            self.current_frame = Some(frame);
        }
    }

    pub fn layers_are_bouncing_back(&self) -> bool {
        match self.current_frame {
            None => false,
            Some(ref current_frame) => !current_frame.layers_bouncing_back.is_empty(),
        }
    }

/*
    fn update_shaders(&mut self) {
        let update_uniforms = !self.pending_shader_updates.is_empty();

        for path in self.pending_shader_updates.drain(..) {
            panic!("todo");
            //self.device.refresh_shader(path);
        }

        if update_uniforms {
            self.update_uniform_locations();
        }
    }
*/

    fn update_texture_cache(&mut self) {
        let mut pending_texture_updates = mem::replace(&mut self.pending_texture_updates, vec![]);
        for update_list in pending_texture_updates.drain(..) {
            for update in update_list.updates {
                match update.op {
                    TextureUpdateOp::Create(width, height, format, filter, mode, maybe_bytes) => {
                        // TODO: clean up match
                        match maybe_bytes {
                            Some(bytes) => {
                                self.device.init_texture(update.id,
                                                         width,
                                                         height,
                                                         format,
                                                         filter,
                                                         mode,
                                                         Some(bytes.as_slice()));
                            }
                            None => {
                                self.device.init_texture(update.id,
                                                         width,
                                                         height,
                                                         format,
                                                         filter,
                                                         mode,
                                                         None);
                            }
                        }
                    }
                    TextureUpdateOp::Grow(new_width,
                                          new_height,
                                          format,
                                          filter,
                                          mode) => {
                        self.device.resize_texture(update.id,
                                                   new_width,
                                                   new_height,
                                                   format,
                                                   filter,
                                                   mode);
                    }
                    TextureUpdateOp::Update(x, y, width, height, details) => {
                        match details {
                            TextureUpdateDetails::Raw => {
                                self.device.update_raw_texture(update.id, x, y, width, height);
                            }
                            TextureUpdateDetails::Blit(bytes) => {
                                self.device.update_texture(
                                    update.id,
                                    x,
                                    y,
                                    width, height,
                                    bytes.as_slice());
                            }
                            TextureUpdateDetails::Blur(bytes,
                                                       glyph_size,
                                                       radius,
                                                       unblurred_glyph_texture_image,
                                                       horizontal_blur_texture_image,
                                                       border_type) => {
                                let radius =
                                    f32::ceil(radius.to_f32_px() * self.device_pixel_ratio) as u32;
                                self.device.update_texture(
                                    unblurred_glyph_texture_image.texture_id,
                                    unblurred_glyph_texture_image.pixel_uv.x,
                                    unblurred_glyph_texture_image.pixel_uv.y,
                                    glyph_size.width,
                                    glyph_size.height,
                                    bytes.as_slice());

                                let blur_program_id = self.blur_program_id;

                                let white = ColorF::new(1.0, 1.0, 1.0, 1.0);
                                let (width, height) = (width as f32, height as f32);

                                let zero_point = Point2D::new(0.0, 0.0);
                                let dest_texture_size = Size2D::new(width as f32, height as f32);
                                let source_texture_size = Size2D::new(glyph_size.width as f32,
                                                                      glyph_size.height as f32);
                                let blur_radius = radius as f32;

                                self.add_rect_to_raster_batch(horizontal_blur_texture_image.texture_id,
                                                              unblurred_glyph_texture_image.texture_id,
                                                              blur_program_id,
                                                              Some(AxisDirection::Horizontal),
                                                              &Rect::new(horizontal_blur_texture_image.pixel_uv,
                                                                         Size2D::new(width as u32, height as u32)),
                                                              border_type,
                                                              |texture_rect| {
                                    [
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.origin,
                                            &white,
                                            &Point2D::new(0.0, 0.0),
                                            &zero_point,
                                            &zero_point,
                                            &unblurred_glyph_texture_image.texel_uv.origin,
                                            &unblurred_glyph_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.top_right(),
                                            &white,
                                            &Point2D::new(1.0, 0.0),
                                            &zero_point,
                                            &zero_point,
                                            &unblurred_glyph_texture_image.texel_uv.origin,
                                            &unblurred_glyph_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.bottom_left(),
                                            &white,
                                            &Point2D::new(0.0, 1.0),
                                            &zero_point,
                                            &zero_point,
                                            &unblurred_glyph_texture_image.texel_uv.origin,
                                            &unblurred_glyph_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.bottom_right(),
                                            &white,
                                            &Point2D::new(1.0, 1.0),
                                            &zero_point,
                                            &zero_point,
                                            &unblurred_glyph_texture_image.texel_uv.origin,
                                            &unblurred_glyph_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                    ]
                                });

                                let source_texture_size = Size2D::new(width as f32, height as f32);

                                self.add_rect_to_raster_batch(update.id,
                                                              horizontal_blur_texture_image.texture_id,
                                                              blur_program_id,
                                                              Some(AxisDirection::Vertical),
                                                              &Rect::new(Point2D::new(x as u32, y as u32),
                                                                         Size2D::new(width as u32, height as u32)),
                                                              border_type,
                                                              |texture_rect| {
                                    [
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.origin,
                                            &white,
                                            &Point2D::new(0.0, 0.0),
                                            &zero_point,
                                            &zero_point,
                                            &horizontal_blur_texture_image.texel_uv.origin,
                                            &horizontal_blur_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.top_right(),
                                            &white,
                                            &Point2D::new(1.0, 0.0),
                                            &zero_point,
                                            &zero_point,
                                            &horizontal_blur_texture_image.texel_uv.origin,
                                            &horizontal_blur_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.bottom_left(),
                                            &white,
                                            &Point2D::new(0.0, 1.0),
                                            &zero_point,
                                            &zero_point,
                                            &horizontal_blur_texture_image.texel_uv.origin,
                                            &horizontal_blur_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.bottom_right(),
                                            &white,
                                            &Point2D::new(1.0, 1.0),
                                            &zero_point,
                                            &zero_point,
                                            &horizontal_blur_texture_image.texel_uv.origin,
                                            &horizontal_blur_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                    ]
                                });
                            }
                        }
                    }
                }
            }
        }

        self.flush_raster_batches();
    }

    fn add_rect_to_raster_batch<F>(&mut self,
                                   dest_texture_id: TextureId,
                                   color_texture_id: TextureId,
                                   program_id: ProgramId,
                                   blur_direction: Option<AxisDirection>,
                                   dest_rect: &Rect<u32>,
                                   border_type: BorderType,
                                   f: F)
                                   where F: Fn(&Rect<f32>) -> [PackedVertexForTextureCacheUpdate; 4] {
        // FIXME(pcwalton): Use a hash table if this linear search shows up in the profile.
        for batch in &mut self.raster_batches {
            if batch.add_rect_if_possible(dest_texture_id,
                                          color_texture_id,
                                          program_id,
                                          blur_direction,
                                          dest_rect,
                                          border_type,
                                          &f) {
                return;
            }
        }

        let raster_op_target = if self.device.texture_has_alpha(dest_texture_id) {
            self.raster_op_target_rgba8
        } else {
            self.raster_op_target_a8
        };

        let mut raster_batch = RasterBatch::new(raster_op_target,
                                                self.max_raster_op_size,
                                                program_id,
                                                blur_direction,
                                                color_texture_id,
                                                dest_texture_id);

        let added = raster_batch.add_rect_if_possible(dest_texture_id,
                                                      color_texture_id,
                                                      program_id,
                                                      blur_direction,
                                                      dest_rect,
                                                      border_type,
                                                      &f);
        debug_assert!(added);
        self.raster_batches.push(raster_batch);
    }

    fn flush_raster_batches(&mut self) {
        let batches = mem::replace(&mut self.raster_batches, vec![]);
        if !batches.is_empty() {
            //println!("flushing {:?} raster batches", batches.len());

            self.device.disable_depth();
            self.device.disable_scissor();
            // Disable MSAA here for raster ops
            self.device.set_multisample(false);

            let projection = Matrix4D::ortho(0.0,
                                             self.max_raster_op_size as f32,
                                             0.0,
                                             self.max_raster_op_size as f32,
                                             ORTHO_NEAR_PLANE,
                                             ORTHO_FAR_PLANE);

            // All horizontal blurs must complete before anything else.
            let mut remaining_batches = vec![];
            for batch in batches.into_iter() {
                if batch.blur_direction != Some(AxisDirection::Horizontal) {
                    remaining_batches.push(batch);
                    continue
                }

                self.set_up_gl_state_for_texture_cache_update(batch.page_allocator.texture_id(),
                                                              batch.color_texture_id,
                                                              batch.program_id,
                                                              batch.blur_direction,
                                                              &projection);
                self.perform_gl_texture_cache_update(batch);
            }

            // Flush the remaining batches.
            for batch in remaining_batches.into_iter() {
                self.set_up_gl_state_for_texture_cache_update(batch.page_allocator.texture_id(),
                                                              batch.color_texture_id,
                                                              batch.program_id,
                                                              batch.blur_direction,
                                                              &projection);
                self.perform_gl_texture_cache_update(batch);
            }
        }
    }

    fn set_up_gl_state_for_texture_cache_update(&mut self,
                                                target_texture_id: TextureId,
                                                color_texture_id: TextureId,
                                                program_id: ProgramId,
                                                blur_direction: Option<AxisDirection>,
                                                projection: &Matrix4D<f32>) {

        self.device.set_blend(!self.device.texture_has_alpha(target_texture_id));
        self.device.set_blend_mode_premultiplied_alpha();

        let dimensions = [self.max_raster_op_size, self.max_raster_op_size];
        self.device.bind_render_target(Some((target_texture_id, 0)), Some(dimensions));

        self.device.bind_program(program_id, &projection);

        self.device.bind_texture(TextureSampler::Color, color_texture_id);
        self.device.bind_texture(TextureSampler::Mask, TextureId::invalid());

        match blur_direction {
            Some(AxisDirection::Horizontal) => {
                self.device.set_uniform_2f(self.u_direction, 1.0, 0.0)
            }
            Some(AxisDirection::Vertical) => {
                self.device.set_uniform_2f(self.u_direction, 0.0, 1.0)
            }
            None => {}
        }
    }

    fn perform_gl_texture_cache_update(&mut self, batch: RasterBatch) {
        let vao_id = match self.raster_op_vao {
            Some(ref mut vao_id) => *vao_id,
            None => {
                let vao_id = self.device.create_vao(VertexFormat::RasterOp, None);
                self.raster_op_vao = Some(vao_id);
                vao_id
            }
        };
        self.device.bind_vao(vao_id);

        self.device.update_vao_indices(vao_id, &batch.indices[..], VertexUsageHint::Dynamic);
        self.device.update_vao_main_vertices(vao_id,
                                             &batch.vertices[..],
                                             VertexUsageHint::Dynamic);

        self.profile_counters.vertices.add(batch.indices.len());
        self.profile_counters.draw_calls.inc();

        //println!("drawing triangles due to GL texture cache update");
        self.device.draw_triangles_u16(0, batch.indices.len() as i32);

        for blit_job in batch.blit_jobs {
            self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                              blit_job.dest_origin.x as i32,
                                              blit_job.dest_origin.y as i32,
                                              blit_job.src_origin.x as i32,
                                              blit_job.src_origin.y as i32,
                                              blit_job.size.width as i32,
                                              blit_job.size.height as i32);

            match blit_job.border_type {
                BorderType::SinglePixel => {
                    // Single pixel corners
                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      blit_job.dest_origin.x as i32 - 1,
                                                      blit_job.dest_origin.y as i32 - 1,
                                                      blit_job.src_origin.x as i32,
                                                      blit_job.src_origin.y as i32,
                                                      1,
                                                      1);

                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      (blit_job.dest_origin.x + blit_job.size.width) as i32,
                                                      blit_job.dest_origin.y as i32 - 1,
                                                      (blit_job.src_origin.x + blit_job.size.width) as i32 - 1,
                                                      blit_job.src_origin.y as i32,
                                                      1,
                                                      1);

                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      blit_job.dest_origin.x as i32 - 1,
                                                      (blit_job.dest_origin.y + blit_job.size.height) as i32,
                                                      blit_job.src_origin.x as i32,
                                                      (blit_job.src_origin.y + blit_job.size.height) as i32 - 1,
                                                      1,
                                                      1);

                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      (blit_job.dest_origin.x + blit_job.size.width) as i32,
                                                      (blit_job.dest_origin.y + blit_job.size.height) as i32,
                                                      (blit_job.src_origin.x + blit_job.size.width) as i32 - 1,
                                                      (blit_job.src_origin.y + blit_job.size.height) as i32 - 1,
                                                      1,
                                                      1);

                    // Horizontal edges
                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      blit_job.dest_origin.x as i32,
                                                      blit_job.dest_origin.y as i32 - 1,
                                                      blit_job.src_origin.x as i32,
                                                      blit_job.src_origin.y as i32,
                                                      blit_job.size.width as i32,
                                                      1);

                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      blit_job.dest_origin.x as i32,
                                                      (blit_job.dest_origin.y + blit_job.size.height) as i32,
                                                      blit_job.src_origin.x as i32,
                                                      (blit_job.src_origin.y + blit_job.size.height) as i32 - 1,
                                                      blit_job.size.width as i32,
                                                      1);

                    // Vertical edges
                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      blit_job.dest_origin.x as i32 - 1,
                                                      blit_job.dest_origin.y as i32,
                                                      blit_job.src_origin.x as i32,
                                                      blit_job.src_origin.y as i32,
                                                      1,
                                                      blit_job.size.height as i32);

                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      (blit_job.dest_origin.x + blit_job.size.width) as i32,
                                                      blit_job.dest_origin.y as i32,
                                                      (blit_job.src_origin.x + blit_job.size.width) as i32 - 1,
                                                      blit_job.src_origin.y as i32,
                                                      1,
                                                      blit_job.size.height as i32);

                }
                BorderType::_NoBorder => {}
            }
        }
    }

    fn add_debug_rect(&mut self,
                      p0: DevicePoint,
                      p1: DevicePoint,
                      label: &str,
                      c: &ColorF) {
        let tile_x0 = p0.x;
        let tile_y0 = p0.y;
        let tile_x1 = p1.x;
        let tile_y1 = p1.y;

        self.debug.add_line(tile_x0,
                            tile_y0,
                            c,
                            tile_x1,
                            tile_y0,
                            c);
        self.debug.add_line(tile_x0,
                            tile_y1,
                            c,
                            tile_x1,
                            tile_y1,
                            c);
        self.debug.add_line(tile_x0,
                            tile_y0,
                            c,
                            tile_x0,
                            tile_y1,
                            c);
        self.debug.add_line(tile_x1,
                            tile_y0,
                            c,
                            tile_x1,
                            tile_y1,
                            c);
        if label.len() > 0 {
            self.debug.add_text((tile_x0 as f32 + tile_x1 as f32) * 0.5,
                                (tile_y0 as f32 + tile_y1 as f32) * 0.5,
                                label,
                                c);
        }
    }

    fn draw_ubo_batch<T>(&mut self,
                         ubo_data: &[T],
                         shader: ProgramId,
                         quads_per_item: usize,
                         color_texture_id: TextureId,
                         mask_texture_id: TextureId,
                         max_prim_items: usize,
                         projection: &Matrix4D<f32>) {
        self.device.bind_program(shader, &projection);
        self.device.bind_vao(self.quad_vao_id);
        self.device.bind_texture(TextureSampler::Color, color_texture_id);
        self.device.bind_texture(TextureSampler::Mask, mask_texture_id);

        for chunk in ubo_data.chunks(max_prim_items) {
            let ubo = self.device.create_ubo(&chunk, UBO_BIND_DATA);

            let quad_count = chunk.len() * quads_per_item;
            self.device.draw_indexed_triangles_instanced_u16(6, quad_count as i32);
            self.profile_counters.vertices.add(6 * (quad_count as usize));
            self.profile_counters.draw_calls.inc();

            self.device.delete_buffer(ubo);
        }
    }

    fn draw_target(&mut self,
                   render_target: Option<(TextureId, i32)>,
                   target: &RenderTarget,
                   target_size: &Size2D<f32>,
                   cache_texture: Option<TextureId>,
                   should_clear: bool) {
        self.gpu_profile.add_marker(GPU_TAG_SETUP_TARGET);

        let dimensions = [target_size.width as u32, target_size.height as u32];
        self.device.bind_render_target(render_target, Some(dimensions));

        self.device.set_blend(false);
        self.device.set_blend_mode_alpha();
        if let Some(cache_texture) = cache_texture {
	        self.device.bind_texture(TextureSampler::Cache, cache_texture);
	    }

        let (color, projection) = match render_target {
            Some(..) => (
                [0.0, 0.0, 0.0, 0.0],
                Matrix4D::ortho(0.0,
                               target_size.width as f32,
                               0.0,
                               target_size.height as f32,
                               ORTHO_NEAR_PLANE,
                               ORTHO_FAR_PLANE)
            ),
            None => (
                [1.0, 1.0, 1.0, 1.0],
                Matrix4D::ortho(0.0,
                               target_size.width as f32,
                               target_size.height as f32,
                               0.0,
                               ORTHO_NEAR_PLANE,
                               ORTHO_FAR_PLANE)
            ),
        };

        // todo(gw): remove me!
        if should_clear {
            self.device.clear_color(color);
        }

        // Draw any cache primitives for this target.
        if !target.box_shadow_cache_prims.is_empty() {
            self.device.set_blend(false);

            self.gpu_profile.add_marker(GPU_TAG_CACHE_BOX_SHADOW);
            let shader = self.cs_box_shadow.get(&mut self.device);
            let max_cache_instances = self.max_cache_instances;
            self.draw_ubo_batch(&target.box_shadow_cache_prims,
                                shader,
                                1,
                                TextureId::invalid(),
                                TextureId::invalid(),
                                max_cache_instances,
                                &projection);
        }

        let mut prev_blend_mode = BlendMode::None;

        for batch in &target.alpha_batcher.batches {
            let color_texture_id = batch.key.color_texture_id;
            let mask_texture_id = batch.key.mask_texture_id;
            let transform_kind = batch.key.flags.transform_kind();
            let has_complex_clip = batch.key.flags.needs_clipping();

            if batch.key.blend_mode != prev_blend_mode {
                match batch.key.blend_mode {
                    // TODO(gw): More blend modes to come with subpixel aa work.
                    BlendMode::None | BlendMode::Alpha => {
                        self.device.set_blend(batch.key.blend_mode == BlendMode::Alpha);
                    }
                }
                prev_blend_mode = batch.key.blend_mode;
            }

            match &batch.data {
                &PrimitiveBatchData::Blend(ref ubo_data) => {
                    self.gpu_profile.add_marker(GPU_TAG_PRIM_BLEND);
                    let shader = self.ps_blend.get(&mut self.device);
                    self.device.bind_program(shader, &projection);
                    self.device.bind_vao(self.quad_vao_id);

                    for chunk in ubo_data.chunks(self.max_prim_blends) {
                        let ubo = self.device.create_ubo(&chunk, UBO_BIND_DATA);

                        self.device.draw_indexed_triangles_instanced_u16(6, chunk.len() as i32);
                        self.profile_counters.vertices.add(6 * chunk.len());
                        self.profile_counters.draw_calls.inc();

                        self.device.delete_buffer(ubo);
                    }
                }
                &PrimitiveBatchData::Composite(ref ubo_data) => {
                    self.gpu_profile.add_marker(GPU_TAG_PRIM_COMPOSITE);
                    let shader = self.ps_composite.get(&mut self.device);
                    self.device.bind_program(shader, &projection);
                    self.device.bind_vao(self.quad_vao_id);

                    for chunk in ubo_data.chunks(self.max_prim_composites) {
                        let ubo = self.device.create_ubo(&chunk, UBO_BIND_DATA);

                        self.device.draw_indexed_triangles_instanced_u16(6, chunk.len() as i32);
                        self.profile_counters.vertices.add(6 * chunk.len());
                        self.profile_counters.draw_calls.inc();

                        self.device.delete_buffer(ubo);
                    }
                }
                &PrimitiveBatchData::Rectangles(ref ubo_data) => {
                    let (shader, max_prim_items) = if has_complex_clip {
                        self.gpu_profile.add_marker(GPU_TAG_PRIM_RECT_CLIP);
                        self.ps_rectangle_clip.get(&mut self.device, transform_kind)
                    } else {
                        self.gpu_profile.add_marker(GPU_TAG_PRIM_RECT);
                        self.ps_rectangle.get(&mut self.device, transform_kind)
                    };
                    self.draw_ubo_batch(ubo_data,
                                        shader,
                                        1,
                                        color_texture_id,
                                        mask_texture_id,
                                        max_prim_items,
                                        &projection);
                }
                &PrimitiveBatchData::Image(ref ubo_data) => {
                    let (shader, max_prim_items) = if has_complex_clip {
                        self.gpu_profile.add_marker(GPU_TAG_PRIM_IMAGE_CLIP);
                        self.ps_image_clip.get(&mut self.device, transform_kind)
                    } else {
                        self.gpu_profile.add_marker(GPU_TAG_PRIM_IMAGE);
                        self.ps_image.get(&mut self.device, transform_kind)
                    };
                    self.draw_ubo_batch(ubo_data,
                                        shader,
                                        1,
                                        color_texture_id,
                                        mask_texture_id,
                                        max_prim_items,
                                        &projection);
                }
                &PrimitiveBatchData::Borders(ref ubo_data) => {
                    self.gpu_profile.add_marker(GPU_TAG_PRIM_BORDER);
                    let (shader, max_prim_items) = self.ps_border.get(&mut self.device, transform_kind);
                    self.draw_ubo_batch(ubo_data,
                                        shader,
                                        1,
                                        color_texture_id,
                                        mask_texture_id,
                                        max_prim_items,
                                        &projection);
                }
                &PrimitiveBatchData::BoxShadow(ref ubo_data) => {
                    self.gpu_profile.add_marker(GPU_TAG_PRIM_BOX_SHADOW);
                    let (shader, max_prim_items) = self.ps_box_shadow.get(&mut self.device, transform_kind);
                    self.draw_ubo_batch(ubo_data,
                                        shader,
                                        1,
                                        color_texture_id,
                                        mask_texture_id,
                                        max_prim_items,
                                        &projection);
                }
                &PrimitiveBatchData::TextRun(ref ubo_data) => {
                    self.gpu_profile.add_marker(GPU_TAG_PRIM_TEXT_RUN);
                    let (shader, max_prim_items) = self.ps_text_run.get(&mut self.device, transform_kind);
                    self.draw_ubo_batch(ubo_data,
                                        shader,
                                        1,
                                        color_texture_id,
                                        mask_texture_id,
                                        max_prim_items,
                                        &projection);
                }
                &PrimitiveBatchData::AlignedGradient(ref ubo_data) => {
                    let (shader, max_prim_items) = if has_complex_clip {
                        self.gpu_profile.add_marker(GPU_TAG_PRIM_GRADIENT_CLIP);
                        self.ps_gradient_clip.get(&mut self.device, transform_kind)
                    } else {
                        self.gpu_profile.add_marker(GPU_TAG_PRIM_GRADIENT);
                        self.ps_gradient.get(&mut self.device, transform_kind)
                    };
                    self.draw_ubo_batch(ubo_data,
                                        shader,
                                        1,
                                        color_texture_id,
                                        mask_texture_id,
                                        max_prim_items,
                                        &projection);
                }
                &PrimitiveBatchData::AngleGradient(ref ubo_data) => {
                    self.gpu_profile.add_marker(GPU_TAG_PRIM_ANGLE_GRADIENT);
                    let (shader, max_prim_items) = self.ps_angle_gradient.get(&mut self.device, transform_kind);
                    self.draw_ubo_batch(ubo_data,
                                        shader,
                                        1,
                                        color_texture_id,
                                        mask_texture_id,
                                        max_prim_items,
                                        &projection);
                }
            }
        }

        self.device.set_blend(false);
    }

    fn draw_tile_frame(&mut self,
                       frame: &mut Frame,
                       framebuffer_size: &Size2D<u32>) {
        // Some tests use a restricted viewport smaller than the main screen size.
        // Ensure we clear the framebuffer in these tests.
        // TODO(gw): Find a better solution for this?
        let viewport_size = Size2D::new(frame.viewport_size.width * self.device_pixel_ratio as i32,
                                        frame.viewport_size.height * self.device_pixel_ratio as i32);
        let needs_clear = viewport_size.width < framebuffer_size.width as i32 ||
                          viewport_size.height < framebuffer_size.height as i32;

        for debug_rect in frame.debug_rects.iter().rev() {
            self.add_debug_rect(debug_rect.rect.origin,
                                debug_rect.rect.bottom_right(),
                                &debug_rect.label,
                                &debug_rect.color);
        }

        self.device.disable_depth_write();
        self.device.disable_stencil();
        self.device.set_blend(false);

        let projection = Matrix4D::ortho(0.0,
                                         framebuffer_size.width as f32,
                                         framebuffer_size.height as f32,
                                         0.0,
                                         ORTHO_NEAR_PLANE,
                                         ORTHO_FAR_PLANE);

        if frame.passes.is_empty() {
            self.device.clear_color([1.0, 1.0, 1.0, 1.0]);
        } else {
            // Add new render targets to the pool if required.
            let needed_targets = frame.passes.len() - 1;     // framebuffer doesn't need a target!
            let current_target_count = self.render_targets.len();
            if needed_targets > current_target_count {
                let new_target_count = needed_targets - current_target_count;
                let new_targets = self.device.create_texture_ids(new_target_count as i32,
                                                                 TextureTarget::Array);
                self.render_targets.extend_from_slice(&new_targets);
            }

            // Init textures and render targets to match this scene. This shouldn't
            // block in drivers, but it might be worth checking...
            for (pass, texture_id) in frame.passes.iter().zip(self.render_targets.iter()) {
                self.device.init_texture(*texture_id,
                                         frame.cache_size.width as u32,
                                         frame.cache_size.height as u32,
                                         ImageFormat::RGBA8,
                                         TextureFilter::Linear,
                                         RenderTargetMode::LayerRenderTarget(pass.targets.len() as i32),
                                         None);
            }

            self.layer_texture.init(&mut self.device, &mut frame.layer_texture_data);
            self.render_task_texture.init(&mut self.device, &mut frame.render_task_data);
            self.data16_texture.init(&mut self.device, &mut frame.gpu_data16);
            self.data32_texture.init(&mut self.device, &mut frame.gpu_data32);
            self.data64_texture.init(&mut self.device, &mut frame.gpu_data64);
            self.data128_texture.init(&mut self.device, &mut frame.gpu_data128);
            self.prim_geom_texture.init(&mut self.device, &mut frame.gpu_geometry);

            self.device.bind_texture(TextureSampler::Layers, self.layer_texture.id);
            self.device.bind_texture(TextureSampler::RenderTasks, self.render_task_texture.id);
            self.device.bind_texture(TextureSampler::Geometry, self.prim_geom_texture.id);
            self.device.bind_texture(TextureSampler::Data16, self.data16_texture.id);
            self.device.bind_texture(TextureSampler::Data32, self.data32_texture.id);
            self.device.bind_texture(TextureSampler::Data64, self.data64_texture.id);
            self.device.bind_texture(TextureSampler::Data128, self.data128_texture.id);

            let mut src_id = None;

            for (pass_index, pass) in frame.passes.iter().enumerate() {
                let (do_clear, size, target_id) = if pass.is_framebuffer {
                    (needs_clear,
                     Size2D::new(framebuffer_size.width as f32, framebuffer_size.height as f32),
                     None)
                } else {
                    (true, frame.cache_size, Some(self.render_targets[pass_index]))
                };

                for (target_index, target) in pass.targets.iter().enumerate() {
                    let render_target = target_id.map(|texture_id| {
                        (texture_id, target_index as i32)
                    });
                    self.draw_target(render_target,
                                     target,
                                     &size,
                                     src_id,
                                     do_clear);

                }

                src_id = target_id;
            }
        }

        self.gpu_profile.add_marker(GPU_TAG_CLEAR_TILES);

        // Clear tiles with no items
        if !frame.clear_tiles.is_empty() {
            let tile_clear_shader = self.tile_clear_shader.get(&mut self.device);
            self.device.bind_program(tile_clear_shader, &projection);
            self.device.bind_vao(self.quad_vao_id);

            for chunk in frame.clear_tiles.chunks(self.max_clear_tiles) {
                let ubo = self.device.create_ubo(&chunk, UBO_BIND_DATA);

                self.device.draw_indexed_triangles_instanced_u16(6, chunk.len() as i32);
                self.profile_counters.vertices.add(6 * chunk.len());
                self.profile_counters.draw_calls.inc();

                self.device.delete_buffer(ubo);
            }
        }
    }

    pub fn debug_renderer<'a>(&'a mut self) -> &'a mut DebugRenderer {
        &mut self.debug
    }

    pub fn set_profiler_enabled(&mut self, enabled: bool) {
        self.enable_profiler = enabled;
    }
}

#[derive(Clone, Debug)]
pub struct RendererOptions {
    pub device_pixel_ratio: f32,
    pub resource_path: PathBuf,
    pub enable_aa: bool,
    pub enable_msaa: bool,
    pub enable_profiler: bool,
    pub debug: bool,
    pub enable_recording: bool,
    pub enable_scrollbars: bool,
    pub precache_shaders: bool,
    pub renderer_kind: RendererKind,
}
