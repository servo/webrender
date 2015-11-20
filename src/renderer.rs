use app_units::Au;
use batch::{RasterBatch, VertexBufferId};
use debug_render::DebugRenderer;
use device::{Device, ProgramId, TextureId, UniformLocation, VertexFormat};
use device::{TextureFilter, VAOId, VertexUsageHint};
use euclid::{Rect, Matrix4, Point2D, Size2D};
use fnv::FnvHasher;
use gleam::gl;
use internal_types::{RendererFrame, ResultMsg, TextureUpdateOp, BatchUpdateOp, BatchUpdateList};
use internal_types::{TextureUpdateDetails, TextureUpdateList, PackedVertex, RenderTargetMode};
use internal_types::{ORTHO_NEAR_PLANE, ORTHO_FAR_PLANE, BoxShadowPart, BasicRotationAngle};
use internal_types::{PackedVertexForTextureCacheUpdate, CompositionOp};
use internal_types::{AxisDirection, LowLevelFilterOp, DrawCommand, ANGLE_FLOAT_TO_FIXED};
use ipc_channel::ipc;
use profiler::{Profiler, BackendProfileCounters};
use profiler::{RendererProfileTimers, RendererProfileCounters};
use render_backend::RenderBackend;
use std::collections::HashMap;
use std::collections::hash_state::DefaultState;
use std::f32;
use std::mem;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{channel, Receiver};
use std::thread;
use tessellator::BorderCornerTessellation;
use texture_cache::{BorderType, TextureCache, TextureInsertOp};
use time::precise_time_ns;
use webrender_traits::{ColorF, Epoch, PipelineId, RenderNotifier};
use webrender_traits::{ImageFormat, MixBlendMode, RenderApiSender};
use offscreen_gl_context::{NativeGLContext, NativeGLContextMethods};
//use util;

pub const BLUR_INFLATION_FACTOR: u32 = 3;

// TODO(gw): HACK! Need to support lighten/darken mix-blend-mode properly on android...

#[cfg(not(any(target_os = "android", target_os = "gonk")))]
const GL_BLEND_MIN: gl::GLuint = gl::MIN;

#[cfg(any(target_os = "android", target_os = "gonk"))]
const GL_BLEND_MIN: gl::GLuint = gl::FUNC_ADD;

#[cfg(not(any(target_os = "android", target_os = "gonk")))]
const GL_BLEND_MAX: gl::GLuint = gl::MAX;

#[cfg(any(target_os = "android", target_os = "gonk"))]
const GL_BLEND_MAX: gl::GLuint = gl::FUNC_ADD;

struct VertexBuffer {
    vao_id: VAOId,
}

struct RenderContext {
    blend_program_id: ProgramId,
    filter_program_id: ProgramId,
    temporary_fb_texture: TextureId,
    projection: Matrix4,
    layer_size: Size2D<u32>,
}

pub struct Renderer {
    result_rx: Receiver<ResultMsg>,
    device: Device,
    pending_texture_updates: Vec<TextureUpdateList>,
    pending_batch_updates: Vec<BatchUpdateList>,
    current_frame: Option<RendererFrame>,
    device_pixel_ratio: f32,
    vertex_buffers: HashMap<VertexBufferId, VertexBuffer, DefaultState<FnvHasher>>,
    raster_batches: Vec<RasterBatch>,

    quad_program_id: ProgramId,
    u_quad_transform_array: UniformLocation,
    u_tile_params: UniformLocation,
    u_atlas_params: UniformLocation,

    blit_program_id: ProgramId,

    border_program_id: ProgramId,

    blend_program_id: ProgramId,
    u_blend_params: UniformLocation,

    filter_program_id: ProgramId,
    u_filter_params: UniformLocation,

    box_shadow_program_id: ProgramId,

    blur_program_id: ProgramId,
    u_direction: UniformLocation,

    clear_program_id: ProgramId,

    notifier: Arc<Mutex<Option<Box<RenderNotifier>>>>,
    viewport_size: Size2D<u32>,
    framebuffer_size: Size2D<u32>,

    enable_profiler: bool,
    debug: DebugRenderer,
    backend_profile_counters: BackendProfileCounters,
    profile_counters: RendererProfileCounters,
    profiler: Profiler,
    last_time: u64,
}

impl Renderer {
    pub fn new(width: u32,
               height: u32,
               framebuffer_size: &Size2D<u32>,
               device_pixel_ratio: f32,
               resource_path: PathBuf,
               enable_aa: bool,
               enable_profiler: bool) -> (Renderer, RenderApiSender) {
        let (api_tx, api_rx) = ipc::channel().unwrap();
        let (result_tx, result_rx) = channel();

        let initial_viewport = Rect::new(Point2D::zero(), Size2D::new(width as i32, height as i32));

        let mut device = Device::new(resource_path, device_pixel_ratio);
        device.begin_frame();

        let quad_program_id = device.create_program("quad.vs.glsl", "quad.fs.glsl");
        let blit_program_id = device.create_program("blit.vs.glsl", "blit.fs.glsl");
        let border_program_id = device.create_program("border.vs.glsl", "border.fs.glsl");
        let blend_program_id = device.create_program("blend.vs.glsl", "blend.fs.glsl");
        let filter_program_id = device.create_program("filter.vs.glsl", "filter.fs.glsl");
        let box_shadow_program_id = device.create_program("box_shadow.vs.glsl",
                                                          "box_shadow.fs.glsl");
        let blur_program_id = device.create_program("blur.vs.glsl", "blur.fs.glsl");
        let clear_program_id = device.create_program("clear.vs.glsl", "clear.fs.glsl");

        let u_quad_transform_array = device.get_uniform_location(quad_program_id, "uMatrixPalette");
        let u_tile_params = device.get_uniform_location(quad_program_id, "uTileParams");
        let u_atlas_params = device.get_uniform_location(quad_program_id, "uAtlasParams");

        let u_blend_params = device.get_uniform_location(blend_program_id, "uBlendParams");

        let u_filter_params = device.get_uniform_location(filter_program_id, "uFilterParams");

        let u_direction = device.get_uniform_location(blur_program_id, "uDirection");

        let texture_ids = device.create_texture_ids(1024);
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

        let debug_renderer = DebugRenderer::new(&mut device);

        device.end_frame();

        let notifier = Arc::new(Mutex::new(None));
        let backend_notifier = notifier.clone();

        // We need a reference to the webrender context from the render backend in order to share
        // texture ids
        let context_handle = NativeGLContext::current_handle();

        thread::spawn(move || {
            let mut backend = RenderBackend::new(api_rx,
                                                 result_tx,
                                                 initial_viewport,
                                                 device_pixel_ratio,
                                                 white_image_id,
                                                 dummy_mask_image_id,
                                                 texture_cache,
                                                 enable_aa,
                                                 backend_notifier,
                                                 context_handle);
            backend.run();
        });

        let renderer = Renderer {
            result_rx: result_rx,
            device: device,
            current_frame: None,
            vertex_buffers: HashMap::with_hash_state(Default::default()),
            raster_batches: Vec::new(),
            pending_texture_updates: Vec::new(),
            pending_batch_updates: Vec::new(),
            border_program_id: border_program_id,
            device_pixel_ratio: device_pixel_ratio,
            blend_program_id: blend_program_id,
            filter_program_id: filter_program_id,
            quad_program_id: quad_program_id,
            blit_program_id: blit_program_id,
            box_shadow_program_id: box_shadow_program_id,
            blur_program_id: blur_program_id,
            clear_program_id: clear_program_id,
            u_blend_params: u_blend_params,
            u_filter_params: u_filter_params,
            u_direction: u_direction,
            u_quad_transform_array: u_quad_transform_array,
            u_atlas_params: u_atlas_params,
            u_tile_params: u_tile_params,
            notifier: notifier,
            viewport_size: Size2D::new(width, height),
            framebuffer_size: *framebuffer_size,
            debug: debug_renderer,
            backend_profile_counters: BackendProfileCounters::new(),
            profile_counters: RendererProfileCounters::new(),
            profiler: Profiler::new(),
            enable_profiler: enable_profiler,
            last_time: 0,
        };

        let sender = RenderApiSender::new(api_tx);

        (renderer, sender)
    }

    pub fn set_render_notifier(&self, notifier: Box<RenderNotifier>) {
        let mut notifier_arc = self.notifier.lock().unwrap();
        *notifier_arc = Some(notifier);
    }

    pub fn current_epoch(&self, pipeline_id: PipelineId) -> Option<Epoch> {
        self.current_frame.as_ref().and_then(|frame| {
            frame.pipeline_epoch_map.get(&pipeline_id).map(|epoch| *epoch)
        })
    }

    pub fn update(&mut self) {
        // Pull any pending results and return the most recent.
        while let Ok(msg) = self.result_rx.try_recv() {
            match msg {
                ResultMsg::UpdateTextureCache(update_list) => {
                    self.pending_texture_updates.push(update_list);
                }
                ResultMsg::UpdateBatches(update_list) => {
                    self.pending_batch_updates.push(update_list);
                }
                ResultMsg::NewFrame(frame, profile_counters) => {
                    self.backend_profile_counters = profile_counters;
                    self.current_frame = Some(frame);
                }
            }
        }
    }

    pub fn render(&mut self) {
        let mut profile_timers = RendererProfileTimers::new();

        profile_timers.total_time.profile(|| {
            self.device.begin_frame();
            self.update_texture_cache();
            self.update_batches();
            self.draw_frame();
        });

        let current_time = precise_time_ns();
        let ns = current_time - self.last_time;
        self.profile_counters.frame_time.set(ns);

        if self.enable_profiler {
            self.profiler.draw_profile(&self.backend_profile_counters,
                                       &self.profile_counters,
                                       &profile_timers,
                                       &mut self.debug);
        }

        self.profile_counters.reset();
        self.profile_counters.frame_counter.inc();

        self.debug.render(&mut self.device, &self.viewport_size);
        self.device.end_frame();
        self.last_time = current_time;
    }

    fn update_batches(&mut self) {
        let mut pending_batch_updates = mem::replace(&mut self.pending_batch_updates, vec![]);
        for update_list in pending_batch_updates.drain(..) {
            for update in update_list.updates {
                match update.op {
                    BatchUpdateOp::Create(vertices, indices) => {
                        let vao_id = self.device.create_vao(VertexFormat::Batch);
                        self.device.bind_vao(vao_id);

                        self.device.update_vao_indices(vao_id,
                                                       &indices,
                                                       VertexUsageHint::Static);
                        self.device.update_vao_vertices(vao_id,
                                                        &vertices,
                                                        VertexUsageHint::Static);

                        self.vertex_buffers.insert(update.id, VertexBuffer {
                            vao_id: vao_id,
                        });
                    }
                    BatchUpdateOp::Destroy => {
                        let vertex_buffer = self.vertex_buffers.remove(&update.id).unwrap();
                        self.device.delete_vao(vertex_buffer.vao_id);
                    }
                }
            }
        }
    }

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
                    TextureUpdateOp::Update(x, y, width, height, details) => {
                        match details {
                            TextureUpdateDetails::Raw => {
                                self.device.update_raw_texture(update.id, x, y, width, height);
                            }
                            TextureUpdateDetails::Blit(bytes) => {
                                self.device.update_texture_for_noncomposite_operation(
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
                                                       horizontal_blur_texture_image) => {
                                let radius =
                                    f32::ceil(radius.to_f32_px() * self.device_pixel_ratio) as u32;
                                self.device.update_texture_for_noncomposite_operation(
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
                                let dest_texture_origin =
                                    Point2D::new(horizontal_blur_texture_image.pixel_uv.x as f32,
                                                 horizontal_blur_texture_image.pixel_uv.y as f32);
                                let dest_texture_size = Size2D::new(width as f32, height as f32);
                                let source_texture_size = Size2D::new(glyph_size.width as f32,
                                                                      glyph_size.height as f32);
                                let blur_radius = radius as f32;

                                let vertices = [
                                    PackedVertexForTextureCacheUpdate::new(
                                        &dest_texture_origin,
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
                                        &Point2D::new(dest_texture_origin.x + width,
                                                      dest_texture_origin.y),
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
                                        &Point2D::new(dest_texture_origin.x,
                                                      dest_texture_origin.y + height),
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
                                        &Point2D::new(dest_texture_origin.x + width,
                                                      dest_texture_origin.y + height),
                                        &white,
                                        &Point2D::new(1.0, 1.0),
                                        &zero_point,
                                        &zero_point,
                                        &unblurred_glyph_texture_image.texel_uv.origin,
                                        &unblurred_glyph_texture_image.texel_uv.bottom_right(),
                                        &dest_texture_size,
                                        &source_texture_size,
                                        blur_radius),
                                ];

                                {
                                    let mut batch = self.get_or_create_raster_batch(
                                        horizontal_blur_texture_image.texture_id,
                                        unblurred_glyph_texture_image.texture_id,
                                        blur_program_id,
                                        Some(AxisDirection::Horizontal));
                                    batch.add_draw_item(horizontal_blur_texture_image.texture_id,
                                                        unblurred_glyph_texture_image.texture_id,
                                                        &vertices);
                                }

                                let (x, y) = (x as f32, y as f32);
                                let source_texture_size = Size2D::new(width as f32, height as f32);
                                let vertices = [
                                    PackedVertexForTextureCacheUpdate::new(
                                        &Point2D::new(x, y),
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
                                        &Point2D::new(x + width, y),
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
                                        &Point2D::new(x, y + height),
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
                                        &Point2D::new(x + width, y + height),
                                        &white,
                                        &Point2D::new(1.0, 1.0),
                                        &zero_point,
                                        &zero_point,
                                        &horizontal_blur_texture_image.texel_uv.origin,
                                        &horizontal_blur_texture_image.texel_uv.bottom_right(),
                                        &dest_texture_size,
                                        &source_texture_size,
                                        blur_radius),
                                ];

                                {
                                    let mut batch = self.get_or_create_raster_batch(
                                        update.id,
                                        horizontal_blur_texture_image.texture_id,
                                        blur_program_id,
                                        Some(AxisDirection::Vertical));
                                    batch.add_draw_item(update.id,
                                                        horizontal_blur_texture_image.texture_id,
                                                        &vertices);
                                }
                            }
                            TextureUpdateDetails::BorderRadius(outer_rx,
                                                               outer_ry,
                                                               inner_rx,
                                                               inner_ry,
                                                               index,
                                                               inverted) => {
                                let x = x as f32;
                                let y = y as f32;
                                let inner_rx = inner_rx.to_f32_px();
                                let inner_ry = inner_ry.to_f32_px();
                                let outer_rx = outer_rx.to_f32_px();
                                let outer_ry = outer_ry.to_f32_px();

                                let border_program_id = self.border_program_id;
                                let color = if inverted {
                                    ColorF::new(0.0, 0.0, 0.0, 1.0)
                                } else {
                                    ColorF::new(1.0, 1.0, 1.0, 1.0)
                                };

                                let border_radii_outer = Point2D::new(outer_rx, outer_ry);
                                let border_radii_inner = Point2D::new(inner_rx, inner_ry);

                                let tessellated_rect =
                                    Rect::new(Point2D::new(0.0, 0.0),
                                              Size2D::new(outer_rx, outer_ry));
                                let tessellated_rect =
                                    tessellated_rect.tessellate_border_corner(
                                        &Size2D::new(outer_rx, outer_ry),
                                        &Size2D::new(inner_rx, inner_ry),
                                        BasicRotationAngle::Upright,
                                        index);
                                let border_position =
                                    Point2D::new(x - tessellated_rect.origin.x + outer_rx,
                                                 y - tessellated_rect.origin.y + outer_ry);
                                let zero_point = Point2D::new(0.0, 0.0);
                                let zero_size = Size2D::new(0.0, 0.0);

                                let (x, y) = (x as f32, y as f32);
                                let (width, height) = (width as f32, height as f32);

                                let vertices: [PackedVertexForTextureCacheUpdate; 4] = [
                                    PackedVertexForTextureCacheUpdate::new(
                                        &Point2D::new(x, y),
                                        &color,
                                        &zero_point,
                                        &border_radii_outer,
                                        &border_radii_inner,
                                        &border_position,
                                        &zero_point,
                                        &zero_size,
                                        &zero_size,
                                        0.0),
                                    PackedVertexForTextureCacheUpdate::new(
                                        &Point2D::new(x + width, y),
                                        &color,
                                        &zero_point,
                                        &border_radii_outer,
                                        &border_radii_inner,
                                        &border_position,
                                        &zero_point,
                                        &zero_size,
                                        &zero_size,
                                        0.0),
                                    PackedVertexForTextureCacheUpdate::new(
                                        &Point2D::new(x, y + height),
                                        &color,
                                        &zero_point,
                                        &border_radii_outer,
                                        &border_radii_inner,
                                        &border_position,
                                        &zero_point,
                                        &zero_size,
                                        &zero_size,
                                        0.0),
                                    PackedVertexForTextureCacheUpdate::new(
                                        &Point2D::new(x + width, y + height),
                                        &color,
                                        &zero_point,
                                        &border_radii_outer,
                                        &border_radii_inner,
                                        &border_position,
                                        &zero_point,
                                        &zero_size,
                                        &zero_size,
                                        0.0),
                                ];

                                let mut batch = self.get_or_create_raster_batch(update.id,
                                                                                TextureId(0),
                                                                                border_program_id,
                                                                                None);
                                batch.add_draw_item(update.id, TextureId(0), &vertices);
                            }
                            TextureUpdateDetails::BoxShadow(blur_radius, part, inverted) => {
                                self.update_texture_cache_for_box_shadow(
                                    update.id,
                                    &Rect::new(Point2D::new(x as f32, y as f32),
                                               Size2D::new(width as f32, height as f32)),
                                    blur_radius,
                                    part,
                                    inverted)
                            }
                        }
                    }
                }
            }
        }

        self.flush_raster_batches();
    }

    fn update_texture_cache_for_box_shadow(&mut self,
                                           update_id: TextureId,
                                           rect: &Rect<f32>,
                                           blur_radius: Au,
                                           box_shadow_part: BoxShadowPart,
                                           inverted: bool) {
        let box_shadow_program_id = self.box_shadow_program_id;

        let blur_radius = blur_radius.to_f32_px();

        let color = if inverted {
            ColorF::new(1.0, 1.0, 1.0, 0.0)
        } else {
            ColorF::new(1.0, 1.0, 1.0, 1.0)
        };

        let zero_point = Point2D::new(0.0, 0.0);
        let zero_size = Size2D::new(0.0, 0.0);

        // `arc_radius_inner` here is just a flag to specify to the shader whether we're an edge
        // (zero) or a corner (nonzero).
        let (arc_radius_outer, arc_radius_inner) = match box_shadow_part {
            BoxShadowPart::Edge => {
                (Point2D::new(rect.size.width, 0.0), Point2D::new(0.0, 0.0))
            }
            BoxShadowPart::Corner(border_radius) => {
                (Point2D::new(border_radius.to_f32_px(), border_radius.to_f32_px()),
                 Point2D::new(1.0, 1.0))
            }
        };

        let vertices: [PackedVertexForTextureCacheUpdate; 4] = [
            PackedVertexForTextureCacheUpdate::new(&rect.origin,
                                                   &color,
                                                   &zero_point,
                                                   &arc_radius_outer,
                                                   &arc_radius_inner,
                                                   &zero_point,
                                                   &rect.origin,
                                                   &rect.size,
                                                   &zero_size,
                                                   blur_radius),
            PackedVertexForTextureCacheUpdate::new(&rect.top_right(),
                                                   &color,
                                                   &zero_point,
                                                   &arc_radius_outer,
                                                   &arc_radius_inner,
                                                   &zero_point,
                                                   &rect.origin,
                                                   &rect.size,
                                                   &zero_size,
                                                   blur_radius),
            PackedVertexForTextureCacheUpdate::new(&rect.bottom_left(),
                                                   &color,
                                                   &zero_point,
                                                   &arc_radius_outer,
                                                   &arc_radius_inner,
                                                   &zero_point,
                                                   &rect.origin,
                                                   &rect.size,
                                                   &zero_size,
                                                   blur_radius),
            PackedVertexForTextureCacheUpdate::new(&rect.bottom_right(),
                                                   &color,
                                                   &zero_point,
                                                   &arc_radius_outer,
                                                   &arc_radius_inner,
                                                   &zero_point,
                                                   &rect.origin,
                                                   &rect.size,
                                                   &zero_size,
                                                   blur_radius),
        ];

        let mut batch = self.get_or_create_raster_batch(update_id,
                                                        TextureId(0),
                                                        box_shadow_program_id,
                                                        None);
        batch.add_draw_item(update_id, TextureId(0), &vertices);
    }

    fn get_or_create_raster_batch(&mut self,
                                  dest_texture_id: TextureId,
                                  color_texture_id: TextureId,
                                  program_id: ProgramId,
                                  blur_direction: Option<AxisDirection>)
                                  -> &mut RasterBatch {
        // FIXME(pcwalton): Use a hash table if this linear search shows up in the profile.
        let mut index = None;
        for (i, batch) in self.raster_batches.iter_mut().enumerate() {
            if batch.can_add_to_batch(dest_texture_id,
                                      color_texture_id,
                                      program_id,
                                      blur_direction) {
                index = Some(i);
                break;
            }
        }

        if index.is_none() {
            index = Some(self.raster_batches.len());
            self.raster_batches.push(RasterBatch::new(program_id,
                                                      blur_direction,
                                                      dest_texture_id,
                                                      color_texture_id));
        }

        &mut self.raster_batches[index.unwrap()]
    }

    fn flush_raster_batches(&mut self) {
        let batches = mem::replace(&mut self.raster_batches, vec![]);
        if batches.len() > 0 {
            println!("flushing {:?} raster batches", batches.len());
        }

        // All horizontal blurs must complete before anything else.
        let mut remaining_batches = vec![];
        for batch in batches.into_iter() {
            if batch.blur_direction != Some(AxisDirection::Horizontal) {
                remaining_batches.push(batch);
                continue
            }

            self.set_up_gl_state_for_texture_cache_update(batch.dest_texture_id,
                                                          batch.color_texture_id,
                                                          batch.program_id,
                                                          batch.blur_direction);
            self.perform_gl_texture_cache_update(batch);
        }

        // Flush the remaining batches.
        for batch in remaining_batches.into_iter() {
            self.set_up_gl_state_for_texture_cache_update(batch.dest_texture_id,
                                                          batch.color_texture_id,
                                                          batch.program_id,
                                                          batch.blur_direction);
            self.perform_gl_texture_cache_update(batch);
        }
    }

    fn set_up_gl_state_for_texture_cache_update(&mut self,
                                                update_id: TextureId,
                                                color_texture_id: TextureId,
                                                program_id: ProgramId,
                                                blur_direction: Option<AxisDirection>) {
        gl::disable(gl::BLEND);
        gl::disable(gl::DEPTH_TEST);
        gl::disable(gl::SCISSOR_TEST);

        let (texture_width, texture_height) = self.device.get_texture_dimensions(update_id);

        let projection = Matrix4::ortho(0.0,
                                        texture_width as f32,
                                        0.0,
                                        texture_height as f32,
                                        ORTHO_NEAR_PLANE,
                                        ORTHO_FAR_PLANE);

        self.device.bind_render_target(Some(update_id));
        gl::viewport(0, 0, texture_width as gl::GLint, texture_height as gl::GLint);

        self.device.bind_program(program_id, &projection);

        self.device.bind_color_texture(color_texture_id);
        self.device.bind_mask_texture(TextureId(0));

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
        let vao_id = self.device.create_vao(VertexFormat::RasterOp);
        self.device.bind_vao(vao_id);

        self.device.update_vao_indices(vao_id, &batch.indices[..], VertexUsageHint::Dynamic);
        self.device.update_vao_vertices(vao_id, &batch.vertices[..], VertexUsageHint::Dynamic);

        self.profile_counters.vertices.add(batch.indices.len());
        self.profile_counters.draw_calls.inc();

        self.device.draw_triangles_u16(0, batch.indices.len() as gl::GLint);
        self.device.delete_vao(vao_id);
    }

    fn draw_frame(&mut self) {
        if let Some(ref frame) = self.current_frame {
            // TODO: cache render targets!

            // Draw render targets in reverse order to ensure dependencies
            // of earlier render targets are already available.
            // TODO: Are there cases where this fails and needs something
            // like a topological sort with dependencies?
            let mut render_context = RenderContext {
                blend_program_id: self.blend_program_id,
                filter_program_id: self.filter_program_id,
                temporary_fb_texture: self.device.create_texture_ids(1)[0],
                projection: Matrix4::identity(),
                layer_size: Size2D::zero(),
            };

            debug_assert!(frame.layers.len() > 0);
            let framebuffer_size = self.framebuffer_size;

            for layer in frame.layers.iter().rev() {
                render_context.layer_size = layer.size;

                let layer_texture_id = layer.render_targets[0].texture.map(|texture| {
                    texture.texture_id
                });
                self.device.bind_render_target(layer_texture_id);

                let mut uv;
                let mut uv_rects = vec![];
                let render_target_size;
                let scale_factor;
                let viewport_y;
                match layer.render_targets[0].texture {
                    None => {
                        let v = layer.size.height as f32;
                        uv = Rect::new(Point2D::new(0, v as u32), layer.size);
                        uv_rects.push(uv);
                        scale_factor = framebuffer_size.width as f32 / layer.size.width as f32;
                        render_target_size = Size2D::new(
                            (layer.size.width as f32 * scale_factor) as u32,
                            (layer.size.height as f32 * scale_factor) as u32);
                        viewport_y = render_target_size.height as f32 - uv.origin.y as f32 *
                            scale_factor;
                    }
                    Some(ref texture) => {
                        render_target_size = texture.texture_size;
                        scale_factor = 1.0;
                        uv = texture.uv_rect;
                        uv_rects.push(uv);

                        for render_target in &layer.render_targets[1..] {
                            if let Some(ref texture) = render_target.texture {
                                uv = uv.union(&texture.uv_rect);
                                uv_rects.push(texture.uv_rect);
                            }
                        }

                        viewport_y = render_target_size.height as f32 -
                            uv.max_y() as f32 * scale_factor;
                    }
                };
                let viewport = Rect::new(
                    Point2D::new((uv.origin.x as f32 * scale_factor) as gl::GLint,
                                 viewport_y as gl::GLint),
                    Size2D::new((uv.size.width as f32 * scale_factor) as gl::GLint,
                                (uv.size.height as f32 * scale_factor) as gl::GLint));
                gl::viewport(viewport.origin.x,
                             viewport.origin.y,
                             viewport.size.width,
                             viewport.size.height);

                // Clear frame buffer
                clear_framebuffer(&mut self.device,
                                  &mut render_context,
                                  &mut self.profile_counters,
                                  &viewport,
                                  &uv,
                                  self.device_pixel_ratio,
                                  &uv_rects[..],
                                  self.clear_program_id,
                                  layer.render_targets[0].texture.is_some());
                gl::enable(gl::DEPTH_TEST);
                gl::depth_func(gl::LEQUAL);

                render_context.projection =
                    Matrix4::ortho(0.0,
                                   viewport.size.width as f32 / self.device_pixel_ratio,
                                   viewport.size.height as f32 / self.device_pixel_ratio,
                                   0.0,
                                   ORTHO_NEAR_PLANE,
                                   ORTHO_FAR_PLANE);

                for cmd in &layer.commands {
                    match cmd {
                        &DrawCommand::Clear(ref info) => {
                            let mut clear_bits = 0;
                            if info.clear_color {
                                clear_bits |= gl::COLOR_BUFFER_BIT;
                            }
                            if info.clear_z {
                                clear_bits |= gl::DEPTH_BUFFER_BIT;
                            }
                            if info.clear_stencil {
                                clear_bits |= gl::STENCIL_BUFFER_BIT;
                            }
                            gl::clear(clear_bits);
                        }
                        &DrawCommand::Batch(ref info) => {
                            // TODO: probably worth sorting front to back to minimize overdraw (if profiling shows fragment / rop bound)

                            gl::enable(gl::BLEND);
                            gl::blend_func_separate(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA,
                                                    gl::ONE, gl::ONE);
                            gl::blend_equation(gl::FUNC_ADD);

                            self.device.bind_program(self.quad_program_id,
                                                     &render_context.projection);

                            self.device.set_uniform_mat4_array(self.u_quad_transform_array,
                                                               &info.matrix_palette);

                            for draw_call in &info.draw_calls {
                                let vao_id = self.vertex_buffers[&draw_call.vertex_buffer_id].vao_id;
                                self.device.bind_vao(vao_id);

                                if draw_call.tile_params.len() > 0 {
                                    // TODO(gw): Avoid alloc here...
                                    let mut floats = Vec::new();
                                    for vec in &draw_call.tile_params {
                                        floats.push(vec.u0);
                                        floats.push(vec.v0);
                                        floats.push(vec.u_size);
                                        floats.push(vec.v_size);
                                    }

                                    self.device.set_uniform_vec4_array(self.u_tile_params,
                                                                       &floats);
                                }

                                self.device.bind_mask_texture(draw_call.mask_texture_id);
                                self.device.bind_color_texture(draw_call.color_texture_id);

                                // TODO(gw): Although a minor cost, this is an extra hashtable lookup for every
                                //           draw call, when the batch textures are (almost) always the same.
                                //           This could probably be cached or provided elsewhere.
                                let color_size = self.device
                                                     .get_texture_dimensions(draw_call.color_texture_id);
                                let mask_size = self.device
                                                    .get_texture_dimensions(draw_call.mask_texture_id);
                                self.device.set_uniform_4f(self.u_atlas_params,
                                                           color_size.0 as f32,
                                                           color_size.1 as f32,
                                                           mask_size.0 as f32,
                                                           mask_size.1 as f32);

                                let index_count = draw_call.index_count as i32;
                                assert!(draw_call.first_vertex <= 65535);

                                self.profile_counters.draw_calls.inc();

                                self.device.draw_triangles_u16(draw_call.first_vertex as i32,
                                                               index_count);
                            }
                        }
                        &DrawCommand::CompositeBatch(ref info) => {
                            let needs_fb = match info.operation {
                                CompositionOp::MixBlend(MixBlendMode::Normal) => unreachable!(),

                                CompositionOp::MixBlend(MixBlendMode::Screen) |
                                CompositionOp::MixBlend(MixBlendMode::Overlay) |
                                CompositionOp::MixBlend(MixBlendMode::ColorDodge) |
                                CompositionOp::MixBlend(MixBlendMode::ColorBurn) |
                                CompositionOp::MixBlend(MixBlendMode::HardLight) |
                                CompositionOp::MixBlend(MixBlendMode::SoftLight) |
                                CompositionOp::MixBlend(MixBlendMode::Difference) |
                                CompositionOp::MixBlend(MixBlendMode::Exclusion) |
                                CompositionOp::MixBlend(MixBlendMode::Hue) |
                                CompositionOp::MixBlend(MixBlendMode::Saturation) |
                                CompositionOp::MixBlend(MixBlendMode::Color) |
                                CompositionOp::MixBlend(MixBlendMode::Luminosity) |
                                CompositionOp::Filter(LowLevelFilterOp::Blur(..)) |
                                CompositionOp::Filter(LowLevelFilterOp::Contrast(_)) |
                                CompositionOp::Filter(LowLevelFilterOp::Grayscale(_)) |
                                CompositionOp::Filter(LowLevelFilterOp::HueRotate(_)) |
                                CompositionOp::Filter(LowLevelFilterOp::Invert(_)) |
                                CompositionOp::Filter(LowLevelFilterOp::Saturate(_)) |
                                CompositionOp::Filter(LowLevelFilterOp::Sepia(_)) => true,

                                CompositionOp::Filter(LowLevelFilterOp::Brightness(_)) |
                                CompositionOp::Filter(LowLevelFilterOp::Opacity(_)) |
                                CompositionOp::MixBlend(MixBlendMode::Multiply) |
                                CompositionOp::MixBlend(MixBlendMode::Darken) |
                                CompositionOp::MixBlend(MixBlendMode::Lighten) => false,
                            };

                            let alpha;
                            if needs_fb {
                                gl::disable(gl::BLEND);

                                // TODO(glennw): No need to re-init this FB working copy texture
                                // every time...
                                for job in &info.jobs {
                                    let x0 = job.rect.origin.x;
                                    let y0 = job.rect.origin.y;

                                    self.device.init_texture(render_context.temporary_fb_texture,
                                                             job.rect.size.width as u32,
                                                             job.rect.size.height as u32,
                                                             ImageFormat::RGBA8,
                                                             TextureFilter::Nearest,
                                                             RenderTargetMode::None,
                                                             None);
                                    self.device.read_framebuffer_rect(
                                        render_context.temporary_fb_texture,
                                        x0,
                                        render_context.layer_size.height as i32 -
                                            job.rect.size.height -
                                            y0,
                                        job.rect.size.width,
                                        job.rect.size.height);
                                }

                                match info.operation {
                                    CompositionOp::MixBlend(blend_mode) => {
                                        self.device.bind_program(render_context.blend_program_id,
                                                                 &render_context.projection);
                                        self.device.set_uniform_4f(self.u_blend_params,
                                                                   blend_mode as i32 as f32,
                                                                   0.0,
                                                                   0.0,
                                                                   0.0);
                                    }
                                    CompositionOp::Filter(filter_op) => {
                                        self.device.bind_program(render_context.filter_program_id,
                                                                 &render_context.projection);

                                        let (opcode, amount, param0, param1) = match filter_op {
                                            LowLevelFilterOp::Blur(radius,
                                                                   AxisDirection::Horizontal) => {
                                                (0.0,
                                                 radius.to_f32_px() * self.device_pixel_ratio,
                                                 1.0,
                                                 0.0)
                                            }
                                            LowLevelFilterOp::Blur(radius,
                                                                   AxisDirection::Vertical) => {
                                                (0.0,
                                                 radius.to_f32_px() * self.device_pixel_ratio,
                                                 0.0,
                                                 1.0)
                                            }
                                            LowLevelFilterOp::Contrast(amount) => {
                                                (1.0, amount.to_f32_px(), 0.0, 0.0)
                                            }
                                            LowLevelFilterOp::Grayscale(amount) => {
                                                (2.0, amount.to_f32_px(), 0.0, 0.0)
                                            }
                                            LowLevelFilterOp::HueRotate(angle) => {
                                                (3.0,
                                                 (angle as f32) / ANGLE_FLOAT_TO_FIXED,
                                                 0.0,
                                                 0.0)
                                            }
                                            LowLevelFilterOp::Invert(amount) => {
                                                (4.0, amount.to_f32_px(), 0.0, 0.0)
                                            }
                                            LowLevelFilterOp::Saturate(amount) => {
                                                (5.0, amount.to_f32_px(), 0.0, 0.0)
                                            }
                                            LowLevelFilterOp::Sepia(amount) => {
                                                (6.0, amount.to_f32_px(), 0.0, 0.0)
                                            }
                                            LowLevelFilterOp::Brightness(_) |
                                            LowLevelFilterOp::Opacity(_) => {
                                                // Expressible using GL blend modes, so not handled
                                                // here.
                                                unreachable!()
                                            }
                                        };

                                        self.device.set_uniform_4f(self.u_filter_params,
                                                                   opcode,
                                                                   amount,
                                                                   param0,
                                                                   param1);
                                    }
                                }
                                self.device.bind_mask_texture(render_context.temporary_fb_texture);
                                alpha = 1.0;
                            } else {
                                gl::enable(gl::BLEND);

                                match info.operation {
                                    CompositionOp::Filter(LowLevelFilterOp::Brightness(
                                            amount)) => {
                                        gl::blend_func(gl::CONSTANT_COLOR, gl::ZERO);
                                        gl::blend_equation(gl::FUNC_ADD);
                                        gl::blend_color(amount.to_f32_px(),
                                                        amount.to_f32_px(),
                                                        amount.to_f32_px(),
                                                        1.0);
                                        alpha = 1.0;
                                    }
                                    CompositionOp::Filter(LowLevelFilterOp::Opacity(amount)) => {
                                        gl::blend_func(gl::SRC_ALPHA,
                                                       gl::ONE_MINUS_SRC_ALPHA);
                                        gl::blend_equation(gl::FUNC_ADD);
                                        alpha = amount.to_f32_px();
                                    }
                                    CompositionOp::MixBlend(MixBlendMode::Multiply) => {
                                        gl::blend_func(gl::DST_COLOR, gl::ZERO);
                                        gl::blend_equation(gl::FUNC_ADD);
                                        alpha = 1.0;
                                    }
                                    CompositionOp::MixBlend(MixBlendMode::Darken) => {
                                        gl::blend_func(gl::ONE, gl::ONE);
                                        gl::blend_equation(GL_BLEND_MIN);
                                        alpha = 1.0;
                                    }
                                    CompositionOp::MixBlend(MixBlendMode::Lighten) => {
                                        gl::blend_func(gl::ONE, gl::ONE);
                                        gl::blend_equation(GL_BLEND_MAX);
                                        alpha = 1.0;
                                    }
                                    _ => unreachable!(),
                                }

                                self.device.bind_program(self.blit_program_id,
                                                         &render_context.projection);
                            }

                            let (mut indices, mut vertices) = (vec![], vec![]);
                            for job in &info.jobs {
                                let x0 = job.rect.origin.x;
                                let y0 = job.rect.origin.y;
                                let x1 = job.rect.max_x();
                                let y1 = job.rect.max_y();

                                let vertex_count = vertices.len() as u16;
                                indices.push(vertex_count + 0);
                                indices.push(vertex_count + 1);
                                indices.push(vertex_count + 2);
                                indices.push(vertex_count + 2);
                                indices.push(vertex_count + 3);
                                indices.push(vertex_count + 1);

                                let color = ColorF::new(1.0, 1.0, 1.0, alpha);

                                let pixel_uv = Rect::new(
                                    Point2D::new(job.render_target_texture.uv_rect.origin.x,
                                                 job.render_target_texture.uv_rect.origin.y),
                                    Size2D::new(
                                        (job.rect.size.width as f32 * self.device_pixel_ratio) as
                                        u32,
                                        (job.rect.size.height as f32 * self.device_pixel_ratio) as
                                        u32));
                                let texture_width =
                                    job.render_target_texture.texture_size.width as f32;
                                let texture_height =
                                    job.render_target_texture.texture_size.height as f32;
                                let texture_uv = Rect::new(
                                    Point2D::new(
                                        pixel_uv.origin.x as f32 / texture_width,
                                        1.0 - (pixel_uv.origin.y + pixel_uv.size.height) as f32 /
                                            texture_height),
                                    Size2D::new(pixel_uv.size.width as f32 / texture_width,
                                                pixel_uv.size.height as f32 / texture_height));

                                vertices.push_all(&[
                                    PackedVertex::from_components_unscaled_muv(
                                        x0 as f32, y0 as f32,
                                        &color,
                                        texture_uv.origin.x, texture_uv.max_y(),
                                        job.rect.size.width as u16, job.rect.size.height as u16),
                                    PackedVertex::from_components_unscaled_muv(
                                        x1 as f32, y0 as f32,
                                        &color,
                                        texture_uv.max_x(), texture_uv.max_y(),
                                        job.rect.size.width as u16, job.rect.size.height as u16),
                                    PackedVertex::from_components_unscaled_muv(
                                        x0 as f32, y1 as f32,
                                        &color,
                                        texture_uv.origin.x, texture_uv.origin.y,
                                        job.rect.size.width as u16, job.rect.size.height as u16),
                                    PackedVertex::from_components_unscaled_muv(
                                        x1 as f32, y1 as f32,
                                        &color,
                                        texture_uv.max_x(), texture_uv.origin.y,
                                        job.rect.size.width as u16, job.rect.size.height as u16),
                                ]);
                            }

                            draw_simple_triangles(&mut self.device,
                                                  &mut render_context,
                                                  &mut self.profile_counters,
                                                  &indices[..],
                                                  &vertices[..],
                                                  info.texture_id);
                        }
                    }
                }
            }
        }
    }

}

fn clear_framebuffer(device: &mut Device,
                     render_context: &mut RenderContext,
                     profile_counters: &mut RendererProfileCounters,
                     viewport: &Rect<gl::GLint>,
                     combined_uv: &Rect<u32>,
                     device_pixel_ratio: f32,
                     uv_rects: &[Rect<u32>],
                     program_id: ProgramId,
                     clear_to_transparent: bool) {
    let clear_color = ColorF {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: if clear_to_transparent { 0.0 } else { 1.0 },
    };

    // Fast path if we only have one rect: just use glClear().
    //
    // TODO(pcwalton): We could take this path too if the rects all union together precisely
    // to the viewport. But I kinda doubt it's worth the trouble.
    if uv_rects.len() < 2 {
        gl::scissor(viewport.origin.x, viewport.origin.y,
                    viewport.size.width, viewport.size.height);
        gl::enable(gl::SCISSOR_TEST);
        gl::clear_color(clear_color.r, clear_color.g, clear_color.b, clear_color.a);
        gl::clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT | gl::STENCIL_BUFFER_BIT);
        gl::disable(gl::SCISSOR_TEST);
        return;
    }

    // Slow path if we have multiple rects.
    //
    // We use [0,1) coordinates here because it's simpler to centralize the scaling to the viewport
    // in one place.
    let (mut indices, mut vertices) = (vec![], vec![]);
    for rect in uv_rects {
        let x0 = (rect.origin.x as f32 - combined_uv.origin.x as f32 * device_pixel_ratio) /
            viewport.size.width as f32;
        let y0 = (rect.origin.y as f32 - combined_uv.origin.y as f32 * device_pixel_ratio) /
            viewport.size.height as f32;
        let x1 = (rect.max_x() as f32 - combined_uv.origin.x as f32 * device_pixel_ratio) /
            viewport.size.width as f32;
        let y1 = (rect.max_y() as f32 - combined_uv.origin.y as f32 * device_pixel_ratio) /
            viewport.size.height as f32;
        indices.extend([0, 1, 2, 1, 3, 2].iter().map(|index| (index + vertices.len()) as u16));
        vertices.push_all(&[
            PackedVertex::from_components(x0, y0, &clear_color, 0.0, 0.0, 0.0, 0.0),
            PackedVertex::from_components(x1, y0, &clear_color, 0.0, 0.0, 0.0, 0.0),
            PackedVertex::from_components(x0, y1, &clear_color, 0.0, 0.0, 0.0, 0.0),
            PackedVertex::from_components(x1, y1, &clear_color, 0.0, 0.0, 0.0, 0.0),
        ]);
    }

    let projection = Matrix4::ortho(0.0, 1.0,
                                    1.0, 0.0,
                                    ORTHO_NEAR_PLANE,
                                    ORTHO_FAR_PLANE);
    device.bind_program(program_id, &projection);
    gl::disable(gl::BLEND);
    draw_simple_triangles(device,
                          render_context,
                          profile_counters,
                          &indices[..],
                          &vertices[..],
                          TextureId(0));

    gl::clear(gl::DEPTH_BUFFER_BIT | gl::STENCIL_BUFFER_BIT);
}

fn draw_simple_triangles(device: &mut Device,
                         render_context: &mut RenderContext,
                         profile_counters: &mut RendererProfileCounters,
                         indices: &[u16],
                         vertices: &[PackedVertex],
                         texture: TextureId) {
    // TODO(glennw): Don't re-create this VAO all the time. Create it once and set positions
    // via uniforms.
    let vao_id = device.create_vao(VertexFormat::Batch);
    device.bind_color_texture(texture);
    device.bind_vao(vao_id);
    device.update_vao_indices(vao_id, &indices[..], VertexUsageHint::Dynamic);
    device.update_vao_vertices(vao_id, &vertices[..], VertexUsageHint::Dynamic);

    profile_counters.vertices.add(indices.len());
    profile_counters.draw_calls.inc();

    device.draw_triangles_u16(0, indices.len() as gl::GLint);
    device.delete_vao(vao_id);
}

