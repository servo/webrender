use app_units::Au;
use batch::{RasterBatch, VertexBufferId};
use device::{Device, ProgramId, TextureId, TextureIndex, UniformLocation, VAOId, VertexUsageHint};
use euclid::{Rect, Matrix4, Point2D, Size2D};
use fnv::FnvHasher;
use gleam::gl;
use internal_types::{RendererFrame, ResultMsg, TextureUpdateOp, BatchUpdateOp, BatchUpdateList};
use internal_types::{TextureUpdateDetails, TextureUpdateList, PackedVertex, RenderTargetMode};
use internal_types::{ORTHO_NEAR_PLANE, ORTHO_FAR_PLANE, BoxShadowPart};
use internal_types::{PackedVertexForTextureCacheUpdate, TextureTarget};
use internal_types::{CompositionOp, LowLevelFilterOp, BlurDirection, DrawCommand};
use render_backend::RenderBackend;
use std::collections::HashMap;
use std::collections::hash_state::DefaultState;
use std::f32;
use std::mem;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};
use std::thread;
use texture_cache::{TextureCache, TextureInsertOp};
use webrender_traits::{ColorF, Epoch, PipelineId, RenderNotifier};
use webrender_traits::{ImageFormat, MixBlendMode, RenderApi};
//use util;

pub const BLUR_INFLATION_FACTOR: u32 = 3;

struct VertexBuffer {
    vao_id: VAOId,
}

struct RenderContext {
    blend_program_id: ProgramId,
    filter_program_id: ProgramId,
    temporary_fb_texture: TextureId,
    projection: Matrix4,
    layer_size: Size2D<u32>,
    draw_calls: usize,
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

    blit_program_id: ProgramId,

    border_program_id: ProgramId,

    blend_program_id: ProgramId,
    u_blend_params: UniformLocation,

    filter_program_id: ProgramId,
    u_filter_params: UniformLocation,
    u_filter_texture_size: UniformLocation,

    box_shadow_program_id: ProgramId,

    blur_program_id: ProgramId,
    u_direction: UniformLocation,

    tile_program_id: ProgramId,
}

impl Renderer {
    pub fn new(notifier: Box<RenderNotifier>,
               width: u32,
               height: u32,
               device_pixel_ratio: f32,
               resource_path: PathBuf) -> (Renderer, RenderApi) {
        let (api_tx, api_rx) = channel();
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
        let tile_program_id = device.create_program("tile.vs.glsl", "tile.fs.glsl");

        let u_quad_transform_array = device.get_uniform_location(quad_program_id, "uMatrixPalette");

        let u_blend_params = device.get_uniform_location(blend_program_id, "uBlendParams");

        let u_filter_params = device.get_uniform_location(filter_program_id, "uFilterParams");
        let u_filter_texture_size = device.get_uniform_location(filter_program_id, "uTextureSize");

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
                             TextureInsertOp::Blit(white_pixels));

        let dummy_mask_image_id = texture_cache.new_item_id();
        texture_cache.insert(dummy_mask_image_id,
                             0,
                             0,
                             2,
                             2,
                             ImageFormat::A8,
                             TextureInsertOp::Blit(mask_pixels));

        device.end_frame();

        let backend_api_tx = api_tx.clone();
        thread::spawn(move || {
            let mut backend = RenderBackend::new(api_rx,
                                                 backend_api_tx,
                                                 result_tx,
                                                 initial_viewport,
                                                 device_pixel_ratio,
                                                 white_image_id,
                                                 dummy_mask_image_id,
                                                 texture_cache);
            backend.run(notifier);
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
            tile_program_id: tile_program_id,
            u_blend_params: u_blend_params,
            u_filter_params: u_filter_params,
            u_filter_texture_size: u_filter_texture_size,
            u_direction: u_direction,
            u_quad_transform_array: u_quad_transform_array,
        };

        let api = RenderApi::new(api_tx);

        (renderer, api)
    }

    pub fn current_epoch(&self, pipeline_id: PipelineId) -> Option<Epoch> {
        self.current_frame.as_ref().and_then(|frame| {
            frame.pipeline_epoch_map.get(&pipeline_id).map(|epoch| *epoch)
        })
    }

    pub fn update(&mut self) {
        // Pull any pending results and return the most recent.
        loop {
            match self.result_rx.try_recv() {
                Ok(msg) => {
                    match msg {
                        ResultMsg::UpdateTextureCache(update_list) => {
                            self.pending_texture_updates.push(update_list);
                        }
                        ResultMsg::UpdateBatches(update_list) => {
                            self.pending_batch_updates.push(update_list);
                        }
                        ResultMsg::NewFrame(frame) => {
                            self.current_frame = Some(frame);
                        }
                    }
                }
                Err(..) => break,
            }
        }
    }

    pub fn render(&mut self) {
        //let _pf = util::ProfileScope::new("render");
        self.device.begin_frame();
        self.update_texture_cache();
        self.update_batches();
        self.draw_frame();
        self.device.end_frame();
    }

    fn update_batches(&mut self) {
        let mut pending_batch_updates = mem::replace(&mut self.pending_batch_updates, vec![]);
        for update_list in pending_batch_updates.drain(..) {
            for update in update_list.updates {
                match update.op {
                    BatchUpdateOp::Create(vertices, indices) => {
                        let vao_id = self.device.create_vao();
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
                    TextureUpdateOp::Create(target,
                                            width, height, levels,
                                            format,
                                            mode,
                                            maybe_bytes) => {
                        // TODO: clean up match
                        match maybe_bytes {
                            Some(bytes) => {
                                self.device.init_texture(target,
                                                         update.id,
                                                         width, height, levels,
                                                         format,
                                                         mode,
                                                         Some(bytes.as_slice()));
                            }
                            None => {
                                self.device.init_texture(target,
                                                         update.id,
                                                         width, height, levels,
                                                         format,
                                                         mode,
                                                         None);
                            }
                        }
                    }
                    TextureUpdateOp::DeinitRenderTarget(id) => {
                        self.device.deinit_texture(TextureTarget::Texture2D, id);
                    }
                    TextureUpdateOp::Update(x, y, width, height, details) => {
                        match details {
                            TextureUpdateDetails::Blit(bytes) => {
                                self.device.update_texture_for_noncomposite_operation(
                                    update.id,
                                    update.index,
                                    x, y,
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
                                    unblurred_glyph_texture_image.texture_index,
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
                                        unblurred_glyph_texture_image.texture_index,
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
                                        unblurred_glyph_texture_image.texture_index,
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
                                        unblurred_glyph_texture_image.texture_index,
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
                                        unblurred_glyph_texture_image.texture_index,
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
                                        horizontal_blur_texture_image.texture_index,
                                        unblurred_glyph_texture_image.texture_id,
                                        blur_program_id,
                                        Some(BlurDirection::Horizontal));
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
                                        horizontal_blur_texture_image.texture_index,
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
                                        horizontal_blur_texture_image.texture_index,
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
                                        horizontal_blur_texture_image.texture_index,
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
                                        horizontal_blur_texture_image.texture_index,
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
                                        update.index,
                                        horizontal_blur_texture_image.texture_id,
                                        blur_program_id,
                                        Some(BlurDirection::Vertical));
                                    batch.add_draw_item(update.id,
                                                        horizontal_blur_texture_image.texture_id,
                                                        &vertices);
                                }
                            }
                            TextureUpdateDetails::BorderRadius(outer_rx,
                                                               outer_ry,
                                                               inner_rx,
                                                               inner_ry,
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
                                let border_position = Point2D::new(x + outer_rx, y + outer_ry);
                                let zero_point = Point2D::new(0.0, 0.0);
                                let zero_size = Size2D::new(0.0, 0.0);
                                let vertices: [PackedVertexForTextureCacheUpdate; 4] = [
                                    PackedVertexForTextureCacheUpdate::new(
                                        &Point2D::new(x, y),
                                        &color,
                                        &zero_point,
                                        TextureIndex(0),
                                        &border_radii_outer,
                                        &border_radii_inner,
                                        &border_position,
                                        &zero_point,
                                        &zero_size,
                                        &zero_size,
                                        0.0),
                                    PackedVertexForTextureCacheUpdate::new(
                                        &Point2D::new(x + outer_rx, y),
                                        &color,
                                        &zero_point,
                                        TextureIndex(0),
                                        &border_radii_outer,
                                        &border_radii_inner,
                                        &border_position,
                                        &zero_point,
                                        &zero_size,
                                        &zero_size,
                                        0.0),
                                    PackedVertexForTextureCacheUpdate::new(
                                        &Point2D::new(x, y + outer_ry),
                                        &color,
                                        &zero_point,
                                        TextureIndex(0),
                                        &border_radii_outer,
                                        &border_radii_inner,
                                        &border_position,
                                        &zero_point,
                                        &zero_size,
                                        &zero_size,
                                        0.0),
                                    PackedVertexForTextureCacheUpdate::new(
                                        &Point2D::new(x + outer_rx, y + outer_ry),
                                        &color,
                                        &zero_point,
                                        TextureIndex(0),
                                        &border_radii_outer,
                                        &border_radii_inner,
                                        &border_position,
                                        &zero_point,
                                        &zero_size,
                                        &zero_size,
                                        0.0),
                                ];

                                let mut batch = self.get_or_create_raster_batch(update.id,
                                                                                update.index,
                                                                                TextureId(0),
                                                                                border_program_id,
                                                                                None);
                                batch.add_draw_item(update.id, TextureId(0), &vertices);
                            }
                            TextureUpdateDetails::BoxShadow(blur_radius, part, inverted) => {
                                self.update_texture_cache_for_box_shadow(
                                    update.id,
                                    update.index,
                                    &Rect::new(Point2D::new(x as f32, y as f32),
                                               Size2D::new(width as f32, height as f32)),
                                    blur_radius,
                                    part,
                                    inverted)
                            }
                            TextureUpdateDetails::Tile(bytes,
                                                       stretch_size,
                                                       scratch_texture_image) => {
                                self.device.update_texture_for_noncomposite_operation(
                                    scratch_texture_image.texture_id,
                                    scratch_texture_image.texture_index,
                                    scratch_texture_image.pixel_uv.x,
                                    scratch_texture_image.pixel_uv.y,
                                    stretch_size.width,
                                    stretch_size.height,
                                    bytes.as_slice());

                                let white = ColorF::new(1.0, 1.0, 1.0, 1.0);
                                let zero_point = Point2D::new(0.0, 0.0);
                                let zero_size = Size2D::new(0.0, 0.0);
                                let scaled_bottom_right =
                                    Point2D::new((width as f32) / (stretch_size.width as f32),
                                                 (height as f32) / (stretch_size.height as f32));

                                let tile_program_id = self.tile_program_id;
                                let mut batch = self.get_or_create_raster_batch(
                                    update.id,
                                    update.index,
                                    scratch_texture_image.texture_id,
                                    tile_program_id,
                                    None);
                                let vertices = [
                                    PackedVertexForTextureCacheUpdate::new(
                                        &Point2D::new(x as f32, y as f32),
                                        &white,
                                        &zero_point,
                                        scratch_texture_image.texture_index,
                                        &Point2D::new(0.0, 0.0),
                                        &zero_point,
                                        &scratch_texture_image.texel_uv.origin,
                                        &scratch_texture_image.texel_uv.bottom_right(),
                                        &zero_size,
                                        &zero_size,
                                        0.0),
                                    PackedVertexForTextureCacheUpdate::new(
                                        &Point2D::new((x + width) as f32, y as f32),
                                        &white,
                                        &zero_point,
                                        scratch_texture_image.texture_index,
                                        &Point2D::new(scaled_bottom_right.x, 0.0),
                                        &zero_point,
                                        &scratch_texture_image.texel_uv.origin,
                                        &scratch_texture_image.texel_uv.bottom_right(),
                                        &zero_size,
                                        &zero_size,
                                        0.0),
                                    PackedVertexForTextureCacheUpdate::new(
                                        &Point2D::new(x as f32, (y + height) as f32),
                                        &white,
                                        &zero_point,
                                        scratch_texture_image.texture_index,
                                        &Point2D::new(0.0, scaled_bottom_right.y),
                                        &zero_point,
                                        &scratch_texture_image.texel_uv.origin,
                                        &scratch_texture_image.texel_uv.bottom_right(),
                                        &zero_size,
                                        &zero_size,
                                        0.0),
                                    PackedVertexForTextureCacheUpdate::new(
                                        &Point2D::new((x + width) as f32, (y + height) as f32),
                                        &white,
                                        &zero_point,
                                        scratch_texture_image.texture_index,
                                        &scaled_bottom_right,
                                        &zero_point,
                                        &scratch_texture_image.texel_uv.origin,
                                        &scratch_texture_image.texel_uv.bottom_right(),
                                        &zero_size,
                                        &zero_size,
                                        0.0),
                                ];
                                batch.add_draw_item(update.id,
                                                    scratch_texture_image.texture_id,
                                                    &vertices);
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
                                           update_index: TextureIndex,
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

        let arc_radius = match box_shadow_part {
            BoxShadowPart::Edge => Point2D::new(rect.size.width, 0.0),
            BoxShadowPart::Corner(border_radius) => {
                Point2D::new(border_radius.to_f32_px(), border_radius.to_f32_px())
            }
        };

        let vertices: [PackedVertexForTextureCacheUpdate; 4] = [
            PackedVertexForTextureCacheUpdate::new(&rect.origin,
                                                   &color,
                                                   &zero_point,
                                                   TextureIndex(0),
                                                   &arc_radius,
                                                   &zero_point,
                                                   &zero_point,
                                                   &rect.origin,
                                                   &rect.size,
                                                   &zero_size,
                                                   blur_radius),
            PackedVertexForTextureCacheUpdate::new(&rect.top_right(),
                                                   &color,
                                                   &zero_point,
                                                   TextureIndex(0),
                                                   &arc_radius,
                                                   &zero_point,
                                                   &zero_point,
                                                   &rect.origin,
                                                   &rect.size,
                                                   &zero_size,
                                                   blur_radius),
            PackedVertexForTextureCacheUpdate::new(&rect.bottom_left(),
                                                   &color,
                                                   &zero_point,
                                                   TextureIndex(0),
                                                   &arc_radius,
                                                   &zero_point,
                                                   &zero_point,
                                                   &rect.origin,
                                                   &rect.size,
                                                   &zero_size,
                                                   blur_radius),
            PackedVertexForTextureCacheUpdate::new(&rect.bottom_right(),
                                                   &color,
                                                   &zero_point,
                                                   TextureIndex(0),
                                                   &arc_radius,
                                                   &zero_point,
                                                   &zero_point,
                                                   &rect.origin,
                                                   &rect.size,
                                                   &zero_size,
                                                   blur_radius),
        ];

        let mut batch = self.get_or_create_raster_batch(update_id,
                                                        update_index,
                                                        TextureId(0),
                                                        box_shadow_program_id,
                                                        None);
        batch.add_draw_item(update_id, TextureId(0), &vertices);
    }

    fn get_or_create_raster_batch(&mut self,
                                  dest_texture_id: TextureId,
                                  dest_texture_index: TextureIndex,
                                  color_texture_id: TextureId,
                                  program_id: ProgramId,
                                  blur_direction: Option<BlurDirection>)
                                  -> &mut RasterBatch {
        // FIXME(pcwalton): Use a hash table if this linear search shows up in the profile.
        let mut index = None;
        for (i, batch) in self.raster_batches.iter_mut().enumerate() {
            if batch.can_add_to_batch(dest_texture_id,
                                      dest_texture_index,
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
                                                      dest_texture_index,
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
            if batch.blur_direction != Some(BlurDirection::Horizontal) {
                remaining_batches.push(batch);
                continue
            }

            self.set_up_gl_state_for_texture_cache_update(batch.dest_texture_id,
                                                          batch.dest_texture_index,
                                                          batch.color_texture_id,
                                                          batch.program_id,
                                                          batch.blur_direction);
            self.perform_gl_texture_cache_update(batch);
        }

        // Flush the remaining batches.
        for batch in remaining_batches.into_iter() {
            self.set_up_gl_state_for_texture_cache_update(batch.dest_texture_id,
                                                          batch.dest_texture_index,
                                                          batch.color_texture_id,
                                                          batch.program_id,
                                                          batch.blur_direction);
            self.perform_gl_texture_cache_update(batch);
        }
    }

    fn set_up_gl_state_for_texture_cache_update(&mut self,
                                                update_id: TextureId,
                                                update_index: TextureIndex,
                                                color_texture_id: TextureId,
                                                program_id: ProgramId,
                                                blur_direction: Option<BlurDirection>) {
        gl::disable(gl::BLEND);
        gl::disable(gl::DEPTH_TEST);

        let (texture_width, texture_height) = self.device.get_texture_dimensions(update_id);

        let projection = Matrix4::ortho(0.0,
                                        texture_width as f32,
                                        0.0,
                                        texture_height as f32,
                                        ORTHO_NEAR_PLANE,
                                        ORTHO_FAR_PLANE);

        self.device.bind_render_target(Some((update_id, update_index)));
        gl::viewport(0, 0, texture_width as gl::GLint, texture_height as gl::GLint);

        self.device.bind_program(program_id, &projection);

        self.device.bind_color_texture_for_noncomposite_operation(color_texture_id);
        self.device.bind_mask_texture_for_noncomposite_operation(TextureId(0));

        match blur_direction {
            Some(BlurDirection::Horizontal) => {
                self.device.set_uniform_2f(self.u_direction, 1.0, 0.0)
            }
            Some(BlurDirection::Vertical) => {
                self.device.set_uniform_2f(self.u_direction, 0.0, 1.0)
            }
            None => {}
        }
    }

    fn perform_gl_texture_cache_update(&mut self, batch: RasterBatch) {
        let vao_id = self.device.create_vao_for_texture_cache_update();
        self.device.bind_vao_for_texture_cache_update(vao_id);

        self.device.update_vao_indices(vao_id, &batch.indices[..], VertexUsageHint::Dynamic);
        self.device.update_vao_vertices(vao_id, &batch.vertices[..], VertexUsageHint::Dynamic);

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
                draw_calls: 0,
            };

            debug_assert!(frame.layers.len() > 0);

            for layer in frame.layers.iter().rev() {
                render_context.layer_size = layer.size;
                render_context.projection = Matrix4::ortho(0.0,
                                                           layer.size.width as f32,
                                                           layer.size.height as f32,
                                                           0.0,
                                                           ORTHO_NEAR_PLANE,
                                                           ORTHO_FAR_PLANE);

                let layer_texture_id_and_index = layer.texture_id.map(|texture_id| {
                    (texture_id, TextureIndex(0))
                });
                self.device.bind_render_target(layer_texture_id_and_index);
                gl::viewport(0,
                             0,
                             (layer.size.width as f32 * self.device_pixel_ratio) as gl::GLint,
                             (layer.size.height as f32 * self.device_pixel_ratio) as gl::GLint);
                gl::enable(gl::DEPTH_TEST);
                gl::depth_func(gl::LEQUAL);

                // Clear frame buffer
                gl::clear_color(1.0, 1.0, 1.0, 1.0);
                gl::clear(gl::COLOR_BUFFER_BIT |
                          gl::DEPTH_BUFFER_BIT |
                          gl::STENCIL_BUFFER_BIT);

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
                            gl::blend_func(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
                            gl::blend_equation(gl::FUNC_ADD);

                            self.device.bind_program(self.quad_program_id,
                                                     &render_context.projection);

                            self.device.set_uniform_mat4_array(self.u_quad_transform_array,
                                                               &info.matrix_palette);

                            for draw_call in &info.draw_calls {
                                let vao_id = self.vertex_buffers[&draw_call.vertex_buffer_id].vao_id;
                                self.device.bind_vao(vao_id);

                                self.device.bind_mask_texture_for_noncomposite_operation(draw_call.mask_texture_id);
                                self.device.bind_color_texture_for_noncomposite_operation(draw_call.color_texture_id);

                                self.device.draw_triangles_u16(draw_call.first_vertex as i32,
                                                               draw_call.index_count as i32);
                                render_context.draw_calls += 1;
                            }
                        }
                        &DrawCommand::Composite(ref info) => {
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

                            let x0 = info.rect.origin.x;
                            let y0 = info.rect.origin.y;
                            let x1 = x0 + info.rect.size.width;
                            let y1 = y0 + info.rect.size.height;

                            if needs_fb {
                                gl::disable(gl::BLEND);

                                // TODO: No need to re-init this FB working copy texture every time...
                                self.device.init_texture(TextureTarget::Texture2D,
                                                         render_context.temporary_fb_texture,
                                                         info.rect.size.width,
                                                         info.rect.size.height,
                                                         1,
                                                         ImageFormat::RGBA8,
                                                         RenderTargetMode::None,
                                                         None);
                                self.device.read_framebuffer_rect(
                                    TextureTarget::Texture2D,
                                    render_context.temporary_fb_texture,
                                    TextureIndex(0),
                                    x0,
                                    render_context.layer_size.height - info.rect.size.height - y0,
                                    info.rect.size.width,
                                    info.rect.size.height);

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
                                                                   BlurDirection::Horizontal) => {
                                                (0.0,
                                                 radius.to_f32_px() * self.device_pixel_ratio,
                                                 1.0,
                                                 0.0)
                                            }
                                            LowLevelFilterOp::Blur(radius,
                                                                   BlurDirection::Vertical) => {
                                                (0.0,
                                                 radius.to_f32_px() * self.device_pixel_ratio,
                                                 0.0,
                                                 1.0)
                                            }
                                            LowLevelFilterOp::Contrast(amount) => {
                                                (1.0, amount, 0.0, 0.0)
                                            }
                                            LowLevelFilterOp::Grayscale(amount) => {
                                                (2.0, amount, 0.0, 0.0)
                                            }
                                            LowLevelFilterOp::HueRotate(angle) => {
                                                (3.0, angle, 0.0, 0.0)
                                            }
                                            LowLevelFilterOp::Invert(amount) => {
                                                (4.0, amount, 0.0, 0.0)
                                            }
                                            LowLevelFilterOp::Saturate(amount) => {
                                                (5.0, amount, 0.0, 0.0)
                                            }
                                            LowLevelFilterOp::Sepia(amount) => {
                                                (6.0, amount, 0.0, 0.0)
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
                                        self.device.set_uniform_2f(self.u_filter_texture_size,
                                                                   info.rect.size.width as f32,
                                                                   info.rect.size.height as f32);
                                    }
                                }
                                self.device.bind_mask_texture(TextureTarget::Texture2D,
                                                              render_context.temporary_fb_texture);
                            } else {
                                gl::enable(gl::BLEND);

                                match info.operation {
                                    CompositionOp::Filter(LowLevelFilterOp::Brightness(
                                            amount)) => {
                                        gl::blend_func(gl::CONSTANT_COLOR, gl::ZERO);
                                        gl::blend_equation(gl::FUNC_ADD);
                                        gl::blend_color(amount, amount, amount, 1.0);
                                    }
                                    CompositionOp::Filter(LowLevelFilterOp::Opacity(amount)) => {
                                        gl::blend_func(gl::CONSTANT_ALPHA,
                                                       gl::ONE_MINUS_CONSTANT_ALPHA);
                                        gl::blend_equation(gl::FUNC_ADD);
                                        gl::blend_color(1.0, 1.0, 1.0, amount);
                                    }
                                    CompositionOp::MixBlend(MixBlendMode::Multiply) => {
                                        gl::blend_func(gl::DST_COLOR, gl::ZERO);
                                        gl::blend_equation(gl::FUNC_ADD);
                                    }
                                    CompositionOp::MixBlend(MixBlendMode::Darken) => {
                                        gl::blend_func(gl::ONE, gl::ONE);
                                        gl::blend_equation(gl::MIN);
                                    }
                                    CompositionOp::MixBlend(MixBlendMode::Lighten) => {
                                        gl::blend_func(gl::ONE, gl::ONE);
                                        gl::blend_equation(gl::MAX);
                                    }
                                    _ => unreachable!(),
                                }

                                self.device.bind_program(self.blit_program_id, &render_context.projection);
                            }

                            let color = ColorF::new(1.0, 1.0, 1.0, 1.0);
                            let indices: [u16; 6] = [ 0, 1, 2, 2, 3, 1 ];
                            let color_texture_index = TextureIndex(0);
                            let vertices: [PackedVertex; 4] = [
                                PackedVertex::from_components(x0 as f32, y0 as f32,
                                                              &color,
                                                              0.0, 1.0,
                                                              0.0, 1.0,
                                                              color_texture_index,
                                                              TextureIndex(0)),
                                PackedVertex::from_components(x1 as f32, y0 as f32,
                                                              &color,
                                                              1.0, 1.0,
                                                              1.0, 1.0,
                                                              color_texture_index,
                                                              TextureIndex(0)),
                                PackedVertex::from_components(x0 as f32, y1 as f32,
                                                              &color,
                                                              0.0, 0.0,
                                                              0.0, 0.0,
                                                              color_texture_index,
                                                              TextureIndex(0)),
                                PackedVertex::from_components(x1 as f32, y1 as f32,
                                                              &color,
                                                              1.0, 0.0,
                                                              1.0, 0.0,
                                                              color_texture_index,
                                                              TextureIndex(0)),
                            ];
                            // TODO: Don't re-create this VAO all the time.
                            // Create it once and set positions via uniforms.
                            let vao_id = self.device.create_vao();
                            self.device.bind_color_texture(TextureTarget::Texture2D,
                                                           info.color_texture_id);
                            self.device.bind_vao(vao_id);
                            self.device.update_vao_indices(vao_id, &indices, VertexUsageHint::Dynamic);
                            self.device.update_vao_vertices(vao_id, &vertices, VertexUsageHint::Dynamic);

                            self.device.draw_triangles_u16(0, indices.len() as gl::GLint);
                            render_context.draw_calls += 1;

                            self.device.delete_vao(vao_id);
                        }
                    }
                }
            }

            //println!("draw_calls {}", render_context.draw_calls);
        }
    }
}
