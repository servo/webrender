use app_units::Au;
use device::{Device, ProgramId, TextureId, UniformLocation, VAOId, VertexUsageHint};
use euclid::{Rect, Matrix4, Point2D, Size2D};
use gleam::gl;
use internal_types::{ApiMsg, Frame, ResultMsg, TextureUpdateOp, BatchUpdateOp, BatchUpdateList};
use internal_types::{TextureUpdateDetails, TextureUpdateList, PackedVertex, RenderTargetMode, BatchId};
use internal_types::{ORTHO_NEAR_PLANE, ORTHO_FAR_PLANE, DrawCommandInfo};
use render_api::RenderApi;
use render_backend::RenderBackend;
use std::collections::HashMap;
use std::f32;
use std::mem;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Sender, Receiver};
use std::thread;
use texture_cache::{TextureCache, TextureInsertOp};
use types::{ColorF, Epoch, PipelineId, RenderNotifier, ImageID, ImageFormat, MixBlendMode};
use types::{CompositionOp, LowLevelFilterOp, BlurDirection};
//use util;

pub const BLUR_INFLATION_FACTOR: u32 = 3;

static RECTANGLE_INDICES: [u16; 6] = [0, 1, 2, 2, 3, 1];

struct Batch {
    program_id: ProgramId,
    color_texture_id: TextureId,
    mask_texture_id: TextureId,
    vao_id: VAOId,
    index_count: gl::GLint,
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
    api_tx: Sender<ApiMsg>,
    result_rx: Receiver<ResultMsg>,
    device: Device,
    pending_texture_updates: Vec<TextureUpdateList>,
    pending_batch_updates: Vec<BatchUpdateList>,
    current_frame: Option<Frame>,
    device_pixel_ratio: f32,
    batches: HashMap<BatchId, Batch>,
    batch_matrices: HashMap<BatchId, Vec<Matrix4>>,

    quad_program_id: ProgramId,
    u_quad_transform_array: UniformLocation,

    glyph_program_id: ProgramId,
    u_glyph_transform_array: UniformLocation,

    border_program_id: ProgramId,
    u_border_radii: UniformLocation,
    u_border_position: UniformLocation,

    blend_program_id: ProgramId,
    u_blend_params: UniformLocation,

    filter_program_id: ProgramId,
    u_filter_params: UniformLocation,
    u_filter_texture_size: UniformLocation,

    box_shadow_corner_program_id: ProgramId,
    u_box_shadow_corner_position: UniformLocation,
    u_box_shadow_blur_radius: UniformLocation,
    u_arc_radius: UniformLocation,

    blur_program_id: ProgramId,
    u_blur_blur_radius: UniformLocation,
    u_dest_texture_size: UniformLocation,
    u_direction: UniformLocation,
    u_source_texture_size: UniformLocation,
}

impl Renderer {
    pub fn new(notifier: Box<RenderNotifier>,
               width: u32,
               height: u32,
               device_pixel_ratio: f32,
               resource_path: PathBuf) -> Renderer {
        let (api_tx, api_rx) = channel();
        let (result_tx, result_rx) = channel();

        let initial_viewport = Rect::new(Point2D::zero(), Size2D::new(width as i32, height as i32));

        let mut device = Device::new(resource_path);
        device.begin_frame();

        let quad_program_id = device.create_program("quad.vs.glsl", "quad.fs.glsl");
        let glyph_program_id = device.create_program("glyph.vs.glsl", "glyph.fs.glsl");
        let border_program_id = device.create_program("border.vs.glsl", "border.fs.glsl");
        let blend_program_id = device.create_program("blend.vs.glsl", "blend.fs.glsl");
        let filter_program_id = device.create_program("filter.vs.glsl", "filter.fs.glsl");
        let box_shadow_corner_program_id = device.create_program("box-shadow-corner.vs.glsl",
                                                                 "box-shadow-corner.fs.glsl");
        let blur_program_id = device.create_program("blur.vs.glsl", "blur.fs.glsl");

        let u_quad_transform_array = device.get_uniform_location(quad_program_id, "uMatrixPalette");

        let u_glyph_transform_array = device.get_uniform_location(glyph_program_id, "uMatrixPalette");

        let u_border_radii = device.get_uniform_location(border_program_id, "uRadii");
        let u_border_position = device.get_uniform_location(border_program_id, "uPosition");

        let u_blend_params = device.get_uniform_location(blend_program_id, "uBlendParams");

        let u_filter_params = device.get_uniform_location(filter_program_id, "uFilterParams");
        let u_filter_texture_size = device.get_uniform_location(filter_program_id, "uTextureSize");

        let u_box_shadow_corner_position =
            device.get_uniform_location(box_shadow_corner_program_id, "uPosition");
        let u_box_shadow_blur_radius = device.get_uniform_location(box_shadow_corner_program_id,
                                                                   "uBlurRadius");
        let u_arc_radius = device.get_uniform_location(box_shadow_corner_program_id,
                                                       "uArcRadius");

        let u_blur_blur_radius = device.get_uniform_location(blur_program_id, "uBlurRadius");
        let u_dest_texture_size = device.get_uniform_location(blur_program_id, "uDestTextureSize");
        let u_direction = device.get_uniform_location(blur_program_id, "uDirection");
        let u_source_texture_size = device.get_uniform_location(blur_program_id,
                                                                "uSourceTextureSize");

        let texture_ids = device.create_texture_ids(1024);
        let mut texture_cache = TextureCache::new(texture_ids);
        let white_pixels: Vec<u8> = vec![
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff,
        ];
        let mask_pixels: Vec<u8> = vec![
            0xff, 0xff,
            0xff, 0xff,
        ];
        // TODO: Ensure that the white texture can never get evicted when the cache supports LRU eviction!
        let white_image_id = ImageID::new();
        texture_cache.insert(white_image_id,
                             0,
                             0,
                             2,
                             2,
                             ImageFormat::RGB8,
                             TextureInsertOp::Blit(white_pixels));

        let dummy_mask_image_id = ImageID::new();
        texture_cache.insert(dummy_mask_image_id,
                             0,
                             0,
                             2,
                             2,
                             ImageFormat::A8,
                             TextureInsertOp::Blit(mask_pixels));

        device.end_frame();

        thread::spawn(move || {
            let mut backend = RenderBackend::new(api_rx,
                                                 result_tx,
                                                 initial_viewport,
                                                 device_pixel_ratio,
                                                 quad_program_id,
                                                 glyph_program_id,
                                                 white_image_id,
                                                 dummy_mask_image_id,
                                                 texture_cache);
            backend.run(notifier);
        });

        Renderer {
            api_tx: api_tx,
            result_rx: result_rx,
            device: device,
            current_frame: None,
            batches: HashMap::new(),
            batch_matrices: HashMap::new(),
            pending_texture_updates: Vec::new(),
            pending_batch_updates: Vec::new(),
            border_program_id: border_program_id,
            device_pixel_ratio: device_pixel_ratio,
            blend_program_id: blend_program_id,
            filter_program_id: filter_program_id,
            quad_program_id: quad_program_id,
            glyph_program_id: glyph_program_id,
            box_shadow_corner_program_id: box_shadow_corner_program_id,
            blur_program_id: blur_program_id,
            u_border_radii: u_border_radii,
            u_border_position: u_border_position,
            u_blend_params: u_blend_params,
            u_filter_params: u_filter_params,
            u_filter_texture_size: u_filter_texture_size,
            u_box_shadow_corner_position: u_box_shadow_corner_position,
            u_box_shadow_blur_radius: u_box_shadow_blur_radius,
            u_arc_radius: u_arc_radius,
            u_blur_blur_radius: u_blur_blur_radius,
            u_dest_texture_size: u_dest_texture_size,
            u_source_texture_size: u_source_texture_size,
            u_direction: u_direction,
            u_quad_transform_array: u_quad_transform_array,
            u_glyph_transform_array: u_glyph_transform_array,
        }
    }

    pub fn new_api(&self) -> RenderApi {
        RenderApi {
            tx: self.api_tx.clone()
        }
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
                    BatchUpdateOp::Create(vertices,
                                          indices,
                                          program_id,
                                          color_texture_id,
                                          mask_texture_id) => {
                        let vao_id = self.device.create_vao();
                        self.device.bind_vao(vao_id);

                        self.device.update_vao_indices(vao_id, &indices, VertexUsageHint::Static);
                        self.device.update_vao_vertices(vao_id, &vertices, VertexUsageHint::Static);

                        self.batches.insert(update.id, Batch {
                            vao_id: vao_id,
                            program_id: program_id,
                            color_texture_id: color_texture_id,
                            mask_texture_id: mask_texture_id,
                            index_count: indices.len() as gl::GLint,
                        });
                    }
                    BatchUpdateOp::UpdateUniforms(matrices) => {
                        self.batch_matrices.insert(update.id, matrices);
                    }
                    BatchUpdateOp::Destroy => {
                        let batch = self.batches.remove(&update.id).unwrap();
                        self.device.delete_vao(batch.vao_id);
                        /*for (_, batch) in self.batches.iter() {
                            self.device.delete_vao(batch.vao_id);
                        }
                        self.batches.clear();*/
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
                    TextureUpdateOp::Create(width, height, format, mode, maybe_bytes) => {
                        // TODO: clean up match
                        match maybe_bytes {
                            Some(bytes) => {
                                self.device.init_texture(update.id,
                                                         width,
                                                         height,
                                                         format,
                                                         mode,
                                                         Some(bytes.as_slice()));
                            }
                            None => {
                                self.device.init_texture(update.id,
                                                         width,
                                                         height,
                                                         format,
                                                         mode,
                                                         None);
                            }
                        }
                    }
                    TextureUpdateOp::DeinitRenderTarget(id) => {
                        self.device.deinit_texture(id);
                    }
                    TextureUpdateOp::Update(x, y, width, height, details) => {
                        match details {
                            TextureUpdateDetails::Blit(bytes) => {
                                self.device.update_texture(update.id,
                                                           x,
                                                           y,
                                                           width,
                                                           height,
                                                           bytes.as_slice());
                            }
                            TextureUpdateDetails::Blur(bytes,
                                                       glyph_size,
                                                       radius,
                                                       unblurred_glyph_texture_id,
                                                       horizontal_blur_texture_id) => {
                                let radius =
                                    f32::ceil(radius.to_f32_px() * self.device_pixel_ratio) as u32;
                                self.device.init_texture(unblurred_glyph_texture_id,
                                                         glyph_size.width,
                                                         glyph_size.height,
                                                         ImageFormat::A8,
                                                         RenderTargetMode::None,
                                                         Some(bytes.as_slice()));
                                self.device.init_texture(horizontal_blur_texture_id,
                                                         width,
                                                         height,
                                                         ImageFormat::A8,
                                                         RenderTargetMode::RenderTarget,
                                                         None);

                                let blur_program_id = self.blur_program_id;
                                self.set_up_gl_state_for_texture_cache_update(
                                    horizontal_blur_texture_id,
                                    blur_program_id);

                                let white = ColorF::new(1.0, 1.0, 1.0, 1.0);
                                let (width, height) = (width as f32, height as f32);
                                self.device.bind_mask_texture(unblurred_glyph_texture_id);
                                self.device.set_uniform_2f(self.u_direction, 1.0, 0.0);

                                // FIXME(pcwalton): This is going to interfere pretty bad with
                                // our batching if we have lots of heterogeneous border radii.
                                // Maybe we should make these varyings instead.
                                self.device.set_uniform_1f(self.u_blur_blur_radius, radius as f32);
                                self.device.set_uniform_2f(self.u_dest_texture_size,
                                                           width as f32,
                                                           height as f32);
                                self.device.set_uniform_2f(self.u_source_texture_size,
                                                           glyph_size.width as f32,
                                                           glyph_size.height as f32);

                                let vertices = [
                                    PackedVertex::from_components(0.0, 0.0,
                                                                  &white,
                                                                  0.0, 0.0, 0.0, 0.0),
                                    PackedVertex::from_components(width, 0.0,
                                                                  &white,
                                                                  0.0, 0.0, 1.0, 0.0),
                                    PackedVertex::from_components(0.0, height,
                                                                  &white,
                                                                  0.0, 0.0, 0.0, 1.0),
                                    PackedVertex::from_components(width, height,
                                                                  &white,
                                                                  0.0, 0.0, 1.0, 1.0),
                                ];

                                self.perform_gl_texture_cache_update(&RECTANGLE_INDICES,
                                                                     &vertices);

                                self.device.deinit_texture(unblurred_glyph_texture_id);

                                self.set_up_gl_state_for_texture_cache_update(
                                    update.id,
                                    blur_program_id);
                                self.device.bind_mask_texture(horizontal_blur_texture_id);
                                self.device.set_uniform_1f(self.u_blur_blur_radius, radius as f32);
                                self.device.set_uniform_2f(self.u_source_texture_size,
                                                           width as f32,
                                                           height as f32);
                                self.device.set_uniform_2f(self.u_dest_texture_size,
                                                           width as f32,
                                                           height as f32);
                                self.device.set_uniform_2f(self.u_direction, 0.0, 1.0);

                                let (x, y) = (x as f32, y as f32);
                                let (max_x, max_y) = (x + width, y + height);
                                let vertices = [
                                    PackedVertex::from_components(x, y,
                                                                  &white,
                                                                  0.0, 0.0, 0.0, 0.0),
                                    PackedVertex::from_components(max_x, y,
                                                                  &white,
                                                                  1.0, 0.0, 1.0, 0.0),
                                    PackedVertex::from_components(x, max_y,
                                                                  &white,
                                                                  0.0, 1.0, 0.0, 1.0),
                                    PackedVertex::from_components(max_x, max_y,
                                                                  &white,
                                                                  1.0, 1.0, 1.0, 1.0),
                                ];
                                self.perform_gl_texture_cache_update(&RECTANGLE_INDICES,
                                                                     &vertices);

                                self.device.deinit_texture(horizontal_blur_texture_id);
                            }
                            TextureUpdateDetails::BorderRadius(outer_rx, outer_ry, inner_rx, inner_ry) => {
                                let x = x as f32;
                                let y = y as f32;
                                let inner_rx = inner_rx.to_f32_px();
                                let inner_ry = inner_ry.to_f32_px();
                                let outer_rx = outer_rx.to_f32_px();
                                let outer_ry = outer_ry.to_f32_px();

                                let border_program_id = self.border_program_id;
                                self.set_up_gl_state_for_texture_cache_update(
                                    update.id,
                                    border_program_id);

                                self.device.set_uniform_4f(self.u_border_radii,
                                                           outer_rx,
                                                           outer_ry,
                                                           inner_rx,
                                                           inner_ry);
                                self.device.set_uniform_4f(self.u_border_position,
                                                           x + outer_rx,
                                                           y + outer_ry,
                                                           0.0,
                                                           0.0);

                                let color0 = ColorF::new(1.0, 0.0, 0.0, 1.0);
                                let color1 = ColorF::new(0.0, 1.0, 0.0, 1.0);
                                let color2 = ColorF::new(0.0, 0.0, 1.0, 1.0);
                                let color3 = ColorF::new(1.0, 1.0, 1.0, 1.0);

                                let indices: [u16; 6] = [ 0, 1, 2, 2, 3, 1 ];
                                let vertices: [PackedVertex; 4] = [
                                    PackedVertex::from_components(x,
                                                                  y,
                                                                  &color0,
                                                                  0.0,
                                                                  0.0,
                                                                  0.0,
                                                                  0.0),
                                    PackedVertex::from_components(x + outer_rx,
                                                                  y,
                                                                  &color1,
                                                                  0.0,
                                                                  0.0,
                                                                  0.0,
                                                                  0.0),
                                    PackedVertex::from_components(x,
                                                                  y + outer_ry,
                                                                  &color2,
                                                                  0.0,
                                                                  0.0,
                                                                  0.0,
                                                                  0.0),
                                    PackedVertex::from_components(x + outer_rx,
                                                                  y + outer_ry,
                                                                  &color3,
                                                                  0.0,
                                                                  0.0,
                                                                  0.0,
                                                                  0.0),
                                ];

                                self.perform_gl_texture_cache_update(&indices, &vertices);
                            }
                            TextureUpdateDetails::BoxShadowCorner(blur_radius, border_radius) => {
                                self.update_texture_cache_for_box_shadow_corner(
                                    update.id,
                                    &Rect::new(Point2D::new(x as f32, y as f32),
                                               Size2D::new(width as f32, height as f32)),
                                    blur_radius,
                                    border_radius)
                            }
                        }
                    }
                }
            }
        }
    }

    fn update_texture_cache_for_box_shadow_corner(&mut self,
                                                  update_id: TextureId,
                                                  rect: &Rect<f32>,
                                                  blur_radius: Au,
                                                  border_radius: Au) {
        let box_shadow_corner_program_id = self.box_shadow_corner_program_id;
        self.set_up_gl_state_for_texture_cache_update(update_id, box_shadow_corner_program_id);

        let blur_radius = blur_radius.to_f32_px();
        let border_radius = border_radius.to_f32_px();
        self.device.set_uniform_4f(self.u_box_shadow_corner_position,
                                   rect.origin.x,
                                   rect.origin.y,
                                   rect.size.width,
                                   rect.size.height);
        self.device.set_uniform_1f(self.u_box_shadow_blur_radius, blur_radius);
        self.device.set_uniform_1f(self.u_arc_radius, border_radius);

        let color = ColorF::new(1.0, 1.0, 1.0, 1.0);
        let vertices: [PackedVertex; 4] = [
            PackedVertex::from_components(rect.origin.x,
                                          rect.origin.y,
                                          &color,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0),
            PackedVertex::from_components(rect.max_x(),
                                          rect.origin.y,
                                          &color,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0),
            PackedVertex::from_components(rect.origin.x,
                                          rect.max_y(),
                                          &color,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0),
            PackedVertex::from_components(rect.max_x(),
                                          rect.max_y(),
                                          &color,
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0),
        ];

        self.perform_gl_texture_cache_update(&RECTANGLE_INDICES, &vertices);
    }

    fn set_up_gl_state_for_texture_cache_update(&mut self,
                                                update_id: TextureId,
                                                program_id: ProgramId) {
        // TODO(gw): Render jobs could also be batched.
        gl::disable(gl::BLEND);
        gl::disable(gl::DEPTH_TEST);

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
    }

    fn perform_gl_texture_cache_update(&mut self,
                                       indices: &[u16],
                                       vertices: &[PackedVertex]) {
        let vao_id = self.device.create_vao();
        self.device.bind_vao(vao_id);

        self.device.update_vao_indices(vao_id, indices, VertexUsageHint::Dynamic);
        self.device.update_vao_vertices(vao_id, vertices, VertexUsageHint::Dynamic);

        self.device.draw_triangles_u16(indices.len() as gl::GLint);
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

            for layer in frame.layers.iter().rev() {
                render_context.layer_size = layer.size;
                render_context.projection = Matrix4::ortho(0.0,
                                                           layer.size.width as f32,
                                                           layer.size.height as f32,
                                                           0.0,
                                                           ORTHO_NEAR_PLANE,
                                                           ORTHO_FAR_PLANE);

                self.device.bind_render_target(layer.texture_id);
                gl::viewport(0,
                             0,
                             (layer.size.width as f32 * self.device_pixel_ratio) as gl::GLint,
                             (layer.size.height as f32 * self.device_pixel_ratio) as gl::GLint);
                gl::disable(gl::DEPTH_TEST);

                // Clear frame buffer
                gl::clear_color(1.0, 1.0, 1.0, 1.0);
                gl::clear(gl::COLOR_BUFFER_BIT);

                for cmd in &layer.commands {
                    match cmd.info {
                        DrawCommandInfo::Batch(batch_id) => {
                            // TODO: probably worth sorting front to back to minimize overdraw (if profiling shows fragment / rop bound)

                            gl::enable(gl::BLEND);
                            gl::blend_func(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
                            gl::blend_equation(gl::FUNC_ADD);

                            let batch = &self.batches[&batch_id];
                            let matrices = &self.batch_matrices[&batch_id];

                            // TODO: hack - bind the uniform locations? this goes away if only one shader anyway...
                            let u_transform_array = if batch.program_id == self.quad_program_id {
                                self.u_quad_transform_array
                            } else if batch.program_id == self.glyph_program_id {
                                self.u_glyph_transform_array
                            } else {
                                panic!("unexpected batch shader!");
                            };

                            Renderer::draw_batch(&mut self.device,
                                                 batch,
                                                 matrices,
                                                 &mut render_context,
                                                 u_transform_array);
                        }
                        DrawCommandInfo::Composite(ref info) => {
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
                                self.device.init_texture(render_context.temporary_fb_texture,
                                                         info.rect.size.width,
                                                         info.rect.size.height,
                                                         ImageFormat::RGBA8,
                                                         RenderTargetMode::None,
                                                         None);
                                self.device.read_framebuffer_rect(render_context.temporary_fb_texture,
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
                                self.device.bind_mask_texture(render_context.temporary_fb_texture);
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

                                self.device.bind_program(self.quad_program_id, &render_context.projection);
                                self.device.set_uniform_mat4_array(self.u_quad_transform_array, &[Matrix4::identity()]);
                            }

                            let color = ColorF::new(1.0, 1.0, 1.0, 1.0);
                            let indices: [u16; 6] = [ 0, 1, 2, 2, 3, 1 ];
                            let vertices: [PackedVertex; 4] = [
                                PackedVertex::from_components(x0 as f32, y0 as f32, &color, 0.0, 1.0, 0.0, 1.0),
                                PackedVertex::from_components(x1 as f32, y0 as f32, &color, 1.0, 1.0, 1.0, 1.0),
                                PackedVertex::from_components(x0 as f32, y1 as f32, &color, 0.0, 0.0, 0.0, 0.0),
                                PackedVertex::from_components(x1 as f32, y1 as f32, &color, 1.0, 0.0, 1.0, 0.0),
                            ];
                            // TODO: Don't re-create this VAO all the time.
                            // Create it once and set positions via uniforms.
                            let vao_id = self.device.create_vao();
                            self.device.bind_color_texture(info.color_texture_id);
                            self.device.bind_vao(vao_id);
                            self.device.update_vao_indices(vao_id, &indices, VertexUsageHint::Dynamic);
                            self.device.update_vao_vertices(vao_id, &vertices, VertexUsageHint::Dynamic);

                            self.device.draw_triangles_u16(indices.len() as gl::GLint);
                            render_context.draw_calls += 1;

                            self.device.delete_vao(vao_id);
                        }
                    }
                }
            }

            //println!("draw_calls {}", render_context.draw_calls);
        }
    }

    fn draw_batch(device: &mut Device,
                  batch: &Batch,
                  matrices: &Vec<Matrix4>,
                  context: &mut RenderContext,
                  u_transform_array: UniformLocation) {
        device.bind_program(batch.program_id, &context.projection);
        device.set_uniform_mat4_array(u_transform_array, matrices);     // The uniform loc here isn't always right!

        device.bind_mask_texture(batch.mask_texture_id);
        device.bind_color_texture(batch.color_texture_id);

        device.bind_vao(batch.vao_id);

        device.draw_triangles_u16(batch.index_count);
        context.draw_calls += 1;
    }
}
