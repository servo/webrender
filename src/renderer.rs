use device::{Device, ProgramId, UniformLocation};
use euclid::{Rect, Matrix4, Point2D, Size2D};
use gleam::gl;
use internal_types::{ApiMsg, Frame, ResultMsg, TextureUpdateOp, TextureUpdateList};
use internal_types::{TextureUpdateDetails, PackedVertex};
use internal_types::{ORTHO_NEAR_PLANE, ORTHO_FAR_PLANE, RenderBatch, VertexFormat};
use render_api::RenderApi;
use render_backend::RenderBackend;
use std::sync::mpsc::{channel, Sender, Receiver};
use std::thread;
use texture_cache::TextureCache;
use types::{ColorF, Epoch, PipelineId, RenderNotifier, ImageID, ImageFormat};
//use util;

pub struct Renderer {
    api_tx: Sender<ApiMsg>,
    result_rx: Receiver<ResultMsg>,
    device: Device,
    pending_texture_updates: Vec<TextureUpdateList>,
    current_frame: Option<Frame>,

    border_program_id: ProgramId,
    u_border_radii: UniformLocation,
    u_border_position: UniformLocation,
}

impl Renderer {
    pub fn new(notifier: Box<RenderNotifier>,
               width: u32,
               height: u32,
               resource_path: String) -> Renderer {
        let (api_tx, api_rx) = channel();
        let (result_tx, result_rx) = channel();

        let initial_viewport = Rect::new(Point2D::zero(), Size2D::new(width as i32, height as i32));

        let mut device = Device::new(resource_path);
        device.begin_frame();

        let quad_program_id = device.create_program("quad.vs.glsl", "quad.fs.glsl");
        let glyph_program_id = device.create_program("glyph.vs.glsl", "glyph.fs.glsl");
        let border_program_id = device.create_program("border.vs.glsl", "border.fs.glsl");

        let u_border_radii = device.get_uniform_location(border_program_id, "uRadii");
        let u_border_position = device.get_uniform_location(border_program_id, "uPosition");

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
        texture_cache.insert(white_image_id, 0, 0, 2, 2, ImageFormat::RGB8, white_pixels);

        let dummy_mask_image_id = ImageID::new();
        texture_cache.insert(dummy_mask_image_id, 0, 0, 2, 2, ImageFormat::A8, mask_pixels);

        device.end_frame();

        thread::spawn(move || {
            let mut backend = RenderBackend::new(api_rx,
                                                 result_tx,
                                                 initial_viewport,
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
            pending_texture_updates: Vec::new(),
            border_program_id: border_program_id,
            u_border_radii: u_border_radii,
            u_border_position: u_border_position,
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
        self.draw_frame();
        self.device.end_frame();
    }

    fn update_texture_cache(&mut self) {
        for update_list in self.pending_texture_updates.drain(..) {
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
                    TextureUpdateOp::FreeRenderTarget(id) => {
                        self.device.free_texture(id);
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
                            TextureUpdateDetails::BorderRadius(outer_rx, outer_ry, inner_rx, inner_ry) => {
                                let x = x as f32;
                                let y = y as f32;
                                let inner_rx = inner_rx.to_f32_px();
                                let inner_ry = inner_ry.to_f32_px();
                                let outer_rx = outer_rx.to_f32_px();
                                let outer_ry = outer_ry.to_f32_px();

                                // TODO: Render jobs could also be batched.
                                gl::disable(gl::BLEND);
                                gl::disable(gl::DEPTH_TEST);

                                let (texture_width, texture_height) =
                                    self.device.get_texture_dimensions(update.id);

                                let projection = Matrix4::ortho(0.0,
                                                                texture_width as f32,
                                                                0.0,
                                                                texture_height as f32,
                                                                ORTHO_NEAR_PLANE,
                                                                ORTHO_FAR_PLANE);

                                self.device.bind_render_target(Some(update.id));
                                gl::viewport(0, 0, texture_width as gl::GLint, texture_height as gl::GLint);
                                self.device.bind_program(self.border_program_id,
                                                         &projection);
                                self.device.set_uniform_4f(self.u_border_radii, outer_rx, outer_ry, inner_rx, inner_ry);
                                self.device.set_uniform_4f(self.u_border_position, x + outer_rx, y + outer_ry, 0.0, 0.0);

                                let vao_id = self.device.create_vao(VertexFormat::Default);
                                self.device.bind_vao(vao_id);

                                let color0 = ColorF::new(1.0, 0.0, 0.0, 1.0);
                                let color1 = ColorF::new(0.0, 1.0, 0.0, 1.0);
                                let color2 = ColorF::new(0.0, 0.0, 1.0, 1.0);
                                let color3 = ColorF::new(1.0, 1.0, 1.0, 1.0);

                                let indices: [u16; 6] = [ 0, 1, 2, 2, 3, 1 ];
                                let vertices: [PackedVertex; 4] = [
                                    PackedVertex::from_components(x, y, &color0),
                                    PackedVertex::from_components(x + outer_rx, y, &color1),
                                    PackedVertex::from_components(x, y + outer_ry, &color2),
                                    PackedVertex::from_components(x + outer_rx, y + outer_ry, &color3),
                                ];
                                self.device.update_vao_indices(vao_id, &indices);
                                self.device.update_vao_vertices(vao_id, &vertices);

                                self.device.draw_triangles_u16(indices.len() as gl::GLint);

                                self.device.delete_vao(vao_id);
                            }
                        }
                    }
                }
            }
        }
    }

    fn draw_frame(&mut self) {
        if let Some(ref frame) = self.current_frame {
            let mut draw_calls = 0;

            for layer in &frame.layers {
                let projection = Matrix4::ortho(0.0,
                                                layer.size.width as f32,
                                                layer.size.height as f32,
                                                0.0,
                                                ORTHO_NEAR_PLANE,
                                                ORTHO_FAR_PLANE);

                self.device.bind_render_target(layer.texture_id);
                gl::viewport(0, 0, layer.size.width as gl::GLint, layer.size.height as gl::GLint);
                gl::enable(gl::DEPTH_TEST);
                gl::blend_func(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
                gl::depth_mask(true);

                // Clear frame buffer
                gl::clear_color(1.0, 1.0, 1.0, 1.0);
                gl::clear(gl::COLOR_BUFFER_BIT |
                          gl::DEPTH_BUFFER_BIT |
                          gl::STENCIL_BUFFER_BIT);

                // standard opaque pass!
                // TODO: probably worth sorting front to back to minimize overdraw (if profiling shows fragment / rop bound)
                gl::disable(gl::BLEND);

                for batch in &layer.opaque_batches {
                    Renderer::draw_batch(&mut self.device, batch, &projection, &mut draw_calls);
                }

                // alpha pass!
                gl::enable(gl::BLEND);
                gl::depth_mask(false);

                for batch in &layer.alpha_batches {
                    Renderer::draw_batch(&mut self.device, batch, &projection, &mut draw_calls);
                }
            }

            //println!("draw_calls {}", draw_calls);
        }
    }

    fn draw_batch(device: &mut Device, batch: &RenderBatch, projection: &Matrix4, draw_calls: &mut usize) {
        device.bind_color_texture(batch.color_texture_id);
        device.bind_mask_texture(batch.mask_texture_id);
        device.bind_program(batch.program_id, &projection);

        let vao_id = device.create_vao(VertexFormat::Default);
        device.bind_vao(vao_id);

        //println!("    draw_batch {:?} {:?}", batch.color_texture_id, batch.vertices);

        device.update_vao_indices(vao_id, &batch.indices);
        device.update_vao_vertices(vao_id, &batch.vertices);

        device.draw_triangles_u16(batch.indices.len() as gl::GLint);
        *draw_calls += 1;

        device.delete_vao(vao_id);
    }
}
