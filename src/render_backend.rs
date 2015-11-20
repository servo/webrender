use euclid::{Rect, Size2D};
use frame::Frame;
use internal_types::{FontTemplate, FrameRenderTarget, ResultMsg, DrawLayer, RendererFrame};
use ipc_channel::ipc::IpcReceiver;
use profiler::BackendProfileCounters;
use resource_cache::ResourceCache;
use scene::Scene;
use scoped_threadpool;
use std::mem;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::Sender;
use texture_cache::{TextureCache, TextureCacheItemId};
use webrender_traits::{ApiMsg, IdNamespace, RenderNotifier, ScrollLayerId};
use webrender_traits::{WebGLContextId, WebGLCommand};
use batch::new_id;
use device::TextureId;
use offscreen_gl_context::{NativeGLContext, GLContext, ColorAttachmentType, NativeGLContextMethods, NativeGLContextHandle};
use gleam::gl;

pub struct RenderBackend {
    api_rx: IpcReceiver<ApiMsg>,
    result_tx: Sender<ResultMsg>,

    viewport: Rect<i32>,
    device_pixel_ratio: f32,
    next_namespace_id: IdNamespace,

    thread_pool: scoped_threadpool::Pool,
    resource_cache: ResourceCache,

    scene: Scene,
    frame: Frame,

    notifier: Arc<Mutex<Option<Box<RenderNotifier>>>>,
    webrender_context_handle: Option<NativeGLContextHandle>,
    webgl_contexts: HashMap<WebGLContextId, GLContext<NativeGLContext>>,
    pending_webgl_commands: Vec<(WebGLContextId, WebGLCommand)>,
}

impl RenderBackend {
    pub fn new(api_rx: IpcReceiver<ApiMsg>,
               result_tx: Sender<ResultMsg>,
               viewport: Rect<i32>,
               device_pixel_ratio: f32,
               white_image_id: TextureCacheItemId,
               dummy_mask_image_id: TextureCacheItemId,
               texture_cache: TextureCache,
               enable_aa: bool,
               notifier: Arc<Mutex<Option<Box<RenderNotifier>>>>,
               webrender_context_handle: Option<NativeGLContextHandle>) -> RenderBackend {
        let mut thread_pool = scoped_threadpool::Pool::new(8);

        let resource_cache = ResourceCache::new(&mut thread_pool,
                                                texture_cache,
                                                white_image_id,
                                                dummy_mask_image_id,
                                                device_pixel_ratio,
                                                enable_aa);

        let backend = RenderBackend {
            thread_pool: thread_pool,
            api_rx: api_rx,
            result_tx: result_tx,
            viewport: viewport,
            device_pixel_ratio: device_pixel_ratio,
            resource_cache: resource_cache,
            scene: Scene::new(),
            frame: Frame::new(),
            next_namespace_id: IdNamespace(1),
            notifier: notifier,
            webrender_context_handle: webrender_context_handle,
            webgl_contexts: HashMap::new(),
            pending_webgl_commands: Vec::new(),
        };

        backend
    }

    pub fn run(&mut self) {
        let mut profile_counters = BackendProfileCounters::new();

        loop {
            let msg = self.api_rx.recv();

            match msg {
                Ok(msg) => {
                    match msg {
                        ApiMsg::AddRawFont(id, bytes) => {
                            profile_counters.font_templates.inc(bytes.len());
                            self.resource_cache
                                .add_font_template(id, FontTemplate::Raw(Arc::new(bytes)));
                        }
                        ApiMsg::AddNativeFont(id, native_font_handle) => {
                            self.resource_cache
                                .add_font_template(id, FontTemplate::Native(native_font_handle));
                        }
                        ApiMsg::AddImage(id, width, height, format, bytes) => {
                            profile_counters.image_templates.inc(bytes.len());
                            self.resource_cache.add_image_template(id,
                                                                   width,
                                                                   height,
                                                                   format,
                                                                   bytes);
                        }
                        ApiMsg::UpdateImage(id, width, height, format, bytes) => {
                            self.resource_cache.update_image_template(id,
                                                                      width,
                                                                      height,
                                                                      format,
                                                                      bytes);
                        }
                        ApiMsg::AddDisplayList(id,
                                               pipeline_id,
                                               epoch,
                                               display_list_builder) => {
                            self.scene.add_display_list(id,
                                                        pipeline_id,
                                                        epoch,
                                                        display_list_builder,
                                                        &mut self.resource_cache);
                        }
                        ApiMsg::AddStackingContext(id,
                                                   pipeline_id,
                                                   epoch,
                                                   stacking_context) => {
                            self.scene.add_stacking_context(id,
                                                            pipeline_id,
                                                            epoch,
                                                            stacking_context);
                        }
                        ApiMsg::CloneApi(sender) => {
                            let result = self.next_namespace_id;

                            let IdNamespace(id_namespace) = self.next_namespace_id;
                            self.next_namespace_id = IdNamespace(id_namespace + 1);

                            sender.send(result).unwrap();
                        }
                        ApiMsg::SetRootStackingContext(stacking_context_id,
                                                       background_color,
                                                       epoch,
                                                       pipeline_id) => {
                            let frame = profile_counters.total_time.profile(|| {
                                self.scene.set_root_stacking_context(pipeline_id,
                                                                     epoch,
                                                                     stacking_context_id,
                                                                     background_color,
                                                                     &mut self.resource_cache);

                                self.build_scene();
                                self.render()
                            });

                            self.publish_frame(frame, &mut profile_counters);
                        }
                        ApiMsg::SetRootPipeline(pipeline_id) => {
                            let frame = profile_counters.total_time.profile(|| {
                                self.scene.set_root_pipeline_id(pipeline_id);

                                self.build_scene();
                                self.render()
                            });

                            self.publish_frame(frame, &mut profile_counters);
                        }
                        ApiMsg::Scroll(delta) => {
                            let frame = profile_counters.total_time.profile(|| {
                                let viewport_size = Size2D::new(self.viewport.size.width as f32,
                                                                self.viewport.size.height as f32);
                                self.frame.scroll(&delta, &viewport_size);
                                self.render()
                            });

                            self.publish_frame(frame, &mut profile_counters);
                        }
                        ApiMsg::TranslatePointToLayerSpace(point, tx) => {
                            // TODO(pcwalton): Select other layers for mouse events.
                            let point = point / self.device_pixel_ratio;
                            match self.frame.layers.get_mut(&ScrollLayerId(0)) {
                                None => tx.send(point).unwrap(),
                                Some(layer) => tx.send(point - layer.scroll_offset).unwrap(),
                            }
                        }
                        ApiMsg::RequestWebGLContext(size, attributes, tx) => {
                            if let Some(ref handle) = self.webrender_context_handle {
                                match GLContext::<NativeGLContext>::new(size, attributes, ColorAttachmentType::Texture, Some(handle)) {
                                    Ok(ctx) => {
                                        let id = WebGLContextId(new_id());

                                        let (real_size, texture_id) = {
                                            let draw_buffer = ctx.borrow_draw_buffer().unwrap();
                                            (draw_buffer.size(), draw_buffer.get_bound_texture_id().unwrap())
                                        };

                                        self.webgl_contexts.insert(id, ctx);

                                        self.resource_cache
                                            .add_webgl_texture(id, TextureId(texture_id), real_size);

                                        tx.send(Ok(id)).unwrap();
                                    },
                                    Err(msg) => {
                                        tx.send(Err(msg.to_owned())).unwrap();
                                    }
                                }
                            } else {
                                tx.send(Err("Not implemented yet".to_owned())).unwrap();
                            }
                        }
                        ApiMsg::WebGLCommand(context_id, command) => {
                            self.pending_webgl_commands.push((context_id, command));
                        }
                    }
                }
                Err(..) => {
                    break;
                }
            }
        }
    }

    fn apply_webgl_command(&self, command: WebGLCommand) {
        match command {
            WebGLCommand::Clear(what)
                => gl::clear(what),
            WebGLCommand::ClearColor(r, g, b, a)
                => gl::clear_color(r, g, b, a),
        }
    }

    fn apply_webgl_commands(&mut self) {
        if self.pending_webgl_commands.is_empty() {
            return; // Early return in easy case
        }

        let mut current_context_id = None;
        let mut current_context: Option<&GLContext<NativeGLContext>> = None;

        // Sort by context id to minimize context switching
        self.pending_webgl_commands.sort_by(|a, b| a.0.cmp(&b.0));

        let mut pending = mem::replace(&mut self.pending_webgl_commands, Vec::new());

        for (context_id, command) in pending.drain(..) {
            if current_context_id != Some(context_id) {
                current_context_id = Some(context_id);
                current_context = Some(self.webgl_contexts.get(&context_id).unwrap());
                current_context.unwrap().make_current().unwrap();
            }

            self.apply_webgl_command(command);
        }

        // Without this the changes in the texture are not seen in the renderer.
        if let Some(ctx) = current_context {
            ctx.unbind().unwrap();
        }
    }


    fn build_scene(&mut self) {
        // Flatten the stacking context hierarchy
        let mut new_pipeline_sizes = HashMap::new();

        self.frame.create(&self.scene,
                          Size2D::new(self.viewport.size.width as u32,
                                      self.viewport.size.height as u32),
                          self.device_pixel_ratio,
                          &mut self.resource_cache,
                          &mut new_pipeline_sizes);

        let mut updated_pipeline_sizes = HashMap::new();

        for (pipeline_id, old_size) in self.scene.pipeline_sizes.drain() {
            let new_size = new_pipeline_sizes.remove(&pipeline_id);

            match new_size {
                Some(new_size) => {
                    // Exists in both old and new -> check if size changed
                    if new_size != old_size {
                        let mut notifier = self.notifier.lock();
                        notifier.as_mut()
                                .unwrap()
                                .as_mut()
                                .unwrap()
                                .pipeline_size_changed(pipeline_id, Some(new_size));
                    }

                    // Re-insert
                    updated_pipeline_sizes.insert(pipeline_id, new_size);
                }
                None => {
                    // Was existing, not in current frame anymore
                        let mut notifier = self.notifier.lock();
                        notifier.as_mut()
                                .unwrap()
                                .as_mut()
                                .unwrap()
                                .pipeline_size_changed(pipeline_id, None);
                }
            }
        }

        // Any remaining items are new pipelines
        for (pipeline_id, new_size) in new_pipeline_sizes.drain() {
            let mut notifier = self.notifier.lock();
            notifier.as_mut()
                    .unwrap()
                    .as_mut()
                    .unwrap()
                    .pipeline_size_changed(pipeline_id, Some(new_size));
            updated_pipeline_sizes.insert(pipeline_id, new_size);
        }

        self.scene.pipeline_sizes = updated_pipeline_sizes;
    }

    fn render(&mut self) -> RendererFrame {
        self.apply_webgl_commands();

        let mut frame = self.frame.build(&self.viewport,
                                         &mut self.resource_cache,
                                         &mut self.thread_pool,
                                         self.device_pixel_ratio);

        // Bit of a hack - if there was nothing visible, at least
        // add one layer to the frame so that the screen gets
        // cleared to the default UA background color. Perhaps
        // there is a better way to handle this...
        if frame.layers.len() == 0 {
            let size = Size2D::new(self.viewport.size.width as u32,
                                   self.viewport.size.height as u32);
            frame.layers.push(DrawLayer {
                render_targets: vec![FrameRenderTarget::new(size, None)],
                texture_id: None,
                size: size,
                commands: Vec::new(),
            });
        }

        let pending_update = self.resource_cache.pending_updates();
        if pending_update.updates.len() > 0 {
            self.result_tx.send(ResultMsg::UpdateTextureCache(pending_update)).unwrap();
        }

        let pending_update = self.frame.pending_updates();
        if pending_update.updates.len() > 0 {
            self.result_tx.send(ResultMsg::UpdateBatches(pending_update)).unwrap();
        }

        frame
    }

    fn publish_frame(&mut self,
                     frame: RendererFrame,
                     profile_counters: &mut BackendProfileCounters) {
        let msg = ResultMsg::NewFrame(frame, profile_counters.clone());
        self.result_tx.send(msg).unwrap();
        profile_counters.reset();

        // TODO(gw): This is kindof bogus to have to lock the notifier
        //           each time it's used. This is due to some nastiness
        //           in initialization order for Servo. Perhaps find a
        //           cleaner way to do this, or use the OnceMutex on crates.io?
        let mut notifier = self.notifier.lock();
        notifier.as_mut().unwrap().as_mut().unwrap().new_frame_ready();
    }
}

