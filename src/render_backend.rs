/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use frame::Frame;
use internal_types::{FontTemplate, ResultMsg, RendererFrame};
use ipc_channel::ipc::IpcReceiver;
use profiler::BackendProfileCounters;
use resource_cache::ResourceCache;
use scene::Scene;
use scoped_threadpool;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::Sender;
use texture_cache::{TextureCache, TextureCacheItemId};
use webrender_traits::{ApiMsg, IdNamespace, RenderNotifier};
use webrender_traits::{WebGLContextId, ScrollLayerId};
use batch::new_id;
use device::TextureId;
use offscreen_gl_context::{NativeGLContext, GLContext, ColorAttachmentType, NativeGLContextMethods, NativeGLContextHandle};

pub struct RenderBackend {
    api_rx: IpcReceiver<ApiMsg>,
    result_tx: Sender<ResultMsg>,

    device_pixel_ratio: f32,
    next_namespace_id: IdNamespace,

    thread_pool: scoped_threadpool::Pool,
    resource_cache: ResourceCache,

    scene: Scene,
    frame: Frame,

    notifier: Arc<Mutex<Option<Box<RenderNotifier>>>>,
    webrender_context_handle: Option<NativeGLContextHandle>,
    webgl_contexts: HashMap<WebGLContextId, GLContext<NativeGLContext>>,
    current_bound_webgl_context_id: Option<WebGLContextId>,
}

impl RenderBackend {
    pub fn new(api_rx: IpcReceiver<ApiMsg>,
               result_tx: Sender<ResultMsg>,
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

        RenderBackend {
            thread_pool: thread_pool,
            api_rx: api_rx,
            result_tx: result_tx,
            device_pixel_ratio: device_pixel_ratio,
            resource_cache: resource_cache,
            scene: Scene::new(),
            frame: Frame::new(),
            next_namespace_id: IdNamespace(1),
            notifier: notifier,
            webrender_context_handle: webrender_context_handle,
            webgl_contexts: HashMap::new(),
            current_bound_webgl_context_id: None,
        }
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
                        ApiMsg::AddDisplayList(id, pipeline_id, epoch, built_display_list) => {
                            self.scene.add_display_list(id,
                                                        pipeline_id,
                                                        epoch,
                                                        built_display_list,
                                                        &mut self.resource_cache);
                        }
                        ApiMsg::AddStackingContext(id, pipeline_id, epoch, stacking_context) => {
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
                                                       pipeline_id,
                                                       viewport_size,
                                                       auxiliary_lists) => {
                            let frame = profile_counters.total_time.profile(|| {
                                self.scene.set_root_stacking_context(pipeline_id,
                                                                     epoch,
                                                                     stacking_context_id,
                                                                     background_color,
                                                                     viewport_size,
                                                                     &mut self.resource_cache,
                                                                     auxiliary_lists);

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
                        ApiMsg::Scroll(delta, cursor, move_phase) => {
                            let frame = profile_counters.total_time.profile(|| {
                                self.frame.scroll(delta, cursor, move_phase);
                                self.render()
                            });

                            self.publish_frame(frame, &mut profile_counters);
                        }
                        ApiMsg::TickScrollingBounce => {
                            let frame = profile_counters.total_time.profile(|| {
                                self.frame.tick_scrolling_bounce_animations();
                                self.render()
                            });

                            self.publish_frame(frame, &mut profile_counters);
                        }
                        ApiMsg::TranslatePointToLayerSpace(point, tx) => {
                            // TODO(pcwalton): Select other layers for mouse events.
                            let point = point / self.device_pixel_ratio;
                            match self.scene.root_pipeline_id {
                                Some(root_pipeline_id) => {
                                    match self.frame.layers.get_mut(&ScrollLayerId::new(root_pipeline_id, 0)) {
                                        None => tx.send(point).unwrap(),
                                        Some(layer) => {
                                            tx.send(point - layer.scrolling.offset).unwrap()
                                        }
                                    }
                                }
                                None => {
                                    tx.send(point).unwrap()
                                }
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
                            // TODO: Buffer the commands and only apply them here if they need to
                            // be synchronous.
                            let ctx = self.webgl_contexts.get(&context_id).unwrap();
                            ctx.make_current().unwrap();
                            command.apply(ctx);
                            self.current_bound_webgl_context_id = Some(context_id);
                        }
                    }
                }
                Err(..) => {
                    break;
                }
            }
        }
    }

    fn build_scene(&mut self) {
        // Flatten the stacking context hierarchy
        let mut new_pipeline_sizes = HashMap::new();

        if let Some(id) = self.current_bound_webgl_context_id {
            self.webgl_contexts.get(&id).unwrap().unbind().unwrap();
        }

        self.frame.create(&self.scene,
                          &mut self.resource_cache,
                          &mut new_pipeline_sizes,
                          self.device_pixel_ratio);

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
        let frame = self.frame.build(&mut self.resource_cache,
                                     &mut self.thread_pool,
                                     self.device_pixel_ratio);

        let pending_update = self.resource_cache.pending_updates();
        if !pending_update.updates.is_empty() {
            self.result_tx.send(ResultMsg::UpdateTextureCache(pending_update)).unwrap();
        }

        frame
    }

    fn publish_frame(&mut self,
                     frame: RendererFrame,
                     profile_counters: &mut BackendProfileCounters) {
        let pending_updates = self.frame.pending_updates();
        let msg = ResultMsg::NewFrame(frame, pending_updates, profile_counters.clone());
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

