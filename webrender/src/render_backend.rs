/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use gleam::gl;
use frame::Frame;
use frame_builder::FrameBuilderConfig;
use gpu_cache::GpuCache;
use internal_types::{SourceTexture, ResultMsg, RendererFrame};
use profiler::{BackendProfileCounters, GpuCacheProfileCounters, TextureCacheProfileCounters};
use record::ApiRecordingReceiver;
use resource_cache::ResourceCache;
use scene::Scene;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{Sender, channel};
use std::thread;
use std::io;
use std::rc::Rc;
use renderer::{Renderer, RendererOptions, InitError};
use texture_cache::TextureCache;
use time::precise_time_ns;
use thread_profiler::register_thread_with_profiler;
use rayon::ThreadPool;
use webgl_types::{GLContextHandleWrapper, GLContextWrapper};
use api::channel::{MsgSender, msg_channel, PayloadReceiver, PayloadReceiverHelperMethods};
use api::channel::{PayloadSender, PayloadSenderHelperMethods};
use api::{ApiMsg, BlobImageRenderer, BuiltDisplayList, DeviceIntPoint};
use api::{DeviceUintPoint, DeviceUintRect, DeviceUintSize, IdNamespace, ImageData};
use api::{LayerPoint, PipelineId, RenderDispatcher, RenderNotifier, RenderApiSender};
use api::{VRCompositorCommand, VRCompositorHandler, WebGLCommand, WebGLContextId};
use api::{FontTemplate, RendererId};

#[cfg(feature = "webgl")]
use offscreen_gl_context::GLContextDispatcher;

#[cfg(not(feature = "webgl"))]
use webgl_types::GLContextDispatcher;

/// The render backend is responsible for transforming high level display lists into
/// GPU-friendly work which is then submitted to the renderer in the form of a frame::Frame.
///
/// The render backend operates on its own thread.
pub struct RenderBackend {
    payload_rx: PayloadReceiver,
    payload_tx: PayloadSender,
    result_tx: Sender<ResultMsg>,

    // TODO(gw): Consider using strongly typed units here.
    hidpi_factor: f32,
    page_zoom_factor: f32,
    pinch_zoom_factor: f32,
    pan: DeviceIntPoint,
    window_size: DeviceUintSize,
    inner_rect: DeviceUintRect,
    next_namespace_id: IdNamespace,

    gpu_cache: GpuCache,
    resource_cache: ResourceCache,

    scene: Scene,
    frame: Frame,

    notifier: Arc<Mutex<Option<Box<RenderNotifier>>>>,
    webrender_context_handle: Option<GLContextHandleWrapper>,
    webgl_contexts: HashMap<WebGLContextId, GLContextWrapper>,
    dirty_webgl_contexts: HashSet<WebGLContextId>,
    current_bound_webgl_context_id: Option<WebGLContextId>,
    recorder: Option<Box<ApiRecordingReceiver>>,
    main_thread_dispatcher: Arc<Mutex<Option<Box<RenderDispatcher>>>>,

    next_webgl_id: usize,

    vr_compositor_handler: Arc<Mutex<Option<Box<VRCompositorHandler>>>>,

    // A helper switch to prevent any frames rendering triggered by scrolling
    // messages between `SetDisplayList` and `GenerateFrame`.
    // If we allow them, then a reftest that scrolls a few layers before generating
    // the first frame would produce inconsistent rendering results, because
    // scroll events are not necessarily received in deterministic order.
    render_on_scroll: bool,
    frame_counter: u32,
}

pub struct RenderBackendInit {
    pub payload_rx: PayloadReceiver,
    pub payload_tx: PayloadSender,
    pub result_tx: Sender<ResultMsg>,
    pub hidpi_factor: f32,
    pub max_texture_size: u32,
    pub workers: Arc<ThreadPool>,
    pub notifier: Arc<Mutex<Option<Box<RenderNotifier>>>>,
    pub webrender_context_handle: Option<GLContextHandleWrapper>,
    pub frame_builder_config: FrameBuilderConfig,
    pub recorder: Option<Box<ApiRecordingReceiver>>,
    pub main_thread_dispatcher: Arc<Mutex<Option<Box<RenderDispatcher>>>>,
    pub blob_image_renderer: Option<Box<BlobImageRenderer>>,
    pub vr_compositor_handler: Arc<Mutex<Option<Box<VRCompositorHandler>>>>,
    pub initial_window_size: DeviceUintSize,
    pub renderer_id: RendererId,
}

impl RenderBackend {
    pub fn new(init: RenderBackendInit) -> Self {

        let resource_cache = ResourceCache::new(
            TextureCache::new(init.max_texture_size),
            init.workers,
            init.blob_image_renderer
        );

        RenderBackend {
            payload_rx: init.payload_rx,
            payload_tx: init.payload_tx,
            result_tx: init.result_tx,
            hidpi_factor: init.hidpi_factor,
            page_zoom_factor: 1.0,
            pinch_zoom_factor: 1.0,
            pan: DeviceIntPoint::zero(),
            resource_cache,
            gpu_cache: GpuCache::new(),
            scene: Scene::new(),
            frame: Frame::new(init.frame_builder_config),
            next_namespace_id: IdNamespace(1),
            notifier: init.notifier,
            webrender_context_handle: init.webrender_context_handle,
            webgl_contexts: HashMap::new(),
            dirty_webgl_contexts: HashSet::new(),
            current_bound_webgl_context_id: None,
            recorder: init.recorder,
            main_thread_dispatcher: init.main_thread_dispatcher,
            next_webgl_id: 0,
            vr_compositor_handler: init.vr_compositor_handler,
            window_size: init.initial_window_size,
            inner_rect: DeviceUintRect::new(DeviceUintPoint::zero(), init.initial_window_size),
            frame_counter: 0,
            render_on_scroll: false,
        }
    }

    fn scroll_frame(&mut self, frame_maybe: Option<RendererFrame>,
                    profile_counters: &mut BackendProfileCounters) {
        match frame_maybe {
            Some(frame) => {
                self.publish_frame(frame, profile_counters);
                self.notify_compositor_of_new_scroll_frame(true)
            }
            None => self.notify_compositor_of_new_scroll_frame(false),
        }
    }

    pub fn process_message(&mut self, msg: ApiMsg, profile_counters: &mut BackendProfileCounters) -> BackendStatus {
        match msg {
            ApiMsg::AddRawFont(id, bytes, index) => {
                profile_counters.resources.font_templates.inc(bytes.len());
                self.resource_cache
                    .add_font_template(id, FontTemplate::Raw(Arc::new(bytes), index));
            }
            ApiMsg::AddNativeFont(id, native_font_handle) => {
                self.resource_cache
                    .add_font_template(id, FontTemplate::Native(native_font_handle));
            }
            ApiMsg::DeleteFont(id) => {
                self.resource_cache.delete_font_template(id);
            }
            ApiMsg::GetGlyphDimensions(glyph_keys, tx) => {
                let mut glyph_dimensions = Vec::with_capacity(glyph_keys.len());
                for glyph_key in &glyph_keys {
                    let glyph_dim = self.resource_cache.get_glyph_dimensions(glyph_key);
                    glyph_dimensions.push(glyph_dim);
                };
                tx.send(glyph_dimensions).unwrap();
            }
            ApiMsg::AddImage(id, descriptor, data, tiling) => {
                if let ImageData::Raw(ref bytes) = data {
                    profile_counters.resources.image_templates.inc(bytes.len());
                }
                self.resource_cache.add_image_template(id, descriptor, data, tiling);
            }
            ApiMsg::UpdateImage(id, descriptor, bytes, dirty_rect) => {
                self.resource_cache.update_image_template(id, descriptor, bytes, dirty_rect);
            }
            ApiMsg::DeleteImage(id) => {
                self.resource_cache.delete_image_template(id);
            }
            ApiMsg::SetPageZoom(factor) => {
                self.page_zoom_factor = factor.get();
            }
            ApiMsg::SetPinchZoom(factor) => {
                self.pinch_zoom_factor = factor.get();
            }
            ApiMsg::SetPan(pan) => {
                self.pan = pan;
            }
            ApiMsg::SetWindowParameters(window_size, inner_rect) => {
                self.window_size = window_size;
                self.inner_rect = inner_rect;
            }
            ApiMsg::CloneApi(sender) => {
                let result = self.next_namespace_id;

                let IdNamespace(id_namespace) = self.next_namespace_id;
                self.next_namespace_id = IdNamespace(id_namespace + 1);

                sender.send(result).unwrap();
            }
            ApiMsg::SetDisplayList(background_color,
                                   epoch,
                                   pipeline_id,
                                   viewport_size,
                                   content_size,
                                   display_list_descriptor,
                                   preserve_frame_state) => {
                profile_scope!("SetDisplayList");
                let mut leftover_data = vec![];
                let mut data;
                loop {
                    data = self.payload_rx.recv_payload().unwrap();
                    {
                        if data.epoch == epoch &&
                           data.pipeline_id == pipeline_id {
                            break
                        }
                    }
                    leftover_data.push(data)
                }
                for leftover_data in leftover_data {
                    self.payload_tx.send_payload(leftover_data).unwrap()
                }
                if let Some(ref mut r) = self.recorder {
                    r.write_payload(self.frame_counter, &data.to_data());
                }

                let built_display_list =
                    BuiltDisplayList::from_data(data.display_list_data,
                                                display_list_descriptor);

                if !preserve_frame_state {
                    self.discard_frame_state_for_pipeline(pipeline_id);
                }

                let display_list_len = built_display_list.data().len();
                let (builder_start_time, builder_finish_time) = built_display_list.times();

                let display_list_received_time = precise_time_ns();

                profile_counters.total_time.profile(|| {
                    self.scene.set_display_list(pipeline_id,
                                                epoch,
                                                built_display_list,
                                                background_color,
                                                viewport_size,
                                                content_size);
                    self.build_scene();
                });

                self.render_on_scroll = false; //wait for `GenerateFrame`

                // Note: this isn't quite right as auxiliary values will be
                // pulled out somewhere in the prim_store, but aux values are
                // really simple and cheap to access, so it's not a big deal.
                let display_list_consumed_time = precise_time_ns();

                profile_counters.ipc.set(builder_start_time, builder_finish_time,
                                         display_list_received_time, display_list_consumed_time,
                                         display_list_len);
            }
            ApiMsg::SetRootPipeline(pipeline_id) => {
                profile_scope!("SetRootPipeline");
                self.scene.set_root_pipeline_id(pipeline_id);

                if self.scene.display_lists.get(&pipeline_id).is_none() {
                    return BackendStatus::Continue;
                }

                profile_counters.total_time.profile(|| {
                    self.build_scene();
                })
            }
            ApiMsg::Scroll(delta, cursor, move_phase) => {
                profile_scope!("Scroll");
                let frame = {
                    let counters = &mut profile_counters.resources.texture_cache;
                    let gpu_cache_counters = &mut profile_counters.resources.gpu_cache;
                    profile_counters.total_time.profile(|| {
                        if self.frame.scroll(delta, cursor, move_phase) {
                            Some(self.render(counters, gpu_cache_counters))
                        } else {
                            None
                        }
                    })
                };

                self.scroll_frame(frame, profile_counters);
            }
            ApiMsg::ScrollNodeWithId(origin, id, clamp) => {
                profile_scope!("ScrollNodeWithScrollId");
                let frame = {
                    let counters = &mut profile_counters.resources.texture_cache;
                    let gpu_cache_counters = &mut profile_counters.resources.gpu_cache;
                    profile_counters.total_time.profile(|| {
                        if self.frame.scroll_node(origin, id, clamp) {
                            Some(self.render(counters, gpu_cache_counters))
                        } else {
                            None
                        }
                    })
                };

                self.scroll_frame(frame, profile_counters);
            }
            ApiMsg::TickScrollingBounce => {
                profile_scope!("TickScrollingBounce");
                let frame = {
                    let counters = &mut profile_counters.resources.texture_cache;
                    let gpu_cache_counters = &mut profile_counters.resources.gpu_cache;
                    profile_counters.total_time.profile(|| {
                        self.frame.tick_scrolling_bounce_animations();
                        if self.render_on_scroll {
                            Some(self.render(counters, gpu_cache_counters))
                        } else {
                            None
                        }
                    })
                };

                self.scroll_frame(frame, profile_counters);
            }
            ApiMsg::TranslatePointToLayerSpace(..) => {
                panic!("unused api - remove from webrender_traits");
            }
            ApiMsg::GetScrollNodeState(tx) => {
                profile_scope!("GetScrollNodeState");
                tx.send(self.frame.get_scroll_node_state()).unwrap()
            }
            ApiMsg::RequestWebGLContext(size, attributes, tx) => {
                if let Some(ref wrapper) = self.webrender_context_handle {
                    let dispatcher: Option<Box<GLContextDispatcher>> = if cfg!(target_os = "windows") {
                        Some(Box::new(WebRenderGLDispatcher {
                            dispatcher: Arc::clone(&self.main_thread_dispatcher)
                        }))
                    } else {
                        None
                    };

                    let result = wrapper.new_context(size, attributes, dispatcher);
                    // Creating a new GLContext may make the current bound context_id dirty.
                    // Clear it to ensure that  make_current() is called in subsequent commands.
                    self.current_bound_webgl_context_id = None;

                    match result {
                        Ok(ctx) => {
                            let id = WebGLContextId(self.next_webgl_id);
                            self.next_webgl_id += 1;

                            let (real_size, texture_id, limits) = ctx.get_info();

                            self.webgl_contexts.insert(id, ctx);

                            self.resource_cache
                                .add_webgl_texture(id, SourceTexture::WebGL(texture_id),
                                                   real_size);

                            tx.send(Ok((id, limits))).unwrap();
                        },
                        Err(msg) => {
                            tx.send(Err(msg.to_owned())).unwrap();
                        }
                    }
                } else {
                    tx.send(Err("Not implemented yet".to_owned())).unwrap();
                }
            }
            ApiMsg::ResizeWebGLContext(context_id, size) => {
                let ctx = self.webgl_contexts.get_mut(&context_id).unwrap();
                if Some(context_id) != self.current_bound_webgl_context_id {
                    ctx.make_current();
                    self.current_bound_webgl_context_id = Some(context_id);
                }
                match ctx.resize(&size) {
                    Ok(_) => {
                        // Update webgl texture size. Texture id may change too.
                        let (real_size, texture_id, _) = ctx.get_info();
                        self.resource_cache
                            .update_webgl_texture(context_id, SourceTexture::WebGL(texture_id),
                                                  real_size);
                    },
                    Err(msg) => {
                        error!("Error resizing WebGLContext: {}", msg);
                    }
                }
            }
            ApiMsg::WebGLCommand(context_id, command) => {
                // TODO: Buffer the commands and only apply them here if they need to
                // be synchronous.
                let ctx = &self.webgl_contexts[&context_id];
                if Some(context_id) != self.current_bound_webgl_context_id {
                    ctx.make_current();
                    self.current_bound_webgl_context_id = Some(context_id);
                }
                ctx.apply_command(command);
            },

            ApiMsg::VRCompositorCommand(context_id, command) => {
                if Some(context_id) != self.current_bound_webgl_context_id {
                    self.webgl_contexts[&context_id].make_current();
                    self.current_bound_webgl_context_id = Some(context_id);
                }
                self.handle_vr_compositor_command(context_id, command);
            }
            ApiMsg::GenerateFrame(property_bindings) => {
                profile_scope!("GenerateFrame");

                // Ideally, when there are property bindings present,
                // we won't need to rebuild the entire frame here.
                // However, to avoid conflicts with the ongoing work to
                // refactor how scroll roots + transforms work, this
                // just rebuilds the frame if there are animated property
                // bindings present for now.
                // TODO(gw): Once the scrolling / reference frame changes
                //           are completed, optimize the internals of
                //           animated properties to not require a full
                //           rebuild of the frame!
                if let Some(property_bindings) = property_bindings {
                    self.scene.properties.set_properties(property_bindings);
                    profile_counters.total_time.profile(|| {
                        self.build_scene();
                    });
                }

                self.render_on_scroll = true;

                let frame = {
                    let counters = &mut profile_counters.resources.texture_cache;
                    let gpu_cache_counters = &mut profile_counters.resources.gpu_cache;
                    profile_counters.total_time.profile(|| {
                        self.render(counters, gpu_cache_counters)
                    })
                };
                if self.scene.root_pipeline_id.is_some() {
                    self.publish_frame_and_notify_compositor(frame, profile_counters);
                    self.frame_counter += 1;
                }
            }
            ApiMsg::ExternalEvent(evt) => {
                let notifier = self.notifier.lock();
                notifier.unwrap()
                        .as_mut()
                        .unwrap()
                        .external_event(evt);
            }
            ApiMsg::ShutDown => {
                let notifier = self.notifier.lock();
                notifier.unwrap()
                        .as_mut()
                        .unwrap()
                        .shut_down();
                return BackendStatus::Remove;
            }
        }

        return BackendStatus::Continue;
    }

    fn discard_frame_state_for_pipeline(&mut self, pipeline_id: PipelineId) {
        self.frame.discard_frame_state_for_pipeline(pipeline_id);
    }

    fn accumulated_scale_factor(&self) -> f32 {
        self.hidpi_factor * self.page_zoom_factor * self.pinch_zoom_factor
    }

    fn build_scene(&mut self) {
        // Flatten the stacking context hierarchy
        if let Some(id) = self.current_bound_webgl_context_id {
            self.webgl_contexts[&id].unbind();
            self.current_bound_webgl_context_id = None;
        }

        // When running in OSMesa mode with texture sharing,
        // a flush is required on any GL contexts to ensure
        // that read-back from the shared texture returns
        // valid data! This should be fine to have run on all
        // implementations - a single flush for each webgl
        // context at the start of a render frame should
        // incur minimal cost.
        // glFlush is not enough in some GPUs.
        // glFlush doesn't guarantee the completion of the GL commands when the shared texture is sampled.
        // This leads to some graphic glitches on some demos or even nothing being rendered at all (GPU Mali-T880).
        // glFinish guarantees the completion of the commands but it may hurt performance a lot.
        // Sync Objects are the recommended way to ensure that textures are ready in OpenGL 3.0+.
        // They are more performant than glFinish and guarantee the completion of the GL commands.
        for (id, webgl_context) in &self.webgl_contexts {
            if self.dirty_webgl_contexts.remove(&id) {
                webgl_context.make_current();
                webgl_context.apply_command(WebGLCommand::FenceAndWaitSync);
                webgl_context.unbind();
            }
        }

        let accumulated_scale_factor = self.accumulated_scale_factor();
        self.frame.create(&self.scene,
                          &mut self.resource_cache,
                          self.window_size,
                          self.inner_rect,
                          accumulated_scale_factor);
    }

    fn render(&mut self,
              texture_cache_profile: &mut TextureCacheProfileCounters,
              gpu_cache_profile: &mut GpuCacheProfileCounters)
              -> RendererFrame {
        let accumulated_scale_factor = self.accumulated_scale_factor();
        let pan = LayerPoint::new(self.pan.x as f32 / accumulated_scale_factor,
                                  self.pan.y as f32 / accumulated_scale_factor);
        let frame = self.frame.build(&mut self.resource_cache,
                                     &mut self.gpu_cache,
                                     &self.scene.display_lists,
                                     accumulated_scale_factor,
                                     pan,
                                     texture_cache_profile,
                                     gpu_cache_profile);
        frame
    }

    fn publish_frame(&mut self,
                     frame: RendererFrame,
                     profile_counters: &mut BackendProfileCounters) {
        let pending_update = self.resource_cache.pending_updates();
        let msg = ResultMsg::NewFrame(frame, pending_update, profile_counters.clone());
        self.result_tx.send(msg).unwrap();
        profile_counters.reset();
    }

    fn publish_frame_and_notify_compositor(&mut self,
                                           frame: RendererFrame,
                                           profile_counters: &mut BackendProfileCounters) {
        self.publish_frame(frame, profile_counters);

        // TODO(gw): This is kindof bogus to have to lock the notifier
        //           each time it's used. This is due to some nastiness
        //           in initialization order for Servo. Perhaps find a
        //           cleaner way to do this, or use the OnceMutex on crates.io?
        let mut notifier = self.notifier.lock();
        notifier.as_mut().unwrap().as_mut().unwrap().new_frame_ready();
    }

    fn notify_compositor_of_new_scroll_frame(&mut self, composite_needed: bool) {
        // TODO(gw): This is kindof bogus to have to lock the notifier
        //           each time it's used. This is due to some nastiness
        //           in initialization order for Servo. Perhaps find a
        //           cleaner way to do this, or use the OnceMutex on crates.io?
        let mut notifier = self.notifier.lock();
        notifier.as_mut().unwrap().as_mut().unwrap().new_scroll_frame_ready(composite_needed);
    }

    fn handle_vr_compositor_command(&mut self, ctx_id: WebGLContextId, cmd: VRCompositorCommand) {
        let texture = match cmd {
            VRCompositorCommand::SubmitFrame(..) => {
                    match self.resource_cache.get_webgl_texture(&ctx_id).id {
                        SourceTexture::WebGL(texture_id) => {
                            let size = self.resource_cache.get_webgl_texture_size(&ctx_id);
                            Some((texture_id, size))
                        },
                        _=> None
                    }
            },
            _ => None
        };
        let mut handler = self.vr_compositor_handler.lock();
        handler.as_mut().unwrap().as_mut().unwrap().handle(cmd, texture);
    }
}

struct WebRenderGLDispatcher {
    dispatcher: Arc<Mutex<Option<Box<RenderDispatcher>>>>
}

impl GLContextDispatcher for WebRenderGLDispatcher {
    fn dispatch(&self, f: Box<Fn() + Send>) {
        let mut dispatcher = self.dispatcher.lock();
        dispatcher.as_mut().unwrap().as_mut().unwrap().dispatch(f);
    }
}

pub enum BackendMsg {
    NewRenderBackend(RenderBackendInit),
    DeleteRenderBackend(RendererId),
}

pub struct RenderBackendThread {
    api_tx: MsgSender<(ApiMsg, RendererId)>,
    init_tx: Sender<BackendMsg>,
    next_renderer_id: RendererId,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BackendStatus {
    Continue,
    Remove,
}

impl RenderBackendThread {
    pub fn new(thread_name: String) -> Result<Self, io::Error> {
        let (api_tx, api_rx) = try! { msg_channel() };
        let (init_tx, init_rx) = channel();

        thread::Builder::new().name(thread_name.clone()).spawn(move || {
            register_thread_with_profiler(thread_name);

            let mut render_backends = HashMap::new();

            while let Ok((api_msg, id)) = api_rx.recv() {

                while let Ok(init_msg) = init_rx.try_recv() {
                    match init_msg {
                        BackendMsg::NewRenderBackend(backend_init) => {
                            let counters = BackendProfileCounters::new();
                            let id = backend_init.renderer_id;
                            let backend = RenderBackend::new(backend_init);
                            render_backends.insert(id, (backend, counters));
                        }
                        BackendMsg::DeleteRenderBackend(id) => {
                            render_backends.remove(&id);
                        }
                    }
                }

                let should_remove = match render_backends.get_mut(&id) {
                    Some(&mut (ref mut backend, ref mut counters)) => {
                        profile_scope!("handle_msg");
                        backend.process_message(api_msg, counters) == BackendStatus::Remove
                    }
                    None => {
                        println!("Failed to deliver message to non-existant render backend!");
                        false
                    }
                };

                if should_remove {
                    render_backends.remove(&id);
                }
            }
        })?;

        Ok(RenderBackendThread {
            api_tx: api_tx,
            init_tx: init_tx,
            next_renderer_id: RendererId(0),
        })
    }

    pub fn new_renderer(
        &mut self,
        gl: Rc<gl::Gl>,
        options: RendererOptions,
        initial_window_size: DeviceUintSize,
    ) -> Result<(Renderer, RenderApiSender), InitError> {
        Renderer::new(gl, options, initial_window_size, self)
    }

    pub fn alloc_renderer_id(&mut self) -> RendererId {
        let id = self.next_renderer_id;
        self.next_renderer_id = RendererId(self.next_renderer_id.0 + 1);

        id
    }

    pub fn clone_api_sender(&self) -> MsgSender<(ApiMsg, RendererId)> {
        self.api_tx.clone()
    }

    pub fn add_render_backend(&self, init: RenderBackendInit) {
        self.init_tx.send(BackendMsg::NewRenderBackend(init)).unwrap();
    }

    pub fn delete_render_backend(&self, id: RendererId) {
        self.init_tx.send(BackendMsg::DeleteRenderBackend(id)).unwrap()
    }
}
