/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{ApiMsg, BuiltDisplayList, ClearCache, DebugCommand};
#[cfg(feature = "debugger")]
use api::{BuiltDisplayListIter, SpecificDisplayItem};
use api::{DeviceIntPoint, DevicePixelScale, DeviceUintPoint, DeviceUintRect, DeviceUintSize};
use api::{DocumentId, DocumentLayer, DocumentMsg, HitTestResult, IdNamespace, PipelineId};
use api::{Epoch, TransactionMsg, ResourceUpdates};
use api::RenderNotifier;
use api::channel::{MsgReceiver, PayloadReceiver, PayloadReceiverHelperMethods};
use api::channel::{PayloadSender, PayloadSenderHelperMethods};
use api::channel::MsgSender;
#[cfg(feature = "capture")]
use api::CaptureBits;
#[cfg(feature = "replay")]
use api::CapturedDocument;
#[cfg(feature = "debugger")]
use debug_server;
use frame::FrameContext;
use frame_builder::{FrameBuilder, FrameBuilderConfig};
use gpu_cache::GpuCache;
use hit_test::{HitTest, HitTester};
use internal_types::{DebugOutput, FastHashMap, FastHashSet, RenderedDocument, ResultMsg};
use profiler::{BackendProfileCounters, IpcProfileCounters, ResourceProfileCounters};
use record::ApiRecordingReceiver;
use resource_cache::ResourceCache;
#[cfg(feature = "replay")]
use resource_cache::PlainCacheOwn;
#[cfg(any(feature = "capture", feature = "replay"))]
use resource_cache::PlainResources;
use scene::{Scene, SceneProperties};
#[cfg(feature = "serialize")]
use serde::{Serialize, Deserialize};
#[cfg(feature = "debugger")]
use serde_json;
#[cfg(any(feature = "capture", feature = "replay"))]
use std::path::PathBuf;
use std::sync::atomic::{ATOMIC_USIZE_INIT, AtomicUsize, Ordering};
use std::mem::replace;
use std::sync::mpsc::{Sender, Receiver, channel};
use std::u32;
use time::precise_time_ns;
use resource_cache::{FontInstanceMap, TiledImageMap};
use clip_scroll_tree::ClipScrollTree;

// WIP: I realize we don't really want to send the entire struct to the scene
// building thread, this will be most likely a private struct again by the time
// I figure the bigger picture out.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Clone)]
pub struct DocumentView {
    pub window_size: DeviceUintSize,
    pub inner_rect: DeviceUintRect,
    pub layer: DocumentLayer,
    pub pan: DeviceIntPoint,
    pub device_pixel_ratio: f32,
    pub page_zoom_factor: f32,
    pub pinch_zoom_factor: f32,
}

impl DocumentView {
    pub fn accumulated_scale_factor(&self) -> DevicePixelScale {
        DevicePixelScale::new(
            self.device_pixel_ratio *
            self.page_zoom_factor *
            self.pinch_zoom_factor
        )
    }
}

struct Document {
    scene: Scene,
    view: DocumentView,
    frame_ctx: FrameContext,
    // the `Option` here is only to deal with borrow checker
    frame_builder: Option<FrameBuilder>,
    // A set of pipelines that the caller has requested be
    // made available as output textures.
    output_pipelines: FastHashSet<PipelineId>,
    // The pipeline removal notifications that will be sent in the next frame.
    // Because of async scene building, removed pipelines should not land here
    // as soon as the render backend receives a DocumentMsg::RemovePipeline.
    // Instead, the notification should be added to this list when the first
    // scene that does not contain the pipeline becomes current.
    removed_pipelines: Vec<PipelineId>,
    // A helper switch to prevent any frames rendering triggered by scrolling
    // messages between `SetDisplayList` and `GenerateFrame`.
    // If we allow them, then a reftest that scrolls a few layers before generating
    // the first frame would produce inconsistent rendering results, because
    // scroll events are not necessarily received in deterministic order.
    render_on_scroll: Option<bool>,
    // A helper flag to prevent any hit-tests from happening between calls
    // to build_scene and rendering the document. In between these two calls,
    // hit-tests produce inconsistent results because the clip_scroll_tree
    // is out of sync with the display list.
    render_on_hittest: bool,

    /// A data structure to allow hit testing against rendered frames. This is updated
    /// every time we produce a fully rendered frame.
    hit_tester: Option<HitTester>,

    /// Properties that are resolved during frame building and can be changed at any time
    /// without requiring the scene to be re-built.
    dynamic_properties: SceneProperties,
}

impl Document {
    pub fn new(
        config: FrameBuilderConfig,
        window_size: DeviceUintSize,
        layer: DocumentLayer,
        enable_render_on_scroll: bool,
        default_device_pixel_ratio: f32,
    ) -> Self {
        let render_on_scroll = if enable_render_on_scroll {
            Some(false)
        } else {
            None
        };
        Document {
            scene: Scene::new(),
            removed_pipelines: Vec::new(),
            view: DocumentView {
                window_size,
                inner_rect: DeviceUintRect::new(DeviceUintPoint::zero(), window_size),
                layer,
                pan: DeviceIntPoint::zero(),
                page_zoom_factor: 1.0,
                pinch_zoom_factor: 1.0,
                device_pixel_ratio: default_device_pixel_ratio,
            },
            frame_ctx: FrameContext::new(config),
            frame_builder: None,
            output_pipelines: FastHashSet::default(),
            render_on_scroll,
            render_on_hittest: false,
            hit_tester: None,
            dynamic_properties: SceneProperties::new(),
        }
    }

    fn can_render(&self) -> bool { self.frame_builder.is_some() }

    // TODO: We will probably get rid of this soon and always forward to the scene building thread.
    fn build_scene(&mut self, resource_cache: &mut ResourceCache) {
        let frame_builder = self.frame_ctx.create_frame_builder(
            self.frame_builder.take().unwrap_or_else(FrameBuilder::empty),
            &self.scene,
            resource_cache,
            self.view.window_size,
            self.view.inner_rect,
            self.view.accumulated_scale_factor(),
            &self.output_pipelines,
        );
        self.removed_pipelines.extend(self.scene.removed_pipelines.drain(..));
        self.frame_builder = Some(frame_builder);
    }

    fn forward_transaction_to_scene_builder(
        &mut self,
        txn: TransactionMsg,
        document_ops: &DocumentOps,
        document_id: DocumentId,
        resource_cache: &ResourceCache,
        scene_tx: &Sender<SceneBuilderRequest>,
    ) {
        // Do as much of the error handling as possible here before dispatching to
        // the scene builder thread.
        let build_scene: bool = document_ops.build
            && self.scene.root_pipeline_id.map(
                |id| { self.scene.pipelines.contains_key(&id) }
            ).unwrap_or(false);

        let scene_request = if build_scene {
            if self.view.window_size.width == 0 || self.view.window_size.height == 0 {
                error!("ERROR: Invalid window dimensions! Please call api.set_window_size()");
            }

            Some(SceneRequest {
                scene: self.scene.clone(),
                view: self.view.clone(),
                font_instances: resource_cache.get_font_instances(),
                tiled_image_map: resource_cache.get_tiled_image_map(),
                output_pipelines: self.output_pipelines.clone(),
            })
        } else {
            None
        };

        scene_tx.send(SceneBuilderRequest::Transaction {
            scene: scene_request,
            resource_updates: txn.resource_updates,
            document_ops: txn.postfix_ops,
            render: txn.generate_frame,
            document_id,
        }).unwrap();
    }

    fn render(
        &mut self,
        resource_cache: &mut ResourceCache,
        gpu_cache: &mut GpuCache,
        resource_profile: &mut ResourceProfileCounters,
    ) -> RenderedDocument {
        let accumulated_scale_factor = self.view.accumulated_scale_factor();
        let pan = self.view.pan.to_f32() / accumulated_scale_factor;
        let (hit_tester, rendered_document) = self.frame_ctx.build_rendered_document(
            self.frame_builder.as_mut().unwrap(),
            resource_cache,
            gpu_cache,
            &self.scene.pipelines,
            accumulated_scale_factor,
            self.view.layer,
            pan,
            &mut resource_profile.texture_cache,
            &mut resource_profile.gpu_cache,
            &self.dynamic_properties,
            replace(&mut self.removed_pipelines, Vec::new()),
        );

        self.hit_tester = Some(hit_tester);

        rendered_document
    }
}

struct DocumentOps {
    scroll: bool,
    build: bool,
    render: bool,
    composite: bool,
}

impl DocumentOps {
    fn nop() -> Self {
        DocumentOps {
            scroll: false,
            build: false,
            render: false,
            composite: false,
        }
    }

    fn build() -> Self {
        DocumentOps {
            build: true,
            ..DocumentOps::nop()
        }
    }

    fn combine(&mut self, other: Self) {
        self.scroll = self.scroll || other.scroll;
        self.build = self.build || other.build;
        self.render = self.render || other.render;
        self.composite = self.composite || other.composite;
    }
}

/// The unique id for WR resource identification.
static NEXT_NAMESPACE_ID: AtomicUsize = ATOMIC_USIZE_INIT;

#[cfg(any(feature = "capture", feature = "replay"))]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
struct PlainRenderBackend {
    default_device_pixel_ratio: f32,
    enable_render_on_scroll: bool,
    frame_config: FrameBuilderConfig,
    documents: FastHashMap<DocumentId, DocumentView>,
    resources: PlainResources,
}

/// The render backend is responsible for transforming high level display lists into
/// GPU-friendly work which is then submitted to the renderer in the form of a frame::Frame.
///
/// The render backend operates on its own thread.
pub struct RenderBackend {
    api_rx: MsgReceiver<ApiMsg>,
    payload_rx: PayloadReceiver,
    payload_tx: PayloadSender,
    result_tx: Sender<ResultMsg>,
    scene_tx: Sender<SceneBuilderRequest>,
    scene_rx: Receiver<SceneBuilderMsg>,

    default_device_pixel_ratio: f32,

    gpu_cache: GpuCache,
    resource_cache: ResourceCache,

    frame_config: FrameBuilderConfig,
    documents: FastHashMap<DocumentId, Document>,

    notifier: Box<RenderNotifier>,
    recorder: Option<Box<ApiRecordingReceiver>>,

    enable_render_on_scroll: bool,
}

impl RenderBackend {
    pub fn new(
        api_rx: MsgReceiver<ApiMsg>,
        payload_rx: PayloadReceiver,
        payload_tx: PayloadSender,
        result_tx: Sender<ResultMsg>,
        scene_tx: Sender<SceneBuilderRequest>,
        scene_rx: Receiver<SceneBuilderMsg>,
        default_device_pixel_ratio: f32,
        resource_cache: ResourceCache,
        notifier: Box<RenderNotifier>,
        frame_config: FrameBuilderConfig,
        recorder: Option<Box<ApiRecordingReceiver>>,
        enable_render_on_scroll: bool,
    ) -> RenderBackend {
        // The namespace_id should start from 1.
        NEXT_NAMESPACE_ID.fetch_add(1, Ordering::Relaxed);

        RenderBackend {
            api_rx,
            payload_rx,
            payload_tx,
            result_tx,
            scene_tx,
            scene_rx,
            default_device_pixel_ratio,
            resource_cache,
            gpu_cache: GpuCache::new(),
            frame_config,
            documents: FastHashMap::default(),
            notifier,
            recorder,
            enable_render_on_scroll,
        }
    }

    fn process_document(
        &mut self,
        document_id: DocumentId,
        message: DocumentMsg,
        frame_counter: u32,
        ipc_profile_counters: &mut IpcProfileCounters,
    ) -> DocumentOps {
        let doc = self.documents.get_mut(&document_id).expect("No document?");

        match message {
            //TODO: move view-related messages in a separate enum?
            DocumentMsg::SetPageZoom(factor) => {
                doc.view.page_zoom_factor = factor.get();
                DocumentOps::nop()
            }
            DocumentMsg::EnableFrameOutput(pipeline_id, enable) => {
                if enable {
                    doc.output_pipelines.insert(pipeline_id);
                } else {
                    doc.output_pipelines.remove(&pipeline_id);
                }
                DocumentOps::nop()
            }
            DocumentMsg::SetPinchZoom(factor) => {
                doc.view.pinch_zoom_factor = factor.get();
                DocumentOps::nop()
            }
            DocumentMsg::SetPan(pan) => {
                doc.view.pan = pan;
                DocumentOps::nop()
            }
            DocumentMsg::SetWindowParameters {
                window_size,
                inner_rect,
                device_pixel_ratio,
            } => {
                doc.view.window_size = window_size;
                doc.view.inner_rect = inner_rect;
                doc.view.device_pixel_ratio = device_pixel_ratio;
                DocumentOps::nop()
            }
            DocumentMsg::SetDisplayList {
                epoch,
                pipeline_id,
                background,
                viewport_size,
                content_size,
                list_descriptor,
                preserve_frame_state,
            } => {
                profile_scope!("SetDisplayList");

                let mut data;
                while {
                    data = self.payload_rx.recv_payload().unwrap();
                    data.epoch != epoch || data.pipeline_id != pipeline_id
                } {
                    self.payload_tx.send_payload(data).unwrap()
                }

                if let Some(ref mut r) = self.recorder {
                    r.write_payload(frame_counter, &data.to_data());
                }

                let built_display_list =
                    BuiltDisplayList::from_data(data.display_list_data, list_descriptor);

                if !preserve_frame_state {
                    doc.frame_ctx.discard_frame_state_for_pipeline(pipeline_id);
                }

                let display_list_len = built_display_list.data().len();
                let (builder_start_time, builder_finish_time, send_start_time) =
                    built_display_list.times();
                let display_list_received_time = precise_time_ns();

                {
                    doc.scene.set_display_list(
                        pipeline_id,
                        epoch,
                        built_display_list,
                        background,
                        viewport_size,
                        content_size,
                    );
                }

                if let Some(ref mut ros) = doc.render_on_scroll {
                    *ros = false; //wait for `GenerateFrame`
                }

                // Note: this isn't quite right as auxiliary values will be
                // pulled out somewhere in the prim_store, but aux values are
                // really simple and cheap to access, so it's not a big deal.
                let display_list_consumed_time = precise_time_ns();

                ipc_profile_counters.set(
                    builder_start_time,
                    builder_finish_time,
                    send_start_time,
                    display_list_received_time,
                    display_list_consumed_time,
                    display_list_len,
                );

                DocumentOps::build()
            }
            DocumentMsg::UpdateEpoch(pipeline_id, epoch) => {
                doc.scene.update_epoch(pipeline_id, epoch);
                doc.frame_ctx.update_epoch(pipeline_id, epoch);
                DocumentOps::nop()
            }
            DocumentMsg::SetRootPipeline(pipeline_id) => {
                profile_scope!("SetRootPipeline");

                doc.scene.set_root_pipeline_id(pipeline_id);
                if doc.scene.pipelines.get(&pipeline_id).is_some() {
                    DocumentOps::build()
                } else {
                    DocumentOps::nop()
                }
            }
            DocumentMsg::RemovePipeline(pipeline_id) => {
                profile_scope!("RemovePipeline");

                doc.scene.remove_pipeline(pipeline_id);
                DocumentOps::nop()
            }
            DocumentMsg::Scroll(delta, cursor, move_phase) => {
                profile_scope!("Scroll");

                let should_render = doc.frame_ctx.scroll(delta, cursor, move_phase)
                    && doc.render_on_scroll == Some(true);

                DocumentOps {
                    scroll: true,
                    render: should_render,
                    composite: should_render,
                    ..DocumentOps::nop()
                }
            }
            DocumentMsg::HitTest(pipeline_id, point, flags, tx) => {

                let result = match doc.hit_tester {
                    Some(ref hit_tester) => {
                        hit_tester.hit_test(HitTest::new(pipeline_id, point, flags))
                    }
                    None => HitTestResult { items: Vec::new() },
                };

                tx.send(result).unwrap();
                DocumentOps::nop()
            }
            DocumentMsg::ScrollNodeWithId(origin, id, clamp) => {
                profile_scope!("ScrollNodeWithScrollId");

                let should_render = doc.frame_ctx.scroll_node(origin, id, clamp)
                    && doc.render_on_scroll == Some(true);

                DocumentOps {
                    scroll: true,
                    render: should_render,
                    composite: should_render,
                    ..DocumentOps::nop()
                }
            }
            DocumentMsg::TickScrollingBounce => {
                profile_scope!("TickScrollingBounce");

                doc.frame_ctx.tick_scrolling_bounce_animations();

                let should_render = doc.render_on_scroll == Some(true);

                DocumentOps {
                    scroll: true,
                    render: should_render,
                    composite: should_render,
                    ..DocumentOps::nop()
                }
            }
            DocumentMsg::GetScrollNodeState(tx) => {
                profile_scope!("GetScrollNodeState");
                tx.send(doc.frame_ctx.get_scroll_node_state()).unwrap();
                DocumentOps::nop()
            }
            DocumentMsg::UpdateDynamicProperties(property_bindings) => {
                doc.dynamic_properties.set_properties(property_bindings);
                DocumentOps::build()
            }
        }
    }

    fn next_namespace_id(&self) -> IdNamespace {
        IdNamespace(NEXT_NAMESPACE_ID.fetch_add(1, Ordering::Relaxed) as u32)
    }

    pub fn run(&mut self, mut profile_counters: BackendProfileCounters) {
        let mut frame_counter: u32 = 0;

        loop {
            profile_scope!("handle_msg");

            while let Ok(msg) = self.scene_rx.try_recv() {
                match msg {
                    SceneBuilderMsg::Transaction {
                        document_id,
                        mut built_scene,
                        document_ops,
                        render,
                        resource_updates,
                    } => {
                        if let Some(doc) = self.documents.get_mut(&document_id) {
                            if let Some(mut built_scene) = built_scene.take() {
                                doc.frame_builder = Some(built_scene.frame_builder);
                                doc.removed_pipelines.extend(built_scene.removed_pipelines.drain(..));
                                doc.frame_ctx.new_async_scene_ready(
                                    built_scene.clip_scroll_tree,
                                    built_scene.pipeline_epoch_map,
                                );
                                doc.render_on_hittest = true;
                            }
                        } else {
                            // The document was removed while we were building it, skip it.
                            // TODO: we might want to just ensure that removed documents are
                            // always forwarded to the scene builder thread to avoid this case.
                            continue;
                        }

                        let txn = TransactionMsg {
                            prefix_ops: Vec::new(),
                            postfix_ops: document_ops,
                            resource_updates,
                            generate_frame: render,
                            use_scene_builder_thread: false,
                        };

                        if !txn.is_empty() {
                            self.update_document(
                                document_id,
                                txn,
                                &mut frame_counter,
                                &mut profile_counters
                            );
                        }
                    }
                }
            }

            let keep_going = match self.api_rx.recv() {
                Ok(msg) => {
                    if let Some(ref mut r) = self.recorder {
                        r.write_msg(frame_counter, &msg);
                    }
                    self.process_api_msg(msg, &mut profile_counters, &mut frame_counter)
                }
                Err(..) => { false }
            };

            if !keep_going {
                let _ = self.scene_tx.send(SceneBuilderRequest::Stop);
                self.notifier.shut_down();
            }
        }
    }

    fn process_api_msg(
        &mut self,
        msg: ApiMsg,
        profile_counters: &mut BackendProfileCounters,
        frame_counter: &mut u32,
    ) -> bool {
        // WIP: reeindent that when the work is closer to land (otherwise rebasing is
        // is a pain).


            match msg {
                ApiMsg::WakeUp => {}
                ApiMsg::UpdateResources(updates) => {
                    self.resource_cache
                        .update_resources(updates, &mut profile_counters.resources);
                }
                ApiMsg::GetGlyphDimensions(instance_key, glyph_keys, tx) => {
                    let mut glyph_dimensions = Vec::with_capacity(glyph_keys.len());
                    if let Some(font) = self.resource_cache.get_font_instance(instance_key) {
                        for glyph_key in &glyph_keys {
                            let glyph_dim = self.resource_cache.get_glyph_dimensions(&font, glyph_key);
                            glyph_dimensions.push(glyph_dim);
                        }
                    }
                    tx.send(glyph_dimensions).unwrap();
                }
                ApiMsg::GetGlyphIndices(font_key, text, tx) => {
                    let mut glyph_indices = Vec::new();
                    for ch in text.chars() {
                        let index = self.resource_cache.get_glyph_index(font_key, ch);
                        glyph_indices.push(index);
                    }
                    tx.send(glyph_indices).unwrap();
                }
                ApiMsg::CloneApi(sender) => {
                    sender.send(self.next_namespace_id()).unwrap();
                }
                ApiMsg::AddDocument(document_id, initial_size, layer) => {
                    let document = Document::new(
                        self.frame_config.clone(),
                        initial_size,
                        layer,
                        self.enable_render_on_scroll,
                        self.default_device_pixel_ratio,
                    );
                    self.documents.insert(document_id, document);
                }
                ApiMsg::DeleteDocument(document_id) => {
                    self.documents.remove(&document_id);
                }
                ApiMsg::ExternalEvent(evt) => {
                    self.notifier.external_event(evt);
                }
                ApiMsg::ClearNamespace(namespace_id) => {
                    self.resource_cache.clear_namespace(namespace_id);
                    let document_ids = self.documents
                        .keys()
                        .filter(|did| did.0 == namespace_id)
                        .cloned()
                        .collect::<Vec<_>>();
                    for document in document_ids {
                        self.documents.remove(&document);
                    }
                }
                ApiMsg::MemoryPressure => {
                    // This is drastic. It will basically flush everything out of the cache,
                    // and the next frame will have to rebuild all of its resources.
                    // We may want to look into something less extreme, but on the other hand this
                    // should only be used in situations where are running low enough on memory
                    // that we risk crashing if we don't do something about it.
                    // The advantage of clearing the cache completely is that it gets rid of any
                    // remaining fragmentation that could have persisted if we kept around the most
                    // recently used resources.
                    self.resource_cache.clear(ClearCache::all());

                    let pending_update = self.resource_cache.pending_updates();
                    let msg = ResultMsg::UpdateResources {
                        updates: pending_update,
                        cancel_rendering: true,
                    };
                    self.result_tx.send(msg).unwrap();
                    self.notifier.wake_up();
                }
                ApiMsg::DebugCommand(option) => {
                    let msg = match option {
                        DebugCommand::EnableDualSourceBlending(enable) => {
                            // Set in the config used for any future documents
                            // that are created.
                            self.frame_config
                                .dual_source_blending_is_enabled = enable;

                            // Set for any existing documents.
                            for (_, doc) in &mut self.documents {
                                doc.frame_ctx
                                   .frame_builder_config
                                   .dual_source_blending_is_enabled = enable;
                            }

                            // We don't want to forward this message to the renderer.
                            return true;
                        }
                        DebugCommand::FetchDocuments => {
                            let json = self.get_docs_for_debugger();
                            ResultMsg::DebugOutput(DebugOutput::FetchDocuments(json))
                        }
                        DebugCommand::FetchClipScrollTree => {
                            let json = self.get_clip_scroll_tree_for_debugger();
                            ResultMsg::DebugOutput(DebugOutput::FetchClipScrollTree(json))
                        }
                        #[cfg(feature = "capture")]
                        DebugCommand::SaveCapture(root, bits) => {
                            let output = self.save_capture(root, bits, profile_counters);
                            ResultMsg::DebugOutput(output)
                        },
                        #[cfg(feature = "replay")]
                        DebugCommand::LoadCapture(root, tx) => {
                            NEXT_NAMESPACE_ID.fetch_add(1, Ordering::Relaxed);
                            *frame_counter += 1;

                            self.load_capture(&root, profile_counters);

                            for (id, doc) in &self.documents {
                                let captured = CapturedDocument {
                                    document_id: *id,
                                    root_pipeline_id: doc.scene.root_pipeline_id,
                                    window_size: doc.view.window_size,
                                };
                                tx.send(captured).unwrap();
                            }
                            // Note: we can't pass `LoadCapture` here since it needs to arrive
                            // before the `PublishDocument` messages sent by `load_capture`.
                            return true;
                        }
                        DebugCommand::ClearCaches(mask) => {
                            self.resource_cache.clear(mask);
                            return true;
                        }
                        _ => ResultMsg::DebugCommand(option),
                    };
                    self.result_tx.send(msg).unwrap();
                    self.notifier.wake_up();
                }
                ApiMsg::ShutDown => {
                    return false;
                }
                ApiMsg::UpdateDocument(document_id, doc_msgs) => {
                    self.update_document(
                        document_id,
                        doc_msgs,
                        frame_counter,
                        profile_counters
                    )
                }
            }

        true
    }

    fn update_document(
        &mut self,
        document_id: DocumentId,
        mut txn: TransactionMsg,
        frame_counter: &mut u32,
        profile_counters: &mut BackendProfileCounters,
    ) {
        let mut op = DocumentOps::nop();

        // TODO: This is a little awkward that we are applying prefix ops here, this
        // will most likely change soon.
        let prefix_ops = replace(&mut txn.prefix_ops, Vec::new());
        for doc_msg in prefix_ops {
            let _timer = profile_counters.total_time.timer();
            op.combine(
                self.process_document(
                    document_id,
                    doc_msg,
                    *frame_counter,
                    &mut profile_counters.ipc,
                )
            );
        }

        if txn.use_scene_builder_thread && !txn.is_empty() {
            let doc = self.documents.get_mut(&document_id).unwrap();
            doc.forward_transaction_to_scene_builder(
                txn,
                &op,
                document_id,
                &self.resource_cache,
                &self.scene_tx,
            );

            return;
        }

        self.resource_cache.update_resources(
            txn.resource_updates,
            &mut profile_counters.resources,
        );

        for doc_msg in txn.postfix_ops {
            let _timer = profile_counters.total_time.timer();
            op.combine(
                self.process_document(
                    document_id,
                    doc_msg,
                    *frame_counter,
                    &mut profile_counters.ipc,
                )
            );
        }

        let doc = self.documents.get_mut(&document_id).unwrap();

        if txn.generate_frame {
            if let Some(ref mut ros) = doc.render_on_scroll {
                *ros = true;
            }

            if doc.scene.root_pipeline_id.is_some() {
                op.render = true;
                op.composite = true;
            }
        }

        debug_assert!(op.render || !op.composite);

        if op.build {
            let _timer = profile_counters.total_time.timer();
            profile_scope!("build scene");

            doc.build_scene(&mut self.resource_cache);
            doc.render_on_hittest = true;
        }

        if !doc.can_render() {
            // WIP: this happens if we are building the first scene asynchronously and
            // scroll at the same time. we should keep track of the fact that we skipped
            // composition here and do it as soon as we receive the scene.
            op.render = false;
            op.composite = false;
        }

        if op.render {
            profile_scope!("generate frame");

            *frame_counter += 1;

            // borrow ck hack for profile_counters
            let (pending_update, rendered_document) = {
                let _timer = profile_counters.total_time.timer();

                let rendered_document = doc.render(
                    &mut self.resource_cache,
                    &mut self.gpu_cache,
                    &mut profile_counters.resources,
                );

                debug!("generated frame for document {:?} with {} passes",
                    document_id, rendered_document.frame.passes.len());

                let msg = ResultMsg::UpdateGpuCache(self.gpu_cache.extract_updates());
                self.result_tx.send(msg).unwrap();

                let pending_update = self.resource_cache.pending_updates();
                (pending_update, rendered_document)
            };

            // Publish the frame
            let msg = ResultMsg::PublishDocument(
                document_id,
                rendered_document,
                pending_update,
                profile_counters.clone()
            );
            self.result_tx.send(msg).unwrap();
            profile_counters.reset();
            doc.render_on_hittest = false;
        }

        if op.render || op.scroll {
            self.notifier.new_document_ready(document_id, op.scroll, op.composite);
        }
    }

    #[cfg(not(feature = "debugger"))]
    fn get_docs_for_debugger(&self) -> String {
        String::new()
    }

    #[cfg(feature = "debugger")]
    fn traverse_items<'a>(
        &self,
        traversal: &mut BuiltDisplayListIter<'a>,
        node: &mut debug_server::TreeNode,
    ) {
        loop {
            let subtraversal = {
                let item = match traversal.next() {
                    Some(item) => item,
                    None => break,
                };

                match *item.item() {
                    display_item @ SpecificDisplayItem::PushStackingContext(..) => {
                        let mut subtraversal = item.sub_iter();
                        let mut child_node =
                            debug_server::TreeNode::new(&display_item.debug_string());
                        self.traverse_items(&mut subtraversal, &mut child_node);
                        node.add_child(child_node);
                        Some(subtraversal)
                    }
                    SpecificDisplayItem::PopStackingContext => {
                        return;
                    }
                    display_item => {
                        node.add_item(&display_item.debug_string());
                        None
                    }
                }
            };

            // If flatten_item created a sub-traversal, we need `traversal` to have the
            // same state as the completed subtraversal, so we reinitialize it here.
            if let Some(subtraversal) = subtraversal {
                *traversal = subtraversal;
            }
        }
    }

    #[cfg(feature = "debugger")]
    fn get_docs_for_debugger(&self) -> String {
        let mut docs = debug_server::DocumentList::new();

        for (_, doc) in &self.documents {
            let mut debug_doc = debug_server::TreeNode::new("document");

            for (_, pipeline) in &doc.scene.pipelines {
                let mut debug_dl = debug_server::TreeNode::new("display-list");
                self.traverse_items(&mut pipeline.display_list.iter(), &mut debug_dl);
                debug_doc.add_child(debug_dl);
            }

            docs.add(debug_doc);
        }

        serde_json::to_string(&docs).unwrap()
    }

    #[cfg(not(feature = "debugger"))]
    fn get_clip_scroll_tree_for_debugger(&self) -> String {
        String::new()
    }

    #[cfg(feature = "debugger")]
    fn get_clip_scroll_tree_for_debugger(&self) -> String {
        let mut debug_root = debug_server::ClipScrollTreeList::new();

        for (_, doc) in &self.documents {
            let debug_node = debug_server::TreeNode::new("document clip-scroll tree");
            let mut builder = debug_server::TreeNodeBuilder::new(debug_node);

            // TODO(gw): Restructure the storage of clip-scroll tree, clip store
            //           etc so this isn't so untidy.
            let clip_store = &doc.frame_builder.as_ref().unwrap().clip_store;
            doc.frame_ctx
                .get_clip_scroll_tree()
                .print_with(clip_store, &mut builder);

            debug_root.add(builder.build());
        }

        serde_json::to_string(&debug_root).unwrap()
    }
}

#[cfg(feature = "debugger")]
trait ToDebugString {
    fn debug_string(&self) -> String;
}

#[cfg(feature = "debugger")]
impl ToDebugString for SpecificDisplayItem {
    fn debug_string(&self) -> String {
        match *self {
            SpecificDisplayItem::Image(..) => String::from("image"),
            SpecificDisplayItem::YuvImage(..) => String::from("yuv_image"),
            SpecificDisplayItem::Text(..) => String::from("text"),
            SpecificDisplayItem::Rectangle(..) => String::from("rectangle"),
            SpecificDisplayItem::ClearRectangle => String::from("clear_rectangle"),
            SpecificDisplayItem::Line(..) => String::from("line"),
            SpecificDisplayItem::Gradient(..) => String::from("gradient"),
            SpecificDisplayItem::RadialGradient(..) => String::from("radial_gradient"),
            SpecificDisplayItem::BoxShadow(..) => String::from("box_shadow"),
            SpecificDisplayItem::Border(..) => String::from("border"),
            SpecificDisplayItem::PushStackingContext(..) => String::from("push_stacking_context"),
            SpecificDisplayItem::Iframe(..) => String::from("iframe"),
            SpecificDisplayItem::Clip(..) => String::from("clip"),
            SpecificDisplayItem::ClipChain(..) => String::from("clip_chain"),
            SpecificDisplayItem::ScrollFrame(..) => String::from("scroll_frame"),
            SpecificDisplayItem::StickyFrame(..) => String::from("sticky_frame"),
            SpecificDisplayItem::SetGradientStops => String::from("set_gradient_stops"),
            SpecificDisplayItem::PopStackingContext => String::from("pop_stacking_context"),
            SpecificDisplayItem::PushShadow(..) => String::from("push_shadow"),
            SpecificDisplayItem::PopAllShadows => String::from("pop_all_shadows"),
        }
    }
}

impl RenderBackend {
    #[cfg(feature = "capture")]
    // Note: the mutable `self` is only needed here for resolving blob images
    fn save_capture(
        &mut self,
        root: PathBuf,
        bits: CaptureBits,
        profile_counters: &mut BackendProfileCounters,
    ) -> DebugOutput {
        use capture::CaptureConfig;

        debug!("capture: saving {:?}", root);
        let (resources, deferred) = self.resource_cache.save_capture(&root);
        let config = CaptureConfig::new(root, bits);

        for (&id, doc) in &mut self.documents {
            debug!("\tdocument {:?}", id);
            if config.bits.contains(CaptureBits::SCENE) {
                let file_name = format!("scene-{}-{}", (id.0).0, id.1);
                config.serialize(&doc.scene, file_name);
            }
            if config.bits.contains(CaptureBits::FRAME) {
                let rendered_document = doc.render(
                    &mut self.resource_cache,
                    &mut self.gpu_cache,
                    &mut profile_counters.resources,
                );
                //TODO: write down full `RenderedDocument`?
                // it has `pipeline_epoch_map` and `layers_bouncing_back`,
                // which may capture necessary details for some cases.
                let file_name = format!("frame-{}-{}", (id.0).0, id.1);
                config.serialize(&rendered_document.frame, file_name);
            }
        }

        info!("\tbackend");
        let backend = PlainRenderBackend {
            default_device_pixel_ratio: self.default_device_pixel_ratio,
            enable_render_on_scroll: self.enable_render_on_scroll,
            frame_config: self.frame_config.clone(),
            documents: self.documents
                .iter()
                .map(|(id, doc)| (*id, doc.view.clone()))
                .collect(),
            resources,
        };

        config.serialize(&backend, "backend");

        if config.bits.contains(CaptureBits::FRAME) {
            // After we rendered the frames, there are pending updates to both
            // GPU cache and resources. Instead of serializing them, we are going to make sure
            // they are applied on the `Renderer` side.
            let msg_update_gpu_cache = ResultMsg::UpdateGpuCache(self.gpu_cache.extract_updates());
            self.result_tx.send(msg_update_gpu_cache).unwrap();
            let msg_update_resources = ResultMsg::UpdateResources {
                updates: self.resource_cache.pending_updates(),
                cancel_rendering: false,
            };
            self.result_tx.send(msg_update_resources).unwrap();
            // Save the texture/glyph/image caches.
            info!("\tresource cache");
            let caches = self.resource_cache.save_caches(&config.root);
            config.serialize(&caches, "resource_cache");
            info!("\tgpu cache");
            config.serialize(&self.gpu_cache, "gpu_cache");
        }

        DebugOutput::SaveCapture(config, deferred)
    }

    #[cfg(feature = "replay")]
    fn load_capture(
        &mut self,
        root: &PathBuf,
        profile_counters: &mut BackendProfileCounters,
    ) {
        use capture::CaptureConfig;
        use tiling::Frame;

        debug!("capture: loading {:?}", root);
        let backend = CaptureConfig::deserialize::<PlainRenderBackend, _>(root, "backend")
            .expect("Unable to open backend.ron");
        let caches_maybe = CaptureConfig::deserialize::<PlainCacheOwn, _>(root, "resource_cache");

        // Note: it would be great to have `RenderBackend` to be split
        // rather explicitly on what's used before and after scene building
        // so that, for example, we never miss anything in the code below:

        let plain_externals = self.resource_cache.load_capture(backend.resources, caches_maybe, root);
        let msg_load = ResultMsg::DebugOutput(
            DebugOutput::LoadCapture(root.clone(), plain_externals)
        );
        self.result_tx.send(msg_load).unwrap();

        self.gpu_cache = match CaptureConfig::deserialize::<GpuCache, _>(root, "gpu_cache") {
            Some(gpu_cache) => gpu_cache,
            None => GpuCache::new(),
        };

        self.documents.clear();
        self.default_device_pixel_ratio = backend.default_device_pixel_ratio;
        self.frame_config = backend.frame_config;
        self.enable_render_on_scroll = backend.enable_render_on_scroll;

        for (id, view) in backend.documents {
            debug!("\tdocument {:?}", id);
            let scene_name = format!("scene-{}-{}", (id.0).0, id.1);
            let scene = CaptureConfig::deserialize::<Scene, _>(root, &scene_name)
                .expect(&format!("Unable to open {}.ron", scene_name));

            let mut doc = Document {
                scene,
                view,
                frame_ctx: FrameContext::new(self.frame_config.clone()),
                frame_builder: Some(FrameBuilder::empty()),
                output_pipelines: FastHashSet::default(),
                render_on_scroll: None,
                render_on_hittest: false,
                removed_pipelines: Vec::new(),
                dynamic_properties: SceneProperties::new(),
                hit_tester: None,
            };

            let frame_name = format!("frame-{}-{}", (id.0).0, id.1);
            let render_doc = match CaptureConfig::deserialize::<Frame, _>(root, frame_name) {
                Some(frame) => {
                    info!("\tloaded a built frame with {} passes", frame.passes.len());
                    doc.frame_ctx.make_rendered_document(frame, Vec::new())
                }
                None => {
                    doc.build_scene(&mut self.resource_cache);
                    doc.render(
                        &mut self.resource_cache,
                        &mut self.gpu_cache,
                        &mut profile_counters.resources,
                    )
                }
            };

            let msg_publish = ResultMsg::PublishDocument(
                id,
                render_doc,
                self.resource_cache.pending_updates(),
                profile_counters.clone(),
            );
            self.result_tx.send(msg_publish).unwrap();
            profile_counters.reset();

            self.notifier.new_document_ready(id, false, true);
            self.documents.insert(id, doc);
        }
    }
}

// Message from render backend to scene builder.
pub enum SceneBuilderRequest {
    Transaction {
        document_id: DocumentId,
        scene: Option<SceneRequest>,
        resource_updates: ResourceUpdates,
        document_ops: Vec<DocumentMsg>,
        render: bool,
    },
    Stop
}

// Message from scene builder to render backend.
pub enum SceneBuilderMsg {
    Transaction {
        document_id: DocumentId,
        built_scene: Option<BuiltScene>,
        resource_updates: ResourceUpdates,
        document_ops: Vec<DocumentMsg>,
        render: bool,
    },
}

/// Contains the the render backend data needed to build a scene.
pub struct SceneRequest {
    pub scene: Scene,
    pub view: DocumentView,
    pub font_instances: FontInstanceMap,
    pub tiled_image_map: TiledImageMap,
    pub output_pipelines: FastHashSet<PipelineId>,
}

pub struct BuiltScene {
    pub frame_builder: FrameBuilder,
    pub clip_scroll_tree: ClipScrollTree,
    pub pipeline_epoch_map: FastHashMap<PipelineId, Epoch>,
    pub removed_pipelines: Vec<PipelineId>,
}

pub struct SceneBuilder {
    rx: Receiver<SceneBuilderRequest>,
    tx: Sender<SceneBuilderMsg>,
    api_tx: MsgSender<ApiMsg>,
    config: FrameBuilderConfig,
}

impl SceneBuilder {
    pub fn new(
        config: FrameBuilderConfig,
        api_tx: MsgSender<ApiMsg>
    ) -> (Self, Sender<SceneBuilderRequest>, Receiver<SceneBuilderMsg>) {
        let (in_tx, in_rx) = channel();
        let (out_tx, out_rx) = channel();
        (
            SceneBuilder {
                rx: in_rx,
                tx: out_tx,
                api_tx,
                config,
            },
            in_tx,
            out_rx,
        )
    }

    pub fn run(&mut self) {
        loop {
            match self.rx.recv() {
                Ok(msg) => {
                    if !self.process_message(msg) {
                        return;
                    }
                }
                Err(_) => {
                    return;
                }
            }
        }
    }

    pub fn process_message(&mut self, msg: SceneBuilderRequest) -> bool {
        match msg {
            SceneBuilderRequest::Transaction {
                document_id,
                scene,
                resource_updates,
                document_ops,
                render,
            } => {
                let built_scene = scene.map(|request|{
                    self.build_scene(request)
                });

                // TODO: pre-rasterization.

                self.tx.send(SceneBuilderMsg::Transaction {
                    document_id,
                    built_scene,
                    resource_updates,
                    document_ops,
                    render,
                }).unwrap();

                let _ = self.api_tx.send(ApiMsg::WakeUp);
            }
            SceneBuilderRequest::Stop => { return false; }
        }

        true
    }

    pub fn build_scene(&mut self, request: SceneRequest) -> BuiltScene {
        FrameContext::create_frame_builder_async(&self.config, request)
    }
}