/* This Source Code Form is subject to the terms of the Mozilla Publi
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{AsyncBlobImageRasterizer, BlobImageResult, DebugFlags, Parameter};
use api::{DocumentId, PipelineId, ExternalEvent, BlobImageRequest};
use api::{NotificationRequest, Checkpoint, IdNamespace, QualitySettings, RenderBackendId};
use api::{GlyphDimensionRequest, GlyphIndexRequest};
use api::channel::{unbounded_channel, single_msg_channel, Receiver, Sender};
use api::units::*;
use crate::render_api::{ApiMsg, FrameMsg, SceneMsg, ResourceUpdate, TransactionMsg, MemoryReport};
use crate::box_shadow::BoxShadow;
use crate::prim_store::rectangle::RectanglePrim;
#[cfg(feature = "capture")]
use crate::capture::CaptureConfig;
use crate::frame_builder::FrameBuilderConfig;
use crate::scene_building::{SceneBuilder, SceneRecycler};
use crate::clip::{ClipIntern, PolygonIntern};
use crate::filterdata::FilterDataIntern;
use glyph_rasterizer::SharedFontResources;
use crate::intern::{Internable, Interner, UpdateList};
use crate::internal_types::{FastHashMap, FastHashSet};
use malloc_size_of::{MallocSizeOf, MallocSizeOfOps};
use crate::prim_store::backdrop::{BackdropCapture, BackdropRender};
use crate::prim_store::borders::{ImageBorder, NormalBorderPrim};
use crate::prim_store::gradient::{LinearGradient, RadialGradient, ConicGradient};
use crate::prim_store::image::{Image, YuvImage};
use crate::prim_store::line_dec::LineDecoration;
use crate::prim_store::picture::Picture;
use crate::prim_store::text_run::TextRun;
use crate::profiler::{self, TransactionProfile};
use crate::render_backend::SceneView;
use crate::renderer::{FullFrameStats, PipelineInfo};
use crate::scene::{BuiltScene, Scene, SceneStats};
use crate::spatial_tree::{SceneSpatialTree, SpatialTreeUpdates};
use crate::telemetry::Telemetry;
use crate::SceneBuilderHooks;
use std::iter;
use crate::util::drain_filter;
use std::thread;
use std::time::Duration;

fn rasterize_blobs(txn: &mut TransactionMsg, is_low_priority: bool, tile_pool: &mut api::BlobTilePool) {
    tracy_rs::profile_scope!("rasterize_blobs");

    if let Some(ref mut rasterizer) = txn.blob_rasterizer {
        let mut rasterized_blobs = rasterizer.rasterize(&txn.blob_requests, is_low_priority, tile_pool);
        // try using the existing allocation if our current list is empty
        if txn.rasterized_blobs.is_empty() {
            txn.rasterized_blobs = rasterized_blobs;
        } else {
            txn.rasterized_blobs.append(&mut rasterized_blobs);
        }
    }
}

/// Represent the remaining work associated to a transaction after the scene building
/// phase as well as the result of scene building itself if applicable.
pub struct BuiltTransaction {
    pub document_id: DocumentId,
    pub built_scene: Option<BuiltScene>,
    pub offscreen_scenes: Vec<OffscreenBuiltScene>,
    pub view: SceneView,
    pub resource_updates: Vec<ResourceUpdate>,
    pub rasterized_blobs: Vec<(BlobImageRequest, BlobImageResult)>,
    pub blob_rasterizer: Option<Box<dyn AsyncBlobImageRasterizer>>,
    pub frame_ops: Vec<FrameMsg>,
    pub removed_pipelines: Vec<(PipelineId, DocumentId)>,
    pub notifications: Vec<NotificationRequest>,
    pub interner_updates: Option<InternerUpdates>,
    pub spatial_tree_updates: Option<SpatialTreeUpdates>,
    pub render_frame: bool,
    pub present: bool,
    pub tracked: bool,
    pub invalidate_rendered_frame: bool,
    pub profile: TransactionProfile,
    pub frame_stats: FullFrameStats,
}

pub struct OffscreenBuiltScene {
    pub scene: BuiltScene,
    pub spatial_tree_updates: SpatialTreeUpdates,
}

#[cfg(feature = "replay")]
pub struct LoadScene {
    pub document_id: DocumentId,
    pub scene: Scene,
    pub fonts: SharedFontResources,
    pub view: SceneView,
    pub config: FrameBuilderConfig,
    pub build_frame: bool,
    pub interners: Interners,
    pub spatial_tree: SceneSpatialTree,
}

/// Message to the scene builder thread.
pub enum SceneBuilderRequest {
    Transactions(Vec<Box<TransactionMsg>>),
    /// Register a new document and the window that owns it. The SB uses
    /// the backend id to dispatch hook calls to the correct per-window
    /// `SceneBuilderHooks` instance (see `SetSceneBuilderHooks`).
    AddDocument(DocumentId, DeviceIntSize, RenderBackendId),
    DeleteDocument(DocumentId),
    GetGlyphDimensions(RenderBackendId, GlyphDimensionRequest),
    GetGlyphIndices(RenderBackendId, GlyphIndexRequest),
    ClearNamespace(RenderBackendId, IdNamespace),
    SimulateLongSceneBuild(u32),
    ExternalEvent(RenderBackendId, ExternalEvent),
    WakeUp,
    ShutDown(Option<Sender<()>>),
    /// Stop producing results for a window without unregistering it.
    ///
    /// Sent on the low-priority scene channel by `stop_render_backend` and
    /// forwarded to the render backend, so that it cannot overtake work queued
    /// earlier on either scene channel. See `RenderApi::stop_render_backend`.
    StopWindow(RenderBackendId, Sender<()>),
    Flush(Sender<()>),
    SetFlags(DebugFlags),
    SetFrameBuilderConfig(RenderBackendId, FrameBuilderConfig),
    SetParameter(RenderBackendId, Parameter),
    ReportMemory(Box<MemoryReport>, Sender<Box<MemoryReport>>),
    /// Install or clear the scene-builder hooks for the given window.
    /// `Some(hooks)` registers them (`hooks.register()` is invoked on the
    /// SB thread, which is the correct thread for APZ's
    /// `apz_register_updater`); `None` deregisters and removes them.
    /// Forwarded from the render backend on window register / unregister.
    SetSceneBuilderHooks(RenderBackendId, Option<Box<dyn SceneBuilderHooks + Send>>),
    #[cfg(feature = "capture")]
    SaveScene(CaptureConfig),
    #[cfg(feature = "replay")]
    LoadScenes(Vec<LoadScene>),
    #[cfg(feature = "capture")]
    StartCaptureSequence(CaptureConfig),
    #[cfg(feature = "capture")]
    StopCaptureSequence,
}

// Message from scene builder to render backend.
pub enum SceneBuilderResult {
    Transactions(Vec<Box<BuiltTransaction>>, Option<Sender<SceneSwapResult>>),
    ExternalEvent(RenderBackendId, ExternalEvent),
    FlushComplete(Sender<()>),
    DeleteDocument(DocumentId),
    ClearNamespace(RenderBackendId, IdNamespace),
    GetGlyphDimensions(RenderBackendId, GlyphDimensionRequest),
    GetGlyphIndices(RenderBackendId, GlyphIndexRequest),
    SetParameter(RenderBackendId, Parameter),
    /// Forwarded `SceneBuilderRequest::StopWindow`. The backend marks the window
    /// stopped and acks on the channel, releasing `stop_render_backend`.
    StopWindow(RenderBackendId, Sender<()>),
    ShutDown(Option<Sender<()>>),

    #[cfg(feature = "capture")]
    /// The same as `Transactions`, but also supplies a `CaptureConfig` that the
    /// render backend should use for sequence capture, until the next
    /// `CapturedTransactions` or `StopCaptureSequence` result.
    CapturedTransactions(Vec<Box<BuiltTransaction>>, CaptureConfig, Option<Sender<SceneSwapResult>>),

    #[cfg(feature = "capture")]
    /// The scene builder has stopped sequence capture, so the render backend
    /// should do the same.
    StopCaptureSequence,
}

// Message from render backend to scene builder to indicate the
// scene swap was completed. We need a separate channel for this
// so that they don't get mixed with SceneBuilderRequest messages.
pub enum SceneSwapResult {
    Complete(Sender<()>),
    Aborted,
}

macro_rules! declare_interners {
    ( $( $name:ident : $ty:ident, )+ ) => {
        /// This struct contains all items that can be shared between
        /// display lists. We want to intern and share the same clips,
        /// primitives and other things between display lists so that:
        /// - GPU cache handles remain valid, reducing GPU cache updates.
        /// - Comparison of primitives and pictures between two
        ///   display lists is (a) fast (b) done during scene building.
        #[cfg_attr(feature = "capture", derive(Serialize))]
        #[cfg_attr(feature = "replay", derive(Deserialize))]
        #[derive(Default)]
        pub struct Interners {
            $(
                pub $name: Interner<$ty>,
            )+
        }

        $(
            impl AsMut<Interner<$ty>> for Interners {
                fn as_mut(&mut self) -> &mut Interner<$ty> {
                    &mut self.$name
                }
            }
        )+

        pub struct InternerUpdates {
            $(
                pub $name: UpdateList<<$ty as Internable>::Key>,
            )+
        }

        impl Interners {
            /// Reports CPU heap memory used by the interners.
            fn report_memory(
                &self,
                ops: &mut MallocSizeOfOps,
                r: &mut MemoryReport,
            ) {
                $(
                    r.interning.interners.$name += self.$name.size_of(ops);
                )+
            }

            fn end_frame_and_get_pending_updates(&mut self) -> InternerUpdates {
                InternerUpdates {
                    $(
                        $name: self.$name.end_frame_and_get_pending_updates(),
                    )+
                }
            }
        }
    }
}

crate::enumerate_interners!(declare_interners);

// A document in the scene builder contains the current scene,
// as well as a persistent clip interner. This allows clips
// to be de-duplicated, and persisted in the GPU cache between
// display lists.
struct Document {
    scene: Scene,
    interners: Interners,
    stats: SceneStats,
    view: SceneView,
    spatial_tree: SceneSpatialTree,
}

impl Document {
    fn new(device_rect: DeviceIntRect) -> Self {
        Document {
            scene: Scene::new(),
            interners: Interners::default(),
            stats: SceneStats::empty(),
            spatial_tree: SceneSpatialTree::new(),
            view: SceneView {
                device_rect,
                quality_settings: QualitySettings::default(),
            },
        }
    }
}

pub struct SceneBuilderThread {
    documents: FastHashMap<DocumentId, Document>,
    /// Which window each document belongs to. Used to dispatch hook
    /// calls to the right per-window `SceneBuilderHooks` instance
    /// when more than one window shares this SB.
    doc_to_window: FastHashMap<DocumentId, RenderBackendId>,
    rx: Receiver<SceneBuilderRequest>,
    tx: Sender<ApiMsg>,
    /// Fallback config used when no per-window config has been registered
    /// yet (initial pool setup, before any window registers).
    config: FrameBuilderConfig,
    /// Per-window frame builder configs. Populated via
    /// `SceneBuilderRequest::SetFrameBuilderConfig(backend_id, cfg)` on
    /// window registration and on any subsequent change. Scene building
    /// looks up the right config via `doc_to_window`.
    window_configs: FastHashMap<RenderBackendId, FrameBuilderConfig>,
    fonts: SharedFontResources,
    size_of_ops: Option<MallocSizeOfOps>,
    /// Per-window scene-builder hooks. Empty for SBs with no registered
    /// hooks. Populated as `SetSceneBuilderHooks(id, Some(_))` arrives
    /// and pruned on `SetSceneBuilderHooks(id, None)` or window removal.
    hooks: FastHashMap<RenderBackendId, Box<dyn SceneBuilderHooks + Send>>,
    simulate_slow_ms: u32,
    removed_pipelines: FastHashSet<PipelineId>,
    #[cfg(feature = "capture")]
    capture_config: Option<CaptureConfig>,
    debug_flags: DebugFlags,
    recycler: SceneRecycler,
    tile_pool: api::BlobTilePool,
}

pub struct SceneBuilderThreadChannels {
    rx: Receiver<SceneBuilderRequest>,
    tx: Sender<ApiMsg>,
}

impl SceneBuilderThreadChannels {
    pub fn new(
        tx: Sender<ApiMsg>
    ) -> (Self, Sender<SceneBuilderRequest>) {
        let (in_tx, in_rx) = unbounded_channel();
        (
            Self {
                rx: in_rx,
                tx,
            },
            in_tx,
        )
    }
}

impl SceneBuilderThread {
    pub fn new(
        config: FrameBuilderConfig,
        fonts: SharedFontResources,
        size_of_ops: Option<MallocSizeOfOps>,
        channels: SceneBuilderThreadChannels,
    ) -> Self {
        let SceneBuilderThreadChannels { rx, tx } = channels;

        Self {
            documents: Default::default(),
            doc_to_window: Default::default(),
            rx,
            tx,
            config,
            window_configs: FastHashMap::default(),
            fonts,
            size_of_ops,
            // Hooks now arrive per-window via `SetSceneBuilderHooks`.
            hooks: FastHashMap::default(),
            simulate_slow_ms: 0,
            removed_pipelines: FastHashSet::default(),
            #[cfg(feature = "capture")]
            capture_config: None,
            debug_flags: DebugFlags::default(),
            recycler: SceneRecycler::new(),
            // TODO: tile size is hard-coded here.
            tile_pool: api::BlobTilePool::new(),
        }
    }

    /// Send a message to the render backend thread.
    ///
    /// We first put something in the result queue and then send a wake-up
    /// message to the api queue that the render backend is blocking on.
    pub fn send(&self, msg: SceneBuilderResult) {
        self.tx.send(ApiMsg::SceneBuilderResult(msg)).unwrap();
    }

    /// The scene builder thread's event loop.
    pub fn run(&mut self) {
        // Hooks register / deregister are tied to window registration
        // (via `SceneBuilderRequest::SetSceneBuilderHooks`), not to the
        // SB thread lifecycle. The SB thread can outlive any single
        // window (shared pool), and can serve multiple windows
        // simultaneously.

        loop {
            tracy_rs::tracy_begin_frame!("scene_builder_thread");

            match self.rx.recv() {
                Ok(SceneBuilderRequest::WakeUp) => {}
                Ok(SceneBuilderRequest::Flush(tx)) => {
                    self.send(SceneBuilderResult::FlushComplete(tx));
                }
                Ok(SceneBuilderRequest::StopWindow(backend_id, tx)) => {
                    self.send(SceneBuilderResult::StopWindow(backend_id, tx));
                }
                Ok(SceneBuilderRequest::SetFlags(debug_flags)) => {
                    self.debug_flags = debug_flags;
                }
                Ok(SceneBuilderRequest::Transactions(txns)) => {
                    let built_txns : Vec<Box<BuiltTransaction>> = txns.into_iter()
                        .map(|txn| self.process_transaction(*txn))
                        .collect();
                    #[cfg(feature = "capture")]
                    match built_txns.iter().any(|txn| txn.built_scene.is_some()) {
                        true => self.save_capture_sequence(),
                        _ => {},
                    }
                    self.forward_built_transactions(built_txns);

                    // Now that we off the critical path, do some memory bookkeeping.
                    self.recycler.recycle_built_scene();
                    self.tile_pool.cleanup();
                }
                Ok(SceneBuilderRequest::AddDocument(document_id, initial_size, backend_id)) => {
                    let old = self.documents.insert(document_id, Document::new(
                        initial_size.into(),
                    ));
                    debug_assert!(old.is_none());
                    self.doc_to_window.insert(document_id, backend_id);
                }
                Ok(SceneBuilderRequest::DeleteDocument(document_id)) => {
                    self.documents.remove(&document_id);
                    self.doc_to_window.remove(&document_id);
                    self.send(SceneBuilderResult::DeleteDocument(document_id));
                }
                Ok(SceneBuilderRequest::ClearNamespace(backend_id, id)) => {
                    self.documents.retain(|doc_id, _doc| doc_id.namespace_id != id);
                    self.doc_to_window.retain(|doc_id, _| doc_id.namespace_id != id);
                    self.send(SceneBuilderResult::ClearNamespace(backend_id, id));
                }
                Ok(SceneBuilderRequest::ExternalEvent(backend_id, evt)) => {
                    self.send(SceneBuilderResult::ExternalEvent(backend_id, evt));
                }
                Ok(SceneBuilderRequest::GetGlyphDimensions(backend_id, request)) => {
                    self.send(SceneBuilderResult::GetGlyphDimensions(backend_id, request));
                }
                Ok(SceneBuilderRequest::GetGlyphIndices(backend_id, request)) => {
                    self.send(SceneBuilderResult::GetGlyphIndices(backend_id, request));
                }
                Ok(SceneBuilderRequest::ShutDown(sync)) => {
                    self.send(SceneBuilderResult::ShutDown(sync));
                    break;
                }
                Ok(SceneBuilderRequest::SimulateLongSceneBuild(time_ms)) => {
                    self.simulate_slow_ms = time_ms
                }
                Ok(SceneBuilderRequest::ReportMemory(mut report, tx)) => {
                    (*report) += self.report_memory();
                    tx.send(report).unwrap();
                }
                Ok(SceneBuilderRequest::SetFrameBuilderConfig(backend_id, cfg)) => {
                    self.window_configs.insert(backend_id, cfg);
                }
                Ok(SceneBuilderRequest::SetParameter(backend_id, prop)) => {
                    self.send(SceneBuilderResult::SetParameter(backend_id, prop));
                }
                Ok(SceneBuilderRequest::SetSceneBuilderHooks(backend_id, hooks)) => {
                    if let Some(old) = self.hooks.remove(&backend_id) {
                        old.deregister();
                    }
                    if let Some(new_hooks) = hooks {
                        new_hooks.register();
                        self.hooks.insert(backend_id, new_hooks);
                    }
                }
                #[cfg(feature = "replay")]
                Ok(SceneBuilderRequest::LoadScenes(msg)) => {
                    self.load_scenes(msg);
                }
                #[cfg(feature = "capture")]
                Ok(SceneBuilderRequest::SaveScene(config)) => {
                    self.save_scene(config);
                }
                #[cfg(feature = "capture")]
                Ok(SceneBuilderRequest::StartCaptureSequence(config)) => {
                    self.start_capture_sequence(config);
                }
                #[cfg(feature = "capture")]
                Ok(SceneBuilderRequest::StopCaptureSequence) => {
                    // FIXME(aosmond): clear config for frames and resource cache without scene
                    // rebuild?
                    self.capture_config = None;
                    self.send(SceneBuilderResult::StopCaptureSequence);
                }
                Err(..) => {
                    break;
                }
            }

            // Poke every registered window's hooks (e.g. APZ's
            // `apz_run_updater`). Each window's updater drives its own
            // APZ tree.
            for (_, hooks) in self.hooks.iter() {
                hooks.poke();
            }

            tracy_rs::tracy_end_frame!("scene_builder_thread");
        }

        // SB is exiting; deregister any still-installed hooks. In a
        // clean shutdown the RB has already sent `SetSceneBuilderHooks(_, None)`
        // for each window, so this is usually empty. The drain is here
        // as a safety net.
        for (_, hooks) in self.hooks.drain() {
            hooks.deregister();
        }
    }

    #[cfg(feature = "capture")]
    fn save_scene(&mut self, config: CaptureConfig) {
        for (id, doc) in &self.documents {
            let interners_name = format!("interners-{}-{}", id.namespace_id.0, id.id);
            config.serialize_for_scene(&doc.interners, interners_name);

            let scene_spatial_tree_name = format!("scene-spatial-tree-{}-{}", id.namespace_id.0, id.id);
            config.serialize_for_scene(&doc.spatial_tree, scene_spatial_tree_name);

            use crate::render_api::CaptureBits;
            if config.bits.contains(CaptureBits::SCENE) {
                let file_name = format!("scene-{}-{}", id.namespace_id.0, id.id);
                config.serialize_for_scene(&doc.scene, file_name);
            }
        }
    }

    #[cfg(feature = "replay")]
    fn load_scenes(&mut self, scenes: Vec<LoadScene>) {
        for mut item in scenes {
            self.config = item.config;

            let mut built_scene = None;
            let mut interner_updates = None;
            let mut spatial_tree_updates = None;

            if item.scene.has_root_pipeline() {
                built_scene = Some(SceneBuilder::build(
                    &item.scene,
                    None,
                    item.fonts,
                    &item.view,
                    &self.config,
                    &mut item.interners,
                    &mut item.spatial_tree,
                    &mut self.recycler,
                    &SceneStats::empty(),
                    self.debug_flags,
                ));

                interner_updates = Some(
                    item.interners.end_frame_and_get_pending_updates()
                );

                spatial_tree_updates = Some(
                    item.spatial_tree.end_frame_and_get_pending_updates()
                );
            }

            self.documents.insert(
                item.document_id,
                Document {
                    scene: item.scene,
                    interners: item.interners,
                    stats: SceneStats::empty(),
                    view: item.view.clone(),
                    spatial_tree: item.spatial_tree,
                },
            );

            let txns = vec![Box::new(BuiltTransaction {
                document_id: item.document_id,
                render_frame: item.build_frame,
                tracked: false,
                present: true,
                invalidate_rendered_frame: false,
                built_scene,
                view: item.view,
                resource_updates: Vec::new(),
                rasterized_blobs: Vec::new(),
                blob_rasterizer: None,
                frame_ops: Vec::new(),
                removed_pipelines: Vec::new(),
                notifications: Vec::new(),
                interner_updates,
                spatial_tree_updates,
                profile: TransactionProfile::new(),
                frame_stats: FullFrameStats::default(),
                offscreen_scenes: Vec::new(),
            })];

            self.forward_built_transactions(txns);
        }
    }

    #[cfg(feature = "capture")]
    fn save_capture_sequence(
        &mut self,
    ) {
        if let Some(ref mut config) = self.capture_config {
            config.prepare_scene();
            for (id, doc) in &self.documents {
                let interners_name = format!("interners-{}-{}", id.namespace_id.0, id.id);
                config.serialize_for_scene(&doc.interners, interners_name);

                let scene_spatial_tree_name = format!("scene-spatial-tree-{}-{}", id.namespace_id.0, id.id);
                config.serialize_for_scene(&doc.spatial_tree, scene_spatial_tree_name);

                use crate::render_api::CaptureBits;
                if config.bits.contains(CaptureBits::SCENE) {
                    let file_name = format!("scene-{}-{}", id.namespace_id.0, id.id);
                    config.serialize_for_scene(&doc.scene, file_name);
                }
            }
        }
    }

    #[cfg(feature = "capture")]
    fn start_capture_sequence(
        &mut self,
        config: CaptureConfig,
    ) {
        self.capture_config = Some(config);
        self.save_capture_sequence();
    }

    /// Do the bulk of the work of the scene builder thread.
    fn process_transaction(&mut self, mut txn: TransactionMsg) -> Box<BuiltTransaction> {
        tracy_rs::profile_scope!("process_transaction");
        let win_id_opt = self.doc_to_window.get(&txn.document_id).copied();

        // Dispatch pre_scene_build to the hooks for this transaction's
        // owning window. With multi-window per SB this is the only way
        // to avoid calling the wrong window's APZ.
        if let Some(win_id) = win_id_opt {
            if let Some(hooks) = self.hooks.get(&win_id) {
                hooks.pre_scene_build();
            }
        }

        let config: FrameBuilderConfig = win_id_opt
            .and_then(|id| self.window_configs.get(&id))
            .unwrap_or(&self.config)
            .clone();

        let doc = self.documents.get_mut(&txn.document_id).unwrap();
        let scene = &mut doc.scene;

        let mut profile = txn.profile.take();

        let scene_build_start = zeitstempel::now();
        let mut removed_pipelines = Vec::new();
        let mut rebuild_scene = false;
        let mut frame_stats = FullFrameStats::default();
        let mut offscreen_scenes = Vec::new();

        for message in txn.scene_ops.drain(..) {
            match message {
                SceneMsg::UpdateEpoch(pipeline_id, epoch) => {
                    scene.update_epoch(pipeline_id, epoch);
                }
                SceneMsg::SetQualitySettings { settings } => {
                    doc.view.quality_settings = settings;
                }
                SceneMsg::SetDocumentView { device_rect } => {
                    doc.view.device_rect = device_rect;
                }
                SceneMsg::SetDisplayList {
                    epoch,
                    pipeline_id,
                    display_list,
                    namespace,
                } => {
                    let (builder_start_time_ns, builder_end_time_ns, send_time_ns) =
                      display_list.times();
                    let content_send_time = profiler::ns_to_ms(zeitstempel::now() - send_time_ns);
                    let dl_build_time = profiler::ns_to_ms(builder_end_time_ns - builder_start_time_ns);
                    profile.set(profiler::CONTENT_SEND_TIME, content_send_time);
                    profile.set(profiler::DISPLAY_LIST_BUILD_TIME, dl_build_time);
                    profile.set(profiler::DISPLAY_LIST_MEM, profiler::bytes_to_mb(display_list.size_in_bytes()));
                    // Should be zero: see profiler::OFF_GRID_COORDS.
                    profile.add(profiler::OFF_GRID_COORDS, display_list.off_grid_coords());

                    let (gecko_display_list_time, full_display_list) = display_list.gecko_display_list_stats();
                    frame_stats.full_display_list = full_display_list;
                    frame_stats.gecko_display_list_time = gecko_display_list_time;
                    frame_stats.wr_display_list_time += dl_build_time;

                    if self.removed_pipelines.contains(&pipeline_id) {
                        continue;
                    }

                    // Note: We could further reduce the amount of unnecessary scene
                    // building by keeping track of which pipelines are used by the
                    // scene (bug 1490751).
                    rebuild_scene = true;

                    scene.set_display_list(
                        pipeline_id,
                        epoch,
                        namespace,
                        display_list,
                    );
                }
                SceneMsg::RenderOffscreen(pipeline_id) => {
                    // Intern the one-shot offscreen scene into the document's
                    // interners so it dedups against existing items and its
                    // handles are valid against the document's data store
                    // (immutable at frame-build time), which the offscreen
                    // frame build borrows rather than allocating a throwaway
                    // store. The interned items are not flushed here; the main
                    // scene's end_frame below emits a single combined delta
                    // (main + offscreen) which the backend applies to the
                    // document data store before processing the offscreen
                    // scene. The transient offscreen items are evicted by a
                    // later end_frame GC once the temporary pipeline is gone.
                    //
                    // This relies on a main scene rebuild happening in the same
                    // transaction; the Gecko caller always issues a
                    // SetDisplayList immediately before RenderOffscreen (see
                    // WebRenderBridgeParent), and the debug_assert after this
                    // loop enforces it.
                    let mut spatial_tree = SceneSpatialTree::new();
                    let built = SceneBuilder::build(
                        &scene,
                        Some(pipeline_id),
                        self.fonts.clone(),
                        &doc.view,
                        &config,
                        &mut doc.interners,
                        &mut spatial_tree,
                        &mut self.recycler,
                        &doc.stats,
                        self.debug_flags,
                    );
                    let spatial_tree_updates = spatial_tree.end_frame_and_get_pending_updates();
                    offscreen_scenes.push(OffscreenBuiltScene {
                        scene: built,
                        spatial_tree_updates,
                    });
                }
                SceneMsg::SetRootPipeline(pipeline_id) => {
                    if scene.root_pipeline_id != Some(pipeline_id) {
                        rebuild_scene = true;
                        scene.set_root_pipeline_id(pipeline_id);
                    }
                }
                SceneMsg::RemovePipeline(pipeline_id) => {
                    scene.remove_pipeline(pipeline_id);
                    self.removed_pipelines.insert(pipeline_id);
                    removed_pipelines.push((pipeline_id, txn.document_id));
                }
            }
        }

        self.removed_pipelines.clear();

        let mut built_scene = None;
        let mut interner_updates = None;
        let mut spatial_tree_updates = None;

        if scene.has_root_pipeline() && rebuild_scene {

            let built = SceneBuilder::build(
                &scene,
                None,
                self.fonts.clone(),
                &doc.view,
                &config,
                &mut doc.interners,
                &mut doc.spatial_tree,
                &mut self.recycler,
                &doc.stats,
                self.debug_flags,
            );

            // Update the allocation stats for next scene
            doc.stats = built.get_stats();

            // Retrieve the list of updates from the clip interner.
            interner_updates = Some(
                doc.interners.end_frame_and_get_pending_updates()
            );

            spatial_tree_updates = Some(
                doc.spatial_tree.end_frame_and_get_pending_updates()
            );

            built_scene = Some(built);
        }

        // Offscreen scenes intern into the document interners but don't flush
        // their own delta; they rely on the main scene rebuild above emitting
        // a combined delta that materializes their items into the document
        // data store. The Gecko caller guarantees a SetDisplayList (hence a
        // rebuild) accompanies every RenderOffscreen.
        debug_assert!(
            offscreen_scenes.is_empty() || interner_updates.is_some(),
            "RenderOffscreen without a main scene rebuild to flush its interned items",
        );

        let scene_build_time_ms =
            profiler::ns_to_ms(zeitstempel::now() - scene_build_start);
        profile.set(profiler::SCENE_BUILD_TIME, scene_build_time_ms);

        frame_stats.scene_build_time += scene_build_time_ms;

        if !txn.blob_requests.is_empty() {
            profile.start_time(profiler::BLOB_RASTERIZATION_TIME);

            let is_low_priority = false;
            rasterize_blobs(&mut txn, is_low_priority, &mut self.tile_pool);

            profile.end_time(profiler::BLOB_RASTERIZATION_TIME);
            Telemetry::record_rasterize_blobs_time(Duration::from_micros((profile.get(profiler::BLOB_RASTERIZATION_TIME).unwrap() * 1000.00) as u64));
        }

        drain_filter(
            &mut txn.notifications,
            |n| { n.when() == Checkpoint::SceneBuilt },
            |n| { n.notify(); },
        );

        if self.simulate_slow_ms > 0 {
            thread::sleep(Duration::from_millis(self.simulate_slow_ms as u64));
        }

        Box::new(BuiltTransaction {
            document_id: txn.document_id,
            render_frame: txn.generate_frame.as_bool(),
            present: txn.generate_frame.present(),
            tracked: txn.generate_frame.tracked(),
            invalidate_rendered_frame: txn.invalidate_rendered_frame,
            built_scene,
            offscreen_scenes,
            view: doc.view,
            rasterized_blobs: txn.rasterized_blobs,
            resource_updates: txn.resource_updates,
            blob_rasterizer: txn.blob_rasterizer,
            frame_ops: txn.frame_ops,
            removed_pipelines,
            notifications: txn.notifications,
            interner_updates,
            spatial_tree_updates,
            profile,
            frame_stats,
        })
    }

    /// Send the results of process_transaction back to the render backend.
    fn forward_built_transactions(&mut self, txns: Vec<Box<BuiltTransaction>>) {
        // Group transactions by their owning window so hooks fire per-window.
        // In practice each batch is usually a single transaction (and hence a
        // single window), but multiple are handled correctly too.
        let mut docs_per_window: FastHashMap<RenderBackendId, Vec<DocumentId>> =
            FastHashMap::default();
        let mut windows_with_built_scene: FastHashSet<RenderBackendId> =
            FastHashSet::default();

        let mut max_sb_time: f64 = 0.0;
        for txn in &txns {
            if let Some(time) = txn.profile.get(profiler::SCENE_BUILD_TIME) {
                max_sb_time = max_sb_time.max(time);
            }

            if let Some(&win_id) = self.doc_to_window.get(&txn.document_id) {
                docs_per_window.entry(win_id).or_default().push(txn.document_id);
                if txn.built_scene.is_some() {
                    windows_with_built_scene.insert(win_id);
                }
            }
        }

        if max_sb_time > 0.0 {
            let duration = Duration::from_nanos((max_sb_time * 1_000_000.0) as u64);
            Telemetry::record_scenebuild_time(duration);
        }

        let has_built_scene = !windows_with_built_scene.is_empty();

        let (pipeline_info, result_tx, result_rx) = if has_built_scene {
            let info = PipelineInfo {
                epochs: txns.iter()
                    .filter(|txn| txn.built_scene.is_some())
                    .map(|txn| {
                        txn.built_scene.as_ref().unwrap()
                            .pipeline_epochs.iter()
                            .zip(iter::repeat(txn.document_id))
                            .map(|((&pipeline_id, &epoch), document_id)| ((pipeline_id, document_id), epoch))
                    }).flatten().collect(),
                removed_pipelines: txns.iter()
                    .map(|txn| txn.removed_pipelines.clone())
                    .flatten().collect(),
            };

            let (tx, rx) = single_msg_channel();

            // Invoke pre_scene_swap for every window that produced a built
            // scene in this batch.
            for win_id in &windows_with_built_scene {
                if let Some(hooks) = self.hooks.get(win_id) {
                    hooks.pre_scene_swap();
                }
            }

            (Some(info), Some(tx), Some(rx))
        } else {
            (None, None, None)
        };

        let timer_id = Telemetry::start_sceneswap_time();
        let have_resources_updates: Vec<DocumentId> = if pipeline_info.is_none() {
            txns.iter()
                .filter(|txn| !txn.resource_updates.is_empty() || txn.invalidate_rendered_frame)
                .map(|txn| txn.document_id)
                .collect()
        } else {
            Vec::new()
        };

        // Unless a transaction generates a frame immediately, the compositor should
        // schedule one whenever appropriate (probably at the next vsync) to present
        // the changes in the scene.
        let compositor_should_schedule_a_frame = !txns.iter().any(|txn| {
            txn.render_frame
        });

        #[cfg(feature = "capture")]
        match self.capture_config {
            Some(ref config) => self.send(SceneBuilderResult::CapturedTransactions(txns, config.clone(), result_tx)),
            None => self.send(SceneBuilderResult::Transactions(txns, result_tx)),
        };

        #[cfg(not(feature = "capture"))]
        self.send(SceneBuilderResult::Transactions(txns, result_tx));

        if let Some(pipeline_info) = pipeline_info {
            // Block until the swap is done, then invoke post_scene_swap.
            let swap_result = result_rx.unwrap().recv();
            Telemetry::stop_and_accumulate_sceneswap_time(timer_id);

            // For each window that contributed a built scene, send it the
            // subset of pipeline info it actually owns. This is what
            // delivers `apz_post_scene_swap(this_window_id, ...)` to the
            // right APZ tree and triggers
            // `wr_schedule_frame_after_scene_build` for that window.
            for win_id in &windows_with_built_scene {
                if let Some(hooks) = self.hooks.get(win_id) {
                    let win_docs = docs_per_window
                        .get(win_id)
                        .cloned()
                        .unwrap_or_default();
                    let win_doc_set: FastHashSet<DocumentId> =
                        win_docs.iter().copied().collect();
                    let win_info = PipelineInfo {
                        epochs: pipeline_info
                            .epochs
                            .iter()
                            .filter(|((_, doc_id), _)| win_doc_set.contains(doc_id))
                            .map(|(k, v)| (*k, *v))
                            .collect(),
                        removed_pipelines: pipeline_info
                            .removed_pipelines
                            .iter()
                            .filter(|(_, doc_id)| win_doc_set.contains(doc_id))
                            .cloned()
                            .collect(),
                    };
                    hooks.post_scene_swap(
                        &win_docs,
                        win_info,
                        compositor_should_schedule_a_frame,
                    );
                }
            }

            // Once the hooks are done, allow the RB thread to resume.
            if let Ok(SceneSwapResult::Complete(resume_tx)) = swap_result {
                let _ = resume_tx.send(());
            }
        } else {
            Telemetry::cancel_sceneswap_time(timer_id);
            // No built scene this batch. Per window, either deliver
            // post_resource_update (with the docs that had updates) or
            // post_empty_scene_build.
            let updates_by_window: FastHashMap<RenderBackendId, Vec<DocumentId>> = {
                let mut map: FastHashMap<RenderBackendId, Vec<DocumentId>> =
                    FastHashMap::default();
                for doc_id in &have_resources_updates {
                    if let Some(&win_id) = self.doc_to_window.get(doc_id) {
                        map.entry(win_id).or_default().push(*doc_id);
                    }
                }
                map
            };
            for (win_id, _doc_ids) in &docs_per_window {
                if let Some(hooks) = self.hooks.get(win_id) {
                    if let Some(updates) = updates_by_window.get(win_id) {
                        hooks.post_resource_update(updates);
                    } else {
                        hooks.post_empty_scene_build();
                    }
                }
            }
        }
    }

    /// Reports CPU heap memory used by the SceneBuilder.
    fn report_memory(&mut self) -> MemoryReport {
        let ops = self.size_of_ops.as_mut().unwrap();
        let mut report = MemoryReport::default();
        for doc in self.documents.values() {
            doc.interners.report_memory(ops, &mut report);
            doc.scene.report_memory(ops, &mut report);
        }

        report
    }
}

/// A scene builder thread which executes expensive operations such as blob rasterization
/// with a lower priority than the normal scene builder thread.
///
/// After rasterizing blobs, the secene building request is forwarded to the normal scene
/// builder where the FrameBuilder is generated.
pub struct LowPrioritySceneBuilderThread {
    pub rx: Receiver<SceneBuilderRequest>,
    pub tx: Sender<SceneBuilderRequest>,
    pub tile_pool: api::BlobTilePool,
}

impl LowPrioritySceneBuilderThread {
    pub fn run(&mut self) {
        loop {
            match self.rx.recv() {
                Ok(SceneBuilderRequest::Transactions(mut txns)) => {
                    let txns : Vec<Box<TransactionMsg>> = txns.drain(..)
                        .map(|txn| self.process_transaction(txn))
                        .collect();
                    self.tx.send(SceneBuilderRequest::Transactions(txns)).unwrap();
                    self.tile_pool.cleanup();
                }
                Ok(SceneBuilderRequest::ShutDown(sync)) => {
                    self.tx.send(SceneBuilderRequest::ShutDown(sync)).unwrap();
                    break;
                }
                Ok(other) => {
                    self.tx.send(other).unwrap();
                }
                Err(..) => break,
            }
        }
    }

    fn process_transaction(&mut self, mut txn: Box<TransactionMsg>) -> Box<TransactionMsg> {
        let is_low_priority = true;
        txn.profile.start_time(profiler::BLOB_RASTERIZATION_TIME);
        rasterize_blobs(&mut txn, is_low_priority, &mut self.tile_pool);
        txn.profile.end_time(profiler::BLOB_RASTERIZATION_TIME);
        Telemetry::record_rasterize_blobs_time(Duration::from_micros((txn.profile.get(profiler::BLOB_RASTERIZATION_TIME).unwrap() * 1000.00) as u64));
        txn.blob_requests = Vec::new();

        txn
    }
}
