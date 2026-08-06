/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::channel::{unbounded_channel, Receiver};
use api::units::{DeviceIntSize, LayoutRect, LayoutSize, WorldPoint};
use api::{Checkpoint, ColorF, CommonItemProperties, DebugFlags, DisplayListBuilder};
use api::{DocumentId, Epoch, FontRenderMode, FramePublishId, FrameReadyParams};
use api::{ImageFormat, NotificationHandler, NotificationRequest, PipelineId};
use api::{PrimitiveFlags, RenderBackendId, RenderNotifier, RenderReasons, SpaceAndClipInfo};
use glyph_rasterizer::SharedFontResources;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::mpsc::{channel, Receiver as StdReceiver, Sender as StdSender};
use std::sync::Arc;
use std::time::Duration;

use crate::bump_allocator::ChunkPool;
use crate::composite::CompositorKind;
use crate::device::{TextureFilter, TextureFormatPair};
use crate::frame_builder::FrameBuilderConfig;
use crate::internal_types::ResultMsg;
use crate::render_api::{ApiMsg, RenderApi, RenderApiSender, ResourceCacheInit};
use crate::render_api::{Transaction, WindowRegistration};
use crate::render_backend::RenderBackend;
use crate::render_backend_pool::{PoolMemberSetup, RenderBackendPool};
use crate::scene_builder_thread::SceneBuilderRequest;
use crate::texture_cache::TextureCacheConfig;

const WINDOW_SIZE: DeviceIntSize = DeviceIntSize::new(64, 64);
const AU_PER_DEV_PX: f32 = 60.0;

struct TestNotifier;

impl RenderNotifier for TestNotifier {
    fn clone(&self) -> Box<dyn RenderNotifier> {
        Box::new(TestNotifier)
    }
    fn wake_up(&self, _composite_needed: bool) {}
    fn new_frame_ready(&self, _: DocumentId, _: FramePublishId, _: &FrameReadyParams) {}
}

/// Reports the checkpoint a transaction reached. `NotificationRequest` is
/// notified exactly once whether the transaction is processed
/// (`FrameBuilt`, on the backend thread) or dropped
/// (`TransactionDropped`), which makes it a barrier that works even for a
/// window whose transactions are being discarded.
struct CheckpointSignal(StdSender<Checkpoint>);

impl NotificationHandler for CheckpointSignal {
    fn notify(&self, when: Checkpoint) {
        let _ = self.0.send(when);
    }
}

/// Waits for a barrier armed by `TestWindow::arm_barrier`.
struct Barrier(StdReceiver<Checkpoint>);

impl Barrier {
    fn wait(self) -> Checkpoint {
        self.0
            .recv_timeout(Duration::from_secs(60))
            .expect("render backend never reported the transaction checkpoint")
    }
}

fn test_frame_config() -> FrameBuilderConfig {
    FrameBuilderConfig {
        default_font_render_mode: FontRenderMode::Mono,
        dual_source_blending_is_supported: false,
        testing: true,
        gpu_supports_fast_clears: false,
        gpu_supports_advanced_blend: false,
        advanced_blend_is_coherent: false,
        gpu_supports_render_target_partial_update: true,
        external_images_require_copy: false,
        batch_lookback_count: 10,
        background_color: None,
        compositor_kind: CompositorKind::default(),
        tile_size_override: None,
        max_surface_override: None,
        max_depth_ids: 1 << 16,
        max_target_size: 2048,
        force_invalidation: false,
        is_software: true,
        low_quality_pinch_zoom: false,
        max_shared_surface_size: 2048,
        enable_dithering: false,
    }
}

/// One window on the shared backend. Stands in for a `Renderer`: holding
/// `result_rx` keeps the publish channel alive, and dropping it is what
/// destroying a `Renderer` does to the backend.
struct TestWindow {
    api: RenderApi,
    document_id: DocumentId,
    pipeline_id: PipelineId,
    result_rx: Option<Receiver<ResultMsg>>,
    epoch: Epoch,
}

impl TestWindow {
    /// Attach a `FrameBuilt` notification so the caller can wait for the
    /// backend to be done with this transaction.
    fn arm_barrier(txn: &mut Transaction) -> Barrier {
        let (tx, rx) = channel();
        txn.notify(NotificationRequest::new(
            Checkpoint::FrameBuilt,
            Box::new(CheckpointSignal(tx)),
        ));
        Barrier(rx)
    }

    /// Push a display list and ask for a frame, going through the scene
    /// builder so this lands in `process_transaction`.
    fn build_scene(&mut self) -> Barrier {
        let mut builder = DisplayListBuilder::new(self.pipeline_id);
        builder.begin(AU_PER_DEV_PX);
        let space_and_clip = SpaceAndClipInfo::root_scroll(self.pipeline_id);
        let rect = LayoutRect::from_size(LayoutSize::new(32.0, 32.0));
        builder.push_rect(
            &CommonItemProperties {
                clip_rect: rect,
                clip_chain_id: space_and_clip.clip_chain_id,
                spatial_id: space_and_clip.spatial_id,
                flags: PrimitiveFlags::default(),
            },
            rect,
            ColorF::new(1.0, 0.0, 0.0, 1.0),
        );

        let mut txn = Transaction::new();
        txn.use_scene_builder_thread();
        txn.set_root_pipeline(self.pipeline_id);
        txn.set_display_list(self.epoch, self.api.get_namespace_id(), builder.end());
        txn.generate_frame(0, true, false, RenderReasons::TESTING);
        self.epoch.0 += 1;
        let barrier = Self::arm_barrier(&mut txn);
        self.api.send_transaction(self.document_id, txn);
        barrier
    }

    /// Ask for a frame without going through the scene builder, so the
    /// message travels the api channel in FIFO order with everything else.
    fn request_frame_without_scene_build(&mut self) -> Barrier {
        let mut txn = Transaction::new();
        txn.generate_frame(0, true, false, RenderReasons::TESTING);
        let barrier = Self::arm_barrier(&mut txn);
        self.api.send_transaction(self.document_id, txn);
        barrier
    }

    /// Drop the publish channel, the way destroying a `Renderer` does.
    fn destroy_renderer(&mut self) {
        self.result_rx = None;
    }

    fn drain_results(&self) -> usize {
        let rx = self.result_rx.as_ref().unwrap();
        let mut count = 0;
        while rx.try_recv().is_ok() {
            count += 1;
        }
        count
    }

    /// Synchronous round-trip through the backend. Doubles as a liveness
    /// probe: if the backend thread died the reply channel is gone and
    /// `hit_test` panics.
    fn round_trip(&self) {
        self.api.hit_test(self.document_id, WorldPoint::zero());
    }
}

struct TestHarness {
    pool: Arc<RenderBackendPool>,
    fonts: SharedFontResources,
    workers: Arc<ThreadPool>,
    next_id: u32,
}

impl TestHarness {
    /// A pool of one backend thread, so every window added below shares it.
    fn new() -> Self {
        let fonts = SharedFontResources::new(RenderBackend::next_namespace_id());
        let pool_fonts = fonts.clone();
        let pool = RenderBackendPool::new(1, move |_idx| PoolMemberSetup {
            frame_builder_config: test_frame_config(),
            fonts: pool_fonts.clone(),
            support_low_priority_transactions: true,
            size_of_op: None,
            enclosing_size_of_op: None,
            render_backend_hooks: None,
            namespace_alloc_by_client: false,
            thread_name_suffix: "test".to_string(),
        })
        .unwrap();

        TestHarness {
            pool,
            fonts,
            workers: Arc::new(ThreadPoolBuilder::new().build().unwrap()),
            next_id: 0,
        }
    }

    fn add_window(&mut self) -> TestWindow {
        let backend_id = RenderBackendId(self.next_id);
        self.next_id += 1;

        let assigned = self.pool.assign();
        let (result_tx, result_rx) = unbounded_channel();
        let config = test_frame_config();

        assigned
            .api_tx
            .send(ApiMsg::RegisterWindow(Box::new(WindowRegistration {
                id: backend_id,
                result_tx,
                notifier: Box::new(TestNotifier),
                sampler: None,
                resource_cache: ResourceCacheInit {
                    max_internal_texture_size: 2048,
                    image_tiling_threshold: 2048,
                    color_cache_formats: TextureFormatPair::from(ImageFormat::BGRA8),
                    swizzle_settings: None,
                    texture_cache_config: TextureCacheConfig::DEFAULT,
                    picture_tile_size: DeviceIntSize::new(512, 512),
                    picture_texture_filter: TextureFilter::Nearest,
                    workers: self.workers.clone(),
                    dedicated_glyph_raster_thread: None,
                    supports_r8_texture_upload: false,
                    fonts: self.fonts.clone(),
                    blob_image_handler: None,
                    enable_multithreading: false,
                },
                chunk_pool: Arc::new(ChunkPool::new()),
                frame_config: config.clone(),
                debug_flags: DebugFlags::empty(),
            })))
            .unwrap();

        assigned
            .scene_tx
            .send(SceneBuilderRequest::SetFrameBuilderConfig(
                backend_id,
                config,
            ))
            .unwrap();

        let api = RenderApiSender::new(
            assigned.api_tx,
            assigned.scene_tx,
            assigned.lp_scene_tx,
            backend_id,
            None,
            self.fonts.clone(),
            self.pool.clone(),
        )
        .create_api();

        let document_id = api.add_document(WINDOW_SIZE);
        let pipeline_id = PipelineId(backend_id.0, 0);

        TestWindow {
            api,
            document_id,
            pipeline_id,
            result_rx: Some(result_rx),
            epoch: Epoch(0),
        }
    }
}

/// A transaction that reaches the backend after the window's `Renderer`
/// has been destroyed must be dropped, not published to a dead channel.
/// Regression test for bug 2058350.
#[test]
fn late_transaction_after_stop_render_backend() {
    let mut harness = TestHarness::new();
    let mut win_a = harness.add_window();
    let mut win_b = harness.add_window();

    assert_eq!(win_a.build_scene().wait(), Checkpoint::FrameBuilt);
    assert_eq!(win_b.build_scene().wait(), Checkpoint::FrameBuilt);
    assert!(win_a.drain_results() > 0);
    assert!(win_b.drain_results() > 0);

    // Tear down window B the way Gecko does: drain the backend, then
    // destroy the Renderer. B stays registered until `shut_down`.
    win_b.api.stop_render_backend();
    win_b.destroy_renderer();

    // Transactions that lose the race with teardown. Before bug 2058350
    // these panicked the backend thread on `result_tx.send().unwrap()`.
    let scene_barrier = win_b.build_scene();
    let frame_barrier = win_b.request_frame_without_scene_build();
    assert_eq!(scene_barrier.wait(), Checkpoint::TransactionDropped);
    assert_eq!(frame_barrier.wait(), Checkpoint::TransactionDropped);

    // The backend thread is still alive and window A is unaffected.
    win_a.round_trip();
    assert_eq!(win_a.build_scene().wait(), Checkpoint::FrameBuilt);
    assert!(
        win_a.drain_results() > 0,
        "window A stopped producing frames after window B was torn down"
    );

    win_b.api.shut_down(true);
    win_a.api.shut_down(true);
}

/// A stopped window must publish nothing more, while the other windows
/// sharing the backend keep working.
#[test]
fn stopped_window_produces_no_results() {
    let mut harness = TestHarness::new();
    let mut win_a = harness.add_window();
    let mut win_b = harness.add_window();

    assert_eq!(win_a.build_scene().wait(), Checkpoint::FrameBuilt);
    assert_eq!(win_b.build_scene().wait(), Checkpoint::FrameBuilt);

    // `stop_render_backend` only returns once the backend has drained
    // everything queued before it, so B's channel is quiet from here on.
    win_b.api.stop_render_backend();
    win_b.drain_results();

    // Here B keeps its `result_rx`, so anything the backend still
    // published for it would show up below.
    assert_eq!(
        win_b.build_scene().wait(),
        Checkpoint::TransactionDropped,
    );
    assert_eq!(
        win_b.drain_results(),
        0,
        "a stopped window still published results"
    );

    assert_eq!(win_a.build_scene().wait(), Checkpoint::FrameBuilt);
    assert!(win_a.drain_results() > 0);

    win_b.api.shut_down(true);
    win_a.api.shut_down(true);
}

/// Process-wide messages fan out to every registered window, so they reach a
/// stopped one without anybody having submitted work for it. They must not
/// publish to it. This is the case a "the embedder stops submitting" contract
/// cannot cover, and the reason `WindowState::stopped` exists.
#[test]
fn memory_pressure_with_stopped_window() {
    let mut harness = TestHarness::new();
    let mut win_a = harness.add_window();
    let mut win_b = harness.add_window();

    assert_eq!(win_a.build_scene().wait(), Checkpoint::FrameBuilt);
    assert_eq!(win_b.build_scene().wait(), Checkpoint::FrameBuilt);

    win_b.api.stop_render_backend();
    win_b.destroy_renderer();
    win_a.drain_results();

    // Sent by A, but the backend acts on every window it still has registered,
    // B included.
    win_a.api.notify_memory_pressure();

    // FIFO on the api channel, so the backend is done with the memory pressure
    // by the time this returns.
    win_a.round_trip();

    assert_eq!(win_a.build_scene().wait(), Checkpoint::FrameBuilt);
    assert!(
        win_a.drain_results() > 0,
        "window A stopped producing frames after memory pressure"
    );

    win_b.api.shut_down(true);
    win_a.api.shut_down(true);
}
