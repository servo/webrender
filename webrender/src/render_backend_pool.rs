/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Pool of render-backend threads.
//!
//! Each pool member owns three threads — a render backend thread and its
//! companion scene builder and (optional) low-priority scene builder. Windows
//! are assigned to a member round-robin via [`RenderBackendPool::assign`],
//! after which a `WindowRegistration` message is sent on the member's api
//! channel to install per-window state.

use std::io;
use std::mem;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use api::channel::{unbounded_channel, Sender};
use crate::frame_builder::FrameBuilderConfig;
use glyph_rasterizer::SharedFontResources;
use malloc_size_of::MallocSizeOfOps;
use tracy_rs::register_thread_with_profiler;

use crate::profiler;
use crate::render_api::ApiMsg;
use crate::render_backend::RenderBackend;
use crate::renderer::init::RenderBackendHooks;
use crate::scene_builder_thread::{
    LowPrioritySceneBuilderThread, SceneBuilderRequest, SceneBuilderThread,
    SceneBuilderThreadChannels,
};
use api::VoidPtrToSizeFn;

/// Setup for one pool member. Provided per-member when constructing the pool.
pub struct PoolMemberSetup {
    /// Frame builder config used to construct the scene builder.
    pub frame_builder_config: FrameBuilderConfig,
    /// Shared font namespace for the scene builder. Members of the same pool
    /// can share these to coalesce font keys.
    pub fonts: SharedFontResources,
    /// Whether to spawn a low-priority scene builder thread.
    pub support_low_priority_transactions: bool,
    /// Function used to compute heap sizes for memory reports.
    pub size_of_op: Option<VoidPtrToSizeFn>,
    /// Enclosing variant of [`Self::size_of_op`].
    pub enclosing_size_of_op: Option<VoidPtrToSizeFn>,
    /// Optional callback hooks for the render backend thread.
    pub render_backend_hooks: Option<Box<dyn RenderBackendHooks + Send>>,
    /// Whether namespaces are allocated by the client (vs. by webrender).
    pub namespace_alloc_by_client: bool,
    /// Suffix used in the spawned thread names (e.g. `"0"` produces
    /// `WRRenderBackend#0`).
    pub thread_name_suffix: String,
}

/// Channels handed back by [`RenderBackendPool::assign`].
pub struct AssignedBackend {
    pub api_tx: Sender<ApiMsg>,
    pub scene_tx: Sender<SceneBuilderRequest>,
    pub lp_scene_tx: Sender<SceneBuilderRequest>,
}

struct PoolMember {
    api_tx: Sender<ApiMsg>,
    scene_tx: Sender<SceneBuilderRequest>,
    lp_scene_tx: Sender<SceneBuilderRequest>,
}

/// A round-robin pool of render-backend threads. See module docs.
pub struct RenderBackendPool {
    members: Vec<PoolMember>,
    /// Join handles of every thread spawned by the pool, waited on when the
    /// pool is dropped.
    handles: Vec<thread::JoinHandle<()>>,
    next: AtomicUsize,
}

impl RenderBackendPool {
    /// Spawn `size` pool members. `member_setup` is invoked `size` times,
    /// once per member, so the caller can supply per-member hooks /
    /// thread-name suffixes.
    pub fn new<F>(size: usize, mut member_setup: F) -> io::Result<Arc<Self>>
    where
        F: FnMut(usize) -> PoolMemberSetup,
    {
        let size = size.max(1).min(16);
        let mut members = Vec::with_capacity(size);
        let mut handles = Vec::with_capacity(size * 3);
        for i in 0..size {
            let (member, member_handles) = Self::spawn_member(member_setup(i))?;
            members.push(member);
            handles.extend(member_handles);
        }
        Ok(Arc::new(Self {
            members,
            handles,
            next: AtomicUsize::new(0),
        }))
    }

    /// Number of backend threads in the pool.
    pub fn size(&self) -> usize {
        self.members.len()
    }

    /// Pick the next member round-robin and return clones of its channels.
    /// The caller is expected to send `ApiMsg::RegisterWindow` on the
    /// returned `api_tx` before any other message that references the new
    /// `RenderBackendId`.
    pub fn assign(&self) -> AssignedBackend {
        let i = self.next.fetch_add(1, Ordering::Relaxed) % self.members.len();
        let m = &self.members[i];
        AssignedBackend {
            api_tx: m.api_tx.clone(),
            scene_tx: m.scene_tx.clone(),
            lp_scene_tx: m.lp_scene_tx.clone(),
        }
    }

    fn spawn_member(
        setup: PoolMemberSetup,
    ) -> io::Result<(PoolMember, Vec<thread::JoinHandle<()>>)> {
        let PoolMemberSetup {
            frame_builder_config,
            fonts,
            support_low_priority_transactions,
            size_of_op,
            enclosing_size_of_op,
            render_backend_hooks,
            namespace_alloc_by_client,
            thread_name_suffix,
        } = setup;

        let rb_thread_name = format!("WRRenderBackend#{}", thread_name_suffix);
        let scene_thread_name = format!("WRSceneBuilder#{}", thread_name_suffix);
        let lp_scene_thread_name = format!("WRSceneBuilderLP#{}", thread_name_suffix);

        let (api_tx, api_rx) = unbounded_channel();
        let (scene_builder_channels, scene_tx) =
            SceneBuilderThreadChannels::new(api_tx.clone());

        let mut handles = Vec::with_capacity(3);

        // Scene builder thread.
        let sb_fonts = fonts.clone();
        let sb_config = frame_builder_config.clone();
        let sb_size_of_ops =
            size_of_op.map(|o| MallocSizeOfOps::new(o, enclosing_size_of_op));
        let sb_thread_name = scene_thread_name.clone();
        handles.push(thread::Builder::new().name(scene_thread_name).spawn(move || {
            register_thread_with_profiler(sb_thread_name.clone());
            profiler::register_thread(&sb_thread_name);

            let mut scene_builder = SceneBuilderThread::new(
                sb_config,
                sb_fonts,
                sb_size_of_ops,
                scene_builder_channels,
            );
            scene_builder.run();

            profiler::unregister_thread();
        })?);

        // Low-priority scene builder thread (optional).
        let lp_scene_tx = if support_low_priority_transactions {
            let (lp_scene_tx, lp_scene_rx) = unbounded_channel();
            let lp_builder = LowPrioritySceneBuilderThread {
                rx: lp_scene_rx,
                tx: scene_tx.clone(),
                tile_pool: api::BlobTilePool::new(),
            };
            let lp_thread_name = lp_scene_thread_name.clone();
            handles.push(thread::Builder::new().name(lp_scene_thread_name).spawn(move || {
                register_thread_with_profiler(lp_thread_name.clone());
                profiler::register_thread(&lp_thread_name);

                let mut scene_builder = lp_builder;
                scene_builder.run();

                profiler::unregister_thread();
            })?);
            lp_scene_tx
        } else {
            scene_tx.clone()
        };

        // Render backend thread.
        let rb_scene_tx = scene_tx.clone();
        let rb_size_of_ops =
            size_of_op.map(|o| MallocSizeOfOps::new(o, enclosing_size_of_op));
        let rb_thread_name_clone = rb_thread_name.clone();
        handles.push(thread::Builder::new().name(rb_thread_name).spawn(move || {
            if let Some(hooks) = render_backend_hooks {
                hooks.init_thread();
            }
            register_thread_with_profiler(rb_thread_name_clone.clone());
            profiler::register_thread(&rb_thread_name_clone);

            let mut backend = RenderBackend::new(
                api_rx,
                rb_scene_tx,
                rb_size_of_ops,
                namespace_alloc_by_client,
            );
            backend.run();
            drop(backend);

            profiler::unregister_thread();
        })?);

        Ok((
            PoolMember {
                api_tx,
                scene_tx,
                lp_scene_tx,
            },
            handles,
        ))
    }
}

impl Drop for RenderBackendPool {
    fn drop(&mut self) {
        // Explicitly tear down each member. Without this we deadlock at
        // process shutdown: `SceneBuilderThread` holds an `api_tx` clone
        // and `RenderBackend` holds a `scene_tx` clone, so even after the
        // pool and every `RenderApi` are dropped, the two threads keep
        // each other's receive channels alive.
        //
        // Send `SceneBuilderRequest::ShutDown(None)` on each scene channel
        // (low-priority, so it routes through the LP scene builder too if
        // one was spawned). SB forwards `SceneBuilderResult::ShutDown` back
        // to RB, RB drains its api channel and exits, threads unwind, and
        // every receiver closes cleanly.
        let members = mem::take(&mut self.members);
        for m in &members {
            let _ = m.lp_scene_tx.send(SceneBuilderRequest::ShutDown(None));
        }

        // Drop the pool's own channel clones *before* joining below. The
        // `SceneBuilderResult::ShutDown` above is consumed by the render
        // backend's main loop, so its drain loop can only end when every
        // `api_tx` clone is gone and `recv()` fails. Holding on to the
        // members while joining deadlocks.
        drop(members);

        // Wait for the threads to exit. They register themselves with the
        // embedder's profiler, which in Gecko lazily creates an nsThread
        // wrapper that is only released once the thread exits, so a thread
        // still winding down at process shutdown is reported as a leak.
        for handle in mem::take(&mut self.handles) {
            let _ = handle.join();
        }
    }
}
