/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! # Visibility pass
//!
//! The first of the two frame building traversals of the picture tree, the
//! second being the [prepare pass](crate::prepare). It is driven by
//! `FrameBuilder::build_layer_screen_rects_and_cull_layers`, which calls
//! [`update_prim_visibility`] once per snapshot picture and once per tile cache
//! slice. From there the pass walks down the picture tree, pushing and popping
//! off-screen surfaces as it goes.
//!
//! For each primitive instance it visits, the pass works out whether the
//! primitive is drawn this frame and under which clips, and records the answer
//! in a [`PrimitiveDrawHeader`], pushed into `scratch.primitive.frame.draws`
//! as each drawn primitive is found.
//! Later passes read those headers instead of re-deriving the information.
//! In addition to visibility calculation, this pass performs snapping and
//! builds clip chain instances.
//!
//! ## Surface bookkeeping
//!
//! Alongside the per-primitive state, the traversal accumulates the exact
//! (clipped) local rect of each off-screen surface from the coverage rects of
//! the primitives drawn into it, and propagates culling rects from parent to
//! child surfaces. The prepare pass sizes the surfaces' render tasks from those
//! accumulated rects, so they must be complete before it runs, which is the
//! main reason visibility is a separate pass.
//!

use api::DebugFlags;
use api::units::*;
use crate::clip::ClipStore;
use crate::composite::CompositeState;
use crate::profiler::{self, TransactionProfile};
use crate::renderer::GpuBufferBuilder;
use crate::spatial_tree::{SpatialTree, SpatialNodeIndex};
use crate::clip::{ClipChainInstance, ClipTree, ClipNodeId};
use crate::composite::CompositorSurfaceKind;
use crate::frame_builder::FrameBuilderConfig;
use crate::picture::ClusterFlags;
use crate::picture_composite_mode::PictureCompositeMode;
use crate::surface::SurfaceInfo;
use crate::tile_cache::TileCacheInstance;
use crate::picture::{PictureScratch, RasterConfig};
use crate::surface::SurfaceIndex;
use crate::tile_cache::SubSliceIndex;
use crate::prim_store::{ClipSnap, ClipTaskIndex, PictureIndex, PrimitiveKind};
use crate::prim_store::{PrimitiveStore, PrimitiveInstance, PrimitiveInstanceIndex};
use crate::prim_store::storage;
use crate::prim_store::text_run::TextRunScratch;
use crate::render_backend::{DataStores, ScratchBuffer};
use crate::render_task_graph::RenderTaskGraphBuilder;
use crate::resource_cache::ResourceCache;
use crate::scene::SceneProperties;
use crate::space::{SpaceMapper, SpaceSnapper};
use crate::util::MaxRect;

pub struct FrameVisibilityContext<'a> {
    pub spatial_tree: &'a SpatialTree,
    pub global_screen_device_rect: DeviceRect,
    pub debug_flags: DebugFlags,
    pub scene_properties: &'a SceneProperties,
    pub config: FrameBuilderConfig,
    pub root_spatial_node_index: SpatialNodeIndex,
}

pub struct FrameVisibilityState<'a> {
    pub clip_store: &'a mut ClipStore,
    pub resource_cache: &'a mut ResourceCache,
    pub frame_gpu_data: &'a mut GpuBufferBuilder,
    pub data_stores: &'a DataStores,
    pub clip_tree: &'a mut ClipTree,
    pub composite_state: &'a mut CompositeState,
    pub rg_builder: &'a mut RenderTaskGraphBuilder,
    pub prim_instances: &'a mut [PrimitiveInstance],
    pub surfaces: &'a mut [SurfaceInfo],
    /// A stack of currently active off-screen surfaces during the
    /// visibility frame traversal.
    pub surface_stack: Vec<(PictureIndex, SurfaceIndex)>,
    pub profile: &'a mut TransactionProfile,
    pub scratch: &'a mut ScratchBuffer,
    pub visited_pictures: &'a mut[bool],
}

impl<'a> FrameVisibilityState<'a> {
    pub fn push_surface(
        &mut self,
        pic_index: PictureIndex,
        surface_index: SurfaceIndex,
    ) {
        self.surface_stack.push((pic_index, surface_index));
    }

    pub fn pop_surface(&mut self) {
        self.surface_stack.pop().unwrap();
    }
}

bitflags! {
    /// A set of bitflags that can be set in the visibility information
    /// for a primitive instance. This can be used to control how primitives
    /// are treated during batching.
    // TODO(gw): We should also move `is_compositor_surface` to be part of
    //           this flags struct.
    #[cfg_attr(feature = "capture", derive(Serialize))]
    #[derive(Debug, Copy, PartialEq, Eq, Clone, PartialOrd, Ord, Hash)]
    pub struct PrimitiveVisibilityFlags: u8 {
        /// Implies that this primitive covers the entire picture cache slice,
        /// and can thus be dropped during batching and drawn with clear color.
        const IS_BACKDROP = 1;
    }
}

/// Contains the current state of the primitive's visibility.
#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "capture", derive(Serialize))]
pub enum DrawState {
    /// No state resolved yet. Only observable between a header being created
    /// and the visibility pass deciding the primitive's fate; a draw that
    /// reaches prepare or batching in this state is a bug.
    Unset,
    /// Culled for being off-screen, or not possible to render (e.g. missing image resource)
    Culled,
    /// A picture that doesn't have a surface - primitives are composed into the
    /// parent picture with a surface.
    PassThrough,
    /// A primitive that has been found to be visible
    Visible {
        /// A set of flags that define how this primitive should be handled
        /// during batching of visible primitives.
        vis_flags: PrimitiveVisibilityFlags,

        /// Sub-slice within the picture cache that this prim exists on
        sub_slice_index: SubSliceIndex,
    },
}

/// Per-draw, per-kind scratch handle. Reaches the appropriate
/// per-frame scratch entry for the drawn primitive's kind. The variant
/// matches the prim's PrimitiveKind. None for kinds without per-frame
/// scratch.
#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "capture", derive(Serialize))]
pub enum KindScratchHandle {
    None,
    TextRun(storage::Index<TextRunScratch>),
    Picture(storage::Index<PictureScratch>),
}

impl KindScratchHandle {
    pub fn unwrap_text_run(&self) -> storage::Index<TextRunScratch> {
        match *self {
            KindScratchHandle::TextRun(h) => h,
            _ => panic!("kind_scratch mismatch: expected TextRun, got {:?}", self),
        }
    }
    pub fn unwrap_picture(&self) -> storage::Index<PictureScratch> {
        match *self {
            KindScratchHandle::Picture(h) => h,
            _ => panic!("kind_scratch mismatch: expected Picture, got {:?}", self),
        }
    }
}

/// Index of a draw in the per-frame `scratch.frame.draws` storage.
///
/// Distinct from `PrimitiveInstanceIndex`, which identifies the scene-relative
/// primitive instance a draw was produced from. Draws are pushed as the
/// visibility pass finds them, so the two are unrelated numbers; cross between
/// them with `PrimitiveFrameScratch::draw_index_for_instance` (instance to draw,
/// fallible) or `PrimitiveDrawHeader::prim_instance_index` (draw to instance).
pub type PrimitiveDrawIndex = storage::Index<PrimitiveDrawHeader>;

/// Information stored for a visible primitive about the visible
/// rect and associated clip information.
#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "capture", derive(Serialize))]
pub struct PrimitiveDrawHeader {
    /// Back-reference to the prim instance this draw belongs to. This is the
    /// only way to get from a draw to its instance: consumers reached via the
    /// command stream hold a `PrimitiveDrawIndex`, which is unrelated to it.
    pub prim_instance_index: PrimitiveInstanceIndex,

    /// The clip chain instance that was built for this primitive.
    pub clip_chain: ClipChainInstance,

    /// Current visibility state of the primitive.
    // TODO(gw): Move more of the fields from this struct into
    //           the state enum.
    pub state: DrawState,

    /// An index into the clip task instances array in the primitive
    /// store. If this is ClipTaskIndex::INVALID, then the primitive
    /// has no clip mask. Otherwise, it may store the offset of the
    /// global clip mask task for this primitive, or the first of
    /// a list of clip task ids (one per segment).
    pub clip_task_index: ClipTaskIndex,

    /// Per-kind scratch handle for this draw. Variant matches the
    /// drawn prim's `PrimitiveKind`; `None` for kinds without per-
    /// frame scratch (e.g. ImageBorder, gradients, BackdropCapture,
    /// BoxShadow, Rectangle/YuvImage).
    pub kind_scratch: KindScratchHandle,

    /// Per-frame compositing decision for Image / YuvImage primitives.
    /// Set during the visibility pass by tile-cache promotion logic;
    /// `Blit` for kinds that aren't candidates for compositor surfaces
    /// or for draws that didn't get promoted this frame.
    pub compositor_surface_kind: CompositorSurfaceKind,

    /// Local-space rect of the primitive after device-pixel snapping has
    /// been applied. Populated for every prim each frame by the visibility
    /// pass (snapping `PrimitiveInstance.unsnapped_pattern_rect` against the
    /// surface raster node) before any visibility / prepare consumer reads it.
    pub snapped_pattern_rect: LayoutRect,
}

impl PrimitiveDrawHeader {
    /// A blank draw header, for the visibility pass to fill in and push once it
    /// knows the primitive is drawn.
    pub fn new() -> Self {
        PrimitiveDrawHeader {
            prim_instance_index: PrimitiveInstanceIndex::INVALID,
            state: DrawState::Unset,
            clip_chain: ClipChainInstance::empty(),
            clip_task_index: ClipTaskIndex::INVALID,
            kind_scratch: KindScratchHandle::None,
            compositor_surface_kind: CompositorSurfaceKind::Blit,
            snapped_pattern_rect: LayoutRect::zero(),
        }
    }

    /// Mark a pushed draw as not drawn after all, for the cases prepare only
    /// discovers late. Clears the fields prepare may already have filled in as
    /// well as the state, so that nothing stale is left reachable through the
    /// header.
    pub fn mark_culled(&mut self) {
        self.state = DrawState::Culled;
        self.clip_task_index = ClipTaskIndex::INVALID;
        self.kind_scratch = KindScratchHandle::None;
        self.compositor_surface_kind = CompositorSurfaceKind::Blit;
    }
}

pub fn update_prim_visibility(
    pic_index: PictureIndex,
    parent_surface_index: Option<SurfaceIndex>,
    root_culling_rect: &DeviceRect,
    store: &PrimitiveStore,
    is_root_tile_cache: bool,
    frame_context: &FrameVisibilityContext,
    frame_state: &mut FrameVisibilityState,
    tile_cache: &mut Option<&mut TileCacheInstance>,
 ) {
    if frame_state.visited_pictures[pic_index.0] {
        return;
    }
    frame_state.visited_pictures[pic_index.0] = true;
    let pic = &store.pictures[pic_index.0];

    let (surface_index, pop_surface) = match pic.raster_config {
        Some(RasterConfig { surface_index, composite_mode: PictureCompositeMode::TileCache { .. }, .. }) => {
            (surface_index, false)
        }
        Some(ref raster_config) => {
            frame_state.push_surface(
                pic_index,
                raster_config.surface_index,
            );

            if let Some(parent_surface_index) = parent_surface_index {
                let parent_culling_rect = frame_state
                    .surfaces[parent_surface_index.0]
                    .culling_rect;

                let surface = &mut frame_state
                    .surfaces[raster_config.surface_index.0 as usize];

                surface.update_culling_rect(
                    parent_culling_rect,
                    &raster_config.composite_mode,
                    frame_context,
                );
            }

            let surface_local_rect = frame_state.surfaces[raster_config.surface_index.0]
                .unclipped_local_rect
                .cast_unit();

            // Let the picture cache know that we are pushing an off-screen
            // surface, so it can treat dependencies of surface atomically.
            if let Some(tile_cache) = tile_cache {
                tile_cache.push_surface(
                    surface_local_rect,
                    pic.spatial_node_index,
                    frame_context.spatial_tree,
                );
            }

            (raster_config.surface_index, true)
        }
        None => {
            (parent_surface_index.expect("bug: pass-through with no parent"), false)
        }
    };

    let surface = &frame_state.surfaces[surface_index.0 as usize];
    let surface_culling_rect = surface.culling_rect;

    let mut map_local_to_picture = surface.map_local_to_picture.clone();

    let map_surface_to_vis = SpaceMapper::new_with_target(
        // TODO: switch from root to raster space.
        frame_context.root_spatial_node_index,
        surface.surface_spatial_node_index,
        surface.culling_rect,
        frame_context.spatial_tree,
    );
    let visibility_spatial_node_index = surface.visibility_spatial_node_index;

    // Snappers into this surface's raster space (the space its content is
    // rasterized in), reused across all clusters/prims in this surface (and a
    // no-op for surfaces that don't snap). `snapper` is re-targeted once per
    // cluster and snaps prim/clip-leaf rects (all prims in a cluster share its
    // spatial node, so it stays a cache hit); `clip_snapper` snaps the per-prim
    // clip chain.
    let mut snapper = SpaceSnapper::new(surface, frame_context.spatial_tree);
    let mut clip_snapper = snapper.clone();

    for cluster in &pic.prim_list.clusters {
        tracy_rs::profile_scope!("cluster");

        // No per-frame reset is needed: a draw exists only if it was pushed
        // this frame, so stale state from a previous frame cannot be observed.

        // Get the cluster and see if is visible
        if !cluster.flags.contains(ClusterFlags::IS_VISIBLE) {
            continue;
        }

        frame_state.profile.add(
            profiler::VISIBILITY_VISITED_PRIMS,
            cluster.prim_range().len(),
        );

        map_local_to_picture.set_target_spatial_node(
            cluster.spatial_node_index,
            frame_context.spatial_tree,
        );

        // Snap each prim's rect and clip-leaf rect from this cluster's
        // spatial-node space into the surface's raster space, before any
        // visibility / prepare / batch consumer reads them.
        snapper.set_target_spatial_node(cluster.spatial_node_index, frame_context.spatial_tree);

        for prim_instance_index in cluster.prim_range() {
            // A prim's snap policy is folded into its clip leaf: device-space
            // prims (text) carry the `INVALID` sentinel and snap nothing - their
            // rect and clips stay at exact sub-pixel positions so the clip keeps
            // matching the glyphs (bug 2050692). Everyone else snaps their rect
            // and own clips to the device grid. How the rect itself is rounded
            // (nearest / round-out for unsnapped text / thickness-preserving for
            // decoration lines) is decided by
            // `PrimitiveInstance::snap_policy`.
            let prim_instance = &frame_state.prim_instances[prim_instance_index];
            let leaf_id = prim_instance.clip_leaf_id;
            let snaps = frame_state.clip_tree.get_leaf(leaf_id).prim_clip_root
                != ClipNodeId::INVALID;

            let policy = prim_instance.snap_policy(snaps, frame_state.data_stores);
            let snapped_pattern_rect =
                snapper.snap_rect_rounded(&prim_instance.unsnapped_pattern_rect, policy.rect);

            // The draw header is accumulated here and pushed only once the
            // primitive is known to be drawn, so culled primitives cost nothing.
            let mut draw = PrimitiveDrawHeader::new();
            draw.prim_instance_index = PrimitiveInstanceIndex(prim_instance_index as u32);
            draw.snapped_pattern_rect = snapped_pattern_rect;

            // Picture / tile-cache leaves carry `max_rect` (snapping it would
            // overflow the snap transform); pass those through. Otherwise the
            // leaf clip rounds per the prim's clip policy: nearest for snapping
            // prims (crisp fill/border edges), exact for device-space prims.
            let leaf = frame_state.clip_tree.get_leaf_mut(leaf_id);
            let unsnapped = leaf.unsnapped_local_clip_rect;
            leaf.snapped_local_clip_rect = if unsnapped == LayoutRect::max_rect() {
                unsnapped
            } else {
                match policy.clip {
                    ClipSnap::Nearest => snapper.snap_rect(&unsnapped),
                    ClipSnap::Exact => unsnapped,
                }
            };

            if let PrimitiveKind::Picture { pic_index, .. } = frame_state.prim_instances[prim_instance_index].kind {
                if !store.pictures[pic_index.0].is_visible(frame_context.spatial_tree) {
                    continue;
                }

                let is_passthrough = match store.pictures[pic_index.0].raster_config {
                    Some(..) => false,
                    None => true,
                };

                if !is_passthrough {
                    let clip_root = store
                        .pictures[pic_index.0]
                        .clip_root
                        .unwrap_or_else(|| {
                            // If we couldn't find a common ancestor then just use the
                            // clip node of the picture primitive itself
                            let leaf_id = frame_state.prim_instances[prim_instance_index].clip_leaf_id;
                            frame_state.clip_tree.get_leaf(leaf_id).node_id
                        }
                    );

                    frame_state.clip_tree.push_clip_root_node(clip_root);
                }

                update_prim_visibility(
                    pic_index,
                    Some(surface_index),
                    root_culling_rect,
                    store,
                    false,
                    frame_context,
                    frame_state,
                    tile_cache,
                );

                if is_passthrough {
                    // Pass through pictures are always considered visible in all dirty tiles.
                    draw.state = DrawState::PassThrough;
                    frame_state.scratch.primitive.frame.push_draw(draw);

                    continue;
                } else {
                    frame_state.clip_tree.pop_clip_root();
                }
            }

            let prim_instance = &mut frame_state.prim_instances[prim_instance_index];

            let local_coverage_rect = frame_state.data_stores.get_local_prim_coverage_rect(
                prim_instance,
                draw.snapped_pattern_rect,
                &store.pictures,
                frame_state.surfaces,
            );

            frame_state.clip_store.set_active_clips(
                cluster.spatial_node_index,
                map_local_to_picture.ref_spatial_node_index,
                visibility_spatial_node_index,
                &mut clip_snapper,
                policy.clip,
                prim_instance.clip_leaf_id,
                &frame_context.spatial_tree,
                &frame_state.data_stores.clip,
                frame_state.clip_tree,
            );

            let clip_chain = frame_state
                .clip_store
                .build_clip_chain_instance(
                    local_coverage_rect,
                    &map_local_to_picture,
                    &map_surface_to_vis,
                    &mut frame_state.frame_gpu_data.f32,
                    frame_state.resource_cache,
                    &surface_culling_rect,
                    &frame_state.data_stores.clip,
                    frame_state.rg_builder,
                    true,
                );

            let clip_chain = match clip_chain {
                Some(clip_chain) => clip_chain,
                None => {
                    continue;
                }
            };
            draw.clip_chain = clip_chain;

            // Everything below needs a draw index (the tile-cache dependency
            // update records one on any compositor surface it promotes), so the
            // draw is pushed here. A primitive that `update_prim_dependencies`
            // then culls keeps its draw, and prepare skips it on `DrawState`.
            let draw_index = frame_state.scratch.primitive.frame.push_draw(draw);

            let is_mix_blend_picture = |prim_instance: &PrimitiveInstance| {
                if let PrimitiveKind::Picture { pic_index, .. } = prim_instance.kind {
                    let pic = &store.pictures[pic_index.0];

                    matches!(
                        pic.composite_mode,
                        Some(PictureCompositeMode::MixBlend(_))
                    )
                } else {
                    false
                }
            };

            if is_root_tile_cache && is_mix_blend_picture(prim_instance) {
                if let Some(tile_cache) = tile_cache {
                    tile_cache.mix_blend_pic_rects.push(clip_chain.pic_coverage_rect);
                }
            }

            {
                let prim_surface_index = frame_state.surface_stack.last().unwrap().1;

                // Accumulate the exact (clipped) local rect into the parent surface.
                let surface = &mut frame_state.surfaces[prim_surface_index.0];
                surface.clipped_local_rect =
                    surface.clipped_local_rect.union(&clip_chain.pic_coverage_rect);
            }

            let new_state = match tile_cache {
                Some(tile_cache) => {
                    tile_cache.update_prim_dependencies(
                        draw_index,
                        prim_instance,
                        cluster.spatial_node_index,
                        // It's OK to pass the local_coverage_rect here as it's only
                        // used by primitives (for compositor surfaces) that don't
                        // have inflation anyway.
                        local_coverage_rect,
                        frame_context,
                        frame_state.data_stores,
                        frame_state.clip_store,
                        &store.pictures,
                        frame_state.resource_cache,
                        &frame_state.surface_stack,
                        &mut frame_state.composite_state,
                        &mut frame_state.frame_gpu_data.f32,
                        &mut frame_state.scratch.primitive,
                        is_root_tile_cache,
                        frame_state.surfaces,
                        frame_state.profile,
                    )
                }
                None => {
                    DrawState::Visible {
                        vis_flags: PrimitiveVisibilityFlags::empty(),
                        sub_slice_index: SubSliceIndex::DEFAULT,
                    }
                }
            };
            frame_state.scratch.primitive.frame.draw_mut(draw_index).state = new_state;
        }
    }

    if let Some(snapshot) = &pic.snapshot {
        if snapshot.detached {
            // If the snapshot is detached, then the contents of the stacking
            // context will only be shown via the snapshot, so there is no point
            // to rendering anything outside of the snapshot area.
            let prim_surface_index = frame_state.surface_stack.last().unwrap().1;
            let surface = &mut frame_state.surfaces[prim_surface_index.0];
            let clip = snapshot.area.round_out().cast_unit();
            surface.clipped_local_rect = surface.clipped_local_rect.intersection_unchecked(&clip);
        }
    }

    if pop_surface {
        frame_state.pop_surface();
    }

    if let Some(ref rc) = pic.raster_config {
        if let Some(tile_cache) = tile_cache {
            match rc.composite_mode {
                PictureCompositeMode::TileCache { .. } => {}
                _ => {
                    // Pop the off-screen surface from the picture cache stack
                    tile_cache.pop_surface();
                }
            }
        }
    }
}

/// The part of a primitive's local rect that the surface it is drawn into needs.
///
/// This is the rect to enumerate repetitions or image tiles against. For a
/// primitive drawn straight onto a picture cache slice the surface's clipping
/// rect is the dirty region, so only the repetitions that will be rasterized
/// are emitted. For a surface that samples outside of its own footprint (a
/// blur) it is the inflated region that surface needs, so the repetitions
/// feeding the blur's margin are kept even though they fall outside the dirty
/// region (bug 2064321).
///
/// `bounds` is the primitive's own extent: the result never exceeds it, and it
/// is the fallback if the primitive's transform cannot be inverted.
pub fn compute_surface_visible_rect(
    surface: &SurfaceInfo,
    clip_chain: &ClipChainInstance,
    prim_spatial_node_index: SpatialNodeIndex,
    bounds: &LayoutRect,
    spatial_tree: &SpatialTree,
) -> LayoutRect {
    let map_prim_to_surface: SpaceMapper<LayoutPixel, PicturePixel> = SpaceMapper::new_with_target(
        surface.surface_spatial_node_index,
        prim_spatial_node_index,
        PictureRect::max_rect(),
        spatial_tree,
    );

    surface.clipping_rect
        .intersection(&clip_chain.pic_coverage_rect)
        .and_then(|rect| map_prim_to_surface.unmap(&rect))
        .unwrap_or(*bounds)
        .intersection_unchecked(bounds)
}
