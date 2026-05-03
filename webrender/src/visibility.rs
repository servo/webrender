/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! # Visibility pass
//!
//! TODO: document what this pass does!
//!

use api::DebugFlags;
use api::units::*;
use std::usize;
use crate::clip::ClipStore;
use crate::composite::CompositeState;
use crate::profiler::TransactionProfile;
use crate::renderer::GpuBufferBuilder;
use crate::spatial_tree::{SpatialTree, SpatialNodeIndex};
use crate::clip::{ClipChainInstance, ClipTree};
use crate::composite::CompositorSurfaceKind;
use crate::frame_builder::FrameBuilderConfig;
use crate::picture::{PictureCompositeMode, ClusterFlags, SurfaceInfo};
use crate::tile_cache::TileCacheInstance;
use crate::picture::{PictureScratch, SurfaceIndex, RasterConfig};
use crate::tile_cache::SubSliceIndex;
use crate::prim_store::{ClipTaskIndex, PictureIndex, PrimitiveKind, SegmentInstanceIndex};
use crate::prim_store::{PrimitiveStore, PrimitiveInstance, PrimitiveInstanceIndex};
use crate::prim_store::backdrop::BackdropRenderScratch;
use crate::prim_store::borders::{ImageBorderScratch, NormalBorderScratch};
use crate::prim_store::gradient::LinearGradientScratch;
use crate::prim_store::image::ImageScratch;
use crate::prim_store::line_dec::LineDecorationScratch;
use crate::prim_store::storage;
use crate::prim_store::text_run::TextRunScratch;
use crate::render_backend::{DataStores, ScratchBuffer};
use crate::render_task_graph::RenderTaskGraphBuilder;
use crate::resource_cache::ResourceCache;
use crate::scene::SceneProperties;
use crate::space::SpaceMapper;
use crate::util::MaxRect;

pub struct FrameVisibilityContext<'a> {
    pub spatial_tree: &'a SpatialTree,
    pub global_screen_world_rect: WorldRect,
    pub global_device_pixel_scale: DevicePixelScale,
    pub debug_flags: DebugFlags,
    pub scene_properties: &'a SceneProperties,
    pub config: FrameBuilderConfig,
    pub root_spatial_node_index: SpatialNodeIndex,
}

pub struct FrameVisibilityState<'a> {
    pub clip_store: &'a mut ClipStore,
    pub resource_cache: &'a mut ResourceCache,
    pub frame_gpu_data: &'a mut GpuBufferBuilder,
    pub data_stores: &'a mut DataStores,
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
    /// Uninitialized - this should never be encountered after prim reset
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
    LineDecoration(storage::Index<LineDecorationScratch>),
    NormalBorder(storage::Index<NormalBorderScratch>),
    ImageBorder(storage::Index<ImageBorderScratch>),
    LinearGradient(storage::Index<LinearGradientScratch>),
    Image(storage::Index<ImageScratch>),
    TextRun(storage::Index<TextRunScratch>),
    Picture(storage::Index<PictureScratch>),
    BackdropRender(storage::Index<BackdropRenderScratch>),
}

impl KindScratchHandle {
    /// Extract the LineDecoration scratch index. Panics if the variant
    /// doesn't match — readers in the LineDecoration arm of the
    /// PrimitiveKind match know the variant by construction.
    pub fn unwrap_line_decoration(&self) -> storage::Index<LineDecorationScratch> {
        match *self {
            KindScratchHandle::LineDecoration(h) => h,
            _ => panic!("kind_scratch mismatch: expected LineDecoration, got {:?}", self),
        }
    }
    pub fn unwrap_normal_border(&self) -> storage::Index<NormalBorderScratch> {
        match *self {
            KindScratchHandle::NormalBorder(h) => h,
            _ => panic!("kind_scratch mismatch: expected NormalBorder, got {:?}", self),
        }
    }
    pub fn unwrap_image_border(&self) -> storage::Index<ImageBorderScratch> {
        match *self {
            KindScratchHandle::ImageBorder(h) => h,
            _ => panic!("kind_scratch mismatch: expected ImageBorder, got {:?}", self),
        }
    }
    pub fn unwrap_image(&self) -> storage::Index<ImageScratch> {
        match *self {
            KindScratchHandle::Image(h) => h,
            _ => panic!("kind_scratch mismatch: expected Image, got {:?}", self),
        }
    }
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
    pub fn unwrap_backdrop_render(&self) -> storage::Index<BackdropRenderScratch> {
        match *self {
            KindScratchHandle::BackdropRender(h) => h,
            _ => panic!("kind_scratch mismatch: expected BackdropRender, got {:?}", self),
        }
    }
}

/// Information stored for a visible primitive about the visible
/// rect and associated clip information.
#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "capture", derive(Serialize))]
pub struct PrimitiveDrawHeader {
    /// Back-reference to the prim instance this draw belongs to.
    /// Currently redundant with the identity-indexed lookup from
    /// `scratch.frame.draws[PrimitiveInstanceIndex.0]`, but reserved
    /// for a follow-up that switches the storage to push-per-draw —
    /// readers iterating draws directly will need this to reach the
    /// instance.
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

    /// Index into PrimitiveFrameScratch.segment_instances for prims
    /// that opt into segmented brush rendering (Rectangle, YuvImage,
    /// non-tiled Image). UNUSED for prims that don't segment, or for
    /// the trivial single-segment case. Built fresh each frame in
    /// build_segments_if_needed.
    pub segment_instance_index: SegmentInstanceIndex,

    /// Per-frame compositing decision for Image / YuvImage primitives.
    /// Set during the visibility pass by tile-cache promotion logic;
    /// `Blit` for kinds that aren't candidates for compositor surfaces
    /// or for draws that didn't get promoted this frame.
    pub compositor_surface_kind: CompositorSurfaceKind,
}

impl PrimitiveDrawHeader {
    pub fn new() -> Self {
        PrimitiveDrawHeader {
            prim_instance_index: PrimitiveInstanceIndex::INVALID,
            state: DrawState::Unset,
            clip_chain: ClipChainInstance::empty(),
            clip_task_index: ClipTaskIndex::INVALID,
            kind_scratch: KindScratchHandle::None,
            segment_instance_index: SegmentInstanceIndex::UNUSED,
            compositor_surface_kind: CompositorSurfaceKind::Blit,
        }
    }

    pub fn reset(&mut self) {
        self.state = DrawState::Culled;
        self.clip_task_index = ClipTaskIndex::INVALID;
        self.kind_scratch = KindScratchHandle::None;
        self.segment_instance_index = SegmentInstanceIndex::UNUSED;
        self.compositor_surface_kind = CompositorSurfaceKind::Blit;
    }
}

pub fn update_prim_visibility(
    pic_index: PictureIndex,
    parent_surface_index: Option<SurfaceIndex>,
    world_culling_rect: &WorldRect,
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

    for cluster in &pic.prim_list.clusters {
        profile_scope!("cluster");

        // Each prim instance must have reset called each frame, to clear
        // indices into various scratch buffers. If this doesn't occur,
        // the primitive may incorrectly be considered visible, which can
        // cause unexpected conditions to occur later during the frame.
        // Primitive instances are normally reset in the main loop below,
        // but we must also reset them in the rare case that the cluster
        // visibility has changed (due to an invalid transform and/or
        // backface visibility changing for this cluster).
        // TODO(gw): This is difficult to test for in CI - as a follow up,
        //           we should add a debug flag that validates the prim
        //           instance is always reset every frame to catch similar
        //           issues in future.
        for idx in cluster.prim_range() {
            frame_state.scratch.primitive.frame.draws[idx].reset();
            frame_state.scratch.primitive.frame.draws[idx].prim_instance_index =
                PrimitiveInstanceIndex(idx as u32);
        }

        // Get the cluster and see if is visible
        if !cluster.flags.contains(ClusterFlags::IS_VISIBLE) {
            continue;
        }

        map_local_to_picture.set_target_spatial_node(
            cluster.spatial_node_index,
            frame_context.spatial_tree,
        );

        for prim_instance_index in cluster.prim_range() {
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
                    world_culling_rect,
                    store,
                    false,
                    frame_context,
                    frame_state,
                    tile_cache,
                );

                if is_passthrough {
                    // Pass through pictures are always considered visible in all dirty tiles.
                    frame_state.scratch.primitive.frame.draws[prim_instance_index].state = DrawState::PassThrough;

                    continue;
                } else {
                    frame_state.clip_tree.pop_clip_root();
                }
            }

            let prim_instance = &mut frame_state.prim_instances[prim_instance_index];

            let local_coverage_rect = frame_state.data_stores.get_local_prim_coverage_rect(
                prim_instance,
                &store.pictures,
                frame_state.surfaces,
            );

            frame_state.clip_store.set_active_clips(
                cluster.spatial_node_index,
                map_local_to_picture.ref_spatial_node_index,
                visibility_spatial_node_index,
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
                    &frame_context.spatial_tree,
                    &mut frame_state.frame_gpu_data.f32,
                    frame_state.resource_cache,
                    &surface_culling_rect,
                    &mut frame_state.data_stores.clip,
                    frame_state.rg_builder,
                    true,
                );

            frame_state.scratch.primitive.frame.draws[prim_instance_index].clip_chain = match clip_chain {
                Some(clip_chain) => clip_chain,
                None => {
                    continue;
                }
            };

            {
                let prim_surface_index = frame_state.surface_stack.last().unwrap().1;
                let prim_clip_chain = &frame_state.scratch.primitive.frame.draws[prim_instance_index].clip_chain;

                // Accumulate the exact (clipped) local rect into the parent surface.
                let surface = &mut frame_state.surfaces[prim_surface_index.0];
                surface.clipped_local_rect = surface.clipped_local_rect.union(&prim_clip_chain.pic_coverage_rect);
            }

            let new_state = match tile_cache {
                Some(tile_cache) => {
                    tile_cache.update_prim_dependencies(
                        PrimitiveInstanceIndex(prim_instance_index as u32),
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
            frame_state.scratch.primitive.frame.draws[prim_instance_index].state = new_state;
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

pub fn compute_conservative_visible_rect(
    clip_chain: &ClipChainInstance,
    culling_rect: VisRect,
    visibility_node_index: SpatialNodeIndex,
    prim_spatial_node_index: SpatialNodeIndex,
    spatial_tree: &SpatialTree,
) -> LayoutRect {
    // Mapping from picture space -> world space
    let map_pic_to_vis: SpaceMapper<PicturePixel, VisPixel> = SpaceMapper::new_with_target(
        visibility_node_index,
        clip_chain.pic_spatial_node_index,
        culling_rect,
        spatial_tree,
    );

    // Mapping from local space -> picture space
    let map_local_to_pic: SpaceMapper<LayoutPixel, PicturePixel> = SpaceMapper::new_with_target(
        clip_chain.pic_spatial_node_index,
        prim_spatial_node_index,
        PictureRect::max_rect(),
        spatial_tree,
    );

    // Unmap the world culling rect from world -> picture space. If this mapping fails due
    // to matrix weirdness, best we can do is use the clip chain's local clip rect.
    let pic_culling_rect = match map_pic_to_vis.unmap(&culling_rect) {
        Some(rect) => rect,
        None => return clip_chain.local_clip_rect,
    };

    // Intersect the unmapped world culling rect with the primitive's clip chain rect that
    // is in picture space (the clip-chain already takes into account the bounds of the
    // primitive local_rect and local_clip_rect). If there is no intersection here, the
    // primitive is not visible at all.
    let pic_culling_rect = match pic_culling_rect.intersection(&clip_chain.pic_coverage_rect) {
        Some(rect) => rect,
        None => return LayoutRect::zero(),
    };

    // Unmap the picture culling rect from picture -> local space. If this mapping fails due
    // to matrix weirdness, best we can do is use the clip chain's local clip rect.
    match map_local_to_pic.unmap(&pic_culling_rect) {
        Some(rect) => rect,
        None => clip_chain.local_clip_rect,
    }
}
