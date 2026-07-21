/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::ColorF;
use api::PropertyBindingId;
use api::units::*;
use smallvec::SmallVec;
use crate::composite::CompositeState;
use crate::internal_types::{FastHashMap, FrameId};
use crate::invalidation::compare::ImageDependency;
use crate::invalidation::compare::{ColorBinding, OpacityBinding, OpacityBindingInfo, PrimitiveComparisonKey};
use crate::invalidation::compare::{PrimitiveComparer, PrimitiveDependency, ColorBindingInfo};
use crate::invalidation::{InvalidationReason, PrimitiveCompareResult, quadtree::TileNode};
use crate::invalidation::vert_buffer::{CornersCache, VertRange};
use crate::intern::ItemUid;
use crate::picture::{PictureCompositeMode, SurfaceIndex, clampf};
use crate::print_tree::PrintTreePrinter;
use crate::resource_cache::ResourceCache;
use crate::space::SpaceMapper;
use crate::visibility::FrameVisibilityContext;
use peek_poke::poke_into_vec;
use std::mem;

pub struct CachedSurface {
    pub current_descriptor: CachedSurfaceDescriptor,
    pub prev_descriptor: CachedSurfaceDescriptor,
    pub is_valid: bool,
    pub local_valid_rect: PictureBox2D,
    pub local_dirty_rect: PictureRect,
    pub local_rect: PictureRect,
    pub root: TileNode,
    pub background_color: Option<ColorF>,
    pub invalidation_reason: Option<InvalidationReason>,
    pub sub_graphs: Vec<(PictureRect, Vec<(PictureCompositeMode, SurfaceIndex)>)>,
}

impl CachedSurface {
    pub fn new() -> Self {
        CachedSurface {
            current_descriptor: CachedSurfaceDescriptor::new(),
            prev_descriptor: CachedSurfaceDescriptor::new(),
            is_valid: false,
            local_valid_rect: PictureBox2D::zero(),
            local_dirty_rect: PictureRect::zero(),
            local_rect: PictureRect::zero(),
            root: TileNode::new_leaf(Vec::new()),
            background_color: None,
            invalidation_reason: None,
            sub_graphs: Vec::new(),
        }
    }

    pub fn print(&self, pt: &mut dyn PrintTreePrinter) {
        pt.add_item(format!("background_color: {:?}", self.background_color));
        pt.add_item(format!("invalidation_reason: {:?}", self.invalidation_reason));
        self.current_descriptor.print(pt);
    }

    /// Setup state before primitive dependency calculation.
    pub fn pre_update(
        &mut self,
        background_color: Option<ColorF>,
        local_tile_rect: PictureRect,
        frame_id: FrameId,
        is_visible: bool,
    ) {
        // TODO(gw): This is a hack / fix for Box2D::union in euclid not working with
        //           zero sized rect accumulation. Once that lands, we'll revert this
        //           to be zero.
        self.local_valid_rect = PictureBox2D::new(
            PicturePoint::new( 1.0e32,  1.0e32),
            PicturePoint::new(-1.0e32, -1.0e32),
        );
        self.invalidation_reason  = None;
        self.sub_graphs.clear();

        // If the tile isn't visible, early exit, skipping the normal set up to
        // validate dependencies. Instead, we will only compare the current tile
        // dependencies the next time it comes into view.
        if !is_visible {
            return;
        }

        if background_color != self.background_color {
            self.invalidate(None, InvalidationReason::BackgroundColor);
            self.background_color = background_color;
        }

        // Clear any dependencies so that when we rebuild them we
        // can compare if the tile has the same content.
        mem::swap(
            &mut self.current_descriptor,
            &mut self.prev_descriptor,
        );
        self.current_descriptor.clear();
        self.root.clear(local_tile_rect);

        self.current_descriptor.last_updated_frame_id = frame_id;
    }

    pub fn add_prim_dependency(
        &mut self,
        info: &PrimitiveDependencyInfo,
        corners_cache: &CornersCache,
        prim_clamp_to_tile: bool,
        local_raster_rect: &RasterRect,
        local_tile_rect: PictureRect,
    ) {
        // Incorporate the bounding rect of the primitive in the local valid rect
        // for this tile. This is used to minimize the size of the scissor rect
        // during rasterization and the draw rect during composition of partial tiles.
        self.local_valid_rect = self.local_valid_rect.union(&info.prim_clip_box);

        // TODO(gw): The prim_clip_rect can be impacted by the clip rect of the display port,
        //           which can cause invalidations when a new display list with changed
        //           display port is received. To work around this, clamp the prim clip rect
        //           to the tile boundaries - if the clip hasn't affected the tile, then the
        //           changed clip can't affect the content of the primitive on this tile.
        //           In future, we could consider supplying the display port clip from Gecko
        //           in a different way (e.g. as a scroll frame clip) which still provides
        //           the desired clip for checkerboarding, but doesn't require this extra
        //           work below.

        // TODO(gw): This is a hot part of the code - we could probably optimize further by:
        //           - Using min/max instead of clamps below (if we guarantee the rects are well formed)

        let pmin = local_tile_rect.min;
        let pmax = local_tile_rect.max;

        let prim_clip_box = PictureBox2D::new(
            PicturePoint::new(
                clampf(info.prim_clip_box.min.x, pmin.x, pmax.x),
                clampf(info.prim_clip_box.min.y, pmin.y, pmax.y),
            ),
            PicturePoint::new(
                clampf(info.prim_clip_box.max.x, pmin.x, pmax.x),
                clampf(info.prim_clip_box.max.y, pmin.y, pmax.y),
            ),
        );

        // Push raster-space corners into this tile's vert_data, clamping if requested.
        let vert_data = &mut self.current_descriptor.vert_data;
        let (prim_corners, coverage_corners) = if prim_clamp_to_tile {
            (
                corners_cache.push_verts_clamped(info.prim_scratch, local_raster_rect, vert_data),
                corners_cache.push_verts_clamped(info.cov_scratch, local_raster_rect, vert_data),
            )
        } else {
            (
                corners_cache.push_verts(info.prim_scratch, vert_data),
                corners_cache.push_verts(info.cov_scratch, vert_data),
            )
        };

        // Update the tile descriptor, used for tile comparison during scene swaps.
        let prim_index = PrimitiveDependencyIndex(self.current_descriptor.prims.len() as u32);

        // Encode the deps for this primitive in the `dep_data` byte buffer.
        let dep_offset = self.current_descriptor.dep_data.len() as u32;
        let mut dep_count = 0;

        for &(clip_uid, clip_scratch) in info.clips.iter() {
            dep_count += 1;
            poke_into_vec(
                &PrimitiveDependency::Clip { prim_uid: clip_uid, vert_range: corners_cache.push_verts(clip_scratch, vert_data) },
                &mut self.current_descriptor.dep_data,
            );
        }

        for image in &info.images {
            dep_count += 1;
            poke_into_vec(
                &PrimitiveDependency::Image {
                    image: *image,
                },
                &mut self.current_descriptor.dep_data,
            );
        }

        for binding in &info.opacity_bindings {
            dep_count += 1;
            poke_into_vec(
                &PrimitiveDependency::OpacityBinding {
                    binding: *binding,
                },
                &mut self.current_descriptor.dep_data,
            );
        }

        if let Some(ref binding) = info.color_binding {
            dep_count += 1;
            poke_into_vec(
                &PrimitiveDependency::ColorBinding {
                    binding: *binding,
                },
                &mut self.current_descriptor.dep_data,
            );
        }

        if info.raster_space_animating {
            dep_count += 1;
            poke_into_vec(
                &PrimitiveDependency::AnimatedRasterSpace {
                    animating: true,
                },
                &mut self.current_descriptor.dep_data,
            );
        }

        self.current_descriptor.prims.push(PrimitiveDescriptor {
            prim_clip_box,
            dep_offset,
            dep_count,
            prim_uid: info.prim_uid,
            prim_corners,
            coverage_corners,
        });

        // Add this primitive to the dirty rect quadtree.
        self.root.add_prim(prim_index, &info.prim_clip_box);
    }

    /// Check if the content of the previous and current tile descriptors match
    fn update_dirty_rects(
        &mut self,
        ctx: &TileUpdateDirtyContext,
        state: &mut TileUpdateDirtyState,
        invalidation_reason: &mut Option<InvalidationReason>,
        frame_context: &FrameVisibilityContext,
    ) -> PictureRect {
        let mut prim_comparer = PrimitiveComparer::new(
            &self.prev_descriptor,
            &self.current_descriptor,
            state.resource_cache,
            ctx.opacity_bindings,
            ctx.color_bindings,
        );

        let mut dirty_rect = PictureBox2D::zero();
        self.root.update_dirty_rects(
            &self.prev_descriptor.prims,
            &self.current_descriptor.prims,
            &mut prim_comparer,
            &mut dirty_rect,
            state.compare_cache,
            invalidation_reason,
            frame_context,
        );

        dirty_rect
    }

    /// Invalidate a tile based on change in content. This
    /// must be called even if the tile is not currently
    /// visible on screen. We might be able to improve this
    /// later by changing how ComparableVec is used.
    pub fn update_content_validity(
        &mut self,
        ctx: &TileUpdateDirtyContext,
        state: &mut TileUpdateDirtyState,
        frame_context: &FrameVisibilityContext,
    ) {
        // Check if the contents of the primitives, clips, and
        // other dependencies are the same.
        state.compare_cache.clear();
        let mut invalidation_reason = None;
        let dirty_rect = self.update_dirty_rects(
            ctx,
            state,
            &mut invalidation_reason,
            frame_context,
        );

        if !dirty_rect.is_empty() {
            self.invalidate(
                Some(dirty_rect),
                invalidation_reason.expect("bug: no invalidation_reason")
            );
        }
        if ctx.invalidate_all {
            self.invalidate(None, InvalidationReason::ScaleChanged);
        }
        // TODO(gw): We can avoid invalidating the whole tile in some cases here,
        //           but it should be a fairly rare invalidation case.
        if self.current_descriptor.local_valid_rect != self.prev_descriptor.local_valid_rect {
            self.invalidate(None, InvalidationReason::ValidRectChanged);
            state.composite_state.dirty_rects_are_valid = false;
        }
    }

    /// Invalidate this tile. If `invalidation_rect` is None, the entire
    /// tile is invalidated.
    pub fn invalidate(
        &mut self,
        invalidation_rect: Option<PictureRect>,
        reason: InvalidationReason,
    ) {
        self.is_valid = false;

        match invalidation_rect {
            Some(rect) => {
                self.local_dirty_rect = self.local_dirty_rect.union(&rect);
            }
            None => {
                self.local_dirty_rect = self.local_rect;
            }
        }

        if self.invalidation_reason.is_none() {
            self.invalidation_reason = Some(reason);
        }
    }
}

// Immutable context passed to picture cache tiles during update_dirty_and_valid_rects
pub struct TileUpdateDirtyContext<'a> {
    /// Maps from picture cache coords -> world space coords.
    pub pic_to_world_mapper: SpaceMapper<PicturePixel, WorldPixel>,

    /// Global scale factor from world -> device pixels.
    pub global_device_pixel_scale: DevicePixelScale,

    /// Information about opacity bindings from the picture cache.
    pub opacity_bindings: &'a FastHashMap<PropertyBindingId, OpacityBindingInfo>,

    /// Information about color bindings from the picture cache.
    pub color_bindings: &'a FastHashMap<PropertyBindingId, ColorBindingInfo>,

    /// The local rect of the overall picture cache
    pub local_rect: PictureRect,

    /// If true, the scale factor of the root transform for this picture
    /// cache changed, so we need to invalidate the tile and re-render.
    pub invalidate_all: bool,
}

// Mutable state passed to picture cache tiles during update_dirty_and_valid_rects
pub struct TileUpdateDirtyState<'a> {
    /// Allow access to the texture cache for requesting tiles
    pub resource_cache: &'a mut ResourceCache,

    /// Current configuration and setup for compositing all the picture cache tiles in renderer.
    pub composite_state: &'a mut CompositeState,

    /// A cache of comparison results to avoid re-computation during invalidation.
    pub compare_cache: &'a mut FastHashMap<PrimitiveComparisonKey, PrimitiveCompareResult>,
}

/// Information about the dependencies of a single primitive instance.
/// Built once per primitive (outside the tile loop); passed to each tile's
/// add_prim_dependency, which does the per-tile clamping and vert-data push.
pub struct PrimitiveDependencyInfo {
    /// The (conservative) clipped area in picture space this primitive occupies.
    /// Used for local_valid_rect accumulation and quadtree binning.
    pub prim_clip_box: PictureBox2D,
    /// Image keys this primitive depends on.
    pub images: SmallVec<[ImageDependency; 8]>,
    /// Opacity bindings this primitive depends on.
    pub opacity_bindings: SmallVec<[OpacityBinding; 4]>,
    /// Color binding this primitive depends on.
    pub color_binding: Option<ColorBinding>,
    /// Intern uid for this primitive instance. Stable across frames and across
    /// content-side scroll events: scene building normalises each primitive's
    /// prim_rect by the accumulated external_scroll_offset before interning,
    /// so the key (and therefore this uid) does not change when the scroll
    /// position changes. If external_scroll_offset is ever removed, this
    /// stability guarantee would need to be preserved by another mechanism.
    pub prim_uid: ItemUid,
    /// Scratch range for the primitive's rect corners in raster space (unquantized).
    /// Quantized into per-tile vert_data inside add_prim_dependency.
    pub prim_scratch: VertRange,
    /// Scratch range for the coverage rect corners (prim ∩ clip) in raster space.
    /// Tracked separately from prim_scratch because merging them into a single
    /// intersection loses prim-rect information: a UV-mapped primitive whose
    /// rect changes size while the clip keeps the visible region constant would
    /// produce an unchanged intersection yet sample different source pixels.
    /// Using coverage_rect (rather than the raw local_clip_rect) avoids
    /// spurious invalidations when the clip changes outside the prim extent.
    pub cov_scratch: VertRange,
    /// Per-clip data: (clip intern uid, scratch range for clip corners).
    /// The uid covers the clip's shape/mode; position is captured in the scratch range.
    pub clips: SmallVec<[(ItemUid, VertRange); 4]>,
    /// Set for a text run whose spatial node (or an ancestor) has an animated
    /// transform, so it is rasterized in local space (bug 2056306). Encoded as
    /// an `AnimatedRasterSpace` dep only when true, so the tile invalidates when
    /// the animation ends and the text returns to the crisp device path.
    pub raster_space_animating: bool,
}

impl PrimitiveDependencyInfo {
    pub fn new(prim_uid: ItemUid, prim_clip_box: PictureBox2D) -> Self {
        PrimitiveDependencyInfo {
            prim_clip_box,
            images: smallvec::SmallVec::new(),
            opacity_bindings: smallvec::SmallVec::new(),
            color_binding: None,
            prim_uid,
            prim_scratch: VertRange::INVALID,
            cov_scratch: VertRange::INVALID,
            clips: smallvec::SmallVec::new(),
            raster_space_animating: false,
        }
    }
}

/// Information about a primitive that is a dependency for a cached surface.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct PrimitiveDescriptor {
    /// Picture-space bounds, clamped to tile boundary. Used for quadtree
    /// binning and local_valid_rect; not used for comparison.
    pub prim_clip_box: PictureBox2D,
    // TODO(gw): These two fields could be packed as a u24/u8
    pub dep_offset: u32,
    pub dep_count: u32,
    /// Intern uid for this primitive. See PrimitiveDependencyInfo::prim_uid for the
    /// scroll-stability guarantee.
    pub prim_uid: ItemUid,
    /// Range into vert_data for this primitive's corners (local_prim_rect).
    pub prim_corners: VertRange,
    /// Range into vert_data for the coverage rect corners (prim ∩ clip).
    pub coverage_corners: VertRange,
}


/// An index into the prims array in a TileDescriptor.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct PrimitiveDependencyIndex(pub u32);

/// Uniquely describes the content of this cached surface, in a way that can be
/// (reasonably) efficiently hashed and compared.
#[cfg_attr(any(feature="capture",feature="replay"), derive(Clone))]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct CachedSurfaceDescriptor {
    /// List of primitive instance unique identifiers. The uid is guaranteed
    /// to uniquely describe the content of the primitive template, while
    /// the other parameters describe the clip chain and instance params.
    pub prims: Vec<PrimitiveDescriptor>,

    /// Picture space rect that contains valid pixels region of this tile.
    pub local_valid_rect: PictureRect,

    /// The last frame this tile had its dependencies updated (dependency updating is
    /// skipped if a tile is off-screen).
    pub last_updated_frame_id: FrameId,

    /// Packed per-prim dependency information
    pub dep_data: Vec<u8>,

    /// Per-tile quantized raster-space vert data. VertRanges stored in
    /// PrimitiveDescriptor and PrimitiveDependency::Clip index into this buffer.
    pub vert_data: Vec<i32>,
}

impl CachedSurfaceDescriptor {
    pub fn new() -> Self {
        CachedSurfaceDescriptor {
            local_valid_rect: PictureRect::zero(),
            dep_data: Vec::new(),
            vert_data: Vec::new(),
            prims: Vec::new(),
            last_updated_frame_id: FrameId::INVALID,
        }
    }

    /// Print debug information about this tile descriptor to a tree printer.
    pub fn print(&self, pt: &mut dyn crate::print_tree::PrintTreePrinter) {
        pt.new_level("current_descriptor".to_string());

        pt.new_level("prims".to_string());
        for prim in &self.prims {
            pt.new_level(format!("prim uid={}", prim.prim_uid.get_uid()));
            pt.add_item(format!("clip: p0={},{} p1={},{}",
                prim.prim_clip_box.min.x,
                prim.prim_clip_box.min.y,
                prim.prim_clip_box.max.x,
                prim.prim_clip_box.max.y,
            ));
            pt.end_level();
        }
        pt.end_level();

        pt.end_level();
    }

    /// Clear the dependency information for a tile, when the dependencies
    /// are being rebuilt.
    pub fn clear(&mut self) {
        self.local_valid_rect = PictureRect::zero();
        self.prims.clear();
        self.dep_data.clear();
        self.vert_data.clear();
    }
}
