/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Invalidation tracking and dirty region management
//!
//! This module contains types and logic for tracking dirty regions and
//! dependencies used to determine what needs to be redrawn each frame.

pub mod compare;
pub mod quadtree;
pub mod cached_surface;
pub mod vert_buffer;

use api::units::*;
use crate::spatial_tree::{SpatialTree, SpatialNodeIndex};
use crate::space::SpaceMapper;
use crate::util::MaxRect;

/// Represents the dirty region of a tile cache picture, relative to a
/// "visibility" spatial node. At the moment the visibility node is
/// world space, but the plan is to switch to raster space.
///
/// The plan is to move away from these world space representation and
/// compute dirty regions in raster space instead.
#[derive(Clone)]
pub struct DirtyRegion {
    /// The overall dirty rect, a combination of dirty_rects
    pub combined: VisRect,

    /// The coordinate space used to do clipping, visibility, and
    /// dirty rect calculations.
    pub visibility_spatial_node: SpatialNodeIndex,
    /// Spatial node of the picture this region represents.
    local_spatial_node: SpatialNodeIndex,
}

impl DirtyRegion {
    /// Construct a new dirty region tracker.
    pub fn new(
        visibility_spatial_node: SpatialNodeIndex,
        local_spatial_node: SpatialNodeIndex,
    ) -> Self {
        DirtyRegion {
            combined: VisRect::zero(),
            visibility_spatial_node,
            local_spatial_node,
        }
    }

    /// Reset the dirty regions back to empty
    pub fn reset(
        &mut self,
        visibility_spatial_node: SpatialNodeIndex,
        local_spatial_node: SpatialNodeIndex,
    ) {
        self.combined = VisRect::zero();
        self.visibility_spatial_node = visibility_spatial_node;
        self.local_spatial_node = local_spatial_node;
    }

    /// Add a dirty region to the tracker. Returns the visibility mask that corresponds to
    /// this region in the tracker.
    pub fn add_dirty_region(
        &mut self,
        rect_in_pic_space: PictureRect,
        spatial_tree: &SpatialTree,
    ) {
        debug_assert_ne!(
            self.visibility_spatial_node,
            SpatialNodeIndex::INVALID,
            "dirty region used before being targeted for this frame",
        );

        let map_pic_to_raster = SpaceMapper::new_with_target(
            self.visibility_spatial_node,
            self.local_spatial_node,
            VisRect::max_rect(),
            spatial_tree,
        );

        let raster_rect = map_pic_to_raster
            .map(&rect_in_pic_space)
            .expect("bug");

        // Include this in the overall dirty rect
        self.combined = self.combined.union(&raster_rect);
    }
}

/// Debugging information about why a tile was invalidated
#[derive(Debug,Clone)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum InvalidationReason {
    /// The background color changed
    BackgroundColor,
    /// The opaque state of the backing native surface changed
    SurfaceOpacityChanged,
    /// There was no backing texture (evicted or never rendered)
    NoTexture,
    /// There was no backing native surface (never rendered, or recreated)
    NoSurface,
    /// The primitive count in the dependency list was different
    PrimCount,
    /// The content of one of the primitives was different
    Content,
    // The compositor type changed
    CompositorKindChanged,
    // The valid region of the tile changed
    ValidRectChanged,
    // The overall scale of the picture cache changed
    ScaleChanged,
    // The content of the sampling surface changed
    SurfaceContentChanged,
    // Cancel underlay
    CancelUnderlay,
}

/// The result of a primitive dependency comparison. Size is a u8
/// since this is a hot path in the code, and keeping the data small
/// is a performance win.
#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[repr(u8)]
pub enum PrimitiveCompareResult {
    /// Primitives match
    Equal,
    /// Something in the PrimitiveDescriptor was different
    Descriptor,
    /// A clip corner changed (vert range values differ)
    Clip,
    /// An image dependency was dirty
    Image,
    /// The value of an opacity binding changed
    OpacityBinding,
    /// The value of a color binding changed
    ColorBinding,
}
