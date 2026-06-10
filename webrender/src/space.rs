/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


//! Utilities to deal with coordinate spaces.

use std::fmt;

use euclid::{Transform3D, Box2D, Point2D, Vector2D};

use api::units::DeviceRect;
use crate::spatial_tree::{CoordinateSystemId, SpatialTree, CoordinateSpaceMapping, SpatialNodeIndex, VisibleFace};
use crate::surface::SurfaceInfo;
use crate::util::project_rect;
use crate::util::{MatrixHelpers, RectHelpers, ScaleOffset};


#[derive(Debug, Clone)]
pub struct SpaceMapper<F, T> {
    kind: CoordinateSpaceMapping<F, T>,
    pub ref_spatial_node_index: SpatialNodeIndex,
    pub current_target_spatial_node_index: SpatialNodeIndex,
    pub bounds: Box2D<f32, T>,
    visible_face: VisibleFace,
}

impl<F, T> SpaceMapper<F, T> where F: fmt::Debug {
    pub fn new(
        ref_spatial_node_index: SpatialNodeIndex,
        bounds: Box2D<f32, T>,
    ) -> Self {
        SpaceMapper {
            kind: CoordinateSpaceMapping::Local,
            ref_spatial_node_index,
            current_target_spatial_node_index: ref_spatial_node_index,
            bounds,
            visible_face: VisibleFace::Front,
        }
    }

    pub fn new_with_target(
        ref_spatial_node_index: SpatialNodeIndex,
        target_node_index: SpatialNodeIndex,
        bounds: Box2D<f32, T>,
        spatial_tree: &SpatialTree,
    ) -> Self {
        let mut mapper = Self::new(ref_spatial_node_index, bounds);
        mapper.set_target_spatial_node(target_node_index, spatial_tree);
        mapper
    }

    pub fn set_target_spatial_node(
        &mut self,
        target_node_index: SpatialNodeIndex,
        spatial_tree: &SpatialTree,
    ) {
        if target_node_index == self.current_target_spatial_node_index {
            return
        }

        let ref_spatial_node = spatial_tree.get_spatial_node(self.ref_spatial_node_index);
        let target_spatial_node = spatial_tree.get_spatial_node(target_node_index);
        self.visible_face = VisibleFace::Front;

        self.kind = if self.ref_spatial_node_index == target_node_index {
            CoordinateSpaceMapping::Local
        } else if ref_spatial_node.coordinate_system_id == target_spatial_node.coordinate_system_id {
            let scale_offset = target_spatial_node.content_transform
                .then(&ref_spatial_node.content_transform.inverse());
            CoordinateSpaceMapping::ScaleOffset(scale_offset)
        } else {
            let transform = spatial_tree
                .get_relative_transform_with_face(
                    target_node_index,
                    self.ref_spatial_node_index,
                    Some(&mut self.visible_face),
                )
                .into_transform()
                .with_source::<F>()
                .with_destination::<T>();
            CoordinateSpaceMapping::Transform(transform)
        };

        self.current_target_spatial_node_index = target_node_index;
    }

    pub fn get_transform(&self) -> Transform3D<f32, F, T> {
        match self.kind {
            CoordinateSpaceMapping::Local => {
                Transform3D::identity()
            }
            CoordinateSpaceMapping::ScaleOffset(ref scale_offset) => {
                scale_offset.to_transform()
            }
            CoordinateSpaceMapping::Transform(transform) => {
                transform
            }
        }
    }

    pub fn unmap(&self, rect: &Box2D<f32, T>) -> Option<Box2D<f32, F>> {
        match self.kind {
            CoordinateSpaceMapping::Local => {
                Some(rect.cast_unit())
            }
            CoordinateSpaceMapping::ScaleOffset(ref scale_offset) => {
                Some(scale_offset.unmap_rect(rect))
            }
            CoordinateSpaceMapping::Transform(ref transform) => {
                transform.inverse_rect_footprint(rect)
            }
        }
    }

    pub fn map(&self, rect: &Box2D<f32, F>) -> Option<Box2D<f32, T>> {
        match self.kind {
            CoordinateSpaceMapping::Local => {
                Some(rect.cast_unit())
            }
            CoordinateSpaceMapping::ScaleOffset(ref scale_offset) => {
                Some(scale_offset.map_rect(rect))
            }
            CoordinateSpaceMapping::Transform(ref transform) => {
                match project_rect(transform, rect, &self.bounds) {
                    Some(bounds) => {
                        Some(bounds)
                    }
                    None => {
                        warn!("parent relative transform can't transform the primitive rect for {:?}", rect);
                        None
                    }
                }
            }
        }
    }

    // Attempt to return a rect that is contained in the mapped rect.
    pub fn map_inner_bounds(&self, rect: &Box2D<f32, F>) -> Option<Box2D<f32, T>> {
        match self.kind {
            CoordinateSpaceMapping::Local => {
                Some(rect.cast_unit())
            }
            CoordinateSpaceMapping::ScaleOffset(ref scale_offset) => {
                Some(scale_offset.map_rect(rect))
            }
            CoordinateSpaceMapping::Transform(..) => {
                // We could figure out a rect that is contained in the transformed rect but
                // for now we do the simple thing here and bail out.
                return None;
            }
        }
    }

    // Map a local space point to the target coordinate space
    pub fn map_point(&self, p: Point2D<f32, F>) -> Option<Point2D<f32, T>> {
        match self.kind {
            CoordinateSpaceMapping::Local => {
                Some(p.cast_unit())
            }
            CoordinateSpaceMapping::ScaleOffset(ref scale_offset) => {
                Some(scale_offset.map_point(&p))
            }
            CoordinateSpaceMapping::Transform(ref transform) => {
                transform.transform_point2d(p)
            }
        }
    }

    pub fn map_vector(&self, v: Vector2D<f32, F>) -> Vector2D<f32, T> {
        match self.kind {
            CoordinateSpaceMapping::Local => {
                v.cast_unit()
            }
            CoordinateSpaceMapping::ScaleOffset(ref scale_offset) => {
                scale_offset.map_vector(&v)
            }
            CoordinateSpaceMapping::Transform(ref transform) => {
                transform.transform_vector2d(v)
            }
        }
    }

    pub fn as_2d_scale_offset(&self) -> Option<ScaleOffset> {
        self.kind.as_2d_scale_offset()
    }
}


/// Snaps rects to the device pixel grid at frame time, in the space they are
/// actually rasterized in. A snapper is bound to a single raster node (the
/// surface the content is rasterized into) at construction and then reused for
/// many targets via `set_target_spatial_node`, which caches the snapping
/// transform for the last target so re-snapping prims/clips that share a
/// spatial node is cheap.
///
/// The snapping transform is derived from each node's resolved
/// `content_transform` (the node-local -> coordinate-system transform the
/// spatial tree already computed, with device-pixel snapping of reference-frame
/// / scroll offsets baked in), so it is consistent with how content is actually
/// placed. Re-deriving it from raw origins / source transforms would snap rects
/// against a different offset than the node transform renders them at, landing
/// content a sub-pixel off (see bug 1580534).
///
/// Snapping into a surface's raster node (rather than always the root) snaps
/// content in the space it is rasterized in — for a tile cache that excludes
/// the scroll above the raster node, matching the cache's own (scroll-stable)
/// content transform.
///
/// A snapper built for a surface that doesn't snap (`allow_snapping == false`)
/// is disabled and passes every rect through unchanged. Such a surface is a
/// non-snapping raster root (preserve-3d / perspective / huge-scale), where
/// snapping against the surface's own scaled node would use only the tiny local
/// scale and collapse content to zero.
#[derive(Clone, Debug)]
pub struct SpaceSnapper {
    /// If false, `snap_rect` passes rects through unchanged.
    enabled: bool,
    /// Inverse of the raster node's `content_transform`, computed once.
    raster_content_inverse: ScaleOffset,
    /// Coordinate system of the raster node. A target in a different coordinate
    /// system has a non-axis-aligned reference frame between it and the raster
    /// node and so cannot be snapped.
    raster_coord_system_id: CoordinateSystemId,
    /// Last target passed to `set_target_spatial_node`, for the cache below.
    current_target_spatial_node_index: SpatialNodeIndex,
    /// Cached snapping transform for `current_target_spatial_node_index`. `None`
    /// when the target cannot be snapped (different coordinate system).
    snapping_transform: Option<ScaleOffset>,
}

impl SpaceSnapper {
    /// Create a snapper that snaps into `surface`'s raster space (the space the
    /// surface's content is rasterized in).
    ///
    /// When the surface snaps (`allow_snapping == true`) content is snapped
    /// against the surface's own raster node.
    ///
    /// A non-snapping raster root (`allow_snapping == false`) whose raster node
    /// is still in the root coordinate system is a resolve target (backdrop
    /// filter): the `DISABLE_SNAPPING` flag keeps it from establishing a
    /// root-snapping raster root, but its content must still be snapped — so we
    /// snap it against the root, mirroring the global snap pass this replaced.
    /// A genuine non-snapping raster root (preserve-3d / perspective, raster
    /// node not in the root coordinate system) stays disabled, since snapping
    /// against its own scaled node would collapse content to zero.
    pub fn new(
        surface: &SurfaceInfo,
        spatial_tree: &SpatialTree,
    ) -> Self {
        let raster_spatial_node_index = surface.raster_spatial_node_index;
        debug_assert!(raster_spatial_node_index != SpatialNodeIndex::INVALID);
        let raster_node = spatial_tree.get_spatial_node(raster_spatial_node_index);
        let raster_in_root = raster_node.coordinate_system_id == CoordinateSystemId::root();

        let (enabled, snap_node_index) = if surface.allow_snapping {
            (true, raster_spatial_node_index)
        } else if raster_in_root {
            (true, spatial_tree.root_reference_frame_index())
        } else {
            (false, raster_spatial_node_index)
        };

        let snap_node = spatial_tree.get_spatial_node(snap_node_index);

        SpaceSnapper {
            enabled,
            raster_content_inverse: snap_node.content_transform.inverse(),
            raster_coord_system_id: snap_node.coordinate_system_id,
            current_target_spatial_node_index: SpatialNodeIndex::INVALID,
            snapping_transform: None,
        }
    }

    /// Set the spatial node whose content subsequent `snap_rect` calls snap.
    /// Cheap to re-set with the same target: the snapping transform is cached.
    pub fn set_target_spatial_node(
        &mut self,
        target_node_index: SpatialNodeIndex,
        spatial_tree: &SpatialTree,
    ) {
        if !self.enabled || target_node_index == self.current_target_spatial_node_index {
            return;
        }

        let target_node = spatial_tree.get_spatial_node(target_node_index);

        self.current_target_spatial_node_index = target_node_index;
        // Can only snap if the target and raster node share a coordinate system
        // (i.e. no non-axis-aligned reference frame between them).
        self.snapping_transform = if target_node.coordinate_system_id != self.raster_coord_system_id {
            None
        } else {
            // target-local -> coordinate-system root -> raster-local (or root).
            Some(target_node.content_transform.then(&self.raster_content_inverse))
        };
    }

    /// Snap a rect to the device pixel grid using the current target's snapping
    /// transform: map the rect into device space, snap it to the integer pixel
    /// grid, then map it back. A target that can't be snapped (or a disabled
    /// snapper) leaves the rect unchanged.
    pub fn snap_rect<F>(&self, rect: &Box2D<f32, F>) -> Box2D<f32, F> where F: fmt::Debug {
        debug_assert!(!self.enabled || self.current_target_spatial_node_index != SpatialNodeIndex::INVALID);
        match self.snapping_transform {
            Some(ref scale_offset) => {
                let snapped_device_rect: DeviceRect = scale_offset.map_rect(rect).snap();
                scale_offset.unmap_rect(&snapped_device_rect)
            }
            None => *rect,
        }
    }
}


