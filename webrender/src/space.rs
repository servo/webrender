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
/// Maps a target node's content into the snap node's (device) space so a rect
/// can be snapped to the device grid there.
///
/// A rotation or reflection by a multiple of 90 degrees (the only
/// cross-coordinate-system case we snap across) is fully described by a
/// `ScaleOffset` plus an optional x/y axis swap: the 0/180-degree case is a
/// `ScaleOffset` directly; the 90/270-degree case is the same after swapping x
/// and y. `swap_xy` is always false for a target in the snap node's own
/// coordinate system.
#[derive(Clone, Debug)]
struct SnapTransform {
    scale_offset: ScaleOffset,
    swap_xy: bool,
}

#[derive(Clone, Debug)]
pub struct SpaceSnapper {
    /// If false, `snap_rect` passes rects through unchanged.
    enabled: bool,
    /// Node content is snapped against (the root, or the surface's raster node).
    snap_node_index: SpatialNodeIndex,
    /// Inverse of the snap node's `content_transform`, computed once.
    raster_content_inverse: ScaleOffset,
    /// Coordinate system of the snap node. A target in the same coordinate
    /// system snaps with a cheap scale + offset; a target in a different one is
    /// only snappable when the reference frame between them is grid-preserving.
    raster_coord_system_id: CoordinateSystemId,
    /// Last target passed to `set_target_spatial_node`, for the cache below.
    current_target_spatial_node_index: SpatialNodeIndex,
    /// Cached snapping transform for `current_target_spatial_node_index`. `None`
    /// when the target cannot be snapped (a non-axis-aligned reference frame
    /// between it and the snap node).
    snapping_transform: Option<SnapTransform>,
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

        let (enabled, snap_node_index) = if raster_in_root {
            (true, spatial_tree.root_reference_frame_index())
        } else if surface.allow_snapping {
            (true, raster_spatial_node_index)
        } else {
            (false, raster_spatial_node_index)
        };

        let snap_node = spatial_tree.get_spatial_node(snap_node_index);

        SpaceSnapper {
            enabled,
            snap_node_index,
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
        self.snapping_transform = if target_node.coordinate_system_id == self.raster_coord_system_id {
            // Same coordinate system: a cheap scale + offset.
            // target-local -> coordinate-system root -> snap-local (or root).
            Some(SnapTransform {
                scale_offset: target_node.content_transform.then(&self.raster_content_inverse),
                swap_xy: false,
            })
        } else {
            // A reference frame between the target and snap node crosses a
            // coordinate system. We can still snap across it if it is a rotation
            // or reflection by a multiple of 90 degrees (with no scaling), because
            // that keeps content on the same pixel grid: snapping in the target's
            // space is identical to snapping in the snap node's, and the content
            // is still rasterized at the same device scale.
            //
            // Anything else can't be grid-snapped: a frame that rescales (e.g. a
            // `rotate-x(45)` that flattens to a y-scale) rasterizes its content in
            // its own local raster space, not the device grid, so snapping against
            // the snap node would shift it; and skew / arbitrary rotation /
            // perspective don't keep the axes aligned at all.
            //
            // TODO: this cross-coordinate-system handling is only needed because
            // a 90/180/270-degree rotation currently establishes a new coordinate
            // system. Such a rotation keeps content on the pixel grid, so it
            // should stay in the parent's coordinate system (as a plain
            // scale/translation frame does); once it does, the target and snap
            // node share a coordinate system and this branch can go away.
            let fwd = spatial_tree
                .get_relative_transform(target_node_index, self.snap_node_index)
                .into_transform();
            fwd.as_grid_aligned_rotation()
                .map(|(scale_offset, swap_xy)| SnapTransform { scale_offset, swap_xy })
        };
    }

    /// Snap a rect to the device pixel grid using the current target's snapping
    /// transform: map the rect into device space, snap it to the integer pixel
    /// grid, then map it back. A target that can't be snapped (or a disabled
    /// snapper) leaves the rect unchanged.
    pub fn snap_rect<F>(&self, rect: &Box2D<f32, F>) -> Box2D<f32, F> where F: fmt::Debug {
        debug_assert!(!self.enabled || self.current_target_spatial_node_index != SpatialNodeIndex::INVALID);
        match self.snapping_transform {
            Some(SnapTransform { ref scale_offset, swap_xy }) => {
                let rect = if swap_xy { swap_box_xy(rect) } else { *rect };
                let snapped_device_rect: DeviceRect = scale_offset.map_rect(&rect).snap();
                let unmapped: Box2D<f32, F> = scale_offset.unmap_rect(&snapped_device_rect);
                if swap_xy { swap_box_xy(&unmapped) } else { unmapped }
            }
            None => *rect,
        }
    }
}

/// Swap the x and y coordinates of a rect, mapping it through the `(x, y) ->
/// (y, x)` reflection. Used to fold a 90/270-degree axis swap into a
/// `ScaleOffset` snap; it is its own inverse.
fn swap_box_xy<F>(r: &Box2D<f32, F>) -> Box2D<f32, F> {
    Box2D::new(
        Point2D::new(r.min.y, r.min.x),
        Point2D::new(r.max.y, r.max.x),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use api::{PipelineId, PropertyBinding, ReferenceFrameKind, StickyOffsetBounds, TransformStyle};
    use api::units::{
        DevicePixelScale, LayoutPoint, LayoutRect, LayoutSize, LayoutTransform, LayoutVector2D,
        WorldPoint, WorldRect, WorldSize,
    };
    use crate::scene::SceneProperties;
    use crate::spatial_node::StickyFrameInfo;
    use crate::spatial_tree::{SceneSpatialTree, SpatialTree};
    use crate::surface::SurfaceInfo;

    #[test]
    fn test_as_grid_aligned_rotation() {
        let deg = |d: f32| euclid::Angle::degrees(d);

        // Snappable: 90/180/270-degree rotations, reflections, identity. The
        // 90/270-degree cases swap x and y; the others do not.
        for (d, expect_swap) in [(0.0, false), (90.0, true), (180.0, false), (270.0, true), (-90.0, true)] {
            let rot = LayoutTransform::rotation(0.0, 0.0, 1.0, deg(d)).as_grid_aligned_rotation();
            assert_eq!(
                rot.map(|(_, swap)| swap),
                Some(expect_swap),
                "rotate-z({d}) should be a grid-aligned rotation with swap_xy={expect_swap}",
            );
        }
        assert!(LayoutTransform::identity().as_grid_aligned_rotation().is_some());
        assert!(LayoutTransform::scale(-1.0, 1.0, 1.0).as_grid_aligned_rotation().is_some());
        assert!(LayoutTransform::rotation(0.0, 0.0, 1.0, deg(90.0))
            .then_translate(euclid::vec3(12.0, -7.0, 0.0))
            .as_grid_aligned_rotation()
            .is_some());

        // Not snappable: a 45-degree z-rotation doesn't keep the axes aligned.
        assert!(LayoutTransform::rotation(0.0, 0.0, 1.0, deg(45.0)).as_grid_aligned_rotation().is_none());
        // A non-unit scale rescales the grid (e.g. a flattened rotate-x).
        assert!(LayoutTransform::scale(1.0, 0.707, 1.0).as_grid_aligned_rotation().is_none());
        // Perspective has identity 2x2 and unit scale, but m34 != 0: must be
        // rejected (the bug that broke the css-transforms/perspective WPTs).
        let mut perspective = LayoutTransform::identity();
        perspective.m34 = -1.0 / 500.0;
        assert!(perspective.as_grid_aligned_rotation().is_none());
        // z-coupling (e.g. rotate-x leaving a residual) must be rejected.
        assert!(LayoutTransform::rotation(1.0, 0.0, 0.0, deg(30.0)).as_grid_aligned_rotation().is_none());
    }

    // Bug 2004666: a snapping surface (e.g. a sticky / scrolled tile cache) whose
    // raster node is in the root coordinate system but offset from root by a
    // fractional, un-snapped amount must snap its content against the root, not
    // against its own node — snapping against its own node is a no-op and leaves
    // content a sub-pixel off the device grid (the cause of the sticky-content
    // jitter). A `should_snap:false` 2d-scale-translation reference frame at a
    // fractional offset reproduces that fractional cs-origin.
    fn assert_snaps_against_root(st: &SpatialTree, raster_node: SpatialNodeIndex) {
        // Precondition: the raster node is in the root coordinate system but
        // offset from root by a fractional amount.
        let node = st.get_spatial_node(raster_node);
        assert_eq!(node.coordinate_system_id, CoordinateSystemId::root());
        assert!(
            (node.content_transform.offset.x - 0.4).abs() < 0.0001,
            "expected fractional cs-origin, got {:?}",
            node.content_transform.offset,
        );

        // A snapping surface rasterized into that node (allow_snapping = true,
        // like a tile cache).
        let surface = SurfaceInfo::new(
            raster_node,
            raster_node,
            WorldRect::from_origin_and_size(WorldPoint::zero(), WorldSize::new(1000.0, 1000.0)),
            st,
            DevicePixelScale::new(1.0),
            (1.0, 1.0),
            (1.0, 1.0),
            true,
            false,
        );

        let mut snapper = SpaceSnapper::new(&surface, st);
        snapper.set_target_spatial_node(raster_node, st);

        // Snapping against root maps the rect to a fractional device rect (offset
        // 0.4), snaps it to the integer grid, and maps it back offset by -0.4.
        // Snapping against the surface's own node would be a no-op (rect stays at
        // its integer local origin), which is the bug.
        let rect = LayoutRect::from_origin_and_size(
            LayoutPoint::new(20.0, 40.0),
            LayoutSize::new(60.0, 20.0),
        );
        let snapped = snapper.snap_rect(&rect);

        assert!(
            (snapped.min.x - 19.6).abs() < 0.01 && (snapped.min.y - 39.6).abs() < 0.01,
            "expected content snapped against root (min ~= 19.6,39.6), got {:?}",
            snapped.min,
        );
    }

    fn add_fractional_ref_frame(
        cst: &mut SceneSpatialTree,
        parent: SpatialNodeIndex,
    ) -> SpatialNodeIndex {
        // A 2d-scale-translation reference frame stays in the root coordinate
        // system, and with should_snap:false its fractional offset is not
        // device-snapped — exactly the situation a sticky tile cache hits under
        // layout.disable-pixel-alignment.
        cst.add_reference_frame(
            parent,
            TransformStyle::Flat,
            PropertyBinding::Value(LayoutTransform::translation(0.4, 0.4, 0.0)),
            ReferenceFrameKind::Transform {
                is_2d_scale_translation: true,
                should_snap: false,
                paired_with_perspective: false,
            },
            LayoutVector2D::zero(),
            PipelineId::dummy(),
            false,
        )
    }

    #[test]
    fn test_root_cs_surface_snaps_against_root() {
        let mut cst = SceneSpatialTree::new();
        let root = cst.root_reference_frame_index();
        let frac = add_fractional_ref_frame(&mut cst, root);

        let mut st = SpatialTree::new();
        st.apply_updates(cst.end_frame_and_get_pending_updates());
        st.update_tree(&SceneProperties::new());

        assert_snaps_against_root(&st, frac);
    }

    #[test]
    fn test_sticky_cache_snaps_against_root() {
        // The same fractional cs-origin reached through a sticky frame (which
        // gets its own tile cache): content in the sticky cache must still snap
        // against root, not the sticky node.
        let mut cst = SceneSpatialTree::new();
        let root = cst.root_reference_frame_index();
        let frac = add_fractional_ref_frame(&mut cst, root);

        let sticky = cst.add_sticky_frame(
            frac,
            StickyFrameInfo {
                frame_rect: LayoutRect::from_size(LayoutSize::new(400.0, 100.0)),
                margins: euclid::SideOffsets2D::new(None, None, None, None),
                vertical_offset_bounds: StickyOffsetBounds::new(0.0, 0.0),
                horizontal_offset_bounds: StickyOffsetBounds::new(0.0, 0.0),
                current_offset: LayoutVector2D::zero(),
                transform: None,
            },
            PipelineId::dummy(),
        );

        let mut st = SpatialTree::new();
        st.apply_updates(cst.end_frame_and_get_pending_updates());
        st.update_tree(&SceneProperties::new());

        assert_snaps_against_root(&st, sticky);
    }

    #[test]
    fn test_grid_preserving_rotation_snaps_across_coord_system() {
        // A 90-degree rotation creates a *new* coordinate system, but it is
        // grid-preserving (maps axis-aligned rects to axis-aligned rects on the
        // device grid), so content under it must still snap to the device grid
        // against root. Earlier the cross-coordinate-system case bailed out of
        // snapping entirely, leaving line decorations under writing-mode /
        // rotation a sub-pixel off (bug 2004666 reftest regression).
        let mut cst = SceneSpatialTree::new();
        let root = cst.root_reference_frame_index();

        let rot = cst.add_reference_frame(
            root,
            TransformStyle::Flat,
            PropertyBinding::Value(LayoutTransform::rotation(
                0.0,
                0.0,
                1.0,
                euclid::Angle::degrees(-90.0),
            )),
            ReferenceFrameKind::Transform {
                is_2d_scale_translation: false,
                should_snap: false,
                paired_with_perspective: false,
            },
            LayoutVector2D::zero(),
            PipelineId::dummy(),
            false,
        );

        let mut st = SpatialTree::new();
        st.apply_updates(cst.end_frame_and_get_pending_updates());
        st.update_tree(&SceneProperties::new());

        // Precondition: the rotation really is in a different coordinate system.
        assert_ne!(
            st.get_spatial_node(rot).coordinate_system_id,
            CoordinateSystemId::root(),
            "expected rotation to establish a new coordinate system",
        );

        // Surface rasterized into root (allow_snapping = true); content's spatial
        // node is the rotation frame.
        let surface = SurfaceInfo::new(
            root,
            root,
            WorldRect::from_origin_and_size(WorldPoint::zero(), WorldSize::new(1000.0, 1000.0)),
            &st,
            DevicePixelScale::new(1.0),
            (1.0, 1.0),
            (1.0, 1.0),
            true,
            false,
        );

        let mut snapper = SpaceSnapper::new(&surface, &st);
        snapper.set_target_spatial_node(rot, &st);

        // A fractional rect in the rotated frame must land on the integer grid
        // once mapped to device space - i.e. it really got snapped, not passed
        // through unchanged.
        let rect = LayoutRect::from_origin_and_size(
            LayoutPoint::new(10.3, 20.7),
            LayoutSize::new(40.4, 2.6),
        );
        let snapped = snapper.snap_rect(&rect);

        let to_root = st.get_relative_transform(rot, root).into_transform();
        for corner in [snapped.min, snapped.max] {
            let device = to_root.transform_point2d(corner).unwrap();
            assert!(
                (device.x - device.x.round()).abs() < 0.01
                    && (device.y - device.y.round()).abs() < 0.01,
                "snapped corner {:?} -> device {:?} not on the integer grid",
                corner,
                device,
            );
        }
    }
}


