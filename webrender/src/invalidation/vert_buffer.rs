/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Quantized raster-space vertex buffer for output-space tile invalidation.
//!
//! Each primitive and clip gets its transformed, raster-space corners stored
//! here as quantized i32 values. The tile descriptor stores a VertRange
//! referencing into this buffer instead of a picture-space prim_clip_box or
//! spatial node dependency.

use api::units::*;
use crate::spatial_tree::{SpatialTree, SpatialNodeIndex, CoordinateSpaceMapping};
use crate::util::{MatrixHelpers, ScaleOffset};

/// Sub-pixel quantization scale: quarter-pixel precision.
pub const VERT_QUANTIZE_SCALE: f32 = 4.0;

pub fn quantize(v: f32) -> i32 {
    (v * VERT_QUANTIZE_SCALE).round() as i32
}

/// A reference into a per-tile vert_data buffer: offset (in i32 elements) and count.
/// count is 4 for an axis-aligned rect (2 corners × 2 coords), 8 for a
/// non-axis-aligned quad (4 corners × 2 coords), or 16 for a transform fingerprint
/// emitted when a perspective-projected rect crosses the camera plane (4 corners ×
/// homogeneous (x, y, z, w) coords).
#[derive(Copy, Clone, Debug, Default, PartialEq, peek_poke::PeekPoke)]
#[cfg_attr(feature = "capture", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct VertRange {
    pub offset: u32,
    pub count: u32,
}

impl VertRange {
    pub const INVALID: VertRange = VertRange { offset: 0, count: 0 };

    pub fn is_valid(self) -> bool {
        self.count > 0
    }
}

/// Persistent per-tile-cache scratch and transform cache for computing
/// raster-space corners.
///
/// Lives on TileCacheInstance and provides two optimisations:
///
/// 1. **Amortised unquantized scratch**: `unquantized` is never dropped between
///    frames, so the heap allocation is paid once after warmup.
///
/// 2. **Spatial-node transform cache**: the relative transform from
///    `prim_spatial_node` → `tile_cache_spatial_node` is cached so that
///    consecutive primitives in the same scroll frame avoid repeated
///    `get_relative_transform` calls.
pub struct CornersCache {
    /// Amortised scratch for unquantized corners.
    /// Cleared once before computing prim + coverage + clips for each primitive.
    unquantized: Vec<RasterPoint>,

    /// The primitive spatial node for which `cached_mapping` was computed.
    /// `None` means the cache is cold (reset at frame start).
    cached_node: Option<SpatialNodeIndex>,

    /// Cached mapping for `cached_node`. Valid only when
    /// `cached_node == Some(current prim_spatial_node)`.
    cached_mapping: CoordinateSpaceMapping<LayoutPixel, LayoutPixel>,
}

impl CornersCache {
    pub fn new() -> Self {
        CornersCache {
            unquantized: Vec::new(),
            cached_node: None,
            cached_mapping: CoordinateSpaceMapping::Local,
        }
    }

    /// Reset the transform cache. Call once at the start of each frame's
    /// dependency update, before any primitives are processed.
    pub fn pre_update(&mut self) {
        self.cached_node = None;
    }

    /// Clear the unquantized scratch. Call once before computing corners for a
    /// single primitive (before prim rect, coverage rect and all clips).
    pub fn clear_scratch(&mut self) {
        self.unquantized.clear();
    }

    /// Compute unquantized raster-space corners for `local_rect` and append
    /// them to the scratch buffer. Returns a VertRange into the scratch, or
    /// VertRange::INVALID if the transform is non-invertible.
    ///
    /// The relative transform for `prim_spatial_node` is cached across calls:
    /// if the same node is passed as the previous call, `get_relative_transform`
    /// is not recomputed.
    pub fn compute_to_scratch(
        &mut self,
        local_rect: LayoutRect,
        prim_spatial_node: SpatialNodeIndex,
        tile_cache_spatial_node: SpatialNodeIndex,
        local_to_raster: ScaleOffset,
        spatial_tree: &SpatialTree,
    ) -> VertRange {
        if Some(prim_spatial_node) != self.cached_node {
            let mapping = spatial_tree.get_relative_transform(
                prim_spatial_node,
                tile_cache_spatial_node,
            );
            self.cached_mapping = match mapping {
                CoordinateSpaceMapping::ScaleOffset(ref so) if so.is_reflection() => {
                    CoordinateSpaceMapping::Transform(so.to_transform())
                }
                other => other,
            };
            self.cached_node = Some(prim_spatial_node);
        }
        self.append_corners_from_mapping(local_rect, local_to_raster)
    }

    fn append_corners_from_mapping(
        &mut self,
        local_rect: LayoutRect,
        local_to_raster: ScaleOffset,
    ) -> VertRange {
        match &self.cached_mapping {
            CoordinateSpaceMapping::Local => {
                let r: RasterRect = local_to_raster.map_rect(&local_rect);
                let offset = self.unquantized.len() as u32;
                self.unquantized.push(r.min);
                self.unquantized.push(r.max);
                VertRange { offset, count: 2 }
            }
            CoordinateSpaceMapping::ScaleOffset(so) => {
                let r: RasterRect = so.then(&local_to_raster).map_rect(&local_rect);
                let offset = self.unquantized.len() as u32;
                self.unquantized.push(r.min);
                self.unquantized.push(r.max);
                VertRange { offset, count: 2 }
            }
            CoordinateSpaceMapping::Transform(m) => {
                let raster_m = m.then(&local_to_raster.to_transform::<LayoutPixel, RasterPixel>());
                let src = [
                    local_rect.min,
                    LayoutPoint::new(local_rect.max.x, local_rect.min.y),
                    LayoutPoint::new(local_rect.min.x, local_rect.max.y),
                    local_rect.max,
                ];
                let offset = self.unquantized.len() as u32;

                // Fast path: no perspective component. transform_point2d can never
                // fail for one corner while succeeding for another, so we don't need
                // homogeneous coords or a fingerprint fallback.
                if !raster_m.has_perspective_component() {
                    for p in &src {
                        match raster_m.transform_point2d(*p) {
                            Some(pt) => self.unquantized.push(pt),
                            None => {
                                self.unquantized.truncate(offset as usize);
                                return VertRange::INVALID;
                            }
                        }
                    }
                    return VertRange { offset, count: 4 };
                }

                // Perspective transform: compute homogeneous coords so we can
                // distinguish "all corners in front of camera" (project them) from
                // "rect crosses the camera plane" (push a stable fingerprint).
                let homogens = [
                    raster_m.transform_point2d_homogeneous(src[0]),
                    raster_m.transform_point2d_homogeneous(src[1]),
                    raster_m.transform_point2d_homogeneous(src[2]),
                    raster_m.transform_point2d_homogeneous(src[3]),
                ];
                if homogens.iter().all(|h| h.w > 0.0) {
                    for h in &homogens {
                        self.unquantized.push(RasterPoint::new(h.x / h.w, h.y / h.w));
                    }
                    VertRange { offset, count: 4 }
                } else {
                    // At least one corner is at or behind the camera plane and can't be
                    // projected to a finite 2D raster point. Falling back to INVALID would
                    // make compare_prim see equal empty slices on every frame, silently
                    // hiding transform animations (bug 2036730). Instead, push the
                    // homogeneous (x, y, z, w) of each corner as a stable per-transform
                    // fingerprint: equal across frames when the transform is unchanged
                    // (no over-invalidation), but different when the transform animates
                    // (correct invalidation). Two RasterPoints encode each corner.
                    for h in &homogens {
                        self.unquantized.push(RasterPoint::new(h.x, h.y));
                        self.unquantized.push(RasterPoint::new(h.z, h.w));
                    }
                    VertRange { offset, count: 8 }
                }
            }
        }
    }

    /// Quantize corners at `scratch_range` from the scratch buffer into `dst`.
    /// Returns a VertRange into `dst`, or INVALID if `scratch_range` is invalid.
    pub fn push_verts(&self, scratch_range: VertRange, dst: &mut Vec<i32>) -> VertRange {
        if !scratch_range.is_valid() {
            return VertRange::INVALID;
        }
        let start = scratch_range.offset as usize;
        let end = (scratch_range.offset + scratch_range.count) as usize;
        let corners = &self.unquantized[start..end];
        debug_assert!(corners.len() == 2 || corners.len() == 4 || corners.len() == 8);
        let offset = dst.len() as u32;
        for p in corners {
            dst.push(quantize(p.x));
            dst.push(quantize(p.y));
        }
        VertRange { offset, count: (corners.len() * 2) as u32 }
    }

    /// Quantize corners at `scratch_range` into `dst`, clamping to `tile_rect`.
    /// Returns a VertRange into `dst`, or INVALID if `scratch_range` is invalid.
    pub fn push_verts_clamped(
        &self,
        scratch_range: VertRange,
        tile_rect: &RasterRect,
        dst: &mut Vec<i32>,
    ) -> VertRange {
        if !scratch_range.is_valid() {
            return VertRange::INVALID;
        }
        let start = scratch_range.offset as usize;
        let end = (scratch_range.offset + scratch_range.count) as usize;
        let corners = &self.unquantized[start..end];
        debug_assert!(corners.len() == 2 || corners.len() == 4 || corners.len() == 8);
        let offset = dst.len() as u32;
        if corners.len() == 8 {
            // Transform fingerprint (homogeneous coords for a perspective-crossing rect).
            // Clamping these to tile bounds would corrupt the fingerprint, so skip the clamp.
            for p in corners {
                dst.push(quantize(p.x));
                dst.push(quantize(p.y));
            }
        } else {
            for p in corners {
                dst.push(quantize(p.x.max(tile_rect.min.x).min(tile_rect.max.x)));
                dst.push(quantize(p.y.max(tile_rect.min.y).min(tile_rect.max.y)));
            }
        }
        VertRange { offset, count: (corners.len() * 2) as u32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use api::units::{LayoutPixel, LayoutPoint, LayoutRect, LayoutTransform};
    use euclid::Angle;

    /// Build a perspective(d) * rotateX(deg) * translate(0, ty, 0) row-vector matrix.
    /// Mirrors the CSS-style transform in bug 2036730's repro: a tall rect rotated
    /// around its top edge so the bottom corners fall past the perspective camera
    /// plane.
    fn perspective_rotate_x_translate_y(deg: f32, d: f32, ty: f32) -> LayoutTransform {
        let translate = LayoutTransform::translation(0.0, ty, 0.0);
        let rotate = LayoutTransform::rotation(1.0, 0.0, 0.0, Angle::degrees(deg));
        let mut perspective = LayoutTransform::identity();
        perspective.m34 = -1.0 / d;
        translate.then(&rotate).then(&perspective)
    }

    /// Bug 2036730 regression test. When a perspective-projected rect has corners
    /// with w <= 0 (the rect crosses the camera plane), compute_to_scratch must
    /// emit a stable per-transform fingerprint instead of returning INVALID. Two
    /// frames of an animated transform must therefore produce different scratch
    /// contents so downstream tile invalidation detects the change. Pre-fix,
    /// both frames returned VertRange::INVALID and compare_prim saw equal empty
    /// slices — silently bypassing invalidation.
    #[test]
    fn perspective_camera_plane_fingerprint_differs_per_transform() {
        // 200 x 2000 rect rotated 80deg around the top edge. With perspective
        // distance 1000, the bottom corners reach z ≈ 2000*sin(80°) ≈ 1969,
        // which is past the camera and gives w ≈ -0.97.
        let local_rect = LayoutRect::new(
            LayoutPoint::new(0.0, 0.0),
            LayoutPoint::new(200.0, 2000.0),
        );
        let local_to_raster = ScaleOffset::identity();

        let mut cache = CornersCache::new();

        cache.cached_mapping = CoordinateSpaceMapping::Transform(
            perspective_rotate_x_translate_y(80.0, 1000.0, 0.0),
        );
        cache.clear_scratch();
        let r1 = cache.append_corners_from_mapping(local_rect, local_to_raster);
        assert!(r1.is_valid(), "fingerprint must not collapse to INVALID");
        assert_eq!(r1.count, 8, "fingerprint encodes 4 corners as 8 RasterPoints");
        let scratch1: Vec<RasterPoint> = cache.unquantized.clone();

        cache.cached_mapping = CoordinateSpaceMapping::Transform(
            perspective_rotate_x_translate_y(80.0, 1000.0, -20.0),
        );
        cache.clear_scratch();
        let r2 = cache.append_corners_from_mapping(local_rect, local_to_raster);
        assert_eq!(r2.count, 8);
        let scratch2: Vec<RasterPoint> = cache.unquantized.clone();

        assert_ne!(
            scratch1, scratch2,
            "different perspective transforms must produce different fingerprints",
        );
    }

    /// The same transform applied twice must produce the same fingerprint, so a
    /// static perspective-crossing primitive does not trip spurious invalidations.
    #[test]
    fn perspective_camera_plane_fingerprint_stable_for_unchanged_transform() {
        let local_rect = LayoutRect::new(
            LayoutPoint::new(0.0, 0.0),
            LayoutPoint::new(200.0, 2000.0),
        );
        let local_to_raster = ScaleOffset::identity();

        let mut cache = CornersCache::new();

        let m = perspective_rotate_x_translate_y(80.0, 1000.0, -40.0);

        cache.cached_mapping = CoordinateSpaceMapping::Transform(m);
        cache.clear_scratch();
        let _ = cache.append_corners_from_mapping(local_rect, local_to_raster);
        let scratch1: Vec<RasterPoint> = cache.unquantized.clone();

        cache.cached_mapping = CoordinateSpaceMapping::Transform(m);
        cache.clear_scratch();
        let _ = cache.append_corners_from_mapping(local_rect, local_to_raster);
        let scratch2: Vec<RasterPoint> = cache.unquantized.clone();

        assert_eq!(
            scratch1, scratch2,
            "the same perspective transform must produce identical fingerprints",
        );
    }

    /// A transform without a perspective component (rotate only) must take the
    /// fast path and emit 4 projected corners, not the 8-element fingerprint.
    #[test]
    fn no_perspective_uses_projected_corners() {
        let local_rect = LayoutRect::new(
            LayoutPoint::new(0.0, 0.0),
            LayoutPoint::new(100.0, 100.0),
        );
        let local_to_raster = ScaleOffset::identity();

        let mut cache = CornersCache::new();
        // Pure rotation around X axis — no perspective component (m14, m24, m34 = 0
        // and m44 = 1), so the fast-path projection should be used.
        cache.cached_mapping = CoordinateSpaceMapping::<LayoutPixel, LayoutPixel>::Transform(
            LayoutTransform::rotation(1.0, 0.0, 0.0, Angle::degrees(45.0)),
        );
        cache.clear_scratch();
        let r = cache.append_corners_from_mapping(local_rect, local_to_raster);
        assert_eq!(r.count, 4, "non-perspective transform must emit 4 corners");
    }
}
