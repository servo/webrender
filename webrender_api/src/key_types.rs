/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Hashable building blocks for WebRender interning keys.
//!
//! These are `f32`-bit-hashed / `Au`-quantized wrappers around geometry and
//! color types, used as fragments of the interning keys for primitives (and,
//! going forward, clips and other interned items). They live in `webrender_api`
//! so that keys built in the content process can reference only api-resident
//! types; `webrender` re-exports them from their former homes.

use crate::serde::{Serialize, Deserialize};
use crate::{ColorU, BorderRadius, BorderSide, BorderStyle, NormalBorder, RepeatMode, GradientStop, PrimitiveFlags};
use crate::{FillRule, GlyphIndex, POLYGON_CLIP_VERTEX_MAX};
use crate::units::{LayoutVector2D, WorldVector2D, LayoutPoint, PicturePoint, WorldPoint};
use crate::units::{LayoutSize, LayoutSizeAu, LayoutPointAu, AuHelpers, LayoutSideOffsets, DeviceIntSideOffsets};
use euclid::{Size2D, SideOffsets2D};
use peek_poke::PeekPoke;
use std::hash::{Hash, Hasher};

/// Per-side mask used to select which primitive edges are anti-aliased. The bit
/// values must match the shader logic in `write_transform_vertex()`.
#[repr(C)]
#[derive(Copy, PartialEq, Eq, Clone, PartialOrd, Ord, Hash, Deserialize, MallocSizeOf, Serialize, PeekPoke)]
pub struct EdgeMask(u8);

bitflags! {
    impl EdgeMask: u8 {
        ///
        const LEFT = 0x1;
        ///
        const TOP = 0x2;
        ///
        const RIGHT = 0x4;
        ///
        const BOTTOM = 0x8;
    }
}

impl core::fmt::Debug for EdgeMask {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        if self.is_empty() {
            write!(f, "{:#x}", Self::empty().bits())
        } else {
            bitflags::parser::to_writer(self, f)
        }
    }
}

impl EdgeMask {
    /// Returns a rectangle with the egdes of `a` or `b`, selected indicidually.
    ///
    /// For each side bit, if it is set then use the edge from `a`, else use the edge from `b`.
    pub fn select<T: Copy, U>(&self, a: euclid::Box2D<T, U>, b: euclid::Box2D<T, U>) -> euclid::Box2D<T, U> {
        let mut rect = b;
        if self.contains(Self::LEFT) {
            rect.min.x = a.min.x;
        }
        if self.contains(Self::TOP) {
            rect.min.y = a.min.y;
        }
        if self.contains(Self::RIGHT) {
            rect.max.x = a.max.x;
        }
        if self.contains(Self::BOTTOM) {
            rect.max.y = a.max.y;
        }

        rect
    }
}

/// Fields common to every interned primitive key.
#[derive(Debug, Clone, Eq, MallocSizeOf, PartialEq, Hash, Deserialize, Serialize)]
pub struct PrimKeyCommonData {
    pub flags: PrimitiveFlags,
    pub aligned_aa_edges: EdgeMask,
    pub transformed_aa_edges: EdgeMask,
}

/// A hashable vector for use as a fragment of an interning key; the raw `f32`
/// bits are hashed so distinct-but-equal floats key consistently.
#[derive(Copy, Debug, Clone, MallocSizeOf, PartialEq, Serialize, Deserialize)]
pub struct VectorKey {
    pub x: f32,
    pub y: f32,
}

impl Eq for VectorKey {}

impl Hash for VectorKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.to_bits().hash(state);
        self.y.to_bits().hash(state);
    }
}

impl From<VectorKey> for LayoutVector2D {
    fn from(key: VectorKey) -> LayoutVector2D {
        LayoutVector2D::new(key.x, key.y)
    }
}

impl From<VectorKey> for WorldVector2D {
    fn from(key: VectorKey) -> WorldVector2D {
        WorldVector2D::new(key.x, key.y)
    }
}

impl From<LayoutVector2D> for VectorKey {
    fn from(vec: LayoutVector2D) -> VectorKey {
        VectorKey { x: vec.x, y: vec.y }
    }
}

impl From<WorldVector2D> for VectorKey {
    fn from(vec: WorldVector2D) -> VectorKey {
        VectorKey { x: vec.x, y: vec.y }
    }
}

/// An `Au`-quantized [`BorderRadius`] for use as a fragment of an interning key
/// (the float radii are quantized so they hash consistently).
#[derive(Copy, Clone, Debug, Hash, MallocSizeOf, PartialEq, Eq, Serialize, Deserialize)]
pub struct BorderRadiusAu {
    pub top_left: LayoutSizeAu,
    pub top_right: LayoutSizeAu,
    pub bottom_left: LayoutSizeAu,
    pub bottom_right: LayoutSizeAu,

    pub shape_top_left: u32,
    pub shape_top_right: u32,
    pub shape_bottom_left: u32,
    pub shape_bottom_right: u32,
}

impl From<BorderRadius> for BorderRadiusAu {
    fn from(radius: BorderRadius) -> BorderRadiusAu {
        BorderRadiusAu {
            top_left: radius.top_left.to_au(),
            top_right: radius.top_right.to_au(),
            bottom_right: radius.bottom_right.to_au(),
            bottom_left: radius.bottom_left.to_au(),
            shape_top_left: radius.shape_top_left.to_bits(),
            shape_top_right: radius.shape_top_right.to_bits(),
            shape_bottom_left: radius.shape_bottom_left.to_bits(),
            shape_bottom_right: radius.shape_bottom_right.to_bits(),
        }
    }
}

impl From<BorderRadiusAu> for BorderRadius {
    fn from(radius: BorderRadiusAu) -> Self {
        BorderRadius {
            top_left: LayoutSize::from_au(radius.top_left),
            top_right: LayoutSize::from_au(radius.top_right),
            bottom_right: LayoutSize::from_au(radius.bottom_right),
            bottom_left: LayoutSize::from_au(radius.bottom_left),
            shape_top_left: f32::from_bits(radius.shape_top_left),
            shape_top_right: f32::from_bits(radius.shape_top_right),
            shape_bottom_left: f32::from_bits(radius.shape_bottom_left),
            shape_bottom_right: f32::from_bits(radius.shape_bottom_right),
        }
    }
}

/// Au-quantized border side, for use as a fragment of an interning key.
#[derive(Clone, Debug, Hash, MallocSizeOf, PartialEq, Eq, Serialize, Deserialize)]
pub struct BorderSideAu {
    pub color: ColorU,
    pub style: BorderStyle,
}

impl From<BorderSide> for BorderSideAu {
    fn from(side: BorderSide) -> Self {
        BorderSideAu {
            color: side.color.into(),
            style: side.style,
        }
    }
}

impl From<BorderSideAu> for BorderSide {
    fn from(side: BorderSideAu) -> Self {
        BorderSide {
            color: side.color.into(),
            style: side.style,
        }
    }
}

/// Au-quantized normal border, for use as an interning key fragment.
#[derive(Debug, Clone, Hash, Eq, MallocSizeOf, PartialEq, Serialize, Deserialize)]
pub struct NormalBorderAu {
    pub left: BorderSideAu,
    pub right: BorderSideAu,
    pub top: BorderSideAu,
    pub bottom: BorderSideAu,
    pub radius: BorderRadiusAu,
    /// Whether to apply anti-aliasing on the border corners.
    ///
    /// Note that for this to be `false` and work, this requires the borders to
    /// be solid, and no border-radius.
    pub do_aa: bool,
}

impl NormalBorderAu {
    // Construct a border based upon self with color
    pub fn with_color(&self, color: ColorU) -> Self {
        let mut b = self.clone();
        b.left.color = color;
        b.right.color = color;
        b.top.color = color;
        b.bottom.color = color;
        b
    }
}

impl From<NormalBorder> for NormalBorderAu {
    fn from(border: NormalBorder) -> Self {
        NormalBorderAu {
            left: border.left.into(),
            right: border.right.into(),
            top: border.top.into(),
            bottom: border.bottom.into(),
            radius: border.radius.into(),
            do_aa: border.do_aa,
        }
    }
}

impl From<NormalBorderAu> for NormalBorder {
    fn from(border: NormalBorderAu) -> Self {
        NormalBorder {
            left: border.left.into(),
            right: border.right.into(),
            top: border.top.into(),
            bottom: border.bottom.into(),
            radius: border.radius.into(),
            do_aa: border.do_aa,
        }
    }
}

/// Clamp border radii so adjacent corners do not overlap given the border's
/// box size, scaling all radii by a uniform ratio. Shared by builder-side
/// interning (to build a stable normal-border key) and frame building.
pub fn ensure_no_corner_overlap(
    radius: &mut BorderRadius,
    size: LayoutSize,
) {
    let mut ratio = 1.0;
    let top_left_radius = &mut radius.top_left;
    let top_right_radius = &mut radius.top_right;
    let bottom_right_radius = &mut radius.bottom_right;
    let bottom_left_radius = &mut radius.bottom_left;

    if size.width > 0.0 {
        let sum = top_left_radius.width + top_right_radius.width;
        if size.width < sum {
            ratio = f32::min(ratio, size.width / sum);
        }

        let sum = bottom_left_radius.width + bottom_right_radius.width;
        if size.width < sum {
            ratio = f32::min(ratio, size.width / sum);
        }
    }

    if size.height > 0.0 {
        let sum = top_left_radius.height + bottom_left_radius.height;
        if size.height < sum {
            ratio = f32::min(ratio, size.height / sum);
        }

        let sum = top_right_radius.height + bottom_right_radius.height;
        if size.height < sum {
            ratio = f32::min(ratio, size.height / sum);
        }
    }

    if ratio < 1. {
        top_left_radius.width *= ratio;
        top_left_radius.height *= ratio;

        top_right_radius.width *= ratio;
        top_right_radius.height *= ratio;

        bottom_left_radius.width *= ratio;
        bottom_left_radius.height *= ratio;

        bottom_right_radius.width *= ratio;
        bottom_right_radius.height *= ratio;
    }
}

/// A hashable point for use as a fragment of an interning key; the raw `f32`
/// bits are hashed.
#[derive(Debug, Copy, Clone, MallocSizeOf, PartialEq, Serialize, Deserialize)]
pub struct PointKey {
    pub x: f32,
    pub y: f32,
}

impl Eq for PointKey {}

impl Hash for PointKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.to_bits().hash(state);
        self.y.to_bits().hash(state);
    }
}

impl From<PointKey> for LayoutPoint {
    fn from(key: PointKey) -> LayoutPoint {
        LayoutPoint::new(key.x, key.y)
    }
}

impl From<LayoutPoint> for PointKey {
    fn from(p: LayoutPoint) -> PointKey {
        PointKey { x: p.x, y: p.y }
    }
}

impl From<PicturePoint> for PointKey {
    fn from(p: PicturePoint) -> PointKey {
        PointKey { x: p.x, y: p.y }
    }
}

impl From<WorldPoint> for PointKey {
    fn from(p: WorldPoint) -> PointKey {
        PointKey { x: p.x, y: p.y }
    }
}

/// A hashable size for use as a fragment of an interning key; the raw `f32`
/// bits are hashed.
#[derive(Copy, Debug, Clone, MallocSizeOf, PartialEq, Serialize, Deserialize)]
pub struct SizeKey {
    w: f32,
    h: f32,
}

impl Eq for SizeKey {}

impl Hash for SizeKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.w.to_bits().hash(state);
        self.h.to_bits().hash(state);
    }
}

impl From<SizeKey> for LayoutSize {
    fn from(key: SizeKey) -> LayoutSize {
        LayoutSize::new(key.w, key.h)
    }
}

impl<U> From<Size2D<f32, U>> for SizeKey {
    fn from(size: Size2D<f32, U>) -> SizeKey {
        SizeKey { w: size.width, h: size.height }
    }
}

/// A hashable image stretch size for use as a fragment of an interning key. The
/// per-axis `fills_*` flags mean the axis fills the primitive rect (resolved at
/// frame build); when both are set the stored `size` is normalised to zero so
/// primitives of different sizes still intern to the same key.
#[derive(Copy, Debug, Clone, MallocSizeOf, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StretchSizeKey {
    pub size: SizeKey,
    pub fills_width: bool,
    pub fills_height: bool,
}

impl StretchSizeKey {
    /// Both axes fill the prim. The stored size is unused; normalised to zero so
    /// different prim sizes still intern to the same key.
    pub fn fills_prim() -> Self {
        StretchSizeKey {
            size: LayoutSize::zero().into(),
            fills_width: true,
            fills_height: true,
        }
    }
}

/// Hashable radial gradient parameters, for use during prim interning. The raw
/// `f32` bits are hashed so distinct-but-equal floats key consistently.
#[derive(Debug, Clone, MallocSizeOf, PartialEq, Serialize, Deserialize)]
pub struct RadialGradientParams {
    pub start_radius: f32,
    pub end_radius: f32,
    pub ratio_xy: f32,
}

impl Eq for RadialGradientParams {}

impl Hash for RadialGradientParams {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.start_radius.to_bits().hash(state);
        self.end_radius.to_bits().hash(state);
        self.ratio_xy.to_bits().hash(state);
    }
}

/// Hashable conic gradient parameters, for use during prim interning. The raw
/// `f32` bits are hashed so distinct-but-equal floats key consistently.
#[derive(Debug, Clone, MallocSizeOf, PartialEq, Serialize, Deserialize)]
pub struct ConicGradientParams {
    pub angle: f32, // in radians
    pub start_offset: f32,
    pub end_offset: f32,
}

impl Eq for ConicGradientParams {}

impl Hash for ConicGradientParams {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.angle.to_bits().hash(state);
        self.start_offset.to_bits().hash(state);
        self.end_offset.to_bits().hash(state);
    }
}

/// A hashable side-offsets for use as a fragment of an interning key; the raw
/// `f32` bits are hashed.
#[derive(Debug, Clone, MallocSizeOf, PartialEq, Serialize, Deserialize)]
pub struct SideOffsetsKey {
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

impl Eq for SideOffsetsKey {}

impl Hash for SideOffsetsKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.top.to_bits().hash(state);
        self.right.to_bits().hash(state);
        self.bottom.to_bits().hash(state);
        self.left.to_bits().hash(state);
    }
}

impl From<SideOffsetsKey> for LayoutSideOffsets {
    fn from(key: SideOffsetsKey) -> LayoutSideOffsets {
        LayoutSideOffsets::new(key.top, key.right, key.bottom, key.left)
    }
}

impl<U> From<SideOffsets2D<f32, U>> for SideOffsetsKey {
    fn from(offsets: SideOffsets2D<f32, U>) -> SideOffsetsKey {
        SideOffsetsKey {
            top: offsets.top,
            right: offsets.right,
            bottom: offsets.bottom,
            left: offsets.left,
        }
    }
}

/// A hashable descriptor for nine-patches, used by image and gradient borders.
#[derive(Debug, Clone, PartialEq, Eq, Hash, MallocSizeOf, Serialize, Deserialize)]
pub struct NinePatchDescriptor {
    pub width: i32,
    pub height: i32,
    pub slice: DeviceIntSideOffsets,
    pub fill: bool,
    pub repeat_horizontal: RepeatMode,
    pub repeat_vertical: RepeatMode,
    pub widths: SideOffsetsKey,
}

/// A hashable gradient stop for use as a fragment of an interning key; the raw
/// `f32` offset bits are hashed.
#[derive(Debug, Copy, Clone, MallocSizeOf, PartialEq, Serialize, Deserialize)]
pub struct GradientStopKey {
    pub offset: f32,
    pub color: ColorU,
}

impl GradientStopKey {
    pub fn empty() -> Self {
        GradientStopKey {
            offset: 0.0,
            color: ColorU::new(0, 0, 0, 0),
        }
    }
}

impl Into<GradientStopKey> for GradientStop {
    fn into(self) -> GradientStopKey {
        GradientStopKey {
            offset: self.offset,
            color: self.color.into(),
        }
    }
}

impl Eq for GradientStopKey {}

impl Hash for GradientStopKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.offset.to_bits().hash(state);
        self.color.hash(state);
    }
}

/// A hashable, fixed-size polygon clip, for use as a fragment of an interning
/// key. To keep a fixed-size representation we use a fixed number of points.
/// Our initialization method restricts us to values <= 32. If our constant
/// `POLYGON_CLIP_VERTEX_MAX` is > 32, the Rust compiler will complain.
#[derive(Copy, Debug, Clone, Hash, MallocSizeOf, PartialEq, Serialize, Deserialize)]
pub struct PolygonKey {
    pub point_count: u8,
    pub points: [PointKey; POLYGON_CLIP_VERTEX_MAX],
    pub fill_rule: FillRule,
}

impl PolygonKey {
    pub fn new(
        points_layout: &[LayoutPoint],
        fill_rule: FillRule,
    ) -> Self {
        // We have to fill fixed-size arrays with data from a Vec.
        // We'll do this by initializing the arrays to known-good
        // values then overwriting those values as long as our
        // iterator provides values.
        let mut points: [PointKey; POLYGON_CLIP_VERTEX_MAX] = [PointKey { x: 0.0, y: 0.0}; POLYGON_CLIP_VERTEX_MAX];

        let mut point_count: u8 = 0;
        for (src, dest) in points_layout.iter().zip(points.iter_mut()) {
            *dest = (*src as LayoutPoint).into();
            point_count = point_count + 1;
        }

        PolygonKey {
            point_count,
            points,
            fill_rule,
        }
    }
}

impl Eq for PolygonKey {}

/// A glyph instance with an `Au`-quantized pen position, for use as a fragment
/// of a text-run interning key. The point is stored in `Au` so the key is
/// stable across sub-pixel-equal positions.
#[derive(Debug, Clone, Eq, MallocSizeOf, PartialEq, Hash, Serialize, Deserialize)]
pub struct GlyphInstanceAu {
    pub index: GlyphIndex,
    pub point: LayoutPointAu,
}
