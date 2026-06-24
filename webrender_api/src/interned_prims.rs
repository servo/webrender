/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Interned primitive scene descriptions.
//!
//! These are the per-primitive "scene description" structs that `webrender`
//! interns (the `T` in its `PrimKey<T>` / `Internable` machinery). Their fields
//! are all api-resident, so they live here in `webrender_api` to let
//! content-process interning in the `DisplayListBuilder` hold them; `webrender`
//! re-exports each from its former home and keeps the `Internable` /
//! `InternablePrimitive` impls and per-frame templates (the trait and templates
//! are webrender-internal). Not part of the public API surface.

use crate::serde::{Serialize, Deserialize};
use crate::{
    AlphaType, BoxShadowClipMode, ColorDepth, ColorRange, ColorU, ExtendMode, ImageKey,
    ImageRendering, LineOrientation, LineStyle, PropertyBinding, YuvColorSpace, YuvFormat,
};
use crate::key_types::{
    BorderRadiusAu, ConicGradientParams, GradientStopKey, NinePatchDescriptor, NormalBorderAu,
    PointKey, RadialGradientParams, SizeKey, StretchSizeKey, VectorKey,
};
use crate::units::LayoutSideOffsetsAu;
use app_units::Au;

#[derive(Debug, Clone, Eq, MallocSizeOf, PartialEq, Hash, Serialize, Deserialize)]
pub struct RectanglePrim {
    pub color: PropertyBinding<ColorU>,
}

#[derive(Debug, Clone, MallocSizeOf, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct BoxShadow {
    pub color: ColorU,
    pub blur_radius: Au,
    pub clip_mode: BoxShadowClipMode,
    pub shadow_radius: BorderRadiusAu,
    pub element_radius: BorderRadiusAu,
    /// `box-shadow` offset of the shadow relative to the element, in
    /// local space.
    pub box_offset: VectorKey,
    /// Signed spread radius. Positive for Outset, negative for Inset
    /// (matches the convention in `add_box_shadow`).
    pub spread_amount: Au,
}

#[derive(Debug, Clone, Eq, PartialEq, MallocSizeOf, Hash, Serialize, Deserialize)]
pub struct Image {
    pub key: ImageKey,
    pub stretch_size: StretchSizeKey,
    pub tile_spacing: SizeKey,
    pub color: ColorU,
    pub image_rendering: ImageRendering,
    pub alpha_type: AlphaType,
}

#[derive(Debug, Clone, Eq, MallocSizeOf, PartialEq, Hash, Serialize, Deserialize)]
pub struct YuvImage {
    pub color_depth: ColorDepth,
    pub yuv_key: [ImageKey; 3],
    pub format: YuvFormat,
    pub color_space: YuvColorSpace,
    pub color_range: ColorRange,
    pub image_rendering: ImageRendering,
}

#[derive(Clone, Debug, Hash, MallocSizeOf, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineDecoration {
    pub style: LineStyle,
    pub orientation: LineOrientation,
    pub wavy_line_thickness: Au,
    pub color: ColorU,
}

#[derive(Debug, Clone, Eq, MallocSizeOf, PartialEq, Hash, Serialize, Deserialize)]
pub struct NormalBorderPrim {
    pub border: NormalBorderAu,
    pub widths: LayoutSideOffsetsAu,
}

#[derive(Clone, Debug, Eq, MallocSizeOf, PartialEq, Hash, Serialize, Deserialize)]
pub struct RadialGradient {
    pub extend_mode: ExtendMode,
    pub center: PointKey,
    pub params: RadialGradientParams,
    /// Per-axis tile size encoded as a fraction of the prim's size. See the
    /// matching `stretch_ratio` field on `RadialGradientKey`.
    pub stretch_ratio: SizeKey,
    pub stops: Vec<GradientStopKey>,
    pub tile_spacing: SizeKey,
    pub nine_patch: Option<Box<NinePatchDescriptor>>,
}

#[derive(Clone, Debug, Eq, MallocSizeOf, PartialEq, Hash, Serialize, Deserialize)]
pub struct ConicGradient {
    pub extend_mode: ExtendMode,
    pub center: PointKey,
    pub params: ConicGradientParams,
    /// Per-axis tile size encoded as a fraction of the prim's size. See the
    /// matching `stretch_ratio` field on `ConicGradientKey`.
    pub stretch_ratio: SizeKey,
    pub stops: Vec<GradientStopKey>,
    pub tile_spacing: SizeKey,
    pub nine_patch: Option<Box<NinePatchDescriptor>>,
}
