/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Radial gradients
//!
//! Specification: https://drafts.csswg.org/css-images-4/#radial-gradients
//!
//! Radial gradients are rendered via cached render tasks and composited with the image brush.

use euclid::{vec2, size2};
use api::{ColorU, ExtendMode, GradientStop};
use api::units::*;
use crate::pattern::gradient::{radial_gradient_pattern};
use crate::pattern::{Pattern, PatternBuilder, PatternBuilderContext, PatternBuilderState};
use crate::scene_building::IsVisible;
use crate::intern::{Internable, InternDebug, Handle as InternHandle};
use crate::internal_types::LayoutPrimitiveInfo;
use crate::prim_store::{InternablePrimitive};
use crate::prim_store::{PrimitiveKind, PrimitiveOpacity};
use crate::prim_store::{PrimKeyCommonData, PrimTemplateCommonData, PrimitiveStore};
use crate::prim_store::{NinePatchDescriptor, PointKey, SizeKey};
use crate::segment::EdgeMask;

use std::{hash, ops::{Deref, DerefMut}};
use super::{
    stops_and_min_alpha, GradientStopKey,
    apply_gradient_local_clip,
};

/// Hashable radial gradient parameters, for use during prim interning.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Clone, MallocSizeOf, PartialEq)]
pub struct RadialGradientParams {
    pub start_radius: f32,
    pub end_radius: f32,
    pub ratio_xy: f32,
}

impl Eq for RadialGradientParams {}

impl hash::Hash for RadialGradientParams {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.start_radius.to_bits().hash(state);
        self.end_radius.to_bits().hash(state);
        self.ratio_xy.to_bits().hash(state);
    }
}

/// Identifying key for a radial gradient.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Clone, Eq, PartialEq, Hash, MallocSizeOf)]
pub struct RadialGradientKey {
    pub common: PrimKeyCommonData,
    pub extend_mode: ExtendMode,
    pub center: PointKey,
    pub params: RadialGradientParams,
    pub stretch_size: SizeKey,
    pub stops: Vec<GradientStopKey>,
    pub tile_spacing: SizeKey,
    pub nine_patch: Option<Box<NinePatchDescriptor>>,
}

impl RadialGradientKey {
    pub fn new(
        info: &LayoutPrimitiveInfo,
        radial_grad: RadialGradient,
    ) -> Self {
        RadialGradientKey {
            common: info.into(),
            extend_mode: radial_grad.extend_mode,
            center: radial_grad.center,
            params: radial_grad.params,
            stretch_size: radial_grad.stretch_size,
            stops: radial_grad.stops,
            tile_spacing: radial_grad.tile_spacing,
            nine_patch: radial_grad.nine_patch,
        }
    }
}

impl InternDebug for RadialGradientKey {}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(MallocSizeOf)]
#[derive(Debug)]
pub struct RadialGradientTemplate {
    pub common: PrimTemplateCommonData,
    pub extend_mode: ExtendMode,
    pub params: RadialGradientParams,
    pub center: LayoutPoint,
    pub task_size: DeviceIntSize,
    pub scale: DeviceVector2D,
    pub stretch_size: LayoutSize,
    pub tile_spacing: LayoutSize,
    pub border_nine_patch: Option<Box<NinePatchDescriptor>>,
    pub stops_opacity: PrimitiveOpacity,
    pub stops: Vec<GradientStop>,
}

impl PatternBuilder for RadialGradientTemplate {
    fn build(
        &self,
        _sub_rect: Option<DeviceRect>,
        offset: LayoutVector2D,
        ctx: &PatternBuilderContext,
        state: &mut PatternBuilderState,
    ) -> Pattern {
        // The scaling parameter is used to compensate for when we reduce the size
        // of the render task for cached gradients. Here we aren't applying any.
        let no_scale = DeviceVector2D::one();

        // RadialGradientTemplate stores the center point relative to the primitive
        // origin, but the shader works with start/end points in "proper" layout
        // coordinates (relative to the primitive's spatial node).
        let center = self.center.cast_unit() + ctx.prim_origin.to_vector() + offset;

        radial_gradient_pattern(
            center,
            no_scale,
            self.params.start_radius,
            self.params.end_radius,
            self.params.ratio_xy,
            self.extend_mode,
            &self.stops,
            ctx.fb_config.is_software,
            state.frame_gpu_data,
        )
    }
}

impl Deref for RadialGradientTemplate {
    type Target = PrimTemplateCommonData;
    fn deref(&self) -> &Self::Target {
        &self.common
    }
}

impl DerefMut for RadialGradientTemplate {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.common
    }
}

impl From<RadialGradientKey> for RadialGradientTemplate {
    fn from(item: RadialGradientKey) -> Self {
        let common = PrimTemplateCommonData::with_key_common(item.common);

        let (stops, min_alpha) = stops_and_min_alpha(&item.stops);

        // Save opacity of the stops for use in
        // selecting which pass this gradient
        // should be drawn in.
        let stops_opacity = PrimitiveOpacity::from_alpha(min_alpha);

        let mut stretch_size: LayoutSize = item.stretch_size.into();
        stretch_size.width = stretch_size.width.min(common.prim_size.width);
        stretch_size.height = stretch_size.height.min(common.prim_size.height);

        // Avoid rendering enormous gradients. Radial gradients are mostly made of soft transitions,
        // so it is unlikely that rendering at a higher resolution that 1024 would produce noticeable
        // differences, especially with 8 bits per channel.
        const MAX_SIZE: f32 = 1024.0;
        let mut task_size: DeviceSize = stretch_size.cast_unit();
        let mut scale = vec2(1.0, 1.0);
        if task_size.width > MAX_SIZE {
            scale.x = task_size.width/ MAX_SIZE;
            task_size.width = MAX_SIZE;
        }
        if task_size.height > MAX_SIZE {
            scale.y = task_size.height /MAX_SIZE;
            task_size.height = MAX_SIZE;
        }

        RadialGradientTemplate {
            common,
            center: item.center.into(),
            extend_mode: item.extend_mode,
            params: item.params,
            stretch_size,
            task_size: task_size.ceil().to_i32(),
            scale,
            tile_spacing: item.tile_spacing.into(),
            border_nine_patch: item.nine_patch,
            stops_opacity,
            stops,
        }
    }
}

pub type RadialGradientDataHandle = InternHandle<RadialGradient>;

#[derive(Debug, MallocSizeOf)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct RadialGradient {
    pub extend_mode: ExtendMode,
    pub center: PointKey,
    pub params: RadialGradientParams,
    pub stretch_size: SizeKey,
    pub stops: Vec<GradientStopKey>,
    pub tile_spacing: SizeKey,
    pub nine_patch: Option<Box<NinePatchDescriptor>>,
}

impl Internable for RadialGradient {
    type Key = RadialGradientKey;
    type StoreData = RadialGradientTemplate;
    type InternData = ();
    const PROFILE_COUNTER: usize = crate::profiler::INTERNED_RADIAL_GRADIENTS;
}

impl InternablePrimitive for RadialGradient {
    fn into_key(
        self,
        info: &LayoutPrimitiveInfo,
    ) -> RadialGradientKey {
        RadialGradientKey::new(info, self)
    }

    fn make_instance_kind(
        _key: RadialGradientKey,
        data_handle: RadialGradientDataHandle,
        _prim_store: &mut PrimitiveStore,
    ) -> PrimitiveKind {
        PrimitiveKind::RadialGradient {
            data_handle,
        }
    }
}

impl IsVisible for RadialGradient {
    fn is_visible(&self) -> bool {
        true
    }
}


/// Avoid invoking the radial gradient shader on large areas where the color is
/// constant.
///
/// If the extend mode is set to clamp, the "interesting" part
/// of the gradient is only in the bounds of the gradient's ellipse, and the rest
/// is the color of the last gradient stop.
///
/// Sometimes we run into radial gradient with a small radius compared to the
/// primitive bounds, which means a large area of the primitive is a constant color
/// This function tries to detect that, potentially shrink the gradient primitive to only
/// the useful part and if needed insert solid color primitives around the gradient where
/// parts of it have been removed.
///
/// If the radial gradient is split into multiple primitives, we must prevent anti-aliasing
/// from being appplied at the edges connecting these primitives to prevent seams. This is
/// done by masking out sides in `aa_mask` for the central gradient primitive and providing
/// an edge mask for each extracted solid primitive.
pub fn optimize_radial_gradient(
    prim_rect: &mut LayoutRect,
    stretch_size: &mut LayoutSize,
    center: &mut LayoutPoint,
    tile_spacing: &mut LayoutSize,
    aa_mask: &mut EdgeMask,
    clip_rect: &LayoutRect,
    radius: LayoutSize,
    end_offset: f32,
    extend_mode: ExtendMode,
    stops: &[GradientStopKey],
    solid_parts: &mut dyn FnMut(&LayoutRect, ColorU, EdgeMask),
) {
    let offset = apply_gradient_local_clip(
        prim_rect,
        stretch_size,
        tile_spacing,
        clip_rect
    );

    *center += offset;

    if extend_mode != ExtendMode::Clamp || stops.is_empty() {
        return;
    }

    // Bounding box of the "interesting" part of the gradient.
    let min = prim_rect.min + center.to_vector() - radius.to_vector() * end_offset;
    let max = prim_rect.min + center.to_vector() + radius.to_vector() * end_offset;

    // The (non-repeated) gradient primitive rect.
    let gradient_rect = LayoutRect::from_origin_and_size(
        prim_rect.min,
        *stretch_size,
    );

    // How much internal margin between the primitive bounds and the gradient's
    // bounding rect (areas that are a constant color).
    let mut l = (min.x - gradient_rect.min.x).max(0.0).floor();
    let mut t = (min.y - gradient_rect.min.y).max(0.0).floor();
    let mut r = (gradient_rect.max.x - max.x).max(0.0).floor();
    let mut b = (gradient_rect.max.y - max.y).max(0.0).floor();

    let is_tiled = prim_rect.width() > stretch_size.width + tile_spacing.width
        || prim_rect.height() > stretch_size.height + tile_spacing.height;

    let bg_color = stops.last().unwrap().color;

    if bg_color.a != 0 && is_tiled {
        // If the primitive has repetitions, it's not enough to insert solid rects around it,
        // so bail out.
        return;
    }

    // If the background is fully transparent, shrinking the primitive bounds as much as possible
    // is always a win. If the background is not transparent, we have to insert solid rectangles
    // around the shrunk parts.
    // If the background is transparent and the primitive is tiled, the optimization may introduce
    // tile spacing which forces the tiling to be manually decomposed.
    // Either way, don't bother optimizing unless it saves a significant amount of pixels.
    if bg_color.a != 0 || (is_tiled && tile_spacing.is_empty()) {
        let threshold = 128.0;
        if l < threshold { l = 0.0 }
        if t < threshold { t = 0.0 }
        if r < threshold { r = 0.0 }
        if b < threshold { b = 0.0 }
    }

    if l + t + r + b == 0.0 {
        // No adjustment to make;
        return;
    }

    // Insert solid rectangles around the gradient, in the places where the primitive will be
    // shrunk.
    if bg_color.a != 0 {
        if l != 0.0 && t != 0.0 {
            let solid_rect = LayoutRect::from_origin_and_size(
                gradient_rect.min,
                size2(l, t),
            );
            solid_parts(&solid_rect, bg_color, EdgeMask::LEFT | EdgeMask::TOP);
        }

        if l != 0.0 && b != 0.0 {
            let solid_rect = LayoutRect::from_origin_and_size(
                gradient_rect.bottom_left() - vec2(0.0, b),
                size2(l, b),
            );
            solid_parts(&solid_rect, bg_color, EdgeMask::LEFT | EdgeMask::BOTTOM);
        }

        if t != 0.0 && r != 0.0 {
            let solid_rect = LayoutRect::from_origin_and_size(
                gradient_rect.top_right() - vec2(r, 0.0),
                size2(r, t),
            );
            solid_parts(&solid_rect, bg_color, EdgeMask::TOP | EdgeMask::RIGHT);
        }

        if r != 0.0 && b != 0.0 {
            let solid_rect = LayoutRect::from_origin_and_size(
                gradient_rect.bottom_right() - vec2(r, b),
                size2(r, b),
            );
            solid_parts(&solid_rect, bg_color, EdgeMask::RIGHT | EdgeMask::BOTTOM);
        }

        if l != 0.0 {
            let solid_rect = LayoutRect::from_origin_and_size(
                gradient_rect.min + vec2(0.0, t),
                size2(l, gradient_rect.height() - t - b),
            );
            let mut solid_aa = EdgeMask::LEFT;
            solid_aa.set(EdgeMask::TOP, t == 0.0);
            solid_aa.set(EdgeMask::BOTTOM, b == 0.0);
            solid_parts(&solid_rect, bg_color, solid_aa);
            aa_mask.remove(EdgeMask::LEFT);
        }

        if r != 0.0 {
            let solid_rect = LayoutRect::from_origin_and_size(
                gradient_rect.top_right() + vec2(-r, t),
                size2(r, gradient_rect.height() - t - b),
            );
            let mut solid_aa = EdgeMask::RIGHT;
            solid_aa.set(EdgeMask::TOP, t == 0.0);
            solid_aa.set(EdgeMask::BOTTOM, b == 0.0);
            solid_parts(&solid_rect, bg_color, solid_aa);
            aa_mask.remove(EdgeMask::RIGHT);
        }

        if t != 0.0 {
            let solid_rect = LayoutRect::from_origin_and_size(
                gradient_rect.min + vec2(l, 0.0),
                size2(gradient_rect.width() - l - r, t),
            );
            let mut solid_aa = EdgeMask::TOP;
            solid_aa.set(EdgeMask::LEFT, l == 0.0);
            solid_aa.set(EdgeMask::RIGHT, r == 0.0);
            solid_parts(&solid_rect, bg_color, solid_aa);
            aa_mask.remove(EdgeMask::TOP);
        }

        if b != 0.0 {
            let solid_rect = LayoutRect::from_origin_and_size(
                gradient_rect.bottom_left() + vec2(l, -b),
                size2(gradient_rect.width() - l - r, b),
            );
            let mut solid_aa = EdgeMask::BOTTOM;
            solid_aa.set(EdgeMask::LEFT, l == 0.0);
            solid_aa.set(EdgeMask::RIGHT, r == 0.0);
            solid_parts(&solid_rect, bg_color, solid_aa);
            aa_mask.remove(EdgeMask::BOTTOM);
        }
    }

    // Shrink the gradient primitive.

    prim_rect.min.x += l;
    prim_rect.min.y += t;

    stretch_size.width -= l + r;
    stretch_size.height -= b + t;

    center.x -= l;
    center.y -= t;

    tile_spacing.width += l + r;
    tile_spacing.height += t + b;
}
