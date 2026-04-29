/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Conic gradients
//!
//! Specification: https://drafts.csswg.org/css-images-4/#conic-gradients
//!
//! Conic gradients are rendered via cached render tasks and composited with the image brush.

use euclid::vec2;
use api::{ExtendMode, GradientStop};
use api::units::*;
use crate::pattern::gradient::{conic_gradient_pattern};
use crate::pattern::{Pattern, PatternBuilder, PatternBuilderContext, PatternBuilderState};
use crate::scene_building::IsVisible;
use crate::intern::{Internable, InternDebug, Handle as InternHandle};
use crate::internal_types::LayoutPrimitiveInfo;
use crate::prim_store::{PrimitiveKind, PrimitiveOpacity};
use crate::prim_store::{PrimKeyCommonData, PrimTemplateCommonData, PrimitiveStore};
use crate::prim_store::{NinePatchDescriptor, PointKey, SizeKey, InternablePrimitive};

use std::{hash, ops::{Deref, DerefMut}};
use super::{stops_and_min_alpha, GradientStopKey};

/// Hashable conic gradient parameters, for use during prim interning.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Clone, MallocSizeOf, PartialEq)]
pub struct ConicGradientParams {
    pub angle: f32, // in radians
    pub start_offset: f32,
    pub end_offset: f32,
}

impl Eq for ConicGradientParams {}

impl hash::Hash for ConicGradientParams {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.angle.to_bits().hash(state);
        self.start_offset.to_bits().hash(state);
        self.end_offset.to_bits().hash(state);
    }
}

/// Identifying key for a line decoration.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Clone, Eq, PartialEq, Hash, MallocSizeOf)]
pub struct ConicGradientKey {
    pub common: PrimKeyCommonData,
    pub extend_mode: ExtendMode,
    pub center: PointKey,
    pub params: ConicGradientParams,
    pub stretch_size: SizeKey,
    pub stops: Vec<GradientStopKey>,
    pub tile_spacing: SizeKey,
    pub nine_patch: Option<Box<NinePatchDescriptor>>,
}

impl ConicGradientKey {
    pub fn new(
        info: &LayoutPrimitiveInfo,
        conic_grad: ConicGradient,
    ) -> Self {
        ConicGradientKey {
            common: info.into(),
            extend_mode: conic_grad.extend_mode,
            center: conic_grad.center,
            params: conic_grad.params,
            stretch_size: conic_grad.stretch_size,
            stops: conic_grad.stops,
            tile_spacing: conic_grad.tile_spacing,
            nine_patch: conic_grad.nine_patch,
        }
    }
}

impl InternDebug for ConicGradientKey {}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(MallocSizeOf)]
pub struct ConicGradientTemplate {
    pub common: PrimTemplateCommonData,
    pub extend_mode: ExtendMode,
    pub center: LayoutPoint,
    pub params: ConicGradientParams,
    pub task_size: DeviceIntSize,
    pub scale: DeviceVector2D,
    pub stretch_size: LayoutSize,
    pub tile_spacing: LayoutSize,
    pub border_nine_patch: Option<Box<NinePatchDescriptor>>,
    pub stops_opacity: PrimitiveOpacity,
    pub stops: Vec<GradientStop>,
}

impl PatternBuilder for ConicGradientTemplate {
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

        // ConicGradientTemplate stores the center point relative to the primitive
        // origin, but the shader works with start/end points in "proper" layout
        // coordinates (relative to the primitive's spatial node).
        let center = self.center + ctx.prim_origin.to_vector() + offset;

        conic_gradient_pattern(
            center,
            no_scale,
            self.params.angle,
            self.params.start_offset,
            self.params.end_offset,
            self.extend_mode,
            &self.stops,
            state.frame_gpu_data,
        )
    }
}

impl Deref for ConicGradientTemplate {
    type Target = PrimTemplateCommonData;
    fn deref(&self) -> &Self::Target {
        &self.common
    }
}

impl DerefMut for ConicGradientTemplate {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.common
    }
}

impl From<ConicGradientKey> for ConicGradientTemplate {
    fn from(item: ConicGradientKey) -> Self {
        let common = PrimTemplateCommonData::with_key_common(item.common);

        let (stops, min_alpha) = stops_and_min_alpha(&item.stops);

        // Save opacity of the stops for use in
        // selecting which pass this gradient
        // should be drawn in.
        let stops_opacity = PrimitiveOpacity::from_alpha(min_alpha);

        let mut stretch_size: LayoutSize = item.stretch_size.into();
        stretch_size.width = stretch_size.width.min(common.prim_size.width);
        stretch_size.height = stretch_size.height.min(common.prim_size.height);

        fn approx_eq(a: f32, b: f32) -> bool { (a - b).abs() < 0.01 }

        // Attempt to detect some of the common configurations with hard gradient stops. Allow
        // those a higher maximum resolution to avoid the worst cases of aliasing artifacts with
        // large conic gradients. A better solution would be to go back to rendering very large
        // conic gradients via a brush shader instead of caching all of them (unclear whether
        // it is important enough to warrant the better solution).
        let mut has_hard_stops = false;
        let mut prev_stop = None;
        let offset_range = item.params.end_offset - item.params.start_offset;
        for stop in &stops {
            if offset_range <= 0.0 {
                break;
            }
            if let Some(prev_offset) = prev_stop {
                // Check whether two consecutive stops are very close (hard stops).
                if stop.offset < prev_offset + 0.005 / offset_range {
                    // a is the angle of the stop normalized into 0-1 space and repeating in the 0-0.25 range.
                    // If close to 0.0 or 0.25 it means the stop is vertical or horizontal. For those, the lower
                    // resolution isn't a big issue.
                    let a = item.params.angle / (2.0 * std::f32::consts::PI)
                        + item.params.start_offset
                        + stop.offset / offset_range;
                    let a = a.rem_euclid(0.25);

                    if !approx_eq(a, 0.0) && !approx_eq(a, 0.25) {
                        has_hard_stops = true;
                        break;
                    }
                }
            }
            prev_stop = Some(stop.offset);
        }

        let max_size = if has_hard_stops {
            2048.0
        } else {
            1024.0
        };

        // Avoid rendering enormous gradients. Radial gradients are mostly made of soft transitions,
        // so it is unlikely that rendering at a higher resolution that 1024 would produce noticeable
        // differences, especially with 8 bits per channel.
        let mut task_size: DeviceSize = stretch_size.cast_unit();
        let mut scale = vec2(1.0, 1.0);
        if task_size.width > max_size {
            scale.x = task_size.width / max_size;
            task_size.width = max_size;
        }
        if task_size.height > max_size {
            scale.y = task_size.height / max_size;
            task_size.height = max_size;
        }

        ConicGradientTemplate {
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

pub type ConicGradientDataHandle = InternHandle<ConicGradient>;

#[derive(Debug, MallocSizeOf)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct ConicGradient {
    pub extend_mode: ExtendMode,
    pub center: PointKey,
    pub params: ConicGradientParams,
    pub stretch_size: SizeKey,
    pub stops: Vec<GradientStopKey>,
    pub tile_spacing: SizeKey,
    pub nine_patch: Option<Box<NinePatchDescriptor>>,
}

impl Internable for ConicGradient {
    type Key = ConicGradientKey;
    type StoreData = ConicGradientTemplate;
    type InternData = ();
    const PROFILE_COUNTER: usize = crate::profiler::INTERNED_CONIC_GRADIENTS;
}

impl InternablePrimitive for ConicGradient {
    fn into_key(
        self,
        info: &LayoutPrimitiveInfo,
    ) -> ConicGradientKey {
        ConicGradientKey::new(info, self)
    }

    fn make_instance_kind(
        _key: ConicGradientKey,
        data_handle: ConicGradientDataHandle,
        _prim_store: &mut PrimitiveStore,
    ) -> PrimitiveKind {
        PrimitiveKind::ConicGradient {
            data_handle,
        }
    }
}

impl IsVisible for ConicGradient {
    fn is_visible(&self) -> bool {
        true
    }
}

