/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Radial gradients
//!
//! Specification: https://drafts.csswg.org/css-images-4/#radial-gradients
//!
//! Radial gradients are rendered via cached render tasks and composited with the image brush.

use api::{ExtendMode, GradientStop};
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

use std::ops::{Deref, DerefMut};
use super::{stops_and_min_alpha, GradientStopKey};

// `RadialGradientParams` now lives in `webrender_api::key_types` so builder-side
// interning keys can reference it. Re-exported to keep existing references
// working.
pub use api::key_types::RadialGradientParams;

/// Identifying key for a radial gradient.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Clone, Eq, PartialEq, Hash, MallocSizeOf)]
pub struct RadialGradientKey {
    pub common: PrimKeyCommonData,
    pub extend_mode: ExtendMode,
    pub center: PointKey,
    pub params: RadialGradientParams,
    /// Per-axis tile size encoded as a fraction of `common.prim_size`. The
    /// runtime `stretch_size` is `stretch_ratio * common.prim_size`.
    pub stretch_ratio: SizeKey,
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
            stretch_ratio: radial_grad.stretch_ratio,
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
    /// Per-axis fraction of `common.prim_size` covered by one tile of the
    /// gradient pattern. Multiply by `common.prim_size` at use to recover the
    /// absolute stretch_size.
    pub stretch_ratio: LayoutSize,
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

        RadialGradientTemplate {
            common,
            center: item.center.into(),
            extend_mode: item.extend_mode,
            params: item.params,
            stretch_ratio: item.stretch_ratio.into(),
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
    /// Per-axis tile size encoded as a fraction of the prim's size. See
    /// [`RadialGradientKey::stretch_ratio`].
    pub stretch_ratio: SizeKey,
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


// `optimize_radial_gradient` now lives in `webrender_api::prim_geometry` so
// content-process interning can share it. Re-exported here to keep existing
// references working.
pub use api::prim_geometry::optimize_radial_gradient;
