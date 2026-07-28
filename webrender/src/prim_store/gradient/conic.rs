/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Conic gradients
//!
//! Specification: https://drafts.csswg.org/css-images-4/#conic-gradients
//!
//! Conic gradients are rendered as quads with the gradient pattern (ps_quad_gradient).

use api::{ExtendMode, GradientStop};
use api::units::*;
use crate::pattern::gradient::{conic_gradient_pattern};
use crate::pattern::{Pattern, PatternBuilder, PatternBuilderContext, PatternBuilderState};
use crate::scene_building::IsVisible;
use crate::intern::{Internable, InternDebug, Handle as InternHandle};
use crate::internal_types::LayoutPrimitiveInfo;
use crate::prim_store::{PrimitiveKind, PrimitiveOpacity};
use crate::prim_store::{PrimTemplateCommonData, PrimitiveStore};
use crate::prim_store::{NinePatchDescriptor, InternablePrimitive};

use std::ops::{Deref, DerefMut};
use super::stops_and_min_alpha;

// `ConicGradientParams` now lives in `webrender_api::key_types` so builder-side
// interning keys can reference it. Re-exported to keep existing references
// working.
pub use api::key_types::ConicGradientParams;

// `ConicGradientKey` lives in `webrender_api::interned_prims` (alongside the
// `ConicGradient` value) so the key can be built from api-resident types. The
// frame-time `ConicGradientTemplate` and interning glue stay here.
pub use api::interned_prims::ConicGradientKey;

impl InternDebug for ConicGradientKey {}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(MallocSizeOf)]
pub struct ConicGradientTemplate {
    pub common: PrimTemplateCommonData,
    pub extend_mode: ExtendMode,
    pub center: LayoutPoint,
    pub params: ConicGradientParams,
    /// Per-axis fraction of `common.prim_size` covered by one tile of the
    /// gradient pattern. Multiply by `common.prim_size` at use to recover the
    /// absolute stretch_size.
    pub stretch_ratio: LayoutSize,
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
        // ConicGradientTemplate stores the center point relative to the primitive
        // origin, but the shader works with start/end points in "proper" layout
        // coordinates (relative to the primitive's spatial node).
        let center = self.center + ctx.prim_origin.to_vector() + offset;

        conic_gradient_pattern(
            center,
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

        ConicGradientTemplate {
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

pub type ConicGradientDataHandle = InternHandle<ConicGradient>;

// `ConicGradient` now lives in `webrender_api::interned_prims` so content-process
// interning can hold it. Re-exported to keep existing references working.
pub use api::interned_prims::ConicGradient;

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
        ConicGradientKey::new(info.into(), self)
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

