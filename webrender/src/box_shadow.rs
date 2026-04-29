/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
use api::{BorderRadius, BoxShadowClipMode, ClipMode, ColorF, ColorU, PropertyBinding};
use api::units::*;
use crate::border::{BorderRadiusAu};
use crate::clip::{ClipItemEntry, ClipItemKey, ClipItemKeyKind, ClipNodeId};
use crate::intern::{Handle as InternHandle, InternDebug, Internable};
use crate::prim_store::{InternablePrimitive, PrimKey, PrimTemplate, PrimTemplateCommonData};
use crate::prim_store::{PrimitiveKind, PrimitiveStore, RectKey};
use crate::prim_store::rectangle::RectanglePrim;
use crate::scene_building::{SceneBuilder, IsVisible};
use crate::spatial_tree::SpatialNodeIndex;
use crate::internal_types::LayoutPrimitiveInfo;

pub type BoxShadowKey = PrimKey<BoxShadow>;

impl BoxShadowKey {
    pub fn new(
        info: &LayoutPrimitiveInfo,
        shadow: BoxShadow,
    ) -> Self {
        BoxShadowKey {
            common: info.into(),
            kind: shadow,
        }
    }
}

impl InternDebug for BoxShadowKey {}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Clone, MallocSizeOf, Hash, Eq, PartialEq)]
pub struct BoxShadow {
    pub color: ColorU,
    pub blur_radius: Au,
    pub clip_mode: BoxShadowClipMode,
    pub inner_shadow_rect: RectKey,
    pub outer_shadow_rect: RectKey,
    pub shadow_radius: BorderRadiusAu,
    /// The element rect (prim_info.rect) in local space. Used to clip the
    /// element analytically in the shader (clip-out for outset, clip-in for inset).
    pub element_rect: RectKey,
    pub element_radius: BorderRadiusAu,
}

impl IsVisible for BoxShadow {
    fn is_visible(&self) -> bool {
        true
    }
}

pub type BoxShadowDataHandle = InternHandle<BoxShadow>;

impl InternablePrimitive for BoxShadow {
    fn into_key(
        self,
        info: &LayoutPrimitiveInfo,
    ) -> BoxShadowKey {
        BoxShadowKey::new(info, self)
    }

    fn make_instance_kind(
        _key: BoxShadowKey,
        data_handle: BoxShadowDataHandle,
        _prim_store: &mut PrimitiveStore,
    ) -> PrimitiveKind {
        PrimitiveKind::BoxShadow {
            data_handle,
        }
    }
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, MallocSizeOf)]
pub struct BoxShadowData {
    pub color: ColorF,
    pub blur_radius: f32,
    pub clip_mode: BoxShadowClipMode,
    pub inner_shadow_rect: LayoutRect,
    pub outer_shadow_rect: LayoutRect,
    pub shadow_radius: BorderRadius,
    pub element_rect: LayoutRect,
    pub element_radius: BorderRadius,
}

impl From<BoxShadow> for BoxShadowData {
    fn from(shadow: BoxShadow) -> Self {
        BoxShadowData {
            color: shadow.color.into(),
            blur_radius: shadow.blur_radius.to_f32_px(),
            clip_mode: shadow.clip_mode,
            inner_shadow_rect: shadow.inner_shadow_rect.into(),
            outer_shadow_rect: shadow.outer_shadow_rect.into(),
            shadow_radius: shadow.shadow_radius.into(),
            element_rect: shadow.element_rect.into(),
            element_radius: shadow.element_radius.into(),
        }
    }
}

pub type BoxShadowTemplate = PrimTemplate<BoxShadowData>;

impl Internable for BoxShadow {
    type Key = BoxShadowKey;
    type StoreData = BoxShadowTemplate;
    type InternData = ();
    const PROFILE_COUNTER: usize = crate::profiler::INTERNED_BOX_SHADOWS;
}

impl From<BoxShadowKey> for BoxShadowTemplate {
    fn from(shadow: BoxShadowKey) -> Self {
        BoxShadowTemplate {
            common: PrimTemplateCommonData::with_key_common(shadow.common),
            kind: shadow.kind.into(),
        }
    }
}

// The blur shader samples BLUR_SAMPLE_SCALE * blur_radius surrounding texels.
pub const BLUR_SAMPLE_SCALE: f32 = 3.0;

// Maximum blur radius for box-shadows (different than blur filters).
// Taken from nsCSSRendering.cpp in Gecko.
pub const MAX_BLUR_RADIUS: f32 = 300.;

// A cache key that uniquely identifies a minimally sized
// and blurred box-shadow rect that can be stored in the
// texture cache and applied to clip-masks.
#[derive(Debug, Clone, Eq, Hash, MallocSizeOf, PartialEq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct BoxShadowCacheKey {
    /// Blur sigma in device pixels at the mask resolution (≤ MAX_BLUR_STD_DEVIATION after Opt B).
    /// Stored as Au for sub-pixel precision; using i32 would round small sigmas to 0.
    pub blur_radius_dp: Au,
    pub clip_mode: BoxShadowClipMode,
    // NOTE(emilio): Only the original allocation size needs to be in the cache
    // key, since the actual size is derived from that.
    pub original_alloc_size: DeviceIntSize,
    pub br_top_left: DeviceIntSize,
    pub br_top_right: DeviceIntSize,
    pub br_bottom_right: DeviceIntSize,
    pub br_bottom_left: DeviceIntSize,
    pub device_pixel_scale: Au,
}

impl<'a> SceneBuilder<'a> {
    pub fn add_box_shadow(
        &mut self,
        spatial_node_index: SpatialNodeIndex,
        clip_node_id: ClipNodeId,
        prim_info: &LayoutPrimitiveInfo,
        box_offset: &LayoutVector2D,
        color: ColorF,
        mut blur_radius: f32,
        spread_radius: f32,
        border_radius: BorderRadius,
        shadow_radius: BorderRadius,
        clip_mode: BoxShadowClipMode,
    ) {
        if color.a == 0.0 {
            return;
        }

        // Inset shadows get smaller as spread radius increases.
        let spread_amount = match clip_mode {
            BoxShadowClipMode::Outset => spread_radius,
            BoxShadowClipMode::Inset => -spread_radius,
        };

        // Ensure the blur radius is somewhat sensible.
        blur_radius = f32::min(blur_radius, MAX_BLUR_RADIUS);

        // Apply parameters that affect where the shadow rect
        // exists in the local space of the primitive.
        let shadow_rect = prim_info
            .rect
            .translate(*box_offset)
            .inflate(spread_amount, spread_amount);

        // If blur radius is zero, we can use a fast path with
        // no blur applied.
        if blur_radius == 0.0 {
            // Trivial reject of box-shadows that are not visible.
            if box_offset.x == 0.0 && box_offset.y == 0.0 && spread_amount == 0.0 {
                return;
            }

            let mut clips = Vec::with_capacity(2);
            let (final_prim_rect, clip_radius) = match clip_mode {
                BoxShadowClipMode::Outset => {
                    if shadow_rect.is_empty() {
                        return;
                    }

                    // TODO(gw): Add a fast path for ClipOut + zero border radius!
                    clips.push(ClipItemEntry {
                        key: ClipItemKey {
                            kind: ClipItemKeyKind::rounded_rect(
                                border_radius,
                                ClipMode::ClipOut,
                            ),
                        },
                        spatial_node_index,
                        clip_rect: prim_info.rect,
                    });

                    (shadow_rect, shadow_radius)
                }
                BoxShadowClipMode::Inset => {
                    if !shadow_rect.is_empty() {
                        clips.push(ClipItemEntry {
                            key: ClipItemKey {
                                kind: ClipItemKeyKind::rounded_rect(
                                    shadow_radius,
                                    ClipMode::ClipOut,
                                ),
                            },
                            spatial_node_index,
                            clip_rect: shadow_rect,
                        });
                    }

                    (prim_info.rect, border_radius)
                }
            };

            clips.push(ClipItemEntry {
                key: ClipItemKey {
                    kind: ClipItemKeyKind::rounded_rect(
                        clip_radius,
                        ClipMode::Clip,
                    ),
                },
                spatial_node_index,
                clip_rect: final_prim_rect,
            });

            self.add_primitive(
                spatial_node_index,
                clip_node_id,
                &LayoutPrimitiveInfo::with_clip_rect(final_prim_rect, prim_info.clip_rect),
                clips,
                RectanglePrim {
                    color: PropertyBinding::Value(color.into()),
                },
            );
        } else {
            // Box-shadows with a valid blur radius use the quad primitive
            // path; element clipping is handled analytically in the shader.
            let blur_offset = (BLUR_SAMPLE_SCALE * blur_radius).ceil();

            // Get the local rect of where the shadow will be drawn,
            // expanded to include room for the blurred region.
            let dest_rect = shadow_rect.inflate(blur_offset, blur_offset);

            match clip_mode {
                BoxShadowClipMode::Outset => {
                    // Certain spread-radii make the shadow invalid.
                    if shadow_rect.is_empty() {
                        return;
                    }

                    // Element clip is handled analytically in the shader.
                    self.add_nonshadowable_primitive(
                        spatial_node_index,
                        clip_node_id,
                        &LayoutPrimitiveInfo::with_clip_rect(dest_rect, prim_info.clip_rect),
                        vec![],
                        BoxShadow {
                            color: color.into(),
                            blur_radius: Au::from_f32_px(blur_radius),
                            clip_mode,
                            inner_shadow_rect: shadow_rect.into(),
                            outer_shadow_rect: dest_rect.into(),
                            shadow_radius: shadow_radius.into(),
                            element_rect: prim_info.rect.into(),
                            element_radius: border_radius.into(),
                        },
                    );
                }
                BoxShadowClipMode::Inset => {
                    // If the inner shadow rect contains the prim
                    // rect, no pixels will be shadowed.
                    if border_radius.is_zero() && shadow_rect
                        .inflate(-blur_radius, -blur_radius)
                        .contains_box(&prim_info.rect)
                    {
                        return;
                    }

                    // Element clip is handled analytically in the shader.
                    self.add_nonshadowable_primitive(
                        spatial_node_index,
                        clip_node_id,
                        &prim_info.clone(),
                        vec![],
                        BoxShadow {
                            color: color.into(),
                            blur_radius: Au::from_f32_px(blur_radius),
                            clip_mode,
                            inner_shadow_rect: shadow_rect.into(),
                            outer_shadow_rect: dest_rect.into(),
                            shadow_radius: shadow_radius.into(),
                            element_rect: prim_info.rect.into(),
                            element_radius: border_radius.into(),
                        },
                    );
                }
            }
        }
    }
}
