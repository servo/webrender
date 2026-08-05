/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
use api::{BorderRadius, BoxShadowClipMode, ClipMode, ColorF};
use api::units::*;
use crate::border;
use crate::command_buffer::CommandBufferIndex;
use crate::clip::{ClipChainInstance, ClipNodeId};
use crate::frame_builder::{FrameBuildingContext, FrameBuildingState, PictureContext};
use crate::intern::{Handle as InternHandle, InternDebug, Internable};
use crate::prim_store::{InternablePrimitive, PrimKey, PrimTemplate, PrimTemplateCommonData, PrimitiveScratchBuffer};
use crate::prim_store::{PrimitiveKind, PrimitiveStore};
use crate::quad::{self, QuadDescriptor, QuadTransformState};
use crate::visibility::PrimitiveDrawIndex;
use crate::pattern::box_shadow::BoxShadowPatternData;
use crate::render_task::{RenderTask, RenderTaskKind, MAX_BLUR_STD_DEVIATION};
use crate::render_backend::DataStores;
use crate::render_task_cache::{RenderTaskCacheKey, RenderTaskCacheKeyKind, RenderTaskParent, to_cache_size};
use crate::render_target::RenderTargetKind;
use crate::gpu_types::BlurEdgeMode;
use crate::scene_building::{SceneBuilder, IsVisible};
use crate::space::SpaceSnapper;
use crate::spatial_tree::SpatialNodeIndex;
use crate::internal_types::LayoutPrimitiveInfo;
use crate::util::clamp_to_scale_factor;

pub type BoxShadowKey = PrimKey<BoxShadow>;

impl InternDebug for BoxShadowKey {}

// `BoxShadow` now lives in `webrender_api::interned_prims` so content-process
// interning can hold it. Re-exported to keep existing references working.
pub use api::interned_prims::BoxShadow;

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
        BoxShadowKey::new(info.into(), self)
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
    pub shadow_radius: BorderRadius,
    pub element_radius: BorderRadius,
    pub box_offset: LayoutVector2D,
    pub spread_amount: f32,
}

impl From<BoxShadow> for BoxShadowData {
    fn from(shadow: BoxShadow) -> Self {
        BoxShadowData {
            color: shadow.color.into(),
            blur_radius: shadow.blur_radius.to_f32_px(),
            clip_mode: shadow.clip_mode,
            shadow_radius: shadow.shadow_radius.into(),
            element_radius: shadow.element_radius.into(),
            box_offset: shadow.box_offset.into(),
            spread_amount: shadow.spread_amount.to_f32_px(),
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
    pub shape_top_left: u32,
    pub shape_top_right: u32,
    pub shape_bottom_left: u32,
    pub shape_bottom_right: u32,
    pub device_pixel_scale: Au,
    pub spread_amount: u32,
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

        // Box-shadows with a valid blur radius use the quad primitive path;
        // element clipping is handled analytically in the shader. Zero-blur
        // box-shadows are desugared into a rectangle plus shaping clips in the
        // DisplayListBuilder and never reach here.
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
                self.add_primitive(
                    spatial_node_index,
                    clip_node_id,
                    &LayoutPrimitiveInfo::with_clip_rect(dest_rect, prim_info.clip_rect),
                    BoxShadow {
                        color: color.into(),
                        blur_radius: Au::from_f32_px(blur_radius),
                        clip_mode,
                        shadow_radius: shadow_radius.into(),
                        element_radius: border_radius.into(),
                        box_offset: (*box_offset).into(),
                        spread_amount: Au::from_f32_px(spread_amount),
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
                self.add_primitive(
                    spatial_node_index,
                    clip_node_id,
                    &prim_info.clone(),
                    BoxShadow {
                        color: color.into(),
                        blur_radius: Au::from_f32_px(blur_radius),
                        clip_mode,
                        shadow_radius: shadow_radius.into(),
                        element_radius: border_radius.into(),
                        box_offset: (*box_offset).into(),
                        spread_amount: Au::from_f32_px(spread_amount),
                    },
                );
            }
        }
    }
}

pub fn prepare_box_shadow(
    shadow_data: &BoxShadowData,
    common_data: &PrimTemplateCommonData,
    unsnapped_pattern_rect: &LayoutRect,
    clip_chain: &ClipChainInstance,
    quad_transform: &mut QuadTransformState,
    frame_context: &FrameBuildingContext,
    pic_context: &PictureContext,
    frame_state: &mut FrameBuildingState,
    scratch: &mut PrimitiveScratchBuffer,
    prim_spatial_node_index: SpatialNodeIndex,
    device_pixel_scale: DevicePixelScale,
    draw_index: PrimitiveDrawIndex,
    cmd_buffer_targets: &[CommandBufferIndex],
    data_stores: &DataStores,
) {
    let blur_radius = shadow_data.blur_radius;
    // Build snapped element/inner/outer rects. The shader expects
    // `inner = element.translate(offset).inflate(spread)` and
    // `outer = inner.inflate(blur_offset)`, with element snapped to
    // the device pixel grid. Because the inflations can have
    // fractional components, snapping the prim's whole rect and
    // then deflating is not equivalent to snapping the element rect
    // directly, so we always snap the element rect itself and
    // re-inflate.
    //
    // The element rect's relation to the per-instance
    // `unsnapped_pattern_rect` differs by clip_mode (set up in
    // `box_shadow::add_box_shadow`):
    //   - Outset: prim rect = element.translate.inflate(spread)
    //                                  .inflate(blur_offset);
    //             recover element by reversing the construction.
    //   - Inset:  prim rect = element directly.
    let blur_offset = (BLUR_SAMPLE_SCALE * blur_radius).ceil();
    let unsnapped_element_rect = match shadow_data.clip_mode {
        BoxShadowClipMode::Outset => unsnapped_pattern_rect
            .inflate(-blur_offset, -blur_offset)
            .inflate(-shadow_data.spread_amount, -shadow_data.spread_amount)
            .translate(-shadow_data.box_offset),
        BoxShadowClipMode::Inset => *unsnapped_pattern_rect,
    };
    let element_rect = {
        // Snap into the prim's surface raster space, matching how the
        // prim's own rect was snapped in the visibility pass.
        let mut snapper = SpaceSnapper::new(
            &frame_state.surfaces[pic_context.surface_index.0],
            frame_context.spatial_tree,
        );
        snapper.set_target_spatial_node(prim_spatial_node_index, frame_context.spatial_tree);
        snapper.snap_rect(&unsnapped_element_rect)
    };
    let inner_shadow_rect = element_rect
        .translate(shadow_data.box_offset)
        .inflate(shadow_data.spread_amount, shadow_data.spread_amount);
    let outer_shadow_rect = inner_shadow_rect.inflate(blur_offset, blur_offset);
    // The shader-facing prim rect mirrors the (re-derived) outer for
    // Outset and the element for Inset — i.e. whichever rect the
    // scene-build path originally registered as `info.rect`. This is
    // what the rest of this block, plus `prepare_quad` below, expects
    // as the prim local-space rect.
    let prim_rect = match shadow_data.clip_mode {
        BoxShadowClipMode::Outset => outer_shadow_rect,
        BoxShadowClipMode::Inset => element_rect,
    };

    let shadow_rect_size = inner_shadow_rect.size();
    let mut shadow_radius = shadow_data.shadow_radius;
    border::ensure_no_corner_overlap(&mut shadow_radius, shadow_rect_size);

    let blur_region = (BLUR_SAMPLE_SCALE * blur_radius).ceil();

    let mut max_corner_width = shadow_radius.top_left.width
        .max(shadow_radius.bottom_left.width)
        .max(shadow_radius.top_right.width)
        .max(shadow_radius.bottom_right.width);
    let mut max_corner_height = shadow_radius.top_left.height
        .max(shadow_radius.bottom_left.height)
        .max(shadow_radius.top_right.height)
        .max(shadow_radius.bottom_right.height);

    if shadow_data.clip_mode == BoxShadowClipMode::Inset && !shadow_radius.shapes_all_round() {
        // Add one extra pixel to avoid stretching the antialiasing tail
        // in extreme cases (e.g corner-shape: notch with a blur radius of 1px).
        max_corner_width -= shadow_data.spread_amount - 1.0;
        max_corner_height -= shadow_data.spread_amount - 1.0;
    }

    let used_corner_width = max_corner_width.max(blur_region);
    let used_corner_height = max_corner_height.max(blur_region);

    let min_shadow_rect_size = LayoutSize::new(
        2.0 * used_corner_width + blur_region,
        2.0 * used_corner_height + blur_region,
    );

    // Compute the nine-patch source rect size per axis (= min_shadow_rect_size when
    // the shadow is large enough to stretch, = shadow_rect_size when corners overlap).
    let src_rect_size = LayoutSize::new(
        if shadow_rect_size.width >= min_shadow_rect_size.width {
            min_shadow_rect_size.width
        } else {
            shadow_rect_size.width
        },
        if shadow_rect_size.height >= min_shadow_rect_size.height {
            min_shadow_rect_size.height
        } else {
            shadow_rect_size.height
        },
    );

    // The full blur alloc size in local pixels. This is the UV denominator passed to
    // the shader: the nine-patch maps shadow_pos/alloc_size so that shadow_pos=blur_region
    // maps exactly to the shadow edge in the texture (preserving the blur falloff).
    let shadow_rect_alloc_size = LayoutSize::new(
        2.0 * blur_region + src_rect_size.width,
        2.0 * blur_region + src_rect_size.height,
    );

    // Scale to device pixels for the render task.
    let blur_radius_dp = blur_radius * 0.5;
    let mut content_scale = LayoutToWorldScale::new(1.0) * device_pixel_scale;
    content_scale.0 = clamp_to_scale_factor(content_scale.0, false);

    // Pre-reduce content_scale so the blur sigma is already within
    // MAX_BLUR_STD_DEVIATION, eliminating downscale passes inside new_blur.
    //
    // The sigma is kept at full sub-pixel precision (no integer rounding).
    // Rounding to an integer sigma quantized the cached mask, so an animated
    // blur transition stepped between a handful of discrete masks instead of
    // interpolating smoothly (bug 2001865). The mask cache key stores the
    // sigma as an Au (1/60px), so nearby static shadows still dedupe while
    // animations get a distinct mask per frame, matching other engines.
    let sigma = blur_radius_dp * content_scale.0;
    let n_downscales = if sigma > MAX_BLUR_STD_DEVIATION {
        (sigma / MAX_BLUR_STD_DEVIATION).log2().ceil() as u32
    } else {
        0
    };
    content_scale.0 /= (1u32 << n_downscales) as f32;

    // Safety cap: reduces content_scale further only for pathological
    // small-blur-huge-element cases where the alloc would exceed the max task size.
    let cache_size_rounded = to_cache_size(shadow_rect_alloc_size, &mut content_scale);

    // Device-space extent the blurred mask content actually occupies. The
    // shader maps nine-patch UV=1.0 to this true content edge rather than to
    // the integer atlas allocation edge; otherwise the sub-texel difference
    // between `content_device_size` and its rounded allocation shifts the
    // nine-patch registration as the blur animates, making the shadow edge
    // jitter / step (bug 2002194 residual). Round the allocation UP so the
    // content always fits inside it (mapping UV=1.0 to a point outside the
    // allocation would sample a neighbouring atlas entry).
    let content_device_size = shadow_rect_alloc_size * content_scale;
    let cache_size = DeviceIntSize::new(
        cache_size_rounded.width.max(content_device_size.width.ceil() as i32),
        cache_size_rounded.height.max(content_device_size.height.ceil() as i32),
    );

    // Blur sigma to pass to new_blur, at the final mask resolution (after both
    // the n_downscales reduction and any to_cache_size safety cap).
    let blur_std_dev = blur_radius_dp * content_scale.0;
    debug_assert!(
        blur_std_dev <= MAX_BLUR_STD_DEVIATION + 1e-3,
        "BoxShadow sigma {blur_std_dev} exceeds MAX_BLUR_STD_DEVIATION \
            (n_downscales={n_downscales}, content_scale={})",
        content_scale.0,
    );

    // We only need the spread amount (-inset) in the cache key when using other
    // corner shapes than 'round'
    let cache_key_spread_amount = if !shadow_radius.shapes_all_round() {
        shadow_data.spread_amount.to_bits()
    } else {
        0
    };

    let bs_cache_key = BoxShadowCacheKey {
        blur_radius_dp: Au::from_f32_px(blur_std_dev),
        clip_mode: shadow_data.clip_mode,
        original_alloc_size: DeviceIntSize::new(
            (shadow_rect_alloc_size * content_scale).round().width as i32,
            (shadow_rect_alloc_size * content_scale).round().height as i32,
        ),
        br_top_left: DeviceIntSize::new(
            (shadow_radius.top_left * content_scale).round().width as i32,
            (shadow_radius.top_left * content_scale).round().height as i32,
        ),
        br_top_right: DeviceIntSize::new(
            (shadow_radius.top_right * content_scale).round().width as i32,
            (shadow_radius.top_right * content_scale).round().height as i32,
        ),
        br_bottom_right: DeviceIntSize::new(
            (shadow_radius.bottom_right * content_scale).round().width as i32,
            (shadow_radius.bottom_right * content_scale).round().height as i32,
        ),
        br_bottom_left: DeviceIntSize::new(
            (shadow_radius.bottom_left * content_scale).round().width as i32,
            (shadow_radius.bottom_left * content_scale).round().height as i32,
        ),
        shape_top_left: shadow_radius.shape_top_left.to_bits(),
        shape_top_right: shadow_radius.shape_top_right.to_bits(),
        shape_bottom_right: shadow_radius.shape_bottom_right.to_bits(),
        shape_bottom_left: shadow_radius.shape_bottom_left.to_bits(),
        device_pixel_scale: Au::from_f32_px(content_scale.0),
        spread_amount: cache_key_spread_amount,
    };

    // The shadow shape is offset by blur_region within the alloc task (local pixels).
    // device_pixel_scale_for_task scales it to the mask resolution.
    let minimal_shadow_rect_origin = LayoutPoint::new(blur_region, blur_region);
    let minimal_shadow_rect = LayoutRect::from_origin_and_size(
        minimal_shadow_rect_origin,
        src_rect_size,
    );
    let device_pixel_scale_for_task = DevicePixelScale::new(content_scale.0);

    let shadow_inset = LayoutSideOffsets::new_all_same(-shadow_data.spread_amount);

    let task_id = frame_state.resource_cache.request_render_task(
        Some(RenderTaskCacheKey {
            origin: DeviceIntPoint::zero(),
            size: cache_size,
            kind: RenderTaskCacheKeyKind::BoxShadow(bs_cache_key),
        }),
        false,
        RenderTaskParent::Surface,
        &mut frame_state.frame_gpu_data.f32,
        frame_state.rg_builder,
        &mut frame_state.surface_builder,
        &mut |rg_builder, _| {
            let mask_task_id = rg_builder.add().init(RenderTask::new_dynamic(
                cache_size,
                RenderTaskKind::new_rounded_rect_mask(
                    minimal_shadow_rect,
                    shadow_radius,
                    shadow_inset,
                    ClipMode::Clip,
                    device_pixel_scale_for_task,
                ),
            ));

            RenderTask::new_blur(
                DeviceSize::new(blur_std_dev, blur_std_dev),
                mask_task_id,
                rg_builder,
                RenderTargetKind::Alpha,
                None,
                cache_size,
                BlurEdgeMode::Duplicate,
            )
        }
    );

    // For outset, prim_rect == dest_rect so offset is zero.
    // For inset, prim_rect is the element rect; dest_rect (outer_shadow_rect)
    // may be offset and smaller, so we pass its size and offset separately.
    let dest_rect = outer_shadow_rect;
    let dest_rect_offset = LayoutVector2D::new(
        dest_rect.min.x - prim_rect.min.x,
        dest_rect.min.y - prim_rect.min.y,
    );
    let dest_rect_size = dest_rect.size();

    let mut element_radius = shadow_data.element_radius;
    border::ensure_no_corner_overlap(&mut element_radius, element_rect.size());
    let element_offset_rel_prim = LayoutVector2D::new(
        element_rect.min.x - prim_rect.min.x,
        element_rect.min.y - prim_rect.min.y,
    );

    let pattern = BoxShadowPatternData {
        color: shadow_data.color,
        render_task: task_id,
        shadow_rect_alloc_size,
        content_device_size,
        dest_rect_size,
        dest_rect_offset,
        clip_mode: shadow_data.clip_mode,
        element_offset_rel_prim,
        element_size: element_rect.size(),
        element_radius,
    };

    quad::prepare_quad(
        &pattern,
        &QuadDescriptor {
            pattern_rect: prim_rect,
            // `prim_rect` is re-derived here by snapping the element rect and
            // re-inflating, so it differs from the prim rect the clip chain was
            // built with; `clip_chain.local_coverage_rect` does not apply.
            bounds: clip_chain.local_clip_rect.intersection_unchecked(&prim_rect),
            aligned_aa_edges: common_data.aligned_aa_edges,
            transformed_aa_edges: common_data.transformed_aa_edges,
        },
        draw_index,
        &None,
        clip_chain,
        quad_transform,
        frame_context,
        pic_context,
        cmd_buffer_targets,
        &data_stores.clip,
        frame_state,
        scratch,
    );

}
