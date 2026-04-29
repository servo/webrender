/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! # Prepare pass
//!
//! TODO: document this!

use api::{ColorF, DebugFlags};
use api::ClipMode;
use crate::util::clamp_to_scale_factor;
use crate::box_shadow::{BoxShadowCacheKey, BLUR_SAMPLE_SCALE};
use crate::pattern::box_shadow::BoxShadowPatternData;
use api::units::*;
use euclid::Scale;
use smallvec::SmallVec;
use crate::composite::CompositorSurfaceKind;
use crate::command_buffer::{CommandBufferIndex, PrimitiveCommand};
use crate::border;
use crate::clip::{ClipStore, ClipNodeRange};
use crate::render_task_graph::RenderTaskId;
use crate::renderer::{GpuBufferAddress, GpuBufferWriterF};
use crate::spatial_tree::SpatialNodeIndex;
use crate::clip::{clamped_radius, ClipNodeFlags, ClipChainInstance, ClipItemKind};
use crate::frame_builder::{FrameBuildingContext, FrameBuildingState, PictureContext, PictureState};
use crate::gpu_types::{BrushFlags, BlurEdgeMode};
use crate::render_target::RenderTargetKind;
use crate::internal_types::{FastHashMap, PlaneSplitAnchor, Filter};
use crate::picture::{ClusterFlags, PictureCompositeMode, PicturePrimitive};
use crate::picture::{PrimitiveList, PrimitiveCluster, SurfaceIndex, SubpixelMode, Picture3DContext};
use crate::tile_cache::{SliceId, TileCacheInstance};
use crate::prim_store::*;
use crate::prim_store::borders::NormalBorderScratch;
use crate::prim_store::line_dec::LineDecorationScratch;
use crate::quad::{self, QuadTransformState};
use crate::render_backend::DataStores;
use crate::render_task_cache::RenderTaskCacheKeyKind;
use crate::render_task_cache::{RenderTaskCacheKey, to_cache_size, RenderTaskParent};
use crate::render_task::{EmptyTask, RenderTask, RenderTaskKind, MAX_BLUR_STD_DEVIATION};
use crate::segment::SegmentBuilder;
use crate::visibility::VisibilityState;


const MAX_MASK_SIZE: i32 = 4096;

const MIN_BRUSH_SPLIT_AREA: f32 = 128.0 * 128.0;

/// The entry point of the preapre pass.
pub fn prepare_picture(
    pic_index: PictureIndex,
    store: &mut PrimitiveStore,
    surface_index: Option<SurfaceIndex>,
    subpixel_mode: SubpixelMode,
    frame_context: &FrameBuildingContext,
    frame_state: &mut FrameBuildingState,
    data_stores: &mut DataStores,
    scratch: &mut PrimitiveScratchBuffer,
    tile_caches: &mut FastHashMap<SliceId, Box<TileCacheInstance>>,
    prim_instances: &mut Vec<PrimitiveInstance>,
) -> bool {
    if frame_state.visited_pictures[pic_index.0] {
        return true;
    }

    frame_state.visited_pictures[pic_index.0] = true;

    let pic = &mut store.pictures[pic_index.0];
    let Some((pic_context, mut pic_state, mut prim_list)) = pic.take_context(
        pic_index,
        surface_index,
        subpixel_mode,
        frame_state,
        frame_context,
        data_stores,
        scratch,
        tile_caches,
    ) else {
        return false;
    };

    prepare_primitives(
        store,
        &mut prim_list,
        &pic_context,
        &mut pic_state,
        frame_context,
        frame_state,
        data_stores,
        scratch,
        tile_caches,
        prim_instances,
    );

    // Restore the dependencies (borrow check dance)
    store.pictures[pic_context.pic_index.0].restore_context(
        pic_context.pic_index,
        prim_list,
        pic_context,
        prim_instances,
        frame_context,
        frame_state,
    );

    true
}

fn prepare_primitives(
    store: &mut PrimitiveStore,
    prim_list: &mut PrimitiveList,
    pic_context: &PictureContext,
    pic_state: &mut PictureState,
    frame_context: &FrameBuildingContext,
    frame_state: &mut FrameBuildingState,
    data_stores: &mut DataStores,
    scratch: &mut PrimitiveScratchBuffer,
    tile_caches: &mut FastHashMap<SliceId, Box<TileCacheInstance>>,
    prim_instances: &mut Vec<PrimitiveInstance>,
) {
    profile_scope!("prepare_primitives");
    let mut cmd_buffer_targets = Vec::new();

    let mut quad_transform = QuadTransformState::new();

    for cluster in &mut prim_list.clusters {
        if !cluster.flags.contains(ClusterFlags::IS_VISIBLE) {
            continue;
        }
        profile_scope!("cluster");
        pic_state.map_local_to_pic.set_target_spatial_node(
            cluster.spatial_node_index,
            frame_context.spatial_tree,
        );

        let device_pixel_scale = frame_state.surfaces[pic_context.surface_index.0].device_pixel_scale;
        quad_transform.set(
            cluster.spatial_node_index,
            pic_context.raster_spatial_node_index,
            frame_context.spatial_tree,
            device_pixel_scale,
        );

        for prim_instance_index in cluster.prim_range() {
            if frame_state.surface_builder.get_cmd_buffer_targets_for_prim(
                &prim_instances[prim_instance_index].vis,
                &mut cmd_buffer_targets,
            ) {
                let plane_split_anchor = PlaneSplitAnchor::new(
                    cluster.spatial_node_index,
                    PrimitiveInstanceIndex(prim_instance_index as u32),
                );

                prepare_prim_for_render(
                    store,
                    prim_instance_index,
                    cluster,
                    &mut quad_transform,
                    pic_context,
                    pic_state,
                    frame_context,
                    frame_state,
                    plane_split_anchor,
                    data_stores,
                    scratch,
                    tile_caches,
                    prim_instances,
                    &cmd_buffer_targets,
                );

                frame_state.num_visible_primitives += 1;
                continue;
            }

            // TODO(gw): Technically no need to clear visibility here, since from this point it
            //           only matters if it got added to a command buffer. Kept here for now to
            //           make debugging simpler, but perhaps we can remove / tidy this up.
            prim_instances[prim_instance_index].clear_visibility();
        }
    }
}

fn can_use_clip_chain_for_quad_path(
    clip_chain: &ClipChainInstance,
    clip_store: &ClipStore,
    data_stores: &DataStores,
) -> bool {
    if !clip_chain.needs_mask {
        return true;
    }

    for i in 0 .. clip_chain.clips_range.count {
        let clip_instance = clip_store.get_instance_from_range(&clip_chain.clips_range, i);
        let clip_node = &data_stores.clip[clip_instance.handle];

        match clip_node.item.kind {
            ClipItemKind::RoundedRectangle { .. } | ClipItemKind::Rectangle { .. } => {}
            ClipItemKind::Image { .. } => {
                panic!("bug: image-masks not expected on rect/quads");
            }
        }
    }

    true
}

fn prepare_prim_for_render(
    store: &mut PrimitiveStore,
    prim_instance_index: usize,
    cluster: &mut PrimitiveCluster,
    quad_transform: &mut QuadTransformState,
    pic_context: &PictureContext,
    pic_state: &mut PictureState,
    frame_context: &FrameBuildingContext,
    frame_state: &mut FrameBuildingState,
    plane_split_anchor: PlaneSplitAnchor,
    data_stores: &mut DataStores,
    scratch: &mut PrimitiveScratchBuffer,
    tile_caches: &mut FastHashMap<SliceId, Box<TileCacheInstance>>,
    prim_instances: &mut Vec<PrimitiveInstance>,
    targets: &[CommandBufferIndex],
) {
    profile_scope!("prepare_prim_for_render");

    // If we have dependencies, we need to prepare them first, in order
    // to know the actual rect of this primitive.
    // For example, scrolling may affect the location of an item in
    // local space, which may force us to render this item on a larger
    // picture target, if being composited.
    let mut is_passthrough = false;
    if let PrimitiveInstanceKind::Picture { pic_index, .. } = prim_instances[prim_instance_index].kind {
        if !prepare_picture(
            pic_index,
            store,
            Some(pic_context.surface_index),
            pic_context.subpixel_mode,
            frame_context,
            frame_state,
            data_stores,
            scratch,
            tile_caches,
            prim_instances
        ) {
            return;
        }

        is_passthrough = store
            .pictures[pic_index.0]
            .composite_mode
            .is_none();
    }

    let prim_instance = &mut prim_instances[prim_instance_index];
    let mut use_legacy_path = true;
    if !is_passthrough {
        match &prim_instance.kind {
            PrimitiveInstanceKind::Rectangle { .. }
            | PrimitiveInstanceKind::RadialGradient { .. }
            | PrimitiveInstanceKind::ConicGradient { .. }
            | PrimitiveInstanceKind::LinearGradient { .. }
            => {
                use_legacy_path = false;
            }
            PrimitiveInstanceKind::Image { data_handle, .. } => {
                use_legacy_path = !crate::prim_store::image::can_use_quad_shaders(
                    &data_stores.image[*data_handle].kind,
                    frame_state.resource_cache,
                );
            }
            _ => {}
        };

        // In this initial patch, we only support non-masked primitives through the new
        // quad rendering path. Follow up patches will extend this to support masks, and
        // then use by other primitives. In the new quad rendering path, we'll still want
        // to skip the entry point to `update_clip_task` as that does old-style segmenting
        // and mask generation.
        let should_update_clip_task = match &mut prim_instance.kind {
            PrimitiveInstanceKind::Rectangle { .. }
            | PrimitiveInstanceKind::Image { .. }
            | PrimitiveInstanceKind::RadialGradient { .. }
            | PrimitiveInstanceKind::ConicGradient { .. }
            | PrimitiveInstanceKind::LinearGradient { .. }
            => {
                use_legacy_path |= !can_use_clip_chain_for_quad_path(
                    &prim_instance.vis.clip_chain,
                    frame_state.clip_store,
                    data_stores,
                );

                use_legacy_path
            }
            PrimitiveInstanceKind::BoxShadow { .. } |
            PrimitiveInstanceKind::Picture { .. } => false,
            _ => true,
        };

        if should_update_clip_task {
            let prim_rect = data_stores.get_local_prim_rect(
                prim_instance,
                &store.pictures,
                frame_state.surfaces,
            );

            if !update_clip_task(
                prim_instance,
                &prim_rect.min,
                cluster.spatial_node_index,
                pic_context.raster_spatial_node_index,
                pic_context.visibility_spatial_node_index,
                pic_context,
                pic_state,
                frame_context,
                frame_state,
                store,
                data_stores,
                scratch,
            ) {
                return;
            }
        }
    }

    prepare_interned_prim_for_render(
        store,
        use_legacy_path,
        PrimitiveInstanceIndex(prim_instance_index as u32),
        prim_instance,
        cluster,
        plane_split_anchor,
        quad_transform,
        pic_context,
        pic_state,
        frame_context,
        frame_state,
        data_stores,
        scratch,
        targets,
    )
}

/// Prepare an interned primitive for rendering, by requesting
/// resources, render tasks etc. This is equivalent to the
/// prepare_prim_for_render_inner call for old style primitives.
fn prepare_interned_prim_for_render(
    store: &mut PrimitiveStore,
    use_legacy_path: bool,
    prim_instance_index: PrimitiveInstanceIndex,
    prim_instance: &mut PrimitiveInstance,
    cluster: &mut PrimitiveCluster,
    plane_split_anchor: PlaneSplitAnchor,
    quad_transform: &mut QuadTransformState,
    pic_context: &PictureContext,
    pic_state: &mut PictureState,
    frame_context: &FrameBuildingContext,
    frame_state: &mut FrameBuildingState,
    data_stores: &mut DataStores,
    scratch: &mut PrimitiveScratchBuffer,
    targets: &[CommandBufferIndex],
) {
    let prim_spatial_node_index = cluster.spatial_node_index;
    let device_pixel_scale = frame_state.surfaces[pic_context.surface_index.0].device_pixel_scale;

    match &mut prim_instance.kind {
        PrimitiveInstanceKind::BoxShadow { data_handle, .. } => {
            profile_scope!("BoxShadow");

            let prim_data = &data_stores.box_shadow[*data_handle];
            let shadow_data = &prim_data.kind;
            let blur_radius = shadow_data.blur_radius;

            let shadow_rect_size = shadow_data.inner_shadow_rect.size();
            let mut shadow_radius = shadow_data.shadow_radius;
            border::ensure_no_corner_overlap(&mut shadow_radius, shadow_rect_size);

            let blur_region = (BLUR_SAMPLE_SCALE * blur_radius).ceil();

            let max_corner_width = shadow_radius.top_left.width
                .max(shadow_radius.bottom_left.width)
                .max(shadow_radius.top_right.width)
                .max(shadow_radius.bottom_right.width);
            let max_corner_height = shadow_radius.top_left.height
                .max(shadow_radius.bottom_left.height)
                .max(shadow_radius.top_right.height)
                .max(shadow_radius.bottom_right.height);

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

            // Opt B: pre-reduce content_scale so the blur sigma is already within
            // MAX_BLUR_STD_DEVIATION, eliminating downscale passes inside new_blur.
            //
            // Use the same rounding as the old code (round to nearest integer) to determine
            // n_downscales, so mask scale exactly matches what old new_blur downscaling would
            // have produced. Exception: if rounded sigma is 0 (tiny sigma from to_cache_size
            // downscaling), use the float sigma to avoid a zero-blur regression.
            let sigma_rounded = (blur_radius_dp * content_scale.0).round();
            let sigma_for_n = if sigma_rounded == 0.0 { blur_radius_dp * content_scale.0 } else { sigma_rounded };
            let n_downscales = if sigma_for_n > MAX_BLUR_STD_DEVIATION {
                (sigma_for_n / MAX_BLUR_STD_DEVIATION).log2().ceil() as u32
            } else {
                0
            };
            content_scale.0 /= (1u32 << n_downscales) as f32;

            // Safety cap: reduces content_scale further only for pathological
            // small-blur-huge-element cases where the alloc would exceed the max task size.
            let cache_size = to_cache_size(shadow_rect_alloc_size, &mut content_scale);

            // Blur sigma to pass to new_blur. Use the same rounded value as the old code
            // (now divided by 2^n instead of being halved inside new_blur), so the blur
            // intensity is byte-for-byte identical to the old pipeline.
            let blur_std_dev = if sigma_rounded == 0.0 {
                blur_radius_dp * content_scale.0
            } else {
                sigma_rounded / (1u32 << n_downscales) as f32
            };
            debug_assert!(
                blur_std_dev <= MAX_BLUR_STD_DEVIATION + 1e-3,
                "BoxShadow sigma {blur_std_dev} exceeds MAX_BLUR_STD_DEVIATION after Opt B \
                 (n_downscales={n_downscales}, content_scale={})",
                content_scale.0,
            );

            let bs_cache_key = BoxShadowCacheKey {
                blur_radius_dp: Au::from_f32_px(blur_std_dev),
                clip_mode: shadow_data.clip_mode,
                original_alloc_size: (shadow_rect_alloc_size * content_scale).round().to_i32(),
                br_top_left: (shadow_radius.top_left * content_scale).round().to_i32(),
                br_top_right: (shadow_radius.top_right * content_scale).round().to_i32(),
                br_bottom_right: (shadow_radius.bottom_right * content_scale).round().to_i32(),
                br_bottom_left: (shadow_radius.bottom_left * content_scale).round().to_i32(),
                device_pixel_scale: Au::from_f32_px(content_scale.0),
            };

            let clip_data = ClipData::rounded_rect(
                src_rect_size,
                &shadow_radius,
                ClipMode::Clip,
            );

            // The shadow shape is offset by blur_region within the alloc task (local pixels).
            // device_pixel_scale_for_task scales it to the mask resolution.
            let minimal_shadow_rect_origin = LayoutPoint::new(blur_region, blur_region);
            let device_pixel_scale_for_task = DevicePixelScale::new(content_scale.0);

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
                            minimal_shadow_rect_origin,
                            clip_data.clone(),
                            device_pixel_scale_for_task,
                            frame_context.fb_config,
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

            let prim_rect = LayoutRect::from_origin_and_size(
                prim_instance.prim_origin,
                prim_data.common.prim_size,
            );

            // For outset, prim_rect == dest_rect so offset is zero.
            // For inset, prim_rect is the element rect; dest_rect (outer_shadow_rect)
            // may be offset and smaller, so we pass its size and offset separately.
            let dest_rect: LayoutRect = shadow_data.outer_shadow_rect.into();
            let dest_rect_offset = LayoutVector2D::new(
                dest_rect.min.x - prim_rect.min.x,
                dest_rect.min.y - prim_rect.min.y,
            );
            let dest_rect_size = dest_rect.size();

            let element_rect: LayoutRect = shadow_data.element_rect.into();
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
                dest_rect_size,
                dest_rect_offset,
                clip_mode: shadow_data.clip_mode,
                element_offset_rel_prim,
                element_size: element_rect.size(),
                element_radius,
            };

            quad::prepare_quad(
                &pattern,
                &prim_rect,
                prim_data.common.aligned_aa_edges,
                prim_data.common.transformed_aa_edges,
                prim_instance_index,
                &None,
                &prim_instance.vis.clip_chain,
                quad_transform,
                frame_context,
                pic_context,
                targets,
                &data_stores.clip,
                frame_state,
                scratch,
            );

            return;
        }
        PrimitiveInstanceKind::LineDecoration { data_handle, ref mut scratch_handle } => {
            profile_scope!("LineDecoration");
            let prim_data = &mut data_stores.line_decoration[*data_handle];
            let common_data = &mut prim_data.common;
            let line_dec_data = &mut prim_data.kind;

            line_dec_data.update(common_data, frame_state);

            let render_task = line_dec_data.prepare_render_task(
                prim_spatial_node_index,
                frame_context,
                frame_state,
            );
            *scratch_handle = scratch.frame.line_decoration.push(LineDecorationScratch { task_id: render_task });
        }
        PrimitiveInstanceKind::TextRun { run_index, data_handle, .. } => {
            profile_scope!("TextRun");
            let prim_data = &mut data_stores.text_run[*data_handle];
            let run = &mut store.text_runs[*run_index];

            prim_data.common.may_need_repetition = false;

            // The glyph transform has to match `glyph_transform` in "ps_text_run" shader.
            // It's relative to the rasterizing space of a glyph.
            let transform = frame_context.spatial_tree
                .get_relative_transform(
                    prim_spatial_node_index,
                    pic_context.raster_spatial_node_index,
                )
                .into_fast_transform();
            let prim_offset = prim_instance.prim_origin.to_vector();

            let surface = &frame_state.surfaces[pic_context.surface_index.0];

            // If subpixel AA is disabled due to the backing surface the glyphs
            // are being drawn onto, disable it (unless we are using the
            // specifial subpixel mode that estimates background color).
            let allow_subpixel = match prim_instance.vis.state {
                VisibilityState::Culled |
                VisibilityState::Unset |
                VisibilityState::PassThrough => {
                    panic!("bug: invalid visibility state");
                }
                VisibilityState::Visible { sub_slice_index, .. } => {
                    // For now, we only allow subpixel AA on primary sub-slices. In future we
                    // may support other sub-slices if we find content that does this.
                    if sub_slice_index.is_primary() {
                        match pic_context.subpixel_mode {
                            SubpixelMode::Allow => true,
                            SubpixelMode::Deny => false,
                            SubpixelMode::Conditional { allowed_rect, prohibited_rect } => {
                                // Conditional mode allows subpixel AA to be enabled for this
                                // text run, so long as it's inside the allowed rect.
                                allowed_rect.contains_box(&prim_instance.vis.clip_chain.pic_coverage_rect) &&
                                !prohibited_rect.intersects(&prim_instance.vis.clip_chain.pic_coverage_rect)
                            }
                        }
                    } else {
                        false
                    }
                }
            };

            run.request_resources(
                prim_offset,
                &prim_data.font,
                &prim_data.glyphs,
                &transform.to_transform().with_destination::<_>(),
                surface,
                prim_spatial_node_index,
                allow_subpixel,
                frame_context.fb_config.low_quality_pinch_zoom,
                frame_state.resource_cache,
                &mut frame_state.frame_gpu_data.f32,
                frame_context.spatial_tree,
                scratch,
            );

            prim_data.update(frame_state);
        }
        PrimitiveInstanceKind::NormalBorder { data_handle, ref mut scratch_handle } => {
            profile_scope!("NormalBorder");
            let prim_data = &mut data_stores.normal_border[*data_handle];
            let common_data = &mut prim_data.common;
            let border_data = &mut prim_data.kind;

            border_data.write_brush_gpu_blocks(common_data, frame_state);

            let segment_count = border_data.border_segments.len();
            let task_ids = scratch.frame.border_task_ids.extend(
                std::iter::repeat(RenderTaskId::INVALID).take(segment_count),
            );
            let handle = scratch.frame.normal_border.push(NormalBorderScratch { task_ids });

            border_data.update(
                common_data,
                prim_spatial_node_index,
                device_pixel_scale,
                frame_context,
                frame_state,
                &mut scratch.frame.border_task_ids[task_ids],
            );

            *scratch_handle = handle;
        }
        PrimitiveInstanceKind::ImageBorder { data_handle, .. } => {
            profile_scope!("ImageBorder");
            let prim_data = &mut data_stores.image_border[*data_handle];

            // TODO: get access to the ninepatch and to check whether we need support
            // for repetitions in the shader.

            // Update the template this instance references, which may refresh the GPU
            // cache with any shared template data.
            prim_data.kind.update(
                &mut prim_data.common,
                frame_state
            );
        }
        PrimitiveInstanceKind::Rectangle { data_handle, segment_instance_index, .. } => {
            profile_scope!("Rectangle");

            if use_legacy_path {
                let prim_data = &mut data_stores.prim[*data_handle];
                prim_data.common.may_need_repetition = false;

                // Update the template this instane references, which may refresh the GPU
                // cache with any shared template data.
                prim_data.update(
                    frame_state,
                    frame_context.scene_properties,
                );

                write_segment(
                    *segment_instance_index,
                    frame_state,
                    &mut scratch.scene.segments,
                    &mut scratch.scene.segment_instances,
                    |request| {
                        request.push_one(frame_context.scene_properties.resolve_color(&prim_data.kind.color).premultiplied());
                    }
                );
            } else {
                let prim_data = &data_stores.prim[*data_handle];
                let prim_rect = LayoutRect::from_origin_and_size(prim_instance.prim_origin, prim_data.common.prim_size);
                let color = prim_data.resolve(frame_context.scene_properties);

                quad::prepare_quad(
                    &color,
                    &prim_rect,
                    prim_data.common.aligned_aa_edges,
                    prim_data.common.transformed_aa_edges,
                    prim_instance_index,
                    &None,
                    &prim_instance.vis.clip_chain,
                    quad_transform,
                    frame_context,
                    pic_context,
                    targets,
                    &data_stores.clip,
                    frame_state,
                    scratch,
                );

                return;
            }
        }
        PrimitiveInstanceKind::YuvImage { data_handle, segment_instance_index, compositor_surface_kind, .. } => {
            profile_scope!("YuvImage");
            let prim_data = &mut data_stores.yuv_image[*data_handle];
            let common_data = &mut prim_data.common;
            let yuv_image_data = &mut prim_data.kind;

            common_data.may_need_repetition = false;

            // Update the template this instane references, which may refresh the GPU
            // cache with any shared template data.
            yuv_image_data.update(
                common_data,
                compositor_surface_kind.is_composited(),
                frame_state,
            );

            write_segment(
                *segment_instance_index,
                frame_state,
                &mut scratch.scene.segments,
                &mut scratch.scene.segment_instances,
                |writer| {
                    yuv_image_data.write_prim_gpu_blocks(writer);
                }
            );
        }
        PrimitiveInstanceKind::Image { data_handle, image_instance_index, .. } => {
            profile_scope!("Image");

            let prim_data = &mut data_stores.image[*data_handle];
            let common_data = &mut prim_data.common;
            let image_data = &mut prim_data.kind;
            let image_instance = &mut store.images[*image_instance_index];

            if !use_legacy_path {
                let prim_rect = LayoutRect::from_origin_and_size(
                    prim_instance.prim_origin,
                    common_data.prim_size,
                );

                crate::prim_store::image::prepare_image_quads(
                    &prim_rect,
                    common_data,
                    image_data,
                    &prim_instance.vis.clip_chain,
                    prim_instance_index,
                    quad_transform,
                    frame_context,
                    pic_context,
                    targets,
                    &data_stores.clip,
                    frame_state,
                    scratch,
                );

                return;
            }

            // Update the template this instance references, which may refresh the GPU
            // cache with any shared template data.
            image_data.update(
                common_data,
                image_instance,
                prim_spatial_node_index,
                frame_state,
                frame_context,
                &mut prim_instance.vis,
                prim_instance.prim_origin,
            );

            write_segment(
                image_instance.segment_instance_index,
                frame_state,
                &mut scratch.scene.segments,
                &mut scratch.scene.segment_instances,
                |request| {
                    image_data.write_prim_gpu_blocks(&image_instance.adjustment, request);
                },
            );
        }
        PrimitiveInstanceKind::LinearGradient { data_handle, .. } => {
            profile_scope!("LinearGradient");
            let prim_data = &mut data_stores.linear_grad[*data_handle];
            let prim_rect = LayoutRect::from_origin_and_size(prim_instance.prim_origin, prim_data.common.prim_size);

            if let Some(nine_patch) = &prim_data.border_nine_patch {
                quad::prepare_border_image_nine_patch(
                    &*nine_patch,
                    prim_data,
                    &prim_rect,
                    prim_data.stretch_size,
                    prim_data.common.aligned_aa_edges,
                    prim_data.common.transformed_aa_edges,
                    prim_instance_index,
                    &prim_instance.vis.clip_chain,
                    quad_transform,
                    frame_context,
                    pic_context,
                    targets,
                    &data_stores.clip,
                    frame_state,
                    scratch,
                );
                return;
            }

            // For SWGL, evaluating the gradient is faster than reading from the texture cache.
            let mut should_cache = !frame_context.fb_config.is_software
                && frame_state.resource_cache.texture_cache.allocated_color_bytes() < 10_000_000;
            if should_cache {
                let surface = &frame_state.surfaces[pic_context.surface_index.0];
                let clipped_surface_rect = surface.get_surface_rect(
                    &prim_instance.vis.clip_chain.pic_coverage_rect,
                    frame_context.spatial_tree,
                );

                should_cache = if let Some(rect) = clipped_surface_rect {
                    rect.width() < 512 && rect.height() < 512
                } else {
                    false
                };
            }

            let cache_key = if should_cache {
                quad::cache_key(
                    data_handle.uid(),
                    quad_transform,
                    &prim_instance.vis.clip_chain,
                    frame_state.clip_store,
                )
            } else {
                None
            };

            let local_rect = LayoutRect::from_origin_and_size(prim_instance.prim_origin, prim_data.common.prim_size);
            quad::prepare_repeatable_quad(
                prim_data,
                &local_rect,
                prim_data.stretch_size,
                prim_data.tile_spacing,
                prim_data.common.aligned_aa_edges,
                prim_data.common.transformed_aa_edges,
                prim_instance_index,
                &cache_key,
                &prim_instance.vis.clip_chain,
                quad_transform,
                frame_context,
                pic_context,
                targets,
                &data_stores.clip,
                frame_state,
                scratch,
            );

            return;
        }
        PrimitiveInstanceKind::RadialGradient { data_handle, .. } => {
            profile_scope!("RadialGradient");
            let prim_data = &mut data_stores.radial_grad[*data_handle];
            let local_rect = LayoutRect::from_origin_and_size(prim_instance.prim_origin, prim_data.common.prim_size);

            if let Some(nine_patch) = &prim_data.border_nine_patch {
                quad::prepare_border_image_nine_patch(
                    &*nine_patch,
                    prim_data,
                    &local_rect,
                    prim_data.stretch_size,
                    prim_data.common.aligned_aa_edges,
                    prim_data.common.transformed_aa_edges,
                    prim_instance_index,
                    &prim_instance.vis.clip_chain,
                    quad_transform,
                    frame_context,
                    pic_context,
                    targets,
                    &data_stores.clip,
                    frame_state,
                    scratch,
                );
                return;
            }

            quad::prepare_repeatable_quad(
                prim_data,
                &local_rect,
                prim_data.stretch_size,
                prim_data.tile_spacing,
                prim_data.common.aligned_aa_edges,
                prim_data.common.transformed_aa_edges,
                prim_instance_index,
                &None,
                &prim_instance.vis.clip_chain,
                quad_transform,
                frame_context,
                pic_context,
                targets,
                &data_stores.clip,
                frame_state,
                scratch,
            );
            return;
        }
        PrimitiveInstanceKind::ConicGradient { data_handle, .. } => {
            profile_scope!("ConicGradient");
            let prim_data = &mut data_stores.conic_grad[*data_handle];
            let prim_rect = LayoutRect::from_origin_and_size(prim_instance.prim_origin, prim_data.common.prim_size);

            if let Some(nine_patch) = &prim_data.border_nine_patch {
                quad::prepare_border_image_nine_patch(
                    &*nine_patch,
                    prim_data,
                    &prim_rect,
                    prim_data.stretch_size,
                    prim_data.common.aligned_aa_edges,
                    prim_data.common.transformed_aa_edges,
                    prim_instance_index,
                    &prim_instance.vis.clip_chain,
                    quad_transform,
                    frame_context,
                    pic_context,
                    targets,
                    &data_stores.clip,
                    frame_state,
                    scratch,
                );
                return;
            }

            // Conic gradients are quite slow with SWGL, so we want to cache
            // them as much as we can, even large ones.
            // TODO: get_surface_rect is not always cheap. We should reorganize
            // the code so that we only call it as much as we really need it,
            // while avoiding this much boilerplate for each primitive that uses
            // caching.
            let mut should_cache = frame_context.fb_config.is_software
                && frame_state.resource_cache.texture_cache.allocated_color_bytes() < 30_000_000;
            if should_cache {
                let surface = &frame_state.surfaces[pic_context.surface_index.0];
                let clipped_surface_rect = surface.get_surface_rect(
                    &prim_instance.vis.clip_chain.pic_coverage_rect,
                    frame_context.spatial_tree,
                );

                should_cache = if let Some(rect) = clipped_surface_rect {
                    rect.width() < 4096 && rect.height() < 4096
                } else {
                    false
                };
            }

            let cache_key = if should_cache {
                quad::cache_key(
                    data_handle.uid(),
                    quad_transform,
                    &prim_instance.vis.clip_chain,
                    frame_state.clip_store,
                )
            } else {
                None
            };

            let local_rect = LayoutRect::from_origin_and_size(prim_instance.prim_origin, prim_data.common.prim_size);
            quad::prepare_repeatable_quad(
                prim_data,
                &local_rect,
                prim_data.stretch_size,
                prim_data.tile_spacing,
                prim_data.common.aligned_aa_edges,
                prim_data.common.transformed_aa_edges,
                prim_instance_index,
                &cache_key,
                &prim_instance.vis.clip_chain,
                quad_transform,
                frame_context,
                pic_context,
                targets,
                &data_stores.clip,
                frame_state,
                scratch,
            );
            return;
        }
        PrimitiveInstanceKind::Picture { pic_index, .. } => {
            profile_scope!("Picture");
            let pic = &mut store.pictures[pic_index.0];

            if prim_instance.vis.clip_chain.needs_mask {
                // TODO(gw): Much of the code in this branch could be moved in to a common
                //           function as we move more primitives to the new clip-mask paths.

                // We are going to split the clip mask tasks in to a list to be rendered
                // on the source picture, and those to be rendered in to a mask for
                // compositing the picture in to the target.
                let mut source_masks = Vec::new();
                let mut target_masks = Vec::new();

                // For some composite modes, we force target mask due to limitations. That
                // might results in artifacts for these modes (which are already an existing
                // problem) but we can handle these cases as follow ups.
                let force_target_mask = match pic.composite_mode {
                    // We can't currently render over top of these filters as their size
                    // may have changed due to downscaling. We could handle this separate
                    // case as a follow up.
                    Some(PictureCompositeMode::Filter(Filter::Blur { .. })) |
                    Some(PictureCompositeMode::Filter(Filter::DropShadows { .. })) |
                    Some(PictureCompositeMode::SVGFEGraph( .. )) => {
                        true
                    }
                    _ => {
                        false
                    }
                };

                // Work out which clips get drawn in to the source / target mask
                for i in 0 .. prim_instance.vis.clip_chain.clips_range.count {
                    let clip_instance = frame_state.clip_store.get_instance_from_range(&prim_instance.vis.clip_chain.clips_range, i);

                    if !force_target_mask && clip_instance.flags.contains(ClipNodeFlags::SAME_COORD_SYSTEM) {
                        source_masks.push(i);
                    } else {
                        target_masks.push(i);
                    }
                }

                let pic_surface_index = pic.raster_config.as_ref().unwrap().surface_index;
                let prim_local_rect: LayoutRect = frame_state
                    .surfaces[pic_surface_index.0]
                    .clipped_local_rect
                    .cast_unit();

                // Handle masks on the source. This is the common case, and occurs for:
                // (a) Any masks in the same coord space as the surface
                // (b) All masks if the surface and parent are axis-aligned
                if !source_masks.is_empty() {
                    let first_clip_node_index = frame_state.clip_store.clip_node_instances.len() as u32;
                    let parent_task_id = pic.primary_render_task_id.expect("bug: no composite mode");

                    // Construct a new clip node range, also add image-mask dependencies as needed
                    for instance in source_masks {
                        let clip_instance = frame_state.clip_store.get_instance_from_range(&prim_instance.vis.clip_chain.clips_range, instance);

                        for tile in frame_state.clip_store.visible_mask_tiles(clip_instance) {
                            frame_state.rg_builder.add_dependency(
                                parent_task_id,
                                tile.task_id,
                            );
                        }

                        frame_state.clip_store.clip_node_instances.push(clip_instance.clone());
                    }

                    let clip_node_range = ClipNodeRange {
                        first: first_clip_node_index,
                        count: frame_state.clip_store.clip_node_instances.len() as u32 - first_clip_node_index,
                    };

                    // Add the mask as a sub-pass of the picture
                    let pic_task_id = pic.primary_render_task_id.expect("uh oh");
                    let pic_task = frame_state.rg_builder.get_task_mut(pic_task_id);

                    let RenderTaskKind::Picture(info) = &pic_task.kind else { unreachable!() };

                    let task_rect = DeviceRect::from_origin_and_size(
                        info.content_origin,
                        pic_task.get_target_size().to_f32(),
                    );

                    quad::prepare_clip_range(
                        clip_node_range,
                        pic_task_id,
                        &task_rect,
                        &prim_local_rect,
                        prim_spatial_node_index,
                        info.raster_spatial_node_index,
                        info.device_pixel_scale,
                        &data_stores.clip,
                        frame_state.clip_store,
                        frame_context.spatial_tree,
                        frame_state.rg_builder,
                        &mut frame_state.frame_gpu_data.f32,
                        frame_state.transforms,
                    );
                }

                // Handle masks on the target. This is the rare case, and occurs for:
                // Masks in parent space when non-axis-aligned to source space
                if !target_masks.is_empty() {
                    let surface = &frame_state.surfaces[pic_context.surface_index.0];
                    let coverage_rect = prim_instance.vis.clip_chain.pic_coverage_rect;

                    let device_pixel_scale = surface.device_pixel_scale;
                    let raster_spatial_node_index = surface.raster_spatial_node_index;

                    let Some(clipped_surface_rect) = surface.get_surface_rect(
                        &coverage_rect,
                        frame_context.spatial_tree,
                    ) else {
                        return;
                    };

                    // Draw a normal screens-space mask to an alpha target that
                    // can be sampled when compositing this picture.
                    let empty_task = EmptyTask {
                        content_origin: clipped_surface_rect.min.to_f32(),
                        device_pixel_scale,
                        raster_spatial_node_index,
                    };

                    let task_size = clipped_surface_rect.size();

                    let clip_task_id = frame_state.rg_builder.add().init(RenderTask::new_dynamic(
                        task_size,
                        RenderTaskKind::Empty(empty_task),
                    ));

                    // Construct a new clip node range, also add image-mask dependencies as needed
                    let first_clip_node_index = frame_state.clip_store.clip_node_instances.len() as u32;
                    for instance in target_masks {
                        let clip_instance = frame_state.clip_store.get_instance_from_range(&prim_instance.vis.clip_chain.clips_range, instance);

                        for tile in frame_state.clip_store.visible_mask_tiles(clip_instance) {
                            frame_state.rg_builder.add_dependency(
                                clip_task_id,
                                tile.task_id,
                            );
                        }

                        frame_state.clip_store.clip_node_instances.push(clip_instance.clone());
                    }

                    let clip_node_range = ClipNodeRange {
                        first: first_clip_node_index,
                        count: frame_state.clip_store.clip_node_instances.len() as u32 - first_clip_node_index,
                    };

                    let task_rect = clipped_surface_rect.to_f32();

                    quad::prepare_clip_range(
                        clip_node_range,
                        clip_task_id,
                        &task_rect,
                        &prim_local_rect,
                        prim_spatial_node_index,
                        raster_spatial_node_index,
                        device_pixel_scale,
                        &data_stores.clip,
                        frame_state.clip_store,
                        frame_context.spatial_tree,
                        frame_state.rg_builder,
                        &mut frame_state.frame_gpu_data.f32,
                        frame_state.transforms,
                    );

                    let clip_task_index = ClipTaskIndex(scratch.frame.clip_mask_instances.len() as _);
                    scratch.frame.clip_mask_instances.push(ClipMaskKind::Mask(clip_task_id));
                    prim_instance.vis.clip_task_index = clip_task_index;
                    frame_state.surface_builder.add_child_render_task(
                        clip_task_id,
                        frame_state.rg_builder,
                    );
                }
            }

            pic.write_gpu_blocks(
                frame_state,
                data_stores,
            );

            if let Picture3DContext::In { root_data: None, plane_splitter_index, .. } = pic.context_3d {
                let dirty_rect = frame_state.current_dirty_region().combined;
                let visibility_node = frame_state.current_dirty_region().visibility_spatial_node;
                let splitter = &mut frame_state.plane_splitters[plane_splitter_index.0];
                let surface_index = pic.raster_config.as_ref().unwrap().surface_index;
                let surface = &frame_state.surfaces[surface_index.0];
                let local_prim_rect = surface.clipped_local_rect.cast_unit();

                PicturePrimitive::add_split_plane(
                    splitter,
                    frame_context.spatial_tree,
                    prim_spatial_node_index,
                    visibility_node,
                    local_prim_rect,
                    &prim_instance.vis.clip_chain.local_clip_rect,
                    dirty_rect,
                    plane_split_anchor,
                );
            }
        }
        PrimitiveInstanceKind::BackdropCapture { .. } => {
            // Register the owner picture of this backdrop primitive as the
            // target for resolve of the sub-graph
            frame_state.surface_builder.register_resolve_source();

            if frame_context.debug_flags.contains(DebugFlags::HIGHLIGHT_BACKDROP_FILTERS) {
                if let Some(world_rect) = pic_state.map_pic_to_vis.map(&prim_instance.vis.clip_chain.pic_coverage_rect) {
                    scratch.push_debug_rect(
                        world_rect.cast_unit(),
                        2,
                        crate::debug_colors::MAGENTA,
                        ColorF::TRANSPARENT,
                    );
                }
            }
        }
        PrimitiveInstanceKind::BackdropRender { pic_index, .. } => {
            match frame_state.surface_builder.sub_graph_output_map.get(pic_index).cloned() {
                Some(sub_graph_output_id) => {
                    frame_state.surface_builder.add_child_render_task(
                        sub_graph_output_id,
                        frame_state.rg_builder,
                    );
                }
                None => {
                    // Backdrop capture was found not visible, didn't produce a sub-graph
                    // so we can just skip drawing
                    prim_instance.clear_visibility();
                }
            }
        }
    }

    match prim_instance.vis.state {
        VisibilityState::Unset => {
            panic!("bug: invalid vis state");
        }
        VisibilityState::Visible { .. } => {
            frame_state.push_prim(
                &PrimitiveCommand::simple(prim_instance_index),
                prim_spatial_node_index,
                targets,
            );
        }
        VisibilityState::PassThrough | VisibilityState::Culled => {}
    }
}


fn write_segment<F>(
    segment_instance_index: SegmentInstanceIndex,
    frame_state: &mut FrameBuildingState,
    segments: &mut SegmentStorage,
    segment_instances: &mut SegmentInstanceStorage,
    f: F,
) where F: Fn(&mut GpuBufferWriterF) {
    debug_assert_ne!(segment_instance_index, SegmentInstanceIndex::INVALID);
    if segment_instance_index != SegmentInstanceIndex::UNUSED {
        let segment_instance = &mut segment_instances[segment_instance_index];

        let segments = &segments[segment_instance.segments_range];
        let mut writer = frame_state.frame_gpu_data.f32.write_blocks(3 + segments.len() * VECS_PER_SEGMENT);

        f(&mut writer);

        for segment in segments {
            segment.write_gpu_blocks(&mut writer);
        }

        segment_instance.gpu_data = writer.finish();
    }
}

fn update_clip_task_for_brush(
    instance: &PrimitiveInstance,
    prim_origin: &LayoutPoint,
    prim_spatial_node_index: SpatialNodeIndex,
    root_spatial_node_index: SpatialNodeIndex,
    visibility_spatial_node_index: SpatialNodeIndex,
    pic_context: &PictureContext,
    pic_state: &mut PictureState,
    frame_context: &FrameBuildingContext,
    frame_state: &mut FrameBuildingState,
    prim_store: &PrimitiveStore,
    data_stores: &mut DataStores,
    segments_store: &mut SegmentStorage,
    segment_instances_store: &mut SegmentInstanceStorage,
    clip_mask_instances: &mut Vec<ClipMaskKind>,
    device_pixel_scale: DevicePixelScale,
) -> Option<ClipTaskIndex> {
    let segments = match instance.kind {
        PrimitiveInstanceKind::BoxShadow { .. } => {
            unreachable!("BUG: box-shadows should not hit legacy brush clip path");
        }
        PrimitiveInstanceKind::Picture { .. } |
        PrimitiveInstanceKind::TextRun { .. } |
        PrimitiveInstanceKind::LineDecoration { .. } |
        PrimitiveInstanceKind::BackdropCapture { .. } |
        PrimitiveInstanceKind::BackdropRender { .. } => {
            return None;
        }
        PrimitiveInstanceKind::Image { image_instance_index, .. } => {
            let segment_instance_index = prim_store
                .images[image_instance_index]
                .segment_instance_index;

            if segment_instance_index == SegmentInstanceIndex::UNUSED {
                return None;
            }

            let segment_instance = &segment_instances_store[segment_instance_index];

            &segments_store[segment_instance.segments_range]
        }
        PrimitiveInstanceKind::YuvImage { segment_instance_index, .. } => {
            debug_assert!(segment_instance_index != SegmentInstanceIndex::INVALID);

            if segment_instance_index == SegmentInstanceIndex::UNUSED {
                return None;
            }

            let segment_instance = &segment_instances_store[segment_instance_index];

            &segments_store[segment_instance.segments_range]
        }
        PrimitiveInstanceKind::Rectangle { segment_instance_index, .. } => {
            debug_assert!(segment_instance_index != SegmentInstanceIndex::INVALID);

            if segment_instance_index == SegmentInstanceIndex::UNUSED {
                return None;
            }

            let segment_instance = &segment_instances_store[segment_instance_index];

            &segments_store[segment_instance.segments_range]
        }
        PrimitiveInstanceKind::ImageBorder { data_handle, .. } => {
            let border_data = &data_stores.image_border[data_handle].kind;

            // TODO: This is quite messy - once we remove legacy primitives we
            //       can change this to be a tuple match on (instance, template)
            border_data.brush_segments.as_slice()
        }
        PrimitiveInstanceKind::NormalBorder { data_handle, .. } => {
            let border_data = &data_stores.normal_border[data_handle].kind;

            // TODO: This is quite messy - once we remove legacy primitives we
            //       can change this to be a tuple match on (instance, template)
            border_data.brush_segments.as_slice()
        }
        PrimitiveInstanceKind::LinearGradient { data_handle, .. } => {
            let prim_data = &data_stores.linear_grad[data_handle];

            // TODO: This is quite messy - once we remove legacy primitives we
            //       can change this to be a tuple match on (instance, template)
            if prim_data.brush_segments.is_empty() {
                return None;
            }

            prim_data.brush_segments.as_slice()
        }
        PrimitiveInstanceKind::RadialGradient { .. } => {
            unreachable!("BUG: radial gradients should always use quad path");
        }
        PrimitiveInstanceKind::ConicGradient { .. } => {
            unreachable!("BUG: conic gradients should always use quad path");
        }
    };

    // If there are no segments, early out to avoid setting a valid
    // clip task instance location below.
    if segments.is_empty() {
        return None;
    }

    // Set where in the clip mask instances array the clip mask info
    // can be found for this primitive. Each segment will push the
    // clip mask information for itself in update_clip_task below.
    let clip_task_index = ClipTaskIndex(clip_mask_instances.len() as _);

    // If we only built 1 segment, there is no point in re-running
    // the clip chain builder. Instead, just use the clip chain
    // instance that was built for the main primitive. This is a
    // significant optimization for the common case.
    if segments.len() == 1 {
        let clip_mask_kind = update_brush_segment_clip_task(
            &segments[0],
            Some(&instance.vis.clip_chain),
            root_spatial_node_index,
            pic_context.surface_index,
            frame_context,
            frame_state,
            device_pixel_scale,
        );
        clip_mask_instances.push(clip_mask_kind);
    } else {
        let dirty_rect = frame_state.current_dirty_region().combined;

        for segment in segments {
            // Build a clip chain for the smaller segment rect. This will
            // often manage to eliminate most/all clips, and sometimes
            // clip the segment completely.
            frame_state.clip_store.set_active_clips_from_clip_chain(
                &instance.vis.clip_chain,
                prim_spatial_node_index,
                visibility_spatial_node_index,
                &frame_context.spatial_tree,
            );

            let segment_clip_chain = frame_state
                .clip_store
                .build_clip_chain_instance(
                    segment.local_rect.translate(prim_origin.to_vector()),
                    &pic_state.map_local_to_pic,
                    &pic_state.map_pic_to_vis,
                    &frame_context.spatial_tree,
                    &mut frame_state.frame_gpu_data.f32,
                    frame_state.resource_cache,
                    &dirty_rect,
                    &mut data_stores.clip,
                    frame_state.rg_builder,
                    false,
                );

            let clip_mask_kind = update_brush_segment_clip_task(
                &segment,
                segment_clip_chain.as_ref(),
                root_spatial_node_index,
                pic_context.surface_index,
                frame_context,
                frame_state,
                device_pixel_scale,
            );
            clip_mask_instances.push(clip_mask_kind);
        }
    }

    Some(clip_task_index)
}

pub fn update_clip_task(
    instance: &mut PrimitiveInstance,
    prim_origin: &LayoutPoint,
    prim_spatial_node_index: SpatialNodeIndex,
    root_spatial_node_index: SpatialNodeIndex,
    visibility_spatial_node_index: SpatialNodeIndex,
    pic_context: &PictureContext,
    pic_state: &mut PictureState,
    frame_context: &FrameBuildingContext,
    frame_state: &mut FrameBuildingState,
    prim_store: &mut PrimitiveStore,
    data_stores: &mut DataStores,
    scratch: &mut PrimitiveScratchBuffer,
) -> bool {
    let device_pixel_scale = frame_state.surfaces[pic_context.surface_index.0].device_pixel_scale;

    build_segments_if_needed(
        instance,
        frame_state,
        prim_store,
        data_stores,
        &mut scratch.scene.segments,
        &mut scratch.scene.segment_instances,
    );

    // First try to  render this primitive's mask using optimized brush rendering.
    instance.vis.clip_task_index = if let Some(clip_task_index) = update_clip_task_for_brush(
        instance,
        prim_origin,
        prim_spatial_node_index,
        root_spatial_node_index,
        visibility_spatial_node_index,
        pic_context,
        pic_state,
        frame_context,
        frame_state,
        prim_store,
        data_stores,
        &mut scratch.scene.segments,
        &mut scratch.scene.segment_instances,
        &mut scratch.frame.clip_mask_instances,
        device_pixel_scale,
    ) {
        clip_task_index
    } else if instance.vis.clip_chain.needs_mask {
        // Get a minimal device space rect, clipped to the screen that we
        // need to allocate for the clip mask, as well as interpolated
        // snap offsets.
        let unadjusted_device_rect = match frame_state.surfaces[pic_context.surface_index.0].get_surface_rect(
            &instance.vis.clip_chain.pic_coverage_rect,
            frame_context.spatial_tree,
        ) {
            Some(rect) => rect,
            None => return false,
        };

        let (device_rect, device_pixel_scale) = adjust_mask_scale_for_max_size(
            unadjusted_device_rect,
            device_pixel_scale,
        );

        if device_rect.size().to_i32().is_empty() {
            log::warn!("Bad adjusted clip task size {:?} (was {:?})", device_rect.size(), unadjusted_device_rect.size());
            return false;
        }

        let clip_task_id = RenderTaskKind::new_mask(
            device_rect,
            instance.vis.clip_chain.clips_range,
            root_spatial_node_index,
            frame_state.rg_builder,
            device_pixel_scale,
            frame_context.fb_config,
        );
        // Set the global clip mask instance for this primitive.
        let clip_task_index = ClipTaskIndex(scratch.frame.clip_mask_instances.len() as _);
        scratch.frame.clip_mask_instances.push(ClipMaskKind::Mask(clip_task_id));
        instance.vis.clip_task_index = clip_task_index;
        frame_state.surface_builder.add_child_render_task(
            clip_task_id,
            frame_state.rg_builder,
        );
        clip_task_index
    } else {
        ClipTaskIndex::INVALID
    };

    true
}

/// Write out to the clip mask instances array the correct clip mask
/// config for this segment.
pub fn update_brush_segment_clip_task(
    segment: &BrushSegment,
    clip_chain: Option<&ClipChainInstance>,
    root_spatial_node_index: SpatialNodeIndex,
    surface_index: SurfaceIndex,
    frame_context: &FrameBuildingContext,
    frame_state: &mut FrameBuildingState,
    device_pixel_scale: DevicePixelScale,
) -> ClipMaskKind {
    let clip_chain = match clip_chain {
        Some(chain) => chain,
        None => return ClipMaskKind::Clipped,
    };
    if !clip_chain.needs_mask ||
       (!segment.may_need_clip_mask && !clip_chain.has_non_local_clips) {
        return ClipMaskKind::None;
    }

    let unadjusted_device_rect = match frame_state.surfaces[surface_index.0].get_surface_rect(
        &clip_chain.pic_coverage_rect,
        frame_context.spatial_tree,
    ) {
        Some(rect) => rect,
        None => return ClipMaskKind::Clipped,
    };

    let (device_rect, device_pixel_scale) = adjust_mask_scale_for_max_size(unadjusted_device_rect, device_pixel_scale);

    if device_rect.size().to_i32().is_empty() {
        log::warn!("Bad adjusted mask size {:?} (was {:?})", device_rect.size(), unadjusted_device_rect.size());
        return ClipMaskKind::Clipped;
    }

    let clip_task_id = RenderTaskKind::new_mask(
        device_rect,
        clip_chain.clips_range,
        root_spatial_node_index,
        frame_state.rg_builder,
        device_pixel_scale,
        frame_context.fb_config,
    );

    frame_state.surface_builder.add_child_render_task(
        clip_task_id,
        frame_state.rg_builder,
    );
    ClipMaskKind::Mask(clip_task_id)
}


fn write_brush_segment_description(
    prim_local_rect: LayoutRect,
    prim_local_clip_rect: LayoutRect,
    clip_chain: &ClipChainInstance,
    segment_builder: &mut SegmentBuilder,
    clip_store: &ClipStore,
    data_stores: &DataStores,
) -> bool {
    // If the brush is small, we want to skip building segments
    // and just draw it as a single primitive with clip mask.
    if prim_local_rect.area() < MIN_BRUSH_SPLIT_AREA {
        return false;
    }

    // NOTE: The local clip rect passed to the segment builder must be the unmodified
    //       local clip rect from the clip leaf, not the local_clip_rect from the
    //       clip-chain instance. The clip-chain instance may have been reduced by
    //       clips that are in the same coordinate system, but not the same spatial
    //       node as the primitive. This can result in the clip for the segment building
    //       being affected by scrolling clips, which we can't handle (since the segments
    //       are not invalidated during frame building after being built).
    segment_builder.initialize(
        prim_local_rect,
        None,
        prim_local_clip_rect,
    );

    // Segment the primitive on all the local-space clip sources that we can.
    for i in 0 .. clip_chain.clips_range.count {
        let clip_instance = clip_store
            .get_instance_from_range(&clip_chain.clips_range, i);
        let clip_node = &data_stores.clip[clip_instance.handle];

        // If this clip item is positioned by another positioning node, its relative position
        // could change during scrolling. This means that we would need to resegment. Instead
        // of doing that, only segment with clips that have the same positioning node.
        // TODO(mrobinson, #2858): It may make sense to include these nodes, resegmenting only
        // when necessary while scrolling.
        if !clip_instance.flags.contains(ClipNodeFlags::SAME_SPATIAL_NODE) {
            continue;
        }

        let (local_clip_rect, radius, mode) = match clip_node.item.kind {
            ClipItemKind::RoundedRectangle { radius, mode } => {
                let radius = clamped_radius(&radius, clip_instance.clip_rect.size());
                (clip_instance.clip_rect, Some(radius), mode)
            }
            ClipItemKind::Rectangle { mode } => {
                (clip_instance.clip_rect, None, mode)
            }
            ClipItemKind::Image { .. } => {
                panic!("bug: masks not supported on old segment path");
            }
        };

        segment_builder.push_clip_rect(local_clip_rect, radius, mode);
    }

    true
}

fn build_segments_if_needed(
    instance: &mut PrimitiveInstance,
    frame_state: &mut FrameBuildingState,
    prim_store: &mut PrimitiveStore,
    data_stores: &DataStores,
    segments_store: &mut SegmentStorage,
    segment_instances_store: &mut SegmentInstanceStorage,
) {
    let prim_clip_chain = &instance.vis.clip_chain;

    // Usually, the primitive rect can be found from information
    // in the instance and primitive template.
    let prim_local_rect = data_stores.get_local_prim_rect(
        instance,
        &prim_store.pictures,
        frame_state.surfaces,
    );

    let segment_instance_index = match instance.kind {
        PrimitiveInstanceKind::Rectangle { ref mut segment_instance_index, .. } => {
            segment_instance_index
        }
        PrimitiveInstanceKind::YuvImage { ref mut segment_instance_index, compositor_surface_kind, .. } => {
            // Only use segments for YUV images if not drawing as a compositor surface
            if !compositor_surface_kind.supports_segments() {
                *segment_instance_index = SegmentInstanceIndex::UNUSED;
                return;
            }

            segment_instance_index
        }
        PrimitiveInstanceKind::Image { data_handle, image_instance_index, compositor_surface_kind, .. } => {
            let image_data = &data_stores.image[data_handle].kind;
            let image_instance = &mut prim_store.images[image_instance_index];

            //Note: tiled images don't support automatic segmentation,
            // they strictly produce one segment per visible tile instead.
            if !compositor_surface_kind.supports_segments() ||
                frame_state.resource_cache
                    .get_image_properties(image_data.key)
                    .and_then(|properties| properties.tiling)
                    .is_some()
            {
                image_instance.segment_instance_index = SegmentInstanceIndex::UNUSED;
                return;
            }
            &mut image_instance.segment_instance_index
        }
        PrimitiveInstanceKind::Picture { .. } |
        PrimitiveInstanceKind::TextRun { .. } |
        PrimitiveInstanceKind::NormalBorder { .. } |
        PrimitiveInstanceKind::ImageBorder { .. } |
        PrimitiveInstanceKind::LinearGradient { .. } |
        PrimitiveInstanceKind::RadialGradient { .. } |
        PrimitiveInstanceKind::ConicGradient { .. } |
        PrimitiveInstanceKind::LineDecoration { .. } |
        PrimitiveInstanceKind::BackdropCapture { .. } |
        PrimitiveInstanceKind::BackdropRender { .. } => {
            // These primitives don't support / need segments.
            return;
        }
        PrimitiveInstanceKind::BoxShadow { .. } => {
            unreachable!("BUG: box-shadows should not hit legacy brush clip path");
        }
    };

    if *segment_instance_index == SegmentInstanceIndex::INVALID {
        let mut segments: SmallVec<[BrushSegment; 8]> = SmallVec::new();
        let clip_leaf = frame_state.clip_tree.get_leaf(instance.clip_leaf_id);

        if write_brush_segment_description(
            prim_local_rect,
            clip_leaf.local_clip_rect,
            prim_clip_chain,
            &mut frame_state.segment_builder,
            frame_state.clip_store,
            data_stores,
        ) {
            frame_state.segment_builder.build(|segment| {
                segments.push(
                    BrushSegment::new(
                        segment.rect.translate(-prim_local_rect.min.to_vector()),
                        segment.has_mask,
                        segment.edge_flags,
                        [0.0; 4],
                        BrushFlags::PERSPECTIVE_INTERPOLATION,
                    ),
                );
            });
        }

        // If only a single segment is produced, there is no benefit to writing
        // a segment instance array. Instead, just use the main primitive rect
        // written into the GPU cache.
        // TODO(gw): This is (sortof) a bandaid - due to a limitation in the current
        //           brush encoding, we can only support a total of up to 2^16 segments.
        //           This should be (more than) enough for any real world case, so for
        //           now we can handle this by skipping cases where we were generating
        //           segments where there is no benefit. The long term / robust fix
        //           for this is to move the segment building to be done as a more
        //           limited nine-patch system during scene building, removing arbitrary
        //           segmentation during frame-building (see bug #1617491).
        if segments.len() <= 1 {
            *segment_instance_index = SegmentInstanceIndex::UNUSED;
        } else {
            let segments_range = segments_store.extend(segments);

            let instance = SegmentedInstance {
                segments_range,
                gpu_data: GpuBufferAddress::INVALID,
            };

            *segment_instance_index = segment_instances_store.push(instance);
        };
    }
}

// Ensures that the size of mask render tasks are within MAX_MASK_SIZE.
fn adjust_mask_scale_for_max_size(device_rect: DeviceIntRect, device_pixel_scale: DevicePixelScale) -> (DeviceIntRect, DevicePixelScale) {
    if device_rect.width() > MAX_MASK_SIZE || device_rect.height() > MAX_MASK_SIZE {
        // round_out will grow by 1 integer pixel if origin is on a
        // fractional position, so keep that margin for error with -1:
        let device_rect_f = device_rect.to_f32();
        let scale = (MAX_MASK_SIZE - 1) as f32 /
            f32::max(device_rect_f.width(), device_rect_f.height());
        let new_device_pixel_scale = device_pixel_scale * Scale::new(scale);
        let new_device_rect = (device_rect_f * Scale::new(scale))
            .round_out()
            .to_i32();
        (new_device_rect, new_device_pixel_scale)
    } else {
        (device_rect, device_pixel_scale)
    }
}

impl CompositorSurfaceKind {
    /// Returns true if the compositor surface strategy supports segment rendering
    fn supports_segments(&self) -> bool {
        match self {
            CompositorSurfaceKind::Underlay | CompositorSurfaceKind::Overlay => false,
            CompositorSurfaceKind::Blit => true,
        }
    }
}
