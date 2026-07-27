/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! # Prepare pass
//!
//! TODO: document this!

use api::{ColorF, DebugFlags, ExtendMode, ExternalImageData, ExternalImageType, GradientStop, ImageBufferKind};
use crate::border_image::prepare_border_image_nine_patch;
use crate::pattern::cutout::Cutout;
use crate::render_task_graph::RenderTaskId;
use crate::util::ScaleOffset;
use crate::util::MaxRect;
use crate::box_shadow::prepare_box_shadow;

use crate::pattern::gradient::linear_gradient_pattern;
use crate::pattern::{Pattern, PatternBuilder, PatternBuilderContext, PatternBuilderState};
use crate::prim_store::gradient::{decompose_axis_aligned_gradient, linear_gradient_decomposes};
use crate::segment::EdgeMask;
use api::units::*;
use euclid::Scale;
use crate::composite::CompositorSurfaceKind;
use crate::command_buffer::{CommandBufferIndex, PrimitiveCommand};

use crate::clip::ClipNodeRange;
use crate::pattern::image::{ImagePattern, ShadowPattern};
use crate::pattern::filter::BlendFilterPattern;
use crate::pattern::yuv::YuvPattern;
use crate::pattern::backdrop::BackdropPattern;
use crate::pattern::mix_blend::{FixedFunctionMixBlendPattern, MixBlendPattern};
use crate::picture::{calculate_screen_uv, prepare_picture_clips};
use crate::space::SpaceMapper;
use crate::renderer::{BlendMode, GpuBufferAddress};
use crate::spatial_tree::SpatialNodeIndex;
use crate::frame_builder::{FrameBuildingContext, FrameBuildingState, PictureContext, PictureState};
use crate::gpu_types::UvRectKind;

use crate::internal_types::{FastHashMap, PlaneSplitAnchor, Filter};
use crate::picture::{ClusterFlags, PictureCompositeMode, PictureInstance, PictureScratch};
use crate::picture::{PrimitiveList, PrimitiveCluster, SurfaceIndex, SubpixelMode, Picture3DContext};
use crate::tile_cache::{SliceId, TileCacheInstance};
use crate::prim_store::*;
use crate::quad::{self, QuadTransformState};
use crate::render_backend::DataStores;


use crate::render_task::{EmptyTask, RenderTask, RenderTaskKind};

use crate::visibility::{DrawState, KindScratchHandle};


const MAX_MASK_SIZE: i32 = 4096;

/// The entry point of the preapre pass.
pub fn prepare_picture(
    pic_index: PictureIndex,
    store: &mut PrimitiveStore,
    surface_index: Option<SurfaceIndex>,
    subpixel_mode: SubpixelMode,
    frame_context: &FrameBuildingContext,
    frame_state: &mut FrameBuildingState,
    data_stores: &DataStores,
    scratch: &mut PrimitiveScratchBuffer,
    tile_caches: &mut FastHashMap<SliceId, Box<TileCacheInstance>>,
    prim_instances: &mut Vec<PrimitiveInstance>,
) -> Option<storage::Index<PictureScratch>> {
    // Only successfully-prepared pictures are cached here, so a cache hit is
    // always a valid scratch handle. Pictures that yielded no scratch are not
    // memoized: take_context's only None path (invisible picture) is a cheap,
    // side-effect-free query that is stable within a frame, so re-consulting it
    // on a repeat visit is fine and avoids caching an invalid handle.
    if let Some(handle) = frame_state.picture_scratch_handles[pic_index.0] {
        return Some(handle);
    }

    let pic = &mut store.pictures[pic_index.0];
    let Some((pic_context, mut pic_state, mut prim_list, scratch_handle)) = pic.take_context(
        pic_index,
        surface_index,
        subpixel_mode,
        frame_state,
        frame_context,
        data_stores,
        scratch,
        tile_caches,
    ) else {
        return None;
    };

    frame_state.picture_scratch_handles[pic_index.0] = Some(scratch_handle);

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
        frame_context,
        frame_state,
        scratch,
    );

    Some(scratch_handle)
}

fn prepare_primitives(
    store: &mut PrimitiveStore,
    prim_list: &mut PrimitiveList,
    pic_context: &PictureContext,
    pic_state: &mut PictureState,
    frame_context: &FrameBuildingContext,
    frame_state: &mut FrameBuildingState,
    data_stores: &DataStores,
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
                &scratch.frame.draws[prim_instance_index],
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
            scratch.frame.draws[prim_instance_index].reset();
        }
    }
}

/// Returns the texture sampler kind used by a YUV image's planes, which selects
/// the matching ps_quad_yuv shader variant. All planes are expected to share the
/// same kind. Texture-cache backed images (raw/blob/buffer) are always Texture2D.
fn yuv_planes_sampler_kind(
    yuv_image_data: &crate::prim_store::image::YuvImageData,
    resource_cache: &crate::resource_cache::ResourceCache,
) -> ImageBufferKind {
    let plane_count = yuv_image_data.format.get_plane_num();
    for key in &yuv_image_data.yuv_key[.. plane_count] {
        if let Some(ExternalImageData { image_type: ExternalImageType::TextureHandle(kind), .. }) =
            resource_cache.get_image_properties(*key).and_then(|props| props.external_image)
        {
            return kind;
        }
    }
    ImageBufferKind::Texture2D
}

/// Maps a filter to the (filter_mode, parameter) pair consumed by the
/// blend shader.
fn blend_filter_param(filter: &Filter, extra_gpu_data: &[GpuBufferAddress]) -> Option<(i32, i32)> {
    let param = match filter {
        Filter::Contrast(amount)
        | Filter::Grayscale(amount)
        | Filter::Invert(amount)
        | Filter::Saturate(amount)
        | Filter::Sepia(amount)
        | Filter::Brightness(amount)
        => (amount * 65536.0) as i32,
        Filter::HueRotate(angle) => (0.01745329251 * angle * 65536.0) as i32,
        Filter::ColorMatrix(..)
        | Filter::Flood(..)
        => extra_gpu_data[0].as_int(),
        Filter::SrgbToLinear
        | Filter::LinearToSrgb
        => 0,
        // Component transfer is handled separately.
        _ => return None,
    };
    Some((filter.as_int(), param))
 }

fn prepare_prim_for_render(
    store: &mut PrimitiveStore,
    prim_instance_index: usize,
    cluster: &mut PrimitiveCluster,
    mut quad_transform: &mut QuadTransformState,
    pic_context: &PictureContext,
    pic_state: &mut PictureState,
    frame_context: &FrameBuildingContext,
    frame_state: &mut FrameBuildingState,
    plane_split_anchor: PlaneSplitAnchor,
    data_stores: &DataStores,
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
    if let PrimitiveKind::Picture { pic_index, .. } = prim_instances[prim_instance_index].kind {
        let Some(scratch_handle) = prepare_picture(
            pic_index,
            store,
            Some(pic_context.surface_index),
            pic_context.subpixel_mode,
            frame_context,
            frame_state,
            data_stores,
            scratch,
            tile_caches,
            prim_instances,
        ) else {
            return;
        };

        scratch.frame.draws[prim_instance_index].kind_scratch =
            KindScratchHandle::Picture(scratch_handle);

        is_passthrough = store
            .pictures[pic_index.0]
            .composite_mode
            .is_none();
    }

    let prim_instance = &mut prim_instances[prim_instance_index];
    let mut use_legacy_path = true;
    if !is_passthrough {
        match &prim_instance.kind {
            PrimitiveKind::Rectangle { .. }
            | PrimitiveKind::RadialGradient { .. }
            | PrimitiveKind::ConicGradient { .. }
            | PrimitiveKind::LinearGradient { .. }
            | PrimitiveKind::Image { .. }
            | PrimitiveKind::NormalBorder { .. }
            | PrimitiveKind::ImageBorder { .. }
            | PrimitiveKind::LineDecoration { .. }
            | PrimitiveKind::BackdropRender { .. }
            | PrimitiveKind::BoxShadow { .. }
            => {
                use_legacy_path = false;
            }
            _ => {}
        };

        // In the new quad rendering path, want to skip the entry point to
        // `update_clip_task` as that does old-style segmenting and mask
        // generation.
        let should_update_clip_task = match &mut prim_instance.kind {
            PrimitiveKind::Picture { .. } => false,
            _ => use_legacy_path,
        };

        if should_update_clip_task {
            let snapped_local_rect = scratch.frame.draws[prim_instance_index].snapped_local_rect;
            let prim_rect = data_stores.get_local_prim_rect(
                prim_instance,
                snapped_local_rect,
                &store.pictures,
                frame_state.surfaces,
            );

            if !update_clip_task(
                PrimitiveInstanceIndex(prim_instance_index as u32),
                prim_rect,
                cluster.spatial_node_index,
                pic_context.raster_spatial_node_index,
                pic_context,
                frame_context,
                frame_state,
                data_stores,
                scratch,
            ) {
                return;
            }
        }
    }

    let prim_instance_index = PrimitiveInstanceIndex(prim_instance_index as u32);

    let prim_spatial_node_index = cluster.spatial_node_index;
    let device_pixel_scale = frame_state.surfaces[pic_context.surface_index.0].device_pixel_scale;
    // Snapshot of the per-frame draw header for this prim. Copy is fine here
    // because the only field this function writes (clip_task_index, in the
    // segmented-clip path) isn't read again in this function — and the other
    // fields (state, clip_chain) aren't written by it.
    let prim_info = scratch.frame.draws[prim_instance_index.0 as usize];

    match &mut prim_instance.kind {
        PrimitiveKind::BoxShadow { data_handle, .. } => {
            profile_scope!("BoxShadow");

            let prim_data = &data_stores.box_shadow[*data_handle];

            prepare_box_shadow(
                &prim_data.kind,
                &prim_data.common,
                &prim_instance.unsnapped_prim_rect,
                &prim_info.clip_chain,
                &mut quad_transform,
                frame_context,
                pic_context,
                frame_state,
                scratch,
                prim_spatial_node_index,
                device_pixel_scale,
                prim_instance_index,
                targets,
                data_stores,
            );

            return;
        }
        PrimitiveKind::LineDecoration { data_handle } => {
            profile_scope!("LineDecoration");
            let prim_data = &data_stores.line_decoration[*data_handle];
            let line_dec_data = &prim_data.kind;

            let task = prim_data.kind.prepare(
                prim_info.snapped_local_rect.size(),
                prim_spatial_node_index,
                frame_context,
                frame_state,
            );

            if let Some((src_task_id, stretch_size)) = task {
                let pattern = ImagePattern {
                    src_task_id,
                    src_is_opaque: false,
                    premultiplied: true,
                    sampler_kind: ImageBufferKind::Texture2D,
                    color: line_dec_data.color,
                };

                quad::prepare_repeatable_quad(
                    &pattern,
                    &prim_info.snapped_local_rect,
                    &prim_info.clip_chain.local_clip_rect,
                    stretch_size,
                    LayoutSize::zero(),
                    prim_data.common.aligned_aa_edges,
                    prim_data.common.transformed_aa_edges,
                    prim_instance_index,
                    &None,
                    &prim_info.clip_chain,
                    quad_transform,
                    frame_context,
                    pic_context,
                    targets,
                    &data_stores.clip,
                    frame_state,
                    scratch,
                );
            } else {
                quad::prepare_quad(
                    &line_dec_data.color,
                    &prim_info.snapped_local_rect,
                    &prim_info.clip_chain.local_clip_rect,
                    prim_data.common.aligned_aa_edges,
                    prim_data.common.transformed_aa_edges,
                    prim_instance_index,
                    &None,
                    &prim_info.clip_chain,
                    quad_transform,
                    frame_context,
                    pic_context,
                    targets,
                    &data_stores.clip,
                    frame_state,
                    scratch,
                );
            }

            return;
        }
        PrimitiveKind::TextRun { data_handle } => {
            profile_scope!("TextRun");

            let prim_data = &data_stores.text_run[*data_handle];

            // The transform has to match the prim -> raster transform applied
            // by "ps_text_run" via `transform.m` + `device_pixel_scale`.
            // `request_resources` uses it to map glyph pen positions into
            // absolute device space for snapping.
            let transform = frame_context.spatial_tree
                .get_relative_transform(
                    prim_spatial_node_index,
                    pic_context.raster_spatial_node_index,
                )
                .into_fast_transform();

            // The run anchor is the normalized prim rect origin; glyph
            // positions in the template are stored relative to it. Use the
            // unsnapped rect so the anchor matches what the shader receives in
            // `PrimitiveHeader.local_rect`.
            let local_rect = prim_instance.unsnapped_prim_rect;

            let surface = &frame_state.surfaces[pic_context.surface_index.0];

            // If subpixel AA is disabled due to the backing surface the glyphs
            // are being drawn onto, disable it (unless we are using the
            // specifial subpixel mode that estimates background color).
            let allow_subpixel = match prim_info.state {
                DrawState::Culled |
                DrawState::Unset |
                DrawState::PassThrough => {
                    panic!("bug: invalid visibility state");
                }
                DrawState::Visible { sub_slice_index, .. } => {
                    // For now, we only allow subpixel AA on primary sub-slices. In future we
                    // may support other sub-slices if we find content that does this.
                    if sub_slice_index.is_primary() {
                        match pic_context.subpixel_mode {
                            SubpixelMode::Allow => true,
                            SubpixelMode::Deny => false,
                            SubpixelMode::Conditional { allowed_rect, prohibited_rect } => {
                                // Conditional mode allows subpixel AA to be enabled for this
                                // text run, so long as it's inside the allowed rect.
                                allowed_rect.contains_box(&prim_info.clip_chain.pic_coverage_rect) &&
                                !prohibited_rect.intersects(&prim_info.clip_chain.pic_coverage_rect)
                            }
                        }
                    } else {
                        false
                    }
                }
            };

            let text_run_handle = prim_data.request_resources(
                local_rect,
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
            scratch.frame.draws[prim_instance_index.0 as usize].kind_scratch =
                KindScratchHandle::TextRun(text_run_handle);
        }
        PrimitiveKind::NormalBorder { data_handle } => {
            profile_scope!("NormalBorder");
            let prim_data = &data_stores.normal_border[*data_handle];
            let aligned_aa_edges = prim_data.common.aligned_aa_edges;
            let transformed_aa_edges = prim_data.common.transformed_aa_edges;
            let border_data = &prim_data.kind;

            border_data.update(
                &prim_info.snapped_local_rect,
                &prim_info.clip_chain,
                prim_spatial_node_index,
                device_pixel_scale,
                aligned_aa_edges,
                transformed_aa_edges,
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
        PrimitiveKind::ImageBorder { data_handle, .. } => {
            profile_scope!("ImageBorder");
            let prim_data = &data_stores.image_border[*data_handle];
            let aligned_aa_edges = prim_data.common.aligned_aa_edges;
            let transformed_aa_edges = prim_data.common.transformed_aa_edges;
            let border_data = &prim_data.kind;

            let (task_id, size, is_opaque) = border_data.update(frame_state);

            let prim_rect = prim_info.snapped_local_rect;

            let src_image = ImagePattern {
                src_task_id: task_id,
                src_is_opaque: is_opaque,
                premultiplied: true,
                sampler_kind: ImageBufferKind::Texture2D,
                color: ColorF::WHITE,
            };

            prepare_border_image_nine_patch(
                &border_data.nine_patch,
                &src_image,
                size,
                &prim_rect,
                aligned_aa_edges,
                transformed_aa_edges,
                prim_instance_index,
                &prim_info.clip_chain,
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
        PrimitiveKind::Rectangle { data_handle, .. } => {
            profile_scope!("Rectangle");

            let prim_data = &data_stores.prim[*data_handle];
            let prim_rect = prim_info.snapped_local_rect;
            let color = prim_data.resolve(frame_context.scene_properties);

            quad::prepare_quad(
                &color,
                &prim_rect,
                &prim_info.clip_chain.local_clip_rect,
                prim_data.common.aligned_aa_edges,
                prim_data.common.transformed_aa_edges,
                prim_instance_index,
                &None,
                &prim_info.clip_chain,
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
        PrimitiveKind::YuvImage { data_handle, .. } => {
            profile_scope!("YuvImage");
            let prim_data = &data_stores.yuv_image[*data_handle];
            let common_data = &prim_data.common;
            let yuv_image_data = &prim_data.kind;

            if prim_info.compositor_surface_kind == CompositorSurfaceKind::Underlay {
                quad::prepare_quad(
                    &Cutout,
                    &prim_info.snapped_local_rect,
                    &prim_info.clip_chain.local_clip_rect,
                    common_data.aligned_aa_edges,
                    common_data.transformed_aa_edges,
                    prim_instance_index,
                    &None,
                    &prim_info.clip_chain,
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

            // Non-composited: draw the YUV image directly through the quad path.
            let planes = yuv_image_data.update(
                prim_info.compositor_surface_kind.is_composited(),
                frame_state,
            );

            let pattern = YuvPattern {
                planes,
                format: yuv_image_data.format,
                color_space: yuv_image_data.color_space.with_range(yuv_image_data.color_range),
                channel_bit_depth: yuv_image_data.color_depth.bit_depth(),
                sampler_kind: yuv_planes_sampler_kind(yuv_image_data, frame_state.resource_cache),
            };

            quad::prepare_quad(
                &pattern,
                &prim_info.snapped_local_rect,
                &prim_info.clip_chain.local_clip_rect,
                common_data.aligned_aa_edges,
                common_data.transformed_aa_edges,
                prim_instance_index,
                &None,
                &prim_info.clip_chain,
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
        PrimitiveKind::Image { data_handle, .. } => {
            profile_scope!("Image");

            let prim_data = &data_stores.image[*data_handle];
            let common_data = &prim_data.common;
            let image_data = &prim_data.kind;

            let prim_rect = prim_info.snapped_local_rect;

            if prim_info.compositor_surface_kind == CompositorSurfaceKind::Underlay {
                quad::prepare_quad(
                    &Cutout,
                    &prim_rect,
                    &prim_info.clip_chain.local_clip_rect,
                    common_data.aligned_aa_edges,
                    common_data.transformed_aa_edges,
                    prim_instance_index,
                    &None,
                    &prim_info.clip_chain,
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

            crate::prim_store::image::prepare_image_quads(
                &prim_rect,
                common_data,
                image_data,
                &prim_info.clip_chain,
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
        PrimitiveKind::LinearGradient { data_handle, .. } => {
            profile_scope!("LinearGradient");
            let prim_data = &data_stores.linear_grad[*data_handle];
            let prim_rect = prim_info.snapped_local_rect;
            let stretch_size = LayoutSize::new(
                prim_data.stretch_ratio.width * prim_rect.size().width,
                prim_data.stretch_ratio.height * prim_rect.size().height,
            );

            if let Some(nine_patch) = &prim_data.border_nine_patch {
                quad::prepare_border_nine_patch(
                    &*nine_patch,
                    prim_data,
                    &prim_rect,
                    stretch_size,
                    prim_data.common.aligned_aa_edges,
                    prim_data.common.transformed_aa_edges,
                    prim_instance_index,
                    &prim_info.clip_chain,
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

            // Fast-path: axis-aligned non-repeating gradients with multiple
            // stops decompose into per-segment two-stop quads so the GPU can
            // take the `sample_gradient_stops_fast` shader path. The
            // decomposition runs at frame-build (against the snapped prim
            // rect) so adjacent segments tile end-to-end at the snapped
            // outer-prim grid, even when the frame-time snap pass nudges
            // the outer rect at fractional DPR.
            //
            // `create_linear_gradient_prim` canonicalises the stored
            // start/end by swapping them when the original gradient line
            // ran "backwards" (and recording that in `reverse_stops`).
            // `LinearGradientTemplate::build` swaps them back at render
            // time; we have to do the same here so the decomposition sees
            // the gecko-original gradient orientation -- otherwise the
            // segment loop produces a gradient with stops in reverse
            // order (e.g. `linear-gradient(to top, red, blue)` rendering
            // as red-on-top instead of red-on-bottom).
            let (effective_start, effective_end) = if prim_data.reverse_stops {
                (prim_data.end_point, prim_data.start_point)
            } else {
                (prim_data.start_point, prim_data.end_point)
            };
            if linear_gradient_decomposes(
                &prim_rect,
                stretch_size,
                prim_data.tile_spacing,
                effective_start,
                effective_end,
                prim_data.extend_mode,
                &prim_data.stops,
                frame_context.fb_config.enable_dithering,
            ) {
                decompose_axis_aligned_gradient(
                    &prim_rect,
                    stretch_size,
                    effective_start,
                    effective_end,
                    &prim_data.stops,
                    &prim_info.clip_chain.local_clip_rect,
                    |seg_rect, seg_start, seg_end, seg_stops, edge_aa_mask| {
                        let pattern = LinearGradientSegmentPattern {
                            start: seg_start,
                            end: seg_end,
                            stops: seg_stops,
                        };
                        quad::prepare_quad(
                            &pattern,
                            seg_rect,
                            &prim_info.clip_chain.local_clip_rect,
                            EdgeMask::empty(),
                            edge_aa_mask,
                            prim_instance_index,
                            &None,
                            &prim_info.clip_chain,
                            quad_transform,
                            frame_context,
                            pic_context,
                            targets,
                            &data_stores.clip,
                            frame_state,
                            scratch,
                        );
                    },
                );
                return;
            }

            // For SWGL, evaluating the gradient is faster than reading from the texture cache.
            let mut should_cache = !frame_context.fb_config.is_software
                && frame_state.resource_cache.texture_cache.allocated_color_bytes() < 10_000_000;
            if should_cache {
                let surface = &frame_state.surfaces[pic_context.surface_index.0];
                let clipped_surface_rect = surface.get_surface_rect(
                    &prim_info.clip_chain.pic_coverage_rect,
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
                    &prim_info.clip_chain,
                    frame_state.clip_store,
                )
            } else {
                None
            };

            let local_rect = prim_info.snapped_local_rect;
            quad::prepare_repeatable_quad(
                prim_data,
                &local_rect,
                &prim_info.clip_chain.local_clip_rect,
                stretch_size,
                prim_data.tile_spacing,
                prim_data.common.aligned_aa_edges,
                prim_data.common.transformed_aa_edges,
                prim_instance_index,
                &cache_key,
                &prim_info.clip_chain,
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
        PrimitiveKind::RadialGradient { data_handle, .. } => {
            profile_scope!("RadialGradient");
            let prim_data = &data_stores.radial_grad[*data_handle];
            let local_rect = prim_info.snapped_local_rect;
            let stretch_size = LayoutSize::new(
                prim_data.stretch_ratio.width * local_rect.size().width,
                prim_data.stretch_ratio.height * local_rect.size().height,
            );

            if let Some(nine_patch) = &prim_data.border_nine_patch {
                quad::prepare_border_nine_patch(
                    &*nine_patch,
                    prim_data,
                    &local_rect,
                    stretch_size,
                    prim_data.common.aligned_aa_edges,
                    prim_data.common.transformed_aa_edges,
                    prim_instance_index,
                    &prim_info.clip_chain,
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
                &prim_info.clip_chain.local_clip_rect,
                stretch_size,
                prim_data.tile_spacing,
                prim_data.common.aligned_aa_edges,
                prim_data.common.transformed_aa_edges,
                prim_instance_index,
                &None,
                &prim_info.clip_chain,
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
        PrimitiveKind::ConicGradient { data_handle, .. } => {
            profile_scope!("ConicGradient");
            let prim_data = &data_stores.conic_grad[*data_handle];
            let prim_rect = prim_info.snapped_local_rect;
            let stretch_size = LayoutSize::new(
                prim_data.stretch_ratio.width * prim_rect.size().width,
                prim_data.stretch_ratio.height * prim_rect.size().height,
            );

            if let Some(nine_patch) = &prim_data.border_nine_patch {
                quad::prepare_border_nine_patch(
                    &*nine_patch,
                    prim_data,
                    &prim_rect,
                    stretch_size,
                    prim_data.common.aligned_aa_edges,
                    prim_data.common.transformed_aa_edges,
                    prim_instance_index,
                    &prim_info.clip_chain,
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
                    &prim_info.clip_chain.pic_coverage_rect,
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
                    &prim_info.clip_chain,
                    frame_state.clip_store,
                )
            } else {
                None
            };

            let local_rect = prim_info.snapped_local_rect;
            quad::prepare_repeatable_quad(
                prim_data,
                &local_rect,
                &prim_info.clip_chain.local_clip_rect,
                stretch_size,
                prim_data.tile_spacing,
                prim_data.common.aligned_aa_edges,
                prim_data.common.transformed_aa_edges,
                prim_instance_index,
                &cache_key,
                &prim_info.clip_chain,
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
        PrimitiveKind::Picture { pic_index, .. } => {
            profile_scope!("Picture");
            let pic_scratch_handle = prim_info.kind_scratch.unwrap_picture();
            let pic = &mut store.pictures[pic_index.0];

            let Some(raster_config) = &pic.raster_config else {
                return;
            };

            let pic_scratch = &mut scratch.frame.pictures[pic_scratch_handle];

            // Write the composite-mode gpu blocks first: the filter eligibility
            // check below reads the resulting extra_gpu_data.
            raster_config.composite_mode.write_gpu_blocks(
                &mut frame_state.frame_gpu_data,
                data_stores,
                &mut pic_scratch.extra_gpu_data,
            );

            // Decide whether this picture's compositing is migrated to the quad
            // path. This is computed before clip-mask handling so that target
            // masks (those applied while compositing, rather than baked onto the
            // source task) can be routed to the quad path rather than the legacy
            // brush path.
            //
            // Pictures that are part of a 3D context are composited through the
            // plane splitter, so they are left on the legacy path here.
            let use_quads = match raster_config.composite_mode {
                PictureCompositeMode::TileCache { .. } => false,
                PictureCompositeMode::IntermediateSurface => false,
                _ => matches!(pic.context_3d, Picture3DContext::Out),
            };

            // Clip masks are split into "source" masks (baked onto the picture's
            // source task) and "target" masks (applied while compositing). When
            // the picture composites via the quad path, target masks are carried
            // here and applied by that path; otherwise the legacy brush path
            // renders a screen-space alpha mask task.
            let mut composite_target_clip_range: Option<ClipNodeRange> = None;

            if prim_info.clip_chain.needs_mask {
                prepare_picture_clips(
                    pic,
                    prim_instance_index,
                    &prim_info.clip_chain,
                    frame_context,
                    frame_state,
                    pic_scratch,
                    &mut scratch.frame.clip_mask_instances,
                    &mut scratch.frame.draws,
                    prim_spatial_node_index,
                    data_stores,
                    use_quads,
                    &mut composite_target_clip_range,
                    pic_context,
                );
            }

            if let Picture3DContext::In { root_data: None, plane_splitter_index, ancestor_index, .. } = pic.context_3d {
                let dirty_rect = frame_state.current_dirty_region().combined;
                let visibility_spatial_node = frame_state.current_dirty_region().visibility_spatial_node;

                let splitter = &mut frame_state.plane_splitters[plane_splitter_index.0];
                let surface_index = raster_config.surface_index;
                let surface = &frame_state.surfaces[surface_index.0];
                let local_prim_rect = surface.clipped_local_rect.cast_unit();

                PictureInstance::add_split_plane(
                    splitter,
                    frame_context.spatial_tree,
                    prim_spatial_node_index,
                    ancestor_index,
                    visibility_spatial_node,
                    local_prim_rect,
                    &prim_info.clip_chain.local_clip_rect,
                    dirty_rect,
                    plane_split_anchor,
                );

                // The PrimitiveCommand is pushed by PictureInstance::restore_context.
                return;
            }

            if !use_quads {
                return;
            }

            // Detached snapshot pictures are not composited.
            let detached = pic.snapshot.map_or(false, |s| s.detached);
            if detached {
                return;
            }

            let pic_task_id = pic_scratch
                .primary_render_task_id
                .expect("bug: no render task for composited picture");

            let surface = &frame_state.surfaces[raster_config.surface_index.0];
            let pic_local_rect = raster_config.composite_mode.get_rect(surface, None);
            let surface_spatial_node_index = surface.surface_spatial_node_index;
            let is_same_coord_system = surface_spatial_node_index == surface.raster_spatial_node_index;

            // For a raster root, the baked raster transform must not
            // be applied again at composite time, so use a dedicated
            // local-to-raster scale-offset transform (and the clip
            // rect it implies) rather than the cluster's transform.
            let mut local_transform;
            let (local_clip_rect, transform) = if is_same_coord_system {
                (prim_info.clip_chain.local_clip_rect, quad_transform)
            } else {
                let map_local_to_raster = SpaceMapper::new_with_target(
                    pic_context.raster_spatial_node_index,
                    surface_spatial_node_index,
                    LayoutRect::max_rect(),
                    frame_context.spatial_tree,
                );

                let raster_rect = map_local_to_raster.map(&pic_local_rect).unwrap();

                // TODO(nical): This matches what the brush code does in batch.rs but
                // it does not make sense to me.
                let sx = raster_rect.width() / pic_local_rect.width();
                let sy = raster_rect.height() / pic_local_rect.height();
                let tx = raster_rect.min.x - sx * pic_local_rect.min.x;
                let ty = raster_rect.min.y - sy * pic_local_rect.min.y;
                let local_to_raster_so = ScaleOffset::new(sx, sy, tx, ty);

                let local_clip_rect = prim_info.clip_chain.local_clip_rect;
                let raster_clip_rect = map_local_to_raster.map(&local_clip_rect).unwrap();
                let adjusted_clip_rect = local_to_raster_so.unmap_rect(&raster_clip_rect);

                local_transform = QuadTransformState::from_scale_offset(
                    local_to_raster_so,
                    prim_spatial_node_index,
                    pic_context.raster_spatial_node_index,
                    quad_transform.device_pixel_scale(),
                );

                (adjusted_clip_rect, &mut local_transform)
            };

            // Source clip masks (if any) were drawn onto the picture's
            // source task above, so the compositing quad must not
            // re-apply them (which would mask twice). Target clip masks
            // are applied here by the quad path via their own clip
            // range.
            let mut composite_clip_chain = prim_info.clip_chain;
            match composite_target_clip_range {
                Some(clips_range) => {
                    composite_clip_chain.needs_mask = true;
                    composite_clip_chain.clips_range = clips_range;
                }
                None => {
                    composite_clip_chain.needs_mask = false;
                }
            }

            let mut opacity = 1.0;
            // (filter_mode, amount-or-gpu-address) for CSS/SVG filters that map
            // to the ps_quad_blend shader.
            let mut filter = None;
            // Software mix-blend mode mapping to the ps_quad_mix_blend shader.
            let mut mix_blend = None;
            // GPU-blend-equation mix-blend mode (Screen/Exclusion/PlusLighter)
            // drawn as a blended image quad.
            let mut hw_blend = None;

            match raster_config.composite_mode {
                PictureCompositeMode::MixBlend(mode) => {
                    match BlendMode::from_mix_blend_mode(
                        mode,
                        frame_context.fb_config.gpu_supports_advanced_blend,
                        frame_context.fb_config.advanced_blend_is_coherent,
                    ) {
                        // No GPU blend equation available: composite via a
                        // software readback of the backdrop (ps_quad_mix_blend).
                        None => {
                            mix_blend = Some(mode);
                        }
                        // Advanced blend equation, or a fixed-function blend
                        // (Screen / Exclusion / PlusLighter): draw the picture
                        // content as a blended image quad.
                        Some(bm) => {
                            hw_blend = Some(bm);
                        }
                    }
                }
                PictureCompositeMode::Filter(Filter::Opacity(_, amount)) => {
                    opacity = amount;
                }
                PictureCompositeMode::Filter(ref f) => {
                    let extra_gpu_data = pic_scratch
                        .extra_gpu_data
                        .as_slice();
                    filter = blend_filter_param(f, extra_gpu_data);
                }
                PictureCompositeMode::ComponentTransferFilter(handle) => {
                    let filter_data = &data_stores.filter_data[handle];
                    let filter_mode: i32 = Filter::ComponentTransfer.as_int()
                        | ((filter_data.data.r_func.to_int() << 28
                            | filter_data.data.g_func.to_int() << 24
                            | filter_data.data.b_func.to_int() << 20
                            | filter_data.data.a_func.to_int() << 16)
                            as i32);
                    let addr = pic_scratch
                        .extra_gpu_data[0]
                        .as_int();
                    filter = Some((filter_mode, addr));
                }
                _ => {}
            };

            let img_pattern;
            let mix_blend_pattern;
            let ff_mix_blend_pattern;
            let filter_pattern;

            let pattern: &dyn PatternBuilder = if let PictureCompositeMode::Filter(Filter::DropShadows(ref shadows)) =
                raster_config.composite_mode
            {
                // Draw each shadow (the blurred source tinted by the
                // shadow color, sampled through its alpha) and then
                // the unblurred content on top.
                for shadow in shadows {
                    let shadow_rect = pic_local_rect.translate(shadow.offset);
                    let shadow_pattern = ShadowPattern {
                        src_task_id: pic_task_id,
                        color: shadow.color,
                    };
                    quad::prepare_quad(
                        &shadow_pattern,
                        &shadow_rect,
                        &local_clip_rect,
                        EdgeMask::empty(),
                        EdgeMask::all(),
                        prim_instance_index,
                        &None,
                        &composite_clip_chain,
                        transform,
                        frame_context,
                        pic_context,
                        targets,
                        &data_stores.clip,
                        frame_state,
                        scratch,
                    );
                }

                let content_task_id = scratch.frame.pictures[pic_scratch_handle]
                    .secondary_render_task_id
                    .expect("bug: no content task for drop shadow");
                img_pattern = ImagePattern {
                    src_task_id: content_task_id,
                    src_is_opaque: false,
                    premultiplied: true,
                    sampler_kind: ImageBufferKind::Texture2D,
                    color: ColorF::WHITE,
                };

                &img_pattern
            } else if let Some(mode) = mix_blend {
                // The backdrop was captured into a readback task during
                // composite-mode setup; blend the picture (source) over it.
                let backdrop_task_id = pic_scratch
                    .secondary_render_task_id
                    .expect("bug: no backdrop readback task for mix-blend");

                mix_blend_pattern = MixBlendPattern {
                    backdrop_task_id,
                    src_task_id: pic_task_id,
                    mode,
                };

                &mix_blend_pattern
            } else if let Some(blend_mode) = hw_blend {
                ff_mix_blend_pattern = FixedFunctionMixBlendPattern {
                    src_task_id: pic_task_id,
                    blend_mode,
                };
                &ff_mix_blend_pattern
            } else if let Some((filter_mode, param)) = filter {
                filter_pattern = BlendFilterPattern {
                    src_task_id: pic_task_id,
                    filter_mode,
                    param,
                };
                &filter_pattern
            } else {
                img_pattern = ImagePattern {
                    src_task_id: pic_task_id,
                    src_is_opaque: false,
                    premultiplied: true,
                    sampler_kind: ImageBufferKind::Texture2D,
                    color: ColorF::new(1.0, 1.0, 1.0, opacity),
                };
                &img_pattern
            };

            quad::prepare_quad(
                pattern,
                &pic_local_rect,
                &local_clip_rect,
                EdgeMask::empty(),
                EdgeMask::all(),
                prim_instance_index,
                &None,
                &composite_clip_chain,
                transform,
                frame_context,
                pic_context,
                targets,
                &data_stores.clip,
                frame_state,
                scratch,
            );

            return;
        }
        PrimitiveKind::BackdropCapture { .. } => {
            // Register the owner picture of this backdrop primitive as the
            // target for resolve of the sub-graph
            frame_state.surface_builder.register_resolve_source();

            if frame_context.debug_flags.contains(DebugFlags::HIGHLIGHT_BACKDROP_FILTERS) {
                if let Some(world_rect) = pic_state.map_pic_to_vis.map(&prim_info.clip_chain.pic_coverage_rect) {
                    scratch.push_debug_rect(
                        world_rect.cast_unit(),
                        2,
                        crate::debug_colors::MAGENTA,
                        ColorF::TRANSPARENT,
                    );
                }
            }
        }
        PrimitiveKind::BackdropRender { pic_index, data_handle, .. } => {
            match frame_state.surface_builder.sub_graph_output_map.get(pic_index).cloned() {
                Some(sub_graph_output_id) => {
                    frame_state.surface_builder.add_child_render_task(
                        sub_graph_output_id,
                        frame_state.rg_builder,
                    );

                    // Compute the four homogeneous screen-space uv corners that map
                    // the primitive rect into the captured backdrop. This mirrors the
                    // legacy brush path in batch.rs.
                    let pic_task = frame_state.rg_builder.get_task(sub_graph_output_id);
                    let uv_rect_kind = pic_task.uv_rect_kind();
                    let RenderTaskKind::Picture(info) = &pic_task.kind else {
                        unreachable!("bug: backdrop sub-graph output is not a picture");
                    };
                    // The shader maps the bilinearly-interpolated screen uv into the
                    // backdrop's texture-cache rect (the segment uv rect), which is
                    // resolved from the source task honoring its uv_rect_kind. When the
                    // backdrop surface is clipped (e.g. by the viewport), that uv rect
                    // is the projection of the *unclipped* surface rect. The screen uvs
                    // must therefore be normalized over the unclipped device rect so the
                    // two are consistent. We reconstruct the unclipped rect from the
                    // clipped rect (content_origin + target size) and the uv_rect_kind.
                    let clipped_origin = info.content_origin;
                    let clipped_size = pic_task.get_target_size().to_f32();
                    let backdrop_rect = match uv_rect_kind {
                        UvRectKind::Rect => {
                            DeviceRect::from_origin_and_size(clipped_origin, clipped_size)
                        }
                        UvRectKind::Quad { top_left, bottom_right, .. } => {
                            DeviceRect {
                                min: clipped_origin + DeviceVector2D::new(
                                    top_left.x * clipped_size.width,
                                    top_left.y * clipped_size.height,
                                ),
                                max: clipped_origin + DeviceVector2D::new(
                                    bottom_right.x * clipped_size.width,
                                    bottom_right.y * clipped_size.height,
                                ),
                            }
                        }
                    };
                    let device_pixel_scale = info.device_pixel_scale;

                    // `content_origin`/`backdrop_rect` are in the surface's raster
                    // device space (world space when the raster root is the scene
                    // root). Map the prim into that same raster space rather than the
                    // surface's own spatial node, otherwise the readback is offset by
                    // the surface node's origin whenever snapping promotes the raster
                    // root away from the surface node.
                    let raster_spatial_node_index = info.raster_spatial_node_index;

                    let map_prim_to_backdrop = SpaceMapper::new_with_target(
                        raster_spatial_node_index,
                        prim_spatial_node_index,
                        WorldRect::max_rect(),
                        frame_context.spatial_tree,
                    );

                    let prim_rect = prim_info.snapped_local_rect;
                    let points = [
                        map_prim_to_backdrop.map_point(prim_rect.top_left()),
                        map_prim_to_backdrop.map_point(prim_rect.top_right()),
                        map_prim_to_backdrop.map_point(prim_rect.bottom_left()),
                        map_prim_to_backdrop.map_point(prim_rect.bottom_right()),
                    ];

                    if points.iter().any(|p| p.is_none()) {
                        scratch.frame.draws[prim_instance_index.0 as usize].reset();
                        return;
                    }

                    let uvs = [
                        calculate_screen_uv(points[0].unwrap() * device_pixel_scale, backdrop_rect),
                        calculate_screen_uv(points[1].unwrap() * device_pixel_scale, backdrop_rect),
                        calculate_screen_uv(points[2].unwrap() * device_pixel_scale, backdrop_rect),
                        calculate_screen_uv(points[3].unwrap() * device_pixel_scale, backdrop_rect),
                    ];

                    let prim_data = &data_stores.backdrop_render[*data_handle];
                    let aligned_aa_edges = prim_data.common.aligned_aa_edges;
                    let transformed_aa_edges = prim_data.common.transformed_aa_edges;

                    let pattern = BackdropPattern {
                        src_task_id: sub_graph_output_id,
                        uvs,
                    };

                    quad::prepare_quad(
                        &pattern,
                        &prim_info.snapped_local_rect,
                        &prim_info.clip_chain.local_clip_rect,
                        aligned_aa_edges,
                        transformed_aa_edges,
                        prim_instance_index,
                        &None,
                        &prim_info.clip_chain,
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
                None => {
                    // Backdrop capture was found not visible, didn't produce a sub-graph
                    // so we can just skip drawing
                    scratch.frame.draws[prim_instance_index.0 as usize].reset();
                }
            }
        }
    }

    match prim_info.state {
        DrawState::Unset => {
            panic!("bug: invalid vis state");
        }
        DrawState::Visible { .. } => {
            frame_state.push_prim(
                &PrimitiveCommand::simple(storage::Index::from_u32(prim_instance_index.0)),
                prim_spatial_node_index,
                targets,
            );
        }
        DrawState::PassThrough | DrawState::Culled => {}
    }
}


/// Create a clip-mask render task by accumulating the clip chain into a blank
/// (white-cleared) alpha target as quad sub-tasks.
fn add_clip_mask_render_task(
    device_rect: DeviceIntRect,
    clip_node_range: ClipNodeRange,
    prim_local_rect: LayoutRect,
    prim_spatial_node_index: SpatialNodeIndex,
    raster_spatial_node_index: SpatialNodeIndex,
    device_pixel_scale: DevicePixelScale,
    data_stores: &DataStores,
    frame_context: &FrameBuildingContext,
    frame_state: &mut FrameBuildingState,
) -> RenderTaskId {
    // A blank render task that just clears its target to white. The clip chain
    // is multiply-blended on top of it by the sub-tasks below.
    let clip_task_id = frame_state.rg_builder.add().init(RenderTask::new_dynamic(
        device_rect.size(),
        RenderTaskKind::Empty(EmptyTask {
            content_origin: device_rect.min.to_f32(),
            device_pixel_scale,
            raster_spatial_node_index,
        }),
    ));

    let task_rect = device_rect.to_f32();

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

    clip_task_id
}

pub fn update_clip_task(
    prim_instance_index: PrimitiveInstanceIndex,
    prim_local_rect: LayoutRect,
    prim_spatial_node_index: SpatialNodeIndex,
    root_spatial_node_index: SpatialNodeIndex,
    pic_context: &PictureContext,
    frame_context: &FrameBuildingContext,
    frame_state: &mut FrameBuildingState,
    data_stores: &DataStores,
    scratch: &mut PrimitiveScratchBuffer,
) -> bool {
    let device_pixel_scale = frame_state.surfaces[pic_context.surface_index.0].device_pixel_scale;

    let new_clip_task_index = if scratch.frame.draws[prim_instance_index.0 as usize].clip_chain.needs_mask {
        // Get a minimal device space rect, clipped to the screen that we
        // need to allocate for the clip mask, as well as interpolated
        // snap offsets.
        let unadjusted_device_rect = match frame_state.surfaces[pic_context.surface_index.0].get_surface_rect(
            &scratch.frame.draws[prim_instance_index.0 as usize].clip_chain.pic_coverage_rect,
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

        let clip_task_id = add_clip_mask_render_task(
            device_rect,
            scratch.frame.draws[prim_instance_index.0 as usize].clip_chain.clips_range,
            prim_local_rect,
            prim_spatial_node_index,
            root_spatial_node_index,
            device_pixel_scale,
            data_stores,
            frame_context,
            frame_state,
        );
        // Set the global clip mask instance for this primitive.
        let clip_task_index = ClipTaskIndex(scratch.frame.clip_mask_instances.len() as _);
        scratch.frame.clip_mask_instances.push(ClipMaskKind::Mask(clip_task_id));
        frame_state.surface_builder.add_child_render_task(
            clip_task_id,
            frame_state.rg_builder,
        );
        clip_task_index
    } else {
        ClipTaskIndex::INVALID
    };
    scratch.frame.draws[prim_instance_index.0 as usize].clip_task_index = new_clip_task_index;

    true
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

/// Pattern builder for a single fast-path two-stop segment emitted by
/// `decompose_axis_aligned_gradient`. Holds the segment's gradient line and
/// stop colors (in segment-local coords); `build` translates start/end into
/// the prim's spatial-node space by adding `ctx.prim_origin`.
struct LinearGradientSegmentPattern {
    start: LayoutPoint,
    end: LayoutPoint,
    stops: [GradientStop; 2],
}

impl PatternBuilder for LinearGradientSegmentPattern {
    fn build(
        &self,
        _sub_rect: Option<DeviceRect>,
        offset: LayoutVector2D,
        ctx: &PatternBuilderContext,
        state: &mut PatternBuilderState,
    ) -> Pattern {
        let prim_offset = offset + ctx.prim_origin.to_vector();
        linear_gradient_pattern(
            self.start + prim_offset,
            self.end + prim_offset,
            ExtendMode::Clamp,
            &self.stops,
            ctx.fb_config.is_software,
            state.frame_gpu_data,
        )
    }
}
