/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::point2;
use api::{ColorF, ImageBufferKind, RepeatMode};
use api::units::*;
use crate::border::compute_border_repetition_1d;
use crate::clip::{ClipChainInstance, ClipIntern};
use crate::command_buffer::CommandBufferIndex;
use crate::frame_builder::{FrameBuildingContext, FrameBuildingState, PictureContext};
use crate::intern::DataStore;
use crate::pattern::{PatternBuilder, PatternBuilderContext, PatternBuilderState};
use crate::pattern::image::ImagePattern;
use crate::quad::{QuadTransformState, prepare_repeatable_quad};
use crate::prim_store::{NinePatchDescriptor, PrimitiveInstanceIndex, PrimitiveScratchBuffer};
use crate::segment::EdgeMask;


pub fn prepare_border_image_nine_patch(
    nine_patch: &NinePatchDescriptor,
    src_image: &ImagePattern,
    src_image_size: DeviceIntSize,
    local_rect: &LayoutRect,
    aligned_aa_edges: EdgeMask,
    transfomed_aa_edges: EdgeMask,
    prim_instance_index: PrimitiveInstanceIndex,
    clip_chain: &ClipChainInstance,
    transform: &mut QuadTransformState,

    frame_context: &FrameBuildingContext,
    pic_context: &PictureContext,
    targets: &[CommandBufferIndex],
    interned_clips: &DataStore<ClipIntern>,

    frame_state: &mut FrameBuildingState,
    scratch: &mut PrimitiveScratchBuffer,
) {
    let pattern_ctx = PatternBuilderContext {
        spatial_tree: frame_context.spatial_tree,
        fb_config: frame_context.fb_config,
        prim_origin: local_rect.min,
    };

    let img_pattern = src_image.build(
        None,
        LayoutVector2D::zero(),
        &pattern_ctx,
        &mut PatternBuilderState {
            frame_gpu_data: frame_state.frame_gpu_data,
            transforms: frame_state.transforms,
        },
    );

    for_each_border_image_segment(nine_patch, local_rect, src_image_size, &mut|src_rect, dst_rect, side, stretch_size, spacing, offset| {
        let segment_src = frame_state.rg_builder.add_sub_rect(src_image.src_task_id, &src_rect);

        let segment_pattern = ImagePattern {
            src_task_id: segment_src,
            src_is_opaque: img_pattern.is_opaque,
            premultiplied: true,
            sampler_kind: ImageBufferKind::Texture2D,
            color: ColorF::WHITE,
        };

        let mut segment_local_rect = *dst_rect;
        segment_local_rect.min += offset;

        // For centered (Repeat) tiling we expand the rect leftwards/upwards
        // so a partial tile spans the gap; clip back to the original dst_rect
        // so the fill doesn't bleed into the surrounding edges and corners.
        let local_clip_rect = clip_chain.local_clip_rect
            .intersection(dst_rect)
            .unwrap_or(LayoutRect::zero());

        prepare_repeatable_quad(
            &segment_pattern,
            &segment_local_rect,
            &local_clip_rect,
            stretch_size,
            spacing,
            aligned_aa_edges & side,
            transfomed_aa_edges & side,
            prim_instance_index,
            &None,
            clip_chain,
            transform,
            frame_context,
            pic_context,
            targets,
            interned_clips,
            frame_state,
            scratch,
        );
    });
}


// Spec: https://drafts.csswg.org/css-backgrounds/#border-image-process
pub fn for_each_border_image_segment(
    nine_patch: &NinePatchDescriptor,
    rect: &LayoutRect,
    src_image_size: DeviceIntSize,
    add_segment: &mut dyn FnMut(
        &DeviceIntRect, // src rect
        &LayoutRect,    // dst rect
        EdgeMask,       // segment side
        LayoutSize,     // stretch size
        LayoutSize,     // spacing
        LayoutVector2D, // offset
    ),
) {
    let w = src_image_size.width;
    let h = src_image_size.height;
    // nine_patch.width/height is not always the actual size of the source image
    // despite being expressed in integer device pixels, so a scale factor needs
    // to be introduced to map to the real device pixel space.
    let sx = src_image_size.width as f32 / nine_patch.width as f32;
    let sy = src_image_size.height as f32 / nine_patch.height as f32;

    // src
    let src_x0 = 0;
    let src_x1 = (nine_patch.slice.left as f32 * sx).round() as i32;
    let src_x2 = (w as f32 - nine_patch.slice.right as f32 * sx).round() as i32;
    let src_x3 = w;

    let src_y0 = 0;
    let src_y1 = (nine_patch.slice.top as f32 * sy).round() as i32;
    let src_y2 = (h as f32 - nine_patch.slice.bottom as f32 * sy).round() as i32;
    let src_y3 = h;

    // dst
    let dst_x0 = rect.min.x;
    let dst_x1 = rect.min.x + nine_patch.widths.left;
    let dst_x2 = rect.max.x - nine_patch.widths.right;
    let dst_x3 = rect.max.x;

    let dst_y0 = rect.min.y;
    let dst_y1 = rect.min.y + nine_patch.widths.top;
    let dst_y2 = rect.max.y - nine_patch.widths.bottom;
    let dst_y3 = rect.max.y;

    // The computation of the parameters for the central segment is based on
    // the left/right/top/bottom ones (dependening on which ones are non-empty).
    let mut center_h_rects = None;
    let mut center_v_rects = None;

    // Top left
    let top_left_src = DeviceIntRect { min: point2(src_x0, src_y0), max: point2(src_x1, src_y1) };
    let top_left_dst = LayoutRect { min: point2(dst_x0, dst_y0), max: point2(dst_x1, dst_y1) };
    add_border_segment(
        &top_left_src,
        &top_left_dst,
        EdgeMask::TOP | EdgeMask::LEFT,
        RepeatMode::Stretch,
        RepeatMode::Stretch,
        add_segment,
    );

    // Top right
    let top_right_src = DeviceIntRect { min: point2(src_x2, src_y0), max: point2(src_x3, src_y1) };
    let top_right_dst = LayoutRect { min: point2(dst_x2, dst_y0), max: point2(dst_x3, dst_y1) };
    add_border_segment(
        &top_right_src,
        &top_right_dst,
        EdgeMask::TOP | EdgeMask::RIGHT,
        RepeatMode::Stretch,
        RepeatMode::Stretch,
        add_segment,
    );

    // Bottom right
    let bottom_right_src = DeviceIntRect { min: point2(src_x2, src_y2), max: point2(src_x3, src_y3) };
    let bottom_right_dst = LayoutRect { min: point2(dst_x2, dst_y2), max: point2(dst_x3, dst_y3) };
    add_border_segment(
        &bottom_right_src,
        &bottom_right_dst,
        EdgeMask::BOTTOM | EdgeMask::RIGHT,
        RepeatMode::Stretch,
        RepeatMode::Stretch,
        add_segment,
    );

    // Bottom left
    let bottom_left_src = DeviceIntRect { min: point2(src_x0, src_y2), max: point2(src_x1, src_y3) };
    let bottom_left_dst = LayoutRect { min: point2(dst_x0, dst_y2), max: point2(dst_x1, dst_y3) };
    add_border_segment(
        &bottom_left_src,
        &bottom_left_dst,
        EdgeMask::BOTTOM | EdgeMask::LEFT,
        RepeatMode::Stretch,
        RepeatMode::Stretch,
        add_segment,
    );

    // Add edge segments.

    // Top
    let top_src = DeviceIntRect { min: point2(src_x1, src_y0), max: point2(src_x2, src_y1) };
    let top_dst = LayoutRect { min: point2(dst_x1, dst_y0), max: point2(dst_x2, dst_y1) };
    if add_border_segment(
        &top_src,
        &top_dst,
        EdgeMask::TOP,
        nine_patch.repeat_horizontal,
        RepeatMode::Stretch,
        add_segment,
    ) {
        center_v_rects = Some((top_src, top_dst));
    }

    // Bottom
    let bottom_src = DeviceIntRect { min: point2(src_x1, src_y2), max: point2(src_x2, src_y3) };
    let bottom_dst = LayoutRect { min: point2(dst_x1, dst_y2), max: point2(dst_x2, dst_y3) };
    if add_border_segment(
        &bottom_src,
        &bottom_dst,
        EdgeMask::BOTTOM,
        nine_patch.repeat_horizontal,
        RepeatMode::Stretch,
        add_segment,
    ) && center_v_rects.is_none() {
        center_v_rects = Some((bottom_src, bottom_dst));
    }

    // Left
    let left_src = DeviceIntRect { min: point2(src_x0, src_y1), max: point2(src_x1, src_y2) };
    let left_dst = LayoutRect { min: point2(dst_x0, dst_y1), max: point2(dst_x1, dst_y2) };
    if add_border_segment(
        &left_src,
        &left_dst,
        EdgeMask::LEFT,
        RepeatMode::Stretch,
        nine_patch.repeat_vertical,
        add_segment,
    ) {
        center_h_rects = Some((left_src, left_dst));
    }

    // Right
    let right_src = DeviceIntRect { min: point2(src_x2, src_y1), max: point2(src_x3, src_y2) };
    let right_dst = LayoutRect { min: point2(dst_x2, dst_y1), max: point2(dst_x3, dst_y2) };
    if add_border_segment(
        &right_src,
        &right_dst,
        EdgeMask::RIGHT,
        RepeatMode::Stretch,
        nine_patch.repeat_vertical,
        add_segment,
    ) && center_h_rects.is_none() {
        center_h_rects = Some((right_src, right_dst));
    }

    // Center

    let center_src = DeviceIntRect { min: point2(src_x1, src_y1), max: point2(src_x2, src_y2) };
    let center_dst = LayoutRect { min: point2(dst_x1, dst_y1), max: point2(dst_x2, dst_y2) };
    if nine_patch.fill && !center_src.is_empty() && !center_dst.is_empty() {
        let mut stretch_size = center_dst.size();
        let mut spacing = LayoutSize::zero();
        let mut offset = LayoutVector2D::zero();

        // Per spec:
        // The middle image’s width is scaled by the same factor as the top
        // image unless that factor is zero or infinity, in which case the
        // scaling factor of the bottom is substituted, and failing that,
        // the width is not scaled.
        let (src_size, dst_size, repeat) = if let Some((src_rect, dst_rect)) = center_v_rects {
            (src_rect.size(), dst_rect.size(), nine_patch.repeat_horizontal)
        } else {
            // No top/bottom edge to inherit a scale factor from, so the middle
            // image is not scaled (scale factor 1). compute_border_repetition_1d
            // derives the tile size from dst_cross / src_cross, so matching the
            // destination cross-size to the source's yields a unit scale; the
            // repeat mode then tiles the natural-size pattern across the area.
            let dst = LayoutSize::new(center_dst.width(), center_src.size().height as f32);
            (center_src.size(), dst, nine_patch.repeat_horizontal)
        };
        compute_border_repetition_1d(
            dst_size,
            src_size.to_f32(),
            repeat,
            &mut stretch_size.width,
            &mut spacing.width,
            &mut offset.x,
        );

        // Similarly, per spec:
        // The height of the middle image is scaled by the same factor as
        // the left image unless that factor is zero or infinity, in which
        // case the scaling factor of the right image is substituted, and
        // failing that, the height is not scaled.
        let (src_size, dst_size, repeat) = if let Some((src_rect, dst_rect)) = center_h_rects {
            (src_rect.size(), dst_rect.size(), nine_patch.repeat_vertical)
        } else {
            // As above: no left/right edge, so the middle image is not scaled
            // vertically. The height call swaps width/height, so matching the
            // destination width to the source width gives a unit vertical scale.
            let dst = LayoutSize::new(center_src.size().width as f32, center_dst.height());
            (center_src.size(), dst, nine_patch.repeat_vertical)
        };
        compute_border_repetition_1d(
            LayoutSize::new(dst_size.height, dst_size.width),
            DeviceIntSize::new(src_size.height, src_size.width).to_f32(),
            repeat,
            &mut stretch_size.height,
            &mut spacing.height,
            &mut offset.y,
        );

        add_segment(
            &center_src,
            &center_dst,
            EdgeMask::empty(),
            stretch_size,
            spacing,
            offset,
        );
    }
}

// Boilerplate for outer segments.
// The central (fill) area needs to be handled differently
fn add_border_segment(
    src_rect: &DeviceIntRect,
    dst_rect: &LayoutRect,
    side: EdgeMask,
    repeat_h: RepeatMode,
    repeat_v: RepeatMode,
    add_segment: &mut dyn FnMut(
        &DeviceIntRect, // src rect
        &LayoutRect,    // dst rect
        EdgeMask,       // segment side
        LayoutSize,     // stretch size
        LayoutSize,     // spacing
        LayoutVector2D, // offset
    ),
) -> bool {
    if src_rect.is_empty() || dst_rect.is_empty() {
        return false
    }

    let mut stretch_size = dst_rect.size();
    let mut spacing = LayoutSize::zero();
    let mut offset = LayoutVector2D::zero();
    crate::border::compute_border_repetition(
        dst_rect.size(),
        src_rect.size().to_f32(),
        repeat_h,
        repeat_v,
        &mut stretch_size,
        &mut spacing,
        &mut offset,
    );

    add_segment(
        src_rect,
        dst_rect,
        side,
        stretch_size,
        spacing,
        offset,
    );

    true
}
