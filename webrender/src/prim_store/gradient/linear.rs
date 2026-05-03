/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Linear gradients
//!
//! Specification: https://drafts.csswg.org/css-images-4/#linear-gradients
//!
//! Linear gradients are rendered via cached render tasks and composited with the image brush.

use euclid::approxeq::ApproxEq;
use euclid::{point2, vec2};
use api::{ExtendMode, GradientStop};
use api::units::*;
use crate::pattern::gradient::linear_gradient_pattern;
use crate::pattern::{Pattern, PatternBuilder, PatternBuilderContext, PatternBuilderState};
use crate::render_backend::DataStores;
use crate::scene_building::IsVisible;
use crate::intern::{Internable, InternDebug, Handle as InternHandle};
use crate::internal_types::LayoutPrimitiveInfo;
use crate::image_tiling::simplify_repeated_primitive;
use crate::prim_store::BrushSegment;
use crate::prim_store::{PrimitiveInstanceIndex, PrimitiveKind, PrimitiveOpacity, PrimitiveScratchBuffer};
use crate::prim_store::{PrimKeyCommonData, PrimTemplateCommonData, PrimitiveStore};
use crate::prim_store::{NinePatchDescriptor, PointKey, SizeKey, InternablePrimitive};
use crate::prim_store::storage;
use crate::segment::EdgeMask;
use crate::visibility::KindScratchHandle;
use super::{stops_and_min_alpha, GradientStopKey, apply_gradient_local_clip};
use std::ops::{Deref, DerefMut};
use std::mem::swap;

pub const MAX_CACHED_SIZE: f32 = 1024.0;

/// Identifying key for a linear gradient.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Clone, Eq, PartialEq, Hash, MallocSizeOf)]
pub struct LinearGradientKey {
    pub common: PrimKeyCommonData,
    pub extend_mode: ExtendMode,
    pub start_point: PointKey,
    pub end_point: PointKey,
    pub stretch_size: SizeKey,
    pub tile_spacing: SizeKey,
    pub stops: Vec<GradientStopKey>,
    pub reverse_stops: bool,
    pub nine_patch: Option<Box<NinePatchDescriptor>>,
    pub enable_dithering: bool,
}

impl LinearGradientKey {
    pub fn new(
        info: &LayoutPrimitiveInfo,
        linear_grad: LinearGradient,
    ) -> Self {
        LinearGradientKey {
            common: info.into(),
            extend_mode: linear_grad.extend_mode,
            start_point: linear_grad.start_point,
            end_point: linear_grad.end_point,
            stretch_size: linear_grad.stretch_size,
            tile_spacing: linear_grad.tile_spacing,
            stops: linear_grad.stops,
            reverse_stops: linear_grad.reverse_stops,
            nine_patch: linear_grad.nine_patch,
            enable_dithering: linear_grad.enable_dithering,
        }
    }
}

impl InternDebug for LinearGradientKey {}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, MallocSizeOf)]
pub struct LinearGradientTemplate {
    pub common: PrimTemplateCommonData,
    pub extend_mode: ExtendMode,
    pub start_point: LayoutPoint,
    pub end_point: LayoutPoint,
    pub task_size: DeviceIntSize,
    pub scale: DeviceVector2D,
    pub stretch_size: LayoutSize,
    pub tile_spacing: LayoutSize,
    pub stops_opacity: PrimitiveOpacity,
    pub stops: Vec<GradientStop>,
    pub border_nine_patch: Option<Box<NinePatchDescriptor>>,
    pub reverse_stops: bool,
}

/// Per-frame scratch data for a LinearGradient primitive whose
/// template has a `border_nine_patch`. Non-nine-patch gradients have
/// no per-frame brush segments and skip the scratch entry entirely.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
pub struct LinearGradientScratch {
    /// Range into `PrimitiveFrameScratch::segments` holding the per-
    /// frame nine-patch brush segments for this gradient. Built fresh
    /// each frame against the prim's current size in
    /// `prepare_prim_for_render`.
    pub brush_segments_range: storage::Range<BrushSegment>,
}

impl LinearGradientScratch {
    /// Build the per-frame nine-patch brush segments for a
    /// LinearGradient prim that has a `border_nine_patch`. No-op for
    /// gradients without a nine_patch (no segments are built and no
    /// scratch entry is pushed).
    ///
    /// Called from the prep early pass before `update_clip_task` runs,
    /// since `update_clip_task_for_brush` reads the brush segments via
    /// the scratch entry allocated here.
    pub fn build_for_prim(
        data_handle: LinearGradientDataHandle,
        prim_instance_index: PrimitiveInstanceIndex,
        data_stores: &DataStores,
        scratch: &mut PrimitiveScratchBuffer,
    ) {
        let prim_data = &data_stores.linear_grad[data_handle];
        let nine_patch = match prim_data.border_nine_patch.as_deref() {
            Some(np) => np,
            None => return,
        };
        let prim_size = prim_data.common.prim_size;

        let brush_open = scratch.frame.segments.open_range();
        scratch.frame.segments.data_mut().extend(
            nine_patch.create_brush_segments(prim_size),
        );
        let brush_segments_range = scratch.frame.segments.close_range(brush_open);

        let handle = scratch.frame.linear_gradient.push(LinearGradientScratch {
            brush_segments_range,
        });
        scratch.frame.draws[prim_instance_index.0 as usize].kind_scratch =
            KindScratchHandle::LinearGradient(handle);
    }
}

impl PatternBuilder for LinearGradientTemplate {
    fn build(
        &self,
        _sub_rect: Option<DeviceRect>,
        offset: LayoutVector2D,
        ctx: &PatternBuilderContext,
        state: &mut PatternBuilderState,
    ) -> Pattern {
        let (start, end) = if self.reverse_stops {
            (self.end_point, self.start_point)
        } else {
            (self.start_point, self.end_point)
        };
        // LinearGradientTemplate stores the start and end points relative to the
        // primitive origin, but the shader works with start/end points in "proper"
        // layout coordinates (relative to the primitive's spatial node).
        let offset = offset + ctx.prim_origin.to_vector();
        linear_gradient_pattern(
            start + offset,
            end + offset,
            self.extend_mode,
            &self.stops,
            ctx.fb_config.is_software,
            state.frame_gpu_data,
        )
    }
}

impl Deref for LinearGradientTemplate {
    type Target = PrimTemplateCommonData;
    fn deref(&self) -> &Self::Target {
        &self.common
    }
}

impl DerefMut for LinearGradientTemplate {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.common
    }
}

/// Perform a few optimizations to the gradient that are relevant to scene building.
///
/// Returns true if the gradient was decomposed into fast-path primitives, indicating
/// that we shouldn't emit a regular gradient primitive after this returns.
pub fn optimize_linear_gradient(
    prim_rect: &mut LayoutRect,
    tile_size: &mut LayoutSize,
    mut tile_spacing: LayoutSize,
    clip_rect: &LayoutRect,
    start: &mut LayoutPoint,
    end: &mut LayoutPoint,
    extend_mode: ExtendMode,
    stops: &mut [GradientStopKey],
    enable_dithering: bool,
    // Callback called for each fast-path segment (rect, start end, stops).
    callback: &mut dyn FnMut(&LayoutRect, LayoutPoint, LayoutPoint, &[GradientStopKey], EdgeMask)
) -> bool {
    // First sanitize the gradient parameters. See if we can remove repetitions,
    // tighten the primitive bounds, etc.

    simplify_repeated_primitive(&tile_size, &mut tile_spacing, prim_rect);

    let vertical = start.x.approx_eq(&end.x);
    let horizontal = start.y.approx_eq(&end.y);

    let mut horizontally_tiled = prim_rect.width() > tile_size.width;
    let mut vertically_tiled = prim_rect.height() > tile_size.height;

    // Check whether the tiling is equivalent to stretching on either axis.
    // Stretching the gradient is more efficient than repeating it.
    if vertically_tiled && horizontal && tile_spacing.height == 0.0 {
        tile_size.height = prim_rect.height();
        vertically_tiled = false;
    }

    if horizontally_tiled && vertical && tile_spacing.width == 0.0 {
        tile_size.width = prim_rect.width();
        horizontally_tiled = false;
    }

    let offset = apply_gradient_local_clip(
        prim_rect,
        &tile_size,
        &tile_spacing,
        &clip_rect
    );

    // The size of gradient render tasks depends on the tile_size. No need to generate
    // large stretch sizes that will be clipped to the bounds of the primitive.
    tile_size.width = tile_size.width.min(prim_rect.width());
    tile_size.height = tile_size.height.min(prim_rect.height());

    *start += offset;
    *end += offset;

    // Next, in the case of axis-aligned gradients, see if it is worth
    // decomposing the gradient into multiple gradients with only two
    // gradient stops per segment to get a faster shader.

    if extend_mode != ExtendMode::Clamp || stops.is_empty() {
        return false;
    }

    if !vertical && !horizontal {
        return false;
    }

    if vertical && horizontal {
        return false;
    }

    if !tile_spacing.is_empty() || vertically_tiled || horizontally_tiled {
        return false;
    }

    // If the gradient is small, no need to bother with decomposing it.
    if !enable_dithering &&
        ((horizontal && tile_size.width < 256.0)
        || (vertical && tile_size.height < 256.0)) {
        return false;
    }

    // Flip x and y if need be so that we only deal with the horizontal case.

    // From now on don't return false. We are going modifying the caller's
    // variables and not bother to restore them. If the control flow changes,
    // Make sure to to restore &mut parameters to sensible values before
    // returning false.

    let adjust_rect = &mut |rect: &mut LayoutRect| {
        if vertical {
            swap(&mut rect.min.x, &mut rect.min.y);
            swap(&mut rect.max.x, &mut rect.max.y);
        }
    };

    let adjust_size = &mut |size: &mut LayoutSize| {
        if vertical { swap(&mut size.width, &mut size.height); }
    };

    let adjust_point = &mut |p: &mut LayoutPoint| {
        if vertical { swap(&mut p.x, &mut p.y); }
    };

    let clip_rect = match clip_rect.intersection(prim_rect) {
        Some(clip) => clip,
        None => {
            return false;
        }
    };

    adjust_rect(prim_rect);
    adjust_point(start);
    adjust_point(end);
    adjust_size(tile_size);

    let length = (end.x - start.x).abs();

    // Decompose the gradient into simple segments. This lets us:
    // - separate opaque from semi-transparent segments,
    // - compress long segments into small render tasks,
    // - make sure hard stops stay so even if the primitive is large.

    let reverse_stops = start.x > end.x;

    // Handle reverse stops so we can assume stops are arranged in increasing x.
    if reverse_stops {
        stops.reverse();
        swap(start, end);
    }

    // Use fake gradient stop to emulate the potential constant color sections
    // before and after the gradient endpoints.
    let mut prev = *stops.first().unwrap();
    let mut last = *stops.last().unwrap();

    // Set the offsets of the fake stops to position them at the edges of the primitive.
    prev.offset = -start.x / length;
    last.offset = (tile_size.width - start.x) / length;
    if reverse_stops {
        prev.offset = 1.0 - prev.offset;
        last.offset = 1.0 - last.offset;
    }

    let (side_edges, first_edge, last_edge) = if vertical {
        (
            EdgeMask::LEFT | EdgeMask::RIGHT,
            EdgeMask::TOP,
            EdgeMask::BOTTOM
        )
    } else {
        (
            EdgeMask::TOP | EdgeMask::BOTTOM,
            EdgeMask::LEFT,
            EdgeMask::RIGHT
        )
    };

    let mut is_first = true;
    let last_offset = last.offset;
    for stop in stops.iter().chain((&[last]).iter()) {
        let prev_stop = prev;
        prev = *stop;

        if prev_stop.color.a == 0 && stop.color.a == 0 {
            continue;
        }


        let prev_offset = if reverse_stops { 1.0 - prev_stop.offset } else { prev_stop.offset };
        let offset = if reverse_stops { 1.0 - stop.offset } else { stop.offset };

        // In layout space, relative to the primitive.
        let segment_start = start.x + prev_offset * length;
        let segment_end = start.x + offset * length;
        let segment_length = segment_end - segment_start;

        if segment_length <= 0.0 {
            continue;
        }

        let mut segment_rect = *prim_rect;
        segment_rect.min.x += segment_start;
        segment_rect.max.x = segment_rect.min.x + segment_length;

        let mut start = point2(0.0, 0.0);
        let mut end = point2(segment_length, 0.0);

        adjust_point(&mut start);
        adjust_point(&mut end);
        adjust_rect(&mut segment_rect);

        let origin_before_clip = segment_rect.min;
        segment_rect = match segment_rect.intersection(&clip_rect) {
            Some(rect) => rect,
            None => {
                continue;
            }
        };
        let offset = segment_rect.min - origin_before_clip;

        // Account for the clipping since start and end are relative to the origin.
        start -= offset;
        end -= offset;

        let mut edge_flags = side_edges;
        if is_first {
            edge_flags |= first_edge;
            is_first = false;
        }
        if stop.offset == last_offset {
            edge_flags |= last_edge;
        }

        callback(
            &segment_rect,
            start,
            end,
            &[
                GradientStopKey { offset: 0.0, .. prev_stop },
                GradientStopKey { offset: 1.0, .. *stop },
            ],
            edge_flags,
        );
    }

    true
}

impl From<LinearGradientKey> for LinearGradientTemplate {
    fn from(item: LinearGradientKey) -> Self {

        let common = PrimTemplateCommonData::with_key_common(item.common);

        let (stops, min_alpha) = stops_and_min_alpha(&item.stops);

        // Save opacity of the stops for use in
        // selecting which pass this gradient
        // should be drawn in.
        let stops_opacity = PrimitiveOpacity::from_alpha(min_alpha);

        let start_point = LayoutPoint::new(item.start_point.x, item.start_point.y);
        let end_point = LayoutPoint::new(item.end_point.x, item.end_point.y);
        let tile_spacing: LayoutSize = item.tile_spacing.into();
        let stretch_size: LayoutSize = item.stretch_size.into();
        let mut task_size: DeviceSize = stretch_size.cast_unit();

        let horizontal = !item.enable_dithering &&
            start_point.y.approx_eq(&end_point.y);
        let vertical = !item.enable_dithering &&
            start_point.x.approx_eq(&end_point.x);

        if horizontal {
            // Completely horizontal, we can stretch the gradient vertically.
            task_size.height = 1.0;
        }

        if vertical {
            // Completely vertical, we can stretch the gradient horizontally.
            task_size.width = 1.0;
        }

        // Avoid rendering enormous gradients. Linear gradients are mostly made of soft transitions,
        // so it is unlikely that rendering at a higher resolution than 1024 would produce noticeable
        // differences, especially with 8 bits per channel.

        let mut scale = vec2(1.0, 1.0);

        if task_size.width > MAX_CACHED_SIZE {
            scale.x = task_size.width / MAX_CACHED_SIZE;
            task_size.width = MAX_CACHED_SIZE;
        }

        if task_size.height > MAX_CACHED_SIZE {
            scale.y = task_size.height / MAX_CACHED_SIZE;
            task_size.height = MAX_CACHED_SIZE;
        }

        LinearGradientTemplate {
            common,
            extend_mode: item.extend_mode,
            start_point,
            end_point,
            task_size: task_size.ceil().to_i32(),
            scale,
            stretch_size,
            tile_spacing,
            stops_opacity,
            stops,
            border_nine_patch: item.nine_patch,
            reverse_stops: item.reverse_stops,
        }
    }
}

pub type LinearGradientDataHandle = InternHandle<LinearGradient>;

#[derive(Debug, MallocSizeOf)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct LinearGradient {
    pub extend_mode: ExtendMode,
    pub start_point: PointKey,
    pub end_point: PointKey,
    pub stretch_size: SizeKey,
    pub tile_spacing: SizeKey,
    pub stops: Vec<GradientStopKey>,
    pub reverse_stops: bool,
    pub nine_patch: Option<Box<NinePatchDescriptor>>,
    pub edge_aa_mask: EdgeMask,
    pub enable_dithering: bool,
}

impl Internable for LinearGradient {
    type Key = LinearGradientKey;
    type StoreData = LinearGradientTemplate;
    type InternData = ();
    const PROFILE_COUNTER: usize = crate::profiler::INTERNED_LINEAR_GRADIENTS;
}

impl InternablePrimitive for LinearGradient {
    fn into_key(
        self,
        info: &LayoutPrimitiveInfo,
    ) -> LinearGradientKey {
        LinearGradientKey::new(info, self)
    }

    fn make_instance_kind(
        _key: LinearGradientKey,
        data_handle: LinearGradientDataHandle,
        _prim_store: &mut PrimitiveStore,
    ) -> PrimitiveKind {
        PrimitiveKind::LinearGradient {
            data_handle,
        }
    }
}

impl IsVisible for LinearGradient {
    fn is_visible(&self) -> bool {
        true
    }
}

