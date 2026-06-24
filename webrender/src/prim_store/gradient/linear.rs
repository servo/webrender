/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Linear gradients
//!
//! Specification: https://drafts.csswg.org/css-images-4/#linear-gradients
//!
//! Linear gradients are rendered via cached render tasks and composited with the image brush.

use euclid::approxeq::ApproxEq;
use euclid::point2;
use api::{ExtendMode, GradientStop};
use api::units::*;
use crate::pattern::gradient::linear_gradient_pattern;
use crate::pattern::{Pattern, PatternBuilder, PatternBuilderContext, PatternBuilderState};
use crate::scene_building::IsVisible;
use crate::intern::{Internable, InternDebug, Handle as InternHandle};
use crate::internal_types::LayoutPrimitiveInfo;
use crate::image_tiling::simplify_repeated_primitive;
use crate::prim_store::{PrimitiveKind, PrimitiveOpacity};
use crate::prim_store::{PrimKeyCommonData, PrimTemplateCommonData, PrimitiveStore};
use crate::prim_store::{NinePatchDescriptor, PointKey, SizeKey, InternablePrimitive};
use crate::segment::EdgeMask;
use super::{stops_and_min_alpha, GradientStopKey, apply_gradient_local_clip};
use std::ops::{Deref, DerefMut};
use std::mem::swap;

/// Identifying key for a linear gradient.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Clone, Eq, PartialEq, Hash, MallocSizeOf)]
pub struct LinearGradientKey {
    pub common: PrimKeyCommonData,
    pub extend_mode: ExtendMode,
    pub start_point: PointKey,
    pub end_point: PointKey,
    /// Per-axis tile size encoded as a fraction of `common.prim_size`. The
    /// runtime `stretch_size` is `stretch_ratio * common.prim_size`.
    pub stretch_ratio: SizeKey,
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
            stretch_ratio: linear_grad.stretch_ratio,
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
    /// Per-axis fraction of `common.prim_size` covered by one tile of the
    /// gradient pattern. Multiply by `common.prim_size` at use to recover the
    /// absolute stretch_size.
    pub stretch_ratio: LayoutSize,
    pub tile_spacing: LayoutSize,
    pub stops_opacity: PrimitiveOpacity,
    pub stops: Vec<GradientStop>,
    pub border_nine_patch: Option<Box<NinePatchDescriptor>>,
    pub reverse_stops: bool,
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
/// Mutates `prim_rect`, `tile_size`, `start`, `end` to bake in the simplifications
/// (repeated-tile collapse, equivalent-to-stretching on either axis, clip-induced
/// offsets). Decomposition into per-segment quads is no longer done here -- the
/// caller emits a single `LinearGradient` prim and prepare-time runs
/// [`decompose_axis_aligned_gradient`] against the snapped prim_rect when the
/// gradient is eligible. Doing the decomposition at frame-build keeps adjacent
/// segments phase-aligned with the snapped outer prim, even when the frame-time
/// snap pass nudges the outer rect.
pub fn optimize_linear_gradient(
    prim_rect: &mut LayoutRect,
    tile_size: &mut LayoutSize,
    mut tile_spacing: LayoutSize,
    clip_rect: &LayoutRect,
    start: &mut LayoutPoint,
    end: &mut LayoutPoint,
) {
    simplify_repeated_primitive(&tile_size, &mut tile_spacing, prim_rect);

    let vertical = start.x.approx_eq(&end.x);
    let horizontal = start.y.approx_eq(&end.y);

    let horizontally_tiled = prim_rect.width() > tile_size.width;
    let vertically_tiled = prim_rect.height() > tile_size.height;

    // Check whether the tiling is equivalent to stretching on either axis.
    // Stretching the gradient is more efficient than repeating it.
    if vertically_tiled && horizontal && tile_spacing.height == 0.0 {
        tile_size.height = prim_rect.height();
    }

    if horizontally_tiled && vertical && tile_spacing.width == 0.0 {
        tile_size.width = prim_rect.width();
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
}

/// Whether a linear gradient is eligible for the fast-path two-stop-per-segment
/// decomposition at prepare time. Inputs are the values produced by
/// `optimize_linear_gradient` (i.e. already simplified and clip-adjusted).
pub fn linear_gradient_decomposes(
    prim_rect: &LayoutRect,
    tile_size: LayoutSize,
    tile_spacing: LayoutSize,
    start: LayoutPoint,
    end: LayoutPoint,
    extend_mode: ExtendMode,
    stops: &[GradientStop],
    enable_dithering: bool,
) -> bool {
    if extend_mode != ExtendMode::Clamp || stops.is_empty() {
        return false;
    }

    let vertical = start.x.approx_eq(&end.x);
    let horizontal = start.y.approx_eq(&end.y);

    if !vertical && !horizontal {
        return false;
    }

    if vertical && horizontal {
        return false;
    }

    if !tile_spacing.is_empty() {
        return false;
    }

    let horizontally_tiled = prim_rect.width() > tile_size.width;
    let vertically_tiled = prim_rect.height() > tile_size.height;
    if vertically_tiled || horizontally_tiled {
        return false;
    }

    if !enable_dithering &&
        ((horizontal && tile_size.width < 256.0)
        || (vertical && tile_size.height < 256.0)) {
        return false;
    }

    true
}

/// Decompose an axis-aligned linear gradient into a sequence of two-stop
/// segments that tile end-to-end across `prim_rect`. Each callback invocation
/// is one segment, ready to be rendered as its own quad via the fast-path
/// gradient shader. Run at frame-build (against the snapped prim_rect) so
/// adjacent segments share a snapped boundary and tile without phase drift.
///
/// Caller must have verified eligibility via [`linear_gradient_decomposes`].
pub fn decompose_axis_aligned_gradient(
    prim_rect: &LayoutRect,
    tile_size: LayoutSize,
    start: LayoutPoint,
    end: LayoutPoint,
    stops: &[GradientStop],
    clip_rect: &LayoutRect,
    mut callback: impl FnMut(&LayoutRect, LayoutPoint, LayoutPoint, [GradientStop; 2], EdgeMask),
) {
    debug_assert!(!stops.is_empty());

    let vertical = start.x.approx_eq(&end.x);

    // Flip x/y when the gradient is vertical so the remaining math treats it
    // as horizontal; un-flip per-segment outputs at the end.
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
        None => return,
    };

    let mut prim_rect = *prim_rect;
    let mut start = start;
    let mut end = end;
    let mut tile_size = tile_size;

    adjust_rect(&mut prim_rect);
    adjust_point(&mut start);
    adjust_point(&mut end);
    adjust_size(&mut tile_size);

    // `clip_rect` stays in the original (un-swapped) space — segment_rect
    // gets `adjust_rect` applied twice (once implicitly via the prim_rect
    // copy, once explicitly after computing per-segment extent) and lands
    // back in original space before this intersection.

    let length = (end.x - start.x).abs();

    // Match the pre-refactor optimiser: when the gradient line points in
    // decreasing-x (post-axis-swap), swap start/end and walk the stop list
    // in reverse, so the loop always processes stops in increasing-x
    // order. The pre-refactor code did this via `stops.reverse()` in
    // place; we can't mutate the template's stops here, so use a reversed
    // iterator and swap which end of the slice supplies the fake-stop
    // colour accordingly.
    let reverse_stops = start.x > end.x;
    if reverse_stops {
        swap(&mut start, &mut end);
    }

    let (first_stop, last_stop) = if reverse_stops {
        (*stops.last().unwrap(), *stops.first().unwrap())
    } else {
        (*stops.first().unwrap(), *stops.last().unwrap())
    };

    let mut prev = first_stop;
    let mut last = last_stop;
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
            EdgeMask::BOTTOM,
        )
    } else {
        (
            EdgeMask::TOP | EdgeMask::BOTTOM,
            EdgeMask::LEFT,
            EdgeMask::RIGHT,
        )
    };

    let mut is_first = true;
    let last_offset = last.offset;

    // Iterate stops in increasing-x order. When reverse_stops is set, walk the
    // backing slice in reverse instead of mutating it.
    let stops_iter: Box<dyn Iterator<Item = &GradientStop>> = if reverse_stops {
        Box::new(stops.iter().rev())
    } else {
        Box::new(stops.iter())
    };

    for stop in stops_iter.chain(std::iter::once(&last)) {
        let prev_stop = prev;
        prev = *stop;

        if prev_stop.color.a == 0.0 && stop.color.a == 0.0 {
            continue;
        }

        let prev_offset = if reverse_stops { 1.0 - prev_stop.offset } else { prev_stop.offset };
        let offset = if reverse_stops { 1.0 - stop.offset } else { stop.offset };

        // Segment_start and segment_end are in the gradient's pre-flip space
        // (relative to the prim's origin); the adjust_* helpers below restore
        // axis orientation when emitting.
        let segment_start = start.x + prev_offset * length;
        let segment_end = start.x + offset * length;
        let segment_length = segment_end - segment_start;

        if segment_length <= 0.0 {
            continue;
        }

        let mut segment_rect = prim_rect;
        segment_rect.min.x += segment_start;
        segment_rect.max.x = segment_rect.min.x + segment_length;

        let mut seg_start = point2(0.0, 0.0);
        let mut seg_end = point2(segment_length, 0.0);

        adjust_point(&mut seg_start);
        adjust_point(&mut seg_end);
        adjust_rect(&mut segment_rect);

        let origin_before_clip = segment_rect.min;
        segment_rect = match segment_rect.intersection(&clip_rect) {
            Some(rect) => rect,
            None => continue,
        };
        let clip_offset = segment_rect.min - origin_before_clip;
        seg_start -= clip_offset;
        seg_end -= clip_offset;

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
            seg_start,
            seg_end,
            [
                GradientStop { offset: 0.0, color: prev_stop.color },
                GradientStop { offset: 1.0, color: stop.color },
            ],
            edge_flags,
        );
    }
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
        let stretch_ratio: LayoutSize = item.stretch_ratio.into();

        LinearGradientTemplate {
            common,
            extend_mode: item.extend_mode,
            start_point,
            end_point,
            stretch_ratio,
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
    /// Per-axis tile size encoded as a fraction of the prim's size. See
    /// [`LinearGradientKey::stretch_ratio`].
    pub stretch_ratio: SizeKey,
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

