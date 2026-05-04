/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{ColorF, ColorU, GradientStop};
use api::units::{LayoutRect, LayoutSize, LayoutVector2D};
use std::hash;

mod linear;
mod radial;
mod conic;

pub use linear::*;
pub use radial::*;
pub use conic::*;

/// A hashable gradient stop that can be used in primitive keys.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Copy, Clone, MallocSizeOf, PartialEq)]
pub struct GradientStopKey {
    pub offset: f32,
    pub color: ColorU,
}

impl GradientStopKey {
    pub fn empty() -> Self {
        GradientStopKey {
            offset: 0.0,
            color: ColorU::new(0, 0, 0, 0),
        }
    }
}

impl Into<GradientStopKey> for GradientStop {
    fn into(self) -> GradientStopKey {
        GradientStopKey {
            offset: self.offset,
            color: self.color.into(),
        }
    }
}

// Convert `stop_keys` into a vector of `GradientStop`s, which is a more
// convenient representation for the current gradient builder. Compute the
// minimum stop alpha along the way.
fn stops_and_min_alpha(stop_keys: &[GradientStopKey]) -> (Vec<GradientStop>, f32) {
    let mut min_alpha: f32 = 1.0;
    let stops = stop_keys.iter().map(|stop_key| {
        let color: ColorF = stop_key.color.into();
        min_alpha = min_alpha.min(color.a);

        GradientStop {
            offset: stop_key.offset,
            color,
        }
    }).collect();

    (stops, min_alpha)
}

impl Eq for GradientStopKey {}

impl hash::Hash for GradientStopKey {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.offset.to_bits().hash(state);
        self.color.hash(state);
    }
}

// If the gradient is not tiled we know that any content outside of the clip will not
// be shown. Applying the clip early reduces how much of the gradient we
// render and cache. We do this optimization separately on each axis.
// Returns the offset between the new and old primitive rect origin, to apply to the
// gradient parameters that are relative to the primitive origin.
pub fn apply_gradient_local_clip(
    prim_rect: &mut LayoutRect,
    stretch_size: &LayoutSize,
    tile_spacing: &LayoutSize,
    clip_rect: &LayoutRect,
) -> LayoutVector2D {
    let w = prim_rect.max.x.min(clip_rect.max.x) - prim_rect.min.x;
    let h = prim_rect.max.y.min(clip_rect.max.y) - prim_rect.min.y;
    let is_tiled_x = w > stretch_size.width + tile_spacing.width;
    let is_tiled_y = h > stretch_size.height + tile_spacing.height;

    let mut offset = LayoutVector2D::new(0.0, 0.0);

    if !is_tiled_x {
        let diff = (clip_rect.min.x - prim_rect.min.x).min(prim_rect.width());
        if diff > 0.0 {
            prim_rect.min.x += diff;
            offset.x = -diff;
        }

        let diff = prim_rect.max.x - clip_rect.max.x;
        if diff > 0.0 {
            prim_rect.max.x -= diff;
        }
    }

    if !is_tiled_y {
        let diff = (clip_rect.min.y - prim_rect.min.y).min(prim_rect.height());
        if diff > 0.0 {
            prim_rect.min.y += diff;
            offset.y = -diff;
        }

        let diff = prim_rect.max.y - clip_rect.max.y;
        if diff > 0.0 {
            prim_rect.max.y -= diff;
        }
    }

    offset
}

#[test]
#[cfg(target_pointer_width = "64")]
fn test_struct_sizes() {
    use std::mem;
    // The sizes of these structures are critical for performance on a number of
    // talos stress tests. If you get a failure here on CI, there's two possibilities:
    // (a) You made a structure smaller than it currently is. Great work! Update the
    //     test expectations and move on.
    // (b) You made a structure larger. This is not necessarily a problem, but should only
    //     be done with care, and after checking if talos performance regresses badly.
    assert_eq!(mem::size_of::<LinearGradient>(), 72, "LinearGradient size changed");
    assert_eq!(mem::size_of::<LinearGradientTemplate>(), 88, "LinearGradientTemplate size changed");
    assert_eq!(mem::size_of::<LinearGradientKey>(), 80, "LinearGradientKey size changed");

    assert_eq!(mem::size_of::<RadialGradient>(), 72, "RadialGradient size changed");
    assert_eq!(mem::size_of::<RadialGradientTemplate>(), 88, "RadialGradientTemplate size changed");
    assert_eq!(mem::size_of::<RadialGradientKey>(), 88, "RadialGradientKey size changed");

    assert_eq!(mem::size_of::<ConicGradient>(), 72, "ConicGradient size changed");
    assert_eq!(mem::size_of::<ConicGradientTemplate>(), 88, "ConicGradientTemplate size changed");
    assert_eq!(mem::size_of::<ConicGradientKey>(), 88, "ConicGradientKey size changed");
}
