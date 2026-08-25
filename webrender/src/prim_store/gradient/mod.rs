/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{ColorF, GradientStop};

mod linear;
mod radial;
mod conic;

pub use linear::*;
pub use radial::*;
pub use conic::*;

// `GradientStopKey` now lives in `webrender_api` so builder-side interning keys
// can reference it. Re-exported here to keep existing references working.
pub use api::key_types::GradientStopKey;

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

// If the gradient is not tiled we know that any content outside of the clip will not
// be shown. Applying the clip early reduces how much of the gradient we
// render and cache. We do this optimization separately on each axis.
// Returns the offset between the new and old primitive rect origin, to apply to the
// gradient parameters that are relative to the primitive origin.
// `apply_gradient_local_clip` now lives in `webrender_api::prim_geometry` so
// content-process interning can share it. Re-exported here to keep existing
// references working.
pub use api::prim_geometry::apply_gradient_local_clip;

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
    assert_eq!(mem::size_of::<LinearGradientTemplate>(), 72, "LinearGradientTemplate size changed");
    assert_eq!(mem::size_of::<LinearGradientKey>(), 72, "LinearGradientKey size changed");

    assert_eq!(mem::size_of::<RadialGradient>(), 72, "RadialGradient size changed");
    assert_eq!(mem::size_of::<RadialGradientTemplate>(), 80, "RadialGradientTemplate size changed");
    assert_eq!(mem::size_of::<RadialGradientKey>(), 72, "RadialGradientKey size changed");

    assert_eq!(mem::size_of::<ConicGradient>(), 72, "ConicGradient size changed");
    assert_eq!(mem::size_of::<ConicGradientTemplate>(), 80, "ConicGradientTemplate size changed");
    assert_eq!(mem::size_of::<ConicGradientKey>(), 72, "ConicGradientKey size changed");
}
