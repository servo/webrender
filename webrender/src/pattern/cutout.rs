/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Cutouts patterns act a mask on top of already rendered content.
//!
//! Their main purpose is to "punch a hole" in the current layer so that
//! an underlay layer can be seen through it.
//! The clip chain controls the shape of the mask. When rendering the
//! cutout, opaque white means pixels will be fully masked, transparent
//! black means pixels will be shown, and values in between represent
//! different levels of transparency.

use api::{ColorF, units::*};

use crate::pattern::{Pattern, PatternBuilder, PatternBuilderContext, PatternBuilderState};
use crate::renderer::BlendMode;

pub struct Cutout;

impl PatternBuilder for Cutout {
    fn build(
        &self,
        _sub_rect: Option<DeviceRect>,
        _offset: LayoutVector2D,
        _ctx: &PatternBuilderContext,
        _state: &mut PatternBuilderState,
    ) -> Pattern {
        Pattern::color(ColorF::WHITE)
            .with_blend_mode(BlendMode::PremultipliedDestOut)
    }
}

// TODO: Cutouts are applied to the destination layer entirely in the alpha
// pass, the fully masked segments could be drawn in the opaque pass.

// TODO: Currently, complex masks for cutouts are drawn into intermediate
// surfaces using the same logic as regular patterns. They could be drawn
// directly into the the destination layer. 
