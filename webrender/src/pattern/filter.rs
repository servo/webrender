/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::ColorF;
use api::units::*;

use crate::pattern::{
    Pattern, PatternBuilder, PatternBuilderContext, PatternBuilderState, PatternKind,
    PatternShaderInput, PatternTextureInput,
};
use crate::render_task_graph::RenderTaskId;
use crate::renderer::BlendMode;

/// Applies a CSS/SVG filter to a source render task via the ps_quad_blend
/// shader.
pub struct BlendFilterPattern {
    pub src_task_id: RenderTaskId,
    /// The filter mode (see `Filter::as_int`) in the low 16 bits, plus the
    /// per-channel component-transfer functions in the high bits.
    pub filter_mode: i32,
    /// Either the filter "amount" (fixed point, scaled by 65536) for scalar
    /// filters, or a GPU buffer address for matrix / component-transfer filters.
    pub param: i32,
}

impl PatternBuilder for BlendFilterPattern {
    fn build(
        &self,
        _sub_rect: Option<DeviceRect>,
        _offset: LayoutVector2D,
        _ctx: &PatternBuilderContext,
        _state: &mut PatternBuilderState,
    ) -> Pattern {
        Pattern {
            kind: PatternKind::Blend,
            shader_input: PatternShaderInput(self.filter_mode, self.param),
            texture_input: PatternTextureInput::new(self.src_task_id),
            base_color: ColorF::WHITE,
            // Filters may change the alpha (e.g. color matrix, flood), so the
            // result is treated as translucent.
            is_opaque: false,
            blend_mode: BlendMode::PremultipliedAlpha,
        }
    }
}
