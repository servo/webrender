/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{ColorF, MixBlendMode};
use api::units::*;

use crate::pattern::{
    Pattern, PatternBuilder, PatternBuilderContext, PatternBuilderState, PatternKind,
    PatternShaderInput, PatternTextureInput,
};
use crate::render_task_graph::RenderTaskId;
use crate::renderer::BlendMode;

/// A mix-blend-mode pattern that can't be expressed with a GPU blend equation.
/// Drawn via the ps_quad_mix_blend shader.
///
/// The backdrop (a readback render task) is sampled from slot 0 (sColor0).
/// The source (the picture's own content) is sampled from slot 1 (sColor1), its
/// texture-cache rect is written to the pattern's gpu block.
pub struct MixBlendPattern {
    pub backdrop_task_id: RenderTaskId,
    pub src_task_id: RenderTaskId,
    pub mode: MixBlendMode,
}

impl PatternBuilder for MixBlendPattern {
    fn build(
        &self,
        _sub_rect: Option<DeviceRect>,
        _offset: LayoutVector2D,
        _ctx: &PatternBuilderContext,
        state: &mut PatternBuilderState,
    ) -> Pattern {
        // See fetch in ps_quad_mix_blend.glsl: the source texture-cache rect,
        // resolved from its render task.
        let mut writer = state.frame_gpu_data.f32.write_blocks(1);
        writer.push_render_task(self.src_task_id);
        let addr = writer.finish();

        Pattern {
            kind: PatternKind::MixBlend,
            shader_input: PatternShaderInput(addr.as_int(), self.mode as u32 as i32),
            texture_input: PatternTextureInput {
                task_ids: [self.backdrop_task_id, self.src_task_id, RenderTaskId::INVALID],
            },
            base_color: ColorF::WHITE,
            is_opaque: false,
            blend_mode: BlendMode::PremultipliedAlpha,
        }
    }
}
