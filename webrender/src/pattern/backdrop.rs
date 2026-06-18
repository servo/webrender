/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::ColorF;
use api::units::*;

use crate::pattern::{Pattern, PatternBuilder, PatternBuilderContext, PatternBuilderState, PatternKind, PatternShaderInput, PatternTextureInput};
use crate::render_task_graph::RenderTaskId;
use crate::renderer::BlendMode;

/// Pattern that samples a captured backdrop texture.
///
/// The backdrop may be in a different coordinate space than the primitive, so
/// the sampling position is described by four homogeneous screen-space uv
/// corners (top-left, top-right, bottom-left, bottom-right) that the shader
/// bilinearly interpolates across the primitive rect. The interpolated value is
/// then mapped into the backdrop's texture-cache rect (provided as the standard
/// quad uv rect via the source render task).
pub struct BackdropPattern {
    pub src_task_id: RenderTaskId,
    /// Homogeneous screen-space uv corners: [top_left, top_right, bottom_left, bottom_right].
    pub uvs: [DeviceHomogeneousVector; 4],
}

impl PatternBuilder for BackdropPattern {
    fn build(
        &self,
        _sub_rect: Option<DeviceRect>,
        _offset: LayoutVector2D,
        _ctx: &PatternBuilderContext,
        state: &mut PatternBuilderState,
    ) -> Pattern {
        // See fetch in ps_quad_backdrop.glsl.
        let mut writer = state.frame_gpu_data.f32.write_blocks(4);
        for uv in &self.uvs {
            writer.push_one([uv.x, uv.y, uv.z, uv.w]);
        }
        let addr = writer.finish();

        Pattern {
            kind: PatternKind::Backdrop,
            shader_input: PatternShaderInput(addr.as_int(), 0),
            texture_input: PatternTextureInput::new(self.src_task_id),
            base_color: ColorF::WHITE,
            is_opaque: false,
            blend_mode: BlendMode::PremultipliedAlpha,
        }
    }
}
