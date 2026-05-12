/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! A pattern tha repeats another pattern.
//! See ps_quad_repeat.glsl.

use api::units::*;
use api::ColorF;
use crate::pattern::PatternBuilder;
use crate::pattern::PatternBuilderContext;
use crate::pattern::PatternBuilderState;
use crate::pattern::{Pattern, PatternKind, PatternShaderInput, PatternTextureInput};
use crate::render_task_graph::RenderTaskId;
use crate::renderer::GpuBufferBuilder;

pub struct RepeatedPattern {
    pub stretch_size: LayoutSize,
    pub spacing: LayoutSize,
    pub src_task_id: RenderTaskId,
    pub src_is_opaque: bool,
}

pub fn repeated_pattern(
    repeat: &RepeatedPattern,
    gpu_buffer_builder: &mut GpuBufferBuilder,
) -> Pattern {
    let mut writer = gpu_buffer_builder.f32.write_blocks(1);
    writer.push_one([
        repeat.stretch_size.width,
        repeat.stretch_size.height,
        repeat.spacing.width,
        repeat.spacing.height,
    ]);
    let repeat_address = writer.finish();

    Pattern {
        kind: PatternKind::Repeat,
        shader_input: PatternShaderInput(
            repeat_address.as_int(),
            0,
        ),
        texture_input: PatternTextureInput::new(repeat.src_task_id),
        base_color: ColorF::WHITE,
        is_opaque: repeat.src_is_opaque
            && repeat.spacing.width <= 0.0
            && repeat.spacing.height <= 0.0,
    }
}

impl PatternBuilder for RepeatedPattern {
    fn build(
        &self,
        _sub_rect: Option<DeviceRect>,
        _offset: LayoutVector2D,
        _ctx: &PatternBuilderContext,
        state: &mut PatternBuilderState,
    ) -> Pattern {
        repeated_pattern(self, state.frame_gpu_data)
    }
}
