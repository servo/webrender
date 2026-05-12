/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{BorderRadius, BoxShadowClipMode};
use api::units::*;
use api::ColorF;
use crate::pattern::{Pattern, PatternBuilder, PatternBuilderContext, PatternBuilderState, PatternKind, PatternShaderInput, PatternTextureInput};
use crate::render_task_graph::RenderTaskId;

pub struct BoxShadowPatternData {
    pub color: ColorF,
    pub render_task: RenderTaskId,
    /// Full blur alloc size in local pixels (= 2*blur_region + src_rect_size per axis).
    /// Used as the UV denominator so shadow_pos/alloc_size maps 1:1 to texture position.
    pub shadow_rect_alloc_size: LayoutSize,
    /// Size of dest_rect in local pixels. For outset this equals shadow_rect_alloc_size
    /// (prim_rect == dest_rect). For inset the prim is the element rect while dest_rect
    /// is smaller (the shadow area), so these differ.
    pub dest_rect_size: LayoutSize,
    /// Offset of dest_rect.p0 relative to prim_rect.p0 in local pixels.
    /// Zero for outset. For inset: dest_rect.min - element_rect.min.
    pub dest_rect_offset: LayoutVector2D,
    pub clip_mode: BoxShadowClipMode,
    /// Offset of the element rect's min corner relative to prim_rect.p0.
    /// Zero for inset (prim_rect IS the element rect). For outset: element_rect.min - dest_rect.min.
    pub element_offset_rel_prim: LayoutVector2D,
    pub element_size: LayoutSize,
    pub element_radius: BorderRadius,
}

impl PatternBuilder for BoxShadowPatternData {
    fn build(
        &self,
        _sub_rect: Option<DeviceRect>,
        _offset: LayoutVector2D,
        _ctx: &PatternBuilderContext,
        state: &mut PatternBuilderState,
    ) -> Pattern {
        let mut writer = state.frame_gpu_data.f32.write_blocks(5);
        writer.push_one([
            self.shadow_rect_alloc_size.width,
            self.shadow_rect_alloc_size.height,
            self.dest_rect_size.width,
            self.dest_rect_size.height,
        ]);
        writer.push_one([
            self.dest_rect_offset.x,
            self.dest_rect_offset.y,
            if self.clip_mode == BoxShadowClipMode::Inset { 1.0 } else { 0.0 },
            0.0,
        ]);
        writer.push_one([
            self.element_offset_rel_prim.x,
            self.element_offset_rel_prim.y,
            self.element_size.width,
            self.element_size.height,
        ]);
        writer.push_one([
            self.element_radius.top_left.width,
            self.element_radius.top_left.height,
            self.element_radius.top_right.width,
            self.element_radius.top_right.height,
        ]);
        writer.push_one([
            self.element_radius.bottom_right.width,
            self.element_radius.bottom_right.height,
            self.element_radius.bottom_left.width,
            self.element_radius.bottom_left.height,
        ]);
        let addr = writer.finish();

        Pattern {
            kind: PatternKind::BoxShadow,
            shader_input: PatternShaderInput(addr.as_int(), 0),
            texture_input: PatternTextureInput::new(self.render_task),
            base_color: self.color,
            is_opaque: false,
        }
    }
}
