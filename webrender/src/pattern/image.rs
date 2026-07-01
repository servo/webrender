/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{ImageBufferKind, ColorF, units::*};

use crate::pattern::{Pattern, PatternBuilder, PatternBuilderContext, PatternBuilderState, PatternKind};
use crate::render_task_graph::RenderTaskId;
use crate::renderer::BlendMode;

pub struct ImagePattern {
    pub src_task_id: RenderTaskId,
    pub src_is_opaque: bool,
    pub premultiplied: bool,
    pub sampler_kind: ImageBufferKind,
    pub color: ColorF,
}

impl PatternBuilder for ImagePattern {
    fn build(
        &self,
        _sub_rect: Option<DeviceRect>,
        _offset: LayoutVector2D,
        _ctx: &PatternBuilderContext,
        _state: &mut PatternBuilderState,
    ) -> Pattern {
        let blend_mode = if self.premultiplied || self.src_is_opaque {
            BlendMode::PremultipliedAlpha
        } else {
            BlendMode::Alpha
        };

        let mut pattern = Pattern::texture(self.src_task_id, self.src_is_opaque)
            .with_base_color(self.color)
            .with_blend_mode(blend_mode);

        pattern.kind = match self.sampler_kind {
            ImageBufferKind::Texture2D => PatternKind::ColorOrTexture,
            ImageBufferKind::TextureExternal => PatternKind::TextureExternal,
            ImageBufferKind::TextureExternalBT709 => PatternKind::TextureExternalBT709,
            ImageBufferKind::TextureRect => PatternKind::TextureRect,
        };

        pattern
    }
}
