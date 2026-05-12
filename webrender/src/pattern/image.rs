/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::units::*;

use crate::pattern::{Pattern, PatternBuilder, PatternBuilderContext, PatternBuilderState};
use crate::render_task_graph::RenderTaskId;

pub struct ImagePattern {
    pub src_task_id: RenderTaskId,
    pub src_is_opaque: bool,
    // pub color: ColorF, // TODO
}

impl PatternBuilder for ImagePattern {
    fn build(
        &self,
        _sub_rect: Option<DeviceRect>,
        _offset: LayoutVector2D,
        _ctx: &PatternBuilderContext,
        _state: &mut PatternBuilderState,
    ) -> Pattern {
        Pattern::texture(self.src_task_id, self.src_is_opaque)
    }
}
