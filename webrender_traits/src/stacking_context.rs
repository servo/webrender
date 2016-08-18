/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use display_list::AuxiliaryListsBuilder;
use euclid::{Matrix4D, Rect};
use {FilterOp, MixBlendMode, ScrollLayerId, ScrollPolicy};
use {ServoStackingContextId, StackingContext};

impl StackingContext {
    pub fn new(servo_id: ServoStackingContextId,
               scroll_layer_id: Option<ScrollLayerId>,
               scroll_policy: ScrollPolicy,
               bounds: Rect<f32>,
               overflow: Rect<f32>,
               z_index: i32,
               transform: &Matrix4D<f32>,
               perspective: &Matrix4D<f32>,
               establishes_3d_context: bool,
               mix_blend_mode: MixBlendMode,
               filters: Vec<FilterOp>,
               auxiliary_lists_builder: &mut AuxiliaryListsBuilder)
               -> StackingContext {
        StackingContext {
            servo_id: servo_id,
            scroll_layer_id: scroll_layer_id,
            scroll_policy: scroll_policy,
            bounds: bounds,
            overflow: overflow,
            z_index: z_index,
            display_lists: Vec::new(),
            transform: transform.clone(),
            perspective: perspective.clone(),
            establishes_3d_context: establishes_3d_context,
            mix_blend_mode: mix_blend_mode,
            filters: auxiliary_lists_builder.add_filters(&filters),
            has_stacking_contexts: false,
        }
    }
}
