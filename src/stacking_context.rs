/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use display_list::{AuxiliaryListsBuilder, ItemRange};
use euclid::{Matrix4, Rect};
use types::{DisplayListId, FilterOp, MixBlendMode, ScrollLayerId, ScrollPolicy};

#[derive(Serialize, Deserialize)]
pub struct StackingContext {
    pub scroll_layer_id: Option<ScrollLayerId>,
    pub scroll_policy: ScrollPolicy,
    pub bounds: Rect<f32>,
    pub overflow: Rect<f32>,
    pub z_index: i32,
    pub display_lists: Vec<DisplayListId>,
    pub transform: Matrix4,
    pub perspective: Matrix4,
    pub establishes_3d_context: bool,
    pub mix_blend_mode: MixBlendMode,
    pub filters: ItemRange,
    pub has_stacking_contexts: bool,
}

impl StackingContext {
    pub fn new(scroll_layer_id: Option<ScrollLayerId>,
               scroll_policy: ScrollPolicy,
               bounds: Rect<f32>,
               overflow: Rect<f32>,
               z_index: i32,
               transform: &Matrix4,
               perspective: &Matrix4,
               establishes_3d_context: bool,
               mix_blend_mode: MixBlendMode,
               filters: Vec<FilterOp>,
               auxiliary_lists_builder: &mut AuxiliaryListsBuilder)
               -> StackingContext {
        StackingContext {
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
