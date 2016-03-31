/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use aabbtree::AABBTreeNode;
use batch::{BatchBuilder, VertexBuffer};
use fnv::FnvHasher;
use frame::{DrawListGroup, FrameId};
use internal_types::{DrawListItemIndex, CompiledNode, StackingContextInfo};
use internal_types::{BatchList, StackingContextIndex};
use internal_types::{DrawListGroupId};
use std::hash::BuildHasherDefault;
use resource_cache::ResourceCache;
use std::collections::HashMap;
use webrender_traits::{AuxiliaryLists, PipelineId, SpecificDisplayItem};

pub trait NodeCompiler {
    fn compile(&mut self,
               resource_cache: &ResourceCache,
               frame_id: FrameId,
               device_pixel_ratio: f32,
               stacking_context_info: &[StackingContextInfo],
               draw_list_groups: &HashMap<DrawListGroupId,
                                          DrawListGroup,
                                          BuildHasherDefault<FnvHasher>>,
               pipeline_auxiliary_lists: &HashMap<PipelineId,
                                                  AuxiliaryLists,
                                                  BuildHasherDefault<FnvHasher>>);
}

impl NodeCompiler for AABBTreeNode {
    fn compile(&mut self,
               resource_cache: &ResourceCache,
               frame_id: FrameId,
               device_pixel_ratio: f32,
               stacking_context_info: &[StackingContextInfo],
               draw_list_groups: &HashMap<DrawListGroupId,
                                          DrawListGroup,
                                          BuildHasherDefault<FnvHasher>>,
               pipeline_auxiliary_lists: &HashMap<PipelineId,
                                                  AuxiliaryLists,
                                                  BuildHasherDefault<FnvHasher>>) {
        let mut compiled_node = CompiledNode::new();
        let mut vertex_buffer = VertexBuffer::new();

        for draw_list_group_segment in &self.draw_list_group_segments {
            let mut builder = BatchBuilder::new(&mut vertex_buffer, device_pixel_ratio);

            // TODO(gw): This is a HACK to fix matrix palette index offsets - there needs to
            //           be no holes in this array to match the draw group matrix palette. It's
            //           noticeable on wikipedia. Find a better solution to this!!!
            let draw_list_group = &draw_list_groups[&draw_list_group_segment.draw_list_group_id];

            for draw_list_id in &draw_list_group.draw_list_ids {
                let draw_list_index_buffer = draw_list_group_segment.index_buffers.iter().find(|ib| {
                    ib.draw_list_id == *draw_list_id
                });

                if let Some(draw_list_index_buffer) = draw_list_index_buffer {
                    let draw_list = resource_cache.get_draw_list(draw_list_index_buffer.draw_list_id);
                    let auxiliary_lists =
                        pipeline_auxiliary_lists.get(&draw_list.pipeline_id)
                                                .expect("No auxiliary lists for pipeline?!");

                    let StackingContextIndex(stacking_context_id) = draw_list.stacking_context_index.unwrap();
                    let context = &stacking_context_info[stacking_context_id];

                    let offset_from_layer = context.offset_from_layer;
                    builder.set_current_clip_rect_offset(offset_from_layer);

                    for index in &draw_list_index_buffer.indices {
                        let DrawListItemIndex(index) = *index;
                        let display_item = &draw_list.items[index as usize];

                        let clip_rect = display_item.clip.main.intersection(&context.local_clip_rect);

                        if let Some(ref clip_rect) = clip_rect {
                            builder.push_clip_in_rect(clip_rect);
                            builder.push_complex_clip(
                                auxiliary_lists.complex_clip_regions(&display_item.clip.complex));

                            match display_item.item {
                                SpecificDisplayItem::WebGL(ref info) => {
                                    builder.add_webgl_rectangle(&display_item.rect,
                                                                resource_cache,
                                                                &info.context_id,
                                                                frame_id);
                                }
                                SpecificDisplayItem::Image(ref info) => {
                                    builder.add_image(&display_item.rect,
                                                      &info.stretch_size,
                                                      info.image_key,
                                                      info.image_rendering,
                                                      resource_cache,
                                                      frame_id);
                                }
                                SpecificDisplayItem::Text(ref info) => {
                                    let glyphs = auxiliary_lists.glyph_instances(&info.glyphs);
                                    builder.add_text(&display_item.rect,
                                                     info.font_key,
                                                     info.size,
                                                     info.blur_radius,
                                                     &info.color,
                                                     &glyphs,
                                                     resource_cache,
                                                     frame_id,
                                                     device_pixel_ratio);
                                }
                                SpecificDisplayItem::Rectangle(ref info) => {
                                    builder.add_color_rectangle(&display_item.rect,
                                                                &info.color,
                                                                resource_cache,
                                                                frame_id);
                                }
                                SpecificDisplayItem::Gradient(ref info) => {
                                    builder.add_gradient(&display_item.rect,
                                                         &info.start_point,
                                                         &info.end_point,
                                                         &info.stops,
                                                         auxiliary_lists,
                                                         resource_cache,
                                                         frame_id);
                                }
                                SpecificDisplayItem::BoxShadow(ref info) => {
                                    builder.add_box_shadow(&info.box_bounds,
                                                           &info.offset,
                                                           &info.color,
                                                           info.blur_radius,
                                                           info.spread_radius,
                                                           info.border_radius,
                                                           info.clip_mode,
                                                           resource_cache,
                                                           frame_id);
                                }
                                SpecificDisplayItem::Border(ref info) => {
                                    builder.add_border(&display_item.rect,
                                                       info,
                                                       resource_cache,
                                                       frame_id,
                                                       device_pixel_ratio);
                                }
                            }

                            builder.pop_complex_clip();
                            builder.pop_clip_in_rect();
                        }
                    }
                }

                builder.next_draw_list();
            }

            let batches = builder.finalize();
            if !batches.is_empty() {
                compiled_node.batch_list.push(BatchList {
                    batches: batches,
                    draw_list_group_id: draw_list_group_segment.draw_list_group_id,
                });
            }
        }

        compiled_node.vertex_buffer = Some(vertex_buffer);
        self.compiled_node = Some(compiled_node);
    }
}
