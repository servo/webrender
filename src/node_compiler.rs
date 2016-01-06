use aabbtree::AABBTreeNode;
use batch::{BatchBuilder, VertexBuffer};
use clipper::{ClipBuffers};
use fnv::FnvHasher;
use frame::DrawListGroup;
use internal_types::{DrawListItemIndex, CompiledNode, StackingContextInfo};
use internal_types::{CombinedClipRegion, BatchList, StackingContextIndex};
use internal_types::{DrawListGroupId};
use resource_cache::ResourceCache;
use std::collections::HashMap;
use std::collections::hash_state::DefaultState;
use webrender_traits::SpecificDisplayItem;

pub trait NodeCompiler {
    fn compile(&mut self,
               resource_cache: &ResourceCache,
               device_pixel_ratio: f32,
               stacking_context_info: &Vec<StackingContextInfo>,
               draw_list_groups: &HashMap<DrawListGroupId, DrawListGroup, DefaultState<FnvHasher>>);
}

impl NodeCompiler for AABBTreeNode {
    fn compile(&mut self,
               resource_cache: &ResourceCache,
               device_pixel_ratio: f32,
               stacking_context_info: &Vec<StackingContextInfo>,
               draw_list_groups: &HashMap<DrawListGroupId, DrawListGroup, DefaultState<FnvHasher>>) {
        let mut compiled_node = CompiledNode::new();
        let mut vertex_buffer = VertexBuffer::new();
        let mut clip_buffers = ClipBuffers::new();

        for draw_list_group_segment in &self.draw_list_group_segments {
            let mut builder = BatchBuilder::new(&mut vertex_buffer);

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

                    let StackingContextIndex(stacking_context_id) = draw_list.stacking_context_index.unwrap();
                    let context = &stacking_context_info[stacking_context_id];

                    for index in &draw_list_index_buffer.indices {
                        let DrawListItemIndex(index) = *index;
                        let display_item = &draw_list.items[index as usize];

                        let clip_rect = display_item.clip.main.intersection(&context.local_overflow);
                        let clip_rect = clip_rect.and_then(|clip_rect| {
                            let split_rect_local_space = self.split_rect.translate(&-context.world_origin);
                            clip_rect.intersection(&split_rect_local_space)
                        });

                        if let Some(ref clip_rect) = clip_rect {
                            let mut clip = CombinedClipRegion::from_clip_in_rect_and_stack(
                                clip_rect,
                                &display_item.clip.complex[..]);

                            match display_item.item {
                                SpecificDisplayItem::WebGL(ref info) => {
                                    builder.add_webgl_rectangle(&display_item.rect,
                                                                &clip,
                                                                resource_cache,
                                                                &mut clip_buffers,
                                                                &info.context_id);
                                }
                                SpecificDisplayItem::Image(ref info) => {
                                    builder.add_image(&display_item.rect,
                                                      &clip,
                                                      &info.stretch_size,
                                                      info.image_key,
                                                      info.image_rendering,
                                                      resource_cache,
                                                      &mut clip_buffers);
                                }
                                SpecificDisplayItem::Text(ref info) => {
                                    builder.add_text(&display_item.rect,
                                                     &clip,
                                                     info.font_key,
                                                     info.size,
                                                     info.blur_radius,
                                                     &info.color,
                                                     &info.glyphs,
                                                     resource_cache,
                                                     &mut clip_buffers,
                                                     device_pixel_ratio);
                                }
                                SpecificDisplayItem::Rectangle(ref info) => {
                                    builder.add_color_rectangle(&display_item.rect,
                                                                &clip,
                                                                resource_cache,
                                                                &mut clip_buffers,
                                                                &info.color);
                                }
                                SpecificDisplayItem::Gradient(ref info) => {
                                    clip.clip_in_rect(&display_item.rect);
                                    builder.add_gradient(&clip,
                                                         &info.start_point,
                                                         &info.end_point,
                                                         &info.stops,
                                                         resource_cache,
                                                         &mut clip_buffers);
                                }
                                SpecificDisplayItem::BoxShadow(ref info) => {
                                    builder.add_box_shadow(&info.box_bounds,
                                                           &clip,
                                                           &info.offset,
                                                           &info.color,
                                                           info.blur_radius,
                                                           info.spread_radius,
                                                           info.border_radius,
                                                           info.clip_mode,
                                                           resource_cache,
                                                           &mut clip_buffers);
                                }
                                SpecificDisplayItem::Border(ref info) => {
                                    builder.add_border(&display_item.rect,
                                                       &clip,
                                                       info,
                                                       resource_cache,
                                                       &mut clip_buffers,
                                                       device_pixel_ratio);
                                }
                            }
                        }
                    }
                }

                builder.next_draw_list();
            }

            let batches = builder.finalize();
            if batches.len() > 0 {
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
