use aabbtree::AABBTreeNode;
use batch::{BatchBuilder, MatrixIndex, VertexBuffer};
use clipper::{ClipBuffers};
use frame::{FrameRenderItem, FrameRenderTarget};
use internal_types::{DrawListItemIndex, CompiledNode, CombinedClipRegion, BatchList};
use resource_cache::ResourceCache;
use webrender_traits::{ColorF, SpecificDisplayItem};

pub trait NodeCompiler {
    fn compile(&mut self,
               resource_cache: &ResourceCache,
               render_targets: &Vec<FrameRenderTarget>,
               device_pixel_ratio: f32);
}

impl NodeCompiler for AABBTreeNode {
    fn compile(&mut self,
               resource_cache: &ResourceCache,
               render_targets: &Vec<FrameRenderTarget>,
               device_pixel_ratio: f32) {
        let mut compiled_node = CompiledNode::new();
        let mut vertex_buffer = VertexBuffer::new();

        let mut clip_buffers = ClipBuffers::new();

        for render_target in render_targets {
            for item in &render_target.items {
                match item {
                    &FrameRenderItem::Clear(..) |
                    &FrameRenderItem::Composite(..) => {}
                    &FrameRenderItem::DrawListBatch(ref batch_info) => {
                        // TODO: Move this to outer loop when combining with >1 draw list!
                        let mut builder = BatchBuilder::new(&mut vertex_buffer);

                        for (index, draw_list_id) in batch_info.draw_lists.iter().enumerate() {
                            let draw_list_id = *draw_list_id;
                            let matrix_index = MatrixIndex(index as u8);

                            let draw_list_index_buffer = self.draw_lists.iter().find(|draw_list| {
                                draw_list.draw_list_id == draw_list_id
                            });

                            if let Some(draw_list_index_buffer) = draw_list_index_buffer {
                                let draw_list = resource_cache.get_draw_list(draw_list_id);

                                for index in &draw_list_index_buffer.indices {
                                    let DrawListItemIndex(index) = *index;
                                    let display_item = &draw_list.items[index as usize];

                                    let context = draw_list.context.as_ref().unwrap();
                                    let clip_rect = display_item.clip.main.intersection(&context.overflow);
                                    let clip_rect = clip_rect.and_then(|clip_rect| {
                                        let split_rect_local_space = self.split_rect.translate(&-context.origin);
                                        clip_rect.intersection(&split_rect_local_space)
                                    });

                                    if let Some(ref clip_rect) = clip_rect {
                                        let mut clip = CombinedClipRegion::from_clip_in_rect_and_stack(
                                            clip_rect,
                                            &display_item.clip.complex[..]);

                                        match display_item.item {
                                            SpecificDisplayItem::Image(ref info) => {
                                                builder.add_image(matrix_index,
                                                                  &display_item.rect,
                                                                  &clip,
                                                                  &info.stretch_size,
                                                                  info.image_key,
                                                                  info.image_rendering,
                                                                  resource_cache,
                                                                  &mut clip_buffers);
                                            }
                                            SpecificDisplayItem::Text(ref info) => {
                                                builder.add_text(matrix_index,
                                                                 &display_item.rect,
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
                                                builder.add_color_rectangle(matrix_index,
                                                                            &display_item.rect,
                                                                            &clip,
                                                                            resource_cache,
                                                                            &mut clip_buffers,
                                                                            &info.color);
                                            }
                                            SpecificDisplayItem::Gradient(ref info) => {
                                                clip.clip_in_rect(&display_item.rect);
                                                builder.add_gradient(matrix_index,
                                                                     &clip,
                                                                     &info.start_point,
                                                                     &info.end_point,
                                                                     &info.stops,
                                                                     resource_cache,
                                                                     &mut clip_buffers);
                                            }
                                            SpecificDisplayItem::BoxShadow(ref info) => {
                                                builder.add_box_shadow(matrix_index,
                                                                       &info.box_bounds,
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
                                              builder.add_border(matrix_index,
                                                                 &display_item.rect,
                                                                 &clip,
                                                                 info,
                                                                 resource_cache,
                                                                 &mut clip_buffers);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        let batches = builder.finalize();

                        compiled_node.batch_list.push(BatchList {
                            batches: batches,
                            first_draw_list_id: *batch_info.draw_lists.first().unwrap(),
                        });
                    }
                }
            }
        }

        compiled_node.vertex_buffer = Some(vertex_buffer);
        self.compiled_node = Some(compiled_node);
    }
}
