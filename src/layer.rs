use aabbtree::{AABBTree, NodeIndex};
use euclid::{Point2D, Rect, Size2D};
use internal_types::{BatchUpdate, BatchUpdateList, BatchUpdateOp};
use internal_types::{DrawListItemIndex, DrawListId, DrawListGroupId};

pub struct Layer {
    // TODO: Remove pub from here if possible in the future
    pub aabb_tree: AABBTree,
    pub scroll_offset: Point2D<f32>,
    pub scroll_boundaries: Size2D<f32>,
}

impl Layer {
    pub fn new(scene_rect: &Rect<f32>, scroll_offset: &Point2D<f32>) -> Layer {
        let aabb_tree = AABBTree::new(1024.0, scene_rect);

        Layer {
            aabb_tree: aabb_tree,
            scroll_offset: *scroll_offset,
            scroll_boundaries: Size2D::zero(),
        }
    }

    pub fn reset(&mut self, pending_updates: &mut BatchUpdateList) {
        for node in &mut self.aabb_tree.nodes {
            if let Some(ref mut compiled_node) = node.compiled_node {
                let vertex_buffer_id = compiled_node.vertex_buffer_id.take().unwrap();
                pending_updates.push(BatchUpdate {
                    id: vertex_buffer_id,
                    op: BatchUpdateOp::Destroy,
                });
            }
        }
    }

    #[inline]
    pub fn insert(&mut self,
                  rect: &Rect<f32>,
                  draw_list_group_id: DrawListGroupId,
                  draw_list_id: DrawListId,
                  item_index: DrawListItemIndex) {
        self.aabb_tree.insert(rect,
                              draw_list_group_id,
                              draw_list_id,
                              item_index);
    }

    pub fn cull(&mut self, viewport_rect: &Rect<f32>) {
        let adjusted_viewport = viewport_rect.translate(&-self.scroll_offset);
        self.aabb_tree.cull(&adjusted_viewport);
    }

    #[allow(dead_code)]
    pub fn print(&self) {
        self.aabb_tree.print(NodeIndex(0), 0);
    }
}
