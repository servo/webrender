use aabbtree::{AABBTree, NodeIndex};
use euclid::{Point2D, Rect, Size2D, Matrix4};
use internal_types::{BatchUpdate, BatchUpdateList, BatchUpdateOp};
use internal_types::{DrawListItemIndex, DrawListId, DrawListGroupId};
use webrender_traits::ScrollLayerId;

pub struct Layer {
    // TODO: Remove pub from here if possible in the future
    pub aabb_tree: AABBTree,
    pub scroll_offset: Point2D<f32>,
    pub viewport_size: Size2D<f32>,
    pub layer_size: Size2D<f32>,
    pub world_origin: Point2D<f32>,
    pub local_transform: Matrix4,
    pub world_transform: Matrix4,
    pub children: Vec<ScrollLayerId>,
}

impl Layer {
    pub fn new(world_origin: Point2D<f32>,
               layer_size: Size2D<f32>,
               viewport_size: Size2D<f32>,
               transform: Matrix4) -> Layer {
        let rect = Rect::new(Point2D::zero(), layer_size);
        let aabb_tree = AABBTree::new(1024.0, &rect);

        Layer {
            aabb_tree: aabb_tree,
            scroll_offset: Point2D::zero(),
            viewport_size: viewport_size,
            world_origin: world_origin,
            layer_size: layer_size,
            local_transform: transform,
            world_transform: transform,
            children: Vec::new(),
        }
    }

    pub fn add_child(&mut self, child: ScrollLayerId) {
        self.children.push(child);
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
                  rect: Rect<f32>,
                  draw_list_group_id: DrawListGroupId,
                  draw_list_id: DrawListId,
                  item_index: DrawListItemIndex) {
        self.aabb_tree.insert(rect,
                              draw_list_group_id,
                              draw_list_id,
                              item_index);
    }

    pub fn finalize(&mut self,
                    initial_scroll_offset: Point2D<f32>) {
        self.scroll_offset = initial_scroll_offset;
        self.aabb_tree.finalize();
    }

    pub fn cull(&mut self) {
        // TODO(gw): Take viewport_size into account here!!!
        let viewport_rect = Rect::new(Point2D::zero(), self.viewport_size);
        let adjusted_viewport = viewport_rect.translate(&-self.scroll_offset);
        self.aabb_tree.cull(&adjusted_viewport);
    }

    #[allow(dead_code)]
    pub fn print(&self) {
        self.aabb_tree.print(NodeIndex(0), 0);
    }
}
