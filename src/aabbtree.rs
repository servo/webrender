use euclid::{Point2D, Rect, Size2D};
use internal_types::{CompiledNode, DisplayItemKey};
use resource_list::ResourceList;
use types::NodeIndex;
use util;

pub struct AABBTreeNode {
    pub split_rect: Rect<f32>,
    pub actual_rect: Rect<f32>,
    pub node_index: NodeIndex,

    // TODO: Use Option + NonZero here
    pub children: Option<NodeIndex>,

    pub is_visible: bool,

    pub src_items: Vec<DisplayItemKey>,

    pub resource_list: Option<ResourceList>,
    pub compiled_node: Option<CompiledNode>,
}

impl AABBTreeNode {
    fn new(split_rect: &Rect<f32>, node_index: NodeIndex) -> AABBTreeNode {
        AABBTreeNode {
            split_rect: split_rect.clone(),
            actual_rect: Rect::zero(),
            node_index: node_index,
            children: None,
            is_visible: false,
            resource_list: None,
            src_items: Vec::new(),
            compiled_node: None,
        }
    }

    #[inline]
    fn append_item(&mut self,
                   draw_list_index: usize,
                   item_index: usize,
                   rect: &Rect<f32>) {
        self.actual_rect = self.actual_rect.union(rect);
        let key = DisplayItemKey::new(draw_list_index, item_index);
        self.src_items.push(key);
    }
}

pub struct AABBTree {
    pub nodes: Vec<AABBTreeNode>,
    pub split_size: f32,
}

pub struct AABBTreeNodeInfo {
    pub rect: Rect<f32>,
    pub is_visible: bool,
}

impl AABBTree {
    pub fn new(split_size: f32) -> AABBTree {
        AABBTree {
            nodes: Vec::new(),
            split_size: split_size,
        }
    }

    pub fn init(&mut self, scene_rect: &Rect<f32>) {
        self.nodes.clear();

        let root_node = AABBTreeNode::new(scene_rect, NodeIndex(0));
        self.nodes.push(root_node);
    }

    #[allow(dead_code)]
    pub fn print(&self, node_index: NodeIndex, level: u32) {
        let mut indent = String::new();
        for _ in 0..level {
            indent.push_str("  ");
        }

        let node = self.node(node_index);
        println!("{}n={:?} sr={:?} ar={:?} c={:?} items={}",
                 indent,
                 node_index,
                 node.split_rect,
                 node.actual_rect,
                 node.children,
                 node.src_items.len());

        if let Some(child_index) = node.children {
            let NodeIndex(child_index) = child_index;
            self.print(NodeIndex(child_index+0), level+1);
            self.print(NodeIndex(child_index+1), level+1);
        }
    }

    #[inline(always)]
    pub fn node(&self, index: NodeIndex) -> &AABBTreeNode {
        let NodeIndex(index) = index;
        &self.nodes[index as usize]
    }

    #[inline(always)]
    pub fn node_mut(&mut self, index: NodeIndex) -> &mut AABBTreeNode {
        let NodeIndex(index) = index;
        &mut self.nodes[index as usize]
    }

    pub fn node_info(&self) -> Vec<AABBTreeNodeInfo> {
        let mut info = Vec::new();
        for node in &self.nodes {
            info.push(AABBTreeNodeInfo {
                rect: node.actual_rect,
                is_visible: node.is_visible,
            });
        }
        info
    }

    #[inline]
    fn find_best_node(&mut self,
                      node_index: NodeIndex,
                      rect: &Rect<f32>) -> Option<NodeIndex> {
        self.split_if_needed(node_index);

        if let Some(child_node_index) = self.node(node_index).children {
            let NodeIndex(child_node_index) = child_node_index;
            let left_node_index = NodeIndex(child_node_index + 0);
            let right_node_index = NodeIndex(child_node_index + 1);

            let left_intersect = self.node(left_node_index).split_rect.intersects(rect);
            let right_intersect = self.node(right_node_index).split_rect.intersects(rect);

            if left_intersect && right_intersect {
                Some(node_index)
            } else if left_intersect {
                self.find_best_node(left_node_index, rect)
            } else if right_intersect {
                self.find_best_node(right_node_index, rect)
            } else {
                None
            }
        } else {
            Some(node_index)
        }
    }

    #[inline]
    pub fn insert(&mut self,
                  rect: &Rect<f32>,
                  draw_list_index: usize,
                  item_index: usize) -> Option<NodeIndex> {
        let node_index = self.find_best_node(NodeIndex(0), rect);
        if let Some(node_index) = node_index {
            let node = self.node_mut(node_index);
            node.append_item(draw_list_index, item_index, rect);
        }
        node_index
    }

    fn split_if_needed(&mut self, node_index: NodeIndex) {
        if self.node(node_index).children.is_none() {
            let rect = self.node(node_index).split_rect.clone();

            let child_rects = if rect.size.width > self.split_size &&
                                 rect.size.width > rect.size.height {
                let new_width = rect.size.width * 0.5;

                let left = Rect::new(rect.origin, Size2D::new(new_width, rect.size.height));
                let right = Rect::new(rect.origin + Point2D::new(new_width, 0.0),
                                      Size2D::new(rect.size.width - new_width, rect.size.height));

                Some((left, right))
            } else if rect.size.height > self.split_size {
                let new_height = rect.size.height * 0.5;

                let left = Rect::new(rect.origin, Size2D::new(rect.size.width, new_height));
                let right = Rect::new(rect.origin + Point2D::new(0.0, new_height),
                                      Size2D::new(rect.size.width, rect.size.height - new_height));

                Some((left, right))
            } else {
                None
            };

            if let Some((left_rect, right_rect)) = child_rects {
                let child_node_index = self.nodes.len() as u32;

                let left_node = AABBTreeNode::new(&left_rect, NodeIndex(child_node_index+0));
                self.nodes.push(left_node);

                let right_node = AABBTreeNode::new(&right_rect, NodeIndex(child_node_index+1));
                self.nodes.push(right_node);

                self.node_mut(node_index).children = Some(NodeIndex(child_node_index));
            }
        }
    }

    fn check_node_visibility(&mut self,
                             node_index: NodeIndex,
                             rect: &Rect<f32>) {
        let children = {
            let node = self.node_mut(node_index);
            if node.split_rect.intersects(rect) {
                if node.src_items.len() > 0 &&
                   node.actual_rect.intersects(rect) {
                    node.is_visible = true;
                }
                node.children
            } else {
                return;
            }
        };

        if let Some(child_index) = children {
            let NodeIndex(child_index) = child_index;
            self.check_node_visibility(NodeIndex(child_index+0), rect);
            self.check_node_visibility(NodeIndex(child_index+1), rect);
        }
    }

    pub fn cull(&mut self, rect: &Rect<f32>) {
        let _pf = util::ProfileScope::new("  cull");
        for node in &mut self.nodes {
            node.is_visible = false;
        }
        if self.nodes.len() > 0 {
            self.check_node_visibility(NodeIndex(0), &rect);
        }
    }
}
