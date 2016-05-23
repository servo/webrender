/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::{Point2D, Rect, Size2D};
use internal_types::{CompiledNode, DrawListId, DrawListItemIndex, DrawListGroupId, MAX_RECT};
use resource_list::ResourceList;
use util;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NodeIndex(pub u32);

pub struct DrawListIndexBuffer {
    pub draw_list_id: DrawListId,
    pub indices: Vec<DrawListItemIndex>,
}

pub struct DrawListGroupSegment {
    pub draw_list_group_id: DrawListGroupId,
    pub index_buffers: Vec<DrawListIndexBuffer>,
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum TreeState {
    Building,
    Finalized,
}

pub struct AABBTreeNode {
    pub split_rect: Rect<f32>,
    pub actual_rect: Rect<f32>,

    // TODO: Use Option + NonZero here
    pub children: Option<NodeIndex>,

    pub is_visible: bool,

    pub draw_list_group_segments: Vec<DrawListGroupSegment>,

    pub resource_list: Option<ResourceList>,
    pub compiled_node: Option<CompiledNode>,
}

impl AABBTreeNode {
    fn new(split_rect: &Rect<f32>) -> AABBTreeNode {
        AABBTreeNode {
            split_rect: split_rect.clone(),
            actual_rect: Rect::zero(),
            children: None,
            is_visible: false,
            resource_list: None,
            draw_list_group_segments: Vec::new(),
            compiled_node: None,
        }
    }

    #[inline]
    fn append_item(&mut self,
                   draw_list_group_id: DrawListGroupId,
                   draw_list_id: DrawListId,
                   item_index: DrawListItemIndex,
                   rect: &Rect<f32>) {
        self.actual_rect = self.actual_rect.union(rect);

        let need_new_group = match self.draw_list_group_segments.last() {
            Some(group) => {
                group.draw_list_group_id != draw_list_group_id
            }
            None => {
                true
            }
        };

        if need_new_group {
            self.draw_list_group_segments.push(DrawListGroupSegment {
                draw_list_group_id: draw_list_group_id,
                index_buffers: Vec::new(),
            })
        }

        let need_new_list = match self.draw_list_group_segments.last().unwrap().index_buffers.last() {
            Some(draw_list) => {
                draw_list.draw_list_id != draw_list_id
            }
            None => {
                true
            }
        };

        if need_new_list {
            self.draw_list_group_segments.last_mut().unwrap().index_buffers.push(DrawListIndexBuffer {
                draw_list_id: draw_list_id,
                indices: Vec::new(),
            });
        }

        self.draw_list_group_segments
            .last_mut()
            .unwrap()
            .index_buffers
            .last_mut()
            .unwrap()
            .indices
            .push(item_index);
    }

    fn item_count(&self) -> usize {
        let mut count = 0;
        for group in &self.draw_list_group_segments {
            for list in &group.index_buffers {
                count += list.indices.len();
            }
        }
        count
    }
}

pub struct AABBTree {
    pub nodes: Vec<AABBTreeNode>,
    pub split_size: f32,

    work_node_indices: Vec<NodeIndex>,

    state: TreeState,
}

impl AABBTree {
    pub fn new(split_size: f32,
               local_bounds: &Rect<f32>) -> AABBTree {
        let mut tree = AABBTree {
            nodes: Vec::new(),
            split_size: split_size,
            work_node_indices: Vec::new(),
            state: TreeState::Building,
        };

        let root_node = AABBTreeNode::new(local_bounds);
        tree.nodes.push(root_node);

        tree
    }

    pub fn finalize(&mut self) {
        debug_assert!(self.state == TreeState::Building);
        self.state = TreeState::Finalized;
    }

    #[allow(dead_code)]
    pub fn print(&self, node_index: NodeIndex, level: u32) {
        let mut indent = String::new();
        for _ in 0..level {
            indent.push_str("  ");
        }

        let node = self.node(node_index);
        println!("{}n={:?} sr={:?} ar={:?} c={:?} lists={} segments={}",
                 indent,
                 node_index,
                 node.split_rect,
                 node.actual_rect,
                 node.children,
                 node.draw_list_group_segments.len(),
                 node.item_count());

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

    #[inline]
    fn find_best_nodes(&mut self,
                       node_index: NodeIndex,
                       rect: &Rect<f32>) {
        self.split_if_needed(node_index);

        if let Some(child_node_index) = self.node(node_index).children {
            let NodeIndex(child_node_index) = child_node_index;
            let left_node_index = NodeIndex(child_node_index + 0);
            let right_node_index = NodeIndex(child_node_index + 1);

            let left_intersect = self.node(left_node_index).split_rect.intersects(rect);
            if left_intersect {
                self.find_best_nodes(left_node_index, rect);
            }

            let right_intersect = self.node(right_node_index).split_rect.intersects(rect);
            if right_intersect {
                self.find_best_nodes(right_node_index, rect);
            }
        } else {
            self.work_node_indices.push(node_index);
        }
    }

    #[inline]
    pub fn insert(&mut self,
                  rect: Rect<f32>,
                  draw_list_group_id: DrawListGroupId,
                  draw_list_id: DrawListId,
                  item_index: DrawListItemIndex) {
        debug_assert!(self.state == TreeState::Building);

        self.find_best_nodes(NodeIndex(0), &rect);
        if self.work_node_indices.is_empty() {
            // TODO(gw): If this happens, it it probably caused by items having
            //           transforms that move them outside the local overflow. According
            //           to the transforms spec, the stacking context overflow should
            //           include transformed elements, however this isn't currently
            //           handled by the layout code! If it's not that, this is an
            //           unexpected condition and should be investigated!
            debug!("WARNING: insert rect {:?} outside bounds, dropped.", rect);
        } else {
            for node_index in self.work_node_indices.drain(..) {
                let NodeIndex(node_index) = node_index;
                let node = &mut self.nodes[node_index as usize];
                node.append_item(draw_list_group_id,
                                 draw_list_id,
                                 item_index,
                                 &rect);
            }
        }
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

                let left_node = AABBTreeNode::new(&left_rect);
                self.nodes.push(left_node);

                let right_node = AABBTreeNode::new(&right_rect);
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
                if !node.draw_list_group_segments.is_empty() &&
                   node.actual_rect.intersects(rect) {
                    debug_assert!(node.children.is_none());
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
        debug_assert!(self.state == TreeState::Finalized);
        for node in &mut self.nodes {
            node.is_visible = false;
        }
        if !self.nodes.is_empty() {
            self.check_node_visibility(NodeIndex(0), &rect);
        }
    }
}
