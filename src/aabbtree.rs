/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/*
use euclid::{Point2D, Rect, Size2D};
use internal_types::{CompiledNode, DrawListGroupId, DrawListId, DrawListItemIndex};
use resource_list::ResourceList;
use util;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NodeIndex(pub u32);

pub struct AABBTreeNode {
    pub split_rect: Rect<f32>,
    pub actual_rect: Rect<f32>,
    pub primitives: Vec<PrimitiveIndex>,
    // TODO: Use Option + NonZero here
    pub children: Option<NodeIndex>,
}

impl AABBTreeNode {
    fn new(split_rect: &Rect<f32>) -> AABBTreeNode {
        AABBTreeNode {
            split_rect: split_rect.clone(),
            actual_rect: Rect::zero(),
            primitives: Vec::new(),
            children: None,
        }
    }

    #[inline]
    fn append_item(&mut self, prim_index: PrimitiveIndex, rect: &Rect<f32>) {
        debug_assert!(self.split_rect.intersects(rect));

        self.actual_rect = self.actual_rect.union(rect);
        self.primitives.push(prim_index);
    }
}

pub struct AABBTree {
    pub nodes: Vec<AABBTreeNode>,
    pub split_size: f32,
    work_node_indices: Vec<NodeIndex>,
}

impl AABBTree {
    pub fn new(split_size: f32,
               local_bounds: &Rect<f32>) -> AABBTree {
        let mut tree = AABBTree {
            nodes: Vec::new(),
            split_size: split_size,
            work_node_indices: Vec::new(),
        };

        let root_node = AABBTreeNode::new(local_bounds);
        tree.nodes.push(root_node);

        tree
    }

    #[allow(dead_code)]
    pub fn print(&self, node_index: NodeIndex, level: u32) {
        let mut indent = String::new();
        for _ in 0..level {
            indent.push_str("  ");
        }

        let node = self.node(node_index);
        println!("{}n={:?} sr={:?} ar={:?} c={:?}",
                 indent,
                 node_index,
                 node.split_rect,
                 node.actual_rect,
                 node.children);

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
    pub fn insert(&mut self, rect: Rect<f32>, prim_index: PrimitiveIndex) {
        self.find_best_nodes(NodeIndex(0), &rect);
        if self.work_node_indices.is_empty() {
            // TODO(gw): If this happens, it it probably caused by items having
            //           transforms that move them outside the local overflow. According
            //           to the transforms spec, the stacking context overflow should
            //           include transformed elements, however this isn't currently
            //           handled by the layout code! If it's not that, this is an
            //           unexpected condition and should be investigated!
            debug!("WARNING: insert rect {:?} outside bounds, dropped.", rect);
            self.work_node_indices.clear();
        } else {
            for node_index in self.work_node_indices.drain(..) {
                let NodeIndex(node_index) = node_index;
                let node = &mut self.nodes[node_index as usize];
                node.append_item(prim_index, &rect);
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

/*
    fn check_node_visibility(&self,
                             node_index: NodeIndex,
                             rect: &Rect<f32>,
                             vis_prims: &mut Vec<PrimitiveIndex>) {
        let children = {
            let node = self.node(node_index);
            if node.split_rect.intersects(rect) {
                if !node.primitives.is_empty() &&
                   node.actual_rect.intersects(rect) {
                    //node.is_visible = true;
                    vis_prims.extend_from_slice(&node.primitives);
                }
                node.children
            } else {
                return;
            }
        };

        if let Some(child_index) = children {
            let NodeIndex(child_index) = child_index;
            self.check_node_visibility(NodeIndex(child_index+0), rect, vis_prims);
            self.check_node_visibility(NodeIndex(child_index+1), rect, vis_prims);
        }
    }

    pub fn vis_query(&self, rect: &Rect<f32>) -> Vec<PrimitiveIndex> {
        debug_assert!(self.state == TreeState::Finalized);
        let mut vis_prims = Vec::new();
        if !self.nodes.is_empty() {
            self.check_node_visibility(NodeIndex(0), &rect, &mut vis_prims);
        }
        vis_prims
    }*/
}
*/
