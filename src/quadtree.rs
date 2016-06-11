/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/*
use euclid::{Point2D, Rect, Size2D};
use std::fmt::Debug;
use util::{MatrixHelpers, rect_contains_rect, RectHelpers};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NodeIndex(pub u32);

pub struct Node<T> {
    pub rect: Rect<f32>,
    pub items: Vec<T>,
    // TODO: Use Option + NonZero here
    children: Option<NodeIndex>,
}

impl<T> Node<T> {
    fn new(rect: Rect<f32>) -> Node<T> {
        Node {
            rect: rect,
            items: Vec::new(),
            children: None,
        }
    }

    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }
}

pub struct Quadtree<T> {
    pub nodes: Vec<Node<T>>,
    pub bounds: Rect<f32>,
    min_size: f32,
    size_threshold: f32,
    split_threshold: usize,
}

impl<T: Copy + Debug> Quadtree<T> {
    pub fn new(rect: Rect<f32>,
               min_size: f32,
               size_threshold: f32,
               split_threshold: usize) -> Quadtree<T> {
        Quadtree {
            bounds: rect,
            min_size: min_size,
            size_threshold: size_threshold,
            split_threshold: split_threshold,
            nodes: vec![Node::new(rect)],
        }
    }

    #[inline]
    fn node(&self, node_index: NodeIndex) -> &Node<T> {
        let NodeIndex(node_index) = node_index;
        let node_index = node_index as usize;
        &self.nodes[node_index]
    }

    #[inline]
    fn node_mut(&mut self, node_index: NodeIndex) -> &mut Node<T> {
        let NodeIndex(node_index) = node_index;
        let node_index = node_index as usize;
        &mut self.nodes[node_index]
    }

    #[inline]
    fn node_info(&self, node_index: NodeIndex) -> (Rect<f32>, usize, bool) {
        let NodeIndex(node_index) = node_index;
        let node_index = node_index as usize;
        let node = &self.nodes[node_index];
        (node.rect, node.items.len(), node.children.is_some())
    }

    fn print_internal(&self, node: NodeIndex, level: usize) {
        let mut indent = String::new();
        for _ in 0..level {
            indent.push_str("    ");
        }

        let node = self.node(node);
        println!("{}{:?} c={:?} i={:?}", indent, node.rect, node.children, node.items);

        if let Some(children) = node.children {
            let NodeIndex(children) = children;
            self.print_internal(NodeIndex(children + 0), level+1);
            self.print_internal(NodeIndex(children + 1), level+1);
        }
    }

    pub fn print(&self) {
        self.print_internal(NodeIndex(0), 0);
    }

    fn insert_internal<F>(&mut self,
                          node_index: NodeIndex,
                          item: T,
                          level: usize,
                          f: &F) where F: Fn(T) -> Rect<f32> {
        let rect = f(item);
        let (node_rect, item_count, has_children) = self.node_info(node_index);
        debug_assert!(node_rect.intersects(&rect));

        let can_split = !has_children && (node_rect.size.width > self.min_size ||
                                          node_rect.size.height > self.min_size);
        let want_split = (item_count == self.split_threshold) ||
                         (self.size_threshold > 0.0 && (node_rect.size.width > self.size_threshold ||
                                                        node_rect.size.height > self.size_threshold));
        if can_split && want_split {
            let x0 = node_rect.origin.x;
            let y0 = node_rect.origin.y;

            let x1 = x0 + node_rect.size.width;
            let y1 = y0 + node_rect.size.height;

            let (r0, r1) = if node_rect.size.width > node_rect.size.height {
                let x_mid = x0 + node_rect.size.width * 0.5;
                let r0 = Rect::from_points(x0, y0, x_mid, y1);
                let r1 = Rect::from_points(x_mid, y0, x1, y1);
                (r0, r1)
            } else {
                let y_mid = y0 + node_rect.size.height * 0.5;
                let r0 = Rect::from_points(x0, y0, x1, y_mid);
                let r1 = Rect::from_points(x0, y_mid, x1, y1);
                (r0, r1)
            };

            let mut c0 = Node::new(r0);
            let mut c1 = Node::new(r1);
            //let mut c2 = Node::new(r2);
            //let mut c3 = Node::new(r3);

            let mut retained_items = Vec::new();
            for move_item in self.node_mut(node_index).items.drain(..) {
                let item_rect = f(move_item);

                let left_int = c0.rect.intersects(&item_rect);
                let right_int = c1.rect.intersects(&item_rect);

                match (left_int, right_int) {
                    (true, true) => retained_items.push(move_item),
                    (true, false) => c0.items.push(move_item),
                    (false, true) => c1.items.push(move_item),
                    (false, false) => unreachable!(),
                }
            }
            self.node_mut(node_index).items = retained_items;
            self.node_mut(node_index).children = Some(NodeIndex(self.nodes.len() as u32));

            self.nodes.push(c0);
            self.nodes.push(c1);
        }

        let child_index = self.node(node_index).children;
        match child_index {
            Some(child_index) => {
                let NodeIndex(child_index) = child_index;
                let child_index = child_index as u32;

                let ci0 = NodeIndex(child_index + 0);
                let ci1 = NodeIndex(child_index + 1);

                let left_int = self.node(ci0).rect.intersects(&rect);
                let right_int = self.node(ci1).rect.intersects(&rect);

                match (left_int, right_int) {
                    (true, true) => self.node_mut(node_index).items.push(item),
                    (true, false) => self.insert_internal(ci0, item, level+1, f),
                    (false, true) => self.insert_internal(ci1, item, level+1, f),
                    (false, false) => unreachable!(),
                }
            }
            None => {
                self.node_mut(node_index).items.push(item);
            }
        }
    }

    pub fn insert<F>(&mut self,
                     item: T,
                     f: &F) where F: Fn(T) -> Rect<f32> {
        self.insert_internal(NodeIndex(0), item, 0, f);
    }

    fn visit_node<F>(&self,
                     node_index: NodeIndex,
                     query_rect: &Rect<f32>,
                     f: &mut F) where F: FnMut(&Rect<f32>, &Vec<T>) {
        let NodeIndex(node_index) = node_index;
        let node_index = node_index as usize;
        let node = &self.nodes[node_index];

        if node.rect.intersects(query_rect) {
            match node.children {
                Some(child_index) => {
                    let NodeIndex(child_index) = child_index;
                    let child_index = child_index as u32;

                    for i in 0..2 {
                        self.visit_node(NodeIndex(child_index + i),
                                        query_rect,
                                        f);
                    }
                }
                None => {
                    f(&node.rect, &node.items);
                }
            }
        }
    }

    pub fn visit<F>(&self,
                    query_rect: &Rect<f32>,
                    f: &mut F) where F: FnMut(&Rect<f32>, &Vec<T>) {
        self.visit_node(NodeIndex(0), query_rect, f);
    }
}
*/
