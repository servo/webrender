/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::{Matrix4D, Point2D, Rect, Size2D};
use spring::{DAMPING, STIFFNESS, Spring};
use webrender_traits::{PipelineId, ScrollLayerId, ServoStackingContextId};

pub struct Layer {
    pub scrolling: ScrollingState,

    /// The viewable region, in world coordinates.
    pub viewport_rect: Rect<f32>,

    /// The transform to apply to the viewable region, in world coordinates.
    ///
    /// TODO(pcwalton): These should really be a stack of clip regions and transforms.
    pub viewport_transform: Matrix4D<f32>,

    pub layer_size: Size2D<f32>,
    pub world_origin: Point2D<f32>,
    pub local_transform: Matrix4D<f32>,
    pub local_perspective: Matrix4D<f32>,
    pub world_transform: Option<Matrix4D<f32>>,
    pub pipeline_id: PipelineId,
    pub stacking_context_id: ServoStackingContextId,
    pub children: Vec<ScrollLayerId>,
}

impl Layer {
    pub fn new(world_origin: Point2D<f32>,
               layer_size: Size2D<f32>,
               viewport_rect: &Rect<f32>,
               viewport_transform: &Matrix4D<f32>,
               local_transform: &Matrix4D<f32>,
               local_perspective: &Matrix4D<f32>,
               pipeline_id: PipelineId,
               stacking_context_id: ServoStackingContextId)
               -> Layer {
        Layer {
            scrolling: ScrollingState::new(),
            viewport_rect: *viewport_rect,
            viewport_transform: *viewport_transform,
            world_origin: world_origin,
            layer_size: layer_size,
            local_transform: *local_transform,
            local_perspective: *local_perspective,
            world_transform: None,
            children: Vec::new(),
            pipeline_id: pipeline_id,
            stacking_context_id: stacking_context_id,
        }
    }

    pub fn add_child(&mut self, child: ScrollLayerId) {
        self.children.push(child);
    }

    pub fn finalize(&mut self, scrolling: &ScrollingState) {
        self.scrolling = *scrolling;
    }

    pub fn overscroll_amount(&self) -> Size2D<f32> {
        let overscroll_x = if self.scrolling.offset.x > 0.0 {
            -self.scrolling.offset.x
        } else if self.scrolling.offset.x < self.viewport_rect.size.width - self.layer_size.width {
            self.viewport_rect.size.width - self.layer_size.width - self.scrolling.offset.x
        } else {
            0.0
        };

        let overscroll_y = if self.scrolling.offset.y > 0.0 {
            -self.scrolling.offset.y
        } else if self.scrolling.offset.y < self.viewport_rect.size.height -
                self.layer_size.height {
            self.viewport_rect.size.height - self.layer_size.height - self.scrolling.offset.y
        } else {
            0.0
        };

        Size2D::new(overscroll_x, overscroll_y)
    }

    pub fn stretch_overscroll_spring(&mut self) {
        let overscroll_amount = self.overscroll_amount();
        self.scrolling.spring.coords(self.scrolling.offset,
                                     self.scrolling.offset,
                                     self.scrolling.offset + overscroll_amount);
    }

    pub fn tick_scrolling_bounce_animation(&mut self) {
        let finished = self.scrolling.spring.animate();
        self.scrolling.offset = self.scrolling.spring.current();
        if finished {
            self.scrolling.bouncing_back = false
        }
    }
}

#[derive(Copy, Clone)]
pub struct ScrollingState {
    pub offset: Point2D<f32>,
    pub spring: Spring,
    pub started_bouncing_back: bool,
    pub bouncing_back: bool,
}

impl ScrollingState {
    pub fn new() -> ScrollingState {
        ScrollingState {
            offset: Point2D::new(0.0, 0.0),
            spring: Spring::at(Point2D::new(0.0, 0.0), STIFFNESS, DAMPING),
            started_bouncing_back: false,
            bouncing_back: false,
        }
    }
}

