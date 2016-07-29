/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::{Rect, Point2D};
use std::f32;
use webrender_traits::{ColorF, BorderStyle};
use webrender_traits::{BorderSide};

//const BORDER_DASH_SIZE: f32 = 3.0;

pub trait BorderSideHelpers {
    fn border_color(&self,
                    scale_factor_0: f32,
                    scale_factor_1: f32,
                    black_color_0: f32,
                    black_color_1: f32) -> ColorF;
}

impl BorderSideHelpers for BorderSide {
    fn border_color(&self,
                    scale_factor_0: f32,
                    scale_factor_1: f32,
                    black_color_0: f32,
                    black_color_1: f32) -> ColorF {
        match self.style {
            BorderStyle::Inset => {
                if self.color.r != 0.0 || self.color.g != 0.0 || self.color.b != 0.0 {
                    self.color.scale_rgb(scale_factor_1)
                } else {
                    ColorF::new(black_color_0, black_color_0, black_color_0, self.color.a)
                }
            }
            BorderStyle::Outset => {
                if self.color.r != 0.0 || self.color.g != 0.0 || self.color.b != 0.0 {
                    self.color.scale_rgb(scale_factor_0)
                } else {
                    ColorF::new(black_color_1, black_color_1, black_color_1, self.color.a)
                }
            }
            _ => self.color,
        }
    }
}

#[derive(Debug)]
pub struct BoxShadowMetrics {
    pub edge_size: f32,
    pub tl_outer: Point2D<f32>,
    pub tl_inner: Point2D<f32>,
    pub tr_outer: Point2D<f32>,
    pub tr_inner: Point2D<f32>,
    pub bl_outer: Point2D<f32>,
    pub bl_inner: Point2D<f32>,
    pub br_outer: Point2D<f32>,
    pub br_inner: Point2D<f32>,
}

impl BoxShadowMetrics {
    pub fn new(box_bounds: &Rect<f32>, border_radius: f32, blur_radius: f32) -> BoxShadowMetrics {
        let outside_edge_size = 2.0 * blur_radius;
        let inside_edge_size = outside_edge_size.max(border_radius);
        let edge_size = outside_edge_size + inside_edge_size;
        let inner_rect = box_bounds.inflate(-inside_edge_size, -inside_edge_size);
        let outer_rect = box_bounds.inflate(outside_edge_size, outside_edge_size);

        BoxShadowMetrics {
            edge_size: edge_size,
            tl_outer: outer_rect.origin,
            tl_inner: inner_rect.origin,
            tr_outer: outer_rect.top_right(),
            tr_inner: inner_rect.top_right(),
            bl_outer: outer_rect.bottom_left(),
            bl_inner: inner_rect.bottom_left(),
            br_outer: outer_rect.bottom_right(),
            br_inner: inner_rect.bottom_right(),
        }
    }
}
