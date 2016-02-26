/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::{Point2D, Rect, Size2D};
use internal_types::BasicRotationAngle;
use webrender_traits::BorderDisplayItem;

pub fn quad_count_for_border_corner(outer_radius: &Size2D<f32>,
                                    device_pixel_ratio: f32) -> u32 {
    let max = 32.0 / device_pixel_ratio;
    if outer_radius.width < max && outer_radius.height < max {
        1
    } else {
        4
    }
}

pub trait BorderCornerTessellation {
    fn tessellate_border_corner(&self,
                                outer_radius: &Size2D<f32>,
                                inner_radius: &Size2D<f32>,
                                device_pixel_ratio: f32,
                                rotation_angle: BasicRotationAngle,
                                index: u32)
                                -> Rect<f32>;
}

impl BorderCornerTessellation for Rect<f32> {
    fn tessellate_border_corner(&self,
                                outer_radius: &Size2D<f32>,
                                inner_radius: &Size2D<f32>,
                                device_pixel_ratio: f32,
                                rotation_angle: BasicRotationAngle,
                                index: u32)
                                -> Rect<f32> {
        let quad_count = quad_count_for_border_corner(outer_radius, device_pixel_ratio);
        if quad_count == 1 {
            return *self
        }

        /*
        // FIXME(pcwalton): This is basically a hack to keep Acid2 working. We don't currently
        // render border corners properly when the corner size is greater than zero but less than
        // the radius, and we'll have to modify this when we do.
        if self.size.width - outer_radius.width > EPSILON ||
                self.size.height - outer_radius.height > EPSILON {
            return Rect::new(Point2D::new(self.origin.x + self.size.width / (quad_count as f32) *
                                          (index as f32),
                                          self.origin.y),
                             Size2D::new(self.size.width / (quad_count as f32),
                                         self.size.height))
        }*/

        let delta = outer_radius.width / (quad_count as f32);
        let prev_x = (delta * (index as f32)).ceil();
        let prev_outer_y = ellipse_y_coordinate(prev_x, outer_radius);

        let next_x = (prev_x + delta).ceil();
        let next_inner_y = ellipse_y_coordinate(next_x, inner_radius);

        let top_left = Point2D::new(prev_x, prev_outer_y);
        let bottom_right = Point2D::new(next_x, next_inner_y);

        let subrect = Rect::new(Point2D::new(top_left.x, bottom_right.y),
                                Size2D::new(bottom_right.x - top_left.x,
                                            top_left.y - bottom_right.y));

        let subrect = match rotation_angle {
            BasicRotationAngle::Upright => {
                Rect::new(Point2D::new(outer_radius.width - subrect.max_x(),
                                       outer_radius.height - subrect.max_y()),
                          subrect.size)
            }
            BasicRotationAngle::Clockwise90 => {
                Rect::new(Point2D::new(subrect.origin.x,
                                       outer_radius.height - subrect.max_y()),
                          subrect.size)
            }
            BasicRotationAngle::Clockwise180 => {
                subrect
            }
            BasicRotationAngle::Clockwise270 => {
                Rect::new(Point2D::new(outer_radius.width - subrect.max_x(),
                                       subrect.origin.y),
                          subrect.size)
            }
        };

        subrect.translate(&self.origin)
    }
}

fn ellipse_y_coordinate(x: f32, radius: &Size2D<f32>) -> f32 {
    if radius.width == 0.0 {
        return x
    }
    let radicand = 1.0 - (x / radius.width) * (x / radius.width);
    if radicand < 0.0 {
        0.0
    } else {
        radius.height * radicand.sqrt()
    }
}

/// FIXME(pcwalton): For now, we don't tessellate multicolored border radii.
pub fn can_tessellate_border(border: &BorderDisplayItem) -> bool {
    border.left.color == border.top.color &&
        border.top.color == border.right.color &&
        border.right.color == border.bottom.color &&
        border.bottom.color == border.left.color
}

