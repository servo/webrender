/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::{Matrix4D, Point2D, Point4D, Rect, Size2D};
use internal_types::{DevicePixel};
use num_traits::Zero;
use time::precise_time_ns;

#[allow(dead_code)]
pub struct ProfileScope {
    name: &'static str,
    t0: u64,
}

impl ProfileScope {
    #[allow(dead_code)]
    pub fn new(name: &'static str) -> ProfileScope {
        ProfileScope {
            name: name,
            t0: precise_time_ns(),
        }
    }
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
            let t1 = precise_time_ns();
            let ms = (t1 - self.t0) as f64 / 1000000f64;
            println!("{} {}", self.name, ms);
    }
}

// TODO: Implement these in euclid!
pub trait MatrixHelpers {
    fn transform_point_and_perspective_project(&self, point: &Point4D<f32>) -> Point2D<f32>;
    fn transform_rect(&self, rect: &Rect<f32>) -> Rect<f32>;

    /// Returns true if this matrix transforms an axis-aligned 2D rectangle to another axis-aligned
    /// 2D rectangle.
    fn can_losslessly_transform_a_2d_rect(&self) -> bool;

    /// Returns true if this matrix will transforms an axis-aligned 2D rectangle to another axis-
    /// aligned 2D rectangle after perspective divide.
    fn can_losslessly_transform_and_perspective_project_a_2d_rect(&self) -> bool;

    /// Clears out the portions of the matrix that `transform_rect()` uses. This allows the use of
    /// `transform_rect()` while keeping the Z/W transform portions of the matrix intact.
    fn reset_after_transforming_rect(&self) -> Matrix4D<f32>;
}

impl MatrixHelpers for Matrix4D<f32> {
    fn transform_point_and_perspective_project(&self, point: &Point4D<f32>) -> Point2D<f32> {
        let point = self.transform_point4d(point);
        Point2D::new(point.x / point.w, point.y / point.w)
    }

    fn transform_rect(&self, rect: &Rect<f32>) -> Rect<f32> {
        let top_left = self.transform_point(&rect.origin);
        let top_right = self.transform_point(&rect.top_right());
        let bottom_left = self.transform_point(&rect.bottom_left());
        let bottom_right = self.transform_point(&rect.bottom_right());
        Rect::from_points(&top_left, &top_right, &bottom_right, &bottom_left)
    }

    fn can_losslessly_transform_a_2d_rect(&self) -> bool {
        self.m12 == 0.0 && self.m14 == 0.0 && self.m21 == 0.0 && self.m24 == 0.0 && self.m44 == 1.0
    }

    fn can_losslessly_transform_and_perspective_project_a_2d_rect(&self) -> bool {
        self.m12 == 0.0 && self.m21 == 0.0
    }

    fn reset_after_transforming_rect(&self) -> Matrix4D<f32> {
        Matrix4D {
            m11: 1.0,      m12: 0.0,      m13: self.m13, m14: 0.0,
            m21: 0.0,      m22: 1.0,      m23: self.m23, m24: 0.0,
            m31: self.m31, m32: self.m32, m33: self.m33, m34: self.m34,
            m41: 0.0,      m42: 0.0,      m43: self.m43, m44: 1.0,
        }
    }
}

pub trait RectHelpers {
    fn from_points(a: &Point2D<f32>, b: &Point2D<f32>, c: &Point2D<f32>, d: &Point2D<f32>) -> Self;
    fn contains_rect(&self, other: &Rect<f32>) -> bool;
    fn from_floats(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect<f32>;
    fn is_well_formed_and_nonempty(&self) -> bool;
}

impl RectHelpers for Rect<f32> {
    fn from_points(a: &Point2D<f32>, b: &Point2D<f32>, c: &Point2D<f32>, d: &Point2D<f32>)
                   -> Rect<f32> {
        let (mut min_x, mut min_y) = (a.x.clone(), a.y.clone());
        let (mut max_x, mut max_y) = (min_x.clone(), min_y.clone());
        for point in &[b, c, d] {
            if point.x < min_x {
                min_x = point.x.clone()
            }
            if point.x > max_x {
                max_x = point.x.clone()
            }
            if point.y < min_y {
                min_y = point.y.clone()
            }
            if point.y > max_y {
                max_y = point.y.clone()
            }
        }
        Rect::new(Point2D::new(min_x.clone(), min_y.clone()),
                  Size2D::new(max_x - min_x, max_y - min_y))
    }

    fn contains_rect(&self, other: &Rect<f32>) -> bool {
        self.origin.x <= other.origin.x &&
        self.origin.y <= other.origin.y &&
        self.max_x() >= other.max_x() &&
        self.max_y() >= other.max_y()
    }

    fn from_floats(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect<f32> {
        Rect::new(Point2D::new(x0, y0),
                  Size2D::new(x1 - x0, y1 - y0))
    }

    fn is_well_formed_and_nonempty(&self) -> bool {
        self.size.width > 0.0 && self.size.height > 0.0
    }
}

// Don't use `euclid`'s `is_empty` because that has effectively has an "and" in the conditional
// below instead of an "or".
pub fn rect_is_empty<N:PartialEq + Zero>(rect: &Rect<N>) -> bool {
    rect.size.width == Zero::zero() || rect.size.height == Zero::zero()
}

/*
#[inline(always)]
pub fn rect_contains_rect(rect: &Rect<Au>, other: &Rect<Au>) -> bool {
    rect.origin.x <= other.origin.x &&
    rect.origin.y <= other.origin.y &&
    rect.max_x() >= other.max_x() &&
    rect.max_y() >= other.max_y()
}*/

#[inline(always)]
pub fn rect_contains_rect(rect: &Rect<DevicePixel>, other: &Rect<DevicePixel>) -> bool {
    rect.origin.x <= other.origin.x &&
    rect.origin.y <= other.origin.y &&
    rect.max_x() >= other.max_x() &&
    rect.max_y() >= other.max_y()
}

#[inline]
pub fn rect_from_points(x0: DevicePixel,
                        y0: DevicePixel,
                        x1: DevicePixel,
                        y1: DevicePixel) -> Rect<DevicePixel> {
    Rect::new(Point2D::new(x0, y0),
              Size2D::new(x1 - x0, y1 - y0))
}

#[inline]
pub fn rect_from_points_f(x0: f32,
                          y0: f32,
                          x1: f32,
                          y1: f32) -> Rect<f32> {
    Rect::new(Point2D::new(x0, y0),
              Size2D::new(x1 - x0, y1 - y0))
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    (b - a) * t + a
}

pub fn subtract_rect(rect: &Rect<f32>,
                     other: &Rect<f32>,
                     results: &mut Vec<Rect<f32>>) {
    results.clear();

    if rect.intersects(other) {
        let rx0 = rect.origin.x;
        let ry0 = rect.origin.y;
        let rx1 = rx0 + rect.size.width;
        let ry1 = ry0 + rect.size.height;

        let ox0 = other.origin.x;
        let oy0 = other.origin.y;
        let ox1 = ox0 + other.size.width;
        let oy1 = oy0 + other.size.height;

        let r = rect_from_points_f(rx0, ry0, ox0, ry1);
        if r.size.width > 0.0 && r.size.height > 0.0 {
            results.push(r);
        }
        let r = rect_from_points_f(ox0, ry0, ox1, oy0);
        if r.size.width > 0.0 && r.size.height > 0.0 {
            results.push(r);
        }
        let r = rect_from_points_f(ox0, oy1, ox1, ry1);
        if r.size.width > 0.0 && r.size.height > 0.0 {
            results.push(r);
        }
        let r = rect_from_points_f(ox1, ry0, rx1, ry1);
        if r.size.width > 0.0 && r.size.height > 0.0 {
            results.push(r);
        }
    } else {
        results.push(*rect);
    }
}