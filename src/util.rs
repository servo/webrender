/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::{Matrix4D, Point2D, Point4D, Rect, Size2D};
use internal_types::{RectColors};
use num_traits::Zero;
use time::precise_time_ns;
use webrender_traits::ColorF;

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
        /*
        if self.name.chars().next() != Some(' ') {
            let t1 = precise_time_ns();
            let ms = (t1 - self.t0) as f64 / 1000000f64;
            println!("{} {}", self.name, ms);
        }*/
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
}

pub trait RectHelpers {
    fn from_points(a: &Point2D<f32>, b: &Point2D<f32>, c: &Point2D<f32>, d: &Point2D<f32>) -> Self;
    fn contains_rect(&self, other: &Rect<f32>) -> bool;
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
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    (b - a) * t + a
}

pub fn bilerp<V>(point: &Point2D<f32>, quad: &Rect<f32>, varyings: &V) -> V::Element
                 where V: RectVaryings, V::Element: VaryingElement {
    let (x1, y1, x2, y2) = (quad.origin.x, quad.origin.y, quad.max_x(), quad.max_y());
    let top_left = varyings.top_left().scale((x2 - point.x) * (y2 - point.y));
    let top_right = varyings.top_right().scale((point.x - x1) * (y2 - point.y));
    let bottom_left = varyings.bottom_left().scale((x2 - point.x) * (point.y - y1));
    let bottom_right = varyings.bottom_right().scale((point.x - x1) * (point.y - y1));
    let sum = VaryingElement::accumulate(&[top_left, top_right, bottom_left, bottom_right]);
    sum.scale(1.0 / ((x2 - x1) * (y2 - y1)))
}

pub fn bilerp_rect<V>(clipped_rect: &Rect<f32>, quad: &Rect<f32>, varyings: &V) -> V
                      where V: RectVaryings, V::Element: VaryingElement {
    let top_left = bilerp(&clipped_rect.origin, quad, varyings);
    let top_right = bilerp(&clipped_rect.top_right(), quad, varyings);
    let bottom_right = bilerp(&clipped_rect.bottom_right(), quad, varyings);
    let bottom_left = bilerp(&clipped_rect.bottom_left(), quad, varyings);
    V::from_elements(&[top_left, top_right, bottom_right, bottom_left])
}

pub trait RectVaryings {
    type Element;
    fn top_left(&self) -> Self::Element;
    fn top_right(&self) -> Self::Element;
    fn bottom_right(&self) -> Self::Element;
    fn bottom_left(&self) -> Self::Element;
    fn from_elements(elements: &[Self::Element; 4]) -> Self;
}

impl RectVaryings for RectColors {
    type Element = ColorF;
    fn top_left(&self) -> ColorF { self.top_left }
    fn top_right(&self) -> ColorF { self.top_right }
    fn bottom_right(&self) -> ColorF { self.bottom_right }
    fn bottom_left(&self) -> ColorF { self.bottom_left }
    fn from_elements(elements: &[ColorF; 4]) -> RectColors {
        RectColors {
            top_left: elements[0],
            top_right: elements[1],
            bottom_right: elements[2],
            bottom_left: elements[3],
        }
    }
}

pub trait VaryingElement : Sized {
    fn scale(&self, factor: f32) -> Self;
    fn accumulate(values: &[Self; 4]) -> Self;
}

impl VaryingElement for ColorF {
    fn scale(&self, factor: f32) -> ColorF {
        ColorF {
            r: self.r * factor,
            g: self.g * factor,
            b: self.b * factor,
            a: self.a * factor,
        }
    }
    fn accumulate(values: &[ColorF; 4]) -> ColorF {
        ColorF {
            r: values[0].r + values[1].r + values[2].r + values[3].r,
            g: values[0].g + values[1].g + values[2].g + values[3].g,
            b: values[0].b + values[1].b + values[2].b + values[3].b,
            a: values[0].a + values[1].a + values[2].a + values[3].a,
        }
    }
}

// Don't use `euclid`'s `is_empty` because that has effectively has an "and" in the conditional
// below instead of an "or".
pub fn rect_is_empty<N:PartialEq + Zero>(rect: &Rect<N>) -> bool {
    rect.size.width == Zero::zero() || rect.size.height == Zero::zero()
}

/// Returns true if the rectangle's width and height are both strictly positive and false
/// otherwise.
pub fn rect_is_well_formed_and_nonempty(rect: &Rect<f32>) -> bool {
    rect.size.width > 0.0 && rect.size.height > 0.0
}

/// Multiplies all non-alpha components of a color by the given value.
pub fn scale_color(color: &ColorF, factor: f32) -> ColorF {
    ColorF {
        r: color.r * factor,
        g: color.g * factor,
        b: color.b * factor,
        a: color.a,
    }
}

/// Subdivides a rectangle into quadrants formed by a point. The quadrants are returned in the
/// order of: top left, top right, bottom right, and bottom left.
pub fn subdivide_rect_into_quadrants(rect: &Rect<f32>, point: &Point2D<f32>) -> (Rect<f32>, Rect<f32>, Rect<f32>, Rect<f32>) {
    let point = Point2D::new(clamp(point.x, rect.origin.x, rect.max_x()),
                             clamp(point.y, rect.origin.y, rect.max_y()));
    let tl_rect = Rect::new(rect.origin,
                            Size2D::new(point.x - rect.origin.x, point.y - rect.origin.y));
    let tr_rect = Rect::new(Point2D::new(point.x, rect.origin.y),
                            Size2D::new(rect.max_x() - point.x, point.y - rect.origin.y));
    let br_rect = Rect::new(point,
                            Size2D::new(rect.max_x() - point.x, rect.max_y() - point.y));
    let bl_rect = Rect::new(Point2D::new(rect.origin.x, point.y),
                            Size2D::new(point.x - rect.origin.x, rect.max_y() - point.y));
    return (tl_rect, tr_rect, br_rect, bl_rect);

    fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
        if x < lo {
            lo
        } else if x > hi {
            hi
        } else {
            x
        }
    }
}

/// Returns the center point of the given rect.
pub fn rect_center(rect: &Rect<f32>) -> Point2D<f32> {
    Point2D::new(rect.origin.x + rect.size.width / 2.0, rect.origin.y + rect.size.height / 2.0)
}

pub fn distance(a: &Point2D<f32>, b: &Point2D<f32>) -> f32 {
    let (x, y) = (b.x - a.x, b.y - a.y);
    (x * x + y * y).sqrt()
}

pub fn lerp_points(a: &Point2D<f32>, b: &Point2D<f32>, t: f32) -> Point2D<f32> {
    Point2D::new(lerp(a.x, b.x, t), lerp(a.y, b.y, t))
}

