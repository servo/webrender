use euclid::{Matrix4, Point2D, Rect, Size2D};
use internal_types::RectUv;
use std::num::Zero;
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
        if self.name.chars().next() != Some(' ') {
            let t1 = precise_time_ns();
            let ms = (t1 - self.t0) as f64 / 1000000f64;
            println!("{} {}", self.name, ms);
        }
    }
}

// TODO: Implement these in euclid!
pub trait MatrixHelpers {
    fn transform_rect(&self, rect: &Rect<f32>) -> Rect<f32>;
}

impl MatrixHelpers for Matrix4 {
    #[inline]
    fn transform_rect(&self, rect: &Rect<f32>) -> Rect<f32> {
        let top_left = self.transform_point(&rect.origin);
        let top_right = self.transform_point(&rect.top_right());
        let bottom_left = self.transform_point(&rect.bottom_left());
        let bottom_right = self.transform_point(&rect.bottom_right());
        let (mut min_x, mut min_y) = (top_left.x.clone(), top_left.y.clone());
        let (mut max_x, mut max_y) = (min_x.clone(), min_y.clone());
        for point in [ top_right, bottom_left, bottom_right ].iter() {
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
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    (b - a) * t + a
}

pub fn bilerp(point: &Point2D<f32>, quad: &Rect<f32>, uv: &RectUv) -> Point2D<f32> {
    let (x1, y1, x2, y2) = (quad.origin.x, quad.origin.y, quad.max_x(), quad.max_y());
    (uv.top_left * (x2 - point.x) * (y2 - point.y) +
     uv.top_right * (point.x - x1) * (y2 - point.y) +
     uv.bottom_left * (x2 - point.x) * (point.y - y1) +
     uv.bottom_right * (point.x - x1) * (point.y - y1)) / ((x2 - x1) * (y2 - y1))
}

// Don't use `euclid`'s `is_empty` because that has effectively has an "and" in the conditional
// below instead of an "or".
pub fn rect_is_empty<N:PartialEq + Zero>(rect: &Rect<N>) -> bool {
    rect.size.width == Zero::zero() || rect.size.height == Zero::zero()
}
