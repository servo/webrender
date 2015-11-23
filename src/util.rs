use euclid::{Matrix4, Point2D, Rect, Size2D};
use internal_types::{PackedVertex, RectColors, RectColorsUv, RectUv};
use std::num::Zero;
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

impl RectVaryings for RectUv {
    type Element = Point2D<f32>;
    fn top_left(&self) -> Point2D<f32> { self.top_left }
    fn top_right(&self) -> Point2D<f32> { self.top_right }
    fn bottom_right(&self) -> Point2D<f32> { self.bottom_right }
    fn bottom_left(&self) -> Point2D<f32> { self.bottom_left }
    fn from_elements(elements: &[Point2D<f32>; 4]) -> RectUv {
        RectUv {
            top_left: elements[0],
            top_right: elements[1],
            bottom_right: elements[2],
            bottom_left: elements[3],
        }
    }
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

impl RectVaryings for RectColorsUv {
    type Element = (ColorF, Point2D<f32>);
    fn top_left(&self) -> (ColorF, Point2D<f32>) {
        (self.colors.top_left, self.uv.top_left)
    }
    fn top_right(&self) -> (ColorF, Point2D<f32>) {
        (self.colors.top_right, self.uv.top_right)
    }
    fn bottom_left(&self) -> (ColorF, Point2D<f32>) {
        (self.colors.bottom_left, self.uv.bottom_left)
    }
    fn bottom_right(&self) -> (ColorF, Point2D<f32>) {
        (self.colors.bottom_right, self.uv.bottom_right)
    }
    fn from_elements(elements: &[(ColorF, Point2D<f32>); 4]) -> RectColorsUv {
        let colors = [elements[0].0, elements[1].0, elements[2].0, elements[3].0];
        let uv = [elements[0].1, elements[1].1, elements[2].1, elements[3].1];
        RectColorsUv {
            colors: RectColors::from_elements(&colors),
            uv: RectUv::from_elements(&uv),
        }
    }
}

pub trait VaryingElement : Sized {
    fn scale(&self, factor: f32) -> Self;
    fn accumulate(values: &[Self; 4]) -> Self;
    fn make_packed_vertex(&self, position: &Point2D<f32>, muv: &Point2D<f32>) -> PackedVertex;
}

impl VaryingElement for Point2D<f32> {
    fn scale(&self, factor: f32) -> Point2D<f32> {
        *self * factor
    }
    fn accumulate(values: &[Point2D<f32>; 4]) -> Point2D<f32> {
        values[0] + values[1] + values[2] + values[3]
    }
    fn make_packed_vertex(&self, position: &Point2D<f32>, muv: &Point2D<f32>) -> PackedVertex {
        static WHITE: ColorF = ColorF {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        };
        PackedVertex::from_points(position, &WHITE, self, muv)
    }
}

impl VaryingElement for (ColorF, Point2D<f32>) {
    fn scale(&self, factor: f32) -> (ColorF, Point2D<f32>) {
        let color = ColorF {
            r: self.0.r * factor,
            g: self.0.g * factor,
            b: self.0.b * factor,
            a: self.0.a * factor,
        };
        (color, self.1 * factor)
    }
    fn accumulate(values: &[(ColorF, Point2D<f32>); 4]) -> (ColorF, Point2D<f32>) {
        let color = ColorF {
            r: values[0].0.r + values[1].0.r + values[2].0.r + values[3].0.r,
            g: values[0].0.g + values[1].0.g + values[2].0.g + values[3].0.g,
            b: values[0].0.b + values[1].0.b + values[2].0.b + values[3].0.b,
            a: values[0].0.a + values[1].0.a + values[2].0.a + values[3].0.a,
        };
        (color, values[0].1 + values[1].1 + values[2].1 + values[3].1)
    }
    fn make_packed_vertex(&self, position: &Point2D<f32>, muv: &Point2D<f32>) -> PackedVertex {
        PackedVertex::from_points(position, &self.0, &self.1, muv)
    }
}

// Don't use `euclid`'s `is_empty` because that has effectively has an "and" in the conditional
// below instead of an "or".
pub fn rect_is_empty<N:PartialEq + Zero>(rect: &Rect<N>) -> bool {
    rect.size.width == Zero::zero() || rect.size.height == Zero::zero()
}
