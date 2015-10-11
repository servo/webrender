use euclid::{Matrix4, Point2D, Rect, Size2D};
use time::precise_time_ns;
use internal_types::{ClipRectToRegionMaskResult, RenderPass};
use types::{ColorF, ImageFormat};

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
            //if ms > 0.1 {
            println!("{} {}", self.name, ms);
        }
    }
}

pub fn get_render_pass(colors: &[ColorF],
                       format: ImageFormat,
                       mask_result: &Option<ClipRectToRegionMaskResult>)
                       -> RenderPass {
    if colors.iter().any(|color| color.a < 1.0) || mask_result.is_some() {
        return RenderPass::Alpha
    }

    match format {
        ImageFormat::A8 => RenderPass::Alpha,
        ImageFormat::RGBA8 => RenderPass::Alpha,
        ImageFormat::RGB8 => RenderPass::Opaque,
        ImageFormat::Invalid => unreachable!(),
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
