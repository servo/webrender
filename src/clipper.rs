use euclid::{Point2D, Rect, Size2D};
use internal_types::{ClipRectResult, ClipRectToRegionMaskResult, ClipRectToRegionResult};
use internal_types::{WorkVertex};
use simd::f32x4;
use std::mem;
use types::{BoxShadowClipMode, ClipRegion, ColorF};

fn is_inside(a: &Point2D<f32>, b: &Point2D<f32>, c: &WorkVertex) -> bool {
    (a.x - c.x) * (b.y - c.y) > (a.y - c.y) * (b.x - c.x)
}

fn intersection(a: &Point2D<f32>, b: &Point2D<f32>, p: &WorkVertex, q: &WorkVertex) -> WorkVertex {
    let denominator = (a.x - b.x) * (p.y - q.y) - (a.y - b.y) * (p.x - q.x);
    let axby = a.x*b.y - b.x*a.y;
    let pxqy = p.x*q.y - p.y*q.x;
    let x = (axby * (p.x-q.x) - (a.x-b.x) * pxqy) / denominator;
    let y = (axby * (p.y-q.y) - (a.y-b.y) * pxqy) / denominator;

    let d1 = ((p.x - x) * (p.x - x) + (p.y - y) * (p.y - y)).sqrt();
    let d2 = ((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y)).sqrt();
    let ratio = d1 / d2;

    // de-simd'd code:
    // let u = p.u + ratio * (q.u - p.u);
    // let v = p.v + ratio * (q.v - p.v);
    // let mu = p.mu + ratio * (q.mu - p.mu);
    // let mv = p.mv + ratio * (q.mv - p.mv);
    // let r = p.r + ratio * (q.r - p.r);
    // let g = p.g + ratio * (q.g - p.g);
    // let b = p.b + ratio * (q.b - p.b);
    // let a = p.a + ratio * (q.a - p.a);
    let mut p_uv = f32x4::new(p.u, p.v, p.mu, p.mv);
    let q_uv = f32x4::new(q.u, q.v, q.mu, q.mv);
    let simd_ratio = f32x4::new(ratio, ratio, ratio, ratio);
    let mut p_rgba = f32x4::new(p.r, p.g, p.b, p.a);
    let q_rgba = f32x4::new(q.r, q.g, q.b, q.a);
    p_uv = p_uv + simd_ratio * (q_uv - p_uv);
    p_rgba = p_rgba + simd_ratio * (q_rgba - p_rgba);

    let color = ColorF::new(p_rgba.extract(0),
                            p_rgba.extract(1),
                            p_rgba.extract(2),
                            p_rgba.extract(0));

    WorkVertex::new(x, y, &color, p_uv.extract(0),
                                  p_uv.extract(1),
                                  p_uv.extract(2),
                                  p_uv.extract(3))
}

// We reuse these buffers for clipping algorithms
pub struct ClipBuffers {
    input: Vec<WorkVertex>,
    result: Vec<WorkVertex>,
}

impl ClipBuffers {
    pub fn new() -> ClipBuffers {
        ClipBuffers {
            input: Vec::new(),
            result: Vec::new(),
        }
    }
}

pub fn clip_polygon<'a>(buffers: &'a mut ClipBuffers, polygon: &[WorkVertex],
                    clip_polygon: &[Point2D<f32>]) -> &'a [WorkVertex] {

    let ClipBuffers {ref mut input, ref mut result} = *buffers;
    input.clear();
    result.clear();
    for vert in polygon {
        result.push(vert.clone());
    }
    let clip_len = clip_polygon.len();

    for i in 0..clip_len {
        input.clone_from(&result);
        let input_len = input.len();
        result.clear();

        let a = &clip_polygon[(i + clip_len-1) % clip_len];
        let b = &clip_polygon[i];

        for j in 0..input.len() {
            let p = &input[(j + input_len-1) % input_len];
            let q = &input[j];

            let p_is_inside = is_inside(a, b, p);

            if is_inside(a, b, q) {
                if !p_is_inside {
                    result.push(intersection(a, b, p, q));
                }
                result.push(q.clone());
            } else if p_is_inside {
                result.push(intersection(a, b, p, q));
            }
        }
    }

    result
}

pub fn clip_rect_pos_uv(pos: &Rect<f32>, uv: &Rect<f32>, clip_rect: &Rect<f32>) -> Option<ClipRectResult> {
    pos.intersection(clip_rect).and_then(|clipped_rect| {
        if rect_is_empty(&clipped_rect) {
            return None
        }

        // de-simd'd code:
        // let cx0 = clipped_rect.origin.x;
        // let cy0 = clipped_rect.origin.y;
        // let cx1 = cx0 + clipped_rect.size.width;
        // let cy1 = cy0 + clipped_rect.size.height;

        // let f0 = (cx0 - pos.origin.x) / pos.size.width;
        // let f1 = (cy0 - pos.origin.y) / pos.size.height;
        // let f2 = (cx1 - pos.origin.x) / pos.size.width;
        // let f3 = (cy1 - pos.origin.y) / pos.size.height;

        // ClipRectResult {
        //     x0: cx0,
        //     y0: cy0,
        //     x1: cx1,
        //     y1: cy1,
        //     u0: uv.origin.x + f0 * uv.size.width,
        //     v0: uv.origin.y + f1 * uv.size.height,
        //     u1: uv.origin.x + f2 * uv.size.width,
        //     v1: uv.origin.y + f3 * uv.size.height,
        // }

        let clip = f32x4::new(clipped_rect.origin.x,
                              clipped_rect.origin.y,
                              clipped_rect.origin.x + clipped_rect.size.width,
                              clipped_rect.origin.y + clipped_rect.size.height);

        let origins = f32x4::new(pos.origin.x, pos.origin.y,
                                 pos.origin.x, pos.origin.y);

        let sizes = f32x4::new(pos.size.width, pos.size.height,
                               pos.size.width, pos.size.height);

        let uv_origins = f32x4::new(uv.origin.x, uv.origin.y,
                                    uv.origin.x, uv.origin.y);
        let uv_sizes = f32x4::new(uv.size.width, uv.size.height,
                                  uv.size.width, uv.size.height);
        let f = ((clip - origins) / sizes) * uv_sizes + uv_origins;

        Some(ClipRectResult {
            x0: clip.extract(0),
            y0: clip.extract(1),
            x1: clip.extract(2),
            y1: clip.extract(3),
            u0: f.extract(0),
            v0: f.extract(1),
            u1: f.extract(2),
            v1: f.extract(3),
        })
    })
}

pub fn clip_out_rect_pos_uv(pos: &Rect<f32>, uv: &Rect<f32>, clip_rect: &Rect<f32>)
                            -> Vec<ClipRectResult> {
    let clip_rect = match pos.intersection(clip_rect) {
        Some(clip_rect) => clip_rect,
        None => return vec![ClipRectResult::from_rects(pos, uv)],
    };

    // FIXME(pcwalton): Clip the u and v too.
    let mut result = vec![];
    push(&mut result, uv, &pos.origin, &Point2D::new(pos.max_x(), clip_rect.origin.y));
    push(&mut result,
         uv,
         &Point2D::new(pos.origin.x, clip_rect.origin.y),
         &clip_rect.bottom_left());
    push(&mut result, uv, &clip_rect.top_right(), &Point2D::new(pos.max_x(), clip_rect.max_y()));
    push(&mut result, uv, &Point2D::new(pos.origin.x, clip_rect.max_y()), &pos.bottom_right());
    return result;

    fn push(result: &mut Vec<ClipRectResult>,
            uv: &Rect<f32>,
            top_left: &Point2D<f32>,
            bottom_right: &Point2D<f32>) {
        if top_left.x >= bottom_right.x || top_left.y >= bottom_right.y {
            return
        }
        result.push(ClipRectResult {
            x0: top_left.x,
            y0: top_left.y,
            x1: bottom_right.x,
            y1: bottom_right.y,
            u0: uv.origin.x,
            v0: uv.origin.y,
            u1: uv.max_x(),
            v1: uv.max_y(),
        })
    }
}

pub fn clip_rect_with_mode_pos_uv(pos: &Rect<f32>,
                                  uv: &Rect<f32>,
                                  clip_rect: &Rect<f32>,
                                  clip_mode: BoxShadowClipMode)
                                  -> Vec<ClipRectResult> {
    match clip_mode {
        BoxShadowClipMode::None => vec![ClipRectResult::from_rects(pos, uv)],
        BoxShadowClipMode::Inset => {
            match clip_rect_pos_uv(pos, uv, clip_rect) {
                Some(clip_result) => vec![clip_result],
                None => vec![],
            }
        }
        BoxShadowClipMode::Outset => {
            clip_out_rect_pos_uv(pos, uv, clip_rect)
        }
    }
}

fn clip_rect_to_region_pos_uv(pos: &Rect<f32>, uv: &Rect<f32>, region: &ClipRegion)
                              -> Vec<ClipRectToRegionResult> {
    let main_result = match clip_rect_pos_uv(pos, uv, &region.main) {
        Some(main_result) => main_result,
        None => return vec![],
    };

    let mut result = vec![ClipRectToRegionResult::new(main_result, None)];
    for complex_region in region.complex.iter() {
        for intermediate_result in mem::replace(&mut result, vec![]) {
            // Quick rejection:
            let intermediate_rect = intermediate_result.rect_result.rect();
            if !complex_region.rect.intersects(&intermediate_rect) {
                continue
            }
            let intermediate_uv_rect = intermediate_result.rect_result.uv_rect();

            // FIXME(pcwalton): This is pretty bogus. I guess we should create a region for the
            // inner area -- which may not be rectangular due to nonequal border radii! -- and use
            // Sutherland-Hodgman clipping for it.
            let border_radius =
                f32::max(f32::max(f32::max(size_max(&complex_region.radii.top_left),
                                           size_max(&complex_region.radii.top_right)),
                                  size_max(&complex_region.radii.bottom_left)),
                         size_max(&complex_region.radii.bottom_right));

            // Compute the middle intersected region:
            //
            //   +--+-----------------+--+
            //   | /|                 |\ | 
            //   +--+-----------------+--+
            //   |#######################|
            //   |#######################|
            //   |#######################|
            //   +--+-----------------+--+
            //   | \|                 |/ | 
            //   +--+-----------------+--+
            let inner_rect = Rect::new(
                Point2D::new(complex_region.rect.origin.x,
                             complex_region.rect.origin.y + border_radius),
                Size2D::new(complex_region.rect.size.width,
                            complex_region.rect.size.height - (border_radius + border_radius)));
            if !rect_is_empty(&inner_rect) {
                if let Some(clip_rect_result) =
                        clip_rect_pos_uv(&intermediate_rect,
                                         &intermediate_result.rect_result.uv_rect(),
                                         &inner_rect) {
                    result.push(ClipRectToRegionResult::new(clip_rect_result, None))
                }
            }

            // Compute the top region:
            //
            //   +--+-----------------+--+
            //   | /|#################|\ |
            //   +--+-----------------+--+
            //   |                       |
            //   |                       |
            //   |                       |
            //   +--+-----------------+--+
            //   | \|                 |/ |
            //   +--+-----------------+--+
            let top_rect = Rect::new(
                Point2D::new(complex_region.rect.origin.x + border_radius,
                             complex_region.rect.origin.y),
                Size2D::new(complex_region.rect.size.width - (border_radius + border_radius),
                            border_radius));
            if !rect_is_empty(&top_rect) {
                if let Some(clip_rect_result) = clip_rect_pos_uv(&intermediate_rect,
                                                                 &intermediate_uv_rect,
                                                                 &top_rect) {
                    result.push(ClipRectToRegionResult::new(clip_rect_result, None))
                }
            }

            // Compute the bottom region:
            //
            //   +--+-----------------+--+
            //   | /|                 |\ |
            //   +--+-----------------+--+
            //   |                       |
            //   |                       |
            //   |                       |
            //   +--+-----------------+--+
            //   | \|#################|/ |
            //   +--+-----------------+--+
            let bottom_rect = Rect::new(
                Point2D::new(complex_region.rect.origin.x + border_radius,
                             complex_region.rect.max_y() - border_radius),
                Size2D::new(complex_region.rect.size.width - (border_radius + border_radius),
                            border_radius));
            if !rect_is_empty(&bottom_rect) {
                if let Some(clip_rect_result) = clip_rect_pos_uv(&intermediate_rect,
                                                                 &intermediate_uv_rect,
                                                                 &bottom_rect) {
                    result.push(ClipRectToRegionResult::new(clip_rect_result, None))
                }
            }

            // Now for the corners:
            //
            //     +--+-----------------+--+
            //   A | /|                 |\ | B
            //     +--+-----------------+--+
            //     |                       |
            //     |                       |
            //     |                       |
            //     +--+-----------------+--+
            //   C | \|                 |/ | D
            //     +--+-----------------+--+
            //
            // FIXME(pcwalton): This should clip the mask u and v properly too. For now we just
            // blindly assume that the border was not clipped.

            // Compute A:
            let mut corner_rect = Rect::new(complex_region.rect.origin,
                                            Size2D::new(border_radius, border_radius));
            if !rect_is_empty(&corner_rect) {
                if let Some(clip_rect_result) = clip_rect_pos_uv(&intermediate_rect,
                                                                 &intermediate_uv_rect,
                                                                 &corner_rect) {
                    let mask_rect = Rect::new(Point2D::new(0.0, 0.0), Size2D::new(1.0, 1.0));
                    result.push(ClipRectToRegionResult::new(
                            clip_rect_result,
                            Some(ClipRectToRegionMaskResult::new(&mask_rect, border_radius))));
                }
            }

            // B:
            corner_rect.origin = Point2D::new(complex_region.rect.max_x() - border_radius,
                                              complex_region.rect.origin.y);
            if !rect_is_empty(&corner_rect) {
                if let Some(clip_rect_result) = clip_rect_pos_uv(&intermediate_rect,
                                                                 &intermediate_uv_rect,
                                                                 &corner_rect) {
                    let mask_rect = Rect::new(Point2D::new(1.0, 0.0), Size2D::new(-1.0, 1.0));
                    result.push(ClipRectToRegionResult::new(
                            clip_rect_result,
                            Some(ClipRectToRegionMaskResult::new(&mask_rect, border_radius))));
                }
            }

            // C:
            corner_rect.origin = Point2D::new(complex_region.rect.origin.x,
                                              complex_region.rect.max_y() - border_radius);
            if !rect_is_empty(&corner_rect) {
                if let Some(clip_rect_result) = clip_rect_pos_uv(&intermediate_rect,
                                                                 &intermediate_uv_rect,
                                                                 &corner_rect) {
                    let mask_rect = Rect::new(Point2D::new(0.0, 1.0), Size2D::new(1.0, -1.0));
                    result.push(ClipRectToRegionResult::new(
                            clip_rect_result,
                            Some(ClipRectToRegionMaskResult::new(&mask_rect, border_radius))));
                }
            }

            // D:
            corner_rect.origin = Point2D::new(complex_region.rect.max_x() - border_radius,
                                              complex_region.rect.max_y() - border_radius);
            if !rect_is_empty(&corner_rect) {
                if let Some(clip_rect_result) = clip_rect_pos_uv(&intermediate_rect,
                                                                 &intermediate_uv_rect,
                                                                 &corner_rect) {
                    let mask_rect = Rect::new(Point2D::new(1.0, 1.0), Size2D::new(-1.0, -1.0));
                    result.push(ClipRectToRegionResult::new(
                            clip_rect_result,
                            Some(ClipRectToRegionMaskResult::new(&mask_rect, border_radius))));
                }
            }

        }
    }

    // Done!
    return result;

    fn size_max(size: &Size2D<f32>) -> f32 {
        f32::max(size.width, size.height)
    }
}

pub fn clip_rect_with_mode_and_to_region_pos_uv(pos: &Rect<f32>,
                                                uv: &Rect<f32>,
                                                clip_rect: &Rect<f32>,
                                                clip_mode: BoxShadowClipMode,
                                                clip_region: &ClipRegion)
                                                -> Vec<ClipRectToRegionResult> {
    let initial_results = clip_rect_with_mode_pos_uv(pos, uv, clip_rect, clip_mode);
    let mut final_results = vec![];
    for initial_clip_result in initial_results.into_iter() {
        final_results.extend(clip_rect_to_region_pos_uv(&initial_clip_result.rect(),
                                                        &initial_clip_result.uv_rect(),
                                                        clip_region).into_iter())
    }
    final_results
}

// Don't use `euclid`'s `is_empty` because that has effectively has an "and" in the conditional
// below instead of an "or".
fn rect_is_empty(rect: &Rect<f32>) -> bool {
    rect.size.width == 0.0 || rect.size.height == 0.0
}

