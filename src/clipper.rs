use euclid::{Point2D, Rect, Size2D};
use internal_types::{ClipRectToRegionMaskResult, ClipRectToRegionResult, PolygonPosColorUv};
use internal_types::{RectPosUv, WorkVertex};
use render_backend::MAX_RECT;
use simd::f32x4;
use std::mem;
use types::{BoxShadowClipMode, ClipRegion, ColorF};
use util;

/// Computes whether the point c is inside the clipping edge ab.
///
/// NB: Assumes clockwise winding for the clipping polygon that the edge comes from.
fn is_inside(a: &Point2D<f32>, b: &Point2D<f32>, c: &WorkVertex) -> bool {
    let ba = *b - *a;
    let cb = c.position() - *b;
    ba.x * cb.y >= ba.y * cb.x
}

fn intersection(a: &Point2D<f32>, b: &Point2D<f32>, p: &WorkVertex, q: &WorkVertex) -> WorkVertex {
    let d1 = *b - *a;
    let d2 = Point2D::new(q.x - p.x, q.y - p.y);
    let dot = d1.x * d2.y - d1.y * d2.x;
    if dot < 0.001 && dot > -0.001 {
        return *p
    }
    let c = Point2D::new(p.x, p.y) - *a;
    let t = (c.x * d2.y - c.y * d2.x) / dot;
    let (x, y) = (a.x + t * d1.x, a.y + t * d1.y);

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
    let mut p_uv = f32x4::new(p.u, p.v, 0.0, 0.0);
    let q_uv = f32x4::new(q.u, q.v, 0.0, 0.0);
    let simd_ratio = f32x4::new(ratio, ratio, ratio, ratio);
    let mut p_rgba = f32x4::new(p.r, p.g, p.b, p.a);
    let q_rgba = f32x4::new(q.r, q.g, q.b, q.a);
    p_uv = p_uv + simd_ratio * (q_uv - p_uv);
    p_rgba = p_rgba + simd_ratio * (q_rgba - p_rgba);

    let color = ColorF::new(p_rgba.extract(0),
                            p_rgba.extract(1),
                            p_rgba.extract(2),
                            p_rgba.extract(3));

    WorkVertex::new(x, y, &color, p_uv.extract(0), p_uv.extract(1))
}

// We reuse these buffers for clipping algorithms
pub struct TypedClipBuffers<P> {
    pub polygon_scratch: Vec<P>,
    pub polygon_output: Vec<P>,
    pub clip_rect_to_region_result_scratch: Vec<ClipRectToRegionResult<P>>,
    pub clip_rect_to_region_result_output: Vec<ClipRectToRegionResult<P>>,
}

impl<P> TypedClipBuffers<P> {
    pub fn new() -> TypedClipBuffers<P> {
        TypedClipBuffers {
            polygon_scratch: Vec::new(),
            polygon_output: Vec::new(),
            clip_rect_to_region_result_scratch: Vec::new(),
            clip_rect_to_region_result_output: Vec::new(),
        }
    }
}

pub struct ClipBuffers {
    pub sh_clip_buffers: ShClipBuffers,
    pub rect_pos_uv: TypedClipBuffers<RectPosUv>,
    pub polygon_pos_color_uv: TypedClipBuffers<PolygonPosColorUv>,
}

impl ClipBuffers {
    pub fn new() -> ClipBuffers {
        ClipBuffers {
            sh_clip_buffers: ShClipBuffers::new(),
            rect_pos_uv: TypedClipBuffers::new(),
            polygon_pos_color_uv: TypedClipBuffers::new(),
        }
    }
}

/// Clip buffers for the Sutherland-Hodgman clipping routine.
pub struct ShClipBuffers {
    pub input: Vec<WorkVertex>,
    pub result: Vec<WorkVertex>,
}

impl ShClipBuffers {
    pub fn new() -> ShClipBuffers {
        ShClipBuffers {
            input: Vec::new(),
            result: Vec::new(),
        }
    }
}

/// Clips the given polygon to a clip polygon.
///
/// NB: Assumes clockwise winding for the clip polygon.
pub fn clip_polygon<'a>(buffers: &'a mut ShClipBuffers,
                        polygon: &[WorkVertex],
                        clip_polygon: &[Point2D<f32>])
                        -> &'a [WorkVertex] {
    let ShClipBuffers {ref mut input, ref mut result} = *buffers;
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

pub fn clip_rect_with_mode<P>(polygon: P,
                              sh_clip_buffers: &mut ShClipBuffers,
                              clip_rect: &Rect<f32>,
                              clip_mode: BoxShadowClipMode,
                              output: &mut Vec<P>)
                              where P: Polygon {
    match clip_mode {
        BoxShadowClipMode::None => output.push(polygon),
        BoxShadowClipMode::Inset => polygon.clip_to_rect(sh_clip_buffers, clip_rect, output),
        BoxShadowClipMode::Outset => polygon.clip_out_rect(sh_clip_buffers, clip_rect, output),
    }
}

fn clip_to_region<P>(polygon: P,
                     sh_clip_buffers: &mut ShClipBuffers,
                     polygon_scratch: &mut Vec<P>,
                     clip_rect_to_region_result_scratch: &mut Vec<ClipRectToRegionResult<P>>,
                     output: &mut Vec<ClipRectToRegionResult<P>>,
                     region: &ClipRegion)
                     where P: Polygon + Clone {
    polygon_scratch.clear();
    polygon.clip_to_rect(sh_clip_buffers, &region.main, polygon_scratch);
    if polygon_scratch.is_empty() {
        return
    }

    for main_result in polygon_scratch.drain(..) {
        output.push(ClipRectToRegionResult::new(main_result, None));
    }

    for complex_region in region.complex.iter() {
        mem::swap(clip_rect_to_region_result_scratch, output);
        for intermediate_result in clip_rect_to_region_result_scratch.drain(..) {
            // Quick rejection:
            let intermediate_polygon = intermediate_result.rect_result.clone();
            if !intermediate_polygon.intersects_rect(&complex_region.rect) {
                continue
            }

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
            if !util::rect_is_empty(&inner_rect) {
                intermediate_polygon.clip_to_rect(sh_clip_buffers, &inner_rect, polygon_scratch);
                push_results(output, polygon_scratch, None);
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
            if !util::rect_is_empty(&top_rect) {
                intermediate_polygon.clip_to_rect(sh_clip_buffers, &top_rect, polygon_scratch);
                push_results(output, polygon_scratch, None);
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
            if !util::rect_is_empty(&bottom_rect) {
                intermediate_polygon.clip_to_rect(sh_clip_buffers, &bottom_rect, polygon_scratch);
                push_results(output, polygon_scratch, None);
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
            if !util::rect_is_empty(&corner_rect) {
                let mask_rect = Rect::new(Point2D::new(0.0, 0.0), Size2D::new(1.0, 1.0));
                intermediate_polygon.clip_to_rect(sh_clip_buffers, &corner_rect, polygon_scratch);
                push_results(output,
                             polygon_scratch,
                             Some(ClipRectToRegionMaskResult::new(&mask_rect,
                                                                  &corner_rect,
                                                                  border_radius)))
            }

            // B:
            corner_rect.origin = Point2D::new(complex_region.rect.max_x() - border_radius,
                                              complex_region.rect.origin.y);
            if !util::rect_is_empty(&corner_rect) {
                intermediate_polygon.clip_to_rect(sh_clip_buffers, &corner_rect, polygon_scratch);
                let mask_rect = Rect::new(Point2D::new(1.0, 0.0), Size2D::new(-1.0, 1.0));
                push_results(output,
                             polygon_scratch,
                             Some(ClipRectToRegionMaskResult::new(&mask_rect,
                                                                  &corner_rect,
                                                                  border_radius)))
            }

            // C:
            corner_rect.origin = Point2D::new(complex_region.rect.origin.x,
                                              complex_region.rect.max_y() - border_radius);
            if !util::rect_is_empty(&corner_rect) {
                intermediate_polygon.clip_to_rect(sh_clip_buffers, &corner_rect, polygon_scratch);
                let mask_rect = Rect::new(Point2D::new(0.0, 1.0), Size2D::new(1.0, -1.0));
                push_results(output,
                             polygon_scratch,
                             Some(ClipRectToRegionMaskResult::new(&mask_rect,
                                                                  &corner_rect,
                                                                  border_radius)))
            }

            // D:
            corner_rect.origin = Point2D::new(complex_region.rect.max_x() - border_radius,
                                              complex_region.rect.max_y() - border_radius);
            if !util::rect_is_empty(&corner_rect) {
                intermediate_polygon.clip_to_rect(sh_clip_buffers, &corner_rect, polygon_scratch);
                let mask_rect = Rect::new(Point2D::new(1.0, 1.0), Size2D::new(-1.0, -1.0));
                push_results(output,
                             polygon_scratch,
                             Some(ClipRectToRegionMaskResult::new(&mask_rect,
                                                                  &corner_rect,
                                                                  border_radius)))
            }
        }
    }

    fn size_max(size: &Size2D<f32>) -> f32 {
        f32::max(size.width, size.height)
    }

    fn push_results<P>(output: &mut Vec<ClipRectToRegionResult<P>>,
                       clip_rect_results: &mut Vec<P>,
                       mask_result: Option<ClipRectToRegionMaskResult>)
                       where P: Polygon {
        output.extend(clip_rect_results.drain(..).map(|clip_rect_result| {
            ClipRectToRegionResult::new(clip_rect_result, mask_result)
        }))
    }
}

/// NB: Clobbers both `polygon_scratch` and `polygon_output` in the typed clip buffers.
pub fn clip_rect_with_mode_and_to_region<P>(polygon: P,
                                            sh_clip_buffers: &mut ShClipBuffers,
                                            typed_clip_buffers: &mut TypedClipBuffers<P>,
                                            clip_rect: &Rect<f32>,
                                            clip_mode: BoxShadowClipMode,
                                            clip_region: &ClipRegion)
                                            where P: Polygon + Clone {
    typed_clip_buffers.polygon_scratch.clear();
    clip_rect_with_mode(polygon,
                        sh_clip_buffers,
                        clip_rect,
                        clip_mode,
                        &mut typed_clip_buffers.polygon_scratch);

    for initial_clip_result in typed_clip_buffers.polygon_scratch.drain(..) {
        clip_to_region(initial_clip_result,
                       sh_clip_buffers,
                       &mut typed_clip_buffers.polygon_output,
                       &mut typed_clip_buffers.clip_rect_to_region_result_scratch,
                       &mut typed_clip_buffers.clip_rect_to_region_result_output,
                       clip_region)
    }
}

pub trait Polygon : Sized {
    fn clip_to_rect(&self,
                    sh_clip_buffers: &mut ShClipBuffers,
                    clip_rect: &Rect<f32>,
                    output: &mut Vec<Self>);
    fn clip_out_rect(&self,
                     sh_clip_buffers: &mut ShClipBuffers,
                     clip_rect: &Rect<f32>,
                     output: &mut Vec<Self>);
    fn intersects_rect(&self, rect: &Rect<f32>) -> bool;
}

impl Polygon for RectPosUv {
    fn clip_to_rect(&self,
                    _: &mut ShClipBuffers,
                    clip_rect: &Rect<f32>,
                    output: &mut Vec<RectPosUv>) {
        for clipped_rect in self.pos.intersection(clip_rect).into_iter() {
            if util::rect_is_empty(&clipped_rect) {
                continue
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

            let origins = f32x4::new(self.pos.origin.x, self.pos.origin.y,
                                     self.pos.origin.x, self.pos.origin.y);

            let sizes = f32x4::new(self.pos.size.width, self.pos.size.height,
                                   self.pos.size.width, self.pos.size.height);

            let uv_origins = f32x4::new(self.uv.origin.x, self.uv.origin.y,
                                        self.uv.origin.x, self.uv.origin.y);
            let uv_sizes = f32x4::new(self.uv.size.width, self.uv.size.height,
                                      self.uv.size.width, self.uv.size.height);
            let f = ((clip - origins) / sizes) * uv_sizes + uv_origins;

            output.push(RectPosUv {
                pos: Rect::new(Point2D::new(clip.extract(0), clip.extract(1)),
                               Size2D::new(clip.extract(2) - clip.extract(0),
                                           clip.extract(3) - clip.extract(1))),
                uv: Rect::new(Point2D::new(f.extract(0), f.extract(1)),
                              Size2D::new(f.extract(2) - f.extract(0),
                                          f.extract(3) - f.extract(1))),
            })
        }
    }

    fn clip_out_rect(&self,
                     _: &mut ShClipBuffers,
                     clip_rect: &Rect<f32>,
                     output: &mut Vec<RectPosUv>) {
        let clip_rect = match self.pos.intersection(clip_rect) {
            Some(clip_rect) => clip_rect,
            None => {
                output.push(*self);
                return
            }
        };

        // FIXME(pcwalton): Clip the u and v too.
        push(output,
             &self.uv,
             &self.pos.origin,
             &Point2D::new(self.pos.max_x(), clip_rect.origin.y));
        push(output,
             &self.uv,
             &Point2D::new(self.pos.origin.x, clip_rect.origin.y),
             &clip_rect.bottom_left());
        push(output,
             &self.uv,
             &clip_rect.top_right(),
             &Point2D::new(self.pos.max_x(), clip_rect.max_y()));
        push(output,
             &self.uv,
             &Point2D::new(self.pos.origin.x, clip_rect.max_y()),
             &self.pos.bottom_right());

        fn push(result: &mut Vec<RectPosUv>,
                uv: &Rect<f32>,
                top_left: &Point2D<f32>,
                bottom_right: &Point2D<f32>) {
            if top_left.x >= bottom_right.x || top_left.y >= bottom_right.y {
                return
            }
            result.push(RectPosUv {
                pos: Rect::new(*top_left,
                               Size2D::new(bottom_right.x - top_left.x,
                                           bottom_right.y - top_left.y)),
                uv: *uv,
            })
        }
    }

    fn intersects_rect(&self, rect: &Rect<f32>) -> bool {
        self.pos.intersects(rect)
    }
}

impl PolygonPosColorUv {
    fn clip_to_polygon(&self,
                       clip_buffers: &mut ShClipBuffers,
                       clip_vertices: &[Point2D<f32>],
                       output: &mut Vec<PolygonPosColorUv>) {
        let clipped_vertices = clip_polygon(clip_buffers,
                                            &self.vertices[..],
                                            clip_vertices).to_vec();
        output.push(PolygonPosColorUv {
            vertices: clipped_vertices,
        })
    }
}

impl Polygon for PolygonPosColorUv {
    fn clip_to_rect(&self,
                    clip_buffers: &mut ShClipBuffers,
                    clip_rect: &Rect<f32>,
                    output: &mut Vec<PolygonPosColorUv>) {
        self.clip_to_polygon(clip_buffers,
                             &[
                                clip_rect.origin,
                                clip_rect.top_right(),
                                clip_rect.bottom_right(),
                                clip_rect.bottom_left(),
                             ],
                             output)
    }

    fn clip_out_rect(&self,
                     clip_buffers: &mut ShClipBuffers,
                     clip_rect: &Rect<f32>,
                     output: &mut Vec<PolygonPosColorUv>) {
        //
        //  +-----------------------+
        //  | 8                     | 9
        //  |                       |
        //  |      +---------+------+
        //  |      | 4       | 1    | 5, 10
        //  |      |         |      |
        //  |      +---------+      |
        //  |        3         2    |
        //  |                       |
        //  +-----------------------+
        //    7                       6
        //

        self.clip_to_polygon(clip_buffers,
                             &[
                                clip_rect.top_right(),                                  // 1
                                clip_rect.bottom_right(),                               // 2
                                clip_rect.bottom_left(),                                // 3
                                clip_rect.origin,                                       // 4
                                Point2D::new(MAX_RECT.max_x(), clip_rect.origin.y),     // 5
                                MAX_RECT.bottom_right(),                                // 6
                                MAX_RECT.bottom_left(),                                 // 7
                                MAX_RECT.origin,                                        // 8
                                MAX_RECT.top_right(),                                   // 9
                                Point2D::new(MAX_RECT.max_x(), clip_rect.origin.y),     // 10
                             ],
                             output)
    }

    fn intersects_rect(&self, rect: &Rect<f32>) -> bool {
        self.vertices.iter().any(|vertex| rect.contains(&Point2D::new(vertex.x, vertex.y)))
    }
}

