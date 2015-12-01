use euclid::{Point2D, Rect, Size2D};
use internal_types::{ClipRectToRegionMaskResult, ClipRectToRegionResult, CombinedClipRegion};
use internal_types::{PolygonPosColorUv, RectColorsUv, RectPolygon, RectUv, WorkVertex};
//use simd::f32x4;
use std::fmt::Debug;
use std::mem;
use webrender_traits::{ColorF, ComplexClipRegion};
use util::{self, RectVaryings, VaryingElement};

pub static MAX_RECT: Rect<f32> = Rect {
    origin: Point2D {
        x: -1000.0,
        y: -1000.0,
    },
    size: Size2D {
        width: 10000.0,
        height: 10000.0,
    },
};

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
    let u = p.u + ratio * (q.u - p.u);
    let v = p.v + ratio * (q.v - p.v);
    let r = p.r + ratio * (q.r - p.r);
    let g = p.g + ratio * (q.g - p.g);
    let b = p.b + ratio * (q.b - p.b);
    let a = p.a + ratio * (q.a - p.a);

    let color = ColorF::new(r, g, b, a);
    WorkVertex::new(x, y, &color, u, v)

    /*
    let mut p_uv = f32x4::new(p.u, p.v, 0.0, 0.0);
    let q_uv = f32x4::new(q.u, q.v, 0.0, 0.0);
    let simd_ratio = f32x4::new(ratio, ratio, ratio, ratio);
    let mut p_rgba = f32x4::new(p.r, p.g, p.b, p.a);
    let q_rgba = f32x4::new(q.r, q.g, q.b, q.a);
    p_uv = p_uv + simd_ratio * (q_uv - p_uv);
    p_rgba = p_rgba + simd_ratio * (q_rgba - p_rgba);*

    let color = ColorF::new(p_rgba.extract(0),
                            p_rgba.extract(1),
                            p_rgba.extract(2),
                            p_rgba.extract(3));

    WorkVertex::new(x, y, &color, p_uv.extract(0), p_uv.extract(1))
    */
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
    pub rect_pos_uv: TypedClipBuffers<RectPolygon<RectUv>>,
    pub rect_pos_colors_uv: TypedClipBuffers<RectPolygon<RectColorsUv>>,
    pub polygon_pos_color_uv: TypedClipBuffers<PolygonPosColorUv>,
}

impl ClipBuffers {
    pub fn new() -> ClipBuffers {
        ClipBuffers {
            sh_clip_buffers: ShClipBuffers::new(),
            rect_pos_uv: TypedClipBuffers::new(),
            rect_pos_colors_uv: TypedClipBuffers::new(),
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

fn clip_to_complex_region<P>(
        sh_clip_buffers: &mut ShClipBuffers,
        polygon_scratch: &mut Vec<P>,
        clip_rect_to_region_result_scratch: &mut Vec<ClipRectToRegionResult<P>>,
        output: &mut Vec<ClipRectToRegionResult<P>>,
        complex_region: &ComplexClipRegion)
        where P: Polygon + Clone {
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

        if border_radius == 0.0 {
            intermediate_polygon.clip_to_rect(sh_clip_buffers,
                                              &complex_region.rect,
                                              polygon_scratch);
            push_results(output, polygon_scratch, None);
            continue
        }

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
pub fn clip_rect_to_combined_region<P>(polygon: P,
                                       sh_clip_buffers: &mut ShClipBuffers,
                                       typed_clip_buffers: &mut TypedClipBuffers<P>,
                                       clip_region: &CombinedClipRegion)
                                       where P: Polygon + Debug + Clone {
    polygon.clip_to_rect(sh_clip_buffers,
                         &clip_region.clip_in_rect,
                         &mut typed_clip_buffers.polygon_scratch);

    if let Some(ref clip_out_complex) = clip_region.clip_out_complex {
        for initial_clip_result in typed_clip_buffers.polygon_scratch.drain(..) {
            initial_clip_result.clip_out_rect(sh_clip_buffers,
                                              &clip_out_complex.rect,
                                              &mut typed_clip_buffers.polygon_output);
        }
        mem::swap(&mut typed_clip_buffers.polygon_output,
                  &mut typed_clip_buffers.polygon_scratch)
    }

    typed_clip_buffers.clip_rect_to_region_result_output
                      .extend(typed_clip_buffers.polygon_scratch.drain(..).map(|clip_rect_result| {
        ClipRectToRegionResult::new(clip_rect_result, None)
    }));

    if let Some(ref clip_in_complex) = clip_region.clip_in_complex {
        clip_to_complex_region(sh_clip_buffers,
                               &mut typed_clip_buffers.polygon_output,
                               &mut typed_clip_buffers.clip_rect_to_region_result_scratch,
                               &mut typed_clip_buffers.clip_rect_to_region_result_output,
                               clip_in_complex);
    }

    for clip_in_complex in clip_region.clip_in_complex_stack {
        clip_to_complex_region(sh_clip_buffers,
                               &mut typed_clip_buffers.polygon_output,
                               &mut typed_clip_buffers.clip_rect_to_region_result_scratch,
                               &mut typed_clip_buffers.clip_rect_to_region_result_output,
                               clip_in_complex)
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

impl<Varyings> RectPolygon<Varyings> where Varyings: RectVaryings,
                                           Varyings::Element: VaryingElement {
    fn push_clipped_rect(&self,
                         clipped_rect: &Rect<f32>,
                         output: &mut Vec<RectPolygon<Varyings>>) {
        if util::rect_is_empty(&clipped_rect) {
            return
        }

        output.push(RectPolygon {
            pos: *clipped_rect,
            varyings: util::bilerp_rect(clipped_rect, &self.pos, &self.varyings),
        });
    }
}

impl<Varyings> Polygon for RectPolygon<Varyings> where Varyings: RectVaryings + Clone,
                                                       Varyings::Element: VaryingElement {
    fn clip_to_rect(&self,
                    _: &mut ShClipBuffers,
                    clip_rect: &Rect<f32>,
                    output: &mut Vec<RectPolygon<Varyings>>) {
        for clipped_rect in self.pos.intersection(clip_rect).iter() {
            self.push_clipped_rect(clipped_rect, output)
        }
    }

    fn clip_out_rect(&self,
                     _: &mut ShClipBuffers,
                     clip_rect: &Rect<f32>,
                     output: &mut Vec<RectPolygon<Varyings>>) {
        let clip_rect = match self.pos.intersection(clip_rect) {
            Some(clip_rect) => clip_rect,
            None => {
                output.push((*self).clone());
                return
            }
        };

        self.push_clipped_rect(&Rect::new(self.pos.origin,
                                          Size2D::new(self.pos.size.width,
                                                      clip_rect.origin.y - self.pos.origin.y)),
                               output);
        self.push_clipped_rect(&Rect::new(Point2D::new(self.pos.origin.x, clip_rect.origin.y),
                                          Size2D::new(clip_rect.origin.x - self.pos.origin.x,
                                                      clip_rect.size.height)),
                               output);
        self.push_clipped_rect(&Rect::new(Point2D::new(clip_rect.max_x(), clip_rect.origin.y),
                                          Size2D::new(self.pos.max_x() - clip_rect.max_x(),
                                                      clip_rect.size.height)),
                               output);
        self.push_clipped_rect(&Rect::new(Point2D::new(self.pos.origin.x, clip_rect.max_y()),
                                          Size2D::new(self.pos.size.width,
                                                      self.pos.max_y() - clip_rect.max_y())),
                               output);
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
