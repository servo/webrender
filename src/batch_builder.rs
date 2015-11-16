use app_units::Au;
use batch::{BatchBuilder, MatrixIndex, TileParams};
use clipper::{self, ClipBuffers, Polygon};
use device::{TextureId, TextureIndex};
use euclid::{Rect, Point2D, Size2D};
use fnv::FnvHasher;
use internal_types::{CombinedClipRegion, RectPosUv};
use internal_types::{RectUv, Primitive, BorderRadiusRasterOp, RasterItem, ClipRectToRegionResult};
use internal_types::{GlyphKey, PackedVertex, WorkVertex};
use internal_types::{PolygonPosColorUv, BorderEdgeDirection};
use internal_types::{BasicRotationAngle, BoxShadowRasterOp};
use renderer::BLUR_INFLATION_FACTOR;
use resource_cache::ResourceCache;
use std::collections::HashMap;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::hash_state::DefaultState;
use tessellator::{self, BorderCornerTessellation};
use texture_cache::{TextureCacheItem};
use webrender_traits::{ColorF, ImageFormat, BorderStyle, BoxShadowClipMode};
use webrender_traits::{BorderRadius, BorderSide, FontKey, GlyphInstance, ImageKey};
use webrender_traits::{BorderDisplayItem, GradientStop, ComplexClipRegion};

const BORDER_DASH_SIZE: f32 = 3.0;

impl<'a> BatchBuilder<'a> {
    #[inline]
    fn add_textured_rectangle(&mut self,
                              matrix_index: MatrixIndex,
                              rect: &Rect<f32>,
                              clip: &CombinedClipRegion,
                              image_info: &TextureCacheItem,
                              resource_cache: &ResourceCache,
                              clip_buffers: &mut ClipBuffers,
                              color: &ColorF) {
        self.add_axis_aligned_gradient_with_texture(matrix_index,
                                                    rect,
                                                    clip,
                                                    image_info,
                                                    resource_cache,
                                                    clip_buffers,
                                                    &[*color, *color, *color, *color])
    }

    #[inline]
    pub fn add_color_rectangle(&mut self,
                               matrix_index: MatrixIndex,
                               rect: &Rect<f32>,
                               clip: &CombinedClipRegion,
                               resource_cache: &ResourceCache,
                               clip_buffers: &mut ClipBuffers,
                               color: &ColorF) {
        self.add_axis_aligned_gradient(matrix_index,
                                       rect,
                                       clip,
                                       resource_cache,
                                       clip_buffers,
                                       &[*color, *color, *color, *color])
    }

    pub fn add_image(&mut self,
                     matrix_index: MatrixIndex,
                     rect: &Rect<f32>,
                     clip: &CombinedClipRegion,
                     stretch_size: &Size2D<f32>,
                     image_key: ImageKey,
                     resource_cache: &ResourceCache,
                     clip_buffers: &mut ClipBuffers,
                     color: &ColorF) {
        // Should be caught higher up
        debug_assert!(stretch_size.width > 0.0 && stretch_size.height > 0.0);
        let image_info = resource_cache.get_image(image_key);

        let u1 = rect.size.width / stretch_size.width;
        let v1 = rect.size.height / stretch_size.height;

        let uv = RectUv {
            top_left: Point2D::zero(),
            top_right: Point2D::new(u1, 0.0),
            bottom_left: Point2D::new(0.0, v1),
            bottom_right: Point2D::new(u1, v1),
        };

        let tile_params = TileParams {
            u0: image_info.u0,
            v0: image_info.v0,
            u_size: image_info.u1 - image_info.u0,
            v_size: image_info.v1 - image_info.v0,
        };

        clipper::clip_rect_to_combined_region(RectPosUv {
                                                pos: *rect,
                                                uv: uv
                                              },
                                              &mut clip_buffers.sh_clip_buffers,
                                              &mut clip_buffers.rect_pos_uv,
                                              clip);
        for clip_region in clip_buffers.rect_pos_uv.clip_rect_to_region_result_output.drain(..) {
            let mask = mask_for_clip_region(resource_cache, &clip_region, false);
            let colors = [*color, *color, *color, *color];
            let mut vertices = clip_region.make_packed_vertices_for_rect(&colors,
                                                                         mask,
                                                                         image_info.texture_index);

            self.add_draw_item(matrix_index,
                               image_info.texture_id,
                               mask.texture_id,
                               Primitive::Rectangles,
                               &mut vertices,
                               Some(tile_params.clone()));
        }
    }

    pub fn add_text(&mut self,
                    matrix_index: MatrixIndex,
                    rect: &Rect<f32>,
                    clip: &CombinedClipRegion,
                    font_key: FontKey,
                    size: Au,
                    blur_radius: Au,
                    color: &ColorF,
                    glyphs: &Vec<GlyphInstance>,
                    resource_cache: &ResourceCache,
                    clip_buffers: &mut ClipBuffers,
                    device_pixel_ratio: f32) {
        let dummy_mask_image = resource_cache.get_dummy_mask_image();

        // Logic below to pick the primary render item depends on len > 0!
        assert!(glyphs.len() > 0);

        let need_text_clip = !clip.clip_in_rect.contains(&rect.origin) ||
                             !clip.clip_in_rect.contains(&rect.bottom_right());

        let mut glyph_key = GlyphKey::new(font_key, size, blur_radius, glyphs[0].index);
        let blur_offset = blur_radius.to_f32_px() * (BLUR_INFLATION_FACTOR as f32) / 2.0;

        let mut text_batches: HashMap<(TextureId, TextureIndex), Vec<RectPosUv>, DefaultState<FnvHasher>> =
            HashMap::with_hash_state(Default::default());

        for glyph in glyphs {
            glyph_key.index = glyph.index;
            let image_info = resource_cache.get_glyph(&glyph_key);
            if let Some(image_info) = image_info {
                let x = glyph.x + image_info.user_data.x0 as f32 / device_pixel_ratio - blur_offset;
                let y = glyph.y - image_info.user_data.y0 as f32 / device_pixel_ratio - blur_offset;
                let width = image_info.width as f32 / device_pixel_ratio;
                let height = image_info.height as f32 / device_pixel_ratio;

                let uv = RectUv {
                    top_left: Point2D::new(image_info.u0, image_info.v0),
                    top_right: Point2D::new(image_info.u1, image_info.v0),
                    bottom_left: Point2D::new(image_info.u0, image_info.v1),
                    bottom_right: Point2D::new(image_info.u1, image_info.v1),
                };

                let rect = RectPosUv {
                    pos: Rect::new(Point2D::new(x, y),
                                   Size2D::new(width, height)),
                    uv: uv,
                };

                let rect_buffer = match text_batches.entry((image_info.texture_id,
                                                            image_info.texture_index)) {
                    Occupied(entry) => {
                        entry.into_mut()
                    }
                    Vacant(entry) => {
                        entry.insert(Vec::new())
                    }
                };

                rect_buffer.push(rect);
            }
        }

        let mut vertex_buffer = Vec::new();
        for ((texture_id, texture_index), mut rect_buffer) in text_batches {
            let rect_buffer = if need_text_clip {
                let mut clipped_rects = Vec::new();
                for rect in rect_buffer.drain(..) {
                    rect.clip_to_rect(&mut clip_buffers.sh_clip_buffers,
                                      &clip.clip_in_rect,
                                      &mut clipped_rects);
                }
                clipped_rects
            } else {
                rect_buffer
            };

            vertex_buffer.clear();

            for rect in rect_buffer {
                let x0 = rect.pos.origin.x;
                let y0 = rect.pos.origin.y;
                let x1 = x0 + rect.pos.size.width;
                let y1 = y0 + rect.pos.size.height;

                vertex_buffer.push(PackedVertex::from_components(
                        x0, y0,
                        color,
                        rect.uv.top_left.x, rect.uv.top_left.y,
                        dummy_mask_image.u0, dummy_mask_image.v0,
                        texture_index,
                        dummy_mask_image.texture_index));
                vertex_buffer.push(PackedVertex::from_components(
                        x1, y0,
                        color,
                        rect.uv.top_right.x, rect.uv.top_right.y,
                        dummy_mask_image.u1, dummy_mask_image.v0,
                        texture_index,
                        dummy_mask_image.texture_index));
                vertex_buffer.push(PackedVertex::from_components(
                        x0, y1,
                        color,
                        rect.uv.bottom_left.x, rect.uv.bottom_left.y,
                        dummy_mask_image.u0, dummy_mask_image.v1,
                        texture_index,
                        dummy_mask_image.texture_index));
                vertex_buffer.push(PackedVertex::from_components(
                        x1, y1,
                        color,
                        rect.uv.bottom_right.x, rect.uv.bottom_right.y,
                        dummy_mask_image.v0, dummy_mask_image.v1,
                        texture_index,
                        dummy_mask_image.texture_index));
            }

            self.add_draw_item(matrix_index,
                               texture_id,
                               dummy_mask_image.texture_id,
                               Primitive::Glyphs,
                               &mut vertex_buffer,
                               None);
        }
    }

    // Colors are in the order: top left, top right, bottom right, bottom left.
    #[inline]
    fn add_axis_aligned_gradient(&mut self,
                                 matrix_index: MatrixIndex,
                                 rect: &Rect<f32>,
                                 clip: &CombinedClipRegion,
                                 resource_cache: &ResourceCache,
                                 clip_buffers: &mut ClipBuffers,
                                 colors: &[ColorF; 4]) {
        let white_image = resource_cache.get_dummy_color_image();
        self.add_axis_aligned_gradient_with_texture(matrix_index,
                                                    rect,
                                                    clip,
                                                    white_image,
                                                    resource_cache,
                                                    clip_buffers,
                                                    colors);
    }

    // Colors are in the order: top left, top right, bottom right, bottom left.
    fn add_axis_aligned_gradient_with_texture(&mut self,
                                              matrix_index: MatrixIndex,
                                              rect: &Rect<f32>,
                                              clip: &CombinedClipRegion,
                                              image_info: &TextureCacheItem,
                                              resource_cache: &ResourceCache,
                                              clip_buffers: &mut ClipBuffers,
                                              colors: &[ColorF; 4]) {
        if rect.size.width == 0.0 || rect.size.height == 0.0 {
            return
        }

        let uv = RectUv {
            top_left: Point2D::new(image_info.u0, image_info.v0),
            top_right: Point2D::new(image_info.u1, image_info.v0),
            bottom_left: Point2D::new(image_info.u0, image_info.v1),
            bottom_right: Point2D::new(image_info.u1, image_info.v1),
        };

        clipper::clip_rect_to_combined_region(
            RectPosUv {
                pos: *rect,
                uv: uv,
            },
            &mut clip_buffers.sh_clip_buffers,
            &mut clip_buffers.rect_pos_uv,
            clip);
        for clip_region in clip_buffers.rect_pos_uv.clip_rect_to_region_result_output.drain(..) {
            let mask = mask_for_clip_region(resource_cache, &clip_region, false);

            let mut vertices = clip_region.make_packed_vertices_for_rect(colors,
                                                                         mask,
                                                                         image_info.texture_index);

            self.add_draw_item(matrix_index,
                               image_info.texture_id,
                               mask.texture_id,
                               Primitive::Rectangles,
                               &mut vertices,
                               None);
        }
    }

    pub fn add_gradient(&mut self,
                        matrix_index: MatrixIndex,
                        clip: &CombinedClipRegion,
                        start_point: &Point2D<f32>,
                        end_point: &Point2D<f32>,
                        stops: &[GradientStop],
                        resource_cache: &ResourceCache,
                        clip_buffers: &mut ClipBuffers) {
        let white_image = resource_cache.get_dummy_color_image();

        debug_assert!(stops.len() >= 2);

        let dir_x = end_point.x - start_point.x;
        let dir_y = end_point.y - start_point.y;
        let dir_len = (dir_x * dir_x + dir_y * dir_y).sqrt();
        let dir_xn = dir_x / dir_len;
        let dir_yn = dir_y / dir_len;
        let perp_xn = -dir_yn;
        let perp_yn = dir_xn;

        for i in 0..stops.len()-1 {
            let stop0 = &stops[i];
            let stop1 = &stops[i+1];

            if stop0.offset == stop1.offset {
                continue;
            }

            let color0 = &stop0.color;
            let color1 = &stop1.color;

            let start_x = start_point.x + stop0.offset * (end_point.x - start_point.x);
            let start_y = start_point.y + stop0.offset * (end_point.y - start_point.y);

            let end_x = start_point.x + stop1.offset * (end_point.x - start_point.x);
            let end_y = start_point.y + stop1.offset * (end_point.y - start_point.y);

            let len_scale = 1000.0;     // todo: determine this properly!!

            let x0 = start_x - perp_xn * len_scale;
            let y0 = start_y - perp_yn * len_scale;

            let x1 = end_x - perp_xn * len_scale;
            let y1 = end_y - perp_yn * len_scale;

            let x2 = end_x + perp_xn * len_scale;
            let y2 = end_y + perp_yn * len_scale;

            let x3 = start_x + perp_xn * len_scale;
            let y3 = start_y + perp_yn * len_scale;

            let gradient_polygon = PolygonPosColorUv {
                vertices: vec![
                    WorkVertex::new(x0, y0, color0, 0.0, 0.0),
                    WorkVertex::new(x1, y1, color1, 0.0, 0.0),
                    WorkVertex::new(x2, y2, color1, 0.0, 0.0),
                    WorkVertex::new(x3, y3, color0, 0.0, 0.0),
                ],
            };

            { // scope for buffers
                clipper::clip_rect_to_combined_region(gradient_polygon,
                                                      &mut clip_buffers.sh_clip_buffers,
                                                      &mut clip_buffers.polygon_pos_color_uv,
                                                      clip);
                for clip_result in clip_buffers.polygon_pos_color_uv
                                               .clip_rect_to_region_result_output
                                               .drain(..) {
                    let mask = mask_for_clip_region(resource_cache, &clip_result, false);

                    let mut packed_vertices = Vec::new();
                    if clip_result.rect_result.vertices.len() >= 3 {
                        for vert in clip_result.rect_result.vertices.iter() {
                            packed_vertices.push(clip_result.make_packed_vertex(
                                    &vert.position(),
                                    &vert.uv(),
                                    &vert.color(),
                                    &mask,
                                    white_image.texture_index));
                        }
                    }

                    if packed_vertices.len() > 0 {
                        self.add_draw_item(matrix_index,
                                           white_image.texture_id,
                                           mask.texture_id,
                                           Primitive::TriangleFan,
                                           &mut packed_vertices,
                                           None);
                    }
                }
            }
        }
    }

    pub fn add_box_shadow(&mut self,
                          matrix_index: MatrixIndex,
                          box_bounds: &Rect<f32>,
                          clip: &CombinedClipRegion,
                          box_offset: &Point2D<f32>,
                          color: &ColorF,
                          blur_radius: f32,
                          spread_radius: f32,
                          border_radius: f32,
                          clip_mode: BoxShadowClipMode,
                          resource_cache: &ResourceCache,
                          clip_buffers: &mut ClipBuffers) {
        let rect = compute_box_shadow_rect(box_bounds, box_offset, spread_radius);

        // Fast path.
        if blur_radius == 0.0 && spread_radius == 0.0 && clip_mode == BoxShadowClipMode::None {
            self.add_color_rectangle(matrix_index, &rect, clip, resource_cache, clip_buffers, color);
            return;
        }

        // Draw the corners.
        self.add_box_shadow_corners(matrix_index,
                                    box_bounds,
                                    box_offset,
                                    color,
                                    blur_radius,
                                    spread_radius,
                                    border_radius,
                                    clip_mode,
                                    clip,
                                    resource_cache,
                                    clip_buffers);

        // Draw the sides.
        self.add_box_shadow_sides(matrix_index,
                                  box_bounds,
                                  clip,
                                  box_offset,
                                  color,
                                  blur_radius,
                                  spread_radius,
                                  border_radius,
                                  clip_mode,
                                  resource_cache,
                                  clip_buffers);

        match clip_mode {
            BoxShadowClipMode::None => {
                // Fill the center area.
                self.add_color_rectangle(matrix_index,
                                         box_bounds,
                                         clip,
                                         resource_cache,
                                         clip_buffers,
                                         color);
            }
            BoxShadowClipMode::Outset => {
                // Fill the center area.
                let metrics = BoxShadowMetrics::new(&rect, border_radius, blur_radius);
                let center_rect = Rect::new(metrics.tl_inner,
                                            Size2D::new(metrics.br_inner.x - metrics.tl_inner.x,
                                                        metrics.br_inner.y - metrics.tl_inner.y));
                let mut clip = *clip;
                clip.clip_out(&ComplexClipRegion::from_rect(&box_bounds));
                self.add_color_rectangle(matrix_index,
                                         &center_rect,
                                         &clip,
                                         resource_cache,
                                         clip_buffers,
                                         color);
            }
            BoxShadowClipMode::Inset => {
                // Fill in the outsides.
                self.fill_outside_area_of_inset_box_shadow(matrix_index,
                                                           box_bounds,
                                                           clip,
                                                           box_offset,
                                                           color,
                                                           blur_radius,
                                                           spread_radius,
                                                           border_radius,
                                                           resource_cache,
                                                           clip_buffers);
            }
        }
    }

    fn add_box_shadow_corners(&mut self,
                              matrix_index: MatrixIndex,
                              box_bounds: &Rect<f32>,
                              box_offset: &Point2D<f32>,
                              color: &ColorF,
                              blur_radius: f32,
                              spread_radius: f32,
                              border_radius: f32,
                              clip_mode: BoxShadowClipMode,
                              clip: &CombinedClipRegion,
                              resource_cache: &ResourceCache,
                              clip_buffers: &mut ClipBuffers) {
        // Draw the corners.
        //
        //      +--+------------------+--+
        //      |##|                  |##|
        //      +--+------------------+--+
        //      |  |                  |  |
        //      |  |                  |  |
        //      |  |                  |  |
        //      +--+------------------+--+
        //      |##|                  |##|
        //      +--+------------------+--+

        let rect = compute_box_shadow_rect(box_bounds, box_offset, spread_radius);
        let metrics = BoxShadowMetrics::new(&rect, border_radius, blur_radius);

        let clip = self.adjust_clip_for_box_shadow_clip_mode(clip,
                                                             box_bounds,
                                                             border_radius,
                                                             clip_mode);

        self.add_box_shadow_corner(matrix_index,
                                   &metrics.tl_outer,
                                   &metrics.tl_inner,
                                   &color,
                                   blur_radius,
                                   border_radius,
                                   clip_mode,
                                   &clip,
                                   resource_cache,
                                   clip_buffers,
                                   BasicRotationAngle::Upright);
        self.add_box_shadow_corner(matrix_index,
                                   &Point2D::new(metrics.tr_inner.x, metrics.tr_outer.y),
                                   &Point2D::new(metrics.tr_outer.x, metrics.tr_inner.y),
                                   &color,
                                   blur_radius,
                                   border_radius,
                                   clip_mode,
                                   &clip,
                                   resource_cache,
                                   clip_buffers,
                                   BasicRotationAngle::Clockwise90);
        self.add_box_shadow_corner(matrix_index,
                                   &metrics.br_inner,
                                   &metrics.br_outer,
                                   &color,
                                   blur_radius,
                                   border_radius,
                                   clip_mode,
                                   &clip,
                                   resource_cache,
                                   clip_buffers,
                                   BasicRotationAngle::Clockwise180);
        self.add_box_shadow_corner(matrix_index,
                                   &Point2D::new(metrics.bl_outer.x, metrics.bl_inner.y),
                                   &Point2D::new(metrics.bl_inner.x, metrics.bl_outer.y),
                                   &color,
                                   blur_radius,
                                   border_radius,
                                   clip_mode,
                                   &clip,
                                   resource_cache,
                                   clip_buffers,
                                   BasicRotationAngle::Clockwise270);
    }

    fn add_box_shadow_sides(&mut self,
                            matrix_index: MatrixIndex,
                            box_bounds: &Rect<f32>,
                            clip: &CombinedClipRegion,
                            box_offset: &Point2D<f32>,
                            color: &ColorF,
                            blur_radius: f32,
                            spread_radius: f32,
                            border_radius: f32,
                            clip_mode: BoxShadowClipMode,
                            resource_cache: &ResourceCache,
                            clip_buffers: &mut ClipBuffers) {
        let rect = compute_box_shadow_rect(box_bounds, box_offset, spread_radius);
        let metrics = BoxShadowMetrics::new(&rect, border_radius, blur_radius);

        let clip = self.adjust_clip_for_box_shadow_clip_mode(clip,
                                                             box_bounds,
                                                             border_radius,
                                                             clip_mode);

        // Draw the sides.
        //
        //      +--+------------------+--+
        //      |  |##################|  |
        //      +--+------------------+--+
        //      |##|                  |##|
        //      |##|                  |##|
        //      |##|                  |##|
        //      +--+------------------+--+
        //      |  |##################|  |
        //      +--+------------------+--+

        let horizontal_size = Size2D::new(metrics.br_inner.x - metrics.tl_inner.x,
                                          metrics.edge_size);
        let vertical_size = Size2D::new(metrics.edge_size,
                                        metrics.br_inner.y - metrics.tl_inner.y);
        let top_rect = Rect::new(metrics.tl_outer + Point2D::new(metrics.edge_size, 0.0),
                                 horizontal_size);
        let right_rect =
            Rect::new(metrics.tr_outer + Point2D::new(-metrics.edge_size, metrics.edge_size),
                      vertical_size);
        let bottom_rect =
            Rect::new(metrics.bl_outer + Point2D::new(metrics.edge_size, -metrics.edge_size),
                      horizontal_size);
        let left_rect = Rect::new(metrics.tl_outer + Point2D::new(0.0, metrics.edge_size),
                                  vertical_size);

        self.add_box_shadow_edge(matrix_index,
                                 &top_rect.origin,
                                 &top_rect.bottom_right(),
                                 color,
                                 blur_radius,
                                 border_radius,
                                 clip_mode,
                                 &clip,
                                 resource_cache,
                                 clip_buffers,
                                 BasicRotationAngle::Clockwise270);
        self.add_box_shadow_edge(matrix_index,
                                 &right_rect.origin,
                                 &right_rect.bottom_right(),
                                 color,
                                 blur_radius,
                                 border_radius,
                                 clip_mode,
                                 &clip,
                                 resource_cache,
                                 clip_buffers,
                                 BasicRotationAngle::Upright);
        self.add_box_shadow_edge(matrix_index,
                                 &bottom_rect.origin,
                                 &bottom_rect.bottom_right(),
                                 color,
                                 blur_radius,
                                 border_radius,
                                 clip_mode,
                                 &clip,
                                 resource_cache,
                                 clip_buffers,
                                 BasicRotationAngle::Clockwise90);
        self.add_box_shadow_edge(matrix_index,
                                 &left_rect.origin,
                                 &left_rect.bottom_right(),
                                 color,
                                 blur_radius,
                                 border_radius,
                                 clip_mode,
                                 &clip,
                                 resource_cache,
                                 clip_buffers,
                                 BasicRotationAngle::Clockwise180);
    }

    fn fill_outside_area_of_inset_box_shadow(&mut self,
                                             matrix_index: MatrixIndex,
                                             box_bounds: &Rect<f32>,
                                             clip: &CombinedClipRegion,
                                             box_offset: &Point2D<f32>,
                                             color: &ColorF,
                                             blur_radius: f32,
                                             spread_radius: f32,
                                             border_radius: f32,
                                             resource_cache: &ResourceCache,
                                             clip_buffers: &mut ClipBuffers) {
        let rect = compute_box_shadow_rect(box_bounds, box_offset, spread_radius);
        let metrics = BoxShadowMetrics::new(&rect, border_radius, blur_radius);

        let clip = self.adjust_clip_for_box_shadow_clip_mode(clip,
                                                             box_bounds,
                                                             border_radius,
                                                             BoxShadowClipMode::Inset);

        // Fill in the outside area of the box.
        //
        //            +------------------------------+
        //      A --> |##############################|
        //            +--+--+------------------+--+--+
        //            |##|  |                  |  |##|
        //            |##+--+------------------+--+##|
        //            |##|  |                  |  |##|
        //      D --> |##|  |                  |  |##| <-- B
        //            |##|  |                  |  |##|
        //            |##+--+------------------+--+##|
        //            |##|  |                  |  |##|
        //            +--+--+------------------+--+--+
        //      C --> |##############################|
        //            +------------------------------+

        // A:
        self.add_color_rectangle(matrix_index,
                                 &Rect::new(box_bounds.origin,
                                            Size2D::new(box_bounds.size.width,
                                                        metrics.tl_outer.y - box_bounds.origin.y)),
                                 &clip,
                                 resource_cache,
                                 clip_buffers,
                                 color);

        // B:
        self.add_color_rectangle(matrix_index,
                                 &Rect::new(metrics.tr_outer,
                                            Size2D::new(box_bounds.max_x() - metrics.tr_outer.x,
                                                        metrics.br_outer.y - metrics.tr_outer.y)),
                                 &clip,
                                 resource_cache,
                                 clip_buffers,
                                 color);

        // C:
        self.add_color_rectangle(matrix_index,
                                 &Rect::new(Point2D::new(box_bounds.origin.x, metrics.bl_outer.y),
                                            Size2D::new(box_bounds.size.width,
                                                        box_bounds.max_y() - metrics.br_outer.y)),
                                 &clip,
                                 resource_cache,
                                 clip_buffers,
                                 color);

        // D:
        self.add_color_rectangle(matrix_index,
                                 &Rect::new(Point2D::new(box_bounds.origin.x, metrics.tl_outer.y),
                                            Size2D::new(metrics.tl_outer.x - box_bounds.origin.x,
                                                        metrics.bl_outer.y - metrics.tl_outer.y)),
                                 &clip,
                                 resource_cache,
                                 clip_buffers,
                                 color);
    }

    fn adjust_clip_for_box_shadow_clip_mode<'b>(&mut self,
                                                clip: &CombinedClipRegion<'b>,
                                                box_bounds: &Rect<f32>,
                                                border_radius: f32,
                                                clip_mode: BoxShadowClipMode)
                                                -> CombinedClipRegion<'b> {
        let mut clip = *clip;
        match clip_mode {
            BoxShadowClipMode::None => {}
            BoxShadowClipMode::Inset => {
                clip.clip_in(&ComplexClipRegion {
                    rect: *box_bounds,
                    radii: BorderRadius::uniform(border_radius),
                });
            }
            BoxShadowClipMode::Outset => {
                // TODO(pcwalton): Support border radii here too, once we have inverted rounded
                // rect clip masks.
                clip.clip_out(&ComplexClipRegion::from_rect(box_bounds));
            }
        }
        clip
    }

    #[inline]
    fn add_border_edge(&mut self,
                       matrix_index: MatrixIndex,
                       rect: &Rect<f32>,
                       clip: &CombinedClipRegion,
                       direction: BorderEdgeDirection,
                       color: &ColorF,
                       border_style: BorderStyle,
                       resource_cache: &ResourceCache,
                       clip_buffers: &mut clipper::ClipBuffers) {
        // TODO: Check for zero width/height borders!
        if color.a <= 0.0 {
            return
        }

        match border_style {
            BorderStyle::Dashed => {
                let (extent, step) = match direction {
                    BorderEdgeDirection::Horizontal => {
                        (rect.size.width, rect.size.height * BORDER_DASH_SIZE)
                    }
                    BorderEdgeDirection::Vertical => {
                        (rect.size.height, rect.size.width * BORDER_DASH_SIZE)
                    }
                };
                let mut origin = 0.0;
                while origin < extent {
                    let dash_rect = match direction {
                        BorderEdgeDirection::Horizontal => {
                            Rect::new(Point2D::new(rect.origin.x + origin, rect.origin.y),
                                      Size2D::new(f32::min(step, extent - origin),
                                                  rect.size.height))
                        }
                        BorderEdgeDirection::Vertical => {
                            Rect::new(Point2D::new(rect.origin.x, rect.origin.y + origin),
                                      Size2D::new(rect.size.width,
                                                  f32::min(step, extent - origin)))
                        }
                    };

                    self.add_color_rectangle(matrix_index,
                                             &dash_rect,
                                             clip,
                                             resource_cache,
                                             clip_buffers,
                                             color);

                    origin += step + step;
                }
            }
            BorderStyle::Dotted => {
                let (extent, step) = match direction {
                    BorderEdgeDirection::Horizontal => (rect.size.width, rect.size.height),
                    BorderEdgeDirection::Vertical => (rect.size.height, rect.size.width),
                };
                let mut origin = 0.0;
                while origin < extent {
                    let (dot_rect, mask_radius) = match direction {
                        BorderEdgeDirection::Horizontal => {
                            (Rect::new(Point2D::new(rect.origin.x + origin, rect.origin.y),
                                       Size2D::new(f32::min(step, extent - origin),
                                                   rect.size.height)),
                             rect.size.height / 2.0)
                        }
                        BorderEdgeDirection::Vertical => {
                            (Rect::new(Point2D::new(rect.origin.x, rect.origin.y + origin),
                                       Size2D::new(rect.size.width,
                                                   f32::min(step, extent - origin))),
                             rect.size.width / 2.0)
                        }
                    };

                    let raster_op =
                        BorderRadiusRasterOp::create(&Size2D::new(mask_radius, mask_radius),
                                                     &Size2D::new(0.0, 0.0),
                                                     false,
                                                     0,
                                                     ImageFormat::RGBA8).expect(
                        "Didn't find border radius mask for dashed border!");
                    let raster_item = RasterItem::BorderRadius(raster_op);
                    let color_image = resource_cache.get_raster(&raster_item);

                    // Top left:
                    self.add_textured_rectangle(matrix_index,
                                                &Rect::new(dot_rect.origin,
                                                           Size2D::new(dot_rect.size.width / 2.0,
                                                                       dot_rect.size.height / 2.0)),
                                                clip,
                                                color_image,
                                                resource_cache,
                                                clip_buffers,
                                                color);

                    // Top right:
                    self.add_textured_rectangle(matrix_index,
                                                &Rect::new(dot_rect.top_right(),
                                                           Size2D::new(-dot_rect.size.width / 2.0,
                                                                       dot_rect.size.height / 2.0)),
                                                clip,
                                                color_image,
                                                resource_cache,
                                                clip_buffers,
                                                color);

                    // Bottom right:
                    self.add_textured_rectangle(matrix_index,
                                                &Rect::new(dot_rect.bottom_right(),
                                                            Size2D::new(-dot_rect.size.width / 2.0,
                                                                        -dot_rect.size.height / 2.0)),
                                                clip,
                                                color_image,
                                                resource_cache,
                                                clip_buffers,
                                                color);

                    // Bottom left:
                    self.add_textured_rectangle(matrix_index,
                                                &Rect::new(dot_rect.bottom_left(),
                                                           Size2D::new(dot_rect.size.width / 2.0,
                                                                       -dot_rect.size.height / 2.0)),
                                                clip,
                                                color_image,
                                                resource_cache,
                                                clip_buffers,
                                                color);

                    origin += step + step;
                }
            }
            BorderStyle::Double => {
                let (outer_rect, inner_rect) = match direction {
                    BorderEdgeDirection::Horizontal => {
                        (Rect::new(rect.origin,
                                   Size2D::new(rect.size.width, rect.size.height / 3.0)),
                         Rect::new(Point2D::new(rect.origin.x,
                                                rect.origin.y + rect.size.height * 2.0 / 3.0),
                                   Size2D::new(rect.size.width, rect.size.height / 3.0)))
                    }
                    BorderEdgeDirection::Vertical => {
                        (Rect::new(rect.origin,
                                   Size2D::new(rect.size.width / 3.0, rect.size.height)),
                         Rect::new(Point2D::new(rect.origin.x + rect.size.width * 2.0 / 3.0,
                                                rect.origin.y),
                                   Size2D::new(rect.size.width / 3.0, rect.size.height)))
                    }
                };
                self.add_color_rectangle(matrix_index,
                                         &outer_rect,
                                         clip,
                                         resource_cache,
                                         clip_buffers,
                                         color);
                self.add_color_rectangle(matrix_index,
                                         &inner_rect,
                                         clip,
                                         resource_cache,
                                         clip_buffers,
                                         color);
            }
            _ => {
                self.add_color_rectangle(matrix_index,
                                         rect,
                                         clip,
                                         resource_cache,
                                         clip_buffers,
                                         color);
            }
        }
    }

    #[inline]
    fn add_border_corner(&mut self,
                         matrix_index: MatrixIndex,
                         clip: &CombinedClipRegion,
                         vertices_rect: &Rect<f32>,
                         color0: &ColorF,
                         color1: &ColorF,
                         outer_radius: &Size2D<f32>,
                         inner_radius: &Size2D<f32>,
                         resource_cache: &ResourceCache,
                         clip_buffers: &mut clipper::ClipBuffers,
                         rotation_angle: BasicRotationAngle) {
        if color0.a <= 0.0 && color1.a <= 0.0 {
            return
        }

        // TODO: Check for zero width/height borders!
        let white_image = resource_cache.get_dummy_color_image();

        for rect_index in 0..tessellator::quad_count_for_border_corner(outer_radius) {
            let tessellated_rect = vertices_rect.tessellate_border_corner(outer_radius,
                                                                          inner_radius,
                                                                          rotation_angle,
                                                                          rect_index);
            let mask_image = match BorderRadiusRasterOp::create(outer_radius,
                                                                inner_radius,
                                                                false,
                                                                rect_index,
                                                                ImageFormat::A8) {
                Some(raster_item) => {
                    let raster_item = RasterItem::BorderRadius(raster_item);
                    resource_cache.get_raster(&raster_item)
                }
                None => {
                    resource_cache.get_dummy_mask_image()
                }
            };

            // FIXME(pcwalton): Either use RGBA8 textures instead of alpha masks here, or implement
            // a mask combiner.
            let mask_uv = RectUv::from_image_and_rotation_angle(mask_image, rotation_angle);
            let tessellated_rect = RectPosUv {
                pos: tessellated_rect,
                uv: mask_uv,
            };

            clipper::clip_rect_to_combined_region(tessellated_rect,
                                                  &mut clip_buffers.sh_clip_buffers,
                                                  &mut clip_buffers.rect_pos_uv,
                                                  clip);

            for clip_region in clip_buffers.rect_pos_uv
                                           .clip_rect_to_region_result_output
                                           .drain(..) {
                let rect_pos_uv = &clip_region.rect_result;
                let v0;
                let v1;
                let muv0;
                let muv1;
                let muv2;
                let muv3;
                match rotation_angle {
                    BasicRotationAngle::Upright => {
                        v0 = rect_pos_uv.pos.origin;
                        v1 = rect_pos_uv.pos.bottom_right();
                        muv0 = rect_pos_uv.uv.top_left;
                        muv1 = rect_pos_uv.uv.top_right;
                        muv2 = rect_pos_uv.uv.bottom_right;
                        muv3 = rect_pos_uv.uv.bottom_left;
                    }
                    BasicRotationAngle::Clockwise90 => {
                        v0 = rect_pos_uv.pos.top_right();
                        v1 = rect_pos_uv.pos.bottom_left();
                        muv0 = rect_pos_uv.uv.top_right;
                        muv1 = rect_pos_uv.uv.top_left;
                        muv2 = rect_pos_uv.uv.bottom_left;
                        muv3 = rect_pos_uv.uv.bottom_right;
                    }
                    BasicRotationAngle::Clockwise180 => {
                        v0 = rect_pos_uv.pos.bottom_right();
                        v1 = rect_pos_uv.pos.origin;
                        muv0 = rect_pos_uv.uv.bottom_right;
                        muv1 = rect_pos_uv.uv.bottom_left;
                        muv2 = rect_pos_uv.uv.top_left;
                        muv3 = rect_pos_uv.uv.top_right;
                    }
                    BasicRotationAngle::Clockwise270 => {
                        v0 = rect_pos_uv.pos.bottom_left();
                        v1 = rect_pos_uv.pos.top_right();
                        muv0 = rect_pos_uv.uv.bottom_left;
                        muv1 = rect_pos_uv.uv.bottom_right;
                        muv2 = rect_pos_uv.uv.top_right;
                        muv3 = rect_pos_uv.uv.top_left;
                    }
                }

                let mut vertices = [
                    PackedVertex::from_components(v0.x, v0.y,
                                                  color0,
                                                  0.0, 0.0,
                                                  muv0.x, muv0.y,
                                                  white_image.texture_index,
                                                  mask_image.texture_index),
                    PackedVertex::from_components(v1.x, v1.y,
                                                  color0,
                                                  0.0, 0.0,
                                                  muv2.x, muv2.y,
                                                  white_image.texture_index,
                                                  mask_image.texture_index),
                    PackedVertex::from_components(v0.x, v1.y,
                                                  color0,
                                                  0.0, 0.0,
                                                  muv3.x, muv3.y,
                                                  white_image.texture_index,
                                                  mask_image.texture_index),
                    PackedVertex::from_components(v0.x, v0.y,
                                                  color1,
                                                  0.0, 0.0,
                                                  muv0.x, muv0.y,
                                                  white_image.texture_index,
                                                  mask_image.texture_index),
                    PackedVertex::from_components(v1.x, v0.y,
                                                  color1,
                                                  0.0, 0.0,
                                                  muv1.x, muv1.y,
                                                  white_image.texture_index,
                                                  mask_image.texture_index),
                    PackedVertex::from_components(v1.x, v1.y,
                                                  color1,
                                                  0.0, 0.0,
                                                  muv2.x, muv2.y,
                                                  white_image.texture_index,
                                                  mask_image.texture_index),
                ];

                self.add_draw_item(matrix_index,
                                   white_image.texture_id,
                                   mask_image.texture_id,
                                   Primitive::Triangles,
                                   &mut vertices,
                                   None);
            }
        }
    }

    fn add_color_image_rectangle(&mut self,
                                 matrix_index: MatrixIndex,
                                 v0: &Point2D<f32>,
                                 v1: &Point2D<f32>,
                                 clip: &CombinedClipRegion,
                                 color0: &ColorF,
                                 color1: &ColorF,
                                 color_image: &TextureCacheItem,
                                 resource_cache: &ResourceCache,
                                 clip_buffers: &mut ClipBuffers,
                                 rotation_angle: BasicRotationAngle) {
        if color0.a <= 0.0 || color1.a <= 0.0 {
            return
        }

        let vertices_rect = Rect::new(*v0, Size2D::new(v1.x - v0.x, v1.y - v0.y));
        let color_uv = RectUv::from_image_and_rotation_angle(color_image, rotation_angle);

        clipper::clip_rect_to_combined_region(RectPosUv {
                                                pos: vertices_rect,
                                                uv: color_uv,
                                              },
                                              &mut clip_buffers.sh_clip_buffers,
                                              &mut clip_buffers.rect_pos_uv,
                                              clip);
        for clip_region in clip_buffers.rect_pos_uv.clip_rect_to_region_result_output.drain(..) {
            let mask = mask_for_clip_region(resource_cache,
                                            &clip_region,
                                            false);
            let colors = [*color0, *color0, *color1, *color1];
            let mut vertices = clip_region.make_packed_vertices_for_rect(&colors,
                                                                         mask,
                                                                         color_image.texture_index);

            self.add_draw_item(matrix_index,
                               color_image.texture_id,
                               mask.texture_id,
                               Primitive::Rectangles,
                               &mut vertices,
                               None);
        }
    }

    pub fn add_border(&mut self,
                      matrix_index: MatrixIndex,
                      rect: &Rect<f32>,
                      clip: &CombinedClipRegion,
                      info: &BorderDisplayItem,
                      resource_cache: &ResourceCache,
                      clip_buffers: &mut ClipBuffers) {
        // TODO: If any border segment is alpha, place all in alpha pass.
        //       Is it ever worth batching at a per-segment level?
        let radius = &info.radius;
        let left = &info.left;
        let right = &info.right;
        let top = &info.top;
        let bottom = &info.bottom;

        let tl_outer = Point2D::new(rect.origin.x, rect.origin.y);
        let tl_inner = tl_outer + Point2D::new(radius.top_left.width.max(left.width),
                                               radius.top_left.height.max(top.width));

        let tr_outer = Point2D::new(rect.origin.x + rect.size.width, rect.origin.y);
        let tr_inner = tr_outer + Point2D::new(-radius.top_right.width.max(right.width),
                                               radius.top_right.height.max(top.width));

        let bl_outer = Point2D::new(rect.origin.x, rect.origin.y + rect.size.height);
        let bl_inner = bl_outer + Point2D::new(radius.bottom_left.width.max(left.width),
                                               -radius.bottom_left.height.max(bottom.width));

        let br_outer = Point2D::new(rect.origin.x + rect.size.width,
                                    rect.origin.y + rect.size.height);
        let br_inner = br_outer - Point2D::new(radius.bottom_right.width.max(right.width),
                                               radius.bottom_right.height.max(bottom.width));

        let left_color = left.border_color(1.0, 2.0/3.0, 0.3, 0.7);
        let top_color = top.border_color(1.0, 2.0/3.0, 0.3, 0.7);
        let right_color = right.border_color(2.0/3.0, 1.0, 0.7, 0.3);
        let bottom_color = bottom.border_color(2.0/3.0, 1.0, 0.7, 0.3);

        // Edges
        self.add_border_edge(matrix_index,
                             &Rect::new(Point2D::new(tl_outer.x, tl_inner.y),
                                        Size2D::new(left.width, bl_inner.y - tl_inner.y)),
                             clip,
                             BorderEdgeDirection::Vertical,
                             &left_color,
                             info.left.style,
                             resource_cache,
                             clip_buffers);

        self.add_border_edge(matrix_index,
                             &Rect::new(Point2D::new(tl_inner.x, tl_outer.y),
                                        Size2D::new(tr_inner.x - tl_inner.x,
                                                    tr_outer.y + top.width - tl_outer.y)),
                             clip,
                             BorderEdgeDirection::Horizontal,
                             &top_color,
                             info.top.style,
                             resource_cache,
                             clip_buffers);

        self.add_border_edge(matrix_index,
                             &Rect::new(Point2D::new(br_outer.x - right.width, tr_inner.y),
                                        Size2D::new(right.width, br_inner.y - tr_inner.y)),
                             clip,
                             BorderEdgeDirection::Vertical,
                             &right_color,
                             info.right.style,
                             resource_cache,
                             clip_buffers);

        self.add_border_edge(matrix_index,
                             &Rect::new(Point2D::new(bl_inner.x, bl_outer.y - bottom.width),
                                        Size2D::new(br_inner.x - bl_inner.x,
                                                    br_outer.y - bl_outer.y + bottom.width)),
                             clip,
                             BorderEdgeDirection::Horizontal,
                             &bottom_color,
                             info.bottom.style,
                             resource_cache,
                             clip_buffers);

        // Corners
        self.add_border_corner(matrix_index,
                               clip,
                               &Rect::new(tl_outer,
                                          Size2D::new(tl_inner.x - tl_outer.x,
                                                      tl_inner.y - tl_outer.y)),
                               &left_color,
                               &top_color,
                               &radius.top_left,
                               &info.top_left_inner_radius(),
                               resource_cache,
                               clip_buffers,
                               BasicRotationAngle::Upright);

        self.add_border_corner(matrix_index,
                               clip,
                               &Rect::new(Point2D::new(tr_inner.x, tr_outer.y),
                                          Size2D::new(tr_outer.x - tr_inner.x,
                                                      tr_inner.y - tr_outer.y)),
                               &right_color,
                               &top_color,
                               &radius.top_right,
                               &info.top_right_inner_radius(),
                               resource_cache,
                               clip_buffers,
                               BasicRotationAngle::Clockwise90);

        self.add_border_corner(matrix_index,
                               clip,
                               &Rect::new(br_inner,
                                          Size2D::new(br_outer.x - br_inner.x,
                                                      br_outer.y - br_inner.y)),
                               &right_color,
                               &bottom_color,
                               &radius.bottom_right,
                               &info.bottom_right_inner_radius(),
                               resource_cache,
                               clip_buffers,
                               BasicRotationAngle::Clockwise180);

        self.add_border_corner(matrix_index,
                               clip,
                               &Rect::new(Point2D::new(bl_outer.x, bl_inner.y),
                                          Size2D::new(bl_inner.x - bl_outer.x,
                                                      bl_outer.y - bl_inner.y)),
                               &left_color,
                               &bottom_color,
                               &radius.bottom_left,
                               &info.bottom_left_inner_radius(),
                               resource_cache,
                               clip_buffers,
                               BasicRotationAngle::Clockwise270);
    }

    // FIXME(pcwalton): Assumes rectangles are well-formed with origin in TL
    fn add_box_shadow_corner(&mut self,
                             matrix_index: MatrixIndex,
                             top_left: &Point2D<f32>,
                             bottom_right: &Point2D<f32>,
                             color: &ColorF,
                             blur_radius: f32,
                             border_radius: f32,
                             clip_mode: BoxShadowClipMode,
                             clip: &CombinedClipRegion,
                             resource_cache: &ResourceCache,
                             clip_buffers: &mut ClipBuffers,
                             rotation_angle: BasicRotationAngle) {
        let inverted = match clip_mode {
            BoxShadowClipMode::Outset | BoxShadowClipMode::None => false,
            BoxShadowClipMode::Inset => true,
        };

        let color_image = match BoxShadowRasterOp::create_corner(blur_radius,
                                                                 border_radius,
                                                                 inverted) {
            Some(raster_item) => {
                let raster_item = RasterItem::BoxShadow(raster_item);
                resource_cache.get_raster(&raster_item)
            }
            None => resource_cache.get_dummy_color_image(),
        };

        self.add_color_image_rectangle(matrix_index,
                                       top_left,
                                       bottom_right,
                                       clip,
                                       color,
                                       color,
                                       &color_image,
                                       resource_cache,
                                       clip_buffers,
                                       rotation_angle)
    }

    fn add_box_shadow_edge(&mut self,
                           matrix_index: MatrixIndex,
                           top_left: &Point2D<f32>,
                           bottom_right: &Point2D<f32>,
                           color: &ColorF,
                           blur_radius: f32,
                           border_radius: f32,
                           clip_mode: BoxShadowClipMode,
                           clip: &CombinedClipRegion,
                           resource_cache: &ResourceCache,
                           clip_buffers: &mut ClipBuffers,
                           rotation_angle: BasicRotationAngle) {
        let inverted = match clip_mode {
            BoxShadowClipMode::Outset | BoxShadowClipMode::None => false,
            BoxShadowClipMode::Inset => true,
        };

        let color_image = match BoxShadowRasterOp::create_edge(blur_radius,
                                                               border_radius,
                                                               inverted) {
            Some(raster_item) => {
                let raster_item = RasterItem::BoxShadow(raster_item);
                resource_cache.get_raster(&raster_item)
            }
            None => resource_cache.get_dummy_color_image(),
        };

        self.add_color_image_rectangle(matrix_index,
                                       top_left,
                                       bottom_right,
                                       clip,
                                       color,
                                       color,
                                       &color_image,
                                       resource_cache,
                                       clip_buffers,
                                       rotation_angle)
    }
}

trait BorderSideHelpers {
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
                    ColorF::new(black_color_1, black_color_1, black_color_1, self.color.a)
                }
            }
            BorderStyle::Outset => {
                if self.color.r != 0.0 || self.color.g != 0.0 || self.color.b != 0.0 {
                    self.color.scale_rgb(scale_factor_0)
                } else {
                    ColorF::new(black_color_0, black_color_0, black_color_0, self.color.a)
                }
            }
            _ => self.color,
        }
    }
}

/// NB: Only returns non-tessellated border radius images!
fn mask_for_border_radius<'a>(resource_cache: &'a ResourceCache,
                              border_radius: f32,
                              inverted: bool)
                              -> &'a TextureCacheItem {
    if border_radius == 0.0 {
        return resource_cache.get_dummy_mask_image()
    }

    let border_radius = Au::from_f32_px(border_radius);
    resource_cache.get_raster(&RasterItem::BorderRadius(BorderRadiusRasterOp {
        outer_radius_x: border_radius,
        outer_radius_y: border_radius,
        inner_radius_x: Au(0),
        inner_radius_y: Au(0),
        inverted: inverted,
        index: 0,
        image_format: ImageFormat::A8,
    }))
}

fn mask_for_clip_region<'a,P>(resource_cache: &'a ResourceCache,
                              clip_region: &ClipRectToRegionResult<P>,
                              inverted: bool)
                              -> &'a TextureCacheItem {
    match clip_region.mask_result {
        None => {
            resource_cache.get_dummy_mask_image()
        }
        Some(ref mask_result) => {
            mask_for_border_radius(resource_cache,
                                   mask_result.border_radius,
                                   inverted)
        }
    }
}

#[derive(Debug)]
struct BoxShadowMetrics {
    edge_size: f32,
    tl_outer: Point2D<f32>,
    tl_inner: Point2D<f32>,
    tr_outer: Point2D<f32>,
    tr_inner: Point2D<f32>,
    bl_outer: Point2D<f32>,
    bl_inner: Point2D<f32>,
    br_outer: Point2D<f32>,
    br_inner: Point2D<f32>,
}

impl BoxShadowMetrics {
    fn new(box_bounds: &Rect<f32>, border_radius: f32, blur_radius: f32) -> BoxShadowMetrics {
        let outside_edge_size = 3.0 * blur_radius;
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

fn compute_box_shadow_rect(box_bounds: &Rect<f32>,
                           box_offset: &Point2D<f32>,
                           spread_radius: f32)
                           -> Rect<f32> {
    let mut rect = (*box_bounds).clone();
    rect.origin.x += box_offset.x;
    rect.origin.y += box_offset.y;
    rect.inflate(spread_radius, spread_radius)
}
