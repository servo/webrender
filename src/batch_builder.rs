/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use batch::{BatchBuilder, TileParams};
use device::TextureId;
use euclid::{Rect, Point2D, Size2D};
use fnv::FnvHasher;
use frame::FrameId;
use internal_types::{AxisDirection, BasicRotationAngle, BorderRadiusRasterOp, BoxShadowRasterOp};
use internal_types::{GlyphKey, PackedVertexColorMode, RasterItem, RectColors, RectPolygon};
use internal_types::{RectSide, RectUv, DevicePixel};
use num_traits::Zero;
use renderer::BLUR_INFLATION_FACTOR;
use resource_cache::ResourceCache;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::f32;
use std::hash::BuildHasherDefault;
use tessellator::{self, BorderCornerTessellation};
use texture_cache::{TextureCacheItem};
use util;
use util::RectVaryings;
use webrender_traits::{AuxiliaryLists, ColorF, ImageFormat, BorderStyle, BoxShadowClipMode};
use webrender_traits::{BorderSide, FontKey, GlyphInstance, ImageKey};
use webrender_traits::{BorderDisplayItem, GradientStop, ImageRendering};
use webrender_traits::{WebGLContextId, ItemRange};

const BORDER_DASH_SIZE: f32 = 3.0;

enum ClipState {
    None,
    ClipIn,
    ClipOut(Option<Rect<f32>>)
}

#[derive(Debug)]
struct GradientRect {
    rect: Rect<f32>,
    rect_uv: RectUv<f32>,
    color0: ColorF,
    color1: ColorF,
}

impl<'a> BatchBuilder<'a> {

    // Colors are in the order: top left, top right, bottom right, bottom left.
    pub fn add_simple_rectangle(&mut self,
                                color_texture_id: TextureId,
                                pos_rect: &Rect<f32>,
                                uv_rect: &RectUv<f32>,
                                mask_texture_id: TextureId,
                                muv_rect: &RectUv<DevicePixel>,
                                colors: &[ColorF; 4],
                                tile_params: Option<TileParams>) {
        if pos_rect.size.width == 0.0 || pos_rect.size.height == 0.0 {
            return
        }

        self.add_rectangle(color_texture_id,
                           mask_texture_id,
                           pos_rect,
                           uv_rect,
                           muv_rect,
                           colors,
                           PackedVertexColorMode::Gradient,
                           tile_params);
    }

    #[inline]
    fn snap_value_to_device_pixel(&self, value: &mut f32) {
        *value = (*value * self.device_pixel_ratio).round() / self.device_pixel_ratio;
    }

    // Colors are in the order: top left, top right, bottom right, bottom left.
    pub fn add_complex_clipped_rectangle(&mut self,
                                         color_texture_id: TextureId,
                                         pos_rect: &Rect<f32>,
                                         uv_rect: &RectUv<f32>,
                                         colors: &[ColorF; 4],
                                         tile_params: Option<TileParams>,
                                         resource_cache: &ResourceCache,
                                         frame_id: FrameId) {
        if pos_rect.size.width == 0.0 || pos_rect.size.height == 0.0 {
            return
        }

        match self.complex_clip {
            Some(complex_clip) => {
                // Get clip rect in local space of the display rect
                let complex_clip_origin = complex_clip.rect.origin - pos_rect.origin;
                let mut local_pos_rect = Rect::new(Point2D::zero(), pos_rect.size);

                // TODO(gw): Perhaps this should be caught higher up, or perhaps we
                //           can even remove this requirement.
                self.snap_value_to_device_pixel(&mut local_pos_rect.size.width);
                self.snap_value_to_device_pixel(&mut local_pos_rect.size.height);

                let mut tl_xr = complex_clip.radii.top_left.width;
                let mut tl_yr = complex_clip.radii.top_left.height;
                let mut tr_xr = complex_clip.radii.top_right.width;
                let mut tr_yr = complex_clip.radii.top_right.height;

                let mut bl_xr = complex_clip.radii.bottom_left.width;
                let mut bl_yr = complex_clip.radii.bottom_left.height;
                let mut br_xr = complex_clip.radii.bottom_right.width;
                let mut br_yr = complex_clip.radii.bottom_right.height;

                self.snap_value_to_device_pixel(&mut tl_xr);
                self.snap_value_to_device_pixel(&mut tl_yr);
                self.snap_value_to_device_pixel(&mut tr_xr);
                self.snap_value_to_device_pixel(&mut tr_yr);
                self.snap_value_to_device_pixel(&mut bl_xr);
                self.snap_value_to_device_pixel(&mut bl_yr);
                self.snap_value_to_device_pixel(&mut br_xr);
                self.snap_value_to_device_pixel(&mut br_yr);

                // Generate all vertices for each line
                let mut x_points = [
                    complex_clip_origin.x + 0.0,
                    complex_clip_origin.x + complex_clip.radii.top_left.width,
                    complex_clip_origin.x + complex_clip.rect.size.width - complex_clip.radii.top_right.width,
                    complex_clip_origin.x + complex_clip.radii.bottom_left.width,
                    complex_clip_origin.x + complex_clip.rect.size.width - complex_clip.radii.bottom_right.width,
                    complex_clip_origin.x + complex_clip.rect.size.width,
                ];

                // Generate all vertices for each line
                let mut y_points = [
                    complex_clip_origin.y + 0.0,
                    complex_clip_origin.y + complex_clip.radii.top_left.height,
                    complex_clip_origin.y + complex_clip.radii.top_right.height,
                    complex_clip_origin.y + complex_clip.rect.size.height - complex_clip.radii.bottom_left.height,
                    complex_clip_origin.y + complex_clip.rect.size.height - complex_clip.radii.bottom_right.height,
                    complex_clip_origin.y + complex_clip.rect.size.height,
                ];

                for x_point in &mut x_points {
                    self.snap_value_to_device_pixel(x_point);
                }

                for y_point in &mut y_points {
                    self.snap_value_to_device_pixel(y_point);
                }

                let tl_clip = Rect::new(Point2D::new(x_points[0], y_points[0]),
                                        Size2D::new(tl_xr, tl_yr));

                let tr_clip = Rect::new(Point2D::new(x_points[2], y_points[0]),
                                        Size2D::new(tr_xr, tr_yr));

                let bl_clip = Rect::new(Point2D::new(x_points[0], y_points[3]),
                                        Size2D::new(bl_xr, bl_yr));

                let br_clip = Rect::new(Point2D::new(x_points[4], y_points[4]),
                                        Size2D::new(br_xr, br_yr));

                x_points.sort_by(|a, b| {
                    a.partial_cmp(b).unwrap()
                });
                y_points.sort_by(|a, b| {
                    a.partial_cmp(b).unwrap()
                });

                for xi in 0..x_points.len()-1 {
                    for yi in 0..y_points.len()-1 {
                        let x0 = x_points[xi+0];
                        let y0 = y_points[yi+0];
                        let x1 = x_points[xi+1];
                        let y1 = y_points[yi+1];

                        if x0 != x1 && y0 != y1 {
                            let sub_clip_rect = Rect::new(Point2D::new(x0, y0),
                                                          Size2D::new(x1-x0, y1-y0));

                            if let Some(clipped_pos_rect) = sub_clip_rect.intersection(&local_pos_rect) {
                                debug_assert!((clipped_pos_rect.origin.x * self.device_pixel_ratio).fract() == 0.0);
                                debug_assert!((clipped_pos_rect.origin.y * self.device_pixel_ratio).fract() == 0.0);
                                debug_assert!((clipped_pos_rect.size.width * self.device_pixel_ratio).fract() == 0.0);
                                debug_assert!((clipped_pos_rect.size.height * self.device_pixel_ratio).fract() == 0.0);

                                let u0_f = clipped_pos_rect.origin.x  / local_pos_rect.size.width;
                                let v0_f = clipped_pos_rect.origin.y  / local_pos_rect.size.height;
                                let u1_f = (clipped_pos_rect.origin.x + clipped_pos_rect.size.width) / local_pos_rect.size.width;
                                let v1_f = (clipped_pos_rect.origin.y + clipped_pos_rect.size.height) / local_pos_rect.size.height;

                                let u_size = uv_rect.top_right.x - uv_rect.top_left.x;
                                let v_size = uv_rect.bottom_right.y - uv_rect.top_left.y;

                                let u0 = uv_rect.top_left.x + u0_f * u_size;
                                let v0 = uv_rect.top_left.y + v0_f * v_size;
                                let u1 = uv_rect.top_left.x + u1_f * u_size;
                                let v1 = uv_rect.top_left.y + v1_f * v_size;

                                let uv_rect = RectUv {
                                    top_left: Point2D::new(u0, v0),
                                    top_right: Point2D::new(u1, v0),
                                    bottom_left: Point2D::new(u0, v1),
                                    bottom_right: Point2D::new(u1, v1),
                                };

                                // TODO(gw): There must be a more efficient way to to
                                //           this (classifying which clip mask we need).
                                let (mask_info, angle) = if sub_clip_rect.intersects(&tl_clip) {
                                    (Some(&tl_clip), BasicRotationAngle::Upright)
                                } else if sub_clip_rect.intersects(&tr_clip) {
                                    (Some(&tr_clip), BasicRotationAngle::Clockwise90)
                                } else if sub_clip_rect.intersects(&bl_clip) {
                                    (Some(&bl_clip), BasicRotationAngle::Clockwise270)
                                } else if sub_clip_rect.intersects(&br_clip) {
                                    (Some(&br_clip), BasicRotationAngle::Clockwise180)
                                } else {
                                    (None, BasicRotationAngle::Upright)
                                };

                                let (mask_texture_id, muv_rect) = match mask_info {
                                    Some(clip_rect) => {
                                        let op = RasterItem::BorderRadius(BorderRadiusRasterOp {
                                            outer_radius_x:
                                                DevicePixel::new(clip_rect.size.width,
                                                                 self.device_pixel_ratio),
                                            outer_radius_y:
                                                DevicePixel::new(clip_rect.size.height,
                                                                 self.device_pixel_ratio),
                                            inner_radius_x: DevicePixel::zero(),
                                            inner_radius_y: DevicePixel::zero(),
                                            inverted: false,
                                            index: None,
                                            image_format: ImageFormat::A8,
                                        });

                                        let mask_image = resource_cache.get_raster(&op, frame_id);

                                        let mut x0_f = (x0 - clip_rect.origin.x) / clip_rect.size.width;
                                        let mut x1_f = (x1 - clip_rect.origin.x) / clip_rect.size.width;
                                        let mut y0_f = (y0 - clip_rect.origin.y) / clip_rect.size.height;
                                        let mut y1_f = (y1 - clip_rect.origin.y) / clip_rect.size.height;

                                        match angle {
                                            BasicRotationAngle::Upright => {}
                                            BasicRotationAngle::Clockwise90 => {
                                                x0_f = 1.0 - x0_f;
                                                x1_f = 1.0 - x1_f;
                                            }
                                            BasicRotationAngle::Clockwise180 => {
                                                x0_f = 1.0 - x0_f;
                                                x1_f = 1.0 - x1_f;
                                                y0_f = 1.0 - y0_f;
                                                y1_f = 1.0 - y1_f;
                                            }
                                            BasicRotationAngle::Clockwise270 => {
                                                y0_f = 1.0 - y0_f;
                                                y1_f = 1.0 - y1_f;
                                            }
                                        }

                                        let mu0 = mask_image.pixel_rect.top_left.x.as_f32();
                                        let mu1 = mask_image.pixel_rect.top_right.x.as_f32();
                                        let mv0 = mask_image.pixel_rect.top_left.y.as_f32();
                                        let mv1 = mask_image.pixel_rect.bottom_left.y.as_f32();

                                        let mu_size = mu1 - mu0;
                                        let mv_size = mv1 - mv0;
                                        let mu1 = mu0 + x1_f * mu_size;
                                        let mu0 = mu0 + x0_f * mu_size;
                                        let mv1 = mv0 + y1_f * mv_size;
                                        let mv0 = mv0 + y0_f * mv_size;

                                        let mu0 = DevicePixel::from_f32(mu0);
                                        let mv0 = DevicePixel::from_f32(mv0);
                                        let mu1 = DevicePixel::from_f32(mu1);
                                        let mv1 = DevicePixel::from_f32(mv1);

                                        let muv_rect = RectUv {
                                            top_left: Point2D::new(mu0, mv0),
                                            top_right: Point2D::new(mu1, mv0),
                                            bottom_left: Point2D::new(mu0, mv1),
                                            bottom_right: Point2D::new(mu1, mv1),
                                        };

                                        (mask_image.texture_id, muv_rect)
                                    }
                                    None => {
                                        let mask_image = resource_cache.get_dummy_mask_image();
                                        (mask_image.texture_id, mask_image.pixel_rect)
                                    }
                                };

                                // TODO(gw): Needless conversions here - just to make it
                                // easier to operate with existing bilerp code - clean this up!
                                let world_clipped_pos_rect = clipped_pos_rect.translate(&pos_rect.origin);

                                let rect_colors = RectColors::from_elements(colors);
                                let rect_colors = util::bilerp_rect(&world_clipped_pos_rect,
                                                                    &pos_rect,
                                                                    &rect_colors);

                                self.add_simple_rectangle(color_texture_id,
                                                          &world_clipped_pos_rect,
                                                          &uv_rect,
                                                          mask_texture_id,
                                                          &muv_rect,
                                                          &[rect_colors.top_left,
                                                            rect_colors.top_right,
                                                            rect_colors.bottom_right,
                                                            rect_colors.bottom_left,
                                                           ],
                                                          tile_params.clone());
                            }
                        }
                    }
                }
            }
            None => {
                let dummy_mask_image = resource_cache.get_dummy_mask_image();

                self.add_simple_rectangle(color_texture_id,
                                          pos_rect,
                                          uv_rect,
                                          dummy_mask_image.texture_id,
                                          &dummy_mask_image.pixel_rect,
                                          colors,
                                          tile_params);
            }
        }
    }

    #[inline]
    pub fn add_color_rectangle(&mut self,
                               rect: &Rect<f32>,
                               color: &ColorF,
                               resource_cache: &ResourceCache,
                               frame_id: FrameId) {
        let white_image = resource_cache.get_dummy_color_image();
        self.add_complex_clipped_rectangle(white_image.texture_id,
                                           rect,
                                           &white_image.uv_rect(),
                                           &[*color, *color, *color, *color],
                                           None,
                                           resource_cache,
                                           frame_id);
    }

    pub fn add_webgl_rectangle(&mut self,
                               rect: &Rect<f32>,
                               resource_cache: &ResourceCache,
                               webgl_context_id: &WebGLContextId,
                               frame_id: FrameId) {
        let texture_id = resource_cache.get_webgl_texture(webgl_context_id);
        let color = ColorF::new(1.0, 1.0, 1.0, 1.0);

        let uv = RectUv {
            top_left: Point2D::new(0.0, 1.0),
            top_right: Point2D::new(1.0, 1.0),
            bottom_left: Point2D::zero(),
            bottom_right: Point2D::new(1.0, 0.0),
        };

        self.add_complex_clipped_rectangle(texture_id,
                                           rect,
                                           &uv,
                                           &[color, color, color, color],
                                           None,
                                           resource_cache,
                                           frame_id);
    }

    pub fn add_image(&mut self,
                     rect: &Rect<f32>,
                     stretch_size: &Size2D<f32>,
                     image_key: ImageKey,
                     image_rendering: ImageRendering,
                     resource_cache: &ResourceCache,
                     frame_id: FrameId) {
        // Should be caught higher up
        debug_assert!(stretch_size.width > 0.0 && stretch_size.height > 0.0);
        let image_info = resource_cache.get_image(image_key, image_rendering, frame_id);

        let u1 = rect.size.width / stretch_size.width;
        let v1 = rect.size.height / stretch_size.height;

        let uv = RectUv {
            top_left: Point2D::zero(),
            top_right: Point2D::new(u1, 0.0),
            bottom_left: Point2D::new(0.0, v1),
            bottom_right: Point2D::new(u1, v1),
        };

        let uv_rect = image_info.uv_rect();
        let uv_size = uv_rect.bottom_right - uv_rect.top_left;

        let tile_params = TileParams {
            u0: image_info.uv_rect().top_left.x,
            v0: image_info.uv_rect().top_left.y,
            u_size: uv_size.x,
            v_size: uv_size.y,
        };

        let color = ColorF::new(1.0, 1.0, 1.0, 1.0);

        self.add_complex_clipped_rectangle(image_info.texture_id,
                                           rect,
                                           &uv,
                                           &[color, color, color, color],
                                           Some(tile_params),
                                           resource_cache,
                                           frame_id);
    }

    pub fn add_text(&mut self,
                    _rect: &Rect<f32>,
                    font_key: FontKey,
                    size: Au,
                    blur_radius: Au,
                    color: &ColorF,
                    glyphs: &[GlyphInstance],
                    resource_cache: &ResourceCache,
                    frame_id: FrameId,
                    device_pixel_ratio: f32) {
        let dummy_mask_image = resource_cache.get_dummy_mask_image();

        // Logic below to pick the primary render item depends on len > 0!
        assert!(glyphs.len() > 0);

        let mut glyph_key = GlyphKey::new(font_key, size, blur_radius, glyphs[0].index);
        let blur_offset = blur_radius.to_f32_px() * (BLUR_INFLATION_FACTOR as f32) / 2.0;

        let mut text_batches: HashMap<TextureId,
                                      Vec<RectPolygon<RectUv<f32>>>,
                                      BuildHasherDefault<FnvHasher>> =
            HashMap::with_hasher(Default::default());

        for glyph in glyphs {
            glyph_key.index = glyph.index;
            let image_info = resource_cache.get_glyph(&glyph_key, frame_id);
            if let Some(image_info) = image_info {
                let x = glyph.x + image_info.user_data.x0 as f32 / device_pixel_ratio - blur_offset;
                let y = glyph.y - image_info.user_data.y0 as f32 / device_pixel_ratio - blur_offset;

                let width = image_info.requested_rect.size.width as f32 / device_pixel_ratio;
                let height = image_info.requested_rect.size.height as f32 / device_pixel_ratio;

                let rect = RectPolygon {
                    pos: Rect::new(Point2D::new(x, y),
                                   Size2D::new(width, height)),
                    varyings: image_info.uv_rect(),
                };

                let rect_buffer = match text_batches.entry(image_info.texture_id) {
                    Occupied(entry) => entry.into_mut(),
                    Vacant(entry) => entry.insert(Vec::new()),
                };

                rect_buffer.push(rect);
            }
        }

        for (texture_id, rect_buffer) in text_batches {
            for rect in rect_buffer {
                self.add_rectangle(texture_id,
                                   dummy_mask_image.texture_id,
                                   &rect.pos,
                                   &rect.varyings,
                                   &dummy_mask_image.pixel_rect,
                                   &[*color, *color, *color, *color],
                                   PackedVertexColorMode::Gradient,
                                   None);
            }

        }
    }

    fn add_axis_aligned_gradient_with_stops(&mut self,
                                            rect: &Rect<f32>,
                                            direction: AxisDirection,
                                            stops: &[GradientStop],
                                            resource_cache: &ResourceCache,
                                            frame_id: FrameId) {
        let white_image = resource_cache.get_dummy_color_image();

        for i in 0..(stops.len() - 1) {
            let (prev_stop, next_stop) = (&stops[i], &stops[i + 1]);
            let piece_rect;
            let piece_colors;
            match direction {
                AxisDirection::Horizontal => {
                    let prev_x = util::lerp(rect.origin.x, rect.max_x(), prev_stop.offset);
                    let next_x = util::lerp(rect.origin.x, rect.max_x(), next_stop.offset);
                    piece_rect = Rect::new(Point2D::new(prev_x, rect.origin.y),
                                           Size2D::new(next_x - prev_x, rect.size.height));
                    piece_colors = [
                        prev_stop.color,
                        next_stop.color,
                        next_stop.color,
                        prev_stop.color
                    ];
                }
                AxisDirection::Vertical => {
                    let prev_y = util::lerp(rect.origin.y, rect.max_y(), prev_stop.offset);
                    let next_y = util::lerp(rect.origin.y, rect.max_y(), next_stop.offset);
                    piece_rect = Rect::new(Point2D::new(rect.origin.x, prev_y),
                                           Size2D::new(rect.size.width, next_y - prev_y));
                    piece_colors = [
                        prev_stop.color,
                        prev_stop.color,
                        next_stop.color,
                        next_stop.color
                    ];
                }
            }

            self.add_complex_clipped_rectangle(white_image.texture_id,
                                               &piece_rect,
                                               &white_image.uv_rect(),
                                               &piece_colors,
                                               None,
                                               resource_cache,
                                               frame_id);
        }
    }

    pub fn add_gradient(&mut self,
                        rect: &Rect<f32>,
                        start_point: &Point2D<f32>,
                        end_point: &Point2D<f32>,
                        stops: &ItemRange,
                        auxiliary_lists: &AuxiliaryLists,
                        resource_cache: &ResourceCache,
                        frame_id: FrameId) {
        let stops = auxiliary_lists.gradient_stops(stops);

        // Fast paths for axis-aligned gradients:
        if start_point.x == end_point.x {
            let rect = Rect::new(Point2D::new(rect.origin.x, start_point.y),
                                 Size2D::new(rect.size.width, end_point.y - start_point.y));
            self.add_axis_aligned_gradient_with_stops(&rect,
                                                      AxisDirection::Vertical,
                                                      stops,
                                                      resource_cache,
                                                      frame_id);
            return
        }
        if start_point.y == end_point.y {
            let rect = Rect::new(Point2D::new(start_point.x, rect.origin.y),
                                 Size2D::new(end_point.x - start_point.x, rect.size.height));
            self.add_axis_aligned_gradient_with_stops(&rect,
                                                      AxisDirection::Horizontal,
                                                      stops,
                                                      resource_cache,
                                                      frame_id);
            return
        }

        let white_image = resource_cache.get_dummy_color_image();
        let dummy_mask_image = resource_cache.get_dummy_mask_image();

        debug_assert!(stops.len() >= 2);

        let mut angle = ((end_point.y - start_point.y) / (end_point.x - start_point.x)).atan() +
            f32::consts::FRAC_PI_2;
        if angle < 0.0 {
            angle += 2.0 * f32::consts::PI
        }

        // A simple way to estimate the length of each strip we'll need. Providing a good estimate
        // saves fragment shader invocations.
        let length_0 = (rect.size.width * angle.sin()).abs() +
            (rect.size.height * angle.cos()).abs();
        let length_1 = (rect.size.width * angle.cos()).abs() +
            (rect.size.height * angle.sin()).abs();
        let length = if length_0 > length_1 {
            length_0
        } else {
            length_1
        };

        let mut rectangles = Vec::new();
        let mut prev_color = stops[0].color;
        let mut prev_offset = 0.0;
        for next in &stops[..] {
            let prev_point = util::lerp_points(start_point, end_point, prev_offset);
            let next_point = util::lerp_points(start_point, end_point, next.offset);
            let midpoint = util::lerp_points(&prev_point, &next_point, 0.5);

            let height = util::distance(&prev_point, &next_point);
            let rect =
                Rect::new(Point2D::new(-length / 2.0 + midpoint.x, midpoint.y - height / 2.0),
                          Size2D::new(length, height));
            let mut rect_uv = white_image.uv_rect();
            rect_uv.bottom_left.x = -angle;

            rectangles.push(GradientRect {
                rect: rect,
                rect_uv: rect_uv,
                color0: next.color,
                color1: prev_color
            });

            prev_color = next.color;
            prev_offset = next.offset
        }

        // This is a bit of a hack to pass reftests. In some cases of angled gradients,
        // there is a one pixel overlap along the stop where two rectangles are drawn.
        // To ensure that these are drawn the same in reftests, ensure the rectangles
        // are always submitted in the same order no matter what angle they were
        // generated from.
        // TODO(gw): This probably deserves a bit more investigation to see if there
        //           is a better solution!
        rectangles.sort_by(|a, b| {
            match a.rect.origin.x.partial_cmp(&b.rect.origin.x).unwrap() {
                Ordering::Equal => a.rect.origin.y.partial_cmp(&b.rect.origin.y).unwrap(),
                ordering => ordering,
            }
        });

        for gradient_rect in rectangles {
            self.add_rectangle(white_image.texture_id,
                               dummy_mask_image.texture_id,
                               &gradient_rect.rect,
                               &gradient_rect.rect_uv,
                               &dummy_mask_image.pixel_rect,
                               &[gradient_rect.color0,
                                 gradient_rect.color0,
                                 gradient_rect.color1,
                                 gradient_rect.color1
                                ],
                               PackedVertexColorMode::Gradient,
                               None);
        }
    }

    pub fn add_box_shadow(&mut self,
                          box_bounds: &Rect<f32>,
                          box_offset: &Point2D<f32>,
                          color: &ColorF,
                          blur_radius: f32,
                          spread_radius: f32,
                          border_radius: f32,
                          clip_mode: BoxShadowClipMode,
                          resource_cache: &ResourceCache,
                          frame_id: FrameId) {
        let rect = compute_box_shadow_rect(box_bounds, box_offset, spread_radius, clip_mode);

        // Fast path.
        if blur_radius == 0.0 && spread_radius == 0.0 && clip_mode == BoxShadowClipMode::None {
            self.add_color_rectangle(&rect, color, resource_cache, frame_id);
            return;
        }

        // Draw the corners.
        self.add_box_shadow_corners(box_bounds,
                                    box_offset,
                                    color,
                                    blur_radius,
                                    spread_radius,
                                    border_radius,
                                    clip_mode,
                                    resource_cache,
                                    frame_id);

        // Draw the sides.
        self.add_box_shadow_sides(box_bounds,
                                  box_offset,
                                  color,
                                  blur_radius,
                                  spread_radius,
                                  border_radius,
                                  clip_mode,
                                  resource_cache,
                                  frame_id);

        match clip_mode {
            BoxShadowClipMode::None => {
                // Fill the center area.
                self.add_color_rectangle(box_bounds, color, resource_cache, frame_id);
            }
            BoxShadowClipMode::Outset => {
                // Fill the center area.
                let metrics = BoxShadowMetrics::new(&rect, border_radius, blur_radius);
                if metrics.br_inner.x > metrics.tl_inner.x &&
                        metrics.br_inner.y > metrics.tl_inner.y {
                    let center_rect =
                        Rect::new(metrics.tl_inner,
                                  Size2D::new(metrics.br_inner.x - metrics.tl_inner.x,
                                              metrics.br_inner.y - metrics.tl_inner.y));

                    // FIXME(pcwalton): This assumes the border radius is zero. That is not always
                    // the case!
                    let old_clip_out_rect = self.set_clip_out_rect(Some(*box_bounds));

                    self.add_color_rectangle(&center_rect, color, resource_cache, frame_id);

                    self.set_clip_out_rect(old_clip_out_rect);
                }
            }
            BoxShadowClipMode::Inset => {
                // Fill in the outsides.
                self.fill_outside_area_of_inset_box_shadow(box_bounds,
                                                           box_offset,
                                                           color,
                                                           blur_radius,
                                                           spread_radius,
                                                           border_radius,
                                                           resource_cache,
                                                           frame_id);
            }
        }
    }

    fn add_box_shadow_corners(&mut self,
                              box_bounds: &Rect<f32>,
                              box_offset: &Point2D<f32>,
                              color: &ColorF,
                              blur_radius: f32,
                              spread_radius: f32,
                              border_radius: f32,
                              clip_mode: BoxShadowClipMode,
                              resource_cache: &ResourceCache,
                              frame_id: FrameId) {
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

        let rect = compute_box_shadow_rect(box_bounds, box_offset, spread_radius, clip_mode);
        let metrics = BoxShadowMetrics::new(&rect, border_radius, blur_radius);

        let clip_state = self.adjust_clip_for_box_shadow_clip_mode(box_bounds,
                                                                   border_radius,
                                                                   clip_mode);

        // Prevent overlap of the box shadow corners when the size of the blur is larger than the
        // size of the box.
        let center = Point2D::new(rect.origin.x + rect.size.width / 2.0,
                                  rect.origin.y + rect.size.height / 2.0);

        self.add_box_shadow_corner(&metrics.tl_outer,
                                   &Point2D::new(metrics.tl_outer.x + metrics.edge_size,
                                                 metrics.tl_outer.y + metrics.edge_size),
                                   &metrics.tl_outer,
                                   &center,
                                   &rect,
                                   &color,
                                   blur_radius,
                                   border_radius,
                                   clip_mode,
                                   resource_cache,
                                   frame_id,
                                   BasicRotationAngle::Upright);
        self.add_box_shadow_corner(&Point2D::new(metrics.tr_outer.x - metrics.edge_size,
                                                 metrics.tr_outer.y),
                                   &Point2D::new(metrics.tr_outer.x,
                                                 metrics.tr_outer.y + metrics.edge_size),
                                   &Point2D::new(center.x, metrics.tr_outer.y),
                                   &Point2D::new(metrics.tr_outer.x, center.y),
                                   &rect,
                                   &color,
                                   blur_radius,
                                   border_radius,
                                   clip_mode,
                                   resource_cache,
                                   frame_id,
                                   BasicRotationAngle::Clockwise90);
        self.add_box_shadow_corner(&Point2D::new(metrics.br_outer.x - metrics.edge_size,
                                                 metrics.br_outer.y - metrics.edge_size),
                                   &Point2D::new(metrics.br_outer.x, metrics.br_outer.y),
                                   &center,
                                   &metrics.br_outer,
                                   &rect,
                                   &color,
                                   blur_radius,
                                   border_radius,
                                   clip_mode,
                                   resource_cache,
                                   frame_id,
                                   BasicRotationAngle::Clockwise180);
        self.add_box_shadow_corner(&Point2D::new(metrics.bl_outer.x,
                                                 metrics.bl_outer.y - metrics.edge_size),
                                   &Point2D::new(metrics.bl_outer.x + metrics.edge_size,
                                                 metrics.bl_outer.y),
                                   &Point2D::new(metrics.bl_outer.x, center.y),
                                   &Point2D::new(center.x, metrics.bl_outer.y),
                                   &rect,
                                   &color,
                                   blur_radius,
                                   border_radius,
                                   clip_mode,
                                   resource_cache,
                                   frame_id,
                                   BasicRotationAngle::Clockwise270);

        self.undo_clip_state(clip_state);
    }

    fn add_box_shadow_sides(&mut self,
                            box_bounds: &Rect<f32>,
                            box_offset: &Point2D<f32>,
                            color: &ColorF,
                            blur_radius: f32,
                            spread_radius: f32,
                            border_radius: f32,
                            clip_mode: BoxShadowClipMode,
                            resource_cache: &ResourceCache,
                            frame_id: FrameId) {
        let rect = compute_box_shadow_rect(box_bounds, box_offset, spread_radius, clip_mode);
        let metrics = BoxShadowMetrics::new(&rect, border_radius, blur_radius);

        let clip_state = self.adjust_clip_for_box_shadow_clip_mode(box_bounds,
                                                                   border_radius,
                                                                   clip_mode);

        if clip_mode == BoxShadowClipMode::Inset && blur_radius == 0.0 {
            return;
        }

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

        // Prevent overlap of the box shadow edges when the size of the blur is larger than the
        // size of the box.
        let center = Point2D::new(rect.origin.x + rect.size.width / 2.0,
                                  rect.origin.y + rect.size.height / 2.0);

        self.add_box_shadow_edge(&top_rect.origin,
                                 &top_rect.bottom_right(),
                                 &top_rect.origin,
                                 &Point2D::new(metrics.tr_inner.x, center.y),
                                 &rect,
                                 color,
                                 blur_radius,
                                 border_radius,
                                 clip_mode,
                                 resource_cache,
                                 frame_id,
                                 BasicRotationAngle::Clockwise90);
        self.add_box_shadow_edge(&right_rect.origin,
                                 &right_rect.bottom_right(),
                                 &Point2D::new(center.x, metrics.tr_inner.y),
                                 &right_rect.bottom_right(),
                                 &rect,
                                 color,
                                 blur_radius,
                                 border_radius,
                                 clip_mode,
                                 resource_cache,
                                 frame_id,
                                 BasicRotationAngle::Clockwise180);
        self.add_box_shadow_edge(&bottom_rect.origin,
                                 &bottom_rect.bottom_right(),
                                 &Point2D::new(metrics.bl_inner.x, center.y),
                                 &bottom_rect.bottom_right(),
                                 &rect,
                                 color,
                                 blur_radius,
                                 border_radius,
                                 clip_mode,
                                 resource_cache,
                                 frame_id,
                                 BasicRotationAngle::Clockwise270);
        self.add_box_shadow_edge(&left_rect.origin,
                                 &left_rect.bottom_right(),
                                 &left_rect.origin,
                                 &Point2D::new(center.x, metrics.bl_inner.y),
                                 &rect,
                                 color,
                                 blur_radius,
                                 border_radius,
                                 clip_mode,
                                 resource_cache,
                                 frame_id,
                                 BasicRotationAngle::Upright);

        self.undo_clip_state(clip_state);
    }

    fn fill_outside_area_of_inset_box_shadow(&mut self,
                                             box_bounds: &Rect<f32>,
                                             box_offset: &Point2D<f32>,
                                             color: &ColorF,
                                             blur_radius: f32,
                                             spread_radius: f32,
                                             border_radius: f32,
                                             resource_cache: &ResourceCache,
                                             frame_id: FrameId) {
        let rect = compute_box_shadow_rect(box_bounds, box_offset, spread_radius, BoxShadowClipMode::Inset);
        let metrics = BoxShadowMetrics::new(&rect, border_radius, blur_radius);

        let clip_state = self.adjust_clip_for_box_shadow_clip_mode(box_bounds,
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
        self.add_color_rectangle(&Rect::new(box_bounds.origin,
                                            Size2D::new(box_bounds.size.width,
                                                        metrics.tl_outer.y - box_bounds.origin.y)),
                                 color,
                                 resource_cache,
                                 frame_id);

        // B:
        self.add_color_rectangle(&Rect::new(metrics.tr_outer,
                                            Size2D::new(box_bounds.max_x() - metrics.tr_outer.x,
                                                        metrics.br_outer.y - metrics.tr_outer.y)),
                                 color,
                                 resource_cache,
                                 frame_id);

        // C:
        self.add_color_rectangle(&Rect::new(Point2D::new(box_bounds.origin.x, metrics.bl_outer.y),
                                            Size2D::new(box_bounds.size.width,
                                                        box_bounds.max_y() - metrics.br_outer.y)),
                                 color,
                                 resource_cache,
                                 frame_id);

        // D:
        self.add_color_rectangle(&Rect::new(Point2D::new(box_bounds.origin.x, metrics.tl_outer.y),
                                            Size2D::new(metrics.tl_outer.x - box_bounds.origin.x,
                                                        metrics.bl_outer.y - metrics.tl_outer.y)),
                                 color,
                                 resource_cache,
                                 frame_id);

        self.undo_clip_state(clip_state);
    }

    fn undo_clip_state(&mut self, clip_state: ClipState) {
        match clip_state {
            ClipState::None => {}
            ClipState::ClipIn => {
                self.pop_clip_in_rect();
            }
            ClipState::ClipOut(old_rect) => {
                self.set_clip_out_rect(old_rect);
            }
        }
    }

    fn adjust_clip_for_box_shadow_clip_mode(&mut self,
                                            box_bounds: &Rect<f32>,
                                            _border_radius: f32,
                                            clip_mode: BoxShadowClipMode) -> ClipState {
        //debug_assert!(border_radius == 0.0);        // TODO(gw): !!!

        match clip_mode {
            BoxShadowClipMode::None => {
                ClipState::None
            }
            BoxShadowClipMode::Inset => {
                self.push_clip_in_rect(box_bounds);
                ClipState::ClipIn
            }
            BoxShadowClipMode::Outset => {
                let old_clip_out_rect = self.set_clip_out_rect(Some(*box_bounds));
                ClipState::ClipOut(old_clip_out_rect)
            }
        }
    }

    #[inline]
    fn add_border_edge(&mut self,
                       rect: &Rect<f32>,
                       side: RectSide,
                       color: &ColorF,
                       border_style: BorderStyle,
                       resource_cache: &ResourceCache,
                       frame_id: FrameId) {
        if color.a <= 0.0 {
            return
        }
        if rect.size.width <= 0.0 || rect.size.height <= 0.0 {
            return
        }

        let dummy_mask_image = resource_cache.get_dummy_mask_image();
        let colors = [*color, *color, *color, *color];

        match border_style {
            BorderStyle::Dashed => {
                let (extent, step) = match side {
                    RectSide::Top | RectSide::Bottom => {
                        (rect.size.width, rect.size.height * BORDER_DASH_SIZE)
                    }
                    RectSide::Left | RectSide::Right => {
                        (rect.size.height, rect.size.width * BORDER_DASH_SIZE)
                    }
                };
                let mut origin = 0.0;
                while origin < extent {
                    let dash_rect = match side {
                        RectSide::Top | RectSide::Bottom => {
                            Rect::new(Point2D::new(rect.origin.x + origin, rect.origin.y),
                                      Size2D::new(f32::min(step, extent - origin),
                                                  rect.size.height))
                        }
                        RectSide::Left | RectSide::Right => {
                            Rect::new(Point2D::new(rect.origin.x, rect.origin.y + origin),
                                      Size2D::new(rect.size.width,
                                                  f32::min(step, extent - origin)))
                        }
                    };

                    self.add_color_rectangle(&dash_rect, color, resource_cache, frame_id);

                    origin += step + step;
                }
            }
            BorderStyle::Dotted => {
                let (extent, step) = match side {
                    RectSide::Top | RectSide::Bottom => (rect.size.width, rect.size.height),
                    RectSide::Left | RectSide::Right => (rect.size.height, rect.size.width),
                };
                let mut origin = 0.0;
                while origin < extent {
                    let (dot_rect, mask_radius) = match side {
                        RectSide::Top | RectSide::Bottom => {
                            (Rect::new(Point2D::new(rect.origin.x + origin, rect.origin.y),
                                       Size2D::new(f32::min(step, extent - origin),
                                                   rect.size.height)),
                             rect.size.height / 2.0)
                        }
                        RectSide::Left | RectSide::Right => {
                            (Rect::new(Point2D::new(rect.origin.x, rect.origin.y + origin),
                                       Size2D::new(rect.size.width,
                                                   f32::min(step, extent - origin))),
                             rect.size.width / 2.0)
                        }
                    };

                    let mask_radius = DevicePixel::new(mask_radius, self.device_pixel_ratio);

                    let raster_op =
                        BorderRadiusRasterOp::create(mask_radius,
                                                     mask_radius,
                                                     DevicePixel::zero(),
                                                     DevicePixel::zero(),
                                                     false,
                                                     None,
                                                     ImageFormat::RGBA8).expect(
                        "Didn't find border radius mask for dashed border!");
                    let raster_item = RasterItem::BorderRadius(raster_op);
                    let color_image = resource_cache.get_raster(&raster_item, frame_id);

                    // Top left:
                    self.add_simple_rectangle(color_image.texture_id,
                                              &Rect::new(dot_rect.origin,
                                                         Size2D::new(dot_rect.size.width / 2.0,
                                                                     dot_rect.size.height / 2.0)),
                                              &color_image.uv_rect(),
                                              dummy_mask_image.texture_id,
                                              &dummy_mask_image.pixel_rect,
                                              &colors,
                                              None);

                    // Top right:
                    self.add_simple_rectangle(color_image.texture_id,
                                              &Rect::new(dot_rect.top_right(),
                                                         Size2D::new(-dot_rect.size.width / 2.0,
                                                                     dot_rect.size.height / 2.0)),
                                              &color_image.uv_rect(),
                                              dummy_mask_image.texture_id,
                                              &dummy_mask_image.pixel_rect,
                                              &colors,
                                              None);

                    // Bottom right:
                    self.add_simple_rectangle(color_image.texture_id,
                                              &Rect::new(dot_rect.bottom_right(),
                                                         Size2D::new(-dot_rect.size.width / 2.0,
                                                                     -dot_rect.size.height / 2.0)),
                                              &color_image.uv_rect(),
                                              dummy_mask_image.texture_id,
                                              &dummy_mask_image.pixel_rect,
                                              &colors,
                                              None);

                    // Bottom left:
                    self.add_simple_rectangle(color_image.texture_id,
                                              &Rect::new(dot_rect.bottom_left(),
                                                         Size2D::new(dot_rect.size.width / 2.0,
                                                                     -dot_rect.size.height / 2.0)),
                                              &color_image.uv_rect(),
                                              dummy_mask_image.texture_id,
                                              &dummy_mask_image.pixel_rect,
                                              &colors,
                                              None);

                    origin += step + step;
                }
            }
            BorderStyle::Double => {
                let (outer_rect, inner_rect) = match side {
                    RectSide::Top | RectSide::Bottom => {
                        (Rect::new(rect.origin,
                                   Size2D::new(rect.size.width, rect.size.height / 3.0)),
                         Rect::new(Point2D::new(rect.origin.x,
                                                rect.origin.y + rect.size.height * 2.0 / 3.0),
                                   Size2D::new(rect.size.width, rect.size.height / 3.0)))
                    }
                    RectSide::Left | RectSide::Right => {
                        (Rect::new(rect.origin,
                                   Size2D::new(rect.size.width / 3.0, rect.size.height)),
                         Rect::new(Point2D::new(rect.origin.x + rect.size.width * 2.0 / 3.0,
                                                rect.origin.y),
                                   Size2D::new(rect.size.width / 3.0, rect.size.height)))
                    }
                };
                self.add_color_rectangle(&outer_rect, color, resource_cache, frame_id);
                self.add_color_rectangle(&inner_rect, color, resource_cache, frame_id);
            }
            BorderStyle::Groove | BorderStyle::Ridge => {
                let (tl_rect, br_rect) = match side {
                    RectSide::Top | RectSide::Bottom => {
                        (Rect::new(rect.origin,
                                   Size2D::new(rect.size.width, rect.size.height / 2.0)),
                         Rect::new(Point2D::new(rect.origin.x,
                                                rect.origin.y + rect.size.height / 2.0),
                                   Size2D::new(rect.size.width, rect.size.height / 2.0)))
                    }
                    RectSide::Left | RectSide::Right => {
                        (Rect::new(rect.origin,
                                   Size2D::new(rect.size.width / 2.0, rect.size.height)),
                         Rect::new(Point2D::new(rect.origin.x + rect.size.width / 2.0,
                                                rect.origin.y),
                                   Size2D::new(rect.size.width / 2.0, rect.size.height)))
                    }
                };
                let (tl_color, br_color) = groove_ridge_border_colors(color, border_style);
                self.add_color_rectangle(&tl_rect, &tl_color, resource_cache, frame_id);
                self.add_color_rectangle(&br_rect, &br_color, resource_cache, frame_id);
            }
            _ => {
                self.add_color_rectangle(rect, color, resource_cache, frame_id);
            }
        }
    }

    /// Draws a border corner.
    ///
    /// The following diagram attempts to explain the parameters to this function. It's an enlarged
    /// version of a border corner that looks like this:
    ///
    /// ```notrust
    /// ╭─
    /// │
    /// ```
    ///
    /// The parameters are as follows:
    ///
    /// ```notrust
    /// ⤹ corner_bounds.origin
    /// ∙┈┈┈┈┈┬┈┈┈┈┈┬┈┈┈┈┈
    /// ┊   ╱ ┊     ┊
    /// ┊  ╱  ┊     ┊
    /// ┊ ╱╲  ┊   ←─┼── color1
    /// ┊╱  ╲ ┊     ┊
    /// ┊    ╲┊     ┊
    /// ├┈┈┈┈┈∙←────┼── radius_extent
    /// ┊     ┊╲    ┊
    /// ┊     ┊ ╲   ┊
    /// ┊     ┊  ╲  ┊
    /// ┊  ↑  ┊   ╲ ┊
    /// ┊  │  ┊    ╲┊
    /// ├┈┈┼┈┈┴┈┈┈┈┈∙┈┈┈┈┈
    /// ┊  │        ┊↖︎
    /// ┊           ┊  corner_bounds.bottom_right()
    /// ┊color0     ┊
    /// ```
    fn add_border_corner(&mut self,
                         border_style: BorderStyle,
                         corner_bounds: &Rect<f32>,
                         radius_extent: &Point2D<f32>,
                         color0: &ColorF,
                         color1: &ColorF,
                         outer_radius: &Size2D<f32>,
                         inner_radius: &Size2D<f32>,
                         resource_cache: &ResourceCache,
                         frame_id: FrameId,
                         can_tessellate: bool,
                         rotation_angle: BasicRotationAngle,
                         device_pixel_ratio: f32) {
        if color0.a <= 0.0 && color1.a <= 0.0 {
            return
        }

        match border_style {
            BorderStyle::Ridge | BorderStyle::Groove => {
                let corner_center = util::rect_center(corner_bounds);
                let (outer_corner_rect, inner_corner_rect, color1_rect, color0_rect) =
                    subdivide_border_corner(corner_bounds, &corner_center, rotation_angle);

                let (tl_color, br_color) = groove_ridge_border_colors(color0, border_style);
                let (color0_outer, color1_outer, color0_inner, color1_inner) =
                    match rotation_angle {
                        BasicRotationAngle::Upright => {
                            (&tl_color, &tl_color, &br_color, &br_color)
                        }
                        BasicRotationAngle::Clockwise90 => {
                            (&br_color, &tl_color, &tl_color, &br_color)
                        }
                        BasicRotationAngle::Clockwise180 => {
                            (&br_color, &br_color, &tl_color, &tl_color)
                        }
                        BasicRotationAngle::Clockwise270 => {
                            (&tl_color, &br_color, &br_color, &tl_color)
                        }
                    };

                // Draw the corner parts:
                self.add_solid_border_corner(&outer_corner_rect,
                                             radius_extent,
                                             &color0_outer,
                                             &color1_outer,
                                             outer_radius,
                                             inner_radius,
                                             resource_cache,
                                             frame_id,
                                             can_tessellate,
                                             rotation_angle,
                                             device_pixel_ratio);
                self.add_solid_border_corner(&inner_corner_rect,
                                             radius_extent,
                                             &color0_inner,
                                             &color1_inner,
                                             outer_radius,
                                             inner_radius,
                                             resource_cache,
                                             frame_id,
                                             can_tessellate,
                                             rotation_angle,
                                             device_pixel_ratio);

                // Draw the solid parts:
                if util::rect_is_well_formed_and_nonempty(&color0_rect) {
                    self.add_color_rectangle(&color0_rect, &color0_outer, resource_cache, frame_id)
                }
                if util::rect_is_well_formed_and_nonempty(&color1_rect) {
                    self.add_color_rectangle(&color1_rect, &color1_outer, resource_cache, frame_id)
                }
            }
            BorderStyle::Double => {
                //      ∙┈┈┈┈┈∙┈┈┈┈┈∙┈┈┈┈┈┐
                //      ┊0    ┊1    ┊2    ┊
                //      ┊     ┊     ┊     ┊
                //      ┊     ┊     ┊     ┊
                //      ∙┈┈┈┈┈∙┈┈┈┈┈∙┈┈┈┈┈┤
                //      ┊3    ┊4    ┊5    ┊
                //      ┊     ┊     ┊     ┊
                //      ┊     ┊     ┊     ┊
                //      ∙┈┈┈┈┈∙┈┈┈┈┈∙┈┈┈┈┈┤
                //      ┊6    ┊7    ┊8    ┊
                //      ┊     ┊     ┊     ┊
                //      ┊     ┊     ┊     ┊
                //      └┈┈┈┈┈┴┈┈┈┈┈┴┈┈┈┈┈┘

                let width_1_3 = corner_bounds.size.width / 3.0;
                let height_1_3 = corner_bounds.size.height / 3.0;
                let width_2_3 = width_1_3 * 2.0;
                let height_2_3 = height_1_3 * 2.0;
                let size_1_3 = Size2D::new(width_1_3, height_1_3);
                let size_width_2_3_height_1_3 = Size2D::new(width_2_3, height_1_3);
                let size_width_1_3_height_2_3 = Size2D::new(width_1_3, height_2_3);

                let p0 = corner_bounds.origin;
                let p1 = Point2D::new(corner_bounds.origin.x + width_1_3, corner_bounds.origin.y);
                let p2 = Point2D::new(corner_bounds.origin.x + width_2_3, corner_bounds.origin.y);
                let p3 = Point2D::new(corner_bounds.origin.x, corner_bounds.origin.y + height_1_3);
                let p5 = Point2D::new(corner_bounds.origin.x + width_2_3,
                                      corner_bounds.origin.y + height_1_3);
                let p6 = Point2D::new(corner_bounds.origin.x, corner_bounds.origin.y + height_2_3);
                let p7 = Point2D::new(corner_bounds.origin.x + width_1_3,
                                      corner_bounds.origin.y + height_2_3);
                let p8 = Point2D::new(corner_bounds.origin.x + width_2_3,
                                      corner_bounds.origin.y + height_2_3);

                let outer_corner_rect;
                let inner_corner_rect;
                let outer_side_rect_0;
                let outer_side_rect_1;
                match rotation_angle {
                    BasicRotationAngle::Upright => {
                        outer_corner_rect = Rect::new(p0, size_1_3);
                        outer_side_rect_1 = Rect::new(p1, size_width_2_3_height_1_3);
                        inner_corner_rect = Rect::new(p8, size_1_3);
                        outer_side_rect_0 = Rect::new(p3, size_width_1_3_height_2_3)
                    }
                    BasicRotationAngle::Clockwise90 => {
                        outer_corner_rect = Rect::new(p2, size_1_3);
                        outer_side_rect_1 = Rect::new(p5, size_width_1_3_height_2_3);
                        inner_corner_rect = Rect::new(p6, size_1_3);
                        outer_side_rect_0 = Rect::new(p0, size_width_2_3_height_1_3)
                    }
                    BasicRotationAngle::Clockwise180 => {
                        outer_corner_rect = Rect::new(p8, size_1_3);
                        outer_side_rect_1 = Rect::new(p6, size_width_2_3_height_1_3);
                        inner_corner_rect = Rect::new(p0, size_1_3);
                        outer_side_rect_0 = Rect::new(p2, size_width_1_3_height_2_3)
                    }
                    BasicRotationAngle::Clockwise270 => {
                        outer_corner_rect = Rect::new(p6, size_1_3);
                        outer_side_rect_1 = Rect::new(p0, size_width_1_3_height_2_3);
                        inner_corner_rect = Rect::new(p2, size_1_3);
                        outer_side_rect_0 = Rect::new(p7, size_width_2_3_height_1_3)
                    }
                }

                self.add_solid_border_corner(&outer_corner_rect,
                                             radius_extent,
                                             color0,
                                             color1,
                                             outer_radius,
                                             &Size2D::new(0.0, 0.0),
                                             resource_cache,
                                             frame_id,
                                             can_tessellate,
                                             rotation_angle,
                                             device_pixel_ratio);

                self.add_color_rectangle(&outer_side_rect_1, &color0, resource_cache, frame_id);

                self.add_solid_border_corner(&inner_corner_rect,
                                             radius_extent,
                                             color0,
                                             color1,
                                             &Size2D::new(0.0, 0.0),
                                             inner_radius,
                                             resource_cache,
                                             frame_id,
                                             can_tessellate,
                                             rotation_angle,
                                             device_pixel_ratio);

                self.add_color_rectangle(&outer_side_rect_0, &color1, resource_cache, frame_id);
            }
            _ => {
                self.add_solid_border_corner(corner_bounds,
                                             radius_extent,
                                             color0,
                                             color1,
                                             outer_radius,
                                             inner_radius,
                                             resource_cache,
                                             frame_id,
                                             can_tessellate,
                                             rotation_angle,
                                             device_pixel_ratio)
            }
        }
    }

    fn add_solid_border_corner(&mut self,
                               corner_bounds: &Rect<f32>,
                               radius_extent: &Point2D<f32>,
                               color0: &ColorF,
                               color1: &ColorF,
                               outer_radius: &Size2D<f32>,
                               inner_radius: &Size2D<f32>,
                               resource_cache: &ResourceCache,
                               frame_id: FrameId,
                               can_tessellate: bool,
                               rotation_angle: BasicRotationAngle,
                               _device_pixel_ratio: f32) {
        // TODO: Check for zero width/height borders!
        // FIXME(pcwalton): It's kind of messy to be matching on the rotation angle here to pick
        // the right rect to draw the rounded corner in. Is there a more elegant way to do this?
        let (outer_corner_rect, inner_corner_rect, color0_rect, color1_rect) =
            subdivide_border_corner(corner_bounds, radius_extent, rotation_angle);

        let dummy_mask_image = resource_cache.get_dummy_mask_image();

        // Draw the rounded part of the corner.
        let quad_count = if can_tessellate {
            tessellator::quad_count_for_border_corner(outer_radius, self.device_pixel_ratio)
        } else {
            1
        };
        for rect_index in 0..quad_count {
            let tessellated_rect;
            let index;
            if can_tessellate {
                tessellated_rect =
                    outer_corner_rect.tessellate_border_corner(outer_radius,
                                                               inner_radius,
                                                               self.device_pixel_ratio,
                                                               rotation_angle,
                                                               rect_index);
                index = Some(rect_index)
            } else {
                tessellated_rect = outer_corner_rect;
                index = None
            };

            let outer_radius_x = DevicePixel::new(outer_radius.width, self.device_pixel_ratio);
            let outer_radius_y = DevicePixel::new(outer_radius.height, self.device_pixel_ratio);
            let inner_radius_x = DevicePixel::new(inner_radius.width, self.device_pixel_ratio);
            let inner_radius_y = DevicePixel::new(inner_radius.height, self.device_pixel_ratio);

            let mask_image = match BorderRadiusRasterOp::create(outer_radius_x,
                                                                outer_radius_y,
                                                                inner_radius_x,
                                                                inner_radius_y,
                                                                false,
                                                                index,
                                                                ImageFormat::A8) {
                Some(raster_item) => {
                    resource_cache.get_raster(&RasterItem::BorderRadius(raster_item), frame_id)
                }
                None => dummy_mask_image,
            };

            // FIXME(pcwalton): Either use RGBA8 textures instead of alpha masks here, or implement
            // a mask combiner.
            let mask_uv = RectUv::from_uv_rect_rotation_angle(&mask_image.pixel_rect,
                                                              rotation_angle,
                                                              true);

            let tessellated_rect = RectPolygon {
                pos: tessellated_rect,
                varyings: mask_uv,
            };

            self.add_border_corner_piece(tessellated_rect,
                                         mask_image,
                                         color0,
                                         color1,
                                         resource_cache,
                                         rotation_angle);
        }

        // Draw the inner rect.
        self.add_border_corner_piece(RectPolygon {
                                        pos: inner_corner_rect,
                                        varyings: RectUv::zero(),
                                     },
                                     dummy_mask_image,
                                     color0,
                                     color1,
                                     resource_cache,
                                     rotation_angle);

        // Draw the two solid rects.
        if util::rect_is_well_formed_and_nonempty(&color0_rect) {
            self.add_color_rectangle(&color0_rect, color0, resource_cache, frame_id)
        }
        if util::rect_is_well_formed_and_nonempty(&color1_rect) {
            self.add_color_rectangle(&color1_rect, color1, resource_cache, frame_id)
        }
    }

    /// Draws one rectangle making up a border corner.
    fn add_border_corner_piece(&mut self,
                               rect_pos_uv: RectPolygon<RectUv<DevicePixel>>,
                               mask_image: &TextureCacheItem,
                               color0: &ColorF,
                               color1: &ColorF,
                               resource_cache: &ResourceCache,
                               rotation_angle: BasicRotationAngle) {
        if !rect_pos_uv.is_well_formed_and_nonempty() {
            return
        }

        let white_image = resource_cache.get_dummy_color_image();

        let v0;
        let v1;
        let muv;
        match rotation_angle {
            BasicRotationAngle::Upright => {
                v0 = rect_pos_uv.pos.origin;
                v1 = rect_pos_uv.pos.bottom_right();
                muv = RectUv {
                    top_left: rect_pos_uv.varyings.top_left,
                    top_right: rect_pos_uv.varyings.top_right,
                    bottom_right: rect_pos_uv.varyings.bottom_right,
                    bottom_left: rect_pos_uv.varyings.bottom_left,
                }
            }
            BasicRotationAngle::Clockwise90 => {
                v0 = rect_pos_uv.pos.top_right();
                v1 = rect_pos_uv.pos.bottom_left();
                muv = RectUv {
                    top_left: rect_pos_uv.varyings.top_right,
                    top_right: rect_pos_uv.varyings.top_left,
                    bottom_right: rect_pos_uv.varyings.bottom_left,
                    bottom_left: rect_pos_uv.varyings.bottom_right,
                }
            }
            BasicRotationAngle::Clockwise180 => {
                v0 = rect_pos_uv.pos.bottom_right();
                v1 = rect_pos_uv.pos.origin;
                muv = RectUv {
                    top_left: rect_pos_uv.varyings.bottom_right,
                    top_right: rect_pos_uv.varyings.bottom_left,
                    bottom_right: rect_pos_uv.varyings.top_left,
                    bottom_left: rect_pos_uv.varyings.top_right,
                }
            }
            BasicRotationAngle::Clockwise270 => {
                v0 = rect_pos_uv.pos.bottom_left();
                v1 = rect_pos_uv.pos.top_right();
                muv = RectUv {
                    top_left: rect_pos_uv.varyings.bottom_left,
                    top_right: rect_pos_uv.varyings.bottom_right,
                    bottom_right: rect_pos_uv.varyings.top_right,
                    bottom_left: rect_pos_uv.varyings.top_left,
                }
            }
        }

        self.add_rectangle(white_image.texture_id,
                           mask_image.texture_id,
                           &Rect::new(v0, Size2D::new(v1.x - v0.x, v1.y - v0.y)),
                           &RectUv::zero(),
                           &muv,
                           &[*color1, *color1, *color0, *color0],
                           PackedVertexColorMode::BorderCorner,
                           None)
    }

    fn add_color_image_rectangle(&mut self,
                                 v0: &Point2D<f32>,
                                 v1: &Point2D<f32>,
                                 color0: &ColorF,
                                 color1: &ColorF,
                                 color_image: &TextureCacheItem,
                                 resource_cache: &ResourceCache,
                                 rotation_angle: BasicRotationAngle,
                                 flip: bool) {
        if color0.a <= 0.0 || color1.a <= 0.0 {
            return
        }

        let vertices_rect = Rect::new(*v0, Size2D::new(v1.x - v0.x, v1.y - v0.y));

        let color_uv = RectUv::from_uv_rect_rotation_angle(&color_image.uv_rect(),
                                                           rotation_angle,
                                                           flip);

        let dummy_mask_image = resource_cache.get_dummy_mask_image();

        self.add_simple_rectangle(color_image.texture_id,
                                  &vertices_rect,
                                  &color_uv,
                                  dummy_mask_image.texture_id,
                                  &dummy_mask_image.pixel_rect,
                                  &[*color0, *color0, *color1, *color1],
                                  None);
    }

    pub fn add_border(&mut self,
                      rect: &Rect<f32>,
                      info: &BorderDisplayItem,
                      resource_cache: &ResourceCache,
                      frame_id: FrameId,
                      device_pixel_ratio: f32) {
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

        let can_tessellate = tessellator::can_tessellate_border(info);

        // Edges
        self.add_border_edge(&Rect::new(Point2D::new(tl_outer.x, tl_inner.y),
                                        Size2D::new(left.width, bl_inner.y - tl_inner.y)),
                             RectSide::Left,
                             &left_color,
                             info.left.style,
                             resource_cache,
                             frame_id);

        self.add_border_edge(&Rect::new(Point2D::new(tl_inner.x, tl_outer.y),
                                        Size2D::new(tr_inner.x - tl_inner.x,
                                                    tr_outer.y + top.width - tl_outer.y)),
                             RectSide::Top,
                             &top_color,
                             info.top.style,
                             resource_cache,
                             frame_id);

        self.add_border_edge(&Rect::new(Point2D::new(br_outer.x - right.width, tr_inner.y),
                                        Size2D::new(right.width, br_inner.y - tr_inner.y)),
                             RectSide::Right,
                             &right_color,
                             info.right.style,
                             resource_cache,
                             frame_id);

        self.add_border_edge(&Rect::new(Point2D::new(bl_inner.x, bl_outer.y - bottom.width),
                                        Size2D::new(br_inner.x - bl_inner.x,
                                                    br_outer.y - bl_outer.y + bottom.width)),
                             RectSide::Bottom,
                             &bottom_color,
                             info.bottom.style,
                             resource_cache,
                             frame_id);

        // Corners
        self.add_border_corner(info.left.style,
                               &Rect::new(tl_outer,
                                          Size2D::new(tl_inner.x - tl_outer.x,
                                                      tl_inner.y - tl_outer.y)),
                               &Point2D::new(tl_outer.x + radius.top_left.width,
                                             tl_outer.y + radius.top_left.height),
                               &left_color,
                               &top_color,
                               &radius.top_left,
                               &info.top_left_inner_radius(),
                               resource_cache,
                               frame_id,
                               can_tessellate,
                               BasicRotationAngle::Upright,
                               device_pixel_ratio);

        self.add_border_corner(info.top.style,
                               &Rect::new(Point2D::new(tr_inner.x, tr_outer.y),
                                          Size2D::new(tr_outer.x - tr_inner.x,
                                                      tr_inner.y - tr_outer.y)),
                               &Point2D::new(tr_outer.x - radius.top_right.width,
                                             tl_outer.y + radius.top_right.height),
                               &right_color,
                               &top_color,
                               &radius.top_right,
                               &info.top_right_inner_radius(),
                               resource_cache,
                               frame_id,
                               can_tessellate,
                               BasicRotationAngle::Clockwise90,
                               device_pixel_ratio);

        self.add_border_corner(info.right.style,
                               &Rect::new(br_inner,
                                          Size2D::new(br_outer.x - br_inner.x,
                                                      br_outer.y - br_inner.y)),
                               &Point2D::new(br_outer.x - radius.bottom_right.width,
                                             br_outer.y - radius.bottom_right.height),
                               &right_color,
                               &bottom_color,
                               &radius.bottom_right,
                               &info.bottom_right_inner_radius(),
                               resource_cache,
                               frame_id,
                               can_tessellate,
                               BasicRotationAngle::Clockwise180,
                               device_pixel_ratio);

        self.add_border_corner(info.bottom.style,
                               &Rect::new(Point2D::new(bl_outer.x, bl_inner.y),
                                          Size2D::new(bl_inner.x - bl_outer.x,
                                                      bl_outer.y - bl_inner.y)),
                               &Point2D::new(bl_outer.x + radius.bottom_left.width,
                                             bl_outer.y - radius.bottom_left.height),
                               &left_color,
                               &bottom_color,
                               &radius.bottom_left,
                               &info.bottom_left_inner_radius(),
                               resource_cache,
                               frame_id,
                               can_tessellate,
                               BasicRotationAngle::Clockwise270,
                               device_pixel_ratio);
    }

    // FIXME(pcwalton): Assumes rectangles are well-formed with origin in TL
    fn add_box_shadow_corner(&mut self,
                             top_left: &Point2D<f32>,
                             bottom_right: &Point2D<f32>,
                             corner_area_top_left: &Point2D<f32>,
                             corner_area_bottom_right: &Point2D<f32>,
                             box_rect: &Rect<f32>,
                             color: &ColorF,
                             blur_radius: f32,
                             border_radius: f32,
                             clip_mode: BoxShadowClipMode,
                             resource_cache: &ResourceCache,
                             frame_id: FrameId,
                             rotation_angle: BasicRotationAngle) {
        let corner_area_rect =
            Rect::new(*corner_area_top_left,
                      Size2D::new(corner_area_bottom_right.x - corner_area_top_left.x,
                                  corner_area_bottom_right.y - corner_area_top_left.y));

        self.push_clip_in_rect(&corner_area_rect);

        let inverted = match clip_mode {
            BoxShadowClipMode::Outset | BoxShadowClipMode::None => false,
            BoxShadowClipMode::Inset => true,
        };

        let color_image = match BoxShadowRasterOp::create_corner(blur_radius,
                                                                 border_radius,
                                                                 box_rect,
                                                                 inverted,
                                                                 self.device_pixel_ratio) {
            Some(raster_item) => {
                let raster_item = RasterItem::BoxShadow(raster_item);
                resource_cache.get_raster(&raster_item, frame_id)
            }
            None => resource_cache.get_dummy_color_image(),
        };

        self.add_color_image_rectangle(top_left,
                                       bottom_right,
                                       color,
                                       color,
                                       &color_image,
                                       resource_cache,
                                       rotation_angle,
                                       true);

        self.pop_clip_in_rect();
    }

    fn add_box_shadow_edge(&mut self,
                           top_left: &Point2D<f32>,
                           bottom_right: &Point2D<f32>,
                           edge_area_top_left: &Point2D<f32>,
                           edge_area_bottom_right: &Point2D<f32>,
                           box_rect: &Rect<f32>,
                           color: &ColorF,
                           blur_radius: f32,
                           border_radius: f32,
                           clip_mode: BoxShadowClipMode,
                           resource_cache: &ResourceCache,
                           frame_id: FrameId,
                           rotation_angle: BasicRotationAngle) {

        if top_left.x >= bottom_right.x || top_left.y >= bottom_right.y {
            return
        }

        let edge_area_rect =
            Rect::new(*edge_area_top_left,
                      Size2D::new(edge_area_bottom_right.x - edge_area_top_left.x,
                                  edge_area_bottom_right.y - edge_area_top_left.y));

        self.push_clip_in_rect(&edge_area_rect);
        let inverted = match clip_mode {
            BoxShadowClipMode::Outset | BoxShadowClipMode::None => false,
            BoxShadowClipMode::Inset => true,
        };

        let color_image = match BoxShadowRasterOp::create_edge(blur_radius,
                                                               border_radius,
                                                               box_rect,
                                                               inverted,
                                                               self.device_pixel_ratio) {
            Some(raster_item) => {
                let raster_item = RasterItem::BoxShadow(raster_item);
                resource_cache.get_raster(&raster_item, frame_id)
            }
            None => resource_cache.get_dummy_color_image(),
        };

        self.add_color_image_rectangle(top_left,
                                       bottom_right,
                                       color,
                                       color,
                                       &color_image,
                                       resource_cache,
                                       rotation_angle,
                                       false);
        self.pop_clip_in_rect();
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

pub fn compute_box_shadow_rect(box_bounds: &Rect<f32>,
                               box_offset: &Point2D<f32>,
                               mut spread_radius: f32,
                               clip_mode: BoxShadowClipMode)
                               -> Rect<f32> {
    let mut rect = (*box_bounds).clone();
    rect.origin.x += box_offset.x;
    rect.origin.y += box_offset.y;

    if clip_mode == BoxShadowClipMode::Inset {
        spread_radius = -spread_radius;
    };

    rect.inflate(spread_radius, spread_radius)
}

/// Returns the top/left and bottom/right colors respectively.
fn groove_ridge_border_colors(color: &ColorF, border_style: BorderStyle) -> (ColorF, ColorF) {
    match (color, border_style) {
        (&ColorF {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: _
        }, BorderStyle::Groove) => {
            // Handle black specially (matching the new browser consensus here).
            (ColorF::new(0.3, 0.3, 0.3, color.a), ColorF::new(0.7, 0.7, 0.7, color.a))
        }
        (&ColorF {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: _
        }, BorderStyle::Ridge) => {
            // As above.
            (ColorF::new(0.7, 0.7, 0.7, color.a), ColorF::new(0.3, 0.3, 0.3, color.a))
        }
        (_, BorderStyle::Groove) => (util::scale_color(color, 1.0 / 3.0), *color),
        (_, _) => (*color, util::scale_color(color, 2.0 / 3.0)),
    }
}

/// Subdivides the border corner into four quadrants and returns them in the order of outer corner,
/// inner corner, color 0 and color 1, respectively. See the diagram in the documentation for
/// `add_border_corner` for more information on what these values represent.
fn subdivide_border_corner(corner_bounds: &Rect<f32>,
                           point: &Point2D<f32>,
                           rotation_angle: BasicRotationAngle)
                           -> (Rect<f32>, Rect<f32>, Rect<f32>, Rect<f32>) {
    let (tl, tr, br, bl) = util::subdivide_rect_into_quadrants(corner_bounds, point);
    match rotation_angle {
        BasicRotationAngle::Upright => (tl, br, bl, tr),
        BasicRotationAngle::Clockwise90 => (tr, bl, tl, br),
        BasicRotationAngle::Clockwise180 => (br, tl, tr, bl),
        BasicRotationAngle::Clockwise270 => (bl, tr, br, tl),
    }
}

