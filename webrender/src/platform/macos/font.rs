/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use core_graphics::base::kCGImageAlphaPremultipliedLast;
use core_graphics::color_space::CGColorSpace;
use core_graphics::context::CGContext;
use core_graphics::data_provider::CGDataProvider;
use core_graphics::font::{CGFont, CGGlyph};
use core_graphics::geometry::CGPoint;
use core_text::font::CTFont;
use core_text::font_descriptor::kCTFontDefaultOrientation;
use core_text;
use internal_types::FontRenderMode;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use webrender_traits::FontKey;

pub type NativeFontHandle = CGFont;

pub struct FontContext {
    cg_fonts: HashMap<FontKey, CGFont>,
    ct_fonts: HashMap<(FontKey, Au), CTFont>,
}

pub struct RasterizedGlyph {
    pub left: i32,
    pub top: i32,
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
}

impl RasterizedGlyph {
    pub fn blank() -> RasterizedGlyph {
        RasterizedGlyph {
            left: 0,
            top: 0,
            width: 0,
            height: 0,
            bytes: vec![],
        }
    }
}

struct GlyphMetrics {
    rasterized_left: i32,
    rasterized_descent: i32,
    rasterized_ascent: i32,
    rasterized_width: u32,
    rasterized_height: u32,
}

fn get_glyph_metrics(ct_font: &CTFont, glyph: CGGlyph) -> GlyphMetrics {
    let bounds = ct_font.get_bounding_rects_for_glyphs(kCTFontDefaultOrientation, &[glyph]);

    let rasterized_left = bounds.origin.x.floor() as i32;
    let rasterized_width =
        (bounds.origin.x - (rasterized_left as f64) + bounds.size.width).ceil() as u32;
    let rasterized_descent = (-bounds.origin.y).ceil() as i32;
    let rasterized_ascent = (bounds.size.height + bounds.origin.y).ceil() as i32;
    let rasterized_height = (rasterized_descent + rasterized_ascent) as u32;

    GlyphMetrics {
        rasterized_ascent: rasterized_ascent,
        rasterized_descent: rasterized_descent,
        rasterized_left: rasterized_left,
        rasterized_width: rasterized_width,
        rasterized_height: rasterized_height,
    }
}

impl FontContext {
    pub fn new() -> FontContext {
        FontContext {
            cg_fonts: HashMap::new(),
            ct_fonts: HashMap::new(),
        }
    }

    pub fn add_raw_font(&mut self, font_key: &FontKey, bytes: &[u8]) {
        if self.cg_fonts.contains_key(font_key) {
            return
        }

        let data_provider = CGDataProvider::from_buffer(bytes);
        let cg_font = match CGFont::from_data_provider(data_provider) {
            Err(_) => return,
            Ok(cg_font) => cg_font,
        };
        self.cg_fonts.insert((*font_key).clone(), cg_font);
    }

    pub fn add_native_font(&mut self, font_key: &FontKey, native_font_handle: CGFont) {
        if self.cg_fonts.contains_key(font_key) {
            return
        }

        self.cg_fonts.insert((*font_key).clone(), native_font_handle);
    }

    fn get_ct_font(&mut self,
                   font_key: FontKey,
                   size: Au,
                   device_pixel_ratio: f32) -> Option<CTFont> {
        match self.ct_fonts.entry(((font_key).clone(), size)) {
            Entry::Occupied(entry) => Some((*entry.get()).clone()),
            Entry::Vacant(entry) => {
                let cg_font = match self.cg_fonts.get(&font_key) {
                    None => return None,
                    Some(cg_font) => cg_font,
                };
                let ct_font = core_text::font::new_from_CGFont(
                        cg_font,
                        size.to_f64_px() * (device_pixel_ratio as f64));
                entry.insert(ct_font.clone());
                Some(ct_font)
            }
        }
    }

    #[allow(dead_code)]     // TODO(gw): Expose this to the public glyph dimensions API.
    pub fn get_glyph_dimensions(&mut self,
                                font_key: FontKey,
                                size: Au,
                                character: u32,
                                device_pixel_ratio: f32) -> Option<(Au, Au)> {
        self.get_ct_font(font_key, size, device_pixel_ratio).map(|ref ct_font| {
            let glyph = character as CGGlyph;
            let metrics = get_glyph_metrics(ct_font, glyph);
            (Au::from_px(metrics.rasterized_width as i32), Au::from_px(metrics.rasterized_height as i32))
        })
    }

    pub fn rasterize_glyph(&mut self,
                           font_key: FontKey,
                           size: Au,
                           character: u32,
                           device_pixel_ratio: f32,
                           render_mode: FontRenderMode) -> Option<RasterizedGlyph> {
        match self.get_ct_font(font_key, size, device_pixel_ratio) {
            Some(ref ct_font) => {
                let glyph = character as CGGlyph;
                let metrics = get_glyph_metrics(ct_font, glyph);
                if metrics.rasterized_width == 0 || metrics.rasterized_height == 0 {
                    return Some(RasterizedGlyph::blank())
                }

                let mut cg_context = CGContext::create_bitmap_context(metrics.rasterized_width as usize,
                                                                      metrics.rasterized_height as usize,
                                                                      8,
                                                                      metrics.rasterized_width as usize * 4,
                                                                      &CGColorSpace::create_device_rgb(),
                                                                      kCGImageAlphaPremultipliedLast);
                // TODO(gw): Add subpixel render mode support on mac.
                let enable_aa = render_mode != FontRenderMode::Mono;
                cg_context.set_allows_font_smoothing(enable_aa);
                cg_context.set_should_smooth_fonts(enable_aa);
                cg_context.set_allows_antialiasing(enable_aa);
                cg_context.set_should_antialias(enable_aa);
                cg_context.set_rgb_fill_color(1.0, 1.0, 1.0, 1.0);

                let rasterization_origin = CGPoint {
                    x: -metrics.rasterized_left as f64,
                    y: metrics.rasterized_descent as f64,
                };
                ct_font.draw_glyphs(&[glyph], &[rasterization_origin], cg_context.clone());

                let rasterized_area = (metrics.rasterized_width * metrics.rasterized_height) as usize;
                let mut rasterized_pixels = cg_context.data().to_vec();
                for i in 0..rasterized_area {
                    let alpha = (rasterized_pixels[i * 4 + 3] as f32) / 255.0;
                    for j in 0..3 {
                        rasterized_pixels[i * 4 + j] =
                            ((rasterized_pixels[i * 4 + j] as f32) / alpha) as u8;
                    }
                }

                Some(RasterizedGlyph {
                    left: metrics.rasterized_left,
                    top: metrics.rasterized_ascent,
                    width: metrics.rasterized_width,
                    height: metrics.rasterized_height,
                    bytes: rasterized_pixels,
                })
            }
            None => {
                return Some(RasterizedGlyph::blank());
            }
        }
    }
}

