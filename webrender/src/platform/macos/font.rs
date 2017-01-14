/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use core_graphics::base::{kCGImageAlphaNoneSkipFirst, kCGImageAlphaPremultipliedLast};
use core_graphics::base::kCGBitmapByteOrder32Little;
use core_graphics::color_space::CGColorSpace;
use core_graphics::context::{CGContext, CGTextDrawingMode};
use core_graphics::data_provider::CGDataProvider;
use core_graphics::font::{CGFont, CGGlyph};
use core_graphics::geometry::{CGPoint, CGSize, CGRect};
use core_text::font::CTFont;
use core_text::font_descriptor::kCTFontDefaultOrientation;
use core_text;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use webrender_traits::{FontKey, FontRenderMode, GlyphDimensions};

pub type NativeFontHandle = CGFont;

pub struct FontContext {
    cg_fonts: HashMap<FontKey, CGFont>,
    ct_fonts: HashMap<(FontKey, Au), CTFont>,
}

pub struct RasterizedGlyph {
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
}

impl RasterizedGlyph {
    pub fn blank() -> RasterizedGlyph {
        RasterizedGlyph {
            width: 0,
            height: 0,
            bytes: vec![],
        }
    }
}

#[derive(Debug)]
struct GlyphMetrics {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

impl GlyphMetrics {
    pub fn width(&self) -> usize {
        return (self.right - self.left) as usize;
    }

    pub fn height(&self) -> usize {
        return (self.top - self.bottom) as usize;
    }
}

// According to the Skia source code, there's no public API to
// determine if subpixel AA is supported. So jrmuizel ported
// this function from Skia which is used to check if a glyph
// can be rendered with subpixel AA.
fn supports_subpixel_aa() -> bool {
    let mut cg_context = CGContext::create_bitmap_context(1, 1, 8, 4,
                                                          &CGColorSpace::create_device_rgb(),
                                                          kCGImageAlphaNoneSkipFirst |
                                                          kCGBitmapByteOrder32Little);
    let ct_font = core_text::font::new_from_name("Helvetica", 16.).unwrap();
    cg_context.set_should_smooth_fonts(true);
    cg_context.set_should_antialias(true);
    cg_context.set_allows_font_smoothing(true);
    cg_context.set_rgb_fill_color(1.0, 1.0, 1.0, 1.0);
    let point = CGPoint {x: -1., y: 0.};
    let glyph = '|' as CGGlyph;
    ct_font.draw_glyphs(&[glyph], &[point], cg_context.clone());
    let data = cg_context.data();
    data[0] != data[1] || data[1] != data[2]
}

fn get_glyph_metrics(ct_font: &CTFont, glyph: CGGlyph) -> GlyphMetrics {
    let bounds = ct_font.get_bounding_rects_for_glyphs(kCTFontDefaultOrientation, &[glyph]);

    // First round out to pixel boundaries
    // CG Origin is bottom left
    let mut left = bounds.origin.x.floor() as i32;
    let mut bottom = bounds.origin.y.floor() as i32;
    let mut right = (bounds.origin.x + bounds.size.width).ceil() as i32;
    let mut top = (bounds.origin.y + bounds.size.height).ceil() as i32;

    // Expand the bounds by 1 pixel, to give CG room for anti-aliasing.
    // Note that this outset is to allow room for LCD smoothed glyphs. However, the correct outset
    // is not currently known, as CG dilates the outlines by some percentage.
    // This is taken from Skia.
    left -= 1;
    bottom -= 1;
    right += 1;
    top += 1;

    let metrics = GlyphMetrics {
        left: left,
        top: top,
        right: right,
        bottom: bottom,
    };

    metrics
}

impl FontContext {
    pub fn new() -> FontContext {
        debug!("Test for subpixel AA support: {}", supports_subpixel_aa());

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
                   size: Au) -> Option<CTFont> {
        match self.ct_fonts.entry(((font_key).clone(), size)) {
            Entry::Occupied(entry) => Some((*entry.get()).clone()),
            Entry::Vacant(entry) => {
                let cg_font = match self.cg_fonts.get(&font_key) {
                    None => return None,
                    Some(cg_font) => cg_font,
                };
                let ct_font = core_text::font::new_from_CGFont(
                        cg_font,
                        size.to_f64_px());
                entry.insert(ct_font.clone());
                Some(ct_font)
            }
        }
    }

    pub fn get_glyph_dimensions(&mut self,
                                font_key: FontKey,
                                size: Au,
                                character: u32) -> Option<GlyphDimensions> {
        self.get_ct_font(font_key, size).and_then(|ref ct_font| {
            let glyph = character as CGGlyph;
            let metrics = get_glyph_metrics(ct_font, glyph);
            if metrics.width() == 0 || metrics.height() == 0 {
                None
            } else {
                Some(GlyphDimensions {
                    left: metrics.left,
                    top: metrics.top,
                    width: metrics.width() as u32,
                    height: metrics.height() as u32,
                })
            }
        })
    }

    #[allow(dead_code)]
    fn print_glyph_data(&mut self, cg_context: &mut CGContext, width: usize, height: usize) {
        let data = cg_context.data();
        // Rust doesn't have step_by support on stable :(
        for i in 0..height {
            let current_height = i * width * 4;

            for pixel in data[current_height .. current_height + (width * 4)].chunks(4) {
                let b = pixel[0];
                let g = pixel[1];
                let r = pixel[2];
                print!("({}, {}, {}) ", r, g, b);
            }
            println!("");
        }
    }

    pub fn rasterize_glyph(&mut self,
                           font_key: FontKey,
                           size: Au,
                           character: u32,
                           render_mode: FontRenderMode) -> Option<RasterizedGlyph> {
        match self.get_ct_font(font_key, size) {
            Some(ref ct_font) => {
                let glyph = character as CGGlyph;
                let metrics = get_glyph_metrics(ct_font, glyph);
                if metrics.width() == 0 || metrics.height() == 0 {
                    return Some(RasterizedGlyph::blank())
                }

                let context_flags = match render_mode {
                    FontRenderMode::Subpixel => kCGBitmapByteOrder32Little | kCGImageAlphaNoneSkipFirst,
                    FontRenderMode::Alpha | FontRenderMode::Mono => kCGImageAlphaPremultipliedLast,
                };

                let mut cg_context = CGContext::create_bitmap_context(metrics.width() as usize,
                                                                      metrics.height() as usize,
                                                                      8,
                                                                      metrics.width() as usize * 4,
                                                                      &CGColorSpace::create_device_rgb(),
                                                                      context_flags);

                let (antialias, smooth) = match render_mode {
                    FontRenderMode::Subpixel => (true, true),
                    FontRenderMode::Alpha => (true, false),
                    FontRenderMode::Mono => (false, false),
                };

                // These are always true in Gecko, even for non-AA fonts
                cg_context.set_allows_font_subpixel_positioning(true);
                cg_context.set_should_subpixel_position_fonts(true);

                cg_context.set_allows_font_smoothing(smooth);
                cg_context.set_should_smooth_fonts(smooth);
                cg_context.set_allows_antialiasing(antialias);
                cg_context.set_should_antialias(antialias);

                let rasterization_origin = CGPoint {
                    x: -metrics.left as f64,
                    y: (-metrics.top + metrics.height() as i32) as f64,
                };

                // CGFonts have to be drawn with black text on a white background
                // After 10.11, there are two different glyphs depending on
                // white on black or black on white. Gecko defaults to black on white.
                cg_context.set_rgb_fill_color(1.0, 1.0, 1.0, 1.0);

                let rect = CGRect {
                    origin: CGPoint {
                        x: 0.0,
                        y: 0.0,
                    },
                    size: CGSize {
                        width: metrics.width() as f64,
                        height: metrics.height() as f64,
                    }
                };

                // Fill white
                cg_context.fill_rect(rect);

                // Now draw black
                cg_context.set_text_drawing_mode(CGTextDrawingMode::CGTextFill);
                cg_context.set_rgb_fill_color(0.0, 0.0, 0.0, 1.0);

                ct_font.draw_glyphs(&[glyph], &[rasterization_origin], cg_context.clone());

                let rasterized_area = (metrics.width() * metrics.height()) as usize;
                let mut rasterized_pixels = cg_context.data().to_vec();

                self.print_glyph_data(&mut cg_context, metrics.width(), metrics.height());

                // We need to invert the pixels back since right now
                // transparent pixels are actually white.
                for i in 0..metrics.height() {
                    let current_height :usize = (i * metrics.width() * 4) as usize;
                    let end_row :usize = current_height + (metrics.width() * 4);

                    for mut pixel in rasterized_pixels[current_height .. end_row ].chunks_mut(4) {
                        pixel[0] = 255 - pixel[0];
                        pixel[1] = 255 - pixel[1];
                        pixel[2] = 255 - pixel[2];
                        pixel[3] = 255;
                    }
                }

                match render_mode {
                    FontRenderMode::Alpha | FontRenderMode::Mono => {
                        for i in 0..rasterized_area {
                            let alpha = (rasterized_pixels[i * 4 + 3] as f32) / 255.0;
                            for channel in &mut rasterized_pixels[(i*4)..(i*4 + 3)] {
                                *channel = ((*channel as f32) / alpha) as u8;
                            }
                        }
                    }
                    FontRenderMode::Subpixel => {}
                }

                Some(RasterizedGlyph {
                    width: metrics.width() as u32,
                    height: metrics.height() as u32,
                    bytes: rasterized_pixels,
                })
            }
            None => {
                return Some(RasterizedGlyph::blank());
            }
        }
    }
}

