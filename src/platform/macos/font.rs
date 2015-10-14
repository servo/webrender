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
use libc::size_t;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use string_cache::Atom;

pub struct FontContext {
    cg_fonts: HashMap<Atom, CGFont>,
    ct_fonts: HashMap<(Atom, Au), CTFont>,
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

impl FontContext {
    pub fn new() -> FontContext {
        FontContext {
            cg_fonts: HashMap::new(),
            ct_fonts: HashMap::new(),
        }
    }

    pub fn add_font(&mut self, font_id: &Atom, bytes: &[u8]) {
        if self.cg_fonts.contains_key(font_id) {
            return
        }

        let data_provider = CGDataProvider::from_buffer(bytes);
        let cg_font = match CGFont::from_data_provider(data_provider) {
            Err(_) => return,
            Ok(cg_font) => cg_font,
        };
        self.cg_fonts.insert((*font_id).clone(), cg_font);
    }

    pub fn get_glyph(&mut self,
                     font_id: &Atom,
                     size: Au,
                     character: u32,
                     device_pixel_ratio: f32)
                     -> Option<RasterizedGlyph> {
        let ct_font = match self.ct_fonts.entry(((*font_id).clone(), size)) {
            Entry::Occupied(entry) => (*entry.get()).clone(),
            Entry::Vacant(entry) => {
                let cg_font = match self.cg_fonts.get(font_id) {
                    None => return Some(RasterizedGlyph::blank()),
                    Some(cg_font) => cg_font,
                };
                let ct_font = core_text::font::new_from_CGFont(
                        cg_font,
                        size.to_f64_px() * (device_pixel_ratio as f64));
                entry.insert(ct_font.clone());
                ct_font
            }
        };

        let glyph = character as CGGlyph;
        let bounds = ct_font.get_bounding_rects_for_glyphs(kCTFontDefaultOrientation, &[glyph]);

        // We add in one extra pixel of width in the horizontal direction to account for
        // antialiasing.
        let rasterized_left = bounds.origin.x.floor() as i32;
        let rasterized_width =
            (bounds.origin.x - (rasterized_left as f64) + bounds.size.width).ceil() as u32 + 1;
        let rasterized_descent = (-bounds.origin.y).ceil() as i32;
        let rasterized_ascent = (bounds.size.height + bounds.origin.y).ceil() as i32;
        let rasterized_height = (rasterized_descent + rasterized_ascent) as u32;
        if rasterized_width == 0 || rasterized_height == 0 {
            return Some(RasterizedGlyph::blank())
        }

        let mut cg_context = CGContext::create_bitmap_context(rasterized_width as size_t,
                                                              rasterized_height as size_t,
                                                              8,
                                                              rasterized_width as size_t * 4,
                                                              &CGColorSpace::create_device_rgb(),
                                                              kCGImageAlphaPremultipliedLast);

        let rasterization_origin = CGPoint {
            x: bounds.origin.x - (rasterized_left as f64),
            y: rasterized_descent as f64,
        };
        ct_font.draw_glyphs(&[glyph], &[rasterization_origin], cg_context.clone());

        let rasterized_area = rasterized_width * rasterized_height;
        let rasterized_pixels = cg_context.data();
        let mut final_pixels = Vec::with_capacity(rasterized_area as usize);
        for i in 0..(rasterized_area as usize) {
            final_pixels.push(rasterized_pixels[i * 4 + 3]);
        }

        Some(RasterizedGlyph {
            left: rasterized_left,
            top: (bounds.size.height + bounds.origin.y).ceil() as i32,
            width: rasterized_width,
            height: rasterized_height,
            bytes: final_pixels,
        })
    }
}

