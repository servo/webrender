use app_units::Au;

use freetype::freetype::{FTErrorMethods, FT_PIXEL_MODE_GRAY};
use freetype::freetype::{FT_Done_FreeType, FT_LOAD_RENDER};
use freetype::freetype::{FT_Library, FT_Set_Char_Size};
use freetype::freetype::{FT_Face, FT_Long, FT_UInt, FT_F26Dot6};
use freetype::freetype::{FT_Init_FreeType, FT_Load_Glyph};
use freetype::freetype::{FT_New_Memory_Face, FT_GlyphSlot};

use std::{mem, ptr, slice};
use std::collections::HashMap;
use string_cache::Atom;
//use util;

/// Native fonts are not used on Linux; all fonts are raw.
pub struct NativeFontHandle;

struct Face {
    face: FT_Face,
}

pub struct FontContext {
    lib: FT_Library,
    faces: HashMap<Atom, Face>,
}

pub struct RasterizedGlyph {
    pub left: i32,
    pub top: i32,
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
}

fn float_to_fixed(before: usize, f: f64) -> i32 {
    ((1i32 << before) as f64 * f) as i32
}

fn float_to_fixed_ft(f: f64) -> i32 {
    float_to_fixed(6, f)
}

impl FontContext {
    pub fn new() -> FontContext {
//        let _pf = util::ProfileScope::new("  FontContext::new");
        let mut lib: FT_Library = ptr::null_mut();
        unsafe {
            let result = FT_Init_FreeType(&mut lib);
            if !result.succeeded() { panic!("Unable to initialize FreeType library {}", result); }
        }

        FontContext {
            lib: lib,
            faces: HashMap::new(),
        }
    }

    pub fn add_font(&mut self, font_id: &Atom, bytes: &[u8]) {
        if !self.faces.contains_key(font_id) {
            let mut face: FT_Face = ptr::null_mut();
            let face_index = 0 as FT_Long;
            let result = unsafe {
                FT_New_Memory_Face(self.lib,
                                   bytes.as_ptr(),
                                   bytes.len() as FT_Long,
                                   face_index,
                                   &mut face)
            };
            if result.succeeded() && !face.is_null() {
                self.faces.insert(font_id.clone(), Face {
                    face: face,
                    //_bytes: bytes
                });
            } else {
                println!("WARN: webrender failed to load font {:?}", font_id);
            }
        }
    }

    pub fn get_glyph(&mut self,
                     font_id: &Atom,
                     size: Au,
                     character: u32,
                     device_pixel_ratio: f32) -> Option<RasterizedGlyph> {
        debug_assert!(self.faces.contains_key(&font_id));

        let face = self.faces.get(&font_id).unwrap();

        unsafe {
            let char_size = float_to_fixed_ft(((0.5f64 + size.to_f64_px()) *
                                               device_pixel_ratio as f64).floor());
            let result = FT_Set_Char_Size(face.face, char_size as FT_F26Dot6, 0, 0, 0);
            assert!(result.succeeded());

            let result =  FT_Load_Glyph(face.face, character as FT_UInt, FT_LOAD_RENDER);
            if result.succeeded() {

                let void_glyph = (*face.face).glyph;
                let slot: FT_GlyphSlot = mem::transmute(void_glyph);
                assert!(!slot.is_null());

                let bitmap = &(*slot).bitmap;
                assert!(bitmap.pixel_mode == FT_PIXEL_MODE_GRAY as i8);

                let buffer = slice::from_raw_parts(
                    bitmap.buffer,
                    (bitmap.width * bitmap.rows) as usize
                );

                let glyph = RasterizedGlyph {
                    left: (*slot).bitmap_left as i32,
                    top: (*slot).bitmap_top as i32,
                    width: bitmap.width as u32,
                    height: bitmap.rows as u32,
                    bytes: buffer.to_vec(),
                };

                Some(glyph)
            } else {
                None
            }
        }
    }
}

impl Drop for FontContext {
    fn drop(&mut self) {
        unsafe {
            FT_Done_FreeType(self.lib);
        }
    }
}
