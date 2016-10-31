/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use internal_types::FontRenderMode;
use webrender_traits::{FontKey, NativeFontHandle};

use freetype::freetype::{FTErrorMethods, FT_PIXEL_MODE_GRAY, FT_PIXEL_MODE_MONO, FT_PIXEL_MODE_LCD};
use freetype::freetype::{FT_Done_FreeType, FT_RENDER_MODE_LCD, FT_Library_SetLcdFilter};
use freetype::freetype::{FT_RENDER_MODE_NORMAL, FT_RENDER_MODE_MONO};
use freetype::freetype::{FT_Library, FT_Set_Char_Size};
use freetype::freetype::{FT_Face, FT_Long, FT_UInt, FT_F26Dot6};
use freetype::freetype::{FT_Init_FreeType, FT_Load_Glyph, FT_Render_Glyph};
use freetype::freetype::{FT_New_Memory_Face, FT_GlyphSlot, FT_LcdFilter};

use std::{mem, ptr, slice};
use std::collections::HashMap;
//use util;

struct Face {
    face: FT_Face,
}

pub struct FontContext {
    lib: FT_Library,
    faces: HashMap<FontKey, Face>,
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

            // TODO(gw): Check result of this to determine if freetype build supports subpixel.
            FT_Library_SetLcdFilter(lib, FT_LcdFilter::FT_LCD_FILTER_DEFAULT);
        }

        FontContext {
            lib: lib,
            faces: HashMap::new(),
        }
    }

    pub fn add_raw_font(&mut self, font_key: &FontKey, bytes: &[u8]) {
        if !self.faces.contains_key(&font_key) {
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
                self.faces.insert(*font_key, Face {
                    face: face,
                    //_bytes: bytes
                });
            } else {
                println!("WARN: webrender failed to load font {:?}", font_key);
            }
        }
    }

    pub fn add_native_font(&mut self, _font_key: &FontKey, _native_font_handle: NativeFontHandle) {
        panic!("TODO: Not supported on Linux");
    }

    fn load_glyph(&self,
                  font_key: FontKey,
                  size: Au,
                  character: u32,
                  device_pixel_ratio: f32) -> Option<FT_GlyphSlot> {
        debug_assert!(self.faces.contains_key(&font_key));
        let face = self.faces.get(&font_key).unwrap();
        let mut slot = None;

        unsafe {
            let char_size = float_to_fixed_ft(((0.5f64 + size.to_f64_px()) *
                                               device_pixel_ratio as f64).floor());
            let result = FT_Set_Char_Size(face.face, char_size as FT_F26Dot6, 0, 0, 0);
            assert!(result.succeeded());

            let result =  FT_Load_Glyph(face.face, character as FT_UInt, 0);
            if result.succeeded() {
                let void_glyph = (*face.face).glyph;
                let slot_ptr: FT_GlyphSlot = mem::transmute(void_glyph);
                assert!(!slot_ptr.is_null());
                slot = Some(slot_ptr);
            }
        }

        slot
    }

    #[allow(dead_code)]     // TODO(gw): Expose this to the public glyph dimensions API.
    pub fn get_glyph_dimensions(&self,
                                font_key: FontKey,
                                size: Au,
                                character: u32,
                                device_pixel_ratio: f32) -> Option<(Au, Au)> {
        self.load_glyph(font_key, size, character, device_pixel_ratio).map(|slot| {
            let metrics = unsafe { &(*slot).metrics };
            (Au::from_px((metrics.width >> 6) as i32), Au::from_px((metrics.height >> 6) as i32))
        })
    }

    pub fn get_glyph(&mut self,
                     font_key: FontKey,
                     size: Au,
                     character: u32,
                     device_pixel_ratio: f32,
                     render_mode: FontRenderMode) -> Option<RasterizedGlyph> {
        let mut glyph = None;

        if let Some(slot) = self.load_glyph(font_key,
                                            size,
                                            character,
                                            device_pixel_ratio) {
            let render_mode = match render_mode {
                FontRenderMode::Mono => FT_RENDER_MODE_MONO,
                FontRenderMode::Alpha => FT_RENDER_MODE_NORMAL,
                FontRenderMode::Subpixel => FT_RENDER_MODE_LCD,
            };

            unsafe {
                let result = FT_Render_Glyph(slot, render_mode);

                if result.succeeded() {
                    let bitmap = &(*slot).bitmap;
                    let bitmap_mode = bitmap.pixel_mode as u32;

                    let width = match bitmap_mode {
                        FT_PIXEL_MODE_MONO | FT_PIXEL_MODE_GRAY => bitmap.width,
                        FT_PIXEL_MODE_LCD => bitmap.width / 3,
                        _ => panic!("Unexpected render mode!"),
                    } as u32;

                    let mut final_buffer = Vec::with_capacity(width as usize * bitmap.rows as usize * 4);

                    match bitmap_mode {
                        FT_PIXEL_MODE_MONO => {
                            // This is not exactly efficient... but it's only used by the
                            // reftest pass when we have AA disabled on glyphs.
                            for y in 0..bitmap.rows {
                                for x in 0..bitmap.width {
                                    let byte_index = (y * bitmap.pitch) + (x >> 3);
                                    let bit_index = x & 7;
                                    let byte_ptr = bitmap.buffer.offset(byte_index as isize);
                                    let bit = (*byte_ptr & (128 >> bit_index)) != 0;
                                    let byte_value = if bit {
                                        0xff
                                    } else {
                                        0
                                    };
                                    final_buffer.extend_from_slice(&[ 0xff, 0xff, 0xff, byte_value ]);
                                }
                            }
                        }
                        FT_PIXEL_MODE_GRAY => {
                            let buffer = slice::from_raw_parts(
                                bitmap.buffer,
                                (bitmap.width * bitmap.rows) as usize
                            );

                            // Convert to RGBA.
                            for &byte in buffer.iter() {
                                final_buffer.extend_from_slice(&[ 0xff, 0xff, 0xff, byte ]);
                            }
                        }
                        FT_PIXEL_MODE_LCD => {
                            for y in 0..bitmap.rows {
                                for x in 0..(bitmap.width / 3) {
                                    let index = (y * bitmap.pitch) + (x * 3);
                                    let ptr = bitmap.buffer.offset(index as isize);
                                    let b = *ptr;
                                    let g = *(ptr.offset(1));
                                    let r = *(ptr.offset(2));
                                    final_buffer.extend_from_slice(&[ r, g, b, 0xff ]);
                                }
                            }
                        }
                        _ => panic!("Unexpected render mode!"),
                    }

                    glyph = Some(RasterizedGlyph {
                        left: (*slot).bitmap_left as i32,
                        top: (*slot).bitmap_top as i32,
                        width: width,
                        height: bitmap.rows as u32,
                        bytes: final_buffer,
                    });
                }
            }
        }

        glyph
    }
}

impl Drop for FontContext {
    fn drop(&mut self) {
        unsafe {
            FT_Done_FreeType(self.lib);
        }
    }
}
