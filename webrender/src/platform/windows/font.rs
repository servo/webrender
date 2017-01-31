/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use std::collections::HashMap;
use webrender_traits::{FontKey, ColorU, FontRenderMode, GlyphDimensions, GlyphOptions};
use webrender_traits::{SubpixelOffset};
use gamma_lut::{GammaLut, Color as ColorLut};

use dwrote;

lazy_static! {
    static ref DEFAULT_FONT_DESCRIPTOR: dwrote::FontDescriptor = dwrote::FontDescriptor {
        family_name: "Arial".to_owned(),
        weight: dwrote::FontWeight::Regular,
        stretch: dwrote::FontStretch::Normal,
        style: dwrote::FontStyle::Normal,
    };
}

pub struct FontContext {
    fonts: HashMap<FontKey, dwrote::FontFace>,
    gamma_lut: GammaLut,
    gdi_gamma_lut: GammaLut,
}

pub struct RasterizedGlyph {
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
}

fn dwrite_texture_type(render_mode: FontRenderMode) ->
                       dwrote::DWRITE_TEXTURE_TYPE {
    match render_mode {
        FontRenderMode::Mono => dwrote::DWRITE_TEXTURE_ALIASED_1x1 ,
        FontRenderMode::Alpha |
        FontRenderMode::Subpixel => dwrote::DWRITE_TEXTURE_CLEARTYPE_3x1,
    }
}

fn dwrite_measure_mode(render_mode: FontRenderMode, options: Option<GlyphOptions>) ->
                       dwrote::DWRITE_MEASURING_MODE {
    if let Some(GlyphOptions{ force_gdi_rendering: true, .. }) = options {
        return dwrote::DWRITE_MEASURING_MODE_GDI_CLASSIC;
    }

    match render_mode {
        FontRenderMode::Mono => dwrote::DWRITE_MEASURING_MODE_GDI_NATURAL,
        FontRenderMode::Alpha |
        FontRenderMode::Subpixel => dwrote::DWRITE_MEASURING_MODE_NATURAL,
    }
}

fn dwrite_render_mode(font_face: &dwrote::FontFace,
                      render_mode: FontRenderMode,
                      em_size: f32,
                      measure_mode: dwrote::DWRITE_MEASURING_MODE,
                      options: Option<GlyphOptions>) ->
                      dwrote::DWRITE_RENDERING_MODE {
    if let Some(GlyphOptions{ force_gdi_rendering: true, .. }) = options {
        return dwrote::DWRITE_RENDERING_MODE_GDI_CLASSIC;
    }

    let dwrite_render_mode = match render_mode {
        FontRenderMode::Mono => dwrote::DWRITE_RENDERING_MODE_ALIASED,
        FontRenderMode::Alpha |
        FontRenderMode::Subpixel => {
            font_face.get_recommended_rendering_mode_default_params(em_size,
                                                                    1.0,
                                                                    measure_mode)
        },
    };

    if dwrite_render_mode  == dwrote::DWRITE_RENDERING_MODE_OUTLINE {
        // Outline mode is not supported
        return dwrote::DWRITE_RENDERING_MODE_CLEARTYPE_NATURAL_SYMMETRIC;
    }

    dwrite_render_mode
}

fn get_glyph_dimensions_with_analysis(analysis: dwrote::GlyphRunAnalysis,
                                      texture_type: dwrote::DWRITE_TEXTURE_TYPE)
                                      -> GlyphDimensions {
    let bounds = analysis.get_alpha_texture_bounds(texture_type);

    let width = (bounds.right - bounds.left) as u32;
    let height = (bounds.bottom - bounds.top) as u32;
    assert!(width > 0 && height > 0);
    GlyphDimensions {
        left: bounds.left,
        top: -bounds.top,
        width: width,
        height: height,
    }
}

impl FontContext {
    pub fn new() -> FontContext {
        // These are the default values we use in Gecko.
        // We use a gamma value of 2.3 for gdi fonts
        // TODO: Fetch this data from Gecko itself.
        let contrast = 1.0;
        let gamma = 1.8;
        let gdi_gamma = 2.3;
        FontContext {
            fonts: HashMap::new(),
            gamma_lut: GammaLut::new(contrast, gamma, gamma),
            gdi_gamma_lut: GammaLut::new(contrast, gdi_gamma, gdi_gamma),
        }
    }

    pub fn add_raw_font(&mut self, font_key: &FontKey, data: &[u8]) {
        if self.fonts.contains_key(font_key) {
            return
        }

        if let Some(font_file) = dwrote::FontFile::new_from_data(data) {
            let face = font_file.create_face(0, dwrote::DWRITE_FONT_SIMULATIONS_NONE);
            self.fonts.insert((*font_key).clone(), face);
        } else {
            // XXX add_raw_font needs to have a way to return an error
            debug!("DWrite WR failed to load font from data, using Arial instead");
            self.add_native_font(font_key, DEFAULT_FONT_DESCRIPTOR.clone());
        }
    }

    pub fn add_native_font(&mut self, font_key: &FontKey, font_handle: dwrote::FontDescriptor) {
        if self.fonts.contains_key(font_key) {
            return
        }

        let system_fc = dwrote::FontCollection::system();
        let font = system_fc.get_font_from_descriptor(&font_handle).unwrap();
        let face = font.create_font_face();
        self.fonts.insert((*font_key).clone(), face);
    }

    // Assumes RGB format from dwrite, which is 3 bytes per pixel as dwrite
    // doesn't output an alpha value via GlyphRunAnalysis::CreateAlphaTexture
    #[allow(dead_code)]
    fn print_glyph_data(&self, data: &Vec<u8>, width: usize, height: usize) {
        // Rust doesn't have step_by support on stable :(
        for i in 0..height {
            let current_height = i * width * 3;

            for pixel in data[current_height .. current_height + (width * 3)].chunks(3) {
                let r = pixel[0];
                let g = pixel[1];
                let b = pixel[2];
                print!("({}, {}, {}) ", r, g, b, );
            }
            println!("");
        }
    }

    fn create_glyph_analysis(&self, font_key: FontKey,
                            size: Au, glyph: u32,
                            render_mode: FontRenderMode,
                            options: Option<GlyphOptions>) ->
                            dwrote::GlyphRunAnalysis {
        let face = self.fonts.get(&font_key).unwrap();
        let glyph = glyph as u16;

        let glyph = glyph as u16;
        let advance = 0.0f32;
        let offset = dwrote::GlyphOffset { advanceOffset: 0.0, ascenderOffset: 0.0 };

        let glyph_run = dwrote::DWRITE_GLYPH_RUN {
            fontFace: unsafe { face.as_ptr() },
            fontEmSize: size.to_f32_px(), // size in DIPs (1/96", same as CSS pixels)
            glyphCount: 1,
            glyphIndices: &glyph,
            glyphAdvances: &advance,
            glyphOffsets: &offset,
            isSideways: 0,
            bidiLevel: 0,
        };

        let dwrite_measure_mode = dwrite_measure_mode(render_mode, options);
        let dwrite_render_mode = dwrite_render_mode(face,
                                                    render_mode,
                                                    size.to_f32_px(),
                                                    dwrite_measure_mode,
                                                    options);

        // XX use the xform to handle subpixel positioning (what skia does), I believe that keeps
        //let xform = dwrote::DWRITE_MATRIX { m11: 1.0, m12: 0.0, m21: 0.0, m22: 1.0, dx: 0.0, dy: 0.0 };
        dwrote::GlyphRunAnalysis::create(&glyph_run, 1.0, None,
                                         dwrite_render_mode,
                                         dwrite_measure_mode,
                                         0.0, 0.0)
    }

    // TODO: Pipe GlyphOptions into glyph_dimensions too
    pub fn get_glyph_dimensions(&self,
                                font_key: FontKey,
                                size: Au,
                                glyph: u32,
                                _x_subpixel: SubpixelOffset,
                                _y_subpixel: SubpixelOffset)
                                -> Option<GlyphDimensions> {
        // Probably have to default to something else here.
        let render_mode = FontRenderMode::Subpixel;
        let analysis = self.create_glyph_analysis(font_key, size,
                                                  glyph, render_mode,
                                                  None);

        let texture_type = dwrite_texture_type(render_mode);
        Some(get_glyph_dimensions_with_analysis(analysis, texture_type))
    }

    // DWRITE gives us values in RGB. WR doesn't really touch it after. Note, CG returns in BGR
    // TODO: Decide whether all fonts should return RGB or BGR
    fn convert_to_rgba(&self, pixels: &Vec<u8>, render_mode: FontRenderMode) -> Vec<u8> {
        match render_mode {
            FontRenderMode::Mono => {
                let mut rgba_pixels: Vec<u8> = vec![0; pixels.len() * 4];
                for i in 0..pixels.len() {
                    rgba_pixels[i*4+0] = pixels[i];
                    rgba_pixels[i*4+1] = pixels[i];
                    rgba_pixels[i*4+2] = pixels[i];
                    rgba_pixels[i*4+3] = pixels[i];
                }
                rgba_pixels
            }
            FontRenderMode::Alpha => {
                let length = pixels.len() / 3;
                let mut rgba_pixels: Vec<u8> = vec![0; length * 4];
                for i in 0..length {
                    // TODO(vlad): we likely need to do something smarter
                    // This is what skia does
                    let alpha = ((pixels[i*3+0] as u32 +
                                pixels[i*3+1] as u32 +
                                pixels[i*3+2] as u32)
                                / 3) as u8;

                    rgba_pixels[i*4+0] = alpha;
                    rgba_pixels[i*4+1] = alpha;
                    rgba_pixels[i*4+2] = alpha;
                    rgba_pixels[i*4+3] = alpha;
                }
                rgba_pixels
            }
            FontRenderMode::Subpixel => {
                let length = pixels.len() / 3;
                let mut rgba_pixels: Vec<u8> = vec![0; length * 4];
                for i in 0..length {
                    rgba_pixels[i*4+0] = pixels[i*3+0];
                    rgba_pixels[i*4+1] = pixels[i*3+1];
                    rgba_pixels[i*4+2] = pixels[i*3+2];
                    rgba_pixels[i*4+3] = 0xff;
                }
                rgba_pixels
            }
        }
    }

    pub fn rasterize_glyph(&mut self,
                           font_key: FontKey,
                           size: Au,
                           color: ColorU,
                           glyph: u32,
                           render_mode: FontRenderMode,
                           _x_suboffset: SubpixelOffset,
                           _y_suboffset: SubpixelOffset,
                           glyph_options: Option<GlyphOptions>)
                           -> Option<RasterizedGlyph> {
        let analysis = self.create_glyph_analysis(font_key, size, glyph,
                                                  render_mode, glyph_options);
        let texture_type = dwrite_texture_type(render_mode);

        let bounds = analysis.get_alpha_texture_bounds(texture_type);
        let width = (bounds.right - bounds.left) as usize;
        let height = (bounds.bottom - bounds.top) as usize;

        let mut pixels = analysis.create_alpha_texture(texture_type, bounds);

        let lut_correction = match glyph_options {
            Some(option) => {
                if option.force_gdi_rendering {
                    &self.gdi_gamma_lut
                } else {
                    &self.gamma_lut
                }
            },
            None => &self.gamma_lut
        };

        lut_correction.preblend_rgb(&mut pixels, width, height,
                                    ColorLut::new(color.r, color.g, color.b, color.a));

        let rgba_pixels = self.convert_to_rgba(&mut pixels, render_mode);

        Some(RasterizedGlyph {
            width: width as u32,
            height: height as u32,
            bytes: rgba_pixels,
        })
    }
}
