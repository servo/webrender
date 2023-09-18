/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{ColorU, GlyphDimensions, FontKey, FontRenderMode, FontSize};
use api::{FontInstanceFlags, NativeFontHandle};
use font_index::{FontCache, FontId, Font};
use zeno::Placement;
use crate::rasterizer::{FontInstance, GlyphKey};
use crate::rasterizer::{
    GlyphFormat, GlyphRasterError, GlyphRasterResult, RasterizedGlyph, FontTransform,
};
use crate::types::FastHashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use swash::FontRef;
use swash::scale::ScaleContext;
use swash::scale::StrikeWith;
use swash::scale::image::{Image as GlyphImage, Content};
use swash::scale::Source;
use swash::scale::Render;
use swash::GlyphId;
use std::mem;

// We rely on Gecko to determine whether the font may have color glyphs to avoid
// needing to load the font ahead of time to query its symbolic traits.
fn is_bitmap_font(font: &FontInstance) -> bool {
    font.flags.contains(FontInstanceFlags::EMBEDDED_BITMAPS)
}

pub struct FontContext {
    fonts: FastHashMap<FontKey, Font>,
    font_cache: FontCache,
    scale_context: ScaleContext,
    cache: FastHashMap<(FontInstance, GlyphKey), GlyphImage>,
}

impl FontContext {
    pub fn distribute_across_threads() -> bool {
        true
    }

    pub fn new() -> FontContext {
        FontContext {
            fonts: FastHashMap::default(),
            font_cache: FontCache::default(),
            cache: FastHashMap::default(),
            scale_context: ScaleContext::new(),
        }
    }

    pub fn add_raw_font(&mut self, font_key: &FontKey, data: Arc<Vec<u8>>, index: u32) {
        if self.fonts.contains_key(font_key) {
            return;
        }
        if let Some(font) = Font::from_data(data.to_vec(), index as usize) {
            self.fonts.insert(*font_key, font);
        }
    }

    pub fn add_native_font(&mut self, font_key: &FontKey, handle: NativeFontHandle) {
        if self.fonts.contains_key(font_key) {
            return;
        }
        if let Some(font) = self.font_cache.get(FontId(handle.0)) {
            self.fonts.insert(*font_key, font);
        }
    }

    pub fn delete_font(&mut self, font_key: &FontKey) {
        if let Some(_) = self.fonts.remove(font_key) {
            self.cache.retain(|k, _| k.0.font_key != *font_key);
        }
    }

    pub fn delete_font_instance(&mut self, instance: &FontInstance) {
        // Remove the Swash image corresponding to this instance.
        self.cache
            .retain(|k, _| k.0.instance_key != instance.instance_key);
    }

    pub fn get_glyph_index(&mut self, font_key: FontKey, ch: char) -> Option<u32> {
        match self.fonts.get(&font_key) {
            None => None,
            Some(font) => {
                let index: u32 = font.charmap().map(ch).into();
                return Some(index);
            }
        }
    }

    pub fn get_glyph_dimensions(
        &mut self,
        instance: &FontInstance,
        key: &GlyphKey,
    ) -> Option<GlyphDimensions> {
        let size = FontSize::from_f64_px(instance.get_transformed_size());
        if let Some(GlyphImage {
            placement:
                Placement {
                    left,
                    top,
                    width,
                    height,
                },
            ..
        }) = self.get_or_create_cache(instance, key)
        {
            if let Some(font) = self.fonts.get(&instance.font_key) {
                let advance = font
                    .as_ref()
                    .glyph_metrics(&[])
                    .scale(size.to_f32_px())
                    .advance_width(key.index() as GlyphId);
                return Some(GlyphDimensions {
                    left: left as i32,
                    top: top as i32,
                    width: width as i32,
                    height: height as i32,
                    advance,
                });
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn prepare_font(font: &mut FontInstance) {
        match font.render_mode {
            FontRenderMode::Mono => {
                // In mono mode the color of the font is irrelevant.
                font.color = ColorU::new(0xFF, 0xFF, 0xFF, 0xFF);
                // Subpixel positioning is disabled in mono mode.
                font.disable_subpixel_position();
            }
            FontRenderMode::Alpha | FontRenderMode::Subpixel => {
                // We don't do any preblending with FreeType currently, so the color is not used.
                font.color = ColorU::new(0xFF, 0xFF, 0xFF, 0xFF);
            }
        }
    }

    pub fn begin_rasterize(_: &FontInstance) {}

    pub fn end_rasterize(_: &FontInstance) {}

    /// Create a swash Image from a cache key, caching results
    pub fn get_or_create_cache(
        &mut self,
        instance: &FontInstance,
        glyph_key: &GlyphKey,
    ) -> Option<GlyphImage> {
        match self.cache.entry((instance.clone(), glyph_key.clone())) {
            Entry::Occupied(entry) => Some(entry.get().clone()),
            Entry::Vacant(entry) => {
                let font = self.fonts.get(&instance.font_key).unwrap();
                if let Some(glyph) =
                    render_glyph(&mut self.scale_context, &font.as_ref(), instance, glyph_key)
                {
                    entry.insert(glyph.clone());
                    return Some(glyph);
                } else {
                    return None;
                }
            }
        }
    }
    pub fn rasterize_glyph(
        &mut self,
        instance: &FontInstance,
        glyph_key: &GlyphKey,
    ) -> GlyphRasterResult {
        info!("todo");
        let glyph_image = self.get_or_create_cache(instance, glyph_key);

        if glyph_image.is_none() {
            return Err(GlyphRasterError::LoadFailed);
        }

        let GlyphImage {
            placement:
                Placement {
                    left,
                    top,
                    width,
                    height,
                },
            data: pixels,
            content,
            scale,
            ..
        } = glyph_image.unwrap();

        // Alpha texture bounds can sometimes return an empty rect
        // Such as for spaces
        if width == 0 || height == 0 {
            return Err(GlyphRasterError::LoadFailed);
        }

        let bgra_pixels = match content {
            Content::Color | Content::SubpixelMask => {
                assert!(width * height * 4 == pixels.len() as u32);
                // let _ = image::RgbaImage::from_raw(width, height, pixels.clone()).unwrap().save("/tmp/emoji_".to_string() + glyph_key.index().to_string().as_str() + ".png");
                let subpixel_bgr = instance.flags.contains(FontInstanceFlags::SUBPIXEL_BGR);
                pixels
                    .chunks_exact(4)
                    .flat_map(|src| {
                        let (mut r, g, mut b, a) = (src[0], src[1], src[2], src[3]);
                        if subpixel_bgr {
                            mem::swap(&mut r, &mut b);
                        }
                        [b, g, r, a]
                    })
                    .collect()
            }
            Content::Mask => pixels
                .chunks_exact(1)
                .flat_map(|src| [src[0], src[0], src[0], src[0]])
                .collect(),
        };

        let format = match content {
            Content::Mask => instance.get_alpha_glyph_format(),
            Content::SubpixelMask => instance.get_subpixel_glyph_format(),
            Content::Color => GlyphFormat::ColorBitmap,
        };

        Ok(RasterizedGlyph {
            left: left as f32,
            top: top as f32,
            width: width as i32,
            height: height as i32,
            scale,
            format,
            bytes: bgra_pixels,
        })
    }
}

fn render_glyph(
    context: &mut ScaleContext,
    font: &FontRef,
    instance: &FontInstance,
    glyph_key: &GlyphKey,
) -> Option<GlyphImage> {
    use zeno::{Format, Vector};
    let (x_scale, y_scale) = instance.transform.compute_scale().unwrap_or((1.0, 1.0));
    let size = instance.size.to_f32_px() * y_scale as f32;

    // Transform
    let (mut _transform, (x_offset, y_offset)) = if is_bitmap_font(instance) {
        (FontTransform::identity(), (0.0, 0.0))
    } else {
        (
            instance.transform.invert_scale(y_scale, y_scale),
            instance.get_subpx_offset(glyph_key),
        )
    };

    // if instance.flags.contains(FontInstanceFlags::FLIP_X) {
    //     transform = transform.flip_x();
    // }
    // if instance.flags.contains(FontInstanceFlags::FLIP_Y) {
    //     transform = transform.flip_y();
    // }
    // if instance.flags.contains(FontInstanceFlags::TRANSPOSE) {
    //     transform = transform.swap_xy();
    // }

    // let (transform, (tx, ty)) = if instance.synthetic_italics.is_enabled() {
    //     instance.synthesize_italics(transform, size as f64)
    // } else {
    //     (transform, (0.0, 0.0))
    // };

    // Strike
    let (strike_scale, _pixel_step) = if is_bitmap_font(instance) {
        (y_scale, 1.0)
    } else {
        (x_scale, y_scale / x_scale)
    };
    let _extra_strikes = instance.get_extra_strikes(
        FontInstanceFlags::SYNTHETIC_BOLD | FontInstanceFlags::MULTISTRIKE_BOLD,
        strike_scale,
    );

    let format = match instance.render_mode {
        FontRenderMode::Mono | FontRenderMode::Alpha => Format::Alpha,
        FontRenderMode::Subpixel => Format::Subpixel,
    };

    // let format = Format::CustomSubpixel([0.3, 0., -0.3]);

    // TODO transform/strike
    // check Render's embolden/style/transform
    // is embolden strike?

    // Build the scaler
    let mut scaler = context
        .builder(*font)
        .size(size)
        .hint(cfg!(not(target_os = "macos")))
        // .variations(instance.variations.clone())
        .build();
    // Compute the fractional offset-- you'll likely want to quantize this
    // in a real renderer
    let offset = Vector::new((x_offset as f32).fract(), (y_offset as f32).fract());
    let embolden = if cfg!(target_os = "macos") { 0.25 } else { 0. };
    // Select our source order
    Render::new(&[
        Source::ColorOutline(0),
        Source::ColorBitmap(StrikeWith::BestFit),
        Source::Outline,
    ])
    // Select a subpixel format
    .format(format)
    // Apply the fractional offset
    .offset(offset)
    .embolden(embolden)
    .default_color([
        instance.color.r,
        instance.color.g,
        instance.color.b,
        instance.color.a,
    ])
    // Render the image
    .render(&mut scaler, glyph_key.index() as GlyphId)
}
