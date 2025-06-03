use std::{path::PathBuf, sync::Arc};
use api::{FontInstanceKey, FontKey, GlyphDimensions, NativeFontHandle};
use memmap2::Mmap;
use skrifa::charmap::Charmap;
use skrifa::instance::Location;
use skrifa::metrics::GlyphMetrics;
use skrifa::outline::{HintingInstance};
use skrifa::prelude::{LocationRef, Size};
use skrifa::raw::{FileRef};
use skrifa::{GlyphId};
use vello_cpu::peniko::{self, Blob};
use vello_cpu::{Level, PaintType, Pixmap, RenderContext, RenderMode, RenderSettings};

use crate::{
    FastHashMap, FontInstance, GlyphFormat, GlyphKey, GlyphRasterError, GlyphRasterResult,
    RasterizedGlyph,
};

type PenikoFont = vello_cpu::peniko::Font;

// struct CachedFont {
//     pub data: Arc<dyn AsRef<[u8]> + Send + Sync>,
//     pub index: u32,
//     pub settings: FontRenderSettings,
// }

// #[derive(Default)]
// struct FontRenderSettings {
//     pub font_size: f32,
// }

#[cfg(any(target_os = "macos", target_os = "ios"))]
struct SimpleFontHandle {
    pub path: PathBuf,
    pub index: u32,
    pub data: Mmap,
}

impl SimpleFontHandle {
    pub(crate) fn new(path: PathBuf, index: u32) -> Self {
        let file = std::fs::File::open(&path).unwrap();
        let mapped = unsafe { Mmap::map(&file) }.unwrap();
        Self {
            path,
            index,
            data: mapped,
        }
    }
}

impl From<NativeFontHandle> for SimpleFontHandle {
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    fn from(native: NativeFontHandle) -> Self {
        let file = std::fs::File::open(&native.path).unwrap();
        let mapped = unsafe { Mmap::map(&file) }.unwrap();

        SimpleFontHandle {
            path: native.path,
            index: native.index,
            data: mapped,
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn from(native: NativeFontHandle) -> Self {
        let file = std::fs::File::open(&native.path).unwrap();
        let mapped = unsafe { Mmap::map(&file) }.unwrap();
        let font_file = FileRef::new(&mapped).unwrap();
        let index = ttc_index_from_postscript_name(font_file, &native.name);

        SimpleFontHandle {
            path: PathBuf::from(native.path),
            index,
            data: mapped,
        }
    }
}

trait RasterContext {
    // Static fns
    fn distribute_across_threads() -> bool;
    fn new() -> Self;
    fn begin_rasterize(font: &FontInstance);
    fn end_rasterize(font: &FontInstance);
    fn prepare_font(font: &mut FontInstance);

    // Font registration
    fn add_raw_font(&mut self, font_key: &FontKey, bytes: Arc<Vec<u8>>, index: u32);
    fn add_native_font(&mut self, font_key: &FontKey, native_font_handle: NativeFontHandle);
    fn delete_font(&mut self, font_key: &FontKey);
    fn delete_font_instance(&mut self, instance: &FontInstance);

    // Methods
    fn get_glyph_index(&mut self, font_key: FontKey, ch: char) -> Option<u32>;
    fn get_glyph_dimensions(
        &mut self,
        font: &FontInstance,
        key: &GlyphKey,
    ) -> Option<GlyphDimensions>;
    fn rasterize_glyph(&mut self, font: &FontInstance, key: &GlyphKey) -> GlyphRasterResult;
}

pub struct FontContext {
    font_cache: FastHashMap<FontKey, PenikoFont>,
    hinting_instance_cache: FastHashMap<FontInstanceKey, HintingInstance>,
}

impl FontContext {
    // Static fns
    pub fn distribute_across_threads() -> bool {
        true
    }
    pub fn new() -> FontContext {
        FontContext {
            font_cache: Default::default(),
            hinting_instance_cache: Default::default(),
        }
    }
    pub fn begin_rasterize(font: &FontInstance) {
        // TODO: apply instance properties
    }
    pub fn end_rasterize(font: &FontInstance) {
        // TODO: clear instance properties
    }

    pub fn prepare_font(font: &mut FontInstance) {
        // Perhaps not needed for fontations?
    }
    pub fn delete_font_instance(&mut self, instance: &FontInstance) {
        self.hinting_instance_cache
            .remove(&instance.base.instance_key);
    }

    pub fn add_raw_font(&mut self, font_key: &FontKey, bytes: Arc<Vec<u8>>, index: u32) {
        let font = PenikoFont {
            data: Blob::new(bytes),
            index: index,
        };
        self.font_cache.insert(*font_key, font);
    }
    pub fn add_native_font(&mut self, font_key: &FontKey, native_font_handle: NativeFontHandle) {
        let handle = SimpleFontHandle::from(native_font_handle);
        let font = PenikoFont {
            data: Blob::new(Arc::new(handle.data)),
            index: handle.index,
        };
        self.font_cache.insert(*font_key, font);
    }
    pub fn delete_font(&mut self, font_key: &FontKey) {
        self.font_cache.remove(font_key);
    }

    // Methods
    pub fn get_glyph_index(&mut self, font_key: FontKey, ch: char) -> Option<u32> {
        let font = self.font_cache.get(&font_key)?;
        let data: &[u8] = font.data.as_ref().as_ref();
        let font_ref = skrifa::FontRef::from_index(data, font.index).ok()?;

        let char_map = Charmap::new(&font_ref);
        let glyph_id = char_map.map(ch)?;

        Some(glyph_id.to_u32())
    }

    pub fn get_glyph_dimensions(
        &mut self,
        font_instance: &FontInstance,
        key: &GlyphKey,
    ) -> Option<GlyphDimensions> {
        let font = self.font_cache.get(&font_instance.font_key)?;
        let data: &[u8] = font.data.as_ref().as_ref();
        let font_ref = skrifa::FontRef::from_index(data, font.index).ok()?;

        // TODO: set variation axis
        //
        // let location = font_ref.axes().location(
        //     variations
        //         .iter()
        //         .map(|v| (Tag::new(&v.tag.to_le_bytes()), v.value)),
        // );
        // let location_ref = LocationRef::from(&location);
        let location = Location::new(0);
        let location_ref = LocationRef::from(&location);

        let font_size = font_instance.size.to_f32_px();
        let (x_scale, y_scale) = font_instance
            .transform
            .compute_scale()
            .unwrap_or((1.0, 1.0));
        let scale = ((x_scale + y_scale) / 2.0) as f32;
        let font_size = font_size * scale;

        let glyph_metrics = GlyphMetrics::new(&font_ref, Size::new(font_size), location_ref);
        let advance = glyph_metrics.advance_width(GlyphId::new(key.index()))?;
        let bounds = glyph_metrics.bounds(GlyphId::new(key.index()))?;

        dbg!(font_size);
        dbg!(bounds);
        dbg!(x_scale, y_scale);
        dbg!(font_instance.flags);

        Some(GlyphDimensions {
            advance,

            // TODO: investigate why Skrifa provides f32 but WebRender expects i32
            // TODO: use hinted metrics
            left: bounds.x_min.floor() as i32,
            top: bounds.y_max.ceil() as i32,
            width: (bounds.x_max - bounds.x_min).ceil() as i32,
            height: (bounds.y_max - bounds.y_min).ceil() as i32,
        })
    }
    pub fn rasterize_glyph(
        &mut self,
        font_instance: &FontInstance,
        key: &GlyphKey,
    ) -> GlyphRasterResult {
        let dimensions = self
            .get_glyph_dimensions(font_instance, key)
            .ok_or(GlyphRasterError::LoadFailed)?;
        let font = self
            .font_cache
            .get(&font_instance.font_key)
            .ok_or(GlyphRasterError::LoadFailed)?;

        // Handle zero-sized glyphs (e.g. space chars)
        if dimensions.width == 0 || dimensions.height == 0 {
            return Err(GlyphRasterError::LoadFailed);
        }

        // let data: &[u8] = font.data.as_ref().as_ref();
        // let font_ref = skrifa::FontRef::from_index(data, font.index).ok().unwrap();

        // let hinting_instance = self
        //     .hinting_instance_cache
        //     .entry(font_instance.instance_key)
        //     .or_insert_with(|| {
        //         let outline_glyphs = font_ref.outline_glyphs();
        //         let size = skrifa::instance::Size::new(font_instance.size.to_f32_px());
        //         let location_ref = LocationRef::default();
        //         let options = HintingOptions::default();
        //         HintingInstance::new(&outline_glyphs, size, location_ref, options).unwrap()
        //     });
        // let draw_settings = DrawSettings::hinted(hinting_instance, false);

        let width = (dimensions.left + dimensions.width) as u16;
        let height = dimensions.height as u16;

        // TODO: reuse RenderContext between glyphs
        let render_settings = RenderSettings {
            render_mode: RenderMode::OptimizeSpeed,
            num_threads: 1,
            level: Level::new(),
        };
        let mut render_context = RenderContext::new_with(width, height, render_settings);
        let color = peniko::Color::from_rgba8(
            font_instance.color.r,
            font_instance.color.g,
            font_instance.color.b,
            font_instance.color.a,
        );
        render_context.set_paint(PaintType::Solid(color));

        let font_size = font_instance.size.to_f32_px();
        let (x_scale, y_scale) = font_instance
            .transform
            .compute_scale()
            .unwrap_or((1.0, 1.0));
        let scale = ((x_scale + y_scale) / 2.0) as f32;
        let font_size = font_size * scale;

        render_context
            .glyph_run(&font)
            .font_size(font_size)
            .hint(true)
            .fill_glyphs(std::iter::once(vello_cpu::Glyph {
                x: 0.0,
                y: dimensions.top as f32 + 1.0,
                id: key.index(),
            }));
        render_context.flush();

        let mut buffer = vec![0; width as usize * height as usize * 4];
        render_context.render_to_buffer(&mut buffer, width, height, RenderMode::OptimizeSpeed);

        // DEBUG: Write out PNG file of glyphs to $CWD/glyphs/glyph_id.png
        //
        // let mut pixmap = Pixmap::new(width, height);
        // render_context.render_to_pixmap(&mut pixmap);
        // let buffer = pixmap.data_as_u8_slice().to_vec();
        // let png = pixmap.into_png().unwrap();
        // std::fs::write(format!("./glyphs/{}.png", key.index()), png).unwrap();

        Ok(RasterizedGlyph {
            top: dimensions.top as f32,
            left: dimensions.left as f32,
            width: width as i32,
            height: height as i32,
            scale: 1.0 / scale,
            format: GlyphFormat::Alpha,
            bytes: buffer,
        })
    }
}

/// CoreText font enumaration gives us a postscript name rather than an index.
/// This functions maps from postscript name to index
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn ttc_index_from_postscript_name(font_file: FileRef<'_>, postscript_name: &str) -> u32 {
    use skrifa::raw::{FileRef, TableProvider as _};
    use skrifa::raw::types::NameId;

    let index = match font_file {
        FileRef::Font(_) => 0,
        FileRef::Collection(collection) => 'idx: {
            for i in 0 .. collection.len() {
                let font = collection.get(i).unwrap();
                let name_table = font.name().unwrap();
                if name_table
                    .name_record()
                    .iter()
                    .filter(|record| record.name_id() == NameId::POSTSCRIPT_NAME)
                    .any(|record| {
                        record
                            .string(name_table.string_data())
                            .unwrap()
                            .chars()
                            .eq(postscript_name.chars())
                    })
                {
                    break 'idx i;
                }
            }

            panic!(
                "Font with postscript_name {} not found in collection",
                postscript_name
            );
        }
    };

    index
}
