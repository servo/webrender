use std::{path::PathBuf, sync::Arc};
use std::collections::HashMap;
use api::{FontKey, GlyphDimensions, NativeFontHandle};
use memmap2::Mmap;
use skrifa::charmap::Charmap;
use skrifa::instance::Location;
use skrifa::metrics::GlyphMetrics;
use skrifa::prelude::{LocationRef, Size};
use skrifa::raw::{FileRef};
use skrifa::GlyphId;

use crate::{FontInstance, GlyphKey, GlyphRasterResult, RasterizedGlyph};

struct CachedFont {
    pub data: Arc<dyn AsRef<[u8]> + Send + Sync>,
    pub index: u32,
    pub settings: FontRenderSettings,
}

#[derive(Default)]
struct FontRenderSettings {
    pub font_size: f32,
}

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
    font_cache: HashMap<FontKey, CachedFont>,
}

impl FontContext {
    // Static fns
    pub fn distribute_across_threads() -> bool {
        true
    }
    pub fn new() -> FontContext {
        FontContext {
            font_cache: HashMap::new(),
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
    pub fn delete_font_instance(&mut self, _instance: &FontInstance) {
        // Do these pair?
    }

    pub fn add_raw_font(&mut self, font_key: &FontKey, bytes: Arc<Vec<u8>>, index: u32) {
        let font = CachedFont {
            data: bytes,
            index: index,
            settings: FontRenderSettings::default(),
        };
        self.font_cache.insert(*font_key, font);
    }
    pub fn add_native_font(&mut self, font_key: &FontKey, native_font_handle: NativeFontHandle) {
        let handle = SimpleFontHandle::from(native_font_handle);
        let font = CachedFont {
            data: Arc::new(handle.data),
            index: handle.index,
            settings: FontRenderSettings::default(),
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
        font: &FontInstance,
        key: &GlyphKey,
    ) -> Option<GlyphDimensions> {
        let font = self.font_cache.get(&font.font_key)?;
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

        let font_size = font.settings.font_size;
        let glyph_metrics = GlyphMetrics::new(&font_ref, Size::new(font_size), location_ref);
        let advance = glyph_metrics.advance_width(GlyphId::new(key.index()))?;
        let bounds = glyph_metrics.bounds(GlyphId::new(key.index()))?;

        Some(GlyphDimensions {
            advance,

            // TODO: investigate why Skrifa provides f32 but WebRender expects i32
            // TODO: use hinted metrics
            left: bounds.x_min as i32,
            top: bounds.y_min as i32,
            width: (bounds.x_max - bounds.x_min) as i32,
            height: (bounds.y_max - bounds.y_min) as i32,
        })
    }
    pub fn rasterize_glyph(&mut self, font: &FontInstance, key: &GlyphKey) -> GlyphRasterResult {
        // todo!()

        Ok(RasterizedGlyph {
            top: (),
            left: (),
            width: (),
            height: (),
            scale: (),
            format: (),
            bytes: (),
        })
    }
}

/// CoreText font enumaration gives us a postscript name rather than an index.
/// This functions maps from postscript name to index
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn ttc_index_from_postscript_name(font_file: FileRef<'_>, postscript_name: &str) -> u32 {
    use skrifa::raw::{FileRef, TableProvider as _};
    use skrifa::raw::types::NameId;

    let mut name_buf = String::with_capacity(100);
    let index = match font_file {
        FileRef::Font(_) => 0,
        FileRef::Collection(collection) => 'idx: {
            for i in 0..collection.len() {
                let font = collection.get(i).unwrap();
                let name_table = font.name().unwrap();
                if name_table
                    .name_record()
                    .iter()
                    .filter(|record| record.name_id() == NameId::POSTSCRIPT_NAME)
                    .any(|record| {
                        name_buf.clear();
                        record
                            .string(name_table.string_data())
                            .unwrap()
                            .chars()
                            .for_each(|c| name_buf.push(c));
                        &name_buf == &postscript_name
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
