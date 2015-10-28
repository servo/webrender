use fnv::FnvHasher;
use internal_types::{FontTemplate, ImageResource, GlyphKey, RasterItem};
use std::collections::HashMap;
use std::collections::hash_state::DefaultState;
use types::{FontKey, ImageID};

pub struct ResourceCache {
    cached_glyphs: HashMap<GlyphKey, ImageID, DefaultState<FnvHasher>>,
    cached_rasters: HashMap<RasterItem, ImageID, DefaultState<FnvHasher>>,

    font_templates: HashMap<FontKey, FontTemplate, DefaultState<FnvHasher>>,
    image_templates: HashMap<ImageID, ImageResource, DefaultState<FnvHasher>>,
}

impl ResourceCache {
    pub fn new() -> ResourceCache {
        ResourceCache {
            cached_glyphs: HashMap::with_hash_state(Default::default()),
            cached_rasters: HashMap::with_hash_state(Default::default()),
            font_templates: HashMap::with_hash_state(Default::default()),
            image_templates: HashMap::with_hash_state(Default::default()),
        }
    }

    pub fn add_font(&mut self, font_key: FontKey, template: FontTemplate) {
        self.font_templates.insert(font_key, template);
    }

    pub fn add_image(&mut self, image_id: ImageID, resource: ImageResource) {
        self.image_templates.insert(image_id, resource);
    }

    pub fn get_font(&self, font_key: FontKey) -> &FontTemplate {
        &self.font_templates[&font_key]
    }

    pub fn get_image(&self, image_id: ImageID) -> &ImageResource {
        &self.image_templates[&image_id]
    }

    pub fn cache_raster_if_required<F>(&mut self,
                                       raster_item: &RasterItem,
                                       mut f: F) where F: FnMut() -> ImageID {
        if !self.cached_rasters.contains_key(raster_item) {
            let image_id = f();
            self.cached_rasters.insert(raster_item.clone(), image_id);
        }
    }

    pub fn cache_glyph_if_required<F>(&mut self,
                                      glyph_key: &GlyphKey,
                                      mut f: F) where F: FnMut() -> ImageID {
        if !self.cached_glyphs.contains_key(glyph_key) {
            let image_id = f();
            self.cached_glyphs.insert(glyph_key.clone(), image_id);
        }
    }

    pub fn get_glyph(&self, glyph_key: &GlyphKey) -> ImageID {
        self.cached_glyphs[glyph_key]
    }

    pub fn get_raster(&self, raster_item: &RasterItem) -> ImageID {
        self.cached_rasters[raster_item]
    }
}