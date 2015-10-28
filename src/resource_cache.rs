use fnv::FnvHasher;
use internal_types::{FontTemplate, ImageResource, GlyphKey, RasterItem};
use std::collections::HashMap;
use std::collections::hash_state::DefaultState;
use texture_cache::TextureCacheItemId;
use types::{FontKey, ImageKey};

pub struct ResourceCache {
    cached_glyphs: HashMap<GlyphKey, TextureCacheItemId, DefaultState<FnvHasher>>,
    cached_rasters: HashMap<RasterItem, TextureCacheItemId, DefaultState<FnvHasher>>,
    cached_images: HashMap<ImageKey, TextureCacheItemId, DefaultState<FnvHasher>>,

    font_templates: HashMap<FontKey, FontTemplate, DefaultState<FnvHasher>>,
    image_templates: HashMap<ImageKey, ImageResource, DefaultState<FnvHasher>>,
}

impl ResourceCache {
    pub fn new() -> ResourceCache {
        ResourceCache {
            cached_glyphs: HashMap::with_hash_state(Default::default()),
            cached_rasters: HashMap::with_hash_state(Default::default()),
            cached_images: HashMap::with_hash_state(Default::default()),
            font_templates: HashMap::with_hash_state(Default::default()),
            image_templates: HashMap::with_hash_state(Default::default()),
        }
    }

    pub fn add_font_template(&mut self, font_key: FontKey, template: FontTemplate) {
        self.font_templates.insert(font_key, template);
    }

    pub fn add_image_template(&mut self, image_key: ImageKey, resource: ImageResource) {
        self.image_templates.insert(image_key, resource);
    }

    pub fn get_font_template(&self, font_key: FontKey) -> &FontTemplate {
        &self.font_templates[&font_key]
    }

    pub fn get_image_template(&self, key: ImageKey) -> &ImageResource {
        &self.image_templates[&key]
    }

    pub fn cache_raster_if_required<F>(&mut self,
                                       raster_item: &RasterItem,
                                       mut f: F) where F: FnMut() -> TextureCacheItemId {
        if !self.cached_rasters.contains_key(raster_item) {
            let image_id = f();
            self.cached_rasters.insert(raster_item.clone(), image_id);
        }
    }

    pub fn cache_glyph_if_required<F>(&mut self,
                                      glyph_key: &GlyphKey,
                                      mut f: F) where F: FnMut() -> TextureCacheItemId {
        if !self.cached_glyphs.contains_key(glyph_key) {
            let image_id = f();
            self.cached_glyphs.insert(glyph_key.clone(), image_id);
        }
    }

    pub fn cache_image_if_required<F>(&mut self,
                                      image_key: ImageKey,
                                      mut f: F) where F: FnMut(&ImageResource) -> TextureCacheItemId {
        if !self.cached_images.contains_key(&image_key) {
            let image_id = {
                let image_template = self.get_image_template(image_key);
                f(image_template)
            };
            self.cached_images.insert(image_key, image_id);
        }
    }

    pub fn get_glyph(&self, glyph_key: &GlyphKey) -> TextureCacheItemId {
        self.cached_glyphs[glyph_key]
    }

    pub fn get_raster(&self, raster_item: &RasterItem) -> TextureCacheItemId {
        self.cached_rasters[raster_item]
    }

    pub fn get_image(&self, image_key: ImageKey) -> TextureCacheItemId {
        self.cached_images[&image_key]
    }
}
