use app_units::Au;
use euclid::Size2D;
use fnv::FnvHasher;
use internal_types::GlyphKey;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::hash_state::DefaultState;
use string_cache::Atom;
use types::{BorderRadiusRasterOp, BoxShadowCornerRasterOp};
use types::{Glyph, ImageFormat, ImageID, RasterItem};

type RequiredImageSet = HashSet<ImageID, DefaultState<FnvHasher>>;
type RequiredGlyphMap = HashMap<Atom, HashSet<Glyph>, DefaultState<FnvHasher>>;
type RequiredRasterSet = HashSet<RasterItem, DefaultState<FnvHasher>>;

pub struct ResourceList {
    required_images: RequiredImageSet,
    required_glyphs: RequiredGlyphMap,
    required_rasters: RequiredRasterSet,
}

impl ResourceList {
    pub fn new() -> ResourceList {
        ResourceList {
            required_glyphs: HashMap::with_hash_state(Default::default()),
            required_images: HashSet::with_hash_state(Default::default()),
            required_rasters: HashSet::with_hash_state(Default::default()),
        }
    }

    pub fn add_image(&mut self, image_id: ImageID) {
        self.required_images.insert(image_id);
    }

    pub fn add_glyph(&mut self, font_id: Atom, glyph: Glyph) {
        match self.required_glyphs.entry(font_id) {
            Occupied(entry) => {
                entry.into_mut().insert(glyph);
            }
            Vacant(entry) => {
                let mut hash_set = HashSet::new();
                hash_set.insert(glyph);
                entry.insert(hash_set);
            }
        }
    }

    pub fn add_radius_raster(&mut self,
                             outer_radius: &Size2D<f32>,
                             inner_radius: &Size2D<f32>,
                             inverted: bool,
                             image_format: ImageFormat) {
        if let Some(raster_item) = BorderRadiusRasterOp::create(outer_radius,
                                                                inner_radius,
                                                                inverted,
                                                                image_format) {
            self.required_rasters.insert(RasterItem::BorderRadius(raster_item));
        }
    }

    pub fn add_box_shadow_corner(&mut self,
                                 blur_radius: f32,
                                 border_radius: f32,
                                 inverted: bool) {
        if let Some(raster_item) = BoxShadowCornerRasterOp::create(blur_radius,
                                                                   border_radius,
                                                                   inverted) {
            self.required_rasters.insert(RasterItem::BoxShadowCorner(raster_item));
        }
    }

    pub fn for_each_image<F>(&self, mut f: F) where F: FnMut(ImageID) {
        for image_id in &self.required_images {
            f(*image_id);
        }
    }

    pub fn for_each_raster<F>(&self, mut f: F) where F: FnMut(&RasterItem) {
        for raster_item in &self.required_rasters {
            f(raster_item);
        }
    }

    pub fn for_each_glyph<F>(&self, mut f: F) where F: FnMut(&GlyphKey) {
        for (font_id, glyphs) in &self.required_glyphs {
            let mut glyph_key = GlyphKey::new(font_id.clone(), Au(0), Au(0), 0);

            for glyph in glyphs {
                glyph_key.size = glyph.size;
                glyph_key.index = glyph.index;
                glyph_key.blur_radius = glyph.blur_radius;

                f(&glyph_key);
            }
        }
    }
}
