use app_units::Au;
use euclid::Size2D;
use fnv::FnvHasher;
use internal_types::{BorderRadiusRasterOp, BoxShadowRasterOp};
use internal_types::{Glyph, GlyphKey, RasterItem, TiledImageKey};
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::hash_state::DefaultState;
use types::{BorderRadius, FontKey, ImageFormat, ImageKey};

type RequiredImageSet = HashSet<ImageKey, DefaultState<FnvHasher>>;
type RequiredGlyphMap = HashMap<FontKey, HashSet<Glyph>, DefaultState<FnvHasher>>;
type RequiredRasterSet = HashSet<RasterItem, DefaultState<FnvHasher>>;
type RequiredTiledImageSet = HashSet<TiledImageKey, DefaultState<FnvHasher>>;

/// The number of repeats of an image we allow within the viewport before we add a tile
/// rasterization op.
const MAX_IMAGE_REPEATS: u32 = 64;

/// The dimensions (horizontal and vertical) of the area that we tile an image to.
const TILE_SIZE: u32 = 128;

/// The size of the virtual viewport used to estimate the number of image repeats we'll have to
/// display.
const APPROXIMATE_VIEWPORT_SIZE: u32 = 1024;

pub struct ResourceList {
    required_images: RequiredImageSet,
    required_glyphs: RequiredGlyphMap,
    required_rasters: RequiredRasterSet,
    required_tiled_images: RequiredTiledImageSet,
}

impl ResourceList {
    pub fn new() -> ResourceList {
        ResourceList {
            required_glyphs: HashMap::with_hash_state(Default::default()),
            required_images: HashSet::with_hash_state(Default::default()),
            required_rasters: HashSet::with_hash_state(Default::default()),
            required_tiled_images: HashSet::with_hash_state(Default::default()),
        }
    }

    pub fn add_image(&mut self,
                     key: ImageKey,
                     tiled_size: &Size2D<f32>,
                     stretch_size: &Size2D<f32>) {
        self.required_images.insert(key);
        self.add_tiled_image(key, tiled_size, stretch_size);
    }

    pub fn add_glyph(&mut self, font_key: FontKey, glyph: Glyph) {
        match self.required_glyphs.entry(font_key) {
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

    pub fn add_radius_raster_for_border_radii(&mut self, radii: &BorderRadius) {
        let zero_size = Size2D::new(0.0, 0.0);
        self.add_radius_raster(&radii.top_left, &zero_size, false, ImageFormat::A8);
        self.add_radius_raster(&radii.top_right, &zero_size, false, ImageFormat::A8);
        self.add_radius_raster(&radii.bottom_left, &zero_size, false, ImageFormat::A8);
        self.add_radius_raster(&radii.bottom_right, &zero_size, false, ImageFormat::A8);
    }

    pub fn add_box_shadow_corner(&mut self, blur_radius: f32, border_radius: f32, inverted: bool) {
        if let Some(raster_item) = BoxShadowRasterOp::create_corner(blur_radius,
                                                                    border_radius,
                                                                    inverted) {
            self.required_rasters.insert(RasterItem::BoxShadow(raster_item));
        }
    }

    pub fn add_box_shadow_edge(&mut self, blur_radius: f32, border_radius: f32, inverted: bool) {
        if let Some(raster_item) = BoxShadowRasterOp::create_edge(blur_radius,
                                                                  border_radius,
                                                                  inverted) {
            self.required_rasters.insert(RasterItem::BoxShadow(raster_item));
        }
    }

    pub fn add_tiled_image(&mut self,
                           image_key: ImageKey,
                           tiled_size: &Size2D<f32>,
                           stretch_size: &Size2D<f32>) {
        if let Some(tiled_image_op) = TiledImageKey::create_if_necessary(image_key,
                                                                         tiled_size,
                                                                         stretch_size) {
            self.required_tiled_images.insert(tiled_image_op);
        }
    }

    pub fn for_each_image<F>(&self, mut f: F) where F: FnMut(ImageKey) {
        for image_id in &self.required_images {
            f(*image_id);
        }
    }

    pub fn for_each_raster<F>(&self, mut f: F) where F: FnMut(&RasterItem) {
        for raster_item in &self.required_rasters {
            f(raster_item);
        }
    }

    pub fn for_each_tiled_image<F>(&self, mut f: F) where F: FnMut(&TiledImageKey) {
        for tiled_image_key in &self.required_tiled_images {
            f(tiled_image_key);
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

impl TiledImageKey {
    pub fn create_if_necessary(image_key: ImageKey,
                               tiled_size: &Size2D<f32>,
                               stretch_size: &Size2D<f32>)
                               -> Option<TiledImageKey> {
        let tiled_size = Size2D::new(tiled_size.width.min(APPROXIMATE_VIEWPORT_SIZE as f32),
                                     tiled_size.height.min(APPROXIMATE_VIEWPORT_SIZE as f32));
        let image_repeats = ((tiled_size.width / stretch_size.width).ceil() *
                (tiled_size.height / stretch_size.height).ceil()) as u32;
        if image_repeats <= MAX_IMAGE_REPEATS {
            return None
        }
        let prerendered_tile_size = Size2D::new(
            (((TILE_SIZE as f32) / stretch_size.width).ceil() * stretch_size.width) as u32,
            (((TILE_SIZE as f32) / stretch_size.height).ceil() * stretch_size.height) as u32);
        Some(TiledImageKey {
            image_key: image_key,
            tiled_width: prerendered_tile_size.width,
            tiled_height: prerendered_tile_size.height,
            stretch_width: stretch_size.width as u32,
            stretch_height: stretch_size.height as u32,
        })
    }
}

