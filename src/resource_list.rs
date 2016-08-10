/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use fnv::FnvHasher;
use internal_types::{Glyph, GlyphKey, RasterItem};
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;
use webrender_traits::{FontKey, ImageKey, ImageRendering};

type RequiredImageSet = HashSet<(ImageKey, ImageRendering), BuildHasherDefault<FnvHasher>>;
type RequiredGlyphMap = HashMap<FontKey,
                                HashSet<Glyph, BuildHasherDefault<FnvHasher>>,
                                BuildHasherDefault<FnvHasher>>;
type RequiredRasterSet = HashSet<RasterItem, BuildHasherDefault<FnvHasher>>;

pub struct ResourceList {
    required_images: RequiredImageSet,
    required_glyphs: RequiredGlyphMap,
    required_rasters: RequiredRasterSet,
}

impl ResourceList {
    pub fn new() -> ResourceList {
        ResourceList {
            required_glyphs: HashMap::with_hasher(Default::default()),
            required_images: HashSet::with_hasher(Default::default()),
            required_rasters: HashSet::with_hasher(Default::default()),
        }
    }

    pub fn add_image(&mut self,
                     key: ImageKey,
                     image_rendering: ImageRendering) {
        self.required_images.insert((key, image_rendering));
    }

    pub fn add_glyph(&mut self, font_key: FontKey, glyph: Glyph) {
        self.required_glyphs.entry(font_key)
                            .or_insert_with(|| HashSet::with_hasher(Default::default()))
                            .insert(glyph);
    }

    pub fn for_each_image<F>(&self, mut f: F) where F: FnMut(ImageKey, ImageRendering) {
        for &(image_id, image_rendering) in &self.required_images {
            f(image_id, image_rendering);
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
