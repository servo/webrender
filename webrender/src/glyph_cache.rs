/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{FontInstanceKey, GlyphKey};
use frame::FrameId;
use internal_types::FastHashMap;
use resource_cache::{Resource, ResourceClassCache};
use texture_cache::{TextureCache, TextureCacheItemId};

pub struct CachedGlyphInfo {
    pub texture_cache_id: Option<TextureCacheItemId>,
    pub last_access: FrameId,
}

impl Resource for CachedGlyphInfo {
    fn free(&self, texture_cache: &mut TextureCache) {
        if let Some(id) = self.texture_cache_id {
            texture_cache.free(id);
        }
    }
    fn get_last_access_time(&self) -> FrameId {
        self.last_access
    }
    fn set_last_access_time(&mut self, frame_id: FrameId) {
        self.last_access = frame_id;
    }
}

pub type GlyphKeyCache = ResourceClassCache<GlyphKey, CachedGlyphInfo>;

pub struct GlyphCache {
    pub glyph_key_caches: FastHashMap<FontInstanceKey, GlyphKeyCache>,
}

impl GlyphCache {
    pub fn new() -> GlyphCache {
        GlyphCache {
            glyph_key_caches: FastHashMap::default(),
        }
    }

    pub fn get_glyph_key_cache_for_font_mut(&mut self,
                                            font: FontInstanceKey) -> &mut GlyphKeyCache {
        self.glyph_key_caches
            .entry(font)
            .or_insert(ResourceClassCache::new())
    }

    pub fn get_glyph_key_cache_for_font(&self,
                                        font: &FontInstanceKey) -> &GlyphKeyCache {
        self.glyph_key_caches
            .get(font)
            .expect("BUG: Unable to find glyph key cache!")
    }

    pub fn expire_old_resources(&mut self, texture_cache: &mut TextureCache, frame_id: FrameId) {
        let mut caches_to_remove = Vec::new();

        for (font, glyph_key_cache) in &mut self.glyph_key_caches {
            glyph_key_cache.expire_old_resources(texture_cache, frame_id);

            if glyph_key_cache.is_empty() {
                caches_to_remove.push(font.clone());
            }
        }

        for key in caches_to_remove {
            self.glyph_key_caches.remove(&key).unwrap();
        }
    }

    pub fn clear_fonts<F>(&mut self, texture_cache: &mut TextureCache, key_fun: F)
    where for<'r> F: Fn(&'r &FontInstanceKey) -> bool
    {
        let caches_to_destroy = self.glyph_key_caches.keys()
            .filter(&key_fun)
            .cloned()
            .collect::<Vec<_>>();
        for key in caches_to_destroy {
            let mut cache = self.glyph_key_caches.remove(&key).unwrap();
            cache.clear(texture_cache);
        }
    }
}
