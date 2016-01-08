use app_units::Au;
use device::{TextureFilter, TextureId};
use euclid::Size2D;
use fnv::FnvHasher;
use freelist::FreeList;
use internal_types::{FontTemplate, GlyphKey, RasterItem};
use internal_types::{TextureUpdateList, DrawListId, DrawList};
use platform::font::{FontContext, RasterizedGlyph};
use renderer::BLUR_INFLATION_FACTOR;
use resource_list::ResourceList;
use scoped_threadpool;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::hash_state::DefaultState;
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT};
use std::sync::atomic::Ordering::SeqCst;
use std::thread;
use std::time::Duration;
use texture_cache::{TextureCache, TextureCacheItem, TextureCacheItemId};
use texture_cache::{BorderType, TextureInsertOp};
use webrender_traits::{Epoch, FontKey, ImageKey, ImageFormat, DisplayItem, ImageRendering};
use webrender_traits::{WebGLContextId};

static FONT_CONTEXT_COUNT: AtomicUsize = ATOMIC_USIZE_INIT;

thread_local!(pub static FONT_CONTEXT: RefCell<FontContext> = RefCell::new(FontContext::new()));

struct ImageResource {
    bytes: Vec<u8>,
    width: u32,
    height: u32,
    format: ImageFormat,
    epoch: Epoch,
}

struct GlyphRasterJob {
    glyph_key: GlyphKey,
    result: Option<RasterizedGlyph>,
}

struct CachedImageInfo {
    texture_cache_id: TextureCacheItemId,
    epoch: Epoch,
}

pub struct ResourceCache {
    cached_glyphs: HashMap<GlyphKey, Option<TextureCacheItemId>, DefaultState<FnvHasher>>,
    cached_rasters: HashMap<RasterItem, TextureCacheItemId, DefaultState<FnvHasher>>,
    cached_images: HashMap<(ImageKey, ImageRendering), CachedImageInfo, DefaultState<FnvHasher>>,
    webgl_textures: HashMap<WebGLContextId, TextureId, DefaultState<FnvHasher>>,

    draw_lists: FreeList<DrawList>,
    font_templates: HashMap<FontKey, FontTemplate, DefaultState<FnvHasher>>,
    image_templates: HashMap<ImageKey, ImageResource, DefaultState<FnvHasher>>,
    device_pixel_ratio: f32,
    enable_aa: bool,

    texture_cache: TextureCache,

    pending_raster_jobs: Vec<GlyphRasterJob>,

    white_image_id: TextureCacheItemId,
    dummy_mask_image_id: TextureCacheItemId,
}

impl ResourceCache {
    pub fn new(thread_pool: &mut scoped_threadpool::Pool,
               texture_cache: TextureCache,
               white_image_id: TextureCacheItemId,
               dummy_mask_image_id: TextureCacheItemId,
               device_pixel_ratio: f32,
               enable_aa: bool) -> ResourceCache {

        let thread_count = thread_pool.thread_count() as usize;
        thread_pool.scoped(|scope| {
            for _ in 0..thread_count {
                scope.execute(|| {
                    FONT_CONTEXT.with(|_| {
                        FONT_CONTEXT_COUNT.fetch_add(1, SeqCst);
                        while FONT_CONTEXT_COUNT.load(SeqCst) != thread_count {
                            thread::sleep(Duration::from_millis(1));
                        }
                    });
                });
            }
        });

        ResourceCache {
            cached_glyphs: HashMap::with_hash_state(Default::default()),
            cached_rasters: HashMap::with_hash_state(Default::default()),
            cached_images: HashMap::with_hash_state(Default::default()),
            webgl_textures: HashMap::with_hash_state(Default::default()),
            draw_lists: FreeList::new(),
            font_templates: HashMap::with_hash_state(Default::default()),
            image_templates: HashMap::with_hash_state(Default::default()),
            texture_cache: texture_cache,
            pending_raster_jobs: Vec::new(),
            device_pixel_ratio: device_pixel_ratio,
            white_image_id: white_image_id,
            dummy_mask_image_id: dummy_mask_image_id,
            enable_aa: enable_aa,
        }
    }

    pub fn add_font_template(&mut self, font_key: FontKey, template: FontTemplate) {
        self.font_templates.insert(font_key, template);
    }

    pub fn add_image_template(&mut self,
                              image_key: ImageKey,
                              width: u32,
                              height: u32,
                              format: ImageFormat,
                              bytes: Vec<u8>) {
        let resource = ImageResource {
            width: width,
            height: height,
            format: format,
            bytes: bytes,
            epoch: Epoch(0),
        };

        self.image_templates.insert(image_key, resource);
    }

    pub fn update_image_template(&mut self,
                                 image_key: ImageKey,
                                 width: u32,
                                 height: u32,
                                 format: ImageFormat,
                                 bytes: Vec<u8>) {
        let next_epoch = match self.image_templates.get(&image_key) {
            Some(image) => {
                let Epoch(current_epoch) = image.epoch;
                Epoch(current_epoch + 1)
            }
            None => {
                Epoch(0)
            }
        };

        let resource = ImageResource {
            width: width,
            height: height,
            format: format,
            bytes: bytes,
            epoch: next_epoch,
        };

        self.image_templates.insert(image_key, resource);
    }

    pub fn add_webgl_texture(&mut self, id: WebGLContextId, texture_id: TextureId, size: Size2D<i32>) {
        self.webgl_textures.insert(id, texture_id);
        self.texture_cache.add_raw_update(texture_id, size);
    }

    pub fn add_resource_list(&mut self, resource_list: &ResourceList) {
        // Update texture cache with any GPU generated procedural items.
        resource_list.for_each_raster(|raster_item| {
            if !self.cached_rasters.contains_key(raster_item) {
                let image_id = self.texture_cache.new_item_id();
                self.texture_cache.insert_raster_op(image_id,
                                                    raster_item,
                                                    self.device_pixel_ratio);
                self.cached_rasters.insert(raster_item.clone(), image_id);
            }
        });

        // Update texture cache with any images that aren't yet uploaded to GPU.
        resource_list.for_each_image(|image_key, image_rendering| {
            let cached_images = &mut self.cached_images;
            let image_template = &self.image_templates[&image_key];

            match cached_images.entry((image_key, image_rendering)) {
                Occupied(entry) => {
                    if entry.get().epoch != image_template.epoch {
                        let image_id = entry.get().texture_cache_id;

                        // TODO: Can we avoid the clone of the bytes here?
                        self.texture_cache.update(image_id,
                                                  image_template.width,
                                                  image_template.height,
                                                  image_template.format,
                                                  image_template.bytes.clone());

                        // Update the cached epoch
                        *entry.into_mut() = CachedImageInfo {
                            texture_cache_id: image_id,
                            epoch: image_template.epoch,
                        };
                    }
                }
                Vacant(entry) => {
                    let image_id = self.texture_cache.new_item_id();

                    let filter = match image_rendering {
                        ImageRendering::Pixelated => TextureFilter::Nearest,
                        ImageRendering::Auto | ImageRendering::CrispEdges => TextureFilter::Linear,
                    };

                    // TODO: Can we avoid the clone of the bytes here?
                    self.texture_cache.insert(image_id,
                                              0,
                                              0,
                                              image_template.width,
                                              image_template.height,
                                              image_template.format,
                                              filter,
                                              TextureInsertOp::Blit(image_template.bytes.clone()),
                                              BorderType::SinglePixel);

                    entry.insert(CachedImageInfo {
                        texture_cache_id: image_id,
                        epoch: image_template.epoch,
                    });
                }
            };
        });

        // Update texture cache with any newly rasterized glyphs.
        resource_list.for_each_glyph(|glyph_key| {
            if !self.cached_glyphs.contains_key(glyph_key) {
                self.pending_raster_jobs.push(GlyphRasterJob {
                    glyph_key: glyph_key.clone(),
                    result: None,
                });
            }
        });
    }

    pub fn raster_pending_glyphs(&mut self,
                                 thread_pool: &mut scoped_threadpool::Pool) {
        // Run raster jobs in parallel
        run_raster_jobs(thread_pool,
                        &mut self.pending_raster_jobs,
                        &self.font_templates,
                        self.device_pixel_ratio,
                        self.enable_aa);

        // Add completed raster jobs to the texture cache
        for job in self.pending_raster_jobs.drain(..) {
            let result = job.result.expect("Failed to rasterize the glyph?");
            let image_id = if result.width > 0 && result.height > 0 {
                let image_id = self.texture_cache.new_item_id();
                let texture_width;
                let texture_height;
                let insert_op;
                match job.glyph_key.blur_radius {
                    Au(0) => {
                        texture_width = result.width;
                        texture_height = result.height;
                        insert_op = TextureInsertOp::Blit(result.bytes);
                    }
                    blur_radius => {
                        let blur_radius_px = f32::ceil(blur_radius.to_f32_px() * self.device_pixel_ratio)
                            as u32;
                        texture_width = result.width + blur_radius_px * BLUR_INFLATION_FACTOR;
                        texture_height = result.height + blur_radius_px * BLUR_INFLATION_FACTOR;
                        insert_op = TextureInsertOp::Blur(result.bytes,
                                                          Size2D::new(result.width, result.height),
                                                          blur_radius);
                    }
                }
                self.texture_cache.insert(image_id,
                                          result.left,
                                          result.top,
                                          texture_width,
                                          texture_height,
                                          ImageFormat::RGBA8,
                                          TextureFilter::Linear,
                                          insert_op,
                                          BorderType::NoBorder);
                Some(image_id)
            } else {
                None
            };
            self.cached_glyphs.insert(job.glyph_key, image_id);
        }
    }

    pub fn add_draw_list(&mut self, items: Vec<DisplayItem>) -> DrawListId {
        self.draw_lists.insert(DrawList::new(items))
    }

    pub fn get_draw_list(&self, draw_list_id: DrawListId) -> &DrawList {
        self.draw_lists.get(draw_list_id)
    }

    pub fn get_draw_list_mut(&mut self, draw_list_id: DrawListId) -> &mut DrawList {
        self.draw_lists.get_mut(draw_list_id)
    }

    pub fn remove_draw_list(&mut self, draw_list_id: DrawListId) {
        self.draw_lists.free(draw_list_id);
    }

    pub fn allocate_render_target(&mut self,
                                  width: u32,
                                  height: u32,
                                  format: ImageFormat)
                                  -> TextureId {
        self.texture_cache.allocate_render_target(width, height, format)
    }

    pub fn free_render_target(&mut self, texture_id: TextureId) {
        self.texture_cache.free_render_target(texture_id)
    }

    pub fn pending_updates(&mut self) -> TextureUpdateList {
        self.texture_cache.pending_updates()
    }

    #[inline]
    pub fn get_dummy_mask_image(&self) -> &TextureCacheItem {
        self.texture_cache.get(self.dummy_mask_image_id)
    }

    #[inline]
    pub fn get_dummy_color_image(&self) -> &TextureCacheItem {
        self.texture_cache.get(self.white_image_id)
    }

    #[inline]
    pub fn get_glyph(&self, glyph_key: &GlyphKey) -> Option<&TextureCacheItem> {
        let image_id = self.cached_glyphs[glyph_key];
        image_id.map(|image_id| self.texture_cache.get(image_id))
    }

    #[inline]
    pub fn get_image(&self,
                     image_key: ImageKey,
                     image_rendering: ImageRendering) -> &TextureCacheItem {
        let image_info = &self.cached_images[&(image_key, image_rendering)];
        self.texture_cache.get(image_info.texture_cache_id)
    }

    #[inline]
    pub fn get_raster(&self, raster_item: &RasterItem) -> &TextureCacheItem {
        let image_id = self.cached_rasters[raster_item];
        self.texture_cache.get(image_id)
    }

    #[inline]
    pub fn get_webgl_texture(&self,
                             context_id: &WebGLContextId) -> TextureId {
        self.webgl_textures.get(context_id).unwrap().clone()
    }

    pub fn device_pixel_ratio(&self) -> f32 {
        self.device_pixel_ratio
    }
}

fn run_raster_jobs(thread_pool: &mut scoped_threadpool::Pool,
                   pending_raster_jobs: &mut Vec<GlyphRasterJob>,
                   font_templates: &HashMap<FontKey, FontTemplate, DefaultState<FnvHasher>>,
                   device_pixel_ratio: f32,
                   enable_aa: bool) {
    if pending_raster_jobs.is_empty() {
        return
    }

    // Run raster jobs in parallel
    thread_pool.scoped(|scope| {
        for job in pending_raster_jobs {
            scope.execute(|| {
                let font_template = &font_templates[&job.glyph_key.font_key];
                FONT_CONTEXT.with(move |font_context| {
                    let mut font_context = font_context.borrow_mut();
                    match *font_template {
                        FontTemplate::Raw(ref bytes) => {
                            font_context.add_raw_font(&job.glyph_key.font_key, &**bytes);
                        }
                        FontTemplate::Native(ref native_font_handle) => {
                            font_context.add_native_font(&job.glyph_key.font_key,
                                                         (*native_font_handle).clone());
                        }
                    }
                    job.result = font_context.get_glyph(job.glyph_key.font_key,
                                                        job.glyph_key.size,
                                                        job.glyph_key.index,
                                                        device_pixel_ratio,
                                                        enable_aa);
                });
            });
        }
    });
}
