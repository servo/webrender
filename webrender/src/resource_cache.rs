/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{AddFont, BlobImageData, BlobImageResources, ResourceUpdate, ResourceUpdates};
use api::{BlobImageDescriptor, BlobImageError, BlobImageRenderer, BlobImageRequest};
use api::{ColorF, FontRenderMode, SubpixelDirection};
use api::{DevicePoint, DeviceUintRect, DeviceUintSize};
use api::{Epoch, FontInstance, FontInstanceKey, FontKey, FontTemplate};
use api::{ExternalImageData, ExternalImageType};
use api::{FontInstanceOptions, FontInstancePlatformOptions, FontVariation};
use api::{GlyphDimensions, GlyphKey, IdNamespace};
use api::{ImageData, ImageDescriptor, ImageKey, ImageRendering};
use api::{TileOffset, TileSize};
use app_units::Au;
use device::TextureFilter;
use frame::FrameId;
use glyph_cache::GlyphCache;
use glyph_rasterizer::{GlyphRasterizer, GlyphRequest};
use gpu_cache::{GpuCache, GpuCacheAddress, GpuCacheHandle};
use internal_types::{FastHashMap, FastHashSet, SourceTexture, TextureUpdateList};
use profiler::{ResourceProfileCounters, TextureCacheProfileCounters};
use rayon::ThreadPool;
use std::collections::hash_map::Entry::{self, Occupied, Vacant};
use std::fmt::Debug;
use std::hash::Hash;
use std::mem;
use std::sync::Arc;
use texture_cache::{TextureCache, TextureCacheHandle};

const DEFAULT_TILE_SIZE: TileSize = 512;

pub struct GlyphFetchResult {
    pub index_in_text_run: i32,
    pub uv_rect_address: GpuCacheAddress,
}

// These coordinates are always in texels.
// They are converted to normalized ST
// values in the vertex shader. The reason
// for this is that the texture may change
// dimensions (e.g. the pages in a texture
// atlas can grow). When this happens, by
// storing the coordinates as texel values
// we don't need to go through and update
// various CPU-side structures.
pub struct CacheItem {
    pub texture_id: SourceTexture,
    pub uv_rect_handle: GpuCacheHandle,
}

#[derive(Debug)]
pub struct ImageProperties {
    pub descriptor: ImageDescriptor,
    pub external_image: Option<ExternalImageData>,
    pub tiling: Option<TileSize>,
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum State {
    Idle,
    AddResources,
    QueryResources,
}

#[derive(Debug)]
struct ImageResource {
    data: ImageData,
    descriptor: ImageDescriptor,
    epoch: Epoch,
    tiling: Option<TileSize>,
    dirty_rect: Option<DeviceUintRect>,
}

#[derive(Debug)]
pub struct ImageTiling {
    pub image_size: DeviceUintSize,
    pub tile_size: TileSize,
}

pub type TiledImageMap = FastHashMap<ImageKey, ImageTiling>;

struct ImageTemplates {
    images: FastHashMap<ImageKey, ImageResource>,
}

impl ImageTemplates {
    fn new() -> Self {
        ImageTemplates {
            images: FastHashMap::default(),
        }
    }

    fn insert(&mut self, key: ImageKey, resource: ImageResource) {
        self.images.insert(key, resource);
    }

    fn remove(&mut self, key: ImageKey) -> Option<ImageResource> {
        self.images.remove(&key)
    }

    fn get(&self, key: ImageKey) -> Option<&ImageResource> {
        self.images.get(&key)
    }

    fn get_mut(&mut self, key: ImageKey) -> Option<&mut ImageResource> {
        self.images.get_mut(&key)
    }
}

struct CachedImageInfo {
    texture_cache_handle: TextureCacheHandle,
    epoch: Epoch,
}

pub struct ResourceClassCache<K, V> {
    resources: FastHashMap<K, V>,
}

impl<K, V> ResourceClassCache<K, V>
where
    K: Clone + Hash + Eq + Debug,
{
    pub fn new() -> ResourceClassCache<K, V> {
        ResourceClassCache {
            resources: FastHashMap::default(),
        }
    }

    fn get(&self, key: &K) -> &V {
        self.resources
            .get(key)
            .expect("Didn't find a cached resource with that ID!")
    }

    pub fn insert(&mut self, key: K, value: V) {
        self.resources.insert(key, value);
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.resources.get_mut(key)
    }

    pub fn entry(&mut self, key: K) -> Entry<K, V> {
        self.resources.entry(key)
    }

    pub fn clear(&mut self) {
        self.resources.clear();
    }

    fn clear_keys<F>(&mut self, key_fun: F)
    where
        for<'r> F: Fn(&'r &K) -> bool,
    {
        let resources_to_destroy = self.resources
            .keys()
            .filter(&key_fun)
            .cloned()
            .collect::<Vec<_>>();
        for key in resources_to_destroy {
            self.resources.remove(&key).unwrap();
        }
    }
}


#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct ImageRequest {
    key: ImageKey,
    rendering: ImageRendering,
    tile: Option<TileOffset>,
}

impl Into<BlobImageRequest> for ImageRequest {
    fn into(self) -> BlobImageRequest {
        BlobImageRequest {
            key: self.key,
            tile: self.tile,
        }
    }
}

struct Resources {
    font_templates: FastHashMap<FontKey, FontTemplate>,
    font_instances: FastHashMap<FontInstanceKey, FontInstance>,
    image_templates: ImageTemplates,
}

impl BlobImageResources for Resources {
    fn get_font_data(&self, key: FontKey) -> &FontTemplate {
        self.font_templates.get(&key).unwrap()
    }
    fn get_image(&self, key: ImageKey) -> Option<(&ImageData, &ImageDescriptor)> {
        self.image_templates
            .get(key)
            .map(|resource| (&resource.data, &resource.descriptor))
    }
}

pub struct ResourceCache {
    cached_glyphs: GlyphCache,
    cached_images: ResourceClassCache<ImageRequest, CachedImageInfo>,

    resources: Resources,
    state: State,
    current_frame_id: FrameId,

    texture_cache: TextureCache,

    // TODO(gw): We should expire (parts of) this cache semi-regularly!
    cached_glyph_dimensions: FastHashMap<GlyphRequest, Option<GlyphDimensions>>,
    glyph_rasterizer: GlyphRasterizer,

    // The set of images that aren't present or valid in the texture cache,
    // and need to be rasterized and/or uploaded this frame. This includes
    // both blobs and regular images.
    pending_image_requests: FastHashSet<ImageRequest>,

    blob_image_renderer: Option<Box<BlobImageRenderer>>,
}

impl ResourceCache {
    pub fn new(
        texture_cache: TextureCache,
        workers: Arc<ThreadPool>,
        blob_image_renderer: Option<Box<BlobImageRenderer>>,
    ) -> ResourceCache {
        ResourceCache {
            cached_glyphs: GlyphCache::new(),
            cached_images: ResourceClassCache::new(),
            resources: Resources {
                font_templates: FastHashMap::default(),
                font_instances: FastHashMap::default(),
                image_templates: ImageTemplates::new(),
            },
            cached_glyph_dimensions: FastHashMap::default(),
            texture_cache,
            state: State::Idle,
            current_frame_id: FrameId(0),
            pending_image_requests: FastHashSet::default(),
            glyph_rasterizer: GlyphRasterizer::new(workers),
            blob_image_renderer,
        }
    }

    pub fn max_texture_size(&self) -> u32 {
        self.texture_cache.max_texture_size()
    }

    fn should_tile(&self, descriptor: &ImageDescriptor, data: &ImageData) -> bool {
        let limit = self.max_texture_size();
        let size_check = descriptor.width > limit || descriptor.height > limit;
        match *data {
            ImageData::Raw(_) | ImageData::Blob(_) => size_check,
            ImageData::External(info) => {
                // External handles already represent existing textures so it does
                // not make sense to tile them into smaller ones.
                info.image_type == ExternalImageType::ExternalBuffer && size_check
            }
        }
    }

    pub fn update_resources(
        &mut self,
        updates: ResourceUpdates,
        profile_counters: &mut ResourceProfileCounters,
    ) {
        // TODO, there is potential for optimization here, by processing updates in
        // bulk rather than one by one (for example by sorting allocations by size or
        // in a way that reduces fragmentation in the atlas).

        for update in updates.updates {
            match update {
                ResourceUpdate::AddImage(img) => {
                    if let ImageData::Raw(ref bytes) = img.data {
                        profile_counters.image_templates.inc(bytes.len());
                    }
                    self.add_image_template(img.key, img.descriptor, img.data, img.tiling);
                }
                ResourceUpdate::UpdateImage(img) => {
                    self.update_image_template(img.key, img.descriptor, img.data, img.dirty_rect);
                }
                ResourceUpdate::DeleteImage(img) => {
                    self.delete_image_template(img);
                }
                ResourceUpdate::AddFont(font) => match font {
                    AddFont::Raw(id, bytes, index) => {
                        profile_counters.font_templates.inc(bytes.len());
                        self.add_font_template(id, FontTemplate::Raw(Arc::new(bytes), index));
                    }
                    AddFont::Native(id, native_font_handle) => {
                        self.add_font_template(id, FontTemplate::Native(native_font_handle));
                    }
                },
                ResourceUpdate::DeleteFont(font) => {
                    self.delete_font_template(font);
                }
                ResourceUpdate::AddFontInstance(instance) => {
                    self.add_font_instance(
                        instance.key,
                        instance.font_key,
                        instance.glyph_size,
                        instance.options,
                        instance.platform_options,
                        instance.variations,
                    );
                }
                ResourceUpdate::DeleteFontInstance(instance) => {
                    self.delete_font_instance(instance);
                }
            }
        }
    }

    pub fn add_font_template(&mut self, font_key: FontKey, template: FontTemplate) {
        // Push the new font to the font renderer, and also store
        // it locally for glyph metric requests.
        self.glyph_rasterizer.add_font(font_key, template.clone());
        self.resources.font_templates.insert(font_key, template);
    }

    pub fn delete_font_template(&mut self, font_key: FontKey) {
        self.glyph_rasterizer.delete_font(font_key);
        self.resources.font_templates.remove(&font_key);
        if let Some(ref mut r) = self.blob_image_renderer {
            r.delete_font(font_key);
        }
    }

    pub fn add_font_instance(
        &mut self,
        instance_key: FontInstanceKey,
        font_key: FontKey,
        glyph_size: Au,
        options: Option<FontInstanceOptions>,
        platform_options: Option<FontInstancePlatformOptions>,
        variations: Vec<FontVariation>,
    ) {
        let mut requested_render_mode = FontRenderMode::Subpixel;
        let mut subpx_dir = SubpixelDirection::Horizontal;
        if let Some(options) = options {
            if let Some(render_mode) = options.render_mode {
                requested_render_mode = render_mode;
            }
        }
        if self.glyph_rasterizer.is_bitmap_font(font_key) {
            requested_render_mode = requested_render_mode.limit_by(FontRenderMode::Bitmap);
        }
        if requested_render_mode == FontRenderMode::Mono {
            subpx_dir = SubpixelDirection::None;
        }
        let instance = FontInstance::new(
            font_key,
            glyph_size,
            ColorF::new(0.0, 0.0, 0.0, 1.0),
            requested_render_mode,
            subpx_dir,
            platform_options,
            variations,
            options.map_or(false, |opts| opts.synthetic_italics),
        );
        self.resources.font_instances.insert(instance_key, instance);
    }

    pub fn delete_font_instance(&mut self, instance_key: FontInstanceKey) {
        self.resources.font_instances.remove(&instance_key);
        if let Some(ref mut r) = self.blob_image_renderer {
            r.delete_font_instance(instance_key);
        }
    }

    pub fn get_font_instance(&self, instance_key: FontInstanceKey) -> Option<&FontInstance> {
        self.resources.font_instances.get(&instance_key)
    }

    pub fn add_image_template(
        &mut self,
        image_key: ImageKey,
        descriptor: ImageDescriptor,
        mut data: ImageData,
        mut tiling: Option<TileSize>,
    ) {
        if tiling.is_none() && self.should_tile(&descriptor, &data) {
            // We aren't going to be able to upload a texture this big, so tile it, even
            // if tiling was not requested.
            tiling = Some(DEFAULT_TILE_SIZE);
        }

        if let ImageData::Blob(ref mut blob) = data {
            self.blob_image_renderer.as_mut().unwrap().add(
                image_key,
                mem::replace(blob, BlobImageData::new()),
                tiling,
            );
        }

        let resource = ImageResource {
            descriptor,
            data,
            epoch: Epoch(0),
            tiling,
            dirty_rect: None,
        };

        self.resources.image_templates.insert(image_key, resource);
    }

    pub fn update_image_template(
        &mut self,
        image_key: ImageKey,
        descriptor: ImageDescriptor,
        mut data: ImageData,
        dirty_rect: Option<DeviceUintRect>,
    ) {
        let resource = if let Some(image) = self.resources.image_templates.get(image_key) {
            let next_epoch = Epoch(image.epoch.0 + 1);

            let mut tiling = image.tiling;
            if tiling.is_none() && self.should_tile(&descriptor, &data) {
                tiling = Some(DEFAULT_TILE_SIZE);
            }

            if let ImageData::Blob(ref mut blob) = data {
                self.blob_image_renderer
                    .as_mut()
                    .unwrap()
                    .update(image_key, mem::replace(blob, BlobImageData::new()));
            }

            ImageResource {
                descriptor,
                data,
                epoch: next_epoch,
                tiling,
                dirty_rect: match (dirty_rect, image.dirty_rect) {
                    (Some(rect), Some(prev_rect)) => Some(rect.union(&prev_rect)),
                    (Some(rect), None) => Some(rect),
                    _ => None,
                },
            }
        } else {
            panic!(
                "Attempt to update non-existant image (key {:?}).",
                image_key
            );
        };

        self.resources.image_templates.insert(image_key, resource);
    }

    pub fn delete_image_template(&mut self, image_key: ImageKey) {
        let value = self.resources.image_templates.remove(image_key);

        self.cached_images
            .clear_keys(|request| request.key == image_key);

        match value {
            Some(image) => if image.data.is_blob() {
                self.blob_image_renderer.as_mut().unwrap().delete(image_key);
            },
            None => {
                println!("Delete the non-exist key:{:?}", image_key);
            }
        }
    }

    pub fn request_image(
        &mut self,
        key: ImageKey,
        rendering: ImageRendering,
        tile: Option<TileOffset>,
        gpu_cache: &mut GpuCache,
    ) {
        debug_assert_eq!(self.state, State::AddResources);
        let request = ImageRequest {
            key,
            rendering,
            tile,
        };

        match self.resources.image_templates.get(key) {
            Some(template) => {
                // Images that don't use the texture cache can early out.
                if !template.data.uses_texture_cache() {
                    return;
                }

                // If this image exists in the texture cache, *and* the epoch
                // in the cache matches that of the template, then it is
                // valid to use as-is.
                let (entry, needs_update) = match self.cached_images.entry(request) {
                    Occupied(entry) => {
                        let needs_update = entry.get().epoch != template.epoch;
                        (entry.into_mut(), needs_update)
                    }
                    Vacant(entry) => (
                        entry.insert(CachedImageInfo {
                            epoch: template.epoch,
                            texture_cache_handle: TextureCacheHandle::new(),
                        }),
                        true,
                    ),
                };

                let needs_upload = self.texture_cache
                    .request(&mut entry.texture_cache_handle, gpu_cache);

                if !needs_upload && !needs_update {
                    return;
                }

                // We can start a worker thread rasterizing right now, if:
                //  - The image is a blob.
                //  - The blob hasn't already been requested this frame.
                if self.pending_image_requests.insert(request) {
                    if template.data.is_blob() {
                        if let Some(ref mut renderer) = self.blob_image_renderer {
                            let (offset, w, h) = match template.tiling {
                                Some(tile_size) => {
                                    let tile_offset = request.tile.unwrap();
                                    let (w, h) = compute_tile_size(
                                        &template.descriptor,
                                        tile_size,
                                        tile_offset,
                                    );
                                    let offset = DevicePoint::new(
                                        tile_offset.x as f32 * tile_size as f32,
                                        tile_offset.y as f32 * tile_size as f32,
                                    );

                                    (offset, w, h)
                                }
                                None => (
                                    DevicePoint::zero(),
                                    template.descriptor.width,
                                    template.descriptor.height,
                                ),
                            };

                            renderer.request(
                                &self.resources,
                                request.into(),
                                &BlobImageDescriptor {
                                    width: w,
                                    height: h,
                                    offset,
                                    format: template.descriptor.format,
                                },
                                template.dirty_rect,
                            );
                        }
                    }
                }
            }
            None => {
                warn!(
                    "ERROR: Trying to render deleted / non-existent key {:?}",
                    key
                );
            }
        }
    }

    pub fn request_glyphs(
        &mut self,
        font: FontInstance,
        glyph_keys: &[GlyphKey],
        gpu_cache: &mut GpuCache,
    ) {
        debug_assert_eq!(self.state, State::AddResources);

        self.glyph_rasterizer.request_glyphs(
            &mut self.cached_glyphs,
            font,
            glyph_keys,
            &mut self.texture_cache,
            gpu_cache,
        );
    }

    pub fn pending_updates(&mut self) -> TextureUpdateList {
        self.texture_cache.pending_updates()
    }

    pub fn fetch_glyphs<F>(
        &self,
        font: FontInstance,
        glyph_keys: &[GlyphKey],
        fetch_buffer: &mut Vec<GlyphFetchResult>,
        gpu_cache: &GpuCache,
        mut f: F,
    ) where
        F: FnMut(SourceTexture, &[GlyphFetchResult]),
    {
        debug_assert_eq!(self.state, State::QueryResources);
        let glyph_key_cache = self.cached_glyphs.get_glyph_key_cache_for_font(&font);

        let mut current_texture_id = SourceTexture::Invalid;
        debug_assert!(fetch_buffer.is_empty());

        for (loop_index, key) in glyph_keys.iter().enumerate() {
            let glyph = glyph_key_cache.get(key);
            let cache_item = glyph
                .as_ref()
                .map(|info| self.texture_cache.get(&info.texture_cache_handle));
            if let Some(cache_item) = cache_item {
                if current_texture_id != cache_item.texture_id {
                    if !fetch_buffer.is_empty() {
                        f(current_texture_id, fetch_buffer);
                        fetch_buffer.clear();
                    }
                    current_texture_id = cache_item.texture_id;
                }
                fetch_buffer.push(GlyphFetchResult {
                    index_in_text_run: loop_index as i32,
                    uv_rect_address: gpu_cache.get_address(&cache_item.uv_rect_handle),
                });
            }
        }

        if !fetch_buffer.is_empty() {
            f(current_texture_id, fetch_buffer);
            fetch_buffer.clear();
        }
    }

    pub fn get_glyph_dimensions(
        &mut self,
        font: &FontInstance,
        key: &GlyphKey,
    ) -> Option<GlyphDimensions> {
        let key = GlyphRequest::new(font, key);

        match self.cached_glyph_dimensions.entry(key.clone()) {
            Occupied(entry) => *entry.get(),
            Vacant(entry) => *entry.insert(
                self.glyph_rasterizer
                    .get_glyph_dimensions(&key.font, &key.key),
            ),
        }
    }

    pub fn get_glyph_index(&mut self, font_key: FontKey, ch: char) -> Option<u32> {
        self.glyph_rasterizer.get_glyph_index(font_key, ch)
    }

    #[inline]
    pub fn get_cached_image(
        &self,
        image_key: ImageKey,
        image_rendering: ImageRendering,
        tile: Option<TileOffset>,
    ) -> CacheItem {
        debug_assert_eq!(self.state, State::QueryResources);
        let key = ImageRequest {
            key: image_key,
            rendering: image_rendering,
            tile,
        };
        let image_info = &self.cached_images.get(&key);
        self.texture_cache.get(&image_info.texture_cache_handle)
    }

    pub fn get_image_properties(&self, image_key: ImageKey) -> Option<ImageProperties> {
        let image_template = &self.resources.image_templates.get(image_key);

        image_template.map(|image_template| {
            let external_image = match image_template.data {
                ImageData::External(ext_image) => {
                    match ext_image.image_type {
                        ExternalImageType::Texture2DHandle |
                        ExternalImageType::Texture2DArrayHandle |
                        ExternalImageType::TextureRectHandle |
                        ExternalImageType::TextureExternalHandle => Some(ext_image),
                        // external buffer uses resource_cache.
                        ExternalImageType::ExternalBuffer => None,
                    }
                }
                // raw and blob image are all using resource_cache.
                ImageData::Raw(..) | ImageData::Blob(..) => None,
            };

            ImageProperties {
                descriptor: image_template.descriptor,
                external_image,
                tiling: image_template.tiling,
            }
        })
    }

    pub fn get_tiled_image_map(&self) -> TiledImageMap {
        self.resources
            .image_templates
            .images
            .iter()
            .filter_map(|(&key, template)| {
                template.tiling.map(|tile_size| {
                    (
                        key,
                        ImageTiling {
                            image_size: DeviceUintSize::new(
                                template.descriptor.width,
                                template.descriptor.height,
                            ),
                            tile_size,
                        },
                    )
                })
            })
            .collect()
    }

    pub fn begin_frame(&mut self, frame_id: FrameId) {
        debug_assert_eq!(self.state, State::Idle);
        self.state = State::AddResources;
        self.texture_cache.begin_frame(frame_id);
        self.current_frame_id = frame_id;
    }

    pub fn block_until_all_resources_added(
        &mut self,
        gpu_cache: &mut GpuCache,
        texture_cache_profile: &mut TextureCacheProfileCounters,
    ) {
        profile_scope!("block_until_all_resources_added");

        debug_assert_eq!(self.state, State::AddResources);
        self.state = State::QueryResources;

        self.glyph_rasterizer.resolve_glyphs(
            &mut self.cached_glyphs,
            &mut self.texture_cache,
            gpu_cache,
            texture_cache_profile,
        );

        // Apply any updates of new / updated images (incl. blobs) to the texture cache.
        self.update_texture_cache(gpu_cache);
        self.texture_cache.end_frame(texture_cache_profile);
    }

    fn update_texture_cache(&mut self, gpu_cache: &mut GpuCache) {
        for request in self.pending_image_requests.drain() {
            let image_template = self.resources.image_templates.get_mut(request.key).unwrap();
            debug_assert!(image_template.data.uses_texture_cache());

            let image_data = match image_template.data {
                ImageData::Raw(..) | ImageData::External(..) => {
                    // Safe to clone here since the Raw image data is an
                    // Arc, and the external image data is small.
                    image_template.data.clone()
                }
                ImageData::Blob(..) => {
                    // Extract the rasterized image from the blob renderer.
                    match self.blob_image_renderer
                        .as_mut()
                        .unwrap()
                        .resolve(request.into())
                    {
                        Ok(image) => ImageData::new(image.data),
                        // TODO(nical): I think that we should handle these somewhat gracefully,
                        // at least in the out-of-memory scenario.
                        Err(BlobImageError::Oom) => {
                            // This one should be recoverable-ish.
                            panic!("Failed to render a vector image (OOM)");
                        }
                        Err(BlobImageError::InvalidKey) => {
                            panic!("Invalid vector image key");
                        }
                        Err(BlobImageError::InvalidData) => {
                            // TODO(nical): If we run into this we should kill the content process.
                            panic!("Invalid vector image data");
                        }
                        Err(BlobImageError::Other(msg)) => {
                            panic!("Vector image error {}", msg);
                        }
                    }
                }
            };

            let filter = match request.rendering {
                ImageRendering::Pixelated => TextureFilter::Nearest,
                ImageRendering::Auto | ImageRendering::CrispEdges => TextureFilter::Linear,
            };

            let descriptor = if let Some(tile) = request.tile {
                let tile_size = image_template.tiling.unwrap();
                let image_descriptor = &image_template.descriptor;

                let (actual_width, actual_height) =
                    compute_tile_size(image_descriptor, tile_size, tile);

                // The tiled image could be stored on the CPU as one large image or be
                // already broken up into tiles. This affects the way we compute the stride
                // and offset.
                let tiled_on_cpu = image_template.data.is_blob();

                let (stride, offset) = if tiled_on_cpu {
                    (image_descriptor.stride, 0)
                } else {
                    let bpp = image_descriptor.format.bytes_per_pixel();
                    let stride = image_descriptor.compute_stride();
                    let offset = image_descriptor.offset +
                        tile.y as u32 * tile_size as u32 * stride +
                        tile.x as u32 * tile_size as u32 * bpp;
                    (Some(stride), offset)
                };

                ImageDescriptor {
                    width: actual_width,
                    height: actual_height,
                    stride,
                    offset,
                    format: image_descriptor.format,
                    is_opaque: image_descriptor.is_opaque,
                }
            } else {
                image_template.descriptor.clone()
            };

            let entry = self.cached_images.get_mut(&request).unwrap();
            self.texture_cache.update(
                &mut entry.texture_cache_handle,
                descriptor,
                filter,
                image_data,
                [0.0; 2],
                image_template.dirty_rect,
                gpu_cache,
            );
            image_template.dirty_rect = None;
        }
    }

    pub fn end_frame(&mut self) {
        debug_assert_eq!(self.state, State::QueryResources);
        self.state = State::Idle;
    }

    pub fn on_memory_pressure(&mut self) {
        // This is drastic. It will basically flush everything out of the cache,
        // and the next frame will have to rebuild all of its resources.
        // We may want to look into something less extreme, but on the other hand this
        // should only be used in situations where are running low enough on memory
        // that we risk crashing if we don't do something about it.
        // The advantage of clearing the cache completely is that it gets rid of any
        // remaining fragmentation that could have persisted if we kept around the most
        // recently used resources.
        self.cached_images.clear();
        self.cached_glyphs.clear();
    }

    pub fn clear_namespace(&mut self, namespace: IdNamespace) {
        //TODO: use `retain` when we are on Rust-1.18
        let image_keys: Vec<_> = self.resources
            .image_templates
            .images
            .keys()
            .filter(|&key| key.0 == namespace)
            .cloned()
            .collect();
        for key in &image_keys {
            self.resources.image_templates.images.remove(key);
        }

        let font_keys: Vec<_> = self.resources
            .font_templates
            .keys()
            .filter(|&key| key.0 == namespace)
            .cloned()
            .collect();
        for key in &font_keys {
            self.resources.font_templates.remove(key);
        }

        self.cached_images
            .clear_keys(|request| request.key.0 == namespace);
        self.cached_glyphs
            .clear_fonts(|font| font.font_key.0 == namespace);
    }
}

// Compute the width and height of a tile depending on its position in the image.
pub fn compute_tile_size(
    descriptor: &ImageDescriptor,
    base_size: TileSize,
    tile: TileOffset,
) -> (u32, u32) {
    let base_size = base_size as u32;
    // Most tiles are going to have base_size as width and height,
    // except for tiles around the edges that are shrunk to fit the mage data
    // (See decompose_tiled_image in frame.rs).
    let actual_width = if (tile.x as u32) < descriptor.width / base_size {
        base_size
    } else {
        descriptor.width % base_size
    };

    let actual_height = if (tile.y as u32) < descriptor.height / base_size {
        base_size
    } else {
        descriptor.height % base_size
    };

    (actual_width, actual_height)
}
