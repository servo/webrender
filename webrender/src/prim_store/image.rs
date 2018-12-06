/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{
    AlphaType, ColorF, ColorU, DeviceIntRect, DeviceIntSideOffsets,
    DeviceIntSize, ImageRendering, LayoutRect, LayoutSize, LayoutPrimitiveInfo,
    PremultipliedColorF, Shadow, TileOffset
};
use api::ImageKey as ApiImageKey;
use display_list_flattener::{AsInstanceKind, CreateShadow, IsVisible};
use frame_builder::FrameBuildingState;
use gpu_cache::{GpuCacheHandle, GpuDataRequest};
use intern::{DataStore, Handle, Internable, Interner, InternDebug, UpdateList};
use picture::SurfaceIndex;
use prim_store::{
    EdgeAaSegmentMask, OpacityBindingIndex, PrimitiveInstanceKind,
    PrimitiveOpacity, PrimitiveSceneData, PrimKeyCommonData,
    PrimTemplateCommonData, PrimitiveStore, SegmentInstanceIndex, SizeKey
};
use render_task::{
    BlitSource, RenderTask, RenderTaskCacheEntryHandle, RenderTaskCacheKey,
    RenderTaskCacheKeyKind
};
use resource_cache::ImageRequest;
use std::ops::{Deref, DerefMut};

#[derive(Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct VisibleImageTile {
    pub tile_offset: TileOffset,
    pub handle: GpuCacheHandle,
    pub edge_flags: EdgeAaSegmentMask,
    pub local_rect: LayoutRect,
    pub local_clip_rect: LayoutRect,
}

// Key that identifies a unique (partial) image that is being
// stored in the render task cache.
#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct ImageCacheKey {
    pub request: ImageRequest,
    pub texel_rect: Option<DeviceIntRect>,
}

/// Instance specific fields for an image primitive. These are
/// currently stored in a separate array to avoid bloating the
/// size of PrimitiveInstance. In the future, we should be able
/// to remove this and store the information inline, by:
/// (a) Removing opacity collapse / binding support completely.
///     Once we have general picture caching, we don't need this.
/// (b) Change visible_tiles to use Storage in the primitive
///     scratch buffer. This will reduce the size of the
///     visible_tiles field here, and save memory allocation
///     when image tiling is used. I've left it as a Vec for
///     now to reduce the number of changes, and because image
///     tiling is very rare on real pages.
#[derive(Debug)]
pub struct ImageInstance {
    pub opacity_binding_index: OpacityBindingIndex,
    pub segment_instance_index: SegmentInstanceIndex,
    pub visible_tiles: Vec<VisibleImageTile>,
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ImageKey {
    pub common: PrimKeyCommonData,
    pub key: ApiImageKey,
    pub stretch_size: SizeKey,
    pub tile_spacing: SizeKey,
    pub color: ColorU,
    pub sub_rect: Option<DeviceIntRect>,
    pub image_rendering: ImageRendering,
    pub alpha_type: AlphaType,
}

impl ImageKey {
    pub fn new(
        is_backface_visible: bool,
        prim_size: LayoutSize,
        prim_relative_clip_rect: LayoutRect,
        image: Image,
    ) -> Self {

        ImageKey {
            common: PrimKeyCommonData {
                is_backface_visible,
                prim_size: prim_size.into(),
                prim_relative_clip_rect: prim_relative_clip_rect.into(),
            },
            key: image.key,
            color: image.color.into(),
            stretch_size: image.stretch_size.into(),
            tile_spacing: image.tile_spacing.into(),
            sub_rect: image.sub_rect,
            image_rendering: image.image_rendering,
            alpha_type: image.alpha_type,
        }
    }
}

impl InternDebug for ImageKey {}

impl AsInstanceKind<ImageDataHandle> for ImageKey {
    /// Construct a primitive instance that matches the type
    /// of primitive key.
    fn as_instance_kind(
        &self,
        data_handle: ImageDataHandle,
        prim_store: &mut PrimitiveStore,
    ) -> PrimitiveInstanceKind {
        // TODO(gw): Refactor this to not need a separate image
        //           instance (see ImageInstance struct).
        let image_instance_index = prim_store.images.push(ImageInstance {
            opacity_binding_index: OpacityBindingIndex::INVALID,
            segment_instance_index: SegmentInstanceIndex::INVALID,
            visible_tiles: Vec::new(),
        });

        PrimitiveInstanceKind::Image {
            data_handle,
            image_instance_index,
        }
    }
}

// Where to find the texture data for an image primitive.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug)]
pub enum ImageSource {
    // A normal image - just reference the texture cache.
    Default,
    // An image that is pre-rendered into the texture cache
    // via a render task.
    Cache {
        size: DeviceIntSize,
        handle: Option<RenderTaskCacheEntryHandle>,
    },
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct ImageTemplate {
    pub common: PrimTemplateCommonData,
    pub key: ApiImageKey,
    pub stretch_size: LayoutSize,
    pub tile_spacing: LayoutSize,
    pub color: ColorF,
    pub source: ImageSource,
    pub image_rendering: ImageRendering,
    pub sub_rect: Option<DeviceIntRect>,
    pub alpha_type: AlphaType,
}

impl From<ImageKey> for ImageTemplate {
    fn from(item: ImageKey) -> Self {
        let common = PrimTemplateCommonData::with_key_common(item.common);
        let color = item.color.into();
        let stretch_size = item.stretch_size.into();
        let tile_spacing = item.tile_spacing.into();

        ImageTemplate {
            common,
            key: item.key,
            color,
            stretch_size,
            tile_spacing,
            source: ImageSource::Default,
            sub_rect: item.sub_rect,
            image_rendering: item.image_rendering,
            alpha_type: item.alpha_type,
        }
    }
}

impl Deref for ImageTemplate {
    type Target = PrimTemplateCommonData;
    fn deref(&self) -> &Self::Target {
        &self.common
    }
}

impl DerefMut for ImageTemplate {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.common
    }
}

fn write_image_gpu_blocks(
    request: &mut GpuDataRequest,
    color: ColorF,
    stretch_size: LayoutSize,
    tile_spacing: LayoutSize,
) {
    // Images are drawn as a white color, modulated by the total
    // opacity coming from any collapsed property bindings.
    request.push(color.premultiplied());
    request.push(PremultipliedColorF::WHITE);
    request.push([
        stretch_size.width + tile_spacing.width,
        stretch_size.height + tile_spacing.height,
        0.0,
        0.0,
    ]);
}

impl ImageTemplate {
    /// Update the GPU cache for a given primitive template. This may be called multiple
    /// times per frame, by each primitive reference that refers to this interned
    /// template. The initial request call to the GPU cache ensures that work is only
    /// done if the cache entry is invalid (due to first use or eviction).
    pub fn update(
        &mut self,
        // TODO(gw): Passing in surface_index here is not ideal. The primitive template
        //           code shouldn't depend on current surface state. This is due to a
        //           limitation in how render task caching works. We should fix this by
        //           allowing render task caching to assign to surfaces implicitly
        //           during pass allocation.
        surface_index: SurfaceIndex,
        frame_state: &mut FrameBuildingState,
    ) {
        if let Some(mut request) =
            frame_state.gpu_cache.request(&mut self.common.gpu_cache_handle) {
                write_image_gpu_blocks(
                    &mut request,
                    self.color,
                    self.stretch_size,
                    self.tile_spacing
                );

                // write_segment_gpu_blocks
            }

        self.opacity = {
            let image_properties = frame_state
                .resource_cache
                .get_image_properties(self.key);

            match image_properties {
                Some(image_properties) => {
                    let is_tiled = image_properties.tiling.is_some();

                    if self.tile_spacing != LayoutSize::zero() && !is_tiled {
                        self.source = ImageSource::Cache {
                            // Size in device-pixels we need to allocate in render task cache.
                            size: image_properties.descriptor.size.to_i32(),
                            handle: None,
                        };
                    }

                    // Work out whether this image is a normal / simple type, or if
                    // we need to pre-render it to the render task cache.
                    if let Some(rect) = self.sub_rect {
                        // We don't properly support this right now.
                        debug_assert!(!is_tiled);
                        self.source = ImageSource::Cache {
                            // Size in device-pixels we need to allocate in render task cache.
                            size: rect.size,
                            handle: None,
                        };
                    }

                    let mut request_source_image = false;
                    let mut is_opaque = image_properties.descriptor.is_opaque;
                    let request = ImageRequest {
                        key: self.key,
                        rendering: self.image_rendering,
                        tile: None,
                    };

                    // Every frame, for cached items, we need to request the render
                    // task cache item. The closure will be invoked on the first
                    // time through, and any time the render task output has been
                    // evicted from the texture cache.
                    match self.source {
                        ImageSource::Cache { ref mut size, ref mut handle } => {
                            let padding = DeviceIntSideOffsets::new(
                                0,
                                (self.tile_spacing.width * size.width as f32 / self.stretch_size.width) as i32,
                                (self.tile_spacing.height * size.height as f32 / self.stretch_size.height) as i32,
                                0,
                            );

                            let inner_size = *size;
                            size.width += padding.horizontal();
                            size.height += padding.vertical();

                            is_opaque &= padding == DeviceIntSideOffsets::zero();

                            let image_cache_key = ImageCacheKey {
                                request,
                                texel_rect: self.sub_rect,
                            };
                            let surfaces = &mut frame_state.surfaces;

                            // Request a pre-rendered image task.
                            *handle = Some(frame_state.resource_cache.request_render_task(
                                RenderTaskCacheKey {
                                    size: *size,
                                    kind: RenderTaskCacheKeyKind::Image(image_cache_key),
                                },
                                frame_state.gpu_cache,
                                frame_state.render_tasks,
                                None,
                                image_properties.descriptor.is_opaque,
                                |render_tasks| {
                                    // We need to render the image cache this frame,
                                    // so will need access to the source texture.
                                    request_source_image = true;

                                    // Create a task to blit from the texture cache to
                                    // a normal transient render task surface. This will
                                    // copy only the sub-rect, if specified.
                                    let cache_to_target_task = RenderTask::new_blit_with_padding(
                                        inner_size,
                                        &padding,
                                        BlitSource::Image { key: image_cache_key },
                                    );
                                    let cache_to_target_task_id = render_tasks.add(cache_to_target_task);

                                    // Create a task to blit the rect from the child render
                                    // task above back into the right spot in the persistent
                                        // render target cache.
                                    let target_to_cache_task = RenderTask::new_blit(
                                        *size,
                                        BlitSource::RenderTask {
                                            task_id: cache_to_target_task_id,
                                        },
                                    );
                                    let target_to_cache_task_id = render_tasks.add(target_to_cache_task);

                                    // Hook this into the render task tree at the right spot.
                                    surfaces[surface_index.0].tasks.push(target_to_cache_task_id);

                                    // Pass the image opacity, so that the cached render task
                                    // item inherits the same opacity properties.
                                    target_to_cache_task_id
                                }
                            ));
                        }
                        ImageSource::Default => {
                            // Normal images just reference the source texture each frame.
                            request_source_image = true;
                        }
                    }

                    if request_source_image && !is_tiled {
                        frame_state.resource_cache.request_image(
                            request,
                            frame_state.gpu_cache,
                        );
                    }

                    if is_opaque {
                        PrimitiveOpacity::from_alpha(self.color.a)
                    } else {
                        PrimitiveOpacity::translucent()
                    }
                }
                None => {
                    PrimitiveOpacity::opaque()
                }
            }
        };
    }

    pub fn write_prim_gpu_blocks(&self, request: &mut GpuDataRequest) {
        write_image_gpu_blocks(
            request,
            self.color,
            self.stretch_size,
            self.tile_spacing
        );
    }
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct ImageDataMarker;

pub type ImageDataStore = DataStore<ImageKey, ImageTemplate, ImageDataMarker>;
pub type ImageDataHandle = Handle<ImageDataMarker>;
pub type ImageDataUpdateList = UpdateList<ImageKey>;
pub type ImageDataInterner = Interner<ImageKey, PrimitiveSceneData, ImageDataMarker>;

pub struct Image {
    pub key: ApiImageKey,
    pub stretch_size: SizeKey,
    pub tile_spacing: SizeKey,
    pub color: ColorU,
    pub sub_rect: Option<DeviceIntRect>,
    pub image_rendering: ImageRendering,
    pub alpha_type: AlphaType,
}

impl Internable for Image {
    type Marker = ImageDataMarker;
    type Source = ImageKey;
    type StoreData = ImageTemplate;
    type InternData = PrimitiveSceneData;

    /// Build a new key from self with `info`.
    fn build_key(
        self,
        info: &LayoutPrimitiveInfo,
        prim_relative_clip_rect: LayoutRect,
    ) -> ImageKey {
        ImageKey::new(
            info.is_backface_visible,
            info.rect.size,
            prim_relative_clip_rect,
            self
        )
    }
}

impl CreateShadow for Image {
    fn create_shadow(&self, shadow: &Shadow) -> Self {
        Image {
            tile_spacing: self.tile_spacing,
            stretch_size: self.stretch_size,
            key: self.key,
            sub_rect: self.sub_rect,
            image_rendering: self.image_rendering,
            alpha_type: self.alpha_type,
            color: shadow.color.into(),
        }
    }
}

impl IsVisible for Image {
    fn is_visible(&self) -> bool {
        true
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_struct_sizes() {
    use std::mem;
    // The sizes of these structures are critical for performance on a number of
    // talos stress tests. If you get a failure here on CI, there's two possibilities:
    // (a) You made a structure smaller than it currently is. Great work! Update the
    //     test expectations and move on.
    // (b) You made a structure larger. This is not necessarily a problem, but should only
    //     be done with care, and after checking if talos performance regresses badly.
    assert_eq!(mem::size_of::<Image>(), 56, "Image size changed");
    assert_eq!(mem::size_of::<ImageTemplate>(), 144, "ImageTemplate size changed");
    assert_eq!(mem::size_of::<ImageKey>(), 84, "ImageKey size changed");
}
