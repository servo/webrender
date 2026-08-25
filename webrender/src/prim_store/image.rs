/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{
    AlphaType, ColorDepth, ColorF, ColorRange, ExternalImageData, ExternalImageType, ImageBufferKind, ImageKey as ApiImageKey, ImageRendering, YuvColorSpace, YuvFormat
};
use api::units::*;
use euclid::point2;
use crate::clip::{ClipChainInstance, ClipIntern};
use crate::command_buffer::CommandBufferIndex;
use crate::pattern::image::ImagePattern;
use crate::quad::{QuadDescriptor, QuadTransformState};
use crate::visibility::PrimitiveDrawIndex;
use crate::scene_building::{IsVisible};
use crate::frame_builder::{FrameBuildingContext, FrameBuildingState, PictureContext};
use crate::intern::{DataStore, Handle as InternHandle, InternDebug, Internable};
use crate::internal_types::LayoutPrimitiveInfo;
use crate::prim_store::{
    EdgeMask, InternablePrimitive, PrimKey, PrimTemplate, PrimTemplateCommonData, PrimitiveKind, PrimitiveScratchBuffer, PrimitiveStore
};
use crate::render_target::RenderTargetKind;
use crate::render_task_graph::RenderTaskId;
use crate::render_task::RenderTask;
use crate::resource_cache::ImageRequest;
use crate::visibility::compute_surface_visible_rect;
use crate::{image_tiling, quad};

// Key that identifies a unique (partial) image that is being
// stored in the render task cache.
#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct ImageCacheKey {
    pub request: ImageRequest,
    pub texel_rect: Option<DeviceIntRect>,
}

// `StretchSizeKey` now lives in `webrender_api::key_types` so builder-side
// interning keys can reference it. The resolved `StretchSize` below (and its
// frame-build `resolve`) stay here. Re-exported to keep existing references
// working.
pub use api::key_types::StretchSizeKey;

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Clone, Copy, MallocSizeOf)]
pub struct StretchSize {
    pub size: LayoutSize,
    pub fills_width: bool,
    pub fills_height: bool,
}

impl From<StretchSizeKey> for StretchSize {
    fn from(k: StretchSizeKey) -> Self {
        StretchSize {
            size: k.size.into(),
            fills_width: k.fills_width,
            fills_height: k.fills_height,
        }
    }
}

impl StretchSize {
    /// Resolve to the LayoutSize used for the GPU shader and tiling math.
    /// Per-axis: an axis flagged `fills_*` resolves to the snapped prim
    /// rect's extent on that axis; the other axis keeps the stored size.
    pub fn resolve(self, prim_rect: &LayoutRect) -> LayoutSize {
        let prim_size = prim_rect.size();
        LayoutSize::new(
            if self.fills_width { prim_size.width } else { self.size.width },
            if self.fills_height { prim_size.height } else { self.size.height },
        )
    }
}

// `Image` now lives in `webrender_api::interned_prims` so content-process
// interning can hold it. Re-exported to keep existing references working.
pub use api::interned_prims::Image;

pub type ImageKey = PrimKey<Image>;

impl InternDebug for ImageKey {}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, MallocSizeOf)]
pub struct ImageData {
    pub key: ApiImageKey,
    pub stretch_size: StretchSize,
    pub tile_spacing: LayoutSize,
    pub color: ColorF,
    pub image_rendering: ImageRendering,
    pub alpha_type: AlphaType,
}

impl From<Image> for ImageData {
    fn from(image: Image) -> Self {
        ImageData {
            key: image.key,
            color: image.color.into(),
            stretch_size: image.stretch_size.into(),
            tile_spacing: image.tile_spacing.into(),
            image_rendering: image.image_rendering,
            alpha_type: image.alpha_type,
        }
    }
}

pub fn prepare_image_quads(
    prim_rect: &LayoutRect,
    common_data: &PrimTemplateCommonData,
    image_data: &ImageData,
    clip_chain: &ClipChainInstance,
    draw_index: PrimitiveDrawIndex,
    quad_transform: &mut QuadTransformState,
    frame_context: &FrameBuildingContext,
    pic_context: &PictureContext,
    targets: &[CommandBufferIndex],
    interned_clips: &DataStore<ClipIntern>,
    frame_state: &mut FrameBuildingState,
    scratch: &mut PrimitiveScratchBuffer,
) {
    let image_properties = frame_state
        .resource_cache
        .get_image_properties(image_data.key);

    let Some(image_properties) = image_properties else {
        return;
    };

    let src_is_opaque = image_properties.descriptor.is_opaque()
        && image_data.color.a >= 0.9999;

    let premultiplied = image_data.alpha_type == AlphaType::PremultipliedAlpha;

    // The coverage rect rather than the clip rect, because decomposing the
    // repeated image can produce primitives that only partially cover the
    // original image rect and we want to clip these extra parts out.
    // We also rely on it being tight in some cases other than tiled/repeated
    // images, for example when rendering a snapshot image where the snapshot
    // area is tighter than the rasterized area.
    let tight_clip_rect = clip_chain.local_coverage_rect;

    let request = ImageRequest {
        key: image_data.key,
        rendering: image_data.image_rendering,
        tile: None,
    };

    let mut sampler_kind = ImageBufferKind::Texture2D;
    if let Some(ExternalImageData { image_type: ExternalImageType::TextureHandle(kind), .. }) = image_properties.external_image {
        sampler_kind = kind;
    }


    match image_properties.tiling {
        // Non-tiled (most common) path.
        None => {
            let size = frame_state.resource_cache.request_image(
                request,
                &mut frame_state.frame_gpu_data.f32,
            );

            let effective_stretch_size = image_data.stretch_size.resolve(prim_rect);
            let prim_rect = image_properties.adjustment.map_local_rect(&prim_rect);
            let stretch_size = image_properties.adjustment.map_stretch_size(effective_stretch_size);

            let mut src_task_id = frame_state.rg_builder.add().init(
                RenderTask::new_image(size, request, false)
            );

            if let Some(external_image) = image_properties.external_image {
                // On some devices we cannot render from an ImageBufferKind::TextureExternal
                // source using most shaders, so must perform a copy to a regular texture first.
                let requires_copy = frame_context.fb_config.external_images_require_copy
                    && external_image.image_type
                        == ExternalImageType::TextureHandle(ImageBufferKind::TextureExternal);

                if requires_copy {
                    let target_kind = if image_properties.descriptor.format.bytes_per_pixel() == 1 {
                        RenderTargetKind::Alpha
                    } else {
                        RenderTargetKind::Color
                    };

                    src_task_id = RenderTask::new_scaling(
                        src_task_id,
                        frame_state.rg_builder,
                        target_kind,
                        size,
                    );

                    frame_state.surface_builder.add_child_render_task(
                        src_task_id,
                        frame_state.rg_builder,
                    );

                    sampler_kind = ImageBufferKind::Texture2D;
                }
            }

            let image_pattern = ImagePattern {
                src_task_id,
                src_is_opaque,
                premultiplied,
                sampler_kind,
                color: image_data.color,
            };

            quad::prepare_repeatable_quad(
                &image_pattern,
                &QuadDescriptor {
                    pattern_rect: prim_rect,
                    bounds: tight_clip_rect.intersection_unchecked(&prim_rect),
                    aligned_aa_edges: common_data.aligned_aa_edges,
                    transformed_aa_edges: common_data.transformed_aa_edges,
                },
                stretch_size,
                image_data.tile_spacing,
                draw_index,
                &None,
                clip_chain,
                quad_transform,
                frame_context,
                pic_context,
                targets,
                interned_clips,
                frame_state,
                scratch,
            );
        }
        Some(tile_size) => {
            // TODO: rename the blob's visible_rect into something that doesn't conflict
            // with the terminology we use during culling since it's not really the same
            // thing.
            let active_rect = image_properties.visible_rect;
            let visible_rect = compute_surface_visible_rect(
                &frame_state.surfaces[pic_context.surface_index.0],
                clip_chain,
                quad_transform.prim_spatial_node_index(),
                &tight_clip_rect,
                frame_context.spatial_tree,
            );

            let effective_stretch_size = image_data.stretch_size.resolve(prim_rect);
            let stride = effective_stretch_size + image_data.tile_spacing;

            let repetitions = image_tiling::repetitions(prim_rect, &visible_rect, stride);

            let base_edge_flags = edge_flags_for_tile_spacing(&image_data.tile_spacing);

            for image_tiling::Repetition { origin, edge_flags } in repetitions {
                let rep_edge_flags = base_edge_flags & edge_flags;

                let layout_image_rect = LayoutRect::from_origin_and_size(
                    origin,
                    effective_stretch_size,
                );

                let tiles = image_tiling::tiles(
                    &layout_image_rect,
                    &visible_rect,
                    &active_rect,
                    tile_size as i32,
                );

                for tile in tiles {
                    let request = request.with_tile(tile.offset);
                    let size = frame_state.resource_cache.request_image(
                        request,
                        &mut frame_state.frame_gpu_data.f32,
                    );

                    let tile_edge_flags = rep_edge_flags & tile.edge_flags;
                    let aligned_aa_edges = tile_edge_flags & common_data.aligned_aa_edges;
                    let transformed_aa_edges = tile_edge_flags & common_data.transformed_aa_edges;

                    let src_task_id = frame_state.rg_builder.add().init(
                        RenderTask::new_image(size, request, false)
                    );

                    let image_pattern = ImagePattern {
                        src_task_id,
                        src_is_opaque,
                        premultiplied,
                        sampler_kind,
                        color: image_data.color,
                    };

                    quad::prepare_quad(
                        &image_pattern,
                        &QuadDescriptor {
                            pattern_rect: tile.rect,
                            bounds: tight_clip_rect.intersection_unchecked(&tile.rect),
                            aligned_aa_edges,
                            transformed_aa_edges,
                        },
                        draw_index,
                        &None,
                        clip_chain,
                        quad_transform,
                        frame_context,
                        pic_context,
                        targets,
                        interned_clips,
                        frame_state,
                        scratch,
                    );
                }
            }
        }
    }
}

fn edge_flags_for_tile_spacing(tile_spacing: &LayoutSize) -> EdgeMask {
    let mut flags = EdgeMask::empty();

    if tile_spacing.width > 0.0 {
        flags |= EdgeMask::LEFT | EdgeMask::RIGHT;
    }
    if tile_spacing.height > 0.0 {
        flags |= EdgeMask::TOP | EdgeMask::BOTTOM;
    }

    flags
}

pub type ImageTemplate = PrimTemplate<ImageData>;

impl From<ImageKey> for ImageTemplate {
    fn from(image: ImageKey) -> Self {
        let common = PrimTemplateCommonData::with_key_common(image.common);

        ImageTemplate {
            common,
            kind: image.kind.into(),
        }
    }
}

pub type ImageDataHandle = InternHandle<Image>;

impl Internable for Image {
    type Key = ImageKey;
    type StoreData = ImageTemplate;
    type InternData = ();
    const PROFILE_COUNTER: usize = crate::profiler::INTERNED_IMAGES;
}

impl InternablePrimitive for Image {
    fn into_key(
        self,
        info: &LayoutPrimitiveInfo,
    ) -> ImageKey {
        ImageKey::new(info.into(), self)
    }

    fn make_instance_kind(
        _key: ImageKey,
        data_handle: ImageDataHandle,
        _prim_store: &mut PrimitiveStore,
    ) -> PrimitiveKind {
        PrimitiveKind::Image {
            data_handle,
        }
    }
}


impl IsVisible for Image {
    fn is_visible(&self) -> bool {
        true
    }
}

/// Represents an adjustment to apply to an image primitive.
/// This can be used to compensate for a difference between the bounds of
/// the images expected by the primitive and the bounds that were actually
/// drawn in the texture cache.
///
/// This happens when rendering snapshot images: A picture is marked so that
/// a specific reference area in layout space can be rendered as an image.
/// However, the bounds of the rasterized area of the picture typically differ
/// from that reference area.
///
/// The adjustment is stored as 4 floats (x0, y0, x1, y1) that represent a
/// transformation of the primitve's local rect such that:
///
/// ```ignore
/// adjusted_rect.min = prim_rect.min + prim_rect.size() * (x0, y0);
/// adjusted_rect.max = prim_rect.max + prim_rect.size() * (x1, y1);
/// ```
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct AdjustedImageSource {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

impl AdjustedImageSource {
    /// The "identity" adjustment.
    pub fn new() -> Self {
        AdjustedImageSource {
            x0: 0.0,
            y0: 0.0,
            x1: 0.0,
            y1: 0.0,
        }
    }

    /// An adjustment to render an image item defined in function of the `reference`
    /// rect whereas the `actual` rect was cached instead.
    pub fn from_rects(reference: &LayoutRect, actual: &LayoutRect) -> Self {
        let ref_size = reference.size();
        let min_offset = reference.min.to_vector();
        let max_offset = reference.max.to_vector();
        AdjustedImageSource {
            x0: (actual.min.x - min_offset.x) / ref_size.width,
            y0: (actual.min.y - min_offset.y) / ref_size.height,
            x1: (actual.max.x - max_offset.x) / ref_size.width,
            y1: (actual.max.y - max_offset.y) / ref_size.height,
        }
    }

    /// Adjust the primitive's local rect.
    pub fn map_local_rect(&self, rect: &LayoutRect) -> LayoutRect {
        let w = rect.width();
        let h = rect.height();
        LayoutRect {
            min: point2(
                rect.min.x + w * self.x0,
                rect.min.y + h * self.y0,
            ),
            max: point2(
                rect.max.x + w * self.x1,
                rect.max.y + h * self.y1,
            ),
        }
    }

    /// The stretch size has to be adjusted as well because it is defined
    /// using the snapshot area as reference but will stretch the rasterized
    /// area instead.
    ///
    /// It has to be scaled by a factor of (adjusted.size() / prim_rect.size()).
    /// We derive the formula in function of the adjustment factors:
    ///
    /// ```ignore
    /// factor = (adjusted.max - adjusted.min) / (w, h)
    ///        = (rect.max + (w, h) * (x1, y1) - (rect.min + (w, h) * (x0, y0))) / (w, h)
    ///        = ((w, h) + (w, h) * (x1, y1) - (w, h) * (x0, y0)) / (w, h)
    ///        = (1.0, 1.0) + (x1, y1) - (x0, y0)
    /// ```
    pub fn map_stretch_size(&self, size: LayoutSize) -> LayoutSize {
        LayoutSize::new(
            size.width * (1.0 + self.x1 - self.x0),
            size.height * (1.0 + self.y1 - self.y0),
        )
    }
}

////////////////////////////////////////////////////////////////////////////////

// `YuvImage` now lives in `webrender_api::interned_prims` so content-process
// interning can hold it. Re-exported to keep existing references working.
pub use api::interned_prims::YuvImage;

pub type YuvImageKey = PrimKey<YuvImage>;

impl InternDebug for YuvImageKey {}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(MallocSizeOf)]
pub struct YuvImageData {
    pub color_depth: ColorDepth,
    pub yuv_key: [ApiImageKey; 3],
    pub src_yuv: [Option<RenderTaskId>; 3],
    pub format: YuvFormat,
    pub color_space: YuvColorSpace,
    pub color_range: ColorRange,
    pub image_rendering: ImageRendering,
}

impl From<YuvImage> for YuvImageData {
    fn from(image: YuvImage) -> Self {
        YuvImageData {
            color_depth: image.color_depth,
            yuv_key: image.yuv_key,
            src_yuv: [None, None, None],
            format: image.format,
            color_space: image.color_space,
            color_range: image.color_range,
            image_rendering: image.image_rendering,
        }
    }
}

impl YuvImageData {
    /// Update the GPU cache for a given primitive template. This may be called multiple
    /// times per frame, by each primitive reference that refers to this interned
    /// template. The initial request call to the GPU cache ensures that work is only
    /// done if the cache entry is invalid (due to first use or eviction).
    pub fn update(
        &self,
        is_composited: bool,
        frame_state: &mut FrameBuildingState,
    ) -> [RenderTaskId; 3] {

        let mut src_yuv = [ RenderTaskId::INVALID; 3 ];

        let channel_num = self.format.get_plane_num();
        debug_assert!(channel_num <= 3);
        for channel in 0 .. channel_num {
            let request = ImageRequest {
                key: self.yuv_key[channel],
                rendering: self.image_rendering,
                tile: None,
            };

            let size = frame_state.resource_cache.request_image(
                request,
                &mut frame_state.frame_gpu_data.f32,
            );

            let task_id = frame_state.rg_builder.add().init(
                RenderTask::new_image(
                    size,
                    request,
                    is_composited,
                )
            );

            src_yuv[channel] = task_id;
        }

        src_yuv
    }
}

pub type YuvImageTemplate = PrimTemplate<YuvImageData>;

impl From<YuvImageKey> for YuvImageTemplate {
    fn from(image: YuvImageKey) -> Self {
        let common = PrimTemplateCommonData::with_key_common(image.common);

        YuvImageTemplate {
            common,
            kind: image.kind.into(),
        }
    }
}

pub type YuvImageDataHandle = InternHandle<YuvImage>;

impl Internable for YuvImage {
    type Key = YuvImageKey;
    type StoreData = YuvImageTemplate;
    type InternData = ();
    const PROFILE_COUNTER: usize = crate::profiler::INTERNED_YUV_IMAGES;
}

impl InternablePrimitive for YuvImage {
    fn into_key(
        self,
        info: &LayoutPrimitiveInfo,
    ) -> YuvImageKey {
        YuvImageKey::new(info.into(), self)
    }

    fn make_instance_kind(
        _key: YuvImageKey,
        data_handle: YuvImageDataHandle,
        _prim_store: &mut PrimitiveStore,
    ) -> PrimitiveKind {
        PrimitiveKind::YuvImage {
            data_handle,
        }
    }
}

impl IsVisible for YuvImage {
    fn is_visible(&self) -> bool {
        true
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
fn test_struct_sizes() {
    use std::mem;
    // The sizes of these structures are critical for performance on a number of
    // talos stress tests. If you get a failure here on CI, there's two possibilities:
    // (a) You made a structure smaller than it currently is. Great work! Update the
    //     test expectations and move on.
    // (b) You made a structure larger. This is not necessarily a problem, but should only
    //     be done with care, and after checking if talos performance regresses badly.
    assert_eq!(mem::size_of::<Image>(), 36, "Image size changed");
    assert_eq!(mem::size_of::<ImageTemplate>(), 52, "ImageTemplate size changed");
    assert_eq!(mem::size_of::<ImageKey>(), 40, "ImageKey size changed");
    assert_eq!(mem::size_of::<YuvImage>(), 32, "YuvImage size changed");
    assert_eq!(mem::size_of::<YuvImageTemplate>(), 72, "YuvImageTemplate size changed");
    assert_eq!(mem::size_of::<YuvImageKey>(), 36, "YuvImageKey size changed");
}
