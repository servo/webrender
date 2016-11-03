/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use device::TextureId;
use euclid::{Point2D, Matrix4D, Rect, Size2D};
use gpu_store::{GpuStore, GpuStoreAddress};
use internal_types::{DeviceRect, DeviceSize};
use renderer::BLUR_INFLATION_FACTOR;
use resource_cache::ResourceCache;
use std::mem;
use std::usize;
use tiling::{Clip, MaskImageSource};
use util::TransformedRect;
use webrender_traits::{AuxiliaryLists, ColorF, ImageKey, ImageRendering, WebGLContextId};
use webrender_traits::{FontKey, ItemRange, ComplexClipRegion, GlyphKey};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct SpecificPrimitiveIndex(pub usize);

impl SpecificPrimitiveIndex {
    pub fn invalid() -> SpecificPrimitiveIndex {
        SpecificPrimitiveIndex(usize::MAX)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PrimitiveIndex(pub usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PrimitiveKind {
    Rectangle,
    TextRun,
    Image,
    Border,
    Gradient,
    BoxShadow,
}

/// Geometry description for simple rectangular primitives, uploaded to the GPU.
#[derive(Debug, Clone)]
pub struct PrimitiveGeometry {
    pub local_rect: Rect<f32>,
    pub local_clip_rect: Rect<f32>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum PrimitiveCacheKey {
    BoxShadow(BoxShadowPrimitiveCacheKey),
}

#[derive(Debug)]
pub struct PrimitiveCacheInfo {
    pub size: DeviceSize,
    pub key: PrimitiveCacheKey,
}

// TODO(gw): Pack the fields here better!
#[derive(Debug)]
pub struct PrimitiveMetadata {
    pub is_opaque: bool,
    pub mask_texture_id: TextureId,
    pub mask_image: Option<ImageKey>,
    pub clip_index: Option<GpuStoreAddress>,
    pub prim_kind: PrimitiveKind,
    pub cpu_prim_index: SpecificPrimitiveIndex,
    pub gpu_prim_index: GpuStoreAddress,
    pub gpu_data_address: GpuStoreAddress,
    pub gpu_data_count: i32,
    pub cache_info: Option<PrimitiveCacheInfo>,
}

#[derive(Debug, Clone)]
pub struct RectanglePrimitive {
    pub color: ColorF,
}

#[derive(Debug)]
pub enum ImagePrimitiveKind {
    Image(ImageKey, ImageRendering, Size2D<f32>),
    WebGL(WebGLContextId),
}

#[derive(Debug)]
pub struct ImagePrimitiveCpu {
    pub kind: ImagePrimitiveKind,
    pub color_texture_id: TextureId,
}

#[derive(Debug, Clone)]
pub struct ImagePrimitiveGpu {
    pub uv0: Point2D<f32>,
    pub uv1: Point2D<f32>,
    pub stretch_size: Size2D<f32>,
    pub tile_spacing: Size2D<f32>,
}

#[derive(Debug, Clone)]
pub struct BorderPrimitiveCpu {
    pub inner_rect: Rect<f32>,
}

#[derive(Debug, Clone)]
pub struct BorderPrimitiveGpu {
    pub style: [f32; 4],
    pub widths: [f32; 4],
    pub colors: [ColorF; 4],
    pub radii: [Size2D<f32>; 4],
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct BoxShadowPrimitiveCacheKey {
    pub shadow_rect_size: Size2D<Au>,
    pub border_radius: Au,
    pub blur_radius: Au,
    pub inverted: bool,
}

#[derive(Debug, Clone)]
pub struct BoxShadowPrimitiveGpu {
    pub src_rect: Rect<f32>,
    pub bs_rect: Rect<f32>,
    pub color: ColorF,
    pub border_radius: f32,
    pub edge_size: f32,
    pub blur_radius: f32,
    pub inverted: f32,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum GradientType {
    Horizontal,
    Vertical,
    Rotated,
}

#[derive(Debug, Clone)]
pub struct GradientStop {
    color: ColorF,
    offset: f32,
    padding: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct GradientPrimitiveGpu {
    pub start_point: Point2D<f32>,
    pub end_point: Point2D<f32>,
    pub kind: f32,
    pub padding: [f32; 3],
}

#[derive(Debug)]
pub struct GradientPrimitiveCpu {
    pub stops_range: ItemRange,
    pub kind: GradientType,
    pub reverse_stops: bool,
    pub cache_dirty: bool,
}

#[derive(Debug, Clone)]
struct InstanceRect {
    rect: Rect<f32>,
}

#[derive(Debug, Clone)]
pub struct TextRunPrimitiveGpu {
    pub color: ColorF,
}

#[derive(Debug, Clone)]
pub struct TextRunPrimitiveCpu {
    pub font_key: FontKey,
    pub font_size: Au,
    pub blur_radius: Au,
    pub glyph_range: ItemRange,
    pub cache_dirty: bool,
    // TODO(gw): Maybe make this an Arc for sharing with resource cache
    pub glyph_indices: Vec<u32>,
    pub color_texture_id: TextureId,
}

#[derive(Debug, Clone)]
struct GlyphPrimitive {
    offset: Point2D<f32>,
    padding: Point2D<f32>,
    uv0: Point2D<f32>,
    uv1: Point2D<f32>,
}

#[derive(Debug, Clone)]
struct ClipRect {
    rect: Rect<f32>,
    padding: [f32; 4],
}

#[derive(Debug, Clone)]
struct ClipCorner {
    rect: Rect<f32>,
    outer_radius_x: f32,
    outer_radius_y: f32,
    inner_radius_x: f32,
    inner_radius_y: f32,
}

impl ClipCorner {
    fn uniform(rect: Rect<f32>, outer_radius: f32, inner_radius: f32) -> ClipCorner {
        ClipCorner {
            rect: rect,
            outer_radius_x: outer_radius,
            outer_radius_y: outer_radius,
            inner_radius_x: inner_radius,
            inner_radius_y: inner_radius,
        }
    }
}

#[derive(Debug, Clone)]
struct ImageMaskInfo {
    uv_rect: Rect<f32>,
    local_rect: Rect<f32>,
}

#[derive(Debug, Clone)]
pub struct ClipInfo {
    rect: ClipRect,
    top_left: ClipCorner,
    top_right: ClipCorner,
    bottom_left: ClipCorner,
    bottom_right: ClipCorner,
    mask_info: ImageMaskInfo,
}

impl ClipInfo {
    pub fn from_clip_region(clip: &ComplexClipRegion) -> ClipInfo {
        ClipInfo {
            rect: ClipRect {
                rect: clip.rect,
                padding: [0.0, 0.0, 0.0, 0.0],
            },
            top_left: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x, clip.rect.origin.y),
                                Size2D::new(clip.radii.top_left.width, clip.radii.top_left.height)),
                outer_radius_x: clip.radii.top_left.width,
                outer_radius_y: clip.radii.top_left.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            top_right: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x + clip.rect.size.width - clip.radii.top_right.width,
                                             clip.rect.origin.y),
                                Size2D::new(clip.radii.top_right.width, clip.radii.top_right.height)),
                outer_radius_x: clip.radii.top_right.width,
                outer_radius_y: clip.radii.top_right.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            bottom_left: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x,
                                             clip.rect.origin.y + clip.rect.size.height - clip.radii.bottom_left.height),
                                Size2D::new(clip.radii.bottom_left.width, clip.radii.bottom_left.height)),
                outer_radius_x: clip.radii.bottom_left.width,
                outer_radius_y: clip.radii.bottom_left.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            bottom_right: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x + clip.rect.size.width - clip.radii.bottom_right.width,
                                             clip.rect.origin.y + clip.rect.size.height - clip.radii.bottom_right.height),
                                Size2D::new(clip.radii.bottom_right.width, clip.radii.bottom_right.height)),
                outer_radius_x: clip.radii.bottom_right.width,
                outer_radius_y: clip.radii.bottom_right.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            mask_info: ImageMaskInfo {
                uv_rect: Rect::zero(),
                local_rect: Rect::zero(),
            },
        }
    }

    pub fn uniform(rect: Rect<f32>, radius: f32) -> ClipInfo {
        ClipInfo {
            rect: ClipRect {
                rect: rect,
                padding: [0.0; 4],
            },
            top_left: ClipCorner::uniform(Rect::new(Point2D::new(rect.origin.x,
                                                                 rect.origin.y),
                                                    Size2D::new(radius, radius)),
                                          radius,
                                          0.0),
            top_right: ClipCorner::uniform(Rect::new(Point2D::new(rect.origin.x + rect.size.width - radius,
                                                                  rect.origin.y),
                                                    Size2D::new(radius, radius)),
                                           radius,
                                           0.0),
            bottom_left: ClipCorner::uniform(Rect::new(Point2D::new(rect.origin.x,
                                                                    rect.origin.y + rect.size.height - radius),
                                                       Size2D::new(radius, radius)),
                                             radius,
                                             0.0),
            bottom_right: ClipCorner::uniform(Rect::new(Point2D::new(rect.origin.x + rect.size.width - radius,
                                                                     rect.origin.y + rect.size.height - radius),
                                                        Size2D::new(radius, radius)),
                                              radius,
                                              0.0),
            mask_info: ImageMaskInfo {
                uv_rect: Rect::zero(),
                local_rect: Rect::zero(),
            },
        }
    }

    pub fn with_mask(self, uv_rect: Rect<f32>, local_rect: Rect<f32>) -> ClipInfo {
        ClipInfo {
            mask_info: ImageMaskInfo {
                uv_rect: uv_rect,
                local_rect: local_rect,
            },
            .. self
        }
    }
}

#[derive(Debug)]
pub enum PrimitiveContainer {
    Rectangle(RectanglePrimitive),
    TextRun(TextRunPrimitiveCpu, TextRunPrimitiveGpu),
    Image(ImagePrimitiveCpu, ImagePrimitiveGpu),
    Border(BorderPrimitiveCpu, BorderPrimitiveGpu),
    Gradient(GradientPrimitiveCpu, GradientPrimitiveGpu),
    BoxShadow(BoxShadowPrimitiveGpu, Vec<Rect<f32>>),
}

pub struct PrimitiveStore {
    // CPU side information only
    pub cpu_bounding_rects: Vec<Option<DeviceRect>>,
    pub cpu_text_runs: Vec<TextRunPrimitiveCpu>,
    pub cpu_images: Vec<ImagePrimitiveCpu>,
    pub cpu_gradients: Vec<GradientPrimitiveCpu>,
    pub cpu_metadata: Vec<PrimitiveMetadata>,
    pub cpu_borders: Vec<BorderPrimitiveCpu>,

    // Gets uploaded directly to GPU via vertex texture
    pub gpu_geometry: GpuStore<PrimitiveGeometry>,
    pub gpu_data16: GpuStore<GpuBlock16>,
    pub gpu_data32: GpuStore<GpuBlock32>,
    pub gpu_data64: GpuStore<GpuBlock64>,
    pub gpu_data128: GpuStore<GpuBlock128>,

    // General
    device_pixel_ratio: f32,
    prims_to_resolve: Vec<PrimitiveIndex>,
}

impl PrimitiveStore {
    pub fn new(device_pixel_ratio: f32) -> PrimitiveStore {
        PrimitiveStore {
            cpu_metadata: Vec::new(),
            cpu_bounding_rects: Vec::new(),
            cpu_text_runs: Vec::new(),
            cpu_images: Vec::new(),
            cpu_gradients: Vec::new(),
            cpu_borders: Vec::new(),
            gpu_geometry: GpuStore::new(),
            gpu_data16: GpuStore::new(),
            gpu_data32: GpuStore::new(),
            gpu_data64: GpuStore::new(),
            gpu_data128: GpuStore::new(),
            device_pixel_ratio: device_pixel_ratio,
            prims_to_resolve: Vec::new(),
        }
    }

    fn populate_clip_data(&mut self, address: GpuStoreAddress, clip: ClipInfo) {
        let data = self.gpu_data32.get_slice_mut(address, 6);
        data[0] = GpuBlock32::from(clip.rect);
        data[1] = GpuBlock32::from(clip.top_left);
        data[2] = GpuBlock32::from(clip.top_right);
        data[3] = GpuBlock32::from(clip.bottom_left);
        data[4] = GpuBlock32::from(clip.bottom_right);
        data[5] = GpuBlock32::from(clip.mask_info);
    }

    pub fn add_primitive(&mut self,
                         rect: &Rect<f32>,
                         clip_rect: &Rect<f32>,
                         clip: Option<Clip>,
                         container: PrimitiveContainer) -> PrimitiveIndex {
        let prim_index = self.cpu_metadata.len();

        self.cpu_bounding_rects.push(None);

        self.gpu_geometry.push(PrimitiveGeometry {
            local_rect: *rect,
            local_clip_rect: *clip_rect,
        });

        let (clip_index, (mask_image, mask_texture_id)) = if let Some(ref masked) = clip {
            let clip = masked.clip.as_ref().clone();
            // TODO(gw): This is slightly inefficient. It
            // pushes default data on when we already have
            // the data we need to push on available now.
            let gpu_address = self.gpu_data32.alloc(6);
            self.populate_clip_data(gpu_address, clip);
            let mask = match masked.mask {
                MaskImageSource::User(image_key) => (Some(image_key), TextureId::invalid()),
                MaskImageSource::Renderer(texture_id) => (None, texture_id),
            };
            (Some(gpu_address), mask)
        } else {
            (None, (None, TextureId::invalid()))
        };

        let metadata = match container {
            PrimitiveContainer::Rectangle(rect) => {
                let is_opaque = rect.color.a == 1.0;
                let gpu_address = self.gpu_data16.push(rect);

                let metadata = PrimitiveMetadata {
                    is_opaque: is_opaque,
                    mask_image: mask_image,
                    mask_texture_id: mask_texture_id,
                    clip_index: clip_index,
                    prim_kind: PrimitiveKind::Rectangle,
                    cpu_prim_index: SpecificPrimitiveIndex::invalid(),
                    gpu_prim_index: gpu_address,
                    gpu_data_address: GpuStoreAddress(0),
                    gpu_data_count: 0,
                    cache_info: None,
                };

                metadata
            }
            PrimitiveContainer::TextRun(text_cpu, text_gpu) => {
                let gpu_address = self.gpu_data16.push(text_gpu);
                let gpu_glyphs_address = self.gpu_data32.alloc(text_cpu.glyph_range.length);

                let metadata = PrimitiveMetadata {
                    is_opaque: false,
                    mask_image: mask_image,
                    mask_texture_id: mask_texture_id,
                    clip_index: clip_index,
                    prim_kind: PrimitiveKind::TextRun,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_text_runs.len()),
                    gpu_prim_index: gpu_address,
                    gpu_data_address: gpu_glyphs_address,
                    gpu_data_count: text_cpu.glyph_range.length as i32,
                    cache_info: None,
                };

                self.cpu_text_runs.push(text_cpu);
                metadata
            }
            PrimitiveContainer::Image(image_cpu, image_gpu) => {
                let gpu_address = self.gpu_data32.push(image_gpu);

                let metadata = PrimitiveMetadata {
                    is_opaque: false,
                    mask_image: mask_image,
                    mask_texture_id: mask_texture_id,
                    clip_index: clip_index,
                    prim_kind: PrimitiveKind::Image,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_images.len()),
                    gpu_prim_index: gpu_address,
                    gpu_data_address: GpuStoreAddress(0),
                    gpu_data_count: 0,
                    cache_info: None,
                };

                self.cpu_images.push(image_cpu);
                metadata
            }
            PrimitiveContainer::Border(border_cpu, border_gpu) => {
                let gpu_address = self.gpu_data128.push(border_gpu);

                let metadata = PrimitiveMetadata {
                    is_opaque: false,
                    mask_image: mask_image,
                    mask_texture_id: mask_texture_id,
                    clip_index: clip_index,
                    prim_kind: PrimitiveKind::Border,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_borders.len()),
                    gpu_prim_index: gpu_address,
                    gpu_data_address: GpuStoreAddress(0),
                    gpu_data_count: 0,
                    cache_info: None,
                };

                self.cpu_borders.push(border_cpu);
                metadata
            }
            PrimitiveContainer::Gradient(gradient_cpu, gradient_gpu) => {
                let gpu_address = self.gpu_data32.push(gradient_gpu);
                let gpu_stops_address = self.gpu_data32.alloc(gradient_cpu.stops_range.length);

                let metadata = PrimitiveMetadata {
                    is_opaque: false,
                    mask_image: mask_image,
                    mask_texture_id: mask_texture_id,
                    clip_index: clip_index,
                    prim_kind: PrimitiveKind::Gradient,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_gradients.len()),
                    gpu_prim_index: gpu_address,
                    gpu_data_address: gpu_stops_address,
                    gpu_data_count: gradient_cpu.stops_range.length as i32,
                    cache_info: None,
                };

                self.cpu_gradients.push(gradient_cpu);
                metadata
            }
            PrimitiveContainer::BoxShadow(box_shadow_gpu, instance_rects) => {
                // TODO(gw): Account for zoom factor!
                // Here, we calculate the size of the patch required in order
                // to create the box shadow corner. First, scale it by the
                // device pixel ratio since the cache shader expects vertices
                // in device space. The shader adds a 1-pixel border around
                // the patch, in order to prevent bilinear filter artifacts as
                // the patch is clamped / mirrored across the box shadow rect.
                let edge_size = box_shadow_gpu.edge_size.ceil() * self.device_pixel_ratio;
                let edge_size = edge_size as i32 + 2;   // Account for bilinear filtering
                let cache_size = DeviceSize::new(edge_size, edge_size);
                let cache_key = PrimitiveCacheKey::BoxShadow(BoxShadowPrimitiveCacheKey {
                    blur_radius: Au::from_f32_px(box_shadow_gpu.blur_radius),
                    border_radius: Au::from_f32_px(box_shadow_gpu.border_radius),
                    inverted: box_shadow_gpu.inverted != 0.0,
                    shadow_rect_size: Size2D::new(Au::from_f32_px(box_shadow_gpu.bs_rect.size.width),
                                                  Au::from_f32_px(box_shadow_gpu.bs_rect.size.height)),
                });

                let gpu_prim_address = self.gpu_data64.push(box_shadow_gpu);
                let gpu_data_address = self.gpu_data16.get_next_address();

                let metadata = PrimitiveMetadata {
                    is_opaque: false,
                    mask_image: mask_image,
                    mask_texture_id: mask_texture_id,
                    clip_index: clip_index,
                    prim_kind: PrimitiveKind::BoxShadow,
                    cpu_prim_index: SpecificPrimitiveIndex::invalid(),
                    gpu_prim_index: gpu_prim_address,
                    gpu_data_address: gpu_data_address,
                    gpu_data_count: instance_rects.len() as i32,
                    cache_info: Some(PrimitiveCacheInfo {
                        size: cache_size,
                        key: cache_key,
                    }),
                };

                for rect in instance_rects {
                    self.gpu_data16.push(InstanceRect {
                        rect: rect,
                    });
                }

                metadata
            }
        };

        self.cpu_metadata.push(metadata);

        PrimitiveIndex(prim_index)
    }

    pub fn resolve_primitives(&mut self, resource_cache: &ResourceCache) {
        for prim_index in self.prims_to_resolve.drain(..) {
            let metadata = &mut self.cpu_metadata[prim_index.0];

            if let Some(mask_key) = metadata.mask_image {
                let tex_cache = resource_cache.get_image(mask_key, ImageRendering::Auto);
                metadata.mask_texture_id = tex_cache.texture_id;
                if let Some(address) = metadata.clip_index {
                    let clip = self.gpu_data32.get_slice_mut(address, 6);
                    let old = clip[5].data; //TODO: avoid retaining the screen rectangle
                    clip[5] = GpuBlock32::from(ImageMaskInfo {
                        uv_rect: Rect::new(tex_cache.uv0,
                                           Size2D::new(tex_cache.uv1.x - tex_cache.uv0.x,
                                                       tex_cache.uv1.y - tex_cache.uv0.x)),
                        local_rect: Rect::new(Point2D::new(old[4], old[5]),
                                              Size2D::new(old[6], old[7])),
                    });
                }
            }

            match metadata.prim_kind {
                PrimitiveKind::Rectangle |
                PrimitiveKind::Border |
                PrimitiveKind::BoxShadow |
                PrimitiveKind::Gradient => {}
                PrimitiveKind::TextRun => {
                    let text = &mut self.cpu_text_runs[metadata.cpu_prim_index.0];

                    let dest_glyphs = self.gpu_data32.get_slice_mut(metadata.gpu_data_address,
                                                                    text.glyph_range.length);

                    let texture_id = resource_cache.get_glyphs(text.font_key,
                                                               text.font_size,
                                                               text.blur_radius,
                                                               &text.glyph_indices, |index, uv0, uv1| {
                        let dest_glyph = &mut dest_glyphs[index];
                        let dest: &mut GlyphPrimitive = unsafe {
                            mem::transmute(dest_glyph)
                        };
                        dest.uv0 = uv0;
                        dest.uv1 = uv1;
                    });

                    text.color_texture_id = texture_id;
                }
                PrimitiveKind::Image => {
                    let image_cpu = &mut self.cpu_images[metadata.cpu_prim_index.0];
                    let image_gpu: &mut ImagePrimitiveGpu = unsafe {
                        mem::transmute(self.gpu_data32.get_mut(metadata.gpu_prim_index))
                    };

                    let cache_item = match image_cpu.kind {
                        ImagePrimitiveKind::Image(image_key, image_rendering, _) => {
                            resource_cache.get_image(image_key, image_rendering)
                        }
                        ImagePrimitiveKind::WebGL(context_id) => {
                            resource_cache.get_webgl_texture(&context_id)
                        }
                    };

                    image_cpu.color_texture_id = cache_item.texture_id;
                    image_gpu.uv0 = cache_item.uv0;
                    image_gpu.uv1 = cache_item.uv1;
                }
            }
        }
    }

    pub fn get_bounding_rect(&self, index: PrimitiveIndex) -> &Option<DeviceRect> {
        &self.cpu_bounding_rects[index.0]
    }

    pub fn set_complex_clip(&mut self, index: PrimitiveIndex, clip: Option<ClipInfo>) {
        self.cpu_metadata[index.0].clip_index = match (self.cpu_metadata[index.0].clip_index, clip) {
            (Some(clip_index), Some(clip)) => {
                self.populate_clip_data(clip_index, clip);
                Some(clip_index)
            }
            (Some(..), None) => {
                // TODO(gw): Add to clip free list!
                None
            }
            (None, Some(clip)) => {
                // TODO(gw): Pull from clip free list!
                let gpu_address = self.gpu_data32.alloc(6);
                self.populate_clip_data(gpu_address, clip);
                Some(gpu_address)
            }
            (None, None) => None,
        }
    }

    pub fn get_metadata(&self, index: PrimitiveIndex) -> &PrimitiveMetadata {
        &self.cpu_metadata[index.0]
    }

    pub fn prim_count(&self) -> usize {
        self.cpu_metadata.len()
    }

    pub fn build_bounding_rect(&mut self,
                               prim_index: PrimitiveIndex,
                               screen_rect: &DeviceRect,
                               layer_transform: &Matrix4D<f32>,
                               layer_combined_local_clip_rect: &Rect<f32>,
                               device_pixel_ratio: f32) -> bool {
        let geom = &self.gpu_geometry.get(GpuStoreAddress(prim_index.0 as i32));

        let bounding_rect = geom.local_rect
                                .intersection(&geom.local_clip_rect)
                                .and_then(|rect| rect.intersection(layer_combined_local_clip_rect))
                                .and_then(|ref local_rect| {
            let xf_rect = TransformedRect::new(local_rect,
                                               layer_transform,
                                               device_pixel_ratio);
            xf_rect.bounding_rect.intersection(screen_rect)
        });

        self.cpu_bounding_rects[prim_index.0] = bounding_rect;
        bounding_rect.is_some()
    }

    pub fn prepare_prim_for_render(&mut self,
                                   prim_index: PrimitiveIndex,
                                   resource_cache: &mut ResourceCache,
                                   device_pixel_ratio: f32,
                                   auxiliary_lists: &AuxiliaryLists) -> bool {
        let metadata = &mut self.cpu_metadata[prim_index.0];

        let mut prim_needs_resolve = false;
        let mut rebuild_bounding_rect = false;

        if let Some(mask_key) = metadata.mask_image {
            resource_cache.request_image(mask_key, ImageRendering::Auto);
            prim_needs_resolve = true;
        }

        match metadata.prim_kind {
            PrimitiveKind::Rectangle |
            PrimitiveKind::Border |
            PrimitiveKind::BoxShadow => {}
            PrimitiveKind::TextRun => {
                let text = &mut self.cpu_text_runs[metadata.cpu_prim_index.0];
                prim_needs_resolve = true;

                if text.cache_dirty {
                    rebuild_bounding_rect = true;
                    text.cache_dirty = false;

                    debug_assert!(metadata.gpu_data_count == text.glyph_range.length as i32);
                    debug_assert!(text.glyph_indices.is_empty());
                    let src_glyphs = auxiliary_lists.glyph_instances(&text.glyph_range);
                    let dest_glyphs = self.gpu_data32.get_slice_mut(metadata.gpu_data_address,
                                                                    text.glyph_range.length);
                    let mut glyph_key = GlyphKey::new(text.font_key,
                                                      text.font_size,
                                                      text.blur_radius,
                                                      src_glyphs[0].index);
                    let blur_offset = text.blur_radius.to_f32_px() *
                        (BLUR_INFLATION_FACTOR as f32) / 2.0;
                    let mut local_rect = Rect::zero();
                    let mut actual_glyph_count = 0;

                    for src in src_glyphs {
                        glyph_key.index = src.index;

                        let dimensions = match resource_cache.get_glyph_dimensions(&glyph_key) {
                            None => continue,
                            Some(dimensions) => dimensions,
                        };

                        // TODO(gw): Check for this and ensure platforms return None in this case!!!
                        debug_assert!(dimensions.width > 0 && dimensions.height > 0);

                        let x = src.x + dimensions.left as f32 / device_pixel_ratio - blur_offset;
                        let y = src.y - dimensions.top as f32 / device_pixel_ratio - blur_offset;

                        let width = dimensions.width as f32 / device_pixel_ratio;
                        let height = dimensions.height as f32 / device_pixel_ratio;

                        let local_glyph_rect = Rect::new(Point2D::new(x, y),
                                                         Size2D::new(width, height));
                        local_rect = local_rect.union(&local_glyph_rect);

                        dest_glyphs[actual_glyph_count] = GpuBlock32::from(GlyphPrimitive {
                            uv0: Point2D::zero(),
                            uv1: Point2D::zero(),
                            padding: Point2D::zero(),
                            offset: local_glyph_rect.origin,
                        });

                        text.glyph_indices.push(src.index);

                        actual_glyph_count += 1;
                    }

                    metadata.gpu_data_count = actual_glyph_count as i32;
                    self.gpu_geometry.get_mut(GpuStoreAddress(prim_index.0 as i32)).local_rect = local_rect;
                }

                resource_cache.request_glyphs(text.font_key,
                                              text.font_size,
                                              text.blur_radius,
                                              &text.glyph_indices);
            }
            PrimitiveKind::Image => {
                let image_cpu = &mut self.cpu_images[metadata.cpu_prim_index.0];
                prim_needs_resolve = true;

                match image_cpu.kind {
                    ImagePrimitiveKind::Image(image_key, image_rendering, tile_spacing) => {
                        resource_cache.request_image(image_key, image_rendering);

                        // TODO(gw): This doesn't actually need to be calculated each frame.
                        // It's cheap enough that it's not worth introducing a cache for images
                        // right now, but if we introduce a cache for images for some other
                        // reason then we might as well cache this with it.
                        let image_properties = resource_cache.get_image_properties(image_key);
                        metadata.is_opaque = image_properties.is_opaque &&
                                             tile_spacing.width == 0.0 &&
                                             tile_spacing.height == 0.0;
                    }
                    ImagePrimitiveKind::WebGL(..) => {}
                }
            }
            PrimitiveKind::Gradient => {
                let gradient = &mut self.cpu_gradients[metadata.cpu_prim_index.0];
                if gradient.cache_dirty {
                    let src_stops = auxiliary_lists.gradient_stops(&gradient.stops_range);

                    debug_assert!(metadata.gpu_data_count == gradient.stops_range.length as i32);
                    let dest_stops = self.gpu_data32.get_slice_mut(metadata.gpu_data_address,
                                                                   gradient.stops_range.length);

                    if gradient.reverse_stops {
                        for (src, dest) in src_stops.iter().rev().zip(dest_stops.iter_mut()) {
                            *dest = GpuBlock32::from(GradientStop {
                                offset: 1.0 - src.offset,
                                color: src.color,
                                padding: [0.0; 3],
                            });
                        }
                    } else {
                        for (src, dest) in src_stops.iter().zip(dest_stops.iter_mut()) {
                            *dest = GpuBlock32::from(GradientStop {
                                offset: src.offset,
                                color: src.color,
                                padding: [0.0; 3],
                            });
                        }
                    }

                    gradient.cache_dirty = false;
                }
            }
        }

        if prim_needs_resolve {
            self.prims_to_resolve.push(prim_index);
        }

        rebuild_bounding_rect
    }
}

#[derive(Clone)]
pub struct GpuBlock16 {
    data: [f32; 4],
}

impl Default for GpuBlock16 {
    fn default() -> GpuBlock16 {
        GpuBlock16 {
            data: unsafe { mem::uninitialized() }
        }
    }
}

impl From<TextRunPrimitiveGpu> for GpuBlock16 {
    fn from(data: TextRunPrimitiveGpu) -> GpuBlock16 {
        unsafe {
            mem::transmute::<TextRunPrimitiveGpu, GpuBlock16>(data)
        }
    }
}

impl From<RectanglePrimitive> for GpuBlock16 {
    fn from(data: RectanglePrimitive) -> GpuBlock16 {
        unsafe {
            mem::transmute::<RectanglePrimitive, GpuBlock16>(data)
        }
    }
}

impl From<InstanceRect> for GpuBlock16 {
    fn from(data: InstanceRect) -> GpuBlock16 {
        unsafe {
            mem::transmute::<InstanceRect, GpuBlock16>(data)
        }
    }
}

#[derive(Clone)]
pub struct GpuBlock32 {
    data: [f32; 8],
}

impl Default for GpuBlock32 {
    fn default() -> GpuBlock32 {
        GpuBlock32 {
            data: unsafe { mem::uninitialized() }
        }
    }
}

impl From<GradientPrimitiveGpu> for GpuBlock32 {
    fn from(data: GradientPrimitiveGpu) -> GpuBlock32 {
        unsafe {
            mem::transmute::<GradientPrimitiveGpu, GpuBlock32>(data)
        }
    }
}

impl From<GradientStop> for GpuBlock32 {
    fn from(data: GradientStop) -> GpuBlock32 {
        unsafe {
            mem::transmute::<GradientStop, GpuBlock32>(data)
        }
    }
}

impl From<GlyphPrimitive> for GpuBlock32 {
    fn from(data: GlyphPrimitive) -> GpuBlock32 {
        unsafe {
            mem::transmute::<GlyphPrimitive, GpuBlock32>(data)
        }
    }
}

impl From<ImagePrimitiveGpu> for GpuBlock32 {
    fn from(data: ImagePrimitiveGpu) -> GpuBlock32 {
        unsafe {
            mem::transmute::<ImagePrimitiveGpu, GpuBlock32>(data)
        }
    }
}

impl From<ClipRect> for GpuBlock32 {
    fn from(data: ClipRect) -> GpuBlock32 {
        unsafe {
            mem::transmute::<ClipRect, GpuBlock32>(data)
        }
    }
}

impl From<ImageMaskInfo> for GpuBlock32 {
    fn from(data: ImageMaskInfo) -> GpuBlock32 {
        unsafe {
            mem::transmute::<ImageMaskInfo, GpuBlock32>(data)
        }
    }
}

impl From<ClipCorner> for GpuBlock32 {
    fn from(data: ClipCorner) -> GpuBlock32 {
        unsafe {
            mem::transmute::<ClipCorner, GpuBlock32>(data)
        }
    }
}

#[derive(Clone)]
pub struct GpuBlock64 {
    data: [f32; 16],
}

impl Default for GpuBlock64 {
    fn default() -> GpuBlock64 {
        GpuBlock64 {
            data: unsafe { mem::uninitialized() }
        }
    }
}

impl From<BoxShadowPrimitiveGpu> for GpuBlock64 {
    fn from(data: BoxShadowPrimitiveGpu) -> GpuBlock64 {
        unsafe {
            mem::transmute::<BoxShadowPrimitiveGpu, GpuBlock64>(data)
        }
    }
}

#[derive(Clone)]
pub struct GpuBlock128 {
    data: [f32; 32],
}

impl Default for GpuBlock128 {
    fn default() -> GpuBlock128 {
        GpuBlock128 {
            data: unsafe { mem::uninitialized() }
        }
    }
}

impl From<BorderPrimitiveGpu> for GpuBlock128 {
    fn from(data: BorderPrimitiveGpu) -> GpuBlock128 {
        unsafe {
            mem::transmute::<BorderPrimitiveGpu, GpuBlock128>(data)
        }
    }
}
