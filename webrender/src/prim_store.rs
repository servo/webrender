/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use euclid::{Point2D, Matrix4D, Rect, Size2D};
use gpu_store::{GpuStore, GpuStoreAddress};
use internal_types::{device_pixel, DeviceRect, DeviceSize, SourceTexture};
use resource_cache::ResourceCache;
use std::mem;
use std::usize;
use texture_cache::TextureCacheItem;
use tiling::RenderTask;
use util::TransformedRect;
use webrender_traits::{AuxiliaryLists, ColorF, ImageKey, ImageRendering};
use webrender_traits::{FontRenderMode, WebGLContextId};
use webrender_traits::{ClipRegion, FontKey, ItemRange, ComplexClipRegion, GlyphKey};

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
    TextShadow(PrimitiveIndex),
}

#[derive(Debug)]
pub enum PrimitiveClipSource {
    NoClip,
    Complex(Rect<f32>, f32),
    Region(ClipRegion),
}

// TODO(gw): Pack the fields here better!
#[derive(Debug)]
pub struct PrimitiveMetadata {
    pub is_opaque: bool,
    pub mask_texture_id: SourceTexture,
    pub clip_index: Option<GpuStoreAddress>,
    pub clip_source: Box<PrimitiveClipSource>,
    pub prim_kind: PrimitiveKind,
    pub cpu_prim_index: SpecificPrimitiveIndex,
    pub gpu_prim_index: GpuStoreAddress,
    pub gpu_data_address: GpuStoreAddress,
    pub gpu_data_count: i32,
    // An optional render task that is a dependency of
    // drawing this primitive. For instance, box shadows
    // use this to draw a portion of the box shadow to
    // a render target to reduce the number of pixels
    // that the box-shadow shader needs to run on. For
    // text-shadow, this creates a render task chain
    // that implements a 2-pass separable blur on a
    // text run.
    pub render_task: Option<RenderTask>,
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
    pub color_texture_id: SourceTexture,
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
    pub color_texture_id: SourceTexture,
    pub color: ColorF,
    pub render_mode: FontRenderMode,
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
struct ImageMaskData {
    uv_rect: Rect<f32>,
    local_rect: Rect<f32>,
}

#[derive(Debug, Clone)]
pub struct ClipData {
    rect: ClipRect,
    top_left: ClipCorner,
    top_right: ClipCorner,
    bottom_left: ClipCorner,
    bottom_right: ClipCorner,
    mask_data: ImageMaskData,
}

impl ClipData {
    pub fn from_clip_region(clip: &ComplexClipRegion) -> ClipData {
        ClipData {
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
            mask_data: ImageMaskData {
                uv_rect: Rect::zero(),
                local_rect: Rect::zero(),
            },
        }
    }

    pub fn uniform(rect: Rect<f32>, radius: f32) -> ClipData {
        ClipData {
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
            mask_data: ImageMaskData {
                uv_rect: Rect::zero(),
                local_rect: Rect::zero(),
            },
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

    fn populate_clip_data(data: &mut [GpuBlock32], clip: ClipData) {
        data[0] = GpuBlock32::from(clip.rect);
        data[1] = GpuBlock32::from(clip.top_left);
        data[2] = GpuBlock32::from(clip.top_right);
        data[3] = GpuBlock32::from(clip.bottom_left);
        data[4] = GpuBlock32::from(clip.bottom_right);
        data[5] = GpuBlock32::from(clip.mask_data);
    }

    pub fn add_primitive(&mut self,
                         rect: &Rect<f32>,
                         clip: &ClipRegion,
                         container: PrimitiveContainer) -> PrimitiveIndex {
        let prim_index = self.cpu_metadata.len();

        self.cpu_bounding_rects.push(None);

        self.gpu_geometry.push(PrimitiveGeometry {
            local_rect: *rect,
            local_clip_rect: clip.main.clone(),
        });

        let clip_source = Box::new(if clip.is_complex() {
            PrimitiveClipSource::Region(clip.clone())
        } else {
            PrimitiveClipSource::NoClip
        });

        let metadata = match container {
            PrimitiveContainer::Rectangle(rect) => {
                let is_opaque = rect.color.a == 1.0;
                let gpu_address = self.gpu_data16.push(rect);

                let metadata = PrimitiveMetadata {
                    is_opaque: is_opaque,
                    mask_texture_id: SourceTexture::Invalid,
                    clip_index: None,
                    clip_source: clip_source,
                    prim_kind: PrimitiveKind::Rectangle,
                    cpu_prim_index: SpecificPrimitiveIndex::invalid(),
                    gpu_prim_index: gpu_address,
                    gpu_data_address: GpuStoreAddress(0),
                    gpu_data_count: 0,
                    render_task: None,
                };

                metadata
            }
            PrimitiveContainer::TextRun(text_cpu, text_gpu) => {
                let gpu_address = self.gpu_data16.push(text_gpu);
                let gpu_glyphs_address = self.gpu_data32.alloc(text_cpu.glyph_range.length);

                let metadata = PrimitiveMetadata {
                    is_opaque: false,
                    mask_texture_id: SourceTexture::Invalid,
                    clip_index: None,
                    clip_source: clip_source,
                    prim_kind: PrimitiveKind::TextRun,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_text_runs.len()),
                    gpu_prim_index: gpu_address,
                    gpu_data_address: gpu_glyphs_address,
                    gpu_data_count: text_cpu.glyph_range.length as i32,
                    render_task: None,
                };

                self.cpu_text_runs.push(text_cpu);
                metadata
            }
            PrimitiveContainer::Image(image_cpu, image_gpu) => {
                let gpu_address = self.gpu_data32.push(image_gpu);

                let metadata = PrimitiveMetadata {
                    is_opaque: false,
                    mask_texture_id: SourceTexture::Invalid,
                    clip_index: None,
                    clip_source: clip_source,
                    prim_kind: PrimitiveKind::Image,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_images.len()),
                    gpu_prim_index: gpu_address,
                    gpu_data_address: GpuStoreAddress(0),
                    gpu_data_count: 0,
                    render_task: None,
                };

                self.cpu_images.push(image_cpu);
                metadata
            }
            PrimitiveContainer::Border(border_cpu, border_gpu) => {
                let gpu_address = self.gpu_data128.push(border_gpu);

                let metadata = PrimitiveMetadata {
                    is_opaque: false,
                    mask_texture_id: SourceTexture::Invalid,
                    clip_index: None,
                    clip_source: clip_source,
                    prim_kind: PrimitiveKind::Border,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_borders.len()),
                    gpu_prim_index: gpu_address,
                    gpu_data_address: GpuStoreAddress(0),
                    gpu_data_count: 0,
                    render_task: None,
                };

                self.cpu_borders.push(border_cpu);
                metadata
            }
            PrimitiveContainer::Gradient(gradient_cpu, gradient_gpu) => {
                let gpu_address = self.gpu_data32.push(gradient_gpu);
                let gpu_stops_address = self.gpu_data32.alloc(gradient_cpu.stops_range.length);

                let metadata = PrimitiveMetadata {
                    is_opaque: false,
                    mask_texture_id: SourceTexture::Invalid,
                    clip_index: None,
                    clip_source: clip_source,
                    prim_kind: PrimitiveKind::Gradient,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_gradients.len()),
                    gpu_prim_index: gpu_address,
                    gpu_data_address: gpu_stops_address,
                    gpu_data_count: gradient_cpu.stops_range.length as i32,
                    render_task: None,
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

                // Create a render task for this box shadow primitive. This renders a small
                // portion of the box shadow to a render target. That portion is then
                // stretched over the actual primitive rect by the box shadow primitive
                // shader, to reduce the number of pixels that the expensive box
                // shadow shader needs to run on.
                // TODO(gw): In the future, we can probably merge the box shadow
                // primitive (stretch) shader with the generic cached primitive shader.
                let render_task = RenderTask::new_prim_cache(cache_key,
                                                             cache_size,
                                                             PrimitiveIndex(prim_index));

                let gpu_prim_address = self.gpu_data64.push(box_shadow_gpu);
                let gpu_data_address = self.gpu_data16.get_next_address();

                let metadata = PrimitiveMetadata {
                    is_opaque: false,
                    mask_texture_id: SourceTexture::Invalid,
                    clip_index: None,
                    clip_source: clip_source,
                    prim_kind: PrimitiveKind::BoxShadow,
                    cpu_prim_index: SpecificPrimitiveIndex::invalid(),
                    gpu_prim_index: gpu_prim_address,
                    gpu_data_address: gpu_data_address,
                    gpu_data_count: instance_rects.len() as i32,
                    render_task: Some(render_task),
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

            if let &PrimitiveClipSource::Region(ClipRegion { image_mask: Some(mask), .. }) = metadata.clip_source.as_ref() {
                let tex_cache = resource_cache.get_image(mask.image, ImageRendering::Auto);
                metadata.mask_texture_id = tex_cache.texture_id;
                if let Some(address) = metadata.clip_index {
                    let clip_data = self.gpu_data32.get_slice_mut(address, 6);
                    clip_data[5] = GpuBlock32::from(ImageMaskData {
                        uv_rect: Rect::new(tex_cache.uv0,
                                           Size2D::new(tex_cache.uv1.x - tex_cache.uv0.x,
                                                       tex_cache.uv1.y - tex_cache.uv0.y)),
                        local_rect: mask.rect,
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
                                                               &text.glyph_indices,
                                                               text.render_mode, |index, uv0, uv1| {
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

    pub fn set_clip_source(&mut self, index: PrimitiveIndex, source: PrimitiveClipSource) {
        let metadata = &mut self.cpu_metadata[index.0];
        let (rect, is_complex) = match source {
            PrimitiveClipSource::NoClip => (None, false),
            PrimitiveClipSource::Complex(rect, radius) => (Some(rect), radius > 0.0),
            PrimitiveClipSource::Region(ref region) => (Some(region.main), region.is_complex()),
        };
        if let Some(rect) = rect {
            self.gpu_geometry.get_mut(GpuStoreAddress(index.0 as i32))
                .local_clip_rect = rect;
            if is_complex && metadata.clip_index.is_none() {
                metadata.clip_index = Some(self.gpu_data32.alloc(6))
            }
        }
        *metadata.clip_source.as_mut() = source;
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

    /// Returns true if the bounding box needs to be updated.
    pub fn prepare_prim_for_render(&mut self,
                                   prim_index: PrimitiveIndex,
                                   resource_cache: &mut ResourceCache,
                                   device_pixel_ratio: f32,
                                   dummy_mask_cache_item: &TextureCacheItem,
                                   auxiliary_lists: &AuxiliaryLists) -> bool {
        let metadata = &mut self.cpu_metadata[prim_index.0];
        let mut prim_needs_resolve = false;
        let mut rebuild_bounding_rect = false;

        if metadata.clip_index.is_none() {
            // if the `clip_index` already exist, we consider the contents up to date
            let clip_data = match metadata.clip_source.as_ref() {
                &PrimitiveClipSource::NoClip => None,
                &PrimitiveClipSource::Complex(rect, radius) => {
                    Some(ClipData::uniform(rect, radius))
                }
                &PrimitiveClipSource::Region(ref clip_region) => {
                    if let Some(mask) = clip_region.image_mask {
                        resource_cache.request_image(mask.image, ImageRendering::Auto);
                    }
                    let clips = auxiliary_lists.complex_clip_regions(&clip_region.complex);
                    //TODO: proper solution to multiple complex clips
                    match clips.len() {
                        0 if clip_region.image_mask.is_none() => None,
                        0 => Some(ClipData::uniform(clip_region.main, 0.0)),
                        1 => Some(ClipData::from_clip_region(&clips[0])),
                        _ => {
                            let internal_clip = clips.last().unwrap();
                            let region = if clips.iter().all(|current_clip| current_clip.might_contain(internal_clip)) {
                                internal_clip
                            } else {
                                &clips[0]
                            };
                            Some(ClipData::from_clip_region(region))
                        },
                    }
                }
            };

            if let Some(data) = clip_data {
                prim_needs_resolve = true;
                let gpu_address = self.gpu_data32.alloc(6);
                let gpu_data = self.gpu_data32.get_slice_mut(gpu_address, 6);
                Self::populate_clip_data(gpu_data, data);
                metadata.clip_index = Some(gpu_address);
                metadata.mask_texture_id = SourceTexture::TextureCache(dummy_mask_cache_item.texture_id);
            }
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
                                                      src_glyphs[0].index);
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

                        let x = src.x + dimensions.left as f32 / device_pixel_ratio;
                        let y = src.y - dimensions.top as f32 / device_pixel_ratio;

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

                    // Expand the rectangle of the text run by the blur radius.
                    let local_rect = local_rect.inflate(text.blur_radius.to_f32_px(),
                                                        text.blur_radius.to_f32_px());

                    let render_task = if text.blur_radius.0 == 0 {
                        None
                    } else {
                        // This is a text-shadow element. Create a render task that will
                        // render the text run to a target, and then apply a gaussian
                        // blur to that text run in order to build the actual primitive
                        // which will be blitted to the framebuffer.
                        let cache_width = (local_rect.size.width * self.device_pixel_ratio).ceil() as i32;
                        let cache_height = (local_rect.size.height * self.device_pixel_ratio).ceil() as i32;
                        let cache_size = DeviceSize::new(cache_width, cache_height);
                        let cache_key = PrimitiveCacheKey::TextShadow(prim_index);
                        let blur_radius = device_pixel(text.blur_radius.to_f32_px(),
                                                       self.device_pixel_ratio);
                        Some(RenderTask::new_blur(cache_key,
                                                  cache_size,
                                                  blur_radius,
                                                  prim_index))
                    };

                    metadata.gpu_data_count = actual_glyph_count as i32;
                    metadata.render_task = render_task;
                    self.gpu_geometry.get_mut(GpuStoreAddress(prim_index.0 as i32)).local_rect = local_rect;
                }

                resource_cache.request_glyphs(text.font_key,
                                              text.font_size,
                                              &text.glyph_indices,
                                              text.render_mode);
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

impl From<ImageMaskData> for GpuBlock32 {
    fn from(data: ImageMaskData) -> GpuBlock32 {
        unsafe {
            mem::transmute::<ImageMaskData, GpuBlock32>(data)
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

//Test for one clip region contains another
trait InsideTest<T> {
    fn might_contain(&self, clip: &T) -> bool;
}

impl InsideTest<ComplexClipRegion> for ComplexClipRegion {
    // Returns true if clip is inside self, can return false negative
    fn might_contain(&self, clip: &ComplexClipRegion) -> bool {
        let delta_left = clip.rect.origin.x - self.rect.origin.x;
        let delta_top = clip.rect.origin.y - self.rect.origin.y;
        let delta_right = self.rect.max_x() - clip.rect.max_x();
        let delta_bottom = self.rect.max_y() - clip.rect.max_y();

        delta_left >= 0f32 &&
        delta_top >= 0f32 &&
        delta_right >= 0f32 &&
        delta_bottom >= 0f32 &&
        clip.radii.top_left.width >= self.radii.top_left.width - delta_left &&
        clip.radii.top_left.height >= self.radii.top_left.height - delta_top &&
        clip.radii.top_right.width >= self.radii.top_right.width - delta_right &&
        clip.radii.top_right.height >= self.radii.top_right.height - delta_top &&
        clip.radii.bottom_left.width >= self.radii.bottom_left.width - delta_left &&
        clip.radii.bottom_left.height >= self.radii.bottom_left.height - delta_bottom &&
        clip.radii.bottom_right.width >= self.radii.bottom_right.width - delta_right &&
        clip.radii.bottom_right.height >= self.radii.bottom_right.height - delta_bottom
    }
}
