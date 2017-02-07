/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use euclid::{Point2D, Size2D};
use gpu_store::GpuStoreAddress;
use internal_types::{SourceTexture, PackedTexel};
use mask_cache::{ClipSource, MaskCacheInfo};
use renderer::{VertexDataStore, GradientDataStore};
use resource_cache::{ImageProperties, ResourceCache};
use std::mem;
use std::usize;
use tiling::{RenderTask, RenderTaskLocation};
use util::TransformedRect;
use webrender_traits::{AuxiliaryLists, ColorF, ImageKey, ImageRendering, YuvColorSpace};
use webrender_traits::{ClipRegion, ComplexClipRegion, ItemRange, GlyphKey};
use webrender_traits::{FontKey, FontRenderMode, WebGLContextId};
use webrender_traits::{device_length, DeviceIntRect, DeviceIntSize};
use webrender_traits::{DeviceRect, DevicePoint, DeviceSize};
use webrender_traits::{LayerRect, LayerSize, LayerPoint};
use webrender_traits::{LayerToWorldTransform, GlyphInstance, GlyphOptions};
use webrender_traits::{ExtendMode, GradientStop};

pub const CLIP_DATA_GPU_SIZE: usize = 5;
pub const MASK_DATA_GPU_SIZE: usize = 1;

/// Stores two coordinates in texel space. The coordinates
/// are stored in texel coordinates because the texture atlas
/// may grow. Storing them as texel coords and normalizing
/// the UVs in the vertex shader means nothing needs to be
/// updated on the CPU when the texture size changes.
#[derive(Clone)]
pub struct TexelRect {
    pub uv0: DevicePoint,
    pub uv1: DevicePoint,
}

impl Default for TexelRect {
    fn default() -> TexelRect {
        TexelRect {
            uv0: DevicePoint::zero(),
            uv1: DevicePoint::zero(),
        }
    }
}

/// For external images, it's not possible to know the
/// UV coords of the image (or the image data itself)
/// until the render thread receives the frame and issues
/// callbacks to the client application. For external
/// images that are visible, a DeferredResolve is created
/// that is stored in the frame. This allows the render
/// thread to iterate this list and update any changed
/// texture data and update the UV rect.
pub struct DeferredResolve {
    pub resource_address: GpuStoreAddress,
    pub image_properties: ImageProperties,
}

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
    YuvImage,
    Border,
    AlignedGradient,
    AngleGradient,
    RadialGradient,
    BoxShadow,
}

/// Geometry description for simple rectangular primitives, uploaded to the GPU.
#[derive(Debug, Clone)]
pub struct PrimitiveGeometry {
    pub local_rect: LayerRect,
    pub local_clip_rect: LayerRect,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum PrimitiveCacheKey {
    BoxShadow(BoxShadowPrimitiveCacheKey),
    TextShadow(PrimitiveIndex),
}

// TODO(gw): Pack the fields here better!
#[derive(Debug)]
pub struct PrimitiveMetadata {
    pub is_opaque: bool,
    pub clip_source: Box<ClipSource>,
    pub clip_cache_info: Option<MaskCacheInfo>,
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
    pub clip_task: Option<RenderTask>,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct RectanglePrimitive {
    pub color: ColorF,
}

#[derive(Debug)]
pub enum ImagePrimitiveKind {
    Image(ImageKey, ImageRendering, LayerSize),
    WebGL(WebGLContextId),
}

#[derive(Debug)]
pub struct ImagePrimitiveCpu {
    pub kind: ImagePrimitiveKind,
    pub color_texture_id: SourceTexture,
    pub resource_address: GpuStoreAddress,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct ImagePrimitiveGpu {
    pub stretch_size: LayerSize,
    pub tile_spacing: LayerSize,
}

#[derive(Debug)]
pub struct YuvImagePrimitiveCpu {
    pub y_key: ImageKey,
    pub u_key: ImageKey,
    pub v_key: ImageKey,
    pub y_texture_id: SourceTexture,
    pub u_texture_id: SourceTexture,
    pub v_texture_id: SourceTexture,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct YuvImagePrimitiveGpu {
    pub y_uv0: DevicePoint,
    pub y_uv1: DevicePoint,
    pub u_uv0: DevicePoint,
    pub u_uv1: DevicePoint,
    pub v_uv0: DevicePoint,
    pub v_uv1: DevicePoint,
    pub size: LayerSize,
    pub color_space: f32,
    pub padding: f32,
}

impl YuvImagePrimitiveGpu {
    pub fn new(size: LayerSize, color_space: YuvColorSpace) -> Self {
        YuvImagePrimitiveGpu {
            y_uv0: DevicePoint::zero(),
            y_uv1: DevicePoint::zero(),
            u_uv0: DevicePoint::zero(),
            u_uv1: DevicePoint::zero(),
            v_uv0: DevicePoint::zero(),
            v_uv1: DevicePoint::zero(),
            size: size,
            color_space: color_space as u32 as f32,
            padding: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BorderPrimitiveCpu {
    pub inner_rect: LayerRect,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct BorderPrimitiveGpu {
    pub style: [f32; 4],
    pub widths: [f32; 4],
    pub colors: [ColorF; 4],
    pub radii: [LayerSize; 4],
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct BoxShadowPrimitiveCacheKey {
    pub shadow_rect_size: Size2D<Au>,
    pub border_radius: Au,
    pub blur_radius: Au,
    pub inverted: bool,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct BoxShadowPrimitiveGpu {
    pub src_rect: LayerRect,
    pub bs_rect: LayerRect,
    pub color: ColorF,
    pub border_radius: f32,
    pub edge_size: f32,
    pub blur_radius: f32,
    pub inverted: f32,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct GradientStopGpu {
    color: ColorF,
    offset: f32,
    padding: [f32; 3],
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct GradientPrimitiveGpu {
    pub start_point: LayerPoint,
    pub end_point: LayerPoint,
    pub extend_mode: f32,
    pub padding: [f32; 3],
}

#[derive(Debug)]
pub struct GradientPrimitiveCpu {
    pub stops_range: ItemRange,
    pub extend_mode: ExtendMode,
    pub reverse_stops: bool,
    pub cache_dirty: bool,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct RadialGradientPrimitiveGpu {
    pub start_center: LayerPoint,
    pub end_center: LayerPoint,
    pub start_radius: f32,
    pub end_radius: f32,
    pub extend_mode: f32,
    pub padding: [f32; 1],
}

#[derive(Debug)]
pub struct RadialGradientPrimitiveCpu {
    pub stops_range: ItemRange,
    pub extend_mode: ExtendMode,
    pub cache_dirty: bool,
}

// The number of entries in a gradient data table.
pub const GRADIENT_DATA_RESOLUTION: usize = 128;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
// An entry in a gradient data table representing a segment of the gradient color space.
pub struct GradientDataEntry {
    pub start_color: PackedTexel,
    pub end_color: PackedTexel,
}

#[repr(C)]
// A table of gradient entries, with two colors per entry, that specify the start and end color
// within the segment of the gradient space represented by that entry. To lookup a gradient result,
// first the entry index is calculated to determine which two colors to interpolate between, then
// the offset within that entry bucket is used to interpolate between the two colors in that entry.
// This layout preserves hard stops, as the end color for a given entry can differ from the start
// color for the following entry, despite them being adjacent. Colors are stored within in BGRA8
// format for texture upload.
pub struct GradientData {
    pub colors: [GradientDataEntry; GRADIENT_DATA_RESOLUTION],
}

impl Default for GradientData {
    fn default() -> GradientData {
        GradientData {
            colors: unsafe { mem::uninitialized() }
        }
    }
}

impl Clone for GradientData {
    fn clone(&self) -> GradientData {
        GradientData {
            colors: self.colors,
        }
    }
}

impl GradientData {
    // Generate a color ramp between the start and end indexes from a start color to an end color.
    fn fill_colors(&mut self, start_idx: usize, end_idx: usize, start_color: &ColorF, end_color: &ColorF) -> usize {
        if start_idx >= end_idx {
            return start_idx;
        }

        // Calculate the color difference for individual steps in the ramp.
        let inv_steps = 1.0 / (end_idx - start_idx) as f32;
        let step_r = (end_color.r - start_color.r) * inv_steps;
        let step_g = (end_color.g - start_color.g) * inv_steps;
        let step_b = (end_color.b - start_color.b) * inv_steps;
        let step_a = (end_color.a - start_color.a) * inv_steps;

        let mut cur_color = *start_color;
        let mut cur_packed_color = PackedTexel::from_color(&cur_color);

        // Walk the ramp writing start and end colors for each entry.
        for entry in &mut self.colors[start_idx..end_idx] {
            entry.start_color = cur_packed_color;

            cur_color.r += step_r;
            cur_color.g += step_g;
            cur_color.b += step_b;
            cur_color.a += step_a;
            cur_packed_color = PackedTexel::from_color(&cur_color);
            entry.end_color = cur_packed_color;
        }

        end_idx
    }

    // Compute an entry index based on a gradient stop offset.
    #[inline]
    fn get_index(offset: f32) -> usize {
        (offset.max(0.0).min(1.0) * GRADIENT_DATA_RESOLUTION as f32).round() as usize
    }

    // Build the gradient data from the supplied stops, reversing them if necessary.
    fn build(&mut self, src_stops: &[GradientStop], reverse_stops: bool) {
        let mut cur_idx = 0usize;
        let mut cur_color = if let Some(src) = src_stops.first() {
            src.color
        } else {
            ColorF::new(0.0, 0.0, 0.0, 0.0)
        };

        if reverse_stops {
            // If the gradient is reversed, then ensure the stops are processed in reverse order
            // and that the offsets are inverted.
            for src in src_stops.iter().rev() {
                cur_idx = self.fill_colors(cur_idx, Self::get_index(1.0 - src.offset),
                                           &cur_color, &src.color);
                cur_color = src.color;
            }
        } else {
            for src in src_stops {
                cur_idx = self.fill_colors(cur_idx, Self::get_index(src.offset),
                                           &cur_color, &src.color);
                cur_color = src.color;
            }
        }

        // Fill out any remaining entries in the gradient.
        self.fill_colors(cur_idx, GRADIENT_DATA_RESOLUTION, &cur_color, &cur_color);
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
struct InstanceRect {
    rect: LayerRect,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct TextRunPrimitiveGpu {
    pub color: ColorF,
}

#[derive(Debug, Clone)]
pub struct TextRunPrimitiveCpu {
    pub font_key: FontKey,
    pub logical_font_size: Au,
    pub blur_radius: Au,
    pub glyph_range: ItemRange,
    pub cache_dirty: bool,
    // TODO(gw): Maybe make this an Arc for sharing with resource cache
    pub glyph_instances: Vec<GlyphInstance>,
    pub color_texture_id: SourceTexture,
    pub color: ColorF,
    pub render_mode: FontRenderMode,
    pub resource_address: GpuStoreAddress,
    pub glyph_options: Option<GlyphOptions>,
}

#[derive(Debug, Clone)]
#[repr(C)]
struct GlyphPrimitive {
    offset: LayerPoint,
    padding: LayerPoint,
}

#[derive(Debug, Clone)]
#[repr(C)]
struct ClipRect {
    rect: LayerRect,
    padding: [f32; 4],
}

#[derive(Debug, Clone)]
#[repr(C)]
struct ClipCorner {
    rect: LayerRect,
    outer_radius_x: f32,
    outer_radius_y: f32,
    inner_radius_x: f32,
    inner_radius_y: f32,
}

impl ClipCorner {
    fn uniform(rect: LayerRect, outer_radius: f32, inner_radius: f32) -> ClipCorner {
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
#[repr(C)]
pub struct ImageMaskData {
    uv_rect: DeviceRect,
    local_rect: LayerRect,
}

#[derive(Debug, Clone)]
pub struct ClipData {
    rect: ClipRect,
    top_left: ClipCorner,
    top_right: ClipCorner,
    bottom_left: ClipCorner,
    bottom_right: ClipCorner,
}

impl ClipData {
    pub fn from_clip_region(clip: &ComplexClipRegion) -> ClipData {
        ClipData {
            rect: ClipRect {
                rect: clip.rect,
                padding: [0.0, 0.0, 0.0, 0.0],
            },
            top_left: ClipCorner {
                rect: LayerRect::new(
                    LayerPoint::new(clip.rect.origin.x, clip.rect.origin.y),
                    LayerSize::new(clip.radii.top_left.width, clip.radii.top_left.height)),
                outer_radius_x: clip.radii.top_left.width,
                outer_radius_y: clip.radii.top_left.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            top_right: ClipCorner {
                rect: LayerRect::new(
                    LayerPoint::new(clip.rect.origin.x + clip.rect.size.width - clip.radii.top_right.width, clip.rect.origin.y),
                    LayerSize::new(clip.radii.top_right.width, clip.radii.top_right.height)),
                outer_radius_x: clip.radii.top_right.width,
                outer_radius_y: clip.radii.top_right.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            bottom_left: ClipCorner {
                rect: LayerRect::new(
                    LayerPoint::new(clip.rect.origin.x, clip.rect.origin.y + clip.rect.size.height - clip.radii.bottom_left.height),
                    LayerSize::new(clip.radii.bottom_left.width, clip.radii.bottom_left.height)),
                outer_radius_x: clip.radii.bottom_left.width,
                outer_radius_y: clip.radii.bottom_left.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            bottom_right: ClipCorner {
                rect: LayerRect::new(
                    LayerPoint::new(clip.rect.origin.x + clip.rect.size.width - clip.radii.bottom_right.width,
                                    clip.rect.origin.y + clip.rect.size.height - clip.radii.bottom_right.height),
                    LayerSize::new(clip.radii.bottom_right.width, clip.radii.bottom_right.height)),
                outer_radius_x: clip.radii.bottom_right.width,
                outer_radius_y: clip.radii.bottom_right.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
        }
    }

    pub fn uniform(rect: LayerRect, radius: f32) -> ClipData {
        ClipData {
            rect: ClipRect {
                rect: rect,
                padding: [0.0; 4],
            },
            top_left: ClipCorner::uniform(
                LayerRect::new(
                    LayerPoint::new(rect.origin.x, rect.origin.y),
                    LayerSize::new(radius, radius)),
                radius, 0.0),
            top_right: ClipCorner::uniform(
                LayerRect::new(
                    LayerPoint::new(rect.origin.x + rect.size.width - radius, rect.origin.y),
                    LayerSize::new(radius, radius)),
                radius, 0.0),
            bottom_left: ClipCorner::uniform(
                LayerRect::new(
                    LayerPoint::new(rect.origin.x, rect.origin.y + rect.size.height - radius),
                    LayerSize::new(radius, radius)),
                radius, 0.0),
            bottom_right: ClipCorner::uniform(
                LayerRect::new(
                    LayerPoint::new(rect.origin.x + rect.size.width - radius, rect.origin.y + rect.size.height - radius),
                    LayerSize::new(radius, radius)),
                radius, 0.0),
        }
    }
}

#[derive(Debug)]
pub enum PrimitiveContainer {
    Rectangle(RectanglePrimitive),
    TextRun(TextRunPrimitiveCpu, TextRunPrimitiveGpu),
    Image(ImagePrimitiveCpu, ImagePrimitiveGpu),
    YuvImage(YuvImagePrimitiveCpu, YuvImagePrimitiveGpu),
    Border(BorderPrimitiveCpu, BorderPrimitiveGpu),
    AlignedGradient(GradientPrimitiveCpu, GradientPrimitiveGpu),
    AngleGradient(GradientPrimitiveCpu, GradientPrimitiveGpu),
    RadialGradient(RadialGradientPrimitiveCpu, RadialGradientPrimitiveGpu),
    BoxShadow(BoxShadowPrimitiveGpu, Vec<LayerRect>),
}

pub struct PrimitiveStore {
    // CPU side information only
    pub cpu_bounding_rects: Vec<Option<DeviceIntRect>>,
    pub cpu_text_runs: Vec<TextRunPrimitiveCpu>,
    pub cpu_images: Vec<ImagePrimitiveCpu>,
    pub cpu_yuv_images: Vec<YuvImagePrimitiveCpu>,
    pub cpu_gradients: Vec<GradientPrimitiveCpu>,
    pub cpu_radial_gradients: Vec<RadialGradientPrimitiveCpu>,
    pub cpu_metadata: Vec<PrimitiveMetadata>,
    pub cpu_borders: Vec<BorderPrimitiveCpu>,

    // Gets uploaded directly to GPU via vertex texture
    pub gpu_geometry: VertexDataStore<PrimitiveGeometry>,
    pub gpu_data16: VertexDataStore<GpuBlock16>,
    pub gpu_data32: VertexDataStore<GpuBlock32>,
    pub gpu_data64: VertexDataStore<GpuBlock64>,
    pub gpu_data128: VertexDataStore<GpuBlock128>,
    pub gpu_gradient_data: GradientDataStore,

    // Resolved resource rects.
    pub gpu_resource_rects: VertexDataStore<TexelRect>,

    // General
    prims_to_resolve: Vec<PrimitiveIndex>,
}

impl PrimitiveStore {
    pub fn new() -> PrimitiveStore {
        PrimitiveStore {
            cpu_metadata: Vec::new(),
            cpu_bounding_rects: Vec::new(),
            cpu_text_runs: Vec::new(),
            cpu_images: Vec::new(),
            cpu_yuv_images: Vec::new(),
            cpu_gradients: Vec::new(),
            cpu_radial_gradients: Vec::new(),
            cpu_borders: Vec::new(),
            gpu_geometry: VertexDataStore::new(),
            gpu_data16: VertexDataStore::new(),
            gpu_data32: VertexDataStore::new(),
            gpu_data64: VertexDataStore::new(),
            gpu_data128: VertexDataStore::new(),
            gpu_gradient_data: GradientDataStore::new(),
            gpu_resource_rects: VertexDataStore::new(),
            prims_to_resolve: Vec::new(),
        }
    }

    pub fn populate_clip_data(data: &mut [GpuBlock32], clip: ClipData) {
        data[0] = GpuBlock32::from(clip.rect);
        data[1] = GpuBlock32::from(clip.top_left);
        data[2] = GpuBlock32::from(clip.top_right);
        data[3] = GpuBlock32::from(clip.bottom_left);
        data[4] = GpuBlock32::from(clip.bottom_right);
    }

    pub fn add_primitive(&mut self,
                         geometry: PrimitiveGeometry,
                         clip_source: Box<ClipSource>,
                         clip_info: Option<MaskCacheInfo>,
                         container: PrimitiveContainer) -> PrimitiveIndex {
        let prim_index = self.cpu_metadata.len();
        self.cpu_bounding_rects.push(None);
        self.gpu_geometry.push(geometry);

        let metadata = match container {
            PrimitiveContainer::Rectangle(rect) => {
                let is_opaque = rect.color.a == 1.0;
                let gpu_address = self.gpu_data16.push(rect);

                let metadata = PrimitiveMetadata {
                    is_opaque: is_opaque,
                    clip_source: clip_source,
                    clip_cache_info: clip_info,
                    prim_kind: PrimitiveKind::Rectangle,
                    cpu_prim_index: SpecificPrimitiveIndex::invalid(),
                    gpu_prim_index: gpu_address,
                    gpu_data_address: GpuStoreAddress(0),
                    gpu_data_count: 0,
                    render_task: None,
                    clip_task: None,
                };

                metadata
            }
            PrimitiveContainer::TextRun(mut text_cpu, text_gpu) => {
                let gpu_address = self.gpu_data16.push(text_gpu);
                let gpu_glyphs_address = self.gpu_data16.alloc(text_cpu.glyph_range.length);
                text_cpu.resource_address = self.gpu_resource_rects.alloc(text_cpu.glyph_range.length);

                let metadata = PrimitiveMetadata {
                    is_opaque: false,
                    clip_source: clip_source,
                    clip_cache_info: clip_info,
                    prim_kind: PrimitiveKind::TextRun,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_text_runs.len()),
                    gpu_prim_index: gpu_address,
                    gpu_data_address: gpu_glyphs_address,
                    gpu_data_count: text_cpu.glyph_range.length as i32,
                    render_task: None,
                    clip_task: None,
                };

                self.cpu_text_runs.push(text_cpu);
                metadata
            }
            PrimitiveContainer::Image(mut image_cpu, image_gpu) => {
                image_cpu.resource_address = self.gpu_resource_rects.alloc(1);

                let gpu_address = self.gpu_data16.push(image_gpu);

                let metadata = PrimitiveMetadata {
                    is_opaque: false,
                    clip_source: clip_source,
                    clip_cache_info: clip_info,
                    prim_kind: PrimitiveKind::Image,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_images.len()),
                    gpu_prim_index: gpu_address,
                    gpu_data_address: GpuStoreAddress(0),
                    gpu_data_count: 0,
                    render_task: None,
                    clip_task: None,
                };

                self.cpu_images.push(image_cpu);
                metadata
            }
            PrimitiveContainer::YuvImage(image_cpu, image_gpu) => {
                let gpu_address = self.gpu_data64.push(image_gpu);

                let metadata = PrimitiveMetadata {
                    is_opaque: true,
                    clip_source: clip_source,
                    clip_cache_info: clip_info,
                    prim_kind: PrimitiveKind::YuvImage,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_yuv_images.len()),
                    gpu_prim_index: gpu_address,
                    gpu_data_address: GpuStoreAddress(0),
                    gpu_data_count: 0,
                    render_task: None,
                    clip_task: None,
                };

                self.cpu_yuv_images.push(image_cpu);
                metadata
            }
            PrimitiveContainer::Border(border_cpu, border_gpu) => {
                let gpu_address = self.gpu_data128.push(border_gpu);

                let metadata = PrimitiveMetadata {
                    is_opaque: false,
                    clip_source: clip_source,
                    clip_cache_info: clip_info,
                    prim_kind: PrimitiveKind::Border,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_borders.len()),
                    gpu_prim_index: gpu_address,
                    gpu_data_address: GpuStoreAddress(0),
                    gpu_data_count: 0,
                    render_task: None,
                    clip_task: None,
                };

                self.cpu_borders.push(border_cpu);
                metadata
            }
            PrimitiveContainer::AlignedGradient(gradient_cpu, gradient_gpu) => {
                let gpu_address = self.gpu_data32.push(gradient_gpu);
                let gpu_stops_address = self.gpu_data32.alloc(gradient_cpu.stops_range.length);

                let metadata = PrimitiveMetadata {
                    // TODO: calculate if the gradient is actually opaque
                    is_opaque: false,
                    clip_source: clip_source,
                    clip_cache_info: clip_info,
                    prim_kind: PrimitiveKind::AlignedGradient,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_gradients.len()),
                    gpu_prim_index: gpu_address,
                    gpu_data_address: gpu_stops_address,
                    gpu_data_count: gradient_cpu.stops_range.length as i32,
                    render_task: None,
                    clip_task: None,
                };

                self.cpu_gradients.push(gradient_cpu);
                metadata
            }
            PrimitiveContainer::AngleGradient(gradient_cpu, gradient_gpu) => {
                let gpu_address = self.gpu_data32.push(gradient_gpu);
                let gpu_gradient_address = self.gpu_gradient_data.alloc(1);

                let metadata = PrimitiveMetadata {
                    // TODO: calculate if the gradient is actually opaque
                    is_opaque: false,
                    clip_source: clip_source,
                    clip_cache_info: clip_info,
                    prim_kind: PrimitiveKind::AngleGradient,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_gradients.len()),
                    gpu_prim_index: gpu_address,
                    gpu_data_address: gpu_gradient_address,
                    gpu_data_count: 1,
                    render_task: None,
                    clip_task: None,
                };

                self.cpu_gradients.push(gradient_cpu);
                metadata
            }
            PrimitiveContainer::RadialGradient(radial_gradient_cpu, radial_gradient_gpu) => {
                let gpu_address = self.gpu_data32.push(radial_gradient_gpu);
                let gpu_gradient_address = self.gpu_gradient_data.alloc(1);

                let metadata = PrimitiveMetadata {
                    // TODO: calculate if the gradient is actually opaque
                    is_opaque: false,
                    clip_source: clip_source,
                    clip_cache_info: clip_info,
                    prim_kind: PrimitiveKind::RadialGradient,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_radial_gradients.len()),
                    gpu_prim_index: gpu_address,
                    gpu_data_address: gpu_gradient_address,
                    gpu_data_count: 1,
                    render_task: None,
                    clip_task: None,
                };

                self.cpu_radial_gradients.push(radial_gradient_cpu);
                metadata
            }
            PrimitiveContainer::BoxShadow(box_shadow_gpu, instance_rects) => {
                let cache_key = PrimitiveCacheKey::BoxShadow(BoxShadowPrimitiveCacheKey {
                    blur_radius: Au::from_f32_px(box_shadow_gpu.blur_radius),
                    border_radius: Au::from_f32_px(box_shadow_gpu.border_radius),
                    inverted: box_shadow_gpu.inverted != 0.0,
                    shadow_rect_size: Size2D::new(Au::from_f32_px(box_shadow_gpu.bs_rect.size.width),
                                                  Au::from_f32_px(box_shadow_gpu.bs_rect.size.height)),
                });

                // The actual cache size is calculated during prepare_prim_for_render().
                // This is necessary since the size may change depending on the device
                // pixel ratio (for example, during zoom or moving the window to a
                // monitor with a different device pixel ratio).
                let cache_size = DeviceIntSize::zero();

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
                    clip_source: clip_source,
                    clip_cache_info: None,
                    prim_kind: PrimitiveKind::BoxShadow,
                    cpu_prim_index: SpecificPrimitiveIndex::invalid(),
                    gpu_prim_index: gpu_prim_address,
                    gpu_data_address: gpu_data_address,
                    gpu_data_count: instance_rects.len() as i32,
                    render_task: Some(render_task),
                    clip_task: None,
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

    fn resolve_clip_cache_internal(gpu_data32: &mut VertexDataStore<GpuBlock32>,
                                   clip_info: &MaskCacheInfo,
                                   resource_cache: &ResourceCache) {
        if let Some((ref mask, gpu_address)) = clip_info.image {
            let cache_item = resource_cache.get_cached_image(mask.image, ImageRendering::Auto);
            let mask_data = gpu_data32.get_slice_mut(gpu_address, MASK_DATA_GPU_SIZE);
            mask_data[0] = GpuBlock32::from(ImageMaskData {
                uv_rect: DeviceRect::new(cache_item.uv0,
                                         DeviceSize::new(cache_item.uv1.x - cache_item.uv0.x,
                                                         cache_item.uv1.y - cache_item.uv0.y)),
                local_rect: mask.rect,
            });
        }
    }

    pub fn resolve_clip_cache(&mut self,
                              clip_info: &MaskCacheInfo,
                              resource_cache: &ResourceCache) {
        Self::resolve_clip_cache_internal(&mut self.gpu_data32, clip_info, resource_cache)
    }

    pub fn resolve_primitives(&mut self,
                              resource_cache: &ResourceCache,
                              device_pixel_ratio: f32) -> Vec<DeferredResolve> {
        let mut deferred_resolves = Vec::new();

        for prim_index in self.prims_to_resolve.drain(..) {
            let metadata = &mut self.cpu_metadata[prim_index.0];
            if let Some(ref clip_info) = metadata.clip_cache_info {
                Self::resolve_clip_cache_internal(&mut self.gpu_data32, clip_info, resource_cache);
            }

            match metadata.prim_kind {
                PrimitiveKind::Rectangle |
                PrimitiveKind::Border |
                PrimitiveKind::BoxShadow |
                PrimitiveKind::AlignedGradient |
                PrimitiveKind::AngleGradient |
                PrimitiveKind::RadialGradient=> {}
                PrimitiveKind::TextRun => {
                    let text = &mut self.cpu_text_runs[metadata.cpu_prim_index.0];

                    let font_size_dp = text.logical_font_size.scale_by(device_pixel_ratio);

                    let dest_rects = self.gpu_resource_rects.get_slice_mut(text.resource_address,
                                                                           text.glyph_range.length);

                    let texture_id = resource_cache.get_glyphs(text.font_key,
                                                               font_size_dp,
                                                               text.color,
                                                               &text.glyph_instances,
                                                               text.render_mode,
                                                               text.glyph_options, |index, uv0, uv1| {
                        let dest_rect = &mut dest_rects[index];
                        dest_rect.uv0 = uv0;
                        dest_rect.uv1 = uv1;
                    });

                    text.color_texture_id = texture_id;
                }
                PrimitiveKind::Image => {
                    let image_cpu = &mut self.cpu_images[metadata.cpu_prim_index.0];

                    let (texture_id, cache_item) = match image_cpu.kind {
                        ImagePrimitiveKind::Image(image_key, image_rendering, _) => {
                            // Check if an external image that needs to be resolved
                            // by the render thread.
                            let image_properties = resource_cache.get_image_properties(image_key);

                            match image_properties.external_id {
                                Some(external_id) => {
                                    // This is an external texture - we will add it to
                                    // the deferred resolves list to be patched by
                                    // the render thread...
                                    deferred_resolves.push(DeferredResolve {
                                        resource_address: image_cpu.resource_address,
                                        image_properties: image_properties,
                                    });

                                    (SourceTexture::External(external_id), None)
                                }
                                None => {
                                    let cache_item = resource_cache.get_cached_image(image_key, image_rendering);
                                    (cache_item.texture_id, Some(cache_item))
                                }
                            }
                        }
                        ImagePrimitiveKind::WebGL(context_id) => {
                            let cache_item = resource_cache.get_webgl_texture(&context_id);
                            (cache_item.texture_id, Some(cache_item))
                        }
                    };

                    if let Some(cache_item) = cache_item {
                        let resource_rect = self.gpu_resource_rects.get_mut(image_cpu.resource_address);
                        resource_rect.uv0 = cache_item.uv0;
                        resource_rect.uv1 = cache_item.uv1;
                    }
                    image_cpu.color_texture_id = texture_id;
                }
                PrimitiveKind::YuvImage => {
                    let image_cpu = &mut self.cpu_yuv_images[metadata.cpu_prim_index.0];
                    let image_gpu: &mut YuvImagePrimitiveGpu = unsafe {
                        mem::transmute(self.gpu_data64.get_mut(metadata.gpu_prim_index))
                    };

                    if image_cpu.y_texture_id == SourceTexture::Invalid {
                        let y_cache_item = resource_cache.get_cached_image(image_cpu.y_key, ImageRendering::Auto);
                        image_cpu.y_texture_id = y_cache_item.texture_id;
                        image_gpu.y_uv0 = y_cache_item.uv0;
                        image_gpu.y_uv1 = y_cache_item.uv1;
                    }

                    if image_cpu.u_texture_id == SourceTexture::Invalid {
                        let u_cache_item = resource_cache.get_cached_image(image_cpu.u_key, ImageRendering::Auto);
                        image_cpu.u_texture_id = u_cache_item.texture_id;
                        image_gpu.u_uv0 = u_cache_item.uv0;
                        image_gpu.u_uv1 = u_cache_item.uv1;
                    }

                    if image_cpu.v_texture_id == SourceTexture::Invalid {
                        let v_cache_item = resource_cache.get_cached_image(image_cpu.v_key, ImageRendering::Auto);
                        image_cpu.v_texture_id = v_cache_item.texture_id;
                        image_gpu.v_uv0 = v_cache_item.uv0;
                        image_gpu.v_uv1 = v_cache_item.uv1;
                    }
                }
            }
        }

        deferred_resolves
    }

    pub fn set_clip_source(&mut self, index: PrimitiveIndex, source: ClipSource) {
        let metadata = &mut self.cpu_metadata[index.0];
        let (rect, is_complex) = match source {
            ClipSource::NoClip => (None, false),
            ClipSource::Complex(rect, radius) => (Some(rect), radius > 0.0),
            ClipSource::Region(ref region) => (Some(region.main), region.is_complex()),
        };
        if let Some(rect) = rect {
            self.gpu_geometry.get_mut(GpuStoreAddress(index.0 as i32))
                .local_clip_rect = rect;
            if is_complex {
                metadata.clip_cache_info = None; //CLIP TODO: re-use the existing GPU allocation
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
                               screen_rect: &DeviceIntRect,
                               layer_transform: &LayerToWorldTransform,
                               layer_combined_local_clip_rect: &LayerRect,
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
                                   layer_transform: &LayerToWorldTransform,
                                   device_pixel_ratio: f32,
                                   auxiliary_lists: &AuxiliaryLists) -> bool {

        let metadata = &mut self.cpu_metadata[prim_index.0];
        let mut prim_needs_resolve = false;
        let mut rebuild_bounding_rect = false;

        if let Some(ref mut clip_info) = metadata.clip_cache_info {
            clip_info.update(&metadata.clip_source,
                             layer_transform,
                             &mut self.gpu_data32,
                             device_pixel_ratio,
                             auxiliary_lists);
            if let &ClipSource::Region(ClipRegion{ image_mask: Some(ref mask), .. }) = metadata.clip_source.as_ref() {
                resource_cache.request_image(mask.image, ImageRendering::Auto);
                prim_needs_resolve = true;
            }
        }

        match metadata.prim_kind {
            PrimitiveKind::Rectangle |
            PrimitiveKind::Border  => {}
            PrimitiveKind::BoxShadow => {
                // TODO(gw): Account for zoom factor!
                // Here, we calculate the size of the patch required in order
                // to create the box shadow corner. First, scale it by the
                // device pixel ratio since the cache shader expects vertices
                // in device space. The shader adds a 1-pixel border around
                // the patch, in order to prevent bilinear filter artifacts as
                // the patch is clamped / mirrored across the box shadow rect.
                let box_shadow_gpu: &BoxShadowPrimitiveGpu = unsafe {
                    mem::transmute(self.gpu_data64.get(metadata.gpu_prim_index))
                };
                let edge_size = box_shadow_gpu.edge_size.ceil() * device_pixel_ratio;
                let edge_size = edge_size as i32 + 2;   // Account for bilinear filtering
                let cache_size = DeviceIntSize::new(edge_size, edge_size);
                let location = RenderTaskLocation::Dynamic(None, cache_size);
                metadata.render_task.as_mut().unwrap().location = location;
            }
            PrimitiveKind::TextRun => {
                let text = &mut self.cpu_text_runs[metadata.cpu_prim_index.0];

                let font_size_dp = text.logical_font_size.scale_by(device_pixel_ratio);
                let src_glyphs = auxiliary_lists.glyph_instances(&text.glyph_range);
                prim_needs_resolve = true;

                if text.cache_dirty {
                    rebuild_bounding_rect = true;
                    text.cache_dirty = false;

                    debug_assert!(metadata.gpu_data_count == text.glyph_range.length as i32);
                    debug_assert!(text.glyph_instances.is_empty());

                    let dest_glyphs = self.gpu_data16.get_slice_mut(metadata.gpu_data_address,
                                                                    text.glyph_range.length);
                    let mut glyph_key = GlyphKey::new(text.font_key,
                                                      font_size_dp,
                                                      text.color,
                                                      src_glyphs[0].index,
                                                      src_glyphs[0].point,
                                                      text.render_mode);
                    let mut local_rect = LayerRect::zero();
                    let mut actual_glyph_count = 0;

                    for src in src_glyphs {
                        glyph_key.index = src.index;
                        glyph_key.subpixel_point.set_offset(src.point, text.render_mode);

                        let dimensions = match resource_cache.get_glyph_dimensions(&glyph_key) {
                            None => continue,
                            Some(dimensions) => dimensions,
                        };

                        // TODO(gw): Check for this and ensure platforms return None in this case!!!
                        debug_assert!(dimensions.width > 0 && dimensions.height > 0);

                        let x = src.point.x + dimensions.left as f32 / device_pixel_ratio;
                        let y = src.point.y - dimensions.top as f32 / device_pixel_ratio;

                        let width = dimensions.width as f32 / device_pixel_ratio;
                        let height = dimensions.height as f32 / device_pixel_ratio;

                        let local_glyph_rect = LayerRect::new(LayerPoint::new(x, y),
                                                              LayerSize::new(width, height));
                        local_rect = local_rect.union(&local_glyph_rect);

                        dest_glyphs[actual_glyph_count] = GpuBlock16::from(GlyphPrimitive {
                            padding: LayerPoint::zero(),
                            offset: local_glyph_rect.origin,
                        });

                        text.glyph_instances.push(GlyphInstance {
                            index: src.index,
                            point: Point2D::new(src.point.x, src.point.y),
                        });

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
                        let cache_width = (local_rect.size.width * device_pixel_ratio).ceil() as i32;
                        let cache_height = (local_rect.size.height * device_pixel_ratio).ceil() as i32;
                        let cache_size = DeviceIntSize::new(cache_width, cache_height);
                        let cache_key = PrimitiveCacheKey::TextShadow(prim_index);
                        let blur_radius = device_length(text.blur_radius.to_f32_px(),
                                                        device_pixel_ratio);
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
                                              font_size_dp,
                                              text.color,
                                              &text.glyph_instances,
                                              text.render_mode,
                                              text.glyph_options);
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
                        metadata.is_opaque = image_properties.descriptor.is_opaque &&
                                             tile_spacing.width == 0.0 &&
                                             tile_spacing.height == 0.0;
                    }
                    ImagePrimitiveKind::WebGL(..) => {}
                }
            }
            PrimitiveKind::YuvImage => {
                let image_cpu = &mut self.cpu_yuv_images[metadata.cpu_prim_index.0];
                prim_needs_resolve = true;

                resource_cache.request_image(image_cpu.y_key, ImageRendering::Auto);
                resource_cache.request_image(image_cpu.u_key, ImageRendering::Auto);
                resource_cache.request_image(image_cpu.v_key, ImageRendering::Auto);

                // TODO(nical): Currently assuming no tile_spacing for yuv images.
                metadata.is_opaque = true;
            }
            PrimitiveKind::AlignedGradient => {
                let gradient = &mut self.cpu_gradients[metadata.cpu_prim_index.0];
                if gradient.cache_dirty {
                    let src_stops = auxiliary_lists.gradient_stops(&gradient.stops_range);

                    debug_assert!(metadata.gpu_data_count == gradient.stops_range.length as i32);
                    let dest_stops = self.gpu_data32.get_slice_mut(metadata.gpu_data_address,
                                                                   gradient.stops_range.length);

                    for (src, dest) in src_stops.iter().zip(dest_stops.iter_mut()) {
                        *dest = GpuBlock32::from(GradientStopGpu {
                            offset: src.offset,
                            color: src.color,
                            padding: [0.0; 3],
                        });
                    }

                    gradient.cache_dirty = false;
                }
            }
            PrimitiveKind::AngleGradient => {
                let gradient = &mut self.cpu_gradients[metadata.cpu_prim_index.0];
                if gradient.cache_dirty {
                    let src_stops = auxiliary_lists.gradient_stops(&gradient.stops_range);
                    let dest_gradient = self.gpu_gradient_data.get_mut(metadata.gpu_data_address);
                    dest_gradient.build(src_stops, gradient.reverse_stops);
                    gradient.cache_dirty = false;
                }
            }
            PrimitiveKind::RadialGradient => {
                let gradient = &mut self.cpu_radial_gradients[metadata.cpu_prim_index.0];
                if gradient.cache_dirty {
                    let src_stops = auxiliary_lists.gradient_stops(&gradient.stops_range);
                    let dest_gradient = self.gpu_gradient_data.get_mut(metadata.gpu_data_address);
                    dest_gradient.build(src_stops, false);
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
#[repr(C)]
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

impl From<ImagePrimitiveGpu> for GpuBlock16 {
    fn from(data: ImagePrimitiveGpu) -> GpuBlock16 {
        unsafe {
            mem::transmute::<ImagePrimitiveGpu, GpuBlock16>(data)
        }
    }
}

impl From<GlyphPrimitive> for GpuBlock16 {
    fn from(data: GlyphPrimitive) -> GpuBlock16 {
        unsafe {
            mem::transmute::<GlyphPrimitive, GpuBlock16>(data)
        }
    }
}

#[derive(Clone)]
#[repr(C)]
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

impl From<GradientStopGpu> for GpuBlock32 {
    fn from(data: GradientStopGpu) -> GpuBlock32 {
        unsafe {
            mem::transmute::<GradientStopGpu, GpuBlock32>(data)
        }
    }
}

impl From<RadialGradientPrimitiveGpu> for GpuBlock32 {
    fn from(data: RadialGradientPrimitiveGpu) -> GpuBlock32 {
        unsafe {
            mem::transmute::<RadialGradientPrimitiveGpu, GpuBlock32>(data)
        }
    }
}

impl From<YuvImagePrimitiveGpu> for GpuBlock64 {
    fn from(data: YuvImagePrimitiveGpu) -> GpuBlock64 {
        unsafe {
            mem::transmute::<YuvImagePrimitiveGpu, GpuBlock64>(data)
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
#[repr(C)]
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
#[repr(C)]
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
