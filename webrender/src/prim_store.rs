/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{BorderRadius, BuiltDisplayList, ColorF, ComplexClipRegion, DeviceIntRect, DeviceIntSize};
use api::{DevicePoint, ExtendMode, FontInstance, FontRenderMode, GlyphInstance, GlyphKey};
use api::{GradientStop, ImageKey, ImageRendering, ItemRange, ItemTag, LayerPoint, LayerRect};
use api::{LayerSize, LayerVector2D, LineOrientation, LineStyle};
use api::{TileOffset, YuvColorSpace, YuvFormat, device_length};
use app_units::Au;
use border::BorderCornerInstance;
use clip::{ClipMode, ClipSourcesHandle, ClipStore, Geometry};
use euclid::Size2D;
use frame_builder::PrimitiveContext;
use gpu_cache::{GpuBlockData, GpuCache, GpuCacheAddress, GpuCacheHandle, GpuDataRequest,
                ToGpuBlocks};
use picture::PicturePrimitive;
use render_task::{ClipWorkItem, RenderTask, RenderTaskId, RenderTaskTree};
use renderer::MAX_VERTEX_TEXTURE_WIDTH;
use resource_cache::{ImageProperties, ResourceCache};
use std::{mem, usize};
use util::{MatrixHelpers, pack_as_float, recycle_vec, TransformedRect};

#[derive(Debug, Copy, Clone)]
pub struct PrimitiveOpacity {
    pub is_opaque: bool,
}

impl PrimitiveOpacity {
    pub fn opaque() -> PrimitiveOpacity {
        PrimitiveOpacity { is_opaque: true }
    }

    pub fn translucent() -> PrimitiveOpacity {
        PrimitiveOpacity { is_opaque: false }
    }

    pub fn from_alpha(alpha: f32) -> PrimitiveOpacity {
        PrimitiveOpacity {
            is_opaque: alpha == 1.0,
        }
    }

    pub fn accumulate(&mut self, alpha: f32) {
        self.is_opaque = self.is_opaque && alpha == 1.0;
    }
}

/// Stores two coordinates in texel space. The coordinates
/// are stored in texel coordinates because the texture atlas
/// may grow. Storing them as texel coords and normalizing
/// the UVs in the vertex shader means nothing needs to be
/// updated on the CPU when the texture size changes.
#[derive(Copy, Clone, Debug)]
pub struct TexelRect {
    pub uv0: DevicePoint,
    pub uv1: DevicePoint,
}

impl TexelRect {
    pub fn new(u0: f32, v0: f32, u1: f32, v1: f32) -> TexelRect {
        TexelRect {
            uv0: DevicePoint::new(u0, v0),
            uv1: DevicePoint::new(u1, v1),
        }
    }

    pub fn invalid() -> TexelRect {
        TexelRect {
            uv0: DevicePoint::new(-1.0, -1.0),
            uv1: DevicePoint::new(-1.0, -1.0),
        }
    }
}

impl Into<GpuBlockData> for TexelRect {
    fn into(self) -> GpuBlockData {
        GpuBlockData {
            data: [self.uv0.x, self.uv0.y, self.uv1.x, self.uv1.y],
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
    pub address: GpuCacheAddress,
    pub image_properties: ImageProperties,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct SpecificPrimitiveIndex(pub usize);

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
    Line,
    Picture,
}

impl GpuCacheHandle {
    pub fn as_int(&self, gpu_cache: &GpuCache) -> i32 {
        gpu_cache.get_address(self).as_int()
    }
}

impl GpuCacheAddress {
    pub fn as_int(&self) -> i32 {
        // TODO(gw): Temporarily encode GPU Cache addresses as a single int.
        //           In the future, we can change the PrimitiveInstance struct
        //           to use 2x u16 for the vertex attribute instead of an i32.
        self.v as i32 * MAX_VERTEX_TEXTURE_WIDTH as i32 + self.u as i32
    }
}

// TODO(gw): Pack the fields here better!
#[derive(Debug)]
pub struct PrimitiveMetadata {
    pub opacity: PrimitiveOpacity,
    pub clip_sources: ClipSourcesHandle,
    pub prim_kind: PrimitiveKind,
    pub cpu_prim_index: SpecificPrimitiveIndex,
    pub gpu_location: GpuCacheHandle,
    pub clip_task_id: Option<RenderTaskId>,

    // TODO(gw): In the future, we should just pull these
    //           directly from the DL item, instead of
    //           storing them here.
    pub local_rect: LayerRect,
    pub local_clip_rect: LayerRect,
    pub is_backface_visible: bool,
    pub screen_rect: Option<DeviceIntRect>,

    /// A tag used to identify this primitive outside of WebRender. This is
    /// used for returning useful data during hit testing.
    pub tag: Option<ItemTag>,
}

#[derive(Debug)]
#[repr(C)]
pub struct RectanglePrimitive {
    pub color: ColorF,
}

impl ToGpuBlocks for RectanglePrimitive {
    fn write_gpu_blocks(&self, mut request: GpuDataRequest) {
        request.push(self.color);
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct LinePrimitive {
    pub color: ColorF,
    pub style: LineStyle,
    pub orientation: LineOrientation,
}

impl ToGpuBlocks for LinePrimitive {
    fn write_gpu_blocks(&self, mut request: GpuDataRequest) {
        request.push(self.color);
        request.push([
            pack_as_float(self.style as u32),
            pack_as_float(self.orientation as u32),
            0.0,
            0.0,
        ]);
    }
}

#[derive(Debug)]
pub struct ImagePrimitiveCpu {
    pub image_key: ImageKey,
    pub image_rendering: ImageRendering,
    pub tile_offset: Option<TileOffset>,
    pub tile_spacing: LayerSize,
    // TODO(gw): Build on demand
    pub gpu_blocks: [GpuBlockData; 2],
}

impl ToGpuBlocks for ImagePrimitiveCpu {
    fn write_gpu_blocks(&self, mut request: GpuDataRequest) {
        request.extend_from_slice(&self.gpu_blocks);
    }
}

#[derive(Debug)]
pub struct YuvImagePrimitiveCpu {
    pub yuv_key: [ImageKey; 3],
    pub format: YuvFormat,
    pub color_space: YuvColorSpace,

    pub image_rendering: ImageRendering,

    // TODO(gw): Generate on demand
    pub gpu_block: GpuBlockData,
}

impl ToGpuBlocks for YuvImagePrimitiveCpu {
    fn write_gpu_blocks(&self, mut request: GpuDataRequest) {
        request.push(self.gpu_block);
    }
}

#[derive(Debug)]
pub struct BorderPrimitiveCpu {
    pub corner_instances: [BorderCornerInstance; 4],
    pub gpu_blocks: [GpuBlockData; 8],
}

impl ToGpuBlocks for BorderPrimitiveCpu {
    fn write_gpu_blocks(&self, mut request: GpuDataRequest) {
        request.extend_from_slice(&self.gpu_blocks);
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct BoxShadowPrimitiveCacheKey {
    pub shadow_rect_size: Size2D<Au>,
    pub border_radius: Au,
    pub blur_radius: Au,
    pub inverted: bool,
}

#[derive(Debug)]
pub struct BoxShadowPrimitiveCpu {
    // todo(gw): generate on demand
    // gpu data
    pub src_rect: LayerRect,
    pub bs_rect: LayerRect,
    pub color: ColorF,
    pub border_radius: f32,
    pub edge_size: f32,
    pub blur_radius: f32,
    pub inverted: f32,
    pub rects: Vec<LayerRect>,
    pub render_task_id: Option<RenderTaskId>,
}

impl ToGpuBlocks for BoxShadowPrimitiveCpu {
    fn write_gpu_blocks(&self, mut request: GpuDataRequest) {
        request.push(self.src_rect);
        request.push(self.bs_rect);
        request.push(self.color);
        request.push([
            self.border_radius,
            self.edge_size,
            self.blur_radius,
            self.inverted,
        ]);
        for &rect in &self.rects {
            request.push(rect);
        }
    }
}

#[derive(Debug)]
pub struct GradientPrimitiveCpu {
    pub stops_range: ItemRange<GradientStop>,
    pub stops_count: usize,
    pub extend_mode: ExtendMode,
    pub reverse_stops: bool,
    pub gpu_blocks: [GpuBlockData; 3],
}

impl GradientPrimitiveCpu {
    fn build_gpu_blocks_for_aligned(
        &self,
        display_list: &BuiltDisplayList,
        mut request: GpuDataRequest,
    ) -> PrimitiveOpacity {
        let mut opacity = PrimitiveOpacity::opaque();
        request.extend_from_slice(&self.gpu_blocks);
        let src_stops = display_list.get(self.stops_range);

        for src in src_stops {
            request.push(src.color.premultiplied());
            request.push([src.offset, 0.0, 0.0, 0.0]);
            opacity.accumulate(src.color.a);
        }

        opacity
    }

    fn build_gpu_blocks_for_angle_radial(
        &self,
        display_list: &BuiltDisplayList,
        mut request: GpuDataRequest,
    ) {
        request.extend_from_slice(&self.gpu_blocks);

        let gradient_builder = GradientGpuBlockBuilder::new(self.stops_range, display_list);
        gradient_builder.build(self.reverse_stops, &mut request);
    }
}

// The gradient entry index for the first color stop
pub const GRADIENT_DATA_FIRST_STOP: usize = 0;
// The gradient entry index for the last color stop
pub const GRADIENT_DATA_LAST_STOP: usize = GRADIENT_DATA_SIZE - 1;

// The start of the gradient data table
pub const GRADIENT_DATA_TABLE_BEGIN: usize = GRADIENT_DATA_FIRST_STOP + 1;
// The exclusive bound of the gradient data table
pub const GRADIENT_DATA_TABLE_END: usize = GRADIENT_DATA_LAST_STOP;
// The number of entries in the gradient data table.
pub const GRADIENT_DATA_TABLE_SIZE: usize = 128;

// The number of entries in a gradient data: GRADIENT_DATA_TABLE_SIZE + first stop entry + last stop entry
pub const GRADIENT_DATA_SIZE: usize = GRADIENT_DATA_TABLE_SIZE + 2;

#[derive(Debug)]
#[repr(C)]
// An entry in a gradient data table representing a segment of the gradient color space.
pub struct GradientDataEntry {
    pub start_color: ColorF,
    pub end_color: ColorF,
}

struct GradientGpuBlockBuilder<'a> {
    stops_range: ItemRange<GradientStop>,
    display_list: &'a BuiltDisplayList,
}

impl<'a> GradientGpuBlockBuilder<'a> {
    fn new(
        stops_range: ItemRange<GradientStop>,
        display_list: &'a BuiltDisplayList,
    ) -> GradientGpuBlockBuilder<'a> {
        GradientGpuBlockBuilder {
            stops_range,
            display_list,
        }
    }

    /// Generate a color ramp filling the indices in [start_idx, end_idx) and interpolating
    /// from start_color to end_color.
    fn fill_colors(
        &self,
        start_idx: usize,
        end_idx: usize,
        start_color: &ColorF,
        end_color: &ColorF,
        entries: &mut [GradientDataEntry; GRADIENT_DATA_SIZE],
    ) {
        // Calculate the color difference for individual steps in the ramp.
        let inv_steps = 1.0 / (end_idx - start_idx) as f32;
        let step_r = (end_color.r - start_color.r) * inv_steps;
        let step_g = (end_color.g - start_color.g) * inv_steps;
        let step_b = (end_color.b - start_color.b) * inv_steps;
        let step_a = (end_color.a - start_color.a) * inv_steps;

        let mut cur_color = *start_color;

        // Walk the ramp writing start and end colors for each entry.
        for index in start_idx .. end_idx {
            let entry = &mut entries[index];
            entry.start_color = cur_color;
            cur_color.r += step_r;
            cur_color.g += step_g;
            cur_color.b += step_b;
            cur_color.a += step_a;
            entry.end_color = cur_color;
        }
    }

    /// Compute an index into the gradient entry table based on a gradient stop offset. This
    /// function maps offsets from [0, 1] to indices in [GRADIENT_DATA_TABLE_BEGIN, GRADIENT_DATA_TABLE_END].
    #[inline]
    fn get_index(offset: f32) -> usize {
        (offset.max(0.0).min(1.0) * GRADIENT_DATA_TABLE_SIZE as f32 +
            GRADIENT_DATA_TABLE_BEGIN as f32)
            .round() as usize
    }

    // Build the gradient data from the supplied stops, reversing them if necessary.
    fn build(&self, reverse_stops: bool, request: &mut GpuDataRequest) {
        let src_stops = self.display_list.get(self.stops_range);

        // Preconditions (should be ensured by DisplayListBuilder):
        // * we have at least two stops
        // * first stop has offset 0.0
        // * last stop has offset 1.0

        let mut src_stops = src_stops.into_iter();
        let first = src_stops.next().unwrap();
        let mut cur_color = first.color.premultiplied();
        debug_assert_eq!(first.offset, 0.0);

        // A table of gradient entries, with two colors per entry, that specify the start and end color
        // within the segment of the gradient space represented by that entry. To lookup a gradient result,
        // first the entry index is calculated to determine which two colors to interpolate between, then
        // the offset within that entry bucket is used to interpolate between the two colors in that entry.
        // This layout preserves hard stops, as the end color for a given entry can differ from the start
        // color for the following entry, despite them being adjacent. Colors are stored within in BGRA8
        // format for texture upload. This table requires the gradient color stops to be normalized to the
        // range [0, 1]. The first and last entries hold the first and last color stop colors respectively,
        // while the entries in between hold the interpolated color stop values for the range [0, 1].
        let mut entries: [GradientDataEntry; GRADIENT_DATA_SIZE] = unsafe { mem::uninitialized() };

        if reverse_stops {
            // Fill in the first entry (for reversed stops) with the first color stop
            self.fill_colors(
                GRADIENT_DATA_LAST_STOP,
                GRADIENT_DATA_LAST_STOP + 1,
                &cur_color,
                &cur_color,
                &mut entries,
            );

            // Fill in the center of the gradient table, generating a color ramp between each consecutive pair
            // of gradient stops. Each iteration of a loop will fill the indices in [next_idx, cur_idx). The
            // loop will then fill indices in [GRADIENT_DATA_TABLE_BEGIN, GRADIENT_DATA_TABLE_END).
            let mut cur_idx = GRADIENT_DATA_TABLE_END;
            for next in src_stops {
                let next_color = next.color.premultiplied();
                let next_idx = Self::get_index(1.0 - next.offset);

                if next_idx < cur_idx {
                    self.fill_colors(next_idx, cur_idx, &next_color, &cur_color, &mut entries);
                    cur_idx = next_idx;
                }

                cur_color = next_color;
            }
            debug_assert_eq!(cur_idx, GRADIENT_DATA_TABLE_BEGIN);

            // Fill in the last entry (for reversed stops) with the last color stop
            self.fill_colors(
                GRADIENT_DATA_FIRST_STOP,
                GRADIENT_DATA_FIRST_STOP + 1,
                &cur_color,
                &cur_color,
                &mut entries,
            );
        } else {
            // Fill in the first entry with the first color stop
            self.fill_colors(
                GRADIENT_DATA_FIRST_STOP,
                GRADIENT_DATA_FIRST_STOP + 1,
                &cur_color,
                &cur_color,
                &mut entries,
            );

            // Fill in the center of the gradient table, generating a color ramp between each consecutive pair
            // of gradient stops. Each iteration of a loop will fill the indices in [cur_idx, next_idx). The
            // loop will then fill indices in [GRADIENT_DATA_TABLE_BEGIN, GRADIENT_DATA_TABLE_END).
            let mut cur_idx = GRADIENT_DATA_TABLE_BEGIN;
            for next in src_stops {
                let next_color = next.color.premultiplied();
                let next_idx = Self::get_index(next.offset);

                if next_idx > cur_idx {
                    self.fill_colors(cur_idx, next_idx, &cur_color, &next_color, &mut entries);
                    cur_idx = next_idx;
                }

                cur_color = next_color;
            }
            debug_assert_eq!(cur_idx, GRADIENT_DATA_TABLE_END);

            // Fill in the last entry with the last color stop
            self.fill_colors(
                GRADIENT_DATA_LAST_STOP,
                GRADIENT_DATA_LAST_STOP + 1,
                &cur_color,
                &cur_color,
                &mut entries,
            );
        }

        for entry in entries.iter() {
            request.push(entry.start_color);
            request.push(entry.end_color);
        }
    }
}

#[derive(Debug)]
pub struct RadialGradientPrimitiveCpu {
    pub stops_range: ItemRange<GradientStop>,
    pub extend_mode: ExtendMode,
    pub gpu_data_count: i32,
    pub gpu_blocks: [GpuBlockData; 3],
}

impl RadialGradientPrimitiveCpu {
    fn build_gpu_blocks_for_angle_radial(
        &self,
        display_list: &BuiltDisplayList,
        mut request: GpuDataRequest,
    ) {
        request.extend_from_slice(&self.gpu_blocks);

        let gradient_builder = GradientGpuBlockBuilder::new(self.stops_range, display_list);
        gradient_builder.build(false, &mut request);
    }
}

#[derive(Debug, Clone)]
pub struct TextRunPrimitiveCpu {
    pub font: FontInstance,
    pub offset: LayerVector2D,
    pub glyph_range: ItemRange<GlyphInstance>,
    pub glyph_count: usize,
    pub glyph_keys: Vec<GlyphKey>,
    pub glyph_gpu_blocks: Vec<GpuBlockData>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum TextRunMode {
    Normal,
    Shadow,
}

impl TextRunPrimitiveCpu {
    pub fn get_font(&self,
                    run_mode: TextRunMode,
                    device_pixel_ratio: f32,
    ) -> FontInstance {
        let mut font = self.font.clone();
        match run_mode {
            TextRunMode::Normal => {}
            TextRunMode::Shadow => {
                // Shadows never use subpixel AA, but need to respect the alpha/mono flag
                // for reftests.
                font.render_mode = font.render_mode.limit_by(FontRenderMode::Alpha);
            }
        };
        font.size = font.size.scale_by(device_pixel_ratio);
        font
    }

    fn prepare_for_render(
        &mut self,
        resource_cache: &mut ResourceCache,
        device_pixel_ratio: f32,
        display_list: &BuiltDisplayList,
        run_mode: TextRunMode,
        gpu_cache: &mut GpuCache,
    ) {
        let font = self.get_font(run_mode, device_pixel_ratio);

        // Cache the glyph positions, if not in the cache already.
        // TODO(gw): In the future, remove `glyph_instances`
        //           completely, and just reference the glyphs
        //           directly from the display list.
        if self.glyph_keys.is_empty() {
            let subpx_dir = font.subpx_dir.limit_by(font.render_mode);
            let src_glyphs = display_list.get(self.glyph_range);

            // TODO(gw): If we support chunks() on AuxIter
            //           in the future, this code below could
            //           be much simpler...
            let mut gpu_block = GpuBlockData::empty();
            for (i, src) in src_glyphs.enumerate() {
                let key = GlyphKey::new(src.index, src.point, font.render_mode, subpx_dir);
                self.glyph_keys.push(key);

                // Two glyphs are packed per GPU block.

                if (i & 1) == 0 {
                    gpu_block.data[0] = src.point.x;
                    gpu_block.data[1] = src.point.y;
                } else {
                    gpu_block.data[2] = src.point.x;
                    gpu_block.data[3] = src.point.y;
                    self.glyph_gpu_blocks.push(gpu_block);
                }
            }

            // Ensure the last block is added in the case
            // of an odd number of glyphs.
            if (self.glyph_keys.len() & 1) != 0 {
                self.glyph_gpu_blocks.push(gpu_block);
            }
        }

        resource_cache.request_glyphs(font, &self.glyph_keys, gpu_cache);
    }

    fn write_gpu_blocks(&self, request: &mut GpuDataRequest) {
        request.push(ColorF::from(self.font.color));
        request.push([
            self.offset.x,
            self.offset.y,
            self.font.subpx_dir.limit_by(self.font.render_mode) as u32 as f32,
            0.0,
        ]);
        request.extend_from_slice(&self.glyph_gpu_blocks);

        assert!(request.current_used_block_num() <= MAX_VERTEX_TEXTURE_WIDTH);
    }
}

#[derive(Debug)]
#[repr(C)]
struct ClipRect {
    rect: LayerRect,
    mode: f32,
}

#[derive(Debug)]
#[repr(C)]
struct ClipCorner {
    rect: LayerRect,
    outer_radius_x: f32,
    outer_radius_y: f32,
    inner_radius_x: f32,
    inner_radius_y: f32,
}

impl ToGpuBlocks for ClipCorner {
    fn write_gpu_blocks(&self, mut request: GpuDataRequest) {
        self.write(&mut request)
    }
}

impl ClipCorner {
    fn write(&self, request: &mut GpuDataRequest) {
        request.push(self.rect);
        request.push([
            self.outer_radius_x,
            self.outer_radius_y,
            self.inner_radius_x,
            self.inner_radius_y,
        ]);
    }

    fn uniform(rect: LayerRect, outer_radius: f32, inner_radius: f32) -> ClipCorner {
        ClipCorner {
            rect,
            outer_radius_x: outer_radius,
            outer_radius_y: outer_radius,
            inner_radius_x: inner_radius,
            inner_radius_y: inner_radius,
        }
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct ImageMaskData {
    pub local_rect: LayerRect,
}

impl ToGpuBlocks for ImageMaskData {
    fn write_gpu_blocks(&self, mut request: GpuDataRequest) {
        request.push(self.local_rect);
    }
}

#[derive(Debug)]
pub struct ClipData {
    rect: ClipRect,
    top_left: ClipCorner,
    top_right: ClipCorner,
    bottom_left: ClipCorner,
    bottom_right: ClipCorner,
}

impl ClipData {
    pub fn rounded_rect(rect: &LayerRect, radii: &BorderRadius, mode: ClipMode) -> ClipData {
        ClipData {
            rect: ClipRect {
                rect: *rect,
                mode: mode as u32 as f32,
            },
            top_left: ClipCorner {
                rect: LayerRect::new(
                    LayerPoint::new(rect.origin.x, rect.origin.y),
                    LayerSize::new(radii.top_left.width, radii.top_left.height),
                ),
                outer_radius_x: radii.top_left.width,
                outer_radius_y: radii.top_left.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            top_right: ClipCorner {
                rect: LayerRect::new(
                    LayerPoint::new(
                        rect.origin.x + rect.size.width - radii.top_right.width,
                        rect.origin.y,
                    ),
                    LayerSize::new(radii.top_right.width, radii.top_right.height),
                ),
                outer_radius_x: radii.top_right.width,
                outer_radius_y: radii.top_right.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            bottom_left: ClipCorner {
                rect: LayerRect::new(
                    LayerPoint::new(
                        rect.origin.x,
                        rect.origin.y + rect.size.height - radii.bottom_left.height,
                    ),
                    LayerSize::new(radii.bottom_left.width, radii.bottom_left.height),
                ),
                outer_radius_x: radii.bottom_left.width,
                outer_radius_y: radii.bottom_left.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            bottom_right: ClipCorner {
                rect: LayerRect::new(
                    LayerPoint::new(
                        rect.origin.x + rect.size.width - radii.bottom_right.width,
                        rect.origin.y + rect.size.height - radii.bottom_right.height,
                    ),
                    LayerSize::new(radii.bottom_right.width, radii.bottom_right.height),
                ),
                outer_radius_x: radii.bottom_right.width,
                outer_radius_y: radii.bottom_right.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
        }
    }

    pub fn uniform(rect: LayerRect, radius: f32, mode: ClipMode) -> ClipData {
        ClipData {
            rect: ClipRect {
                rect,
                mode: mode as u32 as f32,
            },
            top_left: ClipCorner::uniform(
                LayerRect::new(
                    LayerPoint::new(rect.origin.x, rect.origin.y),
                    LayerSize::new(radius, radius),
                ),
                radius,
                0.0,
            ),
            top_right: ClipCorner::uniform(
                LayerRect::new(
                    LayerPoint::new(rect.origin.x + rect.size.width - radius, rect.origin.y),
                    LayerSize::new(radius, radius),
                ),
                radius,
                0.0,
            ),
            bottom_left: ClipCorner::uniform(
                LayerRect::new(
                    LayerPoint::new(rect.origin.x, rect.origin.y + rect.size.height - radius),
                    LayerSize::new(radius, radius),
                ),
                radius,
                0.0,
            ),
            bottom_right: ClipCorner::uniform(
                LayerRect::new(
                    LayerPoint::new(
                        rect.origin.x + rect.size.width - radius,
                        rect.origin.y + rect.size.height - radius,
                    ),
                    LayerSize::new(radius, radius),
                ),
                radius,
                0.0,
            ),
        }
    }

    pub fn write(&self, request: &mut GpuDataRequest) {
        request.push(self.rect.rect);
        request.push([self.rect.mode, 0.0, 0.0, 0.0]);
        for corner in &[
            &self.top_left,
            &self.top_right,
            &self.bottom_left,
            &self.bottom_right,
        ] {
            corner.write(request);
        }
    }
}

#[derive(Debug)]
pub enum PrimitiveContainer {
    Rectangle(RectanglePrimitive),
    TextRun(TextRunPrimitiveCpu),
    Image(ImagePrimitiveCpu),
    YuvImage(YuvImagePrimitiveCpu),
    Border(BorderPrimitiveCpu),
    AlignedGradient(GradientPrimitiveCpu),
    AngleGradient(GradientPrimitiveCpu),
    RadialGradient(RadialGradientPrimitiveCpu),
    BoxShadow(BoxShadowPrimitiveCpu),
    Picture(PicturePrimitive),
    Line(LinePrimitive),
}

pub struct PrimitiveStore {
    /// CPU side information only.
    pub cpu_rectangles: Vec<RectanglePrimitive>,
    pub cpu_text_runs: Vec<TextRunPrimitiveCpu>,
    pub cpu_pictures: Vec<PicturePrimitive>,
    pub cpu_images: Vec<ImagePrimitiveCpu>,
    pub cpu_yuv_images: Vec<YuvImagePrimitiveCpu>,
    pub cpu_gradients: Vec<GradientPrimitiveCpu>,
    pub cpu_radial_gradients: Vec<RadialGradientPrimitiveCpu>,
    pub cpu_metadata: Vec<PrimitiveMetadata>,
    pub cpu_borders: Vec<BorderPrimitiveCpu>,
    pub cpu_box_shadows: Vec<BoxShadowPrimitiveCpu>,
    pub cpu_lines: Vec<LinePrimitive>,
}

impl PrimitiveStore {
    pub fn new() -> PrimitiveStore {
        PrimitiveStore {
            cpu_metadata: Vec::new(),
            cpu_rectangles: Vec::new(),
            cpu_text_runs: Vec::new(),
            cpu_pictures: Vec::new(),
            cpu_images: Vec::new(),
            cpu_yuv_images: Vec::new(),
            cpu_gradients: Vec::new(),
            cpu_radial_gradients: Vec::new(),
            cpu_borders: Vec::new(),
            cpu_box_shadows: Vec::new(),
            cpu_lines: Vec::new(),
        }
    }

    pub fn recycle(self) -> Self {
        PrimitiveStore {
            cpu_metadata: recycle_vec(self.cpu_metadata),
            cpu_rectangles: recycle_vec(self.cpu_rectangles),
            cpu_text_runs: recycle_vec(self.cpu_text_runs),
            cpu_pictures: recycle_vec(self.cpu_pictures),
            cpu_images: recycle_vec(self.cpu_images),
            cpu_yuv_images: recycle_vec(self.cpu_yuv_images),
            cpu_gradients: recycle_vec(self.cpu_gradients),
            cpu_radial_gradients: recycle_vec(self.cpu_radial_gradients),
            cpu_borders: recycle_vec(self.cpu_borders),
            cpu_box_shadows: recycle_vec(self.cpu_box_shadows),
            cpu_lines: recycle_vec(self.cpu_lines),
        }
    }

    pub fn add_primitive(
        &mut self,
        local_rect: &LayerRect,
        local_clip_rect: &LayerRect,
        is_backface_visible: bool,
        clip_sources: ClipSourcesHandle,
        tag: Option<ItemTag>,
        container: PrimitiveContainer,
    ) -> PrimitiveIndex {
        let prim_index = self.cpu_metadata.len();

        let base_metadata = PrimitiveMetadata {
            clip_sources,
            gpu_location: GpuCacheHandle::new(),
            clip_task_id: None,
            local_rect: *local_rect,
            local_clip_rect: *local_clip_rect,
            is_backface_visible: is_backface_visible,
            screen_rect: None,
            tag,

            opacity: PrimitiveOpacity::translucent(),
            prim_kind: PrimitiveKind::Rectangle,
            cpu_prim_index: SpecificPrimitiveIndex(0),
        };

        let metadata = match container {
            PrimitiveContainer::Rectangle(rect) => {
                let metadata = PrimitiveMetadata {
                    opacity: PrimitiveOpacity::from_alpha(rect.color.a),
                    prim_kind: PrimitiveKind::Rectangle,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_rectangles.len()),
                    ..base_metadata
                };

                self.cpu_rectangles.push(rect);

                metadata
            }
            PrimitiveContainer::Line(line) => {
                let metadata = PrimitiveMetadata {
                    opacity: PrimitiveOpacity::translucent(),
                    prim_kind: PrimitiveKind::Line,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_lines.len()),
                    ..base_metadata
                };

                self.cpu_lines.push(line);
                metadata
            }
            PrimitiveContainer::TextRun(text_cpu) => {
                let metadata = PrimitiveMetadata {
                    opacity: PrimitiveOpacity::translucent(),
                    prim_kind: PrimitiveKind::TextRun,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_text_runs.len()),
                    ..base_metadata
                };

                self.cpu_text_runs.push(text_cpu);
                metadata
            }
            PrimitiveContainer::Picture(picture) => {
                let metadata = PrimitiveMetadata {
                    opacity: PrimitiveOpacity::translucent(),
                    prim_kind: PrimitiveKind::Picture,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_pictures.len()),
                    ..base_metadata
                };

                self.cpu_pictures.push(picture);
                metadata
            }
            PrimitiveContainer::Image(image_cpu) => {
                let metadata = PrimitiveMetadata {
                    opacity: PrimitiveOpacity::translucent(),
                    prim_kind: PrimitiveKind::Image,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_images.len()),
                    ..base_metadata
                };

                self.cpu_images.push(image_cpu);
                metadata
            }
            PrimitiveContainer::YuvImage(image_cpu) => {
                let metadata = PrimitiveMetadata {
                    opacity: PrimitiveOpacity::opaque(),
                    prim_kind: PrimitiveKind::YuvImage,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_yuv_images.len()),
                    ..base_metadata
                };

                self.cpu_yuv_images.push(image_cpu);
                metadata
            }
            PrimitiveContainer::Border(border_cpu) => {
                let metadata = PrimitiveMetadata {
                    opacity: PrimitiveOpacity::translucent(),
                    prim_kind: PrimitiveKind::Border,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_borders.len()),
                    ..base_metadata
                };

                self.cpu_borders.push(border_cpu);
                metadata
            }
            PrimitiveContainer::AlignedGradient(gradient_cpu) => {
                let metadata = PrimitiveMetadata {
                    opacity: PrimitiveOpacity::translucent(),
                    prim_kind: PrimitiveKind::AlignedGradient,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_gradients.len()),
                    ..base_metadata
                };

                self.cpu_gradients.push(gradient_cpu);
                metadata
            }
            PrimitiveContainer::AngleGradient(gradient_cpu) => {
                let metadata = PrimitiveMetadata {
                    // TODO: calculate if the gradient is actually opaque
                    opacity: PrimitiveOpacity::translucent(),
                    prim_kind: PrimitiveKind::AngleGradient,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_gradients.len()),
                    ..base_metadata
                };

                self.cpu_gradients.push(gradient_cpu);
                metadata
            }
            PrimitiveContainer::RadialGradient(radial_gradient_cpu) => {
                let metadata = PrimitiveMetadata {
                    // TODO: calculate if the gradient is actually opaque
                    opacity: PrimitiveOpacity::translucent(),
                    prim_kind: PrimitiveKind::RadialGradient,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_radial_gradients.len()),
                    ..base_metadata
                };

                self.cpu_radial_gradients.push(radial_gradient_cpu);
                metadata
            }
            PrimitiveContainer::BoxShadow(box_shadow) => {
                let metadata = PrimitiveMetadata {
                    opacity: PrimitiveOpacity::translucent(),
                    prim_kind: PrimitiveKind::BoxShadow,
                    cpu_prim_index: SpecificPrimitiveIndex(self.cpu_box_shadows.len()),
                    ..base_metadata
                };

                self.cpu_box_shadows.push(box_shadow);
                metadata
            }
        };

        self.cpu_metadata.push(metadata);

        PrimitiveIndex(prim_index)
    }

    pub fn get_metadata(&self, index: PrimitiveIndex) -> &PrimitiveMetadata {
        &self.cpu_metadata[index.0]
    }

    pub fn prim_count(&self) -> usize {
        self.cpu_metadata.len()
    }

    /// Add any task dependencies for this primitive to the provided task.
    pub fn add_render_tasks_for_prim(&self, prim_index: PrimitiveIndex, task: &mut RenderTask) {
        // Add any dynamic render tasks needed to render this primitive
        let metadata = &self.cpu_metadata[prim_index.0];

        let render_task_id = match metadata.prim_kind {
            PrimitiveKind::BoxShadow => {
                let box_shadow = &self.cpu_box_shadows[metadata.cpu_prim_index.0];
                box_shadow.render_task_id
            }
            PrimitiveKind::Picture => {
                let picture = &self.cpu_pictures[metadata.cpu_prim_index.0];
                picture.render_task_id
            }
            PrimitiveKind::Rectangle |
            PrimitiveKind::TextRun |
            PrimitiveKind::Image |
            PrimitiveKind::AlignedGradient |
            PrimitiveKind::YuvImage |
            PrimitiveKind::Border |
            PrimitiveKind::AngleGradient |
            PrimitiveKind::RadialGradient |
            PrimitiveKind::Line => None,
        };

        if let Some(render_task_id) = render_task_id {
            task.children.push(render_task_id);
        }

        if let Some(clip_task_id) = metadata.clip_task_id {
            task.children.push(clip_task_id);
        }
    }

    /// Returns true if the bounding box needs to be updated.
    fn prepare_prim_for_render_inner(
        &mut self,
        prim_index: PrimitiveIndex,
        prim_context: &PrimitiveContext,
        resource_cache: &mut ResourceCache,
        gpu_cache: &mut GpuCache,
        // For some primitives, we need to mark dependencies as needed for rendering
        // without spawning new tasks, since there will be another call to
        // `prepare_prim_for_render_inner` specifically for this primitive later on.
        render_tasks: Option<&mut RenderTaskTree>,
        text_run_mode: TextRunMode,
    ) {
        let metadata = &mut self.cpu_metadata[prim_index.0];
        match metadata.prim_kind {
            PrimitiveKind::Rectangle | PrimitiveKind::Border | PrimitiveKind::Line => {}
            PrimitiveKind::BoxShadow => {
                // TODO(gw): Account for zoom factor!
                // Here, we calculate the size of the patch required in order
                // to create the box shadow corner. First, scale it by the
                // device pixel ratio since the cache shader expects vertices
                // in device space. The shader adds a 1-pixel border around
                // the patch, in order to prevent bilinear filter artifacts as
                // the patch is clamped / mirrored across the box shadow rect.
                let box_shadow = &mut self.cpu_box_shadows[metadata.cpu_prim_index.0];
                let edge_size = box_shadow.edge_size.ceil() * prim_context.device_pixel_ratio;
                let edge_size = edge_size as i32 + 2; // Account for bilinear filtering
                let cache_size = DeviceIntSize::new(edge_size, edge_size);

                let cache_key = BoxShadowPrimitiveCacheKey {
                    blur_radius: Au::from_f32_px(box_shadow.blur_radius),
                    border_radius: Au::from_f32_px(box_shadow.border_radius),
                    inverted: box_shadow.inverted != 0.0,
                    shadow_rect_size: Size2D::new(
                        Au::from_f32_px(box_shadow.bs_rect.size.width),
                        Au::from_f32_px(box_shadow.bs_rect.size.height),
                    ),
                };

                // Create a render task for this box shadow primitive. This renders a small
                // portion of the box shadow to a render target. That portion is then
                // stretched over the actual primitive rect by the box shadow primitive
                // shader, to reduce the number of pixels that the expensive box
                // shadow shader needs to run on.
                // TODO(gw): In the future, we can probably merge the box shadow
                // primitive (stretch) shader with the generic cached primitive shader.
                let render_task = RenderTask::new_box_shadow(
                    cache_key,
                    cache_size,
                    prim_index
                );

                // ignore the new task if we are in a dependency context
                box_shadow.render_task_id = render_tasks.map(|rt| rt.add(render_task));
            }
            PrimitiveKind::Picture => {
                let picture = &mut self.cpu_pictures[metadata.cpu_prim_index.0];

                // This is a shadow element. Create a render task that will
                // render the text run to a target, and then apply a gaussian
                // blur to that text run in order to build the actual primitive
                // which will be blitted to the framebuffer.
                let cache_width =
                    (metadata.local_rect.size.width * prim_context.device_pixel_ratio).ceil() as i32;
                let cache_height =
                    (metadata.local_rect.size.height * prim_context.device_pixel_ratio).ceil() as i32;
                let cache_size = DeviceIntSize::new(cache_width, cache_height);
                let blur_radius = picture.as_shadow().blur_radius;
                let blur_radius = device_length(blur_radius, prim_context.device_pixel_ratio);

                // ignore new tasks if we are in a dependency context
                picture.render_task_id = render_tasks.map(|rt| {
                    let picture_task = RenderTask::new_picture(cache_size, prim_index);
                    let picture_task_id = rt.add(picture_task);
                    let render_task =
                        RenderTask::new_blur(blur_radius, picture_task_id, rt);
                    rt.add(render_task)
                });
            }
            PrimitiveKind::TextRun => {
                let text = &mut self.cpu_text_runs[metadata.cpu_prim_index.0];
                text.prepare_for_render(
                    resource_cache,
                    prim_context.device_pixel_ratio,
                    prim_context.display_list,
                    text_run_mode,
                    gpu_cache,
                );
            }
            PrimitiveKind::Image => {
                let image_cpu = &mut self.cpu_images[metadata.cpu_prim_index.0];

                resource_cache.request_image(
                    image_cpu.image_key,
                    image_cpu.image_rendering,
                    image_cpu.tile_offset,
                    gpu_cache,
                );

                // TODO(gw): This doesn't actually need to be calculated each frame.
                // It's cheap enough that it's not worth introducing a cache for images
                // right now, but if we introduce a cache for images for some other
                // reason then we might as well cache this with it.
                if let Some(image_properties) =
                    resource_cache.get_image_properties(image_cpu.image_key)
                {
                    metadata.opacity.is_opaque = image_properties.descriptor.is_opaque &&
                        image_cpu.tile_spacing.width == 0.0 &&
                        image_cpu.tile_spacing.height == 0.0;
                }
            }
            PrimitiveKind::YuvImage => {
                let image_cpu = &mut self.cpu_yuv_images[metadata.cpu_prim_index.0];

                let channel_num = image_cpu.format.get_plane_num();
                debug_assert!(channel_num <= 3);
                for channel in 0 .. channel_num {
                    resource_cache.request_image(
                        image_cpu.yuv_key[channel],
                        image_cpu.image_rendering,
                        None,
                        gpu_cache,
                    );
                }
            }
            PrimitiveKind::AlignedGradient |
            PrimitiveKind::AngleGradient |
            PrimitiveKind::RadialGradient => {}
        }

        // Mark this GPU resource as required for this frame.
        if let Some(mut request) = gpu_cache.request(&mut metadata.gpu_location) {
            request.push(metadata.local_rect);
            request.push(metadata.local_clip_rect);

            match metadata.prim_kind {
                PrimitiveKind::Rectangle => {
                    let rect = &self.cpu_rectangles[metadata.cpu_prim_index.0];
                    rect.write_gpu_blocks(request);
                }
                PrimitiveKind::Line => {
                    let line = &self.cpu_lines[metadata.cpu_prim_index.0];
                    line.write_gpu_blocks(request);
                }
                PrimitiveKind::Border => {
                    let border = &self.cpu_borders[metadata.cpu_prim_index.0];
                    border.write_gpu_blocks(request);
                }
                PrimitiveKind::BoxShadow => {
                    let box_shadow = &self.cpu_box_shadows[metadata.cpu_prim_index.0];
                    box_shadow.write_gpu_blocks(request);
                }
                PrimitiveKind::Image => {
                    let image = &self.cpu_images[metadata.cpu_prim_index.0];
                    image.write_gpu_blocks(request);
                }
                PrimitiveKind::YuvImage => {
                    let yuv_image = &self.cpu_yuv_images[metadata.cpu_prim_index.0];
                    yuv_image.write_gpu_blocks(request);
                }
                PrimitiveKind::AlignedGradient => {
                    let gradient = &self.cpu_gradients[metadata.cpu_prim_index.0];
                    metadata.opacity = gradient.build_gpu_blocks_for_aligned(prim_context.display_list, request);
                }
                PrimitiveKind::AngleGradient => {
                    let gradient = &self.cpu_gradients[metadata.cpu_prim_index.0];
                    gradient.build_gpu_blocks_for_angle_radial(prim_context.display_list, request);
                }
                PrimitiveKind::RadialGradient => {
                    let gradient = &self.cpu_radial_gradients[metadata.cpu_prim_index.0];
                    gradient.build_gpu_blocks_for_angle_radial(prim_context.display_list, request);
                }
                PrimitiveKind::TextRun => {
                    let text = &self.cpu_text_runs[metadata.cpu_prim_index.0];
                    text.write_gpu_blocks(&mut request);
                }
                PrimitiveKind::Picture => {
                    let picture = &self.cpu_pictures[metadata.cpu_prim_index.0];
                    let shadow = picture.as_shadow();
                    request.push(shadow.color);
                    request.push([
                        shadow.offset.x,
                        shadow.offset.y,
                        shadow.blur_radius,
                        0.0,
                    ]);
                }
            }
        }
    }

    fn update_clip_task(
        &mut self,
        prim_index: PrimitiveIndex,
        prim_context: &PrimitiveContext,
        prim_screen_rect: DeviceIntRect,
        resource_cache: &mut ResourceCache,
        gpu_cache: &mut GpuCache,
        render_tasks: &mut RenderTaskTree,
        clip_store: &mut ClipStore,
    ) -> bool {
        let metadata = &mut self.cpu_metadata[prim_index.0];
        clip_store.get_mut(&metadata.clip_sources).update(
            &prim_context.packed_layer.transform,
            gpu_cache,
            resource_cache,
            prim_context.device_pixel_ratio,
        );

        // Try to create a mask if we may need to.
        let prim_clips = clip_store.get(&metadata.clip_sources);
        let is_axis_aligned = prim_context.packed_layer.transform.preserves_2d_axis_alignment();
        let clip_task = if prim_clips.is_masking() {
            // Take into account the actual clip info of the primitive, and
            // mutate the current bounds accordingly.
            let mask_rect = match prim_clips.bounds.outer {
                Some(ref outer) => match prim_screen_rect.intersection(&outer.device_rect) {
                    Some(rect) => rect,
                    None => {
                        metadata.screen_rect = None;
                        return false;
                    }
                },
                _ => prim_screen_rect,
            };

            let extra = ClipWorkItem {
                layer_index: prim_context.packed_layer_index,
                clip_sources: metadata.clip_sources.weak(),
                apply_rectangles: false,
            };

            RenderTask::new_mask(
                None,
                mask_rect,
                &prim_context.current_clip_stack,
                Some(extra),
                prim_screen_rect,
                clip_store,
                is_axis_aligned,
            )
        } else if !prim_context.current_clip_stack.is_empty() {
            // If the primitive doesn't have a specific clip, key the task ID off the
            // stacking context. This means that two primitives which are only clipped
            // by the stacking context stack can share clip masks during render task
            // assignment to targets.
            RenderTask::new_mask(
                Some(prim_context.clip_id),
                prim_context.clip_bounds,
                &prim_context.current_clip_stack,
                None,
                prim_screen_rect,
                clip_store,
                is_axis_aligned,
            )
        } else {
            None
        };

        metadata.clip_task_id = clip_task.map(|clip_task| render_tasks.add(clip_task));
        true
    }

    /// Returns true if the bounding box needs to be updated.
    pub fn prepare_prim_for_render(
        &mut self,
        prim_index: PrimitiveIndex,
        prim_context: &PrimitiveContext,
        resource_cache: &mut ResourceCache,
        gpu_cache: &mut GpuCache,
        render_tasks: &mut RenderTaskTree,
        clip_store: &mut ClipStore,
    ) -> Option<Geometry> {
        let (geometry, dependent_primitives) = {
            let metadata = &mut self.cpu_metadata[prim_index.0];
            metadata.screen_rect = None;

            if metadata.local_rect.size.width <= 0.0 ||
               metadata.local_rect.size.height <= 0.0 {
                warn!("invalid primitive rect {:?}", metadata.local_rect);
                return None;
            }

            if !metadata.is_backface_visible &&
               prim_context.packed_layer.transform.is_backface_visible() {
                return None;
            }

            let local_rect = metadata
                .local_rect
                .intersection(&metadata.local_clip_rect)
                .and_then(|rect| rect.intersection(&prim_context.packed_layer.local_clip_rect));

            let local_rect = match local_rect {
                Some(local_rect) => local_rect,
                None => return None,
            };

            let xf_rect = TransformedRect::new(
                &local_rect,
                &prim_context.packed_layer.transform,
                prim_context.device_pixel_ratio
            );

            metadata.screen_rect = xf_rect
                .bounding_rect
                .intersection(&prim_context.clip_bounds);

            let geometry = match metadata.screen_rect {
                Some(device_rect) => Geometry {
                    local_rect,
                    device_rect,
                },
                None => return None,
            };

            let dependencies = match metadata.prim_kind {
                PrimitiveKind::Picture =>
                    self.cpu_pictures[metadata.cpu_prim_index.0].prim_runs.clone(),
                _ => Vec::new(),
            };
            (geometry, dependencies)
        };

        // Recurse into any sub primitives and prepare them for rendering first.
        // TODO(gw): This code is a bit hacky to work around the borrow checker.
        //           Specifically, the clone() below on the primitive list for
        //           text shadow primitives. Consider restructuring this code to
        //           avoid borrow checker issues.
        for run in dependent_primitives {
            for i in 0 .. run.count {
                let sub_prim_index = PrimitiveIndex(run.prim_index.0 + i);

                self.prepare_prim_for_render_inner(
                    sub_prim_index,
                    prim_context,
                    resource_cache,
                    gpu_cache,
                    None,
                    TextRunMode::Shadow,
                );
            }
        }

        if !self.update_clip_task(
            prim_index,
            prim_context,
            geometry.device_rect,
            resource_cache,
            gpu_cache,
            render_tasks,
            clip_store,
        ) {
            return None;
        }

        self.prepare_prim_for_render_inner(
            prim_index,
            prim_context,
            resource_cache,
            gpu_cache,
            Some(render_tasks),
            TextRunMode::Normal,
        );

        Some(geometry)
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

        delta_left >= 0f32 && delta_top >= 0f32 && delta_right >= 0f32 && delta_bottom >= 0f32 &&
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
