/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use batch::{VertexBufferId, TileParams};
use device::{TextureId, TextureFilter};
use euclid::{Matrix4D, Point2D, Rect, Size2D};
use fnv::FnvHasher;
use freelist::{FreeListItem, FreeListItemId};
use num_traits::Zero;
use profiler::BackendProfileCounters;
use std::collections::{HashMap, HashSet};
use std::f32;
use std::hash::BuildHasherDefault;
use std::ops::{Add, Sub};
use std::path::PathBuf;
use std::sync::Arc;
use texture_cache::BorderType;
use tiling;
use webrender_traits::{FontKey, Epoch, ColorF, PipelineId};
use webrender_traits::{ImageFormat, MixBlendMode, NativeFontHandle, DisplayItem, ScrollLayerId};

#[derive(Hash, Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct DevicePixel(pub i32);

impl DevicePixel {
    pub fn new(value: f32, device_pixel_ratio: f32) -> DevicePixel {
        DevicePixel((value * device_pixel_ratio).round() as i32)
    }

    pub fn from_u32(value: u32) -> DevicePixel {
        DevicePixel(value as i32)
    }

    // TODO(gw): Remove eventually...
    pub fn as_f32(&self) -> f32 {
        let DevicePixel(value) = *self;
        value as f32
    }

    // TODO(gw): Remove eventually...
    pub fn as_u32(&self) -> u32 {
        let DevicePixel(value) = *self;
        value as u32
    }
}

impl Add for DevicePixel {
    type Output = DevicePixel;

    #[inline]
    fn add(self, other: DevicePixel) -> DevicePixel {
        DevicePixel(self.0 + other.0)
    }
}

impl Sub for DevicePixel {
    type Output = DevicePixel;

    fn sub(self, other: DevicePixel) -> DevicePixel {
        DevicePixel(self.0 - other.0)
    }
}

impl Zero for DevicePixel {
    fn zero() -> DevicePixel {
        DevicePixel(0)
    }

    fn is_zero(&self) -> bool {
        let DevicePixel(value) = *self;
        value == 0
    }
}

const UV_FLOAT_TO_FIXED: f32 = 65535.0;
const COLOR_FLOAT_TO_FIXED: f32 = 255.0;
pub const ANGLE_FLOAT_TO_FIXED: f32 = 65535.0;

pub const ORTHO_NEAR_PLANE: f32 = -1000000.0;
pub const ORTHO_FAR_PLANE: f32 = 1000000.0;


#[inline(always)]
pub fn max_rect() -> Rect<f32> {
    Rect::new(Point2D::new(f32::MIN, f32::MIN), Size2D::new(f32::INFINITY, f32::INFINITY))
}

pub enum FontTemplate {
    Raw(Arc<Vec<u8>>),
    Native(NativeFontHandle),
}

pub type DrawListId = FreeListItemId;

#[derive(Debug, PartialEq, Eq)]
pub enum TextureSampler {
    Color,
    Mask,

    Cache,

    CompositeLayer0,
    CompositeLayer1,
    CompositeLayer2,
    CompositeLayer3,
    CompositeLayer4,
    CompositeLayer5,
    CompositeLayer6,
    CompositeLayer7,
}

pub enum VertexAttribute {
    Position,
    PositionRect,
    ColorRectTL,
    ColorRectTR,
    ColorRectBR,
    ColorRectBL,
    ColorTexCoordRectTop,
    MaskTexCoordRectTop,
    ColorTexCoordRectBottom,
    MaskTexCoordRectBottom,
    BorderRadii,
    BorderPosition,
    BlurRadius,
    DestTextureSize,
    SourceTextureSize,
    Misc,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PackedColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl PackedColor {
    pub fn from_color(color: &ColorF) -> PackedColor {
        PackedColor {
            r: (0.5 + color.r * COLOR_FLOAT_TO_FIXED).floor() as u8,
            g: (0.5 + color.g * COLOR_FLOAT_TO_FIXED).floor() as u8,
            b: (0.5 + color.b * COLOR_FLOAT_TO_FIXED).floor() as u8,
            a: (0.5 + color.a * COLOR_FLOAT_TO_FIXED).floor() as u8,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WorkVertex {
    pub x: f32,
    pub y: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
    pub u: f32,
    pub v: f32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PackedVertexForQuad {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub color_tl: PackedColor,
    pub color_tr: PackedColor,
    pub color_br: PackedColor,
    pub color_bl: PackedColor,
    pub u_tl: f32,
    pub v_tl: f32,
    pub u_tr: f32,
    pub v_tr: f32,
    pub u_br: f32,
    pub v_br: f32,
    pub u_bl: f32,
    pub v_bl: f32,
    pub mu_tl: u16,
    pub mv_tl: u16,
    pub mu_tr: u16,
    pub mv_tr: u16,
    pub mu_br: u16,
    pub mv_br: u16,
    pub mu_bl: u16,
    pub mv_bl: u16,
    pub matrix_index: u8,
    pub clip_in_rect_index: u8,
    pub clip_out_rect_index: u8,
    pub tile_params_index: u8,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PackedVertex {
    pub pos: [f32; 2],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FontVertex {
    pub x: f32,
    pub y: f32,
    pub s: f32,
    pub t: f32,
}

/*
impl PackedVertex {
    pub fn from_components(x: f32,
                           y: f32,
                           color: &ColorF,
                           u: f32,
                           v: f32,
                           mu: f32,
                           mv: f32)
                           -> PackedVertex {
        PackedVertex {
            x: x,
            y: y,
            color: PackedColor::from_color(color),
            u: u,
            v: v,
            mu: (mu * UV_FLOAT_TO_FIXED) as u16,
            mv: (mv * UV_FLOAT_TO_FIXED) as u16,
            matrix_index: 0,
            clip_in_rect_index: 0,
            clip_out_rect_index: 0,
            tile_params_index: 0,
        }
    }

    /// Just like the above function, but doesn't scale the mask uv coordinates. This is useful
    /// for the filter fragment shader, which uses the mask uv coordinates to store the texture
    /// size.
    pub fn from_components_unscaled_muv(x: f32, y: f32,
                                        color: &ColorF,
                                        u: f32, v: f32,
                                        mu: u16, mv: u16)
                                        -> PackedVertex {
        PackedVertex {
            x: x,
            y: y,
            color: PackedColor::from_color(color),
            u: u,
            v: v,
            mu: mu,
            mv: mv,
            matrix_index: 0,
            clip_in_rect_index: 0,
            clip_out_rect_index: 0,
            tile_params_index: 0,
        }
    }
}*/

#[derive(Debug)]
pub struct DebugFontVertex {
    pub x: f32,
    pub y: f32,
    pub color: PackedColor,
    pub u: f32,
    pub v: f32,
}

impl DebugFontVertex {
    pub fn new(x: f32, y: f32, u: f32, v: f32, color: PackedColor) -> DebugFontVertex {
        DebugFontVertex {
            x: x,
            y: y,
            color: color,
            u: u,
            v: v,
        }
    }
}

pub struct DebugColorVertex {
    pub x: f32,
    pub y: f32,
    pub color: PackedColor,
}

impl DebugColorVertex {
    pub fn new(x: f32, y: f32, color: PackedColor) -> DebugColorVertex {
        DebugColorVertex {
            x: x,
            y: y,
            color: color,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum RenderTargetMode {
    None,
    RenderTarget,
}

#[derive(Debug)]
pub enum TextureUpdateDetails {
    Raw,
    Blit(Vec<u8>),
    Blur(Vec<u8>, Size2D<u32>, Au, TextureImage, TextureImage, BorderType),
    /// Blur radius, border radius, box size, raster origin, and whether inverted, respectively.
    BoxShadow(DevicePixel, DevicePixel, Size2D<DevicePixel>, Point2D<DevicePixel>, bool, BorderType),
}

#[derive(Clone, Copy, Debug)]
pub struct TextureImage {
    pub texture_id: TextureId,
    pub texel_uv: Rect<f32>,
    pub pixel_uv: Point2D<u32>,
}

pub enum TextureUpdateOp {
    Create(u32, u32, ImageFormat, TextureFilter, RenderTargetMode, Option<Vec<u8>>),
    Update(u32, u32, u32, u32, TextureUpdateDetails),
    Grow(u32, u32, ImageFormat, TextureFilter, RenderTargetMode),
}

pub struct TextureUpdate {
    pub id: TextureId,
    pub op: TextureUpdateOp,
}

pub struct TextureUpdateList {
    pub updates: Vec<TextureUpdate>,
}

impl TextureUpdateList {
    pub fn new() -> TextureUpdateList {
        TextureUpdateList {
            updates: Vec::new(),
        }
    }

    #[inline]
    pub fn push(&mut self, update: TextureUpdate) {
        self.updates.push(update);
    }
}

// TODO(gw): Use bitflags crate for ClearInfo...
// TODO(gw): Expand clear info to handle color, depth etc as needed.

#[derive(Clone, Debug)]
pub struct ClearInfo {
    pub clear_color: bool,
    pub clear_z: bool,
    pub clear_stencil: bool,
}

#[derive(Clone, Debug)]
pub struct DrawCall {
    pub tile_params: Vec<TileParams>,
    pub clip_rects: Vec<Rect<f32>>,
    pub vertex_buffer_id: VertexBufferId,
    pub color_texture_id: TextureId,
    pub mask_texture_id: TextureId,
    pub first_instance: u32,
    pub instance_count: u32,
}

#[derive(Debug, Clone)]
pub struct CompositeBatchJob {
    pub rect: Rect<f32>,
    pub transform: Matrix4D<f32>,
    pub child_layer_index: ChildLayerIndex,
}

#[derive(Debug, Clone)]
pub struct DrawCompositeBatchJob {
    pub rect: Rect<f32>,
    pub local_transform: Matrix4D<f32>,
    pub world_transform: Matrix4D<f32>,
    pub child_layer_index: ChildLayerIndex,
}

#[derive(Debug, Clone)]
pub struct CompositeBatchInfo {
    pub operation: CompositionOp,
    pub texture_id: TextureId,
    pub scroll_layer_id: ScrollLayerId,
    pub jobs: Vec<CompositeBatchJob>,
}

#[derive(Clone, Copy, Debug, Ord, PartialOrd, PartialEq, Eq, Hash)]
pub struct ChildLayerIndex(pub u32);

pub struct RendererFrame {
    pub pipeline_epoch_map: HashMap<PipelineId, Epoch, BuildHasherDefault<FnvHasher>>,
    pub layers_bouncing_back: HashSet<ScrollLayerId, BuildHasherDefault<FnvHasher>>,
    pub frame: Option<tiling::Frame>,
}

impl RendererFrame {
    pub fn new(pipeline_epoch_map: HashMap<PipelineId, Epoch, BuildHasherDefault<FnvHasher>>,
               layers_bouncing_back: HashSet<ScrollLayerId, BuildHasherDefault<FnvHasher>>,
               frame: Option<tiling::Frame>)
               -> RendererFrame {
        RendererFrame {
            pipeline_epoch_map: pipeline_epoch_map,
            layers_bouncing_back: layers_bouncing_back,
            frame: frame,
        }
    }
}

pub enum ResultMsg {
    UpdateTextureCache(TextureUpdateList),
    RefreshShader(PathBuf),
    NewFrame(RendererFrame, BackendProfileCounters),
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisDirection {
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone, Copy)]
pub struct StackingContextIndex(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DrawListGroupId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderTargetId(pub usize);

#[derive(Debug, Clone)]
pub struct StackingContextInfo {
    pub offset_from_layer: Point2D<f32>,
    pub local_clip_rect: Rect<f32>,
    pub transform: Matrix4D<f32>,
    pub z_clear_needed: bool,
}

#[derive(Debug)]
pub struct DrawList {
    pub items: Vec<DisplayItem>,
    pub stacking_context_index: Option<StackingContextIndex>,
    pub pipeline_id: PipelineId,
    // TODO(gw): Structure squat to remove this field.
    next_free_id: Option<FreeListItemId>,
}

impl DrawList {
    pub fn new(items: Vec<DisplayItem>, pipeline_id: PipelineId) -> DrawList {
        DrawList {
            items: items,
            stacking_context_index: None,
            pipeline_id: pipeline_id,
            next_free_id: None,
        }
    }
}

impl FreeListItem for DrawList {
    fn next_free_id(&self) -> Option<FreeListItemId> {
        self.next_free_id
    }

    fn set_next_free_id(&mut self, id: Option<FreeListItemId>) {
        self.next_free_id = id;
    }
}

#[derive(Clone, Copy, Debug, Ord, PartialOrd, PartialEq, Eq)]
pub struct DrawListItemIndex(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct RectPolygon<Varyings> {
    pub pos: Rect<f32>,
    pub varyings: Varyings,
}

#[derive(Clone, Copy, Debug)]
pub struct RectColors {
    pub top_left: ColorF,
    pub top_right: ColorF,
    pub bottom_right: ColorF,
    pub bottom_left: ColorF,
}

#[derive(Clone, Copy, Debug)]
pub struct RectUv<T> {
    pub top_left: Point2D<T>,
    pub top_right: Point2D<T>,
    pub bottom_left: Point2D<T>,
    pub bottom_right: Point2D<T>,
}

#[derive(Clone, Debug)]
pub struct PolygonPosColorUv {
    pub vertices: Vec<WorkVertex>,
}

#[derive(PartialEq, Eq, Hash)]
pub struct Glyph {
    pub size: Au,
    pub blur_radius: Au,
    pub index: u32,
}

impl Glyph {
    #[inline]
    pub fn new(size: Au, blur_radius: Au, index: u32) -> Glyph {
        Glyph {
            size: size,
            blur_radius: blur_radius,
            index: index,
        }
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct PackedVertexForTextureCacheUpdate {
    pub x: f32,
    pub y: f32,
    pub color: PackedColor,
    pub u: u16,
    pub v: u16,
    pub border_radii_outer_rx: f32,
    pub border_radii_outer_ry: f32,
    pub border_radii_inner_rx: f32,
    pub border_radii_inner_ry: f32,
    pub border_position_x: f32,
    pub border_position_y: f32,
    pub border_position_arc_center_x: f32,
    pub border_position_arc_center_y: f32,
    pub dest_texture_size_x: f32,
    pub dest_texture_size_y: f32,
    pub source_texture_size_x: f32,
    pub source_texture_size_y: f32,
    pub blur_radius: f32,
    pub misc0: u8,
    pub misc1: u8,
    pub misc2: u8,
    pub misc3: u8,
}

impl PackedVertexForTextureCacheUpdate {
    pub fn new(position: &Point2D<f32>,
               color: &ColorF,
               uv: &Point2D<f32>,
               border_radii_outer: &Point2D<f32>,
               border_radii_inner: &Point2D<f32>,
               border_position: &Point2D<f32>,
               border_position_arc_center: &Point2D<f32>,
               dest_texture_size: &Size2D<f32>,
               source_texture_size: &Size2D<f32>,
               blur_radius: f32)
               -> PackedVertexForTextureCacheUpdate {
        PackedVertexForTextureCacheUpdate {
            x: position.x,
            y: position.y,
            color: PackedColor::from_color(color),
            u: (uv.x * UV_FLOAT_TO_FIXED).round() as u16,
            v: (uv.y * UV_FLOAT_TO_FIXED).round() as u16,
            border_radii_outer_rx: border_radii_outer.x,
            border_radii_outer_ry: border_radii_outer.y,
            border_radii_inner_rx: border_radii_inner.x,
            border_radii_inner_ry: border_radii_inner.y,
            border_position_x: border_position.x,
            border_position_y: border_position.y,
            border_position_arc_center_x: border_position_arc_center.x,
            border_position_arc_center_y: border_position_arc_center.y,
            dest_texture_size_x: dest_texture_size.width,
            dest_texture_size_y: dest_texture_size.height,
            source_texture_size_x: source_texture_size.width,
            source_texture_size_y: source_texture_size.height,
            blur_radius: blur_radius,
            misc0: 0,
            misc1: 0,
            misc2: 0,
            misc3: 0,
        }
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct BoxShadowRasterOp {
    pub blur_radius: DevicePixel,
    pub border_radius: DevicePixel,
    // This is a tuple to work around the lack of `Eq` on `Rect`.
    pub box_rect_size: (DevicePixel, DevicePixel),
    pub local_raster_origin: (DevicePixel, DevicePixel),
    pub raster_size: (DevicePixel, DevicePixel),
    pub part: BoxShadowPart,
    pub inverted: bool,
}

/*
impl BoxShadowRasterOp {
    pub fn raster_rect(blur_radius: f32,
                       border_radius: f32,
                       part: BoxShadowPart,
                       box_rect: &Rect<f32>)
                       -> Rect<f32> {
        let outer_extent = blur_radius;
        let inner_extent = outer_extent.max(border_radius);
        let extent = outer_extent + inner_extent;
        match part {
            BoxShadowPart::Corner => {
                Rect::new(Point2D::new(box_rect.origin.x - outer_extent,
                                       box_rect.origin.y - outer_extent),
                          Size2D::new(extent, extent))
            }
            BoxShadowPart::Edge => {
                Rect::new(Point2D::new(box_rect.origin.x - outer_extent,
                                       box_rect.origin.y + box_rect.size.height / 2.0),
                          Size2D::new(extent, 1.0))
            }
        }
    }

    pub fn create_corner(blur_radius: f32,
                         border_radius: f32,
                         box_rect: &Rect<f32>,
                         inverted: bool,
                         device_pixel_ratio: f32)
                         -> Option<BoxShadowRasterOp> {
        if blur_radius > 0.0 || border_radius > 0.0 {
            let raster_rect = BoxShadowRasterOp::raster_rect(blur_radius,
                                                             border_radius,
                                                             BoxShadowPart::Corner,
                                                             box_rect);

            let blur_radius = DevicePixel::new(blur_radius, device_pixel_ratio);
            let border_radius = DevicePixel::new(border_radius, device_pixel_ratio);

            Some(BoxShadowRasterOp {
                blur_radius: blur_radius,
                border_radius: border_radius,
                local_raster_origin: (DevicePixel::new(box_rect.origin.x - raster_rect.origin.x, device_pixel_ratio),
                                      DevicePixel::new(box_rect.origin.y - raster_rect.origin.y, device_pixel_ratio)),
                box_rect_size: (DevicePixel::new(box_rect.size.width, device_pixel_ratio),
                                DevicePixel::new(box_rect.size.height, device_pixel_ratio)),
                raster_size: (DevicePixel::new(raster_rect.size.width, device_pixel_ratio),
                              DevicePixel::new(raster_rect.size.height, device_pixel_ratio)),
                part: BoxShadowPart::Corner,
                inverted: inverted,
            })
        } else {
            None
        }
    }

    pub fn create_edge(blur_radius: f32,
                       border_radius: f32,
                       box_rect: &Rect<f32>,
                       inverted: bool,
                       device_pixel_ratio: f32)
                       -> Option<BoxShadowRasterOp> {
        if blur_radius > 0.0 {
            let raster_rect = BoxShadowRasterOp::raster_rect(blur_radius,
                                                             border_radius,
                                                             BoxShadowPart::Edge,
                                                             box_rect);

            let blur_radius = DevicePixel::new(blur_radius, device_pixel_ratio);
            let border_radius = DevicePixel::new(border_radius, device_pixel_ratio);

            Some(BoxShadowRasterOp {
                blur_radius: blur_radius,
                border_radius: border_radius,
                local_raster_origin: (DevicePixel::new(box_rect.origin.x - raster_rect.origin.x, device_pixel_ratio),
                                      DevicePixel::new(box_rect.origin.y - raster_rect.origin.y, device_pixel_ratio)),
                box_rect_size: (DevicePixel::new(box_rect.size.width, device_pixel_ratio),
                                DevicePixel::new(box_rect.size.height, device_pixel_ratio)),
                raster_size: (DevicePixel::new(raster_rect.size.width, device_pixel_ratio),
                              DevicePixel::new(raster_rect.size.height, device_pixel_ratio)),
                part: BoxShadowPart::Edge,
                inverted: inverted,
            })
        } else {
            None
        }
    }
}*/

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum BoxShadowPart {
    _Corner,
    _Edge,
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct GlyphKey {
    pub font_key: FontKey,
    pub size: Au,
    pub blur_radius: Au,
    pub index: u32,
}

impl GlyphKey {
    pub fn new(font_key: FontKey, size: Au, blur_radius: Au, index: u32) -> GlyphKey {
        GlyphKey {
            font_key: font_key,
            size: size,
            blur_radius: blur_radius,
            index: index,
        }
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum RasterItem {
    _BoxShadow(BoxShadowRasterOp),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LowLevelFilterOp {
    Blur(Au, AxisDirection),
    Brightness(Au),
    Contrast(Au),
    Grayscale(Au),
    /// Fixed-point in `ANGLE_FLOAT_TO_FIXED` units.
    HueRotate(i32),
    Invert(Au),
    Opacity(Au),
    Saturate(Au),
    Sepia(Au),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompositionOp {
    MixBlend(MixBlendMode),
    Filter(LowLevelFilterOp),
}
