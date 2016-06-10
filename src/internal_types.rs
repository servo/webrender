/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use batch::{VertexBuffer, Batch, VertexBufferId, OffsetParams, TileParams};
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
use util::{self, RectVaryings};
use webrender_traits::{FontKey, Epoch, ColorF, PipelineId};
use webrender_traits::{ImageFormat, MixBlendMode, NativeFontHandle, DisplayItem, ScrollLayerId};

#[derive(Hash, Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct DevicePixel(i32);

impl DevicePixel {
    pub fn from_u32(value: u32) -> DevicePixel {
        DevicePixel(value as i32)
    }

    pub fn from_f32(value: f32) -> DevicePixel {
        debug_assert!(value.fract() == 0.0);
        DevicePixel(value as i32)
    }

    pub fn new(value: f32, device_pixel_ratio: f32) -> DevicePixel {
        DevicePixel((value * device_pixel_ratio).round() as i32)
    }

    // TODO(gw): Remove eventually...
    pub fn as_u16(&self) -> u16 {
        let DevicePixel(value) = *self;
        value as u16
    }

    // TODO(gw): Remove eventually...
    pub fn as_u32(&self) -> u32 {
        let DevicePixel(value) = *self;
        value as u32
    }

    // TODO(gw): Remove eventually...
    pub fn as_f32(&self) -> f32 {
        let DevicePixel(value) = *self;
        value as f32
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

pub static MAX_RECT: Rect<f32> = Rect {
    origin: Point2D {
        x: f32::MIN,
        y: f32::MIN,
    },
    size: Size2D {
        width: f32::INFINITY,
        height: f32::INFINITY,
    },
};

pub enum FontTemplate {
    Raw(Arc<Vec<u8>>),
    Native(NativeFontHandle),
}

pub type DrawListId = FreeListItemId;

#[derive(Debug, PartialEq, Eq)]
pub enum TextureSampler {
    Color,
    Mask,
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PackedVertexColorMode {
    Gradient,
    BorderCorner,
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

impl PackedVertexForQuad {
    pub fn new(position: &Rect<f32>,
               colors: &[ColorF; 4],
               uv: &RectUv<f32>,
               muv: &RectUv<DevicePixel>,
               color_mode: PackedVertexColorMode)
               -> PackedVertexForQuad {
        PackedVertexForQuad {
            x: position.origin.x,
            y: position.origin.y,
            width: position.size.width,
            height: position.size.height,
            color_tl: PackedColor::from_color(&colors[0]),
            color_tr: PackedColor::from_color(&colors[1]),
            color_br: PackedColor::from_color(&colors[2]),
            color_bl: PackedColor::from_color(&colors[3]),
            u_tl: uv.top_left.x,
            v_tl: uv.top_left.y,
            u_tr: uv.top_right.x,
            v_tr: uv.top_right.y,
            u_bl: uv.bottom_left.x,
            v_bl: uv.bottom_left.y,
            u_br: uv.bottom_right.x,
            v_br: uv.bottom_right.y,
            mu_tl: muv.top_left.x.as_u16(),
            mv_tl: muv.top_left.y.as_u16(),
            mu_tr: muv.top_right.x.as_u16(),
            mv_tr: muv.top_right.y.as_u16(),
            mu_bl: muv.bottom_left.x.as_u16(),
            mv_bl: muv.bottom_left.y.as_u16(),
            mu_br: muv.bottom_right.x.as_u16(),
            mv_br: muv.bottom_right.y.as_u16(),
            matrix_index: 0,
            clip_in_rect_index: 0,
            clip_out_rect_index: 0,
            tile_params_index: match color_mode {
                PackedVertexColorMode::Gradient => 0x00,
                PackedVertexColorMode::BorderCorner => 0x80,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PackedVertex {
    pub x: f32,
    pub y: f32,
    pub color: PackedColor,
    pub u: f32,
    pub v: f32,
    pub mu: u16,
    pub mv: u16,
    pub matrix_index: u8,
    pub clip_in_rect_index: u8,
    pub clip_out_rect_index: u8,
    pub tile_params_index: u8,
}

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
}

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
    /// All four corners, the tessellation index, and whether inverted, respectively.
    BorderRadius(DevicePixel, DevicePixel, DevicePixel, DevicePixel, Option<u32>, bool, BorderType),
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

pub enum BatchUpdateOp {
    Create(Vec<PackedVertexForQuad>),
    Destroy,
}

pub struct BatchUpdate {
    pub id: VertexBufferId,
    pub op: BatchUpdateOp,
}

pub struct BatchUpdateList {
    pub updates: Vec<BatchUpdate>,
}

impl BatchUpdateList {
    pub fn new() -> BatchUpdateList {
        BatchUpdateList {
            updates: Vec::new(),
        }
    }

    #[inline]
    pub fn push(&mut self, update: BatchUpdate) {
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

#[derive(Clone, Debug)]
pub struct OutputMask {
    pub rect: Rect<f32>,
    pub transform: Matrix4D<f32>,
}

impl OutputMask {
    pub fn new(rect: Rect<f32>,
               transform: Matrix4D<f32>) -> OutputMask {
        OutputMask {
            rect: rect,
            transform: transform,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MaskRegion {
    pub masks: Vec<OutputMask>,
    pub draw_calls: Vec<DrawCall>,
}

impl MaskRegion {
    pub fn new() -> MaskRegion {
        MaskRegion {
            masks: Vec::new(),
            draw_calls: Vec::new(),
        }
    }

    pub fn add_mask(&mut self,
                    rect: Rect<f32>,
                    transform: Matrix4D<f32>) {
        self.masks.push(OutputMask::new(rect, transform));
    }
}

#[derive(Clone, Debug)]
pub struct BatchInfo {
    pub matrix_palette: Vec<Matrix4D<f32>>,
    pub offset_palette: Vec<OffsetParams>,
    pub regions: Vec<MaskRegion>,
}

impl BatchInfo {
    pub fn new(matrix_palette: Vec<Matrix4D<f32>>,
               offset_palette: Vec<OffsetParams>) -> BatchInfo {
        BatchInfo {
            matrix_palette: matrix_palette,
            offset_palette: offset_palette,
            regions: Vec::new(),
        }
    }
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

#[derive(Debug, Clone)]
pub struct DrawCompositeBatchInfo {
    pub operation: CompositionOp,
    pub texture_id: TextureId,
    pub scroll_layer_id: ScrollLayerId,
    pub jobs: Vec<DrawCompositeBatchJob>,
}

#[derive(Clone, Debug)]
pub enum DrawCommand {
    Batch(BatchInfo),
    CompositeBatch(DrawCompositeBatchInfo),
    Clear(ClearInfo),
}

#[derive(Clone, Copy, Debug, Ord, PartialOrd, PartialEq, Eq, Hash)]
pub struct ChildLayerIndex(pub u32);

#[derive(Debug)]
pub struct DrawLayer {
    // This layer
    pub id: RenderTargetId,
    pub commands: Vec<DrawCommand>,
    pub texture_id: Option<TextureId>,
    pub origin: Point2D<f32>,
    pub size: Size2D<f32>,

    // Children
    pub child_layers: Vec<DrawLayer>,
}

impl DrawLayer {
    pub fn new(id: RenderTargetId,
               origin: Point2D<f32>,
               size: Size2D<f32>,
               texture_id: Option<TextureId>,
               commands: Vec<DrawCommand>,
               child_layers: Vec<DrawLayer>)
               -> DrawLayer {
        DrawLayer {
            id: id,
            origin: origin,
            size: size,
            texture_id: texture_id,
            commands: commands,
            child_layers: child_layers,
        }
    }
}

pub struct RendererFrame {
    pub pipeline_epoch_map: HashMap<PipelineId, Epoch, BuildHasherDefault<FnvHasher>>,
    pub layers_bouncing_back: HashSet<ScrollLayerId, BuildHasherDefault<FnvHasher>>,
    pub root_layer: DrawLayer,
}

impl RendererFrame {
    pub fn new(pipeline_epoch_map: HashMap<PipelineId, Epoch, BuildHasherDefault<FnvHasher>>,
               layers_bouncing_back: HashSet<ScrollLayerId, BuildHasherDefault<FnvHasher>>,
               root_layer: DrawLayer)
               -> RendererFrame {
        RendererFrame {
            pipeline_epoch_map: pipeline_epoch_map,
            layers_bouncing_back: layers_bouncing_back,
            root_layer: root_layer,
        }
    }
}

pub enum ResultMsg {
    UpdateTextureCache(TextureUpdateList),
    RefreshShader(PathBuf),
    NewFrame(RendererFrame, BatchUpdateList, BackendProfileCounters),
}

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
    pub perspective: Matrix4D<f32>,
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

#[derive(Debug)]
pub struct BatchList {
    pub batches: Vec<Batch>,
    pub draw_list_group_id: DrawListGroupId,
}

pub struct CompiledNode {
    // TODO(gw): These are mutually exclusive - unify into an enum?
    pub vertex_buffer: Option<VertexBuffer>,
    pub vertex_buffer_id: Option<VertexBufferId>,

    pub batch_list: Vec<BatchList>,
}

impl CompiledNode {
    pub fn new() -> CompiledNode {
        CompiledNode {
            batch_list: Vec::new(),
            vertex_buffer: None,
            vertex_buffer_id: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RectPolygon<Varyings> {
    pub pos: Rect<f32>,
    pub varyings: Varyings,
}

impl<Varyings> RectPolygon<Varyings> {
    pub fn is_well_formed_and_nonempty(&self) -> bool {
        util::rect_is_well_formed_and_nonempty(&self.pos)
    }
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

impl<T: Zero + Copy> RectUv<T> {
    pub fn zero() -> RectUv<T> {
        RectUv {
            top_left: Point2D::zero(),
            top_right: Point2D::zero(),
            bottom_left: Point2D::zero(),
            bottom_right: Point2D::zero(),
        }
    }

    pub fn from_uv_rect_rotation_angle(uv_rect: &RectUv<T>,
                                       rotation_angle: BasicRotationAngle,
                                       flip_90_degree_rotations: bool) -> RectUv<T> {
        match (rotation_angle, flip_90_degree_rotations) {
            (BasicRotationAngle::Upright, _) => {
                RectUv {
                    top_left: uv_rect.top_left,
                    top_right: uv_rect.top_right,
                    bottom_right: uv_rect.bottom_right,
                    bottom_left: uv_rect.bottom_left,
                }
            }
            (BasicRotationAngle::Clockwise90, true) => {
                RectUv {
                    top_right: uv_rect.top_left,
                    top_left: uv_rect.top_right,
                    bottom_left: uv_rect.bottom_right,
                    bottom_right: uv_rect.bottom_left,
                }
            }
            (BasicRotationAngle::Clockwise90, false) => {
                RectUv {
                    top_right: uv_rect.top_left,
                    bottom_right: uv_rect.top_right,
                    bottom_left: uv_rect.bottom_right,
                    top_left: uv_rect.bottom_left,
                }
            }
            (BasicRotationAngle::Clockwise180, _) => {
                RectUv {
                    bottom_right: uv_rect.top_left,
                    bottom_left: uv_rect.top_right,
                    top_left: uv_rect.bottom_right,
                    top_right: uv_rect.bottom_left,
                }
            }
            (BasicRotationAngle::Clockwise270, true) => {
                RectUv {
                    bottom_left: uv_rect.top_left,
                    bottom_right: uv_rect.top_right,
                    top_right: uv_rect.bottom_right,
                    top_left: uv_rect.bottom_left,
                }
            }
            (BasicRotationAngle::Clockwise270, false) => {
                RectUv {
                    bottom_left: uv_rect.top_left,
                    top_left: uv_rect.top_right,
                    top_right: uv_rect.bottom_right,
                    bottom_right: uv_rect.bottom_left,
                }
            }
        }
    }
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
pub struct BorderRadiusRasterOp {
    pub outer_radius_x: DevicePixel,
    pub outer_radius_y: DevicePixel,
    pub inner_radius_x: DevicePixel,
    pub inner_radius_y: DevicePixel,
    pub index: Option<u32>,
    pub image_format: ImageFormat,
    pub inverted: bool,
}

impl BorderRadiusRasterOp {
    pub fn create(outer_radius_x: DevicePixel,
                  outer_radius_y: DevicePixel,
                  inner_radius_x: DevicePixel,
                  inner_radius_y: DevicePixel,
                  inverted: bool,
                  index: Option<u32>,
                  image_format: ImageFormat)
                  -> Option<BorderRadiusRasterOp> {
        if outer_radius_x > DevicePixel::zero() || outer_radius_y > DevicePixel::zero() ||
           inner_radius_x > DevicePixel::zero() || inner_radius_y > DevicePixel::zero() {
            Some(BorderRadiusRasterOp {
                outer_radius_x: outer_radius_x,
                outer_radius_y: outer_radius_y,
                inner_radius_x: inner_radius_x,
                inner_radius_y: inner_radius_y,
                index: index,
                inverted: inverted,
                image_format: image_format,
            })
        } else {
            None
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

impl BoxShadowRasterOp {
    pub fn raster_rect(blur_radius: f32,
                       border_radius: f32,
                       part: BoxShadowPart,
                       box_rect: &Rect<f32>)
                       -> Rect<f32> {
        let outer_extent = 3.0 * blur_radius;
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
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum BoxShadowPart {
    Corner,
    Edge,
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
    BorderRadius(BorderRadiusRasterOp),
    BoxShadow(BoxShadowRasterOp),
}

#[derive(Clone, Copy, Debug)]
pub enum BasicRotationAngle {
    Upright,
    Clockwise90,
    Clockwise180,
    Clockwise270,
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

impl CompositionOp {
    pub fn target_rect(&self, unfiltered_target_rect: &Rect<f32>) -> Rect<f32> {
        match *self {
            CompositionOp::Filter(LowLevelFilterOp::Blur(amount, AxisDirection::Horizontal)) => {
                unfiltered_target_rect.inflate(amount.to_f32_px(), 0.0)
            }
            CompositionOp::Filter(LowLevelFilterOp::Blur(amount, AxisDirection::Vertical)) => {
                unfiltered_target_rect.inflate(0.0, amount.to_f32_px())
            }
            _ => *unfiltered_target_rect,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RectSide {
    Top,
    Right,
    Bottom,
    Left,
}

