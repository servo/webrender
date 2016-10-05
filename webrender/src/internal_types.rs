/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use device::{TextureId, TextureFilter};
use euclid::{Point2D, Rect, Size2D};
use fnv::FnvHasher;
use freelist::{FreeListItem, FreeListItemId};
use num_traits::Zero;
use offscreen_gl_context::{NativeGLContext, NativeGLContextHandle};
use offscreen_gl_context::{GLContext, NativeGLContextMethods, GLContextDispatcher};
use offscreen_gl_context::{OSMesaContext, OSMesaContextHandle};
use offscreen_gl_context::{ColorAttachmentType, GLContextAttributes, GLLimits};
use profiler::BackendProfileCounters;
use std::collections::{HashMap, HashSet};
use std::f32;
use std::hash::BuildHasherDefault;
use std::i32;
use std::ops::{Add, Sub};
use std::path::PathBuf;
use std::sync::Arc;
use texture_cache::BorderType;
use tiling;
use webrender_traits::{Epoch, ColorF, PipelineId};
use webrender_traits::{ImageFormat, MixBlendMode, NativeFontHandle, DisplayItem};
use webrender_traits::{ScrollLayerId, WebGLCommand};

pub enum GLContextHandleWrapper {
    Native(NativeGLContextHandle),
    OSMesa(OSMesaContextHandle),
}

impl GLContextHandleWrapper {
    pub fn current_native_handle() -> Option<GLContextHandleWrapper> {
        NativeGLContext::current_handle().map(GLContextHandleWrapper::Native)
    }

    pub fn current_osmesa_handle() -> Option<GLContextHandleWrapper> {
        OSMesaContext::current_handle().map(GLContextHandleWrapper::OSMesa)
    }

    pub fn new_context(&self,
                       size: Size2D<i32>,
                       attributes: GLContextAttributes,
                       dispatcher: Option<Box<GLContextDispatcher>>) -> Result<GLContextWrapper, &'static str> {
        match *self {
            GLContextHandleWrapper::Native(ref handle) => {
                let ctx = GLContext::<NativeGLContext>::new_shared_with_dispatcher(size,
                                                                                   attributes,
                                                                                   ColorAttachmentType::Texture,
                                                                                   Some(handle),
                                                                                   dispatcher);
                ctx.map(GLContextWrapper::Native)
            }
            GLContextHandleWrapper::OSMesa(ref handle) => {
                let ctx = GLContext::<OSMesaContext>::new_shared_with_dispatcher(size,
                                                                                 attributes,
                                                                                 ColorAttachmentType::Texture,
                                                                                 Some(handle),
                                                                                 dispatcher);
                ctx.map(GLContextWrapper::OSMesa)
            }
        }
    }
}

pub enum GLContextWrapper {
    Native(GLContext<NativeGLContext>),
    OSMesa(GLContext<OSMesaContext>),
}

impl GLContextWrapper {
    pub fn make_current(&self) {
        match *self {
            GLContextWrapper::Native(ref ctx) => {
                ctx.make_current().unwrap();
            }
            GLContextWrapper::OSMesa(ref ctx) => {
                ctx.make_current().unwrap();
            }
        }
    }

    pub fn unbind(&self) {
        match *self {
            GLContextWrapper::Native(ref ctx) => {
                ctx.unbind().unwrap();
            }
            GLContextWrapper::OSMesa(ref ctx) => {
                ctx.unbind().unwrap();
            }
        }
    }

    pub fn apply_command(&self, cmd: WebGLCommand) {
        match *self {
            GLContextWrapper::Native(ref ctx) => {
                cmd.apply(ctx);
            }
            GLContextWrapper::OSMesa(ref ctx) => {
                cmd.apply(ctx);
            }
        }
    }

    pub fn get_info(&self) -> (Size2D<i32>, u32, GLLimits) {
        match *self {
            GLContextWrapper::Native(ref ctx) => {
                let (real_size, texture_id) = {
                    let draw_buffer = ctx.borrow_draw_buffer().unwrap();
                    (draw_buffer.size(), draw_buffer.get_bound_texture_id().unwrap())
                };

                let limits = ctx.borrow_limits().clone();

                (real_size, texture_id, limits)
            }
            GLContextWrapper::OSMesa(ref ctx) => {
                let (real_size, texture_id) = {
                    let draw_buffer = ctx.borrow_draw_buffer().unwrap();
                    (draw_buffer.size(), draw_buffer.get_bound_texture_id().unwrap())
                };

                let limits = ctx.borrow_limits().clone();

                (real_size, texture_id, limits)
            }
        }
    }
}

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
    Layers,
    RenderTasks,
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
