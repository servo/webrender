use app_units::Au;
use device::{ProgramId, TextureId};
use euclid::{Matrix4, Point2D, Rect, Size2D};
use render_backend::DisplayItemKey;
use std::collections::HashMap;
use string_cache::Atom;
use texture_cache::TextureCacheItem;
use types::{MixBlendMode, new_resource_id};
use types::{Epoch, ColorF, PipelineId, ImageFormat, DisplayListID, DrawListID};
use types::{ImageID, StackingContext, DisplayListBuilder, DisplayListMode};

const UV_FLOAT_TO_FIXED: f32 = 65535.0;
const COLOR_FLOAT_TO_FIXED: f32 = 255.0;

pub const ORTHO_NEAR_PLANE: f32 = -1000000.0;
pub const ORTHO_FAR_PLANE: f32 = 1000000.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BatchId(pub usize);

impl BatchId {
    pub fn new() -> BatchId {
        BatchId(new_resource_id())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum TextureSampler {
    Color,
    Mask,
}

pub struct ImageResource {
    pub bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
}

pub enum VertexAttribute {
    Position,
    Color,
    ColorTexCoord,
    MaskTexCoord,
    MatrixIndex,
}

#[derive(Debug, Clone)]
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
            r: (color.r * COLOR_FLOAT_TO_FIXED).round() as u8,
            g: (color.g * COLOR_FLOAT_TO_FIXED).round() as u8,
            b: (color.b * COLOR_FLOAT_TO_FIXED).round() as u8,
            a: (color.a * COLOR_FLOAT_TO_FIXED).round() as u8,
        }
    }

    pub fn from_components(r: f32, g: f32, b: f32, a: f32) -> PackedColor {
        PackedColor {
            r: (r * COLOR_FLOAT_TO_FIXED).round() as u8,
            g: (g * COLOR_FLOAT_TO_FIXED).round() as u8,
            b: (b * COLOR_FLOAT_TO_FIXED).round() as u8,
            a: (a * COLOR_FLOAT_TO_FIXED).round() as u8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorkVertex {
    pub x: f32,
    pub y: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
    pub u: f32,
    pub v: f32,
    pub mu: f32,
    pub mv: f32,
}

impl WorkVertex {
    #[inline]
    pub fn new(x: f32, y: f32, color: &ColorF, u: f32, v: f32, mu: f32, mv: f32) -> WorkVertex {
        debug_assert!(u.is_finite());
        debug_assert!(v.is_finite());

        WorkVertex {
            x: x,
            y: y,
            r: color.r,
            g: color.g,
            b: color.b,
            a: color.a,
            u: u,
            v: v,
            mu: mu,
            mv: mv,
        }
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct PackedVertex {
    pub x: f32,
    pub y: f32,
    pub color: PackedColor,
    pub u: u16,
    pub v: u16,
    pub mu: u16,
    pub mv: u16,
    pub matrix_index: u8,
    pub unused0: u8,        // TODO(gw): For alignment purposes of floats. Profile which GPUs this affects - might be worth having a separate stream.
    pub unused1: u8,
    pub unused2: u8,
}

impl PackedVertex {
    pub fn new(v: &WorkVertex, device_pixel_ratio: f32, matrix_index: u8)
               -> PackedVertex {
        debug_assert!(v.u >= -0.1 && v.u <= 1.1, format!("bad u {:?}", v.u));
        debug_assert!(v.v >= -0.1 && v.v <= 1.1, format!("bad v {:?}", v.v));
        debug_assert!(v.mu >= -0.1 && v.mu <= 1.1, format!("bad mu {:?}", v.mu));
        debug_assert!(v.mv >= -0.1 && v.mv <= 1.1, format!("bad mv {:?}", v.mv));

        // opengl spec f32 -> unorm16
        // round(clamp(c, 0, +1) * 65535.0)

        PackedVertex {
            x: (v.x * device_pixel_ratio).round() / device_pixel_ratio,
            y: (v.y * device_pixel_ratio).round() / device_pixel_ratio,
            color: PackedColor::from_components(v.r, v.g, v.b, v.a),
            u: (v.u * UV_FLOAT_TO_FIXED).round() as u16,
            v: (v.v * UV_FLOAT_TO_FIXED).round() as u16,
            mu: (v.mu * UV_FLOAT_TO_FIXED).round() as u16,
            mv: (v.mv * UV_FLOAT_TO_FIXED).round() as u16,
            matrix_index: matrix_index,
            unused0: 0,
            unused1: 0,
            unused2: 0,
        }
    }

    pub fn from_components(x: f32,
                           y: f32,
                           color: &ColorF,
                           u: f32,
                           v: f32,
                           mu: f32,
                           mv: f32) -> PackedVertex {
        PackedVertex {
            x: x,
            y: y,
            color: PackedColor::from_color(color),
            u: (u * UV_FLOAT_TO_FIXED).round() as u16,
            v: (v * UV_FLOAT_TO_FIXED).round() as u16,
            mu: (mu * UV_FLOAT_TO_FIXED).round() as u16,
            mv: (mv * UV_FLOAT_TO_FIXED).round() as u16,
            matrix_index: 0,
            unused0: 0,
            unused1: 0,
            unused2: 0,
        }
    }
}

/*
#[derive(Debug)]
#[repr(C)]
pub struct DebugVertex {
    x: f32,
    y: f32,
    color: PackedColor,
}*/

/*
impl DebugVertex {
    fn new(x: f32, y: f32, color: &ColorF) -> DebugVertex {
        DebugVertex {
            x: x,
            y: y,
            color: PackedColor::from_color(color),
        }
    }
}*/

#[derive(Debug)]
pub enum RenderTargetMode {
    None,
    RenderTarget,
}

#[derive(Debug)]
pub enum TextureUpdateDetails {
    Blit(Vec<u8>),
    Blur(Vec<u8>, Size2D<u32>, Au, TextureId, TextureId),
    BorderRadius(Au, Au, Au, Au),
    /// Blur radius and border radius, respectively.
    BoxShadowCorner(Au, Au),
}

pub enum TextureUpdateOp {
    Create(u32, u32, ImageFormat, RenderTargetMode, Option<Vec<u8>>),
    Update(u32, u32, u32, u32, TextureUpdateDetails),
    DeinitRenderTarget(TextureId),
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
    Create(Vec<PackedVertex>,
           Vec<u16>,
           ProgramId,
           TextureId,
           TextureId),
    UpdateUniforms(Vec<Matrix4>),
    Destroy,
}

pub struct BatchUpdate {
    pub id: BatchId,
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

#[derive(Clone)]
pub struct CompositeInfo {
    pub blend_mode: MixBlendMode,
    pub rect: Rect<u32>,
    pub color_texture_id: TextureId,
}

#[derive(Clone)]
pub enum DrawCommandInfo {
    Batch(BatchId),
    Composite(CompositeInfo),
}

#[derive(Clone, Copy, Debug, Ord, PartialOrd, PartialEq, Eq, Hash)]
pub struct RenderTargetIndex(pub u32);

#[derive(Clone)]
pub struct DrawCommand {
    pub render_target: RenderTargetIndex,
    pub sort_key: DisplayItemKey,
    pub info: DrawCommandInfo,
}

pub struct DrawLayer {
    pub texture_id: Option<TextureId>,
    pub size: Size2D<u32>,
    pub commands: Vec<DrawCommand>,
}

impl DrawLayer {
    pub fn new(texture_id: Option<TextureId>,
               size: Size2D<u32>,
               commands: Vec<DrawCommand>) -> DrawLayer {
        DrawLayer {
            texture_id: texture_id,
            size: size,
            commands: commands,
        }
    }
}

pub struct Frame {
    pub pipeline_epoch_map: HashMap<PipelineId, Epoch>,
    pub layers: Vec<DrawLayer>,
}

impl Frame {
    pub fn new(pipeline_epoch_map: HashMap<PipelineId, Epoch>) -> Frame {
        Frame {
            pipeline_epoch_map: pipeline_epoch_map,
            layers: Vec::new(),
        }
    }

    pub fn add_layer(&mut self, layer: DrawLayer) {
        self.layers.push(layer);
    }
}

pub enum ApiMsg {
    AddFont(Atom, Vec<u8>),
    AddImage(ImageID, u32, u32, ImageFormat, Vec<u8>),
    AddDisplayList(DisplayListID, PipelineId, Epoch, DisplayListBuilder),
    SetRootStackingContext(StackingContext, ColorF, Epoch, PipelineId),
    Scroll(Point2D<f32>),
}

pub enum ResultMsg {
    UpdateTextureCache(TextureUpdateList),
    UpdateBatches(BatchUpdateList),
    NewFrame(Frame),
}

pub struct DisplayList {
    pub mode: DisplayListMode,

    pub pipeline_id: PipelineId,
    pub epoch: Epoch,

    pub background_and_borders_id: Option<DrawListID>,
    pub block_backgrounds_and_borders_id: Option<DrawListID>,
    pub floats_id: Option<DrawListID>,
    pub content_id: Option<DrawListID>,
    pub positioned_content_id: Option<DrawListID>,
    pub outlines_id: Option<DrawListID>,
}

#[derive(Debug)]
pub struct ClipRectResult {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub u0: f32,
    pub v0: f32,
    pub u1: f32,
    pub v1: f32,
}

impl ClipRectResult {
    pub fn from_rects(rect: &Rect<f32>, uv: &Rect<f32>) -> ClipRectResult {
        ClipRectResult {
            x0: rect.origin.x,
            y0: rect.origin.y,
            x1: rect.max_x(),
            y1: rect.max_y(),
            u0: uv.origin.x,
            v0: uv.origin.y,
            u1: uv.max_x(),
            v1: uv.max_y(),
        }
    }

    pub fn rect(&self) -> Rect<f32> {
        Rect::new(Point2D::new(self.x0, self.y0),
                  Size2D::new(self.x1 - self.x0, self.y1 - self.y0))
    }

    pub fn uv_rect(&self) -> Rect<f32> {
        Rect::new(Point2D::new(self.u0, self.v0),
                  Size2D::new(self.u1 - self.u0, self.v1 - self.v0))
    }
}

#[derive(Debug)]
pub struct ClipRectToRegionMaskResult {
    pub mu0: f32,
    pub mv0: f32,
    pub mu1: f32,
    pub mv1: f32,

    /// For looking up in the texture cache.
    pub border_radius: f32,
}

impl ClipRectToRegionMaskResult {
    pub fn new(rect: &Rect<f32>, border_radius: f32) -> ClipRectToRegionMaskResult {
        ClipRectToRegionMaskResult {
            mu0: rect.origin.x,
            mv0: rect.origin.y,
            mu1: rect.max_x(),
            mv1: rect.max_y(),
            border_radius: border_radius,
        }
    }
}

#[derive(Debug)]
pub struct ClipRectToRegionResult {
    pub rect_result: ClipRectResult,
    pub mask_result: Option<ClipRectToRegionMaskResult>,
}

impl ClipRectToRegionResult {
    pub fn new(rect_result: ClipRectResult, mask_result: Option<ClipRectToRegionMaskResult>)
               -> ClipRectToRegionResult {
        ClipRectToRegionResult {
            rect_result: rect_result,
            mask_result: mask_result,
        }
    }

    pub fn muv(&self, mask: &TextureCacheItem) -> Rect<f32> {
        match self.mask_result {
            None => {
                Rect::new(Point2D::new(mask.u0, mask.v0),
                          Size2D::new(mask.u1 - mask.u0, mask.v1 - mask.v0))
            }
            Some(ref mask_result) => {
                let mask_uv_size = Size2D::new(mask.u1 - mask.u0, mask.v1 - mask.v0);
                Rect::new(Point2D::new(mask.u0 + mask_result.mu0 * mask_uv_size.width,
                                       mask.v0 + mask_result.mv0 * mask_uv_size.height),
                          Size2D::new((mask_result.mu1 - mask_result.mu0) * mask_uv_size.width,
                                      (mask_result.mv1 - mask_result.mv0) * mask_uv_size.height))
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BorderEdgeDirection {
    Horizontal,
    Vertical,
}

