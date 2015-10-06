use device::{ProgramId, TextureId};
use euclid::{Point2D, Size2D, Rect};
use std::collections::HashMap;
use string_cache::Atom;
use types::{BlendMode};
use types::{Epoch, ColorF, PipelineId, ImageFormat, DisplayListID, DrawListID};
use types::{Au, ImageID, StackingContext, DisplayListBuilder, DisplayListMode};

const UV_FLOAT_TO_FIXED: f32 = 65535.0;
const COLOR_FLOAT_TO_FIXED: f32 = 255.0;

pub const ORTHO_NEAR_PLANE: f32 = -1000000.0;
pub const ORTHO_FAR_PLANE: f32 = 1000000.0;

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

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum RenderPass {
    Opaque,
    Alpha,
}

pub enum VertexAttribute {
    Position,
    Color,
    ColorTexCoord,
    MaskTexCoord,
}

pub enum VertexFormat {
    Default,
    //Debug
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
    pub z: f32,
    pub color: PackedColor,
    pub u: u16,
    pub v: u16,
    pub mu: u16,
    pub mv: u16,
}

impl PackedVertex {
    pub fn new(v: &WorkVertex, z: f32, offset: &Point2D<f32>) -> PackedVertex {
        debug_assert!(v.u >= -0.1 && v.u <= 1.1, format!("bad u {:?}", v.u));
        debug_assert!(v.v >= -0.1 && v.v <= 1.1, format!("bad v {:?}", v.v));
        debug_assert!(v.mu >= -0.1 && v.mu <= 1.1, format!("bad mu {:?}", v.mu));
        debug_assert!(v.mv >= -0.1 && v.mv <= 1.1, format!("bad mv {:?}", v.mv));

        // opengl spec f32 -> unorm16
        // round(clamp(c, 0, +1) * 65535.0)

        PackedVertex {
            x: (v.x + offset.x).round(),
            y: (v.y + offset.y).round(),
            z: z,
            color: PackedColor::from_components(v.r, v.g, v.b, v.a),
            u: (v.u * UV_FLOAT_TO_FIXED).round() as u16,
            v: (v.v * UV_FLOAT_TO_FIXED).round() as u16,
            mu: (v.mu * UV_FLOAT_TO_FIXED).round() as u16,
            mv: (v.mv * UV_FLOAT_TO_FIXED).round() as u16,
        }
    }

    pub fn from_components(x: f32,
                           y: f32,
                           z: f32,
                           color: &ColorF,
                           u: f32,
                           v: f32,
                           mu: f32,
                           mv: f32) -> PackedVertex {
        PackedVertex {
            x: x,
            y: y,
            z: z,
            color: PackedColor::from_color(color),
            u: (u * UV_FLOAT_TO_FIXED).round() as u16,
            v: (v * UV_FLOAT_TO_FIXED).round() as u16,
            mu: (mu * UV_FLOAT_TO_FIXED).round() as u16,
            mv: (mv * UV_FLOAT_TO_FIXED).round() as u16,
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
    BorderRadius(Au, Au, Au, Au),
}

pub enum TextureUpdateOp {
    Create(u32, u32, ImageFormat, RenderTargetMode, Option<Vec<u8>>),
    Update(u32, u32, u32, u32, TextureUpdateDetails),
    //FreeRenderTarget(TextureId),
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

pub struct RenderBatch {
    pub program_id: ProgramId,
    pub color_texture_id: TextureId,
    pub mask_texture_id: TextureId,
    pub vertices: Vec<PackedVertex>,
    pub indices: Vec<u16>,
}

pub struct CompositeInfo {
    pub blend_mode: BlendMode,
    pub rect: Rect<u32>,
    pub color_texture_id: TextureId,
    pub z: f32,
}

pub enum DrawCommand {
    Batch(Vec<RenderBatch>, Vec<RenderBatch>),
    Composite(CompositeInfo)
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
