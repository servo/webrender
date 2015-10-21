use app_units::Au;
use batch::RenderBatch;
use device::{ProgramId, TextureId};
use euclid::{Matrix4, Point2D, Rect, Size2D};
use fnv::FnvHasher;
use std::collections::HashMap;
use std::collections::hash_state::DefaultState;
use std::sync::mpsc::Sender;
use string_cache::Atom;
use texture_cache::TextureCacheItem;
use types::{Epoch, ColorF, PipelineId, ImageFormat, DisplayListID, DrawListID};
use types::{ImageID, StackingContext, DisplayListBuilder, DisplayListMode, CompositionOp};
use types::{new_resource_id};
use util;

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
    BorderRadii,
    BorderPosition,
    BlurRadius,
    DestTextureSize,
    SourceTextureSize,
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
            r: (color.r * COLOR_FLOAT_TO_FIXED) as u8,
            g: (color.g * COLOR_FLOAT_TO_FIXED) as u8,
            b: (color.b * COLOR_FLOAT_TO_FIXED) as u8,
            a: (color.a * COLOR_FLOAT_TO_FIXED) as u8,
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

impl WorkVertex {
    #[inline]
    pub fn new(x: f32, y: f32, color: &ColorF, u: f32, v: f32) -> WorkVertex {
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
        }
    }

    pub fn position(&self) -> Point2D<f32> {
        Point2D::new(self.x, self.y)
    }

    pub fn uv(&self) -> Point2D<f32> {
        Point2D::new(self.u, self.v)
    }

    pub fn color(&self) -> ColorF {
        ColorF {
            r: self.r,
            g: self.g,
            b: self.b,
            a: self.a,
        }
    }
}

#[derive(Debug, Clone)]
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
            u: (u * UV_FLOAT_TO_FIXED) as u16,
            v: (v * UV_FLOAT_TO_FIXED) as u16,
            mu: (mu * UV_FLOAT_TO_FIXED) as u16,
            mv: (mv * UV_FLOAT_TO_FIXED) as u16,
            matrix_index: 0,
            unused0: 0,
            unused1: 0,
            unused2: 0,
        }
    }

    pub fn from_points(position: &Point2D<f32>,
                       color: &ColorF,
                       uv: &Point2D<f32>,
                       muv: &Point2D<f32>)
                       -> PackedVertex {
        PackedVertex::from_components(position.x, position.y, color, uv.x, uv.y, muv.x, muv.y)
    }
}

#[derive(Debug)]
pub enum RenderTargetMode {
    None,
    RenderTarget,
}

#[derive(Debug)]
pub enum TextureUpdateDetails {
    Blit(Vec<u8>),
    Blur(Vec<u8>, Size2D<u32>, Au, TextureImage, TextureImage),
    /// All four corners and whether inverted, respectively.
    BorderRadius(Au, Au, Au, Au, bool),
    /// Blur radius border radius, and whether inverted, respectively.
    BoxShadowCorner(Au, Au, bool),
}

#[derive(Clone, Copy, Debug)]
pub struct TextureImage {
    pub texture_id: TextureId,
    pub texel_uv: Rect<f32>,
    pub pixel_uv: Point2D<u32>,
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

// TODO(gw): Use bitflags crate for ClearInfo...
// TODO(gw): Expand clear info to handle color, depth etc as needed.

#[derive(Clone)]
pub struct ClearInfo {
    pub clear_color: bool,
    pub clear_z: bool,
    pub clear_stencil: bool,
}

#[derive(Clone)]
pub struct CompositeInfo {
    pub operation: CompositionOp,
    pub rect: Rect<u32>,
    pub color_texture_id: TextureId,
}

#[derive(Clone)]
pub enum DrawCommandInfo {
    Batch(BatchId),
    Composite(CompositeInfo),
    Clear(ClearInfo),
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
    pub pipeline_epoch_map: HashMap<PipelineId, Epoch, DefaultState<FnvHasher>>,
    pub layers: Vec<DrawLayer>,
}

impl Frame {
    pub fn new(pipeline_epoch_map: HashMap<PipelineId, Epoch, DefaultState<FnvHasher>>) -> Frame {
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
    SetRootPipeline(PipelineId),
    Scroll(Point2D<f32>),
    TranslatePointToLayerSpace(Point2D<f32>, Sender<Point2D<f32>>),
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

#[derive(Debug, Clone, Copy)]
pub struct ClipRectToRegionMaskResult {
    /// The bounding box of the mask, in texture coordinates.
    pub muv_rect: Rect<f32>,

    /// The bounding rect onto which the mask will be applied, in framebuffer coordinates.
    pub position_rect: Rect<f32>,

    /// The border radius in question, for lookup in the texture cache.
    pub border_radius: f32,
}

impl ClipRectToRegionMaskResult {
    pub fn new(muv_rect: &Rect<f32>, position_rect: &Rect<f32>, border_radius: f32)
               -> ClipRectToRegionMaskResult {
        ClipRectToRegionMaskResult {
            muv_rect: *muv_rect,
            position_rect: *position_rect,
            border_radius: border_radius,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ClipRectToRegionResult<P> {
    pub rect_result: P,
    pub mask_result: Option<ClipRectToRegionMaskResult>,
}

impl<P> ClipRectToRegionResult<P> {
    pub fn new(rect_result: P, mask_result: Option<ClipRectToRegionMaskResult>)
               -> ClipRectToRegionResult<P> {
        ClipRectToRegionResult {
            rect_result: rect_result,
            mask_result: mask_result,
        }
    }

    pub fn muv_for_position(&self, position: &Point2D<f32>, mask: &TextureCacheItem)
                            -> Point2D<f32> {
        let mask_uv_size = Size2D::new(mask.u1 - mask.u0, mask.v1 - mask.v0);
        let mask_result = match self.mask_result {
            None => return Point2D::new(0.0, 0.0),
            Some(ref mask_result) => mask_result,
        };

        let muv_rect =
            Rect::new(Point2D::new(mask.u0 + mask_result.muv_rect.origin.x * mask_uv_size.width,
                                   mask.v0 + mask_result.muv_rect.origin.y * mask_uv_size.height),
                      Size2D::new(mask_result.muv_rect.size.width * mask_uv_size.width,
                                  mask_result.muv_rect.size.height * mask_uv_size.height));
        let position_rect = &mask_result.position_rect;

        Point2D::new(util::lerp(muv_rect.origin.x,
                                muv_rect.max_x(),
                                (position.x - position_rect.origin.x) / position_rect.size.width),
                     util::lerp(muv_rect.origin.y,
                                muv_rect.max_y(),
                                (position.y - position_rect.origin.y) / position_rect.size.height))
    }

    pub fn make_packed_vertex(&self,
                              position: &Point2D<f32>,
                              uv: &Point2D<f32>,
                              color: &ColorF,
                              mask: &TextureCacheItem)
                              -> PackedVertex {
        PackedVertex::from_points(position, color, uv, &self.muv_for_position(position, mask))
    }
}

impl ClipRectToRegionResult<RectPosUv> {
    // TODO(pcwalton): Clip colors too!
    pub fn make_packed_vertices_for_rect(&self, colors: &[ColorF; 4], mask: &TextureCacheItem)
                                         -> [PackedVertex; 4] {
        [
            self.make_packed_vertex(&self.rect_result.pos.origin,
                                    &self.rect_result.uv.origin,
                                    &colors[0],
                                    mask),
            self.make_packed_vertex(&self.rect_result.pos.top_right(),
                                    &self.rect_result.uv.top_right(),
                                    &colors[1],
                                    mask),
            self.make_packed_vertex(&self.rect_result.pos.bottom_left(),
                                    &self.rect_result.uv.bottom_left(),
                                    &colors[3],
                                    mask),
            self.make_packed_vertex(&self.rect_result.pos.bottom_right(),
                                    &self.rect_result.uv.bottom_right(),
                                    &colors[2],
                                    mask),
        ]
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BorderEdgeDirection {
    Horizontal,
    Vertical,
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct GlyphKey {
    pub font_id: Atom,
    pub size: Au,
    pub blur_radius: Au,
    pub index: u32,
}

impl GlyphKey {
    pub fn new(font_id: Atom, size: Au, blur_radius: Au, index: u32) -> GlyphKey {
        GlyphKey {
            font_id: font_id,
            size: size,
            blur_radius: blur_radius,
            index: index,
        }
    }
}

#[derive(Clone, Copy, Debug, Ord, PartialOrd, PartialEq, Eq, Hash)]
pub struct DrawListIndex(pub u32);

#[derive(Clone, Copy, Debug, Ord, PartialOrd, PartialEq, Eq)]
pub struct DrawListItemIndex(pub u32);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DisplayItemKey {
    pub draw_list_index: DrawListIndex,
    pub item_index: DrawListItemIndex,
}

impl DisplayItemKey {
    pub fn new(draw_list_index: usize, item_index: usize) -> DisplayItemKey {
        DisplayItemKey {
            draw_list_index: DrawListIndex(draw_list_index as u32),
            item_index: DrawListItemIndex(item_index as u32),
        }
    }
}

#[derive(Debug)]
pub enum Primitive {
    Triangles,
    Rectangles,     // 4 vertices per rect
    TriangleFan,    // simple triangle fan (typically from clipper)
    Glyphs,         // font glyphs (some platforms may specialize shader)
}

pub struct CompiledNode {
    pub batches: Vec<RenderBatch>,
    pub commands: Vec<DrawCommand>,
    pub batch_id_list: Vec<BatchId>,
    pub matrix_maps: HashMap<BatchId,
                             HashMap<DrawListIndex, u8, DefaultState<FnvHasher>>,
                             DefaultState<FnvHasher>>,
}

impl CompiledNode {
    pub fn new() -> CompiledNode {
        CompiledNode {
            batches: Vec::new(),
            commands: Vec::new(),
            batch_id_list: Vec::new(),
            matrix_maps: HashMap::with_hash_state(Default::default()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RectPosUv {
    pub pos: Rect<f32>,
    pub uv: Rect<f32>,
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
    // TODO(pcwalton): For alignment purposes of floats. Does this actually help?
    pub unused: f32,
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
            unused: 0.0,
        }
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct BorderRadiusRasterOp {
    pub outer_radius_x: Au,
    pub outer_radius_y: Au,
    pub inner_radius_x: Au,
    pub inner_radius_y: Au,
    pub inverted: bool,
    pub image_format: ImageFormat,
}

impl BorderRadiusRasterOp {
    pub fn create(outer_radius: &Size2D<f32>,
                  inner_radius: &Size2D<f32>,
                  inverted: bool,
                  image_format: ImageFormat)
                  -> Option<BorderRadiusRasterOp> {
        if outer_radius.width > 0.0 || outer_radius.height > 0.0 {
            Some(BorderRadiusRasterOp {
                outer_radius_x: Au::from_f32_px(outer_radius.width),
                outer_radius_y: Au::from_f32_px(outer_radius.height),
                inner_radius_x: Au::from_f32_px(inner_radius.width),
                inner_radius_y: Au::from_f32_px(inner_radius.height),
                inverted: inverted,
                image_format: image_format,
            })
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct BoxShadowCornerRasterOp {
    pub blur_radius: Au,
    pub border_radius: Au,
    pub inverted: bool,
}

impl BoxShadowCornerRasterOp {
    pub fn create(blur_radius: f32, border_radius: f32, inverted: bool)
                  -> Option<BoxShadowCornerRasterOp> {
        if blur_radius > 0.0 || border_radius > 0.0 {
            Some(BoxShadowCornerRasterOp {
                blur_radius: Au::from_f32_px(blur_radius),
                border_radius: Au::from_f32_px(border_radius),
                inverted: inverted,
            })
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum RasterItem {
    BorderRadius(BorderRadiusRasterOp),
    BoxShadowCorner(BoxShadowCornerRasterOp),
}

