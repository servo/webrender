use app_units::Au;
use batch::{VertexBuffer, Batch, VertexBufferId, OffsetParams, TileParams};
use device::{TextureId, TextureFilter};
use euclid::{Matrix4, Point2D, Rect, Size2D};
use fnv::FnvHasher;
use freelist::{FreeListItem, FreeListItemId};
use profiler::BackendProfileCounters;
use std::collections::HashMap;
use std::collections::hash_state::DefaultState;
use std::sync::Arc;
use texture_cache::TextureCacheItem;
use util::{self, RectVaryings};
use webrender_traits::{FontKey, Epoch, ColorF, PipelineId};
use webrender_traits::{ImageFormat, ScrollLayerId};
use webrender_traits::{MixBlendMode, NativeFontHandle, DisplayItem};

const UV_FLOAT_TO_FIXED: f32 = 65535.0;
const COLOR_FLOAT_TO_FIXED: f32 = 255.0;
pub const ANGLE_FLOAT_TO_FIXED: f32 = 65535.0;

pub const ORTHO_NEAR_PLANE: f32 = -1000000.0;
pub const ORTHO_FAR_PLANE: f32 = 1000000.0;

pub static MAX_RECT: Rect<f32> = Rect {
    origin: Point2D {
        x: -1000.0,
        y: -1000.0,
    },
    size: Size2D {
        width: 10000.0,
        height: 10000.0,
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
    Color,
    ColorTexCoord,
    MaskTexCoord,
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

    pub fn from_points(position: &Point2D<f32>,
                       color: &ColorF,
                       uv: &Point2D<f32>,
                       muv: &Point2D<f32>)
                       -> PackedVertex {
        PackedVertex::from_components(position.x, position.y, color, uv.x, uv.y, muv.x, muv.y)
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

#[derive(Debug)]
pub enum RenderTargetMode {
    None,
    RenderTarget,
}

#[derive(Debug)]
pub enum TextureUpdateDetails {
    Raw,
    Blit(Vec<u8>),
    Blur(Vec<u8>, Size2D<u32>, Au, TextureImage, TextureImage),
    /// All four corners, the tessellation index, and whether inverted, respectively.
    BorderRadius(Au, Au, Au, Au, Option<u32>, bool),
    /// Blur radius, border radius, box rect, raster origin, and whether inverted, respectively.
    BoxShadow(Au, Au, Rect<f32>, Point2D<f32>, bool),
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
    Create(Vec<PackedVertex>, Vec<u16>),
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
    pub batch: Arc<Batch>,
    pub vertex_buffer_id: VertexBufferId,
}

#[derive(Clone, Debug)]
pub struct BatchInfo {
    pub matrix_palette: Vec<Matrix4>,
    pub offset_palette: Vec<OffsetParams>,
    pub draw_calls: Vec<DrawCall>,
}

impl BatchInfo {
    pub fn new(matrix_palette: Vec<Matrix4>,
               offset_palette: Vec<OffsetParams>) -> BatchInfo {
        BatchInfo {
            matrix_palette: matrix_palette,
            offset_palette: offset_palette,
            draw_calls: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompositeBatchJob {
    pub rect: Rect<i32>,
    pub render_target_index: RenderTargetIndex,
}

#[derive(Debug, Clone)]
pub struct CompositeBatchInfo {
    pub operation: CompositionOp,
    pub jobs: Vec<CompositeBatchJob>,
}

#[derive(Clone, Debug)]
pub enum DrawCommand {
    Batch(BatchInfo),
    CompositeBatch(CompositeBatchInfo),
    Clear(ClearInfo),
}

#[derive(Clone, Copy, Debug, Ord, PartialOrd, PartialEq, Eq, Hash)]
pub struct RenderTargetIndex(pub u32);

#[derive(Debug)]
pub struct DrawTargetInfo {
    pub texture_id: TextureId,
    pub size: Size2D<u32>,
}

#[derive(Debug)]
pub struct DrawLayer {
    // This layer
    pub commands: Vec<DrawCommand>,
    pub layer_origin: Point2D<u32>,
    pub layer_size: Size2D<u32>,

    // Children
    pub child_target: Option<DrawTargetInfo>,
    pub child_layers: Vec<DrawLayer>,
}

impl DrawLayer {
    pub fn new(child_target: Option<DrawTargetInfo>,
               child_layers: Vec<DrawLayer>,
               commands: Vec<DrawCommand>,
               size: Size2D<u32>)
               -> DrawLayer {
        DrawLayer {
            child_target: child_target,
            commands: commands,
            child_layers: child_layers,
            layer_origin: Point2D::zero(),
            layer_size: size,
        }
    }
}

pub struct RendererFrame {
    pub pipeline_epoch_map: HashMap<PipelineId, Epoch, DefaultState<FnvHasher>>,
    pub root_layer: DrawLayer,
}

impl RendererFrame {
    pub fn new(pipeline_epoch_map: HashMap<PipelineId, Epoch, DefaultState<FnvHasher>>,
               root_layer: DrawLayer) -> RendererFrame {
        RendererFrame {
            pipeline_epoch_map: pipeline_epoch_map,
            root_layer: root_layer,
        }
    }
}

pub enum ResultMsg {
    UpdateTextureCache(TextureUpdateList),
    UpdateBatches(BatchUpdateList),
    NewFrame(RendererFrame, BackendProfileCounters),
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
    pub world_origin: Point2D<f32>,

    pub local_overflow: Rect<f32>,

    pub world_transform: Matrix4,
    pub world_perspective: Matrix4,

    pub scroll_layer_id: ScrollLayerId,
}

#[derive(Debug)]
pub struct DrawList {
    pub items: Vec<DisplayItem>,
    pub stacking_context_index: Option<StackingContextIndex>,
    // TODO(gw): Structure squat to remove this field.
    next_free_id: Option<FreeListItemId>,
}

impl DrawList {
    pub fn new(items: Vec<DisplayItem>) -> DrawList {
        DrawList {
            items: items,
            stacking_context_index: None,
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

#[derive(Debug, Copy, Clone)]
pub enum Primitive {
    Triangles,
    Rectangles,     // 4 vertices per rect
}

#[derive(Debug)]
pub struct BatchList {
    pub batches: Vec<Arc<Batch>>,
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
pub struct RectColorsUv {
    pub colors: RectColors,
    pub uv: RectUv,
}

#[derive(Clone, Copy, Debug)]
pub struct RectColors {
    pub top_left: ColorF,
    pub top_right: ColorF,
    pub bottom_right: ColorF,
    pub bottom_left: ColorF,
}

#[derive(Clone, Copy, Debug)]
pub struct RectUv {
    pub top_left: Point2D<f32>,
    pub top_right: Point2D<f32>,
    pub bottom_left: Point2D<f32>,
    pub bottom_right: Point2D<f32>,
}

impl RectUv {
    pub fn zero() -> RectUv {
        RectUv {
            top_left: Point2D::new(0.0, 0.0),
            top_right: Point2D::new(0.0, 0.0),
            bottom_left: Point2D::new(0.0, 0.0),
            bottom_right: Point2D::new(0.0, 0.0),
        }
    }

    pub fn from_uv_rect_rotation_angle(uv_rect: &RectUv,
                                       rotation_angle: BasicRotationAngle,
                                       flip_90_degree_rotations: bool) -> RectUv {
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

    pub fn from_image_and_rotation_angle(image: &TextureCacheItem,
                                         rotation_angle: BasicRotationAngle,
                                         flip_90_degree_rotations: bool)
                                         -> RectUv {
        RectUv::from_uv_rect_rotation_angle(&image.uv_rect,
                                            rotation_angle,
                                            flip_90_degree_rotations)
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
    pub outer_radius_x: Au,
    pub outer_radius_y: Au,
    pub inner_radius_x: Au,
    pub inner_radius_y: Au,
    pub index: Option<u32>,
    pub image_format: ImageFormat,
    pub inverted: bool,
}

impl BorderRadiusRasterOp {
    pub fn create(outer_radius: &Size2D<f32>,
                  inner_radius: &Size2D<f32>,
                  inverted: bool,
                  index: Option<u32>,
                  image_format: ImageFormat)
                  -> Option<BorderRadiusRasterOp> {
        if outer_radius.width > 0.0 || outer_radius.height > 0.0 {
            Some(BorderRadiusRasterOp {
                outer_radius_x: Au::from_f32_px(outer_radius.width),
                outer_radius_y: Au::from_f32_px(outer_radius.height),
                inner_radius_x: Au::from_f32_px(inner_radius.width),
                inner_radius_y: Au::from_f32_px(inner_radius.height),
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
    pub blur_radius: Au,
    pub border_radius: Au,
    // This is a tuple to work around the lack of `Eq` on `Rect`.
    pub box_rect_origin: (Au, Au),
    pub box_rect_size: (Au, Au),
    pub raster_origin: (Au, Au),
    pub raster_size: (Au, Au),
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
                         inverted: bool)
                         -> Option<BoxShadowRasterOp> {
        if blur_radius > 0.0 || border_radius > 0.0 {
            let raster_rect = BoxShadowRasterOp::raster_rect(blur_radius,
                                                             border_radius,
                                                             BoxShadowPart::Corner,
                                                             box_rect);
            Some(BoxShadowRasterOp {
                blur_radius: Au::from_f32_px(blur_radius),
                border_radius: Au::from_f32_px(border_radius),
                box_rect_origin: (Au::from_f32_px(box_rect.origin.x),
                                  Au::from_f32_px(box_rect.origin.y)),
                box_rect_size: (Au::from_f32_px(box_rect.size.width),
                                Au::from_f32_px(box_rect.size.height)),
                raster_origin: (Au::from_f32_px(raster_rect.origin.x),
                                Au::from_f32_px(raster_rect.origin.y)),
                raster_size: (Au::from_f32_px(raster_rect.size.width),
                              Au::from_f32_px(raster_rect.size.height)),
                part: BoxShadowPart::Corner,
                inverted: inverted,
            })
        } else {
            None
        }
    }

    pub fn create_edge(blur_radius: f32, border_radius: f32, box_rect: &Rect<f32>, inverted: bool)
                       -> Option<BoxShadowRasterOp> {
        let raster_rect = BoxShadowRasterOp::raster_rect(blur_radius,
                                                         border_radius,
                                                         BoxShadowPart::Edge,
                                                         box_rect);
        if blur_radius > 0.0 {
            Some(BoxShadowRasterOp {
                blur_radius: Au::from_f32_px(blur_radius),
                border_radius: Au::from_f32_px(border_radius),
                box_rect_origin: (Au::from_f32_px(box_rect.origin.x),
                                  Au::from_f32_px(box_rect.origin.y)),
                box_rect_size: (Au::from_f32_px(box_rect.size.width),
                                Au::from_f32_px(box_rect.size.height)),
                raster_origin: (Au::from_f32_px(raster_rect.origin.x),
                                Au::from_f32_px(raster_rect.origin.y)),
                raster_size: (Au::from_f32_px(raster_rect.size.width),
                              Au::from_f32_px(raster_rect.size.height)),
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
    pub fn target_rect(&self, unfiltered_target_rect: &Rect<i32>) -> Rect<i32> {
        match *self {
            CompositionOp::Filter(LowLevelFilterOp::Blur(amount, AxisDirection::Horizontal)) => {
                unfiltered_target_rect.inflate(amount.to_f32_px() as i32, 0)
            }
            CompositionOp::Filter(LowLevelFilterOp::Blur(amount, AxisDirection::Vertical)) => {
                unfiltered_target_rect.inflate(0, amount.to_f32_px() as i32)
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

