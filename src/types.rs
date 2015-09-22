use euclid::{Point2D, Rect, Size2D, Matrix4};
use std::i32;
use std::fmt;
use std::ops::{Add, Sub, Neg, Mul, Div, Rem};
use std::sync::atomic::{AtomicUsize, Ordering, ATOMIC_USIZE_INIT};
use string_cache::Atom;

// An Au is an "App Unit" and represents 1/60th of a CSS pixel.  It was
// originally proposed in 2002 as a standard unit of measure in Gecko.
// See https://bugzilla.mozilla.org/show_bug.cgi?id=177805 for more info.
//
#[derive(Clone, Copy, Hash, PartialEq, PartialOrd, Eq, Ord)]
pub struct Au(pub i32);

impl Default for Au {
    #[inline]
    fn default() -> Au {
        Au(0)
    }
}

pub const MIN_AU: Au = Au(i32::MIN);
pub const MAX_AU: Au = Au(i32::MAX);

impl fmt::Debug for Au {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}px", self.to_f64_px())
    }
}

impl Add for Au {
    type Output = Au;

    #[inline]
    fn add(self, other: Au) -> Au {
        Au(self.0.wrapping_add(other.0))
    }
}

impl Sub for Au {
    type Output = Au;

    #[inline]
    fn sub(self, other: Au) -> Au {
        Au(self.0.wrapping_sub(other.0))
    }

}

impl Mul<i32> for Au {
    type Output = Au;

    #[inline]
    fn mul(self, other: i32) -> Au {
        Au(self.0.wrapping_mul(other))
    }
}

impl Div<i32> for Au {
    type Output = Au;

    #[inline]
    fn div(self, other: i32) -> Au {
        Au(self.0 / other)
    }
}

impl Rem<i32> for Au {
    type Output = Au;

    #[inline]
    fn rem(self, other: i32) -> Au {
        Au(self.0 % other)
    }
}

impl Neg for Au {
    type Output = Au;

    #[inline]
    fn neg(self) -> Au {
        Au(-self.0)
    }
}

impl Au {
    /// FIXME(pcwalton): Workaround for lack of cross crate inlining of newtype structs!
    #[inline]
    pub fn new(value: i32) -> Au {
        Au(value)
    }

    #[inline]
    pub fn scale_by(self, factor: f32) -> Au {
        Au(((self.0 as f32) * factor) as i32)
    }

    #[inline]
    pub fn from_px(px: i32) -> Au {
        Au((px * 60) as i32)
    }

    /// Rounds this app unit down to the pixel towards zero and returns it.
    #[inline]
    pub fn to_px(self) -> i32 {
        self.0 / 60
    }

    /// Rounds this app unit down to the previous (left or top) pixel and returns it.
    #[inline]
    pub fn to_prev_px(self) -> i32 {
        ((self.0 as f64) / 60f64).floor() as i32
    }

    /// Rounds this app unit up to the next (right or bottom) pixel and returns it.
    #[inline]
    pub fn to_next_px(self) -> i32 {
        ((self.0 as f64) / 60f64).ceil() as i32
    }

    #[inline]
    pub fn to_nearest_px(self) -> i32 {
        ((self.0 as f64) / 60f64).round() as i32
    }

    #[inline]
    pub fn to_f32_px(self) -> f32 {
        (self.0 as f32) / 60f32
    }

    #[inline]
    pub fn to_f64_px(self) -> f64 {
        (self.0 as f64) / 60f64
    }

    #[inline]
    pub fn to_snapped(self) -> Au {
        let res = self.0 % 60i32;
        return if res >= 30i32 { return Au(self.0 - res + 60i32) }
                       else { return Au(self.0 - res) };
    }

    #[inline]
    pub fn from_f32_px(px: f32) -> Au {
        Au((px * 60f32) as i32)
    }

    #[inline]
    pub fn from_pt(pt: f64) -> Au {
        Au::from_f64_px(pt_to_px(pt))
    }

    #[inline]
    pub fn from_f64_px(px: f64) -> Au {
        Au((px * 60.) as i32)
    }
}

// assumes 72 points per inch, and 96 px per inch
pub fn pt_to_px(pt: f64) -> f64 {
    pt / 72. * 96.
}

// assumes 72 points per inch, and 96 px per inch
pub fn px_to_pt(px: f64) -> f64 {
    px / 96. * 72.
}

#[derive(Debug, Clone, Copy)]
pub enum ImageFormat {
    Invalid,
    A8,
    RGB8,
    RGBA8,
}

#[derive(Debug, Clone)]
pub struct ColorF {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl ColorF {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> ColorF {
        ColorF {
            r: r,
            g: g,
            b: b,
            a: a,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackingLevel {
    BackgroundAndBorders,
    BlockBackgroundAndBorders,
    Floats,
    Content,
    PositionedContent,
    Outlines,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Epoch(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineId(pub u32);

static RESOURCE_ID_COUNTER: AtomicUsize = ATOMIC_USIZE_INIT;

#[inline]
fn new_resource_id() -> usize {
    RESOURCE_ID_COUNTER.fetch_add(1, Ordering::SeqCst)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DrawListID(usize);

impl DrawListID {
    pub fn new() -> DrawListID {
        DrawListID(new_resource_id())
    }

    pub fn invalid() -> DrawListID {
        DrawListID(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderTargetID(pub u32);     // TODO: HACK HACK HACK this is an alias for device::TextureId

impl RenderTargetID {
    pub fn new(id: u32) -> RenderTargetID {
        RenderTargetID(id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScrollLayerId(pub usize);

impl ScrollLayerId {
    pub fn new(value: usize) -> ScrollLayerId {
        ScrollLayerId(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DisplayListID(usize);

impl DisplayListID {
    pub fn new() -> DisplayListID {
        DisplayListID(new_resource_id())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImageID(usize);

impl ImageID {
    pub fn new() -> ImageID {
        ImageID(new_resource_id())
    }
}

#[derive(Debug, Copy, Clone)]
pub enum BoxShadowClipMode {
    None,
    Outset,
    Inset,
}

#[derive(Debug)]
pub struct BorderSide {
    pub width: f32,
    pub color: ColorF,
    pub style: BorderStyle,
}

#[derive(Debug)]
pub struct GradientStop {
    pub offset: f32,
    pub color: ColorF,
}

#[derive(Debug)]
pub struct BorderRadius {
    pub top_left: Size2D<f32>,
    pub top_right: Size2D<f32>,
    pub bottom_left: Size2D<f32>,
    pub bottom_right: Size2D<f32>,
}

#[derive(Debug)]
pub enum BorderStyle {
    None,
    Solid,
    Double,
    Dotted,
    Dashed,
    Hidden,
    Groove,
    Ridge,
    Inset,
    Outset,
}

#[derive(Debug)]
pub struct GlyphInstance {
    pub index: u32,
    pub x: f32,
    pub y: f32,
}

pub struct StackingContext {
    pub scroll_layer_id: Option<ScrollLayerId>,
    pub bounds: Rect<f32>,
    pub overflow: Rect<f32>,
    pub z_index: i32,
    pub display_lists: Vec<DisplayListID>,
    pub children: Vec<StackingContext>,
    pub transform: Matrix4,
    pub perspective: Matrix4,
    pub establishes_3d_context: bool,
}

impl StackingContext {
    pub fn new(scroll_layer_id: Option<ScrollLayerId>,
               bounds: Rect<f32>,
               overflow: Rect<f32>,
               z_index: i32,
               transform: &Matrix4,
               perspective: &Matrix4,
               establishes_3d_context: bool) -> StackingContext {
        StackingContext {
            scroll_layer_id: scroll_layer_id,
            bounds: bounds,
            overflow: overflow,
            z_index: z_index,
            display_lists: Vec::new(),
            children: Vec::new(),
            transform: transform.clone(),
            perspective: perspective.clone(),
            establishes_3d_context: establishes_3d_context,
        }
    }

    pub fn add_stacking_context(&mut self, stacking_context: StackingContext) {
        self.children.push(stacking_context);
    }

    pub fn add_display_list(&mut self, id: DisplayListID) {
        self.display_lists.push(id);
    }
}

#[derive(Debug)]
pub struct TextDisplayItem {
    pub glyphs: Vec<GlyphInstance>,
    pub font_id: Atom,
    pub size: Au,
    pub color: ColorF,
}

#[derive(Debug)]
pub struct ImageDisplayItem {
    pub image_id: ImageID,
    pub stretch_size: Size2D<f32>,
}

#[derive(Debug)]
pub struct RectangleDisplayItem {
    pub color: ColorF,
}

#[derive(Debug)]
pub struct RenderTargetDisplayItem {
    pub id: RenderTargetID,
}

#[derive(Debug)]
pub struct BorderDisplayItem {
    pub left: BorderSide,
    pub right: BorderSide,
    pub top: BorderSide,
    pub bottom: BorderSide,
    pub radius: BorderRadius,
}

impl BorderDisplayItem {
    pub fn top_left_inner_radius(&self) -> Size2D<f32> {
        Size2D::new((self.radius.top_left.width - self.left.width).max(0.0),
                    (self.radius.top_left.height - self.top.width).max(0.0))
    }

    pub fn top_right_inner_radius(&self) -> Size2D<f32> {
        Size2D::new((self.radius.top_right.width - self.right.width).max(0.0),
                    (self.radius.top_right.height - self.top.width).max(0.0))
    }

    pub fn bottom_left_inner_radius(&self) -> Size2D<f32> {
        Size2D::new((self.radius.bottom_left.width - self.left.width).max(0.0),
                    (self.radius.bottom_left.height - self.bottom.width).max(0.0))
    }

    pub fn bottom_right_inner_radius(&self) -> Size2D<f32> {
        Size2D::new((self.radius.bottom_right.width - self.right.width).max(0.0),
                    (self.radius.bottom_right.height - self.bottom.width).max(0.0))
    }
}

#[derive(Debug)]
pub struct BoxShadowDisplayItem {
    pub box_bounds: Rect<f32>,
    pub offset: Point2D<f32>,
    pub color: ColorF,
    pub blur_radius: f32,
    pub spread_radius: f32,
    pub clip_mode: BoxShadowClipMode,
}

#[derive(Debug)]
pub struct GradientDisplayItem {
    pub start_point: Point2D<f32>,
    pub end_point: Point2D<f32>,
    pub stops: Vec<GradientStop>,
}

#[derive(Debug)]
pub struct IframeDisplayItem {
    pub iframe: PipelineId,
}

#[derive(Debug)]
pub enum SpecificDisplayItem {
    Rectangle(RectangleDisplayItem),
    Text(TextDisplayItem),
    Image(ImageDisplayItem),
    Border(BorderDisplayItem),
    BoxShadow(BoxShadowDisplayItem),
    Gradient(GradientDisplayItem),
    Iframe(IframeDisplayItem),
    RenderTarget(RenderTargetDisplayItem),
}

#[derive(Debug)]
pub struct DisplayItem {
    pub item: SpecificDisplayItem,
    pub rect: Rect<f32>,
    pub clip: ClipRegion,
}

pub enum DisplayListMode {
    Default,
    PseudoFloat,
    PseudoPositionedContent,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct BorderRadiusRasterOp {
    pub outer_radius_x: Au,
    pub outer_radius_y: Au,
    pub inner_radius_x: Au,
    pub inner_radius_y: Au,
}

impl BorderRadiusRasterOp {
    pub fn create(outer_radius: &Size2D<f32>, inner_radius: &Size2D<f32>) -> Option<BorderRadiusRasterOp> {
        if outer_radius.width > 0.0 || outer_radius.height > 0.0 {
            Some(BorderRadiusRasterOp {
                outer_radius_x: Au::from_f32_px(outer_radius.width),
                outer_radius_y: Au::from_f32_px(outer_radius.height),
                inner_radius_x: Au::from_f32_px(inner_radius.width),
                inner_radius_y: Au::from_f32_px(inner_radius.height),
            })
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum RasterItem {
    BorderRadius(BorderRadiusRasterOp),
}

pub struct DrawList {
    pub items: Vec<DisplayItem>,
}

impl DrawList {
    pub fn new() -> DrawList {
        DrawList {
            items: Vec::new(),
        }
    }

    #[inline]
    pub fn push(&mut self, item: DisplayItem) {
        self.items.push(item);
    }

    #[inline]
    pub fn item_count(&self) -> usize {
        self.items.len()
    }
}

pub struct DisplayListBuilder {
    pub mode: DisplayListMode,

    pub background_and_borders: DrawList,
    pub block_backgrounds_and_borders: DrawList,
    pub floats: DrawList,
    pub content: DrawList,
    pub positioned_content: DrawList,
    pub outlines: DrawList,
}

impl DisplayListBuilder {
    pub fn new() -> DisplayListBuilder {
        DisplayListBuilder {
            mode: DisplayListMode::Default,

            background_and_borders: DrawList::new(),
            block_backgrounds_and_borders: DrawList::new(),
            floats: DrawList::new(),
            content: DrawList::new(),
            positioned_content: DrawList::new(),
            outlines: DrawList::new(),
        }
    }

    pub fn set_mode(&mut self, mode: DisplayListMode) {
        self.mode = mode;
    }

    #[inline]
    pub fn item_count(&self) -> usize {
        self.background_and_borders.item_count() +
        self.block_backgrounds_and_borders.item_count() +
        self.floats.item_count() +
        self.content.item_count() +
        self.positioned_content.item_count() +
        self.outlines.item_count()
    }

    pub fn push_rect(&mut self,
                     level: StackingLevel,
                     rect: Rect<f32>,
                     clip: ClipRegion,
                     color: ColorF) {

        let item = RectangleDisplayItem {
            color: color,
        };

        let display_item = DisplayItem {
            item: SpecificDisplayItem::Rectangle(item),
            rect: rect,
            clip: clip,
        };

        self.push_item(level, display_item);
    }

    pub fn push_image(&mut self,
                      level: StackingLevel,
                      rect: Rect<f32>,
                      clip: ClipRegion,
                      stretch_size: Size2D<f32>,
                      id: ImageID) {
        let item = ImageDisplayItem {
            image_id: id,
            stretch_size: stretch_size,
        };

        let display_item = DisplayItem {
            item: SpecificDisplayItem::Image(item),
            rect: rect,
            clip: clip,
        };

        self.push_item(level, display_item);
    }

    pub fn push_text(&mut self,
                     level: StackingLevel,
                     rect: Rect<f32>,
                     clip: ClipRegion,
                     glyphs: Vec<GlyphInstance>,
                     font_id: Atom,
                     color: ColorF,
                     size: Au) {
        let item = TextDisplayItem {
            color: color,
            glyphs: glyphs,
            font_id: font_id,
            size: size,
        };

        let display_item = DisplayItem {
            item: SpecificDisplayItem::Text(item),
            rect: rect,
            clip: clip,
        };

        self.push_item(level, display_item);
    }

    pub fn push_border(&mut self,
                       level: StackingLevel,
                       rect: Rect<f32>,
                       clip: ClipRegion,
                       left: BorderSide,
                       top: BorderSide,
                       right: BorderSide,
                       bottom: BorderSide,
                       radius: BorderRadius) {
        let item = BorderDisplayItem {
            left: left,
            top: top,
            right: right,
            bottom: bottom,
            radius: radius,
        };

        let display_item = DisplayItem {
            item: SpecificDisplayItem::Border(item),
            rect: rect,
            clip: clip,
        };

        self.push_item(level, display_item);
    }

    pub fn push_box_shadow(&mut self,
                           level: StackingLevel,
                           rect: Rect<f32>,
                           clip: ClipRegion,
                           box_bounds: Rect<f32>,
                           offset: Point2D<f32>,
                           color: ColorF,
                           blur_radius: f32,
                           spread_radius: f32,
                           clip_mode: BoxShadowClipMode) {
        let item = BoxShadowDisplayItem {
            box_bounds: box_bounds,
            offset: offset,
            color: color,
            blur_radius: blur_radius,
            spread_radius: spread_radius,
            clip_mode: clip_mode,
        };

        let display_item = DisplayItem {
            item: SpecificDisplayItem::BoxShadow(item),
            rect: rect,
            clip: clip,
        };

        self.push_item(level, display_item);
    }

    pub fn push_gradient(&mut self,
                         level: StackingLevel,
                         rect: Rect<f32>,
                         clip: ClipRegion,
                         start_point: Point2D<f32>,
                         end_point: Point2D<f32>,
                         stops: Vec<GradientStop>) {
        let item = GradientDisplayItem {
            start_point: start_point,
            end_point: end_point,
            stops: stops,
        };

        let display_item = DisplayItem {
            item: SpecificDisplayItem::Gradient(item),
            rect: rect,
            clip: clip,
        };

        self.push_item(level, display_item);
    }

    pub fn push_iframe(&mut self,
                       level: StackingLevel,
                       rect: Rect<f32>,
                       clip: ClipRegion,
                       iframe: PipelineId) {
        assert!(level == StackingLevel::Content ||
                level == StackingLevel::PositionedContent ||
                level == StackingLevel::Floats, format!("push_iframe: level={:?}", level));       // invariant in get_draw_lists_for_stacking_context

        let item = IframeDisplayItem {
            iframe: iframe,
        };

        let display_item = DisplayItem {
            item: SpecificDisplayItem::Iframe(item),
            rect: rect,
            clip: clip,
        };

        self.push_item(level, display_item);
    }

    fn push_item(&mut self, level: StackingLevel, item: DisplayItem) {
        match level {
            StackingLevel::BackgroundAndBorders => {
                self.background_and_borders.push(item);
            }
            StackingLevel::BlockBackgroundAndBorders => {
                self.block_backgrounds_and_borders.push(item);
            }
            StackingLevel::Floats => {
                self.floats.push(item);
            }
            StackingLevel::Content => {
                self.content.push(item);
            }
            StackingLevel::PositionedContent => {
                self.positioned_content.push(item);
            }
            StackingLevel::Outlines => {
                self.outlines.push(item);
            }
        }
    }
}

pub trait RenderNotifier : Send {
    fn new_frame_ready(&mut self);
}

#[derive(PartialEq, Eq, Hash)]
pub struct Glyph {
    pub size: Au,
    pub index: u32,
}

impl Glyph {
    #[inline]
    pub fn new(size: Au, index: u32) -> Glyph {
        Glyph {
            size: size,
            index: index,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClipRegion {
    pub main: Rect<f32>,
}

impl ClipRegion {
    pub fn new(rect: Rect<f32>) -> ClipRegion {
        ClipRegion {
            main: rect,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Copy, Deserialize, Serialize, Debug)]
pub enum ScrollPolicy {
    Scrollable,
    FixedPosition,
}
