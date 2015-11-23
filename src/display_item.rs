use app_units::Au;
use types::{ClipRegion, ColorF, GlyphInstance, FontKey, ImageKey, BorderSide};
use types::{GradientStop, BorderRadius, BoxShadowClipMode, ImageRendering};
use euclid::{Point2D, Rect, Size2D};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct BoxShadowDisplayItem {
    pub box_bounds: Rect<f32>,
    pub offset: Point2D<f32>,
    pub color: ColorF,
    pub blur_radius: f32,
    pub spread_radius: f32,
    pub border_radius: f32,
    pub clip_mode: BoxShadowClipMode,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct GradientDisplayItem {
    pub start_point: Point2D<f32>,
    pub end_point: Point2D<f32>,
    pub stops: Vec<GradientStop>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageDisplayItem {
    pub image_key: ImageKey,
    pub stretch_size: Size2D<f32>,
    pub image_rendering: ImageRendering,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct RectangleDisplayItem {
    pub color: ColorF,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct TextDisplayItem {
    pub glyphs: Vec<GlyphInstance>,
    pub font_key: FontKey,
    pub size: Au,
    pub color: ColorF,
    pub blur_radius: Au,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum SpecificDisplayItem {
    Rectangle(RectangleDisplayItem),
    Text(TextDisplayItem),
    Image(ImageDisplayItem),
    Border(BorderDisplayItem),
    BoxShadow(BoxShadowDisplayItem),
    Gradient(GradientDisplayItem),
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct DisplayItem {
    pub item: SpecificDisplayItem,
    pub rect: Rect<f32>,
    pub clip: ClipRegion,
}
