use app_units::Au;
use display_item::{DisplayItem, SpecificDisplayItem, ImageDisplayItem};
use display_item::{RectangleDisplayItem, TextDisplayItem, GradientDisplayItem};
use display_item::{BorderDisplayItem, BoxShadowDisplayItem};
use euclid::{Point2D, Rect, Size2D};
use std::mem;
use types::{ClipRegion, ColorF, FontKey, ImageKey, PipelineId, StackingLevel};
use types::{BorderRadius, BorderSide, BoxShadowClipMode, GlyphInstance};
use types::{DisplayListMode, GradientStop, StackingContextId};

pub struct DrawListInfo {
    pub items: Vec<DisplayItem>,
}

pub struct StackingContextInfo {
    pub id: StackingContextId,
}

#[derive(Debug, Clone)]
pub struct IframeInfo {
    pub id: PipelineId,
    pub offset: Point2D<f32>,
    pub clip: Rect<f32>,
}

pub enum SpecificDisplayListItem {
    DrawList(DrawListInfo),
    StackingContext(StackingContextInfo),
    Iframe(Box<IframeInfo>),
}

pub struct DisplayListItem {
    pub stacking_level: StackingLevel,
    pub specific: SpecificDisplayListItem,
}

pub struct DisplayListBuilder {
    pub mode: DisplayListMode,
    pub has_stacking_contexts: bool,

    pub work_background_and_borders: Vec<DisplayItem>,
    pub work_block_backgrounds_and_borders: Vec<DisplayItem>,
    pub work_floats: Vec<DisplayItem>,
    pub work_content: Vec<DisplayItem>,
    pub work_positioned_content: Vec<DisplayItem>,
    pub work_outlines: Vec<DisplayItem>,

    pub items: Vec<DisplayListItem>,
}

impl DisplayListBuilder {
    pub fn new() -> DisplayListBuilder {
        DisplayListBuilder {
            mode: DisplayListMode::Default,
            has_stacking_contexts: false,

            work_background_and_borders: Vec::new(),
            work_block_backgrounds_and_borders: Vec::new(),
            work_floats: Vec::new(),
            work_content: Vec::new(),
            work_positioned_content: Vec::new(),
            work_outlines: Vec::new(),

            items: Vec::new(),
        }
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
                      key: ImageKey) {
        let item = ImageDisplayItem {
            image_key: key,
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
                     font_key: FontKey,
                     color: ColorF,
                     size: Au,
                     blur_radius: Au) {
        let item = TextDisplayItem {
            color: color,
            glyphs: glyphs,
            font_key: font_key,
            size: size,
            blur_radius: blur_radius,
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
                           border_radius: f32,
                           clip_mode: BoxShadowClipMode) {
        let item = BoxShadowDisplayItem {
            box_bounds: box_bounds,
            offset: offset,
            color: color,
            blur_radius: blur_radius,
            spread_radius: spread_radius,
            border_radius: border_radius,
            clip_mode: clip_mode,
        };

        let display_item = DisplayItem {
            item: SpecificDisplayItem::BoxShadow(item),
            rect: rect,
            clip: clip,
        };

        self.push_item(level, display_item);
    }

    pub fn push_stacking_context(&mut self,
                                 level: StackingLevel,
                                 stacking_context_id: StackingContextId) {
        self.has_stacking_contexts = true;
        self.flush_list(level);
        let info = StackingContextInfo {
            id: stacking_context_id,
        };
        let item = DisplayListItem {
            stacking_level: level,
            specific: SpecificDisplayListItem::StackingContext(info),
        };
        self.items.push(item);
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
                       _clip: ClipRegion,
                       iframe: PipelineId) {
        self.flush_list(level);
        let info = Box::new(IframeInfo {
            id: iframe,
            offset: rect.origin,
            clip: rect,
        });
        let item = DisplayListItem {
            stacking_level: level,
            specific: SpecificDisplayListItem::Iframe(info),
        };
        self.items.push(item);
    }

    fn push_item(&mut self, level: StackingLevel, item: DisplayItem) {
        match level {
            StackingLevel::BackgroundAndBorders => {
                self.work_background_and_borders.push(item);
            }
            StackingLevel::BlockBackgroundAndBorders => {
                self.work_block_backgrounds_and_borders.push(item);
            }
            StackingLevel::Floats => {
                self.work_floats.push(item);
            }
            StackingLevel::Content => {
                self.work_content.push(item);
            }
            StackingLevel::PositionedContent => {
                self.work_positioned_content.push(item);
            }
            StackingLevel::Outlines => {
                self.work_outlines.push(item);
            }
        }
    }

    fn flush_list(&mut self, level: StackingLevel) {
        let list = match level {
            StackingLevel::BackgroundAndBorders => {
                &mut self.work_background_and_borders
            }
            StackingLevel::BlockBackgroundAndBorders => {
                &mut self.work_block_backgrounds_and_borders
            }
            StackingLevel::Floats => {
                &mut self.work_floats
            }
            StackingLevel::Content => {
                &mut self.work_content
            }
            StackingLevel::PositionedContent => {
                &mut self.work_positioned_content
            }
            StackingLevel::Outlines => {
                &mut self.work_outlines
            }
        };

        let items = mem::replace(list, Vec::new());
        if items.len() > 0 {
            let draw_list = DrawListInfo {
                items: items,
            };
            self.items.push(DisplayListItem {
                stacking_level: level,
                specific: SpecificDisplayListItem::DrawList(draw_list),
            });
        }
    }

    fn flush(&mut self) {
        self.flush_list(StackingLevel::BackgroundAndBorders);
        self.flush_list(StackingLevel::BlockBackgroundAndBorders);
        self.flush_list(StackingLevel::Floats);
        self.flush_list(StackingLevel::Content);
        self.flush_list(StackingLevel::PositionedContent);
        self.flush_list(StackingLevel::Outlines);
    }

    pub fn finalize(&mut self) {
        self.flush();
    }
}
