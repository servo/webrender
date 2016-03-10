/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use display_item::{DisplayItem, SpecificDisplayItem, ImageDisplayItem, WebGLDisplayItem};
use display_item::{RectangleDisplayItem, TextDisplayItem, GradientDisplayItem};
use display_item::{BorderDisplayItem, BoxShadowDisplayItem};
use euclid::{Point2D, Rect, Size2D};
use std::mem;
use std::slice;
use types::{ClipRegion, ColorF, ComplexClipRegion, FontKey, ImageKey, PipelineId, StackingLevel};
use types::{BorderRadius, BorderSide, BoxShadowClipMode, FilterOp, GlyphInstance};
use types::{DisplayListMode, GradientStop, StackingContextId, ImageRendering};
use webgl::{WebGLContextId};

#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct DrawListInfo {
    pub items: ItemRange,
}

#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct StackingContextInfo {
    pub id: StackingContextId,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct IframeInfo {
    pub id: PipelineId,
    pub bounds: Rect<f32>,
}

#[derive(Copy, Clone, Serialize, Deserialize)]
pub enum SpecificDisplayListItem {
    DrawList(DrawListInfo),
    StackingContext(StackingContextInfo),
    Iframe(IframeInfo),
}

#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct DisplayListItem {
    pub stacking_level: StackingLevel,
    pub specific: SpecificDisplayListItem,
}

/// Describes the memory layout of a display list.
///
/// A display list consists of some number of display list items, followed by a number of display
/// items.
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct BuiltDisplayListDescriptor {
    pub mode: DisplayListMode,
    pub has_stacking_contexts: bool,

    /// The size in bytes of the display items in this display list.
    display_list_items_size: usize,
}

/// A display list.
#[derive(Clone, Serialize, Deserialize)]
pub struct BuiltDisplayList {
    data: Vec<u8>,
    descriptor: BuiltDisplayListDescriptor,
}

impl BuiltDisplayList {
    pub fn from_data(data: Vec<u8>, descriptor: BuiltDisplayListDescriptor) -> BuiltDisplayList {
        BuiltDisplayList {
            data: data,
            descriptor: descriptor,
        }
    }

    pub fn data(&self) -> &[u8] {
        &self.data[..]
    }

    pub fn descriptor(&self) -> &BuiltDisplayListDescriptor {
        &self.descriptor
    }

    pub fn display_list_items<'a>(&'a self) -> &'a [DisplayListItem] {
        unsafe {
            convert_blob_to_pod(&self.data[0..self.descriptor.display_list_items_size])
        }
    }

    pub fn display_items<'a>(&'a self, range: &ItemRange) -> &'a [DisplayItem] {
        unsafe {
            range.get(convert_blob_to_pod(&self.data[self.descriptor.display_list_items_size..]))
        }
    }
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

    pub display_list_items: Vec<DisplayListItem>,
    pub display_items: Vec<DisplayItem>,
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

            display_list_items: Vec::new(),
            display_items: Vec::new(),
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
                      image_rendering: ImageRendering,
                      key: ImageKey) {
        let item = ImageDisplayItem {
            image_key: key,
            stretch_size: stretch_size,
            image_rendering: image_rendering,
        };

        let display_item = DisplayItem {
            item: SpecificDisplayItem::Image(item),
            rect: rect,
            clip: clip,
        };

        self.push_item(level, display_item);
    }

    pub fn push_webgl_canvas(&mut self,
                             level: StackingLevel,
                             rect: Rect<f32>,
                             clip: ClipRegion,
                             context_id: WebGLContextId) {
        let item = WebGLDisplayItem {
            context_id: context_id,
        };

        let display_item = DisplayItem {
            item: SpecificDisplayItem::WebGL(item),
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
                     blur_radius: Au,
                     auxiliary_lists_builder: &mut AuxiliaryListsBuilder) {
        // Sanity check - anything with glyphs bigger than this
        // is probably going to consume too much memory to render
        // efficiently anyway. This is specifically to work around
        // the font_advance.html reftest, which creates a very large
        // font as a crash test - the rendering is also ignored
        // by the azure renderer.
        if size < Au::from_px(4096) {
            let item = TextDisplayItem {
                color: color,
                glyphs: auxiliary_lists_builder.add_glyph_instances(&glyphs),
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
        self.display_list_items.push(item);
    }

    pub fn push_gradient(&mut self,
                         level: StackingLevel,
                         rect: Rect<f32>,
                         clip: ClipRegion,
                         start_point: Point2D<f32>,
                         end_point: Point2D<f32>,
                         stops: Vec<GradientStop>,
                         auxiliary_lists_builder: &mut AuxiliaryListsBuilder) {
        let item = GradientDisplayItem {
            start_point: start_point,
            end_point: end_point,
            stops: auxiliary_lists_builder.add_gradient_stops(&stops),
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
        let info = IframeInfo {
            id: iframe,
            bounds: rect,
        };
        let item = DisplayListItem {
            stacking_level: level,
            specific: SpecificDisplayListItem::Iframe(info),
        };
        self.display_list_items.push(item);
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
        if items.is_empty() {
            return
        }

        let draw_list = DrawListInfo {
            items: ItemRange::new(&mut self.display_items, &items),
        };
        self.display_list_items.push(DisplayListItem {
            stacking_level: level,
            specific: SpecificDisplayListItem::DrawList(draw_list),
        });
    }

    fn flush(&mut self) {
        self.flush_list(StackingLevel::BackgroundAndBorders);
        self.flush_list(StackingLevel::BlockBackgroundAndBorders);
        self.flush_list(StackingLevel::Floats);
        self.flush_list(StackingLevel::Content);
        self.flush_list(StackingLevel::PositionedContent);
        self.flush_list(StackingLevel::Outlines);
    }

    pub fn finalize(mut self) -> BuiltDisplayList {
        self.flush();

        unsafe {
            let mut blob = convert_pod_to_blob(&self.display_list_items).to_vec();
            let display_list_items_size = blob.len();
            blob.extend_from_slice(convert_pod_to_blob(&self.display_items));
            BuiltDisplayList {
                descriptor: BuiltDisplayListDescriptor {
                    mode: self.mode,
                    has_stacking_contexts: self.has_stacking_contexts,
                    display_list_items_size: display_list_items_size,
                },
                data: blob,
            }
        }
    }

    pub fn display_items<'a>(&'a self, range: &ItemRange) -> &'a [DisplayItem] {
        range.get(&self.display_items)
    }

    pub fn display_items_mut<'a>(&'a mut self, range: &ItemRange) -> &'a mut [DisplayItem] {
        range.get_mut(&mut self.display_items)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ItemRange {
    pub start: usize,
    pub length: usize,
}

impl ItemRange {
    pub fn new<T>(backing_list: &mut Vec<T>, items: &[T]) -> ItemRange where T: Copy + Clone {
        let start = backing_list.len();
        backing_list.extend_from_slice(items);
        ItemRange {
            start: start,
            length: items.len(),
        }
    }

    pub fn empty() -> ItemRange {
        ItemRange {
            start: 0,
            length: 0,
        }
    }

    pub fn get<'a, T>(&self, backing_list: &'a [T]) -> &'a [T] {
        &backing_list[self.start..(self.start + self.length)]
    }

    pub fn get_mut<'a, T>(&self, backing_list: &'a mut [T]) -> &'a mut [T] {
        &mut backing_list[self.start..(self.start + self.length)]
    }
}

#[derive(Clone)]
pub struct AuxiliaryListsBuilder {
    gradient_stops: Vec<GradientStop>,
    complex_clip_regions: Vec<ComplexClipRegion>,
    filters: Vec<FilterOp>,
    glyph_instances: Vec<GlyphInstance>,
}

impl AuxiliaryListsBuilder {
    pub fn new() -> AuxiliaryListsBuilder {
        AuxiliaryListsBuilder {
            gradient_stops: Vec::new(),
            complex_clip_regions: Vec::new(),
            filters: Vec::new(),
            glyph_instances: Vec::new(),
        }
    }

    pub fn add_gradient_stops(&mut self, gradient_stops: &[GradientStop]) -> ItemRange {
        ItemRange::new(&mut self.gradient_stops, gradient_stops)
    }

    pub fn gradient_stops(&self, gradient_stops_range: &ItemRange) -> &[GradientStop] {
        gradient_stops_range.get(&self.gradient_stops[..])
    }

    pub fn add_complex_clip_regions(&mut self, complex_clip_regions: &[ComplexClipRegion])
                                    -> ItemRange {
        ItemRange::new(&mut self.complex_clip_regions, complex_clip_regions)
    }

    pub fn complex_clip_regions(&self, complex_clip_regions_range: &ItemRange)
                                -> &[ComplexClipRegion] {
        complex_clip_regions_range.get(&self.complex_clip_regions[..])
    }

    pub fn add_filters(&mut self, filters: &[FilterOp]) -> ItemRange {
        ItemRange::new(&mut self.filters, filters)
    }

    pub fn filters(&self, filters_range: &ItemRange) -> &[FilterOp] {
        filters_range.get(&self.filters[..])
    }

    pub fn add_glyph_instances(&mut self, glyph_instances: &[GlyphInstance]) -> ItemRange {
        ItemRange::new(&mut self.glyph_instances, glyph_instances)
    }

    pub fn glyph_instances(&self, glyph_instances_range: &ItemRange) -> &[GlyphInstance] {
        glyph_instances_range.get(&self.glyph_instances[..])
    }

    pub fn finalize(self) -> AuxiliaryLists {
        unsafe {
            let mut blob = convert_pod_to_blob(&self.gradient_stops).to_vec();
            let gradient_stops_size = blob.len();
            blob.extend_from_slice(convert_pod_to_blob(&self.complex_clip_regions));
            let complex_clip_regions_size = blob.len() - gradient_stops_size;
            blob.extend_from_slice(convert_pod_to_blob(&self.filters));
            let filters_size = blob.len() - (complex_clip_regions_size + gradient_stops_size);
            blob.extend_from_slice(convert_pod_to_blob(&self.glyph_instances));

            AuxiliaryLists {
                data: blob,
                descriptor: AuxiliaryListsDescriptor {
                    gradient_stops_size: gradient_stops_size,
                    complex_clip_regions_size: complex_clip_regions_size,
                    filters_size: filters_size,
                },
            }
        }
    }
}

/// Describes the memory layout of the auxiliary lists.
///
/// Auxiliary lists consist of some number of gradient stops, complex clip regions, filters, and
/// glyph instances, in that order.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct AuxiliaryListsDescriptor {
    gradient_stops_size: usize,
    complex_clip_regions_size: usize,
    filters_size: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AuxiliaryLists {
    /// The concatenation of: gradient stops, complex clip regions, filters, and glyph instances,
    /// in that order.
    data: Vec<u8>,
    descriptor: AuxiliaryListsDescriptor,
}

impl AuxiliaryLists {
    /// Creates a new `AuxiliaryLists` instance from a descriptor and data received over a channel.
    pub fn from_data(data: Vec<u8>, descriptor: AuxiliaryListsDescriptor) -> AuxiliaryLists {
        AuxiliaryLists {
            data: data,
            descriptor: descriptor,
        }
    }

    pub fn data(&self) -> &[u8] {
        &self.data[..]
    }

    pub fn descriptor(&self) -> &AuxiliaryListsDescriptor {
        &self.descriptor
    }

    /// Returns the gradient stops described by `gradient_stops_range`.
    pub fn gradient_stops(&self, gradient_stops_range: &ItemRange) -> &[GradientStop] {
        unsafe {
            let end = self.descriptor.gradient_stops_size;
            gradient_stops_range.get(convert_blob_to_pod(&self.data[0..end]))
        }
    }

    /// Returns the complex clipping regions described by `complex_clip_regions_range`.
    pub fn complex_clip_regions(&self, complex_clip_regions_range: &ItemRange)
                                -> &[ComplexClipRegion] {
        let start = self.descriptor.gradient_stops_size;
        let end = start + self.descriptor.complex_clip_regions_size;
        unsafe {
            complex_clip_regions_range.get(convert_blob_to_pod(&self.data[start..end]))
        }
    }

    /// Returns the filters described by `filters_range`.
    pub fn filters(&self, filters_range: &ItemRange) -> &[FilterOp] {
        let start = self.descriptor.gradient_stops_size +
            self.descriptor.complex_clip_regions_size;
        let end = start + self.descriptor.filters_size;
        unsafe {
            filters_range.get(convert_blob_to_pod(&self.data[start..end]))
        }
    }

    /// Returns the glyph instances described by `glyph_instances_range`.
    pub fn glyph_instances(&self, glyph_instances_range: &ItemRange) -> &[GlyphInstance] {
        let start = self.descriptor.gradient_stops_size +
            self.descriptor.complex_clip_regions_size + self.descriptor.filters_size;
        unsafe {
            glyph_instances_range.get(convert_blob_to_pod(&self.data[start..]))
        }
    }
}

unsafe fn convert_pod_to_blob<T>(data: &[T]) -> &[u8] where T: Copy + 'static {
    slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * mem::size_of::<T>())
}

unsafe fn convert_blob_to_pod<T>(blob: &[u8]) -> &[T] where T: Copy + 'static {
    slice::from_raw_parts(blob.as_ptr() as *const T, blob.len() / mem::size_of::<T>())
}

