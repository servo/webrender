/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use display_list::{AuxiliaryListsBuilder, ItemRange};
use euclid::{Point2D, Rect, Size2D};

#[cfg(target_os = "macos")] use core_graphics::font::CGFont;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BorderRadius {
    pub top_left: Size2D<f32>,
    pub top_right: Size2D<f32>,
    pub bottom_left: Size2D<f32>,
    pub bottom_right: Size2D<f32>,
}

impl BorderRadius {
    pub fn zero() -> BorderRadius {
        BorderRadius {
            top_left: Size2D::new(0.0, 0.0),
            top_right: Size2D::new(0.0, 0.0),
            bottom_left: Size2D::new(0.0, 0.0),
            bottom_right: Size2D::new(0.0, 0.0),
        }
    }

    pub fn uniform(radius: f32) -> BorderRadius {
        BorderRadius {
            top_left: Size2D::new(radius, radius),
            top_right: Size2D::new(radius, radius),
            bottom_left: Size2D::new(radius, radius),
            bottom_right: Size2D::new(radius, radius),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BorderSide {
    pub width: f32,
    pub color: ColorF,
    pub style: BorderStyle,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum BoxShadowClipMode {
    None,
    Outset,
    Inset,
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClipRegion {
    pub main: Rect<f32>,
    pub complex: ItemRange,
}

impl ClipRegion {
    pub fn new(rect: &Rect<f32>,
               complex: Vec<ComplexClipRegion>,
               auxiliary_lists_builder: &mut AuxiliaryListsBuilder)
               -> ClipRegion {
        ClipRegion {
            main: *rect,
            complex: auxiliary_lists_builder.add_complex_clip_regions(&complex),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ComplexClipRegion {
    /// The boundaries of the rectangle.
    pub rect: Rect<f32>,
    /// Border radii of this rectangle.
    pub radii: BorderRadius,
}

impl ComplexClipRegion {
    pub fn new(rect: Rect<f32>, radii: BorderRadius) -> ComplexClipRegion {
        ComplexClipRegion {
            rect: rect,
            radii: radii,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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

    pub fn scale_rgb(&self, scale: f32) -> ColorF {
        ColorF {
            r: self.r * scale,
            g: self.g * scale,
            b: self.b * scale,
            a: self.a,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StackingContextId(pub u32, pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ServoStackingContextId(pub FragmentType, pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FragmentType {
    FragmentBody,
    BeforePseudoContent,
    AfterPseudoContent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DisplayListMode {
    Default,
    PseudoFloat,
    PseudoPositionedContent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DisplayListId(pub u32, pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct Epoch(pub u32);

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum FilterOp {
    Blur(Au),
    Brightness(f32),
    Contrast(f32),
    Grayscale(f32),
    HueRotate(f32),
    Invert(f32),
    Opacity(f32),
    Saturate(f32),
    Sepia(f32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct FontKey(u32, u32);

impl FontKey {
    pub fn new(key0: u32, key1: u32) -> FontKey {
        FontKey(key0, key1)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GlyphInstance {
    pub index: u32,
    pub x: f32,
    pub y: f32,
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GradientStop {
    pub offset: f32,
    pub color: ColorF,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageFormat {
    Invalid,
    A8,
    RGB8,
    RGBA8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageRendering {
    Auto,
    CrispEdges,
    Pixelated,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ImageKey(u32, u32);

impl ImageKey {
    pub fn new(key0: u32, key1: u32) -> ImageKey {
        ImageKey(key0, key1)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum MixBlendMode {
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
    Hue,
    Saturation,
    Color,
    Luminosity,
}

#[cfg(target_os = "macos")]
pub type NativeFontHandle = CGFont;

/// Native fonts are not used on Linux; all fonts are raw.
#[cfg(not(target_os = "macos"))]
#[derive(Clone, Serialize, Deserialize)]
pub struct NativeFontHandle;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PipelineId(pub u32, pub u32);

pub trait RenderNotifier : Send {
    fn new_frame_ready(&mut self);
    fn pipeline_size_changed(&mut self,
                             pipeline_id: PipelineId,
                             size: Option<Size2D<f32>>);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScrollLayerInfo {
    Fixed,
    Scrollable(usize)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScrollLayerId {
    pub pipeline_id: PipelineId,
    pub info: ScrollLayerInfo,
}

impl ScrollLayerId {
    pub fn new(pipeline_id: PipelineId, index: usize) -> ScrollLayerId {
        ScrollLayerId {
            pipeline_id: pipeline_id,
            info: ScrollLayerInfo::Scrollable(index),
        }
    }

    pub fn create_fixed(pipeline_id: PipelineId) -> ScrollLayerId {
        ScrollLayerId {
            pipeline_id: pipeline_id,
            info: ScrollLayerInfo::Fixed,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Copy, Deserialize, Serialize, Debug)]
pub enum ScrollPolicy {
    Scrollable,
    Fixed,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ScrollLayerState {
    pub pipeline_id: PipelineId,
    pub stacking_context_id: ServoStackingContextId,
    pub scroll_offset: Point2D<f32>,
}

