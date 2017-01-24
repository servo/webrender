/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// Every serialisable type is defined in this file to only codegen one file
// for the serde implementations.

use app_units::Au;
use channel::{PayloadSender, MsgSender};
#[cfg(feature = "nightly")]
use core::nonzero::NonZero;
#[cfg(feature = "webgl")]
use offscreen_gl_context::{GLContextAttributes, GLLimits};
use std::sync::Arc;

#[cfg(target_os = "macos")] use core_graphics::font::CGFont;
#[cfg(target_os = "windows")] use dwrote::FontDescriptor;

#[derive(Debug, Copy, Clone)]
pub enum RendererKind {
    Native,
    OSMesa,
}

#[derive(Clone, Deserialize, Serialize)]
pub enum ApiMsg {
    AddRawFont(FontKey, Vec<u8>),
    AddNativeFont(FontKey, NativeFontHandle),
    /// Gets the glyph dimensions
    GetGlyphDimensions(Vec<GlyphKey>, MsgSender<Vec<Option<GlyphDimensions>>>),
    /// Adds an image from the resource cache.
    AddImage(ImageKey, ImageDescriptor, ImageData),
    /// Updates the the resource cache with the new image data.
    UpdateImage(ImageKey, ImageDescriptor, Vec<u8>),
    /// Drops an image from the resource cache.
    DeleteImage(ImageKey),
    CloneApi(MsgSender<IdNamespace>),
    /// Supplies a new frame to WebRender.
    ///
    /// After receiving this message, WebRender will read the display list, followed by the
    /// auxiliary lists, from the payload channel.
    SetRootDisplayList(Option<ColorF>,
                       Epoch,
                       PipelineId,
                       LayoutSize,
                       BuiltDisplayListDescriptor,
                       AuxiliaryListsDescriptor),
    SetRootPipeline(PipelineId),
    Scroll(ScrollLocation, WorldPoint, ScrollEventPhase),
    ScrollLayersWithScrollId(LayoutPoint, PipelineId, ServoScrollRootId),
    TickScrollingBounce,
    TranslatePointToLayerSpace(WorldPoint, MsgSender<(LayoutPoint, PipelineId)>),
    GetScrollLayerState(MsgSender<Vec<ScrollLayerState>>),
    #[cfg(feature = "webgl")]
    RequestWebGLContext(DeviceIntSize, GLContextAttributes, MsgSender<Result<(WebGLContextId, GLLimits), String>>),
    #[cfg(feature = "webgl")]
    ResizeWebGLContext(WebGLContextId, DeviceIntSize),
    WebGLCommand(WebGLContextId, WebGLCommand),
    GenerateFrame,
    // WebVR commands that must be called in the WebGL render thread.
    VRCompositorCommand(WebGLContextId, VRCompositorCommand),
    /// An opaque handle that must be passed to the render notifier. It is used by Gecko
    /// to forward gecko-specific messages to the render thread preserving the ordering
    /// within the other messages.
    ExternalEvent(ExternalEvent),
    ShutDown,
}

/// An opaque pointer-sized value.
#[derive(Clone, Deserialize, Serialize)]
pub struct ExternalEvent {
    raw: usize,
}

unsafe impl Send for ExternalEvent {}

impl ExternalEvent {
    pub fn from_raw(raw: usize) -> Self { ExternalEvent { raw: raw } }
    /// Consumes self to make it obvious that the event should be forwarded only once.
    pub fn unwrap(self) -> usize { self.raw }
}

#[derive(Copy, Clone, Deserialize, Serialize, Debug)]
pub struct GlyphDimensions {
    pub left: i32,
    pub top: i32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct AuxiliaryLists {
    /// The concatenation of: gradient stops, complex clip regions, filters, and glyph instances,
    /// in that order.
    data: Vec<u8>,
    descriptor: AuxiliaryListsDescriptor,
}

/// Describes the memory layout of the auxiliary lists.
///
/// Auxiliary lists consist of some number of gradient stops, complex clip regions, filters, and
/// glyph instances, in that order.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct AuxiliaryListsDescriptor {
    gradient_stops_size: usize,
    complex_clip_regions_size: usize,
    filters_size: usize,
    glyph_instances_size: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct BorderDisplayItem {
    pub left: BorderSide,
    pub right: BorderSide,
    pub top: BorderSide,
    pub bottom: BorderSide,
    pub radius: BorderRadius,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct BorderRadius {
    pub top_left: LayoutSize,
    pub top_right: LayoutSize,
    pub bottom_left: LayoutSize,
    pub bottom_right: LayoutSize,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct BorderSide {
    pub width: f32,
    pub color: ColorF,
    pub style: BorderStyle,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
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

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub enum BoxShadowClipMode {
    None,
    Outset,
    Inset,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct BoxShadowDisplayItem {
    pub box_bounds: LayoutRect,
    pub offset: LayoutPoint,
    pub color: ColorF,
    pub blur_radius: f32,
    pub spread_radius: f32,
    pub border_radius: f32,
    pub clip_mode: BoxShadowClipMode,
}

/// A display list.
#[derive(Clone, Deserialize, Serialize)]
pub struct BuiltDisplayList {
    data: Vec<u8>,
    descriptor: BuiltDisplayListDescriptor,
}

/// Describes the memory layout of a display list.
///
/// A display list consists of some number of display list items, followed by a number of display
/// items.
#[derive(Copy, Clone, Deserialize, Serialize)]
pub struct BuiltDisplayListDescriptor {
    pub mode: DisplayListMode,

    /// The size in bytes of the display list items in this display list.
    display_list_items_size: usize,
    /// The size in bytes of the display items in this display list.
    display_items_size: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct ColorF {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}
known_heap_size!(0, ColorF);

#[derive(Clone, Copy, Hash, Eq, Debug, Deserialize, PartialEq, PartialOrd, Ord, Serialize)]
pub struct ColorU {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl From<ColorF> for ColorU {
    fn from(color: ColorF) -> ColorU {
        ColorU {
            r: ColorU::round_to_int(color.r),
            g: ColorU::round_to_int(color.g),
            b: ColorU::round_to_int(color.b),
            a: ColorU::round_to_int(color.a),
        }
    }
}

impl Into<ColorF> for ColorU {
    fn into(self) -> ColorF {
        ColorF {
            r: self.r as f32 / 255.0,
            g: self.g as f32 / 255.0,
            b: self.b as f32 / 255.0,
            a: self.a as f32 / 255.0,
        }
    }
}

impl ColorU {
    fn round_to_int(x: f32) -> u8 {
        debug_assert!((0.0 <= x) && (x <= 1.0));
        let f = (255.0 * x) + 0.5;
        let val = f.floor();
        debug_assert!(val <= 255.0);
        val as u8
    }

    pub fn new(r: u8, g: u8, b: u8, a: u8) -> ColorU {
        ColorU {
            r: r,
            g: g,
            b: b,
            a: a,
        }
    }
}

#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ImageDescriptor {
    pub format: ImageFormat,
    pub width: u32,
    pub height: u32,
    pub stride: Option<u32>,
    pub is_opaque: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct ImageMask {
    pub image: ImageKey,
    pub rect: LayoutRect,
    pub repeat: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct ClipRegion {
    pub main: LayoutRect,
    pub complex: ItemRange,
    pub image_mask: Option<ImageMask>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct ComplexClipRegion {
    /// The boundaries of the rectangle.
    pub rect: LayoutRect,
    /// Border radii of this rectangle.
    pub radii: BorderRadius,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct DisplayItem {
    pub item: SpecificDisplayItem,
    pub rect: LayoutRect,
    pub clip: ClipRegion,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum DisplayListMode {
    Default,
    PseudoFloat,
    PseudoPositionedContent,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Epoch(pub u32);

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize, Ord, PartialOrd)]
pub struct FontKey(u32, u32);

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, Serialize, Deserialize, Ord, PartialOrd)]
pub enum FontRenderMode {
    Mono,
    Alpha,
    Subpixel,
}

#[derive(Clone, Hash, PartialEq, Eq, Debug, Deserialize, Serialize, Ord, PartialOrd)]
pub struct GlyphKey {
    pub font_key: FontKey,
    // The font size is in *device* pixels, not logical pixels.
    // It is stored as an Au since we need sub-pixel sizes, but
    // can't store as a f32 due to use of this type as a hash key.
    // TODO(gw): Perhaps consider having LogicalAu and DeviceAu
    //           or something similar to that.
    pub size: Au,
    pub index: u32,
    pub color: ColorU,
}

impl GlyphKey {
    pub fn new(font_key: FontKey,
               size: Au,
               color: ColorF,
               index: u32) -> GlyphKey {
        GlyphKey {
            font_key: font_key,
            size: size,
            color: ColorU::from(color),
            index: index,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum FragmentType {
    FragmentBody,
    BeforePseudoContent,
    AfterPseudoContent,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct GlyphInstance {
    pub index: u32,
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct GradientDisplayItem {
    pub start_point: LayoutPoint,
    pub end_point: LayoutPoint,
    pub stops: ItemRange,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct GradientStop {
    pub offset: f32,
    pub color: ColorF,
}
known_heap_size!(0, GradientStop);

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct RadialGradientDisplayItem {
    pub start_center: LayoutPoint,
    pub start_radius: f32,
    pub end_center: LayoutPoint,
    pub end_radius: f32,
    pub stops: ItemRange,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct PushStackingContextDisplayItem {
    pub stacking_context: StackingContext,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct PushScrollLayerItem {
    pub content_size: LayoutSize,
    pub id: ScrollLayerId,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct IframeDisplayItem {
    pub pipeline_id: PipelineId,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct IdNamespace(pub u32);

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct ImageDisplayItem {
    pub image_key: ImageKey,
    pub stretch_size: LayoutSize,
    pub tile_spacing: LayoutSize,
    pub image_rendering: ImageRendering,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct YuvImageDisplayItem {
    pub y_image_key: ImageKey,
    pub u_image_key: ImageKey,
    pub v_image_key: ImageKey,
    pub color_space: YuvColorSpace,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum ImageFormat {
    Invalid,
    A8,
    RGB8,
    RGBA8,
    RGBAF32,
}

impl ImageFormat {
    pub fn bytes_per_pixel(self) -> Option<u32> {
        match self {
            ImageFormat::A8 => Some(1),
            ImageFormat::RGB8 => Some(3),
            ImageFormat::RGBA8 => Some(4),
            ImageFormat::RGBAF32 => Some(16),
            ImageFormat::Invalid => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum YuvColorSpace {
    Rec601 = 1, // The values must match the ones in prim_shared.glsl
    Rec709 = 2,
}

/// An arbitrary identifier for an external image provided by the
/// application. It must be a unique identifier for each external
/// image.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ExternalImageId(pub u64);

#[derive(Clone, Serialize, Deserialize)]
pub enum ImageData {
    Raw(Arc<Vec<u8>>),
    External(ExternalImageId),
}

impl ImageData {
    pub fn new(bytes: Vec<u8>) -> ImageData {
        ImageData::Raw(Arc::new(bytes))
    }

    pub fn new_shared(bytes: Arc<Vec<u8>>) -> ImageData {
        ImageData::Raw(bytes)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImageKey(u32, u32);

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum ImageRendering {
    Auto,
    CrispEdges,
    Pixelated,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ItemRange {
    pub start: usize,
    pub length: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
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
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
#[cfg_attr(not(any(target_os = "macos", target_os = "windows")), derive(Clone, Serialize, Deserialize))]
pub struct NativeFontHandle;

#[cfg(target_os = "windows")]
pub type NativeFontHandle = FontDescriptor;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct PipelineId(pub u32, pub u32);

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct RectangleDisplayItem {
    pub color: ColorF,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct RenderApiSender {
    api_sender: MsgSender<ApiMsg>,
    payload_sender: PayloadSender,
}

pub trait RenderNotifier: Send {
    fn new_frame_ready(&mut self);
    fn new_scroll_frame_ready(&mut self, composite_needed: bool);
    fn pipeline_size_changed(&mut self, pipeline_id: PipelineId, size: Option<LayoutSize>);
    fn external_event(&mut self, _evt: ExternalEvent) { unimplemented!() }
    fn shut_down(&mut self) {}
}

// Trait to allow dispatching functions to a specific thread or event loop.
pub trait RenderDispatcher: Send {
    fn dispatch(&self, Box<Fn() + Send>);
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct ResourceId(pub u32);

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub enum ScrollEventPhase {
    /// The user started scrolling.
    Start,
    /// The user performed a scroll. The Boolean flag indicates whether the user's fingers are
    /// down, if a touchpad is in use. (If false, the event is a touchpad fling.)
    Move(bool),
    /// The user ended scrolling.
    End,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ScrollLayerId {
    pub pipeline_id: PipelineId,
    pub info: ScrollLayerInfo,
}

impl ScrollLayerId {
    pub fn root(pipeline_id: PipelineId) -> ScrollLayerId {
        ScrollLayerId {
            pipeline_id: pipeline_id,
            info: ScrollLayerInfo::Scrollable(0, ServoScrollRootId(0)),
        }
    }

    pub fn scroll_root_id(&self) -> Option<ServoScrollRootId> {
        match self.info {
            ScrollLayerInfo::Scrollable(_, scroll_root_id) => Some(scroll_root_id),
            ScrollLayerInfo::Fixed => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum ScrollLayerInfo {
    Fixed,
    Scrollable(usize, ServoScrollRootId)
}

#[derive(Clone, Deserialize, Serialize)]
pub struct ScrollLayerState {
    pub pipeline_id: PipelineId,
    pub scroll_root_id: ServoScrollRootId,
    pub scroll_offset: LayoutPoint,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ScrollPolicy {
    Scrollable,
    Fixed,
}
known_heap_size!(0, ScrollPolicy);

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum ScrollLocation {
    /// Scroll by a certain amount.
    Delta(LayoutPoint),
    /// Scroll to very top of element.
    Start,
    /// Scroll to very bottom of element.
    End
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ServoScrollRootId(pub usize);

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub enum SpecificDisplayItem {
    Rectangle(RectangleDisplayItem),
    Text(TextDisplayItem),
    Image(ImageDisplayItem),
    YuvImage(YuvImageDisplayItem),
    WebGL(WebGLDisplayItem),
    Border(BorderDisplayItem),
    BoxShadow(BoxShadowDisplayItem),
    Gradient(GradientDisplayItem),
    RadialGradient(RadialGradientDisplayItem),
    Iframe(IframeDisplayItem),
    PushStackingContext(PushStackingContextDisplayItem),
    PopStackingContext,
    PushScrollLayer(PushScrollLayerItem),
    PopScrollLayer,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct StackingContext {
    pub scroll_policy: ScrollPolicy,
    pub bounds: LayoutRect,
    pub z_index: i32,
    pub transform: LayoutTransform,
    pub perspective: LayoutTransform,
    pub mix_blend_mode: MixBlendMode,
    pub filters: ItemRange,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct TextDisplayItem {
    pub glyphs: ItemRange,
    pub font_key: FontKey,
    pub size: Au,
    pub color: ColorF,
    pub blur_radius: Au,
}

#[derive(Clone, Deserialize, Serialize)]
pub enum WebGLCommand {
    #[cfg(feature = "webgl")]
    GetContextAttributes(MsgSender<GLContextAttributes>),
    ActiveTexture(u32),
    BlendColor(f32, f32, f32, f32),
    BlendEquation(u32),
    BlendEquationSeparate(u32, u32),
    BlendFunc(u32, u32),
    BlendFuncSeparate(u32, u32, u32, u32),
    AttachShader(WebGLProgramId, WebGLShaderId),
    DetachShader(WebGLProgramId, WebGLShaderId),
    BindAttribLocation(WebGLProgramId, u32, String),
    BufferData(u32, Vec<u8>, u32),
    BufferSubData(u32, isize, Vec<u8>),
    Clear(u32),
    ClearColor(f32, f32, f32, f32),
    ClearDepth(f64),
    ClearStencil(i32),
    ColorMask(bool, bool, bool, bool),
    CullFace(u32),
    FrontFace(u32),
    DepthFunc(u32),
    DepthMask(bool),
    DepthRange(f64, f64),
    Enable(u32),
    Disable(u32),
    CompileShader(WebGLShaderId, String),
    CopyTexImage2D(u32, i32, u32, i32, i32, i32, i32, i32),
    CopyTexSubImage2D(u32, i32, i32, i32, i32, i32, i32, i32),
    CreateBuffer(MsgSender<Option<WebGLBufferId>>),
    CreateFramebuffer(MsgSender<Option<WebGLFramebufferId>>),
    CreateRenderbuffer(MsgSender<Option<WebGLRenderbufferId>>),
    CreateTexture(MsgSender<Option<WebGLTextureId>>),
    CreateProgram(MsgSender<Option<WebGLProgramId>>),
    CreateShader(u32, MsgSender<Option<WebGLShaderId>>),
    DeleteBuffer(WebGLBufferId),
    DeleteFramebuffer(WebGLFramebufferId),
    DeleteRenderbuffer(WebGLRenderbufferId),
    DeleteTexture(WebGLTextureId),
    DeleteProgram(WebGLProgramId),
    DeleteShader(WebGLShaderId),
    BindBuffer(u32, Option<WebGLBufferId>),
    BindFramebuffer(u32, WebGLFramebufferBindingRequest),
    BindRenderbuffer(u32, Option<WebGLRenderbufferId>),
    BindTexture(u32, Option<WebGLTextureId>),
    DisableVertexAttribArray(u32),
    DrawArrays(u32, i32, i32),
    DrawElements(u32, i32, u32, i64),
    EnableVertexAttribArray(u32),
    FramebufferRenderbuffer(u32, u32, u32, Option<WebGLRenderbufferId>),
    FramebufferTexture2D(u32, u32, u32, Option<WebGLTextureId>, i32),
    GetBufferParameter(u32, u32, MsgSender<WebGLResult<WebGLParameter>>),
    GetParameter(u32, MsgSender<WebGLResult<WebGLParameter>>),
    GetProgramParameter(WebGLProgramId, u32, MsgSender<WebGLResult<WebGLParameter>>),
    GetShaderParameter(WebGLShaderId, u32, MsgSender<WebGLResult<WebGLParameter>>),
    GetActiveAttrib(WebGLProgramId, u32, MsgSender<WebGLResult<(i32, u32, String)>>),
    GetActiveUniform(WebGLProgramId, u32, MsgSender<WebGLResult<(i32, u32, String)>>),
    GetAttribLocation(WebGLProgramId, String, MsgSender<Option<i32>>),
    GetUniformLocation(WebGLProgramId, String, MsgSender<Option<i32>>),
    GetVertexAttrib(u32, u32, MsgSender<WebGLResult<WebGLParameter>>),
    GetShaderInfoLog(WebGLShaderId, MsgSender<String>),
    GetProgramInfoLog(WebGLProgramId, MsgSender<String>),
    PolygonOffset(f32, f32),
    RenderbufferStorage(u32, u32, i32, i32),
    ReadPixels(i32, i32, i32, i32, u32, u32, MsgSender<Vec<u8>>),
    SampleCoverage(f32, bool),
    Scissor(i32, i32, i32, i32),
    StencilFunc(u32, i32, u32),
    StencilFuncSeparate(u32, u32, i32, u32),
    StencilMask(u32),
    StencilMaskSeparate(u32, u32),
    StencilOp(u32, u32, u32),
    StencilOpSeparate(u32, u32, u32, u32),
    Hint(u32, u32),
    IsEnabled(u32, MsgSender<bool>),
    LineWidth(f32),
    PixelStorei(u32, i32),
    LinkProgram(WebGLProgramId),
    Uniform1f(i32, f32),
    Uniform1fv(i32, Vec<f32>),
    Uniform1i(i32, i32),
    Uniform1iv(i32, Vec<i32>),
    Uniform2f(i32, f32, f32),
    Uniform2fv(i32, Vec<f32>),
    Uniform2i(i32, i32, i32),
    Uniform2iv(i32, Vec<i32>),
    Uniform3f(i32, f32, f32, f32),
    Uniform3fv(i32, Vec<f32>),
    Uniform3i(i32, i32, i32, i32),
    Uniform3iv(i32, Vec<i32>),
    Uniform4f(i32, f32, f32, f32, f32),
    Uniform4fv(i32, Vec<f32>),
    Uniform4i(i32, i32, i32, i32, i32),
    Uniform4iv(i32, Vec<i32>),
    UniformMatrix2fv(i32, bool, Vec<f32>),
    UniformMatrix3fv(i32, bool, Vec<f32>),
    UniformMatrix4fv(i32, bool, Vec<f32>),
    UseProgram(WebGLProgramId),
    ValidateProgram(WebGLProgramId),
    VertexAttrib(u32, f32, f32, f32, f32),
    VertexAttribPointer(u32, i32, u32, bool, i32, u32),
    VertexAttribPointer2f(u32, i32, bool, i32, u32),
    Viewport(i32, i32, i32, i32),
    TexImage2D(u32, i32, i32, i32, i32, u32, u32, Vec<u8>),
    TexParameteri(u32, u32, i32),
    TexParameterf(u32, u32, f32),
    TexSubImage2D(u32, i32, i32, i32, i32, i32, u32, u32, Vec<u8>),
    DrawingBufferWidth(MsgSender<i32>),
    DrawingBufferHeight(MsgSender<i32>),
    Finish(MsgSender<()>),
    Flush,
    GenerateMipmap(u32),
}

#[cfg(feature = "nightly")]
macro_rules! define_resource_id_struct {
    ($name:ident) => {
        #[derive(Clone, Copy, PartialEq)]
        pub struct $name(NonZero<u32>);

        impl $name {
            #[inline]
            unsafe fn new(id: u32) -> Self {
                $name(NonZero::new(id))
            }

            #[inline]
            fn get(self) -> u32 {
                *self.0
            }
        }

    };
}

#[cfg(not(feature = "nightly"))]
macro_rules! define_resource_id_struct {
    ($name:ident) => {
        #[derive(Clone, Copy, PartialEq)]
        pub struct $name(u32);

        impl $name {
            #[inline]
            unsafe fn new(id: u32) -> Self {
                $name(id)
            }

            #[inline]
            fn get(self) -> u32 {
                self.0
            }
        }
    };
}

macro_rules! define_resource_id {
    ($name:ident) => {
        define_resource_id_struct!($name);

        impl ::serde::Deserialize for $name {
            fn deserialize<D>(deserializer: &mut D) -> Result<Self, D::Error>
                where D: ::serde::Deserializer
            {
                let id = try!(u32::deserialize(deserializer));
                if id == 0 {
                    Err(::serde::Error::invalid_value("expected a non-zero value"))
                } else {
                    Ok(unsafe { $name::new(id) })
                }
            }
        }

        impl ::serde::Serialize for $name {
            fn serialize<S>(&self, serializer: &mut S) -> Result<(), S::Error>
                where S: ::serde::Serializer
            {
                self.get().serialize(serializer)
            }
        }

        impl ::std::fmt::Debug for $name {
            fn fmt(&self, fmt: &mut ::std::fmt::Formatter)
                  -> Result<(), ::std::fmt::Error> {
                fmt.debug_tuple(stringify!($name))
                   .field(&self.get())
                   .finish()
            }
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, fmt: &mut ::std::fmt::Formatter)
                  -> Result<(), ::std::fmt::Error> {
                write!(fmt, "{}", self.get())
            }
        }

        impl ::heapsize::HeapSizeOf for $name {
            fn heap_size_of_children(&self) -> usize { 0 }
        }
    }
}

define_resource_id!(WebGLBufferId);
define_resource_id!(WebGLFramebufferId);
define_resource_id!(WebGLRenderbufferId);
define_resource_id!(WebGLTextureId);
define_resource_id!(WebGLProgramId);
define_resource_id!(WebGLShaderId);

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct WebGLContextId(pub usize);

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct WebGLDisplayItem {
    pub context_id: WebGLContextId,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub enum WebGLError {
    InvalidEnum,
    InvalidFramebufferOperation,
    InvalidOperation,
    InvalidValue,
    OutOfMemory,
    ContextLost,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum WebGLFramebufferBindingRequest {
    Explicit(WebGLFramebufferId),
    Default,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum WebGLParameter {
    Int(i32),
    Bool(bool),
    String(String),
    Float(f32),
    FloatArray(Vec<f32>),
    Invalid,
}

pub type WebGLResult<T> = Result<T, WebGLError>;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum WebGLShaderParameter {
    Int(i32),
    Bool(bool),
    Invalid,
}

pub type VRCompositorId = u64;

// WebVR commands that must be called in the WebGL render thread.
#[derive(Clone, Deserialize, Serialize)]
pub enum VRCompositorCommand {
    Create(VRCompositorId),
    SyncPoses(VRCompositorId, f64, f64, MsgSender<Result<Vec<u8>,()>>),
    SubmitFrame(VRCompositorId, [f32; 4], [f32; 4]),
    Release(VRCompositorId)
}

// Trait object that handles WebVR commands.
// Receives the texture_id associated to the WebGLContext.
pub trait VRCompositorHandler: Send {
    fn handle(&mut self, command: VRCompositorCommand, texture_id: Option<u32>);
}
