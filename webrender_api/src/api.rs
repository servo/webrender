/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use channel::{self, MsgSender, Payload, PayloadSenderHelperMethods, PayloadSender};
#[cfg(feature = "webgl")]
use offscreen_gl_context::{GLContextAttributes, GLLimits};
use std::cell::Cell;
use std::fmt;
use std::marker::PhantomData;
use {BuiltDisplayList, BuiltDisplayListDescriptor, ClipId, ColorF, DeviceIntPoint, DeviceIntSize};
use {DeviceUintRect, DeviceUintSize, FontKey, GlyphDimensions, GlyphKey};
use {ImageData, ImageDescriptor, ImageKey, LayoutPoint, LayoutVector2D, LayoutSize, LayoutTransform};
use {FontInstanceKey, NativeFontHandle, WorldPoint};
#[cfg(feature = "webgl")]
use {WebGLCommand, WebGLContextId};

pub type TileSize = u16;

/// The resource updates for a given transaction (they must be applied in the same frame).
#[derive(Clone, Deserialize, Serialize)]
pub struct ResourceUpdates {
    pub added_images: Vec<AddImageMsg>,
    pub updated_images: Vec<UpdateImageMsg>,
    pub deleted_images: Vec<ImageKey>,
}

impl ResourceUpdates {
    pub fn new() -> Self {
        ResourceUpdates {
            added_images: Vec::new(),
            updated_images: Vec::new(),
            deleted_images: Vec::new(),
        }
    }

    pub fn add_image(
        &mut self,
        key: ImageKey,
        descriptor: ImageDescriptor,
        data: ImageData,
        tiling: Option<TileSize>
    ) {
        self.added_images.push(AddImageMsg { key, descriptor, data, tiling });
    }

    pub fn update_image(
        &mut self,
        key: ImageKey,
        descriptor: ImageDescriptor,
        data: ImageData,
        dirty_rect: Option<DeviceUintRect>
    ) {
        self.updated_images.push(UpdateImageMsg { key, descriptor, data, dirty_rect });
    }

    pub fn delete_image(&mut self, key: ImageKey) {
        self.deleted_images.push(key);
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct AddImageMsg {
    pub key: ImageKey,
    pub descriptor: ImageDescriptor,
    pub data: ImageData,
    pub tiling: Option<TileSize>,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct UpdateImageMsg {
    pub key: ImageKey,
    pub descriptor: ImageDescriptor,
    pub data: ImageData,
    pub dirty_rect: Option<DeviceUintRect>,
}

#[derive(Clone, Deserialize, Serialize)]
pub enum DocumentMsg {
    SetDisplayList {
        list_descriptor: BuiltDisplayListDescriptor,
        epoch: Epoch,
        pipeline_id: PipelineId,
        background: Option<ColorF>,
        viewport_size: LayoutSize,
        content_size: LayoutSize,
        preserve_frame_state: bool,
        resources: ResourceUpdates,
    },
    SetPageZoom(ZoomFactor),
    SetPinchZoom(ZoomFactor),
    SetPan(DeviceIntPoint),
    SetRootPipeline(PipelineId),
    SetWindowParameters {
        window_size: DeviceUintSize,
        inner_rect: DeviceUintRect,
    },
    Scroll(ScrollLocation, WorldPoint, ScrollEventPhase),
    ScrollNodeWithId(LayoutPoint, ClipId, ScrollClamping),
    TickScrollingBounce,
    GetScrollNodeState(MsgSender<Vec<ScrollLayerState>>),
    GenerateFrame(Option<DynamicProperties>),
}

impl fmt::Debug for DocumentMsg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match *self {
            DocumentMsg::SetDisplayList{..} => "DocumentMsg::SetDisplayList",
            DocumentMsg::SetPageZoom(..) => "DocumentMsg::SetPageZoom",
            DocumentMsg::SetPinchZoom(..) => "DocumentMsg::SetPinchZoom",
            DocumentMsg::SetPan(..) => "DocumentMsg::SetPan",
            DocumentMsg::SetRootPipeline(..) => "DocumentMsg::SetRootPipeline",
            DocumentMsg::SetWindowParameters{..} => "DocumentMsg::SetWindowParameters",
            DocumentMsg::Scroll(..) => "DocumentMsg::Scroll",
            DocumentMsg::ScrollNodeWithId(..) => "DocumentMsg::ScrollNodeWithId",
            DocumentMsg::TickScrollingBounce => "DocumentMsg::TickScrollingBounce",
            DocumentMsg::GetScrollNodeState(..) => "DocumentMsg::GetScrollNodeState",
            DocumentMsg::GenerateFrame(..) => "DocumentMsg::GenerateFrame",
        })
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub enum ApiMsg {
    AddRawFont(FontKey, Vec<u8>, u32),
    AddNativeFont(FontKey, NativeFontHandle),
    DeleteFont(FontKey),
    UpdateResources(ResourceUpdates),
    /// Gets the glyph dimensions
    GetGlyphDimensions(FontInstanceKey, Vec<GlyphKey>, MsgSender<Vec<Option<GlyphDimensions>>>),
    /// Gets the glyph indices from a string
    GetGlyphIndices(FontKey, String, MsgSender<Vec<Option<u32>>>),
    /// Adds a new document namespace.
    CloneApi(MsgSender<IdNamespace>),
    /// Adds a new document with given initial size.
    AddDocument(DocumentId, DeviceUintSize),
    /// A message targeted at a particular document.
    UpdateDocument(DocumentId, DocumentMsg),
    /// Deletes an existing document.
    DeleteDocument(DocumentId),
    RequestWebGLContext(DeviceIntSize, GLContextAttributes, MsgSender<Result<(WebGLContextId, GLLimits), String>>),
    ResizeWebGLContext(WebGLContextId, DeviceIntSize),
    WebGLCommand(WebGLContextId, WebGLCommand),
    // WebVR commands that must be called in the WebGL render thread.
    VRCompositorCommand(WebGLContextId, VRCompositorCommand),
    /// An opaque handle that must be passed to the render notifier. It is used by Gecko
    /// to forward gecko-specific messages to the render thread preserving the ordering
    /// within the other messages.
    ExternalEvent(ExternalEvent),
    /// Removes all resources associated with a namespace.
    ClearNamespace(IdNamespace),
    ShutDown,
}

impl fmt::Debug for ApiMsg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match *self {
            ApiMsg::UpdateResources(..) => "ApiMsg::UpdateResources",
            ApiMsg::AddRawFont(..) => "ApiMsg::AddRawFont",
            ApiMsg::AddNativeFont(..) => "ApiMsg::AddNativeFont",
            ApiMsg::DeleteFont(..) => "ApiMsg::DeleteFont",
            ApiMsg::GetGlyphDimensions(..) => "ApiMsg::GetGlyphDimensions",
            ApiMsg::GetGlyphIndices(..) => "ApiMsg::GetGlyphIndices",
            ApiMsg::CloneApi(..) => "ApiMsg::CloneApi",
            ApiMsg::AddDocument(..) => "ApiMsg::AddDocument",
            ApiMsg::UpdateDocument(..) => "ApiMsg::UpdateDocument",
            ApiMsg::DeleteDocument(..) => "ApiMsg::DeleteDocument",
            ApiMsg::RequestWebGLContext(..) => "ApiMsg::RequestWebGLContext",
            ApiMsg::ResizeWebGLContext(..) => "ApiMsg::ResizeWebGLContext",
            ApiMsg::WebGLCommand(..) => "ApiMsg::WebGLCommand",
            ApiMsg::VRCompositorCommand(..) => "ApiMsg::VRCompositorCommand",
            ApiMsg::ExternalEvent(..) => "ApiMsg::ExternalEvent",
            ApiMsg::ClearNamespace(..) => "ApiMsg::ClearNamespace",
            ApiMsg::ShutDown => "ApiMsg::ShutDown",
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Epoch(pub u32);

#[cfg(not(feature = "webgl"))]
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct WebGLContextId(pub usize);

#[cfg(not(feature = "webgl"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GLContextAttributes([u8; 0]);

#[cfg(not(feature = "webgl"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GLLimits([u8; 0]);

#[cfg(not(feature = "webgl"))]
#[derive(Clone, Deserialize, Serialize)]
pub enum WebGLCommand {
    Flush,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Deserialize, Serialize)]
pub struct IdNamespace(pub u32);

#[repr(C)]
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DocumentId(pub IdNamespace, pub u32);

/// This type carries no valuable semantics for WR. However, it reflects the fact that
/// clients (Servo) may generate pipelines by different semi-independent sources.
/// These pipelines still belong to the same `IdNamespace` and the same `DocumentId`.
/// Having this extra Id field enables them to generate `PipelineId` without collision.
pub type PipelineSourceId = u32;

/// From the point of view of WR, `PipelineId` is completely opaque and generic as long as
/// it's clonable, serializable, comparable, and hashable.
#[repr(C)]
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct PipelineId(pub PipelineSourceId, pub u32);

impl PipelineId {
    pub fn dummy() -> Self {
        PipelineId(0, 0)
    }
}


#[repr(C)]
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct ResourceId(pub u32);

/// An opaque pointer-sized value.
#[repr(C)]
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

#[derive(Clone, Deserialize, Serialize)]
pub enum ScrollClamping {
    ToContentBounds,
    NoClamping,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct RenderApiSender {
    api_sender: MsgSender<ApiMsg>,
    payload_sender: PayloadSender,
}

impl RenderApiSender {
    pub fn new(api_sender: MsgSender<ApiMsg>,
               payload_sender: PayloadSender)
               -> Self {
        RenderApiSender {
            api_sender,
            payload_sender,
        }
    }

    /// Creates a new resource API object with a dedicated namespace.
    pub fn create_api(&self) -> RenderApi {
        let (sync_tx, sync_rx) = channel::msg_channel().unwrap();
        let msg = ApiMsg::CloneApi(sync_tx);
        self.api_sender.send(msg).unwrap();
        RenderApi {
            api_sender: self.api_sender.clone(),
            payload_sender: self.payload_sender.clone(),
            namespace_id: sync_rx.recv().unwrap(),
            next_id: Cell::new(ResourceId(0)),
        }
    }
}

pub struct RenderApi {
    api_sender: MsgSender<ApiMsg>,
    payload_sender: PayloadSender,
    namespace_id: IdNamespace,
    next_id: Cell<ResourceId>,
}

impl RenderApi {
    pub fn clone_sender(&self) -> RenderApiSender {
        RenderApiSender::new(self.api_sender.clone(), self.payload_sender.clone())
    }

    pub fn add_document(&self, initial_size: DeviceUintSize) -> DocumentId {
        let new_id = self.next_unique_id();
        let document_id = DocumentId(self.namespace_id, new_id);

        let msg = ApiMsg::AddDocument(document_id, initial_size);
        self.api_sender.send(msg).unwrap();

        document_id
    }

    pub fn delete_document(&self, document_id: DocumentId) {
        let msg = ApiMsg::DeleteDocument(document_id);
        self.api_sender.send(msg).unwrap();
    }

    pub fn generate_font_key(&self) -> FontKey {
        let new_id = self.next_unique_id();
        FontKey::new(self.namespace_id, new_id)
    }

    pub fn add_raw_font(&self, key: FontKey, bytes: Vec<u8>, index: u32) {
        debug_assert_eq!(key.0, self.namespace_id);
        let msg = ApiMsg::AddRawFont(key, bytes, index);
        self.api_sender.send(msg).unwrap();
    }

    pub fn add_native_font(&self, key: FontKey, native_font_handle: NativeFontHandle) {
        debug_assert_eq!(key.0, self.namespace_id);
        let msg = ApiMsg::AddNativeFont(key, native_font_handle);
        self.api_sender.send(msg).unwrap();
    }

    pub fn delete_font(&self, key: FontKey) {
        debug_assert_eq!(key.0, self.namespace_id);
        let msg = ApiMsg::DeleteFont(key);
        self.api_sender.send(msg).unwrap();
    }

    /// Gets the dimensions for the supplied glyph keys
    ///
    /// Note: Internally, the internal texture cache doesn't store
    /// 'empty' textures (height or width = 0)
    /// This means that glyph dimensions e.g. for spaces (' ') will mostly be None.
    pub fn get_glyph_dimensions(&self,
                                font: FontInstanceKey,
                                glyph_keys: Vec<GlyphKey>)
                                -> Vec<Option<GlyphDimensions>> {
        let (tx, rx) = channel::msg_channel().unwrap();
        let msg = ApiMsg::GetGlyphDimensions(font, glyph_keys, tx);
        self.api_sender.send(msg).unwrap();
        rx.recv().unwrap()
    }

    /// Gets the glyph indices for the supplied string. These
    /// can be used to construct GlyphKeys.
    pub fn get_glyph_indices(&self,
                             font_key: FontKey,
                             text: &str) -> Vec<Option<u32>> {
        let (tx, rx) = channel::msg_channel().unwrap();
        let msg = ApiMsg::GetGlyphIndices(font_key, text.to_string(), tx);
        self.api_sender.send(msg).unwrap();
        rx.recv().unwrap()
    }

    /// Creates an `ImageKey`.
    pub fn generate_image_key(&self) -> ImageKey {
        let new_id = self.next_unique_id();
        ImageKey::new(self.namespace_id, new_id)
    }

    /// Adds an image identified by the `ImageKey`.
    pub fn update_resources(&self, resources: ResourceUpdates) {
        self.api_sender.send(ApiMsg::UpdateResources(resources)).unwrap();
    }

    pub fn request_webgl_context(&self, size: &DeviceIntSize, attributes: GLContextAttributes)
                                 -> Result<(WebGLContextId, GLLimits), String> {
        let (tx, rx) = channel::msg_channel().unwrap();
        let msg = ApiMsg::RequestWebGLContext(*size, attributes, tx);
        self.api_sender.send(msg).unwrap();
        rx.recv().unwrap()
    }

    pub fn resize_webgl_context(&self, context_id: WebGLContextId, size: &DeviceIntSize) {
        let msg = ApiMsg::ResizeWebGLContext(context_id, *size);
        self.api_sender.send(msg).unwrap();
    }

    pub fn send_webgl_command(&self, context_id: WebGLContextId, command: WebGLCommand) {
        let msg = ApiMsg::WebGLCommand(context_id, command);
        self.api_sender.send(msg).unwrap();
    }

    pub fn send_vr_compositor_command(&self, context_id: WebGLContextId, command: VRCompositorCommand) {
        let msg = ApiMsg::VRCompositorCommand(context_id, command);
        self.api_sender.send(msg).unwrap();
    }

    pub fn send_external_event(&self, evt: ExternalEvent) {
        let msg = ApiMsg::ExternalEvent(evt);
        self.api_sender.send(msg).unwrap();
    }

    pub fn shut_down(&self) {
        self.api_sender.send(ApiMsg::ShutDown).unwrap();
    }

    /// Create a new unique key that can be used for
    /// animated property bindings.
    pub fn generate_property_binding_key<T: Copy>(&self) -> PropertyBindingKey<T> {
        let new_id = self.next_unique_id();
        PropertyBindingKey {
            id: PropertyBindingId {
                namespace: self.namespace_id,
                uid: new_id,
            },
            _phantom: PhantomData,
        }
    }

    #[inline]
    fn next_unique_id(&self) -> u32 {
        let ResourceId(id) = self.next_id.get();
        self.next_id.set(ResourceId(id + 1));
        id
    }

    // For use in Wrench only
    #[doc(hidden)]
    pub fn send_message(&self, msg: ApiMsg) {
        self.api_sender.send(msg).unwrap();
    }

    // For use in Wrench only
    #[doc(hidden)]
    pub fn send_payload(&self, data: &[u8]) {
        self.payload_sender.send_payload(Payload::from_data(data)).unwrap();
    }

    /// A helper method to send document messages.
    fn send(&self, document_id: DocumentId, msg: DocumentMsg) {
        // This assertion fails on Servo use-cases, because it creates different
        // `RenderApi` instances for layout and compositor.
        //assert_eq!(document_id.0, self.namespace_id);
        self.api_sender.send(ApiMsg::UpdateDocument(document_id, msg)).unwrap()
    }

        /// Sets the root pipeline.
    ///
    /// # Examples
    ///
    /// ```
    /// # use webrender_api::{DeviceUintSize, PipelineId, RenderApiSender};
    /// # fn example(sender: RenderApiSender) {
    /// let api = sender.create_api();
    /// let document_id = api.add_document(DeviceUintSize::zero());
    /// let pipeline_id = PipelineId(0, 0);
    /// api.set_root_pipeline(document_id, pipeline_id);
    /// # }
    /// ```
    pub fn set_root_pipeline(&self, document_id: DocumentId, pipeline_id: PipelineId) {
        self.send(document_id, DocumentMsg::SetRootPipeline(pipeline_id));
    }

    /// Supplies a new frame to WebRender.
    ///
    /// Non-blocking, it notifies a worker process which processes the display list.
    /// When it's done and a RenderNotifier has been set in `webrender::renderer::Renderer`,
    /// [new_frame_ready()][notifier] gets called.
    ///
    /// Note: Scrolling doesn't require an own Frame.
    ///
    /// Arguments:
    ///
    /// * `document_id`: Target Document ID.
    /// * `epoch`: The unique Frame ID, monotonically increasing.
    /// * `background`: The background color of this pipeline.
    /// * `viewport_size`: The size of the viewport for this frame.
    /// * `pipeline_id`: The ID of the pipeline that is supplying this display list.
    /// * `content_size`: The total screen space size of this display list's display items.
    /// * `display_list`: The root Display list used in this frame.
    /// * `preserve_frame_state`: If a previous frame exists which matches this pipeline
    ///                           id, this setting determines if frame state (such as scrolling
    ///                           position) should be preserved for this new display list.
    ///
    /// [notifier]: trait.RenderNotifier.html#tymethod.new_frame_ready
    pub fn set_display_list(
        &self,
        document_id: DocumentId,
        epoch: Epoch,
        background: Option<ColorF>,
        viewport_size: LayoutSize,
        (pipeline_id, content_size, display_list): (PipelineId, LayoutSize, BuiltDisplayList),
        preserve_frame_state: bool,
        resources: ResourceUpdates,
    ) {
        let (display_list_data, list_descriptor) = display_list.into_data();
        self.send(document_id, DocumentMsg::SetDisplayList {
            epoch,
            pipeline_id,
            background,
            viewport_size,
            content_size,
            list_descriptor,
            preserve_frame_state,
            resources,
        });

        self.payload_sender.send_payload(Payload {
            epoch,
            pipeline_id,
            display_list_data,
        }).unwrap();
    }

    /// Scrolls the scrolling layer under the `cursor`
    ///
    /// WebRender looks for the layer closest to the user
    /// which has `ScrollPolicy::Scrollable` set.
    pub fn scroll(&self, document_id: DocumentId, scroll_location: ScrollLocation,
                  cursor: WorldPoint, phase: ScrollEventPhase) {
        self.send(document_id, DocumentMsg::Scroll(scroll_location, cursor, phase));
    }

    pub fn scroll_node_with_id(&self, document_id: DocumentId, origin: LayoutPoint,
                               id: ClipId, clamp: ScrollClamping) {
        self.send(document_id, DocumentMsg::ScrollNodeWithId(origin, id, clamp));
    }

    pub fn set_page_zoom(&self, document_id: DocumentId, page_zoom: ZoomFactor) {
        self.send(document_id, DocumentMsg::SetPageZoom(page_zoom));
    }

    pub fn set_pinch_zoom(&self, document_id: DocumentId, pinch_zoom: ZoomFactor) {
        self.send(document_id, DocumentMsg::SetPinchZoom(pinch_zoom));
    }

    pub fn set_pan(&self, document_id: DocumentId, pan: DeviceIntPoint) {
        self.send(document_id, DocumentMsg::SetPan(pan));
    }

    pub fn set_window_parameters(&self,
                                 document_id: DocumentId,
                                 window_size: DeviceUintSize,
                                 inner_rect: DeviceUintRect) {
        self.send(document_id, DocumentMsg::SetWindowParameters {
            window_size,
            inner_rect,
        });
    }

    pub fn tick_scrolling_bounce_animations(&self, document_id: DocumentId) {
        self.send(document_id, DocumentMsg::TickScrollingBounce);
    }

    pub fn get_scroll_node_state(&self, document_id: DocumentId) -> Vec<ScrollLayerState> {
        let (tx, rx) = channel::msg_channel().unwrap();
        self.send(document_id, DocumentMsg::GetScrollNodeState(tx));
        rx.recv().unwrap()
    }

    /// Generate a new frame. Optionally, supply a list of animated
    /// property bindings that should be used to resolve bindings
    /// in the current display list.
    pub fn generate_frame(&self, document_id: DocumentId,
                          property_bindings: Option<DynamicProperties>) {
        self.send(document_id, DocumentMsg::GenerateFrame(property_bindings));
    }
}

impl Drop for RenderApi {
    fn drop(&mut self) {
        let msg = ApiMsg::ClearNamespace(self.namespace_id);
        let _ = self.api_sender.send(msg);
    }
}

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

#[derive(Clone, Deserialize, Serialize)]
pub struct ScrollLayerState {
    pub id: ClipId,
    pub scroll_offset: LayoutVector2D,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum ScrollLocation {
    /// Scroll by a certain amount.
    Delta(LayoutVector2D),
    /// Scroll to very top of element.
    Start,
    /// Scroll to very bottom of element.
    End
}

/// Represents a zoom factor.
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct ZoomFactor(f32);

impl ZoomFactor {
    /// Construct a new zoom factor.
    pub fn new(scale: f32) -> ZoomFactor {
        ZoomFactor(scale)
    }

    /// Get the zoom factor as an untyped float.
    pub fn get(&self) -> f32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize, Eq, Hash)]
pub struct PropertyBindingId {
    namespace: IdNamespace,
    uid: u32,
}

impl PropertyBindingId {
    pub fn new(value: u64) -> Self {
        PropertyBindingId {
            namespace: IdNamespace((value>>32) as u32),
            uid: value as u32,
        }
    }
}

/// A unique key that is used for connecting animated property
/// values to bindings in the display list.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct PropertyBindingKey<T> {
    pub id: PropertyBindingId,
    _phantom: PhantomData<T>,
}

/// Construct a property value from a given key and value.
impl<T: Copy> PropertyBindingKey<T> {
    pub fn with(&self, value: T) -> PropertyValue<T> {
        PropertyValue {
            key: *self,
            value,
        }
    }
}

impl<T> PropertyBindingKey<T> {
    pub fn new(value: u64) -> Self {
        PropertyBindingKey {
            id: PropertyBindingId::new(value),
            _phantom: PhantomData,
        }
    }
}

/// A binding property can either be a specific value
/// (the normal, non-animated case) or point to a binding location
/// to fetch the current value from.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub enum PropertyBinding<T> {
    Value(T),
    Binding(PropertyBindingKey<T>),
}

impl<T> From<T> for PropertyBinding<T> {
    fn from(value: T) -> PropertyBinding<T> {
        PropertyBinding::Value(value)
    }
}

impl<T> From<PropertyBindingKey<T>> for PropertyBinding<T> {
    fn from(key: PropertyBindingKey<T>) -> PropertyBinding<T> {
        PropertyBinding::Binding(key)
    }
}

/// The current value of an animated property. This is
/// supplied by the calling code.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct PropertyValue<T> {
    pub key: PropertyBindingKey<T>,
    pub value: T,
}

/// When using `generate_frame()`, a list of `PropertyValue` structures
/// can optionally be supplied to provide the current value of any
/// animated properties.
#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct DynamicProperties {
    pub transforms: Vec<PropertyValue<LayoutTransform>>,
    pub floats: Vec<PropertyValue<f32>>,
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
// Receives the texture id and size associated to the WebGLContext.
pub trait VRCompositorHandler: Send {
    fn handle(&mut self, command: VRCompositorCommand, texture: Option<(u32, DeviceIntSize)>);
}

pub trait RenderNotifier: Send {
    fn new_frame_ready(&mut self);
    fn new_scroll_frame_ready(&mut self, composite_needed: bool);
    fn external_event(&mut self, _evt: ExternalEvent) { unimplemented!() }
    fn shut_down(&mut self) {}
}

/// Trait to allow dispatching functions to a specific thread or event loop.
pub trait RenderDispatcher: Send {
    fn dispatch(&self, Box<Fn() + Send>);
}
