/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use byteorder::{LittleEndian, WriteBytesExt};
use display_list::{AuxiliaryLists, AuxiliaryListsDescriptor, BuiltDisplayList};
use display_list::{BuiltDisplayListDescriptor};
use euclid::{Point2D, Size2D};
use ipc_channel::ipc::{self, IpcBytesSender, IpcSender};
use offscreen_gl_context::{GLContextAttributes, GLLimits};
use stacking_context::StackingContext;
use std::cell::Cell;
use types::{ColorF, DisplayListId, Epoch, FontKey, StackingContextId};
use types::{ImageKey, ImageFormat, NativeFontHandle, PipelineId};
use webgl::{WebGLContextId, WebGLCommand};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct IdNamespace(pub u32);

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ResourceId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ScrollEventPhase {
    /// The user started scrolling.
    Start,
    /// The user performed a scroll. The Boolean flag indicates whether the user's fingers are
    /// down, if a touchpad is in use. (If false, the event is a touchpad fling.)
    Move(bool),
    /// The user ended scrolling.
    End,
}

#[derive(Serialize, Deserialize)]
pub enum ApiMsg {
    AddRawFont(FontKey, Vec<u8>),
    AddNativeFont(FontKey, NativeFontHandle),
    AddImage(ImageKey, u32, u32, ImageFormat, Vec<u8>),
    UpdateImage(ImageKey, u32, u32, ImageFormat, Vec<u8>),
    CloneApi(IpcSender<IdNamespace>),
    /// Supplies a new frame to WebRender.
    ///
    /// The first `StackingContextId` describes the root stacking context. The actual stacking
    /// contexts are supplied as the sixth parameter, while the display lists that make up those
    /// stacking contexts are supplied as the seventh parameter.
    ///
    /// After receiving this message, WebRender will read the display lists, followed by the
    /// auxiliary lists, from the payload channel.
    SetRootStackingContext(StackingContextId,
                           ColorF,
                           Epoch,
                           PipelineId,
                           Size2D<f32>,
                           Vec<(StackingContextId, StackingContext)>,
                           Vec<(DisplayListId, BuiltDisplayListDescriptor)>,
                           AuxiliaryListsDescriptor),
    SetRootPipeline(PipelineId),
    Scroll(Point2D<f32>, Point2D<f32>, ScrollEventPhase),
    TickScrollingBounce,
    TranslatePointToLayerSpace(Point2D<f32>, IpcSender<(Point2D<f32>, PipelineId)>),
    RequestWebGLContext(Size2D<i32>, GLContextAttributes, IpcSender<Result<(WebGLContextId, GLLimits), String>>),
    WebGLCommand(WebGLContextId, WebGLCommand),
}

#[derive(Serialize, Deserialize, Clone)]
pub struct RenderApiSender {
    api_sender: IpcSender<ApiMsg>,
    payload_sender: IpcBytesSender,
}

impl RenderApiSender {
    pub fn new(api_sender: IpcSender<ApiMsg>, payload_sender: IpcBytesSender) -> RenderApiSender {
        RenderApiSender {
            api_sender: api_sender,
            payload_sender: payload_sender,
        }
    }

    pub fn create_api(&self) -> RenderApi {
        let RenderApiSender {
            ref api_sender,
            ref payload_sender
        } = *self;
        let (sync_tx, sync_rx) = ipc::channel().unwrap();
        let msg = ApiMsg::CloneApi(sync_tx);
        api_sender.send(msg).unwrap();
        RenderApi {
            api_sender: api_sender.clone(),
            payload_sender: payload_sender.clone(),
            id_namespace: sync_rx.recv().unwrap(),
            next_id: Cell::new(ResourceId(0)),
        }
    }
}

pub struct RenderApi {
    pub api_sender: IpcSender<ApiMsg>,
    pub payload_sender: IpcBytesSender,
    pub id_namespace: IdNamespace,
    pub next_id: Cell<ResourceId>,
}

impl RenderApi {
    pub fn clone_sender(&self) -> RenderApiSender {
        RenderApiSender::new(self.api_sender.clone(), self.payload_sender.clone())
    }

    pub fn add_raw_font(&self, bytes: Vec<u8>) -> FontKey {
        let new_id = self.next_unique_id();
        let key = FontKey::new(new_id.0, new_id.1);
        let msg = ApiMsg::AddRawFont(key, bytes);
        self.api_sender.send(msg).unwrap();
        key
    }

    pub fn add_native_font(&self, native_font_handle: NativeFontHandle) -> FontKey {
        let new_id = self.next_unique_id();
        let key = FontKey::new(new_id.0, new_id.1);
        let msg = ApiMsg::AddNativeFont(key, native_font_handle);
        self.api_sender.send(msg).unwrap();
        key
    }

    pub fn alloc_image(&self) -> ImageKey {
        let new_id = self.next_unique_id();
        ImageKey::new(new_id.0, new_id.1)
    }

    pub fn add_image(&self,
                     width: u32,
                     height: u32,
                     format: ImageFormat,
                     bytes: Vec<u8>) -> ImageKey {
        let new_id = self.next_unique_id();
        let key = ImageKey::new(new_id.0, new_id.1);
        let msg = ApiMsg::AddImage(key, width, height, format, bytes);
        self.api_sender.send(msg).unwrap();
        key
    }

    // TODO: Support changing dimensions (and format) during image update?
    pub fn update_image(&self,
                        key: ImageKey,
                        width: u32,
                        height: u32,
                        format: ImageFormat,
                        bytes: Vec<u8>) {
        let msg = ApiMsg::UpdateImage(key, width, height, format, bytes);
        self.api_sender.send(msg).unwrap();
    }

    pub fn set_root_pipeline(&self, pipeline_id: PipelineId) {
        let msg = ApiMsg::SetRootPipeline(pipeline_id);
        self.api_sender.send(msg).unwrap();
    }

    /// Supplies a new frame to WebRender.
    ///
    /// Arguments:
    /// * `stacking_context_id`: The ID of the root stacking context.
    /// * `background_color`: The background color of this pipeline.
    /// * `epoch`: A monotonically increasing timestamp.
    /// * `pipeline_id`: The ID of the pipeline that is supplying this display list.
    /// * `viewport_size`: The size of the viewport for this frame.
    /// * `stacking_contexts`: Stacking contexts used in this frame.
    /// * `display_lists`: Display lists used in this frame.
    /// * `auxiliary_lists`: Various items that the display lists and stacking contexts reference.
    pub fn set_root_stacking_context(&self,
                                     stacking_context_id: StackingContextId,
                                     background_color: ColorF,
                                     epoch: Epoch,
                                     pipeline_id: PipelineId,
                                     viewport_size: Size2D<f32>,
                                     stacking_contexts: Vec<(StackingContextId, StackingContext)>,
                                     display_lists: Vec<(DisplayListId, BuiltDisplayList)>,
                                     auxiliary_lists: AuxiliaryLists) {
        let display_list_descriptors = display_lists.iter().map(|&(display_list_id,
                                                                   ref built_display_list)| {
            (display_list_id, (*built_display_list.descriptor()).clone())
        }).collect();
        let msg = ApiMsg::SetRootStackingContext(stacking_context_id,
                                                 background_color,
                                                 epoch,
                                                 pipeline_id,
                                                 viewport_size,
                                                 stacking_contexts,
                                                 display_list_descriptors,
                                                 *auxiliary_lists.descriptor());
        self.api_sender.send(msg).unwrap();

        let mut payload = vec![];
        payload.write_u32::<LittleEndian>(stacking_context_id.0).unwrap();
        payload.write_u32::<LittleEndian>(epoch.0).unwrap();

        for &(_, ref built_display_list) in &display_lists {
            payload.extend_from_slice(built_display_list.data());
        }
        payload.extend_from_slice(auxiliary_lists.data());

        self.payload_sender.send(&payload[..]).unwrap();
    }

    pub fn scroll(&self, delta: Point2D<f32>, cursor: Point2D<f32>, phase: ScrollEventPhase) {
        let msg = ApiMsg::Scroll(delta, cursor, phase);
        self.api_sender.send(msg).unwrap();
    }

    pub fn tick_scrolling_bounce_animations(&self) {
        let msg = ApiMsg::TickScrollingBounce;
        self.api_sender.send(msg).unwrap();
    }

    pub fn translate_point_to_layer_space(&self, point: &Point2D<f32>)
                                          -> (Point2D<f32>, PipelineId) {
        let (tx, rx) = ipc::channel().unwrap();
        let msg = ApiMsg::TranslatePointToLayerSpace(*point, tx);
        self.api_sender.send(msg).unwrap();
        rx.recv().unwrap()
    }

    pub fn request_webgl_context(&self, size: &Size2D<i32>, attributes: GLContextAttributes)
                                 -> Result<(WebGLContextId, GLLimits), String> {
        let (tx, rx) = ipc::channel().unwrap();
        let msg = ApiMsg::RequestWebGLContext(*size, attributes, tx);
        self.api_sender.send(msg).unwrap();
        rx.recv().unwrap()
    }

    pub fn send_webgl_command(&self, context_id: WebGLContextId, command: WebGLCommand) {
        let msg = ApiMsg::WebGLCommand(context_id, command);
        self.api_sender.send(msg).unwrap();
    }

    #[inline]
    pub fn next_stacking_context_id(&self) -> StackingContextId {
        let new_id = self.next_unique_id();
        StackingContextId(new_id.0, new_id.1)
    }

    #[inline]
    pub fn next_display_list_id(&self) -> DisplayListId {
        let new_id = self.next_unique_id();
        DisplayListId(new_id.0, new_id.1)
    }

    #[inline]
    fn next_unique_id(&self) -> (u32, u32) {
        let IdNamespace(namespace) = self.id_namespace;
        let ResourceId(id) = self.next_id.get();
        self.next_id.set(ResourceId(id + 1));
        (namespace, id)
    }
}

