use euclid::Point2D;
use internal_types::ApiMsg;
use platform::font::NativeFontHandle;
use std::sync::mpsc::{self, Sender};
use types::{PipelineId, ImageKey, ImageFormat, StackingContext};
use types::{ColorF, DisplayListID, DisplayListBuilder, Epoch, FontKey};

#[derive(Clone)]
pub struct RenderApi {
    pub tx: Sender<ApiMsg>,
}

impl RenderApi {
    pub fn add_raw_font(&self, key: FontKey, bytes: Vec<u8>) {
        let msg = ApiMsg::AddRawFont(key, bytes);
        self.tx.send(msg).unwrap();
    }

    pub fn add_native_font(&self, key: FontKey, native_font_handle: NativeFontHandle) {
        let msg = ApiMsg::AddNativeFont(key, native_font_handle);
        self.tx.send(msg).unwrap();
    }

    pub fn add_image(&self,
                     key: ImageKey,
                     width: u32,
                     height: u32,
                     format: ImageFormat,
                     bytes: Vec<u8>) {
        let msg = ApiMsg::AddImage(key, width, height, format, bytes);
        self.tx.send(msg).unwrap();
    }

    // TODO: Support changing dimensions (and format) during image update?
    pub fn update_image(&self,
                        _image_key: ImageKey,
                        _bytes: Vec<u8>) {
        //println!("TODO!");
        //let msg = ApiMsg::UpdateImage(id, bytes);
        //self.tx.send(msg).unwrap();
    }

    pub fn add_display_list(&self,
                            display_list: DisplayListBuilder,
                            pipeline_id: PipelineId,
                            epoch: Epoch) -> DisplayListID {
        debug_assert!(display_list.item_count() > 0, "Avoid adding empty lists!");
        let id = DisplayListID::new();
        let msg = ApiMsg::AddDisplayList(id, pipeline_id, epoch, display_list);
        self.tx.send(msg).unwrap();
        id
    }

    pub fn set_root_pipeline(&self, pipeline_id: PipelineId) {
        let msg = ApiMsg::SetRootPipeline(pipeline_id);
        self.tx.send(msg).unwrap();
    }

    pub fn set_root_stacking_context(&self,
                                     stacking_context: StackingContext,
                                     background_color: ColorF,
                                     epoch: Epoch,
                                     pipeline_id: PipelineId) {
        let msg = ApiMsg::SetRootStackingContext(stacking_context, background_color, epoch, pipeline_id);
        self.tx.send(msg).unwrap();
    }

    pub fn scroll(&self, delta: Point2D<f32>) {
        let msg = ApiMsg::Scroll(delta);
        self.tx.send(msg).unwrap();
    }

    pub fn translate_point_to_layer_space(&self, point: &Point2D<f32>) -> Point2D<f32> {
        let (tx, rx) = mpsc::channel();
        let msg = ApiMsg::TranslatePointToLayerSpace(*point, tx);
        self.tx.send(msg).unwrap();
        rx.recv().unwrap()
    }
}

