use euclid::Point2D;
use internal_types::ApiMsg;
use std::sync::mpsc::Sender;
use string_cache::Atom;
use types::{PipelineId, ImageID, ImageFormat, StackingContext};
use types::{ColorF, DisplayListID, DisplayListBuilder, Epoch};

#[derive(Clone)]
pub struct RenderApi {
    pub tx: Sender<ApiMsg>,
}

impl RenderApi {
    pub fn add_font(&self, id: Atom, bytes: Vec<u8>) {
        let msg = ApiMsg::AddFont(id, bytes);
        self.tx.send(msg).unwrap();
    }

    pub fn add_image(&self, width: u32, height: u32, format: ImageFormat, bytes: Vec<u8>) -> ImageID {
        let id = ImageID::new();
        let msg = ApiMsg::AddImage(id, width, height, format, bytes);
        self.tx.send(msg).unwrap();
        id
    }

    // TODO: Support changing dimensions (and format) during image update?
    pub fn update_image(&self, _image_id: ImageID, _bytes: Vec<u8>) {
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
}
