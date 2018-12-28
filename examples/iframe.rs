/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate gleam;
extern crate glutin;
extern crate webrender;
extern crate winit;

#[path = "common/boilerplate.rs"]
mod boilerplate;

use boilerplate::{Example, HandyDandyRectBuilder};
use webrender::api::*;

// This example uses the push_iframe API to nest a second pipeline's displaylist
// inside the root pipeline's display list. When it works, a green square is
// shown. If it fails, a red square is shown.

struct App {}

impl Example for App {
    fn render(
        &mut self,
        api: &RenderApi,
        builder: &mut DisplayListBuilder,
        _txn: &mut Transaction,
        _framebuffer_size: DeviceIntSize,
        pipeline_id: PipelineId,
        document_id: DocumentId,
    ) {
        // All the sub_* things are for the nested pipeline
        let sub_size = DeviceIntSize::new(100, 100);
        let sub_bounds = (0, 0).to(sub_size.width as i32, sub_size.height as i32);

        let sub_pipeline_id = PipelineId(pipeline_id.0, 42);
        let mut sub_builder = DisplayListBuilder::new(sub_pipeline_id, sub_bounds.size);
        let mut space_and_clip = SpaceAndClipInfo::root_scroll(pipeline_id);

        let info = LayoutPrimitiveInfo::new(sub_bounds);
        sub_builder.push_stacking_context(
            &info,
            space_and_clip.spatial_id,
            None,
            TransformStyle::Flat,
            MixBlendMode::Normal,
            &[],
            RasterSpace::Screen,
        );

        // green rect visible == success
        sub_builder.push_rect(&info, &space_and_clip, ColorF::new(0.0, 1.0, 0.0, 1.0));
        sub_builder.pop_stacking_context();

        let mut txn = Transaction::new();
        txn.set_display_list(
            Epoch(0),
            None,
            sub_bounds.size,
            sub_builder.finalize(),
            true,
        );
        api.send_transaction(document_id, txn);

        space_and_clip.spatial_id = builder.push_reference_frame(
            &sub_bounds,
            space_and_clip.spatial_id,
            TransformStyle::Flat,
            Some(PropertyBinding::Binding(PropertyBindingKey::new(42), LayoutTransform::identity())),
            None,
        );

        // And this is for the root pipeline
        let info = LayoutPrimitiveInfo::new(sub_bounds);
        builder.push_stacking_context(
            &info,
            space_and_clip.spatial_id,
            None,
            TransformStyle::Flat,
            MixBlendMode::Normal,
            &[],
            RasterSpace::Screen,
        );
        // red rect under the iframe: if this is visible, things have gone wrong
        builder.push_rect(&info, &space_and_clip, ColorF::new(1.0, 0.0, 0.0, 1.0));
        builder.push_iframe(&info, &space_and_clip, sub_pipeline_id, false);
        builder.pop_stacking_context();
        builder.pop_reference_frame();
    }
}

fn main() {
    let mut app = App {};
    boilerplate::main_wrapper(&mut app, None);
}
