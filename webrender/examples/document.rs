/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate euclid;
extern crate gleam;
extern crate glutin;
extern crate webrender;

#[path = "common/boilerplate.rs"]
mod boilerplate;

use boilerplate::Example;
use euclid::ScaleFactor;
use webrender::api::*;

// This example creates multiple documents overlapping each other with
// specified layer indices.

struct Document {
    id: DocumentId,
    pipeline_id: PipelineId,
    content_rect: LayoutRect,
    color: ColorF,
}

struct App {
    documents: Vec<Document>,
}

impl App {
    fn init(
        &mut self,
        api: &RenderApi,
        framebuffer_size: DeviceUintSize,
        device_pixel_ratio: f32,
    ) {
        let init_data = vec![
            (
                PipelineId(1, 0),
                -1,
                ColorF::new(0.0, 1.0, 0.0, 1.0),
                DeviceUintPoint::new(0, 0),
            ),
            (
                PipelineId(2, 0),
                -2,
                ColorF::new(1.0, 1.0, 0.0, 1.0),
                DeviceUintPoint::new(200, 0),
            ),
            (
                PipelineId(3, 0),
                -3,
                ColorF::new(1.0, 0.0, 0.0, 1.0),
                DeviceUintPoint::new(200, 200),
            ),
            (
                PipelineId(4, 0),
                -4,
                ColorF::new(1.0, 0.0, 1.0, 1.0),
                DeviceUintPoint::new(0, 200),
            ),
        ];

        for (pipeline_id, layer, color, offset) in init_data {
            let size = DeviceUintSize::new(250, 250);
            let bounds = DeviceUintRect::new(offset, size);

            let document_id = api.add_document(size, layer);
            api.set_window_parameters(document_id,
                framebuffer_size,
                bounds,
                1.0
            );
            api.set_root_pipeline(document_id, pipeline_id);

            self.documents.push(Document {
                id: document_id,
                pipeline_id,
                content_rect: bounds.to_f32() / ScaleFactor::new(device_pixel_ratio),
                color,
            });
        }
    }
}

impl Example for App {
    fn render(
        &mut self,
        api: &RenderApi,
        base_builder: &mut DisplayListBuilder,
        _: &mut ResourceUpdates,
        framebuffer_size: DeviceUintSize,
        _: PipelineId,
        _: DocumentId,
    ) {
        if self.documents.is_empty() {
            let device_pixel_ratio = framebuffer_size.width as f32 /
                base_builder.content_size().width;
            // this is the first run, hack around the boilerplate,
            // which assumes an example only needs one document
            self.init(api, framebuffer_size, device_pixel_ratio);
        }

        for doc in &self.documents {
            let mut builder = DisplayListBuilder::new(
                doc.pipeline_id,
                doc.content_rect.size,
            );
            let local_rect = LayoutRect::new(
                LayoutPoint::zero(),
                doc.content_rect.size,
            );

            builder.push_stacking_context(
                &LayoutPrimitiveInfo::new(doc.content_rect),
                ScrollPolicy::Fixed,
                None,
                TransformStyle::Flat,
                None,
                MixBlendMode::Normal,
                Vec::new(),
            );
            builder.push_rect(
                &LayoutPrimitiveInfo::new(local_rect),
                doc.color,
            );
            builder.pop_stacking_context();

            api.set_display_list(
                doc.id,
                Epoch(0),
                None,
                doc.content_rect.size,
                builder.finalize(),
                true,
                ResourceUpdates::new(),
            );

            api.generate_frame(doc.id, None);
        }
    }
}

fn main() {
    let mut app = App {
        documents: Vec::new(),
    };
    boilerplate::main_wrapper(&mut app, None);
}
