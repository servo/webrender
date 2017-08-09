/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate gleam;
extern crate glutin;
extern crate webrender;

#[path="common/boilerplate.rs"]
mod boilerplate;

use boilerplate::HandyDandyRectBuilder;
use webrender::api::*;

static IMAGE_KEY: ImageKey = ImageKey(IdNamespace(0), 0);

fn body(_api: &RenderApi,
        _document_id: &DocumentId,
        builder: &mut DisplayListBuilder,
        resources: &mut ResourceUpdates,
        _pipeline_id: &PipelineId,
        _layout_size: &LayoutSize) {

    let mut image_data = Vec::new();
    for y in 0..32 {
        for x in 0..32 {
            let lum = 255 * (((x & 8) == 0) ^ ((y & 8) == 0)) as u8;
            image_data.extend_from_slice(&[lum, lum, lum, 0xff]);
        }
    }

    resources.add_image(
        IMAGE_KEY,
        ImageDescriptor::new(32, 32, ImageFormat::BGRA8, true),
        ImageData::new(image_data),
        None,
    );

    let bounds = (0,0).to(512, 512);
    builder.push_stacking_context(ScrollPolicy::Scrollable,
                                  bounds,
                                  None,
                                  TransformStyle::Flat,
                                  None,
                                  MixBlendMode::Normal,
                                  Vec::new());

    let image_size = LayoutSize::new(100.0, 100.0);

    builder.push_image(
        LayoutRect::new(LayoutPoint::new(100.0, 100.0), image_size),
        Some(LocalClip::from(bounds)),
        image_size,
        LayoutSize::zero(),
        ImageRendering::Auto,
        IMAGE_KEY
    );

    builder.push_image(
        LayoutRect::new(LayoutPoint::new(250.0, 100.0), image_size),
        Some(LocalClip::from(bounds)),
        image_size,
        LayoutSize::zero(),
        ImageRendering::Pixelated,
        IMAGE_KEY
    );

    builder.pop_stacking_context();
}

fn event_handler(event: &glutin::Event,
                 document_id: DocumentId,
                 api: &RenderApi) {
    match *event {
        glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(key)) => {
            match key {
                 glutin::VirtualKeyCode::Space => {
                    let mut image_data = Vec::new();
                    for y in 0..64 {
                        for x in 0..64 {
                            let r = 255 * ((y & 32) == 0) as u8;
                            let g = 255 * ((x & 32) == 0) as u8;
                            image_data.extend_from_slice(&[0, g, r, 0xff]);
                        }
                    }

                    let mut updates = ResourceUpdates::new();
                    updates.update_image(IMAGE_KEY,
                                         ImageDescriptor::new(64, 64, ImageFormat::BGRA8, true),
                                         ImageData::new(image_data),
                                         None);
                    api.update_resources(updates);
                    api.generate_frame(document_id, None);
                 }
                 _ => {}
             }
         }
         _ => {}
     }
}

fn main() {
    boilerplate::main_wrapper(body, event_handler, None);
}
