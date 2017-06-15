/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use WindowWrapper;
use std::sync::mpsc::{channel, Receiver, Sender};
use webrender_traits::*;
use wrench::{Wrench};

use blob;


pub struct RawtestHarness<'a> {
    wrench: &'a mut Wrench,
    window: &'a mut WindowWrapper,
    rx: Receiver<()>,
}

impl<'a> RawtestHarness<'a> {
    pub fn new(wrench: &'a mut Wrench,
               window: &'a mut WindowWrapper) -> RawtestHarness<'a>
    {
        // setup a notifier so we can wait for frames to be finished
        struct Notifier {
            tx: Sender<()>,
        };
        impl RenderNotifier for Notifier {
            fn new_frame_ready(&mut self) {
                self.tx.send(()).unwrap();
            }
            fn new_scroll_frame_ready(&mut self, _composite_needed: bool) {}
        }
        let (tx, rx) = channel();
        wrench.renderer.set_render_notifier(Box::new(Notifier { tx: tx }));

        RawtestHarness {
            wrench: wrench,
            window: window,
            rx: rx,
        }
    }

    pub fn run(mut self) {
        self.retained_blob_images_test();
    }

    fn retained_blob_images_test(&mut self) {
        let blob_img;
        let layout_size = LayoutSize::new(400., 400.);
        {
            let api = &self.wrench.api;

            blob_img = api.generate_image_key();
            api.add_image(blob_img,
                          ImageDescriptor::new(500, 500, ImageFormat::RGBA8, true),
                          ImageData::new_blob_image(blob::serialize_blob(ColorU::new(50, 50, 150, 255))),
                          None,
            );
        }
        let root_background_color = Some(ColorF::new(1.0, 1.0, 1.0, 1.0));
        let bounds = LayoutRect::new(LayoutPoint::zero(), layout_size);
        let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);
        let clip = builder.push_clip_region(&bounds, vec![], None);
        builder.push_image(
            LayoutRect::new(LayoutPoint::new(600.0, 60.0), LayoutSize::new(200.0, 200.0)),
            clip,
            LayoutSize::new(200.0, 200.0),
            LayoutSize::new(0.0, 0.0),
            ImageRendering::Auto,
            blob_img,
        );
        self.wrench.api.set_display_list(root_background_color,
                                         Epoch(0),
                                         layout_size,
                                         builder.finalize(),
                                         false);
        self.wrench.api.generate_frame(None);
        self.rx.recv().unwrap();
        self.wrench.render();
        let window_rect = DeviceUintRect::new(DeviceUintPoint::new(0, 0),
                                              DeviceUintSize::new(400, 400));
        let pixels_first = self.wrench.renderer.read_pixels_rgba8(window_rect);
        let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);
        let clip = builder.push_clip_region(&bounds, vec![], None);
        builder.push_image(
            LayoutRect::new(LayoutPoint::new(0.0, 60.0), LayoutSize::new(200.0, 200.0)),
            clip,
            LayoutSize::new(200.0, 200.0),
            LayoutSize::new(0.0, 0.0),
            ImageRendering::Auto,
            blob_img,
        );
        self.wrench.api.set_display_list(root_background_color,
                                         Epoch(0),
                                         layout_size,
                                         builder.finalize(),
                                         false);
        self.wrench.api.generate_frame(None);
        self.rx.recv().unwrap();
        self.wrench.render();
        let window_rect = DeviceUintRect::new(DeviceUintPoint::new(0, 0),
                                              DeviceUintSize::new(400, 400));
        let pixels_second = self.wrench.renderer.read_pixels_rgba8(window_rect);
        assert!(pixels_first == pixels_second);
    }
}
