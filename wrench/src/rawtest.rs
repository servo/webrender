/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use WindowWrapper;
use blob;
use std::sync::mpsc::{channel, Receiver, Sender};
use webrender::api::*;
use wrench::Wrench;

pub struct RawtestHarness<'a> {
    wrench: &'a mut Wrench,
    rx: Receiver<()>,
    window: &'a mut WindowWrapper,
}

impl<'a> RawtestHarness<'a> {
    pub fn new(wrench: &'a mut Wrench, window: &'a mut WindowWrapper) -> RawtestHarness<'a> {
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
        wrench
            .renderer
            .set_render_notifier(Box::new(Notifier { tx: tx }));

        RawtestHarness {
            wrench: wrench,
            rx: rx,
            window: window,
        }
    }

    pub fn run(mut self) {
        self.retained_blob_images_test();
        self.tile_decomposition();
        self.save_restore();
    }

    fn render_and_get_pixels(&mut self, window_rect: DeviceUintRect) -> Vec<u8> {
        self.rx.recv().unwrap();
        self.wrench.render();
        self.wrench.renderer.read_pixels_rgba8(window_rect)
    }

    fn tile_decomposition(&mut self) {
        // This exposes a crash in tile decomposition
        let layout_size = LayoutSize::new(800., 800.);
        let mut resources = ResourceUpdates::new();

        let blob_img = self.wrench.api.generate_image_key();
        resources.add_image(
            blob_img,
            ImageDescriptor::new(151, 56, ImageFormat::BGRA8, true),
            ImageData::new_blob_image(blob::serialize_blob(ColorU::new(50, 50, 150, 255))),
            Some(128),
        );

        let root_background_color = Some(ColorF::new(1.0, 1.0, 1.0, 1.0));

        let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);

        let info = LayoutPrimitiveInfo::new(
            LayoutRect::new(LayoutPoint::new(448.899994, 74.0), LayoutSize::new(151.000031, 56.)));

        // setup some malicious image size parameters
        builder.push_image(
            &info,
            LayoutSize::new(151., 56.0),
            LayoutSize::new(151.0, 56.0),
            ImageRendering::Auto,
            blob_img,
        );

        self.wrench.api.set_display_list(
            self.wrench.document_id,
            Epoch(0),
            root_background_color,
            layout_size,
            builder.finalize(),
            false,
            resources,
        );
        self.wrench
            .api
            .generate_frame(self.wrench.document_id, None);

        self.rx.recv().unwrap();
        self.wrench.render();
    }

    fn retained_blob_images_test(&mut self) {
        let blob_img;
        let window_size = self.window.get_inner_size_pixels();
        let window_size = DeviceUintSize::new(window_size.0, window_size.1);

        let test_size = DeviceUintSize::new(400, 400);
        let document_id = self.wrench.document_id;

        let window_rect = DeviceUintRect::new(
            DeviceUintPoint::new(0, window_size.height - test_size.height),
            test_size,
        );
        let layout_size = LayoutSize::new(400., 400.);
        let mut resources = ResourceUpdates::new();
        {
            let api = &self.wrench.api;

            blob_img = api.generate_image_key();
            resources.add_image(
                blob_img,
                ImageDescriptor::new(500, 500, ImageFormat::BGRA8, true),
                ImageData::new_blob_image(blob::serialize_blob(ColorU::new(50, 50, 150, 255))),
                None,
            );
        }
        let root_background_color = Some(ColorF::new(1.0, 1.0, 1.0, 1.0));

        // draw the blob the first time
        let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);
        let info = LayoutPrimitiveInfo::new(
            LayoutRect::new(LayoutPoint::new(0.0, 60.0), LayoutSize::new(200.0, 200.0)));

        builder.push_image(
            &info,
            LayoutSize::new(200.0, 200.0),
            LayoutSize::new(0.0, 0.0),
            ImageRendering::Auto,
            blob_img,
        );

        self.wrench.api.set_display_list(
            document_id,
            Epoch(0),
            root_background_color,
            layout_size,
            builder.finalize(),
            false,
            resources,
        );
        self.wrench.api.generate_frame(document_id, None);


        // draw the blob image a second time at a different location

        // make a new display list that refers to the first image
        let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);
        let info = LayoutPrimitiveInfo::new(
            LayoutRect::new(LayoutPoint::new(1.0, 60.0), LayoutSize::new(200.0, 200.0)));
        builder.push_image(
            &info,
            LayoutSize::new(200.0, 200.0),
            LayoutSize::new(0.0, 0.0),
            ImageRendering::Auto,
            blob_img,
        );

        self.wrench.api.set_display_list(
            document_id,
            Epoch(1),
            root_background_color,
            layout_size,
            builder.finalize(),
            false,
            ResourceUpdates::new(),
        );

        self.wrench.api.generate_frame(document_id, None);

        let pixels_first = self.render_and_get_pixels(window_rect);
        let pixels_second = self.render_and_get_pixels(window_rect);

        // use png;
        // png::save_flipped("out1.png", &pixels_first, window_rect.size);
        // png::save_flipped("out2.png", &pixels_second, window_rect.size);

        assert!(pixels_first != pixels_second);
    }


    // Ensures that content doing a save-restore produces the same results as not
    fn save_restore(&mut self) {
        fn rect(x: f32, y: f32, w: f32, h: f32) -> LayoutRect {
            LayoutRect::new(
                    LayoutPoint::new(x, y),
                    LayoutSize::new(w, h))
        }

        let window_size = self.window.get_inner_size_pixels();
        let window_size = DeviceUintSize::new(window_size.0, window_size.1);

        let test_size = DeviceUintSize::new(400, 400);

        let window_rect = DeviceUintRect::new(
            DeviceUintPoint::new(0, window_size.height - test_size.height),
            test_size,
        );
        let layout_size = LayoutSize::new(400., 400.);
        let root_background_color = Some(ColorF::new(1.0, 1.0, 1.0, 1.0));


        let mut do_test = |should_try_and_fail| {
            let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);

            let clip = builder.define_clip(None, rect(110., 120., 200., 200.),
                                           None::<ComplexClipRegion>, None);
            builder.push_clip_id(clip);
            builder.push_rect(&PrimitiveInfo::new(rect(100., 100., 100., 100.)),
                              ColorF::new(0.0, 0.0, 1.0, 1.0));

            if should_try_and_fail {
                builder.save();
                let clip = builder.define_clip(None, rect(80., 80., 90., 90.),
                                           None::<ComplexClipRegion>, None);
                builder.push_clip_id(clip);
                builder.push_rect(&PrimitiveInfo::new(rect(110., 110., 50., 50.)),
                              ColorF::new(0.0, 1.0, 0.0, 1.0));
                builder.push_shadow(&PrimitiveInfo::new(rect(100., 100., 100., 100.)),
                    Shadow {
                        offset: LayoutVector2D::new(1.0, 1.0),
                        blur_radius: 1.0,
                        color: ColorF::new(0.0, 0.0, 0.0, 1.0),
                    });
                builder.push_line(&PrimitiveInfo::new(rect(100., 100., 100., 100.)),
                                  110., 110., 160., LineOrientation::Horizontal, 2.0,
                                  ColorF::new(0.0, 0.0, 0.0, 1.0), LineStyle::Solid);
                builder.restore();
            }

            {
                builder.save();
                let clip = builder.define_clip(None, rect(80., 80., 100., 100.),
                                               None::<ComplexClipRegion>, None);
                builder.push_clip_id(clip);
                builder.push_rect(&PrimitiveInfo::new(rect(150., 150., 100., 100.)),
                                  ColorF::new(0.0, 0.0, 1.0, 1.0));

                builder.pop_clip_id();
                builder.clear_save();
            }

            builder.pop_clip_id();

            self.wrench.api.set_display_list(
                self.wrench.document_id,
                Epoch(0),
                root_background_color,
                layout_size,
                builder.finalize(),
                false,
                ResourceUpdates::new(),
            );
            self.wrench
                .api
                .generate_frame(self.wrench.document_id, None);

            self.render_and_get_pixels(window_rect)
        };


        let first = do_test(false);
        let second = do_test(true);

        assert_eq!(first, second);
    }
}
