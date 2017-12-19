/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use WindowWrapper;
use blob;
use euclid::{TypedRect, TypedSize2D, TypedPoint2D};
use std::sync::Arc;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::mpsc::Receiver;
use webrender::api::*;
use wrench::Wrench;

pub struct RawtestHarness<'a> {
    wrench: &'a mut Wrench,
    rx: Receiver<()>,
    window: &'a mut WindowWrapper,
}

fn point<T: Copy, U>(x: T, y: T) -> TypedPoint2D<T, U> {
    TypedPoint2D::new(x, y)
}

fn size<T: Copy, U>(x: T, y: T) -> TypedSize2D<T, U> {
    TypedSize2D::new(x, y)
}

fn rect<T: Copy, U>(x: T, y: T, width: T, height: T) -> TypedRect<T, U> {
    TypedRect::new(point(x, y), size(width, height))
}

impl<'a> RawtestHarness<'a> {
    pub fn new(wrench: &'a mut Wrench, window: &'a mut WindowWrapper, rx: Receiver<()>) -> Self {
        RawtestHarness {
            wrench,
            rx,
            window,
        }
    }

    pub fn run(mut self) {
        self.retained_blob_images_test();
        self.blob_update_test();
        self.tile_decomposition();
        self.save_restore();
    }

    fn render_and_get_pixels(&mut self, window_rect: DeviceUintRect) -> Vec<u8> {
        self.rx.recv().unwrap();
        self.wrench.render();
        self.wrench.renderer.read_pixels_rgba8(window_rect)
    }

    fn submit_dl(
        &mut self,
        epoch: &mut Epoch,
        layout_size: LayoutSize,
        builder: DisplayListBuilder,
        resources: Option<ResourceUpdates>
    ) {
        let root_background_color = Some(ColorF::new(1.0, 1.0, 1.0, 1.0));
        self.wrench.api.set_display_list(
            self.wrench.document_id,
            *epoch,
            root_background_color,
            layout_size,
            builder.finalize(),
            false,
            resources.unwrap_or(ResourceUpdates::new()),
        );
        epoch.0 += 1;

        self.wrench.api.generate_frame(self.wrench.document_id, None);
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

        let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);

        let info = LayoutPrimitiveInfo::new(rect(448.899994, 74.0, 151.000031, 56.));

        // setup some malicious image size parameters
        builder.push_image(
            &info,
            size(151., 56.0),
            size(151.0, 56.0),
            ImageRendering::Auto,
            blob_img,
        );

        let mut epoch = Epoch(0);

        self.submit_dl(&mut epoch, layout_size, builder, Some(resources));

        self.rx.recv().unwrap();
        self.wrench.render();
    }

    fn retained_blob_images_test(&mut self) {
        let blob_img;
        let window_size = self.window.get_inner_size_pixels();
        let window_size = DeviceUintSize::new(window_size.0, window_size.1);

        let test_size = DeviceUintSize::new(400, 400);

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

        // draw the blob the first time
        let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);
        let info = LayoutPrimitiveInfo::new(rect(0.0, 60.0, 200.0, 200.0));

        builder.push_image(
            &info,
            size(200.0, 200.0),
            size(0.0, 0.0),
            ImageRendering::Auto,
            blob_img,
        );

        let mut epoch = Epoch(0);

        self.submit_dl(&mut epoch, layout_size, builder, Some(resources));

        // draw the blob image a second time at a different location

        // make a new display list that refers to the first image
        let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);
        let info = LayoutPrimitiveInfo::new(rect(1.0, 60.0, 200.0, 200.0));
        builder.push_image(
            &info,
            size(200.0, 200.0),
            size(0.0, 0.0),
            ImageRendering::Auto,
            blob_img,
        );

        self.submit_dl(&mut epoch, layout_size, builder, None);

        let called = Arc::new(AtomicIsize::new(0));
        let called_inner = Arc::clone(&called);

        self.wrench.callbacks.lock().unwrap().request = Box::new(move |_| {
            called_inner.fetch_add(1, Ordering::SeqCst);
        });

        let pixels_first = self.render_and_get_pixels(window_rect);
        assert!(called.load(Ordering::SeqCst) == 1);

        let pixels_second = self.render_and_get_pixels(window_rect);

        // make sure we only requested once
        assert!(called.load(Ordering::SeqCst) == 1);

        // use png;
        // png::save_flipped("out1.png", &pixels_first, window_rect.size);
        // png::save_flipped("out2.png", &pixels_second, window_rect.size);

        assert!(pixels_first != pixels_second);
    }

    fn blob_update_test(&mut self) {
        let blob_img;
        let window_size = self.window.get_inner_size_pixels();
        let window_size = DeviceUintSize::new(window_size.0, window_size.1);

        let test_size = DeviceUintSize::new(400, 400);

        let window_rect = DeviceUintRect::new(
            point(0, window_size.height - test_size.height),
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

        // draw the blob the first time
        let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);
        let info = LayoutPrimitiveInfo::new(rect(0.0, 60.0, 200.0, 200.0));

        builder.push_image(
            &info,
            size(200.0, 200.0),
            size(0.0, 0.0),
            ImageRendering::Auto,
            blob_img,
        );

        let mut epoch = Epoch(0);

        self.submit_dl(&mut epoch, layout_size, builder, Some(resources));
        let pixels_first = self.render_and_get_pixels(window_rect);


        // draw the blob image a second time after updating it with the same color
        let mut resources = ResourceUpdates::new();
        resources.update_image(
            blob_img,
            ImageDescriptor::new(500, 500, ImageFormat::BGRA8, true),
            ImageData::new_blob_image(blob::serialize_blob(ColorU::new(50, 50, 150, 255))),
            Some(rect(100, 100, 100, 100)),
        );

        // make a new display list that refers to the first image
        let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);
        let info = LayoutPrimitiveInfo::new(rect(0.0, 60.0, 200.0, 200.0));
        builder.push_image(
            &info,
            size(200.0, 200.0),
            size(0.0, 0.0),
            ImageRendering::Auto,
            blob_img,
        );

        self.submit_dl(&mut epoch, layout_size, builder, Some(resources));
        let pixels_second = self.render_and_get_pixels(window_rect);


        // draw the blob image a third time after updating it with a different color
        let mut resources = ResourceUpdates::new();
        resources.update_image(
            blob_img,
            ImageDescriptor::new(500, 500, ImageFormat::BGRA8, true),
            ImageData::new_blob_image(blob::serialize_blob(ColorU::new(50, 150, 150, 255))),
            Some(rect(200, 200, 100, 100)),
        );

        // make a new display list that refers to the first image
        let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);
        let info = LayoutPrimitiveInfo::new(rect(0.0, 60.0, 200.0, 200.0));
        builder.push_image(
            &info,
            size(200.0, 200.0),
            size(0.0, 0.0),
            ImageRendering::Auto,
            blob_img,
        );

        self.submit_dl(&mut epoch, layout_size, builder, Some(resources));
        let pixels_third = self.render_and_get_pixels(window_rect);

        assert!(pixels_first == pixels_second);
        assert!(pixels_first != pixels_third);
    }

    // Ensures that content doing a save-restore produces the same results as not
    fn save_restore(&mut self) {
        let window_size = self.window.get_inner_size_pixels();
        let window_size = DeviceUintSize::new(window_size.0, window_size.1);

        let test_size = DeviceUintSize::new(400, 400);

        let window_rect = DeviceUintRect::new(
            DeviceUintPoint::new(0, window_size.height - test_size.height),
            test_size,
        );
        let layout_size = LayoutSize::new(400., 400.);

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
                builder.push_line(&PrimitiveInfo::new(rect(110., 110., 50., 2.)),
                                  0.0, LineOrientation::Horizontal,
                                  &ColorF::new(0.0, 0.0, 0.0, 1.0), LineStyle::Solid);
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

            self.submit_dl(&mut Epoch(0), layout_size, builder, None);

            self.render_and_get_pixels(window_rect)
        };


        let first = do_test(false);
        let second = do_test(true);

        assert_eq!(first, second);
    }
}
