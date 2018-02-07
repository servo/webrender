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
        self.test_tiled_blob_masks();
        self.test_retained_blob_images_test();
        self.test_blob_update_test();
        self.test_blob_update_epoch_test();
        self.test_tile_decomposition();
        self.test_save_restore();
        self.test_capture();
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
        let mut txn = Transaction::new();
        let root_background_color = Some(ColorF::new(1.0, 1.0, 1.0, 1.0));
        if let Some(resources) = resources {
            txn.update_resources(resources);
        }
        txn.set_display_list(
            *epoch,
            root_background_color,
            layout_size,
            builder.finalize(),
            false,
        );
        epoch.0 += 1;

        txn.generate_frame();
        self.wrench.api.send_transaction(self.wrench.document_id, txn);
    }

    fn test_tile_decomposition(&mut self) {
        println!("\ttile decomposition...");
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
            AlphaType::PremultipliedAlpha,
            blob_img,
        );

        let mut epoch = Epoch(0);

        self.submit_dl(&mut epoch, layout_size, builder, Some(resources));

        self.rx.recv().unwrap();
        self.wrench.render();

        // Leaving a tiled blob image in the resource cache
        // confuses the `test_capture`. TODO: remove this
        resources = ResourceUpdates::new();
        resources.delete_image(blob_img);
        self.wrench.api.update_resources(resources);
    }

    fn test_tiled_blob_masks(&mut self) {
        println!("\ttiled blob masks...");
        // This exposes a crash when processing a clip mask that is also a blob-image
        let layout_size = LayoutSize::new(800., 800.);
        let mut resources = ResourceUpdates::new();

        let bounds = rect(448.899994, 74.0, 257.0, 180.0);
        let info = LayoutPrimitiveInfo::new(bounds);

        let blob_img = self.wrench.api.generate_image_key();
        resources.add_image(
            blob_img,
            ImageDescriptor::new(
                bounds.size.width as u32,
                bounds.size.height as u32,
                ImageFormat::BGRA8,
                true),
            ImageData::new_blob_image(blob::serialize_blob(ColorU::new(50, 50, 150, 255))),
            Some(128),
        );

        let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);

        let mask = ImageMask { image: blob_img, rect: bounds, repeat: false };
        let clip = builder.define_clip(
            bounds,
            None::<ComplexClipRegion>,
            Some(mask));

        builder.push_clip_id(clip);
        builder.push_rect(&info, ColorF::new(0.5, 0.5, 0.8, 1.0));
        builder.pop_clip_id();

        let mut epoch = Epoch(0);

        self.submit_dl(&mut epoch, layout_size, builder, Some(resources));

        self.rx.recv().unwrap();
        self.wrench.render();

        // Leaving a tiled blob image in the resource cache
        // confuses the `test_capture`. TODO: remove this
        resources = ResourceUpdates::new();
        resources.delete_image(blob_img);
        self.wrench.api.update_resources(resources);
    }

    fn test_retained_blob_images_test(&mut self) {
        println!("\tretained blob images test...");
        let blob_img;
        let window_size = self.window.get_inner_size();

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
            AlphaType::PremultipliedAlpha,
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
            AlphaType::PremultipliedAlpha,
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

    fn test_blob_update_epoch_test(&mut self) {
        println!("\tblob update epoch test...");
        let (blob_img, blob_img2);
        let window_size = self.window.get_inner_size();

        let test_size = DeviceUintSize::new(400, 400);

        let window_rect = DeviceUintRect::new(
            point(0, window_size.height - test_size.height),
            test_size,
        );
        let layout_size = LayoutSize::new(400., 400.);
        let mut resources = ResourceUpdates::new();
        let (blob_img, blob_img2) = {
            let api = &self.wrench.api;

            blob_img = api.generate_image_key();
            resources.add_image(
                blob_img,
                ImageDescriptor::new(500, 500, ImageFormat::BGRA8, true),
                ImageData::new_blob_image(blob::serialize_blob(ColorU::new(50, 50, 150, 255))),
                None,
            );
            blob_img2 = api.generate_image_key();
            resources.add_image(
                blob_img2,
                ImageDescriptor::new(500, 500, ImageFormat::BGRA8, true),
                ImageData::new_blob_image(blob::serialize_blob(ColorU::new(80, 50, 150, 255))),
                None,
            );
            (blob_img, blob_img2)
        };

        // setup some counters to count how many times each image is requested
        let img1_requested = Arc::new(AtomicIsize::new(0));
        let img1_requested_inner = Arc::clone(&img1_requested);
        let img2_requested = Arc::new(AtomicIsize::new(0));
        let img2_requested_inner = Arc::clone(&img2_requested);

        // track the number of times that the second image has been requested
        self.wrench.callbacks.lock().unwrap().request = Box::new(move |&desc| {
            if desc.key == blob_img {
                img1_requested_inner.fetch_add(1, Ordering::SeqCst);
            }
            if desc.key == blob_img2 {
                img2_requested_inner.fetch_add(1, Ordering::SeqCst);
            }
        });

        // create two blob images and draw them
        let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);
        let info = LayoutPrimitiveInfo::new(rect(0.0, 60.0, 200.0, 200.0));
        let info2 = LayoutPrimitiveInfo::new(rect(200.0, 60.0, 200.0, 200.0));
        let push_images = |builder: &mut DisplayListBuilder| {
            builder.push_image(
                &info,
                size(200.0, 200.0),
                size(0.0, 0.0),
                ImageRendering::Auto,
                AlphaType::PremultipliedAlpha,
                blob_img,
            );
            builder.push_image(
                &info2,
                size(200.0, 200.0),
                size(0.0, 0.0),
                ImageRendering::Auto,
                AlphaType::PremultipliedAlpha,
                blob_img2,
            );
        };

        push_images(&mut builder);

        let mut epoch = Epoch(0);

        self.submit_dl(&mut epoch, layout_size, builder, Some(resources));
        let _pixels_first = self.render_and_get_pixels(window_rect);


        // update and redraw both images
        let mut resources = ResourceUpdates::new();
        resources.update_image(
            blob_img,
            ImageDescriptor::new(500, 500, ImageFormat::BGRA8, true),
            ImageData::new_blob_image(blob::serialize_blob(ColorU::new(50, 50, 150, 255))),
            Some(rect(100, 100, 100, 100)),
        );
        resources.update_image(
            blob_img2,
            ImageDescriptor::new(500, 500, ImageFormat::BGRA8, true),
            ImageData::new_blob_image(blob::serialize_blob(ColorU::new(59, 50, 150, 255))),
            Some(rect(100, 100, 100, 100)),
        );

        let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);
        push_images(&mut builder);
        self.submit_dl(&mut epoch, layout_size, builder, Some(resources));
        let _pixels_second = self.render_and_get_pixels(window_rect);


        // only update the first image
        let mut resources = ResourceUpdates::new();
        resources.update_image(
            blob_img,
            ImageDescriptor::new(500, 500, ImageFormat::BGRA8, true),
            ImageData::new_blob_image(blob::serialize_blob(ColorU::new(50, 150, 150, 255))),
            Some(rect(200, 200, 100, 100)),
        );

        let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);
        push_images(&mut builder);
        self.submit_dl(&mut epoch, layout_size, builder, Some(resources));
        let _pixels_third = self.render_and_get_pixels(window_rect);

        // the first image should be requested 3 times
        assert_eq!(img1_requested.load(Ordering::SeqCst), 3);
        // the second image should've been requested twice
        assert_eq!(img2_requested.load(Ordering::SeqCst), 2);
    }

    fn test_blob_update_test(&mut self) {
        println!("\tblob update test...");
        let window_size = self.window.get_inner_size();

        let test_size = DeviceUintSize::new(400, 400);

        let window_rect = DeviceUintRect::new(
            point(0, window_size.height - test_size.height),
            test_size,
        );
        let layout_size = LayoutSize::new(400., 400.);
        let mut resources = ResourceUpdates::new();

        let blob_img = {
            let img = self.wrench.api.generate_image_key();
            resources.add_image(
                img,
                ImageDescriptor::new(500, 500, ImageFormat::BGRA8, true),
                ImageData::new_blob_image(blob::serialize_blob(ColorU::new(50, 50, 150, 255))),
                None,
            );
            img
        };

        // draw the blobs the first time
        let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);
        let info = LayoutPrimitiveInfo::new(rect(0.0, 60.0, 200.0, 200.0));

        builder.push_image(
            &info,
            size(200.0, 200.0),
            size(0.0, 0.0),
            ImageRendering::Auto,
            AlphaType::PremultipliedAlpha,
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
            AlphaType::PremultipliedAlpha,
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
            AlphaType::PremultipliedAlpha,
            blob_img,
        );

        self.submit_dl(&mut epoch, layout_size, builder, Some(resources));
        let pixels_third = self.render_and_get_pixels(window_rect);

        assert!(pixels_first == pixels_second);
        assert!(pixels_first != pixels_third);
    }

    // Ensures that content doing a save-restore produces the same results as not
    fn test_save_restore(&mut self) {
        println!("\tsave/restore...");
        let window_size = self.window.get_inner_size();

        let test_size = DeviceUintSize::new(400, 400);

        let window_rect = DeviceUintRect::new(
            DeviceUintPoint::new(0, window_size.height - test_size.height),
            test_size,
        );
        let layout_size = LayoutSize::new(400., 400.);

        let mut do_test = |should_try_and_fail| {
            let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);

            let clip = builder.define_clip(
                rect(110., 120., 200., 200.),
                None::<ComplexClipRegion>,
                None
            );
            builder.push_clip_id(clip);
            builder.push_rect(&PrimitiveInfo::new(rect(100., 100., 100., 100.)),
                              ColorF::new(0.0, 0.0, 1.0, 1.0));

            if should_try_and_fail {
                builder.save();
                let clip = builder.define_clip(
                    rect(80., 80., 90., 90.),
                    None::<ComplexClipRegion>,
                    None
                );
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
                let clip = builder.define_clip(
                    rect(80., 80., 100., 100.),
                    None::<ComplexClipRegion>,
                    None
                );
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

    fn test_capture(&mut self) {
        println!("\tcapture...");
        let path = "../captures/test";
        let layout_size = LayoutSize::new(400., 400.);
        let dim = self.window.get_inner_size();
        let window_rect = DeviceUintRect::new(
            point(0, dim.height - layout_size.height as u32),
            size(layout_size.width as u32, layout_size.height as u32),
        );

        // 1. render some scene

        let mut resources = ResourceUpdates::new();
        let image = self.wrench.api.generate_image_key();
        resources.add_image(
            image,
            ImageDescriptor::new(1, 1, ImageFormat::BGRA8, true),
            ImageData::new(vec![0xFF, 0, 0, 0xFF]),
            None,
        );

        let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);

        builder.push_image(
            &LayoutPrimitiveInfo::new(rect(300.0, 70.0, 150.0, 50.0)),
            size(150.0, 50.0),
            size(0.0, 0.0),
            ImageRendering::Auto,
            AlphaType::PremultipliedAlpha,
            image,
        );

        let mut txn = Transaction::new();

        txn.set_display_list(
            Epoch(0),
            Some(ColorF::new(1.0, 1.0, 1.0, 1.0)),
            layout_size,
            builder.finalize(),
            false,
        );
        txn.generate_frame();

        self.wrench.api.send_transaction(self.wrench.document_id, txn);

        let pixels0 = self.render_and_get_pixels(window_rect);

        // 2. capture it
        self.wrench.api.save_capture(path.into(), CaptureBits::all());

        // 3. set a different scene

        builder = DisplayListBuilder::new(self.wrench.root_pipeline_id, layout_size);

        let mut txn = Transaction::new();
        txn.set_display_list(
            Epoch(1),
            Some(ColorF::new(1.0, 0.0, 0.0, 1.0)),
            layout_size,
            builder.finalize(),
            false,
        );
        self.wrench.api.send_transaction(self.wrench.document_id, txn);

        // 4. load the first one

        let mut documents = self.wrench.api.load_capture(path.into());
        let captured = documents.swap_remove(0);

        // 5. render the built frame and compare
        let pixels1 = self.render_and_get_pixels(window_rect);
        assert!(pixels0 == pixels1);

        // 6. rebuild the scene and compare again
        let mut txn = Transaction::new();
        txn.set_root_pipeline(captured.root_pipeline_id.unwrap());
        txn.generate_frame();
        self.wrench.api.send_transaction(captured.document_id, txn);
        let pixels2 = self.render_and_get_pixels(window_rect);
        assert!(pixels0 == pixels2);
    }
}
