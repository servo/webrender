/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use WindowWrapper;
use image::ColorType;
use image::png::PNGEncoder;
use std::fs::File;
use std::path::Path;
use std::sync::mpsc::{channel, Sender};
use webrender::api::*;
use wrench::{Wrench, WrenchThing};
use yaml_frame_reader::YamlFrameReader;

pub fn save_flipped<P: AsRef<Path>>(path: P, orig_pixels: &[u8], size: DeviceUintSize) {
    // flip image vertically (texture is upside down)
    let mut data = orig_pixels.to_owned();
    let stride = size.width as usize * 4;
    for y in 0 .. size.height as usize {
        let dst_start = y * stride;
        let src_start = (size.height as usize - y - 1) * stride;
        let src_slice = &orig_pixels[src_start .. src_start + stride];
        (&mut data[dst_start .. dst_start + stride]).clone_from_slice(&src_slice[.. stride]);
    }

    let encoder = PNGEncoder::new(File::create(path).unwrap());
    encoder
        .encode(&data, size.width, size.height, ColorType::RGBA(8))
        .expect("Unable to encode PNG!");
}

pub fn png(wrench: &mut Wrench, window: &mut WindowWrapper, mut reader: YamlFrameReader) {
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
    reader.do_frame(wrench);

    // wait for the frame
    rx.recv().unwrap();
    wrench.render();

    let size = window.get_inner_size_pixels();
    let device_size = DeviceUintSize::new(size.0, size.1);
    let data = wrench
        .renderer
        .read_pixels_rgba8(DeviceUintRect::new(DeviceUintPoint::zero(), device_size));

    let mut out_path = reader.yaml_path().clone();
    out_path.set_extension("png");

    save_flipped(out_path, &data, device_size);
}
