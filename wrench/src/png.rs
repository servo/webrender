/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use WindowWrapper;
use gleam::gl;
use image::ColorType;
use image::png::PNGEncoder;
use std::fs::File;
use std::sync::mpsc::{channel, Sender};
use webrender_traits::*;
use wrench::{Wrench, WrenchThing};
use yaml_frame_reader::YamlFrameReader;

pub fn png(wrench: &mut Wrench,
           window: &mut WindowWrapper,
           mut reader: YamlFrameReader)
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
    reader.do_frame(wrench);

    // wait for the frame
    rx.recv().unwrap();
    wrench.render();

    let size = window.get_inner_size_pixels();
    let mut data = gl::read_pixels(0,
                                   0,
                                   size.0 as gl::GLsizei,
                                   size.1 as gl::GLsizei,
                                   gl::RGBA,
                                   gl::UNSIGNED_BYTE);
    let width = size.0;
    let height = size.1;

    // flip image vertically (texture is upside down)
    let orig_pixels = data.clone();
    let stride = width as usize * 4;
    for y in 0..height as usize {
        let dst_start = y * stride;
        let src_start = (height as usize - y - 1) * stride;
        let src_slice = &orig_pixels[src_start .. src_start + stride];
        (&mut data[dst_start .. dst_start + stride]).clone_from_slice(&src_slice[..stride]);
    }

    let encoder = PNGEncoder::new(File::create("out.png").unwrap());
    encoder.encode(&data[..],
                   width,
                   height,
                   ColorType::RGBA(8)).expect("Unable to encode PNG!");
}
