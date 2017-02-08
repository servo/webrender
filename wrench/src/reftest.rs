/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use WindowWrapper;
use base64;
use gleam::gl;
use image::ColorType;
use image::png::PNGEncoder;
use std::cmp;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::mpsc::{channel, Receiver, Sender};
use webrender_traits::*;
use wrench::{Wrench, WrenchThing};
use yaml_frame_reader::YamlFrameReader;

pub enum ReftestOp {
    Equal,
    NotEqual,
}

pub struct Reftest<'a> {
    op: ReftestOp,
    test: &'a Path,
    reference: &'a Path,
}

struct ReftestImage {
    data: Vec<u8>,
    size: DeviceUintSize,
}
enum ReftestImageComparison {
    Equal,
    NotEqual { max_difference: usize, count_different: usize },
}
impl ReftestImage {
    fn compare(&self, other: &ReftestImage) -> ReftestImageComparison {
        assert!(self.size == other.size);
        assert!(self.data.len() == other.data.len());
        assert!(self.data.len() % 4 == 0);

        let mut count = 0;
        let mut max = 0;

        for (a, b) in self.data.chunks(4).zip(other.data.chunks(4)) {
            if a != b {
                let pixel_max = a.iter()
                                 .zip(b.iter())
                                 .map(|(x, y)| (*x as isize - *y as isize).abs() as usize)
                                 .max().unwrap();

                count += 1;
                max = cmp::max(max, pixel_max);
            }
        }

        if count != 0 {
            ReftestImageComparison::NotEqual {
                max_difference: max,
                count_different: count,
            }
        } else {
            ReftestImageComparison::Equal
        }
    }

    fn create_data_uri(mut self) -> String {
        let width = self.size.width;
        let height = self.size.height;

        // flip image vertically (texture is upside down)
        let orig_pixels = self.data.clone();
        let stride = width as usize * 4;
        for y in 0..height as usize {
            let dst_start = y * stride;
            let src_start = (height as usize - y - 1) * stride;
            let src_slice = &orig_pixels[src_start .. src_start + stride];
            (&mut self.data[dst_start .. dst_start + stride]).clone_from_slice(&src_slice[..stride]);
        }

        let mut png: Vec<u8> = vec![];
        {
            let encoder = PNGEncoder::new(&mut png);
            encoder.encode(&self.data[..],
                            width,
                            height,
                            ColorType::RGBA(8)).expect("Unable to encode PNG!");
        }
        let png_base64 = base64::encode(&png);
        format!("data:image/png;base64,{}", png_base64)
    }
}


fn parse_reftests<F>(manifest: &Path, runner: &mut F)
    where F: FnMut(Reftest)
{
    let dir = manifest.parent().unwrap();
    let f = File::open(manifest).expect(&format!("couldn't open manifest: {}", manifest.display()));
    let file = BufReader::new(&f);
    for line in file.lines() {
        let l = line.unwrap();

        // strip the comments
        let s = &l[0..l.find("#").unwrap_or(l.len())];
        let s = s.trim();
        if s.len() == 0 {
            continue;
        }

        let mut items = s.split_whitespace();

        match items.next() {
            Some("include") => {
                let include = dir.join(items.next().unwrap());
                parse_reftests(include.as_path(), runner);
            }
            Some(x) => {
                let kind = match x {
                    "==" => ReftestOp::Equal,
                    "!=" => ReftestOp::NotEqual,
                    _ => panic!("unexpected match operator"),
                };
                let test = dir.join(items.next().unwrap());
                let reference = dir.join(items.next().unwrap());
                runner(Reftest {
                    op: kind,
                    test: test.as_path(),
                    reference: reference.as_path(),
                });
            }
            _ => panic!(),
        };
    }

}

fn render_yaml(wrench: &mut Wrench,
               window: &mut WindowWrapper,
               filename: &Path,
               rx: &Receiver<()>)
               -> ReftestImage {
    let mut reader = YamlFrameReader::new(filename);
    reader.do_frame(wrench);
    // wait for the frame
    rx.recv().unwrap();
    wrench.render();

    let size = window.get_inner_size();
    let pixels = gl::read_pixels(0,
                                 0,
                                 size.0 as gl::GLsizei,
                                 size.1 as gl::GLsizei,
                                 gl::RGBA,
                                 gl::UNSIGNED_BYTE);
    window.swap_buffers();

    ReftestImage {
        data: pixels,
        size: DeviceUintSize::new(size.0, size.1)
    }
}

pub fn run_reftests(wrench: &mut Wrench, window: &mut WindowWrapper, filename: &str) {
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

    let mut total_passing = 0;
    let mut total_failing = 0;

    parse_reftests(Path::new(filename), &mut |t: Reftest| {
        let name = match t.op {
            ReftestOp::Equal => format!("{} == {}", t.test.display(), t.reference.display()),
            ReftestOp::NotEqual => format!("{} != {}", t.test.display(), t.reference.display()),
        };

        println!("REFTEST {}", name);

        let test = render_yaml(wrench, window, t.test, &rx);
        let reference = render_yaml(wrench, window, t.reference, &rx);
        let comparison = test.compare(&reference);

        let success = match (t.op, comparison) {
            (ReftestOp::Equal, ReftestImageComparison::Equal) => true,
            (ReftestOp::Equal,
             ReftestImageComparison::NotEqual { max_difference, count_different }) => {
                println!("{} | {} | {}: {}, {}: {}",
                         "REFTEST TEST-UNEXPECTED-FAIL", name,
                         "image comparison, max difference", max_difference,
                         "number of differing pixels", count_different);
                println!("REFTEST   IMAGE 1 (TEST): {}", test.create_data_uri());
                println!("REFTEST   IMAGE 2 (REFERENCE): {}", reference.create_data_uri());
                println!("REFTEST TEST-END | {}", name);

                false
            },
            (ReftestOp::NotEqual, ReftestImageComparison::Equal) => {
                println!("REFTEST TEST-UNEXPECTED-FAIL | {} | image comparison", name);
                println!("REFTEST TEST-END | {}", name);

                false
            },
            (ReftestOp::NotEqual, ReftestImageComparison::NotEqual { .. }) => true,
        };

        if success {
            total_passing += 1;
        } else {
            total_failing += 1;
        }
    });

    println!("REFTEST INFO | {} passing, {} failing", total_passing, total_failing);

    if total_failing > 0 {
        panic!();
    }
}
