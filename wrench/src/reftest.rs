/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use WindowWrapper;
use base64;
use gleam::gl;
use image::ColorType;
use image::png::PNGEncoder;
use parse_function::parse_function;
use std::cmp;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use webrender_traits::*;
use wrench::{Wrench, WrenchThing};
use yaml_frame_reader::YamlFrameReader;

pub enum ReftestOp {
    Equal,
    NotEqual,
}
pub struct Reftest {
    op: ReftestOp,
    test: PathBuf,
    reference: PathBuf,
    max_difference: usize,
    num_differences: usize,
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

struct ReftestManifest {
    reftests: Vec<Reftest>,
}
impl ReftestManifest {
    fn new(manifest: &Path) -> ReftestManifest {
        let dir = manifest.parent().unwrap();
        let f = File::open(manifest).expect(&format!("couldn't open manifest: {}", manifest.display()));
        let file = BufReader::new(&f);

        let mut reftests = Vec::new();

        for line in file.lines() {
            let l = line.unwrap();

            // strip the comments
            let s = &l[0..l.find("#").unwrap_or(l.len())];
            let s = s.trim();
            if s.len() == 0 {
                continue;
            }

            let items: Vec<&str> = s.split_whitespace().collect();

            match items[0] {
                "include" => {
                    let include = dir.join(items[1]);

                    reftests.append(&mut ReftestManifest::new(include.as_path()).reftests);
                }
                item_str => {
                    // If the first item is "fuzzy(<val>,<count>)" the positions of the operator
                    // and paths in the array are offset.
                    // TODO: This is simple but not great because it does not support having spaces
                    // in the fuzzy syntax, like between the arguments.
                    let (max, count, offset) =  if item_str.starts_with("fuzzy(") {
                        let (_, args) = parse_function(item_str);
                        (args[0].parse().unwrap(),  args[1].parse().unwrap(), 1)
                    } else {
                        (0, 0, 0)
                    };
                    reftests.push(Reftest {
                        op: parse_operator(items[offset]).expect("unexpected match operator"),
                        test: dir.join(items[offset + 1]),
                        reference: dir.join(items[offset + 2]),
                        max_difference: max,
                        num_differences: count,
                    });
                }
            };
        }

        ReftestManifest {
            reftests: reftests
        }
    }

    fn find(&self, prefix: &Path) -> Vec<&Reftest> {
        self.reftests.iter().filter(|x| {
            x.test.starts_with(prefix) || x.reference.starts_with(prefix)
        }).collect()
    }
}

fn parse_operator(op_str: &str) -> Option<ReftestOp> {
    match op_str {
        "==" => Some(ReftestOp::Equal),
        "!=" => Some(ReftestOp::NotEqual),
        _ => None,
    }
}

pub struct ReftestHarness<'a> {
    wrench: &'a mut Wrench,
    window: &'a mut WindowWrapper,
    rx: Receiver<()>,
}
impl<'a> ReftestHarness<'a> {
    pub fn new(wrench: &'a mut Wrench,
               window: &'a mut WindowWrapper) -> ReftestHarness<'a>
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

        ReftestHarness {
            wrench: wrench,
            window: window,
            rx: rx,
        }
    }

    pub fn run(mut self, base_manifest: &Path, reftests: Option<&Path>) {
        let manifest = ReftestManifest::new(base_manifest);
        let reftests = manifest.find(reftests.unwrap_or(&PathBuf::new()));

        let mut total_passing = 0;
        let mut total_failing = 0;

        for t in reftests {
            if self.run_reftest(&t) {
                total_passing += 1;
            } else {
                total_failing += 1;
            }
        }

        println!("REFTEST INFO | {} passing, {} failing", total_passing, total_failing);

        if total_failing > 0 {
            // panic here so that we fail CI
            panic!();
        }
    }

    fn run_reftest(&mut self, t: &Reftest) -> bool {
        let name = match t.op {
            ReftestOp::Equal => format!("{} == {}", t.test.display(), t.reference.display()),
            ReftestOp::NotEqual => format!("{} != {}", t.test.display(), t.reference.display()),
        };

        println!("REFTEST {}", name);

        let test = self.render_yaml(t.test.as_path());
        let reference = self.render_yaml(t.reference.as_path());
        let comparison = test.compare(&reference);

        match (&t.op, comparison) {
            (&ReftestOp::Equal, ReftestImageComparison::Equal) => true,
            (&ReftestOp::Equal,
              ReftestImageComparison::NotEqual { max_difference, count_different }) => {
                if max_difference > t.max_difference || count_different > t.num_differences {
                    println!("{} | {} | {}: {}, {}: {}",
                             "REFTEST TEST-UNEXPECTED-FAIL", name,
                             "image comparison, max difference", max_difference,
                             "number of differing pixels", count_different);
                    println!("REFTEST   IMAGE 1 (TEST): {}", test.create_data_uri());
                    println!("REFTEST   IMAGE 2 (REFERENCE): {}", reference.create_data_uri());
                    println!("REFTEST TEST-END | {}", name);

                    false
                } else {
                    true
                }
            },
            (&ReftestOp::NotEqual, ReftestImageComparison::Equal) => {
                println!("REFTEST TEST-UNEXPECTED-FAIL | {} | image comparison", name);
                println!("REFTEST TEST-END | {}", name);

                false
            },
            (&ReftestOp::NotEqual, ReftestImageComparison::NotEqual { .. }) => true,
        }
    }

    fn render_yaml(&mut self, filename: &Path) -> ReftestImage {
        let mut reader = YamlFrameReader::new(filename);
        reader.do_frame(self.wrench);

        // wait for the frame
        self.rx.recv().unwrap();
        self.wrench.render();

        let size = self.window.get_inner_size_pixels();
        let pixels = self.window.gl().read_pixels(0,
                                                  0,
                                                  size.0 as gl::GLsizei,
                                                  size.1 as gl::GLsizei,
                                                  gl::RGBA,
                                                  gl::UNSIGNED_BYTE);
        self.window.swap_buffers();

        ReftestImage {
            data: pixels,
            size: DeviceUintSize::new(size.0, size.1)
        }
    }
}
