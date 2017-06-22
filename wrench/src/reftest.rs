/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use WindowWrapper;
use base64;
use image::load as load_piston_image;
use image::png::PNGEncoder;
use image::{ColorType, ImageFormat};
use parse_function::parse_function;
use png::save_flipped;
use std::cmp;
use std::fmt::{Display, Error, Formatter};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use webrender_traits::*;
use wrench::{Wrench, WrenchThing};
use yaml_frame_reader::YamlFrameReader;

pub struct ReftestOptions {
    // These override values that are lower.
    pub allow_max_difference: usize,
    pub allow_num_differences: usize,
}

impl ReftestOptions {
    pub fn default() -> Self {
        ReftestOptions {
            allow_max_difference: 0,
            allow_num_differences: 0,
        }
    }
}

pub enum ReftestOp {
    Equal,
    NotEqual,
}

impl Display for ReftestOp {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "{}",
               match *self {
                   ReftestOp::Equal => "==".to_owned(),
                   ReftestOp::NotEqual => "!=".to_owned(),
                })
    }
}

pub struct Reftest {
    op: ReftestOp,
    test: PathBuf,
    reference: PathBuf,
    max_difference: usize,
    num_differences: usize,
}

impl Display for Reftest {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "{} {} {}", self.test.display(), self.op, self.reference.display())
    }
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
        assert_eq!(self.size, other.size);
        assert_eq!(self.data.len(), other.data.len());
        assert_eq!(self.data.len() % 4, 0);

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
    fn new(manifest: &Path, options: &ReftestOptions) -> ReftestManifest {
        let dir = manifest.parent().unwrap();
        let f = File::open(manifest).expect(&format!("couldn't open manifest: {}", manifest.display()));
        let file = BufReader::new(&f);

        let mut reftests = Vec::new();

        for line in file.lines() {
            let l = line.unwrap();

            // strip the comments
            let s = &l[0..l.find('#').unwrap_or(l.len())];
            let s = s.trim();
            if s.is_empty() {
                continue;
            }

            let items: Vec<&str> = s.split_whitespace().collect();

            match items[0] {
                "include" => {
                    let include = dir.join(items[1]);

                    reftests.append(&mut ReftestManifest::new(include.as_path(), options).reftests);
                }
                item_str => {
                    // If the first item is "fuzzy(<val>,<count>)" the positions of the operator
                    // and paths in the array are offset.
                    // TODO: This is simple but not great because it does not support having spaces
                    // in the fuzzy syntax, like between the arguments.
                    let (max, count, offset) =  if item_str.starts_with("fuzzy(") {
                        let (_, args, _) = parse_function(item_str);
                        (args[0].parse().unwrap(),  args[1].parse().unwrap(), 1)
                    } else {
                        (0, 0, 0)
                    };

                    reftests.push(Reftest {
                        op: parse_operator(items[offset]).expect("unexpected match operator"),
                        test: dir.join(items[offset + 1]),
                        reference: dir.join(items[offset + 2]),
                        max_difference: cmp::max(max, options.allow_max_difference),
                        num_differences: cmp::max(count, options.allow_num_differences),
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
            wrench,
            window,
            rx,
        }
    }

    pub fn run(mut self, base_manifest: &Path, reftests: Option<&Path>, options: &ReftestOptions) {
        let manifest = ReftestManifest::new(base_manifest, options);
        let reftests = manifest.find(reftests.unwrap_or(&PathBuf::new()));

        let mut total_passing = 0;
        let mut failing = Vec::new();

        for t in reftests {
            if self.run_reftest(t) {
                total_passing += 1;
            } else {
                failing.push(t);
            }
        }

        println!("REFTEST INFO | {} passing, {} failing", total_passing, failing.len());

        if !failing.is_empty() {
            println!("\nReftests with unexpected results:");

            for reftest in &failing {
                println!("\t{}", reftest);
            }
        }

        // panic here so that we fail CI
        assert!(failing.is_empty());
    }

    fn run_reftest(&mut self, t: &Reftest) -> bool {
        println!("REFTEST {}", t);

        let window_size = DeviceUintSize::new(self.window.get_inner_size_pixels().0,
                                              self.window.get_inner_size_pixels().1);
        let reference = match t.reference.extension().unwrap().to_str().unwrap() {
            "yaml" => self.render_yaml(t.reference.as_path(), window_size),
            "png" => self.load_image(t.reference.as_path(), ImageFormat::PNG),
            other => panic!("Unknown reftest extension: {}", other),
        };
        // the reference can be smaller than the window size,
        // in which case we only compare the intersection
        let test = self.render_yaml(t.test.as_path(), reference.size);
        let comparison = test.compare(&reference);

        match (&t.op, comparison) {
            (&ReftestOp::Equal, ReftestImageComparison::Equal) => true,
            (&ReftestOp::Equal,
              ReftestImageComparison::NotEqual { max_difference, count_different }) => {
                if max_difference > t.max_difference || count_different > t.num_differences {
                    println!("{} | {} | {}: {}, {}: {}",
                             "REFTEST TEST-UNEXPECTED-FAIL", t,
                             "image comparison, max difference", max_difference,
                             "number of differing pixels", count_different);
                    println!("REFTEST   IMAGE 1 (TEST): {}", test.create_data_uri());
                    println!("REFTEST   IMAGE 2 (REFERENCE): {}", reference.create_data_uri());
                    println!("REFTEST TEST-END | {}", t);

                    false
                } else {
                    true
                }
            },
            (&ReftestOp::NotEqual, ReftestImageComparison::Equal) => {
                println!("REFTEST TEST-UNEXPECTED-FAIL | {} | image comparison", t);
                println!("REFTEST TEST-END | {}", t);

                false
            },
            (&ReftestOp::NotEqual, ReftestImageComparison::NotEqual { .. }) => true,
        }
    }

    fn load_image(&mut self, filename: &Path, format: ImageFormat) -> ReftestImage {
        let file = BufReader::new(File::open(filename).unwrap());
        let img_raw = load_piston_image(file, format).unwrap();
        let img = img_raw.flipv().to_rgba();
        let size = img.dimensions();
        ReftestImage {
            data: img.into_raw(),
            size: DeviceUintSize::new(size.0, size.1)
        }
    }

    fn render_yaml(&mut self, filename: &Path, size: DeviceUintSize) -> ReftestImage {
        let mut reader = YamlFrameReader::new(filename);
        reader.do_frame(self.wrench);

        // wait for the frame
        self.rx.recv().unwrap();
        self.wrench.render();

        let window_size = self.window.get_inner_size_pixels();
        assert!(size.width <= window_size.0 && size.height <= window_size.1);

        // taking the bottom left sub-rectangle
        let rect = DeviceUintRect::new(DeviceUintPoint::new(0, window_size.1 - size.height),
                                       size);
        let pixels = self.wrench.renderer.read_pixels_rgba8(rect);
        self.window.swap_buffers();

        let write_debug_images = false;
        if write_debug_images {
            let debug_path = filename.with_extension("yaml.png");
            save_flipped(debug_path, &pixels, size);
        }

        ReftestImage {
            data: pixels,
            size,
        }
    }
}
