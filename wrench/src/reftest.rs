use std::cmp;
use std::io::BufReader;
use std::io::BufRead;
use std::fs::File;
use wrench::{Wrench, WrenchThing};
use std::path::{Path, PathBuf};
use gleam::gl;
use std::sync::mpsc::{channel, Sender, Receiver};

use base64;
use image::ColorType;
use image::png::PNGEncoder;

use yaml_frame_reader::YamlFrameReader;
use webrender_traits::*;

use WindowWrapper;

pub enum ReftestOp {
    Equal,
    NotEqual,
}
pub struct Reftest {
    op: ReftestOp,
    test: PathBuf,
    reference: PathBuf,
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

            let mut items = s.split_whitespace();

            match items.next() {
                Some("include") => {
                    let include = dir.join(items.next().unwrap());

                    reftests.append(&mut ReftestManifest::new(include.as_path()).reftests);
                }
                Some(x) => {
                    let kind = match x {
                        "==" => ReftestOp::Equal,
                        "!=" => ReftestOp::NotEqual,
                        _ => panic!("unexpected match operator"),
                    };
                    let test = dir.join(items.next().unwrap());
                    let reference = dir.join(items.next().unwrap());
                    reftests.push(Reftest {
                        op: kind,
                        test: test.clone(),
                        reference: reference.clone(),
                    });
                }
                _ => panic!(),
            };
        }

        ReftestManifest {
            reftests: reftests
        }
    }

    fn find(&self, path: &Path) -> Option<&Reftest> {
        self.reftests.iter().find(|x| x.test == path || x.reference == path)
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

    pub fn run(mut self, base_manifest: &Path, specific_reftest: Option<&Path>) {
        let manifest = ReftestManifest::new(base_manifest);
        let mut total_passing = 0;
        let mut total_failing = 0;

        match specific_reftest {
            Some(path) => {
                let specific_reftest = manifest.find(path).expect("can't find reftest in manifest");
                let success = self.run_reftest(specific_reftest);

                if success {
                    total_passing += 1;
                } else {
                    total_failing += 1;
                }
            }
            None => {
                for t in manifest.reftests.iter() {
                    let success = self.run_reftest(&t);

                    if success {
                        total_passing += 1;
                    } else {
                        total_failing += 1;
                    }
                }
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

        let success = match (&t.op, comparison) {
            (&ReftestOp::Equal, ReftestImageComparison::Equal) => true,
            (&ReftestOp::Equal, ReftestImageComparison::NotEqual { max_difference, count_different }) => {
                println!("REFTEST TEST-UNEXPECTED-FAIL | {} | image comparison, max difference: {}, number of differing pixels: {}",
                         name,
                         max_difference,
                         count_different);
                println!("REFTEST   IMAGE 1 (TEST): {}", test.create_data_uri());
                println!("REFTEST   IMAGE 2 (REFERENCE): {}", reference.create_data_uri());
                println!("REFTEST TEST-END | {}", name);

                false
            },
            (&ReftestOp::NotEqual, ReftestImageComparison::Equal) => {
                println!("REFTEST TEST-UNEXPECTED-FAIL | {} | image comparison", name);
                println!("REFTEST TEST-END | {}", name);

                false
            },
            (&ReftestOp::NotEqual, ReftestImageComparison::NotEqual{..}) => true,
        };

        success
    }

    fn render_yaml(&mut self, filename: &Path) -> ReftestImage {
        let mut reader = YamlFrameReader::new(filename);
        reader.do_frame(self.wrench);

        // wait for the frame
        self.rx.recv().unwrap();
        self.wrench.render();

        let size = self.window.get_inner_size();
        let pixels = gl::read_pixels(0,
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
