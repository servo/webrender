use std::io::BufReader;
use std::io::BufRead;
use std::fs::File;
use wrench::{Wrench, WrenchThing};
use std::path::Path;
use gleam::gl;
use std::sync::mpsc::{channel, Sender, Receiver};

use yaml_frame_reader::YamlFrameReader;
use webrender_traits::*;

use WindowWrapper;

pub enum ReftestOp {
    Equal,
    NotEqual,
}

pub struct Reftest<'a> {
    op: ReftestOp,
    test: &'a Path,
    reference: &'a Path,
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
               -> Vec<u8> {
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
    pixels
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
        fn pipeline_size_changed(&mut self, _: PipelineId, _: Option<LayoutSize>) {}
    }
    let (tx, rx) = channel();
    wrench.renderer.set_render_notifier(Box::new(Notifier { tx: tx }));

    parse_reftests(Path::new(filename), &mut |t: Reftest| {
        println!("{} {}", t.test.display(), t.reference.display());
        let test = render_yaml(wrench, window, t.test, &rx);
        let reference = render_yaml(wrench, window, t.reference, &rx);
        match t.op {
            ReftestOp::Equal => assert!(test == reference),
            ReftestOp::NotEqual => assert!(test != reference),
        }
    });
}
