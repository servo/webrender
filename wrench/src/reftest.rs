use std::io::BufReader;
use std::io::BufRead;
use std::fs::File;
use wrench::{Wrench, WrenchThing};
use std::path::Path;
use gleam::gl;

use yaml_frame_reader::YamlFrameReader;

use glutin;
use WindowWrapper;

pub enum ReftestOp {
    Equal,
    NotEqual
}

pub struct Reftest<'a> {
    op: ReftestOp,
    test: &'a str,
    reference: &'a str
}

pub fn parse_reftests<F>(filename: &str, mut runner: F) where F: FnMut(Reftest)
{
    let f = File::open(filename).unwrap();
    let file = BufReader::new(&f);
    for line in file.lines() {
        let l = line.unwrap();
        if l.starts_with("#") {
            continue;
        }

        // strip the comments
        let s = &l[0..l.find("#").unwrap_or(l.len())];
        let s = s.trim();
        if l.len() == 0 {
            continue;
        }
        let mut items = s.split_whitespace();
        let kind = match items.next() {
            Some("==") => ReftestOp::Equal,
            Some("!=") => ReftestOp::NotEqual,
            _ => panic!()
        };
        let test = items.next().unwrap();
        let reference = items.next().unwrap();
        runner(Reftest{op: kind, test: test, reference: reference});
    }

}

fn render_yaml(wrench: &mut Wrench, window: &mut WindowWrapper, filename: &str) -> Vec<u8>
{
    let mut reader = YamlFrameReader::new(Path::new(filename));
    reader.do_frame(wrench);
    wrench.render();
    window.swap_buffers();
    let size = window.get_inner_size();
    gl::read_pixels(0, 0,
                    size.0 as gl::GLsizei,
                    size.1 as gl::GLsizei,
                    gl::RGB, gl::UNSIGNED_BYTE)
}

pub fn run_reftests(wrench: &mut Wrench, window: &mut WindowWrapper, filename: &str)
{
    parse_reftests(filename, |t: Reftest|
                   {
                       println!("{} {}", t.test, t.reference);
                       let test = render_yaml(wrench, window, t.test);
                       let reference = render_yaml(wrench, window, t.reference);
                       match t.op {
                           ReftestOp::Equal => assert!(test == reference),
                           ReftestOp::NotEqual => assert!(test != reference)
                       }
                   });
}
