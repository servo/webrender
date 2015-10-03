#![feature(test)]

extern crate test;
extern crate euclid;
extern crate webrender;

use test::{black_box, Bencher};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::str::FromStr;

use euclid::{Rect, Point2D, Size2D};

use webrender::bench::{WorkVertex, clip_rect_pos_uv, clip_polygon};

#[bench]
fn bench_clip_rect(b: &mut Bencher) {
    let mut rects = Vec::with_capacity(5000);
    let f = File::open("./benches/rectangles.dat").unwrap();
    let bufread = BufReader::new(f);

    for l in bufread.lines() {
        let line = l.unwrap();
        if line.starts_with('#') {
            // comment lines
            continue;
        }
        let bits = line.split(' ').collect::<Vec<_>>();
        rects.push((new_rect(&bits[0..4]), new_rect(&bits[4..8]), new_rect(&bits[8..12])));
    }

    b.iter(|| {
        for rect in &rects {
            black_box(clip_rect_pos_uv(&rect.0, &rect.1, &rect.2));
        }
    })
}

fn new_rect(arr: &[&str]) -> Rect<f32> {
    Rect::new(Point2D::new(FromStr::from_str(arr[0]).unwrap(), FromStr::from_str(arr[1]).unwrap()),
               Size2D::new(FromStr::from_str(arr[2]).unwrap(), FromStr::from_str(arr[3]).unwrap()))
}

#[bench]
fn bench_clip_poly(b: &mut Bencher) {
    // huge vector of polygons
    let polys = include!("./polygons.in");
    b.iter(|| {
        for poly in &polys {
            black_box(clip_polygon(&poly.0, &poly.1));
        }
    })
}