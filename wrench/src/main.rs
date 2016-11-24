/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate app_units;
extern crate byteorder;
extern crate bincode;
extern crate webrender;
extern crate glutin;
extern crate gleam;
extern crate webrender_traits;
extern crate euclid;
extern crate yaml_rust;
extern crate time;
extern crate image;
#[macro_use]
extern crate clap;
#[macro_use]
extern crate lazy_static;
extern crate serde;
extern crate serde_json;

#[cfg(target_os = "windows")]
extern crate dwrote;
#[cfg(target_os = "linux")]
extern crate font_loader;

use euclid::Size2D;
use glutin::{ElementState, VirtualKeyCode};
use std::path::PathBuf;
use std::cmp::{min, max};
use webrender_traits::*;

mod wrench;
use wrench::{Wrench, WrenchThing};

mod yaml_helper;

mod yaml_frame_reader;
use yaml_frame_reader::YamlFrameReader;

mod yaml_frame_writer;
mod json_frame_writer;

mod binary_frame_reader;
use binary_frame_reader::BinaryFrameReader;

lazy_static! {
    static ref PLATFORM_DEFAULT_FACE_NAME: String =
        if cfg!(target_os = "windows") {
            String::from("Arial")
        } else {
            String::from("Helvetica")
        };

    static ref WHITE_COLOR: ColorF = ColorF::new(1.0, 1.0, 1.0, 1.0);
    static ref BLACK_COLOR: ColorF = ColorF::new(0.0, 0.0, 0.0, 1.0);
}

pub static mut CURRENT_FRAME_NUMBER: u32 = 0;

enum ThingKind {
    YamlFile(YamlFrameReader),
    BinaryFile(BinaryFrameReader),
}

impl ThingKind {
    fn thing<'a>(&'a mut self) -> &'a mut WrenchThing {
        match *self {
            ThingKind::YamlFile(ref mut f) => &mut *f,
            ThingKind::BinaryFile(ref mut f) => &mut *f,
        }
    }
}

fn main() {
    let args_yaml = load_yaml!("args.yaml");
    let args = clap::App::from_yaml(args_yaml)
        .setting(clap::AppSettings::ArgRequiredElseHelp)
        .get_matches();

    // handle some global arguments
    let res_path = args.value_of("shaders").map(|s| PathBuf::from(s));
    let dp_ratio = args.value_of("dp_ratio").map(|v| v.parse::<f32>().unwrap());
    let save_type = args.value_of("save").map(|s| {
        if s == "yaml" { wrench::SaveType::Yaml }
        else if s == "json" { wrench::SaveType::Json }
        else { panic!("Save type must be json or yaml"); }
    });
    let size = args.value_of("size").map(|s| {
        let x = s.find('x').expect("Size must be specified exactly as widthxheight");
        let w = s[0..x].parse::<u32>().expect("Invalid size width");
        let h = s[x+1..].parse::<u32>().expect("Invalid size height");
        Size2D::new(w, h)
    }).unwrap_or(Size2D::<u32>::new(1920, 1080));

    let mut wrench = Wrench::new(res_path,
                                 dp_ratio,
                                 save_type,
                                 size,
                                 args.is_present("subpixel-aa"),
                                 args.is_present("debug"),
                                 args.is_present("vsync"));

    let mut show_help = false;
    let mut profiler = false;

    let mut done = false;

    let mut thing =
        if let Some(subargs) = args.subcommand_matches("show") {
            ThingKind::YamlFile(YamlFrameReader::new_from_args(subargs))
        } else if let Some(subargs) = args.subcommand_matches("replay") {
            ThingKind::BinaryFile(BinaryFrameReader::new_from_args(subargs))
        } else {
            panic!("Should never have gotten here");
        };

    let mut do_loop = false;
    let mut last = time::SteadyTime::now();
    let mut frame_count = 0;

    let mut min_time = time::Duration::max_value();
    let mut min_min_time = time::Duration::max_value();
    let mut max_time = time::Duration::min_value();
    let mut max_max_time = time::Duration::min_value();
    let mut sum_time = time::Duration::zero();

    while !done {
        let mut thing = thing.thing();

        wrench.update();

        let frame_num = thing.do_frame(&mut wrench);
        unsafe {
            CURRENT_FRAME_NUMBER = frame_num;
        }

        if show_help {
            wrench.show_onscreen_help();
        }

        wrench.render();

        let now = time::SteadyTime::now();
        let dur = now - last;

        min_time = min(min_time, dur);
        min_min_time = min(min_min_time, dur);
        max_time = max(max_time, dur);
        max_max_time = max(max_max_time, dur);
        sum_time = sum_time + dur;

        let as_ms = |f: time::Duration| { f.num_microseconds().unwrap() as f64 / 1000. };

        frame_count += 1;
        if frame_count == 60 {
            let avg_time = sum_time / frame_count;
            println!("{:3.6} [{:3.6} .. {:3.6}]  -- {:4.7} fps  -- (global {:3.6} .. {:3.6})",
                     as_ms(avg_time), as_ms(min_time), as_ms(max_time),
                     1000.0 / as_ms(avg_time), as_ms(min_min_time), as_ms(max_max_time));
            min_time = time::Duration::max_value();
            max_time = time::Duration::min_value();
            sum_time = time::Duration::zero();
            frame_count = 0;
        }

        last = now;

        if do_loop {
            thing.next_frame();
        }

        // process any pending events
        for event in wrench.window.poll_events() {
            match event {
                glutin::Event::Closed => {
                    done = true;
                },
                glutin::Event::KeyboardInput(kstate, _scan_code, maybe_vk) => {
                    if kstate == ElementState::Pressed {
                        if let Some(vk) = maybe_vk {
                            match vk {
                                VirtualKeyCode::Escape | VirtualKeyCode::Q => {
                                    done = true;
                                },
                                VirtualKeyCode::P => {
                                    profiler = !profiler;
                                    wrench.renderer.set_profiler_enabled(profiler);
                                },
                                VirtualKeyCode::L => {
                                    do_loop = !do_loop;
                                },
                                VirtualKeyCode::Left => {
                                    thing.prev_frame();
                                },
                                VirtualKeyCode::Right => {
                                    thing.next_frame();
                                },
                                VirtualKeyCode::H => {
                                    show_help = !show_help;
                                },
                                _ => ()
                            }
                        }
                    }
                }
                _ => ()
            }
        }
    }
}
