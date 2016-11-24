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
extern crate image;
#[macro_use]
extern crate clap;
#[macro_use]
extern crate lazy_static;
extern crate serde;
extern crate serde_json;

#[cfg(target_os = "windows")]
extern crate dwrote;

use glutin::{ElementState, VirtualKeyCode};
use std::path::PathBuf;
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
    let size = args.value_of("size");
    let save_type = args.value_of("save").map(|s| {
        if s == "yaml" { wrench::SaveType::Yaml }
        else if s == "json" { wrench::SaveType::Json }
        else { panic!("Save type must be json or yaml"); }
    });

    let mut wrench = Wrench::new(res_path, dp_ratio,
                                 size,
                                 save_type,
                                 args.is_present("subpixel-aa"),
                                 args.is_present("debug"));

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
