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
extern crate crossbeam;

#[cfg(target_os = "windows")]
extern crate dwrote;
#[cfg(target_os = "linux")]
extern crate font_loader;

use gleam::gl;
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

fn percentile(values: &[f64], pct_int: u32) -> f64 {
    let pct = pct_int as f32 / 100.;
    let index_f = (values.len()-1) as f32 * pct;
    let index = f32::floor(index_f) as usize;
    if index == index_f as usize {
        values[index]
    } else {
        (values[index] + values[index+1]) / 2.
    }
}

fn make_window(size: DeviceUintSize,
               dp_ratio: Option<f32>,
               vsync: bool)
               -> glutin::Window
{
        let mut window = glutin::WindowBuilder::new()
            .with_gl(glutin::GlRequest::Specific(glutin::Api::OpenGl, (3, 2)))
            .with_dimensions(size.width, size.height);
        window.opengl.vsync = vsync;
        let window = window.build().unwrap();

        unsafe {
            window.make_current().ok();
            gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);
            gl::clear_color(1.0, 1.0, 1.0, 1.0);
        }

        let gl_version = gl::get_string(gl::VERSION);
        let gl_renderer = gl::get_string(gl::RENDERER);

        let dp_ratio = dp_ratio.unwrap_or(window.hidpi_factor());
        println!("OpenGL version {}, {}", gl_version, gl_renderer);
        println!("hidpi factor: {} (native {})", dp_ratio, window.hidpi_factor());

    window
}

fn main() {
    let args_yaml = load_yaml!("args.yaml");
    let args = clap::App::from_yaml(args_yaml)
        .setting(clap::AppSettings::ArgRequiredElseHelp)
        .get_matches();

    // handle some global arguments
    let res_path = args.value_of("shaders").map(|s| PathBuf::from(s));
    let dp_ratio = args.value_of("dp_ratio").map(|v| v.parse::<f32>().unwrap());
    let limit_seconds = args.value_of("time").map(|s| time::Duration::seconds(s.parse::<i64>().unwrap()));
    let save_type = args.value_of("save").map(|s| {
        if s == "yaml" { wrench::SaveType::Yaml }
        else if s == "json" { wrench::SaveType::Json }
        else { panic!("Save type must be json or yaml"); }
    });
    let size = args.value_of("size").map(|s| {
        if s == "720p" {
            DeviceUintSize::new(1280, 720)
        } else if s == "1080p" {
            DeviceUintSize::new(1920, 1080)
        } else if s == "4k" {
            DeviceUintSize::new(3840, 2160)
        } else {
            let x = s.find('x').expect("Size must be specified exactly as 720p, 1080p, 4k, or widthxheight");
            let w = s[0..x].parse::<u32>().expect("Invalid size width");
            let h = s[x+1..].parse::<u32>().expect("Invalid size height");
            DeviceUintSize::new(w, h)
        }
    }).unwrap_or(DeviceUintSize::new(1920, 1080));

    let mut window = make_window(size, dp_ratio, args.is_present("vsync"));
    let dp_ratio = dp_ratio.unwrap_or(window.hidpi_factor());
    let mut wrench = Wrench::new(&mut window,
                                 res_path,
                                 dp_ratio,
                                 save_type,
                                 size,
                                 args.is_present("rebuild"),
                                 args.is_present("subpixel-aa"),
                                 args.is_present("debug"),
                                 args.is_present("verbose"));

    let mut thing =
        if let Some(subargs) = args.subcommand_matches("show") {
            Box::new(YamlFrameReader::new_from_args(subargs)) as Box<WrenchThing>
        } else if let Some(subargs) = args.subcommand_matches("replay") {
            Box::new(BinaryFrameReader::new_from_args(subargs)) as Box<WrenchThing>
        } else {
            panic!("Should never have gotten here");
        };

    let mut show_help = false;
    let mut profiler = false;
    let mut do_loop = false;

    for _ in 0..thing.queue_frames() {
        let (width, height) = window.get_inner_size().unwrap();
        let dim = DeviceUintSize::new(width, height);
        wrench.update(dim);

        let frame_num = thing.do_frame(&mut wrench);
        unsafe {
            CURRENT_FRAME_NUMBER = frame_num;
        }

        wrench.render();
        window.swap_buffers().ok();
    }

    let time_start = time::SteadyTime::now();
    let mut last = time::SteadyTime::now();
    let mut frame_count = 0;
    let frames_between_dumps = 60;

    let mut min_time = time::Duration::max_value();
    let mut min_min_time = time::Duration::max_value();
    let mut max_time = time::Duration::min_value();
    let mut max_max_time = time::Duration::min_value();
    let mut sum_time = time::Duration::zero();

    let mut block_avg_ms = vec![];
    let mut warmed_up = false;

    fn as_ms(f: time::Duration) -> f64 { f.num_microseconds().unwrap() as f64 / 1000. }
    fn as_fps(f: time::Duration) -> f64 { (1000.*1000.) / f.num_microseconds().unwrap() as f64 }

    for event in window.wait_events() {
        if let Some(window_title) = wrench.take_title() {
            window.set_title(&window_title);
        }

        if let Some(limit) = limit_seconds {
            if (time::SteadyTime::now() - time_start) >= limit {
                let mut block_avg_ms = block_avg_ms.iter().map(|v| as_ms(*v)).collect::<Vec<f64>>();
                block_avg_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let avg_ms = block_avg_ms.iter().fold(0., |sum, v| sum + v) / block_avg_ms.len() as f64;
                let val_10th_pct = percentile(&block_avg_ms, 10);
                let val_90th_pct = percentile(&block_avg_ms, 90);

                println!("-    {:7} {:7} {:7}", "10th", "avg", "90th");
                println!("ms   {:4.3} {:4.3} {:4.3}",
                         val_10th_pct, avg_ms, val_90th_pct);
                println!("fps  {:4.3} {:4.3} {:4.3}",
                         1000. / val_10th_pct, 1000. / avg_ms, 1000. / val_90th_pct);
                break;
            }
        }

        match event {
            glutin::Event::Awakened => {
                let (width, height) = window.get_inner_size().unwrap();
                let dim = DeviceUintSize::new(width, height);
                wrench.update(dim);

                let frame_num = thing.do_frame(&mut wrench);
                unsafe {
                    CURRENT_FRAME_NUMBER = frame_num;
                }

                if show_help {
                    wrench.show_onscreen_help();
                }

                wrench.render();
                window.swap_buffers().ok();

                let now = time::SteadyTime::now();
                let dur = now - last;

                min_time = min(min_time, dur);
                max_time = max(max_time, dur);
                sum_time = sum_time + dur;

                if warmed_up {
                    min_min_time = min(min_min_time, dur);
                    max_max_time = max(max_max_time, dur);
                }

                frame_count += 1;
                if frame_count == frames_between_dumps {
                    let avg_time = sum_time / frame_count;
                    if warmed_up {
                        block_avg_ms.push(avg_time);
                    }

                    if wrench.verbose {
                        if warmed_up {
                            println!("{:3.3} [{:3.3} .. {:3.3}]  -- {:4.3} fps  -- (global {:3.3} .. {:3.3})",
                                    as_ms(avg_time), as_ms(min_time), as_ms(max_time),
                                    as_fps(avg_time), as_ms(min_min_time), as_ms(max_max_time));
                        } else {
                            println!("{:3.3} [{:3.3} .. {:3.3}]  -- {:4.3} fps",
                                    as_ms(avg_time), as_ms(min_time), as_ms(max_time), as_fps(avg_time));
                        }
                    }

                    min_time = time::Duration::max_value();
                    max_time = time::Duration::min_value();
                    sum_time = time::Duration::zero();
                    warmed_up = true;
                    frame_count = 0;
                }

                last = now;

                if do_loop {
                    thing.next_frame();
                }
            }

            glutin::Event::Closed => {
                break;
            }

            glutin::Event::KeyboardInput(kstate, _scan_code, maybe_vk) => {
                if kstate == ElementState::Pressed {
                    if let Some(vk) = maybe_vk {
                        match vk {
                            VirtualKeyCode::Escape | VirtualKeyCode::Q => {
                                break;
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
