/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate app_units;
extern crate base64;
extern crate bincode;
extern crate byteorder;
#[macro_use]
extern crate clap;
extern crate crossbeam;
#[cfg(target_os = "windows")]
extern crate dwrote;
#[cfg(feature = "logging")]
extern crate env_logger;
extern crate euclid;
#[cfg(any(target_os = "linux", target_os = "macos"))]
extern crate font_loader;
extern crate gleam;
extern crate glutin;
extern crate image;
#[macro_use]
extern crate lazy_static;
#[cfg(feature = "headless")]
extern crate osmesa_sys;
extern crate ron;
#[macro_use]
extern crate serde;
extern crate serde_json;
extern crate time;
extern crate webrender;
extern crate yaml_rust;

mod binary_frame_reader;
mod blob;
mod json_frame_writer;
mod parse_function;
mod perf;
mod png;
mod premultiply;
mod rawtest;
mod reftest;
mod ron_frame_writer;
mod scene;
mod wrench;
mod yaml_frame_reader;
mod yaml_frame_writer;
mod yaml_helper;

use binary_frame_reader::BinaryFrameReader;
use gleam::gl;
use glutin::{ElementState, VirtualKeyCode, WindowProxy};
use perf::PerfHarness;
use png::save_flipped;
use rawtest::RawtestHarness;
use reftest::{ReftestHarness, ReftestOptions};
use std::cmp::{max, min};
#[cfg(feature = "headless")]
use std::ffi::CString;
#[cfg(feature = "headless")]
use std::mem;
use std::os::raw::c_void;
use std::path::{Path, PathBuf};
use std::ptr;
use std::rc::Rc;
use webrender::api::*;
use wrench::{Wrench, WrenchThing};
use yaml_frame_reader::YamlFrameReader;

lazy_static! {
    static ref PLATFORM_DEFAULT_FACE_NAME: String = String::from("Arial");
    static ref WHITE_COLOR: ColorF = ColorF::new(1.0, 1.0, 1.0, 1.0);
    static ref BLACK_COLOR: ColorF = ColorF::new(0.0, 0.0, 0.0, 1.0);
}

pub static mut CURRENT_FRAME_NUMBER: u32 = 0;

fn percentile(values: &[f64], pct_int: u32) -> f64 {
    if !values.is_empty() {
        let index_big = (values.len() - 1) * (pct_int as usize);
        let index = index_big / 100;
        if index * 100 == index_big {
            values[index]
        } else {
            (values[index] + values[index + 1]) / 2.
        }
    } else {
        1.0
    }
}

#[cfg(feature = "headless")]
pub struct HeadlessContext {
    width: u32,
    height: u32,
    _context: osmesa_sys::OSMesaContext,
    _buffer: Vec<u32>,
}

#[cfg(not(feature = "headless"))]
pub struct HeadlessContext {
    width: u32,
    height: u32,
}

impl HeadlessContext {
    #[cfg(feature = "headless")]
    fn new(width: u32, height: u32) -> HeadlessContext {
        let mut attribs = Vec::new();

        attribs.push(osmesa_sys::OSMESA_PROFILE);
        attribs.push(osmesa_sys::OSMESA_CORE_PROFILE);
        attribs.push(osmesa_sys::OSMESA_CONTEXT_MAJOR_VERSION);
        attribs.push(3);
        attribs.push(osmesa_sys::OSMESA_CONTEXT_MINOR_VERSION);
        attribs.push(3);
        attribs.push(osmesa_sys::OSMESA_DEPTH_BITS);
        attribs.push(24);
        attribs.push(0);

        let context =
            unsafe { osmesa_sys::OSMesaCreateContextAttribs(attribs.as_ptr(), ptr::null_mut()) };

        assert!(!context.is_null());

        let mut buffer = vec![0; (width * height) as usize];

        unsafe {
            let ret = osmesa_sys::OSMesaMakeCurrent(
                context,
                buffer.as_mut_ptr() as *mut _,
                gl::UNSIGNED_BYTE,
                width as i32,
                height as i32,
            );
            assert!(ret != 0);
        };

        HeadlessContext {
            width,
            height,
            _context: context,
            _buffer: buffer,
        }
    }

    #[cfg(not(feature = "headless"))]
    fn new(width: u32, height: u32) -> HeadlessContext {
        HeadlessContext { width, height }
    }

    #[cfg(feature = "headless")]
    fn get_proc_address(s: &str) -> *const c_void {
        let c_str = CString::new(s).expect("Unable to create CString");
        unsafe { mem::transmute(osmesa_sys::OSMesaGetProcAddress(c_str.as_ptr())) }
    }

    #[cfg(not(feature = "headless"))]
    fn get_proc_address(_: &str) -> *const c_void {
        ptr::null() as *const _
    }
}

pub enum WindowWrapper {
    Window(glutin::Window, Rc<gl::Gl>),
    Headless(HeadlessContext, Rc<gl::Gl>),
}

pub struct HeadlessEventIterater;

impl WindowWrapper {
    fn swap_buffers(&self) {
        match *self {
            WindowWrapper::Window(ref window, _) => window.swap_buffers().unwrap(),
            WindowWrapper::Headless(..) => {}
        }
    }

    fn get_inner_size_pixels(&self) -> (u32, u32) {
        match *self {
            WindowWrapper::Window(ref window, _) => window.get_inner_size_pixels().unwrap(),
            WindowWrapper::Headless(ref context, _) => (context.width, context.height),
        }
    }

    fn hidpi_factor(&self) -> f32 {
        match *self {
            WindowWrapper::Window(ref window, _) => window.hidpi_factor(),
            WindowWrapper::Headless(..) => 1.0,
        }
    }

    fn create_window_proxy(&mut self) -> Option<WindowProxy> {
        match *self {
            WindowWrapper::Window(ref window, _) => Some(window.create_window_proxy()),
            WindowWrapper::Headless(..) => None,
        }
    }

    fn set_title(&mut self, title: &str) {
        match *self {
            WindowWrapper::Window(ref window, _) => window.set_title(title),
            WindowWrapper::Headless(..) => (),
        }
    }

    pub fn gl(&self) -> &gl::Gl {
        match *self {
            WindowWrapper::Window(_, ref gl) | WindowWrapper::Headless(_, ref gl) => &**gl,
        }
    }

    pub fn clone_gl(&self) -> Rc<gl::Gl> {
        match *self {
            WindowWrapper::Window(_, ref gl) | WindowWrapper::Headless(_, ref gl) => gl.clone(),
        }
    }
}

fn make_window(
    size: DeviceUintSize,
    dp_ratio: Option<f32>,
    vsync: bool,
    headless: bool,
) -> WindowWrapper {
    let wrapper = if headless {
        let gl = match gl::GlType::default() {
            gl::GlType::Gl => unsafe {
                gl::GlFns::load_with(|symbol| {
                    HeadlessContext::get_proc_address(symbol) as *const _
                })
            },
            gl::GlType::Gles => unsafe {
                gl::GlesFns::load_with(|symbol| {
                    HeadlessContext::get_proc_address(symbol) as *const _
                })
            },
        };
        WindowWrapper::Headless(HeadlessContext::new(size.width, size.height), gl)
    } else {
        let mut window = glutin::WindowBuilder::new()
            .with_gl(glutin::GlRequest::GlThenGles {
                opengl_version: (3, 2),
                opengles_version: (3, 1),
            })
            .with_dimensions(size.width, size.height);
        window.opengl.vsync = vsync;
        let window = window.build().unwrap();
        unsafe {
            window
                .make_current()
                .expect("unable to make context current!");
        }
        let gl = match gl::GlType::default() {
            gl::GlType::Gl => unsafe {
                gl::GlFns::load_with(|symbol| window.get_proc_address(symbol) as *const _)
            },
            gl::GlType::Gles => unsafe {
                gl::GlesFns::load_with(|symbol| window.get_proc_address(symbol) as *const _)
            },
        };
        WindowWrapper::Window(window, gl)
    };

    wrapper.gl().clear_color(0.3, 0.0, 0.0, 1.0);

    let gl_version = wrapper.gl().get_string(gl::VERSION);
    let gl_renderer = wrapper.gl().get_string(gl::RENDERER);

    let dp_ratio = dp_ratio.unwrap_or(wrapper.hidpi_factor());
    println!("OpenGL version {}, {}", gl_version, gl_renderer);
    println!(
        "hidpi factor: {} (native {})",
        dp_ratio,
        wrapper.hidpi_factor()
    );

    wrapper
}

fn main() {
    #[cfg(feature = "logging")]
    env_logger::init().unwrap();

    let args_yaml = load_yaml!("args.yaml");
    let args = clap::App::from_yaml(args_yaml)
        .setting(clap::AppSettings::ArgRequiredElseHelp)
        .get_matches();

    // handle some global arguments
    let res_path = args.value_of("shaders").map(|s| PathBuf::from(s));
    let dp_ratio = args.value_of("dp_ratio").map(|v| v.parse::<f32>().unwrap());
    let limit_seconds = args.value_of("time")
        .map(|s| time::Duration::seconds(s.parse::<i64>().unwrap()));
    let save_type = args.value_of("save").map(|s| match s {
        "yaml" => wrench::SaveType::Yaml,
        "json" => wrench::SaveType::Json,
        "ron" => wrench::SaveType::Ron,
        "binary" => wrench::SaveType::Binary,
        _ => panic!("Save type must be json, ron, yaml, or binary")
    });
    let size = args.value_of("size")
        .map(|s| if s == "720p" {
            DeviceUintSize::new(1280, 720)
        } else if s == "1080p" {
            DeviceUintSize::new(1920, 1080)
        } else if s == "4k" {
            DeviceUintSize::new(3840, 2160)
        } else {
            let x = s.find('x').expect(
                "Size must be specified exactly as 720p, 1080p, 4k, or widthxheight",
            );
            let w = s[0 .. x].parse::<u32>().expect("Invalid size width");
            let h = s[x + 1 ..].parse::<u32>().expect("Invalid size height");
            DeviceUintSize::new(w, h)
        })
        .unwrap_or(DeviceUintSize::new(1920, 1080));
    let is_headless = args.is_present("headless");

    let mut window = make_window(size, dp_ratio, args.is_present("vsync"), is_headless);
    let dp_ratio = dp_ratio.unwrap_or(window.hidpi_factor());
    let (width, height) = window.get_inner_size_pixels();
    let dim = DeviceUintSize::new(width, height);
    let mut wrench = Wrench::new(
        &mut window,
        res_path,
        dp_ratio,
        save_type,
        dim,
        args.is_present("rebuild"),
        args.is_present("no_subpixel_aa"),
        args.is_present("debug"),
        args.is_present("verbose"),
        args.is_present("no_scissor"),
        args.is_present("no_batch"),
    );

    let mut thing = if let Some(subargs) = args.subcommand_matches("show") {
        Box::new(YamlFrameReader::new_from_args(subargs)) as Box<WrenchThing>
    } else if let Some(subargs) = args.subcommand_matches("replay") {
        Box::new(BinaryFrameReader::new_from_args(subargs)) as Box<WrenchThing>
    } else if let Some(subargs) = args.subcommand_matches("png") {
        let reader = YamlFrameReader::new_from_args(subargs);
        png::png(&mut wrench, &mut window, reader);
        wrench.renderer.deinit();
        return;
    } else if let Some(subargs) = args.subcommand_matches("reftest") {
        let (w, h) = window.get_inner_size_pixels();
        let harness = ReftestHarness::new(&mut wrench, &mut window);
        let base_manifest = Path::new("reftests/reftest.list");
        let specific_reftest = subargs.value_of("REFTEST").map(|x| Path::new(x));
        let mut reftest_options = ReftestOptions::default();
        if let Some(allow_max_diff) = subargs.value_of("fuzz_tolerance") {
            reftest_options.allow_max_difference = allow_max_diff.parse().unwrap_or(1);
            reftest_options.allow_num_differences = w as usize * h as usize;
        }
        harness.run(base_manifest, specific_reftest, &reftest_options);
        return;
    } else if let Some(_) = args.subcommand_matches("rawtest") {
        {
            let harness = RawtestHarness::new(&mut wrench, &mut window);
            harness.run();
        }
        wrench.renderer.deinit();
        return;
    } else if let Some(subargs) = args.subcommand_matches("perf") {
        // Perf mode wants to benchmark the total cost of drawing
        // a new displaty list each frame.
        wrench.rebuild_display_lists = true;
        let harness = PerfHarness::new(&mut wrench, &mut window);
        let base_manifest = Path::new("benchmarks/benchmarks.list");
        let filename = subargs.value_of("filename").unwrap();
        harness.run(base_manifest, filename);
        return;
    } else if let Some(subargs) = args.subcommand_matches("compare_perf") {
        let first_filename = subargs.value_of("first_filename").unwrap();
        let second_filename = subargs.value_of("second_filename").unwrap();
        perf::compare(first_filename, second_filename);
        return;
    } else {
        panic!("Should never have gotten here! {:?}", args);
    };

    let mut show_help = false;
    let mut do_loop = false;

    let queue_frames = thing.queue_frames();
    for _ in 0 .. queue_frames {
        let (width, height) = window.get_inner_size_pixels();
        let dim = DeviceUintSize::new(width, height);
        wrench.update(dim);

        let frame_num = thing.do_frame(&mut wrench);
        unsafe {
            CURRENT_FRAME_NUMBER = frame_num;
        }

        wrench.render();
        window.swap_buffers();
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
    let mut block_avg_time = vec![];
    let mut warmed_up = false;

    fn as_ms(f: time::Duration) -> f64 {
        f.num_microseconds().unwrap() as f64 / 1000.
    }
    fn as_fps(f: time::Duration) -> f64 {
        (1000. * 1000.) / f.num_microseconds().unwrap() as f64
    }

    'outer: loop {
        if let Some(window_title) = wrench.take_title() {
            window.set_title(&window_title);
        }

        if let Some(limit) = limit_seconds {
            if (time::SteadyTime::now() - time_start) >= limit {
                let mut block_avg_ms = block_avg_time
                    .iter()
                    .map(|v| as_ms(*v))
                    .collect::<Vec<f64>>();
                block_avg_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let avg_ms =
                    block_avg_ms.iter().fold(0., |sum, v| sum + v) / block_avg_ms.len() as f64;
                let val_10th_pct = percentile(&block_avg_ms, 10);
                let val_90th_pct = percentile(&block_avg_ms, 90);

                println!("-    {:7} {:7} {:7}", "10th", "avg", "90th");
                println!(
                    "ms   {:4.3} {:4.3} {:4.3}",
                    val_10th_pct,
                    avg_ms,
                    val_90th_pct
                );
                println!(
                    "fps  {:4.3} {:4.3} {:4.3}",
                    1000. / val_10th_pct,
                    1000. / avg_ms,
                    1000. / val_90th_pct
                );
                break;
            }
        }

        let event = match window {
            WindowWrapper::Headless(..) => glutin::Event::Awakened,
            WindowWrapper::Window(ref window, _) => window.wait_events().next().unwrap(),
        };

        if let Some(limit) = limit_seconds {
            if (time::SteadyTime::now() - time_start) >= limit {
                break 'outer;
            }
        }

        match event {
            glutin::Event::Awakened => {
                let (width, height) = window.get_inner_size_pixels();
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
                window.swap_buffers();

                let now = time::SteadyTime::now();
                let dur = now - last;

                min_time = min(min_time, dur);
                min_min_time = min(min_min_time, dur);
                max_time = max(max_time, dur);
                max_max_time = max(max_max_time, dur);
                sum_time = sum_time + dur;

                if warmed_up {
                    min_min_time = min(min_min_time, dur);
                    max_max_time = max(max_max_time, dur);
                }

                frame_count += 1;
                if frame_count == frames_between_dumps {
                    let avg_time = sum_time / frame_count;
                    if warmed_up {
                        block_avg_time.push(avg_time);
                    }

                    if wrench.verbose {
                        print!(
                            "{:3.3} [{:3.3} .. {:3.3}]  -- {:4.3} fps",
                            as_ms(avg_time),
                            as_ms(min_time),
                            as_ms(max_time),
                            as_fps(avg_time)
                        );
                        if warmed_up {
                            println!(
                                "  -- (global {:3.3} .. {:3.3})",
                                as_ms(min_min_time),
                                as_ms(max_max_time)
                            );
                        } else {
                            println!("");
                        }
                    }

                    min_time = time::Duration::max_value();
                    max_time = time::Duration::min_value();
                    sum_time = time::Duration::zero();
                    frame_count = 0;
                    warmed_up = true;
                }

                last = now;

                if do_loop {
                    thing.next_frame();
                }
            }

            glutin::Event::Closed => {
                break 'outer;
            }

            glutin::Event::KeyboardInput(ElementState::Pressed, _scan_code, Some(vk)) => match vk {
                VirtualKeyCode::Escape | VirtualKeyCode::Q => {
                    break 'outer;
                }
                VirtualKeyCode::P => {
                    let mut flags = wrench.renderer.get_debug_flags();
                    flags.toggle(webrender::DebugFlags::PROFILER_DBG);
                    wrench.renderer.set_debug_flags(flags);
                }
                VirtualKeyCode::O => {
                    let mut flags = wrench.renderer.get_debug_flags();
                    flags.toggle(webrender::DebugFlags::RENDER_TARGET_DBG);
                    wrench.renderer.set_debug_flags(flags);
                }
                VirtualKeyCode::I => {
                    let mut flags = wrench.renderer.get_debug_flags();
                    flags.toggle(webrender::DebugFlags::TEXTURE_CACHE_DBG);
                    wrench.renderer.set_debug_flags(flags);
                }
                VirtualKeyCode::B => {
                    let mut flags = wrench.renderer.get_debug_flags();
                    flags.toggle(webrender::DebugFlags::ALPHA_PRIM_DBG);
                    wrench.renderer.set_debug_flags(flags);
                }
                VirtualKeyCode::M => {
                    wrench.api.notify_memory_pressure();
                }
                VirtualKeyCode::L => {
                    do_loop = !do_loop;
                }
                VirtualKeyCode::Left => {
                    thing.prev_frame();
                }
                VirtualKeyCode::Right => {
                    thing.next_frame();
                }
                VirtualKeyCode::H => {
                    show_help = !show_help;
                }
                _ => (),
            },
            _ => (),
        }
    }

    if is_headless {
        let pixels = window.gl().read_pixels(
            0,
            0,
            size.width as gl::GLsizei,
            size.height as gl::GLsizei,
            gl::RGBA,
            gl::UNSIGNED_BYTE,
        );

        save_flipped("screenshot.png", &pixels, size);
    }

    wrench.renderer.deinit();
}
