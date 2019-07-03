/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate base64;
extern crate bincode;
extern crate byteorder;
#[macro_use]
extern crate clap;
#[cfg(target_os = "macos")]
extern crate core_foundation;
#[cfg(target_os = "macos")]
extern crate core_graphics;
extern crate crossbeam;
#[cfg(target_os = "windows")]
extern crate dwrote;
#[cfg(feature = "env_logger")]
extern crate env_logger;
extern crate euclid;
#[cfg(all(unix, not(target_os = "android")))]
extern crate font_loader;
extern crate gleam;
extern crate glutin;
extern crate image;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;
#[cfg(target_os = "windows")]
extern crate mozangle;
#[cfg(feature = "headless")]
extern crate osmesa_sys;
extern crate ron;
#[macro_use]
extern crate serde;
extern crate serde_json;
extern crate time;
extern crate webrender;
extern crate winit;
extern crate yaml_rust;

mod angle;
mod binary_frame_reader;
mod blob;
mod egl;
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
#[cfg(target_os = "macos")]
mod cgfont_to_data;

use crate::binary_frame_reader::BinaryFrameReader;
use gleam::gl;
use crate::perf::PerfHarness;
use crate::png::save_flipped;
use crate::rawtest::RawtestHarness;
use crate::reftest::{ReftestHarness, ReftestOptions};
use std::fs;
#[cfg(feature = "headless")]
use std::ffi::CString;
#[cfg(feature = "headless")]
use std::mem;
use std::os::raw::c_void;
use std::path::{Path, PathBuf};
use std::process;
use std::ptr;
use std::rc::Rc;
use std::sync::mpsc::{channel, Sender, Receiver};
use webrender::DebugFlags;
use webrender::api::*;
use webrender::api::units::*;
use winit::dpi::{LogicalPosition, LogicalSize};
use winit::VirtualKeyCode;
use crate::wrench::{Wrench, WrenchThing};
use crate::yaml_frame_reader::YamlFrameReader;

lazy_static! {
    static ref PLATFORM_DEFAULT_FACE_NAME: String = String::from("Arial");
}

pub static mut CURRENT_FRAME_NUMBER: u32 = 0;

#[cfg(feature = "headless")]
pub struct HeadlessContext {
    width: i32,
    height: i32,
    _context: osmesa_sys::OSMesaContext,
    _buffer: Vec<u32>,
}

#[cfg(not(feature = "headless"))]
pub struct HeadlessContext {
    width: i32,
    height: i32,
}

impl HeadlessContext {
    #[cfg(feature = "headless")]
    fn new(width: i32, height: i32) -> Self {
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
                width,
                height,
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
    fn new(width: i32, height: i32) -> Self {
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
    WindowedContext(glutin::WindowedContext<glutin::PossiblyCurrent>, Rc<dyn gl::Gl>),
    Angle(winit::Window, angle::Context, Rc<dyn gl::Gl>),
    Headless(HeadlessContext, Rc<dyn gl::Gl>),
}

pub struct HeadlessEventIterater;

impl WindowWrapper {
    fn swap_buffers(&self) {
        match *self {
            WindowWrapper::WindowedContext(ref windowed_context, _) => {
                windowed_context.swap_buffers().unwrap()
            }
            WindowWrapper::Angle(_, ref context, _) => context.swap_buffers().unwrap(),
            WindowWrapper::Headless(_, _) => {}
        }
    }

    fn get_inner_size(&self) -> DeviceIntSize {
        fn inner_size(window: &winit::Window) -> DeviceIntSize {
            let size = window
                .get_inner_size()
                .unwrap()
                .to_physical(window.get_hidpi_factor());
            DeviceIntSize::new(size.width as i32, size.height as i32)
        }
        match *self {
            WindowWrapper::WindowedContext(ref windowed_context, _) => {
                inner_size(windowed_context.window())
            }
            WindowWrapper::Angle(ref window, ..) => inner_size(window),
            WindowWrapper::Headless(ref context, _) => DeviceIntSize::new(context.width, context.height),
        }
    }

    fn hidpi_factor(&self) -> f32 {
        match *self {
            WindowWrapper::WindowedContext(ref windowed_context, _) => {
                windowed_context.window().get_hidpi_factor() as f32
            }
            WindowWrapper::Angle(ref window, ..) => window.get_hidpi_factor() as f32,
            WindowWrapper::Headless(_, _) => 1.0,
        }
    }

    fn resize(&mut self, size: DeviceIntSize) {
        match *self {
            WindowWrapper::WindowedContext(ref mut windowed_context, _) => {
                windowed_context.window()
                    .set_inner_size(LogicalSize::new(size.width as f64, size.height as f64))
            },
            WindowWrapper::Angle(ref mut window, ..) => {
                window.set_inner_size(LogicalSize::new(size.width as f64, size.height as f64))
            },
            WindowWrapper::Headless(_, _) => unimplemented!(), // requites Glutin update
        }
    }

    fn set_title(&mut self, title: &str) {
        match *self {
            WindowWrapper::WindowedContext(ref windowed_context, _) => {
                windowed_context.window().set_title(title)
            }
            WindowWrapper::Angle(ref window, ..) => window.set_title(title),
            WindowWrapper::Headless(_, _) => (),
        }
    }

    pub fn gl(&self) -> &dyn gl::Gl {
        match *self {
            WindowWrapper::WindowedContext(_, ref gl) |
            WindowWrapper::Angle(_, _, ref gl) |
            WindowWrapper::Headless(_, ref gl) => &**gl,
        }
    }

    pub fn clone_gl(&self) -> Rc<dyn gl::Gl> {
        match *self {
            WindowWrapper::WindowedContext(_, ref gl) |
            WindowWrapper::Angle(_, _, ref gl) |
            WindowWrapper::Headless(_, ref gl) => gl.clone(),
        }
    }
}

fn make_window(
    size: DeviceIntSize,
    dp_ratio: Option<f32>,
    vsync: bool,
    events_loop: &Option<winit::EventsLoop>,
    angle: bool,
    gl_request: glutin::GlRequest,
) -> WindowWrapper {
    let wrapper = match *events_loop {
        Some(ref events_loop) => {
            let context_builder = glutin::ContextBuilder::new()
                .with_gl(gl_request)
                .with_vsync(vsync);
            let window_builder = winit::WindowBuilder::new()
                .with_title("WRench")
                .with_multitouch()
                .with_dimensions(LogicalSize::new(size.width as f64, size.height as f64));

            if angle {
                let (_window, _context) = angle::Context::with_window(
                    window_builder, context_builder, events_loop
                ).unwrap();

                unsafe {
                    _context
                        .make_current()
                        .expect("unable to make context current!");
                }

                let gl = match _context.get_api() {
                    glutin::Api::OpenGl => unsafe {
                        gl::GlFns::load_with(|symbol| _context.get_proc_address(symbol) as *const _)
                    },
                    glutin::Api::OpenGlEs => unsafe {
                        gl::GlesFns::load_with(|symbol| _context.get_proc_address(symbol) as *const _)
                    },
                    glutin::Api::WebGl => unimplemented!(),
                };

                WindowWrapper::Angle(_window, _context, gl)
            } else {
                let windowed_context = context_builder
                    .build_windowed(window_builder, events_loop)
                    .unwrap();

                let windowed_context = unsafe {
                    windowed_context
                        .make_current()
                        .expect("unable to make context current!")
                };

                let gl = match windowed_context.get_api() {
                    glutin::Api::OpenGl => unsafe {
                        gl::GlFns::load_with(
                            |symbol| windowed_context.get_proc_address(symbol) as *const _
                        )
                    },
                    glutin::Api::OpenGlEs => unsafe {
                        gl::GlesFns::load_with(
                            |symbol| windowed_context.get_proc_address(symbol) as *const _
                        )
                    },
                    glutin::Api::WebGl => unimplemented!(),
                };

                WindowWrapper::WindowedContext(windowed_context, gl)
            }
        }
        None => {
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
        }
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NotifierEvent {
    WakeUp,
    ShutDown,
}

struct Notifier {
    tx: Sender<NotifierEvent>,
}

// setup a notifier so we can wait for frames to be finished
impl RenderNotifier for Notifier {
    fn clone(&self) -> Box<dyn RenderNotifier> {
        Box::new(Notifier {
            tx: self.tx.clone(),
        })
    }

    fn wake_up(&self) {
        self.tx.send(NotifierEvent::WakeUp).unwrap();
    }

    fn shut_down(&self) {
        self.tx.send(NotifierEvent::ShutDown).unwrap();
    }

    fn new_frame_ready(&self,
                       _: DocumentId,
                       _scrolled: bool,
                       composite_needed: bool,
                       _render_time: Option<u64>) {
        if composite_needed {
            self.wake_up();
        }
    }
}

fn create_notifier() -> (Box<dyn RenderNotifier>, Receiver<NotifierEvent>) {
    let (tx, rx) = channel();
    (Box::new(Notifier { tx: tx }), rx)
}

fn rawtest(mut wrench: Wrench, window: &mut WindowWrapper, rx: Receiver<NotifierEvent>) {
    RawtestHarness::new(&mut wrench, window, &rx).run();
    wrench.shut_down(rx);
}

fn reftest<'a>(
    mut wrench: Wrench,
    window: &mut WindowWrapper,
    subargs: &clap::ArgMatches<'a>,
    rx: Receiver<NotifierEvent>
) -> usize {
    let dim = window.get_inner_size();
    let base_manifest = if cfg!(target_os = "android") {
        Path::new("/sdcard/wrench/reftests/reftest.list")
    } else {
        Path::new("reftests/reftest.list")
    };
    let specific_reftest = subargs.value_of("REFTEST").map(|x| Path::new(x));
    let mut reftest_options = ReftestOptions::default();
    if let Some(allow_max_diff) = subargs.value_of("fuzz_tolerance") {
        reftest_options.allow_max_difference = allow_max_diff.parse().unwrap_or(1);
        reftest_options.allow_num_differences = dim.width as usize * dim.height as usize;
    }
    let num_failures = ReftestHarness::new(&mut wrench, window, &rx)
        .run(base_manifest, specific_reftest, &reftest_options);
    wrench.shut_down(rx);
    num_failures
}

fn main() {
    #[cfg(feature = "env_logger")]
    env_logger::init();

    let args_yaml = load_yaml!("args.yaml");
    let clap = clap::App::from_yaml(args_yaml)
        .setting(clap::AppSettings::ArgRequiredElseHelp);

    // On android devices, attempt to read command line arguments
    // from a text file located at /sdcard/wrench/args.
    let args = if cfg!(target_os = "android") {
        // get full backtraces by default because it's hard to request
        // externally on android
        std::env::set_var("RUST_BACKTRACE", "full");

        let mut args = vec!["wrench".to_string()];

        if let Ok(wrench_args) = fs::read_to_string("/sdcard/wrench/args") {
            for line in wrench_args.lines() {
                if line.starts_with("env: ") {
                    let envvar = &line[5..];
                    if let Some(ix) = envvar.find('=') {
                        std::env::set_var(&envvar[0..ix], &envvar[ix + 1..]);
                    } else {
                        std::env::set_var(envvar, "");
                    }
                    continue;
                }
                for arg in line.split_whitespace() {
                    args.push(arg.to_string());
                }
            }
        }

        clap.get_matches_from(&args)
    } else {
        clap.get_matches()
    };

    // handle some global arguments
    let res_path = args.value_of("shaders").map(|s| PathBuf::from(s));
    let dp_ratio = args.value_of("dp_ratio").map(|v| v.parse::<f32>().unwrap());
    let save_type = args.value_of("save").map(|s| match s {
        "yaml" => wrench::SaveType::Yaml,
        "json" => wrench::SaveType::Json,
        "ron" => wrench::SaveType::Ron,
        "binary" => wrench::SaveType::Binary,
        _ => panic!("Save type must be json, ron, yaml, or binary")
    });
    let size = args.value_of("size")
        .map(|s| if s == "720p" {
            DeviceIntSize::new(1280, 720)
        } else if s == "1080p" {
            DeviceIntSize::new(1920, 1080)
        } else if s == "4k" {
            DeviceIntSize::new(3840, 2160)
        } else {
            let x = s.find('x').expect(
                "Size must be specified exactly as 720p, 1080p, 4k, or width x height",
            );
            let w = s[0 .. x].parse::<i32>().expect("Invalid size width");
            let h = s[x + 1 ..].parse::<i32>().expect("Invalid size height");
            DeviceIntSize::new(w, h)
        })
        .unwrap_or(DeviceIntSize::new(1920, 1080));
    let zoom_factor = args.value_of("zoom").map(|z| z.parse::<f32>().unwrap());
    let chase_primitive = match args.value_of("chase") {
        Some(s) => {
            let items = s
                .split(',')
                .map(|s| s.parse::<f32>().unwrap())
                .collect::<Vec<_>>();
            let rect = LayoutRect::new(
                LayoutPoint::new(items[0], items[1]),
                LayoutSize::new(items[2], items[3]),
            );
            webrender::ChasePrimitive::LocalRect(rect)
        },
        None => webrender::ChasePrimitive::Nothing,
    };

    let dump_shader_source = args.value_of("dump_shader_source").map(String::from);

    let mut events_loop = if args.is_present("headless") {
        None
    } else {
        Some(winit::EventsLoop::new())
    };

    let gl_request = match args.value_of("renderer") {
        Some("es3") => {
            glutin::GlRequest::Specific(glutin::Api::OpenGlEs, (3, 0))
        }
        Some("gl3") => {
            glutin::GlRequest::Specific(glutin::Api::OpenGl, (3, 2))
        }
        Some("default") | None => {
            glutin::GlRequest::GlThenGles {
                opengl_version: (3, 2),
                opengles_version: (3, 0),
            }
        }
        Some(api) => {
            panic!("Unexpected renderer string {}", api);
        }
    };

    let mut window = make_window(
        size,
        dp_ratio,
        args.is_present("vsync"),
        &events_loop,
        args.is_present("angle"),
        gl_request,
    );
    let dp_ratio = dp_ratio.unwrap_or(window.hidpi_factor());
    let dim = window.get_inner_size();

    let needs_frame_notifier = ["perf", "reftest", "png", "rawtest"]
        .iter()
        .any(|s| args.subcommand_matches(s).is_some());
    let (notifier, rx) = if needs_frame_notifier {
        let (notifier, rx) = create_notifier();
        (Some(notifier), Some(rx))
    } else {
        (None, None)
    };

    let mut wrench = Wrench::new(
        &mut window,
        events_loop.as_mut().map(|el| el.create_proxy()),
        res_path,
        dp_ratio,
        save_type,
        dim,
        args.is_present("rebuild"),
        args.is_present("no_subpixel_aa"),
        args.is_present("no_picture_caching"),
        args.is_present("verbose"),
        args.is_present("no_scissor"),
        args.is_present("no_batch"),
        args.is_present("precache"),
        args.is_present("slow_subpixel"),
        zoom_factor.unwrap_or(1.0),
        chase_primitive,
        dump_shader_source,
        notifier,
    );

    if let Some(window_title) = wrench.take_title() {
        if !cfg!(windows) {
            window.set_title(&window_title);
        }
    }

    if let Some(subargs) = args.subcommand_matches("show") {
        let no_block = args.is_present("no_block");
        render(
            &mut wrench,
            &mut window,
            size,
            &mut events_loop,
            subargs,
            no_block,
        );
    } else if let Some(subargs) = args.subcommand_matches("png") {
        let surface = match subargs.value_of("surface") {
            Some("screen") | None => png::ReadSurface::Screen,
            Some("gpu-cache") => png::ReadSurface::GpuCache,
            _ => panic!("Unknown surface argument value")
        };
        let output_path = subargs.value_of("OUTPUT").map(|s| PathBuf::from(s));
        let reader = YamlFrameReader::new_from_args(subargs);
        png::png(&mut wrench, surface, &mut window, reader, rx.unwrap(), output_path);
    } else if let Some(subargs) = args.subcommand_matches("reftest") {
        // Exit with an error code in order to ensure the CI job fails.
        process::exit(reftest(wrench, &mut window, subargs, rx.unwrap()) as _);
    } else if let Some(_) = args.subcommand_matches("rawtest") {
        rawtest(wrench, &mut window, rx.unwrap());
        return;
    } else if let Some(subargs) = args.subcommand_matches("perf") {
        // Perf mode wants to benchmark the total cost of drawing
        // a new displaty list each frame.
        wrench.rebuild_display_lists = true;
        let harness = PerfHarness::new(&mut wrench, &mut window, rx.unwrap());
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

    wrench.renderer.deinit();
}

fn render<'a>(
    wrench: &mut Wrench,
    window: &mut WindowWrapper,
    size: DeviceIntSize,
    events_loop: &mut Option<winit::EventsLoop>,
    subargs: &clap::ArgMatches<'a>,
    no_block: bool,
) {
    let input_path = subargs.value_of("INPUT").map(PathBuf::from).unwrap();

    // If the input is a directory, we are looking at a capture.
    let mut thing = if input_path.as_path().is_dir() {
        let mut documents = wrench.api.load_capture(input_path);
        println!("loaded {:?}", documents.iter().map(|cd| cd.document_id).collect::<Vec<_>>());
        let captured = documents.swap_remove(0);
        if let Some(fb_size) = wrench.renderer.device_size() {
            window.resize(fb_size);
        }
        wrench.document_id = captured.document_id;
        Box::new(captured) as Box<dyn WrenchThing>
    } else {
        let extension = input_path
            .extension()
            .expect("Tried to render with an unknown file type.")
            .to_str()
            .expect("Tried to render with an unknown file type.");

        match extension {
            "yaml" => Box::new(YamlFrameReader::new_from_args(subargs)) as Box<dyn WrenchThing>,
            "bin" => Box::new(BinaryFrameReader::new_from_args(subargs)) as Box<dyn WrenchThing>,
            _ => panic!("Tried to render with an unknown file type."),
        }
    };

    let mut show_help = false;
    let mut do_loop = false;
    let mut cpu_profile_index = 0;
    let mut cursor_position = WorldPoint::zero();

    let dim = window.get_inner_size();
    wrench.update(dim);
    thing.do_frame(wrench);

    let mut debug_flags = DebugFlags::empty();

    // Default the profile overlay on for android.
    if cfg!(target_os = "android") {
        debug_flags.toggle(DebugFlags::PROFILER_DBG | DebugFlags::COMPACT_PROFILER);
        wrench.api.send_debug_cmd(DebugCommand::SetFlags(debug_flags));
    }

    let mut body = |wrench: &mut Wrench, events: Vec<winit::Event>| {
        let mut do_frame = false;
        let mut do_render = false;

        for event in events {
            match event {
                winit::Event::Awakened => {
                    do_render = true;
                }
                winit::Event::WindowEvent { event, .. } => match event {
                    winit::WindowEvent::CloseRequested => {
                        return winit::ControlFlow::Break;
                    }
                    winit::WindowEvent::Refresh |
                    winit::WindowEvent::Focused(..) => {
                        do_render = true;
                    }
                    winit::WindowEvent::CursorMoved { position: LogicalPosition { x, y }, .. } => {
                        cursor_position = WorldPoint::new(x as f32, y as f32);
                        wrench.renderer.set_cursor_position(
                            DeviceIntPoint::new(
                                (cursor_position.x * wrench.device_pixel_ratio).round() as i32,
                                (cursor_position.y * wrench.device_pixel_ratio).round() as i32,
                            ),
                        );
                        do_render = true;
                    }
                    winit::WindowEvent::KeyboardInput {
                        input: winit::KeyboardInput {
                            state: winit::ElementState::Pressed,
                            virtual_keycode: Some(vk),
                            ..
                        },
                        ..
                    } => match vk {
                        VirtualKeyCode::Escape => {
                            return winit::ControlFlow::Break;
                        }
                        VirtualKeyCode::P => {
                            debug_flags.toggle(DebugFlags::PROFILER_DBG);
                            wrench.api.send_debug_cmd(DebugCommand::SetFlags(debug_flags));
                            do_render = true;
                        }
                        VirtualKeyCode::O => {
                            debug_flags.toggle(DebugFlags::RENDER_TARGET_DBG);
                            wrench.api.send_debug_cmd(DebugCommand::SetFlags(debug_flags));
                            do_render = true;
                        }
                        VirtualKeyCode::I => {
                            debug_flags.toggle(DebugFlags::TEXTURE_CACHE_DBG);
                            wrench.api.send_debug_cmd(DebugCommand::SetFlags(debug_flags));
                            do_render = true;
                        }
                        VirtualKeyCode::S => {
                            debug_flags.toggle(DebugFlags::COMPACT_PROFILER);
                            wrench.api.send_debug_cmd(DebugCommand::SetFlags(debug_flags));
                            do_render = true;
                        }
                        VirtualKeyCode::D => {
                            debug_flags.toggle(DebugFlags::PICTURE_CACHING_DBG);
                            wrench.api.send_debug_cmd(DebugCommand::SetFlags(debug_flags));
                            do_render = true;
                        }
                        VirtualKeyCode::Q => {
                            debug_flags.toggle(DebugFlags::GPU_TIME_QUERIES | DebugFlags::GPU_SAMPLE_QUERIES);
                            wrench.api.send_debug_cmd(DebugCommand::SetFlags(debug_flags));
                            do_render = true;
                        }
                        VirtualKeyCode::V => {
                            debug_flags.toggle(DebugFlags::SHOW_OVERDRAW);
                            wrench.api.send_debug_cmd(DebugCommand::SetFlags(debug_flags));
                            do_render = true;
                        }
                        VirtualKeyCode::G => {
                            debug_flags.toggle(DebugFlags::GPU_CACHE_DBG);
                            wrench.api.send_debug_cmd(DebugCommand::SetFlags(debug_flags));

                            // force scene rebuild to see the full set of used GPU cache entries
                            let mut txn = Transaction::new();
                            txn.set_root_pipeline(wrench.root_pipeline_id);
                            wrench.api.send_transaction(wrench.document_id, txn);

                            do_frame = true;
                        }
                        VirtualKeyCode::R => {
                            wrench.set_page_zoom(ZoomFactor::new(1.0));
                            do_frame = true;
                        }
                        VirtualKeyCode::M => {
                            wrench.api.notify_memory_pressure();
                            do_render = true;
                        }
                        VirtualKeyCode::L => {
                            do_loop = !do_loop;
                            do_render = true;
                        }
                        VirtualKeyCode::Left => {
                            thing.prev_frame();
                            do_frame = true;
                        }
                        VirtualKeyCode::Right => {
                            thing.next_frame();
                            do_frame = true;
                        }
                        VirtualKeyCode::H => {
                            show_help = !show_help;
                            do_render = true;
                        }
                        VirtualKeyCode::T => {
                            let file_name = format!("profile-{}.json", cpu_profile_index);
                            wrench.renderer.save_cpu_profile(&file_name);
                            cpu_profile_index += 1;
                        }
                        VirtualKeyCode::C => {
                            let path = PathBuf::from("../captures/wrench");
                            wrench.api.save_capture(path, CaptureBits::all());
                        }
                        VirtualKeyCode::Up | VirtualKeyCode::Down => {
                            let mut txn = Transaction::new();

                            let offset = match vk {
                                winit::VirtualKeyCode::Up => LayoutVector2D::new(0.0, 10.0),
                                winit::VirtualKeyCode::Down => LayoutVector2D::new(0.0, -10.0),
                                _ => unreachable!("Should not see non directional keys here.")
                            };

                            txn.scroll(ScrollLocation::Delta(offset), cursor_position);
                            txn.generate_frame();
                            wrench.api.send_transaction(wrench.document_id, txn);

                            do_frame = true;
                        }
                        VirtualKeyCode::Add => {
                            let current_zoom = wrench.get_page_zoom();
                            let new_zoom_factor = ZoomFactor::new(current_zoom.get() + 0.1);
                            wrench.set_page_zoom(new_zoom_factor);
                            do_frame = true;
                        }
                        VirtualKeyCode::Subtract => {
                            let current_zoom = wrench.get_page_zoom();
                            let new_zoom_factor = ZoomFactor::new((current_zoom.get() - 0.1).max(0.1));
                            wrench.set_page_zoom(new_zoom_factor);
                            do_frame = true;
                        }
                        VirtualKeyCode::X => {
                            let results = wrench.api.hit_test(
                                wrench.document_id,
                                None,
                                cursor_position,
                                HitTestFlags::FIND_ALL
                            );

                            println!("Hit test results:");
                            for item in &results.items {
                                println!("  • {:?}", item);
                            }
                            println!("");
                        }
                        VirtualKeyCode::Z => {
                            debug_flags.toggle(DebugFlags::ZOOM_DBG);
                            wrench.api.send_debug_cmd(DebugCommand::SetFlags(debug_flags));
                            do_render = true;
                        }
                        VirtualKeyCode::Y => {
                            println!("Clearing all caches...");
                            wrench.api.send_debug_cmd(DebugCommand::ClearCaches(ClearCache::all()));
                            do_frame = true;
                        }
                        _ => {}
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        let dim = window.get_inner_size();
        wrench.update(dim);

        if do_frame {
            let frame_num = thing.do_frame(wrench);
            unsafe {
                CURRENT_FRAME_NUMBER = frame_num;
            }
        }

        if do_render {
            if show_help {
                wrench.show_onscreen_help();
            }

            wrench.render();
            window.swap_buffers();

            if do_loop {
                thing.next_frame();
            }
        }

        winit::ControlFlow::Continue
    };

    match *events_loop {
        None => {
            while body(wrench, vec![winit::Event::Awakened]) == winit::ControlFlow::Continue {}
            let fb_rect = FramebufferIntSize::new(size.width, size.height).into();
            let pixels = wrench.renderer.read_pixels_rgba8(fb_rect);
            save_flipped("screenshot.png", pixels, size);
        }
        Some(ref mut events_loop) => {
            // We want to ensure that we:
            //
            // (a) Block the thread when no events are present (for CPU/battery purposes)
            // (b) Don't lag the input events by having the event queue back up.
            loop {
                let mut pending_events = Vec::new();

                // Block the thread until at least one event arrives
                // On Android, we are generally profiling when running
                // wrench, and don't want to block on UI events.
                if !no_block && cfg!(not(target_os = "android")) {
                    events_loop.run_forever(|event| {
                        pending_events.push(event);
                        winit::ControlFlow::Break
                    });
                }

                // Collect any other pending events that are also available
                events_loop.poll_events(|event| {
                    pending_events.push(event);
                });

                // Ensure there is at least one event present so that the
                // frame gets rendered.
                if pending_events.is_empty() {
                    pending_events.push(winit::Event::Awakened);
                }

                // Process all of those pending events in the next vsync period
                if body(wrench, pending_events) == winit::ControlFlow::Break {
                    break;
                }
            }
        }
    }
}
