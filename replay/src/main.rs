extern crate bincode;
extern crate byteorder;
extern crate euclid;
extern crate glutin;
extern crate gleam;
extern crate webrender;
extern crate webrender_traits;

use bincode::serde::deserialize;
use byteorder::{LittleEndian, ReadBytesExt};
use euclid::Size2D;
use gleam::gl;
use std::io::Read;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::env;
use webrender_traits::{RenderApi, PipelineId};
use glutin::{Event, ElementState, VirtualKeyCode as Key};


struct Notifier {
    window_proxy: glutin::WindowProxy,
}

impl Notifier {
    fn new(window_proxy: glutin::WindowProxy) -> Notifier {
        Notifier {
            window_proxy: window_proxy,
        }
    }
}

impl webrender_traits::RenderNotifier for Notifier {
    fn new_frame_ready(&mut self) {
        self.window_proxy.wakeup_event_loop();
    }

    fn pipeline_size_changed(&mut self,
                             _pid: PipelineId,
                             _size: Option<Size2D<f32>>) {
    }

    fn new_scroll_frame_ready(&mut self, _composite_needed: bool) {
    }
}

fn read_file(dir: &Path, frame: i32, api: &RenderApi) -> bool {
    let mut filename = PathBuf::from(dir);
    filename.push(format!("frame_{}.bin", frame));
    let mut file = match File::open(&filename) {
        Ok(file) => file,
        Err(_e) => {
            //println!("Failed to open `{}`: {:?}", filename, e);
            return false
        }
    };
    while let Ok(mut len) = file.read_u32::<LittleEndian>() {
        if len > 0 {
            let mut buffer = vec![0; len as usize];
            file.read_exact(&mut buffer).unwrap();
            let msg = deserialize(&buffer).unwrap();
            api.api_sender.send(msg).unwrap();
        } else {
            len = file.read_u32::<LittleEndian>().unwrap();
            let mut buffer = vec![0; len as usize];
            file.read_exact(&mut buffer).unwrap();
            api.payload_sender.send(&buffer[..]).unwrap();
        }
    }
    true
}


fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 && args.len() != 3 {
        println!("{} [<resources_path>] <directory>", args[0]);
        return;
    }

    let (resource_path, dir) = if args.len() == 2 {
        (Some(PathBuf::from(&args[1])), PathBuf::from(&args[2]))
    } else {
        (None, PathBuf::from(&args[1]))
    };

    let window = glutin::WindowBuilder::new()
        .with_title("WebRender Replay")
        .with_gl(glutin::GlRequest::Specific(glutin::Api::OpenGl, (3,2)))
        .build()
        .unwrap();

   unsafe {
        window.make_current().unwrap();
        gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);
    }

    let opts = webrender::RendererOptions {
        device_pixel_ratio: window.hidpi_factor(),
        resource_override_path: resource_path,
        enable_aa: false,
        enable_msaa: false,
        enable_profiler: false,
        enable_recording: false,
        enable_scrollbars: false,
        precache_shaders: false,
        renderer_kind: webrender_traits::RendererKind::Native,
        debug: false,
        enable_subpixel_aa: false,
    };

    let (mut renderer, sender) = webrender::renderer::Renderer::new(opts);
    let api = sender.create_api();
    let notifier = Box::new(Notifier::new(window.create_window_proxy()));
    renderer.set_render_notifier(notifier);
    let (mut width, mut height) = window.get_inner_size().unwrap();

    //read and send the resources file
    let mut frame_num = 0;
    read_file(&dir, frame_num, &api);

    for event in window.wait_events() {
        match event {
            Event::KeyboardInput(ElementState::Pressed, _, Some(Key::Escape)) |
            Event::Closed => break,
            Event::Resized(w, h) => {
                width = w;
                height = h;
            }
            //Event::Refresh |
            Event::Awakened => {
                println!("Rendering frame {}.", frame_num);
                gl::clear(gl::COLOR_BUFFER_BIT);
                renderer.update();
                renderer.render(Size2D::new(width, height));
                window.swap_buffers().unwrap();
            }
            Event::KeyboardInput(ElementState::Pressed, _, Some(Key::Right)) =>{
                frame_num += 1;
                if !read_file(&dir, frame_num, &api) {
                    frame_num -= 1;
                    println!("At last frame.");
                }
            }
            Event::KeyboardInput(ElementState::Pressed, _, Some(Key::Left)) => {
                frame_num -= 1;
                if frame_num < 0 || !read_file(&dir, frame_num, &api) {
                    frame_num +=1;
                    println!("At first frame.");
                }
            }
            _ => ()
        }
    }
}
