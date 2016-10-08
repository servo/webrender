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
use std::fs::File;
use std::io::{BufReader};
use std::io::prelude::*;
use std::path::PathBuf;
use std::env;
use webrender_traits::{ApiMsg, RenderApi, PipelineId};
use glutin::{Event, ElementState, VirtualKeyCode as Key};

struct Notifier {
window_proxy: glutin::WindowProxy,
}

impl Notifier {
	fn new(window_proxy: glutin::WindowProxy) -> Notifier {
		Notifier {window_proxy: window_proxy,
		}
	}
}

impl webrender_traits::RenderNotifier for Notifier {
	fn new_frame_ready(&mut self) {
		self.window_proxy.wakeup_event_loop();
	}

	fn pipeline_size_changed(&mut self,
			pid: PipelineId,
			size: Option<Size2D<f32>>) {
	}

	fn new_scroll_frame_ready(&mut self, composite_needed:bool){}
}


fn get_file(dir:&String, frame: i32) -> Option<File>{
	let filename = format!("{}/frame_{}.bin", dir,frame);
	File::open(filename).ok()
}

fn read_file(file: &mut File, api: &RenderApi){	
	while let Some(len) = file.read_u32::<LittleEndian>().ok(){
		let mut buffer = vec![0; len as usize];
		file.read_exact(&mut buffer).unwrap();
		let msg:Option<ApiMsg> = deserialize(&buffer).ok();
		match msg{
			Some(msg) => {	
				api.api_sender.send(msg).unwrap();
			}
			None => {
				api.payload_sender.send(&buffer[..]).unwrap();
			}
		}
	}
}	


fn main() {
	let args:Vec<String> = env::args().collect();
	if args.len() != 3{
		println!("{}  <resources_path> <directory>", args[0]);
		return;
	}
	let resource_path = &args[1];
	let ref dir= args[2];
	let window = glutin::WindowBuilder::new()
		.with_gl(glutin::GlRequest::Specific(glutin::Api::OpenGl, (3,2)))
		.build()
		.unwrap();

	let (width, height) = window.get_outer_size().unwrap();
	unsafe{
		window.make_current().ok();
		gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);
	}

	let opts = webrender::RendererOptions{device_pixel_ratio: 2.0,
		resource_path: PathBuf::from(resource_path),
		enable_aa: false,
		enable_msaa: false,
		enable_profiler: false,
		enable_recording: false,
		enable_scrollbars: false,
		precache_shaders: false,
		renderer_kind: webrender_traits::RendererKind::Native,
		debug: false,
	};

	let (mut renderer, sender) = webrender::renderer::Renderer::new(opts);	
	let mut api = sender.create_api();
	let notifier = Box::new(Notifier::new(window.create_window_proxy()));
	renderer.set_render_notifier(notifier);
	let (mut width, mut height) = window.get_outer_size().unwrap();
	width *= window.hidpi_factor() as u32;
	height *= window.hidpi_factor() as u32;


	//read and send the resources file
	let mut frame_num = 0;
	if let Some(mut file) = get_file(dir, frame_num){
		read_file(&mut file, &api);
	}	
	for event in window.wait_events(){
		match event {
			Event::Closed => break,
			Event::Awakened => { 				
					gl::clear(gl::COLOR_BUFFER_BIT);
					renderer.update();
					renderer.render(Size2D::new(width, height));
					window.swap_buffers();	
			}
			Event::KeyboardInput(ElementState::Pressed, _, Some(Key::Right)) =>{
				frame_num += 1;
				if let Some(mut file) = get_file(dir, frame_num){
					read_file(&mut file, &api);
				}
				else{
					frame_num -= 1;
					println!("At last frame.");
				}	
			}
			Event::KeyboardInput(ElementState::Pressed, _, Some(Key::Left)) => {
				frame_num -= 1;
				if let Some(mut file) = get_file(dir, frame_num){
					read_file(&mut file, &api);
				}

				else{
					frame_num +=1;
					println!("At first frame.");
				}
			}
			_ => ()
		}
	}
}

