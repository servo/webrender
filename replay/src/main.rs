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
use webrender_traits::{ApiMsg, PipelineId};

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
		println!("new frame ready");		
		self.window_proxy.wakeup_event_loop();
	}

	fn pipeline_size_changed(&mut self,
			_: PipelineId,
			_: Option<Size2D<f32>>) {
	}

	fn new_scroll_frame_ready(&mut self, composite_needed:bool){}
}

fn read_struct(reader: &mut BufReader<File>) -> Option<Vec<u8>>{
	let len = reader.read_u32::<LittleEndian>().ok();
	match len{
		Some(len) => {
			let mut buffer = vec![0; len as usize ];
			reader.read_exact(&mut buffer).unwrap();
			Some(buffer)
		}
		_ => None,
	}
}

fn main() {
	let window = glutin::WindowBuilder::new()
		.with_gl(glutin::GlRequest::Specific(glutin::Api::OpenGl, (3,2)))
		.build()
		.unwrap();

	let (width, height) = window.get_inner_size().unwrap();
	unsafe{
		window.make_current().ok();
		gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);
	}

	let opts = webrender::RendererOptions{device_pixel_ratio: 2.0,
		resource_path: PathBuf::from("/Users/lramjit/work/servo/resources/shaders"),
		enable_aa: false,
		enable_msaa: false,
		enable_profiler: false,
		enable_recording: false,
	};

	let (mut renderer, sender) = webrender::renderer::Renderer::new(opts);	
	let mut api = sender.create_api();
	let notifier = Box::new(Notifier::new(window.create_window_proxy()));
	renderer.set_render_notifier(notifier);
	let (mut width, mut height) = window.get_inner_size().unwrap();
	width *= window.hidpi_factor() as u32;
	height *= window.hidpi_factor() as u32;
	//read and send the resources file
	let f = File::open("resources.bin").unwrap();
	let mut reader = BufReader::new(f);

	while let Some(buffer) = read_struct(&mut reader){
		let msg:ApiMsg = deserialize(&buffer).unwrap();
		api.api_sender.send(msg).unwrap()
	}		
	window.swap_buffers();
	let f = File::open("display_list.bin").unwrap();
	let mut reader = BufReader::new(f);
	if let Some(buff) = read_struct(&mut reader){
		let msg:ApiMsg = deserialize(&buff).unwrap();
		api.api_sender.send(msg).unwrap();
	}

	if let Some(auxiliary_data) = read_struct(&mut reader){
		api.payload_sender.send(&auxiliary_data[..]).unwrap();
	}
	for event in window.wait_events(){
		match event {
			glutin::Event::Closed => break,
				glutin::Event::Awakened => { 				
					gl::clear(gl::COLOR_BUFFER_BIT);
					renderer.update();
					renderer.render(Size2D::new(width, height as u32));
			}
			glutin::Event::KeyboardInput(glutin::ElementState::Released, _, _) =>{
				if let Some(buff) = read_struct(&mut reader){
					let msg:ApiMsg = deserialize(&buff).unwrap();
					api.api_sender.send(msg).unwrap();
				}

				if let Some(auxiliary_data) = read_struct(&mut reader){
					api.payload_sender.send(&auxiliary_data[..]).unwrap();
				}

			}
			_ => ()
		}
	}
}

