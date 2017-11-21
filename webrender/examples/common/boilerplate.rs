/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate env_logger;
extern crate euclid;

use gleam::gl;
use glutin;
use std::env;
use std::path::PathBuf;
use webrender;
use webrender::api::*;

struct Notifier {
    window_proxy: glutin::WindowProxy,
}

impl Notifier {
    fn new(window_proxy: glutin::WindowProxy) -> Notifier {
        Notifier { window_proxy }
    }
}

impl RenderNotifier for Notifier {
    fn clone(&self) -> Box<RenderNotifier> {
        Box::new(Notifier {
            window_proxy: self.window_proxy.clone(),
        })
    }

    fn wake_up(&self) {
        #[cfg(not(target_os = "android"))]
        self.window_proxy.wakeup_event_loop();
    }

    fn new_document_ready(&self, _: DocumentId, _scrolled: bool, _composite_needed: bool) {
        self.wake_up();
    }
}

pub trait HandyDandyRectBuilder {
    fn to(&self, x2: i32, y2: i32) -> LayoutRect;
    fn by(&self, w: i32, h: i32) -> LayoutRect;
}
// Allows doing `(x, y).to(x2, y2)` or `(x, y).by(width, height)` with i32
// values to build a f32 LayoutRect
impl HandyDandyRectBuilder for (i32, i32) {
    fn to(&self, x2: i32, y2: i32) -> LayoutRect {
        LayoutRect::new(
            LayoutPoint::new(self.0 as f32, self.1 as f32),
            LayoutSize::new((x2 - self.0) as f32, (y2 - self.1) as f32),
        )
    }

    fn by(&self, w: i32, h: i32) -> LayoutRect {
        LayoutRect::new(
            LayoutPoint::new(self.0 as f32, self.1 as f32),
            LayoutSize::new(w as f32, h as f32),
        )
    }
}

pub trait Example {
    const TITLE: &'static str = "WebRender Sample App";
    const PRECACHE_SHADERS: bool = false;
    fn render(
        &mut self,
        api: &RenderApi,
        builder: &mut DisplayListBuilder,
        resources: &mut ResourceUpdates,
        framebuffer_size: DeviceUintSize,
        pipeline_id: PipelineId,
        document_id: DocumentId,
    );
    fn on_event(&mut self, glutin::Event, &RenderApi, DocumentId) -> bool {
        false
    }
    fn get_external_image_handler(&self) -> Option<Box<webrender::ExternalImageHandler>> {
        None
    }
    fn get_output_image_handler(
        &mut self,
        _gl: &gl::Gl,
    ) -> Option<Box<webrender::OutputImageHandler>> {
        None
    }
    fn draw_custom(&self, _gl: &gl::Gl) {
    }
}

pub fn main_wrapper<E: Example>(
    example: &mut E,
    options: Option<webrender::RendererOptions>,
) {
    env_logger::init().unwrap();

    let args: Vec<String> = env::args().collect();
    let res_path = if args.len() > 1 {
        Some(PathBuf::from(&args[1]))
    } else {
        None
    };

    let window = glutin::WindowBuilder::new()
        .with_title(E::TITLE)
        .with_multitouch()
        .with_gl(glutin::GlRequest::GlThenGles {
            opengl_version: (3, 2),
            opengles_version: (3, 0),
        })
        .build()
        .unwrap();

    unsafe {
        window.make_current().ok();
    }

    let gl = match gl::GlType::default() {
        gl::GlType::Gl => unsafe {
            gl::GlFns::load_with(|symbol| window.get_proc_address(symbol) as *const _)
        },
        gl::GlType::Gles => unsafe {
            gl::GlesFns::load_with(|symbol| window.get_proc_address(symbol) as *const _)
        },
    };

    println!("OpenGL version {}", gl.get_string(gl::VERSION));
    println!("Shader resource path: {:?}", res_path);
    let device_pixel_ratio = window.hidpi_factor();
    println!("Device pixel ratio: {}", device_pixel_ratio);

    let opts = webrender::RendererOptions {
        resource_override_path: res_path,
        debug: true,
        precache_shaders: E::PRECACHE_SHADERS,
        device_pixel_ratio,
        clear_color: Some(ColorF::new(0.3, 0.0, 0.0, 1.0)),
        ..options.unwrap_or(webrender::RendererOptions::default())
    };

    let framebuffer_size = {
        let (width, height) = window.get_inner_size_pixels().unwrap();
        DeviceUintSize::new(width, height)
    };
    let notifier = Box::new(Notifier::new(window.create_window_proxy()));
    let (mut renderer, sender) = webrender::Renderer::new(gl.clone(), notifier, opts).unwrap();
    let api = sender.create_api();
    let document_id = api.add_document(framebuffer_size, 0);

    if let Some(external_image_handler) = example.get_external_image_handler() {
        renderer.set_external_image_handler(external_image_handler);
    }
    if let Some(output_image_handler) = example.get_output_image_handler(&*gl) {
        renderer.set_output_image_handler(output_image_handler);
    }

    let epoch = Epoch(0);
    let pipeline_id = PipelineId(0, 0);
    let layout_size = framebuffer_size.to_f32() / euclid::ScaleFactor::new(device_pixel_ratio);
    let mut builder = DisplayListBuilder::new(pipeline_id, layout_size);
    let mut resources = ResourceUpdates::new();

    example.render(
        &api,
        &mut builder,
        &mut resources,
        framebuffer_size,
        pipeline_id,
        document_id,
    );
    api.set_display_list(
        document_id,
        epoch,
        None,
        layout_size,
        builder.finalize(),
        true,
        resources,
    );
    api.set_root_pipeline(document_id, pipeline_id);
    api.generate_frame(document_id, None);

    'outer: for event in window.wait_events() {
        let mut events = Vec::new();
        events.push(event);
        events.extend(window.poll_events());

        for event in events {
            match event {
                glutin::Event::Closed |
                glutin::Event::KeyboardInput(_, _, Some(glutin::VirtualKeyCode::Escape)) => break 'outer,

                glutin::Event::KeyboardInput(
                    glutin::ElementState::Pressed,
                    _,
                    Some(glutin::VirtualKeyCode::P),
                ) => {
                    let mut flags = renderer.get_debug_flags();
                    flags.toggle(webrender::DebugFlags::PROFILER_DBG);
                    renderer.set_debug_flags(flags);
                }
                glutin::Event::KeyboardInput(
                    glutin::ElementState::Pressed,
                    _,
                    Some(glutin::VirtualKeyCode::O),
                ) => {
                    let mut flags = renderer.get_debug_flags();
                    flags.toggle(webrender::DebugFlags::RENDER_TARGET_DBG);
                    renderer.set_debug_flags(flags);
                }
                glutin::Event::KeyboardInput(
                    glutin::ElementState::Pressed,
                    _,
                    Some(glutin::VirtualKeyCode::I),
                ) => {
                    let mut flags = renderer.get_debug_flags();
                    flags.toggle(webrender::DebugFlags::TEXTURE_CACHE_DBG);
                    renderer.set_debug_flags(flags);
                }
                glutin::Event::KeyboardInput(
                    glutin::ElementState::Pressed,
                    _,
                    Some(glutin::VirtualKeyCode::B),
                ) => {
                    let mut flags = renderer.get_debug_flags();
                    flags.toggle(webrender::DebugFlags::ALPHA_PRIM_DBG);
                    renderer.set_debug_flags(flags);
                }
                glutin::Event::KeyboardInput(
                    glutin::ElementState::Pressed,
                    _,
                    Some(glutin::VirtualKeyCode::Q),
                ) => {
                    renderer.toggle_queries_enabled();
                }
                glutin::Event::KeyboardInput(
                    glutin::ElementState::Pressed,
                    _,
                    Some(glutin::VirtualKeyCode::Key1),
                ) => {
                    api.set_window_parameters(
                        document_id,
                        framebuffer_size,
                        DeviceUintRect::new(DeviceUintPoint::zero(), framebuffer_size),
                        1.0
                    );
                }
                glutin::Event::KeyboardInput(
                    glutin::ElementState::Pressed,
                    _,
                    Some(glutin::VirtualKeyCode::Key2),
                ) => {
                    api.set_window_parameters(
                        document_id,
                        framebuffer_size,
                        DeviceUintRect::new(DeviceUintPoint::zero(), framebuffer_size),
                        2.0
                    );
                }
                glutin::Event::KeyboardInput(
                    glutin::ElementState::Pressed,
                    _,
                    Some(glutin::VirtualKeyCode::M),
                ) => {
                    api.notify_memory_pressure();
                }
                _ => if example.on_event(event, &api, document_id) {
                    let mut builder = DisplayListBuilder::new(pipeline_id, layout_size);
                    let mut resources = ResourceUpdates::new();

                    example.render(
                        &api,
                        &mut builder,
                        &mut resources,
                        framebuffer_size,
                        pipeline_id,
                        document_id,
                    );
                    api.set_display_list(
                        document_id,
                        epoch,
                        None,
                        layout_size,
                        builder.finalize(),
                        true,
                        resources,
                    );
                    api.generate_frame(document_id, None);
                },
            }
        }

        renderer.update();
        renderer.render(framebuffer_size).unwrap();
        example.draw_custom(&*gl);
        window.swap_buffers().ok();
    }

    renderer.deinit();
}
