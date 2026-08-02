/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate euclid;
extern crate gleam;
extern crate glutin;
extern crate webrender;
extern crate winit;

use gleam::gl;
use glutin::config::{ConfigTemplateBuilder, GlConfig};
use glutin::context::{
    ContextApi, ContextAttributesBuilder, NotCurrentGlContext,
    PossiblyCurrentContext, Version,
};
use glutin::display::{GetGlDisplay, GlDisplay};
use glutin::surface::{GlSurface, Surface, SurfaceAttributesBuilder, WindowSurface};
use glutin_winit::DisplayBuilder;
use std::ffi::CString;
use std::fs::File;
use std::io::Read;
use std::num::NonZeroU32;
use std::rc::Rc;
use webrender::api::*;
use webrender::api::units::*;
use webrender::render_api::*;
use webrender::DebugFlags;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

struct Notifier {
    events_proxy: winit::event_loop::EventLoopProxy<()>,
}

impl Notifier {
    fn new(events_proxy: winit::event_loop::EventLoopProxy<()>) -> Notifier {
        Notifier { events_proxy }
    }
}

impl RenderNotifier for Notifier {
    fn clone(&self) -> Box<dyn RenderNotifier> {
        Box::new(Notifier {
            events_proxy: self.events_proxy.clone(),
        })
    }

    fn wake_up(&self, _composite_needed: bool) {
        #[cfg(not(target_os = "android"))]
        let _ = self.events_proxy.send_event(());
    }

    fn new_frame_ready(&self,
                       _: DocumentId,
                       _: FramePublishId,
                       params: &FrameReadyParams) {
        self.wake_up(params.render);
    }
}

struct Window {
    window: winit::window::Window,
    gl_surface: Surface<WindowSurface>,
    gl_context: PossiblyCurrentContext,
    renderer: webrender::Renderer,
    name: &'static str,
    pipeline_id: PipelineId,
    document_id: DocumentId,
    epoch: Epoch,
    api: RenderApi,
    font_instance_key: FontInstanceKey,
}

fn load_gl_fns(is_gles: bool, gl_display: &impl GlDisplay) -> Rc<dyn gl::Gl> {
    if is_gles {
        unsafe {
            gl::GlesFns::load_with(|symbol| {
                gl_display.get_proc_address(&CString::new(symbol).unwrap()) as *const _
            })
        }
    } else {
        unsafe {
            gl::GlFns::load_with(|symbol| {
                gl_display.get_proc_address(&CString::new(symbol).unwrap()) as *const _
            })
        }
    }
}

impl Window {
    fn new(
        name: &'static str,
        clear_color: ColorF,
        event_loop: &ActiveEventLoop,
        proxy: &winit::event_loop::EventLoopProxy<()>,
    ) -> Self {
        let window_attrs = winit::window::Window::default_attributes()
            .with_title(name)
            .with_inner_size(LogicalSize::new(800.0_f64, 600.0_f64));

        let template = ConfigTemplateBuilder::new().with_depth_size(24);

        let (window, gl_config) = DisplayBuilder::new()
            .with_window_attributes(Some(window_attrs))
            .build(event_loop, template, |configs| {
                configs.reduce(|acc, config| {
                    if config.num_samples() > acc.num_samples() { config } else { acc }
                }).unwrap()
            })
            .expect("failed to create GL display and window");

        let window = window.unwrap();
        let gl_display = gl_config.display();

        let raw_window_handle: RawWindowHandle = window
            .window_handle()
            .expect("failed to get window handle")
            .as_raw();

        let (not_current_ctx, is_gles) = {
            let gl_attrs = ContextAttributesBuilder::new()
                .with_context_api(ContextApi::OpenGl(Some(Version::new(3, 2))))
                .build(Some(raw_window_handle));
            match unsafe { gl_display.create_context(&gl_config, &gl_attrs) } {
                Ok(ctx) => (ctx, false),
                Err(_) => {
                    let gles_attrs = ContextAttributesBuilder::new()
                        .with_context_api(ContextApi::Gles(Some(Version::new(3, 0))))
                        .build(Some(raw_window_handle));
                    let ctx = unsafe { gl_display.create_context(&gl_config, &gles_attrs) }
                        .expect("failed to create GLES context");
                    (ctx, true)
                }
            }
        };

        let physical_size = window.inner_size();
        let surface_attrs = SurfaceAttributesBuilder::<WindowSurface>::new()
            .build(
                raw_window_handle,
                NonZeroU32::new(physical_size.width.max(1)).unwrap(),
                NonZeroU32::new(physical_size.height.max(1)).unwrap(),
            );

        let gl_surface = unsafe {
            gl_display
                .create_window_surface(&gl_config, &surface_attrs)
                .expect("failed to create GL surface")
        };

        let gl_context = not_current_ctx
            .make_current(&gl_surface)
            .expect("failed to make GL context current");

        let gl = load_gl_fns(is_gles, &gl_display);

        let opts = webrender::WebRenderOptions {
            clear_color,
            ..webrender::WebRenderOptions::default()
        };

        let device_size = {
            let size = window.inner_size();
            DeviceIntSize::new(size.width as i32, size.height as i32)
        };
        let notifier = Box::new(Notifier::new(proxy.clone()));
        let (renderer, sender) = webrender::create_webrender_instance(gl.clone(), notifier, opts, None).unwrap();
        let mut api = sender.create_api();
        let document_id = api.add_document(device_size);

        let epoch = Epoch(0);
        let pipeline_id = PipelineId(0, 0);
        let mut txn = Transaction::new();

        let font_key = api.generate_font_key();
        let font_bytes = load_file("../wrench/reftests/text/FreeSans.ttf");
        txn.add_raw_font(font_key, font_bytes, 0);

        let font_instance_key = api.generate_font_instance_key();
        txn.add_font_instance(font_instance_key, font_key, 32.0, None, None, Vec::new());

        api.send_transaction(document_id, txn);

        Window {
            window,
            gl_surface,
            gl_context,
            renderer,
            name,
            epoch,
            pipeline_id,
            document_id,
            api,
            font_instance_key,
        }
    }

    fn tick(&mut self) -> bool {
        let device_size = {
            let size = self.window.inner_size();
            DeviceIntSize::new(size.width as i32, size.height as i32)
        };
        let mut txn = Transaction::new();
        let mut builder = DisplayListBuilder::new(self.pipeline_id);
        let space_and_clip = SpaceAndClipInfo::root_scroll(self.pipeline_id);
        builder.begin();

        builder.push_simple_stacking_context(
            space_and_clip.spatial_id,
            PrimitiveFlags::IS_BACKFACE_VISIBLE,
        );

        builder.push_rect(
            &CommonItemProperties::new(
                LayoutRect::from_origin_and_size(
                    LayoutPoint::new(100.0, 200.0),
                    LayoutSize::new(100.0, 200.0),
                ),
                space_and_clip,
            ),
            LayoutRect::from_origin_and_size(
                LayoutPoint::new(100.0, 200.0),
                LayoutSize::new(100.0, 200.0),
            ),
            ColorF::new(0.0, 1.0, 0.0, 1.0));

        let text_bounds = LayoutRect::from_origin_and_size(
            LayoutPoint::new(100.0, 50.0),
            LayoutSize::new(700.0, 200.0)
        );
        let glyphs = vec![
            GlyphInstance {
                index: 48,
                point: LayoutPoint::new(100.0, 100.0),
            },
            GlyphInstance {
                index: 68,
                point: LayoutPoint::new(150.0, 100.0),
            },
            GlyphInstance {
                index: 80,
                point: LayoutPoint::new(200.0, 100.0),
            },
            GlyphInstance {
                index: 82,
                point: LayoutPoint::new(250.0, 100.0),
            },
            GlyphInstance {
                index: 81,
                point: LayoutPoint::new(300.0, 100.0),
            },
            GlyphInstance {
                index: 3,
                point: LayoutPoint::new(350.0, 100.0),
            },
            GlyphInstance {
                index: 86,
                point: LayoutPoint::new(400.0, 100.0),
            },
            GlyphInstance {
                index: 79,
                point: LayoutPoint::new(450.0, 100.0),
            },
            GlyphInstance {
                index: 72,
                point: LayoutPoint::new(500.0, 100.0),
            },
            GlyphInstance {
                index: 83,
                point: LayoutPoint::new(550.0, 100.0),
            },
            GlyphInstance {
                index: 87,
                point: LayoutPoint::new(600.0, 100.0),
            },
            GlyphInstance {
                index: 17,
                point: LayoutPoint::new(650.0, 100.0),
            },
        ];

        builder.push_text(
            &CommonItemProperties::new(
                text_bounds,
                space_and_clip,
            ),
            text_bounds,
            &glyphs,
            self.font_instance_key,
            ColorF::new(1.0, 1.0, 0.0, 1.0),
            None,
        );

        builder.pop_stacking_context();

        txn.set_display_list(
            self.epoch,
            builder.end(),
        );
        txn.set_root_pipeline(self.pipeline_id);
        txn.generate_frame(0, true, false, RenderReasons::empty());
        self.api.send_transaction(self.document_id, txn);

        self.renderer.update();
        self.renderer.render(device_size, 0).unwrap();
        self.gl_surface.swap_buffers(&self.gl_context).ok();

        false
    }

    fn handle_event(&mut self, event: WindowEvent) -> bool {
        match event {
            WindowEvent::CloseRequested => return true,
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    ref logical_key,
                    ..
                },
                ..
            } => {
                match logical_key.as_ref() {
                    Key::Named(NamedKey::Escape) => return true,
                    Key::Character("p") | Key::Character("P") => {
                        println!("set flags {}", self.name);
                        self.api.send_debug_cmd(DebugCommand::SetFlags(DebugFlags::PROFILER_DBG));
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        false
    }

    fn deinit(self) {
        self.renderer.deinit();
    }
}

struct MultiWindowApp {
    proxy: winit::event_loop::EventLoopProxy<()>,
    win1: Option<Window>,
    win2: Option<Window>,
}

impl ApplicationHandler for MultiWindowApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.win1.is_some() {
            return;
        }
        self.win1 = Some(Window::new("window1", ColorF::new(0.3, 0.0, 0.0, 1.0), event_loop, &self.proxy));
        self.win2 = Some(Window::new("window2", ColorF::new(0.0, 0.3, 0.0, 1.0), event_loop, &self.proxy));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let win = if self.win1.as_ref().map_or(false, |w| w.window.id() == window_id) {
            self.win1.as_mut()
        } else if self.win2.as_ref().map_or(false, |w| w.window.id() == window_id) {
            self.win2.as_mut()
        } else {
            None
        };

        if let Some(win) = win {
            if win.handle_event(event) {
                if let Some(w) = self.win1.take() { w.deinit(); }
                if let Some(w) = self.win2.take() { w.deinit(); }
                event_loop.exit();
            }
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: ()) {
        if let Some(ref mut win) = self.win1 { win.tick(); }
        if let Some(ref mut win) = self.win2 { win.tick(); }
    }
}

fn main() {
    let event_loop = EventLoop::with_user_event()
        .build()
        .expect("failed to create event loop");

    let proxy = event_loop.create_proxy();

    let mut app = MultiWindowApp {
        proxy,
        win1: None,
        win2: None,
    };

    event_loop.run_app(&mut app).expect("event loop error");
}

fn load_file(name: &str) -> Vec<u8> {
    let mut file = File::open(name).unwrap();
    let mut buffer = vec![];
    file.read_to_end(&mut buffer).unwrap();
    buffer
}
