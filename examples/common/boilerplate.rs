/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use gleam::gl;
use glutin::config::{ConfigTemplateBuilder, GlConfig};
use glutin::context::{
    ContextApi, ContextAttributesBuilder, NotCurrentGlContext,
    PossiblyCurrentContext, Version,
};
use glutin::display::{GetGlDisplay, GlDisplay};
use glutin::surface::{GlSurface, Surface, SurfaceAttributesBuilder, WindowSurface};
use glutin_winit::DisplayBuilder;
use std::env;
use std::ffi::CString;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::rc::Rc;
use webrender;
use webrender::{DebugFlags, ShaderPrecacheFlags};
use webrender::api::*;
use webrender::render_api::*;
use webrender::api::units::*;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::Window;

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

#[allow(dead_code)]
pub trait HandyDandyRectBuilder {
    fn to(&self, x2: i32, y2: i32) -> LayoutRect;
    fn by(&self, w: i32, h: i32) -> LayoutRect;
}
// Allows doing `(x, y).to(x2, y2)` or `(x, y).by(width, height)` with i32
// values to build a f32 LayoutRect
impl HandyDandyRectBuilder for (i32, i32) {
    fn to(&self, x2: i32, y2: i32) -> LayoutRect {
        LayoutRect::from_origin_and_size(
            LayoutPoint::new(self.0 as f32, self.1 as f32),
            LayoutSize::new((x2 - self.0) as f32, (y2 - self.1) as f32),
        )
    }

    fn by(&self, w: i32, h: i32) -> LayoutRect {
        LayoutRect::from_origin_and_size(
            LayoutPoint::new(self.0 as f32, self.1 as f32),
            LayoutSize::new(w as f32, h as f32),
        )
    }
}

pub trait Example {
    const TITLE: &'static str = "WebRender Sample App";
    const PRECACHE_SHADER_FLAGS: ShaderPrecacheFlags = ShaderPrecacheFlags::EMPTY;
    const WIDTH: u32 = 1920;
    const HEIGHT: u32 = 1080;

    fn render(
        &mut self,
        api: &mut RenderApi,
        builder: &mut DisplayListBuilder,
        txn: &mut Transaction,
        device_size: DeviceIntSize,
        pipeline_id: PipelineId,
        document_id: DocumentId,
    );
    fn on_event(
        &mut self,
        _: WindowEvent,
        _: &Window,
        _: &mut RenderApi,
        _: DocumentId,
    ) -> bool {
        false
    }
    fn get_image_handler(
        &mut self,
        _gl: &dyn gl::Gl,
    ) -> Option<Box<dyn ExternalImageHandler>> {
        None
    }
    fn draw_custom(&mut self, _gl: &dyn gl::Gl) {
    }
}

struct App<'a, E: Example> {
    example: &'a mut E,
    options: Option<webrender::WebRenderOptions>,
    proxy: winit::event_loop::EventLoopProxy<()>,

    // Created during resumed()
    window: Option<Window>,
    gl_surface: Option<Surface<WindowSurface>>,
    gl_context: Option<PossiblyCurrentContext>,
    gl: Option<Rc<dyn gl::Gl>>,
    renderer: Option<webrender::Renderer>,
    api: Option<RenderApi>,
    document_id: Option<DocumentId>,
    device_size: DeviceIntSize,
    debug_flags: DebugFlags,
    pipeline_id: PipelineId,
    epoch: Epoch,
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

impl<'a, E: Example> ApplicationHandler for App<'a, E> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let args: Vec<String> = env::args().collect();
        let res_path = if args.len() > 1 {
            Some(PathBuf::from(&args[1]))
        } else {
            None
        };

        let window_attrs = Window::default_attributes()
            .with_title(E::TITLE)
            .with_inner_size(winit::dpi::LogicalSize::new(E::WIDTH as f64, E::HEIGHT as f64));

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

        // Try OpenGL first, fall back to GLES.
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

        println!("OpenGL version {}", gl.get_string(gl::VERSION));
        println!("Shader resource path: {:?}", res_path);
        let device_pixel_ratio = window.scale_factor() as f32;
        println!("Device pixel ratio: {}", device_pixel_ratio);

        println!("Loading shaders...");
        let debug_flags = DebugFlags::ECHO_DRIVER_MESSAGES | DebugFlags::TEXTURE_CACHE_DBG;
        let opts = webrender::WebRenderOptions {
            resource_override_path: res_path,
            precache_flags: E::PRECACHE_SHADER_FLAGS,
            clear_color: ColorF::new(0.3, 0.0, 0.0, 1.0),
            debug_flags,
            ..self.options.take().unwrap_or(webrender::WebRenderOptions::default())
        };

        let device_size = {
            let size = window.inner_size();
            DeviceIntSize::new(size.width as i32, size.height as i32)
        };
        let notifier = Box::new(Notifier::new(self.proxy.clone()));
        let (mut renderer, sender) = webrender::create_webrender_instance(
            gl.clone(),
            notifier,
            opts,
            None,
        ).unwrap();
        let mut api = sender.create_api();
        let document_id = api.add_document(device_size);

        let external = self.example.get_image_handler(&*gl);
        if let Some(external_image_handler) = external {
            renderer.set_external_image_handler(external_image_handler);
        }

        let epoch = Epoch(0);
        let pipeline_id = PipelineId(0, 0);
        let mut builder = DisplayListBuilder::new(pipeline_id);
        let mut txn = Transaction::new();
        builder.begin();

        self.example.render(
            &mut api,
            &mut builder,
            &mut txn,
            device_size,
            pipeline_id,
            document_id,
        );
        txn.set_display_list(epoch, builder.end());
        txn.set_root_pipeline(pipeline_id);
        txn.generate_frame(0, true, false, RenderReasons::empty());
        api.send_transaction(document_id, txn);

        self.window = Some(window);
        self.gl_surface = Some(gl_surface);
        self.gl_context = Some(gl_context);
        self.gl = Some(gl);
        self.renderer = Some(renderer);
        self.api = Some(api);
        self.document_id = Some(document_id);
        self.device_size = device_size;
        self.debug_flags = debug_flags;
        self.pipeline_id = pipeline_id;
        self.epoch = epoch;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        win_event: WindowEvent,
    ) {
        if self.window.is_none() {
            return;
        }

        // Handle exit events before borrowing sub-fields.
        match win_event {
            WindowEvent::CloseRequested => {
                if let Some(r) = self.renderer.take() { r.deinit(); }
                event_loop.exit();
                return;
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    ref logical_key,
                    ..
                },
                ..
            } if *logical_key == Key::Named(NamedKey::Escape) => {
                if let Some(r) = self.renderer.take() { r.deinit(); }
                event_loop.exit();
                return;
            }
            _ => {}
        }

        let window = self.window.as_ref().unwrap();
        let api = self.api.as_mut().unwrap();
        let renderer = self.renderer.as_mut().unwrap();
        let gl = self.gl.as_ref().unwrap();
        let gl_surface = self.gl_surface.as_ref().unwrap();
        let gl_context = self.gl_context.as_ref().unwrap();
        let document_id = self.document_id.unwrap();

        let mut txn = Transaction::new();
        let mut custom_event = true;

        let old_flags = self.debug_flags;

        match win_event {
            WindowEvent::CursorMoved { .. } => {
                custom_event = self.example.on_event(
                    win_event,
                    window,
                    api,
                    document_id,
                );
                if !custom_event {
                    return;
                }
            },
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    ref logical_key,
                    ..
                },
                ..
            } => {
                let key = logical_key.clone();
                match key.as_ref() {
                    Key::Character("p") | Key::Character("P") =>
                        self.debug_flags.toggle(DebugFlags::PROFILER_DBG),
                    Key::Character("o") | Key::Character("O") =>
                        self.debug_flags.toggle(DebugFlags::RENDER_TARGET_DBG),
                    Key::Character("i") | Key::Character("I") =>
                        self.debug_flags.toggle(DebugFlags::TEXTURE_CACHE_DBG),
                    Key::Character("t") | Key::Character("T") =>
                        self.debug_flags.toggle(DebugFlags::PICTURE_CACHING_DBG),
                    Key::Character("q") | Key::Character("Q") =>
                        self.debug_flags.toggle(
                            DebugFlags::GPU_TIME_QUERIES | DebugFlags::GPU_SAMPLE_QUERIES
                        ),
                    Key::Character("g") | Key::Character("G") =>
                        self.debug_flags.toggle(DebugFlags::GPU_CACHE_DBG),
                    Key::Character("m") | Key::Character("M") =>
                        api.notify_memory_pressure(),
                    Key::Character("c") | Key::Character("C") => {
                        let path: PathBuf = "../captures/example".into();
                        let bits = CaptureBits::all();
                        api.save_capture(path, bits);
                    },
                    _ => {
                        custom_event = self.example.on_event(
                            win_event,
                            window,
                            api,
                            document_id,
                        )
                    },
                }
            },
            other => custom_event = self.example.on_event(
                other,
                window,
                api,
                document_id,
            ),
        };

        if self.debug_flags != old_flags {
            api.send_debug_cmd(DebugCommand::SetFlags(self.debug_flags));
        }

        if custom_event {
            let mut builder = DisplayListBuilder::new(self.pipeline_id);
            builder.begin();

            self.example.render(
                api,
                &mut builder,
                &mut txn,
                self.device_size,
                self.pipeline_id,
                document_id,
            );
            txn.set_display_list(self.epoch, builder.end());
            txn.generate_frame(0, true, false, RenderReasons::empty());
        }
        api.send_transaction(document_id, txn);

        renderer.update();
        renderer.render(self.device_size, 0).unwrap();
        let _ = renderer.flush_pipeline_info();
        self.example.draw_custom(&**gl);
        gl_surface.swap_buffers(gl_context).ok();
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: ()) {
        if let Some(ref window) = self.window {
            window.request_redraw();
        }
    }
}

pub fn main_wrapper<E: Example>(
    example: &mut E,
    options: Option<webrender::WebRenderOptions>,
) {
    env_logger::init();

    #[cfg(target_os = "macos")]
    {
        use core_foundation::{self as cf, base::TCFType};
        let i = cf::bundle::CFBundle::main_bundle().info_dictionary();
        let mut i = unsafe { i.to_mutable() };
        i.set(
            cf::string::CFString::new("NSSupportsAutomaticGraphicsSwitching"),
            cf::boolean::CFBoolean::true_value().into_CFType(),
        );
    }

    let event_loop = EventLoop::with_user_event()
        .build()
        .expect("failed to create event loop");

    let proxy = event_loop.create_proxy();

    let mut app = App {
        example,
        options,
        proxy,
        window: None,
        gl_surface: None,
        gl_context: None,
        gl: None,
        renderer: None,
        api: None,
        document_id: None,
        device_size: DeviceIntSize::zero(),
        debug_flags: DebugFlags::empty(),
        pipeline_id: PipelineId(0, 0),
        epoch: Epoch(0),
    };

    event_loop.run_app(&mut app).expect("event loop error");
}
