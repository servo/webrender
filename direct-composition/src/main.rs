#[cfg(not(windows))]
compile_error!("This demo only runs on Windows.");

extern crate direct_composition;
extern crate gleam;
extern crate webrender;
extern crate winit;

use direct_composition::DirectComposition;
use webrender::api::{ColorF, DeviceUintSize};
use winit::os::windows::WindowExt;

fn main() {
    let mut events_loop = winit::EventsLoop::new();
    let window = winit::WindowBuilder::new()
        .with_title("Hello, world!")
        .with_dimensions(1024, 768)
        .build(&events_loop)
        .unwrap();

    let composition = direct_composition_from_window(&window);

    let (renderer, _api_sender) = webrender::Renderer::new(
        composition.gleam.clone(),
        Box::new(Notifier { events_proxy: events_loop.create_proxy() }),
        webrender::RendererOptions::default(),
    ).unwrap();

    let mut clicks: usize = 0;
    let mut offset_y = 100.;
    let mut rects = [
        Rectangle::new(&composition, DeviceUintSize::new(300, 200), 0., 0.2, 0.4, 1.),
        Rectangle::new(&composition, DeviceUintSize::new(400, 300), 0., 0.5, 0., 0.5),
    ];
    rects[0].render();
    rects[1].render();

    rects[0].visual.set_offset_x(100.);
    rects[0].visual.set_offset_y(50.);

    rects[1].visual.set_offset_x(200.);
    rects[1].visual.set_offset_y(offset_y);

    composition.commit();

    events_loop.run_forever(|event| {
        if let winit::Event::WindowEvent { event, .. } = event {
            match event {
                winit::WindowEvent::Closed => {
                    return winit::ControlFlow::Break
                }
                winit::WindowEvent::MouseWheel { delta, .. } => {
                    let dy = match delta {
                        winit::MouseScrollDelta::LineDelta(_, dy) => dy,
                        winit::MouseScrollDelta::PixelDelta(_, dy) => dy,
                    };
                    offset_y = (offset_y - 10. * dy).max(0.).min(468.);

                    rects[1].visual.set_offset_y(offset_y);
                    composition.commit();
                }
                winit::WindowEvent::MouseInput {
                    button: winit::MouseButton::Left,
                    state: winit::ElementState::Pressed,
                    ..
                } => {
                    clicks += 1;
                    let rect = &mut rects[clicks % 2];
                    rect.color.g += 0.1;
                    rect.color.g %= 1.;
                    rect.render()
                }
                _ => {}
            }
        }
        winit::ControlFlow::Continue
    });

    renderer.deinit()
}

fn direct_composition_from_window(window: &winit::Window) -> DirectComposition {
    unsafe {
        DirectComposition::new(window.get_hwnd() as _)
    }
}

struct Rectangle {
    visual: direct_composition::AngleVisual,
    color: ColorF,
}

impl Rectangle {
    fn new(composition: &DirectComposition, size: DeviceUintSize,
           r: f32, g: f32, b: f32, a: f32)
           -> Self {
        Rectangle {
            visual: composition.create_angle_visual(size.width, size.height),
            color: ColorF { r, g, b, a },
        }
    }

    fn render(&self) {
        self.visual.make_current();
        self.visual.gleam.clear_color(self.color.r, self.color.g, self.color.b, self.color.a);
        self.visual.gleam.clear(gleam::gl::COLOR_BUFFER_BIT);
        assert_eq!(self.visual.gleam.get_error(), 0);
        self.visual.present();
    }
}


#[derive(Clone)]
struct Notifier {
    events_proxy: winit::EventsLoopProxy,
}

impl webrender::api::RenderNotifier for Notifier {
    fn clone(&self) -> Box<webrender::api::RenderNotifier> {
        Box::new(Clone::clone(self))
    }

    fn wake_up(&self) {
        let _ = self.events_proxy.wakeup();
    }

    fn new_document_ready(&self, _: webrender::api::DocumentId, _: bool, _: bool) {
        self.wake_up();
    }
}
