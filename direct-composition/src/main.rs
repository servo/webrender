#[cfg(not(windows))]
compile_error!("This demo only runs on Windows.");

extern crate direct_composition;
extern crate gleam;
extern crate winit;

use direct_composition::{DirectComposition, D3DVisual};
use winit::os::windows::WindowExt;

fn main() {
    let mut events_loop = winit::EventsLoop::new();
    let window = winit::WindowBuilder::new()
        .with_title("Hello, world!")
        .with_dimensions(1024, 768)
        .build(&events_loop)
        .unwrap();

    let composition = direct_composition_from_window(&window);

    let visual1 = composition.create_d3d_visual(300, 200);
    visual1.set_offset_x(100.);
    visual1.set_offset_y(50.);

    let visual2 = composition.create_d3d_visual(400, 300);
    let mut offset_y = 100.;
    visual2.set_offset_x(200.);
    visual2.set_offset_y(offset_y);

    composition.commit();

    let mut rgba1 = (0., 0.2, 0.4, 1.);
    let mut rgba2 = (0., 0.5, 0., 0.5);
    render_plain_rgba_frame(&visual1, &rgba1);
    render_plain_rgba_frame(&visual2, &rgba2);

    let mut clicks: u32 = 0;

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

                    visual2.set_offset_y(offset_y);
                    composition.commit();
                }
                winit::WindowEvent::MouseInput {
                    button: winit::MouseButton::Left,
                    state: winit::ElementState::Pressed,
                    ..
                } => {
                    clicks += 1;
                    let (rgba, visual) = if clicks % 2 == 0 {
                        (&mut rgba1, &visual1)
                    } else {
                        (&mut rgba2, &visual2)
                    };
                    rgba.1 += 0.1;
                    rgba.1 %= 1.;
                    render_plain_rgba_frame(visual, rgba)
                }
                _ => {}
            }
        }
        winit::ControlFlow::Continue
    });
}

fn direct_composition_from_window(window: &winit::Window) -> DirectComposition {
    unsafe {
        DirectComposition::new(window.get_hwnd() as _)
    }
}

fn render_plain_rgba_frame(visual: &D3DVisual, &(r, g, b, a): &(f32, f32, f32, f32)) {
    visual.make_current();
    visual.gleam.clear_color(r, g, b, a);
    visual.gleam.clear(gleam::gl::COLOR_BUFFER_BIT);
    assert_eq!(visual.gleam.get_error(), 0);
    visual.present();
}
