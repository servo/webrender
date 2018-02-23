#[cfg(not(windows))]
compile_error!("This demo only runs on Windows.");

extern crate directcomposition;
extern crate winit;

use directcomposition::DirectComposition;
use winit::os::windows::WindowExt;

fn main() {
    let mut events_loop = winit::EventsLoop::new();
    let window = winit::WindowBuilder::new()
        .with_title("Hello, world!")
        .with_dimensions(1024, 768)
        .build(&events_loop)
        .unwrap();

    let composition = direct_composition_from_window(&window);

    let visual1 = composition.create_d3d_visual(300, 200).unwrap();
    visual1.set_offset_x(100.).unwrap();
    visual1.set_offset_y(50.).unwrap();

    let visual2 = composition.create_d3d_visual(400, 300).unwrap();
    let mut offset_y = 100.;
    visual2.set_offset_x(200.).unwrap();
    visual2.set_offset_y(offset_y).unwrap();

    composition.commit().unwrap();

    let mut rgba1 = [0., 0.2, 0.4, 1.];
    let mut rgba2 = [0., 0.5, 0., 0.5];
    visual1.render_and_present_solid_frame(&composition, &rgba1).unwrap();
    visual2.render_and_present_solid_frame(&composition, &rgba2).unwrap();

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

                    visual2.set_offset_y(offset_y).unwrap();
                    composition.commit().unwrap();
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
                    rgba[1] += 0.1;
                    rgba[1] %= 1.;
                    visual.render_and_present_solid_frame(&composition, &rgba).unwrap();
                }
                _ => {}
            }
        }
        winit::ControlFlow::Continue
    });
}

fn direct_composition_from_window(window: &winit::Window) -> DirectComposition {
    unsafe {
        DirectComposition::new(window.get_hwnd() as _).unwrap()
    }
}
