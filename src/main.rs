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

    let visual = composition.create_d3d_visual(300, 200).unwrap();
    visual.set_offset_x(100.).unwrap();
    visual.set_offset_y(50.).unwrap();
    composition.commit().unwrap();

    let green_rgba = [0., 0.5, 0., 1.];
    visual.render_and_present_solid_frame(&composition, &green_rgba).unwrap();

    if std::env::var_os("INIT_ONLY").is_some() {
        return
    }

    events_loop.run_forever(|event| match event {
        winit::Event::WindowEvent { event: winit::WindowEvent::Closed, .. } => {
            winit::ControlFlow::Break
        }
        _ => winit::ControlFlow::Continue,
    });
}

fn direct_composition_from_window(window: &winit::Window) -> DirectComposition {
    unsafe {
        DirectComposition::new(window.get_hwnd() as _).unwrap()
    }
}
