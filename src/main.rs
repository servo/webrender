#[cfg(not(windows))]
compile_error!("This demo only runs on Windows.");

extern crate directcomposition;
extern crate winit;

use winit::os::windows::WindowExt;

fn main() {
    let mut events_loop = winit::EventsLoop::new();
    let window = winit::WindowBuilder::new()
        .with_title("Hello, world!")
        .with_dimensions(1024, 768)
        .build(&events_loop)
        .unwrap();

    let _composition = unsafe {
        directcomposition::initialize(window.get_hwnd() as _).unwrap()
    };

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
