#![allow(non_snake_case)]

extern crate gl;
extern crate glutin;
extern crate winapi;

use com::{Com, ToResult};
use glutin::GlContext;
use glutin::os::windows::WindowExt;
use std::ptr;
use winapi::Interface;
use winapi::shared::minwindef::TRUE;
use winapi::shared::winerror::HRESULT;
use winapi::shared::windef::HWND;
use winapi::um::d3d11::ID3D11Device;
use winapi::um::dcomp::IDCompositionDevice;
use winapi::um::dcomp::IDCompositionTarget;

mod com;

fn main() {
    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new()
        .with_title("Hello, world!")
        .with_dimensions(1024, 768);
    let context = glutin::ContextBuilder::new()
        .with_vsync(true);
    let gl_window = glutin::GlWindow::new(window, context, &events_loop).unwrap();

    unsafe {
        gl_window.make_current().unwrap();
    }

    unsafe {
        gl::load_with(|symbol| gl_window.get_proc_address(symbol) as *const _);
        gl::ClearColor(0.0, 1.0, 0.0, 1.0);

        initialize_direct_composition_device(gl_window.window().get_hwnd() as _).unwrap();
    }

//    let mut running = true;
    let mut running = false;
    while running {
        events_loop.poll_events(|event| {
            match event {
                glutin::Event::WindowEvent{ event, .. } => match event {
                    glutin::WindowEvent::Closed => running = false,
                    glutin::WindowEvent::Resized(w, h) => gl_window.resize(w, h),
                    _ => ()
                },
                _ => ()
            }
        });

        unsafe {
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }

        gl_window.swap_buffers().unwrap();
    }
    println!("Ok")
}

/// https://msdn.microsoft.com/en-us/library/windows/desktop/hh449180(v=vs.85).aspx

unsafe fn initialize_direct_composition_device(hwnd: HWND)
    -> Result<(Com<ID3D11Device>, Com<IDCompositionDevice>, Com<IDCompositionTarget>), HRESULT>
{
    let mut d3d_device = Com::<ID3D11Device>::null();
    let mut featureLevelSupported = 0;

    // Create the D3D device object.
    // The D3D11_CREATE_DEVICE_BGRA_SUPPORT flag enables rendering on surfaces using Direct2D.
    winapi::um::d3d11::D3D11CreateDevice(
        ptr::null_mut(),
        winapi::um::d3dcommon::D3D_DRIVER_TYPE_HARDWARE,
        ptr::null_mut(),
        winapi::um::d3d11::D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        ptr::null_mut(),
        0,
        winapi::um::d3d11::D3D11_SDK_VERSION,
        d3d_device.as_ptr_ptr(),
        &mut featureLevelSupported,
        ptr::null_mut(),
    ).to_result()?;

    // Create the DXGI device used to create bitmap surfaces.
    let pDXGIDevice = d3d_device.query_interface::<winapi::shared::dxgi::IDXGIDevice>()?;

    // Create the DirectComposition device object.
    let mut composition_device = Com::<IDCompositionDevice>::null();
    winapi::um::dcomp::DCompositionCreateDevice(
        &*pDXGIDevice,
        &IDCompositionDevice::uuidof(),
        composition_device.as_void_ptr_ptr(),
    ).to_result()?;

    // Create the composition target object based on the
    // specified application window.
    let mut composition_target = Com::<IDCompositionTarget>::null();
    composition_device.CreateTargetForHwnd(
        hwnd,
        TRUE,
        composition_target.as_ptr_ptr(),
    ).to_result()?;

    Ok((d3d_device, composition_device, composition_target))
}
