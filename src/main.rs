#![allow(non_snake_case)]

extern crate gl;
extern crate glutin;
extern crate winapi;
extern crate wio;

use com::OutParam;
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
use wio::com::ComPtr;

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

        DirectComposition::initialize(gl_window.window().get_hwnd() as _).unwrap();
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

#[allow(unused)]
struct DirectComposition {
    d3d_device: ComPtr<ID3D11Device>,
    composition_device: ComPtr<IDCompositionDevice>,
    composition_target: ComPtr<IDCompositionTarget>,
}

impl DirectComposition {
    unsafe fn initialize(hwnd: HWND) -> Result<Self, HRESULT> {
        let mut feature_level_supported = 0;

        // Create the D3D device object.
        // The D3D11_CREATE_DEVICE_BGRA_SUPPORT flag enables rendering on surfaces using Direct2D.
        let d3d_device = ComPtr::from_out_param(|ptr_ptr| winapi::um::d3d11::D3D11CreateDevice(
            ptr::null_mut(),
            winapi::um::d3dcommon::D3D_DRIVER_TYPE_HARDWARE,
            ptr::null_mut(),
            winapi::um::d3d11::D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            ptr::null_mut(),
            0,
            winapi::um::d3d11::D3D11_SDK_VERSION,
            ptr_ptr,
            &mut feature_level_supported,
            ptr::null_mut(),
        ))?;

        // Create the DXGI device used to create bitmap surfaces.
        let pDXGIDevice = d3d_device.cast::<winapi::shared::dxgi::IDXGIDevice>()?;

        // Create the DirectComposition device object.
        let composition_device = ComPtr::<IDCompositionDevice>::from_void_out_param(|ptr_ptr| {
            winapi::um::dcomp::DCompositionCreateDevice(
                &*pDXGIDevice,
                &IDCompositionDevice::uuidof(),
                ptr_ptr,
            )
        })?;

        // Create the composition target object based on the
        // specified application window.
        let composition_target = ComPtr::from_out_param(|ptr_ptr| {
            composition_device.CreateTargetForHwnd(hwnd, TRUE, ptr_ptr)
        })?;

        Ok(DirectComposition { d3d_device, composition_device, composition_target })
    }
}
