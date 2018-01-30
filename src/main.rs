#![allow(non_snake_case)]

extern crate gl;
extern crate glutin;
extern crate winapi;

use glutin::GlContext;
use glutin::os::windows::WindowExt;
use std::ops::Deref;
use std::os::raw::c_void;
use std::ptr;
use winapi::Interface;
use winapi::shared::dxgi::IDXGIDevice;
use winapi::shared::minwindef::TRUE;
use winapi::shared::winerror::HRESULT;
use winapi::shared::winerror::SUCCEEDED;
use winapi::shared::windef::HWND;
use winapi::um::d3d11::D3D11_CREATE_DEVICE_BGRA_SUPPORT;
use winapi::um::d3d11::D3D11_SDK_VERSION;
use winapi::um::d3d11::D3D11CreateDevice;
use winapi::um::d3d11::ID3D11Device;
use winapi::um::d3dcommon::D3D_DRIVER_TYPE_HARDWARE;
use winapi::um::dcomp::IDCompositionDevice;
use winapi::um::dcomp::IDCompositionTarget;
use winapi::um::dcomp::DCompositionCreateDevice;
use winapi::um::unknwnbase::IUnknown;

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

trait ToResult: Sized {
    fn to_result(self) -> Result<(), Self>;
}

impl ToResult for HRESULT {
    fn to_result(self) -> Result<(), Self> {
        if SUCCEEDED(self) {
            Ok(())
        } else {
            Err(self)
        }
    }
}

fn as_void<T>(p: *mut *mut T) -> *mut *mut c_void {
    p as _
}

unsafe fn query_interface<T, U>(p: *mut T) -> Result<*mut U, HRESULT>
    where T: Deref<Target=IUnknown>, U: Interface
{
    let mut ret = ptr::null_mut();
    (*p).QueryInterface(&U::uuidof(), as_void(&mut ret)).to_result()?;
    Ok(ret)
}

/// https://msdn.microsoft.com/en-us/library/windows/desktop/hh449180(v=vs.85).aspx

unsafe fn initialize_direct_composition_device(m_hwnd: HWND)
    -> Result<(*mut ID3D11Device, *mut IDCompositionDevice, *mut IDCompositionTarget), HRESULT>
{
    let mut m_pD3D11Device = ptr::null_mut();
    let mut featureLevelSupported = 0;

    // Create the D3D device object. The D3D11_CREATE_DEVICE_BGRA_SUPPORT
    // flag enables rendering on surfaces using Direct2D.
    D3D11CreateDevice(
        ptr::null_mut(),
        D3D_DRIVER_TYPE_HARDWARE,
        ptr::null_mut(),
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        ptr::null_mut(),
        0,
        D3D11_SDK_VERSION,
        &mut m_pD3D11Device,
        &mut featureLevelSupported,
        ptr::null_mut(),
    ).to_result()?;

    // Create the DXGI device used to create bitmap surfaces.
    let pDXGIDevice: *mut IDXGIDevice = query_interface(m_pD3D11Device)?;

    // Create the DirectComposition device object.
    let mut m_pDCompDevice: *mut IDCompositionDevice = ptr::null_mut();
    DCompositionCreateDevice(
        pDXGIDevice, &IDCompositionDevice::uuidof(), as_void(&mut m_pDCompDevice),
    ).to_result()?;

    // Create the composition target object based on the
    // specified application window.
    let mut m_pDCompTarget = ptr::null_mut();
    (*m_pDCompDevice).CreateTargetForHwnd(m_hwnd, TRUE, &mut m_pDCompTarget).to_result()?;

    // FIXME: make/use a type with a destructor, so that this runs even in case of early return
    (*pDXGIDevice).Release();

    Ok((m_pD3D11Device, m_pDCompDevice, m_pDCompTarget))
}
