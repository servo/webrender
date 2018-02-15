#![allow(non_snake_case)]

extern crate winapi;
extern crate winit;
extern crate wio;

use com::{OutParam, ToResult};
use std::ptr;
use winapi::Interface;
use winapi::shared::minwindef::{TRUE, FALSE};
use winapi::shared::winerror::HRESULT;
use winapi::shared::windef::HWND;
use winapi::um::d3d11::ID3D11Device;
use winapi::um::dcomp::IDCompositionDevice;
use winapi::um::dcomp::IDCompositionTarget;
use winit::os::windows::WindowExt;
use wio::com::ComPtr;

mod com;

fn main() {
    let mut events_loop = winit::EventsLoop::new();
    let window = winit::WindowBuilder::new()
        .with_title("Hello, world!")
        .with_dimensions(1024, 768)
        .build(&events_loop)
        .unwrap();

    let _composition = unsafe {
        DirectComposition::initialize(window.get_hwnd() as _).unwrap()
    };

    events_loop.run_forever(|event| match event {
        winit::Event::WindowEvent { event: winit::WindowEvent::Closed, .. } => {
            winit::ControlFlow::Break
        }
        _ => winit::ControlFlow::Continue,
    });
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
        let dxgi_device = d3d_device.cast::<winapi::shared::dxgi::IDXGIDevice>()?;

        // Create the DirectComposition device object.
        let composition_device = ComPtr::<IDCompositionDevice>::from_void_out_param(|ptr_ptr| {
            winapi::um::dcomp::DCompositionCreateDevice(
                &*dxgi_device,
                &IDCompositionDevice::uuidof(),
                ptr_ptr,
            )
        })?;

        // Create the composition target object based on the
        // specified application window.
        let composition_target = ComPtr::from_out_param(|ptr_ptr| {
            composition_device.CreateTargetForHwnd(hwnd, TRUE, ptr_ptr)
        })?;

        macro_rules! visual { () => {
            ComPtr::from_out_param(|ptr_ptr| composition_device.CreateVisual(ptr_ptr))
        }};
        let root_visual = visual!()?;
        let visual1 = visual!()?;
        let visual2 = visual!()?;

        composition_target.SetRoot(&*root_visual).to_result()?;
        root_visual.AddVisual(&*visual1, FALSE, ptr::null_mut()).to_result()?;
        root_visual.AddVisual(&*visual2, FALSE, ptr::null_mut()).to_result()?;

        let _surface = ComPtr::from_out_param(|ptr_ptr| composition_device.CreateSurface(
            100,
            100,
            winapi::shared::dxgiformat::DXGI_FORMAT_B8G8R8A8_UNORM,
            winapi::shared::dxgi1_2::DXGI_ALPHA_MODE_IGNORE,
            ptr_ptr,
        ))?;

        Ok(DirectComposition { d3d_device, composition_device, composition_target })
    }
}
