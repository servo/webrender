#![cfg(windows)]

extern crate winapi;

use com::{ComPtr, ToResult, HResult, as_ptr};
use std::ptr;
use winapi::shared::dxgi1_2::DXGI_SWAP_CHAIN_DESC1;
use winapi::shared::dxgi1_2::IDXGIFactory2;
use winapi::shared::minwindef::{TRUE, FALSE};
use winapi::shared::windef::HWND;
use winapi::shared::winerror::S_OK;
use winapi::um::d3d11::ID3D11Device;
use winapi::um::dcomp::IDCompositionDevice;
use winapi::um::dcomp::IDCompositionTarget;
use winapi::um::dcomp::IDCompositionVisual;

pub mod com;

pub struct DirectComposition {
    d3d_device: ComPtr<ID3D11Device>,
    dxgi_factory: ComPtr<IDXGIFactory2>,
    d3d_device_context: ComPtr<winapi::um::d3d11::ID3D11DeviceContext>,
    composition_device: ComPtr<IDCompositionDevice>,
    root_visual: ComPtr<IDCompositionVisual>,

    #[allow(unused)]  // Needs to be kept alive
    composition_target: ComPtr<IDCompositionTarget>,
}

impl DirectComposition {
    /// Initialize DirectComposition in the given window
    ///
    /// # Safety
    ///
    /// `hwnd` must be a valid handle to a window.
    pub unsafe fn new(hwnd: HWND) -> HResult<Self> {
        let mut feature_level_supported = 0;

        let d3d_device = ComPtr::new_with(|ptr_ptr| winapi::um::d3d11::D3D11CreateDevice(
            ptr::null_mut(),
            winapi::um::d3dcommon::D3D_DRIVER_TYPE_HARDWARE,
            ptr::null_mut(),
            winapi::um::d3d11::D3D11_CREATE_DEVICE_BGRA_SUPPORT |
            if cfg!(debug_assertions) {
                winapi::um::d3d11::D3D11_CREATE_DEVICE_DEBUG
            } else {
                0
            },
            ptr::null_mut(),
            0,
            winapi::um::d3d11::D3D11_SDK_VERSION,
            ptr_ptr,
            &mut feature_level_supported,
            ptr::null_mut(),
        ))?;

        let dxgi_device = d3d_device.cast::<winapi::shared::dxgi::IDXGIDevice>()?;

        // https://msdn.microsoft.com/en-us/library/windows/desktop/hh404556(v=vs.85).aspx#code-snippet-1
        // “Because you can create a Direct3D device without creating a swap chain,
        //  you might need to retrieve the factory that is used to create the device
        //  in order to create a swap chain.”
        let adapter = ComPtr::new_with(|ptr_ptr| dxgi_device.GetAdapter(ptr_ptr))?;
        let dxgi_factory = ComPtr::<IDXGIFactory2>::new_with_uuid(|uuid, ptr_ptr| {
            adapter.GetParent(uuid, ptr_ptr)
        })?;

        let d3d_device_context = ComPtr::new_with(|ptr_ptr| {
            d3d_device.GetImmediateContext(ptr_ptr);
            S_OK
        })?;

        // Create the DirectComposition device object.
        let composition_device = ComPtr::<IDCompositionDevice>::new_with_uuid(|uuid, ptr_ptr| {
            winapi::um::dcomp::DCompositionCreateDevice(&*dxgi_device, uuid,ptr_ptr)
        })?;

        // Create the composition target object based on the
        // specified application window.
        let composition_target = ComPtr::new_with(|ptr_ptr| {
            composition_device.CreateTargetForHwnd(hwnd, TRUE, ptr_ptr)
        })?;

        let root_visual = ComPtr::new_with(|ptr_ptr| composition_device.CreateVisual(ptr_ptr))?;
        composition_target.SetRoot(&*root_visual).to_result()?;

        Ok(DirectComposition {
            d3d_device, dxgi_factory, d3d_device_context,
            composition_device, composition_target, root_visual,
        })
    }

    /// Execute changes to the DirectComposition scene.
    pub fn commit(&self) -> HResult<()> {
        unsafe {
            self.composition_device.Commit().to_result()
        }
    }

    pub fn create_d3d_visual(&self, width: u32, height: u32) -> HResult<D3DVisual> {
        unsafe {
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: winapi::shared::dxgiformat::DXGI_FORMAT_B8G8R8A8_UNORM,
                Stereo: FALSE,
                SampleDesc: winapi::shared::dxgitype::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: winapi::shared::dxgitype::DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: 2,
                Scaling: winapi::shared::dxgi1_2::DXGI_SCALING_STRETCH,
                SwapEffect: winapi::shared::dxgi::DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
                AlphaMode:  winapi::shared::dxgi1_2::DXGI_ALPHA_MODE_PREMULTIPLIED,
                Flags: 0,
            };
            let swap_chain = ComPtr::<winapi::shared::dxgi1_2::IDXGISwapChain1>::new_with(|ptr_ptr| {
                self.dxgi_factory.CreateSwapChainForComposition(
                    as_ptr(&self.d3d_device),
                    &desc,
                    ptr::null_mut(),
                    ptr_ptr,
                )
            })?;
            let back_buffer = ComPtr::<winapi::um::d3d11::ID3D11Texture2D>::new_with_uuid(|uuid, ptr_ptr| {
                swap_chain.GetBuffer(0, uuid, ptr_ptr)
            })?;
            let render_target_view = ComPtr::new_with(|ptr_ptr| {
                self.d3d_device.CreateRenderTargetView(
                    as_ptr(&back_buffer), ptr::null_mut(), ptr_ptr,
                )
            })?;

            let visual = ComPtr::new_with(|ptr_ptr| self.composition_device.CreateVisual(ptr_ptr))?;
            visual.SetContent(&*****swap_chain).to_result()?;
            self.root_visual.AddVisual(&*visual, FALSE, ptr::null_mut()).to_result()?;

            Ok(D3DVisual { visual, swap_chain, render_target_view })
        }
    }
}

/// A DirectComposition "visual" configured for rendering with Direct3D.
pub struct D3DVisual {
    visual: ComPtr<IDCompositionVisual>,
    swap_chain: ComPtr<winapi::shared::dxgi1_2::IDXGISwapChain1>,
    render_target_view: ComPtr<winapi::um::d3d11::ID3D11RenderTargetView>,
}

impl D3DVisual {
    pub fn set_offset_x(&self, offset_x: f32) -> HResult<()> {
        unsafe {
            self.visual.SetOffsetX_1(offset_x).to_result()
        }
    }

    pub fn set_offset_y(&self, offset_y: f32) -> HResult<()> {
        unsafe {
            self.visual.SetOffsetY_1(offset_y).to_result()
        }
    }

    pub fn render_and_present_solid_frame(&self, composition: &DirectComposition, rgba: &[f32; 4])
                                          -> HResult<()> {
        unsafe {
            // FIXME: arbitrary D3D rendering here?
            composition.d3d_device_context.ClearRenderTargetView(self.render_target_view.as_raw(), &rgba);

            self.swap_chain.Present(0, 0).to_result()
        }
    }
}
