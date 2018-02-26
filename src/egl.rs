use std::ptr;
use winapi;
use winapi::ctypes::{c_void, c_long};
use winapi::um::d3d11::ID3D11Device;
use winapi::um::d3d11::ID3D11Texture2D;

impl Egl {
    pub unsafe fn initialize(&self, d3d_device: *mut ID3D11Device) -> types::EGLDisplay {
        let egl_device = eglCreateDeviceANGLE(
            D3D11_DEVICE_ANGLE,
            d3d_device,
            ptr::null(),
        );
        assert!(!egl_device.is_null());
        let attrib_list = [
            EXPERIMENTAL_PRESENT_PATH_ANGLE, EXPERIMENTAL_PRESENT_PATH_FAST_ANGLE,
            NONE,
        ];
        let egl_display = self.GetPlatformDisplayEXT(
            PLATFORM_DEVICE_EXT,
            egl_device,
            attrib_list.as_ptr() as *const i32,
        );
        assert!(!egl_display.is_null());
        assert!(egl_display != NO_DISPLAY);

        self.Initialize(egl_display, ptr::null_mut(), ptr::null_mut());

        egl_display
    }

    // Adapted from
    // https://searchfox.org/mozilla-central/rev/056a4057/gfx/gl/GLContextProviderEGL.cpp#635
    pub unsafe fn config(&self, display: types::EGLSurface) -> types::EGLConfig {
        let mut configs = [ptr::null(); 64];
        let attrib_list = [
            SURFACE_TYPE, WINDOW_BIT,
            RENDERABLE_TYPE, OPENGL_ES2_BIT,
            RED_SIZE, 8,
            GREEN_SIZE, 8,
            BLUE_SIZE, 8,
            ALPHA_SIZE, 8,
            NONE,
        ];
        let mut num_configs = 0;
        let choose_config_result = self.ChooseConfig(
            display,
            attrib_list.as_ptr() as *const i32,
            configs.as_mut_ptr(),
            configs.len() as i32,
            &mut num_configs,
        );
        assert!(choose_config_result != FALSE);
        assert!(num_configs >= 0);
        // FIXME: pick a preferable config?
        configs[0]
    }

    pub unsafe fn create_surface(&self, display: types::EGLSurface, buffer: *const ID3D11Texture2D,
                                 config: types::EGLConfig, width: u32, height: u32)
                                 -> types::EGLSurface {
        let attrib_list = [
            WIDTH, width,
            HEIGHT, height,
            FLEXIBLE_SURFACE_COMPATIBILITY_SUPPORTED_ANGLE, TRUE,
            NONE,
        ];
        let surface = self.CreatePbufferFromClientBuffer(
            display,
            D3D_TEXTURE_ANGLE,
            buffer as types::EGLClientBuffer,
            config,
            attrib_list.as_ptr() as *const i32,
        );
        assert!(!surface.is_null());
        surface
    }
}

// Adapted from https://github.com/tomaka/glutin/blob/1f3b8360cb/src/api/egl/ffi.rs
#[allow(non_camel_case_types)] pub type khronos_utime_nanoseconds_t = khronos_uint64_t;
#[allow(non_camel_case_types)] pub type khronos_uint64_t = u64;
#[allow(non_camel_case_types)] pub type khronos_ssize_t = c_long;
pub type EGLint = i32;
pub type EGLNativeDisplayType = *const c_void;
pub type EGLNativePixmapType = *const c_void;
pub type EGLNativeWindowType = winapi::shared::windef::HWND;
pub type NativeDisplayType = EGLNativeDisplayType;
pub type NativePixmapType = EGLNativePixmapType;
pub type NativeWindowType = EGLNativeWindowType;

include!(concat!(env!("OUT_DIR"), "/egl_bindings.rs"));


// Adapted from https://chromium.googlesource.com/angle/angle/+/master/include/EGL/eglext_angle.h
type EGLDeviceEXT = *mut c_void;
const EXPERIMENTAL_PRESENT_PATH_ANGLE: types::EGLenum = 0x33A4;
const EXPERIMENTAL_PRESENT_PATH_FAST_ANGLE: types::EGLenum = 0x33A9;
const D3D_TEXTURE_ANGLE: types::EGLenum = 0x33A3;
const FLEXIBLE_SURFACE_COMPATIBILITY_SUPPORTED_ANGLE: types::EGLenum = 0x33A6;

extern "C" {
    pub fn eglCreateDeviceANGLE(
        device_type: types::EGLenum,
        device: *mut ID3D11Device,
        attrib_list: *const types::EGLAttrib,
    ) -> EGLDeviceEXT;
}
