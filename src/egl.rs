use std::ptr;
use winapi;
use winapi::ctypes::{c_void, c_long};
use winapi::um::d3d11::ID3D11Device;

impl Egl {
    pub unsafe fn initialize(&self, d3d_device: *mut ID3D11Device) -> types::EGLDisplay {
        let egl_device = eglCreateDeviceANGLE(
            D3D11_DEVICE_ANGLE,
            d3d_device,
            ptr::null(),
        );
        assert!(!egl_device.is_null());
        let attrib_list = [
            EXPERIMENTAL_PRESENT_PATH_ANGLE,
            EXPERIMENTAL_PRESENT_PATH_FAST_ANGLE,
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
pub type EGLDeviceEXT = *mut c_void;
pub const EXPERIMENTAL_PRESENT_PATH_ANGLE: types::EGLenum = 0x33A4;
pub const EXPERIMENTAL_PRESENT_PATH_FAST_ANGLE: types::EGLenum = 0x33A9;

extern "C" {
    pub fn eglCreateDeviceANGLE(
        device_type: types::EGLenum,
        device: *mut ID3D11Device,
        attrib_list: *const types::EGLAttrib,
    ) -> EGLDeviceEXT;
}
