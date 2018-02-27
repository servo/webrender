use std::os::raw::{c_void, c_long};
use std::ptr;
use winapi;
use winapi::um::d3d11::ID3D11Device;
use winapi::um::d3d11::ID3D11Texture2D;

fn cast_attributes(slice: &[types::EGLenum]) -> &EGLint {
    unsafe {
        &*(slice.as_ptr() as *const EGLint)
    }
}

macro_rules! attributes {
    ($( $key: expr => $value: expr, )*) => {
        cast_attributes(&[
            $( $key, $value, )*
            NONE,
        ])
    }
}
impl Egl {
    fn check_error(&self) {
        unsafe {
            let error = self.GetError() as types::EGLenum;
            assert_eq!(error, SUCCESS, "0x{:x} != 0x{:x}", error, SUCCESS);
        }
    }

    fn check_ptr(&self, p: *const c_void) -> *const c_void {
        self.check_error();
        assert!(!p.is_null());
        p
    }

    fn check_mut_ptr(&self, p: *mut c_void) -> *mut c_void {
        self.check_error();
        assert!(!p.is_null());
        p
    }

    fn check_bool(&self, bool_result: types::EGLBoolean) {
        self.check_error();
        assert_eq!(bool_result, TRUE);
    }

    pub unsafe fn initialize(&self, d3d_device: *mut ID3D11Device) -> types::EGLDisplay {
        let egl_device = self.check_mut_ptr(eglCreateDeviceANGLE(
            D3D11_DEVICE_ANGLE,
            d3d_device,
            ptr::null(),
        ));
        let egl_display = self.check_ptr(self.GetPlatformDisplayEXT(
            PLATFORM_DEVICE_EXT,
            egl_device,
            attributes! [
                EXPERIMENTAL_PRESENT_PATH_ANGLE => EXPERIMENTAL_PRESENT_PATH_FAST_ANGLE,
            ],
        ));
        self.check_bool(self.Initialize(egl_display, ptr::null_mut(), ptr::null_mut()));

        egl_display
    }

    // Adapted from
    // https://searchfox.org/mozilla-central/rev/056a4057/gfx/gl/GLContextProviderEGL.cpp#635
    pub unsafe fn config(&self, display: types::EGLDisplay) -> types::EGLConfig {
        let mut configs = [ptr::null(); 64];
        let mut num_configs = 0;
        self.check_bool(self.ChooseConfig(
            display,
            attributes! [
                SURFACE_TYPE => WINDOW_BIT,
                RENDERABLE_TYPE => OPENGL_ES2_BIT,
                RED_SIZE => 8,
                GREEN_SIZE => 8,
                BLUE_SIZE => 8,
                ALPHA_SIZE => 8,
            ],
            configs.as_mut_ptr(),
            configs.len() as i32,
            &mut num_configs,
        ));
        assert!(num_configs >= 0);
        // FIXME: pick a preferable config?
        configs[0]
    }

    pub unsafe fn create_context(&self, display: types::EGLDisplay, config: types::EGLConfig)
                                 -> types::EGLContext {
        self.check_ptr(self.CreateContext(
            display,
            config,
            NO_CONTEXT,
            attributes![],
        ))
    }

    pub unsafe fn create_surface(&self, display: types::EGLDisplay, buffer: *const ID3D11Texture2D,
                                 config: types::EGLConfig, width: u32, height: u32)
                                 -> types::EGLSurface {
        self.check_ptr(self.CreatePbufferFromClientBuffer(
            display,
            D3D_TEXTURE_ANGLE,
            buffer as types::EGLClientBuffer,
            config,
            attributes! [
                WIDTH => width,
                HEIGHT => height,
                FLEXIBLE_SURFACE_COMPATIBILITY_SUPPORTED_ANGLE => TRUE,
            ],
        ))
    }

    pub unsafe fn make_current(&self, display: types::EGLDisplay, surface: types::EGLSurface,
                               context: types::EGLContext) {
        self.check_bool(self.MakeCurrent(display, surface, surface, context))
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
    fn eglCreateDeviceANGLE(
        device_type: types::EGLenum,
        device: *mut ID3D11Device,
        attrib_list: *const types::EGLAttrib,
    ) -> EGLDeviceEXT;
}
