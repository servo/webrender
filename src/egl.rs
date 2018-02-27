use std::ffi::CString;
use std::os::raw::{c_void, c_long};
use std::ptr;
use std::rc::Rc;
use winapi;
use winapi::um::d3d11::ID3D11Device;
use winapi::um::d3d11::ID3D11Texture2D;

pub struct SharedEglThings {
    functions: Egl,
    display: types::EGLDisplay,
    config: types::EGLConfig,
}

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

impl SharedEglThings {
    pub unsafe fn new(d3d_device: *mut ID3D11Device) -> Rc<Self> {
        let functions = Egl;

        let device = functions.check_mut_ptr(eglCreateDeviceANGLE(
            D3D11_DEVICE_ANGLE,
            d3d_device,
            ptr::null(),
        ));
        let display = functions.check_ptr(functions.GetPlatformDisplayEXT(
            PLATFORM_DEVICE_EXT,
            device,
            attributes! [
                EXPERIMENTAL_PRESENT_PATH_ANGLE => EXPERIMENTAL_PRESENT_PATH_FAST_ANGLE,
            ],
        ));
        functions.check_bool(functions.Initialize(display, ptr::null_mut(), ptr::null_mut()));

        // Adapted from
        // https://searchfox.org/mozilla-central/rev/056a4057/gfx/gl/GLContextProviderEGL.cpp#635
        let mut configs = [ptr::null(); 64];
        let mut num_configs = 0;
        functions.check_bool(functions.ChooseConfig(
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
        let config = configs[0];

        Rc::new(SharedEglThings { functions, display, config })
    }

    pub fn get_proc_address(&self, name: &str) -> *const c_void {
        let name = CString::new(name.as_bytes()).unwrap();
        unsafe {
            self.functions.GetProcAddress(name.as_ptr()) as *const _ as _
        }
    }
}

pub struct PerVisualEglThings {
    shared: Rc<SharedEglThings>,
    context: types::EGLContext,
    surface: types::EGLSurface,
}

impl PerVisualEglThings {
    pub unsafe fn new(shared: Rc<SharedEglThings>, buffer: *const ID3D11Texture2D,
           width: u32, height: u32)
           -> Self {
        let shared = shared.clone();
        let context = shared.functions.check_ptr(shared.functions.CreateContext(
            shared.display,
            shared.config,
            NO_CONTEXT,
            attributes![],
        ));

        let surface = shared.functions.check_ptr(shared.functions.CreatePbufferFromClientBuffer(
            shared.display,
            D3D_TEXTURE_ANGLE,
            buffer as types::EGLClientBuffer,
            shared.config,
            attributes! [
                WIDTH => width,
                HEIGHT => height,
                FLEXIBLE_SURFACE_COMPATIBILITY_SUPPORTED_ANGLE => TRUE,
            ],
        ));

        PerVisualEglThings { shared, context, surface }
    }

    pub fn make_current(&self) {
        unsafe {
            self.shared.functions.check_bool(self.shared.functions.MakeCurrent(
                self.shared.display, self.surface, self.surface, self.context
            ))
        }
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
