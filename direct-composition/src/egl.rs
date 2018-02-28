use std::ffi::CString;
use std::os::raw::{c_void, c_long};
use std::ptr;
use std::rc::Rc;
use winapi;
use winapi::um::d3d11::ID3D11Device;
use winapi::um::d3d11::ID3D11Texture2D;

pub fn get_proc_address(name: &str) -> *const c_void {
    let name = CString::new(name.as_bytes()).unwrap();
    unsafe {
        GetProcAddress(name.as_ptr()) as *const _ as _
    }
}

pub struct SharedEglThings {
    device: EGLDeviceEXT,
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
        let device = eglCreateDeviceANGLE(
            D3D11_DEVICE_ANGLE,
            d3d_device,
            ptr::null(),
        ).check();
        let display = GetPlatformDisplayEXT(
            PLATFORM_DEVICE_EXT,
            device,
            attributes! [
                EXPERIMENTAL_PRESENT_PATH_ANGLE => EXPERIMENTAL_PRESENT_PATH_FAST_ANGLE,
            ],
        ).check();
        Initialize(display, ptr::null_mut(), ptr::null_mut()).check();

        // Adapted from
        // https://searchfox.org/mozilla-central/rev/056a4057/gfx/gl/GLContextProviderEGL.cpp#635
        let mut configs = [ptr::null(); 64];
        let mut num_configs = 0;
        ChooseConfig(
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
        ).check();
        let config = pick_config(&configs[..num_configs as usize]);

        Rc::new(SharedEglThings { device, display, config })
    }
}

fn pick_config(configs: &[types::EGLConfig]) -> types::EGLConfig {
    // FIXME: better criteria to make this choice?
    // Firefox uses GetConfigAttrib to find a config that has the requested r/g/b/a sizes
    // https://searchfox.org/mozilla-central/rev/056a4057/gfx/gl/GLContextProviderEGL.cpp#662-685

    configs[0]
}

impl Drop for SharedEglThings {
    fn drop(&mut self) {
        unsafe {
            // FIXME does EGLDisplay or EGLConfig need clean up? How?
            eglReleaseDeviceANGLE(self.device).check();
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
        let context = CreateContext(
            shared.display,
            shared.config,
            NO_CONTEXT,
            attributes![],
        ).check();

        let surface = CreatePbufferFromClientBuffer(
            shared.display,
            D3D_TEXTURE_ANGLE,
            buffer as types::EGLClientBuffer,
            shared.config,
            attributes! [
                WIDTH => width,
                HEIGHT => height,
                FLEXIBLE_SURFACE_COMPATIBILITY_SUPPORTED_ANGLE => TRUE,
            ],
        ).check();

        PerVisualEglThings { shared, context, surface }
    }

    pub fn make_current(&self) {
        unsafe {
            MakeCurrent(self.shared.display, self.surface, self.surface, self.context).check();
        }
    }
}

impl Drop for PerVisualEglThings {
    fn drop(&mut self) {
        unsafe {
            DestroyContext(self.shared.display, self.context).check();
            DestroySurface(self.shared.display, self.surface).check();
        }
    }
}

fn check_error() {
    unsafe {
        let error = GetError() as types::EGLenum;
        assert_eq!(error, SUCCESS, "0x{:x} != 0x{:x}", error, SUCCESS);
    }
}

trait Check {
    fn check(self) -> Self;
}

impl Check for *const c_void {
    fn check(self) -> Self {
        check_error();
        assert!(!self.is_null());
        self
    }
}

impl Check for *mut c_void {
    fn check(self) -> Self {
        check_error();
        assert!(!self.is_null());
        self
    }
}

impl Check for types::EGLBoolean {
    fn check(self) -> Self {
        check_error();
        assert_eq!(self, TRUE);
        self
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

    fn eglReleaseDeviceANGLE(device: EGLDeviceEXT) -> types::EGLBoolean;
}
