use winapi;
use winapi::ctypes as libc;

include!(concat!(env!("OUT_DIR"), "/egl_bindings.rs"));

// Adapted from https://github.com/tomaka/glutin/blob/1f3b8360cb/src/api/egl/ffi.rs
#[allow(non_camel_case_types)] pub type khronos_utime_nanoseconds_t = khronos_uint64_t;
#[allow(non_camel_case_types)] pub type khronos_uint64_t = u64;
#[allow(non_camel_case_types)] pub type khronos_ssize_t = libc::c_long;
pub type EGLint = i32;
pub type EGLNativeDisplayType = *const libc::c_void;
pub type EGLNativePixmapType = *const libc::c_void;
pub type EGLNativeWindowType = winapi::shared::windef::HWND;
pub type NativeDisplayType = EGLNativeDisplayType;
pub type NativePixmapType = EGLNativePixmapType;
pub type NativeWindowType = EGLNativeWindowType;
