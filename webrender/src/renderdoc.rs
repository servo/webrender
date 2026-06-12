/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! On-demand integration with the RenderDoc in-application capture API.
//!
//! A capture is armed via a `DebugCommand::CaptureRenderDoc` (driven from the
//! WebRender debugger / wrshell); the next composited frame is then wrapped in a
//! RenderDoc frame capture and the path of the written `.rdc` is returned.
//!
//! RenderDoc hooks OpenGL at library-load time, so `librenderdoc` must be loaded
//! *before* the GL context is created. In practice this means launching the
//! host process (e.g. Firefox) with `LD_PRELOAD=.../librenderdoc.so`. We resolve
//! the already-loaded library with `RTLD_NOLOAD` and use the in-app API entry
//! point; we deliberately do not load a fresh copy, since a late load cannot
//! install the GL hooks.

use std::ffi::{c_void, CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;
use std::ptr;

/// Opaque handle to the active graphics device / window. Passing null for both
/// tells RenderDoc to capture whatever device and window are currently active,
/// which is what we want for WebRender's single GL context.
type DevicePointer = *mut c_void;
type WindowHandle = *mut c_void;

type SetCaptureOptionU32Fn = unsafe extern "C" fn(option: u32, val: u32) -> c_int;
type SetCaptureFilePathTemplateFn = unsafe extern "C" fn(path: *const c_char);

// RENDERDOC_CaptureOption values from renderdoc_app.h.
/// Capture all live resources, even those not referenced by the active frame.
const RENDERDOC_OPTION_REF_ALL_RESOURCES: u32 = 8;
type GetNumCapturesFn = unsafe extern "C" fn() -> u32;
type GetCaptureFn = unsafe extern "C" fn(
    idx: u32,
    filename: *mut c_char,
    path_length: *mut u32,
    timestamp: *mut u64,
) -> u32;
type StartFrameCaptureFn = unsafe extern "C" fn(device: DevicePointer, window: WindowHandle);
type EndFrameCaptureFn = unsafe extern "C" fn(device: DevicePointer, window: WindowHandle) -> u32;

/// `eRENDERDOC_API_Version_1_1_2`. We only request this version, so any newer
/// `librenderdoc` returns a struct whose prefix matches the layout below (the
/// RenderDoc ABI is append-only and never reorders existing members).
const RENDERDOC_API_VERSION_1_1_2: c_int = 10102;

/// Mirror of `RENDERDOC_API_1_1_2` from `renderdoc_app.h`. Each member is a
/// function pointer; we only spell out the signatures we actually call and
/// leave the rest as opaque pointers, which keeps the (pointer-sized) layout
/// correct without committing to signatures we don't need.
#[repr(C)]
#[allow(non_snake_case, dead_code)]
struct RenderDocApi {
    GetAPIVersion: *const c_void,
    SetCaptureOptionU32: SetCaptureOptionU32Fn,
    SetCaptureOptionF32: *const c_void,
    GetCaptureOptionU32: *const c_void,
    GetCaptureOptionF32: *const c_void,
    SetFocusToggleKeys: *const c_void,
    SetCaptureKeys: *const c_void,
    GetOverlayBits: *const c_void,
    MaskOverlayBits: *const c_void,
    Shutdown: *const c_void,
    UnloadCrashHandler: *const c_void,
    SetCaptureFilePathTemplate: SetCaptureFilePathTemplateFn,
    GetCaptureFilePathTemplate: *const c_void,
    GetNumCaptures: GetNumCapturesFn,
    GetCapture: GetCaptureFn,
    TriggerCapture: *const c_void,
    IsTargetControlConnected: *const c_void,
    LaunchReplayUI: *const c_void,
    SetActiveWindow: *const c_void,
    StartFrameCapture: StartFrameCaptureFn,
    IsFrameCapturing: *const c_void,
    EndFrameCapture: EndFrameCaptureFn,
    TriggerMultiFrameCapture: *const c_void,
}

type GetApiFn = unsafe extern "C" fn(version: c_int, out: *mut *mut c_void) -> c_int;

#[cfg(unix)]
const RTLD_NOLOAD: i32 = 0x4;

#[cfg(all(unix, not(target_os = "android")))]
const RENDERDOC_LIB: &str = "librenderdoc.so";
#[cfg(target_os = "android")]
const RENDERDOC_LIB: &str = "libVkLayer_GLES_RenderDoc.so";
#[cfg(windows)]
const RENDERDOC_LIB: &str = "renderdoc.dll";

/// A loaded RenderDoc API. The function table pointer is owned by RenderDoc and
/// stays valid for the lifetime of the loaded library, which we keep alive via
/// `_lib`.
struct RenderDocApiHandle {
    api: *const RenderDocApi,
    _lib: libloading::Library,
}

// The render thread is the only user, and the underlying pointer is owned by
// RenderDoc for the lifetime of the process.
unsafe impl Send for RenderDocApiHandle {}

#[cfg(unix)]
fn load_library() -> Result<libloading::Library, libloading::Error> {
    use libloading::os::unix::Library;
    // Only pick up an already-loaded library (injected via LD_PRELOAD); RenderDoc
    // cannot hook GL if loaded after the GL driver, so we don't load it fresh.
    unsafe { Library::open(Some(RENDERDOC_LIB), libloading::os::unix::RTLD_NOW | RTLD_NOLOAD) }
        .map(|lib| lib.into())
}

#[cfg(windows)]
fn load_library() -> Result<libloading::Library, libloading::Error> {
    libloading::os::windows::Library::open_already_loaded(RENDERDOC_LIB).map(|lib| lib.into())
}

fn load_api() -> Result<RenderDocApiHandle, String> {
    let lib = load_library().map_err(|e| {
        format!("{} not loaded ({:?}); launch with LD_PRELOAD={}", RENDERDOC_LIB, e, RENDERDOC_LIB)
    })?;

    let get_api: libloading::Symbol<GetApiFn> = unsafe { lib.get(b"RENDERDOC_GetAPI\0") }
        .map_err(|e| format!("RENDERDOC_GetAPI not found in {}: {:?}", RENDERDOC_LIB, e))?;

    let mut api: *mut c_void = ptr::null_mut();
    let ret = unsafe { get_api(RENDERDOC_API_VERSION_1_1_2, &mut api) };
    if ret != 1 || api.is_null() {
        return Err(format!("RENDERDOC_GetAPI returned {}", ret));
    }

    Ok(RenderDocApiHandle {
        api: api as *const RenderDocApi,
        _lib: lib,
    })
}

/// Drives on-demand RenderDoc frame captures for the WebRender renderer.
pub struct RenderDocCapture {
    handle: Option<RenderDocApiHandle>,
    /// Set when a capture has been requested; consumed by the next render.
    capture_next: bool,
    /// Number of captures RenderDoc had written when the current capture began,
    /// used to detect whether the capture actually produced a new file.
    captures_before: u32,
}

impl RenderDocCapture {
    /// Probe for an already-loaded librenderdoc. Always returns a value; if the
    /// library isn't present, captures will report a helpful error instead.
    pub fn new() -> Self {
        let handle = match load_api() {
            Ok(handle) => {
                info!("RenderDoc: in-app API available");
                // Capture the current contents of all resources so that WebRender's
                // persistent picture-cache tiles (rasterized in earlier frames) replay
                // correctly, even though the captured frame only re-composites them.
                // (SaveAllInitials is implicitly always-on since RenderDoc v1.1.)
                unsafe {
                    ((*handle.api).SetCaptureOptionU32)(RENDERDOC_OPTION_REF_ALL_RESOURCES, 1);
                }
                if let Ok(path) = std::env::var("WR_RENDERDOC_CAPTURE_PATH") {
                    if let Ok(cpath) = CString::new(path.clone()) {
                        unsafe {
                            ((*handle.api).SetCaptureFilePathTemplate)(cpath.as_ptr());
                        }
                        info!("RenderDoc: capture file path template set to {:?}", path);
                    }
                }
                Some(handle)
            }
            Err(reason) => {
                info!("RenderDoc: in-app API unavailable ({})", reason);
                None
            }
        };

        RenderDocCapture {
            handle,
            capture_next: false,
            captures_before: 0,
        }
    }

    /// Request that the next composited frame be captured.
    pub fn arm(&mut self) {
        self.capture_next = true;
    }

    /// Whether the library was found (i.e. captures can succeed).
    pub fn is_available(&self) -> bool {
        self.handle.is_some()
    }

    /// Consume any pending capture request, returning whether the upcoming
    /// frame should be wrapped in a capture.
    pub fn take_request(&mut self) -> bool {
        let armed = self.capture_next && self.handle.is_some();
        self.capture_next = false;
        armed
    }

    /// Begin a frame capture. Must be paired with [`Self::end`].
    pub fn start(&mut self) {
        if let Some(handle) = &self.handle {
            info!("RenderDoc: starting frame capture");
            self.captures_before = unsafe { ((*handle.api).GetNumCaptures)() };
            unsafe {
                ((*handle.api).StartFrameCapture)(ptr::null_mut(), ptr::null_mut());
            }
        }
    }

    /// End a frame capture started with [`Self::start`], returning the path of
    /// the written `.rdc` file on success.
    pub fn end(&mut self) -> Option<PathBuf> {
        let before = self.captures_before;
        let handle = self.handle.as_ref()?;
        unsafe { ((*handle.api).EndFrameCapture)(ptr::null_mut(), ptr::null_mut()) };

        let after = unsafe { ((*handle.api).GetNumCaptures)() };
        if after <= before {
            warn!(
                "RenderDoc: capture produced no file (no active GL capture — is the \
                 host launched with LD_PRELOAD=librenderdoc.so?)"
            );
            return None;
        }
        let path = unsafe { capture_path(handle, after - 1) };
        if let Some(ref p) = path {
            info!("RenderDoc: frame capture written to {:?}", p);
        }
        path
    }
}

/// Query RenderDoc for the path of the capture at `idx`.
unsafe fn capture_path(handle: &RenderDocApiHandle, idx: u32) -> Option<PathBuf> {
    // First call with a null buffer to discover the length (incl. null terminator).
    let mut len: u32 = 0;
    ((*handle.api).GetCapture)(idx, ptr::null_mut(), &mut len, ptr::null_mut());
    if len == 0 {
        return None;
    }
    let mut buf = vec![0u8; len as usize];
    if ((*handle.api).GetCapture)(idx, buf.as_mut_ptr() as *mut c_char, &mut len, ptr::null_mut()) != 1 {
        return None;
    }
    let cstr = CStr::from_ptr(buf.as_ptr() as *const c_char);
    Some(PathBuf::from(cstr.to_string_lossy().into_owned()))
}
