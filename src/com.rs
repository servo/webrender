//! Similar to https://github.com/retep998/wio-rs/blob/44093f7db8/src/com.rs , but can be null

use std::fmt;
use std::ptr;
use winapi::Interface;
use winapi::ctypes::c_void;
use winapi::shared::guiddef::GUID;
use winapi::shared::winerror::HRESULT;
use winapi::shared::winerror::SUCCEEDED;
use winapi::um::unknwnbase::IUnknown;
use wio::com::ComPtr;

pub type HResult<T> = Result<T, HResultError>;

pub struct HResultError(HRESULT);

impl fmt::Debug for HResultError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "0x{:08X}", self.0 as u32)
    }
}

impl From<HRESULT> for HResultError {
    fn from(h: HRESULT) -> Self {
        HResultError(h)
    }
}

pub trait ToResult: Sized {
    fn to_result(self) -> HResult<()>;
}

impl ToResult for HRESULT {
    fn to_result(self) -> HResult<()> {
        if SUCCEEDED(self) {
            Ok(())
        } else {
            Err(HResultError(self))
        }
    }
}

pub trait OutParam<T>: Sized where T: Interface {
    /// For use with APIs that take an interface UUID and
    /// "return" a new COM object through a `*mut *mut c_void` out-parameter.
    ///
    /// # Safety
    ///
    /// `T` must be a COM interface that inherits from `IUnknown`.
    /// If the closure makes the inner pointer non-null,
    /// it must point to a valid COM object that implements `T`.
    /// Ownership of that object is taken.
    unsafe fn new_with_uuid<F>(f: F) -> HResult<Self>
        where F: FnOnce(&GUID, *mut *mut c_void) -> HRESULT
    {
        Self::new_with(|ptr| f(&T::uuidof(), ptr as _))
    }

    /// For use with APIs that "return" a new COM object through a `*mut *mut T` out-parameter.
    ///
    /// # Safety
    ///
    /// `T` must be a COM interface that inherits from `IUnknown`.
    /// If the closure makes the inner pointer non-null,
    /// it must point to a valid COM object that implements `T`.
    /// Ownership of that object is taken.
    unsafe fn new_with<F>(f: F) -> HResult<Self>
        where F: FnOnce(*mut *mut T) -> HRESULT;
}

impl<T> OutParam<T> for ComPtr<T> where T: Interface {
    unsafe fn new_with<F>(f: F) -> HResult<Self>
        where F: FnOnce(*mut *mut T) -> HRESULT
    {
        let mut ptr = ptr::null_mut();
        let status = f(&mut ptr);
        if SUCCEEDED(status) {
            Ok(ComPtr::from_raw(ptr))
        } else {
            if !ptr.is_null() {
                let ptr = ptr as *mut IUnknown;
                (*ptr).Release();
            }
            Err(HResultError(status))
        }
    }
}
