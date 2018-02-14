//! Similar to https://github.com/retep998/wio-rs/blob/44093f7db8/src/com.rs , but can be null

use std::ptr;
use winapi::Interface;
use winapi::ctypes::c_void;
use winapi::shared::winerror::HRESULT;
use winapi::shared::winerror::SUCCEEDED;
use winapi::um::unknwnbase::IUnknown;
use wio::com::ComPtr;

pub trait OutParam<T>: Sized {
    /// For use with APIs that "return" a new COM object through a `*mut *mut c_void` out-parameter.
    ///
    /// # Safety
    ///
    /// `T` must be a COM interface that inherits from `IUnknown`.
    /// If the closure makes the inner pointer non-null,
    /// it must point to a valid COM object that implements `T`.
    /// Ownership of that object is taken.
    unsafe fn from_void_out_param<F>(f: F) -> Result<Self, HRESULT>
        where F: FnOnce(*mut *mut c_void) -> HRESULT
    {
        Self::from_out_param(|ptr| f(ptr as _))
    }

    /// For use with APIs that "return" a new COM object through a `*mut *mut T` out-parameter.
    ///
    /// # Safety
    ///
    /// `T` must be a COM interface that inherits from `IUnknown`.
    /// If the closure makes the inner pointer non-null,
    /// it must point to a valid COM object that implements `T`.
    /// Ownership of that object is taken.
    unsafe fn from_out_param<F>(f: F) -> Result<Self, HRESULT>
        where F: FnOnce(*mut *mut T) -> HRESULT;
}

impl<T> OutParam<T> for ComPtr<T> where T: Interface {
    unsafe fn from_out_param<F>(f: F) -> Result<Self, HRESULT>
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
            Err(status)
        }
    }
}
