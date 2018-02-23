use std::fmt;
use std::ops;
use std::ptr;
use winapi::Interface;
use winapi::ctypes::c_void;
use winapi::shared::guiddef::GUID;
use winapi::shared::winerror::HRESULT;
use winapi::shared::winerror::SUCCEEDED;
use winapi::um::unknwnbase::IUnknown;

pub type HResult<T> = Result<T, HResultError>;

/// An error code returned by a Windows API.
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

pub fn as_ptr<T>(x: &T) -> *mut T {
    x as *const T as _
}

/// Forked from https://github.com/retep998/wio-rs/blob/44093f7db8/src/com.rs
#[derive(PartialEq, Debug)]
pub struct ComPtr<T>(*mut T) where T: Interface;

impl<T> ComPtr<T> where T: Interface {
    /// Creates a `ComPtr` to wrap a raw pointer.
    /// It takes ownership over the pointer which means it does __not__ call `AddRef`.
    /// `T` __must__ be a COM interface that inherits from `IUnknown`.
    pub unsafe fn from_raw(ptr: *mut T) -> ComPtr<T> {
        assert!(!ptr.is_null());
        ComPtr(ptr)
    }

    /// For use with APIs that take an interface UUID and
    /// "return" a new COM object through a `*mut *mut c_void` out-parameter.
    pub unsafe fn new_with_uuid<F>(f: F) -> HResult<Self>
        where F: FnOnce(&GUID, *mut *mut c_void) -> HRESULT
    {
        Self::new_with(|ptr| f(&T::uuidof(), ptr as _))
    }

    /// For use with APIs that "return" a new COM object through a `*mut *mut T` out-parameter.
    pub unsafe fn new_with<F>(f: F) -> HResult<Self>
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

    pub fn as_raw(&self) -> *mut T {
        self.0
    }

    fn as_unknown(&self) -> &IUnknown {
        unsafe {
            &*(self.0 as *mut IUnknown)
        }
    }

    /// Performs QueryInterface fun.
    pub fn cast<U>(&self) -> HResult<ComPtr<U>> where U: Interface {
        unsafe {
            ComPtr::<U>::new_with_uuid(|uuid, ptr| self.as_unknown().QueryInterface(uuid, ptr))
        }
    }
}

impl<T> ops::Deref for ComPtr<T> where T: Interface {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.0 }
    }
}

impl<T> Clone for ComPtr<T> where T: Interface {
    fn clone(&self) -> Self {
        unsafe {
            self.as_unknown().AddRef();
            ComPtr(self.0)
        }
    }
}

impl<T> Drop for ComPtr<T> where T: Interface {
    fn drop(&mut self) {
        unsafe {
            self.as_unknown().Release();
        }
    }
}
