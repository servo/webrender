//! Similar to https://github.com/retep998/wio-rs/blob/44093f7db8/src/com.rs , but can be null

use std::ops::Deref;
use std::ptr;
use winapi;
use winapi::Interface;
use winapi::ctypes::c_void;
use winapi::shared::winerror::HRESULT;
use winapi::um::unknwnbase::IUnknown;

pub trait ToResult: Sized {
    fn to_result(self) -> Result<(), Self>;
}

impl ToResult for HRESULT {
    fn to_result(self) -> Result<(), Self> {
        if winapi::shared::winerror::SUCCEEDED(self) {
            Ok(())
        } else {
            Err(self)
        }
    }
}

/// Nullable pointer to an owned COM object
pub struct Com<T> where T: AsIUnknown {
    ptr: *mut T,
}

impl<T> Com<T> where T: AsIUnknown {
    pub fn null() -> Self {
        Com { ptr: ptr::null_mut() }
    }

    pub fn as_ref(&self) -> Option<&T> {
        unsafe {
            self.ptr.as_ref()
        }
    }

    pub fn as_ptr_ptr(&mut self) -> *mut *mut T {
        &mut self.ptr
    }

    pub fn as_void_ptr_ptr(&mut self) -> *mut *mut c_void {
        self.as_ptr_ptr() as _
    }

    pub fn query_interface<U>(&self) -> Result<Com<U>, HRESULT> where U: AsIUnknown {
        let mut obj = Com::<U>::null();
        unsafe {
            self.as_iunknown().QueryInterface(&U::uuidof(), obj.as_void_ptr_ptr()).to_result()?
        };
        Ok(obj)
    }
}

impl<T> Deref for Com<T> where T: AsIUnknown {
    type Target = T;

    fn deref(&self) -> &T {
        self.as_ref().expect("dereferencing a null COM pointer")
    }
}

impl<T> Drop for Com<T> where T: AsIUnknown {
    fn drop(&mut self) {
        unsafe {
            if let Some(r) = self.ptr.as_ref() {
                r.as_iunknown().Release();
            }
        }
    }
}

/// FIXME: https://github.com/retep998/winapi-rs/issues/571 or some other way to express this generically.
pub trait AsIUnknown: Interface {
    fn as_iunknown(&self) -> &IUnknown;
}

macro_rules! impl_as_iunknown {
    ($( $ty: path, )+) => {
        $(
            impl AsIUnknown for $ty {
                fn as_iunknown(&self) -> &IUnknown { self }
            }
        )+
    }
}

impl_as_iunknown! {
    winapi::shared::dxgi::IDXGIDevice,
    winapi::um::d3d11::ID3D11Device,
    winapi::um::dcomp::IDCompositionDevice,
    winapi::um::dcomp::IDCompositionTarget,
}
