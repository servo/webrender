/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Stubs for the types contained in webgl_types.rs
//!
//! The API surface provided here should be roughly the same to the one provided
//! in webgl_types, modulo completely compiled-out stuff.

use webrender_traits::WebGLCommand;

pub struct GLContextHandleWrapper;

impl GLContextHandleWrapper {
    pub fn current_native_handle() -> Option<GLContextHandleWrapper> {
        None
    }

    pub fn current_osmesa_handle() -> Option<GLContextHandleWrapper> {
        None
    }
}

pub struct GLContextWrapper;

impl GLContextWrapper {
    pub fn make_current(&self) {
        unreachable!()
    }

    pub fn unbind(&self) {
        unreachable!()
    }

    pub fn apply_command(&self, _: WebGLCommand) {
        unreachable!()
    }
}
