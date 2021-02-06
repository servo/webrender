/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#![crate_name = "swgl"]
#![crate_type = "lib"]
#![no_std]

extern crate gleam;
#[macro_use]
extern crate alloc;

mod swgl_fns;

pub use crate::swgl_fns::*;
