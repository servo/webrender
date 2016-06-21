/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#![feature(nonzero)]

extern crate app_units;
extern crate byteorder;
extern crate core;
extern crate euclid;
extern crate gleam;
extern crate ipc_channel;
extern crate offscreen_gl_context;
extern crate serde;

#[cfg(target_os = "macos")] extern crate core_graphics;

include!(concat!(env!("OUT_DIR"), "/types.rs"));

mod api;
mod display_item;
mod display_list;
mod stacking_context;
mod webgl;

pub use api::RenderApi;
pub use display_list::{AuxiliaryListsBuilder, DisplayListBuilder};
