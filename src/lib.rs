/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#![feature(slice_patterns, step_by, zero_one)]
//#![feature(mpsc_select)]

#[macro_use]
extern crate lazy_static;

mod aabbtree;
mod batch;
mod batch_builder;
mod debug_font_data;
mod debug_render;
mod device;
mod frame;
mod freelist;
mod geometry;
mod internal_types;
mod layer;
mod node_compiler;
mod optimizer;
mod profiler;
mod render_backend;
mod resource_cache;
mod resource_list;
mod scene;
mod tessellator;
mod texture_cache;
mod util;

mod platform {
    #[cfg(target_os="macos")]
    pub use platform::macos::font;
    #[cfg(any(target_os="linux", target_os="android"))]
    pub use platform::linux::font;

    #[cfg(target_os="macos")]
    pub mod macos {
        pub mod font;
    }
    #[cfg(any(target_os="linux", target_os="android"))]
    pub mod linux {
        pub mod font;
    }
    // Temporary solution to pass building on Windows
    #[cfg(target_os = "windows")]
    pub mod font {
        pub struct FontContext;
        impl FontContext {
            pub fn new() -> FontContext {
                FontContext
            }
        }
        pub struct RasterizedGlyph;
        impl RasterizedGlyph {}
    }
}

pub mod renderer;

#[cfg(target_os="macos")]
extern crate core_graphics;
#[cfg(target_os="macos")]
extern crate core_text;

#[cfg(not(target_os="macos"))]
extern crate freetype;

extern crate app_units;
extern crate euclid;
extern crate fnv;
extern crate gleam;
extern crate ipc_channel;
extern crate num;
//extern crate notify;
extern crate scoped_threadpool;
extern crate time;
extern crate webrender_traits;
extern crate offscreen_gl_context;

pub use renderer::{Renderer, RendererOptions};
