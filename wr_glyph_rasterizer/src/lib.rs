/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! A glyph rasterizer for webrender
//!
//! ## Overview
//!
//! ## Usage
//!

#![allow(unknown_lints, mismatched_lifetime_syntaxes)]

mod gamma_lut;
mod rasterizer;
mod telemetry;
mod types;

pub mod profiler;

pub use rasterizer::*;
pub use types::*;

#[macro_use]
extern crate malloc_size_of_derive;
#[macro_use]
extern crate tracy_rs;
#[macro_use]
extern crate log;
#[allow(unused_imports)]
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate smallvec;

#[cfg(any(feature = "serde"))]
#[macro_use]
extern crate serde;

extern crate malloc_size_of;

pub mod platform {
    #[cfg(all(any(target_os = "ios", target_os = "macos"), not(feature = "fontations")))]
    pub use crate::platform::macos::font;
    #[cfg(any(target_os = "android", all(unix, not(any(target_os = "ios", target_os = "macos", feature = "fontations")))))]
    pub use crate::platform::unix::font;
    #[cfg(target_os = "windows")]
    pub use crate::platform::windows::font;

    #[cfg(feature = "fontations")]
    pub use crate::platform::fontations::font;

    #[cfg(feature = "fontations")]
    pub mod fontations {
        pub mod font;
    }

    #[cfg(all(any(target_os = "ios", target_os = "macos"), not(feature = "fontations")))]
    pub mod macos {
        pub mod font;
    }
    #[cfg(any(target_os = "android", all(unix, not(any(target_os = "macos", target_os = "ios", feature = "fontations")))))]
    pub mod unix {
        pub mod font;
    }
    #[cfg(target_os = "windows")]
    pub mod windows {
        pub mod font;
    }
}
