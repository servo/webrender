/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#![feature(plugin)]
#![feature(custom_derive)]
#![feature(nonzero)]
#![plugin(serde_macros)]

extern crate app_units;
extern crate euclid;
extern crate ipc_channel;
extern crate serde;
extern crate offscreen_gl_context;
extern crate core;
extern crate gleam;

#[cfg(target_os="macos")]
extern crate core_graphics;

mod api;
mod display_item;
mod display_list;
mod stacking_context;
mod types;
mod webgl;

pub use api::{ApiMsg, IdNamespace, ResourceId, RenderApi, RenderApiSender};
pub use display_list::{DisplayListBuilder, DisplayListItem};
pub use display_list::{SpecificDisplayListItem, IframeInfo};
pub use display_item::{DisplayItem, SpecificDisplayItem, ImageDisplayItem};
pub use display_item::{BorderDisplayItem, GradientDisplayItem, RectangleDisplayItem};
pub use stacking_context::StackingContext;
pub use types::NativeFontHandle;
pub use types::{BorderRadius, BorderSide, BorderStyle, BoxShadowClipMode};
pub use types::{ColorF, ClipRegion, ComplexClipRegion};
pub use types::{DisplayListId, DisplayListMode, ImageRendering};
pub use types::{Epoch, FilterOp, FontKey, GlyphInstance, GradientStop};
pub use types::{ImageFormat, ImageKey, MixBlendMode, PipelineId, RenderNotifier};
pub use types::{ScrollLayerId, ScrollPolicy, StackingLevel, StackingContextId};
pub use webgl::*;
