/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#![feature(plugin)]
#![feature(custom_derive)]
#![feature(nonzero)]
#![plugin(serde_macros)]

extern crate app_units;
extern crate byteorder;
extern crate core;
extern crate euclid;
extern crate gleam;
extern crate ipc_channel;
extern crate offscreen_gl_context;
extern crate serde;

#[cfg(target_os = "macos")] extern crate core_graphics;

mod api;
mod display_item;
mod display_list;
mod stacking_context;
mod types;
mod webgl;

pub use api::{ApiMsg, IdNamespace, ResourceId, RenderApi, RenderApiSender, ScrollEventPhase};
pub use display_list::{AuxiliaryLists, AuxiliaryListsBuilder, BuiltDisplayList};
pub use display_list::{DisplayListBuilder, DisplayListItem, IframeInfo, ItemRange};
pub use display_list::{SpecificDisplayListItem};
pub use display_item::{DisplayItem, SpecificDisplayItem, ImageDisplayItem};
pub use display_item::{BorderDisplayItem, GradientDisplayItem, RectangleDisplayItem};
pub use stacking_context::StackingContext;
pub use types::NativeFontHandle;
pub use types::{BorderRadius, BorderSide, BorderStyle, BoxShadowClipMode};
pub use types::{ColorF, ClipRegion, ComplexClipRegion};
pub use types::{DisplayListId, DisplayListMode, ImageRendering};
pub use types::{Epoch, FilterOp, FontKey, FragmentType, GlyphInstance, GradientStop};
pub use types::{ImageFormat, ImageKey, MixBlendMode, PipelineId, RenderNotifier};
pub use types::{ScrollLayerId, ScrollPolicy, ServoStackingContextId, StackingContextId};
pub use types::{ScrollLayerInfo, ScrollLayerState};
pub use webgl::*;
