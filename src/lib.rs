#![feature(plugin)]
#![feature(custom_derive)]
#![plugin(serde_macros)]

extern crate app_units;
extern crate euclid;
extern crate ipc_channel;
extern crate serde;

#[cfg(target_os="macos")]
extern crate core_graphics;

mod api;
mod display_item;
mod display_list;
mod stacking_context;
mod types;

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
