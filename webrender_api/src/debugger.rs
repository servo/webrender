/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{DebugFlags, PictureRect, DeviceRect, RenderCommandInfo};
use crate::image::ImageFormat;

// Shared type definitions between the WR crate and the debugger

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct ProfileCounterId(pub usize);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProfileCounterDescriptor {
    pub id: ProfileCounterId,
    pub name: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProfileCounterUpdate {
    pub id: ProfileCounterId,
    pub value: f64,
}

#[derive(Serialize, Deserialize)]
pub struct SetDebugFlagsMessage {
    pub flags: DebugFlags,
}

#[derive(Serialize, Deserialize)]
pub struct InitProfileCountersMessage {
    pub counters: Vec<ProfileCounterDescriptor>,
}

#[derive(Serialize, Deserialize)]
pub struct FrameLogMessage {
    pub profile_counters: Option<Vec<ProfileCounterUpdate>>,
    pub render_commands: Option<Vec<RenderCommandInfo>>,
}

#[derive(Serialize, Deserialize)]
pub enum DebuggerMessage {
    SetDebugFlags(SetDebugFlagsMessage),
    InitProfileCounters(InitProfileCountersMessage),
    UpdateFrameLog(FrameLogMessage),
}

#[derive(Serialize, Deserialize)]
pub struct CompositorDebugTile {
    pub local_rect: PictureRect,
    pub device_rect: DeviceRect,
    pub clip_rect: DeviceRect,
    pub z_id: i32,
}

#[derive(Serialize, Deserialize)]
pub struct CompositorDebugInfo {
    pub enabled_z_layers: u64,
    pub tiles: Vec<CompositorDebugTile>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DebuggerTextureContent {
    pub name: String,
    pub category: crate::TextureCacheCategory,
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub data: Vec<u8>,
}
