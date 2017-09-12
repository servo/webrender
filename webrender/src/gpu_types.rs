/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use gpu_cache::GpuCacheAddress;
use render_task::RenderTaskAddress;

// Contains type that must exactly match the same structures declared in GLSL.

// Instance structure for box shadows being drawn into target cache.
#[derive(Debug)]
#[repr(C)]
pub struct BoxShadowCacheInstance {
    pub prim_address: GpuCacheAddress,
    pub task_index: RenderTaskAddress,
}

#[repr(i32)]
#[derive(Debug)]
pub enum BlurDirection {
    Horizontal = 0,
    Vertical,
}

#[derive(Debug)]
#[repr(C)]
pub struct BlurInstance {
    pub task_address: RenderTaskAddress,
    pub src_task_address: RenderTaskAddress,
    pub blur_direction: BlurDirection,
}
