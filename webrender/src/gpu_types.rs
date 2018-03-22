/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{DevicePoint, LayerToWorldTransform, WorldToLayerTransform};
use gpu_cache::{GpuCacheAddress, GpuDataRequest};
use prim_store::EdgeAaSegmentMask;
use render_task::RenderTaskAddress;

// Contains type that must exactly match the same structures declared in GLSL.

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct ZBufferId(i32);

pub struct ZBufferIdGenerator {
    next: i32,
}

impl ZBufferIdGenerator {
    pub fn new() -> ZBufferIdGenerator {
        ZBufferIdGenerator {
            next: 0
        }
    }

    pub fn next(&mut self) -> ZBufferId {
        let id = ZBufferId(self.next);
        self.next += 1;
        id
    }
}

#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[repr(C)]
pub enum RasterizationSpace {
    Local = 0,
    Screen = 1,
}

#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[repr(C)]
pub enum BoxShadowStretchMode {
    Stretch = 0,
    Simple = 1,
}

#[repr(i32)]
#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum BlurDirection {
    Horizontal = 0,
    Vertical,
}

#[derive(Debug)]
#[repr(C)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct BlurInstance {
    pub task_address: RenderTaskAddress,
    pub src_task_address: RenderTaskAddress,
    pub blur_direction: BlurDirection,
}

/// A clipping primitive drawn into the clipping mask.
/// Could be an image or a rectangle, which defines the
/// way `address` is treated.
#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[repr(C)]
pub struct ClipMaskInstance {
    pub render_task_address: RenderTaskAddress,
    pub scroll_node_data_index: ClipScrollNodeIndex,
    pub segment: i32,
    pub clip_data_address: GpuCacheAddress,
    pub resource_address: GpuCacheAddress,
}

// 32 bytes per instance should be enough for anyone!
#[derive(Debug, Clone)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct PrimitiveInstance {
    data: [i32; 8],
}

pub struct SimplePrimitiveInstance {
    pub specific_prim_address: GpuCacheAddress,
    pub task_address: RenderTaskAddress,
    pub clip_task_address: RenderTaskAddress,
    pub clip_chain_rect_index: ClipChainRectIndex,
    pub scroll_id: ClipScrollNodeIndex,
    pub z: ZBufferId,
}

impl SimplePrimitiveInstance {
    pub fn new(
        specific_prim_address: GpuCacheAddress,
        task_address: RenderTaskAddress,
        clip_task_address: RenderTaskAddress,
        clip_chain_rect_index: ClipChainRectIndex,
        scroll_id: ClipScrollNodeIndex,
        z: ZBufferId,
    ) -> Self {
        SimplePrimitiveInstance {
            specific_prim_address,
            task_address,
            clip_task_address,
            clip_chain_rect_index,
            scroll_id,
            z,
        }
    }

    pub fn build(&self, data0: i32, data1: i32, data2: i32) -> PrimitiveInstance {
        PrimitiveInstance {
            data: [
                self.specific_prim_address.as_int(),
                self.task_address.0 as i32,
                self.clip_task_address.0 as i32,
                ((self.clip_chain_rect_index.0 as i32) << 16) | self.scroll_id.0 as i32,
                self.z.0,
                data0,
                data1,
                data2,
            ],
        }
    }
}

pub struct CompositePrimitiveInstance {
    pub task_address: RenderTaskAddress,
    pub src_task_address: RenderTaskAddress,
    pub backdrop_task_address: RenderTaskAddress,
    pub data0: i32,
    pub data1: i32,
    pub z: ZBufferId,
    pub data2: i32,
    pub data3: i32,
}

impl CompositePrimitiveInstance {
    pub fn new(
        task_address: RenderTaskAddress,
        src_task_address: RenderTaskAddress,
        backdrop_task_address: RenderTaskAddress,
        data0: i32,
        data1: i32,
        z: ZBufferId,
        data2: i32,
        data3: i32,
    ) -> Self {
        CompositePrimitiveInstance {
            task_address,
            src_task_address,
            backdrop_task_address,
            data0,
            data1,
            z,
            data2,
            data3,
        }
    }
}

impl From<CompositePrimitiveInstance> for PrimitiveInstance {
    fn from(instance: CompositePrimitiveInstance) -> Self {
        PrimitiveInstance {
            data: [
                instance.task_address.0 as i32,
                instance.src_task_address.0 as i32,
                instance.backdrop_task_address.0 as i32,
                instance.z.0,
                instance.data0,
                instance.data1,
                instance.data2,
                instance.data3,
            ],
        }
    }
}

bitflags! {
    /// Flags that define how the common brush shader
    /// code should process this instance.
    pub struct BrushFlags: u8 {
        const PERSPECTIVE_INTERPOLATION = 0x1;
    }
}

// TODO(gw): While we are comverting things over, we
//           need to have the instance be the same
//           size as an old PrimitiveInstance. In the
//           future, we can compress this vertex
//           format a lot - e.g. z, render task
//           addresses etc can reasonably become
//           a u16 type.
#[repr(C)]
pub struct BrushInstance {
    pub picture_address: RenderTaskAddress,
    pub prim_address: GpuCacheAddress,
    pub clip_chain_rect_index: ClipChainRectIndex,
    pub scroll_id: ClipScrollNodeIndex,
    pub clip_task_address: RenderTaskAddress,
    pub z: ZBufferId,
    pub segment_index: i32,
    pub edge_flags: EdgeAaSegmentMask,
    pub brush_flags: BrushFlags,
    pub user_data: [i32; 3],
}

impl From<BrushInstance> for PrimitiveInstance {
    fn from(instance: BrushInstance) -> Self {
        PrimitiveInstance {
            data: [
                instance.picture_address.0 as i32 | (instance.clip_task_address.0 as i32) << 16,
                instance.prim_address.as_int(),
                ((instance.clip_chain_rect_index.0 as i32) << 16) | instance.scroll_id.0 as i32,
                instance.z.0,
                instance.segment_index |
                    ((instance.edge_flags.bits() as i32) << 16) |
                    ((instance.brush_flags.bits() as i32) << 24),
                instance.user_data[0],
                instance.user_data[1],
                instance.user_data[2],
            ]
        }
    }
}

#[derive(Copy, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[repr(C)]
pub struct ClipScrollNodeIndex(pub u32);

#[derive(Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[repr(C)]
pub struct ClipScrollNodeData {
    pub transform: LayerToWorldTransform,
    pub inv_transform: WorldToLayerTransform,
    pub transform_kind: f32,
    pub padding: [f32; 3],
}

impl ClipScrollNodeData {
    pub fn invalid() -> Self {
        ClipScrollNodeData {
            transform: LayerToWorldTransform::identity(),
            inv_transform: WorldToLayerTransform::identity(),
            transform_kind: 0.0,
            padding: [0.0; 3],
        }
    }
}

#[derive(Copy, Debug, Clone, PartialEq)]
#[repr(C)]
pub struct ClipChainRectIndex(pub usize);

#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[repr(C)]
pub struct ImageSource {
    pub p0: DevicePoint,
    pub p1: DevicePoint,
    pub texture_layer: f32,
    pub user_data: [f32; 3],
}

impl ImageSource {
    pub fn write_gpu_blocks(&self, request: &mut GpuDataRequest) {
        request.push([
            self.p0.x,
            self.p0.y,
            self.p1.x,
            self.p1.y,
        ]);
        request.push([
            self.texture_layer,
            self.user_data[0],
            self.user_data[1],
            self.user_data[2],
        ]);
    }
}
