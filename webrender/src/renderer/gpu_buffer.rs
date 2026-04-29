/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/*
    TODO:
        Efficiently allow writing to buffer (better push interface)
 */

use std::i32;

use crate::gpu_types::UvRectKind;
use crate::internal_types::{FrameId, FrameMemory, FrameVec};
use crate::renderer::MAX_VERTEX_TEXTURE_WIDTH;
use crate::util::ScaleOffset;
use api::units::{DeviceIntPoint, DeviceIntRect, DeviceIntSize, DeviceRect, LayoutRect, PictureRect};
use api::{PremultipliedColorF, ImageFormat};
use crate::device::Texel;
use crate::render_task_graph::{RenderTaskGraph, RenderTaskId};

pub struct GpuBufferBuilder {
    pub i32: GpuBufferBuilderI,
    pub f32: GpuBufferBuilderF,
}

pub type GpuBufferF = GpuBuffer<GpuBufferBlockF>;
pub type GpuBufferBuilderF = GpuBufferBuilderImpl<GpuBufferBlockF>;

pub type GpuBufferI = GpuBuffer<GpuBufferBlockI>;
pub type GpuBufferBuilderI = GpuBufferBuilderImpl<GpuBufferBlockI>;

pub type GpuBufferWriterF<'l> = GpuBufferWriter<'l, GpuBufferBlockF>;
pub type GpuBufferWriterI<'l> = GpuBufferWriter<'l, GpuBufferBlockI>;

unsafe impl Texel for GpuBufferBlockF {
    fn image_format() -> ImageFormat { ImageFormat::RGBAF32 }
}

unsafe impl Texel for GpuBufferBlockI {
    fn image_format() -> ImageFormat { ImageFormat::RGBAI32 }
}

impl Default for GpuBufferBlockF {
    fn default() -> Self {
        GpuBufferBlockF::EMPTY
    }
}

impl Default for GpuBufferBlockI {
    fn default() -> Self {
        GpuBufferBlockI::EMPTY
    }
}

/// A single texel in RGBAF32 texture - 16 bytes.
#[derive(Copy, Clone, Debug, MallocSizeOf)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct GpuBufferBlockF {
    data: [f32; 4],
}

/// A single texel in RGBAI32 texture - 16 bytes.
#[derive(Copy, Clone, Debug, MallocSizeOf)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct GpuBufferBlockI {
    data: [i32; 4],
}

/// GpuBuffer handle is similar to GpuBufferAddress with additional checks
/// to avoid accidentally using the same handle in multiple frames.
///
/// Do not send GpuBufferHandle to the GPU directly. Instead use a GpuBuffer
/// or GpuBufferBuilder to resolve the handle into a GpuBufferAddress that
/// can be placed into GPU data.
///
/// The extra checks consists into storing an 8 bit epoch in the upper 8 bits
/// of the handle. The epoch will be reused every 255 frames so this is not
/// a mechanism that one can rely on to store and reuse handles over multiple
/// frames. It is only a mechanism to catch mistakes where a handle is
/// accidentally used in the wrong frame and panic.
#[repr(transparent)]
#[derive(Copy, Clone, MallocSizeOf, Eq, PartialEq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct GpuBufferHandle(u32);

impl GpuBufferHandle {
    pub const INVALID: GpuBufferHandle = GpuBufferHandle(u32::MAX - 1);
    const EPOCH_MASK: u32 = 0xFC000000; // Leading 6 bits

    fn new(addr: u32, epoch: u32) -> Self {
        Self(addr | epoch)
    }

    pub fn address_unchecked(&self) -> GpuBufferAddress {
        GpuBufferAddress(self.0 & !Self::EPOCH_MASK)
    }
}

impl std::fmt::Debug for GpuBufferHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let addr = self.0 & !Self::EPOCH_MASK;
        let epoch = (self.0 & Self::EPOCH_MASK) >> 26;
        write!(f, "#{addr}@{epoch}")
    }
}

// TODO(gw): Temporarily encode GPU Cache addresses as a single int.
//           In the future, we can change the PrimitiveInstanceData struct
//           to use 2x u16 for the vertex attribute instead of an i32.
#[repr(transparent)]
#[derive(Copy, Clone, MallocSizeOf, Eq, PartialEq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct GpuBufferAddress(u32);

impl GpuBufferAddress {
    pub fn new(u: u16, v: u16) -> Self {
        GpuBufferAddress(
            v as u32 * MAX_VERTEX_TEXTURE_WIDTH as u32 + u as u32
        )
    }

    pub fn is_valid(&self) -> bool {
        *self != Self::INVALID
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }

    pub fn from_u32(val: u32) -> Self {
        GpuBufferAddress(val)
    }

    #[allow(dead_code)]
    pub fn as_int(self) -> i32 {
        self.0 as i32
    }

    #[allow(dead_code)]
    pub fn uv(self) -> (u16, u16) {
        (
            (self.0 as usize % MAX_VERTEX_TEXTURE_WIDTH) as u16,
            (self.0 as usize / MAX_VERTEX_TEXTURE_WIDTH) as u16,
        )
    }

    pub const INVALID: GpuBufferAddress = GpuBufferAddress(u32::MAX - 1);
}

impl std::fmt::Debug for GpuBufferAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if *self == Self::INVALID {
            write!(f, "<invalid>")
        } else {
            write!(f, "#{}", self.0)
        }
    }
}

impl GpuBufferBlockF {
    pub const EMPTY: Self = GpuBufferBlockF { data: [0.0; 4] };
}

impl GpuBufferBlockI {
    pub const EMPTY: Self = GpuBufferBlockI { data: [0; 4] };
}

impl Into<GpuBufferBlockF> for LayoutRect {
    fn into(self) -> GpuBufferBlockF {
        GpuBufferBlockF {
            data: [
                self.min.x,
                self.min.y,
                self.max.x,
                self.max.y,
            ],
        }
    }
}

impl Into<GpuBufferBlockF> for crate::quad::LayoutOrDeviceRect {
    fn into(self) -> GpuBufferBlockF {
        GpuBufferBlockF {
            data: [
                self.min.x,
                self.min.y,
                self.max.x,
                self.max.y,
            ],
        }
    }
}

impl Into<GpuBufferBlockF> for ScaleOffset {
    fn into(self) -> GpuBufferBlockF {
        GpuBufferBlockF {
            data: [
                self.scale.x,
                self.scale.y,
                self.offset.x,
                self.offset.y,
            ],
        }
    }
}

impl Into<GpuBufferBlockF> for PictureRect {
    fn into(self) -> GpuBufferBlockF {
        GpuBufferBlockF {
            data: [
                self.min.x,
                self.min.y,
                self.max.x,
                self.max.y,
            ],
        }
    }
}

impl Into<GpuBufferBlockF> for DeviceRect {
    fn into(self) -> GpuBufferBlockF {
        GpuBufferBlockF {
            data: [
                self.min.x,
                self.min.y,
                self.max.x,
                self.max.y,
            ],
        }
    }
}

impl Into<GpuBufferBlockF> for PremultipliedColorF {
    fn into(self) -> GpuBufferBlockF {
        GpuBufferBlockF {
            data: [
                self.r,
                self.g,
                self.b,
                self.a,
            ],
        }
    }
}

impl From<DeviceIntRect> for GpuBufferBlockF {
    fn from(rect: DeviceIntRect) -> Self {
        GpuBufferBlockF {
            data: [
                rect.min.x as f32,
                rect.min.y as f32,
                rect.max.x as f32,
                rect.max.y as f32,
            ],
        }
    }
}

impl From<DeviceIntRect> for GpuBufferBlockI {
    fn from(rect: DeviceIntRect) -> Self {
        GpuBufferBlockI {
            data: [
                rect.min.x,
                rect.min.y,
                rect.max.x,
                rect.max.y,
            ],
        }
    }
}

impl Into<GpuBufferBlockF> for [f32; 4] {
    fn into(self) -> GpuBufferBlockF {
        GpuBufferBlockF {
            data: self,
        }
    }
}

impl Into<GpuBufferBlockI> for [i32; 4] {
    fn into(self) -> GpuBufferBlockI {
        GpuBufferBlockI {
            data: self,
        }
    }
}

pub trait GpuBufferDataF {
    const NUM_BLOCKS: usize;
    fn write(&self, writer: &mut GpuBufferWriterF);
}

pub trait GpuBufferDataI {
    const NUM_BLOCKS: usize;
    fn write(&self, writer: &mut GpuBufferWriterI);
}

impl GpuBufferDataF for [f32; 4] {
    const NUM_BLOCKS: usize = 1;
    fn write(&self, writer: &mut GpuBufferWriterF) {
        writer.push_one(*self);
    }
}

impl GpuBufferDataI for [i32; 4] {
    const NUM_BLOCKS: usize = 1;
    fn write(&self, writer: &mut GpuBufferWriterI) {
        writer.push_one(*self);
    }
}

/// Record a patch to the GPU buffer for a render task
struct DeferredBlock {
    task_id: RenderTaskId,
    index: usize,
}

/// Interface to allow writing multiple GPU blocks, possibly of different types
pub struct GpuBufferWriter<'a, T> {
    buffer: &'a mut FrameVec<T>,
    deferred: &'a mut Vec<DeferredBlock>,
    index: usize,
    max_block_count: usize,
    epoch: u32,
}

impl<'a, T> GpuBufferWriter<'a, T> where T: Texel {
    fn new(
        buffer: &'a mut FrameVec<T>,
        deferred: &'a mut Vec<DeferredBlock>,
        index: usize,
        max_block_count: usize,
        epoch: u32,
    ) -> Self {
        GpuBufferWriter {
            buffer,
            deferred,
            index,
            max_block_count,
            epoch,
        }
    }

    /// Push one (16 byte) block of data in to the writer
    pub fn push_one<B>(&mut self, block: B) where B: Into<T> {
        self.buffer.push(block.into());
    }

    /// Push a reference to a render task in to the writer. Once the render
    /// task graph is resolved, this will be patched with the UV rect of the task
    pub fn push_render_task(&mut self, task_id: RenderTaskId) {
        if task_id != RenderTaskId::INVALID {
            self.deferred.push(DeferredBlock {
                task_id,
                index: self.buffer.len(),
            });
        }

        self.buffer.push(T::default());
    }

    /// Close this writer, returning the GPU address of this set of block(s).
    pub fn finish(self) -> GpuBufferAddress {
        assert!(self.buffer.len() <= self.index + self.max_block_count);

        GpuBufferAddress(self.index as u32)
    }

    /// Close this writer, returning the GPU address of this set of block(s).
    pub fn finish_with_handle(self) -> GpuBufferHandle {
        assert!(self.buffer.len() <= self.index + self.max_block_count);
        assert_eq!(self.index & (GpuBufferHandle::EPOCH_MASK as usize), 0);

        GpuBufferHandle::new(self.index as u32, self.epoch)
    }
}

impl<'a> GpuBufferWriterF<'a> {
    pub fn push<Data: GpuBufferDataF>(&mut self, data: &Data) {
        let _start_index = self.buffer.len();
        data.write(self);
        debug_assert_eq!(self.buffer.len() - _start_index, Data::NUM_BLOCKS);
    }
}

impl<'a> GpuBufferWriterI<'a> {
    pub fn push<Data: GpuBufferDataI>(&mut self, data: &Data) {
        data.write(self);
    }
}

impl<'a, T> Drop for GpuBufferWriter<'a, T> {
    fn drop(&mut self) {
        assert!(self.buffer.len() <= self.index + self.max_block_count, "Attempt to write too many GpuBuffer blocks");
    }
}

pub struct GpuBufferBuilderImpl<T> {
    // `data` will become the backing store of the GpuBuffer sent along
    // with the frame so it uses the frame allocator.
    data: FrameVec<T>,
    // `deferred` is only used during frame building and not sent with the
    // built frame, so it does not use the same allocator.
    deferred: Vec<DeferredBlock>,

    epoch: u32,
}

impl<T> GpuBufferBuilderImpl<T> where T: Texel + std::convert::From<DeviceIntRect> {
    pub fn new(memory: &FrameMemory, capacity: usize, frame_id: FrameId) -> Self {
        // Pick the first 8 bits of the frame id and store them in the upper bits
        // of the handles.
        let epoch = ((frame_id.as_u64() % 62) as u32 + 1) << 26;
        GpuBufferBuilderImpl {
            data: memory.new_vec_with_capacity(capacity),
            deferred: Vec::new(),
            epoch,
        }
    }

    #[allow(dead_code)]
    pub fn push_blocks(
        &mut self,
        blocks: &[T],
    ) -> GpuBufferAddress {
        assert!(blocks.len() <= MAX_VERTEX_TEXTURE_WIDTH);

        ensure_row_capacity(&mut self.data, blocks.len());

        let index = self.data.len();

        self.data.extend_from_slice(blocks);

        GpuBufferAddress(index as u32 | self.epoch)
    }

    /// Begin writing a specific number of blocks
    pub fn write_blocks(
        &mut self,
        max_block_count: usize,
    ) -> GpuBufferWriter<T> {
        assert!(max_block_count <= MAX_VERTEX_TEXTURE_WIDTH);

        ensure_row_capacity(&mut self.data, max_block_count);

        let index = self.data.len();

        GpuBufferWriter::new(
            &mut self.data,
            &mut self.deferred,
            index,
            max_block_count,
            self.epoch,
        )
    }

    // Reserve space in the gpu buffer for data that will be written by the
    // renderer.
    pub fn reserve_renderer_deferred_blocks(&mut self, block_count: usize) -> GpuBufferHandle {
        ensure_row_capacity(&mut self.data, block_count);

        let index = self.data.len();

        self.data.reserve(block_count);
        for _ in 0 ..block_count {
            self.data.push(Default::default());
        }

        GpuBufferHandle::new(index as u32, self.epoch)
    }

    pub fn finalize(
        mut self,
        render_tasks: &RenderTaskGraph,
    ) -> GpuBuffer<T> {
        finish_row(&mut self.data);

        let len = self.data.len();
        assert!(len % MAX_VERTEX_TEXTURE_WIDTH == 0);

        // At this point, we know that the render task graph has been built, and we can
        // query the location of any dynamic (render target) or static (texture cache)
        // task. This allows us to patch the UV rects in to the GPU buffer before upload
        // to the GPU.
        for block in self.deferred.drain(..) {
            let render_task = &render_tasks[block.task_id];
            let target_rect = render_task.get_target_rect();

            let uv_rect = match render_task.uv_rect_kind() {
                UvRectKind::Rect => {
                    target_rect
                }
                UvRectKind::Quad { top_left, bottom_right, .. } => {
                    let size = target_rect.size();

                    DeviceIntRect::new(
                        DeviceIntPoint::new(
                            target_rect.min.x + (top_left.x * size.width as f32).round() as i32,
                            target_rect.min.y + (top_left.y * size.height as f32).round() as i32,
                        ),
                        DeviceIntPoint::new(
                            target_rect.min.x + (bottom_right.x * size.width as f32).round() as i32,
                            target_rect.min.y + (bottom_right.y * size.height as f32).round() as i32,
                        ),
                    )
                }
            };

            self.data[block.index] = uv_rect.into();
        }

        GpuBuffer {
            data: self.data,
            size: DeviceIntSize::new(MAX_VERTEX_TEXTURE_WIDTH as i32, (len / MAX_VERTEX_TEXTURE_WIDTH) as i32),
            format: T::image_format(),
            epoch: self.epoch,
        }
    }

    pub fn resolve_handle(&self, handle: GpuBufferHandle) -> GpuBufferAddress {
        if handle == GpuBufferHandle::INVALID {
            return GpuBufferAddress::INVALID;
        }

        let epoch = handle.0 & GpuBufferHandle::EPOCH_MASK;
        assert!(self.epoch == epoch);

        GpuBufferAddress(handle.0 & !GpuBufferHandle::EPOCH_MASK)
    }

    /// Panics if the handle cannot be used this frame.
    #[allow(unused)]
    pub fn check_handle(&self, handle: GpuBufferHandle) {
        if handle == GpuBufferHandle::INVALID {
            return;
        }
        let epoch = handle.0 & GpuBufferHandle::EPOCH_MASK;
        assert!(self.epoch == epoch);
    }
}

impl GpuBufferBuilderF {
    pub fn push<D>(&mut self, data: &D) -> GpuBufferAddress
        where D: GpuBufferDataF
    {
        let mut writer = self.write_blocks(D::NUM_BLOCKS);
        data.write(&mut writer);

        writer.finish()
    }
}

impl GpuBufferBuilderI {
    pub fn push<D>(&mut self, data: &D) -> GpuBufferAddress
        where D: GpuBufferDataI
    {
        let mut writer = self.write_blocks(D::NUM_BLOCKS);
        data.write(&mut writer);

        writer.finish()
    }
}

fn ensure_row_capacity<T: Default>(data: &mut FrameVec<T>, cap: usize) {
    if (data.len() % MAX_VERTEX_TEXTURE_WIDTH) + cap > MAX_VERTEX_TEXTURE_WIDTH {
        finish_row(data);
    }
}

fn finish_row<T: Default>(data: &mut FrameVec<T>) {
    let required_len = (data.len() + MAX_VERTEX_TEXTURE_WIDTH-1) & !(MAX_VERTEX_TEXTURE_WIDTH-1);
    for _ in 0 .. required_len - data.len() {
        data.push(T::default());
    }
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct GpuBuffer<T> {
    pub data: FrameVec<T>,
    pub size: DeviceIntSize,
    pub format: ImageFormat,
    epoch: u32,
}

impl<T> GpuBuffer<T> {
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn resolve_handle(&self, handle: GpuBufferHandle) -> GpuBufferAddress {
        if handle == GpuBufferHandle::INVALID {
            return GpuBufferAddress::INVALID;
        }

        let epoch = handle.0 & GpuBufferHandle::EPOCH_MASK;
        assert!(self.epoch == epoch);

        GpuBufferAddress(handle.0 & !GpuBufferHandle::EPOCH_MASK)
    }
}

#[test]
fn test_gpu_buffer_sizing_push() {
    let frame_memory = FrameMemory::fallback();
    let render_task_graph = RenderTaskGraph::new_for_testing();
    let mut builder = GpuBufferBuilderF::new(&frame_memory, 0, FrameId::first());

    let row = vec![GpuBufferBlockF::EMPTY; MAX_VERTEX_TEXTURE_WIDTH];
    builder.push_blocks(&row);

    builder.push_blocks(&[GpuBufferBlockF::EMPTY]);
    builder.push_blocks(&[GpuBufferBlockF::EMPTY]);

    let buffer = builder.finalize(&render_task_graph);
    assert_eq!(buffer.data.len(), MAX_VERTEX_TEXTURE_WIDTH * 2);
}

#[test]
fn test_gpu_buffer_sizing_writer() {
    let frame_memory = FrameMemory::fallback();
    let render_task_graph = RenderTaskGraph::new_for_testing();
    let mut builder = GpuBufferBuilderF::new(&frame_memory, 0, FrameId::first());

    let mut writer = builder.write_blocks(MAX_VERTEX_TEXTURE_WIDTH);
    for _ in 0 .. MAX_VERTEX_TEXTURE_WIDTH {
        writer.push_one(GpuBufferBlockF::EMPTY);
    }
    writer.finish();

    let mut writer = builder.write_blocks(1);
    writer.push_one(GpuBufferBlockF::EMPTY);
    writer.finish();

    let mut writer = builder.write_blocks(1);
    writer.push_one(GpuBufferBlockF::EMPTY);
    writer.finish();

    let buffer = builder.finalize(&render_task_graph);
    assert_eq!(buffer.data.len(), MAX_VERTEX_TEXTURE_WIDTH * 2);
}
