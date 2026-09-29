/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Texture uploads through pooled, fence-recycled upload buffers, built on
//! the upload buffer and fence operations of `GpuBackend`.

use api::ImageFormat;
use api::units::*;
use crate::render_api::MemoryReport;
use smallvec::SmallVec;
use std::{mem, ptr, slice, thread};
use super::{Device, Fence, FenceStatus, Texture, TransferBuffer, UploadBufferMapping, UploadChunk, UploadMethod, VertexUsageHint};

#[derive(Debug)]
struct PixelBuffer<'a> {
    size_used: usize,
    // small vector avoids heap allocation for a single chunk
    chunks: SmallVec<[UploadChunk<'a>; 1]>,
    inner: UploadPBO,
    mapping: &'a mut [mem::MaybeUninit<u8>],
}

impl<'a> PixelBuffer<'a> {
    fn new(
        pbo: UploadPBO,
    ) -> Self {
        let mapping = unsafe {
            slice::from_raw_parts_mut(pbo.mapping.get_ptr().as_ptr(), pbo.pbo.reserved_size)
        };
        Self {
            size_used: 0,
            chunks: SmallVec::new(),
            inner: pbo,
            mapping,
        }
    }
}

impl<'a> Drop for PixelBuffer<'a> {
    fn drop(&mut self) {
        assert_eq!(self.chunks.len(), 0, "PixelBuffer must be flushed before dropping.");
    }
}

impl UploadBufferMapping {
    fn get_ptr(&self) -> ptr::NonNull<mem::MaybeUninit<u8>> {
        match self {
            UploadBufferMapping::Unmapped => unreachable!("Cannot get pointer to unmapped TransferBuffer."),
            UploadBufferMapping::Transient(ptr) => *ptr,
            UploadBufferMapping::Persistent(ptr) => *ptr,
        }
    }
}

/// A TransferBuffer for uploading texture data, managed by UploadBufferPool.
#[derive(Debug)]
struct UploadPBO {
    pbo: TransferBuffer,
    mapping: UploadBufferMapping,
    can_recycle: bool,
}

impl UploadPBO {
    fn empty() -> Self {
        Self {
            pbo: TransferBuffer {
                id: 0,
                reserved_size: 0,
            },
            mapping: UploadBufferMapping::Unmapped,
            can_recycle: false,
        }
    }
}

/// Allocates and recycles PBOs used for uploading texture data.
/// Tries to allocate and recycle PBOs of a fixed size, but will make exceptions when
/// a larger buffer is required or to work around driver bugs.
pub struct UploadBufferPool {
    /// Usage hint to provide to the driver for optimizations.
    usage_hint: VertexUsageHint,
    /// The preferred size, in bytes, of the buffers to allocate.
    default_size: usize,
    /// List of allocated PBOs ready to be re-used.
    available_buffers: Vec<UploadPBO>,
    /// PBOs which have been returned during the current frame,
    /// and do not yet have an associated sync object.
    returned_buffers: Vec<UploadPBO>,
    /// PBOs which are waiting until their fence is signalled,
    /// indicating they can are ready to be re-used.
    waiting_buffers: Vec<(Fence, Vec<UploadPBO>)>,
    /// PBOs which have been orphaned.
    /// We can recycle their IDs but must reallocate their storage.
    orphaned_buffers: Vec<TransferBuffer>,
}

impl UploadBufferPool {
    pub fn new(device: &mut Device, default_size: usize) -> Self {
        let usage_hint = match *device.upload_method() {
            UploadMethod::Immediate => VertexUsageHint::Stream,
            UploadMethod::PixelBuffer(usage_hint) => usage_hint,
        };
        Self {
            usage_hint,
            default_size,
            available_buffers: Vec::new(),
            returned_buffers: Vec::new(),
            waiting_buffers: Vec::new(),
            orphaned_buffers: Vec::new(),
        }
    }

    /// To be called at the beginning of a series of uploads.
    /// Moves any buffers which are now ready to be used from the waiting list to the ready list.
    pub fn begin_frame(&mut self, device: &mut Device) {
        // Iterate through the waiting buffers and check if each fence has been signalled.
        // If a fence is signalled, move its corresponding buffers to the available list.
        // On error, delete the buffers. Stop when we find the first non-signalled fence,
        // and clean up the signalled fences.
        let mut first_not_signalled = self.waiting_buffers.len();
        for (i, (fence, buffers)) in self.waiting_buffers.iter_mut().enumerate() {
            match device.poll_fence(fence) {
                FenceStatus::Pending => {
                    first_not_signalled = i;
                    break;
                },
                FenceStatus::Signaled => {
                    self.available_buffers.extend(buffers.drain(..));
                }
                FenceStatus::Error => {
                    warn!("fence poll error in UploadBufferPool::begin_frame()");
                    for buffer in buffers.drain(..) {
                        device.delete_transfer_buffer(buffer.pbo);
                    }
                }
            }
        }

        // Delete signalled fences, and remove their now-empty Vecs from waiting_buffers.
        for (fence, _) in self.waiting_buffers.drain(0..first_not_signalled) {
            device.delete_fence(fence);
        }
    }

    // To be called at the end of a series of uploads.
    // Creates a fence, and adds the buffers returned during this frame to waiting_buffers.
    pub fn end_frame(&mut self, device: &mut Device) {
        if !self.returned_buffers.is_empty() {
            match device.create_fence() {
                Some(fence) => {
                    self.waiting_buffers.push((fence, mem::replace(&mut self.returned_buffers, Vec::new())))
                }
                None => {
                    warn!("fence creation error in UploadBufferPool::end_frame()");

                    for buffer in self.returned_buffers.drain(..) {
                        device.delete_transfer_buffer(buffer.pbo);
                    }
                }
            }
        }
    }

    /// Obtain a PBO, either by reusing an existing PBO or allocating a new one.
    /// min_size specifies the minimum required size of the PBO. The returned PBO
    /// may be larger than required.
    fn get_pbo(&mut self, device: &mut Device, min_size: usize) -> Result<UploadPBO, String> {

        // If min_size is smaller than our default size, then use the default size.
        // The exception to this is when due to driver bugs we cannot upload from
        // offsets other than zero within a PBO. In this case, there is no point in
        // allocating buffers larger than required, as they cannot be shared.
        let (can_recycle, size) = if min_size <= self.default_size && device.get_capabilities().supports_nonzero_pbo_offsets {
            (true, self.default_size)
        } else {
            (false, min_size)
        };

        // Try to recycle an already allocated PBO.
        if can_recycle {
            if let Some(mut buffer) = self.available_buffers.pop() {
                assert_eq!(buffer.pbo.reserved_size, size);
                assert!(buffer.can_recycle);

                match buffer.mapping {
                    UploadBufferMapping::Unmapped => {
                        // If buffer was unmapped then transiently map it.
                        let ptr = device.map_upload_buffer(&buffer.pbo)?;
                        buffer.mapping = UploadBufferMapping::Transient(ptr);
                    }
                    UploadBufferMapping::Transient(_) => {
                        unreachable!("Transiently mapped UploadPBO must be unmapped before returning to pool.");
                    }
                    UploadBufferMapping::Persistent(_) => {
                    }
                }

                return Ok(buffer);
            }
        }

        // Try to recycle a PBO ID (but not its allocation) from a previously allocated PBO.
        // If there are none available, create a new PBO.
        let mut pbo = match self.orphaned_buffers.pop() {
            Some(pbo) => pbo,
            None => device.create_transfer_buffer(),
        };

        let persistent = device.get_capabilities().supports_buffer_storage && can_recycle;
        let mapping = device.allocate_upload_buffer(&mut pbo, size, self.usage_hint, persistent)?;

        Ok(UploadPBO { pbo, mapping, can_recycle })
    }

    /// Returns a PBO to the pool. If the PBO is recyclable it is placed in the waiting list.
    /// Otherwise we orphan the allocation immediately, and will subsequently reuse just the ID.
    fn return_pbo(&mut self, device: &mut Device, mut buffer: UploadPBO) {
        assert!(
            !matches!(buffer.mapping, UploadBufferMapping::Transient(_)),
            "Transiently mapped UploadPBO must be unmapped before returning to pool.",
        );

        if buffer.can_recycle {
            self.returned_buffers.push(buffer);
        } else {
            device.orphan_upload_buffer(&mut buffer.pbo);
            self.orphaned_buffers.push(buffer.pbo);
        }
    }

    /// Frees all allocated buffers in response to a memory pressure event.
    pub fn on_memory_pressure(&mut self, device: &mut Device) {
        for buffer in self.available_buffers.drain(..) {
            device.delete_transfer_buffer(buffer.pbo);
        }
        for buffer in self.returned_buffers.drain(..) {
            device.delete_transfer_buffer(buffer.pbo)
        }
        for (fence, buffers) in self.waiting_buffers.drain(..) {
            device.delete_fence(fence);
            for buffer in buffers {
                device.delete_transfer_buffer(buffer.pbo)
            }
        }
        // There is no need to delete orphaned PBOs on memory pressure.
    }

    /// Generates a memory report.
    pub fn report_memory(&self) -> MemoryReport {
        let mut report = MemoryReport::default();
        for buffer in &self.available_buffers {
            report.texture_upload_pbos += buffer.pbo.reserved_size;
        }
        for buffer in &self.returned_buffers {
            report.texture_upload_pbos += buffer.pbo.reserved_size;
        }
        for (_, buffers) in &self.waiting_buffers {
            for buffer in buffers {
                report.texture_upload_pbos += buffer.pbo.reserved_size;
            }
        }
        report
    }

    pub fn deinit(&mut self, device: &mut Device) {
        for buffer in self.available_buffers.drain(..) {
            device.delete_transfer_buffer(buffer.pbo);
        }
        for buffer in self.returned_buffers.drain(..) {
            device.delete_transfer_buffer(buffer.pbo)
        }
        for (fence, buffers) in self.waiting_buffers.drain(..) {
            device.delete_fence(fence);
            for buffer in buffers {
                device.delete_transfer_buffer(buffer.pbo)
            }
        }
        for pbo in self.orphaned_buffers.drain(..) {
            device.delete_transfer_buffer(pbo);
        }
    }
}

/// Used to perform a series of texture uploads.
/// Create using Device::upload_texture(). Perform a series of uploads using either
/// upload(), or stage() and upload_staged(), then call flush().
pub struct TextureUploader<'a> {
    /// A list of buffers containing uploads that need to be flushed.
    buffers: Vec<PixelBuffer<'a>>,
    /// Pool used to obtain PBOs to fill with texture data.
    pub pbo_pool: &'a mut UploadBufferPool,
}

impl<'a> Drop for TextureUploader<'a> {
    fn drop(&mut self) {
        assert!(
            thread::panicking() || self.buffers.is_empty(),
            "TextureUploader must be flushed before it is dropped."
        );
    }
}

/// A buffer used to manually stage data to be uploaded to a texture.
/// Created by calling TextureUploader::stage(), the data can then be written to via get_mapping().
#[derive(Debug)]
pub struct UploadStagingBuffer<'a> {
    /// The PixelBuffer containing this upload.
    buffer: PixelBuffer<'a>,
    /// The offset of this upload within the PixelBuffer.
    offset: usize,
    /// The size of this upload.
    size: usize,
    /// The stride of the data within the buffer.
    stride: usize,
}

impl<'a> UploadStagingBuffer<'a> {
    /// Returns the required stride of the data to be written to the buffer.
    pub fn get_stride(&self) -> usize {
        self.stride
    }

    /// Returns a mapping of the data in the buffer, to be written to.
    pub fn get_mapping(&mut self) -> &mut [mem::MaybeUninit<u8>] {
        &mut self.buffer.mapping[self.offset..self.offset + self.size]
    }
}

impl<'a> TextureUploader<'a> {
    /// Starts a series of uploads. Once they have been performed the uploader
    /// must be flushed with `flush()`.
    pub fn new(device: &mut Device, pbo_pool: &'a mut UploadBufferPool) -> Self {
        pbo_pool.begin_frame(device);

        TextureUploader {
            buffers: Vec::new(),
            pbo_pool,
        }
    }

    /// Returns an UploadStagingBuffer which can be used to manually stage data to be uploaded.
    /// Once the data has been staged, it can be uploaded with upload_staged().
    pub fn stage(
        &mut self,
        device: &mut Device,
        format: ImageFormat,
        size: DeviceIntSize,
    ) -> Result<UploadStagingBuffer<'a>, String> {
        assert!(matches!(device.upload_method(), UploadMethod::PixelBuffer(_)), "Texture uploads should only be staged when using pixel buffers.");

        // for optimal PBO texture uploads the offset and stride of the data in
        // the buffer may have to be a multiple of a certain value.
        let (dst_size, dst_stride) = device.required_upload_size_and_stride(
            size,
            format,
        );

        // Find a pixel buffer with enough space remaining, creating a new one if required.
        let buffer_index = self.buffers.iter().position(|buffer| {
            buffer.size_used + dst_size <= buffer.inner.pbo.reserved_size
        });
        let buffer = match buffer_index {
            Some(i) => self.buffers.swap_remove(i),
            None => PixelBuffer::new(self.pbo_pool.get_pbo(device, dst_size)?),
        };

        if !device.get_capabilities().supports_nonzero_pbo_offsets {
            assert_eq!(buffer.size_used, 0, "TransferBuffer uploads from non-zero offset are not supported.");
        }
        assert!(buffer.size_used + dst_size <= buffer.inner.pbo.reserved_size, "PixelBuffer is too small");

        let offset = buffer.size_used;

        Ok(UploadStagingBuffer {
            buffer,
            offset,
            size: dst_size,
            stride: dst_stride,
        })
    }

    /// Uploads manually staged texture data to the specified texture.
    pub fn upload_staged(
        &mut self,
        device: &mut Device,
        texture: &'a Texture,
        rect: DeviceIntRect,
        format_override: Option<ImageFormat>,
        mut staging_buffer: UploadStagingBuffer<'a>,
    ) -> usize {
        let size = staging_buffer.size;

        staging_buffer.buffer.chunks.push(UploadChunk {
            rect,
            stride: Some(staging_buffer.stride as i32),
            offset: staging_buffer.offset,
            format_override,
            texture,
        });
        staging_buffer.buffer.size_used += staging_buffer.size;

        // Flush the buffer if it is full, otherwise return it to the uploader for further use.
        if staging_buffer.buffer.size_used < staging_buffer.buffer.inner.pbo.reserved_size {
            self.buffers.push(staging_buffer.buffer);
        } else {
            Self::flush_buffer(device, self.pbo_pool, staging_buffer.buffer);
        }

        size
    }

    /// Uploads texture data to the specified texture.
    pub fn upload<T>(
        &mut self,
        device: &mut Device,
        texture: &'a Texture,
        mut rect: DeviceIntRect,
        stride: Option<i32>,
        format_override: Option<ImageFormat>,
        data: *const T,
        len: usize,
    ) -> usize {
        // Textures dimensions may have been clamped by the hardware. Crop the
        // upload region to match.
        let cropped = rect.intersection(
            &DeviceIntRect::from_size(texture.get_dimensions())
        );
        if cfg!(debug_assertions) && cropped.map_or(true, |r| r != rect) {
            warn!("Cropping texture upload {:?} to {:?}", rect, cropped);
        }
        rect = match cropped {
            None => return 0,
            Some(r) => r,
        };

        let bytes_pp = texture.format.bytes_per_pixel() as usize;
        let width_bytes = rect.width() as usize * bytes_pp;

        let src_stride = stride.map_or(width_bytes, |stride| {
            assert!(stride >= 0);
            stride as usize
        });
        let src_size = (rect.height() as usize - 1) * src_stride + width_bytes;
        assert!(src_size <= len * mem::size_of::<T>());

        match *device.upload_method() {
            UploadMethod::Immediate => {
                let src = unsafe { slice::from_raw_parts(data as *const u8, src_size) };
                device.upload_texture_region(
                    texture,
                    rect,
                    Some(src_stride as i32),
                    format_override,
                    src,
                );

                width_bytes * rect.height() as usize
            }
            UploadMethod::PixelBuffer(_) => {
                let mut staging_buffer = match self.stage(device, texture.format, rect.size()) {
                    Ok(staging_buffer) => staging_buffer,
                    Err(_) => return 0,
                };
                let dst_stride = staging_buffer.get_stride();

                unsafe {
                    let src: &[mem::MaybeUninit<u8>] = slice::from_raw_parts(data as *const _, src_size);

                    if src_stride == dst_stride {
                        // the stride is already optimal, so simply copy
                        // the data as-is in to the buffer
                        staging_buffer.get_mapping()[..src_size].copy_from_slice(src);
                    } else {
                        // copy the data line-by-line in to the buffer so
                        // that it has an optimal stride
                        for y in 0..rect.height() as usize {
                            let src_start = y * src_stride;
                            let src_end = src_start + width_bytes;
                            let dst_start = y * staging_buffer.get_stride();
                            let dst_end = dst_start + width_bytes;

                            staging_buffer.get_mapping()[dst_start..dst_end].copy_from_slice(&src[src_start..src_end])
                        }
                    }
                }

                self.upload_staged(device, texture, rect, format_override, staging_buffer)
            }
        }
    }

    fn flush_buffer(device: &mut Device, pbo_pool: &mut UploadBufferPool, mut buffer: PixelBuffer) {
        device.flush_upload_buffer(
            &buffer.inner.pbo,
            &buffer.inner.mapping,
            buffer.size_used,
            &buffer.chunks,
        );
        buffer.chunks.clear();
        if let UploadBufferMapping::Transient(_) = buffer.inner.mapping {
            buffer.inner.mapping = UploadBufferMapping::Unmapped;
        }
        let pbo = mem::replace(&mut buffer.inner, UploadPBO::empty());
        pbo_pool.return_pbo(device, pbo);
    }

    /// Flushes all pending texture uploads. Must be called after all
    /// required upload() or upload_staged() calls have been made.
    pub fn flush(mut self, device: &mut Device) {
        for buffer in self.buffers.drain(..) {
            Self::flush_buffer(device, self.pbo_pool, buffer);
        }
    }
}
