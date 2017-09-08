/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::ImageMask;
use border::BorderCornerClipSource;
use clip::{ClipMode, ClipSource};
use gpu_cache::{GpuCache, GpuCacheHandle, ToGpuBlocks};
use prim_store::{CLIP_DATA_GPU_BLOCKS, ClipData, ImageMaskData};

#[derive(Debug, Copy, Clone)]
pub struct ClipAddressRange {
    pub location: GpuCacheHandle,
    item_count: usize,
}

impl ClipAddressRange {
    fn new(count: usize) -> Self {
        ClipAddressRange {
            location: GpuCacheHandle::new(),
            item_count: count,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.item_count == 0
    }

    pub fn get_count(&self) -> usize {
        self.item_count
    }

    fn get_block_count(&self) -> Option<usize> {
        if self.item_count != 0 {
            Some(self.item_count * CLIP_DATA_GPU_BLOCKS)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct MaskCacheInfo {
    /// Clip items that are always applied
    pub complex_clip_range: ClipAddressRange,
    /// Clip items that are only applied if the clip space is transformed from
    /// the local space of target primitive/layer.
    pub layer_clip_range: ClipAddressRange,
    pub image: Option<(ImageMask, GpuCacheHandle)>,
    pub border_corners: Vec<(BorderCornerClipSource, GpuCacheHandle)>,
}

impl MaskCacheInfo {
    /// Create a new mask cache info. It allocates the GPU store data but leaves
    /// it uninitialized for the following `update()` call to deal with.
    pub fn new(clips: &[ClipSource]) -> MaskCacheInfo {
        let mut image = None;
        let mut border_corners = Vec::new();
        let mut complex_clip_count = 0;
        let mut layer_clip_count = 0;

        // Work out how much clip data space we need to allocate
        // and if we have an image mask.
        for clip in clips {
            match *clip {
                ClipSource::RoundedRectangle(..) => {
                    complex_clip_count += 1;
                }
                ClipSource::Rectangle(..) => {
                    layer_clip_count += 1;
                }
                ClipSource::Image(image_mask) => {
                    debug_assert!(image.is_none());     // TODO(gw): Support >1 image mask!
                    image = Some((image_mask, GpuCacheHandle::new()));
                }
                ClipSource::BorderCorner(ref source) => {
                    border_corners.push((source.clone(), GpuCacheHandle::new()));
                }
            }
        }

        MaskCacheInfo {
            complex_clip_range: ClipAddressRange::new(complex_clip_count),
            layer_clip_range: ClipAddressRange::new(layer_clip_count),
            image,
            border_corners,
        }
    }

    pub fn update(&mut self,
                  sources: &[ClipSource],
                  gpu_cache: &mut GpuCache) {
        // update GPU cache data
        if let Some(block_count) = self.complex_clip_range.get_block_count() {
            if let Some(mut request) = gpu_cache.request(&mut self.complex_clip_range.location) {
                for source in sources {
                    if let ClipSource::RoundedRectangle(ref rect, ref radius, mode) = *source {
                        let data = ClipData::rounded_rect(rect, radius, mode);
                        data.write(&mut request);
                    }
                }
                assert_eq!(request.close(), block_count);
            }
        }

        if let Some(block_count) = self.layer_clip_range.get_block_count() {
            if let Some(mut request) = gpu_cache.request(&mut self.layer_clip_range.location) {
                for source in sources {
                    if let ClipSource::Rectangle(rect) = *source {
                        let data = ClipData::uniform(rect, 0.0, ClipMode::Clip);
                        data.write(&mut request);
                    }
                }
                assert_eq!(request.close(), block_count);
            }
        }

        for &mut (ref mut border_source, ref mut gpu_location) in &mut self.border_corners {
            if let Some(request) = gpu_cache.request(gpu_location) {
                border_source.write(request);
            }
        }

        if let Some((ref mask, ref mut gpu_location)) = self.image {
            if let Some(request) = gpu_cache.request(gpu_location) {
                let data = ImageMaskData {
                    local_rect: mask.rect,
                };
                data.write_gpu_blocks(request);
            }
        }
    }

    /// Check if this `MaskCacheInfo` actually carries any masks.
    pub fn is_masking(&self) -> bool {
        self.image.is_some() ||
        self.complex_clip_range.item_count != 0 ||
        self.layer_clip_range.item_count != 0 ||
        !self.border_corners.is_empty()
    }
}
