/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use gpu_store::{GpuStore, GpuStoreAddress};
use prim_store::{ClipData, GpuBlock32, PrimitiveStore};
use prim_store::{CLIP_DATA_GPU_SIZE, MASK_DATA_GPU_SIZE};
use util::TransformedRect;
use webrender_traits::{AuxiliaryLists, BorderRadius, ClipRegion, ComplexClipRegion, ImageMask};
use webrender_traits::{DeviceIntRect, DeviceIntSize, LayerRect, LayerToWorldTransform};

#[derive(Clone, Debug)]
pub enum ClipSource {
    NoClip,
    Complex(LayerRect, f32),
    Region(ClipRegion),
}

impl ClipSource {
    pub fn to_rect(&self) -> Option<LayerRect> {
        match self {
            &ClipSource::NoClip => None,
            &ClipSource::Complex(rect, _) => Some(rect),
            &ClipSource::Region(ref region) => Some(region.main),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ClipAddressRange {
    pub start: GpuStoreAddress,
    pub item_count: u32,
}

#[derive(Clone, Debug)]
pub struct MaskCacheInfo {
    pub clip_range: ClipAddressRange,
    pub image: Option<(ImageMask, GpuStoreAddress)>,
    pub local_rect: Option<LayerRect>,
    pub local_inner: Option<LayerRect>,
    pub inner_rect: DeviceIntRect,
    pub outer_rect: DeviceIntRect,
}

impl MaskCacheInfo {
    /// Create a new mask cache info. It allocates the GPU store data but leaves
    /// it unitialized for the following `update()` call to deal with.
    pub fn new(source: &ClipSource,
               is_transformed: bool,
               clip_store: &mut GpuStore<GpuBlock32>)
               -> Option<MaskCacheInfo> {
        let (image, clip_range) = match source {
            &ClipSource::NoClip => return None,
            &ClipSource::Complex(..) => {
                (None,
                ClipAddressRange {
                    start: clip_store.alloc(CLIP_DATA_GPU_SIZE),
                    item_count: 1,
                })
            },
            &ClipSource::Region(ref region) => {
                let count = region.complex.length + if is_transformed {1} else {0};
                (region.image_mask.map(|info|
                    (info, clip_store.alloc(MASK_DATA_GPU_SIZE))),
                ClipAddressRange {
                    start: if count > 0 {
                        clip_store.alloc(CLIP_DATA_GPU_SIZE * count)
                    } else {
                        GpuStoreAddress(0)
                    },
                    item_count: count as u32,
                })
            },
        };

        Some(MaskCacheInfo {
            clip_range: clip_range,
            image: image,
            local_rect: None,
            local_inner: None,
            inner_rect: DeviceIntRect::zero(),
            outer_rect: DeviceIntRect::zero(),
        })
    }

    pub fn update(&mut self,
                  source: &ClipSource,
                  transform: &LayerToWorldTransform,
                  clip_store: &mut GpuStore<GpuBlock32>,
                  device_pixel_ratio: f32,
                  aux_lists: &AuxiliaryLists) {

        if self.local_rect.is_none() {
            let mut local_rect;
            let mut local_inner: Option<LayerRect>;
            match source {
                &ClipSource::NoClip => unreachable!(),
                &ClipSource::Complex(rect, radius) => {
                    let slice = clip_store.get_slice_mut(self.clip_range.start, CLIP_DATA_GPU_SIZE);
                    let data = ClipData::uniform(rect, radius);
                    PrimitiveStore::populate_clip_data(slice, data);
                    debug_assert_eq!(self.clip_range.item_count, 1);
                    local_rect = Some(rect);
                    local_inner = ComplexClipRegion::new(rect, BorderRadius::uniform(radius))
                                                    .get_inner_rect();
                }
                &ClipSource::Region(ref region) => {
                    local_rect = Some(region.main);
                    local_inner = match region.image_mask {
                        Some(ref mask) if !mask.repeat => {
                            local_rect = local_rect.and_then(|r| r.intersection(&mask.rect));
                            None
                        },
                        Some(_) => None,
                        None => local_rect,
                    };
                    let clips = aux_lists.complex_clip_regions(&region.complex);
                    assert_eq!(self.clip_range.item_count, clips.len() as u32 + 1);
                    let slice = clip_store.get_slice_mut(self.clip_range.start, CLIP_DATA_GPU_SIZE * (clips.len() + 1));
                    for (clip, chunk) in clips.iter().zip(slice.chunks_mut(CLIP_DATA_GPU_SIZE)) {
                        let data = ClipData::from_clip_region(clip);
                        PrimitiveStore::populate_clip_data(chunk, data);
                        local_rect = local_rect.and_then(|r| r.intersection(&clip.rect));
                        local_inner = local_inner.and_then(|r| clip.get_inner_rect()
                                                                   .and_then(|ref inner| r.intersection(&inner)));
                    }
                    if slice.len() > CLIP_DATA_GPU_SIZE * clips.len() {
                        // we have an extra clip rect coming from the transformed layer
                        PrimitiveStore::populate_clip_data(&mut slice[CLIP_DATA_GPU_SIZE * clips.len() ..],
                                                           ClipData::uniform(region.main, 0.0));
                    }
                }
            };
            self.local_rect = Some(local_rect.unwrap_or(LayerRect::zero()));
            self.local_inner = local_inner;
        }

        let transformed = TransformedRect::new(self.local_rect.as_ref().unwrap(),
                                               &transform,
                                               device_pixel_ratio);
        self.outer_rect = transformed.bounding_rect;

        self.inner_rect = if let Some(ref inner_rect) = self.local_inner {
            let transformed = TransformedRect::new(inner_rect,
                                                   &transform,
                                                   device_pixel_ratio);
            transformed.inner_rect
        } else {
            DeviceIntRect::new(self.outer_rect.origin, DeviceIntSize::zero())
        }
    }
}
