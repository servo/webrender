/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use gpu_store::{GpuStore, GpuStoreAddress};
use prim_store::{ClipData, GpuBlock32, PrimitiveStore};
use prim_store::{CLIP_DATA_GPU_SIZE, MASK_DATA_GPU_SIZE};
use util::{rect_from_points_f, TransformedRect};
use webrender_traits::{AuxiliaryLists, ClipRegion, ImageMask};
use webrender_traits::{DeviceIntRect, LayerRect, LayerToWorldTransform};

const MAX_COORD: f32 = 1.0e+16;

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
impl<'a> From<&'a ClipRegion> for ClipSource {
    fn from(clip_region: &'a ClipRegion) -> ClipSource {
        if clip_region.is_complex() {
            ClipSource::Region(clip_region.clone())
        } else {
            ClipSource::NoClip
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
    pub outer_rect: DeviceIntRect,
}

impl MaskCacheInfo {
    /// Create a new mask cache info. It allocates the GPU store data but leaves
    /// it unitialized for the following `update()` call to deal with.
    pub fn new(source: &ClipSource,
               clip_store: &mut GpuStore<GpuBlock32>)
               -> Option<MaskCacheInfo> {
        let (image, clip_range) = match source {
            &ClipSource::NoClip => return None,
            &ClipSource::Complex(..) => (
                None,
                ClipAddressRange {
                    start: clip_store.alloc(CLIP_DATA_GPU_SIZE),
                    item_count: 1,
                }
            ),
            &ClipSource::Region(ref region) => (
                region.image_mask.map(|info|
                    (info, clip_store.alloc(MASK_DATA_GPU_SIZE))
                ),
                ClipAddressRange {
                    start: if region.complex.length > 0 {
                        clip_store.alloc(CLIP_DATA_GPU_SIZE * region.complex.length)
                    } else {
                        GpuStoreAddress(0)
                    },
                    item_count: region.complex.length as u32,
                }
            ),
        };

        Some(MaskCacheInfo {
            clip_range: clip_range,
            image: image,
            local_rect: None,
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
            match source {
                &ClipSource::NoClip => unreachable!(),
                &ClipSource::Complex(rect, radius) => {
                    let slice = clip_store.get_slice_mut(self.clip_range.start, CLIP_DATA_GPU_SIZE);
                    let data = ClipData::uniform(rect, radius);
                    PrimitiveStore::populate_clip_data(slice, data);
                    debug_assert_eq!(self.clip_range.item_count, 1);
                    local_rect = Some(rect);
                }
                &ClipSource::Region(ref region) => {
                    local_rect = Some(LayerRect::from_untyped(&rect_from_points_f(-MAX_COORD, -MAX_COORD, MAX_COORD, MAX_COORD)));
                    let clips = aux_lists.complex_clip_regions(&region.complex);
                    assert_eq!(self.clip_range.item_count, clips.len() as u32);
                    let slice = clip_store.get_slice_mut(self.clip_range.start, CLIP_DATA_GPU_SIZE * clips.len());
                    for (clip, chunk) in clips.iter().zip(slice.chunks_mut(CLIP_DATA_GPU_SIZE)) {
                        let data = ClipData::from_clip_region(clip);
                        PrimitiveStore::populate_clip_data(chunk, data);
                        local_rect = local_rect.and_then(|r| r.intersection(&clip.rect));
                    }
                }
            };
            self.local_rect = Some(local_rect.unwrap_or(LayerRect::zero()));
        }

        let transformed = TransformedRect::new(self.local_rect.as_ref().unwrap(),
                                               &transform,
                                               device_pixel_ratio);
        self.outer_rect = transformed.bounding_rect;
    }
}
