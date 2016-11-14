/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::collections::HashMap;
use euclid::{Rect, Matrix4D};
use gpu_store::{GpuStore, GpuStoreAddress};
use internal_types::DeviceRect;
use prim_store::{ClipData, GpuBlock32, PrimitiveClipSource, PrimitiveStore};
use prim_store::{CLIP_DATA_GPU_SIZE, MASK_DATA_GPU_SIZE};
use tiling::StackingContextIndex;
use util::TransformedRect;
use webrender_traits::{AuxiliaryLists, ImageMask};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ClipAddressRange {
    pub start: GpuStoreAddress,
    pub item_count: u32,
}

type ImageMaskIndex = u16;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct MaskCacheKey {
    pub layer_id: StackingContextIndex,
    pub clip_range: ClipAddressRange,
    pub image: Option<GpuStoreAddress>,
}

impl MaskCacheKey {
    pub fn empty(layer_id: StackingContextIndex) -> MaskCacheKey {
        MaskCacheKey {
            layer_id: layer_id,
            clip_range: ClipAddressRange {
                start: GpuStoreAddress(0),
                item_count: 0,
            },
            image: None,
        }
    }
}

#[derive(Debug)]
pub struct MaskCacheInfo {
    pub key: MaskCacheKey,
    // this is needed to update the ImageMaskData after the
    // ResourceCache allocates/load the actual data
    // will be simplified after the TextureCache upgrade
    pub image: Option<ImageMask>,
    pub device_rect: DeviceRect,
    // Vec<layer transforms>
    // Vec<layer mask images>
}

struct LayerInfo {
    world_transform: Matrix4D<f32>,
    _transformed_rect: TransformedRect,
    local_rect: Rect<f32>,
    parent_id: Option<StackingContextIndex>,
}

pub struct ClipRegionStack {
    layers: HashMap<StackingContextIndex, LayerInfo>,
    image_masks: Vec<ImageMask>,
    current_layer_id: Option<StackingContextIndex>,
    device_pixel_ratio: f32,
}

impl ClipRegionStack {
    pub fn new(device_pixel_ratio: f32) -> ClipRegionStack {
        ClipRegionStack {
            layers: HashMap::new(),
            image_masks: Vec::new(),
            current_layer_id: None,
            device_pixel_ratio: device_pixel_ratio,
        }
    }

    pub fn reset(&mut self) {
        self.layers.clear();
        self.image_masks.clear();
        self.current_layer_id = None;
    }

    pub fn generate(&mut self,
                    source: &PrimitiveClipSource,
                    clip_store: &mut GpuStore<GpuBlock32>,
                    aux_lists: &AuxiliaryLists)
                    -> Option<MaskCacheInfo> {
        let mut clip_key = MaskCacheKey::empty(match self.current_layer_id {
            Some(lid) => lid,
            None => return None,
        });
        let layer = &self.layers[&clip_key.layer_id];
        let mut local_rect = Some(layer.local_rect);

        let image = match source {
            &PrimitiveClipSource::NoClip => None,
            &PrimitiveClipSource::Complex(rect, radius) => {
                let address = clip_store.alloc(CLIP_DATA_GPU_SIZE);
                let slice = clip_store.get_slice_mut(address, CLIP_DATA_GPU_SIZE);
                let data = ClipData::uniform(rect, radius);
                PrimitiveStore::populate_clip_data(slice, data);
                clip_key.clip_range.item_count = 1;
                clip_key.clip_range.start = address;
                local_rect = local_rect.and_then(|r| r.intersection(&rect));
                None
            }
            &PrimitiveClipSource::Region(ref region) => {
                let clips = aux_lists.complex_clip_regions(&region.complex);
                if !clips.is_empty() {
                    let address = clip_store.alloc(CLIP_DATA_GPU_SIZE * clips.len());
                    let slice = clip_store.get_slice_mut(address, CLIP_DATA_GPU_SIZE * clips.len());
                    for (clip, chunk) in clips.iter().zip(slice.chunks_mut(CLIP_DATA_GPU_SIZE)) {
                        let data = ClipData::from_clip_region(clip);
                        PrimitiveStore::populate_clip_data(chunk, data);
                        local_rect = local_rect.and_then(|r| r.intersection(&clip.rect));
                    }
                    clip_key.clip_range.item_count = clips.len() as u32;
                    clip_key.clip_range.start = address;
                }
                if let Some(ref mask) = region.image_mask {
                    let address = clip_store.alloc(MASK_DATA_GPU_SIZE);
                    clip_key.image = Some(address);
                    local_rect = local_rect.and_then(|r| r.intersection(&mask.rect));
                }
                region.image_mask
            }
        };

        if clip_key.clip_range.item_count != 0 || clip_key.image.is_some() {
            let rect = local_rect.unwrap_or(Rect::zero());
            let transformed = TransformedRect::new(&rect,
                                                   &layer.world_transform,
                                                   self.device_pixel_ratio);
            Some(MaskCacheInfo {
                key: clip_key,
                image: image,
                device_rect: transformed.bounding_rect,
            })
        } else {
            None
        }
    }

    pub fn push_layer(&mut self,
                      sc_index: StackingContextIndex,
                      local_transform: &Matrix4D<f32>,
                      rect: &Rect<f32>) {

        let (world_transform, transformed_rect) = {
            let indentity_transform = Matrix4D::identity();
            let current_transform = match self.current_layer_id {
                Some(lid) => &self.layers[&lid].world_transform,
                None => &indentity_transform,
            };
            let tr = TransformedRect::new(rect,
                                          current_transform,
                                          self.device_pixel_ratio);
            let wt = current_transform.pre_mul(local_transform);
            (wt, tr)
        };

        self.layers.insert(sc_index, LayerInfo {
            world_transform: world_transform,
            _transformed_rect: transformed_rect,
            local_rect: rect.clone(),
            parent_id: self.current_layer_id,
        });

        self.current_layer_id = Some(sc_index);
    }

    pub fn pop_layer(&mut self) {
        let lid = self.current_layer_id.unwrap();
        self.current_layer_id = self.layers[&lid].parent_id;
    }
}
