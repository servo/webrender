/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::collections::HashMap;
use device::TextureId;
use euclid::{Rect, Matrix4D};
use gpu_store::{GpuStore, GpuStoreAddress};
use prim_store::{ClipData, GpuBlock32, PrimitiveClipSource, PrimitiveStore};
use prim_store::{CLIP_DATA_GPU_SIZE, MASK_DATA_GPU_SIZE};
use tiling::StackingContextIndex;
use util::{TransformedRect, TransformedRectKind};
use webrender_traits::{AuxiliaryLists, ImageMask};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ClipAddressRange {
    pub start: GpuStoreAddress, // start GPU address
    pub count: u32, // number of items, not bytes
}

type ImageMaskIndex = u16;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct MaskCacheKey {
    pub layer_id: StackingContextIndex,
    pub clip_range: ClipAddressRange,
    pub image: Option<GpuStoreAddress>,
}

#[derive(Debug)]
pub struct MaskCacheInfo {
    pub key: MaskCacheKey,
    pub mask_texture_id: TextureId,
    // Vec<layer transforms>
    // Vec<layer mask images>
}

struct LayerInfo {
    world_transform: Matrix4D<f32>,
    transformed_rect: TransformedRect,
    mask_image: Option<ImageMask>,
    //mask_texture_id: TextureId,
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

    fn need_mask(&self) -> bool {
        let mut layer_id = self.current_layer_id;
        while let Some(lid) = layer_id {
            let layer = &self.layers[&lid];
            if layer.transformed_rect.kind == TransformedRectKind::Complex ||
               layer.mask_image.is_some() {
                return true
            }
            layer_id = layer.parent_id;
        }
        false
    }

    pub fn generate(&mut self,
                    source: &PrimitiveClipSource,
                    clip_store: &mut GpuStore<GpuBlock32>,
                    aux_lists: &AuxiliaryLists)
                    -> Option<MaskCacheInfo> {
        let mut clip_key = MaskCacheKey {
            layer_id: match self.current_layer_id {
                Some(lid) => lid,
                None => return None,
            },
            clip_range: ClipAddressRange {
                start: GpuStoreAddress(0),
                count: 0,
            },
            image: None,
        };
        match source {
            &PrimitiveClipSource::NoClip => (),
            &PrimitiveClipSource::Complex(rect, radius) => {
                let address = clip_store.alloc(CLIP_DATA_GPU_SIZE);
                let slice = clip_store.get_slice_mut(address, CLIP_DATA_GPU_SIZE);
                let data = ClipData::uniform(rect, radius);
                PrimitiveStore::populate_clip_data(slice, data);
                clip_key.clip_range.count = 1;
                clip_key.clip_range.start = address;
            },
            &PrimitiveClipSource::Region(ref region) => {
                let clips = aux_lists.complex_clip_regions(&region.complex);
                if !clips.is_empty() {
                    let address = clip_store.alloc(CLIP_DATA_GPU_SIZE * clips.len());
                    let slice = clip_store.get_slice_mut(address, CLIP_DATA_GPU_SIZE * clips.len());
                    for (clip, chunk) in clips.iter().zip(slice.chunks_mut(CLIP_DATA_GPU_SIZE)) {
                        let data = ClipData::from_clip_region(clip);
                        PrimitiveStore::populate_clip_data(chunk, data);
                    }
                    clip_key.clip_range.count = clips.len() as u32;
                    clip_key.clip_range.start = address;
                }
                if region.image_mask.is_some() {
                    let address = clip_store.alloc(MASK_DATA_GPU_SIZE);
                    clip_key.image = Some(address);
                }
            },
        };
        if self.need_mask() ||
           clip_key.clip_range.count != 0 ||
           clip_key.image.is_some() {
            Some(MaskCacheInfo {
                key: clip_key,
                mask_texture_id: TextureId::invalid(),
            })
        } else {
            None
        }
    }

    pub fn push_layer(&mut self,
                      sc_index: StackingContextIndex,
                      local_transform: &Matrix4D<f32>,
                      rect: &Rect<f32>,
                      mask_image: Option<ImageMask>) {
                      //CLIP TODO: -> Option<MaskCacheInfo> ?
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
            transformed_rect: transformed_rect,
            mask_image: mask_image,
            parent_id: self.current_layer_id,
        });

        self.current_layer_id = Some(sc_index);
    }

    pub fn pop_layer(&mut self) {
        let lid = self.current_layer_id.unwrap();
        self.current_layer_id = self.layers[&lid].parent_id;
    }
}
