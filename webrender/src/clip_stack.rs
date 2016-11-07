/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::{Rect, Matrix4D};
use gpu_store::GpuStoreAddress;
use prim_store::{ClipData, PrimitiveClipSource};
use util::{TransformedRect, TransformedRectKind};
use webrender_traits::{AuxiliaryLists, BorderRadius, ComplexClipRegion, ImageMask};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ClipItemRange {
    pub start: u32,
    pub length: u32,
}

type ImageMaskIndex = u16;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct MaskCacheKey {
    pub layer_id: u32,
    pub item_range: ClipItemRange,
    pub mask_id: Option<ImageMaskIndex>,
}

#[derive(Debug)]
pub struct MaskCacheInfo {
    pub key: MaskCacheKey,
}

struct LayerInfo {
    world_transform: Matrix4D<f32>,
    transformed_rect: TransformedRect,
    mask_image: Option<ImageMask>,
    _index: u32,
}

pub struct ClipRegionStack {
    layers: Vec<LayerInfo>,
    complex_regions: Vec<ComplexClipRegion>,
    image_masks: Vec<ImageMask>,
    current_layer_id: u32,
    device_pixel_ratio: f32,
}

impl ClipRegionStack {
    pub fn new(device_pixel_ratio: f32) -> ClipRegionStack {
        ClipRegionStack {
            layers: Vec::new(),
            complex_regions: Vec::new(),
            image_masks: Vec::new(),
            current_layer_id: 0,
            device_pixel_ratio: device_pixel_ratio,
        }
    }

    pub fn reset(&mut self) {
        self.layers.clear();
        self.complex_regions.clear();
        self.image_masks.clear();
        self.current_layer_id = 0;
    }

    fn need_mask(&self) -> bool {
        self.layers.iter().find(|layer|
            layer.transformed_rect.kind == TransformedRectKind::Complex ||
            layer.mask_image.is_some() // TODO: parent layers masks?
            ).is_some()
    }

    pub fn generate<F>(&mut self,
                    source: &PrimitiveClipSource,
                    aux_lists: &AuxiliaryLists,
                    fun_data: F)
                    -> Option<MaskCacheInfo>
    where F: FnMut(ClipData) -> GpuStoreAddress {
        let mut clip_key = MaskCacheKey {
            layer_id: self.current_layer_id,
            item_range: ClipItemRange {
                start: self.complex_regions.len() as u32,
                length: 0,
            },
            mask_id: None,
        };
        match source {
            &PrimitiveClipSource::NoClip => (),
            &PrimitiveClipSource::Complex(rect, radius) => {
                clip_key.item_range.length = 1;
                let radius = BorderRadius::uniform(radius);
                self.complex_regions.push(ComplexClipRegion::new(rect, radius));
            },
            &PrimitiveClipSource::Region(ref region) => {
                let clips = aux_lists.complex_clip_regions(&region.complex);
                clip_key.item_range.length += clips.len() as u32;
                self.complex_regions.extend_from_slice(clips);
                if let Some(ref mask) = region.image_mask {
                    clip_key.mask_id = Some(self.image_masks.len() as ImageMaskIndex);
                    self.image_masks.push(mask.clone());
                }
            },
        };
        if self.need_mask() || clip_key.mask_id.is_some() || clip_key.item_range.length != 0 {
            Some(MaskCacheInfo {
                key: clip_key,
            })
        } else {
            None
        }
    }

    pub fn push_layer(&mut self,
                      local_transform: &Matrix4D<f32>,
                      rect: &Rect<f32>,
                      mask_image: Option<ImageMask>) {
                      //TODO: -> Option<MaskCacheInfo> ?
        let (world_transform, transformed_rect) = {
            let indentity_transform = Matrix4D::identity();
            let current_transform = match self.layers.last() {
                Some(ref layer) => &layer.world_transform,
                None => &indentity_transform,
            };
            let tr = TransformedRect::new(rect,
                                          current_transform,
                                          self.device_pixel_ratio);
            let wt = current_transform.pre_mul(local_transform);
            (wt, tr)
        };

        self.current_layer_id += 1;

        self.layers.push(LayerInfo {
            world_transform: world_transform,
            transformed_rect: transformed_rect,
            mask_image: mask_image,
            _index: self.current_layer_id,
        });
    }

    pub fn pop_layer(&mut self) -> Option<()> {
        self.layers.pop()
            .map(|_| ())
    }
}
