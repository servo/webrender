/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use device::TextureId;
use euclid::{Rect, Matrix4D};
use internal_types::DeviceSize;
use prim_store::PrimitiveClipSource;
use util::{TransformedRect, TransformedRectKind};
use webrender_traits::{ClipRegion, ComplexClipRegion, ImageMask, ItemRange};


#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct MaskCacheKey {
    layer_id: u32,
    item_range: ItemRange,
    mask_image: Option<ImageMask>,
}

#[derive(Debug)]
pub struct MaskCacheInfo {
    pub size: DeviceSize,
    pub key: MaskCacheKey,
}

struct LayerInfo {
    world_transform: Matrix4D<f32>,
    transformed_rect: TransformedRect,
    mask_image: Option<ImageMask>,
    _index: u32,
}

pub struct ClipRegionStack {
    last_token_index: u32,
    layers: Vec<LayerInfo>,
    next_layer_id: u32,
    device_pixel_ratio: f32,
}

impl ClipRegionStack {
    pub fn new(device_pixel_ratio: f32) -> ClipRegionStack {
        ClipRegionStack {
            last_token_index: 0,
            layers: Vec::new(),
            next_layer_id: 0,
            device_pixel_ratio: device_pixel_ratio,
        }
    }

    fn need_mask(&self) -> bool {
        self.layers.iter().find(|layer|
            layer.transformed_rect.kind == TransformedRectKind::Complex ||
            layer.mask_image.is_some()
            ).is_some()
    }

    pub fn generate_source(&mut self, clip_region: &ClipRegion) -> PrimitiveClipSource {
        if clip_region.is_complex() {
            PrimitiveClipSource::Region(clip_region.clone())
        } else {
            PrimitiveClipSource::NoClip
        };
        /*
        if self.need_mask() || !complex.is_empty() {
            self.last_token_index += 1;
            let t_rect = TransformedRect::new(rect,
                                              &self.layers.last().unwrap().world_transform,
                                              self.device_pixel_ratio);
            ClipMask::Cached(MaskCacheInfo {
                size: t_rect.bounding_rect.size,
                key: MaskCacheKey {
                    index: self.last_token_index,
                },
            })
        } else if let &Some(image) = mask_image {
            ClipMask::Image(image)
        } else {
            ClipMask::Dummy(TextureId::invalid()) //self.dummy_textute_id
        }*/
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

        self.layers.push(LayerInfo {
            world_transform: world_transform,
            transformed_rect: transformed_rect,
            mask_image: mask_image,
            _index: self.next_layer_id,
        });
        self.next_layer_id += 1;
    }

    pub fn pop_layer(&mut self) -> Option<()> {
        self.layers.pop()
            .map(|_| ())
    }
}
