/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use gpu_store::{GpuStore, GpuStoreAddress};
use prim_store::{ClipData, GpuBlock32};
use util::{rect_from_points_f, TransformedRect};
use webrender_traits::{AuxiliaryLists, ClipRegion, ImageMask};
use webrender_traits::{DeviceIntRect, LayerRect, LayerToWorldTransform};

const MAX_COORD: f32 = 1.0e+16;

pub const CORNERS_PER_CLIP_REGION: usize = 4;

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

#[derive(Clone, Debug)]
pub struct ImageMaskComponent {
    pub gpu_address: GpuStoreAddress,
    pub image_mask: ImageMask,
}

#[derive(Clone, Debug)]
pub struct CornerMaskComponent {
    pub gpu_address: GpuStoreAddress,

    // TODO(gw): We will store a local rect and world
    //           bounding rect for each clip component.
    //           This is how we can work out the region
    //           of interest for each primitive with
    //           a complex clip region, and then we
    //           will only allocate those regions of
    //           clip interest in the mask rendering.
}

#[derive(Clone, Debug)]
pub struct MaskCacheInfo {
    pub local_rect: Option<LayerRect>,
    pub bounding_rect: DeviceIntRect,
    pub corner_components: Vec<CornerMaskComponent>,
    pub image_components: Vec<ImageMaskComponent>,
}

impl MaskCacheInfo {
    /// Create a new mask cache info. It allocates the GPU store data but leaves
    /// it unitialized for the following `update()` call to deal with.
    pub fn new(source: &ClipSource,
               clip_store: &mut GpuStore<GpuBlock32>)
               -> Option<MaskCacheInfo> {
        let mut mask = MaskCacheInfo {
            corner_components: Vec::new(),
            image_components: Vec::new(),
            local_rect: None,
            bounding_rect: DeviceIntRect::zero(),
        };

        match source {
            &ClipSource::NoClip => {
                return None;
            }
            &ClipSource::Complex(..) => {
                for _ in 0..CORNERS_PER_CLIP_REGION {
                    mask.corner_components.push(CornerMaskComponent {
                        gpu_address: clip_store.alloc(1),
                    });
                }
            }
            &ClipSource::Region(ref region) => {
                if let Some(image_mask) = region.image_mask {
                    mask.image_components.push(ImageMaskComponent {
                        gpu_address: clip_store.alloc(1),
                        image_mask: image_mask,
                    });
                }

                for _ in 0..region.complex.length * CORNERS_PER_CLIP_REGION {
                    mask.corner_components.push(CornerMaskComponent {
                        gpu_address: clip_store.alloc(1),
                    });
                }
            }
        };

        Some(mask)
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
                    let data = ClipData::uniform(rect, radius);
                    for (corner, component) in data.corners.iter().zip(self.corner_components.iter()) {
                        let gpu_block = clip_store.get_mut(component.gpu_address);
                        *gpu_block = GpuBlock32::from(*corner);
                    }
                    local_rect = Some(rect);
                }
                &ClipSource::Region(ref region) => {
                    local_rect = Some(LayerRect::from_untyped(&rect_from_points_f(-MAX_COORD, -MAX_COORD, MAX_COORD, MAX_COORD)));
                    let clips = aux_lists.complex_clip_regions(&region.complex);
                    debug_assert_eq!(self.corner_components.len(), clips.len() * CORNERS_PER_CLIP_REGION);
                    for (clip, chunk) in clips.iter().zip(self.corner_components.chunks_mut(CORNERS_PER_CLIP_REGION)) {
                        let data = ClipData::from_clip_region(clip);
                        for (corner, component) in data.corners.iter().zip(chunk) {
                            let gpu_block = clip_store.get_mut(component.gpu_address);
                            *gpu_block = GpuBlock32::from(*corner);
                        }
                        local_rect = local_rect.and_then(|r| r.intersection(&clip.rect));
                    }
                }
            };
            self.local_rect = Some(local_rect.unwrap_or(LayerRect::zero()));
        }

        let transformed = TransformedRect::new(self.local_rect.as_ref().unwrap(),
                                               &transform,
                                               device_pixel_ratio);
        self.bounding_rect = transformed.bounding_rect;
    }
}
