/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use border::BorderCornerClipSource;
use gpu_store::GpuStoreAddress;
use prim_store::{ClipData, GpuBlock32, ImageMaskData, PrimitiveStore};
use prim_store::{CLIP_DATA_GPU_SIZE, MASK_DATA_GPU_SIZE};
use renderer::VertexDataStore;
use util::{ComplexClipRegionHelpers, TransformedRect};
use webrender_traits::{BorderRadius, BuiltDisplayList, ClipRegion, ComplexClipRegion, ImageMask};
use webrender_traits::{DeviceIntRect, LayerToWorldTransform};
use webrender_traits::{DeviceRect, LayerRect, LayerPoint, LayerSize};
use std::ops::Not;

const MAX_CLIP: f32 = 1000000.0;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ClipMode {
    Clip,           // Pixels inside the region are visible.
    ClipOut,        // Pixels outside the region are visible.
}

impl Not for ClipMode {
    type Output = ClipMode;

    fn not(self) -> ClipMode {
        match self {
            ClipMode::Clip => ClipMode::ClipOut,
            ClipMode::ClipOut => ClipMode::Clip
        }
    }
}

#[derive(Clone, Debug)]
pub enum ClipSource {
    Complex(LayerRect, f32, ClipMode),
    Region(ClipRegion),
    /// TODO(gw): This currently only handles dashed style
    /// clips, where the border style is dashed for both
    /// adjacent border edges. Expand to handle dotted style
    /// and different styles per edge.
    BorderCorner(BorderCornerClipSource),
}

impl ClipSource {
    pub fn image_mask(&self) -> Option<ImageMask> {
        match *self {
            ClipSource::Complex(..) |
            ClipSource::BorderCorner(..) => None,
            ClipSource::Region(ref region) => region.image_mask,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ClipAddressRange {
    pub start: GpuStoreAddress,
    item_count: usize,
}

impl ClipAddressRange {
    pub fn get_count(&self) -> usize {
        self.item_count
    }
}

/// Represents a local rect and a device space
/// rectangles that are either outside or inside bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    pub local_rect: LayerRect,
    pub device_rect: DeviceIntRect,
}

impl From<LayerRect> for Geometry {
    fn from(local_rect: LayerRect) -> Self {
        Geometry {
            local_rect: local_rect,
            device_rect: DeviceIntRect::zero(),
        }
    }
}

/// Depending on the complexity of the clip, we may either
/// know the outer and/or inner rect, or neither or these.
/// In the case of a clip-out, we currently set the mask
/// bounds to be unknown. This is conservative, but ensures
/// correctness. In the future we can make this a lot
/// more clever with some proper region handling.
#[derive(Clone, Debug, PartialEq)]
pub struct MaskBounds {
    pub outer: Option<Geometry>,
    pub inner: Option<Geometry>,
}

impl MaskBounds {
    pub fn update(&mut self, transform: &LayerToWorldTransform, device_pixel_ratio: f32) {
        if let Some(ref mut outer) = self.outer {
            let transformed = TransformedRect::new(&outer.local_rect,
                                                   transform,
                                                   device_pixel_ratio);
            outer.device_rect = transformed.bounding_rect;
        }
        if let Some(ref mut inner) = self.inner {
            let transformed = TransformedRect::new(&inner.local_rect,
                                                   transform,
                                                   device_pixel_ratio);
            inner.device_rect = transformed.inner_rect;
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
    pub image: Option<(ImageMask, GpuStoreAddress)>,
    pub border_corners: Vec<(BorderCornerClipSource, GpuStoreAddress)>,
    pub bounds: MaskBounds,
}

impl MaskCacheInfo {
    /// Create a new mask cache info. It allocates the GPU store data but leaves
    /// it uninitialized for the following `update()` call to deal with.
    pub fn new(clips: &[ClipSource],
               clip_store: &mut VertexDataStore<GpuBlock32>)
               -> MaskCacheInfo {
        let mut image = None;
        let mut border_corners = Vec::new();
        let mut complex_clip_count = 0;
        let mut layer_clip_count = 0;

        // Work out how much clip data space we need to allocate
        // and if we have an image mask.
        for clip in clips {
            match *clip {
                ClipSource::Complex(..) => {
                    complex_clip_count += 1;
                }
                ClipSource::Region(ref region) => {
                    if let Some(info) = region.image_mask {
                        debug_assert!(image.is_none());     // TODO(gw): Support >1 image mask!
                        image = Some((info, clip_store.alloc(MASK_DATA_GPU_SIZE)));
                    }
                    complex_clip_count += region.complex_clip_count;
                    layer_clip_count += 1;
                }
                ClipSource::BorderCorner(ref source) => {
                    // One block for the corner header, plus one
                    // block per dash to clip out.
                    let gpu_address = clip_store.alloc(1 + source.max_clip_count);
                    border_corners.push((source.clone(), gpu_address));
                }
            }
        }

        MaskCacheInfo {
            complex_clip_range: ClipAddressRange {
                start: if complex_clip_count > 0 {
                    clip_store.alloc(CLIP_DATA_GPU_SIZE * complex_clip_count)
                } else {
                    GpuStoreAddress(0)
                },
                item_count: complex_clip_count,
            },
            layer_clip_range: ClipAddressRange {
                start: if layer_clip_count > 0 {
                    clip_store.alloc(CLIP_DATA_GPU_SIZE * layer_clip_count)
                } else {
                    GpuStoreAddress(0)
                },
                item_count: layer_clip_count,
            },
            image: image,
            border_corners: border_corners,
            bounds: MaskBounds {
                inner: None,
                outer: None,
            },
        }
    }

    pub fn update(&mut self,
                  sources: &[ClipSource],
                  transform: &LayerToWorldTransform,
                  clip_store: &mut VertexDataStore<GpuBlock32>,
                  device_pixel_ratio: f32,
                  display_list: &BuiltDisplayList) -> &MaskBounds {
        //TODO: move to initialization stage?
        if self.bounds.inner.is_none() {
            let mut local_rect = Some(LayerRect::new(LayerPoint::new(-MAX_CLIP, -MAX_CLIP),
                                                     LayerSize::new(2.0 * MAX_CLIP, 2.0 * MAX_CLIP)));
            let mut local_inner: Option<LayerRect> = None;
            let mut has_clip_out = false;
            let mut has_border_clip = false;

            let mut complex_clip_count = 0;
            let mut layer_clip_count = 0;

            for source in sources {
                match *source {
                    ClipSource::Complex(rect, radius, mode) => {
                        // Once we encounter a clip-out, we just assume the worst
                        // case clip mask size, for now.
                        if mode == ClipMode::ClipOut {
                            has_clip_out = true;
                        }
                        let address = self.complex_clip_range.start + complex_clip_count * CLIP_DATA_GPU_SIZE;
                        complex_clip_count += 1;

                        let slice = clip_store.get_slice_mut(address, CLIP_DATA_GPU_SIZE);
                        let data = ClipData::uniform(rect, radius, mode);
                        PrimitiveStore::populate_clip_data(slice, data);
                        local_rect = local_rect.and_then(|r| r.intersection(&rect));
                        local_inner = ComplexClipRegion::new(rect, BorderRadius::uniform(radius))
                                                        .get_inner_rect_safe();
                    }
                    ClipSource::Region(ref region) => {
                        local_rect = local_rect.and_then(|r| r.intersection(&region.main));
                        local_inner = match region.image_mask {
                            Some(ref mask) => {
                                if !mask.repeat {
                                    local_rect = local_rect.and_then(|r| r.intersection(&mask.rect));
                                }
                                None
                            },
                            None => local_rect,
                        };

                        {// Add an extra clip for the main rectangle,
                            let address = self.layer_clip_range.start + layer_clip_count * CLIP_DATA_GPU_SIZE;
                            layer_clip_count += 1;
                            let slice = clip_store.get_slice_mut(address, CLIP_DATA_GPU_SIZE);
                            PrimitiveStore::populate_clip_data(slice, ClipData::uniform(region.main, 0.0, ClipMode::Clip));
                        }

                        let clips = display_list.get(region.complex_clips);
                        let address = self.complex_clip_range.start + complex_clip_count * CLIP_DATA_GPU_SIZE;
                        complex_clip_count += clips.len();

                        let slice = clip_store.get_slice_mut(address, CLIP_DATA_GPU_SIZE * clips.len());
                        for (clip, chunk) in clips.zip(slice.chunks_mut(CLIP_DATA_GPU_SIZE)) {
                            let data = ClipData::from_clip_region(&clip);
                            PrimitiveStore::populate_clip_data(chunk, data);
                            local_rect = local_rect.and_then(|r| r.intersection(&clip.rect));
                            local_inner = local_inner.and_then(|r| clip.get_inner_rect_safe()
                                                                       .and_then(|ref inner| r.intersection(inner)));
                        }
                    }
                    ClipSource::BorderCorner{..} => {}
                }
            }

            debug_assert_eq!(complex_clip_count, self.complex_clip_range.item_count);
            debug_assert_eq!(layer_clip_count, self.layer_clip_range.item_count);

            for &mut (ref mut source, gpu_address) in &mut self.border_corners {
                has_border_clip = true;
                let slice = clip_store.get_slice_mut(gpu_address,
                                                     1 + source.max_clip_count);
                source.populate_gpu_data(slice);
            }

            if let Some((ref mask, gpu_address)) = self.image {
                let mask_data = clip_store.get_slice_mut(gpu_address, MASK_DATA_GPU_SIZE);
                mask_data[0] = GpuBlock32::from(ImageMaskData {
                    padding: DeviceRect::zero(),
                    local_rect: mask.rect,
                });
            }

            // Work out the type of mask geometry we have, based on the
            // list of clip sources above.
            self.bounds = if has_clip_out || has_border_clip {
                // For clip-out, the mask rect is not known.
                MaskBounds {
                    outer: None,
                    inner: Some(LayerRect::zero().into()),
                }
            } else {
                MaskBounds {
                    outer: Some(local_rect.unwrap_or(LayerRect::zero()).into()),
                    inner: Some(local_inner.unwrap_or(LayerRect::zero()).into()),
                }
            };
        }

        // Update the device space bounding rects of the mask geometry.
        self.bounds.update(transform, device_pixel_ratio);
        &self.bounds
    }

    /// Check if this `MaskCacheInfo` actually carries any masks.
    pub fn is_masking(&self) -> bool {
        self.image.is_some() ||
        self.complex_clip_range.item_count != 0 ||
        self.layer_clip_range.item_count != 0 ||
        !self.border_corners.is_empty()
    }

    /// Return a clone of this object without any layer-aligned clip items
    pub fn strip_aligned(&self) -> Self {
        MaskCacheInfo {
            layer_clip_range: ClipAddressRange {
                start: GpuStoreAddress(0),
                item_count: 0,
            },
            .. self.clone()
        }
    }
}
