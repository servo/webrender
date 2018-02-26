/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{BorderRadius, ClipMode, ComplexClipRegion, DeviceIntRect, DevicePixelScale, ImageMask};
use api::{ImageRendering, LayerRect, LayoutPoint, LayoutVector2D, LocalClip};
use border::{BorderCornerClipSource, ensure_no_corner_overlap};
use clip_scroll_tree::{ClipChainIndex, CoordinateSystemId};
use ellipse::Ellipse;
use freelist::{FreeList, FreeListHandle, WeakFreeListHandle};
use gpu_cache::{GpuCache, GpuCacheHandle, ToGpuBlocks};
use gpu_types::ClipScrollNodeIndex;
use prim_store::{ClipData, ImageMaskData};
use resource_cache::{ImageRequest, ResourceCache};
use util::{LayerToWorldFastTransform, MaxRect, calculate_screen_bounding_rect};
use util::extract_inner_rect_safe;
use std::sync::Arc;

pub type ClipStore = FreeList<ClipSources>;
pub type ClipSourcesHandle = FreeListHandle<ClipSources>;
pub type ClipSourcesWeakHandle = WeakFreeListHandle<ClipSources>;

#[derive(Clone, Debug)]
pub struct ClipRegion {
    pub main: LayerRect,
    pub image_mask: Option<ImageMask>,
    pub complex_clips: Vec<ComplexClipRegion>,
}

impl ClipRegion {
    pub fn create_for_clip_node(
        rect: LayerRect,
        mut complex_clips: Vec<ComplexClipRegion>,
        mut image_mask: Option<ImageMask>,
        reference_frame_relative_offset: &LayoutVector2D,
    ) -> ClipRegion {
        let rect = rect.translate(reference_frame_relative_offset);

        if let Some(ref mut image_mask) = image_mask {
            image_mask.rect = image_mask.rect.translate(reference_frame_relative_offset);
        }

        for complex_clip in complex_clips.iter_mut() {
            complex_clip.rect = complex_clip.rect.translate(reference_frame_relative_offset);
        }

        ClipRegion {
            main: rect,
            image_mask,
            complex_clips,
        }
    }

    pub fn create_for_clip_node_with_local_clip(
        local_clip: &LocalClip,
        reference_frame_relative_offset: &LayoutVector2D
    ) -> ClipRegion {
        let complex_clips = match local_clip {
            &LocalClip::Rect(_) => Vec::new(),
            &LocalClip::RoundedRect(_, ref region) => vec![region.clone()],
        };
        ClipRegion::create_for_clip_node(
            *local_clip.clip_rect(),
            complex_clips,
            None,
            reference_frame_relative_offset
        )
    }
}

#[derive(Debug)]
pub enum ClipSource {
    Rectangle(LayerRect),
    RoundedRectangle(LayerRect, BorderRadius, ClipMode),
    Image(ImageMask),
    /// TODO(gw): This currently only handles dashed style
    /// clips, where the border style is dashed for both
    /// adjacent border edges. Expand to handle dotted style
    /// and different styles per edge.
    BorderCorner(BorderCornerClipSource),
}

impl From<ClipRegion> for ClipSources {
    fn from(region: ClipRegion) -> ClipSources {
        let mut clips = Vec::new();

        if let Some(info) = region.image_mask {
            clips.push(ClipSource::Image(info));
        }

        clips.push(ClipSource::Rectangle(region.main));

        for complex in region.complex_clips {
            clips.push(ClipSource::new_rounded_rect(
                complex.rect,
                complex.radii,
                complex.mode,
            ));
        }

        ClipSources::new(clips)
    }
}

impl ClipSource {
    pub fn new_rounded_rect(
        rect: LayerRect,
        mut radii: BorderRadius,
        clip_mode: ClipMode
    ) -> ClipSource {
        ensure_no_corner_overlap(&mut radii, &rect);
        ClipSource::RoundedRectangle(
            rect,
            radii,
            clip_mode,
        )
    }
}

#[derive(Debug)]
pub struct ClipSources {
    pub clips: Vec<(ClipSource, GpuCacheHandle)>,
    pub local_inner_rect: LayerRect,
    pub local_outer_rect: Option<LayerRect>
}

impl ClipSources {
    pub fn new(clips: Vec<ClipSource>) -> ClipSources {
        let (local_inner_rect, local_outer_rect) = Self::calculate_inner_and_outer_rects(&clips);

        let clips = clips
            .into_iter()
            .map(|clip| (clip, GpuCacheHandle::new()))
            .collect();

        ClipSources {
            clips,
            local_inner_rect,
            local_outer_rect,
        }
    }

    pub fn clips(&self) -> &[(ClipSource, GpuCacheHandle)] {
        &self.clips
    }

    fn calculate_inner_and_outer_rects(clips: &Vec<ClipSource>) -> (LayerRect, Option<LayerRect>) {
        if clips.is_empty() {
            return (LayerRect::zero(), None);
        }

        // Depending on the complexity of the clip, we may either know the outer and/or inner
        // rect, or neither or these.  In the case of a clip-out, we currently set the mask bounds
        // to be unknown. This is conservative, but ensures correctness. In the future we can make
        // this a lot more clever with some proper region handling.
        let mut local_outer = Some(LayerRect::max_rect());
        let mut local_inner = local_outer;
        let mut can_calculate_inner_rect = true;
        let mut can_calculate_outer_rect = false;
        for source in clips {
            match *source {
                ClipSource::Image(ref mask) => {
                    if !mask.repeat {
                        can_calculate_outer_rect = true;
                        local_outer = local_outer.and_then(|r| r.intersection(&mask.rect));
                    }
                    local_inner = None;
                }
                ClipSource::Rectangle(rect) => {
                    can_calculate_outer_rect = true;
                    local_outer = local_outer.and_then(|r| r.intersection(&rect));
                    local_inner = local_inner.and_then(|r| r.intersection(&rect));
                }
                ClipSource::RoundedRectangle(ref rect, ref radius, mode) => {
                    // Once we encounter a clip-out, we just assume the worst
                    // case clip mask size, for now.
                    if mode == ClipMode::ClipOut {
                        can_calculate_inner_rect = false;
                        break;
                    }

                    can_calculate_outer_rect = true;
                    local_outer = local_outer.and_then(|r| r.intersection(rect));

                    let inner_rect = extract_inner_rect_safe(rect, radius);
                    local_inner = local_inner
                        .and_then(|r| inner_rect.and_then(|ref inner| r.intersection(inner)));
                }
                ClipSource::BorderCorner { .. } => {
                    can_calculate_inner_rect = false;
                    break;
                }
            }
        }

        let outer = match can_calculate_outer_rect {
            true => Some(local_outer.unwrap_or_else(LayerRect::zero)),
            false => None,
        };

        let inner = match can_calculate_inner_rect {
            true => local_inner.unwrap_or_else(LayerRect::zero),
            false => LayerRect::zero(),
        };

        (inner, outer)
    }

    pub fn update(
        &mut self,
        gpu_cache: &mut GpuCache,
        resource_cache: &mut ResourceCache,
    ) {
        for &mut (ref mut source, ref mut handle) in &mut self.clips {
            if let Some(mut request) = gpu_cache.request(handle) {
                match *source {
                    ClipSource::Image(ref mask) => {
                        let data = ImageMaskData { local_rect: mask.rect };
                        data.write_gpu_blocks(request);
                    }
                    ClipSource::Rectangle(rect) => {
                        let data = ClipData::uniform(rect, 0.0, ClipMode::Clip);
                        data.write(&mut request);
                    }
                    ClipSource::RoundedRectangle(ref rect, ref radius, mode) => {
                        let data = ClipData::rounded_rect(rect, radius, mode);
                        data.write(&mut request);
                    }
                    ClipSource::BorderCorner(ref mut source) => {
                        source.write(request);
                    }
                }
            }

            if let ClipSource::Image(ref mask) = *source {
                resource_cache.request_image(
                    ImageRequest {
                        key: mask.image,
                        rendering: ImageRendering::Auto,
                        tile: None,
                    },
                    gpu_cache,
                );
            }
        }
    }

    /// Whether or not this ClipSources has any clips (does any clipping).
    pub fn has_clips(&self) -> bool {
        !self.clips.is_empty()
    }

    pub fn get_screen_bounds(
        &self,
        transform: &LayerToWorldFastTransform,
        device_pixel_scale: DevicePixelScale,
    ) -> (DeviceIntRect, Option<DeviceIntRect>) {
        // If this translation isn't axis aligned or has a perspective component, don't try to
        // calculate the inner rectangle. The rectangle that we produce would include potentially
        // clipped screen area.
        // TODO(mrobinson): We should eventually try to calculate an inner region or some inner
        // rectangle so that we can do screen inner rectangle optimizations for these kind of
        // cilps.
        let can_calculate_inner_rect =
            transform.preserves_2d_axis_alignment() && !transform.has_perspective_component();
        let screen_inner_rect = if can_calculate_inner_rect {
            calculate_screen_bounding_rect(transform, &self.local_inner_rect, device_pixel_scale)
        } else {
            DeviceIntRect::zero()
        };

        let screen_outer_rect = self.local_outer_rect.map(|outer_rect|
            calculate_screen_bounding_rect(transform, &outer_rect, device_pixel_scale)
        );

        (screen_inner_rect, screen_outer_rect)
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
            local_rect,
            device_rect: DeviceIntRect::zero(),
        }
    }
}

pub trait Contains {
    fn contains(&self, point: &LayoutPoint) -> bool;
}

impl Contains for LocalClip {
    fn contains(&self, point: &LayoutPoint) -> bool {
        if !self.clip_rect().contains(point) {
            return false;
        }
        match self {
            &LocalClip::Rect(..) => true,
            &LocalClip::RoundedRect(_, complex_clip) => complex_clip.contains(point),
        }
    }
}

impl Contains for ComplexClipRegion {
    fn contains(&self, point: &LayoutPoint) -> bool {
        rounded_rectangle_contains_point(point, &self.rect, &self.radii)
    }
}

pub fn rounded_rectangle_contains_point(point: &LayoutPoint,
                                        rect: &LayerRect,
                                        radii: &BorderRadius)
                                        -> bool {
    if !rect.contains(point) {
        return false;
    }

    let top_left_center = rect.origin + radii.top_left.to_vector();
    if top_left_center.x > point.x && top_left_center.y > point.y &&
       !Ellipse::new(radii.top_left).contains(*point - top_left_center.to_vector()) {
        return false;
    }

    let bottom_right_center = rect.bottom_right() - radii.bottom_right.to_vector();
    if bottom_right_center.x < point.x && bottom_right_center.y < point.y &&
       !Ellipse::new(radii.bottom_right).contains(*point - bottom_right_center.to_vector()) {
        return false;
    }

    let top_right_center = rect.top_right() +
                           LayoutVector2D::new(-radii.top_right.width, radii.top_right.height);
    if top_right_center.x < point.x && top_right_center.y > point.y &&
       !Ellipse::new(radii.top_right).contains(*point - top_right_center.to_vector()) {
        return false;
    }

    let bottom_left_center = rect.bottom_left() +
                             LayoutVector2D::new(radii.bottom_left.width, -radii.bottom_left.height);
    if bottom_left_center.x > point.x && bottom_left_center.y < point.y &&
       !Ellipse::new(radii.bottom_left).contains(*point - bottom_left_center.to_vector()) {
        return false;
    }

    true
}

pub type ClipChainNodeRef = Option<Arc<ClipChainNode>>;

#[derive(Debug, Clone)]
pub struct ClipChainNode {
    pub work_item: ClipWorkItem,
    pub local_clip_rect: LayerRect,
    pub screen_outer_rect: DeviceIntRect,
    pub screen_inner_rect: DeviceIntRect,
    pub prev: ClipChainNodeRef,
}

#[derive(Debug, Clone)]
pub struct ClipChain {
    pub parent_index: Option<ClipChainIndex>,
    pub combined_outer_screen_rect: DeviceIntRect,
    pub combined_inner_screen_rect: DeviceIntRect,
    pub nodes: ClipChainNodeRef,
}

impl ClipChain {
    pub fn empty(screen_rect: &DeviceIntRect) -> ClipChain {
        ClipChain {
            parent_index: None,
            combined_inner_screen_rect: *screen_rect,
            combined_outer_screen_rect: *screen_rect,
            nodes: None,
        }
    }

    pub fn new_with_added_node(&self, new_node: &ClipChainNode) -> ClipChain {
        // If the new node's inner rectangle completely surrounds our outer rectangle,
        // we can discard the new node entirely since it isn't going to affect anything.
        if new_node.screen_inner_rect.contains_rect(&self.combined_outer_screen_rect) {
            return self.clone();
        }

        let mut new_chain = self.clone();
        new_chain.add_node(new_node.clone());
        new_chain
    }

    pub fn add_node(&mut self, mut new_node: ClipChainNode) {
        new_node.prev = self.nodes.clone();

        // If this clip's outer rectangle is completely enclosed by the clip
        // chain's inner rectangle, then the only clip that matters from this point
        // on is this clip. We can disconnect this clip from the parent clip chain.
        if self.combined_inner_screen_rect.contains_rect(&new_node.screen_outer_rect) {
            new_node.prev = None;
        }

        self.combined_outer_screen_rect =
            self.combined_outer_screen_rect.intersection(&new_node.screen_outer_rect)
            .unwrap_or_else(DeviceIntRect::zero);
        self.combined_inner_screen_rect =
            self.combined_inner_screen_rect.intersection(&new_node.screen_inner_rect)
            .unwrap_or_else(DeviceIntRect::zero);

        self.nodes = Some(Arc::new(new_node));
    }
}

pub struct ClipChainNodeIter {
    pub current: ClipChainNodeRef,
}

impl Iterator for ClipChainNodeIter {
    type Item = Arc<ClipChainNode>;

    fn next(&mut self) -> ClipChainNodeRef {
        let previous = self.current.clone();
        self.current = match self.current {
            Some(ref item) => item.prev.clone(),
            None => return None,
        };
        previous
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct ClipWorkItem {
    pub scroll_node_data_index: ClipScrollNodeIndex,
    pub clip_sources: ClipSourcesWeakHandle,
    pub coordinate_system_id: CoordinateSystemId,
}

