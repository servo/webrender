/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{BorderRadius, ClipMode, ComplexClipRegion, DeviceIntRect, DevicePixelScale, ImageMask};
use api::{ImageRendering, LayoutRect, LayoutSize, LayoutPoint, LayoutVector2D, LocalClip};
use api::{BoxShadowClipMode, LayoutToWorldScale, LineOrientation, LineStyle};
use api::{LayoutToWorldTransform, WorldPixel, WorldRect, WorldPoint, WorldSize};
use border::{ensure_no_corner_overlap};
use box_shadow::{BLUR_SAMPLE_SCALE, BoxShadowClipSource, BoxShadowCacheKey};
use clip_scroll_tree::{ClipScrollTree, CoordinateSystemId, SpatialNodeIndex};
use ellipse::Ellipse;
use gpu_cache::{GpuCache, GpuCacheHandle, ToGpuBlocks};
use gpu_types::BoxShadowStretchMode;
use plane_split::{Clipper, Polygon};
use prim_store::{ClipData, ImageMaskData};
use render_task::to_cache_size;
use resource_cache::{ImageRequest, ResourceCache};
use std::{cmp, u32};
use util::{extract_inner_rect_safe, pack_as_float, recycle_vec, MaxRect};

/*

 Module Overview

 There are a number of data structures involved in the clip module:

 ClipStore - Main interface used by other modules.

 ClipItem - A single clip item (e.g. a rounded rect, or a box shadow).
            These are an exposed API type, stored inline in a ClipNode.

 ClipNode - A ClipItem with attached positioning information (a spatial node index).
            Stored as a contiguous array of nodes within the ClipStore.

    +-----------------------+-----------------------+-----------------------+
    | ClipItem              | ClipItem              | ClipItem              |
    | Spatial Node Index    | Spatial Node Index    | Spatial Node Index    |
    | GPU cache handle      | GPU cache handle      | GPU cache handle      |
    +-----------------------+-----------------------+-----------------------+
               0                        1                       2

       +----------------+    |                                              |
       | ClipItemRange  |____|                                              |
       |    index: 1    |                                                   |
       |    count: 2    |___________________________________________________|
       +----------------+

 ClipItemRange - A clip item range identifies a range of clip nodes. It is stored
                 as an (index, count).

 ClipChain - A clip chain node contains a range of ClipNodes (a ClipItemRange)
             and a parent link to an optional ClipChain. Both legacy hierchical clip
             chains and user defined API clip chains use the same data structure.
             ClipChainId is an index into an array, or ClipChainId::NONE for no parent.

    +----------------+    ____+----------------+    ____+----------------+    ____+----------------+
    | ClipChain      |   |    | ClipChain      |   |    | ClipChain      |   |    | ClipChain      |
    +----------------+   |    +----------------+   |    +----------------+   |    +----------------+
    | ClipItemRange  |   |    | ClipItemRange  |   |    | ClipItemRange  |   |    | ClipItemRange  |
    | Parent Id      |___|    | Parent Id      |___|    | Parent Id      |___|    | Parent Id      |
    +----------------+        +----------------+        +----------------+        +----------------+

 ClipChainInstance - A ClipChain that has been built for a specific primitive + positioning node.

    When given a clip chain ID, and a local primitive rect + spatial node, the clip module
    creates a clip chain instance. This is a struct with various pieces of useful information
    (such as a local clip rect and affected local bounding rect). It also contains a (index, count)
    range specifier into an index buffer of the ClipNode structures that are actually relevant
    for this clip chain instance. The index buffer structure allows a single array to be used for
    all of the clip-chain instances built in a single frame. Each entry in the index buffer
    also stores some flags relevant to the clip node in this positioning context.

    +----------------------+
    | ClipChainInstance    |
    +----------------------+
    | local_clip_rect      |
    | local_bounding_rect  |________________________________________________________________________
    | clips_range          |_______________                                                        |
    +----------------------+              |                                                        |
                                          |                                                        |
    +------------------+------------------+------------------+------------------+------------------+
    | ClipNodeInstance | ClipNodeInstance | ClipNodeInstance | ClipNodeInstance | ClipNodeInstance |
    +------------------+------------------+------------------+------------------+------------------+
    | flags            | flags            | flags            | flags            | flags            |
    | ClipNodeIndex    | ClipNodeIndex    | ClipNodeIndex    | ClipNodeIndex    | ClipNodeIndex    |
    +------------------+------------------+------------------+------------------+------------------+

 */

// Result of comparing a clip node instance against a local rect.
#[derive(Debug)]
enum ClipResult {
    // The clip does not affect the region at all.
    Accept,
    // The clip prevents the region from being drawn.
    Reject,
    // The clip affects part of the region. This may
    // require a clip mask, depending on other factors.
    Partial,
}

// A clip item range represents one or more ClipItem structs.
// They are stored in a contiguous array inside the ClipStore,
// and identified by an (offset, count).
#[derive(Debug, Copy, Clone)]
pub struct ClipItemRange {
    pub index: ClipNodeIndex,
    pub count: u32,
}

// A clip node is a single clip source, along with some
// positioning information and implementation details
// that control where the GPU data for this clip source
// can be found.
pub struct ClipNode {
    pub spatial_node_index: SpatialNodeIndex,
    pub item: ClipItem,
    pub gpu_cache_handle: GpuCacheHandle,
}

// Flags that are attached to instances of clip nodes.
bitflags! {
    pub struct ClipNodeFlags: u8 {
        const SAME_SPATIAL_NODE = 0x1;
        const SAME_COORD_SYSTEM = 0x2;
    }
}

// Identifier for a clip chain. Clip chains are stored
// in a contiguous array in the clip store. They are
// identified by a simple index into that array.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClipChainId(pub u32);

// The root of each clip chain is the NONE id. The
// value is specifically set to u32::MAX so that if
// any code accidentally tries to access the root
// node, a bounds error will occur.
impl ClipChainId {
    pub const NONE: Self = ClipChainId(u32::MAX);
}

// A clip chain node is an id for a range of clip sources,
// and a link to a parent clip chain node, or ClipChainId::NONE.
#[derive(Clone)]
pub struct ClipChainNode {
    pub clip_item_range: ClipItemRange,
    pub parent_clip_chain_id: ClipChainId,
}

// An index into the clip_nodes array.
#[derive(Clone, Copy, Debug, PartialEq, Hash, Eq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct ClipNodeIndex(pub u32);

// When a clip node is found to be valid for a
// clip chain instance, it's stored in an index
// buffer style structure. This struct contains
// an index to the node data itself, as well as
// some flags describing how this clip node instance
// is positioned.
#[derive(Clone, Copy, Debug, PartialEq, Hash, Eq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct ClipNodeInstance(pub u32);

impl ClipNodeInstance {
    fn new(index: ClipNodeIndex, flags: ClipNodeFlags) -> ClipNodeInstance {
        ClipNodeInstance(
            (index.0 & 0x00ffffff) | ((flags.bits() as u32) << 24)
        )
    }

    fn flags(&self) -> ClipNodeFlags {
        ClipNodeFlags::from_bits_truncate((self.0 >> 24) as u8)
    }

    fn index(&self) -> usize {
        (self.0 & 0x00ffffff) as usize
    }
}

// A range of clip node instances that were found by
// building a clip chain instance.
#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct ClipNodeRange {
    pub first: u32,
    pub count: u32,
}

// A helper struct for converting between coordinate systems
// of clip sources and primitives.
// todo(gw): optimize:
//  separate arrays for matrices
//  cache and only build as needed.
#[derive(Debug)]
enum ClipSpaceConversion {
    Local,
    Offset(LayoutVector2D),
    Transform(LayoutToWorldTransform),
}

// Temporary information that is cached and reused
// during building of a clip chain instance.
struct ClipNodeInfo {
    conversion: ClipSpaceConversion,
    node_index: ClipNodeIndex,
    has_non_root_coord_system: bool,
}

impl ClipNode {
    pub fn update(
        &mut self,
        gpu_cache: &mut GpuCache,
        resource_cache: &mut ResourceCache,
        device_pixel_scale: DevicePixelScale,
    ) {
        if let Some(mut request) = gpu_cache.request(&mut self.gpu_cache_handle) {
            match self.item {
                ClipItem::Image(ref mask) => {
                    let data = ImageMaskData { local_rect: mask.rect };
                    data.write_gpu_blocks(request);
                }
                ClipItem::BoxShadow(ref info) => {
                    request.push([
                        info.shadow_rect_alloc_size.width,
                        info.shadow_rect_alloc_size.height,
                        info.clip_mode as i32 as f32,
                        0.0,
                    ]);
                    request.push([
                        info.stretch_mode_x as i32 as f32,
                        info.stretch_mode_y as i32 as f32,
                        0.0,
                        0.0,
                    ]);
                    request.push(info.prim_shadow_rect);
                }
                ClipItem::Rectangle(rect, mode) => {
                    let data = ClipData::uniform(rect, 0.0, mode);
                    data.write(&mut request);
                }
                ClipItem::RoundedRectangle(ref rect, ref radius, mode) => {
                    let data = ClipData::rounded_rect(rect, radius, mode);
                    data.write(&mut request);
                }
                ClipItem::LineDecoration(ref info) => {
                    request.push(info.rect);
                    request.push([
                        info.wavy_line_thickness,
                        pack_as_float(info.style as u32),
                        pack_as_float(info.orientation as u32),
                        0.0,
                    ]);
                }
            }
        }

        match self.item {
            ClipItem::Image(ref mask) => {
                resource_cache.request_image(
                    ImageRequest {
                        key: mask.image,
                        rendering: ImageRendering::Auto,
                        tile: None,
                    },
                    gpu_cache,
                );
            }
            ClipItem::BoxShadow(ref mut info) => {
                // Quote from https://drafts.csswg.org/css-backgrounds-3/#shadow-blur
                // "the image that would be generated by applying to the shadow a
                // Gaussian blur with a standard deviation equal to half the blur radius."
                let blur_radius_dp = (info.blur_radius * 0.5 * device_pixel_scale.0).round();

                // Create the cache key for this box-shadow render task.
                let content_scale = LayoutToWorldScale::new(1.0) * device_pixel_scale;
                let cache_size = to_cache_size(info.shadow_rect_alloc_size * content_scale);
                let bs_cache_key = BoxShadowCacheKey {
                    blur_radius_dp: blur_radius_dp as i32,
                    clip_mode: info.clip_mode,
                    rect_size: (info.shadow_rect_alloc_size * content_scale).round().to_i32(),
                    br_top_left: (info.shadow_radius.top_left * content_scale).round().to_i32(),
                    br_top_right: (info.shadow_radius.top_right * content_scale).round().to_i32(),
                    br_bottom_right: (info.shadow_radius.bottom_right * content_scale).round().to_i32(),
                    br_bottom_left: (info.shadow_radius.bottom_left * content_scale).round().to_i32(),
                };

                info.cache_key = Some((cache_size, bs_cache_key));

                if let Some(mut request) = gpu_cache.request(&mut info.clip_data_handle) {
                    let data = ClipData::rounded_rect(
                        &info.minimal_shadow_rect,
                        &info.shadow_radius,
                        ClipMode::Clip,
                    );

                    data.write(&mut request);
                }
            }
            ClipItem::Rectangle(..) |
            ClipItem::RoundedRectangle(..) |
            ClipItem::LineDecoration(..) => {}
        }
    }
}

// The main clipping public interface that other modules access.
pub struct ClipStore {
    pub clip_nodes: Vec<ClipNode>,
    pub clip_chain_nodes: Vec<ClipChainNode>,
    clip_node_indices: Vec<ClipNodeInstance>,
    clip_node_info: Vec<ClipNodeInfo>,
}

// A clip chain instance is what gets built for a given clip
// chain id + local primitive region + positioning node.
#[derive(Debug)]
pub struct ClipChainInstance {
    pub clips_range: ClipNodeRange,
    pub local_clip_rect: LayoutRect,
    pub has_non_root_coord_system: bool,
    pub world_clip_rect: WorldRect,
}

impl ClipStore {
    pub fn new() -> Self {
        ClipStore {
            clip_nodes: Vec::new(),
            clip_chain_nodes: Vec::new(),
            clip_node_indices: Vec::new(),
            clip_node_info: Vec::new(),
        }
    }

    pub fn recycle(self) -> Self {
        ClipStore {
            clip_nodes: recycle_vec(self.clip_nodes),
            clip_chain_nodes: recycle_vec(self.clip_chain_nodes),
            clip_node_indices: recycle_vec(self.clip_node_indices),
            clip_node_info: recycle_vec(self.clip_node_info),
        }
    }

    pub fn add_clip_items(
        &mut self,
        clip_items: Vec<ClipItem>,
        spatial_node_index: SpatialNodeIndex,
    ) -> ClipItemRange {
        debug_assert!(!clip_items.is_empty());

        let range = ClipItemRange {
            index: ClipNodeIndex(self.clip_nodes.len() as u32),
            count: clip_items.len() as u32,
        };

        let nodes = clip_items
            .into_iter()
            .map(|item| {
                ClipNode {
                    item,
                    spatial_node_index,
                    gpu_cache_handle: GpuCacheHandle::new(),
                }
            });

        self.clip_nodes.extend(nodes);
        range
    }

    pub fn get_clip_chain(&self, clip_chain_id: ClipChainId) -> &ClipChainNode {
        &self.clip_chain_nodes[clip_chain_id.0 as usize]
    }

    pub fn add_clip_chain(
        &mut self,
        clip_item_range: ClipItemRange,
        parent_clip_chain_id: ClipChainId,
    ) -> ClipChainId {
        let id = ClipChainId(self.clip_chain_nodes.len() as u32);
        self.clip_chain_nodes.push(ClipChainNode {
            clip_item_range,
            parent_clip_chain_id,
        });
        id
    }

    pub fn get_node_from_range(
        &self,
        node_range: &ClipNodeRange,
        index: u32,
    ) -> (&ClipNode, ClipNodeFlags) {
        let instance = self.clip_node_indices[(node_range.first + index) as usize];
        (&self.clip_nodes[instance.index()], instance.flags())
    }

    pub fn get_node_from_range_mut(
        &mut self,
        node_range: &ClipNodeRange,
        index: u32,
    ) -> (&mut ClipNode, ClipNodeFlags) {
        let instance = self.clip_node_indices[(node_range.first + index) as usize];
        (&mut self.clip_nodes[instance.index()], instance.flags())
    }

    // The main interface other code uses. Given a local primitive, positioning
    // information, and a clip chain id, build an optimized clip chain instance.
    pub fn build_clip_chain_instance(
        &mut self,
        clip_chain_id: ClipChainId,
        local_prim_rect: LayoutRect,
        local_prim_clip_rect: LayoutRect,
        spatial_node_index: SpatialNodeIndex,
        clip_scroll_tree: &ClipScrollTree,
        gpu_cache: &mut GpuCache,
        resource_cache: &mut ResourceCache,
        device_pixel_scale: DevicePixelScale,
    ) -> Option<ClipChainInstance> {
        let mut local_clip_rect = local_prim_clip_rect;
        let mut world_clip_rect = WorldRect::max_rect();
        let spatial_nodes = &clip_scroll_tree.spatial_nodes;

        // Walk the clip chain to build local rects, and collect the
        // smallest possible local/device clip area.

        self.clip_node_info.clear();
        let ref_spatial_node = &spatial_nodes[spatial_node_index.0];
        let mut current_clip_chain_id = clip_chain_id;

        // for each clip chain node
        while current_clip_chain_id != ClipChainId::NONE {
            let clip_chain_node = &self.clip_chain_nodes[current_clip_chain_id.0 as usize];
            let node_count = clip_chain_node.clip_item_range.count;

            // for each clip node (clip source) in this clip chain node
            for i in 0 .. node_count {
                let clip_node_index = ClipNodeIndex(clip_chain_node.clip_item_range.index.0 + i);
                let clip_node = &self.clip_nodes[clip_node_index.0 as usize];
                let clip_spatial_node = &spatial_nodes[clip_node.spatial_node_index.0 as usize];

                // Determine the most efficient way to convert between coordinate
                // systems of the primitive and clip node.
                let conversion = if spatial_node_index == clip_node.spatial_node_index {
                    Some(ClipSpaceConversion::Local)
                } else if ref_spatial_node.coordinate_system_id == clip_spatial_node.coordinate_system_id {
                    let offset = clip_spatial_node.coordinate_system_relative_offset -
                                 ref_spatial_node.coordinate_system_relative_offset;
                    Some(ClipSpaceConversion::Offset(offset))
                } else {
                    let xf = clip_scroll_tree.get_relative_transform(
                        clip_node.spatial_node_index,
                        SpatialNodeIndex(0),
                    );

                    xf.map(|xf| {
                        ClipSpaceConversion::Transform(xf.with_destination::<WorldPixel>())
                    })
                };

                // If we can convert spaces, try to reduce the size of the region
                // requested, and cache the conversion information for the next step.
                if let Some(conversion) = conversion {
                    if let Some(clip_rect) = clip_node.item.get_local_clip_rect() {
                        match conversion {
                            ClipSpaceConversion::Local => {
                                local_clip_rect = match local_clip_rect.intersection(&clip_rect) {
                                    Some(local_clip_rect) => local_clip_rect,
                                    None => return None,
                                };
                            }
                            ClipSpaceConversion::Offset(ref offset) => {
                                let clip_rect = clip_rect.translate(offset);
                                local_clip_rect = match local_clip_rect.intersection(&clip_rect) {
                                    Some(local_clip_rect) => local_clip_rect,
                                    None => return None,
                                };
                            }
                            ClipSpaceConversion::Transform(ref transform) => {
                                let world_clip_rect_for_item = match project_rect(
                                    transform,
                                    &clip_rect,
                                ) {
                                    Some(rect) => rect,
                                    None => return None,
                                };

                                world_clip_rect = match world_clip_rect.intersection(&world_clip_rect_for_item) {
                                    Some(world_clip_rect) => world_clip_rect,
                                    None => return None,
                                };
                            }
                        }
                    }
                    self.clip_node_info.push(ClipNodeInfo {
                        conversion,
                        node_index: clip_node_index,
                        has_non_root_coord_system: clip_spatial_node.coordinate_system_id != CoordinateSystemId::root(),
                    })
                }
            }

            current_clip_chain_id = clip_chain_node.parent_clip_chain_id;
        }

        let local_bounding_rect = match local_prim_rect.intersection(&local_clip_rect) {
            Some(rect) => rect,
            None => return None,
        };

        let world_bounding_rect = match project_rect(
            &ref_spatial_node.world_content_transform.to_transform(),
            &local_bounding_rect,
        ) {
            Some(world_bounding_rect) => world_bounding_rect,
            None => return None,
        };

        let world_clip_rect = match world_clip_rect.intersection(&world_bounding_rect) {
            Some(world_clip_rect) => world_clip_rect,
            None => return None,
        };

        // Now, we've collected all the clip nodes that *potentially* affect this
        // primitive region, and reduced the size of the prim region as much as possible.

        // Run through the clip nodes, and see which ones affect this prim region.

        let first_clip_node_index = self.clip_node_indices.len() as u32;
        let mut has_non_root_coord_system = false;

        // For each potential clip node
        for node_info in self.clip_node_info.drain(..) {
            let node = &mut self.clip_nodes[node_info.node_index.0 as usize];

            // See how this clip affects the prim region.
            let clip_result = match node_info.conversion {
                ClipSpaceConversion::Local => {
                    node.item.get_clip_result(&local_bounding_rect)
                }
                ClipSpaceConversion::Offset(offset) => {
                    node.item.get_clip_result(&local_bounding_rect.translate(&-offset))
                }
                ClipSpaceConversion::Transform(ref transform) => {
                    node.item.get_clip_result_complex(
                        transform,
                        &world_bounding_rect,
                    )
                }
            };

            match clip_result {
                ClipResult::Accept => {
                    // Doesn't affect the primitive at all, so skip adding to list
                }
                ClipResult::Reject => {
                    // Completely clips the supplied prim rect
                    return None;
                }
                ClipResult::Partial => {
                    // Needs a mask -> add to clip node indices

                    // TODO(gw): Ensure this only runs once on each node per frame?
                    node.update(
                        gpu_cache,
                        resource_cache,
                        device_pixel_scale,
                    );

                    // Calculate some flags that are required for the segment
                    // building logic.
                    let flags = match node_info.conversion {
                        ClipSpaceConversion::Local => {
                            ClipNodeFlags::SAME_SPATIAL_NODE | ClipNodeFlags::SAME_COORD_SYSTEM
                        }
                        ClipSpaceConversion::Offset(..) => {
                            ClipNodeFlags::SAME_COORD_SYSTEM
                        }
                        ClipSpaceConversion::Transform(..) => {
                            ClipNodeFlags::empty()
                        }
                    };

                    // Store this in the index buffer for this clip chain instance.
                    self.clip_node_indices
                        .push(ClipNodeInstance::new(node_info.node_index, flags));

                    has_non_root_coord_system |= node_info.has_non_root_coord_system;
                }
            }
        }

        // Get the range identifying the clip nodes in the index buffer.
        let clips_range = ClipNodeRange {
            first: first_clip_node_index,
            count: self.clip_node_indices.len() as u32 - first_clip_node_index,
        };

        // Return a valid clip chain instance
        Some(ClipChainInstance {
            clips_range,
            has_non_root_coord_system,
            local_clip_rect,
            world_clip_rect,
        })
    }
}

#[derive(Debug)]
pub struct LineDecorationClipSource {
    rect: LayoutRect,
    style: LineStyle,
    orientation: LineOrientation,
    wavy_line_thickness: f32,
}


pub struct ComplexTranslateIter<I> {
    source: I,
    offset: LayoutVector2D,
}

impl<I: Iterator<Item = ComplexClipRegion>> Iterator for ComplexTranslateIter<I> {
    type Item = ComplexClipRegion;
    fn next(&mut self) -> Option<Self::Item> {
        self.source
            .next()
            .map(|mut complex| {
                complex.rect = complex.rect.translate(&self.offset);
                complex
            })
    }
}

#[derive(Clone, Debug)]
pub struct ClipRegion<I> {
    pub main: LayoutRect,
    pub image_mask: Option<ImageMask>,
    pub complex_clips: I,
}

impl<J> ClipRegion<ComplexTranslateIter<J>> {
    pub fn create_for_clip_node(
        rect: LayoutRect,
        complex_clips: J,
        mut image_mask: Option<ImageMask>,
        reference_frame_relative_offset: &LayoutVector2D,
    ) -> Self
    where
        J: Iterator<Item = ComplexClipRegion>
    {
        if let Some(ref mut image_mask) = image_mask {
            image_mask.rect = image_mask.rect.translate(reference_frame_relative_offset);
        }

        ClipRegion {
            main: rect.translate(reference_frame_relative_offset),
            image_mask,
            complex_clips: ComplexTranslateIter {
                source: complex_clips,
                offset: *reference_frame_relative_offset,
            },
        }
    }
}

impl ClipRegion<Option<ComplexClipRegion>> {
    pub fn create_for_clip_node_with_local_clip(
        local_clip: &LocalClip,
        reference_frame_relative_offset: &LayoutVector2D
    ) -> Self {
        ClipRegion {
            main: local_clip
                .clip_rect()
                .translate(reference_frame_relative_offset),
            image_mask: None,
            complex_clips: match *local_clip {
                LocalClip::Rect(_) => None,
                LocalClip::RoundedRect(_, ref region) => {
                    Some(ComplexClipRegion {
                        rect: region.rect.translate(reference_frame_relative_offset),
                        radii: region.radii,
                        mode: region.mode,
                    })
                },
            }
        }
    }
}

#[derive(Debug)]
pub enum ClipItem {
    Rectangle(LayoutRect, ClipMode),
    RoundedRectangle(LayoutRect, BorderRadius, ClipMode),
    Image(ImageMask),
    BoxShadow(BoxShadowClipSource),
    LineDecoration(LineDecorationClipSource),
}

impl ClipItem {
    pub fn new_rounded_rect(
        rect: LayoutRect,
        mut radii: BorderRadius,
        clip_mode: ClipMode
    ) -> Self {
        if radii.is_zero() {
            ClipItem::Rectangle(rect, clip_mode)
        } else {
            ensure_no_corner_overlap(&mut radii, &rect);
            ClipItem::RoundedRectangle(
                rect,
                radii,
                clip_mode,
            )
        }
    }

    pub fn new_line_decoration(
        rect: LayoutRect,
        style: LineStyle,
        orientation: LineOrientation,
        wavy_line_thickness: f32,
    ) -> Self {
        ClipItem::LineDecoration(
            LineDecorationClipSource {
                rect,
                style,
                orientation,
                wavy_line_thickness,
            }
        )
    }

    pub fn new_box_shadow(
        shadow_rect: LayoutRect,
        shadow_radius: BorderRadius,
        prim_shadow_rect: LayoutRect,
        blur_radius: f32,
        clip_mode: BoxShadowClipMode,
    ) -> Self {
        // Get the fractional offsets required to match the
        // source rect with a minimal rect.
        let fract_offset = LayoutPoint::new(
            shadow_rect.origin.x.fract().abs(),
            shadow_rect.origin.y.fract().abs(),
        );
        let fract_size = LayoutSize::new(
            shadow_rect.size.width.fract().abs(),
            shadow_rect.size.height.fract().abs(),
        );

        // Create a minimal size primitive mask to blur. In this
        // case, we ensure the size of each corner is the same,
        // to simplify the shader logic that stretches the blurred
        // result across the primitive.
        let max_corner_width = shadow_radius.top_left.width
                                    .max(shadow_radius.bottom_left.width)
                                    .max(shadow_radius.top_right.width)
                                    .max(shadow_radius.bottom_right.width);
        let max_corner_height = shadow_radius.top_left.height
                                    .max(shadow_radius.bottom_left.height)
                                    .max(shadow_radius.top_right.height)
                                    .max(shadow_radius.bottom_right.height);

        // Get maximum distance that can be affected by given blur radius.
        let blur_region = (BLUR_SAMPLE_SCALE * blur_radius).ceil();

        // If the largest corner is smaller than the blur radius, we need to ensure
        // that it's big enough that the corners don't affect the middle segments.
        let used_corner_width = max_corner_width.max(blur_region);
        let used_corner_height = max_corner_height.max(blur_region);

        // Minimal nine-patch size, corner + internal + corner.
        let min_shadow_rect_size = LayoutSize::new(
            2.0 * used_corner_width + blur_region,
            2.0 * used_corner_height + blur_region,
        );

        // The minimal rect to blur.
        let mut minimal_shadow_rect = LayoutRect::new(
            LayoutPoint::new(
                blur_region + fract_offset.x,
                blur_region + fract_offset.y,
            ),
            LayoutSize::new(
                min_shadow_rect_size.width + fract_size.width,
                min_shadow_rect_size.height + fract_size.height,
            ),
        );

        // If the width or height ends up being bigger than the original
        // primitive shadow rect, just blur the entire rect along that
        // axis and draw that as a simple blit. This is necessary for
        // correctness, since the blur of one corner may affect the blur
        // in another corner.
        let mut stretch_mode_x = BoxShadowStretchMode::Stretch;
        if shadow_rect.size.width < minimal_shadow_rect.size.width {
            minimal_shadow_rect.size.width = shadow_rect.size.width;
            stretch_mode_x = BoxShadowStretchMode::Simple;
        }

        let mut stretch_mode_y = BoxShadowStretchMode::Stretch;
        if shadow_rect.size.height < minimal_shadow_rect.size.height {
            minimal_shadow_rect.size.height = shadow_rect.size.height;
            stretch_mode_y = BoxShadowStretchMode::Simple;
        }

        // Expand the shadow rect by enough room for the blur to take effect.
        let shadow_rect_alloc_size = LayoutSize::new(
            2.0 * blur_region + minimal_shadow_rect.size.width.ceil(),
            2.0 * blur_region + minimal_shadow_rect.size.height.ceil(),
        );

        ClipItem::BoxShadow(BoxShadowClipSource {
            shadow_rect_alloc_size,
            shadow_radius,
            prim_shadow_rect,
            blur_radius,
            clip_mode,
            stretch_mode_x,
            stretch_mode_y,
            cache_handle: None,
            cache_key: None,
            clip_data_handle: GpuCacheHandle::new(),
            minimal_shadow_rect,
        })
    }

    // Return a modified clip source that is the same as self
    // but offset in local-space by a specified amount.
    pub fn offset(&self, offset: &LayoutVector2D) -> Self {
        match *self {
            ClipItem::LineDecoration(ref info) => {
                ClipItem::LineDecoration(LineDecorationClipSource {
                    rect: info.rect.translate(offset),
                    ..*info
                })
            }
            _ => {
                panic!("bug: other clip sources not expected here yet");
            }
        }
    }

    // Get an optional clip rect that a clip source can provide to
    // reduce the size of a primitive region. This is typically
    // used to eliminate redundant clips, and reduce the size of
    // any clip mask that eventually gets drawn.
    fn get_local_clip_rect(&self) -> Option<LayoutRect> {
        match *self {
            ClipItem::Rectangle(clip_rect, ClipMode::Clip) => Some(clip_rect),
            ClipItem::Rectangle(_, ClipMode::ClipOut) => None,
            ClipItem::RoundedRectangle(clip_rect, _, ClipMode::Clip) => Some(clip_rect),
            ClipItem::RoundedRectangle(_, _, ClipMode::ClipOut) => None,
            ClipItem::Image(ref mask) if mask.repeat => None,
            ClipItem::Image(ref mask) => Some(mask.rect),
            ClipItem::BoxShadow(..) => None,
            ClipItem::LineDecoration(..) => None,
        }
    }

    fn get_clip_result_complex(
        &self,
        transform: &LayoutToWorldTransform,
        prim_rect: &WorldRect,
    ) -> ClipResult {
        match *self {
            ClipItem::Rectangle(ref clip_rect, ClipMode::Clip) => {
                if let Some(inner_clip_rect) = project_inner_rect(transform, clip_rect) {
                    if inner_clip_rect.contains_rect(prim_rect) {
                        return ClipResult::Accept;
                    }
                }

                let outer_clip_rect = match project_rect(transform, clip_rect) {
                    Some(outer_clip_rect) => outer_clip_rect,
                    None => return ClipResult::Partial,
                };

                match outer_clip_rect.intersection(prim_rect) {
                    Some(..) => {
                        ClipResult::Partial
                    }
                    None => {
                        ClipResult::Reject
                    }
                }
            }
            ClipItem::RoundedRectangle(ref clip_rect, ref radius, ClipMode::Clip) => {
                let inner_clip_rect = extract_inner_rect_safe(clip_rect, radius)
                    .and_then(|ref inner_clip_rect| {
                        project_inner_rect(transform, inner_clip_rect)
                    });

                if let Some(inner_clip_rect) = inner_clip_rect {
                    if inner_clip_rect.contains_rect(prim_rect) {
                        return ClipResult::Accept;
                    }
                }

                let outer_clip_rect = match project_rect(transform, clip_rect) {
                    Some(outer_clip_rect) => outer_clip_rect,
                    None => return ClipResult::Partial,
                };

                match outer_clip_rect.intersection(prim_rect) {
                    Some(..) => {
                        ClipResult::Partial
                    }
                    None => {
                        ClipResult::Reject
                    }
                }
            }
            ClipItem::Rectangle(_, ClipMode::ClipOut) |
            ClipItem::RoundedRectangle(_, _, ClipMode::ClipOut) |
            ClipItem::Image(..) |
            ClipItem::BoxShadow(..) |
            ClipItem::LineDecoration(..) => {
                ClipResult::Partial
            }
        }
    }

    // Check how a given clip source affects a local primitive region.
    fn get_clip_result(
        &self,
        prim_rect: &LayoutRect,
    ) -> ClipResult {
        match *self {
            ClipItem::Rectangle(ref clip_rect, ClipMode::Clip) => {
                if clip_rect.contains_rect(prim_rect) {
                    return ClipResult::Accept;
                }

                match clip_rect.intersection(prim_rect) {
                    Some(..) => {
                        ClipResult::Partial
                    }
                    None => {
                        ClipResult::Reject
                    }
                }
            }
            ClipItem::Rectangle(ref clip_rect, ClipMode::ClipOut) => {
                if clip_rect.contains_rect(prim_rect) {
                    return ClipResult::Reject;
                }

                match clip_rect.intersection(prim_rect) {
                    Some(_) => {
                        ClipResult::Partial
                    }
                    None => {
                        ClipResult::Accept
                    }
                }
            }
            ClipItem::RoundedRectangle(ref clip_rect, ref radius, ClipMode::Clip) => {
                // TODO(gw): Consider caching this in the ClipNode
                //           if it ever shows in profiles.
                // TODO(gw): extract_inner_rect_safe is overly
                //           conservative for this code!
                let inner_clip_rect = extract_inner_rect_safe(clip_rect, radius);
                if let Some(inner_clip_rect) = inner_clip_rect {
                    if inner_clip_rect.contains_rect(prim_rect) {
                        return ClipResult::Accept;
                    }
                }

                match clip_rect.intersection(prim_rect) {
                    Some(..) => {
                        ClipResult::Partial
                    }
                    None => {
                        ClipResult::Reject
                    }
                }
            }
            ClipItem::RoundedRectangle(ref clip_rect, ref radius, ClipMode::ClipOut) => {
                // TODO(gw): Consider caching this in the ClipNode
                //           if it ever shows in profiles.
                // TODO(gw): extract_inner_rect_safe is overly
                //           conservative for this code!
                let inner_clip_rect = extract_inner_rect_safe(clip_rect, radius);
                if let Some(inner_clip_rect) = inner_clip_rect {
                    if inner_clip_rect.contains_rect(prim_rect) {
                        return ClipResult::Reject;
                    }
                }

                match clip_rect.intersection(prim_rect) {
                    Some(_) => {
                        ClipResult::Partial
                    }
                    None => {
                        ClipResult::Accept
                    }
                }
            }
            ClipItem::Image(ref mask) => {
                if mask.repeat {
                    ClipResult::Partial
                } else {
                    match mask.rect.intersection(prim_rect) {
                        Some(..) => {
                            ClipResult::Partial
                        }
                        None => {
                            ClipResult::Reject
                        }
                    }
                }
            }
            ClipItem::BoxShadow(..) |
            ClipItem::LineDecoration(..) => {
                ClipResult::Partial
            }
        }
    }
}

/// Represents a local rect and a device space
/// rectangles that are either outside or inside bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    pub local_rect: LayoutRect,
    pub device_rect: DeviceIntRect,
}

impl From<LayoutRect> for Geometry {
    fn from(local_rect: LayoutRect) -> Self {
        Geometry {
            local_rect,
            device_rect: DeviceIntRect::zero(),
        }
    }
}

pub fn rounded_rectangle_contains_point(
    point: &LayoutPoint,
    rect: &LayoutRect,
    radii: &BorderRadius
) -> bool {
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

fn project_rect(
    transform: &LayoutToWorldTransform,
    rect: &LayoutRect,
) -> Option<WorldRect> {
    let homogens = [
        transform.transform_point2d_homogeneous(&rect.origin),
        transform.transform_point2d_homogeneous(&rect.top_right()),
        transform.transform_point2d_homogeneous(&rect.bottom_left()),
        transform.transform_point2d_homogeneous(&rect.bottom_right()),
    ];

    // Note: we only do the full frustum collision when the polygon approaches the camera plane.
    // Otherwise, it will be clamped to the screen bounds anyway.
    if homogens.iter().any(|h| h.w <= 0.0) {
        let mut clipper = Clipper::new();
        clipper.add_frustum(
            transform,
            None,
        );

        let polygon = Polygon::from_rect(*rect, 1);
        let results = clipper.clip(polygon);
        if results.is_empty() {
            return None
        }

        Some(WorldRect::from_points(results
            .into_iter()
            // filter out parts behind the view plane
            .flat_map(|poly| &poly.points)
            .map(|p| {
                let mut homo = transform.transform_point2d_homogeneous(&p.to_2d());
                homo.w = homo.w.max(0.00000001); // avoid infinite values
                homo.to_point2d().unwrap()
            })
        ))
    } else {
        // we just checked for all the points to be in positive hemisphere, so `unwrap` is valid
        Some(WorldRect::from_points(&[
            homogens[0].to_point2d().unwrap(),
            homogens[1].to_point2d().unwrap(),
            homogens[2].to_point2d().unwrap(),
            homogens[3].to_point2d().unwrap(),
        ]))
    }
}

pub fn project_inner_rect(
    transform: &LayoutToWorldTransform,
    rect: &LayoutRect,
) -> Option<WorldRect> {
    let homogens = [
        transform.transform_point2d_homogeneous(&rect.origin),
        transform.transform_point2d_homogeneous(&rect.top_right()),
        transform.transform_point2d_homogeneous(&rect.bottom_left()),
        transform.transform_point2d_homogeneous(&rect.bottom_right()),
    ];

    // Note: we only do the full frustum collision when the polygon approaches the camera plane.
    // Otherwise, it will be clamped to the screen bounds anyway.
    if homogens.iter().any(|h| h.w <= 0.0) {
        None
    } else {
        // we just checked for all the points to be in positive hemisphere, so `unwrap` is valid
        let points = [
            homogens[0].to_point2d().unwrap(),
            homogens[1].to_point2d().unwrap(),
            homogens[2].to_point2d().unwrap(),
            homogens[3].to_point2d().unwrap(),
        ];
        let mut xs = [points[0].x, points[1].x, points[2].x, points[3].x];
        let mut ys = [points[0].y, points[1].y, points[2].y, points[3].y];
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(cmp::Ordering::Equal));
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(cmp::Ordering::Equal));
        Some(WorldRect::new(
            WorldPoint::new(xs[1], ys[1]),
            WorldSize::new(xs[2] - xs[1], ys[2] - ys[1]),
        ))
    }
}
