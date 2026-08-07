/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Quadtree-based dirty region tracking for tiles
//!
//! This module implements a quadtree data structure that tracks which regions
//! of a tile have been invalidated. The quadtree can dynamically split and merge
//! nodes based on invalidation patterns to optimize tracking.

use api::units::*;
use crate::debug_colors;
use crate::internal_types::FastHashMap;
use crate::prim_store::PrimitiveScratchBuffer;
use crate::space::SpaceMapper;
use crate::invalidation::{InvalidationReason, PrimitiveCompareResult};
use crate::invalidation::cached_surface::{PrimitiveDescriptor, PrimitiveDependencyIndex};
use crate::invalidation::compare::{PrimitiveComparer, PrimitiveComparisonKey};
use crate::visibility::FrameVisibilityContext;
use std::mem;


/// Details for a node in a quadtree that tracks dirty rects for a tile.
#[cfg_attr(any(feature="capture",feature="replay"), derive(Clone))]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum TileNodeKind {
    Leaf {
        /// The index buffer of primitives that affected this tile previous frame
        #[cfg_attr(any(feature = "capture", feature = "replay"), serde(skip))]
        prev_indices: Vec<PrimitiveDependencyIndex>,
        /// The index buffer of primitives that affect this tile on this frame
        #[cfg_attr(any(feature = "capture", feature = "replay"), serde(skip))]
        curr_indices: Vec<PrimitiveDependencyIndex>,
        /// A bitset of which of the last 64 frames have been dirty for this leaf.
        #[cfg_attr(any(feature = "capture", feature = "replay"), serde(skip))]
        dirty_tracker: u64,
        /// The number of frames since this node split or merged.
        #[cfg_attr(any(feature = "capture", feature = "replay"), serde(skip))]
        frames_since_modified: usize,
    },
    Node {
        /// The four children of this node
        children: Vec<TileNode>,
    },
}

/// The kind of modification that a tile wants to do
#[derive(Copy, Clone, PartialEq, Debug)]
enum TileModification {
    Split,
    Merge,
}

/// A node in the dirty rect tracking quadtree.
#[cfg_attr(any(feature="capture",feature="replay"), derive(Clone))]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct TileNode {
    /// Leaf or internal node
    pub kind: TileNodeKind,
    /// Rect of this node in the same space as the tile cache picture
    pub rect: PictureBox2D,
}

impl TileNode {
    /// Construct a new leaf node, with the given primitive dependency index buffer
    pub fn new_leaf(curr_indices: Vec<PrimitiveDependencyIndex>) -> Self {
        TileNode {
            kind: TileNodeKind::Leaf {
                prev_indices: Vec::new(),
                curr_indices,
                dirty_tracker: 0,
                frames_since_modified: 0,
            },
            rect: PictureBox2D::zero(),
        }
    }

    /// Draw debug information about this tile node
    pub fn draw_debug_rects(
        &self,
        pic_to_world_mapper: &SpaceMapper<PicturePixel, WorldPixel>,
        is_opaque: bool,
        local_valid_rect: PictureRect,
        scratch: &mut PrimitiveScratchBuffer,
        global_device_pixel_scale: DevicePixelScale,
    ) {
        match self.kind {
            TileNodeKind::Leaf { dirty_tracker, .. } => {
                let color = if (dirty_tracker & 1) != 0 {
                    debug_colors::RED
                } else if is_opaque {
                    debug_colors::GREEN
                } else {
                    debug_colors::YELLOW
                };

                if let Some(local_rect) = local_valid_rect.intersection(&self.rect) {
                    let world_rect = pic_to_world_mapper
                        .map(&local_rect)
                        .unwrap();
                    let device_rect = world_rect * global_device_pixel_scale;

                    let outer_color = color.scale_alpha(0.3);
                    let inner_color = outer_color.scale_alpha(0.5);
                    scratch.push_debug_rect(
                        device_rect.inflate(-3.0, -3.0),
                        1,
                        outer_color,
                        inner_color
                    );
                }
            }
            TileNodeKind::Node { ref children, .. } => {
                for child in children.iter() {
                    child.draw_debug_rects(
                        pic_to_world_mapper,
                        is_opaque,
                        local_valid_rect,
                        scratch,
                        global_device_pixel_scale,
                    );
                }
            }
        }
    }

    /// Calculate the four child rects for a given node
    fn get_child_rects(
        rect: &PictureBox2D,
        result: &mut [PictureBox2D; 4],
    ) {
        let p0 = rect.min;
        let p1 = rect.max;
        let pc = p0 + rect.size() * 0.5;

        *result = [
            PictureBox2D::new(
                p0,
                pc,
            ),
            PictureBox2D::new(
                PicturePoint::new(pc.x, p0.y),
                PicturePoint::new(p1.x, pc.y),
            ),
            PictureBox2D::new(
                PicturePoint::new(p0.x, pc.y),
                PicturePoint::new(pc.x, p1.y),
            ),
            PictureBox2D::new(
                pc,
                p1,
            ),
        ];
    }

    /// Called during pre_update, to clear the current dependencies
    pub fn clear(
        &mut self,
        rect: PictureBox2D,
    ) {
        self.rect = rect;

        match self.kind {
            TileNodeKind::Leaf { ref mut prev_indices, ref mut curr_indices, ref mut dirty_tracker, ref mut frames_since_modified } => {
                // Swap current dependencies to be the previous frame
                mem::swap(prev_indices, curr_indices);
                curr_indices.clear();
                // Note that another frame has passed in the dirty bit trackers
                *dirty_tracker = *dirty_tracker << 1;
                *frames_since_modified += 1;
            }
            TileNodeKind::Node { ref mut children, .. } => {
                let mut child_rects = [PictureBox2D::zero(); 4];
                TileNode::get_child_rects(&rect, &mut child_rects);
                assert_eq!(child_rects.len(), children.len());

                for (child, rect) in children.iter_mut().zip(child_rects.iter()) {
                    child.clear(*rect);
                }
            }
        }
    }

    /// Add a primitive dependency to this node
    pub fn add_prim(
        &mut self,
        index: PrimitiveDependencyIndex,
        prim_rect: &PictureBox2D,
    ) {
        match self.kind {
            TileNodeKind::Leaf { ref mut curr_indices, .. } => {
                curr_indices.push(index);
            }
            TileNodeKind::Node { ref mut children, .. } => {
                for child in children.iter_mut() {
                    if child.rect.intersects(prim_rect) {
                        child.add_prim(index, prim_rect);
                    }
                }
            }
        }
    }

    /// Apply a merge or split operation to this tile, if desired
    pub fn maybe_merge_or_split(
        &mut self,
        level: i32,
        curr_prims: &[PrimitiveDescriptor],
        max_split_levels: i32,
    ) {
        // Determine if this tile wants to split or merge
        let mut tile_mod = None;

        fn get_dirty_frames(
            dirty_tracker: u64,
            frames_since_modified: usize,
        ) -> Option<u32> {
            // Only consider splitting or merging at least 64 frames since we last changed
            if frames_since_modified > 64 {
                // Each bit in the tracker is a frame that was recently invalidated
                Some(dirty_tracker.count_ones())
            } else {
                None
            }
        }

        match self.kind {
            TileNodeKind::Leaf { dirty_tracker, frames_since_modified, .. } => {
                // Only consider splitting if the tree isn't too deep.
                if level < max_split_levels {
                    if let Some(dirty_frames) = get_dirty_frames(dirty_tracker, frames_since_modified) {
                        // If the tile has invalidated > 50% of the recent number of frames, split.
                        if dirty_frames > 32 {
                            tile_mod = Some(TileModification::Split);
                        }
                    }
                }
            }
            TileNodeKind::Node { ref children, .. } => {
                // There's two conditions that cause a node to merge its children:
                // (1) If _all_ the child nodes are constantly invalidating, then we are wasting
                //     CPU time tracking dependencies for each child, so merge them.
                // (2) If _none_ of the child nodes are recently invalid, then the page content
                //     has probably changed, and we no longer need to track fine grained dependencies here.

                let mut static_count = 0;
                let mut changing_count = 0;

                for child in children {
                    // Only consider merging nodes at the edge of the tree.
                    if let TileNodeKind::Leaf { dirty_tracker, frames_since_modified, .. } = child.kind {
                        if let Some(dirty_frames) = get_dirty_frames(dirty_tracker, frames_since_modified) {
                            if dirty_frames == 0 {
                                // Hasn't been invalidated for some time
                                static_count += 1;
                            } else if dirty_frames == 64 {
                                // Is constantly being invalidated
                                changing_count += 1;
                            }
                        }
                    }

                    // Only merge if all the child tiles are in agreement. Otherwise, we have some
                    // that are invalidating / static, and it's worthwhile tracking dependencies for
                    // them individually.
                    if static_count == 4 || changing_count == 4 {
                        tile_mod = Some(TileModification::Merge);
                    }
                }
            }
        }

        match tile_mod {
            Some(TileModification::Split) => {
                // To split a node, take the current dependency index buffer for this node, and
                // split it into child index buffers.
                let curr_indices = match self.kind {
                    TileNodeKind::Node { .. } => {
                        unreachable!("bug - only leaves can split");
                    }
                    TileNodeKind::Leaf { ref mut curr_indices, .. } => {
                        mem::take(curr_indices)
                    }
                };

                let mut child_rects = [PictureBox2D::zero(); 4];
                TileNode::get_child_rects(&self.rect, &mut child_rects);

                let mut child_indices = [
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                ];

                // Step through the index buffer, and add primitives to each of the children
                // that they intersect.
                for index in curr_indices {
                    let prim = &curr_prims[index.0 as usize];
                    for (child_rect, indices) in child_rects.iter().zip(child_indices.iter_mut()) {
                        if prim.prim_clip_box.intersects(child_rect) {
                            indices.push(index);
                        }
                    }
                }

                // Create the child nodes and switch from leaf -> node.
                let children = child_indices
                    .iter_mut()
                    .map(|i| TileNode::new_leaf(mem::replace(i, Vec::new())))
                    .collect();

                self.kind = TileNodeKind::Node {
                    children,
                };
            }
            Some(TileModification::Merge) => {
                // Construct a merged index buffer by collecting the dependency index buffers
                // from each child, and merging them into a de-duplicated index buffer.
                let merged_indices = match self.kind {
                    TileNodeKind::Node { ref mut children, .. } => {
                        let mut merged_indices = Vec::new();

                        for child in children.iter() {
                            let child_indices = match child.kind {
                                TileNodeKind::Leaf { ref curr_indices, .. } => {
                                    curr_indices
                                }
                                TileNodeKind::Node { .. } => {
                                    unreachable!("bug: child is not a leaf");
                                }
                            };
                            merged_indices.extend_from_slice(child_indices);
                        }

                        merged_indices.sort();
                        merged_indices.dedup();

                        merged_indices
                    }
                    TileNodeKind::Leaf { .. } => {
                        unreachable!("bug - trying to merge a leaf");
                    }
                };

                // Switch from a node to a leaf, with the combined index buffer
                self.kind = TileNodeKind::Leaf {
                    prev_indices: Vec::new(),
                    curr_indices: merged_indices,
                    dirty_tracker: 0,
                    frames_since_modified: 0,
                };
            }
            None => {
                // If this node didn't merge / split, then recurse into children
                // to see if they want to split / merge.
                if let TileNodeKind::Node { ref mut children, .. } = self.kind {
                    for child in children.iter_mut() {
                        child.maybe_merge_or_split(
                            level+1,
                            curr_prims,
                            max_split_levels,
                        );
                    }
                }
            }
        }
    }

    /// Update the dirty state of this node, building the overall dirty rect
    pub fn update_dirty_rects(
        &mut self,
        prev_prims: &[PrimitiveDescriptor],
        curr_prims: &[PrimitiveDescriptor],
        prim_comparer: &mut PrimitiveComparer,
        dirty_rect: &mut PictureBox2D,
        compare_cache: &mut FastHashMap<PrimitiveComparisonKey, PrimitiveCompareResult>,
        invalidation_reason: &mut Option<InvalidationReason>,
        frame_context: &FrameVisibilityContext,
    ) {
        match self.kind {
            TileNodeKind::Node { ref mut children, .. } => {
                for child in children.iter_mut() {
                    child.update_dirty_rects(
                        prev_prims,
                        curr_prims,
                        prim_comparer,
                        dirty_rect,
                        compare_cache,
                        invalidation_reason,
                        frame_context,
                    );
                }
            }
            TileNodeKind::Leaf { ref prev_indices, ref curr_indices, ref mut dirty_tracker, .. } => {
                // If the index buffers are of different length, they must be different
                if prev_indices.len() == curr_indices.len() {
                    // Walk each index buffer, comparing primitives
                    for (prev_index, curr_index) in prev_indices.iter().zip(curr_indices.iter()) {
                        let i0 = prev_index.0 as usize;
                        let i1 = curr_index.0 as usize;

                        // Compare the primitives, caching the result in a hash map
                        // to save comparisons in other tree nodes.
                        let key = PrimitiveComparisonKey {
                            prev_index: *prev_index,
                            curr_index: *curr_index,
                        };

                        let prim_compare_result = *compare_cache
                            .entry(key)
                            .or_insert_with(|| {
                                let prev = &prev_prims[i0];
                                let curr = &curr_prims[i1];
                                prim_comparer.compare_prim(prev, curr)
                            });

                        // If not the same, mark this node as dirty and update the dirty rect
                        if prim_compare_result != PrimitiveCompareResult::Equal {
                            if invalidation_reason.is_none() {
                                *invalidation_reason = Some(InvalidationReason::Content);
                            }
                            *dirty_rect = self.rect.union(dirty_rect);
                            *dirty_tracker = *dirty_tracker | 1;
                            break;
                        }
                    }
                } else {
                    if invalidation_reason.is_none() {
                        *invalidation_reason = Some(InvalidationReason::PrimCount);
                    }
                    *dirty_rect = self.rect.union(dirty_rect);
                    *dirty_tracker = *dirty_tracker | 1;
                }
            }
        }
    }
}
