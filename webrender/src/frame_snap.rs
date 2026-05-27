/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Frame-time pass that snaps prim, cluster, and clip-tree rects to the
//! device pixel grid. Runs once near the start of frame building, after
//! `spatial_tree.update_tree`, and writes the `snapped_*` rect fields that
//! downstream visibility / prepare / batch consumers read.

use api::units::{LayoutRect, RasterPixelScale};
use crate::clip::ClipTree;
use crate::prim_store::{PrimitiveInstance, PrimitiveStore};
use crate::space::SpaceSnapper;
use crate::spatial_tree::{SpatialNodeIndex, SpatialTree};
use crate::util::MaxRect;
use crate::visibility::PrimitiveDrawHeader;

/// Snap all per-frame rect state that depends on the current spatial tree:
/// each prim's `snapped_local_rect` (into `draws`), each cluster's
/// `snapped_bounding_rect`, each prim's clip-leaf rect, and each clip-tree
/// node's rect. All snapping happens against the root reference frame in
/// the rect's own spatial-node space, mirroring the scene-build helper this
/// replaces.
///
/// Per-prim clip leaves are snapped inside the cluster loop, against the
/// cluster's spatial node — the leaf's local clip rect lives in the prim's
/// local space, and clusters group prims by spatial node, so the cluster's
/// (resolved) spatial node is the correct snap target. This matters for
/// backdrop-filter capture leaves where the prim's clip-leaf spatial node
/// would otherwise be `UNKNOWN`: the cluster is resolved by
/// `finalize_picture` before this pass runs.
pub fn snap_frame_rects(
    prim_store: &mut PrimitiveStore,
    prim_instances: &[PrimitiveInstance],
    clip_tree: &mut ClipTree,
    draws: &mut [PrimitiveDrawHeader],
    spatial_tree: &SpatialTree,
) {
    profile_scope!("snap_frame_rects");

    let root = spatial_tree.root_reference_frame_index();
    let mut snapper = SpaceSnapper::new(root, RasterPixelScale::new(1.0));

    for pic in &mut prim_store.pictures {
        for cluster in &mut pic.prim_list.clusters {
            snapper.set_target_spatial_node(cluster.spatial_node_index, spatial_tree);
            cluster.snapped_bounding_rect = snapper.snap_rect(&cluster.unsnapped_bounding_rect);

            for prim_idx in cluster.prim_range() {
                draws[prim_idx].snapped_local_rect =
                    snapper.snap_rect(&prim_instances[prim_idx].unsnapped_prim_rect);

                // Snap the prim's clip-leaf in the same spatial-node space.
                // Picture / tile-cache leaves carry `max_rect`; snapping
                // `max_rect` would overflow through the snapper's transforms,
                // so pass those through unchanged.
                let leaf = clip_tree.get_leaf_mut(prim_instances[prim_idx].clip_leaf_id);
                if leaf.unsnapped_local_clip_rect == LayoutRect::max_rect() {
                    leaf.snapped_local_clip_rect = leaf.unsnapped_local_clip_rect;
                } else {
                    let unsnapped = leaf.unsnapped_local_clip_rect;
                    leaf.snapped_local_clip_rect = snapper.snap_rect(&unsnapped);
                }
            }
        }
    }

    for node in clip_tree.nodes_mut() {
        if node.spatial_node_index == SpatialNodeIndex::INVALID {
            node.snapped_clip_rect = node.unsnapped_clip_rect;
            continue;
        }
        snapper.set_target_spatial_node(node.spatial_node_index, spatial_tree);
        node.snapped_clip_rect = snapper.snap_rect(&node.unsnapped_clip_rect);
    }
}
