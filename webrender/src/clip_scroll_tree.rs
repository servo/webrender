/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{ClipId, DeviceIntRect, LayerPoint, LayerRect, LayerToScrollTransform};
use api::{LayerToWorldTransform, LayerVector2D, PipelineId, ScrollClamping, ScrollEventPhase};
use api::{ScrollLayerState, ScrollLocation, WorldPoint};
use clip::ClipStore;
use clip_scroll_node::{ClipScrollNode, NodeType, ScrollingState, StickyFrameInfo};
use gpu_cache::GpuCache;
use gpu_types::ClipScrollNodeData;
use internal_types::{FastHashMap, FastHashSet};
use print_tree::{PrintTree, PrintTreePrinter};
use render_task::ClipChain;
use resource_cache::ResourceCache;

pub type ScrollStates = FastHashMap<ClipId, ScrollingState>;

/// An id that identifies coordinate systems in the ClipScrollTree. Each
/// coordinate system has an id and those ids will be shared when the coordinates
/// system are the same or are in the same axis-aligned space. This allows
/// for optimizing mask generation.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CoordinateSystemId(pub u32);

impl CoordinateSystemId {
    pub fn next(&self) -> CoordinateSystemId {
        let CoordinateSystemId(id) = *self;
        CoordinateSystemId(id + 1)
    }
}

pub struct ClipScrollTree {
    pub nodes: FastHashMap<ClipId, ClipScrollNode>,
    pub pending_scroll_offsets: FastHashMap<ClipId, (LayerPoint, ScrollClamping)>,

    /// The ClipId of the currently scrolling node. Used to allow the same
    /// node to scroll even if a touch operation leaves the boundaries of that node.
    pub currently_scrolling_node_id: Option<ClipId>,

    /// The current frame id, used for giving a unique id to all new dynamically
    /// added frames and clips. The ClipScrollTree increments this by one every
    /// time a new dynamic frame is created.
    current_new_node_item: u64,

    /// The root reference frame, which is the true root of the ClipScrollTree. Initially
    /// this ID is not valid, which is indicated by ```node``` being empty.
    pub root_reference_frame_id: ClipId,

    /// The root scroll node which is the first child of the root reference frame.
    /// Initially this ID is not valid, which is indicated by ```nodes``` being empty.
    pub topmost_scrolling_node_id: ClipId,

    /// A set of pipelines which should be discarded the next time this
    /// tree is drained.
    pub pipelines_to_discard: FastHashSet<PipelineId>,
}

#[derive(Clone)]
pub struct TransformUpdateState {
    pub parent_reference_frame_transform: LayerToWorldTransform,
    pub parent_combined_viewport_rect: LayerRect,
    pub parent_accumulated_scroll_offset: LayerVector2D,
    pub nearest_scrolling_ancestor_offset: LayerVector2D,
    pub nearest_scrolling_ancestor_viewport: LayerRect,
    pub parent_clip_chain: ClipChain,
    pub combined_outer_clip_bounds: DeviceIntRect,

    /// An id for keeping track of the axis-aligned space of this node. This is used in
    /// order to to track what kinds of clip optimizations can be done for a particular
    /// display list item, since optimizations can usually only be done among
    /// coordinate systems which are relatively axis aligned.
    pub current_coordinate_system_id: CoordinateSystemId,
    pub next_coordinate_system_id: CoordinateSystemId,
}

impl ClipScrollTree {
    pub fn new() -> ClipScrollTree {
        let dummy_pipeline = PipelineId::dummy();
        ClipScrollTree {
            nodes: FastHashMap::default(),
            pending_scroll_offsets: FastHashMap::default(),
            currently_scrolling_node_id: None,
            root_reference_frame_id: ClipId::root_reference_frame(dummy_pipeline),
            topmost_scrolling_node_id: ClipId::root_scroll_node(dummy_pipeline),
            current_new_node_item: 1,
            pipelines_to_discard: FastHashSet::default(),
        }
    }

    pub fn root_reference_frame_id(&self) -> ClipId {
        // TODO(mrobinson): We should eventually make this impossible to misuse.
        debug_assert!(!self.nodes.is_empty());
        debug_assert!(self.nodes.contains_key(&self.root_reference_frame_id));
        self.root_reference_frame_id
    }

    pub fn topmost_scrolling_node_id(&self) -> ClipId {
        // TODO(mrobinson): We should eventually make this impossible to misuse.
        debug_assert!(!self.nodes.is_empty());
        debug_assert!(self.nodes.contains_key(&self.topmost_scrolling_node_id));
        self.topmost_scrolling_node_id
    }

    pub fn collect_nodes_bouncing_back(&self) -> FastHashSet<ClipId> {
        let mut nodes_bouncing_back = FastHashSet::default();
        for (clip_id, node) in self.nodes.iter() {
            if let NodeType::ScrollFrame(ref scrolling) = node.node_type {
                if scrolling.bouncing_back {
                    nodes_bouncing_back.insert(*clip_id);
                }
            }
        }
        nodes_bouncing_back
    }

    fn find_scrolling_node_at_point_in_node(
        &self,
        cursor: &WorldPoint,
        clip_id: ClipId,
    ) -> Option<ClipId> {
        self.nodes.get(&clip_id).and_then(|node| {
            for child_layer_id in node.children.iter().rev() {
                if let Some(layer_id) =
                    self.find_scrolling_node_at_point_in_node(cursor, *child_layer_id)
                {
                    return Some(layer_id);
                }
            }

            match node.node_type {
                NodeType::ScrollFrame(state) if state.sensitive_to_input_events() => {}
                _ => return None,
            }

            if node.ray_intersects_node(cursor) {
                Some(clip_id)
            } else {
                None
            }
        })
    }

    pub fn find_scrolling_node_at_point(&self, cursor: &WorldPoint) -> ClipId {
        self.find_scrolling_node_at_point_in_node(cursor, self.root_reference_frame_id())
            .unwrap_or(self.topmost_scrolling_node_id())
    }

    pub fn is_point_clipped_in_for_node(
        &self,
        point: WorldPoint,
        node_id: &ClipId,
        cache: &mut FastHashMap<ClipId, Option<LayerPoint>>,
        clip_store: &ClipStore
    ) -> bool {
        if let Some(point) = cache.get(node_id) {
            return point.is_some();
        }

        let node = self.nodes.get(node_id).unwrap();
        let parent_clipped_in = match node.parent {
            None => true, // This is the root node.
            Some(ref parent_id) => {
                self.is_point_clipped_in_for_node(point, parent_id, cache, clip_store)
            }
        };

        if !parent_clipped_in {
            cache.insert(*node_id, None);
            return false;
        }

        let transform = node.world_viewport_transform;
        let transformed_point = match transform.inverse() {
            Some(inverted) => inverted.transform_point2d(&point),
            None => {
                cache.insert(*node_id, None);
                return false;
            }
        };

        let point_in_layer = transformed_point - node.local_viewport_rect.origin.to_vector();
        let clip_info = match node.node_type {
            NodeType::Clip(ref info) => info,
            _ => {
                cache.insert(*node_id, Some(point_in_layer));
                return true;
            }
        };

        if !node.local_clip_rect.contains(&transformed_point) {
            cache.insert(*node_id, None);
            return false;
        }

        let point_in_clips = transformed_point - node.local_clip_rect.origin.to_vector();
        for &(ref clip, _) in clip_store.get(&clip_info.clip_sources).clips() {
            if !clip.contains(&point_in_clips) {
                cache.insert(*node_id, None);
                return false;
            }
        }

        cache.insert(*node_id, Some(point_in_layer));

        true
    }

    pub fn get_scroll_node_state(&self) -> Vec<ScrollLayerState> {
        let mut result = vec![];
        for (id, node) in self.nodes.iter() {
            if let NodeType::ScrollFrame(scrolling) = node.node_type {
                result.push(ScrollLayerState {
                    id: *id,
                    scroll_offset: scrolling.offset,
                })
            }
        }
        result
    }

    pub fn drain(&mut self) -> ScrollStates {
        self.current_new_node_item = 1;

        let mut scroll_states = FastHashMap::default();
        for (layer_id, old_node) in &mut self.nodes.drain() {
            if self.pipelines_to_discard.contains(&layer_id.pipeline_id()) {
                continue;
            }

            if let NodeType::ScrollFrame(scrolling) = old_node.node_type {
                scroll_states.insert(layer_id, scrolling);
            }
        }

        self.pipelines_to_discard.clear();
        scroll_states
    }

    pub fn scroll_node(&mut self, origin: LayerPoint, id: ClipId, clamp: ScrollClamping) -> bool {
        if self.nodes.is_empty() {
            self.pending_scroll_offsets.insert(id, (origin, clamp));
            return false;
        }

        if let Some(node) = self.nodes.get_mut(&id) {
            return node.set_scroll_origin(&origin, clamp);
        }

        self.pending_scroll_offsets.insert(id, (origin, clamp));
        false
    }

    pub fn scroll(
        &mut self,
        scroll_location: ScrollLocation,
        cursor: WorldPoint,
        phase: ScrollEventPhase,
    ) -> bool {
        if self.nodes.is_empty() {
            return false;
        }

        let clip_id = match (
            phase,
            self.find_scrolling_node_at_point(&cursor),
            self.currently_scrolling_node_id,
        ) {
            (ScrollEventPhase::Start, scroll_node_at_point_id, _) => {
                self.currently_scrolling_node_id = Some(scroll_node_at_point_id);
                scroll_node_at_point_id
            }
            (_, scroll_node_at_point_id, Some(cached_clip_id)) => {
                let clip_id = match self.nodes.get(&cached_clip_id) {
                    Some(_) => cached_clip_id,
                    None => {
                        self.currently_scrolling_node_id = Some(scroll_node_at_point_id);
                        scroll_node_at_point_id
                    }
                };
                clip_id
            }
            (_, _, None) => return false,
        };

        let topmost_scrolling_node_id = self.topmost_scrolling_node_id();
        let non_root_overscroll = if clip_id != topmost_scrolling_node_id {
            self.nodes.get(&clip_id).unwrap().is_overscrolling()
        } else {
            false
        };

        let mut switch_node = false;
        if let Some(node) = self.nodes.get_mut(&clip_id) {
            if let NodeType::ScrollFrame(ref mut scrolling) = node.node_type {
                match phase {
                    ScrollEventPhase::Start => {
                        // if this is a new gesture, we do not switch node,
                        // however we do save the state of non_root_overscroll,
                        // for use in the subsequent Move phase.
                        scrolling.should_handoff_scroll = non_root_overscroll;
                    }
                    ScrollEventPhase::Move(_) => {
                        // Switch node if movement originated in a new gesture,
                        // from a non root node in overscroll.
                        switch_node = scrolling.should_handoff_scroll && non_root_overscroll
                    }
                    ScrollEventPhase::End => {
                        // clean-up when gesture ends.
                        scrolling.should_handoff_scroll = false;
                    }
                }
            }
        }

        let clip_id = if switch_node {
            topmost_scrolling_node_id
        } else {
            clip_id
        };

        self.nodes
            .get_mut(&clip_id)
            .unwrap()
            .scroll(scroll_location, phase)
    }

    pub fn update_all_node_transforms(
        &mut self,
        screen_rect: &DeviceIntRect,
        device_pixel_ratio: f32,
        clip_store: &mut ClipStore,
        resource_cache: &mut ResourceCache,
        gpu_cache: &mut GpuCache,
        pan: LayerPoint,
        node_data: &mut Vec<ClipScrollNodeData>,
    ) {
        if self.nodes.is_empty() {
            return;
        }

        let root_reference_frame_id = self.root_reference_frame_id();
        let root_viewport = self.nodes[&root_reference_frame_id].local_clip_rect;

        let mut state = TransformUpdateState {
            parent_reference_frame_transform: LayerToWorldTransform::create_translation(
                pan.x,
                pan.y,
                0.0,
            ),
            parent_combined_viewport_rect: root_viewport,
            parent_accumulated_scroll_offset: LayerVector2D::zero(),
            nearest_scrolling_ancestor_offset: LayerVector2D::zero(),
            nearest_scrolling_ancestor_viewport: LayerRect::zero(),
            parent_clip_chain: None,
            combined_outer_clip_bounds: *screen_rect,
            current_coordinate_system_id: CoordinateSystemId(0),
            next_coordinate_system_id: CoordinateSystemId(0).next(),
        };
        self.update_node_transform(
            root_reference_frame_id,
            &mut state,
            device_pixel_ratio,
            clip_store,
            resource_cache,
            gpu_cache,
            node_data,
        );
    }

    fn update_node_transform(
        &mut self,
        layer_id: ClipId,
        state: &mut TransformUpdateState,
        device_pixel_ratio: f32,
        clip_store: &mut ClipStore,
        resource_cache: &mut ResourceCache,
        gpu_cache: &mut GpuCache,
        node_data: &mut Vec<ClipScrollNodeData>,
    ) {
        // TODO(gw): This is an ugly borrow check workaround to clone these.
        //           Restructure this to avoid the clones!
        let mut state = state.clone();
        let node_children = {
            let node = match self.nodes.get_mut(&layer_id) {
                Some(node) => node,
                None => return,
            };

            node.update_transform(
                &mut state,
                node_data
            );
            node.update_clip_work_item(
                &mut state,
                device_pixel_ratio,
                clip_store,
                resource_cache,
                gpu_cache,
            );

            node.children.clone()
        };

        for child_layer_id in node_children {
            self.update_node_transform(
                child_layer_id,
                &mut state,
                device_pixel_ratio,
                clip_store,
                resource_cache,
                gpu_cache,
                node_data,
            );
        }
    }

    pub fn tick_scrolling_bounce_animations(&mut self) {
        for (_, node) in &mut self.nodes {
            node.tick_scrolling_bounce_animation()
        }
    }

    pub fn finalize_and_apply_pending_scroll_offsets(&mut self, old_states: ScrollStates) {
        // TODO(gw): These are all independent - can be run through thread pool if it shows up
        // in the profile!
        for (clip_id, node) in &mut self.nodes {
            if let Some(scrolling_state) = old_states.get(clip_id) {
                node.apply_old_scrolling_state(scrolling_state);
            }

            if let Some((pending_offset, clamping)) = self.pending_scroll_offsets.remove(clip_id) {
                node.set_scroll_origin(&pending_offset, clamping);
            }
        }
    }

    pub fn generate_new_clip_id(&mut self, pipeline_id: PipelineId) -> ClipId {
        let new_id = ClipId::DynamicallyAddedNode(self.current_new_node_item, pipeline_id);
        self.current_new_node_item += 1;
        new_id
    }

    pub fn add_reference_frame(
        &mut self,
        rect: &LayerRect,
        transform: &LayerToScrollTransform,
        origin_in_parent_reference_frame: LayerVector2D,
        pipeline_id: PipelineId,
        parent_id: Option<ClipId>,
        root_for_pipeline: bool,
    ) -> ClipId {
        let reference_frame_id = if root_for_pipeline {
            ClipId::root_reference_frame(pipeline_id)
        } else {
            self.generate_new_clip_id(pipeline_id)
        };

        let node = ClipScrollNode::new_reference_frame(
            parent_id,
            rect,
            transform,
            origin_in_parent_reference_frame,
            pipeline_id,
        );
        self.add_node(node, reference_frame_id);
        reference_frame_id
    }

    pub fn add_sticky_frame(
        &mut self,
        id: ClipId,
        parent_id: ClipId,
        frame_rect: LayerRect,
        sticky_frame_info: StickyFrameInfo,
    ) {
        let node = ClipScrollNode::new_sticky_frame(
            parent_id,
            frame_rect,
            sticky_frame_info,
            id.pipeline_id(),
        );
        self.add_node(node, id);
    }

    pub fn add_node(&mut self, node: ClipScrollNode, id: ClipId) {
        // When the parent node is None this means we are adding the root.
        match node.parent {
            Some(parent_id) => self.nodes.get_mut(&parent_id).unwrap().add_child(id),
            None => self.root_reference_frame_id = id,
        }

        debug_assert!(!self.nodes.contains_key(&id));
        self.nodes.insert(id, node);
    }

    pub fn discard_frame_state_for_pipeline(&mut self, pipeline_id: PipelineId) {
        self.pipelines_to_discard.insert(pipeline_id);

        match self.currently_scrolling_node_id {
            Some(id) if id.pipeline_id() == pipeline_id => self.currently_scrolling_node_id = None,
            _ => {}
        }
    }

    fn print_node<T: PrintTreePrinter>(&self, id: &ClipId, pt: &mut T, clip_store: &ClipStore) {
        let node = self.nodes.get(id).unwrap();

        match node.node_type {
            NodeType::Clip(ref info) => {
                pt.new_level("Clip".to_owned());

                pt.add_item(format!("id: {:?}", id));
                let clips = clip_store.get(&info.clip_sources).clips();
                pt.new_level(format!("Clip Sources [{}]", clips.len()));
                for source in clips {
                    pt.add_item(format!("{:?}", source));
                }
                pt.end_level();
            }
            NodeType::ReferenceFrame(ref info) => {
                pt.new_level(format!("ReferenceFrame {:?}", info.transform));
                pt.add_item(format!("id: {:?}", id));
            }
            NodeType::ScrollFrame(scrolling_info) => {
                pt.new_level(format!("ScrollFrame"));
                pt.add_item(format!("id: {:?}", id));
                pt.add_item(format!("scrollable_size: {:?}", scrolling_info.scrollable_size));
                pt.add_item(format!("scroll.offset: {:?}", scrolling_info.offset));
            }
            NodeType::StickyFrame(ref sticky_frame_info) => {
                pt.new_level(format!("StickyFrame"));
                pt.add_item(format!("id: {:?}", id));
                pt.add_item(format!("sticky info: {:?}", sticky_frame_info));
            }
        }

        pt.add_item(format!(
            "local_viewport_rect: {:?}",
            node.local_viewport_rect
        ));
        pt.add_item(format!("local_clip_rect: {:?}", node.local_clip_rect));
        pt.add_item(format!(
            "combined_local_viewport_rect: {:?}",
            node.combined_local_viewport_rect
        ));
        pt.add_item(format!(
            "world_viewport_transform: {:?}",
            node.world_viewport_transform
        ));
        pt.add_item(format!(
            "world_content_transform: {:?}",
            node.world_content_transform
        ));

        for child_id in &node.children {
            self.print_node(child_id, pt, clip_store);
        }

        pt.end_level();
    }

    #[allow(dead_code)]
    pub fn print(&self, clip_store: &ClipStore) {
        if !self.nodes.is_empty() {
            let mut pt = PrintTree::new("clip_scroll tree");
            self.print_with(clip_store, &mut pt);
        }
    }

    pub fn print_with<T: PrintTreePrinter>(&self, clip_store: &ClipStore, pt: &mut T) {
        if !self.nodes.is_empty() {
            self.print_node(&self.root_reference_frame_id, pt, clip_store);
        }
    }

    pub fn make_node_relative_point_absolute(
        &self,
        pipeline_id: Option<PipelineId>,
        point: &LayerPoint
    ) -> WorldPoint {
        pipeline_id.and_then(|id| self.nodes.get(&ClipId::root_reference_frame(id)))
                   .map(|node| node.world_viewport_transform.transform_point2d(point))
                   .unwrap_or_else(|| WorldPoint::new(point.x, point.y))

    }
}
