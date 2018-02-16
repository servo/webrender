
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{BuiltDisplayListIter, ClipAndScrollInfo, ClipId, ColorF, ComplexClipRegion};
use api::{DeviceUintSize, DisplayItemRef, Epoch, ExternalScrollId, FilterOp, IframeDisplayItem};
use api::{ImageDisplayItem, ItemRange, LayerPoint, LayerPrimitiveInfo, LayerRect, LayerSize};
use api::{LayerVector2D, LayoutSize, PipelineId, ScrollFrameDisplayItem, ScrollPolicy};
use api::{ScrollSensitivity, SpecificDisplayItem, StackingContext, StickyFrameDisplayItem};
use api::{TileOffset, TransformStyle};
use clip::ClipRegion;
use clip_scroll_node::StickyFrameInfo;
use clip_scroll_tree::{ClipChainIndex, ClipScrollNodeIndex, ClipScrollTree};
use euclid::rect;
use frame_builder::{FrameBuilder, FrameBuilderConfig, ScrollbarInfo};
use internal_types::{FastHashMap, FastHashSet};
use prim_store::ScrollNodeAndClipChain;
use resource_cache::{FontInstanceMap, TiledImageMap};
use scene::{Scene, ScenePipeline, StackingContextHelpers};
use scene_builder::{SceneRequest, BuiltScene};
use tiling::{CompositeOps};
use render_backend::DocumentView;

static DEFAULT_SCROLLBAR_COLOR: ColorF = ColorF {
    r: 0.3,
    g: 0.3,
    b: 0.3,
    a: 0.6,
};

/// A data structure that keeps track of mapping between API ClipIds and the indices used
/// internally in the ClipScrollTree to avoid having to do HashMap lookups. ClipIdToIndexMapper is
/// responsible for mapping both ClipId to ClipChainIndex and ClipId to ClipScrollNodeIndex.  We
/// also include two small LRU caches. Currently the caches are small (1 entry), but in the future
/// we could use uluru here to do something more involved.
#[derive(Default)]
pub struct ClipIdToIndexMapper {
    /// A map which converts a ClipId for a clipping node or an API-defined ClipChain into
    /// a ClipChainIndex, which is the index used internally in the ClipScrollTree to
    /// identify ClipChains.
    clip_chain_map: FastHashMap<ClipId, ClipChainIndex>,

    /// The last mapped ClipChainIndex, used to avoid having to do lots of consecutive
    /// HashMap lookups.
    cached_clip_chain_index: Option<(ClipId, ClipChainIndex)>,

    /// The offset in the ClipScrollTree's array of ClipScrollNodes for a particular pipeline.
    /// This is used to convert a ClipId into a ClipScrollNodeIndex.
    pipeline_offsets: FastHashMap<PipelineId, usize>,

    /// The last mapped pipeline offset for this mapper. This is used to avoid having to
    /// consult `pipeline_offsets` repeatedly when flattening the display list.
    cached_pipeline_offset: Option<(PipelineId, usize)>,

    /// The next available pipeline offset for ClipScrollNodeIndex. When we encounter a pipeline
    /// we will use this value and increment it by the total number of ClipScrollNodes in the
    /// pipeline's display list.
    next_available_offset: usize,
}

impl ClipIdToIndexMapper {
    pub fn add_clip_chain(&mut self, id: ClipId, index: ClipChainIndex) {
        debug_assert!(!self.clip_chain_map.contains_key(&id));
        self.clip_chain_map.insert(id, index);
    }

    pub fn map_to_parent_clip_chain(&mut self, id: ClipId, parent_id: &ClipId) {
        let parent_chain_index = self.get_clip_chain_index(parent_id);
        self.add_clip_chain(id, parent_chain_index);
    }

    pub fn get_clip_chain_index(&mut self, id: &ClipId) -> ClipChainIndex {
        match self.cached_clip_chain_index {
            Some((cached_id, cached_clip_chain_index)) if cached_id == *id =>
                return cached_clip_chain_index,
            _ => {}
        }

        self.clip_chain_map[id]
    }

    pub fn get_clip_chain_index_and_cache_result(&mut self, id: &ClipId) -> ClipChainIndex {
        let index = self.get_clip_chain_index(id);
        self.cached_clip_chain_index = Some((*id, index));
        index
    }

    pub fn map_clip_and_scroll(&mut self, info: &ClipAndScrollInfo) -> ScrollNodeAndClipChain {
        ScrollNodeAndClipChain::new(
            self.get_node_index(info.scroll_node_id),
            self.get_clip_chain_index_and_cache_result(&info.clip_node_id())
        )
    }

    pub fn simple_scroll_and_clip_chain(&mut self, id: &ClipId) -> ScrollNodeAndClipChain {
        self.map_clip_and_scroll(&ClipAndScrollInfo::simple(*id))
    }

    pub fn initialize_for_pipeline(&mut self, pipeline: &ScenePipeline) {
        debug_assert!(!self.pipeline_offsets.contains_key(&pipeline.pipeline_id));
        self.pipeline_offsets.insert(pipeline.pipeline_id, self.next_available_offset);
        self.next_available_offset += pipeline.display_list.total_clip_ids();
    }

    pub fn get_node_index(&mut self, id: ClipId) -> ClipScrollNodeIndex {
        let (index, pipeline_id) = match id {
            ClipId::Clip(index, pipeline_id) => (index, pipeline_id),
            ClipId::ClipChain(_) => panic!("Tried to use ClipChain as scroll node."),
        };

        let pipeline_offset = match self.cached_pipeline_offset {
            Some((last_used_id, offset)) if last_used_id == pipeline_id => offset,
            _ => {
                let offset = self.pipeline_offsets[&pipeline_id];
                self.cached_pipeline_offset = Some((pipeline_id, offset));
                offset
            }
        };

        ClipScrollNodeIndex(pipeline_offset + index)
    }
}

pub struct FlattenContext<'a> {
    scene: &'a Scene,
    builder: FrameBuilder,
    clip_scroll_tree: &'a mut ClipScrollTree,
    font_instances: FontInstanceMap,
    tiled_image_map: TiledImageMap,
    pipeline_epochs: Vec<(PipelineId, Epoch)>,
    replacements: Vec<(ClipId, ClipId)>,
    output_pipelines: &'a FastHashSet<PipelineId>,
    id_to_index_mapper: ClipIdToIndexMapper,
}

impl<'a> FlattenContext<'a> {
    // When changing this, please make the same modification to build_scene,
    // which will soon replace this method completely.
    pub fn create_frame_builder(
        old_builder: FrameBuilder,
        scene: &Scene,
        clip_scroll_tree: &mut ClipScrollTree,
        font_instances: FontInstanceMap,
        tiled_image_map: TiledImageMap,
        view: &DocumentView,
        output_pipelines: &FastHashSet<PipelineId>,
        frame_builder_config: &FrameBuilderConfig,
        pipeline_epochs: &mut FastHashMap<PipelineId, Epoch>,
    ) -> FrameBuilder {
        // We checked that the root pipeline is available on the render backend.
        let root_pipeline_id = scene.root_pipeline_id.unwrap();
        let root_pipeline = scene.pipelines.get(&root_pipeline_id).unwrap();

        let root_epoch = scene.pipeline_epochs[&root_pipeline_id];
        pipeline_epochs.insert(root_pipeline_id, root_epoch);

        let background_color = root_pipeline
            .background_color
            .and_then(|color| if color.a > 0.0 { Some(color) } else { None });

        let frame_builder = {
            let mut roller = FlattenContext {
                scene,
                builder: old_builder.recycle(
                    view.inner_rect,
                    background_color,
                    view.window_size,
                    *frame_builder_config,
                ),
                clip_scroll_tree,
                font_instances,
                tiled_image_map,
                pipeline_epochs: Vec::new(),
                replacements: Vec::new(),
                output_pipelines,
                id_to_index_mapper: ClipIdToIndexMapper::default(),
            };

            roller.id_to_index_mapper.initialize_for_pipeline(root_pipeline);

            roller.builder.push_root(
                root_pipeline_id,
                &root_pipeline.viewport_size,
                &root_pipeline.content_size,
                roller.clip_scroll_tree,
                &mut roller.id_to_index_mapper,
            );

            roller.builder.setup_viewport_offset(
                view.inner_rect,
                view.accumulated_scale_factor(),
                roller.clip_scroll_tree,
            );

            roller.flatten_root(root_pipeline, &root_pipeline.viewport_size);

            debug_assert!(roller.builder.picture_stack.is_empty());

            pipeline_epochs.extend(roller.pipeline_epochs.drain(..));

            roller.builder
        };

        frame_builder
    }

    /// Since WebRender still handles fixed position and reference frame content internally
    /// we need to apply this table of id replacements only to the id that affects the
    /// position of a node. We can eventually remove this when clients start handling
    /// reference frames themselves. This method applies these replacements.
    fn apply_scroll_frame_id_replacement(&self, index: ClipId) -> ClipId {
        match self.replacements.last() {
            Some(&(to_replace, replacement)) if to_replace == index => replacement,
            _ => index,
        }
    }

    fn get_complex_clips(
        &self,
        pipeline_id: PipelineId,
        complex_clips: ItemRange<ComplexClipRegion>,
    ) -> Vec<ComplexClipRegion> {
        if complex_clips.is_empty() {
            return vec![];
        }

        self.scene
            .pipelines
            .get(&pipeline_id)
            .expect("No display list?")
            .display_list
            .get(complex_clips)
            .collect()
    }

    fn get_clip_chain_items(
        &self,
        pipeline_id: PipelineId,
        items: ItemRange<ClipId>,
    ) -> Vec<ClipId> {
        if items.is_empty() {
            return vec![];
        }

        self.scene
            .pipelines
            .get(&pipeline_id)
            .expect("No display list?")
            .display_list
            .get(items)
            .collect()
    }

    fn flatten_root(&mut self, pipeline: &'a ScenePipeline, frame_size: &LayoutSize) {
        let pipeline_id = pipeline.pipeline_id;
        let reference_frame_info = self.id_to_index_mapper.simple_scroll_and_clip_chain(
            &ClipId::root_reference_frame(pipeline_id)
        );
        let scroll_frame_info = self.id_to_index_mapper.simple_scroll_and_clip_chain(
            &ClipId::root_scroll_node(pipeline_id)
        );

        self.builder.push_stacking_context(
            pipeline_id,
            CompositeOps::default(),
            TransformStyle::Flat,
            true,
            true,
            scroll_frame_info,
            self.output_pipelines,
        );

        // For the root pipeline, there's no need to add a full screen rectangle
        // here, as it's handled by the framebuffer clear.
        if self.scene.root_pipeline_id != Some(pipeline_id) {
            if let Some(pipeline) = self.scene.pipelines.get(&pipeline_id) {
                if let Some(bg_color) = pipeline.background_color {
                    let root_bounds = LayerRect::new(LayerPoint::zero(), *frame_size);
                    let info = LayerPrimitiveInfo::new(root_bounds);
                    self.builder.add_solid_rectangle(
                        reference_frame_info,
                        &info,
                        bg_color,
                        None,
                    );
                }
            }
        }

        self.flatten_items(&mut pipeline.display_list.iter(), pipeline_id, LayerVector2D::zero());

        if self.builder.config.enable_scrollbars {
            let scrollbar_rect = LayerRect::new(LayerPoint::zero(), LayerSize::new(10.0, 70.0));
            let container_rect = LayerRect::new(LayerPoint::zero(), *frame_size);
            self.builder.add_scroll_bar(
                reference_frame_info,
                &LayerPrimitiveInfo::new(scrollbar_rect),
                DEFAULT_SCROLLBAR_COLOR,
                ScrollbarInfo(scroll_frame_info.scroll_node_id, container_rect),
            );
        }

        self.builder.pop_stacking_context();
    }

    fn flatten_items(
        &mut self,
        traversal: &mut BuiltDisplayListIter<'a>,
        pipeline_id: PipelineId,
        reference_frame_relative_offset: LayerVector2D,
    ) {
        loop {
            let subtraversal = {
                let item = match traversal.next() {
                    Some(item) => item,
                    None => break,
                };

                if SpecificDisplayItem::PopStackingContext == *item.item() {
                    return;
                }

                self.flatten_item(
                    item,
                    pipeline_id,
                    reference_frame_relative_offset,
                )
            };

            // If flatten_item created a sub-traversal, we need `traversal` to have the
            // same state as the completed subtraversal, so we reinitialize it here.
            if let Some(subtraversal) = subtraversal {
                *traversal = subtraversal;
            }
        }
    }

    fn flatten_sticky_frame(
        &mut self,
        item: &DisplayItemRef,
        info: &StickyFrameDisplayItem,
        clip_and_scroll: &ScrollNodeAndClipChain,
        parent_id: &ClipId,
        reference_frame_relative_offset: &LayerVector2D,
    ) {
        let frame_rect = item.rect().translate(&reference_frame_relative_offset);
        let sticky_frame_info = StickyFrameInfo::new(
            info.margins,
            info.vertical_offset_bounds,
            info.horizontal_offset_bounds,
            info.previously_applied_offset,
        );

        let index = self.id_to_index_mapper.get_node_index(info.id);
        self.clip_scroll_tree.add_sticky_frame(
            index,
            clip_and_scroll.scroll_node_id, /* parent id */
            frame_rect,
            sticky_frame_info,
            info.id.pipeline_id(),
        );
        self.id_to_index_mapper.map_to_parent_clip_chain(info.id, &parent_id);
    }

    fn flatten_scroll_frame(
        &mut self,
        item: &DisplayItemRef,
        info: &ScrollFrameDisplayItem,
        pipeline_id: PipelineId,
        clip_and_scroll_ids: &ClipAndScrollInfo,
        reference_frame_relative_offset: &LayerVector2D,
    ) {
        let complex_clips = self.get_complex_clips(pipeline_id, item.complex_clip().0);
        let clip_region = ClipRegion::create_for_clip_node(
            *item.local_clip().clip_rect(),
            complex_clips,
            info.image_mask,
            &reference_frame_relative_offset,
        );
        // Just use clip rectangle as the frame rect for this scroll frame.
        // This is useful when calculating scroll extents for the
        // ClipScrollNode::scroll(..) API as well as for properly setting sticky
        // positioning offsets.
        let frame_rect = item.local_clip()
            .clip_rect()
            .translate(&reference_frame_relative_offset);
        let content_rect = item.rect().translate(&reference_frame_relative_offset);

        debug_assert!(info.clip_id != info.scroll_frame_id);

        self.builder.add_clip_node(
            info.clip_id,
            clip_and_scroll_ids.scroll_node_id,
            clip_region,
            self.clip_scroll_tree,
            &mut self.id_to_index_mapper,
        );

        self.builder.add_scroll_frame(
            info.scroll_frame_id,
            info.clip_id,
            info.external_id,
            pipeline_id,
            &frame_rect,
            &content_rect.size,
            info.scroll_sensitivity,
            self.clip_scroll_tree,
            &mut self.id_to_index_mapper,
        );
    }

    fn flatten_stacking_context(
        &mut self,
        traversal: &mut BuiltDisplayListIter<'a>,
        pipeline_id: PipelineId,
        unreplaced_scroll_id: ClipId,
        scroll_node_id: ClipId,
        mut reference_frame_relative_offset: LayerVector2D,
        bounds: &LayerRect,
        stacking_context: &StackingContext,
        filters: ItemRange<FilterOp>,
        is_backface_visible: bool,
    ) {
        // Avoid doing unnecessary work for empty stacking contexts.
        if traversal.current_stacking_context_empty() {
            traversal.skip_current_stacking_context();
            return;
        }

        let composition_operations = {
            // TODO(optimization?): self.traversal.display_list()
            let display_list = &self
                .scene
                .pipelines
                .get(&pipeline_id)
                .expect("No display list?!")
                .display_list;
            CompositeOps::new(
                stacking_context.filter_ops_for_compositing(display_list, filters),
                stacking_context.mix_blend_mode_for_compositing(),
            )
        };

        if stacking_context.scroll_policy == ScrollPolicy::Fixed {
            self.replacements.push((
                unreplaced_scroll_id,
                self.builder.current_reference_frame_id(),
            ));
        }

        reference_frame_relative_offset += bounds.origin.to_vector();

        // If we have a transformation or a perspective, we should have been assigned a new
        // reference frame id. This means this stacking context establishes a new reference frame.
        // Descendant fixed position content will be positioned relative to us.
        if let Some(reference_frame_id) = stacking_context.reference_frame_id {
            debug_assert!(
                stacking_context.transform.is_some() ||
                stacking_context.perspective.is_some()
            );

            let reference_frame_bounds = LayerRect::new(LayerPoint::zero(), bounds.size);
            self.builder.push_reference_frame(
                reference_frame_id,
                Some(scroll_node_id),
                pipeline_id,
                &reference_frame_bounds,
                stacking_context.transform,
                stacking_context.perspective,
                reference_frame_relative_offset,
                self.clip_scroll_tree,
                &mut self.id_to_index_mapper,
            );
            self.replacements.push((unreplaced_scroll_id, reference_frame_id));
            reference_frame_relative_offset = LayerVector2D::zero();
        }

        // We apply the replacements one more time in case we need to set it to a replacement
        // that we just pushed above.
        let sc_scroll_node = self.apply_scroll_frame_id_replacement(unreplaced_scroll_id);
        let stacking_context_clip_and_scroll =
            self.id_to_index_mapper.simple_scroll_and_clip_chain(&sc_scroll_node);
        self.builder.push_stacking_context(
            pipeline_id,
            composition_operations,
            stacking_context.transform_style,
            is_backface_visible,
            false,
            stacking_context_clip_and_scroll,
            self.output_pipelines,
        );

        self.flatten_items(
            traversal,
            pipeline_id,
            reference_frame_relative_offset,
        );

        if stacking_context.scroll_policy == ScrollPolicy::Fixed {
            self.replacements.pop();
        }

        if stacking_context.reference_frame_id.is_some() {
            self.replacements.pop();
            self.builder.pop_reference_frame();
        }

        self.builder.pop_stacking_context();
    }

    fn flatten_iframe(
        &mut self,
        item: &DisplayItemRef,
        info: &IframeDisplayItem,
        clip_and_scroll_ids: &ClipAndScrollInfo,
        reference_frame_relative_offset: &LayerVector2D,
    ) {
        let iframe_pipeline_id = info.pipeline_id;
        let pipeline = match self.scene.pipelines.get(&iframe_pipeline_id) {
            Some(pipeline) => pipeline,
            None => return,
        };

        self.id_to_index_mapper.initialize_for_pipeline(pipeline);

        self.builder.add_clip_node(
            info.clip_id,
            clip_and_scroll_ids.scroll_node_id,
            ClipRegion::create_for_clip_node_with_local_clip(
                &item.local_clip(),
                &reference_frame_relative_offset
            ),
            self.clip_scroll_tree,
            &mut self.id_to_index_mapper,
        );

        let epoch = self.scene.pipeline_epochs[&iframe_pipeline_id];
        self.pipeline_epochs.push((iframe_pipeline_id, epoch));

        let bounds = item.rect();
        let iframe_rect = LayerRect::new(LayerPoint::zero(), bounds.size);
        let origin = *reference_frame_relative_offset + bounds.origin.to_vector();
        self.builder.push_reference_frame(
            ClipId::root_reference_frame(iframe_pipeline_id),
            Some(info.clip_id),
            iframe_pipeline_id,
            &iframe_rect,
            None,
            None,
            origin,
            self.clip_scroll_tree,
            &mut self.id_to_index_mapper,
        );

        self.builder.add_scroll_frame(
            ClipId::root_scroll_node(iframe_pipeline_id),
            ClipId::root_reference_frame(iframe_pipeline_id),
            Some(ExternalScrollId(0, iframe_pipeline_id)),
            iframe_pipeline_id,
            &iframe_rect,
            &pipeline.content_size,
            ScrollSensitivity::ScriptAndInputEvents,
            self.clip_scroll_tree,
            &mut self.id_to_index_mapper,
        );

        self.flatten_root(pipeline, &iframe_rect.size);

        self.builder.pop_reference_frame();
    }

    fn flatten_item<'b>(
        &'b mut self,
        item: DisplayItemRef<'a, 'b>,
        pipeline_id: PipelineId,
        reference_frame_relative_offset: LayerVector2D,
    ) -> Option<BuiltDisplayListIter<'a>> {
        let mut clip_and_scroll_ids = item.clip_and_scroll();
        let unreplaced_scroll_id = clip_and_scroll_ids.scroll_node_id;
        clip_and_scroll_ids.scroll_node_id =
            self.apply_scroll_frame_id_replacement(clip_and_scroll_ids.scroll_node_id);
        let clip_and_scroll = self.id_to_index_mapper.map_clip_and_scroll(&clip_and_scroll_ids);

        let prim_info = item.get_layer_primitive_info(&reference_frame_relative_offset);
        match *item.item() {
            SpecificDisplayItem::Image(ref info) => {
                match self.tiled_image_map.get(&info.image_key).cloned() {
                    Some(tiling) => {
                        // The image resource is tiled. We have to generate an image primitive
                        // for each tile.
                        self.decompose_image(
                            clip_and_scroll,
                            &prim_info,
                            info,
                            tiling.image_size,
                            tiling.tile_size as u32,
                        );
                    }
                    None => {
                        self.builder.add_image(
                            clip_and_scroll,
                            &prim_info,
                            info.stretch_size,
                            info.tile_spacing,
                            None,
                            info.image_key,
                            info.image_rendering,
                            info.alpha_type,
                            None,
                        );
                    }
                }
            }
            SpecificDisplayItem::YuvImage(ref info) => {
                self.builder.add_yuv_image(
                    clip_and_scroll,
                    &prim_info,
                    info.yuv_data,
                    info.color_space,
                    info.image_rendering,
                );
            }
            SpecificDisplayItem::Text(ref text_info) => {
                let instance_map = self.font_instances
                    .read()
                    .unwrap();
                match instance_map.get(&text_info.font_key) {
                    Some(instance) => {
                        self.builder.add_text(
                            clip_and_scroll,
                            reference_frame_relative_offset,
                            &prim_info,
                            instance,
                            &text_info.color,
                            item.glyphs(),
                            item.display_list().get(item.glyphs()).count(),
                            text_info.glyph_options,
                        );
                    }
                    None => {
                        warn!("Unknown font instance key");
                        debug!("key={:?}", text_info.font_key);
                    }
                }
            }
            SpecificDisplayItem::Rectangle(ref info) => {
                self.builder.add_solid_rectangle(
                    clip_and_scroll,
                    &prim_info,
                    info.color,
                    None,
                );
            }
            SpecificDisplayItem::ClearRectangle => {
                self.builder.add_clear_rectangle(
                    clip_and_scroll,
                    &prim_info,
                );
            }
            SpecificDisplayItem::Line(ref info) => {
                self.builder.add_line(
                    clip_and_scroll,
                    &prim_info,
                    info.wavy_line_thickness,
                    info.orientation,
                    &info.color,
                    info.style,
                );
            }
            SpecificDisplayItem::Gradient(ref info) => {
                self.builder.add_gradient(
                    clip_and_scroll,
                    &prim_info,
                    info.gradient.start_point,
                    info.gradient.end_point,
                    item.gradient_stops(),
                    item.display_list().get(item.gradient_stops()).count(),
                    info.gradient.extend_mode,
                    info.tile_size,
                    info.tile_spacing,
                );
            }
            SpecificDisplayItem::RadialGradient(ref info) => {
                self.builder.add_radial_gradient(
                    clip_and_scroll,
                    &prim_info,
                    info.gradient.start_center,
                    info.gradient.start_radius,
                    info.gradient.end_center,
                    info.gradient.end_radius,
                    info.gradient.ratio_xy,
                    item.gradient_stops(),
                    info.gradient.extend_mode,
                    info.tile_size,
                    info.tile_spacing,
                );
            }
            SpecificDisplayItem::BoxShadow(ref box_shadow_info) => {
                let bounds = box_shadow_info
                    .box_bounds
                    .translate(&reference_frame_relative_offset);
                let mut prim_info = prim_info.clone();
                prim_info.rect = bounds;
                self.builder.add_box_shadow(
                    pipeline_id,
                    clip_and_scroll,
                    &prim_info,
                    &box_shadow_info.offset,
                    &box_shadow_info.color,
                    box_shadow_info.blur_radius,
                    box_shadow_info.spread_radius,
                    box_shadow_info.border_radius,
                    box_shadow_info.clip_mode,
                );
            }
            SpecificDisplayItem::Border(ref info) => {
                self.builder.add_border(
                    clip_and_scroll,
                    &prim_info,
                    info,
                    item.gradient_stops(),
                    item.display_list().get(item.gradient_stops()).count(),
                );
            }
            SpecificDisplayItem::PushStackingContext(ref info) => {
                let mut subtraversal = item.sub_iter();
                self.flatten_stacking_context(
                    &mut subtraversal,
                    pipeline_id,
                    unreplaced_scroll_id,
                    clip_and_scroll_ids.scroll_node_id,
                    reference_frame_relative_offset,
                    &item.rect(),
                    &info.stacking_context,
                    item.filters(),
                    prim_info.is_backface_visible,
                );
                return Some(subtraversal);
            }
            SpecificDisplayItem::Iframe(ref info) => {
                self.flatten_iframe(
                    &item,
                    info,
                    &clip_and_scroll_ids,
                    &reference_frame_relative_offset
                );
            }
            SpecificDisplayItem::Clip(ref info) => {
                let complex_clips = self.get_complex_clips(pipeline_id, item.complex_clip().0);
                let clip_region = ClipRegion::create_for_clip_node(
                    *item.local_clip().clip_rect(),
                    complex_clips,
                    info.image_mask,
                    &reference_frame_relative_offset,
                );
                self.builder.add_clip_node(
                    info.id,
                    clip_and_scroll_ids.scroll_node_id,
                    clip_region,
                    &mut self.clip_scroll_tree,
                    &mut self.id_to_index_mapper,
                );
            }
            SpecificDisplayItem::ClipChain(ref info) => {
                let items = self.get_clip_chain_items(pipeline_id, item.clip_chain_items())
                                .iter()
                                .map(|id| self.id_to_index_mapper.get_node_index(*id))
                                .collect();
                let parent = info.parent.map(|id|
                     self.id_to_index_mapper.get_clip_chain_index(&ClipId::ClipChain(id))
                );
                let clip_chain_index =
                    self.clip_scroll_tree.add_clip_chain_descriptor(parent, items);
                self.id_to_index_mapper.add_clip_chain(ClipId::ClipChain(info.id), clip_chain_index);
            },
            SpecificDisplayItem::ScrollFrame(ref info) => {
                self.flatten_scroll_frame(
                    &item,
                    info,
                    pipeline_id,
                    &clip_and_scroll_ids,
                    &reference_frame_relative_offset
                );
            }
            SpecificDisplayItem::StickyFrame(ref info) => {
                self.flatten_sticky_frame(
                    &item,
                    info,
                    &clip_and_scroll,
                    &clip_and_scroll_ids.scroll_node_id,
                    &reference_frame_relative_offset
                );
            }

            // Do nothing; these are dummy items for the display list parser
            SpecificDisplayItem::SetGradientStops => {}

            SpecificDisplayItem::PopStackingContext => {
                unreachable!("Should have returned in parent method.")
            }
            SpecificDisplayItem::PushShadow(shadow) => {
                let mut prim_info = prim_info.clone();
                prim_info.rect = LayerRect::zero();
                self.builder
                    .push_shadow(shadow, clip_and_scroll, &prim_info);
            }
            SpecificDisplayItem::PopAllShadows => {
                self.builder.pop_all_shadows();
            }
        }
        None
    }

    /// Decomposes an image display item that is repeated into an image per individual repetition.
    /// We need to do this when we are unable to perform the repetition in the shader,
    /// for example if the image is tiled.
    ///
    /// In all of the "decompose" methods below, we independently handle horizontal and vertical
    /// decomposition. This lets us generate the minimum amount of primitives by, for  example,
    /// decompositing the repetition horizontally while repeating vertically in the shader (for
    /// an image where the width is too bug but the height is not).
    ///
    /// decompose_image and decompose_image_row handle image repetitions while decompose_tiled_image
    /// takes care of the decomposition required by the internal tiling of the image.
    fn decompose_image(
        &mut self,
        clip_and_scroll: ScrollNodeAndClipChain,
        prim_info: &LayerPrimitiveInfo,
        info: &ImageDisplayItem,
        image_size: DeviceUintSize,
        tile_size: u32,
    ) {
        let no_vertical_tiling = image_size.height <= tile_size;
        let no_vertical_spacing = info.tile_spacing.height == 0.0;
        let item_rect = prim_info.rect;
        if no_vertical_tiling && no_vertical_spacing {
            self.decompose_image_row(
                clip_and_scroll,
                prim_info,
                info,
                image_size,
                tile_size,
            );
            return;
        }

        // Decompose each vertical repetition into rows.
        let layout_stride = info.stretch_size.height + info.tile_spacing.height;
        let num_repetitions = (item_rect.size.height / layout_stride).ceil() as u32;
        for i in 0 .. num_repetitions {
            if let Some(row_rect) = rect(
                item_rect.origin.x,
                item_rect.origin.y + (i as f32) * layout_stride,
                item_rect.size.width,
                info.stretch_size.height,
            ).intersection(&item_rect)
            {
                let mut prim_info = prim_info.clone();
                prim_info.rect = row_rect;
                self.decompose_image_row(
                    clip_and_scroll,
                    &prim_info,
                    info,
                    image_size,
                    tile_size,
                );
            }
        }
    }

    fn decompose_image_row(
        &mut self,
        clip_and_scroll: ScrollNodeAndClipChain,
        prim_info: &LayerPrimitiveInfo,
        info: &ImageDisplayItem,
        image_size: DeviceUintSize,
        tile_size: u32,
    ) {
        let no_horizontal_tiling = image_size.width <= tile_size;
        let no_horizontal_spacing = info.tile_spacing.width == 0.0;
        if no_horizontal_tiling && no_horizontal_spacing {
            self.decompose_tiled_image(
                clip_and_scroll,
                prim_info,
                info,
                image_size,
                tile_size,
            );
            return;
        }

        // Decompose each horizontal repetition.
        let item_rect = prim_info.rect;
        let layout_stride = info.stretch_size.width + info.tile_spacing.width;
        let num_repetitions = (item_rect.size.width / layout_stride).ceil() as u32;
        for i in 0 .. num_repetitions {
            if let Some(decomposed_rect) = rect(
                item_rect.origin.x + (i as f32) * layout_stride,
                item_rect.origin.y,
                info.stretch_size.width,
                item_rect.size.height,
            ).intersection(&item_rect)
            {
                let mut prim_info = prim_info.clone();
                prim_info.rect = decomposed_rect;
                self.decompose_tiled_image(
                    clip_and_scroll,
                    &prim_info,
                    info,
                    image_size,
                    tile_size,
                );
            }
        }
    }

    fn decompose_tiled_image(
        &mut self,
        clip_and_scroll: ScrollNodeAndClipChain,
        prim_info: &LayerPrimitiveInfo,
        info: &ImageDisplayItem,
        image_size: DeviceUintSize,
        tile_size: u32,
    ) {
        // The image resource is tiled. We have to generate an image primitive
        // for each tile.
        // We need to do this because the image is broken up into smaller tiles in the texture
        // cache and the image shader is not able to work with this type of sparse representation.

        // The tiling logic works as follows:
        //
        //  ###################-+  -+
        //  #    |    |    |//# |   | image size
        //  #    |    |    |//# |   |
        //  #----+----+----+--#-+   |  -+
        //  #    |    |    |//# |   |   | regular tile size
        //  #    |    |    |//# |   |   |
        //  #----+----+----+--#-+   |  -+-+
        //  #////|////|////|//# |   |     | "leftover" height
        //  ################### |  -+  ---+
        //  #----+----+----+----+
        //
        // In the ascii diagram above, a large image is plit into tiles of almost regular size.
        // The tiles on the right and bottom edges (hatched in the diagram) are smaller than
        // the regular tiles and are handled separately in the code see leftover_width/height.
        // each generated image primitive corresponds to a tile in the texture cache, with the
        // assumption that the smaller tiles with leftover sizes are sized to fit their own
        // irregular size in the texture cache.
        //
        // For the case where we don't tile along an axis, we can still perform the repetition in
        // the shader (for this particular axis), and it is worth special-casing for this to avoid
        // generating many primitives.
        // This can happen with very tall and thin images used as a repeating background.
        // Apparently web authors do that...

        let item_rect = prim_info.rect;
        let needs_repeat_x = info.stretch_size.width < item_rect.size.width;
        let needs_repeat_y = info.stretch_size.height < item_rect.size.height;

        let tiled_in_x = image_size.width > tile_size;
        let tiled_in_y = image_size.height > tile_size;

        // If we don't actually tile in this dimension, repeating can be done in the shader.
        let shader_repeat_x = needs_repeat_x && !tiled_in_x;
        let shader_repeat_y = needs_repeat_y && !tiled_in_y;

        let tile_size_f32 = tile_size as f32;

        // Note: this rounds down so it excludes the partially filled tiles on the right and
        // bottom edges (we handle them separately below).
        let num_tiles_x = (image_size.width / tile_size) as u16;
        let num_tiles_y = (image_size.height / tile_size) as u16;

        // Ratio between (image space) tile size and image size.
        let img_dw = tile_size_f32 / (image_size.width as f32);
        let img_dh = tile_size_f32 / (image_size.height as f32);

        // Strected size of the tile in layout space.
        let stretched_tile_size = LayerSize::new(
            img_dw * info.stretch_size.width,
            img_dh * info.stretch_size.height,
        );

        // The size in pixels of the tiles on the right and bottom edges, smaller
        // than the regular tile size if the image is not a multiple of the tile size.
        // Zero means the image size is a multiple of the tile size.
        let leftover =
            DeviceUintSize::new(image_size.width % tile_size, image_size.height % tile_size);

        for ty in 0 .. num_tiles_y {
            for tx in 0 .. num_tiles_x {
                self.add_tile_primitive(
                    clip_and_scroll,
                    prim_info,
                    info,
                    TileOffset::new(tx, ty),
                    stretched_tile_size,
                    1.0,
                    1.0,
                    shader_repeat_x,
                    shader_repeat_y,
                );
            }
            if leftover.width != 0 {
                // Tiles on the right edge that are smaller than the tile size.
                self.add_tile_primitive(
                    clip_and_scroll,
                    prim_info,
                    info,
                    TileOffset::new(num_tiles_x, ty),
                    stretched_tile_size,
                    (leftover.width as f32) / tile_size_f32,
                    1.0,
                    shader_repeat_x,
                    shader_repeat_y,
                );
            }
        }

        if leftover.height != 0 {
            for tx in 0 .. num_tiles_x {
                // Tiles on the bottom edge that are smaller than the tile size.
                self.add_tile_primitive(
                    clip_and_scroll,
                    prim_info,
                    info,
                    TileOffset::new(tx, num_tiles_y),
                    stretched_tile_size,
                    1.0,
                    (leftover.height as f32) / tile_size_f32,
                    shader_repeat_x,
                    shader_repeat_y,
                );
            }

            if leftover.width != 0 {
                // Finally, the bottom-right tile with a "leftover" size.
                self.add_tile_primitive(
                    clip_and_scroll,
                    prim_info,
                    info,
                    TileOffset::new(num_tiles_x, num_tiles_y),
                    stretched_tile_size,
                    (leftover.width as f32) / tile_size_f32,
                    (leftover.height as f32) / tile_size_f32,
                    shader_repeat_x,
                    shader_repeat_y,
                );
            }
        }
    }

    fn add_tile_primitive(
        &mut self,
        clip_and_scroll: ScrollNodeAndClipChain,
        prim_info: &LayerPrimitiveInfo,
        info: &ImageDisplayItem,
        tile_offset: TileOffset,
        stretched_tile_size: LayerSize,
        tile_ratio_width: f32,
        tile_ratio_height: f32,
        shader_repeat_x: bool,
        shader_repeat_y: bool,
    ) {
        // If the the image is tiled along a given axis, we can't have the shader compute
        // the image repetition pattern. In this case we base the primitive's rectangle size
        // on the stretched tile size which effectively cancels the repetion (and repetition
        // has to be emulated by generating more primitives).
        // If the image is not tiled along this axis, we can perform the repetition in the
        // shader. in this case we use the item's size in the primitive (on that particular
        // axis).
        // See the shader_repeat_x/y code below.

        let stretched_size = LayerSize::new(
            stretched_tile_size.width * tile_ratio_width,
            stretched_tile_size.height * tile_ratio_height,
        );

        let mut prim_rect = LayerRect::new(
            prim_info.rect.origin +
                LayerVector2D::new(
                    tile_offset.x as f32 * stretched_tile_size.width,
                    tile_offset.y as f32 * stretched_tile_size.height,
                ),
            stretched_size,
        );

        if shader_repeat_x {
            assert_eq!(tile_offset.x, 0);
            prim_rect.size.width = prim_info.rect.size.width;
        }

        if shader_repeat_y {
            assert_eq!(tile_offset.y, 0);
            prim_rect.size.height = prim_info.rect.size.height;
        }

        // Fix up the primitive's rect if it overflows the original item rect.
        if let Some(prim_rect) = prim_rect.intersection(&prim_info.rect) {
            let mut prim_info = prim_info.clone();
            prim_info.rect = prim_rect;
            self.builder.add_image(
                clip_and_scroll,
                &prim_info,
                stretched_size,
                info.tile_spacing,
                None,
                info.image_key,
                info.image_rendering,
                info.alpha_type,
                Some(tile_offset),
            );
        }
    }
}

pub fn build_scene(config: &FrameBuilderConfig, request: SceneRequest) -> BuiltScene {
    let mut pipeline_epoch_map = FastHashMap::default();
    let mut clip_scroll_tree = ClipScrollTree::new();

    let frame_builder = FlattenContext::create_frame_builder(
        FrameBuilder::empty(), // WIP, we're not really recycling anything here, clean this up.
        &request.scene,
        &mut clip_scroll_tree,
        request.font_instances,
        request.tiled_image_map,
        &request.view,
        &request.output_pipelines,
        config,
        &mut pipeline_epoch_map
    );

    BuiltScene {
        scene: request.scene,
        frame_builder,
        clip_scroll_tree,
        pipeline_epoch_map,
        removed_pipelines: request.removed_pipelines,
    }
}
