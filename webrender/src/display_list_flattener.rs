
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{AlphaType, BorderDetails, BorderDisplayItem, BuiltDisplayListIter};
use api::{ClipAndScrollInfo, ClipId, ColorF, ComplexClipRegion, DeviceIntPoint, DeviceIntRect};
use api::{DeviceIntSize, DevicePixelScale, DeviceUintRect, DeviceUintSize};
use api::{DisplayItemRef, Epoch, ExtendMode, ExternalScrollId, FilterOp};
use api::{FontInstanceKey, FontRenderMode, GlyphInstance, GlyphOptions, GradientStop};
use api::{IframeDisplayItem, ImageDisplayItem, ImageKey, ImageRendering, ItemRange, LayerPoint};
use api::{LayerPrimitiveInfo, LayerRect, LayerSize, LayerVector2D, LayoutSize, LayoutTransform};
use api::{LayoutVector2D, LineOrientation, LineStyle, LocalClip, PipelineId};
use api::{PropertyBinding, RepeatMode, ScrollFrameDisplayItem, ScrollPolicy, ScrollSensitivity};
use api::{Shadow, SpecificDisplayItem, StackingContext, StickyFrameDisplayItem, TexelRect};
use api::{TileOffset, TransformStyle, YuvColorSpace, YuvData};
use app_units::Au;
use border::ImageBorderSegment;
use clip::{ClipRegion, ClipSource, ClipSources, ClipStore};
use clip_scroll_node::{ClipScrollNode, NodeType, StickyFrameInfo};
use clip_scroll_tree::{ClipChainIndex, ClipScrollNodeIndex, ClipScrollTree};
use euclid::{SideOffsets2D, rect, vec2};
use frame_builder::{FrameBuilder, FrameBuilderConfig};
use glyph_rasterizer::FontInstance;
use hit_test::{HitTestingItem, HitTestingRun};
use internal_types::{FastHashMap, FastHashSet};
use picture::{PictureCompositeMode, PictureKind, PicturePrimitive};
use prim_store::{BrushKind, BrushPrimitive, BrushSegmentDescriptor, CachedGradient};
use prim_store::{CachedGradientIndex, ImageCacheKey, ImagePrimitiveCpu, ImageSource};
use prim_store::{PrimitiveContainer, PrimitiveIndex, PrimitiveKind, PrimitiveStore};
use prim_store::{ScrollNodeAndClipChain, TextRunPrimitiveCpu};
use render_backend::{DocumentView};
use resource_cache::{FontInstanceMap, ImageRequest, TiledImageMap};
use scene::{Scene, ScenePipeline, StackingContextHelpers};
use scene_builder::{BuiltScene, SceneRequest};
use std::{f32, mem, usize};
use tiling::{CompositeOps, ScrollbarPrimitive};
use util::{MaxRect, RectHelpers, recycle_vec};

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

/// A structure that converts a serialized display list into a form that WebRender
/// can use to later build a frame. This structure produces a FrameBuilder. Public
/// members are typically those that are destructured into the FrameBuilder.
pub struct DisplayListFlattener<'a> {
    /// The scene that we are currently flattening.
    scene: &'a Scene,

    /// The ClipScrollTree that we are currently building during flattening.
    clip_scroll_tree: &'a mut ClipScrollTree,

    /// The map of all font instances.
    font_instances: FontInstanceMap,

    /// The map of tiled images.
    tiled_image_map: TiledImageMap,

    /// Used to track the latest flattened epoch for each pipeline.
    pipeline_epochs: Vec<(PipelineId, Epoch)>,

    /// A set of pipelines that the caller has requested be made available as
    /// output textures.
    output_pipelines: &'a FastHashSet<PipelineId>,

    /// A list of replacements to make in order to properly handle fixed position
    /// content as well as stacking contexts that create reference frames.
    replacements: Vec<(ClipId, ClipId)>,

    /// The data structure that converting between ClipId and the various index
    /// types that the ClipScrollTree uses.
    id_to_index_mapper: ClipIdToIndexMapper,

    /// A stack of the current shadow primitives.  The sub-Vec stores
    /// a buffer of fast-path primitives to be appended on pop.
    shadow_prim_stack: Vec<(PrimitiveIndex, Vec<(PrimitiveIndex, ScrollNodeAndClipChain)>)>,

    /// A buffer of "real" content when doing fast-path shadows. This is appended
    /// when the shadow stack is empty.
    pending_shadow_contents: Vec<(PrimitiveIndex, ScrollNodeAndClipChain, LayerPrimitiveInfo)>,

    /// A stack of scroll nodes used during display list processing to properly
    /// parent new scroll nodes.
    reference_frame_stack: Vec<(ClipId, ClipScrollNodeIndex)>,

    /// A stack of stacking context properties.
    sc_stack: Vec<FlattenedStackingContext>,

    /// A stack of the current pictures.
    picture_stack: Vec<PrimitiveIndex>,

    /// A list of scrollbar primitives.
    pub scrollbar_prims: Vec<ScrollbarPrimitive>,

    /// The store of primitives.
    pub prim_store: PrimitiveStore,

    /// Information about all primitives involved in hit testing.
    pub hit_testing_runs: Vec<HitTestingRun>,

    /// The store which holds all complex clipping information.
    pub clip_store: ClipStore,

    /// The configuration to use for the FrameBuilder. We consult this in
    /// order to determine the default font.
    pub config: FrameBuilderConfig,

    /// The gradients collecting during display list flattening.
    pub cached_gradients: Vec<CachedGradient>,
}

impl<'a> DisplayListFlattener<'a> {
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

        let mut flattener = DisplayListFlattener {
            scene,
            clip_scroll_tree,
            font_instances,
            tiled_image_map,
            config: *frame_builder_config,
            pipeline_epochs: Vec::new(),
            replacements: Vec::new(),
            output_pipelines,
            id_to_index_mapper: ClipIdToIndexMapper::default(),
            hit_testing_runs: recycle_vec(old_builder.hit_testing_runs),
            shadow_prim_stack: Vec::new(),
            cached_gradients: recycle_vec(old_builder.cached_gradients),
            pending_shadow_contents: Vec::new(),
            scrollbar_prims: recycle_vec(old_builder.scrollbar_prims),
            reference_frame_stack: Vec::new(),
            picture_stack: Vec::new(),
            sc_stack: Vec::new(),
            prim_store: old_builder.prim_store.recycle(),
            clip_store: old_builder.clip_store.recycle(),
        };

        flattener.id_to_index_mapper.initialize_for_pipeline(root_pipeline);
        flattener.push_root(
            root_pipeline_id,
            &root_pipeline.viewport_size,
            &root_pipeline.content_size,
        );
        flattener.setup_viewport_offset(view.inner_rect, view.accumulated_scale_factor());
        flattener.flatten_root(root_pipeline, &root_pipeline.viewport_size);

        debug_assert!(flattener.picture_stack.is_empty());
        pipeline_epochs.extend(flattener.pipeline_epochs.drain(..));

        FrameBuilder::with_display_list_flattener(
            view.inner_rect,
            background_color,
            view.window_size,
            flattener
        )
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

        self.push_stacking_context(
            pipeline_id,
            CompositeOps::default(),
            TransformStyle::Flat,
            true,
            true,
            scroll_frame_info,
        );

        // For the root pipeline, there's no need to add a full screen rectangle
        // here, as it's handled by the framebuffer clear.
        if self.scene.root_pipeline_id != Some(pipeline_id) {
            if let Some(pipeline) = self.scene.pipelines.get(&pipeline_id) {
                if let Some(bg_color) = pipeline.background_color {
                    let root_bounds = LayerRect::new(LayerPoint::zero(), *frame_size);
                    let info = LayerPrimitiveInfo::new(root_bounds);
                    self.add_solid_rectangle(
                        reference_frame_info,
                        &info,
                        bg_color,
                        None,
                    );
                }
            }
        }

        self.flatten_items(&mut pipeline.display_list.iter(), pipeline_id, LayerVector2D::zero());

        if self.config.enable_scrollbars {
            let scrollbar_rect = LayerRect::new(LayerPoint::zero(), LayerSize::new(10.0, 70.0));
            let container_rect = LayerRect::new(LayerPoint::zero(), *frame_size);
            self.add_scroll_bar(
                reference_frame_info,
                &LayerPrimitiveInfo::new(scrollbar_rect),
                DEFAULT_SCROLLBAR_COLOR,
                ScrollbarInfo(scroll_frame_info.scroll_node_id, container_rect),
            );
        }

        self.pop_stacking_context();
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

        self.add_clip_node(info.clip_id, clip_and_scroll_ids.scroll_node_id, clip_region);

        self.add_scroll_frame(
            info.scroll_frame_id,
            info.clip_id,
            info.external_id,
            pipeline_id,
            &frame_rect,
            &content_rect.size,
            info.scroll_sensitivity,
        );
    }

    fn flatten_stacking_context(
        &mut self,
        traversal: &mut BuiltDisplayListIter<'a>,
        pipeline_id: PipelineId,
        unreplaced_scroll_id: ClipId,
        mut scroll_node_id: ClipId,
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
            scroll_node_id = self.current_reference_frame_id();
            self.replacements.push((unreplaced_scroll_id, scroll_node_id));
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
            self.push_reference_frame(
                reference_frame_id,
                Some(scroll_node_id),
                pipeline_id,
                &reference_frame_bounds,
                stacking_context.transform,
                stacking_context.perspective,
                reference_frame_relative_offset,
            );
            self.replacements.push((unreplaced_scroll_id, reference_frame_id));
            reference_frame_relative_offset = LayerVector2D::zero();
        }

        // We apply the replacements one more time in case we need to set it to a replacement
        // that we just pushed above.
        let sc_scroll_node = self.apply_scroll_frame_id_replacement(unreplaced_scroll_id);
        let stacking_context_clip_and_scroll =
            self.id_to_index_mapper.simple_scroll_and_clip_chain(&sc_scroll_node);
        self.push_stacking_context(
            pipeline_id,
            composition_operations,
            stacking_context.transform_style,
            is_backface_visible,
            false,
            stacking_context_clip_and_scroll,
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
            self.pop_reference_frame();
        }

        self.pop_stacking_context();
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

        self.add_clip_node(
            info.clip_id,
            clip_and_scroll_ids.scroll_node_id,
            ClipRegion::create_for_clip_node_with_local_clip(
                &item.local_clip(),
                &reference_frame_relative_offset
            ),
        );

        let epoch = self.scene.pipeline_epochs[&iframe_pipeline_id];
        self.pipeline_epochs.push((iframe_pipeline_id, epoch));

        let bounds = item.rect();
        let iframe_rect = LayerRect::new(LayerPoint::zero(), bounds.size);
        let origin = *reference_frame_relative_offset + bounds.origin.to_vector();
        self.push_reference_frame(
            ClipId::root_reference_frame(iframe_pipeline_id),
            Some(info.clip_id),
            iframe_pipeline_id,
            &iframe_rect,
            None,
            None,
            origin,
        );

        self.add_scroll_frame(
            ClipId::root_scroll_node(iframe_pipeline_id),
            ClipId::root_reference_frame(iframe_pipeline_id),
            Some(ExternalScrollId(0, iframe_pipeline_id)),
            iframe_pipeline_id,
            &iframe_rect,
            &pipeline.content_size,
            ScrollSensitivity::ScriptAndInputEvents,
        );

        self.flatten_root(pipeline, &iframe_rect.size);

        self.pop_reference_frame();
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
                        self.add_image(
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
                self.add_yuv_image(
                    clip_and_scroll,
                    &prim_info,
                    info.yuv_data,
                    info.color_space,
                    info.image_rendering,
                );
            }
            SpecificDisplayItem::Text(ref text_info) => {
                self.add_text(
                    clip_and_scroll,
                    reference_frame_relative_offset,
                    &prim_info,
                    &text_info.font_key,
                    &text_info.color,
                    item.glyphs(),
                    item.display_list().get(item.glyphs()).count(),
                    text_info.glyph_options,
                );
            }
            SpecificDisplayItem::Rectangle(ref info) => {
                self.add_solid_rectangle(
                    clip_and_scroll,
                    &prim_info,
                    info.color,
                    None,
                );
            }
            SpecificDisplayItem::ClearRectangle => {
                self.add_clear_rectangle(
                    clip_and_scroll,
                    &prim_info,
                );
            }
            SpecificDisplayItem::Line(ref info) => {
                self.add_line(
                    clip_and_scroll,
                    &prim_info,
                    info.wavy_line_thickness,
                    info.orientation,
                    &info.color,
                    info.style,
                );
            }
            SpecificDisplayItem::Gradient(ref info) => {
                self.add_gradient(
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
                self.add_radial_gradient(
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
                self.add_box_shadow(
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
                self.add_border(
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
                self.add_clip_node(info.id, clip_and_scroll_ids.scroll_node_id, clip_region);
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
                self
                    .push_shadow(shadow, clip_and_scroll, &prim_info);
            }
            SpecificDisplayItem::PopAllShadows => {
                self.pop_all_shadows();
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
            self.add_image(
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

    /// Create a primitive and add it to the prim store. This method doesn't
    /// add the primitive to the draw list, so can be used for creating
    /// sub-primitives.
    pub fn create_primitive(
        &mut self,
        info: &LayerPrimitiveInfo,
        mut clip_sources: Vec<ClipSource>,
        container: PrimitiveContainer,
    ) -> PrimitiveIndex {
        if let &LocalClip::RoundedRect(main, region) = &info.local_clip {
            clip_sources.push(ClipSource::Rectangle(main));

            clip_sources.push(ClipSource::new_rounded_rect(
                region.rect,
                region.radii,
                region.mode,
            ));
        }

        let stacking_context = self.sc_stack.last().expect("bug: no stacking context!");

        let clip_sources = self.clip_store.insert(ClipSources::new(clip_sources));
        let prim_index = self.prim_store.add_primitive(
            &info.rect,
            &info.local_clip.clip_rect(),
            info.is_backface_visible && stacking_context.is_backface_visible,
            clip_sources,
            info.tag,
            container,
        );

        prim_index
    }

    pub fn add_primitive_to_hit_testing_list(
        &mut self,
        info: &LayerPrimitiveInfo,
        clip_and_scroll: ScrollNodeAndClipChain
    ) {
        let tag = match info.tag {
            Some(tag) => tag,
            None => return,
        };

        let new_item = HitTestingItem::new(tag, info);
        match self.hit_testing_runs.last_mut() {
            Some(&mut HitTestingRun(ref mut items, prev_clip_and_scroll))
                if prev_clip_and_scroll == clip_and_scroll => {
                items.push(new_item);
                return;
            }
            _ => {}
        }

        self.hit_testing_runs.push(HitTestingRun(vec![new_item], clip_and_scroll));
    }

    /// Add an already created primitive to the draw lists.
    pub fn add_primitive_to_draw_list(
        &mut self,
        prim_index: PrimitiveIndex,
        clip_and_scroll: ScrollNodeAndClipChain,
    ) {
        // Add primitive to the top-most Picture on the stack.
        // TODO(gw): Let's consider removing the extra indirection
        //           needed to get a specific primitive index...
        let pic_prim_index = self.picture_stack.last().unwrap();
        let metadata = &self.prim_store.cpu_metadata[pic_prim_index.0];
        let pic = &mut self.prim_store.cpu_pictures[metadata.cpu_prim_index.0];
        pic.add_primitive(
            prim_index,
            clip_and_scroll
        );
    }

    /// Convenience interface that creates a primitive entry and adds it
    /// to the draw list.
    pub fn add_primitive(
        &mut self,
        clip_and_scroll: ScrollNodeAndClipChain,
        info: &LayerPrimitiveInfo,
        clip_sources: Vec<ClipSource>,
        container: PrimitiveContainer,
    ) -> PrimitiveIndex {
        self.add_primitive_to_hit_testing_list(info, clip_and_scroll);
        let prim_index = self.create_primitive(info, clip_sources, container);

        self.add_primitive_to_draw_list(prim_index, clip_and_scroll);
        prim_index
    }

    pub fn push_stacking_context(
        &mut self,
        pipeline_id: PipelineId,
        composite_ops: CompositeOps,
        transform_style: TransformStyle,
        is_backface_visible: bool,
        is_pipeline_root: bool,
        clip_and_scroll: ScrollNodeAndClipChain,
    ) {
        // Construct the necessary set of Picture primitives
        // to draw this stacking context.
        let current_reference_frame_index = self.current_reference_frame_index();

        // An arbitrary large clip rect. For now, we don't
        // specify a clip specific to the stacking context.
        // However, now that they are represented as Picture
        // primitives, we can apply any kind of clip mask
        // to them, as for a normal primitive. This is needed
        // to correctly handle some CSS cases (see #1957).
        let max_clip = LayerRect::max_rect();

        // If there is no root picture, create one for the main framebuffer.
        if self.sc_stack.is_empty() {
            // Should be no pictures at all if the stack is empty...
            debug_assert!(self.prim_store.cpu_pictures.is_empty());
            debug_assert_eq!(transform_style, TransformStyle::Flat);

            // This picture stores primitive runs for items on the
            // main framebuffer.
            let pic = PicturePrimitive::new_image(
                None,
                false,
                pipeline_id,
                current_reference_frame_index,
                None,
            );

            // No clip sources needed for the main framebuffer.
            let clip_sources = self.clip_store.insert(ClipSources::new(Vec::new()));

            // Add root picture primitive. The provided layer rect
            // is zero, because we don't yet know the size of the
            // picture. Instead, this is calculated recursively
            // when we cull primitives.
            let prim_index = self.prim_store.add_primitive(
                &LayerRect::zero(),
                &max_clip,
                true,
                clip_sources,
                None,
                PrimitiveContainer::Picture(pic),
            );

            self.picture_stack.push(prim_index);
        } else if composite_ops.mix_blend_mode.is_some() && self.sc_stack.len() > 2 {
            // If we have a mix-blend-mode, and we aren't the primary framebuffer,
            // the stacking context needs to be isolated to blend correctly as per
            // the CSS spec.
            // TODO(gw): The way we detect not being the primary framebuffer (len > 2)
            //           is hacky and depends on how we create a root stacking context
            //           during flattening.
            let current_pic_prim_index = self.picture_stack.last().unwrap();
            let pic_cpu_prim_index = self.prim_store.cpu_metadata[current_pic_prim_index.0].cpu_prim_index;
            let parent_pic = &mut self.prim_store.cpu_pictures[pic_cpu_prim_index.0];

            match parent_pic.kind {
                PictureKind::Image { ref mut composite_mode, .. } => {
                    // If not already isolated for some other reason,
                    // make this picture as isolated.
                    if composite_mode.is_none() {
                        *composite_mode = Some(PictureCompositeMode::Blit);
                    }
                }
                PictureKind::TextShadow { .. } |
                PictureKind::BoxShadow { .. } => {
                    panic!("bug: text/box pictures invalid here");
                }
            }
        }

        // Get the transform-style of the parent stacking context,
        // which determines if we *might* need to draw this on
        // an intermediate surface for plane splitting purposes.
        let parent_transform_style = match self.sc_stack.last() {
            Some(sc) => sc.transform_style,
            None => TransformStyle::Flat,
        };

        // If this is preserve-3d *or* the parent is, then this stacking
        // context is participating in the 3d rendering context. In that
        // case, hoist the picture up to the 3d rendering context
        // container, so that it's rendered as a sibling with other
        // elements in this context.
        let participating_in_3d_context =
            composite_ops.count() == 0 &&
            (parent_transform_style == TransformStyle::Preserve3D ||
             transform_style == TransformStyle::Preserve3D);

        // If this is participating in a 3d context *and* the
        // parent was not a 3d context, then this must be the
        // element that establishes a new 3d context.
        let establishes_3d_context =
            participating_in_3d_context &&
            parent_transform_style == TransformStyle::Flat;

        let rendering_context_3d_prim_index = if establishes_3d_context {
            // If establishing a 3d context, we need to add a picture
            // that will be the container for all the planes and any
            // un-transformed content.
            let container = PicturePrimitive::new_image(
                None,
                false,
                pipeline_id,
                current_reference_frame_index,
                None,
            );

            let clip_sources = self.clip_store.insert(ClipSources::new(Vec::new()));

            let prim_index = self.prim_store.add_primitive(
                &LayerRect::zero(),
                &max_clip,
                is_backface_visible,
                clip_sources,
                None,
                PrimitiveContainer::Picture(container),
            );

            let parent_pic_prim_index = *self.picture_stack.last().unwrap();
            let pic_prim_index = self.prim_store.cpu_metadata[parent_pic_prim_index.0].cpu_prim_index;
            let pic = &mut self.prim_store.cpu_pictures[pic_prim_index.0];
            pic.add_primitive(
                prim_index,
                clip_and_scroll,
            );

            self.picture_stack.push(prim_index);

            Some(prim_index)
        } else {
            None
        };

        let mut parent_pic_prim_index = if !establishes_3d_context && participating_in_3d_context {
            // If we're in a 3D context, we will parent the picture
            // to the first stacking context we find that is a
            // 3D rendering context container. This follows the spec
            // by hoisting these items out into the same 3D context
            // for plane splitting.
            self.sc_stack
                .iter()
                .rev()
                .find(|sc| sc.rendering_context_3d_prim_index.is_some())
                .map(|sc| sc.rendering_context_3d_prim_index.unwrap())
                .unwrap()
        } else {
            *self.picture_stack.last().unwrap()
        };

        // For each filter, create a new image with that composite mode.
        for filter in composite_ops.filters.iter().rev() {
            let src_prim = PicturePrimitive::new_image(
                Some(PictureCompositeMode::Filter(*filter)),
                false,
                pipeline_id,
                current_reference_frame_index,
                None,
            );
            let src_clip_sources = self.clip_store.insert(ClipSources::new(Vec::new()));

            let src_prim_index = self.prim_store.add_primitive(
                &LayerRect::zero(),
                &max_clip,
                is_backface_visible,
                src_clip_sources,
                None,
                PrimitiveContainer::Picture(src_prim),
            );

            let pic_prim_index = self.prim_store.cpu_metadata[parent_pic_prim_index.0].cpu_prim_index;
            parent_pic_prim_index = src_prim_index;
            let pic = &mut self.prim_store.cpu_pictures[pic_prim_index.0];
            pic.add_primitive(
                src_prim_index,
                clip_and_scroll,
            );

            self.picture_stack.push(src_prim_index);
        }

        // Same for mix-blend-mode.
        if let Some(mix_blend_mode) = composite_ops.mix_blend_mode {
            let src_prim = PicturePrimitive::new_image(
                Some(PictureCompositeMode::MixBlend(mix_blend_mode)),
                false,
                pipeline_id,
                current_reference_frame_index,
                None,
            );
            let src_clip_sources = self.clip_store.insert(ClipSources::new(Vec::new()));

            let src_prim_index = self.prim_store.add_primitive(
                &LayerRect::zero(),
                &max_clip,
                is_backface_visible,
                src_clip_sources,
                None,
                PrimitiveContainer::Picture(src_prim),
            );

            let pic_prim_index = self.prim_store.cpu_metadata[parent_pic_prim_index.0].cpu_prim_index;
            parent_pic_prim_index = src_prim_index;
            let pic = &mut self.prim_store.cpu_pictures[pic_prim_index.0];
            pic.add_primitive(
                src_prim_index,
                clip_and_scroll,
            );

            self.picture_stack.push(src_prim_index);
        }

        // By default, this picture will be collapsed into
        // the owning target.
        let mut composite_mode = None;
        let mut frame_output_pipeline_id = None;

        // If this stacking context if the root of a pipeline, and the caller
        // has requested it as an output frame, create a render task to isolate it.
        if is_pipeline_root && self.output_pipelines.contains(&pipeline_id) {
            composite_mode = Some(PictureCompositeMode::Blit);
            frame_output_pipeline_id = Some(pipeline_id);
        }

        if participating_in_3d_context {
            // TODO(gw): For now, as soon as this picture is in
            //           a 3D context, we draw it to an intermediate
            //           surface and apply plane splitting. However,
            //           there is a large optimization opportunity here.
            //           During culling, we can check if there is actually
            //           perspective present, and skip the plane splitting
            //           completely when that is not the case.
            composite_mode = Some(PictureCompositeMode::Blit);
        }

        // Add picture for this actual stacking context contents to render into.
        let sc_prim = PicturePrimitive::new_image(
            composite_mode,
            participating_in_3d_context,
            pipeline_id,
            current_reference_frame_index,
            frame_output_pipeline_id,
        );

        let sc_clip_sources = self.clip_store.insert(ClipSources::new(Vec::new()));
        let sc_prim_index = self.prim_store.add_primitive(
            &LayerRect::zero(),
            &max_clip,
            is_backface_visible,
            sc_clip_sources,
            None,
            PrimitiveContainer::Picture(sc_prim),
        );

        let pic_prim_index = self.prim_store.cpu_metadata[parent_pic_prim_index.0].cpu_prim_index;
        let sc_pic = &mut self.prim_store.cpu_pictures[pic_prim_index.0];
        sc_pic.add_primitive(
            sc_prim_index,
            clip_and_scroll,
        );

        // Add this as the top-most picture for primitives to be added to.
        self.picture_stack.push(sc_prim_index);

        // TODO(gw): This is super conservative. We can expand on this a lot
        //           once all the picture code is in place and landed.
        let allow_subpixel_aa = composite_ops.count() == 0 &&
                                transform_style == TransformStyle::Flat;

        // Push the SC onto the stack, so we know how to handle things in
        // pop_stacking_context.
        let sc = FlattenedStackingContext {
            composite_ops,
            is_backface_visible,
            pipeline_id,
            allow_subpixel_aa,
            transform_style,
            rendering_context_3d_prim_index,
        };

        self.sc_stack.push(sc);
    }

    pub fn pop_stacking_context(&mut self) {
        let sc = self.sc_stack.pop().unwrap();

        // Always pop at least the main picture for this stacking context.
        let mut pop_count = 1;

        // Remove the picture for any filter/mix-blend-mode effects.
        pop_count += sc.composite_ops.count();

        // Remove the 3d context container if created
        if sc.rendering_context_3d_prim_index.is_some() {
            pop_count += 1;
        }

        for _ in 0 .. pop_count {
            self.picture_stack.pop().expect("bug: mismatched picture stack");
        }

        // By the time the stacking context stack is empty, we should
        // also have cleared the picture stack.
        if self.sc_stack.is_empty() {
            self.picture_stack.pop().expect("bug: picture stack invalid");
            debug_assert!(self.picture_stack.is_empty());
        }

        assert!(
            self.shadow_prim_stack.is_empty(),
            "Found unpopped text shadows when popping stacking context!"
        );
    }

    pub fn push_reference_frame(
        &mut self,
        reference_frame_id: ClipId,
        parent_id: Option<ClipId>,
        pipeline_id: PipelineId,
        rect: &LayerRect,
        source_transform: Option<PropertyBinding<LayoutTransform>>,
        source_perspective: Option<LayoutTransform>,
        origin_in_parent_reference_frame: LayerVector2D,
    ) -> ClipScrollNodeIndex {
        let index = self.id_to_index_mapper.get_node_index(reference_frame_id);
        let node = ClipScrollNode::new_reference_frame(
            parent_id.map(|id| self.id_to_index_mapper.get_node_index(id)),
            rect,
            source_transform,
            source_perspective,
            origin_in_parent_reference_frame,
            pipeline_id,
        );
        self.clip_scroll_tree.add_node(node, index);
        self.reference_frame_stack.push((reference_frame_id, index));

        match parent_id {
            Some(ref parent_id) =>
                self.id_to_index_mapper.map_to_parent_clip_chain(reference_frame_id, parent_id),
            _ => self.id_to_index_mapper.add_clip_chain(reference_frame_id, ClipChainIndex(0)),
        }
        index
    }

    pub fn current_reference_frame_index(&self) -> ClipScrollNodeIndex {
        self.reference_frame_stack.last().unwrap().1
    }

    pub fn current_reference_frame_id(&self) -> ClipId{
        self.reference_frame_stack.last().unwrap().0
    }

    pub fn setup_viewport_offset(
        &mut self,
        inner_rect: DeviceUintRect,
        device_pixel_scale: DevicePixelScale,
    ) {
        let viewport_offset = (inner_rect.origin.to_vector().to_f32() / device_pixel_scale).round();
        let root_id = self.clip_scroll_tree.root_reference_frame_index();
        let root_node = &mut self.clip_scroll_tree.nodes[root_id.0];
        if let NodeType::ReferenceFrame(ref mut info) = root_node.node_type {
            info.resolved_transform =
                LayerVector2D::new(viewport_offset.x, viewport_offset.y).into();
        }
    }

    pub fn push_root(
        &mut self,
        pipeline_id: PipelineId,
        viewport_size: &LayerSize,
        content_size: &LayerSize,
    ) {
        let viewport_rect = LayerRect::new(LayerPoint::zero(), *viewport_size);

        self.push_reference_frame(
            ClipId::root_reference_frame(pipeline_id),
            None,
            pipeline_id,
            &viewport_rect,
            None,
            None,
            LayerVector2D::zero(),
        );

        self.add_scroll_frame(
            ClipId::root_scroll_node(pipeline_id),
            ClipId::root_reference_frame(pipeline_id),
            Some(ExternalScrollId(0, pipeline_id)),
            pipeline_id,
            &viewport_rect,
            content_size,
            ScrollSensitivity::ScriptAndInputEvents,
        );
    }

    pub fn add_clip_node(
        &mut self,
        new_node_id: ClipId,
        parent_id: ClipId,
        clip_region: ClipRegion,
    ) -> ClipScrollNodeIndex {
        let clip_rect = clip_region.main;
        let clip_sources = ClipSources::from(clip_region);

        debug_assert!(clip_sources.has_clips());
        let handle = self.clip_store.insert(clip_sources);

        let node_index = self.id_to_index_mapper.get_node_index(new_node_id);
        let clip_chain_index = self.clip_scroll_tree.add_clip_node(
            node_index,
            self.id_to_index_mapper.get_node_index(parent_id),
            handle,
            clip_rect,
            new_node_id.pipeline_id(),
        );
        self.id_to_index_mapper.add_clip_chain(new_node_id, clip_chain_index);
        node_index
    }

    pub fn add_scroll_frame(
        &mut self,
        new_node_id: ClipId,
        parent_id: ClipId,
        external_id: Option<ExternalScrollId>,
        pipeline_id: PipelineId,
        frame_rect: &LayerRect,
        content_size: &LayerSize,
        scroll_sensitivity: ScrollSensitivity,
    ) -> ClipScrollNodeIndex {
        let node_index = self.id_to_index_mapper.get_node_index(new_node_id);
        let node = ClipScrollNode::new_scroll_frame(
            pipeline_id,
            self.id_to_index_mapper.get_node_index(parent_id),
            external_id,
            frame_rect,
            content_size,
            scroll_sensitivity,
        );

        self.clip_scroll_tree.add_node(node, node_index);
        self.id_to_index_mapper.map_to_parent_clip_chain(new_node_id, &parent_id);
        node_index
    }

    pub fn pop_reference_frame(&mut self) {
        self.reference_frame_stack.pop();
    }

    pub fn push_shadow(
        &mut self,
        shadow: Shadow,
        clip_and_scroll: ScrollNodeAndClipChain,
        info: &LayerPrimitiveInfo,
    ) {
        let pipeline_id = self.sc_stack.last().unwrap().pipeline_id;
        let prim = PicturePrimitive::new_text_shadow(shadow, pipeline_id);

        // Create an empty shadow primitive. Insert it into
        // the draw lists immediately so that it will be drawn
        // before any visual text elements that are added as
        // part of this shadow context.
        let prim_index = self.create_primitive(
            info,
            Vec::new(),
            PrimitiveContainer::Picture(prim),
        );

        let pending = vec![(prim_index, clip_and_scroll)];
        self.shadow_prim_stack.push((prim_index, pending));
    }

    pub fn pop_all_shadows(&mut self) {
        assert!(self.shadow_prim_stack.len() > 0, "popped shadows, but none were present");

        // Borrowcheck dance
        let mut shadows = mem::replace(&mut self.shadow_prim_stack, Vec::new());
        for (_, pending_primitives) in shadows.drain(..) {
            // Push any fast-path shadows now
            for (prim_index, clip_and_scroll) in pending_primitives {
                self.add_primitive_to_draw_list(prim_index, clip_and_scroll);
            }
        }

        let mut pending_primitives = mem::replace(&mut self.pending_shadow_contents, Vec::new());
        for (prim_index, clip_and_scroll, info) in pending_primitives.drain(..) {
            self.add_primitive_to_hit_testing_list(&info, clip_and_scroll);
            self.add_primitive_to_draw_list(prim_index, clip_and_scroll);
        }

        mem::replace(&mut self.pending_shadow_contents, pending_primitives);
        mem::replace(&mut self.shadow_prim_stack, shadows);
    }

    pub fn add_solid_rectangle(
        &mut self,
        clip_and_scroll: ScrollNodeAndClipChain,
        info: &LayerPrimitiveInfo,
        color: ColorF,
        segments: Option<BrushSegmentDescriptor>,
    ) {
        if color.a == 0.0 {
            // Don't add transparent rectangles to the draw list, but do consider them for hit
            // testing. This allows specifying invisible hit testing areas.
            self.add_primitive_to_hit_testing_list(info, clip_and_scroll);
            return;
        }

        let prim = BrushPrimitive::new(
            BrushKind::Solid {
                color,
            },
            segments,
        );

        self.add_primitive(
            clip_and_scroll,
            info,
            Vec::new(),
            PrimitiveContainer::Brush(prim),
        );
    }

    pub fn add_clear_rectangle(
        &mut self,
        clip_and_scroll: ScrollNodeAndClipChain,
        info: &LayerPrimitiveInfo,
    ) {
        let prim = BrushPrimitive::new(
            BrushKind::Clear,
            None,
        );

        self.add_primitive(
            clip_and_scroll,
            info,
            Vec::new(),
            PrimitiveContainer::Brush(prim),
        );
    }

    pub fn add_scroll_bar(
        &mut self,
        clip_and_scroll: ScrollNodeAndClipChain,
        info: &LayerPrimitiveInfo,
        color: ColorF,
        scrollbar_info: ScrollbarInfo,
    ) {
        if color.a == 0.0 {
            return;
        }

        let prim = BrushPrimitive::new(
            BrushKind::Solid {
                color,
            },
            None,
        );

        let prim_index = self.add_primitive(
            clip_and_scroll,
            info,
            Vec::new(),
            PrimitiveContainer::Brush(prim),
        );

        self.scrollbar_prims.push(ScrollbarPrimitive {
            prim_index,
            scroll_frame_index: scrollbar_info.0,
            frame_rect: scrollbar_info.1,
        });
    }

    pub fn add_line(
        &mut self,
        clip_and_scroll: ScrollNodeAndClipChain,
        info: &LayerPrimitiveInfo,
        wavy_line_thickness: f32,
        orientation: LineOrientation,
        line_color: &ColorF,
        style: LineStyle,
    ) {
        let line = BrushPrimitive::new(
            BrushKind::Line {
                wavy_line_thickness,
                color: line_color.premultiplied(),
                style,
                orientation,
            },
            None,
        );

        let mut fast_shadow_prims = Vec::new();
        for (idx, &(shadow_prim_index, _)) in self.shadow_prim_stack.iter().enumerate() {
            let shadow_metadata = &self.prim_store.cpu_metadata[shadow_prim_index.0];
            let picture = &self.prim_store.cpu_pictures[shadow_metadata.cpu_prim_index.0];
            match picture.kind {
                PictureKind::TextShadow { offset, color, blur_radius, .. } if blur_radius == 0.0 => {
                    fast_shadow_prims.push((idx, offset, color));
                }
                _ => {}
            }
        }

        for (idx, shadow_offset, shadow_color) in fast_shadow_prims {
            let line = BrushPrimitive::new(
                BrushKind::Line {
                    wavy_line_thickness,
                    color: shadow_color.premultiplied(),
                    style,
                    orientation,
                },
                None,
            );
            let mut info = info.clone();
            info.rect = info.rect.translate(&shadow_offset);
            info.local_clip =
              LocalClip::from(info.local_clip.clip_rect().translate(&shadow_offset));
            let prim_index = self.create_primitive(
                &info,
                Vec::new(),
                PrimitiveContainer::Brush(line),
            );
            self.shadow_prim_stack[idx].1.push((prim_index, clip_and_scroll));
        }

        let prim_index = self.create_primitive(
            &info,
            Vec::new(),
            PrimitiveContainer::Brush(line),
        );

        if line_color.a > 0.0 {
            if self.shadow_prim_stack.is_empty() {
                self.add_primitive_to_hit_testing_list(&info, clip_and_scroll);
                self.add_primitive_to_draw_list(prim_index, clip_and_scroll);
            } else {
                self.pending_shadow_contents.push((prim_index, clip_and_scroll, *info));
            }
        }

        for &(shadow_prim_index, _) in &self.shadow_prim_stack {
            let shadow_metadata = &mut self.prim_store.cpu_metadata[shadow_prim_index.0];
            debug_assert_eq!(shadow_metadata.prim_kind, PrimitiveKind::Picture);
            let picture =
                &mut self.prim_store.cpu_pictures[shadow_metadata.cpu_prim_index.0];

            match picture.kind {
                // Only run real blurs here (fast path zero blurs are handled above).
                PictureKind::TextShadow { blur_radius, .. } if blur_radius > 0.0 => {
                    picture.add_primitive(
                        prim_index,
                        clip_and_scroll,
                    );
                }
                _ => {}
            }
        }
    }

    pub fn add_border(
        &mut self,
        clip_and_scroll: ScrollNodeAndClipChain,
        info: &LayerPrimitiveInfo,
        border_item: &BorderDisplayItem,
        gradient_stops: ItemRange<GradientStop>,
        gradient_stops_count: usize,
    ) {
        let rect = info.rect;
        let create_segments = |outset: SideOffsets2D<f32>| {
            // Calculate the modified rect as specific by border-image-outset
            let origin = LayerPoint::new(rect.origin.x - outset.left, rect.origin.y - outset.top);
            let size = LayerSize::new(
                rect.size.width + outset.left + outset.right,
                rect.size.height + outset.top + outset.bottom,
            );
            let rect = LayerRect::new(origin, size);

            let tl_outer = LayerPoint::new(rect.origin.x, rect.origin.y);
            let tl_inner = tl_outer + vec2(border_item.widths.left, border_item.widths.top);

            let tr_outer = LayerPoint::new(rect.origin.x + rect.size.width, rect.origin.y);
            let tr_inner = tr_outer + vec2(-border_item.widths.right, border_item.widths.top);

            let bl_outer = LayerPoint::new(rect.origin.x, rect.origin.y + rect.size.height);
            let bl_inner = bl_outer + vec2(border_item.widths.left, -border_item.widths.bottom);

            let br_outer = LayerPoint::new(
                rect.origin.x + rect.size.width,
                rect.origin.y + rect.size.height,
            );
            let br_inner = br_outer - vec2(border_item.widths.right, border_item.widths.bottom);

            // Build the list of gradient segments
            vec![
                // Top left
                LayerRect::from_floats(tl_outer.x, tl_outer.y, tl_inner.x, tl_inner.y),
                // Top right
                LayerRect::from_floats(tr_inner.x, tr_outer.y, tr_outer.x, tr_inner.y),
                // Bottom right
                LayerRect::from_floats(br_inner.x, br_inner.y, br_outer.x, br_outer.y),
                // Bottom left
                LayerRect::from_floats(bl_outer.x, bl_inner.y, bl_inner.x, bl_outer.y),
                // Top
                LayerRect::from_floats(tl_inner.x, tl_outer.y, tr_inner.x, tl_inner.y),
                // Bottom
                LayerRect::from_floats(bl_inner.x, bl_inner.y, br_inner.x, bl_outer.y),
                // Left
                LayerRect::from_floats(tl_outer.x, tl_inner.y, tl_inner.x, bl_inner.y),
                // Right
                LayerRect::from_floats(tr_inner.x, tr_inner.y, br_outer.x, br_inner.y),
            ]
        };

        match border_item.details {
            BorderDetails::Image(ref border) => {
                // Calculate the modified rect as specific by border-image-outset
                let origin = LayerPoint::new(
                    rect.origin.x - border.outset.left,
                    rect.origin.y - border.outset.top,
                );
                let size = LayerSize::new(
                    rect.size.width + border.outset.left + border.outset.right,
                    rect.size.height + border.outset.top + border.outset.bottom,
                );
                let rect = LayerRect::new(origin, size);

                // Calculate the local texel coords of the slices.
                let px0 = 0.0;
                let px1 = border.patch.slice.left as f32;
                let px2 = border.patch.width as f32 - border.patch.slice.right as f32;
                let px3 = border.patch.width as f32;

                let py0 = 0.0;
                let py1 = border.patch.slice.top as f32;
                let py2 = border.patch.height as f32 - border.patch.slice.bottom as f32;
                let py3 = border.patch.height as f32;

                let tl_outer = LayerPoint::new(rect.origin.x, rect.origin.y);
                let tl_inner = tl_outer + vec2(border_item.widths.left, border_item.widths.top);

                let tr_outer = LayerPoint::new(rect.origin.x + rect.size.width, rect.origin.y);
                let tr_inner = tr_outer + vec2(-border_item.widths.right, border_item.widths.top);

                let bl_outer = LayerPoint::new(rect.origin.x, rect.origin.y + rect.size.height);
                let bl_inner = bl_outer + vec2(border_item.widths.left, -border_item.widths.bottom);

                let br_outer = LayerPoint::new(
                    rect.origin.x + rect.size.width,
                    rect.origin.y + rect.size.height,
                );
                let br_inner = br_outer - vec2(border_item.widths.right, border_item.widths.bottom);

                fn add_segment(
                    segments: &mut Vec<ImageBorderSegment>,
                    rect: LayerRect,
                    uv_rect: TexelRect,
                    repeat_horizontal: RepeatMode,
                    repeat_vertical: RepeatMode) {
                    if uv_rect.uv1.x > uv_rect.uv0.x &&
                       uv_rect.uv1.y > uv_rect.uv0.y {
                        segments.push(ImageBorderSegment::new(
                            rect,
                            uv_rect,
                            repeat_horizontal,
                            repeat_vertical,
                        ));
                    }
                }

                // Build the list of image segments
                let mut segments = vec![];

                // Top left
                add_segment(
                    &mut segments,
                    LayerRect::from_floats(tl_outer.x, tl_outer.y, tl_inner.x, tl_inner.y),
                    TexelRect::new(px0, py0, px1, py1),
                    RepeatMode::Stretch,
                    RepeatMode::Stretch
                );
                // Top right
                add_segment(
                    &mut segments,
                    LayerRect::from_floats(tr_inner.x, tr_outer.y, tr_outer.x, tr_inner.y),
                    TexelRect::new(px2, py0, px3, py1),
                    RepeatMode::Stretch,
                    RepeatMode::Stretch
                );
                // Bottom right
                add_segment(
                    &mut segments,
                    LayerRect::from_floats(br_inner.x, br_inner.y, br_outer.x, br_outer.y),
                    TexelRect::new(px2, py2, px3, py3),
                    RepeatMode::Stretch,
                    RepeatMode::Stretch
                );
                // Bottom left
                add_segment(
                    &mut segments,
                    LayerRect::from_floats(bl_outer.x, bl_inner.y, bl_inner.x, bl_outer.y),
                    TexelRect::new(px0, py2, px1, py3),
                    RepeatMode::Stretch,
                    RepeatMode::Stretch
                );

                // Center
                if border.fill {
                    add_segment(
                        &mut segments,
                        LayerRect::from_floats(tl_inner.x, tl_inner.y, tr_inner.x, bl_inner.y),
                        TexelRect::new(px1, py1, px2, py2),
                        border.repeat_horizontal,
                        border.repeat_vertical
                    );
                }

                // Add edge segments.

                // Top
                add_segment(
                    &mut segments,
                    LayerRect::from_floats(tl_inner.x, tl_outer.y, tr_inner.x, tl_inner.y),
                    TexelRect::new(px1, py0, px2, py1),
                    border.repeat_horizontal,
                    RepeatMode::Stretch,
                );
                // Bottom
                add_segment(
                    &mut segments,
                    LayerRect::from_floats(bl_inner.x, bl_inner.y, br_inner.x, bl_outer.y),
                    TexelRect::new(px1, py2, px2, py3),
                    border.repeat_horizontal,
                    RepeatMode::Stretch,
                );
                // Left
                add_segment(
                    &mut segments,
                    LayerRect::from_floats(tl_outer.x, tl_inner.y, tl_inner.x, bl_inner.y),
                    TexelRect::new(px0, py1, px1, py2),
                    RepeatMode::Stretch,
                    border.repeat_vertical,
                );
                // Right
                add_segment(
                    &mut segments,
                    LayerRect::from_floats(tr_inner.x, tr_inner.y, br_outer.x, br_inner.y),
                    TexelRect::new(px2, py1, px3, py2),
                    RepeatMode::Stretch,
                    border.repeat_vertical,
                );

                for segment in segments {
                    let mut info = info.clone();
                    info.rect = segment.geom_rect;
                    self.add_image(
                        clip_and_scroll,
                        &info,
                        segment.stretch_size,
                        segment.tile_spacing,
                        Some(segment.sub_rect),
                        border.image_key,
                        ImageRendering::Auto,
                        AlphaType::PremultipliedAlpha,
                        None,
                    );
                }
            }
            BorderDetails::Normal(ref border) => {
                self.add_normal_border(info, border, &border_item.widths, clip_and_scroll);
            }
            BorderDetails::Gradient(ref border) => for segment in create_segments(border.outset) {
                let segment_rel = segment.origin - rect.origin;
                let mut info = info.clone();
                info.rect = segment;

                self.add_gradient(
                    clip_and_scroll,
                    &info,
                    border.gradient.start_point - segment_rel,
                    border.gradient.end_point - segment_rel,
                    gradient_stops,
                    gradient_stops_count,
                    border.gradient.extend_mode,
                    segment.size,
                    LayerSize::zero(),
                );
            },
            BorderDetails::RadialGradient(ref border) => {
                for segment in create_segments(border.outset) {
                    let segment_rel = segment.origin - rect.origin;
                    let mut info = info.clone();
                    info.rect = segment;

                    self.add_radial_gradient(
                        clip_and_scroll,
                        &info,
                        border.gradient.start_center - segment_rel,
                        border.gradient.start_radius,
                        border.gradient.end_center - segment_rel,
                        border.gradient.end_radius,
                        border.gradient.ratio_xy,
                        gradient_stops,
                        border.gradient.extend_mode,
                        segment.size,
                        LayerSize::zero(),
                    );
                }
            }
        }
    }

    fn add_gradient_impl(
        &mut self,
        clip_and_scroll: ScrollNodeAndClipChain,
        info: &LayerPrimitiveInfo,
        start_point: LayerPoint,
        end_point: LayerPoint,
        stops: ItemRange<GradientStop>,
        stops_count: usize,
        extend_mode: ExtendMode,
        gradient_index: CachedGradientIndex,
    ) {
        // Try to ensure that if the gradient is specified in reverse, then so long as the stops
        // are also supplied in reverse that the rendered result will be equivalent. To do this,
        // a reference orientation for the gradient line must be chosen, somewhat arbitrarily, so
        // just designate the reference orientation as start < end. Aligned gradient rendering
        // manages to produce the same result regardless of orientation, so don't worry about
        // reversing in that case.
        let reverse_stops = start_point.x > end_point.x ||
            (start_point.x == end_point.x && start_point.y > end_point.y);

        // To get reftests exactly matching with reverse start/end
        // points, it's necessary to reverse the gradient
        // line in some cases.
        let (sp, ep) = if reverse_stops {
            (end_point, start_point)
        } else {
            (start_point, end_point)
        };

        let prim = BrushPrimitive::new(
            BrushKind::LinearGradient {
                stops_range: stops,
                stops_count,
                extend_mode,
                reverse_stops,
                start_point: sp,
                end_point: ep,
                gradient_index,
            },
            None,
        );

        let prim = PrimitiveContainer::Brush(prim);

        self.add_primitive(clip_and_scroll, info, Vec::new(), prim);
    }

    pub fn add_gradient(
        &mut self,
        clip_and_scroll: ScrollNodeAndClipChain,
        info: &LayerPrimitiveInfo,
        start_point: LayerPoint,
        end_point: LayerPoint,
        stops: ItemRange<GradientStop>,
        stops_count: usize,
        extend_mode: ExtendMode,
        tile_size: LayerSize,
        tile_spacing: LayerSize,
    ) {
        let gradient_index = CachedGradientIndex(self.cached_gradients.len());
        self.cached_gradients.push(CachedGradient::new());

        let prim_infos = info.decompose(
            tile_size,
            tile_spacing,
            64 * 64,
        );

        if prim_infos.is_empty() {
            self.add_gradient_impl(
                clip_and_scroll,
                info,
                start_point,
                end_point,
                stops,
                stops_count,
                extend_mode,
                gradient_index,
            );
        } else {
            for prim_info in prim_infos {
                self.add_gradient_impl(
                    clip_and_scroll,
                    &prim_info,
                    start_point,
                    end_point,
                    stops,
                    stops_count,
                    extend_mode,
                    gradient_index,
                );
            }
        }
    }

    fn add_radial_gradient_impl(
        &mut self,
        clip_and_scroll: ScrollNodeAndClipChain,
        info: &LayerPrimitiveInfo,
        start_center: LayerPoint,
        start_radius: f32,
        end_center: LayerPoint,
        end_radius: f32,
        ratio_xy: f32,
        stops: ItemRange<GradientStop>,
        extend_mode: ExtendMode,
        gradient_index: CachedGradientIndex,
    ) {
        let prim = BrushPrimitive::new(
            BrushKind::RadialGradient {
                stops_range: stops,
                extend_mode,
                start_center,
                end_center,
                start_radius,
                end_radius,
                ratio_xy,
                gradient_index,
            },
            None,
        );

        self.add_primitive(
            clip_and_scroll,
            info,
            Vec::new(),
            PrimitiveContainer::Brush(prim),
        );
    }

    pub fn add_radial_gradient(
        &mut self,
        clip_and_scroll: ScrollNodeAndClipChain,
        info: &LayerPrimitiveInfo,
        start_center: LayerPoint,
        start_radius: f32,
        end_center: LayerPoint,
        end_radius: f32,
        ratio_xy: f32,
        stops: ItemRange<GradientStop>,
        extend_mode: ExtendMode,
        tile_size: LayerSize,
        tile_spacing: LayerSize,
    ) {
        let gradient_index = CachedGradientIndex(self.cached_gradients.len());
        self.cached_gradients.push(CachedGradient::new());

        let prim_infos = info.decompose(
            tile_size,
            tile_spacing,
            64 * 64,
        );

        if prim_infos.is_empty() {
            self.add_radial_gradient_impl(
                clip_and_scroll,
                info,
                start_center,
                start_radius,
                end_center,
                end_radius,
                ratio_xy,
                stops,
                extend_mode,
                gradient_index,
            );
        } else {
            for prim_info in prim_infos {
                self.add_radial_gradient_impl(
                    clip_and_scroll,
                    &prim_info,
                    start_center,
                    start_radius,
                    end_center,
                    end_radius,
                    ratio_xy,
                    stops,
                    extend_mode,
                    gradient_index,
                );
            }
        }
    }

    pub fn add_text(
        &mut self,
        clip_and_scroll: ScrollNodeAndClipChain,
        run_offset: LayoutVector2D,
        info: &LayerPrimitiveInfo,
        font_instance_key: &FontInstanceKey,
        text_color: &ColorF,
        glyph_range: ItemRange<GlyphInstance>,
        glyph_count: usize,
        glyph_options: Option<GlyphOptions>,
    ) {
        let prim = {
            let instance_map = self.font_instances.read().unwrap();
            let font_instance = match instance_map.get(&font_instance_key) {
                Some(instance) => instance,
                None => {
                    warn!("Unknown font instance key");
                    debug!("key={:?}", font_instance_key);
                    return;
                }
            };

            // Trivial early out checks
            if font_instance.size.0 <= 0 {
                return;
            }

            // Sanity check - anything with glyphs bigger than this
            // is probably going to consume too much memory to render
            // efficiently anyway. This is specifically to work around
            // the font_advance.html reftest, which creates a very large
            // font as a crash test - the rendering is also ignored
            // by the azure renderer.
            if font_instance.size >= Au::from_px(4096) {
                return;
            }

            // TODO(gw): Use a proper algorithm to select
            // whether this item should be rendered with
            // subpixel AA!
            let mut render_mode = self.config
                .default_font_render_mode
                .limit_by(font_instance.render_mode);
            let mut flags = font_instance.flags;
            if let Some(options) = glyph_options {
                render_mode = render_mode.limit_by(options.render_mode);
                flags |= options.flags;
            }

            // There are some conditions under which we can't use
            // subpixel text rendering, even if enabled.
            if render_mode == FontRenderMode::Subpixel {
                // text on a picture that has filters
                // (e.g. opacity) can't use sub-pixel.
                // TODO(gw): It's possible we can relax this in
                //           the future, if we modify the way
                //           we handle subpixel blending.
                if let Some(ref stacking_context) = self.sc_stack.last() {
                    if !stacking_context.allow_subpixel_aa {
                        render_mode = FontRenderMode::Alpha;
                    }
                }
            }

            let prim_font = FontInstance::new(
                font_instance.font_key,
                font_instance.size,
                *text_color,
                font_instance.bg_color,
                render_mode,
                font_instance.subpx_dir,
                flags,
                font_instance.platform_options,
                font_instance.variations.clone(),
            );
            TextRunPrimitiveCpu {
                font: prim_font,
                glyph_range,
                glyph_count,
                glyph_gpu_blocks: Vec::new(),
                glyph_keys: Vec::new(),
                offset: run_offset,
                shadow: false,
            }
        };

        // Text shadows that have a blur radius of 0 need to be rendered as normal
        // text elements to get pixel perfect results for reftests. It's also a big
        // performance win to avoid blurs and render target allocations where
        // possible. For any text shadows that have zero blur, create a normal text
        // primitive with the shadow's color and offset. These need to be added
        // *before* the visual text primitive in order to get the correct paint
        // order. Store them in a Vec first to work around borrowck issues.
        // TODO(gw): Refactor to avoid having to store them in a Vec first.
        let mut fast_shadow_prims = Vec::new();
        for (idx, &(shadow_prim_index, _)) in self.shadow_prim_stack.iter().enumerate() {
            let shadow_metadata = &self.prim_store.cpu_metadata[shadow_prim_index.0];
            let picture_prim = &self.prim_store.cpu_pictures[shadow_metadata.cpu_prim_index.0];
            match picture_prim.kind {
                PictureKind::TextShadow { offset, color, blur_radius, .. } if blur_radius == 0.0 => {
                    let mut text_prim = prim.clone();
                    text_prim.font.color = color.into();
                    text_prim.shadow = true;
                    text_prim.offset += offset;
                    fast_shadow_prims.push((idx, text_prim));
                }
                _ => {}
            }
        }

        for (idx, text_prim) in fast_shadow_prims {
            let rect = info.rect;
            let mut info = info.clone();
            info.rect = rect.translate(&text_prim.offset);
            info.local_clip =
              LocalClip::from(info.local_clip.clip_rect().translate(&text_prim.offset));
            let prim_index = self.create_primitive(
                &info,
                Vec::new(),
                PrimitiveContainer::TextRun(text_prim),
            );
            self.shadow_prim_stack[idx].1.push((prim_index, clip_and_scroll));
        }

        // Create (and add to primitive store) the primitive that will be
        // used for both the visual element and also the shadow(s).
        let prim_index = self.create_primitive(
            info,
            Vec::new(),
            PrimitiveContainer::TextRun(prim),
        );

        // Only add a visual element if it can contribute to the scene.
        if text_color.a > 0.0 {
            if self.shadow_prim_stack.is_empty() {
                self.add_primitive_to_hit_testing_list(info, clip_and_scroll);
                self.add_primitive_to_draw_list(prim_index, clip_and_scroll);
            } else {
                self.pending_shadow_contents.push((prim_index, clip_and_scroll, *info));
            }
        }

        // Now add this primitive index to all the currently active text shadow
        // primitives. Although we're adding the indices *after* the visual
        // primitive here, they will still draw before the visual text, since
        // the shadow primitive itself has been added to the draw cmd
        // list *before* the visual element, during push_shadow. We need
        // the primitive index of the visual element here before we can add
        // the indices as sub-primitives to the shadow primitives.
        for &(shadow_prim_index, _) in &self.shadow_prim_stack {
            let shadow_metadata = &mut self.prim_store.cpu_metadata[shadow_prim_index.0];
            debug_assert_eq!(shadow_metadata.prim_kind, PrimitiveKind::Picture);
            let picture =
                &mut self.prim_store.cpu_pictures[shadow_metadata.cpu_prim_index.0];

            match picture.kind {
                // Only run real blurs here (fast path zero blurs are handled above).
                PictureKind::TextShadow { blur_radius, .. } if blur_radius > 0.0 => {
                    picture.add_primitive(
                        prim_index,
                        clip_and_scroll,
                    );
                }
                _ => {}
            }
        }
    }

    pub fn add_image(
        &mut self,
        clip_and_scroll: ScrollNodeAndClipChain,
        info: &LayerPrimitiveInfo,
        stretch_size: LayerSize,
        mut tile_spacing: LayerSize,
        sub_rect: Option<TexelRect>,
        image_key: ImageKey,
        image_rendering: ImageRendering,
        alpha_type: AlphaType,
        tile_offset: Option<TileOffset>,
    ) {
        // If the tile spacing is the same as the rect size,
        // then it is effectively zero. We use this later on
        // in prim_store to detect if an image can be considered
        // opaque.
        if tile_spacing == info.rect.size {
            tile_spacing = LayerSize::zero();
        }

        let request = ImageRequest {
            key: image_key,
            rendering: image_rendering,
            tile: tile_offset,
        };

        // See if conditions are met to run through the new
        // image brush shader, which supports segments.
        if tile_spacing == LayerSize::zero() &&
           stretch_size == info.rect.size &&
           sub_rect.is_none() &&
           tile_offset.is_none() {
            let prim = BrushPrimitive::new(
                BrushKind::Image {
                    request,
                    current_epoch: Epoch::invalid(),
                    alpha_type,
                },
                None,
            );

            self.add_primitive(
                clip_and_scroll,
                info,
                Vec::new(),
                PrimitiveContainer::Brush(prim),
            );
        } else {
            let prim_cpu = ImagePrimitiveCpu {
                tile_spacing,
                alpha_type,
                stretch_size,
                current_epoch: Epoch::invalid(),
                source: ImageSource::Default,
                key: ImageCacheKey {
                    request,
                    texel_rect: sub_rect.map(|texel_rect| {
                        DeviceIntRect::new(
                            DeviceIntPoint::new(
                                texel_rect.uv0.x as i32,
                                texel_rect.uv0.y as i32,
                            ),
                            DeviceIntSize::new(
                                (texel_rect.uv1.x - texel_rect.uv0.x) as i32,
                                (texel_rect.uv1.y - texel_rect.uv0.y) as i32,
                            ),
                        )
                    }),
                },
            };

            self.add_primitive(
                clip_and_scroll,
                info,
                Vec::new(),
                PrimitiveContainer::Image(prim_cpu),
            );
        }
    }

    pub fn add_yuv_image(
        &mut self,
        clip_and_scroll: ScrollNodeAndClipChain,
        info: &LayerPrimitiveInfo,
        yuv_data: YuvData,
        color_space: YuvColorSpace,
        image_rendering: ImageRendering,
    ) {
        let format = yuv_data.get_format();
        let yuv_key = match yuv_data {
            YuvData::NV12(plane_0, plane_1) => [plane_0, plane_1, ImageKey::DUMMY],
            YuvData::PlanarYCbCr(plane_0, plane_1, plane_2) => [plane_0, plane_1, plane_2],
            YuvData::InterleavedYCbCr(plane_0) => [plane_0, ImageKey::DUMMY, ImageKey::DUMMY],
        };

        let prim = BrushPrimitive::new(
            BrushKind::YuvImage {
                yuv_key,
                format,
                color_space,
                image_rendering,
            },
            None,
        );

        self.add_primitive(
            clip_and_scroll,
            info,
            Vec::new(),
            PrimitiveContainer::Brush(prim),
        );
    }
}

pub fn build_scene(config: &FrameBuilderConfig, request: SceneRequest) -> BuiltScene {
    // TODO: mutably pass the scene and update its own pipeline epoch map instead of
    // creating a new one here.
    let mut pipeline_epoch_map = FastHashMap::default();
    let mut clip_scroll_tree = ClipScrollTree::new();

    let frame_builder = DisplayListFlattener::create_frame_builder(
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

    let mut scene = request.scene;
    scene.pipeline_epochs = pipeline_epoch_map;

    BuiltScene {
        scene,
        frame_builder,
        clip_scroll_tree,
        removed_pipelines: request.removed_pipelines,
    }
}

trait PrimitiveInfoTiler {
    fn decompose(
        &self,
        tile_size: LayerSize,
        tile_spacing: LayerSize,
        max_prims: usize,
    ) -> Vec<LayerPrimitiveInfo>;
}

impl PrimitiveInfoTiler for LayerPrimitiveInfo {
    fn decompose(
        &self,
        tile_size: LayerSize,
        tile_spacing: LayerSize,
        max_prims: usize,
    ) -> Vec<LayerPrimitiveInfo> {
        let mut prims = Vec::new();
        let tile_repeat = tile_size + tile_spacing;

        if tile_repeat.width <= 0.0 ||
           tile_repeat.height <= 0.0 {
            return prims;
        }

        if tile_repeat.width < self.rect.size.width ||
           tile_repeat.height < self.rect.size.height {
            let local_clip = self.local_clip.clip_by(&self.rect);
            let rect_p0 = self.rect.origin;
            let rect_p1 = self.rect.bottom_right();

            let mut y0 = rect_p0.y;
            while y0 < rect_p1.y {
                let mut x0 = rect_p0.x;

                while x0 < rect_p1.x {
                    prims.push(LayerPrimitiveInfo {
                        rect: LayerRect::new(
                            LayerPoint::new(x0, y0),
                            tile_size,
                        ),
                        local_clip,
                        is_backface_visible: self.is_backface_visible,
                        tag: self.tag,
                    });

                    // Mostly a safety against a crazy number of primitives
                    // being generated. If we exceed that amount, just bail
                    // out and only draw the maximum amount.
                    if prims.len() > max_prims {
                        warn!("too many prims found due to repeat/tile. dropping extra prims!");
                        return prims;
                    }

                    x0 += tile_repeat.width;
                }

                y0 += tile_repeat.height;
            }
        }

        prims
    }
}

/// Properties of a stacking context that are maintained
/// during creation of the scene. These structures are
/// not persisted after the initial scene build.
struct FlattenedStackingContext {
    /// Pipeline this stacking context belongs to.
    pipeline_id: PipelineId,

    /// Filters / mix-blend-mode effects
    composite_ops: CompositeOps,

    /// If true, visible when backface is visible.
    is_backface_visible: bool,

    /// Allow subpixel AA for text runs on this stacking context.
    /// This is a temporary hack while we don't support subpixel AA
    /// on transparent stacking contexts.
    allow_subpixel_aa: bool,

    /// CSS transform-style property.
    transform_style: TransformStyle,

    /// If Some(..), this stacking context establishes a new
    /// 3d rendering context, and the value is the primitive
    // index of the 3d context container.
    rendering_context_3d_prim_index: Option<PrimitiveIndex>,
}

#[derive(Debug)]
pub struct ScrollbarInfo(pub ClipScrollNodeIndex, pub LayerRect);
