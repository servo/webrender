use batch::MAX_MATRICES_PER_BATCH;
use device::{TextureId};
use euclid::{Rect, Point2D, Size2D, Matrix4};
use fnv::FnvHasher;
use internal_types::{BlurDirection, LowLevelFilterOp, CompositionOp, DrawListItemIndex};
use internal_types::{BatchUpdateList, DrawListId, TextureTarget};
use internal_types::{RendererFrame, DrawListContext, BatchInfo, DrawCall};
use internal_types::{BatchUpdate, BatchUpdateOp, DrawLayer};
use internal_types::{DrawCommand, ClearInfo, CompositeInfo};
use layer::Layer;
use node_compiler::NodeCompiler;
use resource_cache::ResourceCache;
use resource_list::BuildRequiredResources;
use scene::{SceneStackingContext, ScenePipeline, Scene, SceneItem, SpecificSceneItem};
use scoped_threadpool;
use std::collections::HashMap;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::hash_state::DefaultState;
use std::mem;
use util;
use util::MatrixHelpers;
use webrender_traits::{PipelineId, Epoch, ScrollPolicy, ScrollLayerId, StackingContext};
use webrender_traits::{FilterOp, ImageFormat, MixBlendMode, StackingLevel};

#[derive(Clone, Copy, Debug, Ord, PartialOrd, PartialEq, Eq, Hash)]
pub struct RenderTargetIndex(pub u32);

#[derive(Debug)]
pub struct DrawListBatchInfo {
    pub scroll_layer_id: ScrollLayerId,
    pub draw_lists: Vec<DrawListId>,
}

#[derive(Debug)]
pub enum FrameRenderItem {
    Clear(ClearInfo),
    Composite(CompositeInfo),
    DrawListBatch(DrawListBatchInfo),
}

#[derive(Debug)]
pub struct FrameRenderTarget {
    pub size: Size2D<u32>,
    pub texture_id: Option<TextureId>,
    pub items: Vec<FrameRenderItem>,
    current_batch: Option<DrawListBatchInfo>,
}

impl FrameRenderTarget {
    pub fn new(size: Size2D<u32>,
           texture_id: Option<TextureId>) -> FrameRenderTarget {
        FrameRenderTarget {
            size: size,
            items: Vec::new(),
            texture_id: texture_id,
            current_batch: None,
        }
    }

    fn finalize(&mut self) {
        self.flush();
    }

    fn flush(&mut self) {
        if let Some(batch) = self.current_batch.take() {
            self.items.push(FrameRenderItem::DrawListBatch(batch));
        }
    }

    fn push_clear(&mut self, clear_info: ClearInfo) {
        self.flush();
        self.items.push(FrameRenderItem::Clear(clear_info));
    }

    fn push_composite(&mut self, composite_info: CompositeInfo) {
        self.flush();
        self.items.push(FrameRenderItem::Composite(composite_info));
    }

    fn push_draw_list(&mut self,
                      draw_list_id: DrawListId,
                      scroll_layer_id: ScrollLayerId) {
        let need_new_batch = match self.current_batch {
            Some(ref batch) => {
                batch.scroll_layer_id != scroll_layer_id ||
                batch.draw_lists.len() == MAX_MATRICES_PER_BATCH
            }
            None => {
                true
            }
        };

        if need_new_batch {
            self.flush();

            self.current_batch = Some(DrawListBatchInfo {
                scroll_layer_id: scroll_layer_id,
                draw_lists: Vec::new(),
            });
        }

        self.current_batch.as_mut().unwrap().draw_lists.push(draw_list_id);
    }
}

pub struct Frame {
    pub layers: HashMap<ScrollLayerId, Layer, DefaultState<FnvHasher>>,
    pub pipeline_epoch_map: HashMap<PipelineId, Epoch, DefaultState<FnvHasher>>,
    pub render_targets: Vec<FrameRenderTarget>,
    pub render_target_stack: Vec<RenderTargetIndex>,
    pub pending_updates: BatchUpdateList,
}

enum SceneItemKind<'a> {
    StackingContext(&'a SceneStackingContext),
    Pipeline(&'a ScenePipeline)
}

#[derive(Clone)]
struct SceneItemWithZOrder {
    item: SceneItem,
    z_index: i32,
}

impl<'a> SceneItemKind<'a> {
    fn collect_scene_items(&self, scene: &Scene) -> Vec<SceneItem> {
        let mut background_and_borders = Vec::new();
        let mut positioned_content = Vec::new();
        let mut block_background_and_borders = Vec::new();
        let mut floats = Vec::new();
        let mut content = Vec::new();
        let mut outlines = Vec::new();

        let stacking_context = match *self {
            SceneItemKind::StackingContext(stacking_context) => {
                &stacking_context.stacking_context
            }
            SceneItemKind::Pipeline(pipeline) => {
                if let Some(background_draw_list) = pipeline.background_draw_list {
                    background_and_borders.push(SceneItem {
                        stacking_level: StackingLevel::BackgroundAndBorders,
                        specific: SpecificSceneItem::DrawList(background_draw_list),
                    });
                }

                &scene.stacking_context_map
                      .get(&pipeline.root_stacking_context_id)
                      .unwrap()
                      .stacking_context
            }
        };

        for display_list_id in &stacking_context.display_lists {
            let display_list = &scene.display_list_map[display_list_id];
            for item in &display_list.items {
                match item.stacking_level {
                    StackingLevel::BackgroundAndBorders => {
                        background_and_borders.push(item.clone());
                    }
                    StackingLevel::BlockBackgroundAndBorders => {
                        block_background_and_borders.push(item.clone());
                    }
                    StackingLevel::PositionedContent => {
                        let z_index = match item.specific {
                            SpecificSceneItem::StackingContext(id) => {
                                scene.stacking_context_map
                                     .get(&id)
                                     .unwrap()
                                     .stacking_context
                                     .z_index
                            }
                            SpecificSceneItem::DrawList(..) |
                            SpecificSceneItem::Iframe(..) => {
                                // TODO(gw): Probably wrong for an iframe?
                                0
                            }
                        };

                        positioned_content.push(SceneItemWithZOrder {
                            item: item.clone(),
                            z_index: z_index,
                        });
                    }
                    StackingLevel::Floats => {
                        floats.push(item.clone());
                    }
                    StackingLevel::Content => {
                        content.push(item.clone());
                    }
                    StackingLevel::Outlines => {
                        outlines.push(item.clone());
                    }
                }
            }
        }

        positioned_content.sort_by(|a, b| {
            a.z_index.cmp(&b.z_index)
        });

        let mut result = Vec::new();
        result.extend(background_and_borders);
        result.extend(positioned_content.iter().filter_map(|item| {
            if item.z_index < 0 {
                Some(item.item.clone())
            } else {
                None
            }
        }));
        result.extend(block_background_and_borders);
        result.extend(floats);
        result.extend(content);
        result.extend(positioned_content.iter().filter_map(|item| {
            if item.z_index < 0 {
                None
            } else {
                Some(item.item.clone())
            }
        }));
        result.extend(outlines);
        result
    }
}

trait StackingContextHelpers {
    fn needs_composition_operation_for_mix_blend_mode(&self) -> bool;
}

impl StackingContextHelpers for StackingContext {
    fn needs_composition_operation_for_mix_blend_mode(&self) -> bool {
        match self.mix_blend_mode {
            MixBlendMode::Normal => false,
            MixBlendMode::Multiply |
            MixBlendMode::Screen |
            MixBlendMode::Overlay |
            MixBlendMode::Darken |
            MixBlendMode::Lighten |
            MixBlendMode::ColorDodge |
            MixBlendMode::ColorBurn |
            MixBlendMode::HardLight |
            MixBlendMode::SoftLight |
            MixBlendMode::Difference |
            MixBlendMode::Exclusion |
            MixBlendMode::Hue |
            MixBlendMode::Saturation |
            MixBlendMode::Color |
            MixBlendMode::Luminosity => true,
        }
    }
}

impl Frame {
    pub fn new() -> Frame {
        Frame {
            layers: HashMap::with_hash_state(Default::default()),
            pipeline_epoch_map: HashMap::with_hash_state(Default::default()),
            render_targets: Vec::new(),
            render_target_stack: Vec::new(),
            pending_updates: BatchUpdateList::new(),
        }
    }

    pub fn reset(&mut self,
             resource_cache: &mut ResourceCache) -> HashMap<ScrollLayerId, Layer, DefaultState<FnvHasher>> {
        self.pipeline_epoch_map.clear();

        // Free any render targets from last frame.
        // TODO: This should really re-use existing targets here...
        for render_target in self.render_targets.drain(..) {
            if let Some(texture_id) = render_target.texture_id {
                resource_cache.free_render_target(texture_id);
            }
        }

        mem::replace(&mut self.layers, HashMap::with_hash_state(Default::default()))
    }

    pub fn pending_updates(&mut self) -> BatchUpdateList {
        mem::replace(&mut self.pending_updates, BatchUpdateList::new())
    }

    pub fn scroll(&mut self, delta: &Point2D<f32>, viewport_size: &Size2D<f32>) {
        // TODO: Select other layers for scrolling!
        let layer = self.layers.get_mut(&ScrollLayerId(0));

        if let Some(layer) = layer {
            layer.scroll_offset = layer.scroll_offset + *delta;

            layer.scroll_offset.x = layer.scroll_offset.x.min(0.0);
            layer.scroll_offset.y = layer.scroll_offset.y.min(0.0);

            layer.scroll_offset.x = layer.scroll_offset.x.max(-layer.scroll_boundaries.width +
                                                              viewport_size.width);
            layer.scroll_offset.y = layer.scroll_offset.y.max(-layer.scroll_boundaries.height +
                                                              viewport_size.height);
        } else {
            println!("unable to find root scroll layer (may be an empty stacking context)");
        }
    }

    pub fn create(&mut self,
              scene: &Scene,
              viewport_size: Size2D<u32>,
              resource_cache: &mut ResourceCache) {
        if let Some(root_pipeline_id) = scene.root_pipeline_id {
            if let Some(root_pipeline) = scene.pipeline_map.get(&root_pipeline_id) {
                let mut old_layers = self.reset(resource_cache);

                let root_stacking_context = scene.stacking_context_map
                                                 .get(&root_pipeline.root_stacking_context_id)
                                                 .unwrap();

                let root_scroll_layer_id = root_stacking_context.stacking_context
                                                                .scroll_layer_id
                                                                .expect("root layer must be a scroll layer!");

                debug_assert!(self.render_target_stack.len() == 0);
                self.push_render_target(viewport_size, None);
                self.flatten(SceneItemKind::Pipeline(root_pipeline),
                             &Point2D::zero(),
                             &Matrix4::identity(),
                             &Matrix4::identity(),
                             root_scroll_layer_id,
                             resource_cache,
                             &root_stacking_context.stacking_context.overflow,
                             scene,
                             &old_layers,
                             &root_stacking_context.stacking_context.overflow);
                self.pop_render_target();
                debug_assert!(self.render_target_stack.len() == 0);

                // TODO(gw): This should be moved elsewhere!
                if let Some(root_scroll_layer) = self.layers.get_mut(&root_scroll_layer_id) {
                    root_scroll_layer.scroll_boundaries = root_stacking_context.stacking_context.overflow.size;
                }

                for (_, old_layer) in &mut old_layers {
                    old_layer.reset(&mut self.pending_updates)
                }
            }
        }
    }

    pub fn flatten(&mut self,
                   item_kind: SceneItemKind,
                   parent_offset: &Point2D<f32>,
                   parent_transform: &Matrix4,
                   parent_perspective: &Matrix4,
                   parent_scroll_layer: ScrollLayerId,
                   resource_cache: &mut ResourceCache,
                   clip_rect: &Rect<f32>,
                   scene: &Scene,
                   old_layers: &HashMap<ScrollLayerId, Layer, DefaultState<FnvHasher>>,
                   scene_rect: &Rect<f32>) {
        let _pf = util::ProfileScope::new("  flatten");

        let stacking_context = match item_kind {
            SceneItemKind::StackingContext(stacking_context) => {
                &stacking_context.stacking_context
            }
            SceneItemKind::Pipeline(pipeline) => {
                self.pipeline_epoch_map.insert(pipeline.pipeline_id, pipeline.epoch);

                &scene.stacking_context_map
                      .get(&pipeline.root_stacking_context_id)
                      .unwrap()
                      .stacking_context
            }
        };

        let (this_scroll_layer, parent_scroll_layer) = match stacking_context.scroll_policy {
            ScrollPolicy::Scrollable => {
                let scroll_layer = stacking_context.scroll_layer_id.unwrap_or(parent_scroll_layer);
                (scroll_layer, scroll_layer)
            }
            ScrollPolicy::Fixed => {
                debug_assert!(stacking_context.scroll_layer_id.is_none());
                (ScrollLayerId::fixed_layer(), parent_scroll_layer)
            }
        };

        // TODO: Account for scroll offset with transforms!

        // Build world space transform
        let origin = &stacking_context.bounds.origin;
        let child_offset = *parent_offset + *origin;
        let local_transform = Matrix4::identity().translate(origin.x, origin.y, 0.0)
                                                 .mul(&stacking_context.transform);

        let mut final_transform = parent_perspective.mul(&parent_transform)
                                                    .mul(&local_transform);

        // Build world space perspective transform
        let perspective_transform = Matrix4::identity().translate(origin.x, origin.y, 0.0)
                                                       .mul(&stacking_context.perspective)
                                                       .translate(-origin.x, -origin.y, 0.0);

        let overflow = clip_rect.translate(&stacking_context.overflow.origin)
                                .intersection(&stacking_context.overflow);

        if let Some(overflow) = overflow {
            // When establishing a new 3D context, clear Z. This is only needed if there
            // are child stacking contexts, otherwise it is a redundant clear.
            if stacking_context.establishes_3d_context &&
               stacking_context.has_stacking_contexts {
                self.push_clear(ClearInfo {
                    clear_color: false,
                    clear_z: true,
                    clear_stencil: true,
                });
            }

            let mut composition_operations = vec![];
            if stacking_context.needs_composition_operation_for_mix_blend_mode() {
                composition_operations.push(CompositionOp::MixBlend(stacking_context.mix_blend_mode));
            }
            for filter in stacking_context.filters.iter() {
                match *filter {
                    FilterOp::Blur(radius) => {
                        composition_operations.push(CompositionOp::Filter(LowLevelFilterOp::Blur(
                            radius,
                            BlurDirection::Horizontal)));
                        composition_operations.push(CompositionOp::Filter(LowLevelFilterOp::Blur(
                            radius,
                            BlurDirection::Vertical)));
                    }
                    FilterOp::Brightness(amount) => {
                        composition_operations.push(CompositionOp::Filter(LowLevelFilterOp::Brightness(
                            amount)));
                    }
                    FilterOp::Contrast(amount) => {
                        composition_operations.push(CompositionOp::Filter(LowLevelFilterOp::Contrast(
                            amount)));
                    }
                    FilterOp::Grayscale(amount) => {
                        composition_operations.push(CompositionOp::Filter(LowLevelFilterOp::Grayscale(
                            amount)));
                    }
                    FilterOp::HueRotate(angle) => {
                        composition_operations.push(CompositionOp::Filter(LowLevelFilterOp::HueRotate(
                            angle)));
                    }
                    FilterOp::Invert(amount) => {
                        composition_operations.push(CompositionOp::Filter(LowLevelFilterOp::Invert(
                            amount)));
                    }
                    FilterOp::Opacity(amount) => {
                        composition_operations.push(CompositionOp::Filter(LowLevelFilterOp::Opacity(
                            amount)));
                    }
                    FilterOp::Saturate(amount) => {
                        composition_operations.push(CompositionOp::Filter(LowLevelFilterOp::Saturate(
                            amount)));
                    }
                    FilterOp::Sepia(amount) => {
                        composition_operations.push(CompositionOp::Filter(LowLevelFilterOp::Sepia(
                            amount)));
                    }
                }
            }

            for composition_operation in composition_operations.iter() {
                let size = Size2D::new(stacking_context.overflow.size.width as u32,
                                       stacking_context.overflow.size.height as u32);
                // TODO(gw): Get composition ops working with transforms
                let origin = Point2D::new(child_offset.x as u32, child_offset.y as u32);

                let texture_id = resource_cache.allocate_render_target(TextureTarget::Texture2D,
                                                                       size.width,
                                                                       size.height,
                                                                       1,
                                                                       ImageFormat::RGBA8);

                self.push_composite(CompositeInfo {
                    operation: *composition_operation,
                    color_texture_id: texture_id,
                    rect: Rect::new(origin, size),
                });

                self.push_render_target(size, Some(texture_id));
                final_transform = Matrix4::identity();
            }

            let scene_items = item_kind.collect_scene_items(scene);

            for item in scene_items {
                match item.specific {
                    SpecificSceneItem::DrawList(draw_list_id) => {
                        self.push_draw_list(draw_list_id, this_scroll_layer);

                        let layer = match self.layers.entry(this_scroll_layer) {
                            Occupied(entry) => {
                                entry.into_mut()
                            }
                            Vacant(entry) => {
                                let scroll_offset = match old_layers.get(&this_scroll_layer) {
                                    Some(ref old_layer) => old_layer.scroll_offset,
                                    None => Point2D::zero(),
                                };

                                entry.insert(Layer::new(scene_rect, &scroll_offset))
                            }
                        };

                        let draw_list = resource_cache.get_draw_list_mut(draw_list_id);

                        // Store draw context
                        draw_list.context = Some(DrawListContext {
                            origin: child_offset,
                            overflow: overflow,
                            final_transform: final_transform,
                        });

                        for (item_index, item) in draw_list.items.iter().enumerate() {
                            // Node index may already be Some(..). This can occur when a page has iframes
                            // and a new root stacking context is received. In this case, the node index
                            // may already be set for draw lists from other iframe(s) that weren't updated
                            // as part of this new stacking context.
                            let item_index = DrawListItemIndex(item_index as u32);
                            let rect = final_transform.transform_rect(&item.rect);
                            layer.insert(&rect, draw_list_id, item_index);
                        }
                    }
                    SpecificSceneItem::StackingContext(id) => {
                        let stacking_context = scene.stacking_context_map
                                                    .get(&id)
                                                    .unwrap();

                        let clip_rect = clip_rect.translate(&-child_offset);

                        self.flatten(SceneItemKind::StackingContext(stacking_context),
                                     &child_offset,
                                     &final_transform,
                                     &perspective_transform,
                                     parent_scroll_layer,
                                     resource_cache,
                                     &clip_rect,
                                     scene,
                                     old_layers,
                                     scene_rect);
                    }
                    SpecificSceneItem::Iframe(iframe_info) => {
                        let pipeline = scene.pipeline_map
                                            .get(&iframe_info.id);

                        if let Some(pipeline) = pipeline {
                            // TODO: Doesn't handle transforms on iframes yet!
                            let child_offset = child_offset + iframe_info.offset;

                            let iframe_transform = Matrix4::identity().translate(child_offset.x,
                                                                                 child_offset.y,
                                                                                 0.0);

                            let clip_rect = clip_rect.intersection(&iframe_info.clip);

                            if let Some(clip_rect) = clip_rect {
                                let clip_rect = clip_rect.translate(&-iframe_info.offset);
                                self.flatten(SceneItemKind::Pipeline(pipeline),
                                             &child_offset,
                                             &iframe_transform,
                                             &perspective_transform,
                                             parent_scroll_layer,
                                             resource_cache,
                                             &clip_rect,
                                             scene,
                                             old_layers,
                                             scene_rect);
                            }
                        }
                    }
                }
            }

            for _ in composition_operations.iter() {
                self.pop_render_target();
            }
        }
    }

    pub fn push_render_target(&mut self,
                          size: Size2D<u32>,
                          texture_id: Option<TextureId>) {
        let rt_index = RenderTargetIndex(self.render_targets.len() as u32);
        self.render_target_stack.push(rt_index);

        let render_target = FrameRenderTarget::new(size, texture_id);
        self.render_targets.push(render_target);
    }

    #[inline]
    fn push_clear(&mut self, clear_info: ClearInfo) {
        self.current_render_target().push_clear(clear_info);
    }

    #[inline]
    fn push_composite(&mut self, composite_info: CompositeInfo) {
        self.current_render_target().push_composite(composite_info);
    }

    #[inline]
    fn push_draw_list(&mut self,
                      draw_list_id: DrawListId,
                      scroll_layer_id: ScrollLayerId) {
        self.current_render_target().push_draw_list(draw_list_id,
                                                    scroll_layer_id);
    }

    fn current_render_target(&mut self) -> &mut FrameRenderTarget {
        let RenderTargetIndex(index) = *self.render_target_stack.last().unwrap();
        &mut self.render_targets[index as usize]
    }

    pub fn pop_render_target(&mut self) {
        self.current_render_target().finalize();
        self.render_target_stack.pop().unwrap();
    }

    pub fn build(&mut self,
             viewport: &Rect<i32>,
             resource_cache: &mut ResourceCache,
             thread_pool: &mut scoped_threadpool::Pool,
             device_pixel_ratio: f32) -> RendererFrame {
        let origin = Point2D::new(viewport.origin.x as f32, viewport.origin.y as f32);
        let size = Size2D::new(viewport.size.width as f32, viewport.size.height as f32);
        let viewport_rect = Rect::new(origin, size);

        // Traverse layer trees to calculate visible nodes
        for (_, layer) in &mut self.layers {
            layer.cull(&viewport_rect);
        }

        // Build resource list for newly visible nodes
        self.update_resource_lists(resource_cache, thread_pool);

        // Update texture cache and build list of raster jobs.
        self.update_texture_cache_and_build_raster_jobs(resource_cache);

        // Rasterize needed glyphs on worker threads
        self.raster_glyphs(thread_pool,
                           resource_cache);

        // Compile nodes that have become visible
        self.compile_visible_nodes(thread_pool,
                                   resource_cache,
                                   device_pixel_ratio);

        // Update the batch cache from newly compiled nodes
        self.update_batch_cache();

        // Collect the visible batches into a frame
        self.collect_and_sort_visible_batches(resource_cache)
    }

    pub fn update_resource_lists(&mut self,
                             resource_cache: &ResourceCache,
                             thread_pool: &mut scoped_threadpool::Pool) {
        let _pf = util::ProfileScope::new("  update_resource_lists");

        for (_, layer) in &mut self.layers {
            let nodes = &mut layer.aabb_tree.nodes;

            thread_pool.scoped(|scope| {
                for node in nodes {
                    if node.is_visible && node.compiled_node.is_none() {
                        scope.execute(move || {
                            node.build_resource_list(resource_cache);
                        });
                    }
                }
            });
        }
    }

    pub fn update_texture_cache_and_build_raster_jobs(&mut self, resource_cache: &mut ResourceCache) {
        let _pf = util::ProfileScope::new("  update_texture_cache_and_build_raster_jobs");

        for (_, layer) in &self.layers {
            for node in &layer.aabb_tree.nodes {
                if node.is_visible {
                    let resource_list = node.resource_list.as_ref().unwrap();
                    resource_cache.add_resource_list(resource_list);
                }
            }
        }
    }

    pub fn raster_glyphs(&mut self,
                     thread_pool: &mut scoped_threadpool::Pool,
                     resource_cache: &mut ResourceCache) {
        let _pf = util::ProfileScope::new("  raster_glyphs");
        resource_cache.raster_pending_glyphs(thread_pool);
    }

    pub fn compile_visible_nodes(&mut self,
                             thread_pool: &mut scoped_threadpool::Pool,
                             resource_cache: &ResourceCache,
                             device_pixel_ratio: f32) {
        let _pf = util::ProfileScope::new("  compile_visible_nodes");

        let layers = &mut self.layers;
        let render_targets = &self.render_targets;

        thread_pool.scoped(|scope| {
            for (_, layer) in layers {
                let nodes = &mut layer.aabb_tree.nodes;
                for node in nodes {
                    if node.is_visible && node.compiled_node.is_none() {
                        scope.execute(move || {
                            node.compile(resource_cache,
                                         render_targets,
                                         device_pixel_ratio);
                        });
                    }
                }
            }
        });
    }

    pub fn update_batch_cache(&mut self) {
        // Allocate and update VAOs
        for (_, layer) in &mut self.layers {
            for node in &mut layer.aabb_tree.nodes {
                if node.is_visible {
                    let compiled_node = node.compiled_node.as_mut().unwrap();
                    if let Some(vertex_buffer) = compiled_node.vertex_buffer.take() {
                        debug_assert!(compiled_node.vertex_buffer_id.is_none());

                        self.pending_updates.push(BatchUpdate {
                            id: vertex_buffer.id,
                            op: BatchUpdateOp::Create(vertex_buffer.vertices,
                                                      vertex_buffer.indices),
                        });

                        compiled_node.vertex_buffer_id = Some(vertex_buffer.id);
                    }
                }
            }
        }
    }

    pub fn collect_and_sort_visible_batches(&mut self,
                                            resource_cache: &ResourceCache) -> RendererFrame {
        let mut frame = RendererFrame::new(self.pipeline_epoch_map.clone());

        for render_target in &self.render_targets {
            let mut commands = Vec::new();

            for item in &render_target.items {
                match item {
                    &FrameRenderItem::Clear(ref info) => {
                        commands.push(DrawCommand::Clear(info.clone()));
                    }
                    &FrameRenderItem::Composite(ref info) => {
                        commands.push(DrawCommand::Composite(info.clone()));
                    }
                    &FrameRenderItem::DrawListBatch(ref batch_info) => {
                        let layer = &self.layers[&batch_info.scroll_layer_id];
                        let first_draw_list_id = *batch_info.draw_lists.first().unwrap();
                        debug_assert!(batch_info.draw_lists.len() < MAX_MATRICES_PER_BATCH);
                        let mut matrix_palette = vec![Matrix4::identity(); batch_info.draw_lists.len()];

                        // Update batch matrices
                        for (index, draw_list_id) in batch_info.draw_lists.iter().enumerate() {
                            let draw_list = resource_cache.get_draw_list(*draw_list_id);

                            let transform = draw_list.context.as_ref().unwrap().final_transform;
                            let transform = transform.translate(layer.scroll_offset.x,
                                                                layer.scroll_offset.y,
                                                                0.0);
                            matrix_palette[index] = transform;
                        }

                        let mut batch_info = BatchInfo::new(matrix_palette);

                        // Collect relevant draws from each node in the tree.
                        for node in &layer.aabb_tree.nodes {
                            if node.is_visible {
                                debug_assert!(node.compiled_node.is_some());
                                let compiled_node = node.compiled_node.as_ref().unwrap();

                                let batch_list = compiled_node.batch_list.iter().find(|batch_list| {
                                    batch_list.first_draw_list_id == first_draw_list_id
                                });

                                if let Some(batch_list) = batch_list {
                                    let vertex_buffer_id = compiled_node.vertex_buffer_id.unwrap();

                                    for batch in &batch_list.batches {
                                        batch_info.draw_calls.push(DrawCall {
                                            tile_params: batch.tile_params.clone(),     // TODO(gw): Move this instead?
                                            vertex_buffer_id: vertex_buffer_id,
                                            color_texture_id: batch.color_texture_id,
                                            mask_texture_id: batch.mask_texture_id,
                                            first_vertex: batch.first_vertex,
                                            index_count: batch.index_count,
                                        });
                                    }
                                }
                            }
                        }

                        // Finally, add the batch + draw calls
                        commands.push(DrawCommand::Batch(batch_info));
                    }
                }
            }

            let layer = DrawLayer::new(render_target.texture_id,
                                       render_target.size,
                                       commands);
            frame.layers.push(layer);
        }

        frame
    }
}
