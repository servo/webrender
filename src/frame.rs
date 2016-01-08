use app_units::Au;
use batch::{MAX_MATRICES_PER_BATCH, OffsetParams};
use device::{TextureId, TextureFilter};
use euclid::{Rect, Point2D, Size2D, Matrix4};
use fnv::FnvHasher;
use internal_types::{AxisDirection, LowLevelFilterOp, CompositionOp, DrawListItemIndex};
use internal_types::{BatchUpdateList, RenderTargetIndex, DrawListId};
use internal_types::{CompositeBatchInfo, CompositeBatchJob};
use internal_types::{RendererFrame, StackingContextInfo, BatchInfo, DrawCall, StackingContextIndex};
use internal_types::{ANGLE_FLOAT_TO_FIXED, BatchUpdate, BatchUpdateOp, DrawLayer};
use internal_types::{DrawCommand, ClearInfo, DrawTargetInfo, RenderTargetId, DrawListGroupId};
use layer::Layer;
use node_compiler::NodeCompiler;
use renderer::CompositionOpHelpers;
use resource_cache::ResourceCache;
use resource_list::BuildRequiredResources;
use scene::{SceneStackingContext, ScenePipeline, Scene, SceneItem, SpecificSceneItem};
use scoped_threadpool;
use std::collections::HashMap;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::hash_state::DefaultState;
use std::mem;
use texture_cache::TexturePage;
use util;
use util::MatrixHelpers;
use webrender_traits::{PipelineId, Epoch, ScrollPolicy, ScrollLayerId, StackingContext};
use webrender_traits::{FilterOp, ImageFormat, MixBlendMode, StackingLevel};

pub struct DrawListGroup {
    pub id: DrawListGroupId,

    // Together, these define the granularity that batches
    // can be created at. When compiling nodes, if either
    // the scroll layer or render target are different from
    // the current batch, it must be broken and a new batch started.
    // This automatically handles the case of CompositeBatch, because
    // for a composite batch to be present, the next draw list must be
    // in a different render target!
    pub scroll_layer_id: ScrollLayerId,
    pub render_target_id: RenderTargetId,

    pub draw_list_ids: Vec<DrawListId>,
}

impl DrawListGroup {
    fn new(id: DrawListGroupId,
           scroll_layer_id: ScrollLayerId,
           render_target_id: RenderTargetId) -> DrawListGroup {
        DrawListGroup {
            id: id,
            scroll_layer_id: scroll_layer_id,
            render_target_id: render_target_id,
            draw_list_ids: Vec::new(),
        }
    }

    fn can_add(&self,
               scroll_layer_id: ScrollLayerId,
               render_target_id: RenderTargetId) -> bool {
        let scroll_ok = scroll_layer_id == self.scroll_layer_id;
        let target_ok = render_target_id == self.render_target_id;
        let size_ok = self.draw_list_ids.len() < MAX_MATRICES_PER_BATCH;
        scroll_ok && target_ok && size_ok
    }

    fn push(&mut self, draw_list_id: DrawListId) {
        self.draw_list_ids.push(draw_list_id);
    }
}

struct FlattenContext<'a> {
    resource_cache: &'a mut ResourceCache,
    scene: &'a Scene,
    pipeline_sizes: &'a mut HashMap<PipelineId, Size2D<f32>>,
    current_draw_list_group: Option<DrawListGroup>,
}

#[derive(Debug)]
pub enum FrameRenderItem {
    Clear(ClearInfo),
    CompositeBatch(CompositeBatchInfo),
    DrawListBatch(DrawListGroupId),
}

pub struct RenderTarget {
    id: RenderTargetId,

    // Child render targets
    children: Vec<RenderTarget>,

    // Outputs
    items: Vec<FrameRenderItem>,

    // Texture id for any child render targets to use
    child_texture_id: Option<TextureId>,

    size: Size2D<u32>,
}

impl RenderTarget {
    fn new(id: RenderTargetId, size: Size2D<u32>) -> RenderTarget {
        RenderTarget {
            id: id,
            children: Vec::new(),
            items: Vec::new(),
            child_texture_id: None,
            size: size,
        }
    }

    fn collect_and_sort_visible_batches(&mut self,
                                        resource_cache: &mut ResourceCache,
                                        draw_list_groups: &HashMap<DrawListGroupId, DrawListGroup, DefaultState<FnvHasher>>,
                                        layers: &HashMap<ScrollLayerId, Layer, DefaultState<FnvHasher>>,
                                        stacking_context_info: &Vec<StackingContextInfo>,
                                        device_pixel_ratio: f32) -> DrawLayer {
        let mut commands = vec![];
        for item in &self.items {
            match item {
                &FrameRenderItem::Clear(ref info) => {
                    commands.push(DrawCommand::Clear(info.clone()));
                }
                &FrameRenderItem::CompositeBatch(ref info) => {
                    commands.push(DrawCommand::CompositeBatch(info.clone()));
                }
                &FrameRenderItem::DrawListBatch(draw_list_group_id) => {
                    let draw_list_group = &draw_list_groups[&draw_list_group_id];
                    debug_assert!(draw_list_group.draw_list_ids.len() <= MAX_MATRICES_PER_BATCH);

                    let layer = &layers[&draw_list_group.scroll_layer_id];
                    let mut matrix_palette =
                        vec![Matrix4::identity(); draw_list_group.draw_list_ids.len()];
                    let mut offset_palette =
                        vec![OffsetParams::identity(); draw_list_group.draw_list_ids.len()];

                    // Update batch matrices
                    for (index, draw_list_id) in draw_list_group.draw_list_ids.iter().enumerate() {
                        let draw_list = resource_cache.get_draw_list(*draw_list_id);

                        let StackingContextIndex(stacking_context_id) = draw_list.stacking_context_index.unwrap();
                        let context = &stacking_context_info[stacking_context_id];
                        let mut transform = context.world_transform;
                        transform = transform.translate(layer.scroll_offset.x,
                                                        layer.scroll_offset.y,
                                                        0.0);
                        matrix_palette[index] = transform;

                        offset_palette[index].stacking_context_x0 = context.world_origin.x;
                        offset_palette[index].stacking_context_y0 = context.world_origin.y;
                    }

                    let mut batch_info = BatchInfo::new(matrix_palette, offset_palette);

                    // Collect relevant draws from each node in the tree.
                    for node in &layer.aabb_tree.nodes {
                        if node.is_visible {
                            debug_assert!(node.compiled_node.is_some());
                            let compiled_node = node.compiled_node.as_ref().unwrap();

                            let batch_list = compiled_node.batch_list.iter().find(|batch_list| {
                                batch_list.draw_list_group_id == draw_list_group_id
                            });

                            if let Some(batch_list) = batch_list {
                                let vertex_buffer_id = compiled_node.vertex_buffer_id.unwrap();

                                for batch in &batch_list.batches {
                                    batch_info.draw_calls.push(DrawCall {
                                        batch: (*batch).clone(),
                                        vertex_buffer_id: vertex_buffer_id,
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

        let mut child_layers = Vec::new();

        let draw_target_info = if self.children.is_empty() {
            None
        } else {
            let texture_size = 2048;
            let device_pixel_size = texture_size * device_pixel_ratio as u32;

            // TODO(gw): This doesn't handle not having enough space to store
            //           draw all child render targets. However, this will soon
            //           be changing to do the RT allocation in a smarter way
            //           that greatly reduces the # of active RT allocations.
            //           When that happens, ensure it handles this case!
            if let Some(child_texture_id) = self.child_texture_id.take() {
                resource_cache.free_render_target(child_texture_id);
            }

            self.child_texture_id = Some(resource_cache.allocate_render_target(device_pixel_size,
                                                                               device_pixel_size,
                                                                               ImageFormat::RGBA8));

            // TODO(gw): Move this texture page allocator based on the suggested changes above.
            let mut page = TexturePage::new(self.child_texture_id.unwrap(), texture_size);

            for child in &mut self.children {
                let mut child_layer = child.collect_and_sort_visible_batches(resource_cache,
                                                                             draw_list_groups,
                                                                             layers,
                                                                             stacking_context_info,
                                                                             device_pixel_ratio);

                child_layer.layer_origin = page.allocate(&child_layer.layer_size,
                                                         TextureFilter::Linear).unwrap();
                child_layers.push(child_layer);
            }

            Some(DrawTargetInfo {
                size: Size2D::new(texture_size, texture_size),
                texture_id: self.child_texture_id.unwrap(),
            })
        };

        DrawLayer::new(draw_target_info,
                       child_layers,
                       commands,
                       self.size)
    }

    fn reset(&mut self,
             pending_updates: &mut BatchUpdateList,
             resource_cache: &mut ResourceCache) {
        if let Some(child_texture_id) = self.child_texture_id.take() {
            resource_cache.free_render_target(child_texture_id);
        }

        for mut child in &mut self.children.drain(..) {
            child.reset(pending_updates,
                        resource_cache);
        }

        self.items.clear();
    }

    fn push_clear(&mut self, clear_info: ClearInfo) {
        self.items.push(FrameRenderItem::Clear(clear_info));
    }

    fn push_composite(&mut self,
                      op: CompositionOp,
                      target: Rect<i32>,
                      render_target_index: RenderTargetIndex) {
        // TODO(gw): Relax the restriction on batch breaks for FB reads
        //           once the proper render target allocation code is done!
        let need_new_batch = op.needs_framebuffer() || match self.items.last() {
            Some(&FrameRenderItem::CompositeBatch(ref info)) => {
                info.operation != op
            }
            Some(&FrameRenderItem::Clear(..)) |
            Some(&FrameRenderItem::DrawListBatch(..)) |
            None => {
                true
            }
        };

        if need_new_batch {
            self.items.push(FrameRenderItem::CompositeBatch(CompositeBatchInfo {
                operation: op,
                jobs: Vec::new(),
            }));
        }

        // TODO(gw): This seems a little messy - restructure how current batch works!
        match self.items.last_mut().unwrap() {
            &mut FrameRenderItem::CompositeBatch(ref mut batch) => {
                let job = CompositeBatchJob {
                    rect: target,
                    render_target_index: render_target_index
                };
                batch.jobs.push(job);
            }
            _ => {
                unreachable!();
            }
        }
    }

    fn push_draw_list_group(&mut self, draw_list_group_id: DrawListGroupId) {
        self.items.push(FrameRenderItem::DrawListBatch(draw_list_group_id));
    }
}

pub struct Frame {
    pub layers: HashMap<ScrollLayerId, Layer, DefaultState<FnvHasher>>,
    pub pipeline_epoch_map: HashMap<PipelineId, Epoch, DefaultState<FnvHasher>>,
    pub pending_updates: BatchUpdateList,
    pub root: Option<RenderTarget>,
    pub stacking_context_info: Vec<StackingContextInfo>,
    next_render_target_id: RenderTargetId,
    next_draw_list_group_id: DrawListGroupId,
    draw_list_groups: HashMap<DrawListGroupId, DrawListGroup, DefaultState<FnvHasher>>,
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
    fn composition_operations(&self) -> Vec<CompositionOp>;
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

    fn composition_operations(&self) -> Vec<CompositionOp> {
        let mut composition_operations = vec![];
        if self.needs_composition_operation_for_mix_blend_mode() {
            composition_operations.push(CompositionOp::MixBlend(self.mix_blend_mode));
        }
        for filter in self.filters.iter() {
            match *filter {
                FilterOp::Blur(radius) => {
                    composition_operations.push(CompositionOp::Filter(LowLevelFilterOp::Blur(
                        radius,
                        AxisDirection::Horizontal)));
                    composition_operations.push(CompositionOp::Filter(LowLevelFilterOp::Blur(
                        radius,
                        AxisDirection::Vertical)));
                }
                FilterOp::Brightness(amount) => {
                    composition_operations.push(CompositionOp::Filter(
                            LowLevelFilterOp::Brightness(Au::from_f32_px(amount))));
                }
                FilterOp::Contrast(amount) => {
                    composition_operations.push(CompositionOp::Filter(
                            LowLevelFilterOp::Contrast(Au::from_f32_px(amount))));
                }
                FilterOp::Grayscale(amount) => {
                    composition_operations.push(CompositionOp::Filter(
                            LowLevelFilterOp::Grayscale(Au::from_f32_px(amount))));
                }
                FilterOp::HueRotate(angle) => {
                    composition_operations.push(CompositionOp::Filter(
                            LowLevelFilterOp::HueRotate(f32::round(
                                    angle * ANGLE_FLOAT_TO_FIXED) as i32)));
                }
                FilterOp::Invert(amount) => {
                    composition_operations.push(CompositionOp::Filter(
                            LowLevelFilterOp::Invert(Au::from_f32_px(amount))));
                }
                FilterOp::Opacity(amount) => {
                    composition_operations.push(CompositionOp::Filter(
                            LowLevelFilterOp::Opacity(Au::from_f32_px(amount))));
                }
                FilterOp::Saturate(amount) => {
                    composition_operations.push(CompositionOp::Filter(
                            LowLevelFilterOp::Saturate(Au::from_f32_px(amount))));
                }
                FilterOp::Sepia(amount) => {
                    composition_operations.push(CompositionOp::Filter(
                            LowLevelFilterOp::Sepia(Au::from_f32_px(amount))));
                }
            }
        }

        composition_operations
    }
}

impl Frame {
    pub fn new() -> Frame {
        Frame {
            pipeline_epoch_map: HashMap::with_hash_state(Default::default()),
            pending_updates: BatchUpdateList::new(),
            root: None,
            layers: HashMap::with_hash_state(Default::default()),
            stacking_context_info: Vec::new(),
            next_render_target_id: RenderTargetId(0),
            next_draw_list_group_id: DrawListGroupId(0),
            draw_list_groups: HashMap::with_hash_state(Default::default()),
        }
    }

    pub fn reset(&mut self, resource_cache: &mut ResourceCache)
                 -> HashMap<ScrollLayerId, Point2D<f32>, DefaultState<FnvHasher>> {
        self.draw_list_groups.clear();
        self.pipeline_epoch_map.clear();
        self.stacking_context_info.clear();

        if let Some(mut root) = self.root.take() {
            root.reset(&mut self.pending_updates, resource_cache);
        }

        // Free any render targets from last frame.
        // TODO: This should really re-use existing targets here...
        let mut old_layer_offsets = HashMap::with_hash_state(Default::default());
        for (layer_id, mut old_layer) in &mut self.layers.drain() {
            old_layer.reset(&mut self.pending_updates);
            old_layer_offsets.insert(layer_id, old_layer.scroll_offset);
        }

        old_layer_offsets
    }

    fn next_render_target_id(&mut self) -> RenderTargetId {
        let RenderTargetId(render_target_id) = self.next_render_target_id;
        self.next_render_target_id = RenderTargetId(render_target_id + 1);
        RenderTargetId(render_target_id)
    }

    fn next_draw_list_group_id(&mut self) -> DrawListGroupId {
        let DrawListGroupId(draw_list_group_id) = self.next_draw_list_group_id;
        self.next_draw_list_group_id = DrawListGroupId(draw_list_group_id + 1);
        DrawListGroupId(draw_list_group_id)
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
                  resource_cache: &mut ResourceCache,
                  pipeline_sizes: &mut HashMap<PipelineId, Size2D<f32>>,
                  framebuffer_size: Size2D<u32>) {
        if let Some(root_pipeline_id) = scene.root_pipeline_id {
            if let Some(root_pipeline) = scene.pipeline_map.get(&root_pipeline_id) {
                let old_layer_offsets = self.reset(resource_cache);

                let root_stacking_context = scene.stacking_context_map
                                                 .get(&root_pipeline.root_stacking_context_id)
                                                 .unwrap();

                let root_scroll_layer_id = root_stacking_context.stacking_context
                                                                .scroll_layer_id
                                                                .expect("root layer must be a scroll layer!");

                let root_target_id = self.next_render_target_id();

                let mut root_target = RenderTarget::new(root_target_id, framebuffer_size);

                // Work around borrow check on resource cache
                {
                    let mut context = FlattenContext {
                        resource_cache: resource_cache,
                        scene: scene,
                        pipeline_sizes: pipeline_sizes,
                        current_draw_list_group: None,
                    };

                    let parent_info = StackingContextInfo {
                        world_origin: Point2D::zero(),
                        scroll_layer_id: root_scroll_layer_id,
                        world_perspective: Matrix4::identity(),
                        world_transform: Matrix4::identity(),
                        local_overflow: root_stacking_context.stacking_context.overflow,
                    };

                    let root_pipeline = SceneItemKind::Pipeline(root_pipeline);
                    self.flatten(root_pipeline,
                                 &parent_info,
                                 &mut context,
                                 &mut root_target);
                    self.root = Some(root_target);

                    if let Some(last_draw_list_group) = context.current_draw_list_group.take() {
                        self.draw_list_groups.insert(last_draw_list_group.id,
                                                     last_draw_list_group);
                    }
                }

                let scene_rect = root_stacking_context.stacking_context.overflow;

                for (draw_list_group_id, draw_list_group) in &self.draw_list_groups {
                    let layer = match self.layers.entry(draw_list_group.scroll_layer_id) {
                        Occupied(entry) => {
                            entry.into_mut()
                        }
                        Vacant(entry) => {
                            let scroll_offset = match old_layer_offsets.get(&draw_list_group.scroll_layer_id) {
                                Some(old_offset) => *old_offset,
                                None => Point2D::zero(),
                            };

                            entry.insert(Layer::new(&scene_rect, &scroll_offset))
                        }
                    };

                    for draw_list_id in &draw_list_group.draw_list_ids {
                        let draw_list_id = *draw_list_id;
                        let draw_list = resource_cache.get_draw_list(draw_list_id);

                        let StackingContextIndex(stacking_context_id) = draw_list.stacking_context_index.unwrap();
                        let sc_info = &self.stacking_context_info[stacking_context_id];

                        for (item_index, item) in draw_list.items.iter().enumerate() {
                            // Node index may already be Some(..). This can occur when a page has iframes
                            // and a new root stacking context is received. In this case, the node index
                            // may already be set for draw lists from other iframe(s) that weren't updated
                            // as part of this new stacking context.
                            let item_index = DrawListItemIndex(item_index as u32);
                            let world_space_rect = item.rect.translate(&sc_info.world_origin);
                            let rect = sc_info.world_transform.transform_rect(&world_space_rect);
                            layer.insert(&rect,
                                         *draw_list_group_id,
                                         draw_list_id,
                                         item_index);
                        }
                    }
                }

                // TODO(gw): This should be moved elsewhere!
                if let Some(root_scroll_layer) = self.layers.get_mut(&root_scroll_layer_id) {
                    root_scroll_layer.scroll_boundaries = root_stacking_context.stacking_context.overflow.size;
                }
            }
        }
    }

    fn add_items_to_target(&mut self,
                           scene_items: &Vec<SceneItem>,
                           target: &mut RenderTarget,
                           sc_info: StackingContextInfo,
                           context: &mut FlattenContext) {
        let stacking_context_index = StackingContextIndex(self.stacking_context_info.len());
        self.stacking_context_info.push(sc_info.clone()); // TODO(gw): Avoid clone?

        for item in scene_items {
            match item.specific {
                SpecificSceneItem::DrawList(draw_list_id) => {
                    let draw_list = context.resource_cache.get_draw_list_mut(draw_list_id);

                    // Store draw context
                    draw_list.stacking_context_index = Some(stacking_context_index);

                    let needs_new_draw_group = match context.current_draw_list_group {
                        Some(ref draw_list_group) => {
                            !draw_list_group.can_add(sc_info.scroll_layer_id,
                                                     target.id)
                        }
                        None => {
                            true
                        }
                    };

                    if needs_new_draw_group {
                        if let Some(draw_list_group) = context.current_draw_list_group.take() {
                            self.draw_list_groups.insert(draw_list_group.id,
                                                         draw_list_group);
                        }

                        let draw_list_group_id = self.next_draw_list_group_id();

                        let new_draw_list_group = DrawListGroup::new(draw_list_group_id,
                                                                     sc_info.scroll_layer_id,
                                                                     target.id);

                        target.push_draw_list_group(draw_list_group_id);

                        context.current_draw_list_group = Some(new_draw_list_group);
                    }

                    context.current_draw_list_group.as_mut().unwrap().push(draw_list_id);
                }
                SpecificSceneItem::StackingContext(id) => {
                    let stacking_context = context.scene
                                                  .stacking_context_map
                                                  .get(&id)
                                                  .unwrap();

                    let child = SceneItemKind::StackingContext(stacking_context);
                    self.flatten(child,
                                 &sc_info,
                                 context,
                                 target);
                }
                SpecificSceneItem::Iframe(ref iframe_info) => {
                    let pipeline = context.scene
                                          .pipeline_map
                                          .get(&iframe_info.id);

                    context.pipeline_sizes.insert(iframe_info.id,
                                                  iframe_info.bounds.size);

                    if let Some(pipeline) = pipeline {
                        let iframe = SceneItemKind::Pipeline(pipeline);

                        // TODO(gw): Doesn't handle transforms on iframes yet!
                        let world_origin = sc_info.world_origin + iframe_info.bounds.origin;
                        let iframe_transform = Matrix4::identity();

                        let overflow = sc_info.local_overflow
                                              .translate(&-sc_info.local_overflow.origin)
                                              .intersection(&iframe_info.bounds);

                        if let Some(overflow) = overflow {
                            let overflow = overflow.translate(&-iframe_info.bounds.origin);

                            let iframe_info = StackingContextInfo {
                                scroll_layer_id: sc_info.scroll_layer_id,
                                world_origin: world_origin,
                                world_transform: iframe_transform,
                                local_overflow: overflow,
                                world_perspective: Matrix4::identity(),
                            };

                            self.flatten(iframe,
                                         &iframe_info,
                                         context,
                                         target);
                        }
                    }
                }
            }
        }

        //target.flush();
    }

    pub fn flatten(&mut self,
                   scene_item: SceneItemKind,
                   parent: &StackingContextInfo,
                   context: &mut FlattenContext,
                   target: &mut RenderTarget) {
        let _pf = util::ProfileScope::new("  flatten");

        let stacking_context = match scene_item {
            SceneItemKind::StackingContext(stacking_context) => {
                &stacking_context.stacking_context
            }
            SceneItemKind::Pipeline(pipeline) => {
                self.pipeline_epoch_map.insert(pipeline.pipeline_id, pipeline.epoch);

                &context.scene.stacking_context_map
                        .get(&pipeline.root_stacking_context_id)
                        .unwrap()
                        .stacking_context
            }
        };

        let this_scroll_layer = match stacking_context.scroll_policy {
            ScrollPolicy::Scrollable => {
                stacking_context.scroll_layer_id.unwrap_or(parent.scroll_layer_id)
            }
            ScrollPolicy::Fixed => {
                debug_assert!(stacking_context.scroll_layer_id.is_none());
                ScrollLayerId::fixed_layer()
            }
        };

        let overflow = parent.local_overflow
                             .translate(&-stacking_context.bounds.origin)
                             .translate(&-stacking_context.overflow.origin)
                             .intersection(&stacking_context.overflow);

        if let Some(overflow) = overflow {
            let scene_items = scene_item.collect_scene_items(&context.scene);
            if !scene_items.is_empty() {

                // When establishing a new 3D context, clear Z. This is only needed if there
                // are child stacking contexts, otherwise it is a redundant clear.
                if stacking_context.establishes_3d_context &&
                   stacking_context.has_stacking_contexts {
                    target.push_clear(ClearInfo {
                        clear_color: false,
                        clear_z: true,
                        clear_stencil: true,
                    });
                }

                // TODO: Account for scroll offset with transforms!
                let composition_operations = stacking_context.composition_operations();
                if composition_operations.is_empty() {
                    // Build world space transform
                    let origin = stacking_context.bounds.origin;

                    let local_transform = Matrix4::identity().translate(origin.x, origin.y, 0.0)
                                                             .mul(&stacking_context.transform)
                                                             .translate(-origin.x, -origin.y, 0.0);

                    let transform = parent.world_perspective.mul(&parent.world_transform)
                                                            .mul(&local_transform);

                    // Build world space perspective transform
                    let perspective_transform = Matrix4::identity().translate(origin.x, origin.y, 0.0)
                                                                   .mul(&stacking_context.perspective)
                                                                   .translate(-origin.x, -origin.y, 0.0);

                    let info = StackingContextInfo {
                        world_origin: parent.world_origin + origin,
                        scroll_layer_id: this_scroll_layer,
                        local_overflow: overflow,
                        world_transform: transform,
                        world_perspective: perspective_transform,
                    };

                    self.add_items_to_target(&scene_items,
                                             target,
                                             info,
                                             context);
                } else {
                    let target_size = Size2D::new(overflow.size.width as i32,
                                                  overflow.size.height as i32);
                    let origin = stacking_context.bounds.origin;
                    let target_origin =
                        Point2D::new(parent.world_origin.x as i32 + origin.x as i32,
                                     parent.world_origin.y as i32 + origin.y as i32);
                    let unfiltered_target_rect = Rect::new(target_origin, target_size);
                    let mut target_rect = unfiltered_target_rect;
                    for composition_operation in &composition_operations {
                        target_rect = composition_operation.target_rect(&target_rect);
                    }

                    let render_target_index = RenderTargetIndex(target.children.len() as u32);

                    let render_target_size = Size2D::new(target_rect.size.width as u32,
                                                         target_rect.size.height as u32);
                    let render_target_id = self.next_render_target_id();
                    let mut new_target = RenderTarget::new(render_target_id,
                                                           render_target_size);

                    let world_origin = Point2D::new(
                            (unfiltered_target_rect.origin.x - target_rect.origin.x) as f32,
                            (unfiltered_target_rect.origin.y - target_rect.origin.y) as f32);
                    let world_transform = stacking_context.transform.translate(world_origin.x,
                                                                               world_origin.y,
                                                                               0.0);
                    let info = StackingContextInfo {
                        world_origin: Point2D::zero(),
                        scroll_layer_id: this_scroll_layer,
                        local_overflow: overflow,
                        world_transform: world_transform,
                        world_perspective: stacking_context.perspective,
                    };

                    // TODO(gw): Handle transforms + composition ops...
                    for composition_operation in composition_operations {
                        target.push_composite(composition_operation,
                                              target_rect,
                                              render_target_index);
                    }

                    self.add_items_to_target(&scene_items,
                                             &mut new_target,
                                             info,
                                             context);

                    target.children.push(new_target);
                }
            }
        }
    }

    pub fn build(&mut self,
                 viewport: &Rect<i32>,
                 resource_cache: &mut ResourceCache,
                 thread_pool: &mut scoped_threadpool::Pool,
                 device_pixel_ratio: f32)
                 -> RendererFrame {
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
        let frame = self.collect_and_sort_visible_batches(resource_cache, device_pixel_ratio);

        frame
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

    pub fn update_texture_cache_and_build_raster_jobs(&mut self,
                                                      resource_cache: &mut ResourceCache) {
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
        let stacking_context_info = &self.stacking_context_info;
        let draw_list_groups = &self.draw_list_groups;

        thread_pool.scoped(|scope| {
            for (_, layer) in layers {
                let nodes = &mut layer.aabb_tree.nodes;
                for node in nodes {
                    if node.is_visible && node.compiled_node.is_none() {
                        scope.execute(move || {
                            node.compile(resource_cache,
                                         device_pixel_ratio,
                                         stacking_context_info,
                                         draw_list_groups);
                        });
                    }
                }
            }
        });
    }

    pub fn update_batch_cache(&mut self) {
        let _pf = util::ProfileScope::new("  update_batch_cache");

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
                                            resource_cache: &mut ResourceCache,
                                            device_pixel_ratio: f32)
                                            -> RendererFrame {
        let root_layer = match self.root {
            Some(ref mut root) => {
                 root.collect_and_sort_visible_batches(resource_cache,
                                                       &self.draw_list_groups,
                                                       &self.layers,
                                                       &self.stacking_context_info,
                                                       device_pixel_ratio)
            }
            None => {
                DrawLayer::new(None,
                               Vec::new(),
                               Vec::new(),
                               Size2D::zero())
            }
        };

        RendererFrame::new(self.pipeline_epoch_map.clone(), root_layer)
    }
}
