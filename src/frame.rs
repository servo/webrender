/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use batch::{MAX_MATRICES_PER_BATCH, OffsetParams};
use device::{TextureId, TextureFilter};
use euclid::{Matrix4D, Point2D, Point3D, Point4D, Rect, Size2D};
use fnv::FnvHasher;
use geometry::ray_intersects_rect;
use internal_types::{AxisDirection, LowLevelFilterOp, CompositionOp, DrawListItemIndex};
use internal_types::{BatchUpdateList, ChildLayerIndex, DrawListId, DrawCompositeBatchInfo};
use internal_types::{CompositeBatchInfo, CompositeBatchJob, DrawCompositeBatchJob, MaskRegion};
use internal_types::{RendererFrame, StackingContextInfo, BatchInfo, DrawCall, StackingContextIndex};
use internal_types::{ANGLE_FLOAT_TO_FIXED, MAX_RECT, BatchUpdate, BatchUpdateOp, DrawLayer};
use internal_types::{DrawCommand, ClearInfo, RenderTargetId, DrawListGroupId};
use internal_types::{ORTHO_NEAR_PLANE, ORTHO_FAR_PLANE};
use layer::{Layer, ScrollingState};
use node_compiler::NodeCompiler;
use renderer::CompositionOpHelpers;
use resource_cache::ResourceCache;
use resource_list::BuildRequiredResources;
use scene::{SceneStackingContext, ScenePipeline, Scene, SceneItem, SpecificSceneItem};
use scoped_threadpool;
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;
use std::mem;
use texture_cache::TexturePage;
use util::{self, MatrixHelpers};
use webrender_traits::{AuxiliaryLists, PipelineId, Epoch, ScrollPolicy, ScrollLayerId};
use webrender_traits::{StackingContext, FilterOp, ImageFormat, MixBlendMode};
use webrender_traits::{ScrollEventPhase, ScrollLayerInfo, ScrollLayerState};

#[cfg(target_os = "macos")]
const CAN_OVERSCROLL: bool = true;

#[cfg(not(target_os = "macos"))]
const CAN_OVERSCROLL: bool = false;

#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct FrameId(pub u32);

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
    device_pixel_ratio: f32,
}

#[derive(Debug)]
struct FlattenInfo {
    /// The viewable region, in world coordinates.
    viewport_rect: Rect<f32>,

    /// The transform to apply to the viewable region, in world coordinates.
    ///
    /// TODO(pcwalton): These should really be a stack of clip regions and transforms.
    viewport_transform: Matrix4D<f32>,

    current_clip_rect: Rect<f32>,
    default_scroll_layer_id: ScrollLayerId,
    actual_scroll_layer_id: ScrollLayerId,
    fixed_scroll_layer_id: ScrollLayerId,
    offset_from_origin: Point2D<f32>,
    offset_from_current_layer: Point2D<f32>,
    transform: Matrix4D<f32>,
    perspective: Matrix4D<f32>,
    pipeline_id: PipelineId,
}

#[derive(Debug)]
pub enum FrameRenderItem {
    /// The extra boolean indicates whether a Z-buffer clear is needed.
    CompositeBatch(CompositeBatchInfo, bool),
    DrawListBatch(DrawListGroupId),
}

pub struct RenderTarget {
    id: RenderTargetId,
    size: Size2D<f32>,
    /// The origin in render target space.
    origin: Point2D<f32>,
    /// The origin in world space.
    world_origin: Point2D<f32>,
    items: Vec<FrameRenderItem>,
    texture_id: Option<TextureId>,
    children: Vec<RenderTarget>,

    page_allocator: Option<TexturePage>,
    texture_id_list: Vec<TextureId>,
}

impl RenderTarget {
    fn new(id: RenderTargetId,
           origin: Point2D<f32>,
           world_origin: Point2D<f32>,
           size: Size2D<f32>,
           texture_id: Option<TextureId>) -> RenderTarget {
        RenderTarget {
            id: id,
            size: size,
            origin: origin,
            world_origin: world_origin,
            items: Vec::new(),
            texture_id: texture_id,
            children: Vec::new(),
            texture_id_list: Vec::new(),
            page_allocator: None,
        }
    }

    fn allocate_target_rect(&mut self,
                            width: f32,
                            height: f32,
                            device_pixel_ratio: f32,
                            resource_cache: &mut ResourceCache,
                            frame_id: FrameId) -> (Point2D<f32>, TextureId) {
        // If the target is more than 512x512 (an arbitrary choice), assign it
        // to an exact sized render target - assuming that there probably aren't
        // many of them. This minimises GPU memory wastage if there are just a small
        // number of large targets. Otherwise, attempt to allocate inside a shared render
        // target texture - this allows composite batching to take place when
        // there are a lot of small targets (which is more efficient).
        if width < 512.0 && height < 512.0 {
            if self.page_allocator.is_none() {
                let texture_size = 2048;
                let device_pixel_size = texture_size * device_pixel_ratio as u32;

                let texture_id = resource_cache.allocate_render_target(device_pixel_size,
                                                                       device_pixel_size,
                                                                       ImageFormat::RGBA8,
                                                                       frame_id);
                self.texture_id_list.push(texture_id);
                self.page_allocator = Some(TexturePage::new(texture_id, texture_size));
            }

            // TODO(gw): This has accuracy issues if the size of a rendertarget is
            //           not scene pixel aligned!
            let size = Size2D::new(width as u32, height as u32);
            let allocated_origin = self.page_allocator
                                       .as_mut()
                                       .unwrap()
                                       .allocate(&size, TextureFilter::Linear);
            if let Some(allocated_origin) = allocated_origin {
                let origin = Point2D::new(allocated_origin.x as f32,
                                          allocated_origin.y as f32);
                return (origin, self.page_allocator.as_ref().unwrap().texture_id())
            }
        }

        let device_pixel_width = width as u32 * device_pixel_ratio as u32;
        let device_pixel_height = height as u32 * device_pixel_ratio as u32;

        let texture_id = resource_cache.allocate_render_target(device_pixel_width,
                                                               device_pixel_height,
                                                               ImageFormat::RGBA8,
                                                               frame_id);
        self.texture_id_list.push(texture_id);

        (Point2D::zero(), texture_id)
    }

    fn collect_and_sort_visible_batches(&mut self,
                                        resource_cache: &mut ResourceCache,
                                        draw_list_groups: &HashMap<DrawListGroupId, DrawListGroup, BuildHasherDefault<FnvHasher>>,
                                        layers: &HashMap<ScrollLayerId, Layer, BuildHasherDefault<FnvHasher>>,
                                        stacking_context_info: &[StackingContextInfo],
                                        device_pixel_ratio: f32) -> DrawLayer {
        let mut commands = vec![];
        for item in &self.items {
            match item {
                &FrameRenderItem::CompositeBatch(ref info, z_clear_needed) => {
                    let layer = &layers[&info.scroll_layer_id];
                    let transform = layer.world_transform;
                    if z_clear_needed && !commands.is_empty() {
                        commands.push(DrawCommand::Clear(ClearInfo {
                            clear_color: false,
                            clear_z: true,
                            clear_stencil: true,
                        }))
                    }

                    let mut draw_jobs = vec![];
                    for job in &info.jobs {
                        draw_jobs.push(DrawCompositeBatchJob {
                            rect: job.rect,
                            local_transform: job.transform,
                            world_transform: job.transform.mul(&transform),
                            child_layer_index: job.child_layer_index,
                        })
                    }

                    commands.push(DrawCommand::CompositeBatch(DrawCompositeBatchInfo {
                        operation: info.operation,
                        texture_id: info.texture_id,
                        scroll_layer_id: info.scroll_layer_id,
                        jobs: draw_jobs,
                    }))
                }
                &FrameRenderItem::DrawListBatch(draw_list_group_id) => {
                    let draw_list_group = &draw_list_groups[&draw_list_group_id];
                    debug_assert!(draw_list_group.draw_list_ids.len() <= MAX_MATRICES_PER_BATCH);

                    let layer = &layers[&draw_list_group.scroll_layer_id];
                    let mut matrix_palette =
                        vec![Matrix4D::identity(); draw_list_group.draw_list_ids.len()];
                    let mut offset_palette =
                        vec![OffsetParams::identity(); draw_list_group.draw_list_ids.len()];

                    // Update batch matrices
                    let mut z_clear_needed = false;
                    for (index, draw_list_id) in draw_list_group.draw_list_ids.iter().enumerate() {
                        let draw_list = resource_cache.get_draw_list(*draw_list_id);

                        match draw_list.stacking_context_index {
                            Some(StackingContextIndex(stacking_context_id)) => {
                                let context = &stacking_context_info[stacking_context_id];
                                if context.z_clear_needed {
                                    z_clear_needed = true
                                }

                                let mut world_transform_in_render_target_space =
                                    layer.world_transform;
                                if self.texture_id.is_some() {
                                    // If we're rendering to a temporary target, the X/Y
                                    // translation part of the transform will be applied by whoever
                                    // composites us.
                                    world_transform_in_render_target_space.m41 = 0.0;
                                    world_transform_in_render_target_space.m42 = 0.0;
                                };
                                let transform =
                                    world_transform_in_render_target_space.mul(&context.transform);
                                matrix_palette[index] = transform;

                                offset_palette[index].stacking_context_x0 =
                                    context.offset_from_layer.x;
                                offset_palette[index].stacking_context_y0 =
                                    context.offset_from_layer.y;
                            }
                            None => {
                                // This can happen if the root pipeline was set before any stacking
                                // context was set for it (during navigation, usually). In that
                                // case we just render nothing.
                                continue
                            }
                        }
                    }

                    let mut batch_info = BatchInfo::new(matrix_palette, offset_palette);

                    // Collect relevant draws from each node in the tree.
                    let mut any_were_visible = false;
                    for node in &layer.aabb_tree.nodes {
                        if node.is_visible {
                            any_were_visible = true;

                            debug_assert!(node.compiled_node.is_some());
                            let compiled_node = node.compiled_node.as_ref().unwrap();

                            let batch_list = compiled_node.batch_list.iter().find(|batch_list| {
                                batch_list.draw_list_group_id == draw_list_group_id
                            });

                            if let Some(batch_list) = batch_list {
                                let mut region = MaskRegion::new();

                                let vertex_buffer_id = compiled_node.vertex_buffer_id.unwrap();

                                // TODO(gw): Support mask regions for nested render targets
                                //           with transforms.
                                if self.texture_id.is_none() {
                                    // Mask out anything outside this AABB tree node.
                                    // This is a requirement to ensure paint order is correctly
                                    // maintained since the batches are built in parallel.
                                    region.add_mask(node.split_rect, layer.world_transform);

                                    // Mask out anything outside this viewport. This is used
                                    // for things like clipping content that is outside a
                                    // transformed iframe.
                                    let mask_rect = layer.viewport_rect;
                                    region.add_mask(mask_rect, layer.viewport_transform);
                                }

                                for batch in &batch_list.batches {
                                    region.draw_calls.push(DrawCall {
                                        tile_params: batch.tile_params.clone(),     // TODO(gw): Move this instead?
                                        clip_rects: batch.clip_rects.clone(),
                                        vertex_buffer_id: vertex_buffer_id,
                                        color_texture_id: batch.color_texture_id,
                                        mask_texture_id: batch.mask_texture_id,
                                        first_instance: batch.first_instance,
                                        instance_count: batch.instance_count,
                                    });
                                }

                                batch_info.regions.push(region);
                            }
                        }
                    }

                    if any_were_visible {
                        // Add a clear command if necessary.
                        if z_clear_needed && !commands.is_empty() {
                            commands.push(DrawCommand::Clear(ClearInfo {
                                clear_color: false,
                                clear_z: true,
                                clear_stencil: true,
                            }))
                        }

                        // Finally, add the batch + draw calls
                        commands.push(DrawCommand::Batch(batch_info));
                    }
                }
            }
        }

        let mut child_layers = Vec::new();

        for child in &mut self.children {
            let child_layer = child.collect_and_sort_visible_batches(resource_cache,
                                                                     draw_list_groups,
                                                                     layers,
                                                                     stacking_context_info,
                                                                     device_pixel_ratio);

            child_layers.push(child_layer);
        }

        DrawLayer::new(self.id,
                       self.origin,
                       self.size,
                       self.texture_id,
                       commands,
                       child_layers)
    }

    fn reset(&mut self,
             pending_updates: &mut BatchUpdateList,
             resource_cache: &mut ResourceCache) {
        self.texture_id_list.clear();
        resource_cache.free_old_render_targets();

        for mut child in &mut self.children.drain(..) {
            child.reset(pending_updates,
                        resource_cache);
        }

        self.items.clear();
        self.page_allocator = None;
    }

    fn push_composite(&mut self,
                      op: CompositionOp,
                      texture_id: TextureId,
                      target: Rect<f32>,
                      transform: &Matrix4D<f32>,
                      child_layer_index: ChildLayerIndex,
                      scroll_layer_id: ScrollLayerId,
                      z_clear_needed: bool) {
        // TODO(gw): Relax the restriction on batch breaks for FB reads
        //           once the proper render target allocation code is done!
        let need_new_batch = op.needs_framebuffer() || match self.items.last() {
            Some(&FrameRenderItem::CompositeBatch(ref info, _)) => {
                info.operation != op || info.texture_id != texture_id
            }
            Some(&FrameRenderItem::DrawListBatch(..)) | None => true,
        };

        if need_new_batch {
            self.items.push(FrameRenderItem::CompositeBatch(CompositeBatchInfo {
                operation: op,
                texture_id: texture_id,
                jobs: Vec::new(),
                scroll_layer_id: scroll_layer_id,
            }, z_clear_needed));
        }

        // TODO(gw): This seems a little messy - restructure how current batch works!
        match self.items.last_mut().unwrap() {
            &mut FrameRenderItem::CompositeBatch(ref mut batch, ref mut old_z_clear_needed) => {
                let job = CompositeBatchJob {
                    rect: target,
                    transform: *transform,
                    child_layer_index: child_layer_index,
                };
                batch.jobs.push(job);

                if !*old_z_clear_needed && z_clear_needed {
                    *old_z_clear_needed = true
                }
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

pub type LayerMap = HashMap<ScrollLayerId, Layer, BuildHasherDefault<FnvHasher>>;

pub struct Frame {
    pub layers: LayerMap,
    pub pipeline_epoch_map: HashMap<PipelineId, Epoch, BuildHasherDefault<FnvHasher>>,
    pub pipeline_auxiliary_lists: HashMap<PipelineId,
                                          AuxiliaryLists,
                                          BuildHasherDefault<FnvHasher>>,
    pub pending_updates: BatchUpdateList,
    pub root: Option<RenderTarget>,
    pub stacking_context_info: Vec<StackingContextInfo>,
    next_render_target_id: RenderTargetId,
    next_draw_list_group_id: DrawListGroupId,
    draw_list_groups: HashMap<DrawListGroupId, DrawListGroup, BuildHasherDefault<FnvHasher>>,
    pub root_scroll_layer_id: Option<ScrollLayerId>,
    id: FrameId,
}

enum SceneItemKind<'a> {
    StackingContext(&'a SceneStackingContext, PipelineId),
    Pipeline(&'a ScenePipeline)
}

#[derive(Clone)]
struct SceneItemWithZOrder {
    item: SceneItem,
    z_index: i32,
}

impl<'a> SceneItemKind<'a> {
    fn collect_scene_items(&self, scene: &Scene) -> Vec<SceneItem> {
        let mut result = Vec::new();
        let stacking_context = match *self {
            SceneItemKind::StackingContext(stacking_context, _) => {
                &stacking_context.stacking_context
            }
            SceneItemKind::Pipeline(pipeline) => {
                if let Some(background_draw_list) = pipeline.background_draw_list {
                    result.push(SceneItem {
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
                result.push(item.clone());
            }
        }
        result
    }
}

trait StackingContextHelpers {
    fn needs_composition_operation_for_mix_blend_mode(&self) -> bool;
    fn composition_operations(&self, auxiliary_lists: &AuxiliaryLists) -> Vec<CompositionOp>;
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

    fn composition_operations(&self, auxiliary_lists: &AuxiliaryLists) -> Vec<CompositionOp> {
        let mut composition_operations = vec![];
        if self.needs_composition_operation_for_mix_blend_mode() {
            composition_operations.push(CompositionOp::MixBlend(self.mix_blend_mode));
        }
        for filter in auxiliary_lists.filters(&self.filters) {
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
            pipeline_epoch_map: HashMap::with_hasher(Default::default()),
            pending_updates: BatchUpdateList::new(),
            pipeline_auxiliary_lists: HashMap::with_hasher(Default::default()),
            root: None,
            layers: HashMap::with_hasher(Default::default()),
            stacking_context_info: Vec::new(),
            next_render_target_id: RenderTargetId(0),
            next_draw_list_group_id: DrawListGroupId(0),
            draw_list_groups: HashMap::with_hasher(Default::default()),
            root_scroll_layer_id: None,
            id: FrameId(0),
        }
    }

    pub fn reset(&mut self, resource_cache: &mut ResourceCache)
                 -> HashMap<ScrollLayerId, ScrollingState, BuildHasherDefault<FnvHasher>> {
        self.draw_list_groups.clear();
        self.pipeline_epoch_map.clear();
        self.stacking_context_info.clear();

        if let Some(mut root) = self.root.take() {
            root.reset(&mut self.pending_updates, resource_cache);
        }

        // Free any render targets from last frame.
        // TODO: This should really re-use existing targets here...
        let mut old_layer_scrolling_states = HashMap::with_hasher(Default::default());
        for (layer_id, mut old_layer) in &mut self.layers.drain() {
            old_layer.reset(&mut self.pending_updates);
            old_layer_scrolling_states.insert(layer_id, old_layer.scrolling);
        }

        // Advance to the next frame.
        self.id.0 += 1;

        old_layer_scrolling_states
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

    pub fn get_scroll_layer(&self,
                            cursor: &Point2D<f32>,
                            scroll_layer_id: ScrollLayerId,
                            parent_transform: &Matrix4D<f32>) -> Option<ScrollLayerId> {
        self.layers.get(&scroll_layer_id).and_then(|layer| {
            let transform = parent_transform.mul(&layer.local_transform);

            for child_layer_id in layer.children.iter().rev() {
                if let Some(layer_id) = self.get_scroll_layer(cursor,
                                                              *child_layer_id,
                                                              &transform) {
                    return Some(layer_id);
                }
            }

            match scroll_layer_id.info {
                ScrollLayerInfo::Fixed => {
                    None
                }
                ScrollLayerInfo::Scrollable(..) => {
                    let inv = transform.invert();
                    let z0 = -10000.0;
                    let z1 =  10000.0;

                    let p0 = inv.transform_point4d(&Point4D::new(cursor.x, cursor.y, z0, 1.0));
                    let p0 = Point3D::new(p0.x / p0.w,
                                          p0.y / p0.w,
                                          p0.z / p0.w);
                    let p1 = inv.transform_point4d(&Point4D::new(cursor.x, cursor.y, z1, 1.0));
                    let p1 = Point3D::new(p1.x / p1.w,
                                          p1.y / p1.w,
                                          p1.z / p1.w);

                    if ray_intersects_rect(p0, p1, layer.viewport_rect) {
                        Some(scroll_layer_id)
                    } else {
                        None
                    }
                }
            }
        })
    }

    pub fn get_scroll_layer_state(&self, device_pixel_ratio: f32) -> Vec<ScrollLayerState> {
        let mut result = vec![];
        for (scroll_layer_id, scroll_layer) in &self.layers {
            match scroll_layer_id.info {
                ScrollLayerInfo::Scrollable(_) => {
                    result.push(ScrollLayerState {
                        pipeline_id: scroll_layer.pipeline_id,
                        stacking_context_id: scroll_layer.stacking_context_id,
                        scroll_offset: scroll_layer.scrolling.offset,
                    })
                }
                ScrollLayerInfo::Fixed => {}
            }
        }
        result
    }

    pub fn scroll(&mut self,
                  mut delta: Point2D<f32>,
                  cursor: Point2D<f32>,
                  phase: ScrollEventPhase) {
        let root_scroll_layer_id = match self.root_scroll_layer_id {
            Some(root_scroll_layer_id) => root_scroll_layer_id,
            None => return,
        };

        let scroll_layer_id = match self.get_scroll_layer(&cursor,
                                                          root_scroll_layer_id,
                                                          &Matrix4D::identity()) {
            Some(scroll_layer_id) => scroll_layer_id,
            None => return,
        };

        let layer = self.layers.get_mut(&scroll_layer_id).unwrap();
        if layer.scrolling.started_bouncing_back && phase == ScrollEventPhase::Move(false) {
            return
        }

        let overscroll_amount = layer.overscroll_amount();
        let overscrolling = CAN_OVERSCROLL && (overscroll_amount.width != 0.0 ||
                                               overscroll_amount.height != 0.0);
        if overscrolling {
            if overscroll_amount.width != 0.0 {
                delta.x /= overscroll_amount.width.abs()
            }
            if overscroll_amount.height != 0.0 {
                delta.y /= overscroll_amount.height.abs()
            }
        }

        let is_unscrollable = layer.layer_size.width <= layer.viewport_rect.size.width &&
            layer.layer_size.height <= layer.viewport_rect.size.height;

        if layer.layer_size.width > layer.viewport_rect.size.width {
            layer.scrolling.offset.x = layer.scrolling.offset.x + delta.x;
            if is_unscrollable || !CAN_OVERSCROLL {
                layer.scrolling.offset.x = layer.scrolling.offset.x.min(0.0);
                layer.scrolling.offset.x =
                    layer.scrolling.offset.x.max(-layer.layer_size.width +
                                                 layer.viewport_rect.size.width);
            }
        }

        if layer.layer_size.height > layer.viewport_rect.size.height {
            layer.scrolling.offset.y = layer.scrolling.offset.y + delta.y;
            if is_unscrollable || !CAN_OVERSCROLL {
                layer.scrolling.offset.y = layer.scrolling.offset.y.min(0.0);
                layer.scrolling.offset.y =
                    layer.scrolling.offset.y.max(-layer.layer_size.height +
                                                 layer.viewport_rect.size.height);
            }
        }

        if phase == ScrollEventPhase::Start || phase == ScrollEventPhase::Move(true) {
            layer.scrolling.started_bouncing_back = false
        } else if overscrolling &&
                ((delta.x < 1.0 && delta.y < 1.0) || phase == ScrollEventPhase::End) {
            layer.scrolling.started_bouncing_back = true
        }

        layer.scrolling.offset.x = layer.scrolling.offset.x.round();
        layer.scrolling.offset.y = layer.scrolling.offset.y.round();

        if CAN_OVERSCROLL {
            layer.stretch_overscroll_spring();
        }
    }

    pub fn tick_scrolling_bounce_animations(&mut self) {
        for (_, layer) in &mut self.layers {
            layer.tick_scrolling_bounce_animation()
        }
    }

    pub fn create(&mut self,
                  scene: &Scene,
                  resource_cache: &mut ResourceCache,
                  pipeline_sizes: &mut HashMap<PipelineId, Size2D<f32>>,
                  device_pixel_ratio: f32) {
        if let Some(root_pipeline_id) = scene.root_pipeline_id {
            if let Some(root_pipeline) = scene.pipeline_map.get(&root_pipeline_id) {
                let old_layer_scrolling_states = self.reset(resource_cache);

                self.pipeline_auxiliary_lists = scene.pipeline_auxiliary_lists.clone();

                let root_stacking_context = scene.stacking_context_map
                                                 .get(&root_pipeline.root_stacking_context_id)
                                                 .unwrap();

                let root_scroll_layer_id = root_stacking_context.stacking_context
                                                                .scroll_layer_id
                                                                .expect("root layer must be a scroll layer!");
                self.root_scroll_layer_id = Some(root_scroll_layer_id);

                let root_target_id = self.next_render_target_id();

                let mut root_target = RenderTarget::new(root_target_id,
                                                        Point2D::zero(),
                                                        Point2D::zero(),
                                                        root_pipeline.viewport_size,
                                                        None);

                // Insert global position: fixed elements layer
                debug_assert!(self.layers.is_empty());
                let root_fixed_layer_id = ScrollLayerId::create_fixed(root_pipeline_id);
                self.layers.insert(
                    root_fixed_layer_id,
                    Layer::new(root_stacking_context.stacking_context.overflow.origin,
                               root_stacking_context.stacking_context.overflow.size,
                               &Rect::new(Point2D::zero(), root_pipeline.viewport_size),
                               &Matrix4D::identity(),
                               Matrix4D::identity(),
                               root_pipeline_id,
                               root_stacking_context.stacking_context.servo_id));

                // Work around borrow check on resource cache
                {
                    let mut context = FlattenContext {
                        resource_cache: resource_cache,
                        scene: scene,
                        pipeline_sizes: pipeline_sizes,
                        current_draw_list_group: None,
                        device_pixel_ratio: device_pixel_ratio,
                    };

                    let parent_info = FlattenInfo {
                        viewport_rect: Rect::new(Point2D::zero(), root_pipeline.viewport_size),
                        viewport_transform: Matrix4D::identity(),
                        offset_from_origin: Point2D::zero(),
                        offset_from_current_layer: Point2D::zero(),
                        default_scroll_layer_id: root_scroll_layer_id,
                        actual_scroll_layer_id: root_scroll_layer_id,
                        fixed_scroll_layer_id: root_fixed_layer_id,
                        current_clip_rect: MAX_RECT,
                        transform: Matrix4D::identity(),
                        perspective: Matrix4D::identity(),
                        pipeline_id: root_pipeline_id,
                    };

                    let root_pipeline = SceneItemKind::Pipeline(root_pipeline);
                    self.flatten(root_pipeline,
                                 &parent_info,
                                 &mut context,
                                 &mut root_target,
                                 0);
                    self.root = Some(root_target);

                    if let Some(last_draw_list_group) = context.current_draw_list_group.take() {
                        self.draw_list_groups.insert(last_draw_list_group.id,
                                                     last_draw_list_group);
                    }
                }

                // TODO(gw): These are all independent - can be run through thread pool if it shows up in the profile!
                for (scroll_layer_id, layer) in &mut self.layers {
                    let scrolling_state = match old_layer_scrolling_states.get(&scroll_layer_id) {
                        Some(old_scrolling_state) => *old_scrolling_state,
                        None => ScrollingState::new(),
                    };

                    layer.finalize(&scrolling_state);
                }
            }
        }
    }

    fn add_items_to_target(&mut self,
                           scene_items: &[SceneItem],
                           info: &FlattenInfo,
                           target: &mut RenderTarget,
                           context: &mut FlattenContext,
                           _level: i32,
                           z_clear_needed: bool) {
        let stacking_context_index = StackingContextIndex(self.stacking_context_info.len());
        self.stacking_context_info.push(StackingContextInfo {
            offset_from_layer: info.offset_from_current_layer,
            local_clip_rect: info.current_clip_rect,
            transform: info.transform,
            perspective: info.perspective,
            z_clear_needed: z_clear_needed,
        });

        for item in scene_items {
            match item.specific {
                SpecificSceneItem::DrawList(draw_list_id) => {
                    let draw_list = context.resource_cache.get_draw_list_mut(draw_list_id);

                    // Store draw context
                    draw_list.stacking_context_index = Some(stacking_context_index);

                    let needs_new_draw_group = match context.current_draw_list_group {
                        Some(ref draw_list_group) => {
                            !draw_list_group.can_add(info.actual_scroll_layer_id,
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
                                                                     info.actual_scroll_layer_id,
                                                                     target.id);

                        target.push_draw_list_group(draw_list_group_id);

                        context.current_draw_list_group = Some(new_draw_list_group);
                    }

                    context.current_draw_list_group.as_mut().unwrap().push(draw_list_id);

                    let draw_list_group_id = context.current_draw_list_group.as_ref().unwrap().id;
                    let layer = self.layers.get_mut(&info.actual_scroll_layer_id).unwrap();
                    for (item_index, item) in draw_list.items.iter().enumerate() {
                        let item_index = DrawListItemIndex(item_index as u32);
                        let rect = item.rect
                                       .translate(&info.offset_from_current_layer);
                        layer.insert(rect,
                                     draw_list_group_id,
                                     draw_list_id,
                                     item_index);
                    }
                }
                SpecificSceneItem::StackingContext(id, pipeline_id) => {
                    let stacking_context = context.scene
                                                  .stacking_context_map
                                                  .get(&id)
                                                  .unwrap();

                    let child = SceneItemKind::StackingContext(stacking_context, pipeline_id);
                    self.flatten(child,
                                 info,
                                 context,
                                 target,
                                 _level+1);
                }
                SpecificSceneItem::Iframe(ref iframe_info) => {
                    let pipeline = context.scene
                                          .pipeline_map
                                          .get(&iframe_info.id);

                    context.pipeline_sizes.insert(iframe_info.id,
                                                  iframe_info.bounds.size);

                    if let Some(pipeline) = pipeline {
                        let iframe = SceneItemKind::Pipeline(pipeline);

                        let iframe_fixed_layer_id = ScrollLayerId::create_fixed(pipeline.pipeline_id);

                        // TODO(servo/servo#9983, pcwalton): Support rounded rectangle clipping.
                        // Currently we take the main part of the clip rect only.
                        let offset_from_origin = info.offset_from_origin +
                            iframe_info.bounds.origin;
                        let clipped_iframe_bounds =
                            iframe_info.bounds
                                       .intersection(&iframe_info.clip.main)
                                       .unwrap_or(Rect::new(Point2D::zero(), Size2D::zero()))
                                       .translate(&info.offset_from_origin);
                        let iframe_info = FlattenInfo {
                            viewport_rect: clipped_iframe_bounds,
                            viewport_transform: Matrix4D::identity(),
                            offset_from_origin: offset_from_origin,
                            offset_from_current_layer: info.offset_from_current_layer + iframe_info.bounds.origin,
                            default_scroll_layer_id: info.default_scroll_layer_id,
                            actual_scroll_layer_id: info.actual_scroll_layer_id,
                            fixed_scroll_layer_id: iframe_fixed_layer_id,
                            current_clip_rect: MAX_RECT,
                            transform: info.transform,
                            perspective: info.perspective,
                            pipeline_id: pipeline.pipeline_id,
                        };

                        let iframe_stacking_context = context.scene
                                                             .stacking_context_map
                                                             .get(&pipeline.root_stacking_context_id)
                                                             .unwrap();

                        let layer_origin = iframe_info.offset_from_origin;
                        let layer_size = iframe_stacking_context.stacking_context.overflow.size;

                        self.layers.insert(iframe_fixed_layer_id,
                                           Layer::new(layer_origin,
                                                      layer_size,
                                                      &iframe_info.viewport_rect,
                                                      &iframe_info.transform,
                                                      iframe_info.transform,
                                                      pipeline.pipeline_id,
                                                      iframe_stacking_context.stacking_context
                                                                             .servo_id));

                        self.flatten(iframe,
                                     &iframe_info,
                                     context,
                                     target,
                                     _level+1);
                    }
                }
            }
        }
    }

    fn flatten(&mut self,
               scene_item: SceneItemKind,
               parent_info: &FlattenInfo,
               context: &mut FlattenContext,
               target: &mut RenderTarget,
               level: i32) {
        let _pf = util::ProfileScope::new("  flatten");

        let (stacking_context, pipeline_id) = match scene_item {
            SceneItemKind::StackingContext(stacking_context, pipeline_id) => {
                (&stacking_context.stacking_context, pipeline_id)
            }
            SceneItemKind::Pipeline(pipeline) => {
                self.pipeline_epoch_map.insert(pipeline.pipeline_id, pipeline.epoch);

                let stacking_context = &context.scene.stacking_context_map
                                               .get(&pipeline.root_stacking_context_id)
                                               .unwrap()
                                               .stacking_context;

                (stacking_context, pipeline.pipeline_id)
            }
        };

        // FIXME(pcwalton): This is a not-great partial solution to servo/servo#10164. A better
        // solution would be to do this only if the transform consists of a translation+scale only
        // and fall back to stenciling if the object has a more complex transform.
        let local_clip_rect =
            match (stacking_context.scroll_policy, stacking_context.scroll_layer_id) {
                (ScrollPolicy::Scrollable, Some(ScrollLayerId {
                    info: ScrollLayerInfo::Scrollable(index),
                    ..
                })) if index > 0 => {
                    Some(stacking_context.transform
                                         .invert()
                                         .transform_rect(&stacking_context.overflow))
                }
                _ => {
                    stacking_context.transform
                                    .invert()
                                    .transform_rect(&parent_info.current_clip_rect)
                                    .translate(&-stacking_context.bounds.origin)
                                    .intersection(&stacking_context.overflow)
                }
            };

        if let Some(local_clip_rect) = local_clip_rect {
            let scene_items = scene_item.collect_scene_items(&context.scene);
            if !scene_items.is_empty() {
                let composition_operations = {
                    let auxiliary_lists = self.pipeline_auxiliary_lists
                                              .get(&pipeline_id)
                                              .expect("No auxiliary lists?!");
                    stacking_context.composition_operations(auxiliary_lists)
                };

                // Detect composition operations that will make us invisible.
                for composition_operation in &composition_operations {
                    match *composition_operation {
                        CompositionOp::Filter(LowLevelFilterOp::Opacity(Au(0))) => return,
                        _ => {}
                    }
                }

                // Build world space transform
                let origin = parent_info.offset_from_current_layer + stacking_context.bounds.origin;
                let local_transform = if composition_operations.is_empty() {
                    Matrix4D::identity().translate(origin.x, origin.y, 0.0)
                                        .mul(&stacking_context.transform)
                                        .translate(-origin.x, -origin.y, 0.0)
                } else {
                    Matrix4D::identity()
                };

                let transform = parent_info.perspective.mul(&parent_info.transform)
                                                       .mul(&local_transform);

                // Build world space perspective transform
                let perspective = Matrix4D::identity().translate(origin.x, origin.y, 0.0)
                                                      .mul(&stacking_context.perspective)
                                                      .translate(-origin.x, -origin.y, 0.0);

                let viewport_rect = if composition_operations.is_empty() {
                    parent_info.viewport_rect
                } else {
                    Rect::new(Point2D::new(0.0, 0.0), parent_info.viewport_rect.size)
                };

                let viewport_transform = transform;

                let mut info = FlattenInfo {
                    viewport_rect: viewport_rect,
                    viewport_transform: viewport_transform,
                    offset_from_origin: parent_info.offset_from_origin + stacking_context.bounds.origin,
                    offset_from_current_layer: parent_info.offset_from_current_layer + stacking_context.bounds.origin,
                    default_scroll_layer_id: parent_info.default_scroll_layer_id,
                    actual_scroll_layer_id: parent_info.default_scroll_layer_id,
                    fixed_scroll_layer_id: parent_info.fixed_scroll_layer_id,
                    current_clip_rect: local_clip_rect,
                    transform: transform,
                    perspective: perspective,
                    pipeline_id: parent_info.pipeline_id,
                };

                match (stacking_context.scroll_policy, stacking_context.scroll_layer_id) {
                    (ScrollPolicy::Fixed, _scroll_layer_id) => {
                        debug_assert!(_scroll_layer_id.is_none());
                        info.actual_scroll_layer_id = info.fixed_scroll_layer_id;
                    }
                    (ScrollPolicy::Scrollable, Some(scroll_layer_id)) => {
                        debug_assert!(!self.layers.contains_key(&scroll_layer_id));
                        let (viewport_rect, viewport_transform) = match scroll_layer_id.info {
                            ScrollLayerInfo::Scrollable(index) if index > 0 => {
                                let mut stacking_context_rect =
                                    Rect::new(parent_info.offset_from_origin,
                                              stacking_context.bounds.size);
                                (parent_info.viewport_rect
                                            .intersection(&stacking_context_rect)
                                            .unwrap_or(Rect::new(Point2D::new(0.0, 0.0),
                                                                 Size2D::new(0.0, 0.0))),
                                 Matrix4D::identity())
                            }
                            _ if transform.can_losslessly_transform_a_2d_rect() => {
                                // FIXME(pcwalton): This is pretty much just a hack for
                                // browser.html to stave off `viewport_rect` becoming a full stack
                                // of matrices and clipping regions as long as we can.
                                (transform.transform_rect(&parent_info.viewport_rect),
                                 Matrix4D::identity())
                            }
                            _ => (parent_info.viewport_rect, transform),
                        };
                        let layer = Layer::new(parent_info.offset_from_origin,
                                               stacking_context.overflow.size,
                                               &viewport_rect,
                                               &viewport_transform,
                                               transform,
                                               parent_info.pipeline_id,
                                               stacking_context.servo_id);
                        if parent_info.actual_scroll_layer_id != scroll_layer_id {
                            self.layers.get_mut(&parent_info.actual_scroll_layer_id).unwrap().add_child(scroll_layer_id);
                        }
                        self.layers.insert(scroll_layer_id, layer);
                        info.viewport_rect = viewport_rect;
                        info.default_scroll_layer_id = scroll_layer_id;
                        info.actual_scroll_layer_id = scroll_layer_id;
                        info.offset_from_current_layer = Point2D::zero();
                        info.transform = Matrix4D::identity();
                        info.perspective = Matrix4D::identity();
                    }
                    (ScrollPolicy::Scrollable, None) => {
                        // Nothing to do - use defaults as set above.
                    }
                }

                // When establishing a new 3D context, clear Z. This is only needed if there
                // are child stacking contexts, otherwise it is a redundant clear.
                let z_clear_needed = stacking_context.establishes_3d_context &&
                   stacking_context.has_stacking_contexts;

                // TODO: Account for scroll offset with transforms!
                if composition_operations.is_empty() {
                    self.add_items_to_target(&scene_items,
                                             &info,
                                             target,
                                             context,
                                             level,
                                             z_clear_needed);
                } else {
                    // TODO(gw): This makes the reftests work (mix_blend_mode) and
                    //           inline_stacking_context, but it seems wrong.
                    //           Need to investigate those and see what the root
                    //           issue is...
                    let empty_stacking_context = stacking_context.bounds.size.width == 0.0 ||
                                                 stacking_context.bounds.size.height == 0.0;
                    let target_size = if empty_stacking_context {
                        stacking_context.overflow.size
                    } else {
                        stacking_context.bounds.size
                    };
                    let target_origin = Point2D::new(info.offset_from_origin.x,
                                                     info.offset_from_origin.y);
                    let unfiltered_target_rect = Rect::new(target_origin, target_size);
                    let mut target_rect = unfiltered_target_rect;
                    for composition_operation in &composition_operations {
                        target_rect = composition_operation.target_rect(&target_rect);
                    }

                    let child_layer_index = ChildLayerIndex(target.children.len() as u32);

                    let render_target_size = Size2D::new(target_rect.size.width,
                                                         target_rect.size.height);
                    let render_target_id = self.next_render_target_id();

                    let (origin, texture_id) =
                        target.allocate_target_rect(target_rect.size.width,
                                                    target_rect.size.height,
                                                    context.device_pixel_ratio,
                                                    context.resource_cache,
                                                    self.id);

                    let mut new_target = RenderTarget::new(render_target_id,
                                                           origin,
                                                           target_rect.origin,
                                                           render_target_size,
                                                           Some(texture_id));

                    let local_transform =
                        Matrix4D::identity().translate(origin.x, origin.y, 0.0)
                                            .mul(&stacking_context.transform)
                                            .translate(-origin.x, -origin.y, 0.0);
                    for composition_operation in composition_operations {
                        target.push_composite(composition_operation,
                                              texture_id,
                                              target_rect,
                                              &local_transform,
                                              child_layer_index,
                                              info.actual_scroll_layer_id,
                                              z_clear_needed);
                    }

                    info.offset_from_current_layer = Point2D::zero();

                    self.add_items_to_target(&scene_items,
                                             &info,
                                             &mut new_target,
                                             context,
                                             level,
                                             z_clear_needed);

                    target.children.push(new_target);
                }
            }
        }
    }

    pub fn build(&mut self,
                 resource_cache: &mut ResourceCache,
                 thread_pool: &mut scoped_threadpool::Pool,
                 device_pixel_ratio: f32)
                 -> RendererFrame {
        // Traverse layer trees to calculate visible nodes
        for (_, layer) in &mut self.layers {
            layer.cull();
        }

        // Build resource list for newly visible nodes
        self.update_resource_lists(resource_cache, thread_pool);

        // Update texture cache and build list of raster jobs.
        self.update_texture_cache_and_build_raster_jobs(resource_cache);

        // Rasterize needed glyphs on worker threads
        self.raster_glyphs(thread_pool, resource_cache);

        // Compile nodes that have become visible
        self.compile_visible_nodes(thread_pool, resource_cache, device_pixel_ratio);

        // Update the batch cache from newly compiled nodes
        self.update_batch_cache();

        // Update the layer transform matrices
        self.update_layer_transforms();

        // Collect the visible batches into a frame
        let frame = self.collect_and_sort_visible_batches(resource_cache, device_pixel_ratio);

        resource_cache.expire_old_resources(self.id);

        frame
    }

    fn update_layer_transform(&mut self,
                              layer_id: ScrollLayerId,
                              parent_transform: &Matrix4D<f32>) {
        // TODO(gw): This is an ugly borrow check workaround to clone these.
        //           Restructure this to avoid the clones!
        let (layer_transform_for_children, layer_children) = {
            match self.layers.get_mut(&layer_id) {
                Some(layer) => {
                    let layer_transform_for_children =
                        parent_transform.mul(&layer.local_transform);
                    layer.world_transform =
                        layer_transform_for_children.translate(layer.world_origin.x,
                                                               layer.world_origin.y,
                                                               0.0)
                                                    .translate(layer.scrolling.offset.x,
                                                               layer.scrolling.offset.y,
                                                               0.0);
                    (layer_transform_for_children, layer.children.clone())
                }
                None => {
                    return;
                }
            }
        };

        for child_layer_id in layer_children {
            self.update_layer_transform(child_layer_id, &layer_transform_for_children);
        }
    }

    fn update_layer_transforms(&mut self) {
        if let Some(root_scroll_layer_id) = self.root_scroll_layer_id {
            self.update_layer_transform(root_scroll_layer_id, &Matrix4D::identity());
        }

        // Update any fixed layers
        let mut fixed_layers = Vec::new();
        for (layer_id, _) in &self.layers {
            match layer_id.info {
                ScrollLayerInfo::Scrollable(..) => {}
                ScrollLayerInfo::Fixed => {
                    fixed_layers.push(*layer_id);
                }
            }
        }

        for layer_id in fixed_layers {
            self.update_layer_transform(layer_id, &Matrix4D::identity());
        }
    }

    pub fn update_resource_lists(&mut self,
                                 resource_cache: &ResourceCache,
                                 thread_pool: &mut scoped_threadpool::Pool) {
        let _pf = util::ProfileScope::new("  update_resource_lists");

        for (_, layer) in &mut self.layers {
            let nodes = &mut layer.aabb_tree.nodes;
            let pipeline_auxiliary_lists = &self.pipeline_auxiliary_lists;

            thread_pool.scoped(|scope| {
                for node in nodes {
                    if node.is_visible && node.compiled_node.is_none() {
                        scope.execute(move || {
                            node.build_resource_list(resource_cache, pipeline_auxiliary_lists);
                        });
                    }
                }
            });
        }
    }

    pub fn update_texture_cache_and_build_raster_jobs(&mut self,
                                                      resource_cache: &mut ResourceCache) {
        let _pf = util::ProfileScope::new("  update_texture_cache_and_build_raster_jobs");

        let frame_id = self.id;
        for (_, layer) in &self.layers {
            for node in &layer.aabb_tree.nodes {
                if node.is_visible {
                    let resource_list = node.resource_list.as_ref().unwrap();
                    resource_cache.add_resource_list(resource_list, frame_id);
                }
            }
        }
    }

    pub fn raster_glyphs(&mut self,
                         thread_pool: &mut scoped_threadpool::Pool,
                         resource_cache: &mut ResourceCache) {
        let _pf = util::ProfileScope::new("  raster_glyphs");
        resource_cache.raster_pending_glyphs(thread_pool, self.id);
    }

    pub fn compile_visible_nodes(&mut self,
                                 thread_pool: &mut scoped_threadpool::Pool,
                                 resource_cache: &ResourceCache,
                                 device_pixel_ratio: f32) {
        let _pf = util::ProfileScope::new("  compile_visible_nodes");

        let layers = &mut self.layers;
        let stacking_context_info = &self.stacking_context_info;
        let draw_list_groups = &self.draw_list_groups;
        let frame_id = self.id;
        let pipeline_auxiliary_lists = &self.pipeline_auxiliary_lists;

        thread_pool.scoped(|scope| {
            for (_, layer) in layers {
                let nodes = &mut layer.aabb_tree.nodes;
                for node in nodes {
                    if node.is_visible && node.compiled_node.is_none() {
                        scope.execute(move || {
                            node.compile(resource_cache,
                                         frame_id,
                                         device_pixel_ratio,
                                         stacking_context_info,
                                         draw_list_groups,
                                         pipeline_auxiliary_lists);
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
                            op: BatchUpdateOp::Create(vertex_buffer.vertices),
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
                DrawLayer::new(RenderTargetId(0),
                               Point2D::zero(),
                               Size2D::zero(),
                               None,
                               Vec::new(),
                               Vec::new())
            }
        };

        let layers_bouncing_back = self.collect_layers_bouncing_back();
        RendererFrame::new(self.pipeline_epoch_map.clone(), layers_bouncing_back, root_layer)
    }

    fn collect_layers_bouncing_back(&self)
                                    -> HashSet<ScrollLayerId, BuildHasherDefault<FnvHasher>> {
        let mut layers_bouncing_back = HashSet::with_hasher(Default::default());
        for (scroll_layer_id, layer) in &self.layers {
            if layer.scrolling.started_bouncing_back {
                layers_bouncing_back.insert(*scroll_layer_id);
            }
        }
        layers_bouncing_back
    }

    pub fn root_scroll_layer_for_pipeline(&self, pipeline_id: PipelineId)
                                          -> Option<ScrollLayerId> {
        let root_scroll_layer_id = match self.root_scroll_layer_id {
            Some(root_scroll_layer_id) => root_scroll_layer_id,
            None => return None,
        };
        return search(&self.layers, root_scroll_layer_id, pipeline_id);

        fn search(layers: &LayerMap, layer_id: ScrollLayerId, query: PipelineId)
                  -> Option<ScrollLayerId> {
            let layer = layers.get(&layer_id).expect("No layer with that ID!");
            if layer.pipeline_id == query {
                return Some(layer_id)
            }
            for &kid in &layer.children {
                if let Some(layer_id) = search(layers, kid, query) {
                    return Some(layer_id)
                }
            }
            None
        }
    }
}

