/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use euclid::{Matrix4D, Point2D, Point3D, Point4D, Rect, Size2D};
use fnv::FnvHasher;
use geometry::ray_intersects_rect;
use internal_types::{ANGLE_FLOAT_TO_FIXED, AxisDirection};
use internal_types::{CompositionOp};
use internal_types::{LowLevelFilterOp, max_rect};
use internal_types::{RendererFrame};
use layer::{Layer, ScrollingState};
use resource_cache::ResourceCache;
use scene::{SceneStackingContext, ScenePipeline, Scene, SceneItem, SpecificSceneItem};
use scoped_threadpool;
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;
use tiling::{Clip, FrameBuilder, FrameBuilderConfig, InsideTest};
use util::{MatrixHelpers};
use webrender_traits::{AuxiliaryLists, PipelineId, Epoch, ScrollPolicy, ScrollLayerId};
use webrender_traits::{StackingContext, FilterOp, MixBlendMode};
use webrender_traits::{ScrollEventPhase, ScrollLayerInfo, SpecificDisplayItem, ScrollLayerState};

#[cfg(target_os = "macos")]
const CAN_OVERSCROLL: bool = true;

#[cfg(not(target_os = "macos"))]
const CAN_OVERSCROLL: bool = false;

#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct FrameId(pub u32);

struct FlattenContext<'a> {
    resource_cache: &'a mut ResourceCache,
    scene: &'a Scene,
    pipeline_sizes: &'a mut HashMap<PipelineId, Size2D<f32>>,
    builder: &'a mut FrameBuilder,
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

    /// The transform relative to the parent layer.
    local_transform: Matrix4D<f32>,

    /// The perspective relative to the parent layer.
    local_perspective: Matrix4D<f32>,

    /// The final concatenated transform, taking perspective into account.
    world_transform: Matrix4D<f32>,

    pipeline_id: PipelineId,
}

pub type LayerMap = HashMap<ScrollLayerId, Layer, BuildHasherDefault<FnvHasher>>;

pub struct Frame {
    pub layers: LayerMap,
    pub pipeline_epoch_map: HashMap<PipelineId, Epoch, BuildHasherDefault<FnvHasher>>,
    pub pipeline_auxiliary_lists: HashMap<PipelineId,
                                          AuxiliaryLists,
                                          BuildHasherDefault<FnvHasher>>,
    pub root_scroll_layer_id: Option<ScrollLayerId>,
    id: FrameId,
    debug: bool,
    frame_builder_config: FrameBuilderConfig,
    frame_builder: Option<FrameBuilder>,
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
    pub fn new(debug: bool, config: FrameBuilderConfig) -> Frame {
        Frame {
            pipeline_epoch_map: HashMap::with_hasher(Default::default()),
            pipeline_auxiliary_lists: HashMap::with_hasher(Default::default()),
            layers: HashMap::with_hasher(Default::default()),
            root_scroll_layer_id: None,
            id: FrameId(0),
            debug: debug,
            frame_builder: None,
            frame_builder_config: config,
        }
    }

    pub fn reset(&mut self)
                 -> HashMap<ScrollLayerId, ScrollingState, BuildHasherDefault<FnvHasher>> {
        self.pipeline_epoch_map.clear();

        // Free any render targets from last frame.
        // TODO: This should really re-use existing targets here...
        let mut old_layer_scrolling_states = HashMap::with_hasher(Default::default());
        for (layer_id, old_layer) in &mut self.layers.drain() {
            old_layer_scrolling_states.insert(layer_id, old_layer.scrolling);
        }

        // Advance to the next frame.
        self.id.0 += 1;

        old_layer_scrolling_states
    }

    pub fn get_scroll_layer(&self, cursor: &Point2D<f32>, scroll_layer_id: ScrollLayerId)
                            -> Option<ScrollLayerId> {
        self.layers.get(&scroll_layer_id).and_then(|layer| {
            for child_layer_id in layer.children.iter().rev() {
                if let Some(layer_id) = self.get_scroll_layer(cursor, *child_layer_id) {
                    return Some(layer_id);
                }
            }

            match scroll_layer_id.info {
                ScrollLayerInfo::Fixed => {
                    None
                }
                ScrollLayerInfo::Scrollable(..) => {
                    let inv = layer.viewport_transform.invert();
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

                    let is_unscrollable = layer.layer_size.width <= layer.viewport_rect.size.width &&
                        layer.layer_size.height <= layer.viewport_rect.size.height;
                    if is_unscrollable {
                        None
                    } else {
                        let result = ray_intersects_rect(p0, p1, layer.viewport_rect);
                        if result {
                            Some(scroll_layer_id)
                        } else {
                            None
                        }
                    }
                }
            }
        })
    }

    pub fn get_scroll_layer_state(&self) -> Vec<ScrollLayerState> {
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

    /// Returns true if any layers actually changed position or false otherwise.
    pub fn scroll(&mut self,
                  mut delta: Point2D<f32>,
                  cursor: Point2D<f32>,
                  phase: ScrollEventPhase)
                  -> bool {
        let root_scroll_layer_id = match self.root_scroll_layer_id {
            Some(root_scroll_layer_id) => root_scroll_layer_id,
            None => return false,
        };

        let scroll_layer_id = match self.get_scroll_layer(&cursor, root_scroll_layer_id) {
            Some(scroll_layer_id) => scroll_layer_id,
            None => return false,
        };

        let layer = self.layers.get_mut(&scroll_layer_id).unwrap();
        if layer.scrolling.started_bouncing_back && phase == ScrollEventPhase::Move(false) {
            return false
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

        let original_layer_scroll_offset = layer.scrolling.offset;

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
            layer.scrolling.started_bouncing_back = true;
            layer.scrolling.bouncing_back = true
        }

        layer.scrolling.offset.x = layer.scrolling.offset.x.round();
        layer.scrolling.offset.y = layer.scrolling.offset.y.round();

        if CAN_OVERSCROLL {
            layer.stretch_overscroll_spring();
        }

        layer.scrolling.offset != original_layer_scroll_offset || layer.scrolling
                                                                       .started_bouncing_back
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
                let old_layer_scrolling_states = self.reset();

                self.pipeline_auxiliary_lists = scene.pipeline_auxiliary_lists.clone();

                let root_stacking_context = scene.stacking_context_map
                                                 .get(&root_pipeline.root_stacking_context_id)
                                                 .unwrap();

                let root_scroll_layer_id = root_stacking_context.stacking_context
                                                                .scroll_layer_id
                                                                .expect("root layer must be a scroll layer!");
                self.root_scroll_layer_id = Some(root_scroll_layer_id);

                // Insert global position: fixed elements layer
                debug_assert!(self.layers.is_empty());
                let root_fixed_layer_id = ScrollLayerId::create_fixed(root_pipeline_id);
                self.layers.insert(
                    root_fixed_layer_id,
                    Layer::new(root_stacking_context.stacking_context.overflow.origin,
                               root_stacking_context.stacking_context.overflow.size,
                               &Rect::new(Point2D::zero(), root_pipeline.viewport_size),
                               &Matrix4D::identity(),
                               &Matrix4D::identity(),
                               &Matrix4D::identity(),
                               root_pipeline_id,
                               root_stacking_context.stacking_context.servo_id));

                // Work around borrow check on resource cache
                {
                    let mut frame_builder = FrameBuilder::new(root_pipeline.viewport_size,
                                                              device_pixel_ratio,
                                                              self.debug,
                                                              self.frame_builder_config);

                    {
                        let mut context = FlattenContext {
                            resource_cache: resource_cache,
                            scene: scene,
                            pipeline_sizes: pipeline_sizes,
                            builder: &mut frame_builder,
                        };

                        let parent_info = FlattenInfo {
                            viewport_rect: Rect::new(Point2D::zero(), root_pipeline.viewport_size),
                            viewport_transform: Matrix4D::identity(),
                            offset_from_origin: Point2D::zero(),
                            offset_from_current_layer: Point2D::zero(),
                            default_scroll_layer_id: root_scroll_layer_id,
                            actual_scroll_layer_id: root_scroll_layer_id,
                            fixed_scroll_layer_id: root_fixed_layer_id,
                            current_clip_rect: max_rect(),
                            local_transform: Matrix4D::identity(),
                            local_perspective: Matrix4D::identity(),
                            world_transform: Matrix4D::identity(),
                            pipeline_id: root_pipeline_id,
                        };

                        let root_pipeline = SceneItemKind::Pipeline(root_pipeline);
                        self.flatten(root_pipeline,
                                     &parent_info,
                                     &mut context);
                    }

                    self.frame_builder = Some(frame_builder);
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
                           context: &mut FlattenContext,
                           sc_rect: Rect<f32>,
                           composition_ops: Vec<CompositionOp>) {
        context.builder.push_layer(sc_rect,
                                   info.current_clip_rect,
                                   info.world_transform,
                                   info.pipeline_id,
                                   info.actual_scroll_layer_id,
                                   info.offset_from_current_layer,
                                   composition_ops);

        for item in scene_items {
            match item.specific {
                SpecificSceneItem::DrawList(draw_list_id) => {
                    let draw_list = context.resource_cache.get_draw_list(draw_list_id);
                    let builder = &mut context.builder;

                    let auxiliary_lists = {
                        self.pipeline_auxiliary_lists
                                              .get(&info.pipeline_id)
                                              .expect("No auxiliary lists?!")
                    };

                    for item in &draw_list.items {
                        let clips = auxiliary_lists.complex_clip_regions(&item.clip.complex);
                        let clip = match clips.len() {
                            0 => None,
                            1 => Some(Box::new(Clip::from_clip_region(&clips[0]))),
                            _ => {
                                let internal_clip = clips.last().unwrap();
                                if clips.iter().all(|current_clip| current_clip.might_contain(internal_clip)) {
                                    Some(Box::new(Clip::from_clip_region(internal_clip)))
                                } else {
                                    Some(Box::new(Clip::from_clip_region(&clips[0])))
                                }
                            },
                        };

                        match item.item {
                            SpecificDisplayItem::WebGL(ref info) => {
                                builder.add_webgl_rectangle(item.rect,
                                                            &item.clip.main,
                                                            clip,
                                                            info.context_id);
                            }
                            SpecificDisplayItem::Image(ref info) => {
                                builder.add_image(item.rect,
                                                  &item.clip.main,
                                                  clip,
                                                  &info.stretch_size,
                                                  info.image_key,
                                                  info.image_rendering);
                            }
                            SpecificDisplayItem::Text(ref text_info) => {
                                builder.add_text(item.rect,
                                                 &item.clip.main,
                                                 clip,
                                                 text_info.font_key,
                                                 text_info.size,
                                                 text_info.blur_radius,
                                                 &text_info.color,
                                                 text_info.glyphs);
                            }
                            SpecificDisplayItem::Rectangle(ref info) => {
                                builder.add_solid_rectangle(&item.rect,
                                                            &item.clip.main,
                                                            clip,
                                                            &info.color);
                            }
                            SpecificDisplayItem::Gradient(ref info) => {
                                builder.add_gradient(item.rect,
                                                     &item.clip.main,
                                                     clip,
                                                     info.start_point,
                                                     info.end_point,
                                                     info.stops);
                            }
                            SpecificDisplayItem::BoxShadow(ref box_shadow_info) => {
                                builder.add_box_shadow(&box_shadow_info.box_bounds,
                                                       &item.clip.main,
                                                       clip,
                                                       &box_shadow_info.offset,
                                                       &box_shadow_info.color,
                                                       box_shadow_info.blur_radius,
                                                       box_shadow_info.spread_radius,
                                                       box_shadow_info.border_radius,
                                                       box_shadow_info.clip_mode);
                            }
                            SpecificDisplayItem::Border(ref info) => {
                                builder.add_border(item.rect,
                                                   &item.clip.main,
                                                   clip,
                                                   info);
                            }
                        }
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
                                 context);
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

                        let viewport_transform = info.world_transform.mul(&info.local_perspective)
                                                                     .mul(&info.local_transform);

                        // TODO(servo/servo#9983, pcwalton): Support rounded rectangle clipping.
                        // Currently we take the main part of the clip rect only.
                        // TODO(pcwalton): Allow this to work even when transformed
                        let offset_from_origin = info.offset_from_origin +
                            iframe_info.bounds.origin;
                        let clipped_iframe_bounds =
                            iframe_info.bounds
                                       .intersection(&iframe_info.clip.main)
                                       .unwrap_or(Rect::zero())
                                       .translate(&info.offset_from_origin)
                                       .intersection(&info.viewport_rect)
                                       .unwrap_or(Rect::zero());
                        let iframe_info = FlattenInfo {
                            viewport_rect: clipped_iframe_bounds,
                            viewport_transform: viewport_transform,
                            offset_from_origin: offset_from_origin,
                            offset_from_current_layer: info.offset_from_current_layer +
                                iframe_info.bounds.origin,
                            default_scroll_layer_id: info.default_scroll_layer_id,
                            actual_scroll_layer_id: info.actual_scroll_layer_id,
                            fixed_scroll_layer_id: iframe_fixed_layer_id,
                            current_clip_rect: max_rect(),
                            local_transform: info.local_transform,
                            local_perspective: info.local_perspective,
                            world_transform: info.world_transform,
                            pipeline_id: pipeline.pipeline_id,
                        };

                        let iframe_stacking_context = context.scene
                                                             .stacking_context_map
                                                             .get(&pipeline.root_stacking_context_id)
                                                             .unwrap();

                        let layer_origin = iframe_info.offset_from_origin;
                        let layer_size = iframe_stacking_context.stacking_context.overflow.size;

                        self.layers
                            .insert(iframe_fixed_layer_id,
                                    Layer::new(layer_origin,
                                               layer_size,
                                               &iframe_info.viewport_rect,
                                               &iframe_info.viewport_transform,
                                               &iframe_info.local_transform,
                                               &iframe_info.local_perspective,
                                               pipeline.pipeline_id,
                                               iframe_stacking_context.stacking_context.servo_id));

                        self.flatten(iframe,
                                     &iframe_info,
                                     context);
                    }
                }
            }
        }

        context.builder.pop_layer();
    }

    fn flatten(&mut self,
               scene_item: SceneItemKind,
               parent_info: &FlattenInfo,
               context: &mut FlattenContext) {
        //let _pf = util::ProfileScope::new("  flatten");

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

        // TODO(gw): This is broken for 3d transforms ref test.
        // Removing it fixes that but it needs a proper solution before
        // merging which will coming when the clipping branch merges!

        let local_clip_rect = Some(stacking_context.overflow);
        /*
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
            };*/

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

                // Build local and world space transform
                let origin = parent_info.offset_from_current_layer + stacking_context.bounds.origin;
                //let local_transform = if composition_operations.is_empty() {
                //    Matrix4D::identity().translate(origin.x, origin.y, 0.0)
                //                        .mul(&stacking_context.transform)
                //                        .translate(-origin.x, -origin.y, 0.0)
                //} else {
                //    Matrix4D::identity()
                //};
                let local_transform = Matrix4D::identity().translate(origin.x, origin.y, 0.0)
                                                          .mul(&stacking_context.transform)
                                                          .translate(-origin.x, -origin.y, 0.0);

                // Build local space perspective transform
                let local_perspective = Matrix4D::identity().translate(origin.x, origin.y, 0.0)
                                                            .mul(&stacking_context.perspective)
                                                            .translate(-origin.x, -origin.y, 0.0);

                // Build world space transform
                let world_transform = parent_info.world_transform.mul(&local_perspective)
                                                                 .mul(&local_transform);

                /*
                let viewport_rect = if composition_operations.is_empty() {
                    parent_info.viewport_rect
                } else {
                    Rect::new(Point2D::zero(), parent_info.viewport_rect.size)
                };
                */
                let viewport_rect = parent_info.viewport_rect;

                let mut info = FlattenInfo {
                    viewport_rect: viewport_rect,
                    viewport_transform: world_transform,
                    offset_from_origin: parent_info.offset_from_origin + stacking_context.bounds.origin,
                    offset_from_current_layer: parent_info.offset_from_current_layer + stacking_context.bounds.origin,
                    default_scroll_layer_id: parent_info.default_scroll_layer_id,
                    actual_scroll_layer_id: parent_info.default_scroll_layer_id,
                    fixed_scroll_layer_id: parent_info.fixed_scroll_layer_id,
                    current_clip_rect: local_clip_rect,
                    local_transform: local_transform,
                    local_perspective: local_perspective,
                    world_transform: world_transform,
                    pipeline_id: parent_info.pipeline_id,
                };

                match (stacking_context.scroll_policy, stacking_context.scroll_layer_id) {
                    (ScrollPolicy::Fixed, _scroll_layer_id) => {
                        debug_assert!(_scroll_layer_id.is_none());
                        info.actual_scroll_layer_id = info.fixed_scroll_layer_id;
                    }
                    (ScrollPolicy::Scrollable, Some(scroll_layer_id)) => {
                        debug_assert!(!self.layers.contains_key(&scroll_layer_id));
                        let mut viewport_transform =
                            world_transform.mul(&parent_info.local_perspective)
                                           .mul(&parent_info.local_transform);
                        let mut viewport_rect = parent_info.viewport_rect;

                        if let ScrollLayerInfo::Scrollable(index) = scroll_layer_id.info {
                            if index > 0 && viewport_transform == Matrix4D::identity() {
                                let stacking_context_rect =
                                    Rect::new(parent_info.offset_from_origin,
                                              stacking_context.bounds.size);
                                viewport_rect = viewport_rect.intersection(&stacking_context_rect)
                                                             .unwrap_or(Rect::zero())
                            }
                        }

                        if viewport_transform.can_losslessly_transform_a_2d_rect() {
                            // FIXME(pcwalton): This is pretty much just a hack for
                            // browser.html to stave off `viewport_rect` becoming a full stack
                            // of matrices and clipping regions as long as we can.
                            viewport_rect = viewport_transform.transform_rect(&viewport_rect);
                            viewport_transform = viewport_transform.reset_after_transforming_rect()
                        }

                        let layer = Layer::new(parent_info.offset_from_origin,
                                               stacking_context.overflow.size,
                                               &viewport_rect,
                                               &viewport_transform,
                                               &local_transform,
                                               &local_perspective,
                                               parent_info.pipeline_id,
                                               stacking_context.servo_id);
                        if parent_info.actual_scroll_layer_id != scroll_layer_id {
                            self.layers.get_mut(&parent_info.actual_scroll_layer_id).unwrap().add_child(scroll_layer_id);
                        }
                        self.layers.insert(scroll_layer_id, layer);
                        info.viewport_rect = viewport_rect;
                        info.viewport_transform = viewport_transform;
                        info.default_scroll_layer_id = scroll_layer_id;
                        info.actual_scroll_layer_id = scroll_layer_id;
                        info.offset_from_current_layer = Point2D::zero();
                    }
                    (ScrollPolicy::Scrollable, None) => {
                        // Nothing to do - use defaults as set above.
                    }
                }

                // TODO: Account for scroll offset with transforms!
                self.add_items_to_target(&scene_items,
                                         &info,
                                         context,
                                         stacking_context.overflow,
                                         composition_operations);
            }
        }
    }

    pub fn build(&mut self,
                 resource_cache: &mut ResourceCache,
                 _thread_pool: &mut scoped_threadpool::Pool,
                 _device_pixel_ratio: f32)
                 -> RendererFrame {
        self.update_layer_transforms();
        let frame = self.build_frame(resource_cache);
        resource_cache.expire_old_resources(self.id);
        frame
    }

    fn update_layer_transform(&mut self,
                              layer_id: ScrollLayerId,
                              parent_world_transform: &Matrix4D<f32>) {
        // TODO(gw): This is an ugly borrow check workaround to clone these.
        //           Restructure this to avoid the clones!
        let (layer_transform_for_children, layer_children) = {
            match self.layers.get_mut(&layer_id) {
                Some(layer) => {
                    let layer_transform_for_children =
                        parent_world_transform.mul(&layer.local_transform)
                                              .translate(layer.scrolling.offset.x,
                                                         layer.scrolling.offset.y,
                                                         0.0);
                    layer.viewport_transform = parent_world_transform.mul(&layer.local_transform);
                    layer.world_transform =
                        Some(layer_transform_for_children.translate(layer.world_origin.x,
                                                                    layer.world_origin.y,
                                                                    0.0));
                    (layer_transform_for_children, layer.children.clone())
                }
                None => return,
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

    fn build_frame(&mut self,
                   resource_cache: &mut ResourceCache) -> RendererFrame {
        let frame_builder = self.frame_builder.take();
        let frame = frame_builder.map(|mut builder| {
            builder.build(resource_cache,
                          self.id,
                          &self.pipeline_auxiliary_lists,
                          &self.layers)
        });

        let layers_bouncing_back = self.collect_layers_bouncing_back();
        RendererFrame::new(self.pipeline_epoch_map.clone(),
                           layers_bouncing_back,
                           frame)
    }

    fn collect_layers_bouncing_back(&self)
                                    -> HashSet<ScrollLayerId, BuildHasherDefault<FnvHasher>> {
        let mut layers_bouncing_back = HashSet::with_hasher(Default::default());
        for (scroll_layer_id, layer) in &self.layers {
            if layer.scrolling.bouncing_back {
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
