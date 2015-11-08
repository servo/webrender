use aabbtree::{AABBTreeNode, AABBTreeNodeInfo};
use app_units::Au;
use batch::RenderBatch;
use clipper::{self, ClipBuffers};
use device::{ProgramId, TextureId};
use euclid::{Rect, Point2D, Size2D, Matrix4};
use fnv::FnvHasher;
use internal_types::{ApiMsg, Frame, ResultMsg, DrawLayer, Primitive, ClearInfo};
use internal_types::{BorderRadiusRasterOp, BoxShadowRasterOp, DrawListID, RasterItem};
use internal_types::{BatchUpdateList, BatchId, BatchUpdate, BatchUpdateOp, CompiledNode};
use internal_types::{PackedVertex, WorkVertex, DisplayList, DrawCommand, DrawCommandInfo};
use internal_types::{ClipRectToRegionResult, DrawListIndex, DrawListItemIndex, DisplayItemKey};
use internal_types::{CompositeInfo, BorderEdgeDirection, RenderTargetIndex, GlyphKey};
use internal_types::{FontTemplate, Glyph, PolygonPosColorUv, RectPosUv, TextureTarget};
use internal_types::{ResourceId, IdNamespace, TiledImageKey, BasicRotationAngle};
use internal_types::{RectUv, CombinedClipRegion};
use layer::Layer;
use optimizer;
use render_api::RenderApi;
use renderer::BLUR_INFLATION_FACTOR;
use resource_cache::ResourceCache;
use resource_list::ResourceList;
use std::cell::Cell;
use std::collections::HashMap;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::hash_state::DefaultState;
use std::cmp::Ordering;
use std::f32;
use std::mem;
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT};
use std::sync::atomic::Ordering::SeqCst;
use std::sync::Arc;
use std::sync::mpsc::{Sender, Receiver};
use texture_cache::{TextureCache, TextureCacheItem, TextureCacheItemId};
use types::{DisplayListID, Epoch, FontKey, ImageKey, BorderDisplayItem, ScrollPolicy};
use types::{RectangleDisplayItem, ScrollLayerId, ClearDisplayItem};
use types::{GradientStop, DisplayListMode, ClipRegion};
use types::{GlyphInstance, DrawList, ImageFormat, BoxShadowClipMode, DisplayItem};
use types::{PipelineId, RenderNotifier, StackingContext, SpecificDisplayItem, ColorF};
use types::{RenderTargetID, MixBlendMode, CompositeDisplayItem, BorderSide, BorderStyle};
use types::{NodeIndex, CompositionOp, FilterOp, LowLevelFilterOp, BlurDirection};
use types::{ComplexClipRegion, BorderRadius};
use util;
use util::MatrixHelpers;
use scoped_threadpool;

type DisplayListMap = HashMap<DisplayListID, DisplayList, DefaultState<FnvHasher>>;
type DrawListMap = HashMap<DrawListID, DrawList, DefaultState<FnvHasher>>;
type FlatDrawListArray = Vec<FlatDrawList>;
type StackingContextMap = HashMap<PipelineId, RootStackingContext, DefaultState<FnvHasher>>;

static ID_COUNTER: AtomicUsize = ATOMIC_USIZE_INIT;

#[inline]
pub fn new_id() -> usize {
    ID_COUNTER.fetch_add(1, SeqCst)
}

pub static MAX_RECT: Rect<f32> = Rect {
    origin: Point2D {
        x: -1000.0,
        y: -1000.0,
    },
    size: Size2D {
        width: 10000.0,
        height: 10000.0,
    },
};

const BORDER_DASH_SIZE: f32 = 3.0;

impl BatchId {
    fn new() -> BatchId {
        BatchId(new_id())
    }
}

#[derive(Debug)]
struct RenderTarget {
    size: Size2D<u32>,
    draw_list_indices: Vec<DrawListIndex>,
    texture_id: Option<TextureId>,
}

impl RenderTarget {
    fn new(size: Size2D<u32>, texture_id: Option<TextureId>) -> RenderTarget {
        RenderTarget {
            size: size,
            draw_list_indices: Vec::new(),
            texture_id: texture_id,
        }
    }
}

struct DisplayItemIterator<'a> {
    flat_draw_lists: &'a FlatDrawListArray,
    current_key: DisplayItemKey,
    last_key: DisplayItemKey,
}

impl<'a> DisplayItemIterator<'a> {
    fn new(flat_draw_lists: &'a FlatDrawListArray,
           src_items: &Vec<DisplayItemKey>) -> DisplayItemIterator<'a> {

        match (src_items.first(), src_items.last()) {
            (Some(first), Some(last)) => {
                let current_key = first.clone();
                let mut last_key = last.clone();

                let DrawListItemIndex(last_item_index) = last_key.item_index;
                last_key.item_index = DrawListItemIndex(last_item_index + 1);

                DisplayItemIterator {
                    current_key: current_key,
                    last_key: last_key,
                    flat_draw_lists: flat_draw_lists,
                }
            }
            (None, None) => {
                DisplayItemIterator {
                    current_key: DisplayItemKey::new(0, 0),
                    last_key: DisplayItemKey::new(0, 0),
                    flat_draw_lists: flat_draw_lists
                }
            }
            _ => unreachable!(),
        }
    }
}

impl<'a> Iterator for DisplayItemIterator<'a> {
    type Item = DisplayItemKey;

    fn next(&mut self) -> Option<DisplayItemKey> {
        if self.current_key == self.last_key {
            return None;
        }

        let key = self.current_key.clone();
        let DrawListItemIndex(item_index) = key.item_index;
        let DrawListIndex(list_index) = key.draw_list_index;

        self.current_key.item_index = DrawListItemIndex(item_index + 1);

        if key.draw_list_index != self.last_key.draw_list_index {
            let last_item_index = DrawListItemIndex(self.flat_draw_lists[list_index as usize].draw_list.items.len() as u32);
            if self.current_key.item_index == last_item_index {
                self.current_key.draw_list_index = DrawListIndex(list_index + 1);
                self.current_key.item_index = DrawListItemIndex(0);
            }
        }

        Some(key)
    }
}

trait GetDisplayItemHelper {
    fn get_item(&self, key: &DisplayItemKey) -> &DisplayItem;
    fn get_item_and_draw_context(&self, key: &DisplayItemKey) -> (&DisplayItem, &DrawContext);
}

impl GetDisplayItemHelper for FlatDrawListArray {
    fn get_item(&self, key: &DisplayItemKey) -> &DisplayItem {
        let DrawListIndex(list_index) = key.draw_list_index;
        let DrawListItemIndex(item_index) = key.item_index;
        &self[list_index as usize].draw_list.items[item_index as usize]
    }

    fn get_item_and_draw_context(&self, key: &DisplayItemKey) -> (&DisplayItem, &DrawContext) {
        let DrawListIndex(list_index) = key.draw_list_index;
        let DrawListItemIndex(item_index) = key.item_index;
        let list = &self[list_index as usize];
        (&list.draw_list.items[item_index as usize], &list.draw_context)
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

#[derive(Clone)]
pub struct DrawContext {
    render_target_index: RenderTargetIndex,
    overflow: Rect<f32>,
    device_pixel_ratio: f32,
    final_transform: Matrix4,
    scroll_layer_id: ScrollLayerId,
}

pub struct FlatDrawList {
    pub id: Option<DrawListID>,
    pub draw_context: DrawContext,
    pub draw_list: DrawList,
}

struct StackingContextDrawLists {
    background_and_borders: Vec<DrawListID>,
    block_background_and_borders: Vec<DrawListID>,
    floats: Vec<DrawListID>,
    content: Vec<DrawListID>,
    positioned_content: Vec<DrawListID>,
    outlines: Vec<DrawListID>,
}

impl StackingContextDrawLists {
    fn new() -> StackingContextDrawLists {
        StackingContextDrawLists {
            background_and_borders: Vec::new(),
            block_background_and_borders: Vec::new(),
            floats: Vec::new(),
            content: Vec::new(),
            positioned_content: Vec::new(),
            outlines: Vec::new(),
        }
    }

    #[inline(always)]
    fn push_draw_list_id(id: Option<DrawListID>, list: &mut Vec<DrawListID>) {
        if let Some(id) = id {
            list.push(id);
        }
    }
}

trait CollectDrawListsForStackingContext {
    fn collect_draw_lists(&self, display_lists: &DisplayListMap) -> StackingContextDrawLists;
}

impl CollectDrawListsForStackingContext for StackingContext {
    fn collect_draw_lists(&self, display_lists: &DisplayListMap) -> StackingContextDrawLists {
        let mut result = StackingContextDrawLists::new();

        for display_list_id in &self.display_lists {
            let display_list = &display_lists[display_list_id];
            match display_list.mode {
                DisplayListMode::Default => {
                    StackingContextDrawLists::push_draw_list_id(display_list.background_and_borders_id,
                                                                &mut result.background_and_borders);
                    StackingContextDrawLists::push_draw_list_id(display_list.block_backgrounds_and_borders_id,
                                                                &mut result.block_background_and_borders);
                    StackingContextDrawLists::push_draw_list_id(display_list.floats_id,
                                                                &mut result.floats);
                    StackingContextDrawLists::push_draw_list_id(display_list.content_id,
                                                                &mut result.content);
                    StackingContextDrawLists::push_draw_list_id(display_list.positioned_content_id,
                                                                &mut result.positioned_content);
                    StackingContextDrawLists::push_draw_list_id(display_list.outlines_id,
                                                                &mut result.outlines);
                }
                DisplayListMode::PseudoFloat => {
                    StackingContextDrawLists::push_draw_list_id(display_list.background_and_borders_id,
                                                                &mut result.floats);
                    StackingContextDrawLists::push_draw_list_id(display_list.block_backgrounds_and_borders_id,
                                                                &mut result.floats);
                    StackingContextDrawLists::push_draw_list_id(display_list.floats_id,
                                                                &mut result.floats);
                    StackingContextDrawLists::push_draw_list_id(display_list.content_id,
                                                                &mut result.floats);
                    StackingContextDrawLists::push_draw_list_id(display_list.positioned_content_id,
                                                                &mut result.floats);
                    StackingContextDrawLists::push_draw_list_id(display_list.outlines_id,
                                                                &mut result.floats);
                }
                DisplayListMode::PseudoPositionedContent => {
                    StackingContextDrawLists::push_draw_list_id(display_list.background_and_borders_id,
                                                                &mut result.positioned_content);
                    StackingContextDrawLists::push_draw_list_id(display_list.block_backgrounds_and_borders_id,
                                                                &mut result.positioned_content);
                    StackingContextDrawLists::push_draw_list_id(display_list.floats_id,
                                                                &mut result.positioned_content);
                    StackingContextDrawLists::push_draw_list_id(display_list.content_id,
                                                                &mut result.positioned_content);
                    StackingContextDrawLists::push_draw_list_id(display_list.positioned_content_id,
                                                                &mut result.positioned_content);
                    StackingContextDrawLists::push_draw_list_id(display_list.outlines_id,
                                                                &mut result.positioned_content);
                }
            }
        }

        result
    }
}

struct Scene {
    // Internal state
    layers: HashMap<ScrollLayerId, Layer, DefaultState<FnvHasher>>,

    // Source data
    flat_draw_lists: Vec<FlatDrawList>,

    // Outputs
    pipeline_epoch_map: HashMap<PipelineId, Epoch, DefaultState<FnvHasher>>,
    render_targets: Vec<RenderTarget>,
    render_target_stack: Vec<RenderTargetIndex>,
    pending_updates: BatchUpdateList,

    // Constants
    quad_program_id: ProgramId,
}

impl Scene {
    fn new(quad_program_id: ProgramId) -> Scene {
        Scene {
            layers: HashMap::with_hash_state(Default::default()),

            flat_draw_lists: Vec::new(),

            pipeline_epoch_map: HashMap::with_hash_state(Default::default()),
            render_targets: Vec::new(),
            render_target_stack: Vec::new(),
            pending_updates: BatchUpdateList::new(),

            quad_program_id: quad_program_id,
        }
    }

    pub fn pending_updates(&mut self) -> BatchUpdateList {
        mem::replace(&mut self.pending_updates, BatchUpdateList::new())
    }

    fn reset(&mut self, resource_cache: &mut ResourceCache) {
        debug_assert!(self.render_target_stack.len() == 0);
        self.pipeline_epoch_map.clear();

        // Free any render targets from last frame.
        // TODO: This should really re-use existing targets here...
        for render_target in &mut self.render_targets {
            if let Some(texture_id) = render_target.texture_id {
                resource_cache.free_render_target(texture_id);
            }
        }

        self.render_targets.clear();
    }

    fn push_render_target(&mut self,
                          size: Size2D<u32>,
                          texture_id: Option<TextureId>) {
        let rt_index = RenderTargetIndex(self.render_targets.len() as u32);
        self.render_target_stack.push(rt_index);

        let render_target = RenderTarget::new(size, texture_id);
        self.render_targets.push(render_target);
    }

    fn current_render_target(&self) -> RenderTargetIndex {
        *self.render_target_stack.last().unwrap()
    }

    fn pop_render_target(&mut self) {
        self.render_target_stack.pop().unwrap();
    }

    fn push_draw_list(&mut self,
                      id: Option<DrawListID>,
                      draw_list: DrawList,
                      draw_context: &DrawContext) {
        let RenderTargetIndex(current_render_target) = *self.render_target_stack.last().unwrap();
        let render_target = &mut self.render_targets[current_render_target as usize];

        let draw_list_index = DrawListIndex(self.flat_draw_lists.len() as u32);
        render_target.draw_list_indices.push(draw_list_index);

        self.flat_draw_lists.push(FlatDrawList {
            id: id,
            draw_context: draw_context.clone(),
            draw_list: draw_list,
        });
    }

    fn add_draw_list(&mut self,
                     draw_list_id: DrawListID,
                     draw_context: &DrawContext,
                     draw_list_map: &mut DrawListMap,
                     iframes: &mut Vec<IframeInfo>) {
        let draw_list = draw_list_map.remove(&draw_list_id).expect(&format!("unable to remove draw list {:?}", draw_list_id));

        // TODO: DrawList should set a flag if iframes are added, to avoid this loop in the common case of no iframes.
        for item in &draw_list.items {
            match item.item {
                SpecificDisplayItem::Iframe(ref info) => {
                    let iframe_offset = draw_context.final_transform.transform_point(&item.rect.origin);
                    iframes.push(IframeInfo::new(info.iframe, iframe_offset, item.rect));
                }
                _ => {}
            }
        }

        self.push_draw_list(Some(draw_list_id),
                            draw_list,
                            draw_context);
    }

    fn flatten_stacking_context(&mut self,
                                stacking_context_kind: StackingContextKind,
                                parent_transform: &Matrix4,
                                parent_perspective: &Matrix4,
                                display_list_map: &DisplayListMap,
                                draw_list_map: &mut DrawListMap,
                                parent_scroll_layer: ScrollLayerId,
                                stacking_contexts: &StackingContextMap,
                                device_pixel_ratio: f32,
                                resource_cache: &mut ResourceCache,
                                clip_rect: &Rect<f32>) {
        let _pf = util::ProfileScope::new("  flatten_stacking_context");
        let stacking_context = match stacking_context_kind {
            StackingContextKind::Normal(stacking_context) => stacking_context,
            StackingContextKind::Root(root) => &root.stacking_context,
        };

        let mut iframes = Vec::new();

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
        let local_transform = Matrix4::identity().translate(origin.x, origin.y, 0.0)
                                                 .mul(&stacking_context.transform);

        let mut final_transform = parent_perspective.mul(&parent_transform)
                                                    .mul(&local_transform);

        // Build world space perspective transform
        let perspective_transform = Matrix4::identity().translate(origin.x, origin.y, 0.0)
                                                       .mul(&stacking_context.perspective)
                                                       .translate(-origin.x, -origin.y, 0.0);

        let overflow = stacking_context.overflow.intersection(&clip_rect);

        if let Some(overflow) = overflow {
            let mut draw_context = DrawContext {
                render_target_index: self.current_render_target(),
                overflow: overflow,
                device_pixel_ratio: device_pixel_ratio,
                final_transform: final_transform,
                scroll_layer_id: this_scroll_layer,
            };

            // When establishing a new 3D context, clear Z. This is only needed if there
            // are child stacking contexts, otherwise it is a redundant clear.
            if stacking_context.establishes_3d_context && stacking_context.children.len() > 0 {
                let mut clear_draw_list = DrawList::new();
                let clear_item = ClearDisplayItem {
                    clear_color: false,
                    clear_z: true,
                    clear_stencil: true,
                };
                let clip = ClipRegion {
                    main: stacking_context.overflow,
                    complex: vec![],
                };
                let display_item = DisplayItem {
                    item: SpecificDisplayItem::Clear(clear_item),
                    rect: stacking_context.overflow,
                    clip: clip,
                    node_index: None,
                };
                clear_draw_list.push(display_item);
                self.push_draw_list(None, clear_draw_list, &draw_context);
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
                let texture_id = resource_cache.allocate_render_target(TextureTarget::Texture2D,
                                                                       size.width,
                                                                       size.height,
                                                                       1,
                                                                       ImageFormat::RGBA8);
                let TextureId(render_target_id) = texture_id;

                let mut composite_draw_list = DrawList::new();
                let composite_item = CompositeDisplayItem {
                    operation: *composition_operation,
                    texture_id: RenderTargetID(render_target_id),
                };
                let clip = ClipRegion {
                    main: stacking_context.overflow,
                    complex: vec![],
                };
                let composite_item = DisplayItem {
                    item: SpecificDisplayItem::Composite(composite_item),
                    rect: stacking_context.overflow,
                    clip: clip,
                    node_index: None,
                };
                composite_draw_list.push(composite_item);
                self.push_draw_list(None, composite_draw_list, &draw_context);

                self.push_render_target(size, Some(texture_id));
                final_transform = Matrix4::identity();
                draw_context.final_transform = final_transform;
                draw_context.render_target_index = self.current_render_target();
            }

            match stacking_context_kind {
                StackingContextKind::Normal(..) => {}
                StackingContextKind::Root(root) => {
                    self.pipeline_epoch_map.insert(root.pipeline_id, root.epoch);

                    if root.background_color.a > 0.0 {
                        let mut root_draw_list = DrawList::new();
                        let rectangle_item = RectangleDisplayItem {
                            color: root.background_color.clone(),
                        };
                        let clip = ClipRegion {
                            main: stacking_context.overflow,
                            complex: vec![],
                        };
                        let root_bg_color_item = DisplayItem {
                            item: SpecificDisplayItem::Rectangle(rectangle_item),
                            rect: stacking_context.overflow,
                            clip: clip,
                            node_index: None,
                        };
                        root_draw_list.push(root_bg_color_item);

                        self.push_draw_list(None, root_draw_list, &draw_context);
                    }
                }
            }

            let draw_list_ids = stacking_context.collect_draw_lists(display_list_map);

            for id in &draw_list_ids.background_and_borders {
                self.add_draw_list(*id, &draw_context, draw_list_map, &mut iframes);
            }

            // TODO: Sort children (or store in two arrays) to avoid having
            //       to iterate this list twice.
            for child in &stacking_context.children {
                if child.z_index >= 0 {
                    continue;
                }
                self.flatten_stacking_context(StackingContextKind::Normal(child),
                                              &final_transform,
                                              &perspective_transform,
                                              display_list_map,
                                              draw_list_map,
                                              parent_scroll_layer,
                                              stacking_contexts,
                                              device_pixel_ratio,
                                              resource_cache,
                                              clip_rect);
            }

            for id in &draw_list_ids.block_background_and_borders {
                self.add_draw_list(*id, &draw_context, draw_list_map, &mut iframes);
            }

            for id in &draw_list_ids.floats {
                self.add_draw_list(*id, &draw_context, draw_list_map, &mut iframes);
            }

            for id in &draw_list_ids.content {
                self.add_draw_list(*id, &draw_context, draw_list_map, &mut iframes);
            }

            for id in &draw_list_ids.positioned_content {
                self.add_draw_list(*id, &draw_context, draw_list_map, &mut iframes);
            }

            for child in &stacking_context.children {
                if child.z_index < 0 {
                    continue;
                }
                self.flatten_stacking_context(StackingContextKind::Normal(child),
                                              &final_transform,
                                              &perspective_transform,
                                              display_list_map,
                                              draw_list_map,
                                              parent_scroll_layer,
                                              stacking_contexts,
                                              device_pixel_ratio,
                                              resource_cache,
                                              clip_rect);
            }

            // TODO: This ordering isn't quite right - it should look
            //       at the z-index in the iframe root stacking context.
            for iframe_info in &iframes {
                let iframe = stacking_contexts.get(&iframe_info.id);
                if let Some(iframe) = iframe {
                    // TODO: DOesn't handle transforms on iframes yet!
                    let iframe_transform = Matrix4::identity().translate(iframe_info.offset.x,
                                                                         iframe_info.offset.y,
                                                                         0.0);

                    let clip_rect = clip_rect.intersection(&iframe_info.clip_rect);

                    if let Some(clip_rect) = clip_rect {
                        let clip_rect = clip_rect.translate(&-iframe_info.offset);
                        self.flatten_stacking_context(StackingContextKind::Root(iframe),
                                                      &iframe_transform,
                                                      &perspective_transform,
                                                      display_list_map,
                                                      draw_list_map,
                                                      parent_scroll_layer,
                                                      stacking_contexts,
                                                      device_pixel_ratio,
                                                      resource_cache,
                                                      &clip_rect);
                    }
                }
            }

            for id in &draw_list_ids.outlines {
                self.add_draw_list(*id, &draw_context, draw_list_map, &mut iframes);
            }

            for _ in composition_operations.iter() {
                self.pop_render_target();
            }
        }
    }

    fn build_layers(&mut self, scene_rect: &Rect<f32>, old_draw_lists: &Vec<DrawList>) {
        let _pf = util::ProfileScope::new("  build_layers");

        let mut old_layers = mem::replace(&mut self.layers,
                                          HashMap::with_hash_state(Default::default()));

        // push all visible draw lists into aabb tree
        for (draw_list_index, flat_draw_list) in self.flat_draw_lists.iter_mut().enumerate() {
            let scroll_offset = match old_layers.get(&flat_draw_list.draw_context
                                                                    .scroll_layer_id) {
                Some(ref old_layer) => old_layer.scroll_offset,
                None => Point2D::zero(),
            };

            let layer = match self.layers.entry(flat_draw_list.draw_context.scroll_layer_id) {
                Occupied(entry) => {
                    entry.into_mut()
                }
                Vacant(entry) => {
                    entry.insert(Layer::new(scene_rect, &scroll_offset))
                }
            };

            for (item_index, item) in flat_draw_list.draw_list.items.iter_mut().enumerate() {
                // Node index may already be Some(..). This can occur when a page has iframes
                // and a new root stacking context is received. In this case, the node index
                // may already be set for draw lists from other iframe(s) that weren't updated
                // as part of this new stacking context.
                let rect = flat_draw_list.draw_context.final_transform.transform_rect(&item.rect);
                item.node_index = layer.insert(&rect, draw_list_index, item_index);
            }
        }

        for (scroll_layer_id, new_layer) in &mut self.layers {
            if let Some(ref mut old_layer) = old_layers.get_mut(&scroll_layer_id) {
                new_layer.reuse_compiled_data_from_old_layer_if_possible(old_layer,
                                                                         &mut self.flat_draw_lists,
                                                                         old_draw_lists)
            }
        }

        for (_, old_layer) in &mut old_layers {
            old_layer.reset(&mut self.pending_updates)
        }
    }

    fn build_frame(&mut self,
                   viewport: &Rect<i32>,
                   resource_cache: &mut ResourceCache,
                   thread_pool: &mut scoped_threadpool::Pool) -> Frame {
        let origin = Point2D::new(viewport.origin.x as f32, viewport.origin.y as f32);
        let size = Size2D::new(viewport.size.width as f32, viewport.size.height as f32);
        let viewport_rect = Rect::new(origin, size);

        // Traverse layer trees to calculate visible nodes
        for (_, layer) in &mut self.layers {
            layer.cull(&viewport_rect);
        }

        // Build resource list for newly visible nodes
        self.update_resource_lists(thread_pool);

        // Update texture cache and build list of raster jobs.
        self.update_texture_cache_and_build_raster_jobs(resource_cache);

        // Rasterize needed glyphs on worker threads
        self.raster_glyphs(thread_pool,
                           resource_cache);

        // Compile nodes that have become visible
        self.compile_visible_nodes(thread_pool,
                                   resource_cache);

        // Update the batch cache from newly compiled nodes
        self.update_batch_cache();

        // Collect the visible batches into a frame
        self.collect_and_sort_visible_batches()
    }

    fn collect_and_sort_visible_batches(&mut self) -> Frame {
        let mut frame = Frame::new(self.pipeline_epoch_map.clone());

        let mut render_layers = Vec::new();

        for render_target in &self.render_targets {
            render_layers.push(DrawLayer::new(render_target.texture_id,
                                              render_target.size,
                                              Vec::new()));
        }

        for (_, layer) in &self.layers {
            for node in &layer.aabb_tree.nodes {
                if node.is_visible {
                    debug_assert!(node.compiled_node.is_some());
                    let compiled_node = node.compiled_node.as_ref().unwrap();

                    // Update batch matrices
                    for (batch_id, matrix_map) in &compiled_node.matrix_maps {
                        // TODO: Could cache these matrices rather than generate for every batch.
                        let mut matrix_palette = vec![Matrix4::identity(); matrix_map.len()];

                        for (draw_list_index, matrix_index) in matrix_map {
                            let DrawListIndex(draw_list_index) = *draw_list_index;
                            let transform = self.flat_draw_lists[draw_list_index as usize].draw_context.final_transform;
                            let transform = transform.translate(layer.scroll_offset.x,
                                                                layer.scroll_offset.y,
                                                                0.0);
                            let matrix_index = *matrix_index as usize;
                            matrix_palette[matrix_index] = transform;
                        }

                        self.pending_updates.push(BatchUpdate {
                            id: *batch_id,
                            op: BatchUpdateOp::UpdateUniforms(matrix_palette),
                        });
                    }

                    for command in &compiled_node.commands {
                        let RenderTargetIndex(render_target) = command.render_target;
                        render_layers[render_target as usize].commands.push(command.clone());
                    }
                }
            }
        }

        for mut render_layer in render_layers {
            if render_layer.commands.len() > 0 {
                render_layer.commands.sort_by(|a, b| {
                    let draw_list_order = a.sort_key.draw_list_index.cmp(&b.sort_key.draw_list_index);
                    match draw_list_order {
                        Ordering::Equal => {
                            a.sort_key.item_index.cmp(&b.sort_key.item_index)
                        }
                        order => {
                            order
                        }
                    }
                });

                frame.add_layer(render_layer);
            }
        }

        frame
    }

    fn compile_visible_nodes(&mut self,
                             thread_pool: &mut scoped_threadpool::Pool,
                             resource_cache: &ResourceCache) {
        let _pf = util::ProfileScope::new("  compile_visible_nodes");

        // TODO(gw): This is a bit messy with layers - work out a cleaner interface
        // for detecting node overlaps...
        let mut node_info_map = HashMap::with_hash_state(Default::default());
        for (scroll_layer_id, layer) in &self.layers {
            node_info_map.insert(*scroll_layer_id, layer.aabb_tree.node_info());
        }

        let flat_draw_list_array = &self.flat_draw_lists;
        let layers = &mut self.layers;
        let node_info_map = &node_info_map;
        let quad_program_id = self.quad_program_id;

        thread_pool.scoped(|scope| {
            for (scroll_layer_id, layer) in layers {
                let nodes = &mut layer.aabb_tree.nodes;
                for node in nodes {
                    if node.is_visible && node.compiled_node.is_none() {
                        scope.execute(move || {
                            node.compile(flat_draw_list_array,
                                         resource_cache,
                                         node_info_map,
                                         quad_program_id,
                                         *scroll_layer_id);
                        });
                    }
                }
            }
        });
    }

    fn update_batch_cache(&mut self) {
        // Allocate and update VAOs
        for (_, layer) in &mut self.layers {
            for node in &mut layer.aabb_tree.nodes {
                if node.is_visible {
                    let compiled_node = node.compiled_node.as_mut().unwrap();
                    for batch in compiled_node.batches.drain(..) {
                        self.pending_updates.push(BatchUpdate {
                            id: batch.batch_id,
                            op: BatchUpdateOp::Create(batch.vertices,
                                                      batch.indices,
                                                      batch.program_id,
                                                      batch.color_texture_id,
                                                      batch.mask_texture_id),
                        });
                        compiled_node.batch_id_list.push(batch.batch_id);
                        compiled_node.matrix_maps.insert(batch.batch_id, batch.matrix_map);
                    }
                }
            }
        }
    }

    fn update_texture_cache_and_build_raster_jobs(&mut self, resource_cache: &mut ResourceCache) {
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

    fn raster_glyphs(&mut self,
                     thread_pool: &mut scoped_threadpool::Pool,
                     resource_cache: &mut ResourceCache) {
        let _pf = util::ProfileScope::new("  raster_glyphs");
        resource_cache.raster_pending_glyphs(thread_pool);
    }

    fn update_resource_lists(&mut self,
                             thread_pool: &mut scoped_threadpool::Pool) {
        let _pf = util::ProfileScope::new("  update_resource_lists");

        let flat_draw_lists = &self.flat_draw_lists;

        for (_, layer) in &mut self.layers {
            let nodes = &mut layer.aabb_tree.nodes;

            thread_pool.scoped(|scope| {
                for node in nodes {
                    if node.is_visible && node.compiled_node.is_none() {
                        scope.execute(move || {
                            node.build_resource_list(flat_draw_lists);
                        });
                    }
                }
            });
        }
    }

    fn scroll(&mut self, delta: &Point2D<f32>, viewport_size: &Size2D<f32>) {
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
}

struct DrawCommandBuilder {
    quad_program_id: ProgramId,
    render_target_index: RenderTargetIndex,
    current_batch: Option<RenderBatch>,
    draw_commands: Vec<DrawCommand>,
    batches: Vec<RenderBatch>,
}

impl DrawCommandBuilder {
    fn new(quad_program_id: ProgramId, render_target_index: RenderTargetIndex)
           -> DrawCommandBuilder {
        DrawCommandBuilder {
            render_target_index: render_target_index,
            quad_program_id: quad_program_id,
            current_batch: None,
            draw_commands: Vec::new(),
            batches: Vec::new(),
        }
    }

    fn flush_current_batch(&mut self) {
        // When a clear/composite is encountered - always flush any batches that are pending.
        // TODO: It may be possible to be smarter about this in the future and avoid
        // flushing the batches in some cases.
        if let Some(current_batch) = self.current_batch.take() {
            self.draw_commands.push(DrawCommand {
                render_target: self.render_target_index,
                sort_key: current_batch.sort_key.clone(),
                info: DrawCommandInfo::Batch(current_batch.batch_id),
            });
            self.batches.push(current_batch);
        }
    }

    fn add_clear(&mut self,
                 sort_key: &DisplayItemKey,
                 clear_color: bool,
                 clear_z: bool,
                 clear_stencil: bool) {
        self.flush_current_batch();

        let clear_info = ClearInfo {
            clear_color: clear_color,
            clear_z: clear_z,
            clear_stencil: clear_stencil,
        };
        let cmd = DrawCommand {
            render_target: self.render_target_index,
            sort_key: sort_key.clone(),
            info: DrawCommandInfo::Clear(clear_info),
        };
        self.draw_commands.push(cmd);
    }

    fn add_composite_item(&mut self,
                          operation: CompositionOp,
                          color_texture_id: TextureId,
                          rect: Rect<u32>,
                          sort_key: &DisplayItemKey) {
        self.flush_current_batch();

        let composite_info = CompositeInfo {
            operation: operation,
            rect: rect,
            color_texture_id: color_texture_id,
        };
        let cmd = DrawCommand {
            render_target: self.render_target_index,
            sort_key: sort_key.clone(),
            info: DrawCommandInfo::Composite(composite_info)
        };
        self.draw_commands.push(cmd);
    }

    fn add_draw_item(&mut self,
                     sort_key: &DisplayItemKey,
                     color_texture_id: TextureId,
                     mask_texture_id: TextureId,
                     primitive: Primitive,
                     vertices: &mut [PackedVertex]) {
        let program_id = self.quad_program_id;
        let need_new_batch = self.current_batch.is_none() ||
                             !self.current_batch.as_ref().unwrap().can_add_to_batch(color_texture_id,
                                                                                    mask_texture_id,
                                                                                    sort_key,
                                                                                    program_id);

        if need_new_batch {
            if let Some(current_batch) = self.current_batch.take() {
                self.draw_commands.push(DrawCommand {
                    render_target: self.render_target_index,
                    sort_key: current_batch.sort_key.clone(),
                    info: DrawCommandInfo::Batch(current_batch.batch_id),
                });
                self.batches.push(current_batch);
            }
            self.current_batch = Some(RenderBatch::new(BatchId::new(),
                                                       sort_key.clone(),
                                                       program_id,
                                                       color_texture_id,
                                                       mask_texture_id));
        }

        let batch = self.current_batch.as_mut().unwrap();
        batch.add_draw_item(color_texture_id,
                            mask_texture_id,
                            primitive,
                            vertices,
                            sort_key);
    }

    fn finalize(mut self) -> (Vec<RenderBatch>, Vec<DrawCommand>) {
        if let Some(current_batch) = self.current_batch.take() {
            self.draw_commands.push(DrawCommand {
                render_target: self.render_target_index,
                sort_key: current_batch.sort_key.clone(),
                info: DrawCommandInfo::Batch(current_batch.batch_id),
            });
            self.batches.push(current_batch);
        }

        (self.batches, self.draw_commands)
    }
}

#[derive(Debug)]
struct IframeInfo {
    offset: Point2D<f32>,
    clip_rect: Rect<f32>,
    id: PipelineId,
}

impl IframeInfo {
    fn new(id: PipelineId,
           offset: Point2D<f32>,
           clip_rect: Rect<f32>) -> IframeInfo {
        IframeInfo {
            offset: offset,
            id: id,
            clip_rect: clip_rect,
        }
    }
}

struct RootStackingContext {
    pipeline_id: PipelineId,
    epoch: Epoch,
    background_color: ColorF,
    stacking_context: StackingContext,
}

enum StackingContextKind<'a> {
    Normal(&'a StackingContext),
    Root(&'a RootStackingContext)
}

pub struct RenderBackend {
    thread_pool: scoped_threadpool::Pool,
    api_rx: Receiver<ApiMsg>,
    api_tx: Sender<ApiMsg>,
    result_tx: Sender<ResultMsg>,
    viewport: Rect<i32>,
    device_pixel_ratio: f32,
    root_pipeline_id: Option<PipelineId>,
    resource_cache: ResourceCache,
    display_list_map: DisplayListMap,
    draw_list_map: DrawListMap,
    stacking_contexts: StackingContextMap,
    next_draw_list_id: DrawListID,
    next_namespace_id: IdNamespace,

    scene: Scene,
}

impl RenderBackend {
    pub fn new(api_rx: Receiver<ApiMsg>,
               api_tx: Sender<ApiMsg>,
               result_tx: Sender<ResultMsg>,
               viewport: Rect<i32>,
               device_pixel_ratio: f32,
               quad_program_id: ProgramId,
               white_image_id: TextureCacheItemId,
               dummy_mask_image_id: TextureCacheItemId,
               texture_cache: TextureCache) -> RenderBackend {
        let mut thread_pool = scoped_threadpool::Pool::new(8);

        let resource_cache = ResourceCache::new(&mut thread_pool,
                                                texture_cache,
                                                white_image_id,
                                                dummy_mask_image_id,
                                                device_pixel_ratio);

        let backend = RenderBackend {
            thread_pool: thread_pool,
            api_rx: api_rx,
            api_tx: api_tx,
            result_tx: result_tx,
            viewport: viewport,
            device_pixel_ratio: device_pixel_ratio,
            root_pipeline_id: None,
            resource_cache: resource_cache,
            scene: Scene::new(quad_program_id),
            display_list_map: HashMap::with_hash_state(Default::default()),
            draw_list_map: HashMap::with_hash_state(Default::default()),
            stacking_contexts: HashMap::with_hash_state(Default::default()),

            next_namespace_id: IdNamespace(1),
            next_draw_list_id: DrawListID(0),
        };

        backend
    }

    fn remove_and_save_draw_list(&mut self,
                                 saved_old_draw_lists: &mut HashMap<DrawListID, DrawList>,
                                 draw_list_id: Option<DrawListID>) {
        if let Some(id) = draw_list_id {
            saved_old_draw_lists.insert(id, self.draw_list_map.remove(&id).unwrap());
        }
    }

    fn add_draw_list(&mut self, draw_list: DrawList) -> Option<DrawListID> {
        if draw_list.item_count() > 0 {
            let id = self.next_draw_list_id;

            let DrawListID(current_id) = id;
            self.next_draw_list_id = DrawListID(current_id+1);

            self.draw_list_map.insert(id, draw_list);
            Some(id)
        } else {
            None
        }
    }

    pub fn run(&mut self, notifier: Box<RenderNotifier>) {
        let mut notifier = notifier;

        loop {
            let msg = self.api_rx.recv();

            match msg {
                Ok(msg) => {
                    match msg {
                        ApiMsg::AddRawFont(id, bytes) => {
                            self.resource_cache
                                .add_font_template(id, FontTemplate::Raw(Arc::new(bytes)));
                        }
                        ApiMsg::AddNativeFont(id, native_font_handle) => {
                            self.resource_cache
                                .add_font_template(id, FontTemplate::Native(native_font_handle));
                        }
                        ApiMsg::AddImage(id, width, height, format, bytes) => {
                            self.resource_cache.add_image_template(id,
                                                                   width,
                                                                   height,
                                                                   format,
                                                                   bytes);
                        }
                        ApiMsg::UpdateImage(id, width, height, format, bytes) => {
                            self.resource_cache.update_image_template(id,
                                                                      width,
                                                                      height,
                                                                      format,
                                                                      bytes);
                        }
                        ApiMsg::AddDisplayList(id,
                                               pipeline_id,
                                               epoch,
                                               mut display_list_builder) => {
                            optimizer::optimize_display_list_builder(&mut display_list_builder);

                            let display_list = DisplayList {
                                mode: display_list_builder.mode,
                                pipeline_id: pipeline_id,
                                epoch: epoch,
                                background_and_borders_id: self.add_draw_list(display_list_builder.background_and_borders),
                                block_backgrounds_and_borders_id: self.add_draw_list(display_list_builder.block_backgrounds_and_borders),
                                floats_id: self.add_draw_list(display_list_builder.floats),
                                content_id: self.add_draw_list(display_list_builder.content),
                                positioned_content_id: self.add_draw_list(display_list_builder.positioned_content),
                                outlines_id: self.add_draw_list(display_list_builder.outlines),
                            };

                            self.display_list_map.insert(id, display_list);
                        }
                        ApiMsg::CloneApi(sender) => {
                            let new_api = RenderApi {
                                tx: self.api_tx.clone(),
                                id_namespace: self.next_namespace_id,
                                next_id: Cell::new(ResourceId(0)),
                            };

                            let IdNamespace(id_namespace) = self.next_namespace_id;
                            self.next_namespace_id = IdNamespace(id_namespace + 1);

                            sender.send(new_api).unwrap();
                        }
                        ApiMsg::SetRootStackingContext(stacking_context, background_color, epoch, pipeline_id) => {
                            let _pf = util::ProfileScope::new("SetRootStackingContext");

                            let old_draw_lists = mem::replace(&mut self.scene.flat_draw_lists,
                                                              Vec::new());
                            let old_display_list_keys: Vec<_> = self.display_list_map.iter()
                                                                    .filter(|&(_, ref v)| {
                                                                        v.pipeline_id == pipeline_id &&
                                                                        v.epoch < epoch
                                                                    })
                                                                    .map(|(k, _)| k.clone())
                                                                    .collect();

                            self.stacking_contexts.insert(pipeline_id, RootStackingContext {
                                pipeline_id: pipeline_id,
                                epoch: epoch,
                                background_color: background_color,
                                stacking_context: stacking_context,
                            });

                            self.build_scene(old_draw_lists, &old_display_list_keys[..]);
                            self.render(&mut *notifier);
                        }
                        ApiMsg::SetRootPipeline(pipeline_id) => {
                            let _pf = util::ProfileScope::new("SetRootPipeline");

                            let old_draw_lists = mem::replace(&mut self.scene.flat_draw_lists,
                                                              Vec::new());
                            self.root_pipeline_id = Some(pipeline_id);
                            self.build_scene(old_draw_lists, &[]);

                            self.render(&mut *notifier);
                        }
                        ApiMsg::Scroll(delta) => {
                            let _pf = util::ProfileScope::new("Scroll");

                            self.scroll(&delta);
                            self.render(&mut *notifier);
                        }
                        ApiMsg::TranslatePointToLayerSpace(point, tx) => {
                            // TODO(pcwalton): Select other layers for mouse events.
                            let point = point / self.device_pixel_ratio;
                            match self.scene.layers.get_mut(&ScrollLayerId(0)) {
                                None => tx.send(point).unwrap(),
                                Some(layer) => tx.send(point - layer.scroll_offset).unwrap(),
                            }
                        }
                    }
                }
                Err(..) => {
                    break;
                }
            }
        }
    }

    fn build_scene(&mut self,
                   old_draw_lists: Vec<FlatDrawList>,
                   old_display_list_keys: &[DisplayListID]) {
        // Return all current draw lists to the hash
        let mut old_draw_lists_or_ids = Vec::new();
        for flat_draw_list in old_draw_lists.into_iter() {
            match flat_draw_list.id {
                Some(id) => {
                    old_draw_lists_or_ids.push(DrawListOrId::Id(id));
                    self.draw_list_map.insert(id, flat_draw_list.draw_list);
                }
                None => {
                    old_draw_lists_or_ids.push(DrawListOrId::DrawList(flat_draw_list.draw_list));
                }
            }
        }

        // Remove any old draw lists and display lists for this pipeline
        let mut saved_old_draw_lists = HashMap::new();
        for key in old_display_list_keys {
            let display_list = self.display_list_map.remove(key).unwrap();
            self.remove_and_save_draw_list(&mut saved_old_draw_lists,
                                           display_list.background_and_borders_id);
            self.remove_and_save_draw_list(&mut saved_old_draw_lists,
                                           display_list.block_backgrounds_and_borders_id);
            self.remove_and_save_draw_list(&mut saved_old_draw_lists, display_list.floats_id);
            self.remove_and_save_draw_list(&mut saved_old_draw_lists, display_list.content_id);
            self.remove_and_save_draw_list(&mut saved_old_draw_lists,
                                           display_list.positioned_content_id);
            self.remove_and_save_draw_list(&mut saved_old_draw_lists, display_list.outlines_id);
        }

        // Recreate the old flat draw lists.
        let old_draw_lists: Vec<_> = old_draw_lists_or_ids.into_iter().flat_map(|draw_list_or_id| {
            match draw_list_or_id {
                DrawListOrId::DrawList(draw_list) => Some(draw_list),
                DrawListOrId::Id(id) => saved_old_draw_lists.remove(&id),
            }
        }).collect();

        // Flatten the stacking context hierarchy
        if let Some(root_pipeline_id) = self.root_pipeline_id {
            if let Some(root_sc) = self.stacking_contexts.get(&root_pipeline_id) {
                // Clear out any state and return draw lists (if needed)
                self.scene.reset(&mut self.resource_cache);

                let size = Size2D::new(self.viewport.size.width as u32,
                                       self.viewport.size.height as u32);
                let root_scroll_layer_id = root_sc.stacking_context
                                                  .scroll_layer_id
                                                  .expect("root layer must be a scroll layer!");

                self.scene.push_render_target(size, None);
                self.scene.flatten_stacking_context(StackingContextKind::Root(root_sc),
                                                    &Matrix4::identity(),
                                                    &Matrix4::identity(),
                                                    &self.display_list_map,
                                                    &mut self.draw_list_map,
                                                    root_scroll_layer_id,
                                                    &self.stacking_contexts,
                                                    self.device_pixel_ratio,
                                                    &mut self.resource_cache,
                                                    &root_sc.stacking_context.overflow);
                self.scene.pop_render_target();
            }
        }

        if let Some(root_pipeline_id) = self.root_pipeline_id {
            if let Some(root_sc) = self.stacking_contexts.get(&root_pipeline_id) {
                // Init the AABB culling tree(s)
                self.scene.build_layers(&root_sc.stacking_context.overflow, &old_draw_lists);

                // FIXME(pcwalton): This should be done somewhere else, probably when building the
                // layer.
                let root_scroll_layer_id = root_sc.stacking_context
                                                  .scroll_layer_id
                                                  .expect("root layer must be a scroll layer!");
                if let Some(root_scroll_layer) = self.scene.layers.get_mut(&root_scroll_layer_id) {
                    root_scroll_layer.scroll_boundaries = root_sc.stacking_context.overflow.size;
                }
            }
        }
    }

    fn render(&mut self, notifier: &mut RenderNotifier) {
        let mut frame = self.scene.build_frame(&self.viewport,
                                               &mut self.resource_cache,
                                               &mut self.thread_pool);

        // Bit of a hack - if there was nothing visible, at least
        // add one layer to the frame so that the screen gets
        // cleared to the default UA background color. Perhaps
        // there is a better way to handle this...
        if frame.layers.len() == 0 {
            frame.layers.push(DrawLayer {
                texture_id: None,
                size: Size2D::new(self.viewport.size.width as u32,
                                   self.viewport.size.height as u32),
                commands: Vec::new(),
            });
        }

        let pending_update = self.resource_cache.pending_updates();
        if pending_update.updates.len() > 0 {
            self.result_tx.send(ResultMsg::UpdateTextureCache(pending_update)).unwrap();
        }

        let pending_update = self.scene.pending_updates();
        if pending_update.updates.len() > 0 {
            self.result_tx.send(ResultMsg::UpdateBatches(pending_update)).unwrap();
        }

        self.result_tx.send(ResultMsg::NewFrame(frame)).unwrap();
        notifier.new_frame_ready();
    }

    fn scroll(&mut self, delta: &Point2D<f32>) {
        let viewport_size = Size2D::new(self.viewport.size.width as f32,
                                        self.viewport.size.height as f32);
        self.scene.scroll(delta, &viewport_size);
    }

}

impl DrawCommandBuilder {
    #[inline]
    fn add_textured_rectangle(&mut self,
                              sort_key: &DisplayItemKey,
                              rect: &Rect<f32>,
                              clip: &CombinedClipRegion,
                              image_info: &TextureCacheItem,
                              resource_cache: &ResourceCache,
                              clip_buffers: &mut ClipBuffers,
                              color: &ColorF) {
        self.add_axis_aligned_gradient_with_texture(sort_key,
                                                    rect,
                                                    clip,
                                                    image_info,
                                                    resource_cache,
                                                    clip_buffers,
                                                    &[*color, *color, *color, *color])
    }

    #[inline]
    fn add_color_rectangle(&mut self,
                           sort_key: &DisplayItemKey,
                           rect: &Rect<f32>,
                           clip: &CombinedClipRegion,
                           resource_cache: &ResourceCache,
                           clip_buffers: &mut ClipBuffers,
                           color: &ColorF) {
        self.add_axis_aligned_gradient(sort_key,
                                       rect,
                                       clip,
                                       resource_cache,
                                       clip_buffers,
                                       &[*color, *color, *color, *color])
    }

    fn add_composite(&mut self,
                     sort_key: &DisplayItemKey,
                     draw_context: &DrawContext,
                     rect: &Rect<f32>,
                     texture_id: RenderTargetID,
                     operation: CompositionOp) {
        let RenderTargetID(texture_id) = texture_id;

        let origin = draw_context.final_transform.transform_point(&rect.origin);
        let origin = Point2D::new(origin.x as u32, origin.y as u32);
        let size = Size2D::new(rect.size.width as u32, rect.size.height as u32);

        self.add_composite_item(operation,
                                TextureId(texture_id),
                                Rect::new(origin, size),
                                sort_key);
    }

    fn add_image(&mut self,
                 sort_key: &DisplayItemKey,
                 rect: &Rect<f32>,
                 clip: &CombinedClipRegion,
                 stretch_size: &Size2D<f32>,
                 image_key: ImageKey,
                 resource_cache: &ResourceCache,
                 clip_buffers: &mut ClipBuffers,
                 color: &ColorF) {
        // Should be caught higher up
        debug_assert!(stretch_size.width > 0.0 && stretch_size.height > 0.0);

        let image_info = resource_cache.get_image(image_key);

        if rect.size.width == stretch_size.width && rect.size.height == stretch_size.height {
            let uv = RectUv {
                top_left: Point2D::new(image_info.u0, image_info.v0),
                top_right: Point2D::new(image_info.u1, image_info.v0),
                bottom_left: Point2D::new(image_info.u0, image_info.v1),
                bottom_right: Point2D::new(image_info.u1, image_info.v1),
            };

            self.push_image_rect(color,
                                 image_info,
                                 clip,
                                 &sort_key,
                                 resource_cache,
                                 clip_buffers,
                                 rect,
                                 &uv);
            return
        }

        let (image_info, tiled_size) = match TiledImageKey::create_if_necessary(image_key,
                                                                                &rect.size,
                                                                                stretch_size) {
            Some(ref tiled_image_key) => {
                (resource_cache.get_tiled_image(tiled_image_key),
                 Size2D::new(tiled_image_key.tiled_width as f32,
                             tiled_image_key.tiled_height as f32))
            }
            None => (image_info, *stretch_size),
        };

        let uv = RectUv {
            top_left: Point2D::new(image_info.u0, image_info.v0),
            top_right: Point2D::new(image_info.u1, image_info.v0),
            bottom_left: Point2D::new(image_info.u0, image_info.v1),
            bottom_right: Point2D::new(image_info.u1, image_info.v1),
        };

        let mut y_offset = 0.0;
        while y_offset < rect.size.height {
            let mut x_offset = 0.0;
            while x_offset < rect.size.width {
                let origin = Point2D::new(rect.origin.x + x_offset, rect.origin.y + y_offset);
                let tiled_rect = Rect::new(origin, tiled_size.clone());

                self.push_image_rect(color,
                                     image_info,
                                     clip,
                                     &sort_key,
                                     resource_cache,
                                     clip_buffers,
                                     &tiled_rect,
                                     &uv);

                x_offset = x_offset + tiled_size.width;
            }

            y_offset = y_offset + tiled_size.height;
        }
    }

    fn push_image_rect(&mut self,
                       color: &ColorF,
                       image_info: &TextureCacheItem,
                       clip: &CombinedClipRegion,
                       sort_key: &DisplayItemKey,
                       resource_cache: &ResourceCache,
                       clip_buffers: &mut ClipBuffers,
                       rect: &Rect<f32>,
                       uv: &RectUv) {
        clipper::clip_rect_to_combined_region(RectPosUv {
                                                pos: *rect,
                                                uv: *uv
                                              },
                                              &mut clip_buffers.sh_clip_buffers,
                                              &mut clip_buffers.rect_pos_uv,
                                              clip);
        for clip_region in clip_buffers.rect_pos_uv.clip_rect_to_region_result_output.drain(..) {
            let mask = mask_for_clip_region(resource_cache, &clip_region, false);
            let colors = [*color, *color, *color, *color];
            let mut vertices = clip_region.make_packed_vertices_for_rect(&colors,
                                                                         mask,
                                                                         image_info.texture_index);

            self.add_draw_item(sort_key,
                               image_info.texture_id,
                               mask.texture_id,
                               Primitive::Rectangles,
                               &mut vertices);
        }
    }

    fn add_text(&mut self,
                sort_key: &DisplayItemKey,
                draw_context: &DrawContext,
                font_key: FontKey,
                size: Au,
                blur_radius: Au,
                color: &ColorF,
                glyphs: &Vec<GlyphInstance>,
                resource_cache: &ResourceCache) {
        let dummy_mask_image = resource_cache.get_dummy_mask_image();
        //let dummy_mask_image = texture_cache.get(resource_cache.dummy_mask_image_id);

        // Logic below to pick the primary render item depends on len > 0!
        assert!(glyphs.len() > 0);

        let device_pixel_ratio = draw_context.device_pixel_ratio;

        let mut glyph_key = GlyphKey::new(font_key, size, blur_radius, glyphs[0].index);

        let blur_offset = blur_radius.to_f32_px() * (BLUR_INFLATION_FACTOR as f32) / 2.0;

        let mut text_batches: HashMap<TextureId, Vec<PackedVertex>, DefaultState<FnvHasher>> =
            HashMap::with_hash_state(Default::default());

        for glyph in glyphs {
            glyph_key.index = glyph.index;
            let image_info = resource_cache.get_glyph(&glyph_key);

            if image_info.width > 0 && image_info.height > 0 {
                let x0 = glyph.x + image_info.user_x0 as f32 / device_pixel_ratio - blur_offset;
                let y0 = glyph.y - image_info.user_y0 as f32 / device_pixel_ratio - blur_offset;

                let x1 = x0 + image_info.width as f32 / device_pixel_ratio;
                let y1 = y0 + image_info.height as f32 / device_pixel_ratio;

                let vertex_buffer = match text_batches.entry(image_info.texture_id) {
                    Occupied(entry) => {
                        entry.into_mut()
                    }
                    Vacant(entry) => {
                        entry.insert(Vec::new())
                    }
                };
                vertex_buffer.push(PackedVertex::from_components(
                        x0, y0,
                        color,
                        image_info.u0, image_info.v0,
                        dummy_mask_image.u0, dummy_mask_image.v0,
                        image_info.texture_index,
                        dummy_mask_image.texture_index));
                vertex_buffer.push(PackedVertex::from_components(
                        x1, y0,
                        color,
                        image_info.u1, image_info.v0,
                        dummy_mask_image.u1, dummy_mask_image.v0,
                        image_info.texture_index,
                        dummy_mask_image.texture_index));
                vertex_buffer.push(PackedVertex::from_components(
                        x0, y1,
                        color,
                        image_info.u0, image_info.v1,
                        dummy_mask_image.u0, dummy_mask_image.v1,
                        image_info.texture_index,
                        dummy_mask_image.texture_index));
                vertex_buffer.push(PackedVertex::from_components(
                        x1, y1,
                        color,
                        image_info.u1, image_info.v1,
                        dummy_mask_image.v0, dummy_mask_image.v1,
                        image_info.texture_index,
                        dummy_mask_image.texture_index));
            }
        }

        for (color_texture_id, mut vertex_buffer) in text_batches {
            self.add_draw_item(sort_key,
                               color_texture_id,
                               dummy_mask_image.texture_id,
                               Primitive::Glyphs,
                               &mut vertex_buffer);
        }
    }

    // Colors are in the order: top left, top right, bottom right, bottom left.
    #[inline]
    fn add_axis_aligned_gradient(&mut self,
                                 sort_key: &DisplayItemKey,
                                 rect: &Rect<f32>,
                                 clip: &CombinedClipRegion,
                                 resource_cache: &ResourceCache,
                                 clip_buffers: &mut ClipBuffers,
                                 colors: &[ColorF; 4]) {
        let white_image = resource_cache.get_dummy_color_image();
        self.add_axis_aligned_gradient_with_texture(sort_key,
                                                    rect,
                                                    clip,
                                                    white_image,
                                                    resource_cache,
                                                    clip_buffers,
                                                    colors);
    }

    // Colors are in the order: top left, top right, bottom right, bottom left.
    fn add_axis_aligned_gradient_with_texture(&mut self,
                                              sort_key: &DisplayItemKey,
                                              rect: &Rect<f32>,
                                              clip: &CombinedClipRegion,
                                              image_info: &TextureCacheItem,
                                              resource_cache: &ResourceCache,
                                              clip_buffers: &mut ClipBuffers,
                                              colors: &[ColorF; 4]) {
        if rect.size.width == 0.0 || rect.size.height == 0.0 {
            return
        }

        let uv = RectUv {
            top_left: Point2D::new(image_info.u0, image_info.v0),
            top_right: Point2D::new(image_info.u1, image_info.v0),
            bottom_left: Point2D::new(image_info.u0, image_info.v1),
            bottom_right: Point2D::new(image_info.u1, image_info.v1),
        };

        clipper::clip_rect_to_combined_region(
            RectPosUv {
                pos: *rect,
                uv: uv,
            },
            &mut clip_buffers.sh_clip_buffers,
            &mut clip_buffers.rect_pos_uv,
            clip);
        for clip_region in clip_buffers.rect_pos_uv.clip_rect_to_region_result_output.drain(..) {
            let mask = mask_for_clip_region(resource_cache, &clip_region, false);

            let mut vertices = clip_region.make_packed_vertices_for_rect(colors,
                                                                         mask,
                                                                         image_info.texture_index);

            self.add_draw_item(sort_key,
                               image_info.texture_id,
                               mask.texture_id,
                               Primitive::Rectangles,
                               &mut vertices);
        }
    }

    fn add_gradient(&mut self,
                    sort_key: &DisplayItemKey,
                    clip: &CombinedClipRegion,
                    start_point: &Point2D<f32>,
                    end_point: &Point2D<f32>,
                    stops: &[GradientStop],
                    resource_cache: &ResourceCache,
                    clip_buffers: &mut ClipBuffers) {
        let white_image = resource_cache.get_dummy_color_image();

        debug_assert!(stops.len() >= 2);

        let dir_x = end_point.x - start_point.x;
        let dir_y = end_point.y - start_point.y;
        let dir_len = (dir_x * dir_x + dir_y * dir_y).sqrt();
        let dir_xn = dir_x / dir_len;
        let dir_yn = dir_y / dir_len;
        let perp_xn = -dir_yn;
        let perp_yn = dir_xn;

        for i in 0..stops.len()-1 {
            let stop0 = &stops[i];
            let stop1 = &stops[i+1];

            if stop0.offset == stop1.offset {
                continue;
            }

            let color0 = &stop0.color;
            let color1 = &stop1.color;

            let start_x = start_point.x + stop0.offset * (end_point.x - start_point.x);
            let start_y = start_point.y + stop0.offset * (end_point.y - start_point.y);

            let end_x = start_point.x + stop1.offset * (end_point.x - start_point.x);
            let end_y = start_point.y + stop1.offset * (end_point.y - start_point.y);

            let len_scale = 1000.0;     // todo: determine this properly!!

            let x0 = start_x - perp_xn * len_scale;
            let y0 = start_y - perp_yn * len_scale;

            let x1 = end_x - perp_xn * len_scale;
            let y1 = end_y - perp_yn * len_scale;

            let x2 = end_x + perp_xn * len_scale;
            let y2 = end_y + perp_yn * len_scale;

            let x3 = start_x + perp_xn * len_scale;
            let y3 = start_y + perp_yn * len_scale;

            let gradient_polygon = PolygonPosColorUv {
                vertices: vec![
                    WorkVertex::new(x0, y0, color0, 0.0, 0.0),
                    WorkVertex::new(x1, y1, color1, 0.0, 0.0),
                    WorkVertex::new(x2, y2, color1, 0.0, 0.0),
                    WorkVertex::new(x3, y3, color0, 0.0, 0.0),
                ],
            };

            { // scope for buffers
                clipper::clip_rect_to_combined_region(gradient_polygon,
                                                      &mut clip_buffers.sh_clip_buffers,
                                                      &mut clip_buffers.polygon_pos_color_uv,
                                                      clip);
                for clip_result in clip_buffers.polygon_pos_color_uv
                                               .clip_rect_to_region_result_output
                                               .drain(..) {
                    let mask = mask_for_clip_region(resource_cache, &clip_result, false);

                    let mut packed_vertices = Vec::new();
                    if clip_result.rect_result.vertices.len() >= 3 {
                        for vert in clip_result.rect_result.vertices.iter() {
                            packed_vertices.push(clip_result.make_packed_vertex(
                                    &vert.position(),
                                    &vert.uv(),
                                    &vert.color(),
                                    &mask,
                                    white_image.texture_index));
                        }
                    }

                    if packed_vertices.len() > 0 {
                        self.add_draw_item(sort_key,
                                           white_image.texture_id,
                                           mask.texture_id,
                                           Primitive::TriangleFan,
                                           &mut packed_vertices);
                    }
                }
            }
        }
    }

    fn add_box_shadow(&mut self,
                      sort_key: &DisplayItemKey,
                      box_bounds: &Rect<f32>,
                      clip: &CombinedClipRegion,
                      box_offset: &Point2D<f32>,
                      color: &ColorF,
                      blur_radius: f32,
                      spread_radius: f32,
                      border_radius: f32,
                      clip_mode: BoxShadowClipMode,
                      resource_cache: &ResourceCache,
                      clip_buffers: &mut ClipBuffers) {
        let rect = compute_box_shadow_rect(box_bounds, box_offset, spread_radius);

        // Fast path.
        if blur_radius == 0.0 && spread_radius == 0.0 && clip_mode == BoxShadowClipMode::None {
            self.add_color_rectangle(sort_key, &rect, clip, resource_cache, clip_buffers, color);
            return;
        }

        // Draw the corners.
        self.add_box_shadow_corners(sort_key,
                                    box_bounds,
                                    box_offset,
                                    color,
                                    blur_radius,
                                    spread_radius,
                                    border_radius,
                                    clip_mode,
                                    clip,
                                    resource_cache,
                                    clip_buffers);

        // Draw the sides.
        self.add_box_shadow_sides(sort_key,
                                  box_bounds,
                                  clip,
                                  box_offset,
                                  color,
                                  blur_radius,
                                  spread_radius,
                                  border_radius,
                                  clip_mode,
                                  resource_cache,
                                  clip_buffers);

        match clip_mode {
            BoxShadowClipMode::None => {
                // Fill the center area.
                self.add_color_rectangle(sort_key,
                                         box_bounds,
                                         clip,
                                         resource_cache,
                                         clip_buffers,
                                         color);
            }
            BoxShadowClipMode::Outset => {
                // Fill the center area.
                let metrics = BoxShadowMetrics::new(&rect, border_radius, blur_radius);
                let center_rect = Rect::new(metrics.tl_inner,
                                            Size2D::new(metrics.br_inner.x - metrics.tl_inner.x,
                                                        metrics.br_inner.y - metrics.tl_inner.y));
                let mut clip = *clip;
                clip.clip_out(&ComplexClipRegion::from_rect(&box_bounds));
                self.add_color_rectangle(sort_key,
                                         &center_rect,
                                         &clip,
                                         resource_cache,
                                         clip_buffers,
                                         color);
            }
            BoxShadowClipMode::Inset => {
                // Fill in the outsides.
                self.fill_outside_area_of_inset_box_shadow(sort_key,
                                                           box_bounds,
                                                           clip,
                                                           box_offset,
                                                           color,
                                                           blur_radius,
                                                           spread_radius,
                                                           border_radius,
                                                           resource_cache,
                                                           clip_buffers);
            }
        }
    }

    fn add_box_shadow_corners(&mut self,
                              sort_key: &DisplayItemKey,
                              box_bounds: &Rect<f32>,
                              box_offset: &Point2D<f32>,
                              color: &ColorF,
                              blur_radius: f32,
                              spread_radius: f32,
                              border_radius: f32,
                              clip_mode: BoxShadowClipMode,
                              clip: &CombinedClipRegion,
                              resource_cache: &ResourceCache,
                              clip_buffers: &mut ClipBuffers) {
        // Draw the corners.
        //
        //      +--+------------------+--+
        //      |##|                  |##|
        //      +--+------------------+--+
        //      |  |                  |  |
        //      |  |                  |  |
        //      |  |                  |  |
        //      +--+------------------+--+
        //      |##|                  |##|
        //      +--+------------------+--+

        let rect = compute_box_shadow_rect(box_bounds, box_offset, spread_radius);
        let metrics = BoxShadowMetrics::new(&rect, border_radius, blur_radius);

        let clip = self.adjust_clip_for_box_shadow_clip_mode(clip,
                                                             box_bounds,
                                                             border_radius,
                                                             clip_mode);

        self.add_box_shadow_corner(sort_key,
                                   &metrics.tl_outer,
                                   &metrics.tl_inner,
                                   &color,
                                   blur_radius,
                                   border_radius,
                                   clip_mode,
                                   &clip,
                                   resource_cache,
                                   clip_buffers,
                                   BasicRotationAngle::Upright);
        self.add_box_shadow_corner(sort_key,
                                   &Point2D::new(metrics.tr_inner.x, metrics.tr_outer.y),
                                   &Point2D::new(metrics.tr_outer.x, metrics.tr_inner.y),
                                   &color,
                                   blur_radius,
                                   border_radius,
                                   clip_mode,
                                   &clip,
                                   resource_cache,
                                   clip_buffers,
                                   BasicRotationAngle::Clockwise90);
        self.add_box_shadow_corner(sort_key,
                                   &metrics.br_inner,
                                   &metrics.br_outer,
                                   &color,
                                   blur_radius,
                                   border_radius,
                                   clip_mode,
                                   &clip,
                                   resource_cache,
                                   clip_buffers,
                                   BasicRotationAngle::Clockwise180);
        self.add_box_shadow_corner(sort_key,
                                   &Point2D::new(metrics.bl_outer.x, metrics.bl_inner.y),
                                   &Point2D::new(metrics.bl_inner.x, metrics.bl_outer.y),
                                   &color,
                                   blur_radius,
                                   border_radius,
                                   clip_mode,
                                   &clip,
                                   resource_cache,
                                   clip_buffers,
                                   BasicRotationAngle::Clockwise270);
    }

    fn add_box_shadow_sides(&mut self,
                            sort_key: &DisplayItemKey,
                            box_bounds: &Rect<f32>,
                            clip: &CombinedClipRegion,
                            box_offset: &Point2D<f32>,
                            color: &ColorF,
                            blur_radius: f32,
                            spread_radius: f32,
                            border_radius: f32,
                            clip_mode: BoxShadowClipMode,
                            resource_cache: &ResourceCache,
                            clip_buffers: &mut ClipBuffers) {
        let rect = compute_box_shadow_rect(box_bounds, box_offset, spread_radius);
        let metrics = BoxShadowMetrics::new(&rect, border_radius, blur_radius);

        let clip = self.adjust_clip_for_box_shadow_clip_mode(clip,
                                                             box_bounds,
                                                             border_radius,
                                                             clip_mode);

        // Draw the sides.
        //
        //      +--+------------------+--+
        //      |  |##################|  |
        //      +--+------------------+--+
        //      |##|                  |##|
        //      |##|                  |##|
        //      |##|                  |##|
        //      +--+------------------+--+
        //      |  |##################|  |
        //      +--+------------------+--+

        let horizontal_size = Size2D::new(metrics.br_inner.x - metrics.tl_inner.x,
                                          metrics.edge_size);
        let vertical_size = Size2D::new(metrics.edge_size,
                                        metrics.br_inner.y - metrics.tl_inner.y);
        let top_rect = Rect::new(metrics.tl_outer + Point2D::new(metrics.edge_size, 0.0),
                                 horizontal_size);
        let right_rect =
            Rect::new(metrics.tr_outer + Point2D::new(-metrics.edge_size, metrics.edge_size),
                      vertical_size);
        let bottom_rect =
            Rect::new(metrics.bl_outer + Point2D::new(metrics.edge_size, -metrics.edge_size),
                      horizontal_size);
        let left_rect = Rect::new(metrics.tl_outer + Point2D::new(0.0, metrics.edge_size),
                                  vertical_size);

        self.add_box_shadow_edge(sort_key,
                                 &top_rect.origin,
                                 &top_rect.bottom_right(),
                                 color,
                                 blur_radius,
                                 border_radius,
                                 clip_mode,
                                 &clip,
                                 resource_cache,
                                 clip_buffers,
                                 BasicRotationAngle::Clockwise270);
        self.add_box_shadow_edge(sort_key,
                                 &right_rect.origin,
                                 &right_rect.bottom_right(),
                                 color,
                                 blur_radius,
                                 border_radius,
                                 clip_mode,
                                 &clip,
                                 resource_cache,
                                 clip_buffers,
                                 BasicRotationAngle::Upright);
        self.add_box_shadow_edge(sort_key,
                                 &bottom_rect.origin,
                                 &bottom_rect.bottom_right(),
                                 color,
                                 blur_radius,
                                 border_radius,
                                 clip_mode,
                                 &clip,
                                 resource_cache,
                                 clip_buffers,
                                 BasicRotationAngle::Clockwise90);
        self.add_box_shadow_edge(sort_key,
                                 &left_rect.origin,
                                 &left_rect.bottom_right(),
                                 color,
                                 blur_radius,
                                 border_radius,
                                 clip_mode,
                                 &clip,
                                 resource_cache,
                                 clip_buffers,
                                 BasicRotationAngle::Clockwise180);
    }

    fn fill_outside_area_of_inset_box_shadow(&mut self,
                                             sort_key: &DisplayItemKey,
                                             box_bounds: &Rect<f32>,
                                             clip: &CombinedClipRegion,
                                             box_offset: &Point2D<f32>,
                                             color: &ColorF,
                                             blur_radius: f32,
                                             spread_radius: f32,
                                             border_radius: f32,
                                             resource_cache: &ResourceCache,
                                             clip_buffers: &mut ClipBuffers) {
        let rect = compute_box_shadow_rect(box_bounds, box_offset, spread_radius);
        let metrics = BoxShadowMetrics::new(&rect, border_radius, blur_radius);

        let clip = self.adjust_clip_for_box_shadow_clip_mode(clip,
                                                             box_bounds,
                                                             border_radius,
                                                             BoxShadowClipMode::Inset);

        // Fill in the outside area of the box.
        //
        //            +------------------------------+
        //      A --> |##############################|
        //            +--+--+------------------+--+--+
        //            |##|  |                  |  |##|
        //            |##+--+------------------+--+##|
        //            |##|  |                  |  |##|
        //      D --> |##|  |                  |  |##| <-- B
        //            |##|  |                  |  |##|
        //            |##+--+------------------+--+##|
        //            |##|  |                  |  |##|
        //            +--+--+------------------+--+--+
        //      C --> |##############################|
        //            +------------------------------+

        // A:
        self.add_color_rectangle(sort_key,
                                 &Rect::new(box_bounds.origin,
                                            Size2D::new(box_bounds.size.width,
                                                        metrics.tl_outer.y - box_bounds.origin.y)),
                                 &clip,
                                 resource_cache,
                                 clip_buffers,
                                 color);

        // B:
        self.add_color_rectangle(sort_key,
                                 &Rect::new(metrics.tr_outer,
                                            Size2D::new(box_bounds.max_x() - metrics.tr_outer.x,
                                                        metrics.br_outer.y - metrics.tr_outer.y)),
                                 &clip,
                                 resource_cache,
                                 clip_buffers,
                                 color);

        // C:
        self.add_color_rectangle(sort_key,
                                 &Rect::new(Point2D::new(box_bounds.origin.x, metrics.bl_outer.y),
                                            Size2D::new(box_bounds.size.width,
                                                        box_bounds.max_y() - metrics.br_outer.y)),
                                 &clip,
                                 resource_cache,
                                 clip_buffers,
                                 color);

        // D:
        self.add_color_rectangle(sort_key,
                                 &Rect::new(Point2D::new(box_bounds.origin.x, metrics.tl_outer.y),
                                            Size2D::new(metrics.tl_outer.x - box_bounds.origin.x,
                                                        metrics.bl_outer.y - metrics.tl_outer.y)),
                                 &clip,
                                 resource_cache,
                                 clip_buffers,
                                 color);
    }

    fn adjust_clip_for_box_shadow_clip_mode<'a>(&mut self,
                                                clip: &CombinedClipRegion<'a>,
                                                box_bounds: &Rect<f32>,
                                                border_radius: f32,
                                                clip_mode: BoxShadowClipMode)
                                                -> CombinedClipRegion<'a> {
        let mut clip = *clip;
        match clip_mode {
            BoxShadowClipMode::None => {}
            BoxShadowClipMode::Inset => {
                clip.clip_in(&ComplexClipRegion {
                    rect: *box_bounds,
                    radii: BorderRadius::uniform(border_radius),
                });
            }
            BoxShadowClipMode::Outset => {
                // TODO(pcwalton): Support border radii here too, once we have inverted rounded
                // rect clip masks.
                clip.clip_out(&ComplexClipRegion::from_rect(box_bounds));
            }
        }
        clip
    }

    #[inline]
    fn add_border_edge(&mut self,
                       sort_key: &DisplayItemKey,
                       rect: &Rect<f32>,
                       clip: &CombinedClipRegion,
                       direction: BorderEdgeDirection,
                       color: &ColorF,
                       border_style: BorderStyle,
                       resource_cache: &ResourceCache,
                       clip_buffers: &mut clipper::ClipBuffers) {
        // TODO: Check for zero width/height borders!
        if color.a <= 0.0 {
            return
        }

        match border_style {
            BorderStyle::Dashed => {
                let (extent, step) = match direction {
                    BorderEdgeDirection::Horizontal => {
                        (rect.size.width, rect.size.height * BORDER_DASH_SIZE)
                    }
                    BorderEdgeDirection::Vertical => {
                        (rect.size.height, rect.size.width * BORDER_DASH_SIZE)
                    }
                };
                let mut origin = 0.0;
                while origin < extent {
                    let dash_rect = match direction {
                        BorderEdgeDirection::Horizontal => {
                            Rect::new(Point2D::new(rect.origin.x + origin, rect.origin.y),
                                      Size2D::new(f32::min(step, extent - origin),
                                                  rect.size.height))
                        }
                        BorderEdgeDirection::Vertical => {
                            Rect::new(Point2D::new(rect.origin.x, rect.origin.y + origin),
                                      Size2D::new(rect.size.width,
                                                  f32::min(step, extent - origin)))
                        }
                    };

                    self.add_color_rectangle(sort_key,
                                             &dash_rect,
                                             clip,
                                             resource_cache,
                                             clip_buffers,
                                             color);

                    origin += step + step;
                }
            }
            BorderStyle::Dotted => {
                let (extent, step) = match direction {
                    BorderEdgeDirection::Horizontal => (rect.size.width, rect.size.height),
                    BorderEdgeDirection::Vertical => (rect.size.height, rect.size.width),
                };
                let mut origin = 0.0;
                while origin < extent {
                    let (dot_rect, mask_radius) = match direction {
                        BorderEdgeDirection::Horizontal => {
                            (Rect::new(Point2D::new(rect.origin.x + origin, rect.origin.y),
                                       Size2D::new(f32::min(step, extent - origin),
                                                   rect.size.height)),
                             rect.size.height / 2.0)
                        }
                        BorderEdgeDirection::Vertical => {
                            (Rect::new(Point2D::new(rect.origin.x, rect.origin.y + origin),
                                       Size2D::new(rect.size.width,
                                                   f32::min(step, extent - origin))),
                             rect.size.width / 2.0)
                        }
                    };

                    let raster_op =
                        BorderRadiusRasterOp::create(&Size2D::new(mask_radius, mask_radius),
                                                     &Size2D::new(0.0, 0.0),
                                                     false,
                                                     ImageFormat::RGBA8).expect(
                        "Didn't find border radius mask for dashed border!");
                    let raster_item = RasterItem::BorderRadius(raster_op);
                    let color_image = resource_cache.get_raster(&raster_item);

                    // Top left:
                    self.add_textured_rectangle(sort_key,
                                                &Rect::new(dot_rect.origin,
                                                           Size2D::new(dot_rect.size.width / 2.0,
                                                                       dot_rect.size.height / 2.0)),
                                                clip,
                                                color_image,
                                                resource_cache,
                                                clip_buffers,
                                                color);

                    // Top right:
                    self.add_textured_rectangle(sort_key,
                                                &Rect::new(dot_rect.top_right(),
                                                           Size2D::new(-dot_rect.size.width / 2.0,
                                                                       dot_rect.size.height / 2.0)),
                                                clip,
                                                color_image,
                                                resource_cache,
                                                clip_buffers,
                                                color);

                    // Bottom right:
                    self.add_textured_rectangle(sort_key,
                                                &Rect::new(dot_rect.bottom_right(),
                                                            Size2D::new(-dot_rect.size.width / 2.0,
                                                                        -dot_rect.size.height / 2.0)),
                                                clip,
                                                color_image,
                                                resource_cache,
                                                clip_buffers,
                                                color);

                    // Bottom left:
                    self.add_textured_rectangle(sort_key,
                                                &Rect::new(dot_rect.bottom_left(),
                                                           Size2D::new(dot_rect.size.width / 2.0,
                                                                       -dot_rect.size.height / 2.0)),
                                                clip,
                                                color_image,
                                                resource_cache,
                                                clip_buffers,
                                                color);

                    origin += step + step;
                }
            }
            BorderStyle::Double => {
                let (outer_rect, inner_rect) = match direction {
                    BorderEdgeDirection::Horizontal => {
                        (Rect::new(rect.origin,
                                   Size2D::new(rect.size.width, rect.size.height / 3.0)),
                         Rect::new(Point2D::new(rect.origin.x,
                                                rect.origin.y + rect.size.height * 2.0 / 3.0),
                                   Size2D::new(rect.size.width, rect.size.height / 3.0)))
                    }
                    BorderEdgeDirection::Vertical => {
                        (Rect::new(rect.origin,
                                   Size2D::new(rect.size.width / 3.0, rect.size.height)),
                         Rect::new(Point2D::new(rect.origin.x + rect.size.width * 2.0 / 3.0,
                                                rect.origin.y),
                                   Size2D::new(rect.size.width / 3.0, rect.size.height)))
                    }
                };
                self.add_color_rectangle(sort_key,
                                         &outer_rect,
                                         clip,
                                         resource_cache,
                                         clip_buffers,
                                         color);
                self.add_color_rectangle(sort_key,
                                         &inner_rect,
                                         clip,
                                         resource_cache,
                                         clip_buffers,
                                         color);
            }
            _ => {
                self.add_color_rectangle(sort_key,
                                         rect,
                                         clip,
                                         resource_cache,
                                         clip_buffers,
                                         color);
            }
        }
    }

    #[inline]
    fn add_border_corner(&mut self,
                         sort_key: &DisplayItemKey,
                         clip: &CombinedClipRegion,
                         vertices_rect: &Rect<f32>,
                         color0: &ColorF,
                         color1: &ColorF,
                         outer_radius: &Size2D<f32>,
                         inner_radius: &Size2D<f32>,
                         resource_cache: &ResourceCache,
                         clip_buffers: &mut clipper::ClipBuffers,
                         rotation_angle: BasicRotationAngle) {
        if color0.a <= 0.0 && color1.a <= 0.0 {
            return
        }

        // TODO: Check for zero width/height borders!
        let white_image = resource_cache.get_dummy_color_image();
        let mask_image = match BorderRadiusRasterOp::create(outer_radius,
                                                            inner_radius,
                                                            false,
                                                            ImageFormat::A8) {
            Some(raster_item) => {
                let raster_item = RasterItem::BorderRadius(raster_item);
                resource_cache.get_raster(&raster_item)
            }
            None => {
                resource_cache.get_dummy_mask_image()
            }
        };

        // FIXME(pcwalton): Either use RGBA8 textures instead of alpha masks here, or implement
        // a mask combiner.
        let mask_uv = RectUv::from_image_and_rotation_angle(mask_image, rotation_angle);
        clipper::clip_rect_to_combined_region(
            RectPosUv {
                pos: *vertices_rect,
                uv: mask_uv,
            },
            &mut clip_buffers.sh_clip_buffers,
            &mut clip_buffers.rect_pos_uv,
            clip);
        for clip_region in clip_buffers.rect_pos_uv.clip_rect_to_region_result_output.drain(..) {
            let v0;
            let v1;
            let muv0;
            let muv1;
            match rotation_angle {
                BasicRotationAngle::Upright => {
                    v0 = clip_region.rect_result.pos.origin;
                    muv0 = clip_region.rect_result.uv.top_left;
                    v1 = clip_region.rect_result.pos.bottom_right();
                    muv1 = clip_region.rect_result.uv.bottom_right;
                }
                BasicRotationAngle::Clockwise90 => {
                    v0 = clip_region.rect_result.pos.top_right();
                    muv0 = clip_region.rect_result.uv.top_right;
                    v1 = clip_region.rect_result.pos.bottom_left();
                    muv1 = clip_region.rect_result.uv.bottom_left;
                }
                BasicRotationAngle::Clockwise180 => {
                    v0 = clip_region.rect_result.pos.bottom_right();
                    muv0 = clip_region.rect_result.uv.bottom_right;
                    v1 = clip_region.rect_result.pos.origin;
                    muv1 = clip_region.rect_result.uv.top_left;
                }
                BasicRotationAngle::Clockwise270 => {
                    v0 = clip_region.rect_result.pos.bottom_left();
                    muv0 = clip_region.rect_result.uv.bottom_left;
                    v1 = clip_region.rect_result.pos.top_right();
                    muv1 = clip_region.rect_result.uv.top_right;
                }
            }

            let mut vertices = [
                PackedVertex::from_components(v0.x, v0.y,
                                              color0,
                                              0.0, 0.0,
                                              muv0.x, muv0.y,
                                              white_image.texture_index,
                                              mask_image.texture_index),
                PackedVertex::from_components(v1.x, v1.y,
                                              color0,
                                              0.0, 0.0,
                                              muv1.x, muv1.y,
                                              white_image.texture_index, mask_image.texture_index),
                PackedVertex::from_components(v0.x, v1.y,
                                              color0,
                                              0.0, 0.0,
                                              muv0.x, muv1.y,
                                              white_image.texture_index, mask_image.texture_index),
                PackedVertex::from_components(v0.x, v0.y,
                                              color1,
                                              0.0, 0.0,
                                              muv0.x, muv0.y,
                                              white_image.texture_index, mask_image.texture_index),
                PackedVertex::from_components(v1.x, v0.y,
                                              color1,
                                              0.0, 0.0,
                                              muv1.x, muv0.y,
                                              white_image.texture_index, mask_image.texture_index),
                PackedVertex::from_components(v1.x, v1.y,
                                              color1,
                                              0.0, 0.0,
                                              muv1.x, muv1.y,
                                              white_image.texture_index, mask_image.texture_index),
            ];

            self.add_draw_item(sort_key,
                               white_image.texture_id,
                               mask_image.texture_id,
                               Primitive::Triangles,
                               &mut vertices);
        }
    }

    fn add_color_image_rectangle(&mut self,
                                 sort_key: &DisplayItemKey,
                                 v0: &Point2D<f32>,
                                 v1: &Point2D<f32>,
                                 clip: &CombinedClipRegion,
                                 color0: &ColorF,
                                 color1: &ColorF,
                                 color_image: &TextureCacheItem,
                                 resource_cache: &ResourceCache,
                                 clip_buffers: &mut ClipBuffers,
                                 rotation_angle: BasicRotationAngle) {
        if color0.a <= 0.0 || color1.a <= 0.0 {
            return
        }

        let vertices_rect = Rect::new(*v0, Size2D::new(v1.x - v0.x, v1.y - v0.y));
        let color_uv = RectUv::from_image_and_rotation_angle(color_image, rotation_angle);

        clipper::clip_rect_to_combined_region(RectPosUv {
                                                pos: vertices_rect,
                                                uv: color_uv,
                                              },
                                              &mut clip_buffers.sh_clip_buffers,
                                              &mut clip_buffers.rect_pos_uv,
                                              clip);
        for clip_region in clip_buffers.rect_pos_uv.clip_rect_to_region_result_output.drain(..) {
            let mask = mask_for_clip_region(resource_cache,
                                            &clip_region,
                                            false);
            let colors = [*color0, *color0, *color1, *color1];
            let mut vertices = clip_region.make_packed_vertices_for_rect(&colors,
                                                                         mask,
                                                                         color_image.texture_index);

            self.add_draw_item(sort_key,
                               color_image.texture_id,
                               mask.texture_id,
                               Primitive::Rectangles,
                               &mut vertices);
        }
    }

    fn add_border(&mut self,
                  sort_key: &DisplayItemKey,
                  rect: &Rect<f32>,
                  clip: &CombinedClipRegion,
                  info: &BorderDisplayItem,
                  resource_cache: &ResourceCache,
                  clip_buffers: &mut ClipBuffers) {
        // TODO: If any border segment is alpha, place all in alpha pass.
        //       Is it ever worth batching at a per-segment level?
        let radius = &info.radius;
        let left = &info.left;
        let right = &info.right;
        let top = &info.top;
        let bottom = &info.bottom;

        let tl_outer = Point2D::new(rect.origin.x, rect.origin.y);
        let tl_inner = tl_outer + Point2D::new(radius.top_left.width.max(left.width),
                                               radius.top_left.height.max(top.width));

        let tr_outer = Point2D::new(rect.origin.x + rect.size.width, rect.origin.y);
        let tr_inner = tr_outer + Point2D::new(-radius.top_right.width.max(right.width),
                                               radius.top_right.height.max(top.width));

        let bl_outer = Point2D::new(rect.origin.x, rect.origin.y + rect.size.height);
        let bl_inner = bl_outer + Point2D::new(radius.bottom_left.width.max(left.width),
                                               -radius.bottom_left.height.max(bottom.width));

        let br_outer = Point2D::new(rect.origin.x + rect.size.width,
                                    rect.origin.y + rect.size.height);
        let br_inner = br_outer - Point2D::new(radius.bottom_right.width.max(right.width),
                                               radius.bottom_right.height.max(bottom.width));

        let left_color = left.border_color(1.0, 2.0/3.0, 0.3, 0.7);
        let top_color = top.border_color(1.0, 2.0/3.0, 0.3, 0.7);
        let right_color = right.border_color(2.0/3.0, 1.0, 0.7, 0.3);
        let bottom_color = bottom.border_color(2.0/3.0, 1.0, 0.7, 0.3);

        // Edges
        self.add_border_edge(sort_key,
                             &Rect::new(Point2D::new(tl_outer.x, tl_inner.y),
                                        Size2D::new(left.width, bl_inner.y - tl_inner.y)),
                             clip,
                             BorderEdgeDirection::Vertical,
                             &left_color,
                             info.left.style,
                             resource_cache,
                             clip_buffers);

        self.add_border_edge(sort_key,
                             &Rect::new(Point2D::new(tl_inner.x, tl_outer.y),
                                        Size2D::new(tr_inner.x - tl_inner.x,
                                                    tr_outer.y + top.width - tl_outer.y)),
                             clip,
                             BorderEdgeDirection::Horizontal,
                             &top_color,
                             info.top.style,
                             resource_cache,
                             clip_buffers);

        self.add_border_edge(sort_key,
                             &Rect::new(Point2D::new(br_outer.x - right.width, tr_inner.y),
                                        Size2D::new(right.width, br_inner.y - tr_inner.y)),
                             clip,
                             BorderEdgeDirection::Vertical,
                             &right_color,
                             info.right.style,
                             resource_cache,
                             clip_buffers);

        self.add_border_edge(sort_key,
                             &Rect::new(Point2D::new(bl_inner.x, bl_outer.y - bottom.width),
                                        Size2D::new(br_inner.x - bl_inner.x,
                                                    br_outer.y - bl_outer.y + bottom.width)),
                             clip,
                             BorderEdgeDirection::Horizontal,
                             &bottom_color,
                             info.bottom.style,
                             resource_cache,
                             clip_buffers);

        // Corners
        self.add_border_corner(sort_key,
                               clip,
                               &Rect::new(tl_outer,
                                          Size2D::new(tl_inner.x - tl_outer.x,
                                                      tl_inner.y - tl_outer.y)),
                               &left_color,
                               &top_color,
                               &radius.top_left,
                               &info.top_left_inner_radius(),
                               resource_cache,
                               clip_buffers,
                               BasicRotationAngle::Upright);

        self.add_border_corner(sort_key,
                               clip,
                               &Rect::new(Point2D::new(tr_inner.x, tr_outer.y),
                                          Size2D::new(tr_outer.x - tr_inner.x,
                                                      tr_inner.y - tr_outer.y)),
                               &right_color,
                               &top_color,
                               &radius.top_right,
                               &info.top_right_inner_radius(),
                               resource_cache,
                               clip_buffers,
                               BasicRotationAngle::Clockwise90);

        self.add_border_corner(sort_key,
                               clip,
                               &Rect::new(br_inner,
                                          Size2D::new(br_outer.x - br_inner.x,
                                                      br_outer.y - br_inner.y)),
                               &right_color,
                               &bottom_color,
                               &radius.bottom_right,
                               &info.bottom_right_inner_radius(),
                               resource_cache,
                               clip_buffers,
                               BasicRotationAngle::Clockwise180);

        self.add_border_corner(sort_key,
                               clip,
                               &Rect::new(Point2D::new(bl_outer.x, bl_inner.y),
                                          Size2D::new(bl_inner.x - bl_outer.x,
                                                      bl_outer.y - bl_inner.y)),
                               &left_color,
                               &bottom_color,
                               &radius.bottom_left,
                               &info.bottom_left_inner_radius(),
                               resource_cache,
                               clip_buffers,
                               BasicRotationAngle::Clockwise270);
    }

    // FIXME(pcwalton): Assumes rectangles are well-formed with origin in TL
    fn add_box_shadow_corner(&mut self,
                             sort_key: &DisplayItemKey,
                             top_left: &Point2D<f32>,
                             bottom_right: &Point2D<f32>,
                             color: &ColorF,
                             blur_radius: f32,
                             border_radius: f32,
                             clip_mode: BoxShadowClipMode,
                             clip: &CombinedClipRegion,
                             resource_cache: &ResourceCache,
                             clip_buffers: &mut ClipBuffers,
                             rotation_angle: BasicRotationAngle) {
        let inverted = match clip_mode {
            BoxShadowClipMode::Outset | BoxShadowClipMode::None => false,
            BoxShadowClipMode::Inset => true,
        };

        let color_image = match BoxShadowRasterOp::create_corner(blur_radius,
                                                                 border_radius,
                                                                 inverted) {
            Some(raster_item) => {
                let raster_item = RasterItem::BoxShadow(raster_item);
                resource_cache.get_raster(&raster_item)
            }
            None => resource_cache.get_dummy_color_image(),
        };

        self.add_color_image_rectangle(sort_key,
                                       top_left,
                                       bottom_right,
                                       clip,
                                       color,
                                       color,
                                       &color_image,
                                       resource_cache,
                                       clip_buffers,
                                       rotation_angle)
    }

    fn add_box_shadow_edge(&mut self,
                           sort_key: &DisplayItemKey,
                           top_left: &Point2D<f32>,
                           bottom_right: &Point2D<f32>,
                           color: &ColorF,
                           blur_radius: f32,
                           border_radius: f32,
                           clip_mode: BoxShadowClipMode,
                           clip: &CombinedClipRegion,
                           resource_cache: &ResourceCache,
                           clip_buffers: &mut ClipBuffers,
                           rotation_angle: BasicRotationAngle) {
        let inverted = match clip_mode {
            BoxShadowClipMode::Outset | BoxShadowClipMode::None => false,
            BoxShadowClipMode::Inset => true,
        };

        let color_image = match BoxShadowRasterOp::create_edge(blur_radius,
                                                               border_radius,
                                                               inverted) {
            Some(raster_item) => {
                let raster_item = RasterItem::BoxShadow(raster_item);
                resource_cache.get_raster(&raster_item)
            }
            None => resource_cache.get_dummy_color_image(),
        };

        self.add_color_image_rectangle(sort_key,
                                       top_left,
                                       bottom_right,
                                       clip,
                                       color,
                                       color,
                                       &color_image,
                                       resource_cache,
                                       clip_buffers,
                                       rotation_angle)
    }
}

trait BuildRequiredResources {
    fn build_resource_list(&mut self, flat_draw_lists: &FlatDrawListArray);
}

impl BuildRequiredResources for AABBTreeNode {
    fn build_resource_list(&mut self, flat_draw_lists: &FlatDrawListArray) {
        //let _pf = util::ProfileScope::new("  build_resource_list");
        let mut resource_list = ResourceList::new();

        for item_key in &self.src_items {
            let display_item = flat_draw_lists.get_item(item_key);

            // Handle border radius for complex clipping regions.
            for complex_clip_region in display_item.clip.complex.iter() {
                resource_list.add_radius_raster_for_border_radii(&complex_clip_region.radii);
            }

            match display_item.item {
                SpecificDisplayItem::Image(ref info) => {
                    resource_list.add_image(info.image_key,
                                            &display_item.rect.size,
                                            &info.stretch_size);
                }
                SpecificDisplayItem::Text(ref info) => {
                    for glyph in &info.glyphs {
                        let glyph = Glyph::new(info.size, info.blur_radius, glyph.index);
                        resource_list.add_glyph(info.font_key, glyph);
                    }
                }
                SpecificDisplayItem::Rectangle(..) => {}
                SpecificDisplayItem::Iframe(..) => {}
                SpecificDisplayItem::Gradient(..) => {}
                SpecificDisplayItem::Composite(..) => {}
                SpecificDisplayItem::Clear(..) => {}
                SpecificDisplayItem::BoxShadow(ref info) => {
                    resource_list.add_radius_raster_for_border_radii(
                        &BorderRadius::uniform(info.border_radius));
                    resource_list.add_box_shadow_corner(info.blur_radius,
                                                        info.border_radius,
                                                        false);
                    resource_list.add_box_shadow_edge(info.blur_radius, info.border_radius, false);
                    if info.clip_mode == BoxShadowClipMode::Inset {
                        resource_list.add_box_shadow_corner(info.blur_radius,
                                                            info.border_radius,
                                                            true);
                        resource_list.add_box_shadow_edge(info.blur_radius,
                                                          info.border_radius,
                                                          true);
                    }
                }
                SpecificDisplayItem::Border(ref info) => {
                    resource_list.add_radius_raster(&info.radius.top_left,
                                                    &info.top_left_inner_radius(),
                                                    false,
                                                    ImageFormat::A8);
                    resource_list.add_radius_raster(&info.radius.top_right,
                                                    &info.top_right_inner_radius(),
                                                    false,
                                                    ImageFormat::A8);
                    resource_list.add_radius_raster(&info.radius.bottom_left,
                                                    &info.bottom_left_inner_radius(),
                                                    false,
                                                    ImageFormat::A8);
                    resource_list.add_radius_raster(&info.radius.bottom_right,
                                                    &info.bottom_right_inner_radius(),
                                                    false,
                                                    ImageFormat::A8);

                    if info.top.style == BorderStyle::Dotted {
                        resource_list.add_radius_raster(&Size2D::new(info.top.width / 2.0,
                                                                     info.top.width / 2.0),
                                                        &Size2D::new(0.0, 0.0),
                                                        false,
                                                        ImageFormat::RGBA8);
                    }
                    if info.right.style == BorderStyle::Dotted {
                        resource_list.add_radius_raster(&Size2D::new(info.right.width / 2.0,
                                                                     info.right.width / 2.0),
                                                        &Size2D::new(0.0, 0.0),
                                                        false,
                                                        ImageFormat::RGBA8);
                    }
                    if info.bottom.style == BorderStyle::Dotted {
                        resource_list.add_radius_raster(&Size2D::new(info.bottom.width / 2.0,
                                                                     info.bottom.width / 2.0),
                                                        &Size2D::new(0.0, 0.0),
                                                        false,
                                                        ImageFormat::RGBA8);
                    }
                    if info.left.style == BorderStyle::Dotted {
                        resource_list.add_radius_raster(&Size2D::new(info.left.width / 2.0,
                                                                     info.left.width / 2.0),
                                                        &Size2D::new(0.0, 0.0),
                                                        false,
                                                        ImageFormat::RGBA8);
                    }
                }
            }
        }

        self.resource_list = Some(resource_list);
    }
}

trait BorderSideHelpers {
    fn border_color(&self,
                    scale_factor_0: f32,
                    scale_factor_1: f32,
                    black_color_0: f32,
                    black_color_1: f32) -> ColorF;
}

impl BorderSideHelpers for BorderSide {
    fn border_color(&self,
                    scale_factor_0: f32,
                    scale_factor_1: f32,
                    black_color_0: f32,
                    black_color_1: f32) -> ColorF {
        match self.style {
            BorderStyle::Inset => {
                if self.color.r != 0.0 || self.color.g != 0.0 || self.color.b != 0.0 {
                    self.color.scale_rgb(scale_factor_1)
                } else {
                    ColorF::new(black_color_1, black_color_1, black_color_1, self.color.a)
                }
            }
            BorderStyle::Outset => {
                if self.color.r != 0.0 || self.color.g != 0.0 || self.color.b != 0.0 {
                    self.color.scale_rgb(scale_factor_0)
                } else {
                    ColorF::new(black_color_0, black_color_0, black_color_0, self.color.a)
                }
            }
            _ => self.color,
        }
    }
}

fn mask_for_border_radius<'a>(resource_cache: &'a ResourceCache,
                              border_radius: f32,
                              inverted: bool)
                              -> &'a TextureCacheItem {
    if border_radius == 0.0 {
        return resource_cache.get_dummy_mask_image()
    }

    let border_radius = Au::from_f32_px(border_radius);
    resource_cache.get_raster(&RasterItem::BorderRadius(BorderRadiusRasterOp {
        outer_radius_x: border_radius,
        outer_radius_y: border_radius,
        inner_radius_x: Au(0),
        inner_radius_y: Au(0),
        inverted: inverted,
        image_format: ImageFormat::A8,
    }))
}

fn mask_for_clip_region<'a,P>(resource_cache: &'a ResourceCache,
                              clip_region: &ClipRectToRegionResult<P>,
                              inverted: bool)
                              -> &'a TextureCacheItem {
    match clip_region.mask_result {
        None => {
            resource_cache.get_dummy_mask_image()
        }
        Some(ref mask_result) => {
            mask_for_border_radius(resource_cache,
                                   mask_result.border_radius,
                                   inverted)
        }
    }
}

trait NodeCompiler {
    fn compile(&mut self,
               flat_draw_lists: &FlatDrawListArray,
               resource_cache: &ResourceCache,
               node_info_map: &HashMap<ScrollLayerId,
                                       Vec<AABBTreeNodeInfo>,
                                       DefaultState<FnvHasher>>,
               quad_program_id: ProgramId,
               node_scroll_layer_id: ScrollLayerId);
}

impl NodeCompiler for AABBTreeNode {
    fn compile(&mut self,
               flat_draw_lists: &FlatDrawListArray,
               resource_cache: &ResourceCache,
               node_info_map: &HashMap<ScrollLayerId,
                                       Vec<AABBTreeNodeInfo>,
                                       DefaultState<FnvHasher>>,
               quad_program_id: ProgramId,
               node_scroll_layer_id: ScrollLayerId) {
        let color_white = ColorF::new(1.0, 1.0, 1.0, 1.0);
        let mut compiled_node = CompiledNode::new();

        let mut draw_cmd_builders = HashMap::new();
        let mut clip_buffers = ClipBuffers::new();

        let iter = DisplayItemIterator::new(flat_draw_lists, &self.src_items);
        for key in iter {
            let (display_item, draw_context) = flat_draw_lists.get_item_and_draw_context(&key);

            if let Some(item_node_index) = display_item.node_index {
                if item_node_index == self.node_index &&
                   node_scroll_layer_id == draw_context.scroll_layer_id {
                    let clip_rect = display_item.clip.main.intersection(&draw_context.overflow);

                    if let Some(ref clip_rect) = clip_rect {
                        let mut clip = CombinedClipRegion::from_clip_in_rect_and_stack(
                            clip_rect,
                            &display_item.clip.complex[..]);

                        let builder = match draw_cmd_builders.entry(draw_context.render_target_index) {
                            Vacant(entry) => {
                                entry.insert(DrawCommandBuilder::new(
                                    quad_program_id,
                                    draw_context.render_target_index))
                            }
                            Occupied(entry) => entry.into_mut(),
                        };

                        match display_item.item {
                            SpecificDisplayItem::Image(ref info) => {
                                builder.add_image(&key,
                                                  &display_item.rect,
                                                  &clip,
                                                  &info.stretch_size,
                                                  info.image_key,
                                                  resource_cache,
                                                  &mut clip_buffers,
                                                  &color_white);
                            }
                            SpecificDisplayItem::Text(ref info) => {
                                builder.add_text(&key,
                                                 draw_context,
                                                 info.font_key,
                                                 info.size,
                                                 info.blur_radius,
                                                 &info.color,
                                                 &info.glyphs,
                                                 resource_cache);
                            }
                            SpecificDisplayItem::Rectangle(ref info) => {
                                builder.add_color_rectangle(&key,
                                                            &display_item.rect,
                                                            &clip,
                                                            resource_cache,
                                                            &mut clip_buffers,
                                                            &info.color);
                            }
                            SpecificDisplayItem::Iframe(..) => {}
                            SpecificDisplayItem::Gradient(ref info) => {
                                clip.clip_in_rect(&display_item.rect);
                                builder.add_gradient(&key,
                                                     &clip,
                                                     &info.start_point,
                                                     &info.end_point,
                                                     &info.stops,
                                                     resource_cache,
                                                     &mut clip_buffers);
                            }
                            SpecificDisplayItem::BoxShadow(ref info) => {
                                builder.add_box_shadow(&key,
                                                       &info.box_bounds,
                                                       &clip,
                                                       &info.offset,
                                                       &info.color,
                                                       info.blur_radius,
                                                       info.spread_radius,
                                                       info.border_radius,
                                                       info.clip_mode,
                                                       resource_cache,
                                                       &mut clip_buffers);
                            }
                            SpecificDisplayItem::Border(ref info) => {
                                builder.add_border(&key,
                                                   &display_item.rect,
                                                   &clip,
                                                   info,
                                                   resource_cache,
                                                   &mut clip_buffers);
                            }
                            SpecificDisplayItem::Composite(ref info) => {
                                builder.add_composite(&key,
                                                      draw_context,
                                                      &display_item.rect,
                                                      info.texture_id,
                                                      info.operation);
                            }
                            SpecificDisplayItem::Clear(ref info) => {
                                builder.add_clear(&key,
                                                  info.clear_color,
                                                  info.clear_z,
                                                  info.clear_stencil);
                            }
                        }
                    }
                } else {
                    // TODO: Cache this information!!!
                    let NodeIndex(node_index_for_item) = item_node_index;
                    let NodeIndex(node_index_for_node) = self.node_index;

                    let info_list_for_item = node_info_map.get(&draw_context.scroll_layer_id).unwrap();
                    let info_list_for_node = node_info_map.get(&node_scroll_layer_id).unwrap();

                    let info_for_item = &info_list_for_item[node_index_for_item as usize];
                    let info_for_node = &info_list_for_node[node_index_for_node as usize];

                    // This node should be visible, else it shouldn't be getting compiled!
                    debug_assert!(info_for_node.is_visible);

                    if info_for_item.is_visible {
                        let rect_for_info = &info_for_item.rect;
                        let rect_for_node = &info_for_node.rect;

                        let nodes_overlap = rect_for_node.intersects(rect_for_info);
                        if nodes_overlap {
                            if let Some(builder) = draw_cmd_builders.remove(&draw_context.render_target_index) {
                                let (batches, commands) = builder.finalize();
                                compiled_node.batches.extend(batches.into_iter());
                                compiled_node.commands.extend(commands.into_iter());
                            }
                        }
                    }
                }
            }
        }

        for (_, builder) in draw_cmd_builders.into_iter() {
            let (batches, commands) = builder.finalize();
            compiled_node.batches.extend(batches.into_iter());
            compiled_node.commands.extend(commands.into_iter());
        }

        self.compiled_node = Some(compiled_node);
    }
}

#[derive(Debug)]
struct BoxShadowMetrics {
    edge_size: f32,
    tl_outer: Point2D<f32>,
    tl_inner: Point2D<f32>,
    tr_outer: Point2D<f32>,
    tr_inner: Point2D<f32>,
    bl_outer: Point2D<f32>,
    bl_inner: Point2D<f32>,
    br_outer: Point2D<f32>,
    br_inner: Point2D<f32>,
}

impl BoxShadowMetrics {
    fn new(box_bounds: &Rect<f32>, border_radius: f32, blur_radius: f32) -> BoxShadowMetrics {
        let outside_edge_size = 3.0 * blur_radius;
        let inside_edge_size = outside_edge_size.max(border_radius);
        let edge_size = outside_edge_size + inside_edge_size;
        let inner_rect = box_bounds.inflate(-inside_edge_size, -inside_edge_size);
        let outer_rect = box_bounds.inflate(outside_edge_size, outside_edge_size);

        BoxShadowMetrics {
            edge_size: edge_size,
            tl_outer: outer_rect.origin,
            tl_inner: inner_rect.origin,
            tr_outer: outer_rect.top_right(),
            tr_inner: inner_rect.top_right(),
            bl_outer: outer_rect.bottom_left(),
            bl_inner: inner_rect.bottom_left(),
            br_outer: outer_rect.bottom_right(),
            br_inner: inner_rect.bottom_right(),
        }
    }
}

fn compute_box_shadow_rect(box_bounds: &Rect<f32>,
                           box_offset: &Point2D<f32>,
                           spread_radius: f32)
                           -> Rect<f32> {
    let mut rect = (*box_bounds).clone();
    rect.origin.x += box_offset.x;
    rect.origin.y += box_offset.y;
    rect.inflate(spread_radius, spread_radius)
}

enum DrawListOrId {
    DrawList(DrawList),
    Id(DrawListID),
}

