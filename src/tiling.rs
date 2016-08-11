/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::{Au};
use batch_builder::{BorderSideHelpers, BoxShadowMetrics};
use device::{TextureId};
use euclid::{Point2D, Rect, Matrix4D, Size2D, Point4D};
use fnv::FnvHasher;
use frame::FrameId;
use internal_types::{Glyph, GlyphKey, DevicePixel, CompositionOp};
use internal_types::{ANGLE_FLOAT_TO_FIXED, LowLevelFilterOp, RectUv};
use layer::Layer;
use renderer::{BLUR_INFLATION_FACTOR};
use resource_cache::ResourceCache;
use resource_list::ResourceList;
use std::cmp;
use std::collections::{HashMap};
use std::f32;
use std::mem;
use std::hash::{BuildHasherDefault};
use texture_cache::{TexturePage};
use util::{self, rect_from_points, rect_from_points_f, MatrixHelpers, subtract_rect};
use webrender_traits::{ColorF, FontKey, ImageKey, ImageRendering, ComplexClipRegion};
use webrender_traits::{BorderDisplayItem, BorderStyle, ItemRange, AuxiliaryLists, BorderRadius, BorderSide};
use webrender_traits::{BoxShadowClipMode, PipelineId, ScrollLayerId, WebGLContextId};

#[repr(u32)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum GradientType {
    Horizontal,
    Vertical,
    Rotated,
}

#[derive(Debug, Copy, Clone)]
struct TaskIndex(usize);

struct AlphaBatchTask {
    items: Vec<AlphaRenderItem>,
    target_rect: Rect<DevicePixel>,
    actual_rect: Rect<DevicePixel>,
    child_rects: Vec<Rect<DevicePixel>>,
}

pub struct AlphaBatcher {
    pub layer_ubos: Vec<Vec<PackedLayer>>,
    pub tile_ubos: Vec<Vec<PackedTile>>,
    pub batches: Vec<PrimitiveBatch>,
    layer_to_ubo_map: Vec<Option<usize>>,
    tile_to_ubo_map: Vec<Option<usize>>,
    tasks: Vec<AlphaBatchTask>,
}

impl AlphaBatcher {
    fn new() -> AlphaBatcher {
        AlphaBatcher {
            layer_ubos: Vec::new(),
            tile_ubos: Vec::new(),
            batches: Vec::new(),
            layer_to_ubo_map: Vec::new(),
            tile_to_ubo_map: Vec::new(),
            tasks: Vec::new(),
        }
    }

    fn add_tile_to_ubo(tile_ubos: &mut Vec<Vec<PackedTile>>,
                       tile_to_ubo_map: &mut Vec<Option<usize>>,
                       task_index: TaskIndex,
                       task: &AlphaBatchTask,
                       ctx: &RenderTargetContext) -> (usize, u32) {
        let index_in_ubo = match tile_to_ubo_map[task_index.0] {
            Some(index_in_ubo) => {
                index_in_ubo
            }
            None => {
                let need_new_ubo = tile_ubos.is_empty() ||
                                   tile_ubos.last().unwrap().len() == ctx.alpha_batch_max_tiles;

                if need_new_ubo {
                    for i in 0..tile_to_ubo_map.len() {
                        tile_to_ubo_map[i] = None;
                    }
                    tile_ubos.push(Vec::new());
                }

                let tile_ubo = tile_ubos.last_mut().unwrap();
                let index = tile_ubo.len();
                tile_ubo.push(PackedTile {
                    actual_rect: task.actual_rect,
                    target_rect: task.target_rect,
                });
                tile_to_ubo_map[task_index.0] = Some(index);
                index
            }
        };

        (tile_ubos.len() - 1, index_in_ubo as u32)
    }

    fn add_layer_to_ubo(layer_ubos: &mut Vec<Vec<PackedLayer>>,
                        layer_to_ubo_map: &mut Vec<Option<usize>>,
                        layer_index: StackingContextIndex,
                        ctx: &RenderTargetContext) -> (usize, u32) {
        let index_in_ubo = match layer_to_ubo_map[layer_index.0] {
            Some(index_in_ubo) => {
                index_in_ubo
            }
            None => {
                let need_new_ubo = layer_ubos.is_empty() ||
                                   layer_ubos.last().unwrap().len() == ctx.alpha_batch_max_layers;

                if need_new_ubo {
                    for i in 0..layer_to_ubo_map.len() {
                        layer_to_ubo_map[i] = None;
                    }
                    layer_ubos.push(Vec::new());
                }

                let layer_ubo = layer_ubos.last_mut().unwrap();
                let index = layer_ubo.len();
                let sc = &ctx.layer_store[layer_index.0];
                layer_ubo.push(PackedLayer {
                    transform: sc.transform,
                    inv_transform: sc.transform.invert(),
                    screen_vertices: sc.xf_rect.as_ref().unwrap().vertices,
                    world_clip_rect: sc.world_clip_rect.unwrap(),
                });
                layer_to_ubo_map[layer_index.0] = Some(index);
                index
            }
        };

        (layer_ubos.len() - 1, index_in_ubo as u32)
    }

    fn add_task(&mut self, task: AlphaBatchTask) {
        self.tasks.push(task);
    }

    fn build(&mut self, packed_primitive_cache: &PackedPrimitiveCache, ctx: &RenderTargetContext) {
        for _ in 0..ctx.layer_store.len() {
            self.layer_to_ubo_map.push(None);
        }
        for _ in 0..self.tasks.len() {
            self.tile_to_ubo_map.push(None);
        }

        loop {
            // Pull next primitive
            let mut batch = None;
            for (task_index, task) in self.tasks.iter_mut().enumerate() {
                let next_item = match task.items.pop() {
                    Some(next_item) => next_item,
                    None => continue,
                };
                match next_item {
                    AlphaRenderItem::Composite(info) => {
                        batch = Some(PrimitiveBatch::composite(task.child_rects[0],
                                                               task.child_rects[1],
                                                               task.target_rect,
                                                               info));
                        break;
                    }
                    AlphaRenderItem::Blend(child_index, opacity) => {
                        batch = Some(PrimitiveBatch::blend(task.child_rects[child_index],
                                                           task.target_rect,
                                                           opacity));
                        break;
                    }
                    AlphaRenderItem::Primitive(sc_index, prim_index) => {
                        // See if this task fits into the tile UBO
                        let layer = &ctx.layer_store[sc_index.0];
                        let prim = &ctx.prim_store[prim_index.0];
                        let transform_kind = layer.xf_rect.as_ref().unwrap().kind;
                        let (layer_ubo_index, index_in_layer_ubo) =
                            AlphaBatcher::add_layer_to_ubo(&mut self.layer_ubos,
                                                           &mut self.layer_to_ubo_map,
                                                           sc_index,
                                                           ctx);
                        let (tile_ubo_index, index_in_tile_ubo) =
                            AlphaBatcher::add_tile_to_ubo(&mut self.tile_ubos,
                                                          &mut self.tile_to_ubo_map,
                                                          TaskIndex(task_index),
                                                          task,
                                                          ctx);
                        let needs_blending = !prim.is_opaque(ctx.resource_cache, ctx.frame_id);
                        let mut new_batch = PrimitiveBatch::new(prim,
                                                                transform_kind,
                                                                layer_ubo_index,
                                                                tile_ubo_index,
                                                                needs_blending);
                        let ok = packed_primitive_cache.add_to_batch(prim_index,
                                                                     &mut new_batch,
                                                                     index_in_layer_ubo,
                                                                     index_in_tile_ubo,
                                                                     transform_kind,
                                                                     needs_blending);
                        debug_assert!(ok);
                        batch = Some(new_batch);
                        break;
                    }
                }
            }

            let mut batch = match batch {
                Some(batch) => batch,
                None => break,
            };
            for (task_index, task) in self.tasks.iter_mut().enumerate() {
                loop {
                    let next_item = match task.items.pop() {
                        Some(next_item) => next_item,
                        None => break,
                    };
                    match next_item {
                        AlphaRenderItem::Composite(info) => {
                            if !batch.pack_composite(task.child_rects[0],
                                                     task.child_rects[1],
                                                     task.target_rect,
                                                     info) {
                                task.items.push(next_item);
                                break;
                            }
                        }
                        AlphaRenderItem::Blend(child_index, opacity) => {
                            if !batch.pack_blend(task.child_rects[child_index],
                                                 task.target_rect,
                                                 opacity) {
                                task.items.push(next_item);
                                break;
                            }
                        }
                        AlphaRenderItem::Primitive(sc_index, prim_index) => {
                            let layer = &ctx.layer_store[sc_index.0];
                            let prim = &ctx.prim_store[prim_index.0];
                            let transform_kind = layer.xf_rect.as_ref().unwrap().kind;
                            let (layer_ubo_index, index_in_layer_ubo) =
                                AlphaBatcher::add_layer_to_ubo(&mut self.layer_ubos,
                                                               &mut self.layer_to_ubo_map,
                                                               sc_index,
                                                               ctx);
                            let (tile_ubo_index, index_in_tile_ubo) =
                                AlphaBatcher::add_tile_to_ubo(&mut self.tile_ubos,
                                                              &mut self.tile_to_ubo_map,
                                                              TaskIndex(task_index),
                                                              task,
                                                              ctx);

                            if layer_ubo_index != batch.layer_ubo_index ||
                               tile_ubo_index != batch.tile_ubo_index ||
                               !packed_primitive_cache.add_to_batch(prim_index,
                                                                    &mut batch,
                                                                    index_in_layer_ubo,
                                                                    index_in_tile_ubo,
                                                                    transform_kind,
                                                                    !prim.is_opaque(ctx.resource_cache, ctx.frame_id)) {
                                task.items.push(next_item);
                                break;
                            }
                        }
                    }
                }
            }

            self.batches.push(batch);
        }
    }
}

struct RenderTargetContext<'a> {
    layer_store: &'a Vec<StackingContext>,
    prim_store: &'a Vec<Primitive>,
    resource_cache: &'a ResourceCache,
    device_pixel_ratio: f32,
    frame_id: FrameId,
    alpha_batch_max_tiles: usize,
    alpha_batch_max_layers: usize,
}

pub struct RenderTarget {
    pub is_framebuffer: bool,
    page_allocator: TexturePage,
    tasks: Vec<RenderTask>,

    pub alpha_batchers: Vec<AlphaBatcher>,
}

impl RenderTarget {
    fn new(is_framebuffer: bool) -> RenderTarget {
        RenderTarget {
            is_framebuffer: is_framebuffer,
            page_allocator: TexturePage::new(TextureId(0), RENDERABLE_CACHE_SIZE.0 as u32),
            tasks: Vec::new(),

            alpha_batchers: Vec::new(),
        }
    }

    fn add_render_task(&mut self, task: RenderTask) {
        self.tasks.push(task);
    }

    fn build(&mut self, packed_primitive_cache: &PackedPrimitiveCache, ctx: &RenderTargetContext) {
        // Step through each task, adding to batches as appropriate.

        for task in self.tasks.drain(..) {
            let target_rect = task.get_target_rect();

            match task.kind {
                RenderTaskKind::Alpha(info) => {
                    let need_new_batcher = self.alpha_batchers.is_empty() ||
                                           self.alpha_batchers.last().unwrap().tasks.len() == 64;

                    if need_new_batcher {
                        self.alpha_batchers.push(AlphaBatcher::new());
                    }

                    self.alpha_batchers.last_mut().unwrap().add_task(AlphaBatchTask {
                        target_rect: target_rect,
                        actual_rect: info.actual_rect,
                        items: info.items,
                        child_rects: task.child_locations.clone(),  // TODO(gw): Remove clone somehow!?
                    });
                }
            }
        }

        for ab in &mut self.alpha_batchers {
            ab.build(packed_primitive_cache, ctx);
        }
    }
}

pub struct RenderPhase {
    pub targets: Vec<RenderTarget>,
}

impl RenderPhase {
    fn new(max_target_count: usize) -> RenderPhase {
        let mut targets = Vec::with_capacity(max_target_count);
        for index in 0..max_target_count {
            targets.push(RenderTarget::new(index == max_target_count-1));
        }

        RenderPhase {
            targets: targets,
        }
    }

    fn add_compiled_screen_tile(&mut self,
                                mut tile: CompiledScreenTile) -> Option<CompiledScreenTile> {
        debug_assert!(tile.required_target_count <= self.targets.len());

        let ok = tile.main_render_task.alloc_if_required(self.targets.len() - 1,
                                                         &mut self.targets);

        if ok {
            tile.main_render_task.assign_to_targets(self.targets.len() - 1,
                                                    &mut self.targets);
            None
        } else {
            Some(tile)
        }
    }

    fn build(&mut self, packed_primitive_cache: &PackedPrimitiveCache, ctx: &RenderTargetContext) {
        for target in &mut self.targets {
            target.build(packed_primitive_cache, ctx);
        }
    }
}

#[derive(Debug)]
enum RenderTaskLocation {
    Fixed(Rect<DevicePixel>),
    Dynamic(Option<Point2D<DevicePixel>>, Size2D<DevicePixel>),
}

#[derive(Debug)]
enum AlphaRenderItem {
    Primitive(StackingContextIndex, PrimitiveIndex),
    Blend(usize, f32),
    Composite(PackedCompositeInfo),
}

#[derive(Debug)]
struct AlphaRenderTask {
    actual_rect: Rect<DevicePixel>,
    items: Vec<AlphaRenderItem>,
    children: Vec<AlphaRenderTask>,
}

impl AlphaRenderTask {
    fn new(actual_rect: Rect<DevicePixel>) -> AlphaRenderTask {
        AlphaRenderTask {
            actual_rect: actual_rect,
            items: Vec::new(),
            children: Vec::new(),
        }
    }
}

#[derive(Debug)]
enum RenderTaskKind {
    Alpha(AlphaRenderTask),
}

#[derive(Debug)]
struct RenderTask {
    location: RenderTaskLocation,
    children: Vec<RenderTask>,
    child_locations: Vec<Rect<DevicePixel>>,
    kind: RenderTaskKind,
}

impl RenderTask {
    fn from_primitives(mut task: AlphaRenderTask,
                       location: RenderTaskLocation,
                       size: Size2D<DevicePixel>) -> RenderTask {
        let mut children = Vec::new();
        for child in task.children.drain(..) {
            let location = RenderTaskLocation::Dynamic(None, size);
            children.push(RenderTask::from_primitives(child, location, size));
        }

        task.items.reverse();

        RenderTask {
            children: children,
            child_locations: Vec::new(),
            location: location,
            kind: RenderTaskKind::Alpha(task),
        }
    }

    fn get_target_rect(&self) -> Rect<DevicePixel> {
        match self.location {
            RenderTaskLocation::Fixed(rect) => rect,
            RenderTaskLocation::Dynamic(origin, size) => {
                Rect::new(origin.expect("Should have been allocated by now!"),
                          size)
            }
        }
    }

    fn assign_to_targets(mut self,
                         target_index: usize,
                         targets: &mut Vec<RenderTarget>) {
        for child in self.children.drain(..) {
            self.child_locations.push(child.get_target_rect());
            child.assign_to_targets(target_index - 1, targets);
        }

        // Sanity check - can be relaxed if needed
        match self.location {
            RenderTaskLocation::Fixed(..) => {
                debug_assert!(target_index == targets.len() - 1);
            }
            RenderTaskLocation::Dynamic(..) => {
                debug_assert!(target_index < targets.len() - 1);
            }
        }

        let target = &mut targets[target_index];
        target.add_render_task(self);
    }

    fn alloc_if_required(&mut self,
                         target_index: usize,
                         targets: &mut Vec<RenderTarget>) -> bool {
        match self.location {
            RenderTaskLocation::Fixed(..) => {}
            RenderTaskLocation::Dynamic(ref mut origin, ref size) => {
                let target = &mut targets[target_index];

                let alloc_size = Size2D::new(size.width.0 as u32,
                                             size.height.0 as u32);

                let alloc_origin = target.page_allocator.allocate(&alloc_size);

                match alloc_origin {
                    Some(alloc_origin) => {
                        *origin = Some(Point2D::new(DevicePixel(alloc_origin.x as i32),
                                                    DevicePixel(alloc_origin.y as i32)));
                    }
                    None => {
                        return false;
                    }
                }
            }
        }

        for child in &mut self.children {
            if !child.alloc_if_required(target_index - 1,
                                        targets) {
                return false;
            }
        }

        true
    }

    fn max_depth(&self,
                 depth: usize,
                 max_depth: &mut usize) {
        let depth = depth + 1;
        *max_depth = cmp::max(*max_depth, depth);
        for child in &self.children {
            child.max_depth(depth, max_depth);
        }
    }
}

pub const SCREEN_TILE_SIZE: i32 = 64;
pub const RENDERABLE_CACHE_SIZE: DevicePixel = DevicePixel(2048);
const MAX_STOPS_PER_ANGLE_GRADIENT: usize = 8;

#[derive(Debug, Clone)]
pub struct DebugRect {
    pub label: String,
    pub color: ColorF,
    pub rect: Rect<DevicePixel>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TransformedRectKind {
    AxisAligned,
    Complex,
}

#[derive(Debug, Clone)]
struct TransformedRect {
    local_rect: Rect<f32>,
    bounding_rect: Rect<DevicePixel>,
    vertices: [Point4D<f32>; 4],
    kind: TransformedRectKind,
}

impl TransformedRect {
    fn new(rect: &Rect<f32>,
           transform: &Matrix4D<f32>,
           device_pixel_ratio: f32) -> TransformedRect {

        let kind = if transform.can_losslessly_transform_and_perspective_project_a_2d_rect() {
            TransformedRectKind::AxisAligned
        } else {
            TransformedRectKind::Complex
        };

/*
        match kind {
            TransformedRectKind::AxisAligned => {
                let v0 = transform.transform_point(&rect.origin);
                let v1 = transform.transform_point(&rect.top_right());
                let v2 = transform.transform_point(&rect.bottom_left());
                let v3 = transform.transform_point(&rect.bottom_right());

                let screen_min_dp = Point2D::new(DevicePixel((v0.x * device_pixel_ratio).floor() as i32),
                                                 DevicePixel((v0.y * device_pixel_ratio).floor() as i32));
                let screen_max_dp = Point2D::new(DevicePixel((v3.x * device_pixel_ratio).ceil() as i32),
                                                 DevicePixel((v3.y * device_pixel_ratio).ceil() as i32));

                let screen_rect_dp = Rect::new(screen_min_dp, Size2D::new(screen_max_dp.x - screen_min_dp.x,
                                                                          screen_max_dp.y - screen_min_dp.y));

                TransformedRect {
                    local_rect: *rect,
                    vertices: [
                        Point4D::new(v0.x, v0.y, 0.0, 1.0),
                        Point4D::new(v1.x, v1.y, 0.0, 1.0),
                        Point4D::new(v2.x, v2.y, 0.0, 1.0),
                        Point4D::new(v3.x, v3.y, 0.0, 1.0),
                    ],
                    bounding_rect: screen_rect_dp,
                    kind: kind,
                }
            }
            TransformedRectKind::Complex => {
                */
                let vertices = [
                    transform.transform_point4d(&Point4D::new(rect.origin.x,
                                                              rect.origin.y,
                                                              0.0,
                                                              1.0)),
                    transform.transform_point4d(&Point4D::new(rect.bottom_left().x,
                                                              rect.bottom_left().y,
                                                              0.0,
                                                              1.0)),
                    transform.transform_point4d(&Point4D::new(rect.bottom_right().x,
                                                              rect.bottom_right().y,
                                                              0.0,
                                                              1.0)),
                    transform.transform_point4d(&Point4D::new(rect.top_right().x,
                                                              rect.top_right().y,
                                                              0.0,
                                                              1.0)),
                ];


                let mut screen_min : Point2D<f32> = Point2D::new(10000000.0, 10000000.0);
                let mut screen_max : Point2D<f32>  = Point2D::new(-10000000.0, -10000000.0);

                for vertex in &vertices {
                    let inv_w = 1.0 / vertex.w;
                    let vx = vertex.x * inv_w;
                    let vy = vertex.y * inv_w;
                    screen_min.x = screen_min.x.min(vx);
                    screen_min.y = screen_min.y.min(vy);
                    screen_max.x = screen_max.x.max(vx);
                    screen_max.y = screen_max.y.max(vy);
                }

                let screen_min_dp = Point2D::new(DevicePixel((screen_min.x * device_pixel_ratio).floor() as i32),
                                                 DevicePixel((screen_min.y * device_pixel_ratio).floor() as i32));
                let screen_max_dp = Point2D::new(DevicePixel((screen_max.x * device_pixel_ratio).ceil() as i32),
                                                 DevicePixel((screen_max.y * device_pixel_ratio).ceil() as i32));

                let screen_rect_dp = Rect::new(screen_min_dp, Size2D::new(screen_max_dp.x - screen_min_dp.x,
                                                                          screen_max_dp.y - screen_min_dp.y));

                TransformedRect {
                    local_rect: *rect,
                    vertices: vertices,
                    bounding_rect: screen_rect_dp,
                    kind: kind,
                }
                /*
            }
        }*/
    }
}

#[derive(Debug)]
struct RectanglePrimitive {
    color: ColorF,
}

#[derive(Debug)]
struct TextPrimitive {
    color: ColorF,
    font_key: FontKey,
    size: Au,
    blur_radius: Au,
    glyph_range: ItemRange,
}

#[derive(Debug)]
struct BoxShadowPrimitive {
    src_rect: Rect<f32>,
    bs_rect: Rect<f32>,
    color: ColorF,
    blur_radius: f32,
    spread_radius: f32,
    border_radius: f32,
    clip_mode: BoxShadowClipMode,
    metrics: BoxShadowMetrics,
}

#[derive(Debug)]
struct BorderPrimitive {
    tl_outer: Point2D<f32>,
    tl_inner: Point2D<f32>,
    tr_outer: Point2D<f32>,
    tr_inner: Point2D<f32>,
    bl_outer: Point2D<f32>,
    bl_inner: Point2D<f32>,
    br_outer: Point2D<f32>,
    br_inner: Point2D<f32>,
    left_width: f32,
    top_width: f32,
    right_width: f32,
    bottom_width: f32,
    radius: BorderRadius,
    left_color: ColorF,
    top_color: ColorF,
    right_color: ColorF,
    bottom_color: ColorF,
    left_style: BorderStyle,
    top_style: BorderStyle,
    right_style: BorderStyle,
    bottom_style: BorderStyle,
}

#[derive(Debug)]
enum ImagePrimitiveKind {
    Image(ImageKey, ImageRendering, Size2D<f32>),
    WebGL(WebGLContextId),
}

#[derive(Debug)]
struct ImagePrimitive {
    kind: ImagePrimitiveKind,
}

#[derive(Debug)]
struct GradientPrimitive {
    stops_range: ItemRange,
    kind: GradientType,
    start_point: Point2D<f32>,
    end_point: Point2D<f32>,
}

#[derive(Debug)]
enum PrimitiveDetails {
    Rectangle(RectanglePrimitive),
    Text(TextPrimitive),
    Image(ImagePrimitive),
    Border(BorderPrimitive),
    Gradient(GradientPrimitive),
    BoxShadow(BoxShadowPrimitive),
}

#[derive(Copy, Clone, Debug)]
struct LayerPackedPrimitiveRangeStartOffsets {
    rectangles: usize,
    rectangles_clip: usize,
    borders: usize,
    box_shadows: usize,
    text: usize,
    images: usize,
    gradients: usize,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PrimitiveIndex(usize);

#[derive(Debug)]
struct Primitive {
    rect: Rect<f32>,
    local_clip_rect: Rect<f32>,
    complex_clip: Option<Box<Clip>>,
    xf_rect: Option<TransformedRect>,
    details: PrimitiveDetails,
}

impl Primitive {
    fn is_opaque(&self, resource_cache: &ResourceCache, frame_id: FrameId) -> bool {
        match self.details {
            PrimitiveDetails::Rectangle(ref primitive) => primitive.color.a == 1.0,
            PrimitiveDetails::Image(ImagePrimitive {
                kind: ImagePrimitiveKind::Image(image_key, image_rendering, _),
            }) => resource_cache.get_image(image_key, image_rendering, frame_id).is_opaque,
            _ => false,
        }
    }

    fn pack(&self,
            index: PrimitiveIndex,
            cache: &mut PackedPrimitiveCache,
            auxiliary_lists: &AuxiliaryLists,
            ctx: &RenderTargetContext) {
        // TODO(pcwalton): Only pack visible primitives!
        cache.init_packed_primitive(index);

        if self.xf_rect.is_none() {
            return;
        }

        match self.details {
            PrimitiveDetails::Rectangle(ref details) => {
                match self.complex_clip {
                    Some(ref clip) => {
                        let packed = PackedPrimitive::RectangleClip(PackedRectanglePrimitiveClip {
                            common: PackedPrimitiveInfo {
                                padding: 0,
                                tile_index: 0,
                                layer_index: 0,
                                part: PrimitivePart::Invalid,
                                local_clip_rect: self.local_clip_rect,
                                local_rect: self.rect,
                            },
                            color: details.color,
                            clip: (**clip).clone(),
                        });
                        cache.add_packed_primitive(index, packed, TextureId(0))
                    }
                    None => {
                        let packed = PackedPrimitive::Rectangle(PackedRectanglePrimitive {
                            common: PackedPrimitiveInfo {
                                padding: 0,
                                tile_index: 0,
                                layer_index: 0,
                                part: PrimitivePart::Invalid,
                                local_clip_rect: self.local_clip_rect,
                                local_rect: self.rect,
                            },
                            color: details.color,
                        });
                        cache.add_packed_primitive(index, packed, TextureId(0))
                    }
                }
            }
            PrimitiveDetails::Image(ref details) => {
                let (texture_id, uv_rect, stretch_size) = match details.kind {
                    ImagePrimitiveKind::Image(image_key, image_rendering, stretch_size) => {
                        let info = ctx.resource_cache.get_image(image_key,
                                                                image_rendering,
                                                                ctx.frame_id);
                        (info.texture_id, info.uv_rect(), stretch_size)
                    }
                    ImagePrimitiveKind::WebGL(context_id) => {
                        let texture_id = ctx.resource_cache.get_webgl_texture(&context_id);
                        let uv = RectUv {
                            top_left: Point2D::new(0.0, 1.0),
                            top_right: Point2D::new(1.0, 1.0),
                            bottom_left: Point2D::zero(),
                            bottom_right: Point2D::new(1.0, 0.0),
                        };
                        (texture_id, uv, self.rect.size)
                    }
                };

                let packed = match self.complex_clip {
                    Some(ref clip) => {
                        PackedPrimitive::ImageClip(PackedImagePrimitiveClip {
                            common: PackedPrimitiveInfo {
                                padding: 0,
                                tile_index: 0,
                                layer_index: 0,
                                part: PrimitivePart::Invalid,
                                local_clip_rect: self.local_clip_rect,
                                local_rect: self.rect,
                            },
                            st0: uv_rect.top_left,
                            st1: uv_rect.bottom_right,
                            stretch_size: stretch_size,
                            padding: [0, 0],
                            clip: (**clip).clone(),
                        })
                    }
                    None => {
                         PackedPrimitive::Image(PackedImagePrimitive {
                            common: PackedPrimitiveInfo {
                                padding: 0,
                                tile_index: 0,
                                layer_index: 0,
                                part: PrimitivePart::Invalid,
                                local_clip_rect: self.local_clip_rect,
                                local_rect: self.rect,
                            },
                            st0: uv_rect.top_left,
                            st1: uv_rect.bottom_right,
                            stretch_size: stretch_size,
                            padding: [0, 0],
                        })
                    }
                };

                cache.add_packed_primitive(index, packed, texture_id)
            }
            PrimitiveDetails::Border(ref details) => {
                let inner_radius = BorderRadius {
                    top_left: Size2D::new(details.radius.top_left.width - details.left_width,
                                          details.radius.top_left.height - details.top_width),
                    top_right: Size2D::new(details.radius.top_right.width - details.right_width,
                                           details.radius.top_right.height - details.top_width),
                    bottom_left:
                        Size2D::new(details.radius.bottom_left.width - details.left_width,
                                    details.radius.bottom_left.height - details.bottom_width),
                    bottom_right:
                        Size2D::new(details.radius.bottom_right.width - details.right_width,
                                    details.radius.bottom_right.height - details.bottom_width),
                };

                cache.add_packed_primitive(index, PackedPrimitive::Border(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        tile_index: 0,
                        layer_index: 0,
                        part: PrimitivePart::TopLeft,
                        local_clip_rect: self.local_clip_rect,
                        local_rect: rect_from_points_f(details.tl_outer.x,
                                                       details.tl_outer.y,
                                                       details.tl_inner.x,
                                                       details.tl_inner.y),
                    },
                    vertical_color: details.top_color,
                    horizontal_color: details.left_color,
                    outer_radius_x: details.radius.top_left.width,
                    outer_radius_y: details.radius.top_left.height,
                    inner_radius_x: inner_radius.top_left.width,
                    inner_radius_y: inner_radius.top_left.height,
                    top_style: details.top_style as u32,
                    right_style: details.right_style as u32,
                    bottom_style: details.bottom_style as u32,
                    left_style: details.bottom_style as u32,
                }), TextureId(0));

                cache.add_packed_primitive(index, PackedPrimitive::Border(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        tile_index: 0,
                        layer_index: 0,
                        part: PrimitivePart::TopRight,
                        local_clip_rect: self.local_clip_rect,
                        local_rect: rect_from_points_f(details.tr_inner.x,
                                                       details.tr_outer.y,
                                                       details.tr_outer.x,
                                                       details.tr_inner.y),
                    },
                    vertical_color: details.right_color,
                    horizontal_color: details.top_color,
                    outer_radius_x: details.radius.top_right.width,
                    outer_radius_y: details.radius.top_right.height,
                    inner_radius_x: inner_radius.top_right.width,
                    inner_radius_y: inner_radius.top_right.height,
                    top_style: details.top_style as u32,
                    right_style: details.right_style as u32,
                    bottom_style: details.bottom_style as u32,
                    left_style: details.bottom_style as u32,
                }), TextureId(0));

                cache.add_packed_primitive(index, PackedPrimitive::Border(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        tile_index: 0,
                        layer_index: 0,
                        part: PrimitivePart::BottomLeft,
                        local_clip_rect: self.local_clip_rect,
                        local_rect: rect_from_points_f(details.bl_outer.x,
                                                       details.bl_inner.y,
                                                       details.bl_inner.x,
                                                       details.bl_outer.y),
                    },
                    vertical_color: details.left_color,
                    horizontal_color: details.bottom_color,
                    outer_radius_x: details.radius.bottom_left.width,
                    outer_radius_y: details.radius.bottom_left.height,
                    inner_radius_x: inner_radius.bottom_left.width,
                    inner_radius_y: inner_radius.bottom_left.height,
                    top_style: details.top_style as u32,
                    right_style: details.right_style as u32,
                    bottom_style: details.bottom_style as u32,
                    left_style: details.bottom_style as u32,
                }), TextureId(0));

                cache.add_packed_primitive(index, PackedPrimitive::Border(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        tile_index: 0,
                        layer_index: 0,
                        part: PrimitivePart::BottomRight,
                        local_clip_rect: self.local_clip_rect,
                        local_rect: rect_from_points_f(details.br_inner.x,
                                                       details.br_inner.y,
                                                       details.br_outer.x,
                                                       details.br_outer.y),
                    },
                    vertical_color: details.right_color,
                    horizontal_color: details.bottom_color,
                    outer_radius_x: details.radius.bottom_right.width,
                    outer_radius_y: details.radius.bottom_right.height,
                    inner_radius_x: inner_radius.bottom_right.width,
                    inner_radius_y: inner_radius.bottom_right.height,
                    top_style: details.top_style as u32,
                    right_style: details.right_style as u32,
                    bottom_style: details.bottom_style as u32,
                    left_style: details.bottom_style as u32,
                }), TextureId(0));

                cache.add_packed_primitive(index, PackedPrimitive::Border(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        tile_index: 0,
                        layer_index: 0,
                        part: PrimitivePart::Left,
                        local_clip_rect: self.local_clip_rect,
                        local_rect: rect_from_points_f(details.tl_outer.x,
                                                       details.tl_inner.y,
                                                       details.tl_outer.x + details.left_width,
                                                       details.bl_inner.y),
                    },
                    vertical_color: details.left_color,
                    horizontal_color: details.left_color,
                    outer_radius_x: 0.0,
                    outer_radius_y: 0.0,
                    inner_radius_x: 0.0,
                    inner_radius_y: 0.0,
                    top_style: details.top_style as u32,
                    right_style: details.right_style as u32,
                    bottom_style: details.bottom_style as u32,
                    left_style: details.bottom_style as u32,
                }), TextureId(0));

                cache.add_packed_primitive(index, PackedPrimitive::Border(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        tile_index: 0,
                        layer_index: 0,
                        part: PrimitivePart::Right,
                        local_clip_rect: self.local_clip_rect,
                        local_rect: rect_from_points_f(details.tr_outer.x - details.right_width,
                                                       details.tr_inner.y,
                                                       details.br_outer.x,
                                                       details.br_inner.y),
                    },
                    vertical_color: details.right_color,
                    horizontal_color: details.right_color,
                    outer_radius_x: 0.0,
                    outer_radius_y: 0.0,
                    inner_radius_x: 0.0,
                    inner_radius_y: 0.0,
                    top_style: details.top_style as u32,
                    right_style: details.right_style as u32,
                    bottom_style: details.bottom_style as u32,
                    left_style: details.bottom_style as u32,
                }), TextureId(0));

                cache.add_packed_primitive(index, PackedPrimitive::Border(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        tile_index: 0,
                        layer_index: 0,
                        part: PrimitivePart::Top,
                        local_clip_rect: self.local_clip_rect,
                        local_rect: rect_from_points_f(details.tl_inner.x,
                                                       details.tl_outer.y,
                                                       details.tr_inner.x,
                                                       details.tr_outer.y + details.top_width),
                    },
                    vertical_color: details.top_color,
                    horizontal_color: details.top_color,
                    outer_radius_x: 0.0,
                    outer_radius_y: 0.0,
                    inner_radius_x: 0.0,
                    inner_radius_y: 0.0,
                    top_style: details.top_style as u32,
                    right_style: details.right_style as u32,
                    bottom_style: details.bottom_style as u32,
                    left_style: details.bottom_style as u32,
                }), TextureId(0));

                cache.add_packed_primitive(index, PackedPrimitive::Border(PackedBorderPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: 0,
                        tile_index: 0,
                        layer_index: 0,
                        part: PrimitivePart::Bottom,
                        local_clip_rect: self.local_clip_rect,
                        local_rect: rect_from_points_f(details.bl_inner.x,
                                                       details.bl_outer.y - details.bottom_width,
                                                       details.br_inner.x,
                                                       details.br_outer.y),
                    },
                    vertical_color: details.bottom_color,
                    horizontal_color: details.bottom_color,
                    outer_radius_x: 0.0,
                    outer_radius_y: 0.0,
                    inner_radius_x: 0.0,
                    inner_radius_y: 0.0,
                    top_style: details.top_style as u32,
                    right_style: details.right_style as u32,
                    bottom_style: details.bottom_style as u32,
                    left_style: details.bottom_style as u32,
                }), TextureId(0));
            }
            PrimitiveDetails::Gradient(ref details) => {
                match details.kind {
                    GradientType::Horizontal | GradientType::Vertical => {
                        let stops = auxiliary_lists.gradient_stops(&details.stops_range);
                        for i in 0..(stops.len() - 1) {
                            let (prev_stop, next_stop) = (&stops[i], &stops[i + 1]);
                            let piece_origin;
                            let piece_size;
                            match details.kind {
                                GradientType::Horizontal => {
                                    let prev_x = util::lerp(details.start_point.x,
                                                            details.end_point.x,
                                                            prev_stop.offset);
                                    let next_x = util::lerp(details.start_point.x,
                                                            details.end_point.x,
                                                            next_stop.offset);
                                    piece_origin = Point2D::new(prev_x, self.rect.origin.y);
                                    piece_size = Size2D::new(next_x - prev_x,
                                                             self.rect.size.height);
                                }
                                GradientType::Vertical => {
                                    let prev_y = util::lerp(details.start_point.y,
                                                            details.end_point.y,
                                                            prev_stop.offset);
                                    let next_y = util::lerp(details.start_point.y,
                                                            details.end_point.y,
                                                            next_stop.offset);
                                    piece_origin = Point2D::new(self.rect.origin.x, prev_y);
                                    piece_size = Size2D::new(self.rect.size.width, next_y - prev_y);
                                }
                                GradientType::Rotated => unreachable!(),
                            }

                            let piece_rect = Rect::new(piece_origin, piece_size);
                            let mut clip = Clip::invalid(piece_rect);

                            if let Some(ref prim_clip) = self.complex_clip {
                                if i == 0 {
                                    clip.top_left.outer_radius_x = prim_clip.top_left
                                                                            .outer_radius_x;
                                    clip.top_left.outer_radius_y = prim_clip.top_left
                                                                            .outer_radius_y;

                                    match details.kind {
                                        GradientType::Horizontal => {
                                            clip.bottom_left.outer_radius_x =
                                                prim_clip.bottom_left.outer_radius_x;
                                            clip.bottom_left.outer_radius_y =
                                                prim_clip.bottom_left.outer_radius_y;
                                        }
                                        GradientType::Vertical => {
                                            clip.top_right.outer_radius_x =
                                                prim_clip.top_right.outer_radius_x;
                                            clip.top_right.outer_radius_y =
                                                prim_clip.top_right.outer_radius_y;
                                        }
                                        GradientType::Rotated => unreachable!(),
                                    }
                                }

                                if i == stops.len() - 2 {
                                    clip.bottom_right.outer_radius_x = prim_clip.bottom_right
                                                                                .outer_radius_x;
                                    clip.bottom_right.outer_radius_y = prim_clip.bottom_right
                                                                                .outer_radius_y;

                                    match details.kind {
                                        GradientType::Horizontal => {
                                            clip.top_right.outer_radius_x =
                                                prim_clip.top_right.outer_radius_x;
                                            clip.top_right.outer_radius_y =
                                                prim_clip.top_right.outer_radius_y;
                                        }
                                        GradientType::Vertical => {
                                            clip.bottom_left.outer_radius_x =
                                                prim_clip.bottom_left.outer_radius_x;
                                            clip.bottom_left.outer_radius_y =
                                                prim_clip.bottom_left.outer_radius_y;
                                        }
                                        GradientType::Rotated => unreachable!(),
                                    }
                                }
                            }

                            cache.add_packed_primitive(
                                index,
                                PackedPrimitive::AlignedGradient(PackedAlignedGradientPrimitive {
                                    common: PackedPrimitiveInfo {
                                        padding: 0,
                                        tile_index: 0,
                                        layer_index: 0,
                                        part: PrimitivePart::Bottom,
                                        local_clip_rect: self.local_clip_rect,
                                        local_rect: piece_rect,
                                    },
                                    color0: prev_stop.color,
                                    color1: next_stop.color,
                                    padding: [0, 0, 0],
                                    kind: details.kind,
                                    clip: clip,
                                }),
                                TextureId(0));
                        }
                    }
                    GradientType::Rotated => {
                        let src_stops = auxiliary_lists.gradient_stops(&details.stops_range);
                        if src_stops.len() > MAX_STOPS_PER_ANGLE_GRADIENT {
                            println!("TODO: Angle gradients with > {} stops",
                                     MAX_STOPS_PER_ANGLE_GRADIENT);
                            return
                        }

                        let mut stops: [f32; MAX_STOPS_PER_ANGLE_GRADIENT] = unsafe {
                            mem::uninitialized()
                        };
                        let mut colors: [ColorF; MAX_STOPS_PER_ANGLE_GRADIENT] = unsafe {
                            mem::uninitialized()
                        };

                        let sx = details.start_point.x;
                        let ex = details.end_point.x;

                        let (sp, ep) = if sx > ex {
                            for (stop_index, stop) in src_stops.iter().rev().enumerate() {
                                stops[stop_index] = 1.0 - stop.offset;
                                colors[stop_index] = stop.color;
                            }

                            (details.end_point, details.start_point)
                        } else {
                            for (stop_index, stop) in src_stops.iter().enumerate() {
                                stops[stop_index] = stop.offset;
                                colors[stop_index] = stop.color;
                            }

                            (details.start_point, details.end_point)
                        };

                        cache.add_packed_primitive(
                            index,
                            PackedPrimitive::AngleGradient(PackedAngleGradientPrimitive {
                                common: PackedPrimitiveInfo {
                                    padding: 0,
                                    tile_index: 0,
                                    layer_index: 0,
                                    part: PrimitivePart::Invalid,
                                    local_clip_rect: self.local_clip_rect,
                                    local_rect: self.rect,
                                },
                                padding: [0, 0, 0],
                                start_point: sp,
                                end_point: ep,
                                stop_count: src_stops.len() as u32,
                                stops: stops,
                                colors: colors,
                            }),
                            TextureId(0));
                    }
                }
            }
            PrimitiveDetails::BoxShadow(ref details) => {
                let mut rects = Vec::new();
                let inverted = match details.clip_mode {
                    BoxShadowClipMode::None | BoxShadowClipMode::Outset => {
                        subtract_rect(&self.rect, &details.src_rect, &mut rects);
                        0.0
                    }
                    BoxShadowClipMode::Inset => {
                        subtract_rect(&self.rect, &details.bs_rect, &mut rects);
                        1.0
                    }
                };

                for rect in rects {
                    cache.add_packed_primitive(
                        index,
                        PackedPrimitive::BoxShadow(PackedBoxShadowPrimitive {
                            common: PackedPrimitiveInfo {
                                padding: 0,
                                tile_index: 0,
                                layer_index: 0,
                                part: PrimitivePart::Invalid,
                                local_clip_rect: self.local_clip_rect,
                                local_rect: rect,
                            },
                            color: details.color,

                            border_radii: Point2D::new(details.border_radius,
                                                       details.border_radius),
                            blur_radius: details.blur_radius,
                            inverted: inverted,
                            bs_rect: details.bs_rect,
                            src_rect: details.src_rect,
                        }),
                        TextureId(0))
                }
            }
            PrimitiveDetails::Text(ref details) => {
                let src_glyphs = auxiliary_lists.glyph_instances(&details.glyph_range);
                let mut glyph_key = GlyphKey::new(details.font_key,
                                                  details.size,
                                                  details.blur_radius,
                                                  src_glyphs[0].index);
                let blur_offset = details.blur_radius.to_f32_px() *
                    (BLUR_INFLATION_FACTOR as f32) / 2.0;

                for glyph in src_glyphs {
                    glyph_key.index = glyph.index;

                    let image_info = match ctx.resource_cache.get_glyph(&glyph_key, ctx.frame_id) {
                        None => continue,
                        Some(image_info) => image_info,
                    };

                    // TODO(gw): Need a general solution to handle multiple texture pages per tile
                    // in WR2!
                    let texture_id = image_info.texture_id;
                    assert!(texture_id == TextureId(0) || texture_id == image_info.texture_id);

                    let x = glyph.x + image_info.user_data.x0 as f32 / ctx.device_pixel_ratio -
                        blur_offset;
                    let y = glyph.y - image_info.user_data.y0 as f32 / ctx.device_pixel_ratio -
                        blur_offset;

                    let width = image_info.requested_rect.size.width as f32 /
                        ctx.device_pixel_ratio;
                    let height = image_info.requested_rect.size.height as f32 /
                        ctx.device_pixel_ratio;

                    let uv_rect = image_info.uv_rect();
                    let local_rect = Rect::new(Point2D::new(x, y), Size2D::new(width, height));

                    cache.add_packed_primitive(
                        index,
                        PackedPrimitive::Text(PackedGlyphPrimitive {
                            common: PackedPrimitiveInfo {
                                padding: 0,
                                tile_index: 0,
                                layer_index: 0,
                                part: PrimitivePart::Invalid,
                                local_clip_rect: self.local_clip_rect,
                                local_rect: local_rect,
                            },
                            color: details.color,
                            st0: uv_rect.top_left,
                            st1: uv_rect.bottom_right,
                        }),
                        texture_id)
                }
            }
        }
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone)]
enum PrimitivePart {
    Invalid = 0,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Top,
    Left,
    Bottom,
    Right,
}

// All Packed Primitives below must be 16 byte aligned.
#[derive(Debug)]
pub struct PackedTile {
    actual_rect: Rect<DevicePixel>,
    target_rect: Rect<DevicePixel>,
}

#[derive(Debug)]
pub struct PackedLayer {
    transform: Matrix4D<f32>,
    inv_transform: Matrix4D<f32>,
    world_clip_rect: Rect<DevicePixel>,
    screen_vertices: [Point4D<f32>; 4],
}

#[derive(Debug, Clone)]
pub struct PackedPrimitiveInfo {
    layer_index: u32,
    tile_index: u32,
    part: PrimitivePart,
    padding: u32,
    local_clip_rect: Rect<f32>,
    local_rect: Rect<f32>,
}

#[derive(Debug, Clone)]
pub struct PackedRectanglePrimitiveClip {
    common: PackedPrimitiveInfo,
    color: ColorF,
    clip: Clip,
}

#[derive(Debug, Clone)]
pub struct PackedRectanglePrimitive {
    common: PackedPrimitiveInfo,
    color: ColorF,
}

#[derive(Debug, Clone)]
pub struct PackedGlyphPrimitive {
    common: PackedPrimitiveInfo,
    color: ColorF,
    st0: Point2D<f32>,
    st1: Point2D<f32>,
}

#[derive(Debug, Clone)]
pub struct PackedImagePrimitive {
    common: PackedPrimitiveInfo,
    st0: Point2D<f32>,
    st1: Point2D<f32>,
    stretch_size: Size2D<f32>,
    padding: [u32; 2],
}

#[derive(Debug, Clone)]
pub struct PackedImagePrimitiveClip {
    common: PackedPrimitiveInfo,
    st0: Point2D<f32>,
    st1: Point2D<f32>,
    stretch_size: Size2D<f32>,
    padding: [u32; 2],
    clip: Clip,
}

#[derive(Debug, Clone)]
pub struct PackedAlignedGradientPrimitive {
    common: PackedPrimitiveInfo,
    color0: ColorF,
    color1: ColorF,
    kind: GradientType,
    padding: [u32; 3],
    clip: Clip,
}

// TODO(gw): Angle gradient only support 8 stops due
//           to limits of interpolators. FIXME!
#[derive(Debug, Clone)]
pub struct PackedAngleGradientPrimitive {
    common: PackedPrimitiveInfo,
    start_point: Point2D<f32>,
    end_point: Point2D<f32>,
    stop_count: u32,
    padding: [u32; 3],
    colors: [ColorF; MAX_STOPS_PER_ANGLE_GRADIENT],
    stops: [f32; MAX_STOPS_PER_ANGLE_GRADIENT],
}

#[derive(Debug, Clone)]
pub struct PackedBorderPrimitive {
    common: PackedPrimitiveInfo,
    vertical_color:     ColorF,
    horizontal_color:   ColorF,
    outer_radius_x: f32,
    outer_radius_y: f32,
    inner_radius_x: f32,
    inner_radius_y: f32,
    top_style: u32,
    right_style: u32,
    bottom_style: u32,
    left_style: u32,
}

#[derive(Debug, Clone)]
pub struct PackedBoxShadowPrimitive {
    common: PackedPrimitiveInfo,
    color: ColorF,
    border_radii: Point2D<f32>,
    blur_radius: f32,
    inverted: f32,
    bs_rect: Rect<f32>,
    src_rect: Rect<f32>,
}

#[derive(Debug, Clone)]
pub struct PackedBlendPrimitive {
    target_rect: Rect<DevicePixel>,
    src_rect: Rect<DevicePixel>,
    opacity: f32,
    padding: [u32; 3],
}

#[derive(Debug, Copy, Clone)]
struct PackedCompositeInfo {
    kind: u32,
    op: u32,
    padding: [u32; 2],
    amount: f32,
    padding1: [u32; 3],
}

impl PackedCompositeInfo {
    fn new(ops: &Vec<CompositionOp>) -> PackedCompositeInfo {
        // TODO(gw): Support chained filters
        let op = &ops[0];

        let (kind, op, amount) = match op {
            &CompositionOp::MixBlend(mode) => {
                (0, mode as u32, 0.0)
            }
            &CompositionOp::Filter(filter) => {
                let (filter_mode, amount) = match filter {
                    LowLevelFilterOp::Blur(..) => (0, 0.0),
                    LowLevelFilterOp::Contrast(amount) => (1, amount.to_f32_px()),
                    LowLevelFilterOp::Grayscale(amount) => (2, amount.to_f32_px()),
                    LowLevelFilterOp::HueRotate(angle) => (3, (angle as f32) / ANGLE_FLOAT_TO_FIXED),
                    LowLevelFilterOp::Invert(amount) => (4, amount.to_f32_px()),
                    LowLevelFilterOp::Saturate(amount) => (5, amount.to_f32_px()),
                    LowLevelFilterOp::Sepia(amount) => (6, amount.to_f32_px()),
                    LowLevelFilterOp::Brightness(_) |
                    LowLevelFilterOp::Opacity(_) => {
                        // Expressible using GL blend modes, so not handled
                        // here.
                        unreachable!()
                    }
                };

                (1, filter_mode, amount)
            }
        };

        PackedCompositeInfo {
            kind: kind,
            op: op,
            padding: [0, 0],
            amount: amount,
            padding1: [0, 0, 0],
        }
    }
}

#[derive(Debug)]
pub struct PackedCompositePrimitive {
    rect0: Rect<DevicePixel>,
    rect1: Rect<DevicePixel>,
    target_rect: Rect<DevicePixel>,
    info: PackedCompositeInfo,
}

#[derive(Debug)]
pub enum PrimitiveBatchData {
    Rectangles(Vec<PackedRectanglePrimitive>),
    RectanglesClip(Vec<PackedRectanglePrimitiveClip>),
    Borders(Vec<PackedBorderPrimitive>),
    BoxShadows(Vec<PackedBoxShadowPrimitive>),
    Text(Vec<PackedGlyphPrimitive>),
    Image(Vec<PackedImagePrimitive>),
    ImageClip(Vec<PackedImagePrimitiveClip>),
    Blend(Vec<PackedBlendPrimitive>),
    Composite(Vec<PackedCompositePrimitive>),
    AlignedGradient(Vec<PackedAlignedGradientPrimitive>),
    AngleGradient(Vec<PackedAngleGradientPrimitive>),
}

enum PackedPrimitive {
    Rectangle(PackedRectanglePrimitive),
    RectangleClip(PackedRectanglePrimitiveClip),
    Border(PackedBorderPrimitive),
    BoxShadow(PackedBoxShadowPrimitive),
    Text(PackedGlyphPrimitive),
    Image(PackedImagePrimitive),
    ImageClip(PackedImagePrimitiveClip),
    AlignedGradient(PackedAlignedGradientPrimitive),
    AngleGradient(PackedAngleGradientPrimitive),
}

#[derive(Debug)]
pub struct PrimitiveBatch {
    pub transform_kind: TransformedRectKind,
    pub color_texture_id: TextureId,        // TODO(gw): Expand to sampler array to handle all glyphs!
    pub layer_ubo_index: usize,
    pub tile_ubo_index: usize,
    pub blending_enabled: bool,
    pub data: PrimitiveBatchData,
}

impl PrimitiveBatch {
    fn blend(src_rect: Rect<DevicePixel>,
             target_rect: Rect<DevicePixel>,
             opacity: f32) -> PrimitiveBatch {
        let blend = PackedBlendPrimitive {
            src_rect: src_rect,
            target_rect: target_rect,
            opacity: opacity,
            padding: [0, 0, 0],
        };

        PrimitiveBatch {
            color_texture_id: TextureId(0),
            transform_kind: TransformedRectKind::AxisAligned,
            layer_ubo_index: 0,
            tile_ubo_index: 0,
            blending_enabled: true,
            data: PrimitiveBatchData::Blend(vec![blend]),
        }
    }

    fn composite(first_src_rect: Rect<DevicePixel>,
                 second_src_rect: Rect<DevicePixel>,
                 target_rect: Rect<DevicePixel>,
                 info: PackedCompositeInfo) -> PrimitiveBatch {
        let composite = PackedCompositePrimitive {
            rect0: first_src_rect,
            rect1: second_src_rect,
            target_rect: target_rect,
            info: info,
        };

        PrimitiveBatch {
            color_texture_id: TextureId(0),
            transform_kind: TransformedRectKind::AxisAligned,
            layer_ubo_index: 0,
            tile_ubo_index: 0,
            blending_enabled: true,
            data: PrimitiveBatchData::Composite(vec![composite]),
        }
    }

    fn pack_blend(&mut self,
                  src_rect: Rect<DevicePixel>,
                  target_rect: Rect<DevicePixel>,
                  opacity: f32) -> bool {
        match &mut self.data {
            &mut PrimitiveBatchData::Blend(ref mut ubo_data) => {
                ubo_data.push(PackedBlendPrimitive {
                    opacity: opacity,
                    padding: [0, 0, 0],
                    src_rect: src_rect,
                    target_rect: target_rect,
                });

                true
            }
            _ => false
        }
    }

    fn pack_composite(&mut self,
                      rect0: Rect<DevicePixel>,
                      rect1: Rect<DevicePixel>,
                      target_rect: Rect<DevicePixel>,
                      info: PackedCompositeInfo) -> bool {
        match &mut self.data {
            &mut PrimitiveBatchData::Composite(ref mut ubo_data) => {
                ubo_data.push(PackedCompositePrimitive {
                    rect0: rect0,
                    rect1: rect1,
                    target_rect: target_rect,
                    info: info,
                });

                true
            }
            _ => false
        }
    }

    fn new(prim: &Primitive,
           transform_kind: TransformedRectKind,
           layer_ubo_index: usize,
           tile_ubo_index: usize,
           blending_enabled: bool) -> PrimitiveBatch {
        let data = match prim.details {
            PrimitiveDetails::Rectangle(..) => {
                match prim.complex_clip {
                    Some(..) => PrimitiveBatchData::RectanglesClip(Vec::new()),
                    None => PrimitiveBatchData::Rectangles(Vec::new()),
                }
            }
            PrimitiveDetails::Border(..) => {
                PrimitiveBatchData::Borders(Vec::new())
            }
            PrimitiveDetails::BoxShadow(..) => {
                PrimitiveBatchData::BoxShadows(Vec::new())
            }
            PrimitiveDetails::Text(..) => {
                PrimitiveBatchData::Text(Vec::new())
            }
            PrimitiveDetails::Image(..) => {
                match prim.complex_clip {
                    Some(..) => PrimitiveBatchData::ImageClip(Vec::new()),
                    None => PrimitiveBatchData::Image(Vec::new()),
                }
            }
            PrimitiveDetails::Gradient(ref details) => {
                match details.kind {
                    GradientType::Rotated => {
                        PrimitiveBatchData::AngleGradient(Vec::new())
                    }
                    GradientType::Horizontal | GradientType::Vertical => {
                        PrimitiveBatchData::AlignedGradient(Vec::new())
                    }
                }
            }
        };

        PrimitiveBatch {
            color_texture_id: TextureId(0),
            transform_kind: transform_kind,
            layer_ubo_index: layer_ubo_index,
            tile_ubo_index: tile_ubo_index,
            blending_enabled: blending_enabled,
            data: data,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ScreenTileLayerIndex(usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct StackingContextIndex(usize);

enum StackingContextItem {
    StackingContext(StackingContextIndex),
    Primitive(PrimitiveIndex),
}

struct StackingContext {
    pipeline_id: PipelineId,
    local_transform: Matrix4D<f32>,
    local_rect: Rect<f32>,
    local_offset: Point2D<f32>,
    items: Vec<StackingContextItem>,
    scroll_layer_id: ScrollLayerId,
    transform: Matrix4D<f32>,
    xf_rect: Option<TransformedRect>,
    is_valid: bool,
    composition_ops: Vec<CompositionOp>,
    local_clip_rect: Rect<f32>,
    world_clip_rect: Option<Rect<DevicePixel>>,
    parent: Option<StackingContextIndex>,
}

#[derive(Debug, Copy, Clone)]
enum CompositeKind {
    None,
    Simple(f32),
    Complex(PackedCompositeInfo),
}

impl StackingContext {
    fn build_resource_list(&self,
                           resource_list: &mut ResourceList,
                           auxiliary_lists: &AuxiliaryLists,
                           prim_store: &Vec<Primitive>) {
        for item in &self.items {
            let prim_index = match item {
                &StackingContextItem::Primitive(prim_index) => prim_index,
                &StackingContextItem::StackingContext(..) => continue,
            };
            let prim = &prim_store[prim_index.0];
            if prim.xf_rect.is_none() {
                continue;
            }
            match prim.details {
                PrimitiveDetails::Rectangle(..) => {}
                PrimitiveDetails::Gradient(..) => {}
                PrimitiveDetails::Border(..) => {}
                PrimitiveDetails::BoxShadow(..) => {}
                PrimitiveDetails::Image(ref details) => {
                    match details.kind {
                        ImagePrimitiveKind::Image(image_key, image_rendering, _) => {
                            resource_list.add_image(image_key, image_rendering);
                        }
                        ImagePrimitiveKind::WebGL(..) => {}
                    }
                }
                PrimitiveDetails::Text(ref details) => {
                    let glyphs = auxiliary_lists.glyph_instances(&details.glyph_range);
                    for glyph in glyphs {
                        let glyph = Glyph::new(details.size, details.blur_radius, glyph.index);
                        resource_list.add_glyph(details.font_key, glyph);
                    }
                }
            }
        }
    }

    fn can_contribute_to_scene(&self) -> bool {
        for op in &self.composition_ops {
            match op {
                &CompositionOp::Filter(filter_op) => {
                    match filter_op {
                        LowLevelFilterOp::Opacity(opacity) => {
                            if opacity == Au(0) {
                                return false
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        true
    }

    fn composite_kind(&self) -> CompositeKind {
        if self.composition_ops.is_empty() {
            return CompositeKind::None;
        }

        if self.composition_ops.len() == 1 {
            match self.composition_ops.first().unwrap() {
                &CompositionOp::Filter(filter_op) => {
                    match filter_op {
                        LowLevelFilterOp::Opacity(opacity) => {
                            let opacity = opacity.to_f32_px();
                            if opacity == 1.0 {
                                return CompositeKind::None;
                            } else {
                                return CompositeKind::Simple(opacity);
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        let info = PackedCompositeInfo::new(&self.composition_ops);
        CompositeKind::Complex(info)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct ClipIndex(usize);

#[derive(Debug, Clone)]
pub struct ClipCorner {
    rect: Rect<f32>,
    outer_radius_x: f32,
    outer_radius_y: f32,
    inner_radius_x: f32,
    inner_radius_y: f32,
}

impl ClipCorner {
    fn invalid(rect: Rect<f32>) -> ClipCorner {
        ClipCorner {
            rect: rect,
            outer_radius_x: 0.0,
            outer_radius_y: 0.0,
            inner_radius_x: 0.0,
            inner_radius_y: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Clip {
    rect: Rect<f32>,
    top_left: ClipCorner,
    top_right: ClipCorner,
    bottom_left: ClipCorner,
    bottom_right: ClipCorner,
}

#[derive(Debug, Clone)]
pub struct ClearTile {
    pub rect: Rect<DevicePixel>,
}

#[derive(Clone, Copy)]
pub struct FrameBuilderConfig {
    max_prim_layers: usize,
    max_prim_tiles: usize,
}

impl FrameBuilderConfig {
    pub fn new(max_prim_layers: usize,
               max_prim_tiles: usize) -> FrameBuilderConfig {
        FrameBuilderConfig {
            max_prim_layers: max_prim_layers,
            max_prim_tiles: max_prim_tiles,
        }
    }
}

pub struct FrameBuilder {
    screen_rect: Rect<i32>,
    prim_store: Vec<Primitive>,
    layer_store: Vec<StackingContext>,
    layer_stack: Vec<StackingContextIndex>,
    device_pixel_ratio: f32,
    debug: bool,
    config: FrameBuilderConfig,
}

pub struct Frame {
    pub debug_rects: Vec<DebugRect>,
    pub cache_size: Size2D<f32>,
    pub phases: Vec<RenderPhase>,
    pub clear_tiles: Vec<ClearTile>,
}

impl Clip {
    pub fn from_clip_region(clip: &ComplexClipRegion) -> Clip {
        Clip {
            rect: clip.rect,
            top_left: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x, clip.rect.origin.y),
                                Size2D::new(clip.radii.top_left.width, clip.radii.top_left.height)),
                outer_radius_x: clip.radii.top_left.width,
                outer_radius_y: clip.radii.top_left.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            top_right: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x + clip.rect.size.width - clip.radii.top_right.width,
                                             clip.rect.origin.y),
                                Size2D::new(clip.radii.top_right.width, clip.radii.top_right.height)),
                outer_radius_x: clip.radii.top_right.width,
                outer_radius_y: clip.radii.top_right.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            bottom_left: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x,
                                             clip.rect.origin.y + clip.rect.size.height - clip.radii.bottom_left.height),
                                Size2D::new(clip.radii.bottom_left.width, clip.radii.bottom_left.height)),
                outer_radius_x: clip.radii.bottom_left.width,
                outer_radius_y: clip.radii.bottom_left.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            bottom_right: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x + clip.rect.size.width - clip.radii.bottom_right.width,
                                             clip.rect.origin.y + clip.rect.size.height - clip.radii.bottom_right.height),
                                Size2D::new(clip.radii.bottom_right.width, clip.radii.bottom_right.height)),
                outer_radius_x: clip.radii.bottom_right.width,
                outer_radius_y: clip.radii.bottom_right.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
        }
    }

    fn invalid(rect: Rect<f32>) -> Clip {
        Clip {
            rect: rect,
            top_left: ClipCorner::invalid(rect),
            top_right: ClipCorner::invalid(rect),
            bottom_left: ClipCorner::invalid(rect),
            bottom_right: ClipCorner::invalid(rect),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ScreenTileIndex(usize);

#[derive(Debug)]
struct CompiledScreenTile {
    main_render_task: RenderTask,
    required_target_count: usize,
}

impl CompiledScreenTile {
    fn new(main_render_task: RenderTask) -> CompiledScreenTile {
        let mut required_target_count = 0;
        main_render_task.max_depth(0, &mut required_target_count);

        CompiledScreenTile {
            main_render_task: main_render_task,
            required_target_count: required_target_count,
        }
    }
}

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
enum TileCommand {
    PushLayer(StackingContextIndex),
    PopLayer,
    DrawPrimitive(PrimitiveIndex),
}

#[derive(Debug)]
struct ScreenTile {
    rect: Rect<DevicePixel>,
    cmds: Vec<TileCommand>,
    prim_count: usize,
}

impl ScreenTile {
    fn new(rect: Rect<DevicePixel>) -> ScreenTile {
        ScreenTile {
            rect: rect,
            cmds: Vec::new(),
            prim_count: 0,
        }
    }

    #[inline(always)]
    fn push_layer(&mut self, sc_index: StackingContextIndex) {
        self.cmds.push(TileCommand::PushLayer(sc_index));
    }

    #[inline(always)]
    fn push_primitive(&mut self, prim_index: PrimitiveIndex) {
        self.cmds.push(TileCommand::DrawPrimitive(prim_index));
        self.prim_count += 1;
    }

    #[inline(always)]
    fn pop_layer(&mut self, sc_index: StackingContextIndex) {
        let last_cmd = *self.cmds.last().unwrap();
        if last_cmd == TileCommand::PushLayer(sc_index) {
            self.cmds.pop();
        } else {
            self.cmds.push(TileCommand::PopLayer);
        }
    }

    fn compile(self, layer_store: &Vec<StackingContext>) -> Option<CompiledScreenTile> {
        if self.prim_count == 0 {
            return None;
        }

        let mut sc_stack = Vec::new();
        let mut current_task = AlphaRenderTask::new(self.rect);
        let mut alpha_task_stack = Vec::new();

        for cmd in self.cmds {
            match cmd {
                TileCommand::PushLayer(sc_index) => {
                    sc_stack.push(sc_index);

                    let layer = &layer_store[sc_index.0];
                    match layer.composite_kind() {
                        CompositeKind::None => {}
                        CompositeKind::Simple(..) | CompositeKind::Complex(..) => {
                            let prev_task = mem::replace(&mut current_task, AlphaRenderTask::new(self.rect));
                            alpha_task_stack.push(prev_task);
                        }
                    }
                }
                TileCommand::PopLayer => {
                    let sc_index = sc_stack.pop().unwrap();

                    let layer = &layer_store[sc_index.0];
                    match layer.composite_kind() {
                        CompositeKind::None => {}
                        CompositeKind::Simple(opacity) => {
                            let mut prev_task = alpha_task_stack.pop().unwrap();
                            prev_task.items.push(AlphaRenderItem::Blend(prev_task.children.len(),
                                                                        opacity));
                            prev_task.children.push(current_task);
                            current_task = prev_task;
                        }
                        CompositeKind::Complex(info) => {
                            let backdrop = alpha_task_stack.pop().unwrap();

                            let mut composite_task = AlphaRenderTask::new(self.rect);
                            composite_task.children.push(backdrop);
                            composite_task.children.push(current_task);

                            composite_task.items.push(AlphaRenderItem::Composite(info));

                            current_task = composite_task;
                        }
                    }
                }
                TileCommand::DrawPrimitive(prim_index) => {
                    let sc_index = *sc_stack.last().unwrap();
                    current_task.items.push(AlphaRenderItem::Primitive(sc_index, prim_index));
                }
            }
        }

        debug_assert!(alpha_task_stack.is_empty());

        let task = RenderTask::from_primitives(current_task,
                                               RenderTaskLocation::Fixed(self.rect),
                                               self.rect.size);
        Some(CompiledScreenTile::new(task))
    }
}

impl FrameBuilder {
    pub fn new(viewport_size: Size2D<f32>,
               device_pixel_ratio: f32,
               debug: bool,
               config: FrameBuilderConfig) -> FrameBuilder {
        let viewport_size = Size2D::new(viewport_size.width as i32, viewport_size.height as i32);
        FrameBuilder {
            screen_rect: Rect::new(Point2D::zero(), viewport_size),
            layer_store: Vec::new(),
            prim_store: Vec::new(),
            layer_stack: Vec::new(),
            device_pixel_ratio: device_pixel_ratio,
            debug: debug,
            config: config,
        }
    }

    fn add_primitive(&mut self,
                     rect: &Rect<f32>,
                     clip_rect: &Rect<f32>,
                     clip: Option<Box<Clip>>,
                     details: PrimitiveDetails) {
        let current_layer = *self.layer_stack.last().unwrap();
        let StackingContextIndex(layer_index) = current_layer;
        let layer = &mut self.layer_store[layer_index as usize];

        let prim = Primitive {
            rect: *rect,
            complex_clip: clip,
            local_clip_rect: *clip_rect,
            details: details,
            xf_rect: None,
        };
        let prim_index = self.prim_store.len();
        self.prim_store.push(prim);

        layer.items.push(StackingContextItem::Primitive(PrimitiveIndex(prim_index)));
    }

    pub fn push_layer(&mut self,
                      rect: Rect<f32>,
                      clip_rect: Rect<f32>,
                      transform: Matrix4D<f32>,
                      pipeline_id: PipelineId,
                      scroll_layer_id: ScrollLayerId,
                      offset: Point2D<f32>,
                      composition_operations: Vec<CompositionOp>) {
        let sc_index = StackingContextIndex(self.layer_store.len());

        let sc = StackingContext {
            items: Vec::new(),
            local_rect: rect,
            local_transform: transform,
            local_offset: offset,
            scroll_layer_id: scroll_layer_id,
            pipeline_id: pipeline_id,
            xf_rect: None,
            transform: Matrix4D::identity(),
            is_valid: false,
            composition_ops: composition_operations,
            local_clip_rect: clip_rect,
            world_clip_rect: None,
            parent: self.layer_stack.last().map(|index| *index),
        };
        self.layer_store.push(sc);

        if !self.layer_stack.is_empty() {
            let current_layer = *self.layer_stack.last().unwrap();
            let StackingContextIndex(layer_index) = current_layer;
            let layer = &mut self.layer_store[layer_index as usize];
            layer.items.push(StackingContextItem::StackingContext(sc_index));
        }

        self.layer_stack.push(sc_index);
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
    }

    pub fn add_solid_rectangle(&mut self,
                               rect: &Rect<f32>,
                               clip_rect: &Rect<f32>,
                               clip: Option<Box<Clip>>,
                               color: &ColorF) {
        if color.a == 0.0 {
            return;
        }

        let prim = RectanglePrimitive {
            color: *color,
        };

        self.add_primitive(rect,
                           clip_rect,
                           clip,
                           PrimitiveDetails::Rectangle(prim));
    }

    pub fn supported_style(&mut self, border: &BorderSide) -> bool {
        match border.style {
            BorderStyle::Solid |
            BorderStyle::None |
            BorderStyle::Dotted |
            BorderStyle::Dashed |
            BorderStyle::Inset |
            BorderStyle::Outset => {
                return true;
            }
            _ => {
                println!("TODO: Other border styles {:?}", border.style);
                return false;
            }
        }
    }

    pub fn add_border(&mut self,
                      rect: Rect<f32>,
                      clip_rect: &Rect<f32>,
                      clip: Option<Box<Clip>>,
                      border: &BorderDisplayItem) {
        let radius = &border.radius;
        let left = &border.left;
        let right = &border.right;
        let top = &border.top;
        let bottom = &border.bottom;

        if !self.supported_style(left) || !self.supported_style(right) ||
           !self.supported_style(top) || !self.supported_style(bottom) {
            println!("Unsupported border style, not rendering border");
            return;
        }

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

        // These colors are used during inset/outset scaling.
        let left_color      = left.border_color(1.0, 2.0/3.0, 0.3, 0.7);
        let top_color       = top.border_color(1.0, 2.0/3.0, 0.3, 0.7);
        let right_color     = right.border_color(2.0/3.0, 1.0, 0.7, 0.3);
        let bottom_color    = bottom.border_color(2.0/3.0, 1.0, 0.7, 0.3);

        let prim = BorderPrimitive {
            tl_outer: tl_outer,
            tl_inner: tl_inner,
            tr_outer: tr_outer,
            tr_inner: tr_inner,
            bl_outer: bl_outer,
            bl_inner: bl_inner,
            br_outer: br_outer,
            br_inner: br_inner,
            radius: radius.clone(),
            left_width: left.width,
            top_width: top.width,
            bottom_width: bottom.width,
            right_width: right.width,
            left_color: left_color,
            top_color: top_color,
            bottom_color: bottom_color,
            right_color: right_color,
            left_style: left.style,
            top_style: top.style,
            right_style: right.style,
            bottom_style: bottom.style,
        };

        self.add_primitive(&rect,
                           clip_rect,
                           clip,
                           PrimitiveDetails::Border(prim));
    }

    pub fn add_gradient(&mut self,
                        rect: Rect<f32>,
                        clip_rect: &Rect<f32>,
                        clip: Option<Box<Clip>>,
                        start_point: Point2D<f32>,
                        end_point: Point2D<f32>,
                        stops: ItemRange) {
        // Fast paths for axis-aligned gradients:
        if start_point.x == end_point.x {
            let prim = GradientPrimitive {
                stops_range: stops,
                kind: GradientType::Vertical,
                start_point: start_point,
                end_point: end_point,
            };
            self.add_primitive(&rect,
                               clip_rect,
                               clip,
                               PrimitiveDetails::Gradient(prim));
        } else if start_point.y == end_point.y {
            let prim = GradientPrimitive {
                stops_range: stops,
                kind: GradientType::Horizontal,
                start_point: start_point,
                end_point: end_point,
            };
            self.add_primitive(&rect,
                               clip_rect,
                               clip,
                               PrimitiveDetails::Gradient(prim));
        } else {
            let prim = GradientPrimitive {
                stops_range: stops,
                kind: GradientType::Rotated,
                start_point: start_point,
                end_point: end_point,
            };
            self.add_primitive(&rect,
                               clip_rect,
                               clip,
                               PrimitiveDetails::Gradient(prim));
        }
    }

    pub fn add_text(&mut self,
                    rect: Rect<f32>,
                    clip_rect: &Rect<f32>,
                    clip: Option<Box<Clip>>,
                    font_key: FontKey,
                    size: Au,
                    blur_radius: Au,
                    color: &ColorF,
                    glyph_range: ItemRange) {
        if color.a == 0.0 {
            return
        }

        let prim = TextPrimitive {
            color: *color,
            font_key: font_key,
            size: size,
            blur_radius: blur_radius,
            glyph_range: glyph_range,
        };

        self.add_primitive(&rect,
                           clip_rect,
                           clip,
                           PrimitiveDetails::Text(prim));
    }

    pub fn add_box_shadow(&mut self,
                          box_bounds: &Rect<f32>,
                          clip_rect: &Rect<f32>,
                          clip: Option<Box<Clip>>,
                          box_offset: &Point2D<f32>,
                          color: &ColorF,
                          blur_radius: f32,
                          spread_radius: f32,
                          border_radius: f32,
                          clip_mode: BoxShadowClipMode) {
        if color.a == 0.0 {
            return
        }

        // Fast path.
        if blur_radius == 0.0 && spread_radius == 0.0 && clip_mode == BoxShadowClipMode::None {
            self.add_solid_rectangle(&box_bounds,
                                     clip_rect,
                                     None,
                                     color);
            return;
        }

        let bs_rect = compute_box_shadow_rect(box_bounds,
                                              box_offset,
                                              spread_radius);

        let metrics = BoxShadowMetrics::new(&bs_rect,
                                            border_radius,
                                            blur_radius);

        let prim_rect = match clip_mode {
            BoxShadowClipMode::Outset | BoxShadowClipMode::None => {
                 Rect::new(metrics.tl_outer,
                           Size2D::new(metrics.br_outer.x - metrics.tl_outer.x,
                                       metrics.br_outer.y - metrics.tl_outer.y))
            }
            BoxShadowClipMode::Inset => {
                *box_bounds
            }
        };

        let prim = BoxShadowPrimitive {
            metrics: metrics,
            src_rect: *box_bounds,
            bs_rect: bs_rect,
            color: *color,
            blur_radius: blur_radius,
            spread_radius: spread_radius,
            border_radius: border_radius,
            clip_mode: clip_mode,
        };

        self.add_primitive(&prim_rect,
                           clip_rect,
                           clip,
                           PrimitiveDetails::BoxShadow(prim));
    }

    pub fn add_webgl_rectangle(&mut self,
                               rect: Rect<f32>,
                               clip_rect: &Rect<f32>,
                               clip: Option<Box<Clip>>,
                               context_id: WebGLContextId) {
        let prim = ImagePrimitive {
            kind: ImagePrimitiveKind::WebGL(context_id),
        };

        self.add_primitive(&rect,
                           clip_rect,
                           clip,
                           PrimitiveDetails::Image(prim));
    }

    pub fn add_image(&mut self,
                     rect: Rect<f32>,
                     clip_rect: &Rect<f32>,
                     clip: Option<Box<Clip>>,
                     stretch_size: &Size2D<f32>,
                     image_key: ImageKey,
                     image_rendering: ImageRendering) {
        let prim = ImagePrimitive {
            kind: ImagePrimitiveKind::Image(image_key,
                                            image_rendering,
                                            stretch_size.clone()),
        };

        self.add_primitive(&rect,
                           clip_rect,
                           clip,
                           PrimitiveDetails::Image(prim));
    }

    fn cull_layers(&mut self,
                   screen_rect: &Rect<DevicePixel>,
                   layer_map: &HashMap<ScrollLayerId, Layer, BuildHasherDefault<FnvHasher>>) {
        // Remove layers that are transparent.

        // Build layer screen rects.
        // TODO(gw): This can be done earlier once update_layer_transforms() is fixed.
        for layer_index in 0..self.layer_store.len() {
            let parent_index = self.layer_store[layer_index].parent;
            let parent_clip_rect = parent_index.map_or(Some(*screen_rect), |parent_index| {
                self.layer_store[parent_index.0].world_clip_rect
            });
            if parent_clip_rect.is_none() {
                continue;
            }
            let layer = &mut self.layer_store[layer_index];

            if layer.can_contribute_to_scene() {
                let scroll_layer = &layer_map[&layer.scroll_layer_id];
                let offset_transform = Matrix4D::identity().translate(layer.local_offset.x,
                                                                      layer.local_offset.y,
                                                                      0.0);
                let transform = scroll_layer.world_transform
                                            .as_ref()
                                            .unwrap()
                                            .mul(&layer.local_transform)
                                            .mul(&offset_transform);
                layer.transform = transform;
                layer.xf_rect = Some(TransformedRect::new(&layer.local_rect,
                                                          &transform,
                                                          self.device_pixel_ratio));

                let world_clip_rect = TransformedRect::new(&layer.local_clip_rect,
                                                           &transform,
                                                           self.device_pixel_ratio);

                // TODO(gw): This gets the iframe reftests passing but is questionable.
                //           Need to refactor the whole layer viewport_rect code once
                //           WR2 lands since it can be simplified now.
                let origin = Point2D::new(DevicePixel::new(scroll_layer.viewport_rect.origin.x,
                                                           self.device_pixel_ratio),
                                          DevicePixel::new(scroll_layer.viewport_rect.origin.y,
                                                           self.device_pixel_ratio));
                let size = Size2D::new(DevicePixel::new(scroll_layer.viewport_rect.size.width,
                                                        self.device_pixel_ratio),
                                       DevicePixel::new(scroll_layer.viewport_rect.size.height,
                                                        self.device_pixel_ratio));
                let viewport_rect = Rect::new(origin, size);

                layer.world_clip_rect = world_clip_rect.bounding_rect
                                                       .intersection(&parent_clip_rect.unwrap())
                                                       .and_then(|cr| {
                                                         cr.intersection(&viewport_rect)
                                                       });

                if layer.world_clip_rect.is_some() {
                    if layer.xf_rect
                            .as_ref()
                            .unwrap()
                            .bounding_rect
                            .intersects(&screen_rect) {

                        for item in &mut layer.items {
                            match item {
                                &mut StackingContextItem::StackingContext(..) => {
                                    // TODO(gw): Worth removing these to reduce cmd list size?
                                }
                                &mut StackingContextItem::Primitive(prim_index) => {
                                    let prim = &mut self.prim_store[prim_index.0];

                                    let xf_rect = TransformedRect::new(&prim.rect,
                                                                       &layer.transform,
                                                                       self.device_pixel_ratio);

                                    if xf_rect.bounding_rect.intersects(&screen_rect) {
                                        prim.xf_rect = Some(xf_rect);
                                    }
                                }
                            }
                        }

                        layer.is_valid = true;
                    }
                }
            }
        }
    }

    fn create_screen_tiles(&self) -> (i32, i32, Vec<ScreenTile>) {
        let dp_size = Size2D::new(DevicePixel::new(self.screen_rect.size.width as f32,
                                                   self.device_pixel_ratio),
                                  DevicePixel::new(self.screen_rect.size.height as f32,
                                                   self.device_pixel_ratio));

        let x_tile_size = DevicePixel(SCREEN_TILE_SIZE);
        let y_tile_size = DevicePixel(SCREEN_TILE_SIZE);
        let x_tile_count = (dp_size.width + x_tile_size - DevicePixel(1)).0 / x_tile_size.0;
        let y_tile_count = (dp_size.height + y_tile_size - DevicePixel(1)).0 / y_tile_size.0;

        // Build screen space tiles, which are individual BSP trees.
        let mut screen_tiles = Vec::new();
        for y in 0..y_tile_count {
            let y0 = DevicePixel(y * y_tile_size.0);
            let y1 = y0 + y_tile_size;

            for x in 0..x_tile_count {
                let x0 = DevicePixel(x * x_tile_size.0);
                let x1 = x0 + x_tile_size;

                let tile_rect = rect_from_points(x0, y0, x1, y1);

                screen_tiles.push(ScreenTile::new(tile_rect));
            }
        }

        (x_tile_count, y_tile_count, screen_tiles)
    }


    fn assign_prims_to_screen_tiles(&self,
                                    stacking_context_index: StackingContextIndex,
                                    x_tile_count: i32,
                                    y_tile_count: i32,
                                    screen_tiles: &mut Vec<ScreenTile>) {
        let layer = &self.layer_store[stacking_context_index.0];
        if !layer.is_valid {
            return;
        }

        let l_rect = &layer.xf_rect.as_ref().unwrap().bounding_rect;

        let l_tile_x0 = l_rect.origin.x.0 / SCREEN_TILE_SIZE;
        let l_tile_y0 = l_rect.origin.y.0 / SCREEN_TILE_SIZE;
        let l_tile_x1 = (l_rect.origin.x.0 + l_rect.size.width.0 + SCREEN_TILE_SIZE - 1) / SCREEN_TILE_SIZE;
        let l_tile_y1 = (l_rect.origin.y.0 + l_rect.size.height.0 + SCREEN_TILE_SIZE - 1) / SCREEN_TILE_SIZE;

        let l_tile_x0 = cmp::min(l_tile_x0, x_tile_count);
        let l_tile_x0 = cmp::max(l_tile_x0, 0);
        let l_tile_x1 = cmp::min(l_tile_x1, x_tile_count);
        let l_tile_x1 = cmp::max(l_tile_x1, 0);

        let l_tile_y0 = cmp::min(l_tile_y0, y_tile_count);
        let l_tile_y0 = cmp::max(l_tile_y0, 0);
        let l_tile_y1 = cmp::min(l_tile_y1, y_tile_count);
        let l_tile_y1 = cmp::max(l_tile_y1, 0);

        for ly in l_tile_y0..l_tile_y1 {
            for lx in l_tile_x0..l_tile_x1 {
                let tile = &mut screen_tiles[(ly * x_tile_count + lx) as usize];
                tile.push_layer(stacking_context_index);
            }
        }

        for item in &layer.items {
            match item {
                &StackingContextItem::StackingContext(sc_index) => {
                    self.assign_prims_to_screen_tiles(sc_index,
                                                      x_tile_count,
                                                      y_tile_count,
                                                      screen_tiles);
                }
                &StackingContextItem::Primitive(prim_index) => {
                    let prim = &self.prim_store[prim_index.0];

                    if let Some(ref p_rect) = prim.xf_rect {
                        let p_rect = &p_rect.bounding_rect;

                        // TODO(gw): Ensure that certain primitives (such as background-image) only get
                        //           assigned to tiles where their containing layer intersects with.
                        //           Does this cause any problems / demonstrate other bugs?
                        //           Restrict the tiles by clamping to the layer tile indices...

                        let p_tile_x0 = p_rect.origin.x.0 / SCREEN_TILE_SIZE;
                        let p_tile_y0 = p_rect.origin.y.0 / SCREEN_TILE_SIZE;
                        let p_tile_x1 = (p_rect.origin.x.0 + p_rect.size.width.0 + SCREEN_TILE_SIZE - 1) / SCREEN_TILE_SIZE;
                        let p_tile_y1 = (p_rect.origin.y.0 + p_rect.size.height.0 + SCREEN_TILE_SIZE - 1) / SCREEN_TILE_SIZE;

                        let p_tile_x0 = cmp::min(p_tile_x0, l_tile_x1);
                        let p_tile_x0 = cmp::max(p_tile_x0, l_tile_x0);
                        let p_tile_x1 = cmp::min(p_tile_x1, l_tile_x1);
                        let p_tile_x1 = cmp::max(p_tile_x1, l_tile_x0);

                        let p_tile_y0 = cmp::min(p_tile_y0, l_tile_y1);
                        let p_tile_y0 = cmp::max(p_tile_y0, l_tile_y0);
                        let p_tile_y1 = cmp::min(p_tile_y1, l_tile_y1);
                        let p_tile_y1 = cmp::max(p_tile_y1, l_tile_y0);

                        for py in p_tile_y0..p_tile_y1 {
                            for px in p_tile_x0..p_tile_x1 {
                                let tile = &mut screen_tiles[(py * x_tile_count + px) as usize];
                                tile.push_primitive(prim_index);
                            }
                        }
                    }
                }
            }
        }

        for ly in l_tile_y0..l_tile_y1 {
            for lx in l_tile_x0..l_tile_x1 {
                let tile = &mut screen_tiles[(ly * x_tile_count + lx) as usize];
                tile.pop_layer(stacking_context_index);
            }
        }
    }

    fn build_resource_list(&mut self,
                           resource_cache: &mut ResourceCache,
                           frame_id: FrameId,
                           pipeline_auxiliary_lists: &HashMap<PipelineId, AuxiliaryLists, BuildHasherDefault<FnvHasher>>) {
        let mut resource_list = ResourceList::new();

        // Non-visible layers have been marked invalid by now
        for layer in &self.layer_store {
            if layer.is_valid {
                let auxiliary_lists = pipeline_auxiliary_lists.get(&layer.pipeline_id)
                                                              .expect("No auxiliary lists?!");

                // Non-visible chunks have also been removed by now
                layer.build_resource_list(&mut resource_list,
                                          auxiliary_lists,
                                          &self.prim_store);
            }
        }

        resource_cache.add_resource_list(&resource_list,
                                         frame_id);
        resource_cache.raster_pending_glyphs(frame_id);
    }

    fn pack_primitives_for_layer(
            &self,
            stacking_context_index: StackingContextIndex,
            packed_primitive_cache: &mut PackedPrimitiveCache,
            pipeline_auxiliary_lists: &HashMap<PipelineId,
                                               AuxiliaryLists,
                                               BuildHasherDefault<FnvHasher>>,
            render_target_context: &RenderTargetContext) {
        let layer = &self.layer_store[stacking_context_index.0];
        if !layer.is_valid {
            return;
        }

        for item in &layer.items {
            match item {
                &StackingContextItem::StackingContext(sc_index) => {
                    self.pack_primitives_for_layer(sc_index,
                                                   packed_primitive_cache,
                                                   pipeline_auxiliary_lists,
                                                   render_target_context)
                }
                &StackingContextItem::Primitive(prim_index) => {
                    let prim = &self.prim_store[prim_index.0];
                    let auxiliary_lists = pipeline_auxiliary_lists.get(&layer.pipeline_id)
                                                                  .expect("No auxiliary lists?!");
                    prim.pack(prim_index,
                              packed_primitive_cache,
                              auxiliary_lists,
                              render_target_context)
                }
            }
        }
    }

    fn pack_primitives(&self,
                       pipeline_auxiliary_lists: &HashMap<PipelineId,
                                                          AuxiliaryLists,
                                                          BuildHasherDefault<FnvHasher>>,
                       render_target_context: &RenderTargetContext)
                       -> PackedPrimitiveCache {
        let mut packed_primitive_cache = PackedPrimitiveCache::new();
        if !self.layer_store.is_empty() {
            self.pack_primitives_for_layer(StackingContextIndex(0),
                                           &mut packed_primitive_cache,
                                           pipeline_auxiliary_lists,
                                           render_target_context);
        }
        packed_primitive_cache
    }

    pub fn build(&mut self,
                 resource_cache: &mut ResourceCache,
                 frame_id: FrameId,
                 pipeline_auxiliary_lists: &HashMap<PipelineId, AuxiliaryLists, BuildHasherDefault<FnvHasher>>,
                 layer_map: &HashMap<ScrollLayerId, Layer, BuildHasherDefault<FnvHasher>>) -> Frame {
        let screen_rect = Rect::new(Point2D::zero(),
                                    Size2D::new(DevicePixel::new(self.screen_rect.size.width as f32, self.device_pixel_ratio),
                                                DevicePixel::new(self.screen_rect.size.height as f32, self.device_pixel_ratio)));

        self.cull_layers(&screen_rect, layer_map);

        let mut debug_rects = Vec::new();

        self.build_resource_list(resource_cache, frame_id, pipeline_auxiliary_lists);
        let (x_tile_count, y_tile_count, mut screen_tiles) = self.create_screen_tiles();

        let ctx = RenderTargetContext {
            layer_store: &self.layer_store,
            prim_store: &self.prim_store,
            resource_cache: resource_cache,
            device_pixel_ratio: self.device_pixel_ratio,
            frame_id: frame_id,
            alpha_batch_max_layers: self.config.max_prim_layers,
            alpha_batch_max_tiles: self.config.max_prim_tiles,
        };

        let packed_primitive_cache = self.pack_primitives(pipeline_auxiliary_lists, &ctx);
        if !self.layer_store.is_empty() {
            let root_sc_index = StackingContextIndex(0);
            self.assign_prims_to_screen_tiles(root_sc_index,
                                              x_tile_count,
                                              y_tile_count,
                                              &mut screen_tiles);
        }

        if self.debug {
            for r in &screen_tiles {
                debug_rects.push(DebugRect {
                    label: format!("{}|{}", r.cmds.len(), r.prim_count),
                    color: ColorF::new(1.0, 0.0, 0.0, 1.0),
                    rect: r.rect,
                });
            }
        }

        let mut clear_tiles = Vec::new();

        // Build list of passes, target allocs that each tile needs.
        let mut compiled_screen_tiles = Vec::new();
        for screen_tile in screen_tiles {
            let rect = screen_tile.rect;        // TODO(gw): Remove clone here
            match screen_tile.compile(&self.layer_store) {
                Some(compiled_screen_tile) => {
                    compiled_screen_tiles.push(compiled_screen_tile);
                }
                None => {
                    clear_tiles.push(ClearTile {
                        rect: rect,
                    });
                }
            }
        }

        let mut phases = Vec::new();

        if !compiled_screen_tiles.is_empty() {
            // Sort by pass count to minimize render target switches.
            compiled_screen_tiles.sort_by(|a, b| {
                let a_passes = a.required_target_count;
                let b_passes = b.required_target_count;
                b_passes.cmp(&a_passes)
            });

            // Do the allocations now, assigning each tile to a render
            // phase as required.

            let mut current_phase = RenderPhase::new(compiled_screen_tiles[0].required_target_count);

            for compiled_screen_tile in compiled_screen_tiles {
                if let Some(failed_tile) = current_phase.add_compiled_screen_tile(compiled_screen_tile) {
                    let full_phase = mem::replace(&mut current_phase,
                                                  RenderPhase::new(failed_tile.required_target_count));
                    phases.push(full_phase);

                    let result = current_phase.add_compiled_screen_tile(failed_tile);
                    assert!(result.is_none(), "TODO: Handle single tile not fitting in render phase.");
                }
            }

            phases.push(current_phase);

            for phase in &mut phases {
                phase.build(&packed_primitive_cache, &ctx);
            }
        }

        Frame {
            debug_rects: debug_rects,
            phases: phases,
            clear_tiles: clear_tiles,
            cache_size: Size2D::new(RENDERABLE_CACHE_SIZE.0 as f32,
                                    RENDERABLE_CACHE_SIZE.0 as f32),
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

#[derive(Clone, Copy, PartialEq, Debug)]
struct PackedPrimitiveMetadata {
    start: usize,
    end: usize,
    texture_id: TextureId,
}

impl PackedPrimitiveMetadata {
    fn new(start: usize, end: usize, texture_id: TextureId) -> PackedPrimitiveMetadata {
        PackedPrimitiveMetadata {
            start: start,
            end: end,
            texture_id: texture_id,
        }
    }

    fn none() -> PackedPrimitiveMetadata {
        PackedPrimitiveMetadata::new(0, 0, TextureId(0))
    }
}

/// To find the packed primitives for a primitive index, first look up the metadata within the
/// `metadata` field, and then consult the indices defined by the range specified by that metadata
/// in the `primitives` field.
struct PackedPrimitiveCache {
    /// A mapping from primitive index to range in the `primitives` vector below.
    metadata: Vec<PackedPrimitiveMetadata>,
    /// A list of packed primitives.
    primitives: Vec<PackedPrimitive>,
}

impl PackedPrimitiveCache {
    fn new() -> PackedPrimitiveCache {
        PackedPrimitiveCache {
            metadata: vec![],
            primitives: vec![],
        }
    }

    /// Reserves space for the packed primitive with the given index.
    ///
    /// This must be called before `add_packed_primitive` below.
    fn init_packed_primitive(&mut self, primitive_index: PrimitiveIndex) {
        while self.metadata.len() < primitive_index.0 + 1 {
            self.metadata.push(PackedPrimitiveMetadata::none())
        }
    }

    fn add_packed_primitive(&mut self,
                            primitive_index: PrimitiveIndex,
                            packed_primitive: PackedPrimitive,
                            texture_id: TextureId) {
        let mut metadata = &mut self.metadata[primitive_index.0];
        if *metadata == PackedPrimitiveMetadata::none() {
            metadata.start = self.primitives.len();
        } else {
            debug_assert!(metadata.end == self.primitives.len());
        }
        metadata.end = self.primitives.len() + 1;
        metadata.texture_id = texture_id;

        self.primitives.push(packed_primitive)
    }

    fn add_to_batch(&self,
                    primitive_index: PrimitiveIndex,
                    batch: &mut PrimitiveBatch,
                    layer_index_in_ubo: u32,
                    tile_index_in_ubo: u32,
                    transform_kind: TransformedRectKind,
                    needs_blending: bool)
                    -> bool {
        if transform_kind != batch.transform_kind ||
           needs_blending != batch.blending_enabled {
            return false
        }

        // TODO(gw): Tidy the support for batch breaks up...
        let metadata = self.metadata[primitive_index.0];
        if metadata.texture_id != TextureId(0) {
            if batch.color_texture_id != TextureId(0) &&
                    batch.color_texture_id != metadata.texture_id {
                return false
            }
            batch.color_texture_id = metadata.texture_id;
        }

        for packed_primitive_index in metadata.start..metadata.end {
            match (&mut batch.data, &self.primitives[packed_primitive_index]) {
                (&mut PrimitiveBatchData::Blend(..), _) => return false,
                (&mut PrimitiveBatchData::Composite(..), _) => return false,
                (&mut PrimitiveBatchData::Rectangles(ref mut data),
                 &PackedPrimitive::Rectangle(ref rectangle)) => {
                    // FIXME(pcwalton): Don't clone here!
                    // TODO(pcwalton): Check rect intersection.
                    let mut rectangle = (*rectangle).clone();
                    rectangle.common.tile_index = tile_index_in_ubo;
                    rectangle.common.layer_index = layer_index_in_ubo;
                    data.push(rectangle)
                }
                (&mut PrimitiveBatchData::Rectangles(..), _) => return false,
                (&mut PrimitiveBatchData::RectanglesClip(ref mut data),
                 &PackedPrimitive::RectangleClip(ref rectangle)) => {
                    // FIXME(pcwalton): Don't clone here!
                    // TODO(pcwalton): Check rect intersection.
                    let mut rectangle = (*rectangle).clone();
                    rectangle.common.tile_index = tile_index_in_ubo;
                    rectangle.common.layer_index = layer_index_in_ubo;
                    data.push(rectangle)
                }
                (&mut PrimitiveBatchData::RectanglesClip(..), _) => return false,
                (&mut PrimitiveBatchData::Image(ref mut data),
                 &PackedPrimitive::Image(ref image)) => {
                    // FIXME(pcwalton): Don't clone here!
                    let mut image = (*image).clone();
                    image.common.tile_index = tile_index_in_ubo;
                    image.common.layer_index = layer_index_in_ubo;
                    data.push(image)
                }
                (&mut PrimitiveBatchData::Image(..), _) => return false,
                (&mut PrimitiveBatchData::ImageClip(ref mut data),
                 &PackedPrimitive::ImageClip(ref image)) => {
                    // FIXME(pcwalton): Don't clone here!
                    // TODO(pcwalton): Check rect intersection.
                    let mut image = (*image).clone();
                    image.common.tile_index = tile_index_in_ubo;
                    image.common.layer_index = layer_index_in_ubo;
                    data.push(image)
                }
                (&mut PrimitiveBatchData::ImageClip(..), _) => return false,
                (&mut PrimitiveBatchData::Borders(ref mut data),
                 &PackedPrimitive::Border(ref border)) => {
                    // FIXME(pcwalton): Don't clone here!
                    // TODO(pcwalton): Check rect intersection.
                    let mut border = (*border).clone();
                    border.common.tile_index = tile_index_in_ubo;
                    border.common.layer_index = layer_index_in_ubo;
                    data.push(border)
                }
                (&mut PrimitiveBatchData::Borders(..), _) => return false,
                (&mut PrimitiveBatchData::AlignedGradient(ref mut data),
                 &PackedPrimitive::AlignedGradient(ref gradient)) => {
                    // FIXME(pcwalton): Don't clone here!
                    // TODO(pcwalton): Check rect intersection.
                    let mut gradient = (*gradient).clone();
                    gradient.common.tile_index = tile_index_in_ubo;
                    gradient.common.layer_index = layer_index_in_ubo;
                    data.push(gradient)
                }
                (&mut PrimitiveBatchData::AlignedGradient(..), _) => return false,
                (&mut PrimitiveBatchData::AngleGradient(ref mut data),
                 &PackedPrimitive::AngleGradient(ref gradient)) => {
                    // FIXME(pcwalton): Don't clone here!
                    let mut gradient = (*gradient).clone();
                    gradient.common.tile_index = tile_index_in_ubo;
                    gradient.common.layer_index = layer_index_in_ubo;
                    data.push(gradient)
                }
                (&mut PrimitiveBatchData::AngleGradient(..), _) => return false,
                (&mut PrimitiveBatchData::BoxShadows(ref mut data),
                 &PackedPrimitive::BoxShadow(ref shadow)) => {
                    // FIXME(pcwalton): Don't clone here!
                    // TODO(pcwalton): Check rect intersection.
                    let mut shadow = (*shadow).clone();
                    shadow.common.tile_index = tile_index_in_ubo;
                    shadow.common.layer_index = layer_index_in_ubo;
                    data.push(shadow)
                }
                (&mut PrimitiveBatchData::BoxShadows(..), _) => return false,
                (&mut PrimitiveBatchData::Text(ref mut data),
                 &PackedPrimitive::Text(ref glyph)) => {
                    // FIXME(pcwalton): Don't clone here!
                    // TODO(pcwalton): Check rect intersection. Binary search to find the start
                    // point, maybe?
                    let mut glyph = (*glyph).clone();
                    glyph.common.tile_index = tile_index_in_ubo;
                    glyph.common.layer_index = layer_index_in_ubo;
                    data.push(glyph)
                }
                (&mut PrimitiveBatchData::Text(..), _) => return false,
            }
        }
        true
    }
}

//Test for one clip region contains another
pub trait InsideTest<T> {
    fn might_contain(&self, clip: &T) -> bool;
}

impl InsideTest<ComplexClipRegion> for ComplexClipRegion {
    // Returns true if clip is inside self, can return false negative
    fn might_contain(&self, clip: &ComplexClipRegion) -> bool {
        let delta_left = clip.rect.origin.x - self.rect.origin.x;
        let delta_top = clip.rect.origin.y - self.rect.origin.y;
        let delta_right = self.rect.max_x() - clip.rect.max_x();
        let delta_bottom = self.rect.max_y() - clip.rect.max_y();

        delta_left >= 0f32 &&
        delta_top >= 0f32 &&
        delta_right >= 0f32 &&
        delta_bottom >= 0f32 &&
        clip.radii.top_left.width >= self.radii.top_left.width - delta_left &&
        clip.radii.top_left.height >= self.radii.top_left.height - delta_top &&
        clip.radii.top_right.width >= self.radii.top_right.width - delta_right &&
        clip.radii.top_right.height >= self.radii.top_right.height - delta_top &&
        clip.radii.bottom_left.width >= self.radii.bottom_left.width - delta_left &&
        clip.radii.bottom_left.height >= self.radii.bottom_left.height - delta_bottom &&
        clip.radii.bottom_right.width >= self.radii.bottom_right.width - delta_right &&
        clip.radii.bottom_right.height >= self.radii.bottom_right.height - delta_bottom
    }
}

