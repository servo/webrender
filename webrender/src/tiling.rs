/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::{Au};
use batch_builder::{BorderSideHelpers, BoxShadowMetrics};
use device::{TextureId};
use euclid::{Point2D, Point4D, Rect, Matrix4D, Size2D};
use fnv::FnvHasher;
use frame::FrameId;
use internal_types::{Glyph, DevicePixel, CompositionOp};
use internal_types::{ANGLE_FLOAT_TO_FIXED, LowLevelFilterOp};
use layer::Layer;
use profiler::FrameProfileCounters;
use renderer::{BLUR_INFLATION_FACTOR};
use resource_cache::ResourceCache;
use resource_list::ResourceList;
use std::cmp;
use std::collections::{HashMap};
use std::f32;
use std::mem;
use std::hash::{BuildHasherDefault};
use std::sync::atomic::{AtomicUsize, Ordering};
use texture_cache::TexturePage;
use util::{self, rect_from_points, rect_from_points_f, MatrixHelpers, subtract_rect};
use webrender_traits::{ColorF, FontKey, GlyphKey, ImageKey, ImageRendering, ComplexClipRegion};
use webrender_traits::{BorderDisplayItem, BorderStyle, ItemRange, AuxiliaryLists, BorderRadius, BorderSide};
use webrender_traits::{BoxShadowClipMode, PipelineId, ScrollLayerId, WebGLContextId};

pub const GLYPHS_PER_TEXT_RUN: usize = 8;
pub const ELEMENTS_PER_BORDER: usize = 8;

const ALPHA_BATCHERS_PER_RENDER_TARGET: usize = 4;
const MIN_TASKS_PER_ALPHA_BATCHER: usize = 64;
const FLOATS_PER_RENDER_TASK_INFO: usize = 8;

#[derive(Debug)]
struct ScrollbarPrimitive {
    scroll_layer_id: ScrollLayerId,
    prim_index: PrimitiveIndex,
    border_radius: f32,
}

#[inline(always)]
fn pack_as_float(value: u32) -> f32 {
    value as f32 + 0.5
}

trait PackRectAsFloat {
    fn pack_as_float(&self) -> Rect<f32>;
}

impl PackRectAsFloat for Rect<DevicePixel> {
    fn pack_as_float(&self) -> Rect<f32> {
        Rect::new(Point2D::new(self.origin.x.0 as f32,
                               self.origin.y.0 as f32),
                  Size2D::new(self.size.width.0 as f32,
                              self.size.height.0 as f32))
    }
}

enum PrimitiveRunCmd {
    PushStackingContext(StackingContextIndex),
    PrimitiveRun(PrimitiveIndex, usize),
    PopStackingContext,
}

#[derive(Debug, Copy, Clone)]
pub enum PrimitiveFlags {
    None,
    Scrollbar(ScrollLayerId, f32)
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum GradientType {
    Horizontal,
    Vertical,
    Rotated,
}

#[derive(Debug, Copy, Clone)]
struct RenderTaskIndex(usize);

#[derive(Debug, Copy, Clone)]
enum RenderTaskId {
    Static(RenderTaskIndex),
    //Dynamic(RenderTaskKey),
}

struct RenderTaskCollection {
    render_task_data: Vec<RenderTaskData>,
}

impl RenderTaskCollection {
    fn new(static_render_task_count: usize) -> RenderTaskCollection {
        RenderTaskCollection {
            render_task_data: vec![RenderTaskData::empty(); static_render_task_count],
        }
    }

    fn add(&mut self, task: &RenderTask) {
        match task.id {
            RenderTaskId::Static(index) => {
                self.render_task_data[index.0] = task.write_task_data();
            }
        }
    }

    fn get_task_index(&self, id: &RenderTaskId) -> RenderTaskIndex {
        match id {
            &RenderTaskId::Static(index) => index,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RenderTaskData {
    data: [f32; FLOATS_PER_RENDER_TASK_INFO],
}

impl RenderTaskData {
    fn empty() -> RenderTaskData {
        RenderTaskData {
            data: unsafe { mem::uninitialized() }
        }
    }
}

impl Default for RenderTaskData {
    fn default() -> RenderTaskData {
        RenderTaskData {
            data: unsafe { mem::uninitialized() },
        }
    }
}

struct AlphaBatchTask {
    task_id: RenderTaskId,
    items: Vec<AlphaRenderItem>,
}

pub struct AlphaBatcher {
    pub batches: Vec<PrimitiveBatch>,
    tasks: Vec<AlphaBatchTask>,
}

impl AlphaBatcher {
    fn new() -> AlphaBatcher {
        AlphaBatcher {
            batches: Vec::new(),
            tasks: Vec::new(),
        }
    }

    fn add_task(&mut self, task: AlphaBatchTask) {
        self.tasks.push(task);
    }

    fn build(&mut self,
             ctx: &RenderTargetContext,
             render_tasks: &mut RenderTaskCollection) {
        let mut batches: Vec<(AlphaBatchKey, PrimitiveBatch)> = vec![];
        for task in &mut self.tasks {
            let task_index = render_tasks.get_task_index(&task.task_id);
            let task_index = pack_as_float(task_index.0 as u32);

            let mut existing_batch_index = 0;
            let items = mem::replace(&mut task.items, vec![]);
            for item in items.into_iter().rev() {
                let batch_key;
                match item {
                    AlphaRenderItem::Composite(..) => {
                        batch_key = AlphaBatchKey::composite();
                    }
                    AlphaRenderItem::Blend(..) => {
                        batch_key = AlphaBatchKey::blend();
                    }
                    AlphaRenderItem::Primitive(sc_index, prim_index) => {
                        // See if this task fits into the tile UBO
                        let layer = &ctx.layer_store[sc_index.0];
                        let prim = &ctx.prim_store[prim_index.0];
                        let transform_kind = layer.xf_rect.as_ref().unwrap().kind;
                        let needs_blending = transform_kind == TransformedRectKind::Complex ||
                                             !prim.is_opaque(ctx.resource_cache, ctx.frame_id);
                        let batch_kind = prim.batch_kind();
                        let color_texture_id = prim.color_texture_id(ctx.resource_cache,
                                                                     ctx.frame_id);
                        let flags = AlphaBatchKeyFlags::new(transform_kind, needs_blending);
                        batch_key = AlphaBatchKey::primitive(batch_kind,
                                                             flags,
                                                             color_texture_id);
                    }
                }

                while existing_batch_index < batches.len() &&
                        !batches[existing_batch_index].0.is_compatible_with(&batch_key) {
                    existing_batch_index += 1
                }

                if existing_batch_index == batches.len() {
                    let new_batch = match item {
                        AlphaRenderItem::Composite(..) => {
                            PrimitiveBatch::composite()
                        }
                        AlphaRenderItem::Blend(..) => {
                            PrimitiveBatch::blend()
                        }
                        AlphaRenderItem::Primitive(_, prim_index) => {
                            // See if this task fits into the tile UBO
                            PrimitiveBatch::new(&ctx.prim_store[prim_index.0],
                                                batch_key.flags.transform_kind(),
                                                batch_key.flags.needs_blending())
                        }
                    };
                    batches.push((batch_key, new_batch))
                }

                let batch = &mut batches[existing_batch_index].1;
                match item {
                    AlphaRenderItem::Composite(src0_id, src1_id, info) => {
                        let ok = batch.pack_composite(render_tasks.get_task_index(&src0_id),
                                                      render_tasks.get_task_index(&src1_id),
                                                      render_tasks.get_task_index(&task.task_id),
                                                      info);
                        debug_assert!(ok)
                    }
                    AlphaRenderItem::Blend(src_id, info) => {
                        let (opacity, brightness) = match info {
                            SimpleCompositeInfo::Opacity(opacity) => (opacity, 1.0),
                            SimpleCompositeInfo::Brightness(brightness) => (1.0, brightness),
                        };
                        let ok = batch.pack_blend(render_tasks.get_task_index(&src_id),
                                                  render_tasks.get_task_index(&task.task_id),
                                                  opacity,
                                                  brightness);
                        debug_assert!(ok)
                    }
                    AlphaRenderItem::Primitive(sc_index, prim_index) => {
                        let prim = &ctx.prim_store[prim_index.0];
                        let ok = prim.add_to_batch(batch,
                                                   sc_index,
                                                   task_index,
                                                   batch_key.flags.transform_kind(),
                                                   batch_key.flags.needs_blending());
                        debug_assert!(ok);

                        let color_texture_id = prim.color_texture_id(ctx.resource_cache,
                                                                     ctx.frame_id);
                        if color_texture_id != TextureId(0) &&
                                batch.color_texture_id != color_texture_id {
                            debug_assert!(batch.color_texture_id == TextureId(0));
                            batch.color_texture_id = color_texture_id
                        }
                    }
                }
            }
        }

        self.batches.extend(batches.into_iter().map(|(_, batch)| batch))
    }
}

struct RenderTargetContext<'a> {
    layer_store: &'a Vec<StackingContext>,
    prim_store: &'a Vec<Primitive>,
    resource_cache: &'a ResourceCache,
    frame_id: FrameId,
    render_task_id_counter: AtomicUsize,
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

    fn build(&mut self,
             ctx: &RenderTargetContext,
             render_tasks: &mut RenderTaskCollection) {
        // Step through each task, adding to batches as appropriate.
        let tasks_per_batcher =
            cmp::max((self.tasks.len() + ALPHA_BATCHERS_PER_RENDER_TARGET - 1) /
                     ALPHA_BATCHERS_PER_RENDER_TARGET,
                     MIN_TASKS_PER_ALPHA_BATCHER);
        for task in self.tasks.drain(..) {
            match task.kind {
                RenderTaskKind::Alpha(info) => {
                    let need_new_batcher =
                        self.alpha_batchers.is_empty() ||
                        self.alpha_batchers.last().unwrap().tasks.len() == tasks_per_batcher;

                    if need_new_batcher {
                        self.alpha_batchers.push(AlphaBatcher::new());
                    }

                    self.alpha_batchers.last_mut().unwrap().add_task(AlphaBatchTask {
                        task_id: task.id,
                        items: info.items,
                    });
                }
            }
        }

        //println!("+ + start render target");
        for ab in &mut self.alpha_batchers {
            ab.build(ctx, render_tasks);
        }
    }
}

pub struct RenderPhase {
    pub targets: Vec<RenderTarget>,
}

impl RenderPhase {
    fn new(max_target_count: usize) -> RenderPhase {
        //println!("+ start render phase: targets={}", max_target_count);
        let mut targets = Vec::with_capacity(max_target_count);
        for index in 0..max_target_count {
            targets.push(RenderTarget::new(index == max_target_count-1));
        }

        RenderPhase {
            targets: targets,
        }
    }

    fn add_compiled_screen_tile(&mut self,
                                mut tile: CompiledScreenTile,
                                render_tasks: &mut RenderTaskCollection) -> Option<CompiledScreenTile> {
        debug_assert!(tile.required_target_count <= self.targets.len());

        let ok = tile.main_render_task.alloc_if_required(self.targets.len() - 1,
                                                         &mut self.targets);

        if ok {
            tile.main_render_task.assign_to_targets(self.targets.len() - 1,
                                                    &mut self.targets,
                                                    render_tasks);
            None
        } else {
            Some(tile)
        }
    }

    fn build(&mut self,
             ctx: &RenderTargetContext,
             render_tasks: &mut RenderTaskCollection) {
        for target in &mut self.targets {
            target.build(ctx, render_tasks);
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
    Blend(RenderTaskId, SimpleCompositeInfo),
    Composite(RenderTaskId, RenderTaskId, PackedCompositeInfo),
}

#[derive(Debug)]
struct AlphaRenderTask {
    actual_rect: Rect<DevicePixel>,
    items: Vec<AlphaRenderItem>,
}

#[derive(Debug)]
enum RenderTaskKind {
    Alpha(AlphaRenderTask),
}

#[derive(Debug)]
struct RenderTask {
    id: RenderTaskId,
    location: RenderTaskLocation,
    children: Vec<RenderTask>,
    kind: RenderTaskKind,
}

impl RenderTask {
    fn new_alpha_batch(actual_rect: Rect<DevicePixel>, ctx: &RenderTargetContext) -> RenderTask {
        let task_index = ctx.render_task_id_counter.fetch_add(1, Ordering::Relaxed);

        RenderTask {
            id: RenderTaskId::Static(RenderTaskIndex(task_index)),
            children: Vec::new(),
            location: RenderTaskLocation::Dynamic(None, actual_rect.size),
            kind: RenderTaskKind::Alpha(AlphaRenderTask {
                actual_rect: actual_rect,
                items: Vec::new(),
            }),
        }
    }

    fn as_alpha_batch<'a>(&'a mut self) -> &'a mut AlphaRenderTask {
        match self.kind {
            RenderTaskKind::Alpha(ref mut task) => task,
        }
    }

    fn write_task_data(&self) -> RenderTaskData {
        match self.kind {
            RenderTaskKind::Alpha(ref task) => {
                let target_rect = self.get_target_rect();

                RenderTaskData {
                    data: [
                        task.actual_rect.origin.x.0 as f32,
                        task.actual_rect.origin.y.0 as f32,
                        task.actual_rect.size.width.0 as f32,
                        task.actual_rect.size.height.0 as f32,
                        target_rect.origin.x.0 as f32,
                        target_rect.origin.y.0 as f32,
                        target_rect.size.width.0 as f32,
                        target_rect.size.height.0 as f32,
                    ],
                }
            }
        }
    }

    fn finalize(&mut self) {
        match self.kind {
            RenderTaskKind::Alpha(ref mut task) => {
                task.items.reverse();
            }
        }
        for child in &mut self.children {
            child.finalize();
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
                         targets: &mut Vec<RenderTarget>,
                         render_tasks: &mut RenderTaskCollection) {
        for child in self.children.drain(..) {
            child.assign_to_targets(target_index - 1,
                                    targets,
                                    render_tasks);
        }

        render_tasks.add(&self);

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
#[repr(u8)]
pub enum TransformedRectKind {
    AxisAligned = 0,
    Complex = 1,
}

#[derive(Debug, Clone)]
pub struct TransformedRect {
    local_rect: Rect<f32>,
    pub bounding_rect: Rect<DevicePixel>,
    vertices: [Point4D<f32>; 4],
    kind: TransformedRectKind,
}

impl TransformedRect {
    pub fn new(rect: &Rect<f32>,
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
struct TextPrimitiveCache {
    color_texture_id: TextureId,
    glyph: Option<PackedGlyphPrimitive>,
}

impl TextPrimitiveCache {
    fn new() -> TextPrimitiveCache {
        TextPrimitiveCache {
            color_texture_id: TextureId(0),
            glyph: None,
        }
    }
}

#[derive(Debug)]
struct TextRunPrimitiveCache {
    color_texture_id: TextureId,
    glyphs: Option<PackedTextRunPrimitive>,
}

impl TextRunPrimitiveCache {
    fn new() -> TextRunPrimitiveCache {
        TextRunPrimitiveCache {
            color_texture_id: TextureId(0),
            glyphs: None,
        }
    }
}

#[derive(Debug)]
struct TextPrimitive {
    color: ColorF,
    font_key: FontKey,
    size: Au,
    blur_radius: Au,
    glyph_index: u32,
    cache: Option<TextPrimitiveCache>,
}

#[derive(Debug)]
struct TextRunPrimitive {
    color: ColorF,
    font_key: FontKey,
    size: Au,
    blur_radius: Au,
    glyph_range: ItemRange,
    cache: Option<TextRunPrimitiveCache>,
}

#[derive(Debug)]
struct BoxShadowPrimitiveCache {
    elements: Vec<PackedBoxShadowPrimitive>,
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
    cache: Option<BoxShadowPrimitiveCache>,
}

#[derive(Debug)]
struct BorderPrimitiveCache {
    elements: [PackedBorderPrimitive; ELEMENTS_PER_BORDER],
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
    cache: Option<Box<BorderPrimitiveCache>>,
}

impl BorderPrimitive {
    fn pack_style(&self) -> [f32; 4] {
        [
            pack_as_float(self.top_style as u32),
            pack_as_float(self.right_style as u32),
            pack_as_float(self.bottom_style as u32),
            pack_as_float(self.left_style as u32),
        ]
    }
}

#[derive(Debug)]
enum ImagePrimitiveKind {
    Image(ImageKey, ImageRendering, Size2D<f32>, Size2D<f32>),
    WebGL(WebGLContextId),
}

#[derive(Debug)]
enum ImagePrimitiveCache {
    Normal(PackedImagePrimitive),
    Clip(PackedImagePrimitiveClip),
}

#[derive(Debug)]
struct ImagePrimitive {
    kind: ImagePrimitiveKind,
    cache: Option<ImagePrimitiveCache>,
}

#[derive(Debug)]
enum GradientPrimitiveCache {
    Aligned(Vec<PackedAlignedGradientPrimitive>),
    Angle(PackedAngleGradientPrimitive),
}

#[derive(Debug)]
struct GradientPrimitive {
    stops_range: ItemRange,
    kind: GradientType,
    start_point: Point2D<f32>,
    end_point: Point2D<f32>,
    cache: Option<GradientPrimitiveCache>,
}

#[derive(Debug)]
enum PrimitiveDetails {
    Rectangle(RectanglePrimitive),
    Text(TextPrimitive),
    TextRun(TextRunPrimitive),
    Image(ImagePrimitive),
    Border(BorderPrimitive),
    Gradient(GradientPrimitive),
    BoxShadow(BoxShadowPrimitive),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
enum AlphaBatchKind {
    Composite = 0,
    Blend = 1,
    Rectangle = 2,
    RectangleClip = 3,
    Text = 4,
    TextRun = 5,
    Image = 6,
    ImageClip = 7,
    Border = 8,
    AlignedGradient = 9,
    AngleGradient = 10,
    BoxShadow = 11,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PrimitiveIndex(usize);

#[derive(Debug)]
struct Primitive {
    rect: Rect<f32>,
    local_clip_rect: Rect<f32>,
    complex_clip: Option<Box<Clip>>,
    bounding_rect: Option<Rect<DevicePixel>>,
    details: PrimitiveDetails,
}

impl Primitive {
    fn is_opaque(&self, resource_cache: &ResourceCache, frame_id: FrameId) -> bool {
        if self.complex_clip.is_some() {
            return false;
        }

        match self.details {
            PrimitiveDetails::Rectangle(ref primitive) => primitive.color.a == 1.0,
            PrimitiveDetails::Image(ImagePrimitive {
                kind: ImagePrimitiveKind::Image(image_key, image_rendering, _, tile_spacing),
                ..
            }) => {
                tile_spacing.width == 0.0 && tile_spacing.height == 0.0 &&
                    resource_cache.get_image(image_key, image_rendering, frame_id).is_opaque
            }
            _ => false,
        }
    }

    fn prepare_for_render(&mut self,
                          screen_rect: &Rect<DevicePixel>,
                          layer_index: StackingContextIndex,
                          layer_transform: &Matrix4D<f32>,
                          layer_combined_local_clip_rect: &Rect<f32>,
                          resource_cache: &ResourceCache,
                          frame_id: FrameId,
                          device_pixel_ratio: f32,
                          auxiliary_lists: &AuxiliaryLists) {
        let layer_index = pack_as_float(layer_index.0 as u32);

        match self.details {
            PrimitiveDetails::Rectangle(..) => {
                // not cached by build_resource_list
                unreachable!()
            }
            PrimitiveDetails::BoxShadow(ref mut shadow) => {
                let mut rects = Vec::new();
                let inverted = match shadow.clip_mode {
                    BoxShadowClipMode::None | BoxShadowClipMode::Outset => {
                        subtract_rect(&self.rect, &shadow.src_rect, &mut rects);
                        0.0
                    }
                    BoxShadowClipMode::Inset => {
                        subtract_rect(&self.rect, &shadow.bs_rect, &mut rects);
                        1.0
                    }
                };

                let mut elements = Vec::new();
                for rect in rects {
                    elements.push(PackedBoxShadowPrimitive {
                        common: PackedPrimitiveInfo {
                            padding: [0, 0],
                            task_id: 0.0,
                            layer_index: layer_index,
                            local_clip_rect: self.local_clip_rect,
                            local_rect: rect,
                        },
                        color: shadow.color,

                        border_radii: Point2D::new(shadow.border_radius,
                                                   shadow.border_radius),
                        blur_radius: shadow.blur_radius,
                        inverted: inverted,
                        bs_rect: shadow.bs_rect,
                        src_rect: shadow.src_rect,
                    });
                }

                shadow.cache = Some(BoxShadowPrimitiveCache {
                    elements: elements,
                });
            }
            PrimitiveDetails::Image(ref mut image) => {
                let ImageInfo {
                    color_texture_id: texture_id,
                    uv0,
                    uv1,
                    stretch_size,
                    tile_spacing,
                    uv_kind,
                } = image.image_info(resource_cache, frame_id);

                match self.complex_clip {
                    Some(ref complex_clip) => {
                        let element = PackedImagePrimitiveClip {
                            common: PackedPrimitiveInfo {
                                padding: [0, 0],
                                task_id: 0.0,
                                layer_index: layer_index,
                                local_clip_rect: self.local_clip_rect,
                                local_rect: self.rect,
                            },
                            uv0: uv0,
                            uv1: uv1,
                            stretch_size: stretch_size.unwrap_or(self.rect.size),
                            tile_spacing: tile_spacing,
                            uv_kind: pack_as_float(uv_kind as u32),
                            texture_id: texture_id,
                            padding: [0, 0],
                            clip: complex_clip.as_ref().clone(),
                        };

                        image.cache = Some(ImagePrimitiveCache::Clip(element));
                    }
                    None => {
                        let element = PackedImagePrimitive {
                            common: PackedPrimitiveInfo {
                                padding: [0, 0],
                                task_id: 0.0,
                                layer_index: layer_index,
                                local_clip_rect: self.local_clip_rect,
                                local_rect: self.rect,
                            },
                            uv0: uv0,
                            uv1: uv1,
                            stretch_size: stretch_size.unwrap_or(self.rect.size),
                            tile_spacing: tile_spacing,
                            uv_kind: pack_as_float(uv_kind as u32),
                            texture_id: texture_id,
                            padding: [0, 0],
                        };

                        image.cache = Some(ImagePrimitiveCache::Normal(element));
                    }
                }
            }
            PrimitiveDetails::Gradient(ref mut gradient) => {
                match gradient.kind {
                    GradientType::Horizontal | GradientType::Vertical => {
                        let stops = auxiliary_lists.gradient_stops(&gradient.stops_range);
                        let mut pieces = Vec::new();
                        for i in 0..(stops.len() - 1) {
                            let (prev_stop, next_stop) = (&stops[i], &stops[i + 1]);
                            let piece_origin;
                            let piece_size;
                            match gradient.kind {
                                GradientType::Horizontal => {
                                    let prev_x = util::lerp(gradient.start_point.x,
                                                            gradient.end_point.x,
                                                            prev_stop.offset);
                                    let next_x = util::lerp(gradient.start_point.x,
                                                            gradient.end_point.x,
                                                            next_stop.offset);
                                    piece_origin = Point2D::new(prev_x, self.rect.origin.y);
                                    piece_size = Size2D::new(next_x - prev_x,
                                                             self.rect.size.height);
                                }
                                GradientType::Vertical => {
                                    let prev_y = util::lerp(gradient.start_point.y,
                                                            gradient.end_point.y,
                                                            prev_stop.offset);
                                    let next_y = util::lerp(gradient.start_point.y,
                                                            gradient.end_point.y,
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

                                    match gradient.kind {
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

                                    match gradient.kind {
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

                            pieces.push(PackedAlignedGradientPrimitive {
                                common: PackedPrimitiveInfo {
                                    padding: [0, 0],
                                    task_id: 0.0,
                                    layer_index: layer_index,
                                    local_clip_rect: self.local_clip_rect,
                                    local_rect: piece_rect,
                                },
                                color0: prev_stop.color,
                                color1: next_stop.color,
                                padding: [0, 0, 0],
                                kind: pack_as_float(gradient.kind as u32),
                                clip: clip,
                            });
                        }

                        gradient.cache = Some(GradientPrimitiveCache::Aligned(pieces));
                    }
                    GradientType::Rotated => {
                        let src_stops = auxiliary_lists.gradient_stops(&gradient.stops_range);
                        if src_stops.len() > MAX_STOPS_PER_ANGLE_GRADIENT {
                            println!("TODO: Angle gradients with > {} stops",
                                     MAX_STOPS_PER_ANGLE_GRADIENT);
                            return;
                        }

                        let mut stops: [f32; MAX_STOPS_PER_ANGLE_GRADIENT] = unsafe {
                            mem::uninitialized()
                        };
                        let mut colors: [ColorF; MAX_STOPS_PER_ANGLE_GRADIENT] = unsafe {
                            mem::uninitialized()
                        };

                        let sx = gradient.start_point.x;
                        let ex = gradient.end_point.x;

                        let (sp, ep) = if sx > ex {
                            for (stop_index, stop) in src_stops.iter().rev().enumerate() {
                                stops[stop_index] = 1.0 - stop.offset;
                                colors[stop_index] = stop.color;
                            }

                            (gradient.end_point, gradient.start_point)
                        } else {
                            for (stop_index, stop) in src_stops.iter().enumerate() {
                                stops[stop_index] = stop.offset;
                                colors[stop_index] = stop.color;
                            }

                            (gradient.start_point, gradient.end_point)
                        };

                        let packed_prim = PackedAngleGradientPrimitive {
                            common: PackedPrimitiveInfo {
                                padding: [0, 0],
                                task_id: 0.0,
                                layer_index: layer_index,
                                local_clip_rect: self.local_clip_rect,
                                local_rect: self.rect,
                            },
                            padding: [0, 0, 0],
                            start_point: sp,
                            end_point: ep,
                            stop_count: pack_as_float(src_stops.len() as u32),
                            stops: stops,
                            colors: colors,
                        };

                        gradient.cache = Some(GradientPrimitiveCache::Angle(packed_prim));
                    }
                }
            }
            PrimitiveDetails::Border(ref mut border) => {
                let inner_radius = BorderRadius {
                    top_left: Size2D::new(border.radius.top_left.width - border.left_width,
                                          border.radius.top_left.height - border.top_width),
                    top_right: Size2D::new(border.radius.top_right.width - border.right_width,
                                           border.radius.top_right.height - border.top_width),
                    bottom_left:
                        Size2D::new(border.radius.bottom_left.width - border.left_width,
                                    border.radius.bottom_left.height - border.bottom_width),
                    bottom_right:
                        Size2D::new(border.radius.bottom_right.width - border.right_width,
                                    border.radius.bottom_right.height - border.bottom_width),
                };

                border.cache = Some(Box::new(BorderPrimitiveCache {
                    elements: [
                        PackedBorderPrimitive {
                            common: PackedPrimitiveInfo {
                                padding: [0, 0],
                                task_id: 0.0,
                                layer_index: layer_index,
                                local_clip_rect: self.local_clip_rect,
                                local_rect: rect_from_points_f(border.tl_outer.x,
                                                               border.tl_outer.y,
                                                               border.tl_inner.x,
                                                               border.tl_inner.y),
                            },
                            vertical_color: border.top_color,
                            horizontal_color: border.left_color,
                            outer_radius_x: border.radius.top_left.width,
                            outer_radius_y: border.radius.top_left.height,
                            inner_radius_x: inner_radius.top_left.width,
                            inner_radius_y: inner_radius.top_left.height,
                            style: border.pack_style(),
                            part: [pack_as_float(PrimitivePart::TopLeft as u32), 0.0, 0.0, 0.0],
                        },
                        PackedBorderPrimitive {
                            common: PackedPrimitiveInfo {
                                padding: [0, 0],
                                task_id: 0.0,
                                layer_index: layer_index,
                                local_clip_rect: self.local_clip_rect,
                                local_rect: rect_from_points_f(border.tr_inner.x,
                                                               border.tr_outer.y,
                                                               border.tr_outer.x,
                                                               border.tr_inner.y),
                            },
                            vertical_color: border.right_color,
                            horizontal_color: border.top_color,
                            outer_radius_x: border.radius.top_right.width,
                            outer_radius_y: border.radius.top_right.height,
                            inner_radius_x: inner_radius.top_right.width,
                            inner_radius_y: inner_radius.top_right.height,
                            style: border.pack_style(),
                            part: [pack_as_float(PrimitivePart::TopRight as u32), 0.0, 0.0, 0.0],
                        },
                        PackedBorderPrimitive {
                            common: PackedPrimitiveInfo {
                                padding: [0, 0],
                                task_id: 0.0,
                                layer_index: layer_index,
                                local_clip_rect: self.local_clip_rect,
                                local_rect: rect_from_points_f(border.bl_outer.x,
                                                               border.bl_inner.y,
                                                               border.bl_inner.x,
                                                               border.bl_outer.y),
                            },
                            vertical_color: border.left_color,
                            horizontal_color: border.bottom_color,
                            outer_radius_x: border.radius.bottom_left.width,
                            outer_radius_y: border.radius.bottom_left.height,
                            inner_radius_x: inner_radius.bottom_left.width,
                            inner_radius_y: inner_radius.bottom_left.height,
                            style: border.pack_style(),
                            part: [pack_as_float(PrimitivePart::BottomLeft as u32), 0.0, 0.0, 0.0],
                        },
                        PackedBorderPrimitive {
                            common: PackedPrimitiveInfo {
                                padding: [0, 0],
                                task_id: 0.0,
                                layer_index: layer_index,
                                local_clip_rect: self.local_clip_rect,
                                local_rect: rect_from_points_f(border.br_inner.x,
                                                               border.br_inner.y,
                                                               border.br_outer.x,
                                                               border.br_outer.y),
                            },
                            vertical_color: border.right_color,
                            horizontal_color: border.bottom_color,
                            outer_radius_x: border.radius.bottom_right.width,
                            outer_radius_y: border.radius.bottom_right.height,
                            inner_radius_x: inner_radius.bottom_right.width,
                            inner_radius_y: inner_radius.bottom_right.height,
                            style: border.pack_style(),
                            part: [pack_as_float(PrimitivePart::BottomRight as u32), 0.0, 0.0, 0.0],
                        },
                        PackedBorderPrimitive {
                            common: PackedPrimitiveInfo {
                                padding: [0, 0],
                                task_id: 0.0,
                                layer_index: layer_index,
                                local_clip_rect: self.local_clip_rect,
                                local_rect: rect_from_points_f(border.tl_outer.x,
                                                               border.tl_inner.y,
                                                               border.tl_outer.x + border.left_width,
                                                               border.bl_inner.y),
                            },
                            vertical_color: border.left_color,
                            horizontal_color: border.left_color,
                            outer_radius_x: 0.0,
                            outer_radius_y: 0.0,
                            inner_radius_x: 0.0,
                            inner_radius_y: 0.0,
                            style: border.pack_style(),
                            part: [pack_as_float(PrimitivePart::Left as u32), 0.0, 0.0, 0.0],
                        },
                        PackedBorderPrimitive {
                            common: PackedPrimitiveInfo {
                                padding: [0, 0],
                                task_id: 0.0,
                                layer_index: layer_index,
                                local_clip_rect: self.local_clip_rect,
                                local_rect: rect_from_points_f(border.tr_outer.x - border.right_width,
                                                               border.tr_inner.y,
                                                               border.br_outer.x,
                                                               border.br_inner.y),
                            },
                            vertical_color: border.right_color,
                            horizontal_color: border.right_color,
                            outer_radius_x: 0.0,
                            outer_radius_y: 0.0,
                            inner_radius_x: 0.0,
                            inner_radius_y: 0.0,
                            style: border.pack_style(),
                            part: [pack_as_float(PrimitivePart::Right as u32), 0.0, 0.0, 0.0],
                        },
                        PackedBorderPrimitive {
                            common: PackedPrimitiveInfo {
                                padding: [0, 0],
                                task_id: 0.0,
                                layer_index: layer_index,
                                local_clip_rect: self.local_clip_rect,
                                local_rect: rect_from_points_f(border.tl_inner.x,
                                                               border.tl_outer.y,
                                                               border.tr_inner.x,
                                                               border.tr_outer.y + border.top_width),
                            },
                            vertical_color: border.top_color,
                            horizontal_color: border.top_color,
                            outer_radius_x: 0.0,
                            outer_radius_y: 0.0,
                            inner_radius_x: 0.0,
                            inner_radius_y: 0.0,
                            style: border.pack_style(),
                            part: [pack_as_float(PrimitivePart::Top as u32), 0.0, 0.0, 0.0],
                        },
                        PackedBorderPrimitive {
                            common: PackedPrimitiveInfo {
                                padding: [0, 0],
                                task_id: 0.0,
                                layer_index: layer_index,
                                local_clip_rect: self.local_clip_rect,
                                local_rect: rect_from_points_f(border.bl_inner.x,
                                                               border.bl_outer.y - border.bottom_width,
                                                               border.br_inner.x,
                                                               border.br_outer.y),
                            },
                            vertical_color: border.bottom_color,
                            horizontal_color: border.bottom_color,
                            outer_radius_x: 0.0,
                            outer_radius_y: 0.0,
                            inner_radius_x: 0.0,
                            inner_radius_y: 0.0,
                            style: border.pack_style(),
                            part: [pack_as_float(PrimitivePart::Bottom as u32), 0.0, 0.0, 0.0],
                        },
                    ],
                }));
            }
            PrimitiveDetails::Text(ref mut text) => {
                let mut cache = TextPrimitiveCache::new();
                let glyph_range = ItemRange {
                    start: text.glyph_index as usize,
                    length: 1,
                };
                let glyph = auxiliary_lists.glyph_instances(&glyph_range)[0];
                let glyph_key = GlyphKey::new(text.font_key,
                                              text.size,
                                              text.blur_radius,
                                              glyph.index);
                let blur_offset = text.blur_radius.to_f32_px() *
                    (BLUR_INFLATION_FACTOR as f32) / 2.0;

                let image_info = match resource_cache.get_glyph(&glyph_key, frame_id) {
                    None => return,
                    Some(image_info) => image_info,
                };

                debug_assert!(cache.color_texture_id == TextureId(0) ||
                              cache.color_texture_id == image_info.texture_id);
                cache.color_texture_id = image_info.texture_id;

                let x = glyph.x + image_info.user_data.x0 as f32 / device_pixel_ratio -
                    blur_offset;
                let y = glyph.y - image_info.user_data.y0 as f32 / device_pixel_ratio -
                    blur_offset;

                let width = image_info.requested_rect.size.width as f32 / device_pixel_ratio;
                let height = image_info.requested_rect.size.height as f32 / device_pixel_ratio;

                self.rect = Rect::new(Point2D::new(x, y), Size2D::new(width, height));
                cache.glyph = Some(PackedGlyphPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: [0, 0],
                        task_id: 0.0,
                        layer_index: layer_index,
                        local_clip_rect: self.local_clip_rect,
                        local_rect: self.rect,
                    },
                    color: text.color,
                    uv0: Point2D::new(image_info.pixel_rect.top_left.x.0 as f32,
                                      image_info.pixel_rect.top_left.y.0 as f32),
                    uv1: Point2D::new(image_info.pixel_rect.bottom_right.x.0 as f32,
                                      image_info.pixel_rect.bottom_right.y.0 as f32),
                });

                text.cache = Some(cache);
            }
            PrimitiveDetails::TextRun(ref mut text_run) => {
                debug_assert!(text_run.cache.is_none());
                let mut cache = TextRunPrimitiveCache::new();

                let src_glyphs = auxiliary_lists.glyph_instances(&text_run.glyph_range);
                let mut glyph_key = GlyphKey::new(text_run.font_key,
                                                  text_run.size,
                                                  text_run.blur_radius,
                                                  src_glyphs[0].index);
                let blur_offset = text_run.blur_radius.to_f32_px() *
                    (BLUR_INFLATION_FACTOR as f32) / 2.0;

                let mut glyphs: [PackedTextRunGlyph; GLYPHS_PER_TEXT_RUN] = unsafe {
                    mem::zeroed()
                };

                self.rect = Rect::zero();
                for (glyph_index, glyph) in src_glyphs.iter().enumerate() {
                    glyph_key.index = glyph.index;

                    let image_info = match resource_cache.get_glyph(&glyph_key, frame_id) {
                        None => continue,
                        Some(image_info) => image_info,
                    };

                    debug_assert!(cache.color_texture_id == TextureId(0) ||
                                  cache.color_texture_id == image_info.texture_id);
                    cache.color_texture_id = image_info.texture_id;

                    let x = glyph.x + image_info.user_data.x0 as f32 / device_pixel_ratio -
                        blur_offset;
                    let y = glyph.y - image_info.user_data.y0 as f32 / device_pixel_ratio -
                        blur_offset;

                    let width = image_info.requested_rect.size.width as f32 /
                        device_pixel_ratio;
                    let height = image_info.requested_rect.size.height as f32 /
                        device_pixel_ratio;

                    let local_glyph_rect = Rect::new(Point2D::new(x, y),
                                                     Size2D::new(width, height));
                    self.rect = self.rect.union(&local_glyph_rect);

                    glyphs[glyph_index] = PackedTextRunGlyph {
                        local_rect: local_glyph_rect,
                        uv0: Point2D::new(image_info.pixel_rect.top_left.x.0 as f32,
                                          image_info.pixel_rect.top_left.y.0 as f32),
                        uv1: Point2D::new(image_info.pixel_rect.bottom_right.x.0 as f32,
                                          image_info.pixel_rect.bottom_right.y.0 as f32),
                    }
                }

                cache.glyphs = Some(PackedTextRunPrimitive {
                    common: PackedPrimitiveInfo {
                        padding: [0, 0],
                        task_id: 0.0,
                        layer_index: layer_index,
                        local_clip_rect: self.local_clip_rect,
                        local_rect: self.rect,
                    },
                    color: text_run.color,
                    glyphs: glyphs,
                });

                text_run.cache = Some(cache);
            }
        }

        self.rebuild_bounding_rect(screen_rect,
                                   layer_transform,
                                   layer_combined_local_clip_rect,
                                   device_pixel_ratio);
    }

    fn build_resource_list(&mut self,
                           resource_list: &mut ResourceList,
                           auxiliary_lists: &AuxiliaryLists) -> bool {
        match self.details {
            PrimitiveDetails::Rectangle(..) => false,
            PrimitiveDetails::BoxShadow(ref details) => {
                details.cache.is_none()
            }
            PrimitiveDetails::Gradient(ref details) => {
                details.cache.is_none()
            }
            PrimitiveDetails::Border(ref details) => {
                details.cache.is_none()
            }
            PrimitiveDetails::Image(ref details) => {
                match details.kind {
                    ImagePrimitiveKind::Image(image_key, image_rendering, _, _) => {
                        resource_list.add_image(image_key, image_rendering);
                    }
                    ImagePrimitiveKind::WebGL(..) => {}
                }
                details.cache.is_none()
            }
            PrimitiveDetails::Text(ref details) => {
                let glyphs = auxiliary_lists.glyph_instances(&ItemRange {
                    start: details.glyph_index as usize,
                    length: 1,
                });
                for glyph in glyphs {
                    let glyph = Glyph::new(details.size, details.blur_radius, glyph.index);
                    resource_list.add_glyph(details.font_key, glyph);
                }
                details.cache.is_none()
            }
            PrimitiveDetails::TextRun(ref details) => {
                let glyphs = auxiliary_lists.glyph_instances(&details.glyph_range);
                for glyph in glyphs {
                    let glyph = Glyph::new(details.size, details.blur_radius, glyph.index);
                    resource_list.add_glyph(details.font_key, glyph);
                }
                details.cache.is_none()
            }
        }
    }

    // Optional narrow phase intersection test, depending on primitive type.
    fn affects_tile(&self,
                    tile_rect: &Rect<DevicePixel>,
                    transform: &Matrix4D<f32>,
                    device_pixel_ratio: f32) -> bool {
        match self.details {
            PrimitiveDetails::Rectangle(..) => true,
            PrimitiveDetails::Text(..) => true,
            PrimitiveDetails::TextRun(..) => true,
            PrimitiveDetails::Image(..) => true,
            PrimitiveDetails::Gradient(..) => true,
            PrimitiveDetails::BoxShadow(..) => true,
            PrimitiveDetails::Border(ref border) => {
                let inner_rect = rect_from_points_f(border.tl_inner.x.max(border.bl_inner.x),
                                                    border.tl_inner.y.max(border.tr_inner.y),
                                                    border.tr_inner.x.min(border.br_inner.x),
                                                    border.bl_inner.y.min(border.br_inner.y));
                let inner_rect = TransformedRect::new(&inner_rect, transform, device_pixel_ratio);

                !inner_rect.bounding_rect.contains_rect(tile_rect)
            }
        }
    }

    fn add_to_batch(&self,
                    batch: &mut PrimitiveBatch,
                    layer_index: StackingContextIndex,
                    task_id: f32,
                    transform_kind: TransformedRectKind,
                    needs_blending: bool) -> bool {
        if transform_kind != batch.transform_kind ||
           needs_blending != batch.blending_enabled {
            return false
        }

        let layer_index = pack_as_float(layer_index.0 as u32);

        match (&mut batch.data, &self.details) {
            (&mut PrimitiveBatchData::Blend(..), _) => return false,
            (&mut PrimitiveBatchData::Composite(..), _) => return false,
            (&mut PrimitiveBatchData::Rectangles(ref mut data),
             &PrimitiveDetails::Rectangle(ref rectangle)) => {
                if self.complex_clip.is_some() {
                    return false;
                }
                data.push(PackedRectanglePrimitive {
                    common: PackedPrimitiveInfo {
                        padding: [0, 0],
                        task_id: task_id,
                        layer_index: layer_index,
                        local_clip_rect: self.local_clip_rect,
                        local_rect: self.rect,
                    },
                    color: rectangle.color,
                });
            }
            (&mut PrimitiveBatchData::Rectangles(..), _) => return false,
            (&mut PrimitiveBatchData::RectanglesClip(ref mut data),
             &PrimitiveDetails::Rectangle(ref rectangle)) => {
                if self.complex_clip.is_none() {
                    return false;
                }
                data.push(PackedRectanglePrimitiveClip {
                    common: PackedPrimitiveInfo {
                        padding: [0, 0],
                        task_id: task_id,
                        layer_index: layer_index,
                        local_clip_rect: self.local_clip_rect,
                        local_rect: self.rect,
                    },
                    color: rectangle.color,
                    clip: *self.complex_clip.as_ref().unwrap().clone(),
                });
            }
            (&mut PrimitiveBatchData::RectanglesClip(..), _) => return false,
            (&mut PrimitiveBatchData::Image(ref mut data),
             &PrimitiveDetails::Image(ref image)) => {
                let cache = image.cache.as_ref().expect("No image cache!");

                match cache {
                    &ImagePrimitiveCache::Normal(ref element) => {
                        if batch.color_texture_id != TextureId(0) && element.texture_id != batch.color_texture_id {
                            return false
                        }

                        let mut element = element.clone();
                        element.common.task_id = task_id;
                        data.push(element);
                    }
                    &ImagePrimitiveCache::Clip(..) => return false,
                }

            }
            (&mut PrimitiveBatchData::Image(..), _) => return false,
            (&mut PrimitiveBatchData::ImageClip(ref mut data),
             &PrimitiveDetails::Image(ref image)) => {
                let cache = image.cache.as_ref().expect("No image cache!");

                match cache {
                    &ImagePrimitiveCache::Normal(..) => return false,
                    &ImagePrimitiveCache::Clip(ref element) => {
                        if batch.color_texture_id != TextureId(0) && element.texture_id != batch.color_texture_id {
                            return false
                        }

                        let mut element = element.clone();
                        element.common.task_id = task_id;
                        data.push(element);
                    }
                }
            }
            (&mut PrimitiveBatchData::ImageClip(..), _) => return false,
            (&mut PrimitiveBatchData::Borders(ref mut data),
             &PrimitiveDetails::Border(ref border)) => {
                let cache = border.cache.as_ref().expect("No cache for border present!");

                for element in &cache.elements {
                    let mut element = element.clone();
                    element.common.task_id = task_id;
                    data.push(element);
                }
            }
            (&mut PrimitiveBatchData::Borders(..), _) => return false,
            (&mut PrimitiveBatchData::AlignedGradient(ref mut data),
             &PrimitiveDetails::Gradient(ref gradient)) => {
                match gradient.cache {
                    Some(GradientPrimitiveCache::Aligned(ref pieces)) => {
                        for piece in pieces {
                            let mut piece = piece.clone();
                            piece.common.task_id = task_id;
                            data.push(piece);
                        }
                    }
                    Some(GradientPrimitiveCache::Angle(..)) | None => return false,
                }
            }
            (&mut PrimitiveBatchData::AlignedGradient(..), _) => return false,
            (&mut PrimitiveBatchData::AngleGradient(ref mut data),
             &PrimitiveDetails::Gradient(ref gradient)) => {
                match gradient.cache {
                    Some(GradientPrimitiveCache::Angle(ref piece)) => {
                        let mut piece = piece.clone();
                        piece.common.task_id = task_id;
                        data.push(piece);
                    }
                    Some(GradientPrimitiveCache::Aligned(..)) | None => return false,
                }
            }
            (&mut PrimitiveBatchData::AngleGradient(..), _) => return false,
            (&mut PrimitiveBatchData::BoxShadows(ref mut data),
             &PrimitiveDetails::BoxShadow(ref shadow)) => {
                let cache = shadow.cache.as_ref().expect("No cache for box shadow present!");

                for element in &cache.elements {
                    let mut element = element.clone();
                    element.common.task_id = task_id;
                    data.push(element);
                }
            }
            (&mut PrimitiveBatchData::BoxShadows(..), _) => return false,
            (&mut PrimitiveBatchData::Text(ref mut data),
             &PrimitiveDetails::Text(ref text)) => {
                let cache = match text.cache.as_ref() {
                    None => {
                        // This can happen if the resource cache failed to rasterize a glyph,
                        // perhaps because the font doesn't contain that glyph. In this case,
                        // render nothing (successfully).
                        return true
                    }
                    Some(cache) => cache,
                };

                if batch.color_texture_id != TextureId(0) &&
                        cache.color_texture_id != batch.color_texture_id {
                    return false;
                }

                batch.color_texture_id = cache.color_texture_id;

                for glyph in &cache.glyph {
                    let mut glyph = glyph.clone();
                    glyph.common.task_id = task_id;
                    data.push(glyph);
                }
            }
            (&mut PrimitiveBatchData::Text(..), _) => return false,
            (&mut PrimitiveBatchData::TextRun(ref mut data),
             &PrimitiveDetails::TextRun(ref text)) => {
                let cache = text.cache.as_ref().expect("No cache for text run present!");

                if batch.color_texture_id != TextureId(0) &&
                        cache.color_texture_id != batch.color_texture_id {
                    return false;
                }

                for glyphs in &cache.glyphs {
                    let mut glyphs = glyphs.clone();
                    glyphs.common.task_id = task_id;
                    data.push(glyphs);
                }
            }
            (&mut PrimitiveBatchData::TextRun(..), _) => return false,
        }

        true
    }

    fn rebuild_bounding_rect(&mut self,
                             screen_rect: &Rect<DevicePixel>,
                             layer_transform: &Matrix4D<f32>,
                             layer_combined_local_clip_rect: &Rect<f32>,
                             device_pixel_ratio: f32) {
        self.bounding_rect = None;

        let local_rect;
        match self.rect
                  .intersection(&self.local_clip_rect)
                  .and_then(|rect| rect.intersection(layer_combined_local_clip_rect)) {
            Some(rect) => local_rect = rect,
            None => return,
        };

        let xf_rect = TransformedRect::new(&local_rect, layer_transform, device_pixel_ratio);
        if !xf_rect.bounding_rect.intersects(screen_rect) {
            return
        }

        self.bounding_rect = Some(xf_rect.bounding_rect)
    }

    fn batch_kind(&self) -> AlphaBatchKind {
        match (&self.details, &self.complex_clip) {
            (&PrimitiveDetails::Rectangle(_), &None) => AlphaBatchKind::Rectangle,
            (&PrimitiveDetails::Rectangle(_), &Some(_)) => AlphaBatchKind::RectangleClip,
            (&PrimitiveDetails::Text(_), _) => AlphaBatchKind::Text,
            (&PrimitiveDetails::TextRun(_), _) => AlphaBatchKind::TextRun,
            (&PrimitiveDetails::Image(_), &None) => AlphaBatchKind::Image,
            (&PrimitiveDetails::Image(_), &Some(_)) => AlphaBatchKind::ImageClip,
            (&PrimitiveDetails::Border(_), _) => AlphaBatchKind::Border,
            (&PrimitiveDetails::Gradient(ref gradient), _) => {
                match gradient.kind {
                    GradientType::Horizontal | GradientType::Vertical => AlphaBatchKind::AlignedGradient,
                    GradientType::Rotated => AlphaBatchKind::AngleGradient,
                }
            }
            (&PrimitiveDetails::BoxShadow(_), _) => AlphaBatchKind::BoxShadow,
        }
    }

    fn color_texture_id(&self, resource_cache: &ResourceCache, frame_id: FrameId) -> TextureId {
        match self.details {
            PrimitiveDetails::Rectangle(_) |
            PrimitiveDetails::Border(_) |
            PrimitiveDetails::Gradient(_) |
            PrimitiveDetails::BoxShadow(_) => TextureId(0),
            PrimitiveDetails::Text(ref text) => {
                match text.cache {
                    Some(ref cache) => cache.color_texture_id,
                    None => TextureId(0),
                }
            }
            PrimitiveDetails::TextRun(ref text_run) => {
                match text_run.cache {
                    Some(ref cache) => cache.color_texture_id,
                    None => TextureId(0),
                }
            }
            PrimitiveDetails::Image(ref image) => {
                image.image_info(resource_cache, frame_id).color_texture_id
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct AlphaBatchKey {
    kind: AlphaBatchKind,
    flags: AlphaBatchKeyFlags,
    color_texture_id: TextureId,
}

impl AlphaBatchKey {
    fn blend() -> AlphaBatchKey {
        AlphaBatchKey {
            kind: AlphaBatchKind::Blend,
            flags: AlphaBatchKeyFlags(0),
            color_texture_id: TextureId(0),
        }
    }

    fn composite() -> AlphaBatchKey {
        AlphaBatchKey {
            kind: AlphaBatchKind::Composite,
            flags: AlphaBatchKeyFlags(0),
            color_texture_id: TextureId(0),
        }
    }

    fn primitive(kind: AlphaBatchKind,
                 flags: AlphaBatchKeyFlags,
                 color_texture_id: TextureId)
                 -> AlphaBatchKey {
        AlphaBatchKey {
            kind: kind,
            flags: flags,
            color_texture_id: color_texture_id,
        }
    }

    fn is_compatible_with(&self, other: &AlphaBatchKey) -> bool {
        self.kind == other.kind &&
            self.flags == other.flags &&
            (self.color_texture_id == TextureId(0) || other.color_texture_id == TextureId(0) ||
             self.color_texture_id == other.color_texture_id)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct AlphaBatchKeyFlags(u8);

impl AlphaBatchKeyFlags {
    fn new(transform_kind: TransformedRectKind, needs_blending: bool) -> AlphaBatchKeyFlags {
        AlphaBatchKeyFlags(((transform_kind as u8) << 1) | (needs_blending as u8))
    }

    fn transform_kind(&self) -> TransformedRectKind {
        if ((self.0 >> 1) & 1) == 0 {
            TransformedRectKind::AxisAligned
        } else {
            TransformedRectKind::Complex
        }
    }

    fn needs_blending(&self) -> bool {
        (self.0 & 1) != 0
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone)]
enum PrimitivePart {
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
#[derive(Debug, Clone)]
pub struct PackedPrimitiveInfo {
    layer_index: f32,
    task_id: f32,
    padding: [u32; 2],
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
    uv0: Point2D<f32>,
    uv1: Point2D<f32>,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct PackedTextRunPrimitive {
    common: PackedPrimitiveInfo,
    color: ColorF,
    glyphs: [PackedTextRunGlyph; GLYPHS_PER_TEXT_RUN],
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct PackedTextRunGlyph {
    local_rect: Rect<f32>,
    uv0: Point2D<f32>,
    uv1: Point2D<f32>,
}

#[derive(Debug, Clone, Copy)]
pub enum TextureCoordKind {
    Normalized = 0,
    Pixel,
}

#[derive(Debug, Clone)]
pub struct PackedImagePrimitive {
    common: PackedPrimitiveInfo,
    uv0: Point2D<f32>,
    uv1: Point2D<f32>,
    stretch_size: Size2D<f32>,
    tile_spacing: Size2D<f32>,
    uv_kind: f32,
    texture_id: TextureId,
    padding: [u32; 2],
}

#[derive(Debug, Clone)]
pub struct PackedImagePrimitiveClip {
    common: PackedPrimitiveInfo,
    uv0: Point2D<f32>,
    uv1: Point2D<f32>,
    stretch_size: Size2D<f32>,
    tile_spacing: Size2D<f32>,
    uv_kind: f32,
    texture_id: TextureId,
    padding: [u32; 2],
    clip: Clip,
}

#[derive(Debug, Clone)]
pub struct PackedAlignedGradientPrimitive {
    common: PackedPrimitiveInfo,
    color0: ColorF,
    color1: ColorF,
    kind: f32,
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
    stop_count: f32,
    padding: [u32; 3],
    colors: [ColorF; MAX_STOPS_PER_ANGLE_GRADIENT],
    stops: [f32; MAX_STOPS_PER_ANGLE_GRADIENT],
}

#[derive(Debug, Clone)]
pub struct PackedBorderPrimitive {
    common: PackedPrimitiveInfo,
    vertical_color: ColorF,
    horizontal_color: ColorF,
    outer_radius_x: f32,
    outer_radius_y: f32,
    inner_radius_x: f32,
    inner_radius_y: f32,
    style: [f32; 4],
    part: [f32; 4],
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
    src_task_id: f32,
    target_task_id: f32,
    brightness: f32,
    opacity: f32,
}

#[derive(Debug, Copy, Clone)]
struct PackedCompositeInfo {
    kind: f32,
    op: f32,
    amount: f32,
    padding: f32,
}

impl PackedCompositeInfo {
    fn new(ops: &[CompositionOp]) -> PackedCompositeInfo {
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
            kind: pack_as_float(kind),
            op: pack_as_float(op),
            amount: amount,
            padding: 0.0,
        }
    }
}

#[derive(Debug)]
pub struct PackedCompositePrimitive {
    src0_task_id: f32,
    src1_task_id: f32,
    target_task_id: f32,
    padding: f32,
    info: PackedCompositeInfo,
}

#[derive(Debug)]
pub enum PrimitiveBatchData {
    Rectangles(Vec<PackedRectanglePrimitive>),
    RectanglesClip(Vec<PackedRectanglePrimitiveClip>),
    Borders(Vec<PackedBorderPrimitive>),
    BoxShadows(Vec<PackedBoxShadowPrimitive>),
    Text(Vec<PackedGlyphPrimitive>),
    TextRun(Vec<PackedTextRunPrimitive>),
    Image(Vec<PackedImagePrimitive>),
    ImageClip(Vec<PackedImagePrimitiveClip>),
    Blend(Vec<PackedBlendPrimitive>),
    Composite(Vec<PackedCompositePrimitive>),
    AlignedGradient(Vec<PackedAlignedGradientPrimitive>),
    AngleGradient(Vec<PackedAngleGradientPrimitive>),
}

#[derive(Debug)]
pub struct PrimitiveBatch {
    pub transform_kind: TransformedRectKind,
    pub color_texture_id: TextureId,        // TODO(gw): Expand to sampler array to handle all glyphs!
    pub blending_enabled: bool,
    pub data: PrimitiveBatchData,
}

impl PrimitiveBatch {
    fn blend() -> PrimitiveBatch {
        PrimitiveBatch {
            color_texture_id: TextureId(0),
            transform_kind: TransformedRectKind::AxisAligned,
            blending_enabled: true,
            data: PrimitiveBatchData::Blend(Vec::new()),
        }
    }

    fn composite() -> PrimitiveBatch {
        PrimitiveBatch {
            color_texture_id: TextureId(0),
            transform_kind: TransformedRectKind::AxisAligned,
            blending_enabled: true,
            data: PrimitiveBatchData::Composite(Vec::new()),
        }
    }

    fn pack_blend(&mut self,
                  src_rect_index: RenderTaskIndex,
                  target_rect_index: RenderTaskIndex,
                  opacity: f32,
                  brightness: f32) -> bool {
        match &mut self.data {
            &mut PrimitiveBatchData::Blend(ref mut ubo_data) => {
                ubo_data.push(PackedBlendPrimitive {
                    src_task_id: pack_as_float(src_rect_index.0 as u32),
                    target_task_id: pack_as_float(target_rect_index.0 as u32),
                    opacity: opacity,
                    brightness: brightness,
                });

                true
            }
            _ => false
        }
    }

    fn pack_composite(&mut self,
                      rect0_index: RenderTaskIndex,
                      rect1_index: RenderTaskIndex,
                      target_rect_index: RenderTaskIndex,
                      info: PackedCompositeInfo) -> bool {
        match &mut self.data {
            &mut PrimitiveBatchData::Composite(ref mut ubo_data) => {
                ubo_data.push(PackedCompositePrimitive {
                    src0_task_id: pack_as_float(rect0_index.0 as u32),
                    src1_task_id: pack_as_float(rect1_index.0 as u32),
                    target_task_id: pack_as_float(target_rect_index.0 as u32),
                    padding: 0.0,
                    info: info,
                });

                true
            }
            _ => false
        }
    }

    fn new(prim: &Primitive,
           transform_kind: TransformedRectKind,
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
            PrimitiveDetails::TextRun(..) => {
                PrimitiveBatchData::TextRun(Vec::new())
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
            blending_enabled: blending_enabled,
            data: data,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ScreenTileLayerIndex(usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct StackingContextIndex(usize);

#[derive(Debug)]
struct TileRange {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
}

struct StackingContext {
    index: StackingContextIndex,
    pipeline_id: PipelineId,
    local_transform: Matrix4D<f32>,
    local_rect: Rect<f32>,
    scroll_layer_id: ScrollLayerId,
    xf_rect: Option<TransformedRect>,
    composite_kind: CompositeKind,
    local_clip_rect: Rect<f32>,
    prims_to_prepare: Vec<PrimitiveIndex>,
    tile_range: Option<TileRange>,
}

#[derive(Debug, Clone)]
pub struct PackedStackingContext {
    transform: Matrix4D<f32>,
    inv_transform: Matrix4D<f32>,
    local_clip_rect: Rect<f32>,
    screen_vertices: [Point4D<f32>; 4],
}

impl Default for PackedStackingContext {
    fn default() -> PackedStackingContext {
        PackedStackingContext {
            transform: Matrix4D::identity(),
            inv_transform: Matrix4D::identity(),
            local_clip_rect: Rect::new(Point2D::zero(), Size2D::zero()),
            screen_vertices: [Point4D::zero(); 4],
        }
    }
}

#[derive(Debug, Copy, Clone)]
enum SimpleCompositeInfo {
    Opacity(f32),
    Brightness(f32),
}

#[derive(Debug, Copy, Clone)]
enum CompositeKind {
    None,
    // Requires only a single texture as input (e.g. most filters)
    Simple(SimpleCompositeInfo),
    // Requires two source textures (e.g. mix-blend-mode)
    Complex(PackedCompositeInfo),
}

impl CompositeKind {
    fn new(composition_ops: &[CompositionOp]) -> CompositeKind {
        if composition_ops.is_empty() {
            return CompositeKind::None;
        }

        if composition_ops.len() == 1 {
            match composition_ops.first().unwrap() {
                &CompositionOp::Filter(filter_op) => {
                    match filter_op {
                        LowLevelFilterOp::Opacity(opacity) => {
                            let opacity = opacity.to_f32_px();
                            if opacity == 1.0 {
                                return CompositeKind::None;
                            } else {
                                return CompositeKind::Simple(SimpleCompositeInfo::Opacity(opacity));
                            }
                        }
                        LowLevelFilterOp::Brightness(amount) => {
                            let amount = amount.to_f32_px();
                            return CompositeKind::Simple(SimpleCompositeInfo::Brightness(amount));
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        let info = PackedCompositeInfo::new(composition_ops);
        CompositeKind::Complex(info)
    }
}

impl StackingContext {
    fn is_visible(&self) -> bool {
        self.xf_rect.is_some()
    }

    fn can_contribute_to_scene(&self) -> bool {
        match self.composite_kind {
            CompositeKind::None | CompositeKind::Complex(..) => true,
            CompositeKind::Simple(SimpleCompositeInfo::Brightness(..)) => true,
            CompositeKind::Simple(SimpleCompositeInfo::Opacity(opacity)) => opacity > 0.0,
        }
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

    fn uniform(rect: Rect<f32>, outer_radius: f32, inner_radius: f32) -> ClipCorner {
        ClipCorner {
            rect: rect,
            outer_radius_x: outer_radius,
            outer_radius_y: outer_radius,
            inner_radius_x: inner_radius,
            inner_radius_y: inner_radius,
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
    pub enable_scrollbars: bool,
}

impl FrameBuilderConfig {
    pub fn new(enable_scrollbars: bool) -> FrameBuilderConfig {
        FrameBuilderConfig {
            enable_scrollbars: enable_scrollbars,
        }
    }
}

pub struct FrameBuilder {
    screen_rect: Rect<i32>,
    prim_store: Vec<Primitive>,
    cmds: Vec<PrimitiveRunCmd>,
    device_pixel_ratio: f32,
    debug: bool,

    layer_store: Vec<StackingContext>,
    packed_layers: Vec<PackedStackingContext>,

    scrollbar_prims: Vec<ScrollbarPrimitive>,
}

pub struct Frame {
    pub viewport_size: Size2D<i32>,
    pub debug_rects: Vec<DebugRect>,
    pub cache_size: Size2D<f32>,
    pub phases: Vec<RenderPhase>,
    pub clear_tiles: Vec<ClearTile>,
    pub profile_counters: FrameProfileCounters,
    pub layer_texture_data: Vec<PackedStackingContext>,
    pub render_task_data: Vec<RenderTaskData>,
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

    pub fn uniform(rect: Rect<f32>, radius: f32) -> Clip {
        Clip {
            rect: rect,
            top_left: ClipCorner::uniform(Rect::new(Point2D::new(rect.origin.x,
                                                                 rect.origin.y),
                                                    Size2D::new(radius, radius)),
                                          radius,
                                          0.0),
            top_right: ClipCorner::uniform(Rect::new(Point2D::new(rect.origin.x + rect.size.width - radius,
                                                                  rect.origin.y),
                                                    Size2D::new(radius, radius)),
                                           radius,
                                           0.0),
            bottom_left: ClipCorner::uniform(Rect::new(Point2D::new(rect.origin.x,
                                                                    rect.origin.y + rect.size.height - radius),
                                                       Size2D::new(radius, radius)),
                                             radius,
                                             0.0),
            bottom_right: ClipCorner::uniform(Rect::new(Point2D::new(rect.origin.x + rect.size.width - radius,
                                                                     rect.origin.y + rect.size.height - radius),
                                                        Size2D::new(radius, radius)),
                                              radius,
                                              0.0),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ScreenTileIndex(usize);

#[derive(Debug)]
enum CompiledScreenTileInfo {
    SimpleAlpha(usize),
    ComplexAlpha(usize, usize),
}

#[derive(Debug)]
struct CompiledScreenTile {
    main_render_task: RenderTask,
    required_target_count: usize,
    info: CompiledScreenTileInfo,
}

impl CompiledScreenTile {
    fn new(main_render_task: RenderTask,
           info: CompiledScreenTileInfo) -> CompiledScreenTile {
        let mut required_target_count = 0;
        main_render_task.max_depth(0, &mut required_target_count);

        CompiledScreenTile {
            main_render_task: main_render_task,
            required_target_count: required_target_count,
            info: info,
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
    is_simple: bool,
}

impl ScreenTile {
    fn new(rect: Rect<DevicePixel>) -> ScreenTile {
        ScreenTile {
            rect: rect,
            cmds: Vec::new(),
            prim_count: 0,
            is_simple: true,
        }
    }

    #[inline(always)]
    fn push_layer(&mut self,
                  sc_index: StackingContextIndex,
                  layers: &[StackingContext]) {
        self.cmds.push(TileCommand::PushLayer(sc_index));

        let layer = &layers[sc_index.0];
        match layer.composite_kind {
            CompositeKind::None => {}
            CompositeKind::Simple(..) | CompositeKind::Complex(..) => {
                // Bail out on tiles with composites
                // for now. This can be handled in the future!
                self.is_simple = false;
            }
        }
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

    fn compile(self, ctx: &RenderTargetContext) -> Option<CompiledScreenTile> {
        if self.prim_count == 0 {
            return None;
        }

        let cmd_count = self.cmds.len();
        let mut actual_prim_count = 0;

        let mut sc_stack = Vec::new();
        let mut current_task = RenderTask::new_alpha_batch(self.rect, ctx);
        let mut alpha_task_stack = Vec::new();

        for cmd in self.cmds {
            match cmd {
                TileCommand::PushLayer(sc_index) => {
                    sc_stack.push(sc_index);

                    let layer = &ctx.layer_store[sc_index.0];
                    match layer.composite_kind {
                        CompositeKind::None => {}
                        CompositeKind::Simple(..) | CompositeKind::Complex(..) => {
                            let prev_task = mem::replace(&mut current_task, RenderTask::new_alpha_batch(self.rect, ctx));
                            alpha_task_stack.push(prev_task);
                        }
                    }
                }
                TileCommand::PopLayer => {
                    let sc_index = sc_stack.pop().unwrap();

                    let layer = &ctx.layer_store[sc_index.0];
                    match layer.composite_kind {
                        CompositeKind::None => {}
                        CompositeKind::Simple(info) => {
                            let mut prev_task = alpha_task_stack.pop().unwrap();
                            prev_task.as_alpha_batch().items.push(AlphaRenderItem::Blend(current_task.id,
                                                                                         info));
                            prev_task.children.push(current_task);
                            current_task = prev_task;
                        }
                        CompositeKind::Complex(info) => {
                            let backdrop = alpha_task_stack.pop().unwrap();

                            let mut composite_task = RenderTask::new_alpha_batch(self.rect, ctx);

                            composite_task.as_alpha_batch().items.push(AlphaRenderItem::Composite(backdrop.id,
                                                                                                  current_task.id,
                                                                                                  info));

                            composite_task.children.push(backdrop);
                            composite_task.children.push(current_task);

                            current_task = composite_task;
                        }
                    }
                }
                TileCommand::DrawPrimitive(prim_index) => {
                    let sc_index = *sc_stack.last().unwrap();

                    // TODO(gw): Complex tiles don't currently get
                    // any occlusion culling!
                    if self.is_simple {
                        let layer = &ctx.layer_store[sc_index.0];
                        let prim = &ctx.prim_store[prim_index.0];

                        if layer.xf_rect.as_ref().unwrap().kind == TransformedRectKind::AxisAligned &&
                           prim.complex_clip.is_none() &&
                           prim.is_opaque(ctx.resource_cache, ctx.frame_id) &&
                           prim.bounding_rect.as_ref().unwrap().contains_rect(&self.rect) {
                            current_task.as_alpha_batch().items.clear();
                        }
                    }

                    actual_prim_count += 1;
                    current_task.as_alpha_batch().items.push(AlphaRenderItem::Primitive(sc_index, prim_index));
                }
            }
        }

        debug_assert!(alpha_task_stack.is_empty());

        let info = if self.is_simple {
            CompiledScreenTileInfo::SimpleAlpha(actual_prim_count)
        } else {
            CompiledScreenTileInfo::ComplexAlpha(cmd_count, actual_prim_count)
        };

        current_task.location = RenderTaskLocation::Fixed(self.rect);
        current_task.finalize();
        Some(CompiledScreenTile::new(current_task, info))
    }
}

impl FrameBuilder {
    pub fn new(viewport_size: Size2D<f32>,
               device_pixel_ratio: f32,
               debug: bool,
               _config: FrameBuilderConfig) -> FrameBuilder {
        let viewport_size = Size2D::new(viewport_size.width as i32, viewport_size.height as i32);
        FrameBuilder {
            screen_rect: Rect::new(Point2D::zero(), viewport_size),
            layer_store: Vec::new(),
            prim_store: Vec::new(),
            cmds: Vec::new(),
            device_pixel_ratio: device_pixel_ratio,
            debug: debug,
            packed_layers: Vec::new(),
            scrollbar_prims: Vec::new(),
        }
    }

    fn add_primitive(&mut self,
                     rect: &Rect<f32>,
                     clip_rect: &Rect<f32>,
                     clip: Option<Box<Clip>>,
                     details: PrimitiveDetails) -> PrimitiveIndex {
        let prim = Primitive {
            rect: *rect,
            complex_clip: clip,
            local_clip_rect: *clip_rect,
            details: details,
            bounding_rect: None,
        };
        let prim_index = PrimitiveIndex(self.prim_store.len());
        self.prim_store.push(prim);

        match self.cmds.last_mut().unwrap() {
            &mut PrimitiveRunCmd::PrimitiveRun(_run_prim_index, ref mut count) => {
                debug_assert!(_run_prim_index.0 + *count == prim_index.0);
                *count += 1;
                return prim_index;
            }
            &mut PrimitiveRunCmd::PushStackingContext(..) |
            &mut PrimitiveRunCmd::PopStackingContext => {}
        }

        self.cmds.push(PrimitiveRunCmd::PrimitiveRun(prim_index, 1));

        prim_index
    }

    pub fn push_layer(&mut self,
                      rect: Rect<f32>,
                      clip_rect: Rect<f32>,
                      transform: Matrix4D<f32>,
                      pipeline_id: PipelineId,
                      scroll_layer_id: ScrollLayerId,
                      composition_operations: &[CompositionOp]) {
        let sc_index = StackingContextIndex(self.layer_store.len());

        let sc = StackingContext {
            index: sc_index,
            local_rect: rect,
            local_transform: transform,
            scroll_layer_id: scroll_layer_id,
            pipeline_id: pipeline_id,
            xf_rect: None,
            composite_kind: CompositeKind::new(composition_operations),
            local_clip_rect: clip_rect,
            prims_to_prepare: Vec::new(),
            tile_range: None,
        };
        self.layer_store.push(sc);

        self.packed_layers.push(PackedStackingContext {
            transform: Matrix4D::identity(),
            inv_transform: Matrix4D::identity(),
            screen_vertices: [Point4D::zero(); 4],
            local_clip_rect: Rect::new(Point2D::zero(), Size2D::zero()),
        });

        self.cmds.push(PrimitiveRunCmd::PushStackingContext(sc_index));
    }

    pub fn pop_layer(&mut self) {
        self.cmds.push(PrimitiveRunCmd::PopStackingContext);
    }

    pub fn add_solid_rectangle(&mut self,
                               rect: &Rect<f32>,
                               clip_rect: &Rect<f32>,
                               clip: Option<Box<Clip>>,
                               color: &ColorF,
                               flags: PrimitiveFlags) {
        if color.a == 0.0 {
            return;
        }

        let prim = RectanglePrimitive {
            color: *color,
        };

        let prim_index = self.add_primitive(rect,
                                            clip_rect,
                                            clip,
                                            PrimitiveDetails::Rectangle(prim));

        match flags {
            PrimitiveFlags::None => {}
            PrimitiveFlags::Scrollbar(scroll_layer_id, border_radius) => {
                self.scrollbar_prims.push(ScrollbarPrimitive {
                    prim_index: prim_index,
                    scroll_layer_id: scroll_layer_id,
                    border_radius: border_radius,
                });
            }
        }
    }

    pub fn supported_style(&mut self, border: &BorderSide) -> bool {
        match border.style {
            BorderStyle::Solid |
            BorderStyle::None |
            BorderStyle::Dotted |
            BorderStyle::Dashed |
            BorderStyle::Inset |
            BorderStyle::Outset |
            BorderStyle::Double => {
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
            cache: None,
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
                cache: None,
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
                cache: None,
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
                cache: None,
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

        if size.0 <= 0 {
            return
        }

        let text_run_count = glyph_range.length / GLYPHS_PER_TEXT_RUN;
        let text_count = glyph_range.length % GLYPHS_PER_TEXT_RUN;

        for text_run_index in 0..text_run_count {
            let prim = TextRunPrimitive {
                color: *color,
                font_key: font_key,
                size: size,
                blur_radius: blur_radius,
                glyph_range: ItemRange {
                    start: glyph_range.start + (text_run_index * GLYPHS_PER_TEXT_RUN),
                    length: GLYPHS_PER_TEXT_RUN,
                },
                cache: None,
            };

            self.add_primitive(&rect, clip_rect, clip.clone(), PrimitiveDetails::TextRun(prim));
        }

        for text_index in 0..text_count {
            let prim = TextPrimitive {
                color: *color,
                font_key: font_key,
                size: size,
                blur_radius: blur_radius,
                glyph_index: (glyph_range.start +
                              text_run_count * GLYPHS_PER_TEXT_RUN +
                              text_index) as u32,
                cache: None,
            };

            self.add_primitive(&rect, clip_rect, clip.clone(), PrimitiveDetails::Text(prim));
        }
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
                                     color,
                                     PrimitiveFlags::None);
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
            src_rect: *box_bounds,
            bs_rect: bs_rect,
            color: *color,
            blur_radius: blur_radius,
            spread_radius: spread_radius,
            border_radius: border_radius,
            clip_mode: clip_mode,
            cache: None,
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
            cache: None,
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
                     tile_spacing: &Size2D<f32>,
                     image_key: ImageKey,
                     image_rendering: ImageRendering) {
        let prim = ImagePrimitive {
            kind: ImagePrimitiveKind::Image(image_key,
                                            image_rendering,
                                            stretch_size.clone(),
                                            tile_spacing.clone()),
            cache: None,
        };

        self.add_primitive(&rect,
                           clip_rect,
                           clip,
                           PrimitiveDetails::Image(prim));
    }

    fn cull_layers(&mut self,
                   screen_rect: &Rect<DevicePixel>,
                   layer_map: &HashMap<ScrollLayerId, Layer, BuildHasherDefault<FnvHasher>>,
                   pipeline_auxiliary_lists: &HashMap<PipelineId, AuxiliaryLists, BuildHasherDefault<FnvHasher>>,
                   resource_list: &mut ResourceList,
                   x_tile_count: i32,
                   y_tile_count: i32,
                   profile_counters: &mut FrameProfileCounters) {
        // Build layer screen rects.
        // TODO(gw): This can be done earlier once update_layer_transforms() is fixed.

        // TODO(gw): Remove this stack once the layers refactor is done!
        let mut layer_stack: Vec<StackingContextIndex> = Vec::new();

        for cmd in &self.cmds {
            match cmd {
                &PrimitiveRunCmd::PushStackingContext(sc_index) => {
                    layer_stack.push(sc_index);
                    let layer = &mut self.layer_store[sc_index.0];
                    let packed_layer = &mut self.packed_layers[sc_index.0];

                    layer.xf_rect = None;
                    layer.tile_range = None;

                    if !layer.can_contribute_to_scene() {
                        continue;
                    }

                    let scroll_layer = &layer_map[&layer.scroll_layer_id];
                    packed_layer.transform = scroll_layer.world_content_transform
                                                         .pre_mul(&layer.local_transform);
                    packed_layer.inv_transform = packed_layer.transform.inverse().unwrap();

                    let inv_layer_transform = layer.local_transform.inverse().unwrap();
                    let local_viewport_rect = scroll_layer.combined_local_viewport_rect;
                    let viewport_rect = inv_layer_transform.transform_rect(&local_viewport_rect);
                    let layer_local_rect = layer.local_rect
                                                .intersection(&viewport_rect)
                                                .and_then(|rect| rect.intersection(&layer.local_clip_rect));

                    if let Some(layer_local_rect) = layer_local_rect {
                        let layer_xf_rect = TransformedRect::new(&layer_local_rect,
                                                                 &packed_layer.transform,
                                                                 self.device_pixel_ratio);

                        if layer_xf_rect.bounding_rect.intersects(&screen_rect) {
                            packed_layer.screen_vertices = layer_xf_rect.vertices.clone();
                            packed_layer.local_clip_rect = layer_local_rect;

                            let layer_rect = layer_xf_rect.bounding_rect;
                            layer.xf_rect = Some(layer_xf_rect);

                            let tile_x0 = layer_rect.origin.x.0 / SCREEN_TILE_SIZE;
                            let tile_y0 = layer_rect.origin.y.0 / SCREEN_TILE_SIZE;
                            let tile_x1 = (layer_rect.origin.x.0 + layer_rect.size.width.0 + SCREEN_TILE_SIZE - 1) / SCREEN_TILE_SIZE;
                            let tile_y1 = (layer_rect.origin.y.0 + layer_rect.size.height.0 + SCREEN_TILE_SIZE - 1) / SCREEN_TILE_SIZE;

                            let tile_x0 = cmp::min(tile_x0, x_tile_count);
                            let tile_x0 = cmp::max(tile_x0, 0);
                            let tile_x1 = cmp::min(tile_x1, x_tile_count);
                            let tile_x1 = cmp::max(tile_x1, 0);

                            let tile_y0 = cmp::min(tile_y0, y_tile_count);
                            let tile_y0 = cmp::max(tile_y0, 0);
                            let tile_y1 = cmp::min(tile_y1, y_tile_count);
                            let tile_y1 = cmp::max(tile_y1, 0);

                            layer.tile_range = Some(TileRange {
                                x0: tile_x0,
                                y0: tile_y0,
                                x1: tile_x1,
                                y1: tile_y1,
                            });
                        }
                    }
                }
                &PrimitiveRunCmd::PrimitiveRun(prim_index, prim_count) => {
                    let sc_index = layer_stack.last().unwrap();
                    let layer = &mut self.layer_store[sc_index.0];
                    if !layer.is_visible() {
                        continue;
                    }

                    let packed_layer = &self.packed_layers[sc_index.0];
                    let auxiliary_lists = pipeline_auxiliary_lists.get(&layer.pipeline_id)
                                                                  .expect("No auxiliary lists?!");

                    for i in 0..prim_count {
                        let prim_index = PrimitiveIndex(prim_index.0 + i);
                        let prim = &mut self.prim_store[prim_index.0];
                        prim.rebuild_bounding_rect(screen_rect,
                                                   &packed_layer.transform,
                                                   &packed_layer.local_clip_rect,
                                                   self.device_pixel_ratio);
                        if prim.bounding_rect.is_some() {
                            profile_counters.visible_primitives.inc();

                            if prim.build_resource_list(resource_list, auxiliary_lists) {
                                layer.prims_to_prepare.push(prim_index)
                            }
                        }
                    }
                }
                &PrimitiveRunCmd::PopStackingContext => {
                    layer_stack.pop().unwrap();
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
                                    screen_tiles: &mut Vec<ScreenTile>,
                                    x_tile_count: i32) {
        let mut layer_stack: Vec<StackingContextIndex> = Vec::new();

        for cmd in &self.cmds {
            match cmd {
                &PrimitiveRunCmd::PushStackingContext(sc_index) => {
                    layer_stack.push(sc_index);

                    let layer = &self.layer_store[sc_index.0];
                    if !layer.is_visible() {
                        continue;
                    }

                    let tile_range = layer.tile_range.as_ref().unwrap();
                    for ly in tile_range.y0..tile_range.y1 {
                        for lx in tile_range.x0..tile_range.x1 {
                            let tile = &mut screen_tiles[(ly * x_tile_count + lx) as usize];
                            tile.push_layer(sc_index, &self.layer_store);
                        }
                    }
                }
                &PrimitiveRunCmd::PrimitiveRun(first_prim_index, prim_count) => {
                    let sc_index = layer_stack.last().unwrap();

                    let layer = &self.layer_store[sc_index.0];
                    if !layer.is_visible() {
                        continue;
                    }
                    let packed_layer = &self.packed_layers[sc_index.0];

                    let tile_range = layer.tile_range.as_ref().unwrap();
                    let xf_rect = &layer.xf_rect.as_ref().unwrap();

                    for i in 0..prim_count {
                        let prim_index = PrimitiveIndex(first_prim_index.0 + i);
                        let prim = &self.prim_store[prim_index.0];

                        if let Some(ref p_rect) = prim.bounding_rect {
                            // TODO(gw): Ensure that certain primitives (such as background-image) only get
                            //           assigned to tiles where their containing layer intersects with.
                            //           Does this cause any problems / demonstrate other bugs?
                            //           Restrict the tiles by clamping to the layer tile indices...

                            let p_tile_x0 = p_rect.origin.x.0 / SCREEN_TILE_SIZE;
                            let p_tile_y0 = p_rect.origin.y.0 / SCREEN_TILE_SIZE;
                            let p_tile_x1 = (p_rect.origin.x.0 + p_rect.size.width.0 + SCREEN_TILE_SIZE - 1) / SCREEN_TILE_SIZE;
                            let p_tile_y1 = (p_rect.origin.y.0 + p_rect.size.height.0 + SCREEN_TILE_SIZE - 1) / SCREEN_TILE_SIZE;

                            let p_tile_x0 = cmp::min(p_tile_x0, tile_range.x1);
                            let p_tile_x0 = cmp::max(p_tile_x0, tile_range.x0);
                            let p_tile_x1 = cmp::min(p_tile_x1, tile_range.x1);
                            let p_tile_x1 = cmp::max(p_tile_x1, tile_range.x0);

                            let p_tile_y0 = cmp::min(p_tile_y0, tile_range.y1);
                            let p_tile_y0 = cmp::max(p_tile_y0, tile_range.y0);
                            let p_tile_y1 = cmp::min(p_tile_y1, tile_range.y1);
                            let p_tile_y1 = cmp::max(p_tile_y1, tile_range.y0);

                            for py in p_tile_y0..p_tile_y1 {
                                for px in p_tile_x0..p_tile_x1 {
                                    let tile = &mut screen_tiles[(py * x_tile_count + px) as usize];

                                    // TODO(gw): Support narrow phase for 3d transform elements!
                                    if xf_rect.kind == TransformedRectKind::Complex ||
                                       prim.affects_tile(&tile.rect,
                                                         &packed_layer.transform,
                                                         self.device_pixel_ratio) {
                                        tile.push_primitive(prim_index);
                                    }
                                }
                            }
                        }
                    }
                }
                &PrimitiveRunCmd::PopStackingContext => {
                    let sc_index = layer_stack.pop().unwrap();

                    let layer = &self.layer_store[sc_index.0];
                    if !layer.is_visible() {
                        continue;
                    }

                    let tile_range = layer.tile_range.as_ref().unwrap();
                    for ly in tile_range.y0..tile_range.y1 {
                        for lx in tile_range.x0..tile_range.x1 {
                            let tile = &mut screen_tiles[(ly * x_tile_count + lx) as usize];
                            tile.pop_layer(sc_index);
                        }
                    }
                }
            }
        }
    }

    fn prepare_primitives(&mut self,
                          screen_rect: &Rect<DevicePixel>,
                          resource_cache: &ResourceCache,
                          frame_id: FrameId,
                          pipeline_auxiliary_lists: &HashMap<PipelineId, AuxiliaryLists, BuildHasherDefault<FnvHasher>>) {
        for (layer, packed_layer) in self.layer_store
                                         .iter_mut()
                                         .zip(self.packed_layers.iter()) {
            if !layer.is_visible() {
                continue;
            }

            let auxiliary_lists = pipeline_auxiliary_lists.get(&layer.pipeline_id)
                                                              .expect("No auxiliary lists?!");

            for prim_index in layer.prims_to_prepare.drain(..) {
                let prim = &mut self.prim_store[prim_index.0];
                prim.prepare_for_render(screen_rect,
                                        layer.index,
                                        &packed_layer.transform,
                                        &packed_layer.local_clip_rect,
                                        resource_cache,
                                        frame_id,
                                        self.device_pixel_ratio,
                                        auxiliary_lists);
            }
        }
    }

    fn update_scroll_bars(&mut self,
                          layer_map: &HashMap<ScrollLayerId, Layer, BuildHasherDefault<FnvHasher>>) {
        let distance_from_edge = 8.0;

        for scrollbar_prim in &self.scrollbar_prims {
            let prim = &mut self.prim_store[scrollbar_prim.prim_index.0];
            let scroll_layer = &layer_map[&scrollbar_prim.scroll_layer_id];

            let scrollable_distance = scroll_layer.content_size.height - scroll_layer.local_viewport_rect.size.height;

            if scrollable_distance <= 0.0 {
                prim.local_clip_rect.size = Size2D::zero();
                continue;
            }

            let f = -scroll_layer.scrolling.offset.y / scrollable_distance;

            let min_y = scroll_layer.local_viewport_rect.origin.y -
                        scroll_layer.scrolling.offset.y +
                        distance_from_edge;

            let max_y = scroll_layer.local_viewport_rect.origin.y +
                        scroll_layer.local_viewport_rect.size.height -
                        scroll_layer.scrolling.offset.y -
                        prim.rect.size.height -
                        distance_from_edge;

            prim.rect.origin.x = scroll_layer.local_viewport_rect.origin.x +
                                 scroll_layer.local_viewport_rect.size.width -
                                 prim.rect.size.width -
                                 distance_from_edge;

            prim.rect.origin.y = util::lerp(min_y, max_y, f);
            prim.local_clip_rect = prim.rect;

            if scrollbar_prim.border_radius == 0.0 {
                prim.complex_clip = None;
            } else {
                prim.complex_clip = Some(Box::new(Clip::uniform(prim.rect,
                                                                scrollbar_prim.border_radius)));
            }
        }
    }

    pub fn build(&mut self,
                 resource_cache: &mut ResourceCache,
                 frame_id: FrameId,
                 pipeline_auxiliary_lists: &HashMap<PipelineId, AuxiliaryLists, BuildHasherDefault<FnvHasher>>,
                 layer_map: &HashMap<ScrollLayerId, Layer, BuildHasherDefault<FnvHasher>>) -> Frame {
        let mut profile_counters = FrameProfileCounters::new();
        profile_counters.total_primitives.set(self.prim_store.len());

        let screen_rect = Rect::new(Point2D::zero(),
                                    Size2D::new(DevicePixel::new(self.screen_rect.size.width as f32, self.device_pixel_ratio),
                                                DevicePixel::new(self.screen_rect.size.height as f32, self.device_pixel_ratio)));

        let mut resource_list = ResourceList::new();
        let mut debug_rects = Vec::new();

        let (x_tile_count, y_tile_count, mut screen_tiles) = self.create_screen_tiles();

        self.update_scroll_bars(layer_map);

        self.cull_layers(&screen_rect,
                         layer_map,
                         pipeline_auxiliary_lists,
                         &mut resource_list,
                         x_tile_count,
                         y_tile_count,
                         &mut profile_counters);

        resource_cache.add_resource_list(&resource_list, frame_id);
        resource_cache.raster_pending_glyphs(frame_id);

        self.prepare_primitives(&screen_rect,
                                resource_cache,
                                frame_id,
                                pipeline_auxiliary_lists);

        let ctx = RenderTargetContext {
            layer_store: &self.layer_store,
            prim_store: &self.prim_store,
            resource_cache: resource_cache,
            frame_id: frame_id,

            // This doesn't need to be atomic right now (all the screen tiles are
            // compiled on a single thread). However, in the future each of the
            // compile steps below will be run on a worker thread, which will
            // require an atomic int here anyway.
            render_task_id_counter: AtomicUsize::new(0),
        };

        if !self.layer_store.is_empty() {
            self.assign_prims_to_screen_tiles(&mut screen_tiles, x_tile_count);
        }

        let mut clear_tiles = Vec::new();

        // Build list of passes, target allocs that each tile needs.
        let mut compiled_screen_tiles = Vec::new();
        for screen_tile in screen_tiles {
            let rect = screen_tile.rect;        // TODO(gw): Remove clone here
            match screen_tile.compile(&ctx) {
                Some(compiled_screen_tile) => {
                    if self.debug {
                        let (label, color) = match &compiled_screen_tile.info {
                            &CompiledScreenTileInfo::SimpleAlpha(prim_count) => {
                                (format!("{}", prim_count), ColorF::new(1.0, 0.0, 1.0, 1.0))
                            }
                            &CompiledScreenTileInfo::ComplexAlpha(cmd_count, prim_count) => {
                                (format!("{}|{}", cmd_count, prim_count), ColorF::new(1.0, 0.0, 0.0, 1.0))
                            }
                        };
                        debug_rects.push(DebugRect {
                            label: label,
                            color: color,
                            rect: rect,
                        });
                    }
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
        let static_render_task_count = ctx.render_task_id_counter.load(Ordering::SeqCst);
        let mut render_tasks = RenderTaskCollection::new(static_render_task_count);

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
                if let Some(failed_tile) = current_phase.add_compiled_screen_tile(compiled_screen_tile,
                                                                                  &mut render_tasks) {
                    let full_phase = mem::replace(&mut current_phase,
                                                  RenderPhase::new(failed_tile.required_target_count));
                    phases.push(full_phase);

                    let result = current_phase.add_compiled_screen_tile(failed_tile,
                                                                        &mut render_tasks);
                    assert!(result.is_none(), "TODO: Handle single tile not fitting in render phase.");
                }
            }

            phases.push(current_phase);

            //println!("rendering: phase count={}", phases.len());
            for phase in &mut phases {
                phase.build(&ctx, &mut render_tasks);

                profile_counters.phases.inc();
                profile_counters.targets.add(phase.targets.len());
            }
        }

        Frame {
            viewport_size: self.screen_rect.size,
            debug_rects: debug_rects,
            profile_counters: profile_counters,
            phases: phases,
            clear_tiles: clear_tiles,
            cache_size: Size2D::new(RENDERABLE_CACHE_SIZE.0 as f32,
                                    RENDERABLE_CACHE_SIZE.0 as f32),
            layer_texture_data: self.packed_layers.clone(),
            render_task_data: render_tasks.render_task_data,
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

struct ImageInfo {
    color_texture_id: TextureId,
    uv0: Point2D<f32>,
    uv1: Point2D<f32>,
    stretch_size: Option<Size2D<f32>>,
    uv_kind: TextureCoordKind,
    tile_spacing: Size2D<f32>,
}

impl ImagePrimitive {
    fn image_info(&self, resource_cache: &ResourceCache, frame_id: FrameId) -> ImageInfo {
        match self.kind {
            ImagePrimitiveKind::Image(image_key, image_rendering, stretch_size, tile_spacing) => {
                let info = resource_cache.get_image(image_key, image_rendering, frame_id);
                ImageInfo {
                    color_texture_id: info.texture_id,
                    uv0: Point2D::new(info.pixel_rect.top_left.x.0 as f32,
                                      info.pixel_rect.top_left.y.0 as f32),
                    uv1: Point2D::new(info.pixel_rect.bottom_right.x.0 as f32,
                                      info.pixel_rect.bottom_right.y.0 as f32),
                    stretch_size: Some(stretch_size),
                    uv_kind: TextureCoordKind::Pixel,
                    tile_spacing: tile_spacing,
                }
            }
            ImagePrimitiveKind::WebGL(context_id) => {
                ImageInfo {
                    color_texture_id: resource_cache.get_webgl_texture(&context_id),
                    uv0: Point2D::new(0.0, 1.0),
                    uv1: Point2D::new(1.0, 0.0),
                    stretch_size: None,
                    uv_kind: TextureCoordKind::Normalized,
                    tile_spacing: Size2D::zero(),
                }
            }
        }
    }
}
