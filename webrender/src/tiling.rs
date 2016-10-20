/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use batch_builder::{BorderSideHelpers, BoxShadowMetrics};
use device::{TextureId};
use euclid::{Point2D, Point4D, Rect, Matrix4D, Size2D};
use fnv::FnvHasher;
use frame::FrameId;
use gpu_store::GpuStoreAddress;
use internal_types::{DevicePixel, CompositionOp};
use internal_types::{ANGLE_FLOAT_TO_FIXED, LowLevelFilterOp};
use layer::Layer;
use prim_store::{PrimitiveGeometry, RectanglePrimitive, PrimitiveContainer};
use prim_store::{BorderPrimitiveCpu, BorderPrimitiveGpu, BoxShadowPrimitive};
use prim_store::{Clip, ImagePrimitiveCpu, ImagePrimitiveKind};
use prim_store::{PrimitiveKind, PrimitiveIndex, PrimitiveMetadata};
use prim_store::{GradientPrimitiveCpu, GradientPrimitiveGpu, GradientType};
use prim_store::{TextRunPrimitiveGpu, TextRunPrimitiveCpu};
use prim_store::{PrimitiveStore, GpuBlock16, GpuBlock32, GpuBlock64, GpuBlock128};
use profiler::FrameProfileCounters;
use resource_cache::ResourceCache;
use resource_list::ResourceList;
use std::cmp;
use std::collections::{HashMap};
use std::f32;
use std::mem;
use std::hash::{BuildHasherDefault};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::usize;
use texture_cache::TexturePage;
use util::{self, rect_from_points, MatrixHelpers, rect_from_points_f};
use util::{TransformedRect, TransformedRectKind, subtract_rect, pack_as_float};
use webrender_traits::{ColorF, FontKey, ImageKey, ImageRendering, ComplexClipRegion};
use webrender_traits::{BorderDisplayItem, BorderStyle, ItemRange, AuxiliaryLists, BorderSide};
use webrender_traits::{BoxShadowClipMode, PipelineId, ScrollLayerId, WebGLContextId};

const ALPHA_BATCHERS_PER_RENDER_TARGET: usize = 4;
const MIN_TASKS_PER_ALPHA_BATCHER: usize = 64;
const FLOATS_PER_RENDER_TASK_INFO: usize = 8;

trait AlphaBatchHelpers {
    fn get_batch_info(&self, metadata: &PrimitiveMetadata) -> (AlphaBatchKind, TextureId);
    fn prim_affects_tile(&self,
                         prim_index: PrimitiveIndex,
                         tile_rect: &Rect<DevicePixel>,
                         transform: &Matrix4D<f32>,
                         device_pixel_ratio: f32) -> bool;
    fn add_prim_to_batch(&self,
                         prim_index: PrimitiveIndex,
                         batch: &mut PrimitiveBatch,
                         layer_index: StackingContextIndex,
                         task_id: i32,
                         transform_kind: TransformedRectKind,
                         needs_blending: bool);
}

impl AlphaBatchHelpers for PrimitiveStore {
    fn get_batch_info(&self, metadata: &PrimitiveMetadata) -> (AlphaBatchKind, TextureId) {
        let batch_kind = match metadata.prim_kind {
            PrimitiveKind::Border => AlphaBatchKind::Border,
            PrimitiveKind::BoxShadow => AlphaBatchKind::BoxShadow,
            PrimitiveKind::Image => AlphaBatchKind::Image,
            PrimitiveKind::Rectangle => AlphaBatchKind::Rectangle,
            PrimitiveKind::TextRun => AlphaBatchKind::TextRun,
            PrimitiveKind::Gradient => {
                let gradient = &self.cpu_gradients[metadata.cpu_prim_index.0];
                match gradient.kind {
                    GradientType::Horizontal | GradientType::Vertical => {
                        AlphaBatchKind::AlignedGradient
                    }
                    GradientType::Rotated => {
                        AlphaBatchKind::AngleGradient
                    }
                }
            }
        };

        (batch_kind, metadata.color_texture_id)
    }

    // Optional narrow phase intersection test, depending on primitive type.
    fn prim_affects_tile(&self,
                         prim_index: PrimitiveIndex,
                         tile_rect: &Rect<DevicePixel>,
                         transform: &Matrix4D<f32>,
                         device_pixel_ratio: f32) -> bool {
        let metadata = self.get_metadata(prim_index);

        match metadata.prim_kind {
            PrimitiveKind::Rectangle |
            PrimitiveKind::TextRun |
            PrimitiveKind::Image |
            PrimitiveKind::Gradient |
            PrimitiveKind::BoxShadow => true,

            PrimitiveKind::Border => {
                let border = &self.cpu_borders[metadata.cpu_prim_index.0];
                let inner_rect = TransformedRect::new(&border.inner_rect,
                                                      transform,
                                                      device_pixel_ratio);

                !inner_rect.bounding_rect.contains_rect(tile_rect)
            }
        }
    }

    fn add_prim_to_batch(&self,
                         prim_index: PrimitiveIndex,
                         batch: &mut PrimitiveBatch,
                         layer_index: StackingContextIndex,
                         task_id: i32,
                         transform_kind: TransformedRectKind,
                         needs_blending: bool) {
        debug_assert!(transform_kind == batch.transform_kind);
        debug_assert!(needs_blending == batch.blending_enabled);

        let metadata = self.get_metadata(prim_index);
        let layer_index = layer_index.0 as i32;
        let global_prim_id = prim_index.0 as i32;
        let prim_address = metadata.gpu_prim_index;
        let clip_address = metadata.clip_index.unwrap_or(GpuStoreAddress(0));

        match &mut batch.data {
            &mut PrimitiveBatchData::Blend(..) |
            &mut PrimitiveBatchData::Composite(..) => unreachable!(),

            &mut PrimitiveBatchData::Rectangles(ref mut data) => {
                data.push(PrimitiveInstance {
                    task_id: task_id,
                    layer_index: layer_index,
                    global_prim_id: global_prim_id,
                    prim_address: prim_address,
                    clip_address: clip_address,
                    user_data: [0, 0, 0],
                });
            }
            &mut PrimitiveBatchData::TextRun(ref mut data) => {
                let metadata = self.get_metadata(prim_index);

                for glyph_index in 0..metadata.gpu_data_count {
                    data.push(PrimitiveInstance {
                        task_id: task_id,
                        layer_index: layer_index,
                        global_prim_id: global_prim_id,
                        prim_address: prim_address,
                        clip_address: clip_address,
                        user_data: [ metadata.gpu_data_address.0, glyph_index, 0 ],
                    });
                }
            }
            &mut PrimitiveBatchData::Image(ref mut data) => {
                data.push(PrimitiveInstance {
                    task_id: task_id,
                    layer_index: layer_index,
                    global_prim_id: global_prim_id,
                    prim_address: prim_address,
                    clip_address: clip_address,
                    user_data: [ 0, 0, 0 ],
                });
            }
            &mut PrimitiveBatchData::Borders(ref mut data) => {
                for border_segment in 0..8 {
                    data.push(PrimitiveInstance {
                        task_id: task_id,
                        layer_index: layer_index,
                        global_prim_id: global_prim_id,
                        prim_address: prim_address,
                        clip_address: clip_address,
                        user_data: [ border_segment, 0, 0 ],
                    });
                }
            }
            &mut PrimitiveBatchData::AlignedGradient(ref mut data) => {
                let metadata = self.get_metadata(prim_index);

                for part_index in 0..(metadata.gpu_data_count - 1) {
                    data.push(PrimitiveInstance {
                        task_id: task_id,
                        layer_index: layer_index,
                        global_prim_id: global_prim_id,
                        prim_address: prim_address,
                        clip_address: clip_address,
                        user_data: [ metadata.gpu_data_address.0, part_index, 0 ],
                    });
                }
            }
            &mut PrimitiveBatchData::AngleGradient(ref mut data) => {
                let metadata = self.get_metadata(prim_index);

                data.push(PrimitiveInstance {
                    task_id: task_id,
                    layer_index: layer_index,
                    global_prim_id: global_prim_id,
                    prim_address: prim_address,
                    clip_address: clip_address,
                    user_data: [ metadata.gpu_data_address.0, metadata.gpu_data_count, 0 ],
                });
            }
             &mut PrimitiveBatchData::BoxShadow(ref mut data) => {
                let metadata = self.get_metadata(prim_index);

                for rect_index in 0..metadata.gpu_data_count {
                    data.push(PrimitiveInstance {
                        task_id: task_id,
                        layer_index: layer_index,
                        global_prim_id: global_prim_id,
                        prim_address: prim_address,
                        clip_address: clip_address,
                        user_data: [ metadata.gpu_data_address.0, rect_index, 0 ],
                    });
                }
            }
        }
    }
}

#[derive(Debug)]
struct ScrollbarPrimitive {
    scroll_layer_id: ScrollLayerId,
    prim_index: PrimitiveIndex,
    border_radius: f32,
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

impl Default for PrimitiveGeometry {
    fn default() -> PrimitiveGeometry {
        PrimitiveGeometry {
            local_rect: unsafe { mem::uninitialized() },
            local_clip_rect: unsafe { mem::uninitialized() },
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
            let task_index = task_index.0 as i32;

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
                        let prim_metadata = ctx.prim_store.get_metadata(prim_index);
                        let transform_kind = layer.xf_rect.as_ref().unwrap().kind;
                        let needs_clipping = prim_metadata.clip_index.is_some();
                        let needs_blending = transform_kind == TransformedRectKind::Complex ||
                                             !prim_metadata.is_opaque ||
                                             needs_clipping;
                        let flags = AlphaBatchKeyFlags::new(transform_kind,
                                                            needs_blending,
                                                            needs_clipping);
                        let (batch_kind, color_texture_id) = ctx.prim_store
                                                                .get_batch_info(prim_metadata);
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
                            let prim_metadata = ctx.prim_store.get_metadata(prim_index);
                            let (batch_kind, color_texture_id) = ctx.prim_store
                                                                    .get_batch_info(prim_metadata);
                            PrimitiveBatch::new(batch_kind,
                                                batch_key.flags.transform_kind(),
                                                batch_key.flags.needs_blending(),
                                                batch_key.flags.needs_clipping(),
                                                color_texture_id)
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
                        ctx.prim_store.add_prim_to_batch(prim_index,
                                                         batch,
                                                         sc_index,
                                                         task_index,
                                                         batch_key.flags.transform_kind(),
                                                         batch_key.flags.needs_blending());
                    }
                }
            }
        }

        self.batches.extend(batches.into_iter().map(|(_, batch)| batch))
    }
}

struct RenderTargetContext<'a> {
    layer_store: &'a Vec<StackingContext>,
    prim_store: &'a PrimitiveStore,
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

#[derive(Debug, Clone)]
pub struct DebugRect {
    pub label: String,
    pub color: ColorF,
    pub rect: Rect<DevicePixel>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
enum AlphaBatchKind {
    Composite = 0,
    Blend,
    Rectangle,
    TextRun,
    Image,
    Border,
    AlignedGradient,
    AngleGradient,
    BoxShadow,
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

// FIXME(gw): Change these to use the bitflags!()

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct AlphaBatchKeyFlags(u8);

impl AlphaBatchKeyFlags {
    fn new(transform_kind: TransformedRectKind,
           needs_blending: bool,
           needs_clipping: bool) -> AlphaBatchKeyFlags {
        AlphaBatchKeyFlags( ((needs_clipping as u8) << 2) |
                            ((transform_kind as u8) << 1) |
                            ((needs_blending as u8) << 0) )
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

    fn needs_clipping(&self) -> bool {
        (self.0 & 4) != 0
    }
}

// All Packed Primitives below must be 16 byte aligned.
#[derive(Debug, Clone)]
pub struct PrimitiveInstance {
    global_prim_id: i32,
    prim_address: GpuStoreAddress,
    task_id: i32,
    layer_index: i32,
    clip_address: GpuStoreAddress,
    user_data: [i32; 3],
}

#[derive(Debug, Clone)]
pub struct PackedBlendPrimitive {
    src_task_id: i32,
    target_task_id: i32,
    brightness: i32,
    opacity: i32,
}

#[derive(Debug, Copy, Clone)]
struct PackedCompositeInfo {
    kind: i32,
    op: i32,
    amount: i32,
    padding: i32,
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
            kind: kind,
            op: op as i32,
            amount: (amount * 65536.0).round() as i32,
            padding: 0,
        }
    }
}

#[derive(Debug)]
pub struct PackedCompositePrimitive {
    src0_task_id: i32,
    src1_task_id: i32,
    target_task_id: i32,
    padding: i32,
    info: PackedCompositeInfo,
}

#[derive(Debug)]
pub enum PrimitiveBatchData {
    Rectangles(Vec<PrimitiveInstance>),
    TextRun(Vec<PrimitiveInstance>),
    Image(Vec<PrimitiveInstance>),
    Borders(Vec<PrimitiveInstance>),
    AlignedGradient(Vec<PrimitiveInstance>),
    AngleGradient(Vec<PrimitiveInstance>),
    BoxShadow(Vec<PrimitiveInstance>),
    Blend(Vec<PackedBlendPrimitive>),
    Composite(Vec<PackedCompositePrimitive>),
}

#[derive(Debug)]
pub struct PrimitiveBatch {
    pub transform_kind: TransformedRectKind,
    pub has_complex_clip: bool,
    pub color_texture_id: TextureId,        // TODO(gw): Expand to sampler array to handle all glyphs!
    pub blending_enabled: bool,
    pub data: PrimitiveBatchData,
}

impl PrimitiveBatch {
    fn blend() -> PrimitiveBatch {
        PrimitiveBatch {
            color_texture_id: TextureId(0),
            transform_kind: TransformedRectKind::AxisAligned,
            has_complex_clip: false,
            blending_enabled: true,
            data: PrimitiveBatchData::Blend(Vec::new()),
        }
    }

    fn composite() -> PrimitiveBatch {
        PrimitiveBatch {
            color_texture_id: TextureId(0),
            transform_kind: TransformedRectKind::AxisAligned,
            has_complex_clip: false,
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
                    src_task_id: src_rect_index.0 as i32,
                    target_task_id: target_rect_index.0 as i32,
                    opacity: (opacity * 65535.0).round() as i32,
                    brightness: (brightness * 65535.0).round() as i32,
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
                    src0_task_id: rect0_index.0 as i32,
                    src1_task_id: rect1_index.0 as i32,
                    target_task_id: target_rect_index.0 as i32,
                    padding: 0,
                    info: info,
                });

                true
            }
            _ => false
        }
    }

    fn new(batch_kind: AlphaBatchKind,
           transform_kind: TransformedRectKind,
           blending_enabled: bool,
           has_complex_clip: bool,
           color_texture_id: TextureId) -> PrimitiveBatch {
        let data = match batch_kind {
            AlphaBatchKind::Rectangle => PrimitiveBatchData::Rectangles(Vec::new()),
            AlphaBatchKind::TextRun => PrimitiveBatchData::TextRun(Vec::new()),
            AlphaBatchKind::Image => PrimitiveBatchData::Image(Vec::new()),
            AlphaBatchKind::Border => PrimitiveBatchData::Borders(Vec::new()),
            AlphaBatchKind::AlignedGradient => PrimitiveBatchData::AlignedGradient(Vec::new()),
            AlphaBatchKind::AngleGradient => PrimitiveBatchData::AngleGradient(Vec::new()),
            AlphaBatchKind::BoxShadow => PrimitiveBatchData::BoxShadow(Vec::new()),
            AlphaBatchKind::Blend | AlphaBatchKind::Composite => unreachable!(),
        };

        PrimitiveBatch {
            color_texture_id: color_texture_id,
            transform_kind: transform_kind,
            blending_enabled: blending_enabled,
            has_complex_clip: has_complex_clip,
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
    prim_store: PrimitiveStore,
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
    pub gpu_data16: Vec<GpuBlock16>,
    pub gpu_data32: Vec<GpuBlock32>,
    pub gpu_data64: Vec<GpuBlock64>,
    pub gpu_data128: Vec<GpuBlock128>,
    pub gpu_geometry: Vec<PrimitiveGeometry>,
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

                        let prim_metadata = ctx.prim_store.get_metadata(prim_index);
                        let prim_bounding_rect = ctx.prim_store.get_bounding_rect(prim_index);

                        if layer.xf_rect.as_ref().unwrap().kind == TransformedRectKind::AxisAligned &&
                           prim_metadata.clip_index.is_none() &&
                           prim_metadata.is_opaque &&
                           prim_bounding_rect.as_ref().unwrap().contains_rect(&self.rect) {
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
            prim_store: PrimitiveStore::new(),
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
                     container: PrimitiveContainer) -> PrimitiveIndex {
        let prim_index = self.prim_store.add_primitive(rect,
                                                       clip_rect,
                                                       clip,
                                                       container);

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
                                            PrimitiveContainer::Rectangle(prim));

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

        // These colors are used during inset/outset scaling.
        let left_color      = left.border_color(1.0, 2.0/3.0, 0.3, 0.7);
        let top_color       = top.border_color(1.0, 2.0/3.0, 0.3, 0.7);
        let right_color     = right.border_color(2.0/3.0, 1.0, 0.7, 0.3);
        let bottom_color    = bottom.border_color(2.0/3.0, 1.0, 0.7, 0.3);

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

        let inner_rect = rect_from_points_f(tl_inner.x.max(bl_inner.x),
                                            tl_inner.y.max(tr_inner.y),
                                            tr_inner.x.min(br_inner.x),
                                            bl_inner.y.min(br_inner.y));

        let prim_cpu = BorderPrimitiveCpu {
            inner_rect: inner_rect,
        };

        let prim_gpu = BorderPrimitiveGpu {
            colors: [ left_color, top_color, right_color, bottom_color ],
            widths: [ left.width, top.width, right.width, bottom.width ],
            style: [
                pack_as_float(left.style as u32),
                pack_as_float(top.style as u32),
                pack_as_float(right.style as u32),
                pack_as_float(bottom.style as u32),
            ],
            radii: [
                radius.top_left,
                radius.top_right,
                radius.bottom_right,
                radius.bottom_left,
            ],
        };

        self.add_primitive(&rect,
                           clip_rect,
                           clip,
                           PrimitiveContainer::Border(prim_cpu, prim_gpu));
    }

    pub fn add_gradient(&mut self,
                        rect: Rect<f32>,
                        clip_rect: &Rect<f32>,
                        clip: Option<Box<Clip>>,
                        start_point: Point2D<f32>,
                        end_point: Point2D<f32>,
                        stops: ItemRange) {
        // Fast paths for axis-aligned gradients:
        let mut reverse_stops = false;
        let kind = if start_point.x == end_point.x {
            GradientType::Vertical
        } else if start_point.y == end_point.y {
            GradientType::Horizontal
        } else {
            reverse_stops = start_point.x > end_point.x;
            GradientType::Rotated
        };

        let gradient_cpu = GradientPrimitiveCpu {
            stops_range: stops,
            kind: kind,
            reverse_stops: reverse_stops,
        };

        // To get reftests exactly matching with reverse start/end
        // points, it's necessary to reverse the gradient
        // line in some cases.
        let (sp, ep) = if reverse_stops {
            (end_point, start_point)
        } else {
            (start_point, end_point)
        };

        // TODO(gw): The gradient shader only has a clip variant
        // right now. So add an invalid clip if none is provided.
        // Remove this when a non-clip gradient shader is added.
        let clip = Some(clip.unwrap_or(Box::new(Clip::invalid(rect))));

        let gradient_gpu = GradientPrimitiveGpu {
            start_point: sp,
            end_point: ep,
            padding: [0.0, 0.0, 0.0],
            kind: pack_as_float(kind as u32),
        };

        self.add_primitive(&rect,
                           clip_rect,
                           clip,
                           PrimitiveContainer::Gradient(gradient_cpu, gradient_gpu));
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

        let text_run_count = (glyph_range.length + 7) / 8;
        for run_index in 0..text_run_count {
            let start = run_index * 8;
            let end = cmp::min(start + 8, glyph_range.length);
            let sub_range = ItemRange {
                start: glyph_range.start + start,
                length: end - start,
            };

            let prim_cpu = TextRunPrimitiveCpu {
                font_key: font_key,
                font_size: size,
                blur_radius: blur_radius,
                glyph_range: sub_range,
            };

            let prim_gpu = TextRunPrimitiveGpu {
                color: *color,
            };

            self.add_primitive(&rect,
                               clip_rect,
                               clip.clone(),
                               PrimitiveContainer::TextRun(prim_cpu, prim_gpu));
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

        let mut instance_rects = Vec::new();
        let (prim_rect, inverted) = match clip_mode {
            BoxShadowClipMode::Outset | BoxShadowClipMode::None => {
                let prim_rect = Rect::new(metrics.tl_outer, Size2D::new(metrics.br_outer.x - metrics.tl_outer.x,
                                                                        metrics.br_outer.y - metrics.tl_outer.y));
                subtract_rect(&prim_rect, box_bounds, &mut instance_rects);
                (prim_rect, 0.0)
            }
            BoxShadowClipMode::Inset => {
                subtract_rect(box_bounds, &bs_rect, &mut instance_rects);
                (*box_bounds, 1.0)
            }
        };

        let prim = BoxShadowPrimitive {
            src_rect: *box_bounds,
            bs_rect: bs_rect,
            color: *color,
            blur_radius: blur_radius,
            border_radii: Point2D::new(border_radius, border_radius),
            inverted: inverted,
        };

        self.add_primitive(&prim_rect,
                           clip_rect,
                           clip,
                           PrimitiveContainer::BoxShadow(prim, instance_rects));
    }

    pub fn add_webgl_rectangle(&mut self,
                               rect: Rect<f32>,
                               clip_rect: &Rect<f32>,
                               clip: Option<Box<Clip>>,
                               context_id: WebGLContextId) {
        let prim_cpu = ImagePrimitiveCpu {
            kind: ImagePrimitiveKind::WebGL(context_id),
        };

        self.add_primitive(&rect,
                           clip_rect,
                           clip,
                           PrimitiveContainer::Image(prim_cpu));
    }

    pub fn add_image(&mut self,
                     rect: Rect<f32>,
                     clip_rect: &Rect<f32>,
                     clip: Option<Box<Clip>>,
                     stretch_size: &Size2D<f32>,
                     tile_spacing: &Size2D<f32>,
                     image_key: ImageKey,
                     image_rendering: ImageRendering) {
        let prim_cpu = ImagePrimitiveCpu {
            kind: ImagePrimitiveKind::Image(image_key,
                                            image_rendering,
                                            stretch_size.clone(),
                                            tile_spacing.clone()),
        };

        self.add_primitive(&rect,
                           clip_rect,
                           clip,
                           PrimitiveContainer::Image(prim_cpu));
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
                        if self.prim_store.build_bounding_rect(prim_index,
                                                               screen_rect,
                                                               &packed_layer.transform,
                                                               &packed_layer.local_clip_rect,
                                                               self.device_pixel_ratio) {
                            profile_counters.visible_primitives.inc();

                            if self.prim_store.build_resource_list(prim_index,
                                                                   resource_list,
                                                                   auxiliary_lists) {
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
                        if let &Some(p_rect) = self.prim_store.get_bounding_rect(prim_index) {
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
                                       self.prim_store.prim_affects_tile(prim_index,
                                                                         &tile.rect,
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
                if self.prim_store.prepare_prim_for_render(prim_index,
                                                           resource_cache,
                                                           frame_id,
                                                           self.device_pixel_ratio,
                                                           auxiliary_lists) {
                    self.prim_store.build_bounding_rect(prim_index,
                                                        screen_rect,
                                                        &packed_layer.transform,
                                                        &packed_layer.local_clip_rect,
                                                        self.device_pixel_ratio);
                }
            }
        }
    }

    fn update_scroll_bars(&mut self,
                          layer_map: &HashMap<ScrollLayerId, Layer, BuildHasherDefault<FnvHasher>>) {
        let distance_from_edge = 8.0;

        for scrollbar_prim in &self.scrollbar_prims {
            let mut geom = (*self.prim_store.gpu_geometry.get(GpuStoreAddress(scrollbar_prim.prim_index.0 as i32))).clone();
            let scroll_layer = &layer_map[&scrollbar_prim.scroll_layer_id];

            let scrollable_distance = scroll_layer.content_size.height - scroll_layer.local_viewport_rect.size.height;

            if scrollable_distance <= 0.0 {
                geom.local_clip_rect.size = Size2D::zero();
                *self.prim_store.gpu_geometry.get_mut(GpuStoreAddress(scrollbar_prim.prim_index.0 as i32)) = geom;
                continue;
            }

            let f = -scroll_layer.scrolling.offset.y / scrollable_distance;

            let min_y = scroll_layer.local_viewport_rect.origin.y -
                        scroll_layer.scrolling.offset.y +
                        distance_from_edge;

            let max_y = scroll_layer.local_viewport_rect.origin.y +
                        scroll_layer.local_viewport_rect.size.height -
                        scroll_layer.scrolling.offset.y -
                        geom.local_rect.size.height -
                        distance_from_edge;

            geom.local_rect.origin.x = scroll_layer.local_viewport_rect.origin.x +
                                       scroll_layer.local_viewport_rect.size.width -
                                       geom.local_rect.size.width -
                                       distance_from_edge;

            geom.local_rect.origin.y = util::lerp(min_y, max_y, f);
            geom.local_clip_rect = geom.local_rect;

            if scrollbar_prim.border_radius == 0.0 {
                self.prim_store.set_complex_clip(scrollbar_prim.prim_index, None);
            } else {
                let clip = Clip::uniform(geom.local_rect, scrollbar_prim.border_radius);
                self.prim_store.set_complex_clip(scrollbar_prim.prim_index, Some(clip));
            }
            *self.prim_store.gpu_geometry.get_mut(GpuStoreAddress(scrollbar_prim.prim_index.0 as i32)) = geom;
        }
    }

    pub fn build(&mut self,
                 resource_cache: &mut ResourceCache,
                 frame_id: FrameId,
                 pipeline_auxiliary_lists: &HashMap<PipelineId, AuxiliaryLists, BuildHasherDefault<FnvHasher>>,
                 layer_map: &HashMap<ScrollLayerId, Layer, BuildHasherDefault<FnvHasher>>) -> Frame {
        let mut profile_counters = FrameProfileCounters::new();
        profile_counters.total_primitives.set(self.prim_store.prim_count());

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
            gpu_data16: self.prim_store.gpu_data16.build(),
            gpu_data32: self.prim_store.gpu_data32.build(),
            gpu_data64: self.prim_store.gpu_data64.build(),
            gpu_data128: self.prim_store.gpu_data128.build(),
            gpu_geometry: self.prim_store.gpu_geometry.build(),
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
