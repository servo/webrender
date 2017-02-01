/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use batch_builder::BorderSideHelpers;
use fnv::FnvHasher;
use frame::FrameId;
use gpu_store::GpuStoreAddress;
use internal_types::{ANGLE_FLOAT_TO_FIXED, LowLevelFilterOp};
use internal_types::{BatchTextures, CacheTextureId, SourceTexture};
use mask_cache::{ClipSource, MaskCacheInfo};
use prim_store::{PrimitiveGeometry, RectanglePrimitive, PrimitiveContainer};
use prim_store::{BorderPrimitiveCpu, BorderPrimitiveGpu, BoxShadowPrimitiveGpu};
use prim_store::{ImagePrimitiveCpu, ImagePrimitiveGpu, YuvImagePrimitiveCpu, YuvImagePrimitiveGpu, ImagePrimitiveKind, };
use prim_store::{PrimitiveKind, PrimitiveIndex, PrimitiveMetadata, TexelRect};
use prim_store::{CLIP_DATA_GPU_SIZE, DeferredResolve};
use prim_store::{GradientPrimitiveCpu, GradientPrimitiveGpu, GradientData};
use prim_store::{RadialGradientPrimitiveCpu, RadialGradientPrimitiveGpu};
use prim_store::{PrimitiveCacheKey, TextRunPrimitiveGpu, TextRunPrimitiveCpu};
use prim_store::{PrimitiveStore, GpuBlock16, GpuBlock32, GpuBlock64, GpuBlock128};
use profiler::FrameProfileCounters;
use renderer::BlendMode;
use resource_cache::ResourceCache;
use scroll_tree::ScrollTree;
use std::cmp;
use std::collections::{HashMap};
use std::{i32, f32};
use std::mem;
use std::hash::{BuildHasherDefault};
use std::usize;
use texture_cache::TexturePage;
use util::{self, pack_as_float, rect_from_points_f, subtract_rect};
use util::{TransformedRect, TransformedRectKind};
use webrender_traits::{ColorF, ExtendMode, FontKey, ImageKey, ImageRendering, MixBlendMode};
use webrender_traits::{BorderDisplayItem, BorderSide, BorderStyle, YuvColorSpace};
use webrender_traits::{AuxiliaryLists, ItemRange, BoxShadowClipMode, ClipRegion};
use webrender_traits::{PipelineId, ScrollLayerId, WebGLContextId, FontRenderMode};
use webrender_traits::{DeviceIntRect, DeviceIntPoint, DeviceIntSize, DeviceIntLength, device_length};
use webrender_traits::{DeviceUintSize, DeviceUintPoint};
use webrender_traits::{LayerRect, LayerPoint, LayerSize};
use webrender_traits::{LayerToScrollTransform, LayerToWorldTransform, WorldToLayerTransform};
use webrender_traits::{WorldPoint4D, ScrollLayerPixel, as_scroll_parent_rect};
use webrender_traits::{GlyphOptions};

// Special sentinel value recognized by the shader. It is considered to be
// a dummy task that doesn't mask out anything.
const OPAQUE_TASK_INDEX: RenderTaskIndex = RenderTaskIndex(i32::MAX as usize);

const FLOATS_PER_RENDER_TASK_INFO: usize = 12;

pub type AuxiliaryListsMap = HashMap<PipelineId,
                                     AuxiliaryLists,
                                     BuildHasherDefault<FnvHasher>>;

trait AlphaBatchHelpers {
    fn get_batch_kind(&self, metadata: &PrimitiveMetadata) -> AlphaBatchKind;
    fn get_color_textures(&self, metadata: &PrimitiveMetadata) -> [SourceTexture; 3];
    fn get_blend_mode(&self, needs_blending: bool, metadata: &PrimitiveMetadata) -> BlendMode;
    fn add_prim_to_batch(&self,
                         prim_index: PrimitiveIndex,
                         batch: &mut PrimitiveBatch,
                         packed_layer_index: PackedLayerIndex,
                         task_index: RenderTaskIndex,
                         render_tasks: &RenderTaskCollection,
                         pass_index: RenderPassIndex,
                         z_sort_index: i32);
    fn add_blend_to_batch(&self,
                          stacking_context_index: StackingContextIndex,
                          batch: &mut PrimitiveBatch,
                          task_index: RenderTaskIndex,
                          src_task_index: RenderTaskIndex,
                          filter: LowLevelFilterOp,
                          z_sort_index: i32);
}

impl AlphaBatchHelpers for PrimitiveStore {
    fn get_batch_kind(&self, metadata: &PrimitiveMetadata) -> AlphaBatchKind {
        let batch_kind = match metadata.prim_kind {
            PrimitiveKind::Border => AlphaBatchKind::Border,
            PrimitiveKind::BoxShadow => AlphaBatchKind::BoxShadow,
            PrimitiveKind::Image => AlphaBatchKind::Image,
            PrimitiveKind::YuvImage => AlphaBatchKind::YuvImage,
            PrimitiveKind::Rectangle => AlphaBatchKind::Rectangle,
            PrimitiveKind::AlignedGradient => AlphaBatchKind::AlignedGradient,
            PrimitiveKind::AngleGradient => AlphaBatchKind::AngleGradient,
            PrimitiveKind::RadialGradient => AlphaBatchKind::RadialGradient,
            PrimitiveKind::TextRun => {
                let text_run_cpu = &self.cpu_text_runs[metadata.cpu_prim_index.0];
                if text_run_cpu.blur_radius.0 == 0 {
                    AlphaBatchKind::TextRun
                } else {
                    // Select a generic primitive shader that can blit the
                    // results of the cached text blur to the framebuffer,
                    // applying tile clipping etc.
                    AlphaBatchKind::CacheImage
                }
            }
        };

        batch_kind
    }

    fn get_color_textures(&self, metadata: &PrimitiveMetadata) -> [SourceTexture; 3] {
        let invalid = SourceTexture::Invalid;
        match metadata.prim_kind {
            PrimitiveKind::Border |
            PrimitiveKind::BoxShadow |
            PrimitiveKind::Rectangle |
            PrimitiveKind::AlignedGradient |
            PrimitiveKind::AngleGradient |
            PrimitiveKind::RadialGradient => [invalid; 3],
            PrimitiveKind::Image => {
                let image_cpu = &self.cpu_images[metadata.cpu_prim_index.0];
                [image_cpu.color_texture_id, invalid, invalid]
            }
            PrimitiveKind::YuvImage => {
                let image_cpu = &self.cpu_yuv_images[metadata.cpu_prim_index.0];
                [image_cpu.y_texture_id, image_cpu.u_texture_id, image_cpu.v_texture_id]
            }
            PrimitiveKind::TextRun => {
                let text_run_cpu = &self.cpu_text_runs[metadata.cpu_prim_index.0];
                [text_run_cpu.color_texture_id, invalid, invalid]
            }
        }
    }

    fn get_blend_mode(&self, needs_blending: bool, metadata: &PrimitiveMetadata) -> BlendMode {
        match metadata.prim_kind {
            PrimitiveKind::TextRun => {
                let text_run_cpu = &self.cpu_text_runs[metadata.cpu_prim_index.0];
                if text_run_cpu.blur_radius.0 == 0 {
                    match text_run_cpu.render_mode {
                        FontRenderMode::Subpixel => BlendMode::Subpixel(text_run_cpu.color),
                        FontRenderMode::Alpha | FontRenderMode::Mono => BlendMode::Alpha,
                    }
                } else {
                    // Text runs drawn to blur never get drawn with subpixel AA.
                    BlendMode::Alpha
                }
            }
            _ => {
                if needs_blending {
                    BlendMode::Alpha
                } else {
                    BlendMode::None
                }
            }
        }
    }

    fn add_blend_to_batch(&self,
                          stacking_context_index: StackingContextIndex,
                          batch: &mut PrimitiveBatch,
                          task_index: RenderTaskIndex,
                          src_task_index: RenderTaskIndex,
                          filter: LowLevelFilterOp,
                          z_sort_index: i32) {
        let (filter_mode, amount) = match filter {
            LowLevelFilterOp::Blur(..) => (0, 0.0),
            LowLevelFilterOp::Contrast(amount) => (1, amount.to_f32_px()),
            LowLevelFilterOp::Grayscale(amount) => (2, amount.to_f32_px()),
            LowLevelFilterOp::HueRotate(angle) => (3, (angle as f32) / ANGLE_FLOAT_TO_FIXED),
            LowLevelFilterOp::Invert(amount) => (4, amount.to_f32_px()),
            LowLevelFilterOp::Saturate(amount) => (5, amount.to_f32_px()),
            LowLevelFilterOp::Sepia(amount) => (6, amount.to_f32_px()),
            LowLevelFilterOp::Brightness(amount) => (7, amount.to_f32_px()),
            LowLevelFilterOp::Opacity(amount) => (8, amount.to_f32_px()),
        };

        let amount = (amount * 65535.0).round() as i32;

        batch.items.push(PrimitiveBatchItem::StackingContext(stacking_context_index));

        match batch.data {
            PrimitiveBatchData::Instances(ref mut data) => {
                data.push(PrimitiveInstance {
                    global_prim_id: -1,
                    prim_address: GpuStoreAddress(0),
                    task_index: task_index.0 as i32,
                    clip_task_index: -1,
                    layer_index: -1,
                    sub_index: filter_mode,
                    user_data: [src_task_index.0 as i32, amount],
                    z_sort_index: z_sort_index,
                });
            }
            _ => unreachable!(),
        }
    }

    fn add_prim_to_batch(&self,
                         prim_index: PrimitiveIndex,
                         batch: &mut PrimitiveBatch,
                         packed_layer_index: PackedLayerIndex,
                         task_index: RenderTaskIndex,
                         render_tasks: &RenderTaskCollection,
                         child_pass_index: RenderPassIndex,
                         z_sort_index: i32) {
        let metadata = self.get_metadata(prim_index);
        let packed_layer_index = packed_layer_index.0 as i32;
        let global_prim_id = prim_index.0 as i32;
        let prim_address = metadata.gpu_prim_index;
        let clip_task_index = match metadata.clip_task {
            Some(ref clip_task) => {
                render_tasks.get_task_index(&clip_task.id, child_pass_index)
            }
            None => {
                OPAQUE_TASK_INDEX
            }
        };
        let task_index = task_index.0 as i32;
        let clip_task_index = clip_task_index.0 as i32;
        batch.items.push(PrimitiveBatchItem::Primitive(prim_index));

        match &mut batch.data {
            &mut PrimitiveBatchData::Composite(..) => unreachable!(),
            &mut PrimitiveBatchData::Instances(ref mut data) => {
                match batch.key.kind {
                    AlphaBatchKind::Composite => unreachable!(),
                    AlphaBatchKind::Blend => unreachable!(),
                    AlphaBatchKind::Rectangle => {
                        data.push(PrimitiveInstance {
                            task_index: task_index,
                            clip_task_index: clip_task_index,
                            layer_index: packed_layer_index,
                            global_prim_id: global_prim_id,
                            prim_address: prim_address,
                            sub_index: 0,
                            user_data: [0, 0],
                            z_sort_index: z_sort_index,
                        });
                    }
                    AlphaBatchKind::TextRun => {
                        let text_cpu = &self.cpu_text_runs[metadata.cpu_prim_index.0];

                        for glyph_index in 0..metadata.gpu_data_count {
                            data.push(PrimitiveInstance {
                                task_index: task_index,
                                clip_task_index: clip_task_index,
                                layer_index: packed_layer_index,
                                global_prim_id: global_prim_id,
                                prim_address: prim_address,
                                sub_index: metadata.gpu_data_address.0 + glyph_index,
                                user_data: [ text_cpu.resource_address.0 + glyph_index, 0 ],
                                z_sort_index: z_sort_index,
                            });
                        }
                    }
                    AlphaBatchKind::Image => {
                        let image_cpu = &self.cpu_images[metadata.cpu_prim_index.0];

                        data.push(PrimitiveInstance {
                            task_index: task_index,
                            clip_task_index: clip_task_index,
                            layer_index: packed_layer_index,
                            global_prim_id: global_prim_id,
                            prim_address: prim_address,
                            sub_index: 0,
                            user_data: [ image_cpu.resource_address.0, 0 ],
                            z_sort_index: z_sort_index,
                        });
                    }
                    AlphaBatchKind::YuvImage => {
                        data.push(PrimitiveInstance {
                            task_index: task_index,
                            clip_task_index: clip_task_index,
                            layer_index: packed_layer_index,
                            global_prim_id: global_prim_id,
                            prim_address: prim_address,
                            sub_index: 0,
                            user_data: [ 0, 0 ],
                            z_sort_index: z_sort_index,
                        });
                    }
                    AlphaBatchKind::Border => {
                        for border_segment in 0..8 {
                            data.push(PrimitiveInstance {
                                task_index: task_index,
                                clip_task_index: clip_task_index,
                                layer_index: packed_layer_index,
                                global_prim_id: global_prim_id,
                                prim_address: prim_address,
                                sub_index: border_segment,
                                user_data: [ 0, 0 ],
                                z_sort_index: z_sort_index,
                            });
                        }
                    }
                    AlphaBatchKind::AlignedGradient => {
                        for part_index in 0..(metadata.gpu_data_count - 1) {
                            data.push(PrimitiveInstance {
                                task_index: task_index,
                                clip_task_index: clip_task_index,
                                layer_index: packed_layer_index,
                                global_prim_id: global_prim_id,
                                prim_address: prim_address,
                                sub_index: metadata.gpu_data_address.0 + part_index,
                                user_data: [ 0, 0 ],
                                z_sort_index: z_sort_index,
                            });
                        }
                    }
                    AlphaBatchKind::AngleGradient => {
                        data.push(PrimitiveInstance {
                            task_index: task_index,
                            clip_task_index: clip_task_index,
                            layer_index: packed_layer_index,
                            global_prim_id: global_prim_id,
                            prim_address: prim_address,
                            sub_index: metadata.gpu_data_address.0,
                            user_data: [ metadata.gpu_data_count, 0 ],
                            z_sort_index: z_sort_index,
                        });
                    }
                    AlphaBatchKind::RadialGradient => {
                        data.push(PrimitiveInstance {
                            task_index: task_index,
                            clip_task_index: clip_task_index,
                            layer_index: packed_layer_index,
                            global_prim_id: global_prim_id,
                            prim_address: prim_address,
                            sub_index: metadata.gpu_data_address.0,
                            user_data: [ metadata.gpu_data_count, 0 ],
                            z_sort_index: z_sort_index,
                        });
                    }
                    AlphaBatchKind::BoxShadow => {
                        let cache_task_id = &metadata.render_task.as_ref().unwrap().id;
                        let cache_task_index = render_tasks.get_task_index(cache_task_id,
                                                                           child_pass_index);

                        for rect_index in 0..metadata.gpu_data_count {
                            data.push(PrimitiveInstance {
                                task_index: task_index,
                                clip_task_index: clip_task_index,
                                layer_index: packed_layer_index,
                                global_prim_id: global_prim_id,
                                prim_address: prim_address,
                                sub_index: metadata.gpu_data_address.0 + rect_index,
                                user_data: [ cache_task_index.0 as i32, 0 ],
                                z_sort_index: z_sort_index,
                            });
                        }
                    }
                    AlphaBatchKind::CacheImage => {
                        // Find the render task index for the render task
                        // that this primitive depends on. Pass it to the
                        // shader so that it can sample from the cache texture
                        // at the correct location.
                        let cache_task_id = &metadata.render_task.as_ref().unwrap().id;
                        let cache_task_index = render_tasks.get_task_index(cache_task_id,
                                                                           child_pass_index);

                        data.push(PrimitiveInstance {
                            task_index: task_index,
                            clip_task_index: clip_task_index,
                            layer_index: packed_layer_index,
                            global_prim_id: global_prim_id,
                            prim_address: prim_address,
                            sub_index: 0,
                            user_data: [ cache_task_index.0 as i32, 0 ],
                            z_sort_index: z_sort_index,
                        });
                    }
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

// TODO(gw): I've had to make several of these types below public
//           with the changes for text-shadow. The proper solution
//           is to split the render task and render target code into
//           its own module. However, I'm avoiding that for now since
//           this PR is large enough already, and other people are working
//           on PRs that make use of render tasks.

#[derive(Debug, Copy, Clone)]
pub struct RenderTargetIndex(usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct RenderPassIndex(isize);

#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
pub struct RenderTaskIndex(usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum MaskCacheKey {
    Primitive(PrimitiveIndex),
    StackingContext(StackingContextIndex),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum RenderTaskKey {
    /// Draw this primitive to a cache target.
    CachePrimitive(PrimitiveCacheKey),
    /// Draw the alpha mask for a primitive.
    CacheMask(MaskCacheKey),
    /// Apply a vertical blur pass of given radius for this primitive.
    VerticalBlur(i32, PrimitiveIndex),
    /// Apply a horizontal blur pass of given radius for this primitive.
    HorizontalBlur(i32, PrimitiveIndex),
    /// Allocate a block of space in target for framebuffer copy.
    CopyFramebuffer(StackingContextIndex),
}

#[derive(Debug, Copy, Clone)]
pub enum RenderTaskId {
    Static(RenderTaskIndex),
    Dynamic(RenderTaskKey),
}

struct DynamicTaskInfo {
    index: RenderTaskIndex,
    rect: DeviceIntRect,
}

struct RenderTaskCollection {
    render_task_data: Vec<RenderTaskData>,
    dynamic_tasks: HashMap<(RenderTaskKey, RenderPassIndex), DynamicTaskInfo, BuildHasherDefault<FnvHasher>>,
}

impl RenderTaskCollection {
    fn new(static_render_task_count: usize) -> RenderTaskCollection {
        RenderTaskCollection {
            render_task_data: vec![RenderTaskData::empty(); static_render_task_count],
            dynamic_tasks: HashMap::with_hasher(Default::default()),
        }
    }

    fn add(&mut self, task: &RenderTask, pass: RenderPassIndex) -> RenderTaskIndex {
        match task.id {
            RenderTaskId::Static(index) => {
                self.render_task_data[index.0] = task.write_task_data();
                index
            }
            RenderTaskId::Dynamic(key) => {
                let index = RenderTaskIndex(self.render_task_data.len());
                let key = (key, pass);
                debug_assert!(self.dynamic_tasks.contains_key(&key) == false);
                self.dynamic_tasks.insert(key, DynamicTaskInfo {
                    index: index,
                    rect: match task.location {
                        RenderTaskLocation::Fixed => panic!("Dynamic tasks should not have fixed locations!"),
                        RenderTaskLocation::Dynamic(Some((origin, _)), size) => DeviceIntRect::new(origin, size),
                        RenderTaskLocation::Dynamic(None, _) => panic!("Expect the task to be already allocated here"),
                    },
                });
                self.render_task_data.push(task.write_task_data());
                index
            }
        }
    }

    fn get_dynamic_allocation(&self, pass_index: RenderPassIndex, key: RenderTaskKey) -> Option<&DeviceIntRect> {
        let key = (key, pass_index);
        self.dynamic_tasks.get(&key)
                          .map(|task| &task.rect)
    }

    fn get_static_task_index(&self, id: &RenderTaskId) -> RenderTaskIndex {
        match id {
            &RenderTaskId::Static(index) => index,
            &RenderTaskId::Dynamic(..) => panic!("This is a bug - expected a static render task!"),
        }
    }

    fn get_task_index(&self, id: &RenderTaskId, pass_index: RenderPassIndex) -> RenderTaskIndex {
        match id {
            &RenderTaskId::Static(index) => index,
            &RenderTaskId::Dynamic(key) => {
                self.dynamic_tasks[&(key, pass_index)].index
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct RenderTaskData {
    pub data: [f32; FLOATS_PER_RENDER_TASK_INFO],
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
    opaque_items: Vec<AlphaRenderItem>,
    alpha_items: Vec<AlphaRenderItem>,
}

/// Encapsulates the logic of building batches for items that are blended.
pub struct AlphaBatcher {
    pub alpha_batches: Vec<PrimitiveBatch>,
    pub opaque_batches: Vec<PrimitiveBatch>,
    tasks: Vec<AlphaBatchTask>,
}

impl AlphaBatcher {
    fn new() -> AlphaBatcher {
        AlphaBatcher {
            alpha_batches: Vec::new(),
            opaque_batches: Vec::new(),
            tasks: Vec::new(),
        }
    }

    fn add_task(&mut self, task: AlphaBatchTask) {
        self.tasks.push(task);
    }

    fn build(&mut self,
             ctx: &RenderTargetContext,
             render_tasks: &RenderTaskCollection,
             child_pass_index: RenderPassIndex) {
        let mut alpha_batches: Vec<PrimitiveBatch> = vec![];
        let mut opaque_batches: Vec<PrimitiveBatch> = vec![];

        for task in &mut self.tasks {
            let task_index = render_tasks.get_static_task_index(&task.task_id);
            let mut existing_opaque_batch_index = 0;

            for item in &task.alpha_items {
                let (batch_key, item_bounding_rect) = match item {
                    &AlphaRenderItem::Blend(stacking_context_index, ..) => {
                        let stacking_context = &ctx.stacking_context_store[stacking_context_index.0];
                        (AlphaBatchKey::new(AlphaBatchKind::Blend,
                                            AlphaBatchKeyFlags::empty(),
                                            BlendMode::Alpha,
                                            BatchTextures::no_texture()),
                         &stacking_context.xf_rect.as_ref().unwrap().bounding_rect)
                    }
                    &AlphaRenderItem::Composite(stacking_context_index,
                                                backdrop_id,
                                                src_id,
                                                info,
                                                z) => {
                        // Composites always get added to their own batch.
                        // This is because the result of a composite can affect
                        // the input to the next composite. Perhaps we can
                        // optimize this in the future.
                        let batch = PrimitiveBatch::new_composite(stacking_context_index,
                                                                  task_index,
                                                                  render_tasks.get_task_index(&backdrop_id, child_pass_index),
                                                                  render_tasks.get_static_task_index(&src_id),
                                                                  info,
                                                                  z);
                        alpha_batches.push(batch);
                        continue;
                    }
                    &AlphaRenderItem::Primitive(stacking_context_index, prim_index, _) => {
                        let stacking_context =
                            &ctx.stacking_context_store[stacking_context_index.0];
                        let prim_metadata = ctx.prim_store.get_metadata(prim_index);
                        let transform_kind = stacking_context.xf_rect.as_ref().unwrap().kind;
                        let needs_clipping = prim_metadata.clip_task.is_some();
                        let needs_blending = transform_kind == TransformedRectKind::Complex ||
                                             !prim_metadata.is_opaque ||
                                             needs_clipping;
                        let blend_mode = ctx.prim_store.get_blend_mode(needs_blending, prim_metadata);
                        let needs_clipping_flag = if needs_clipping {
                            NEEDS_CLIPPING
                        } else {
                            AlphaBatchKeyFlags::empty()
                        };
                        let flags = match transform_kind {
                            TransformedRectKind::AxisAligned => AXIS_ALIGNED | needs_clipping_flag,
                            _ => needs_clipping_flag,
                        };
                        let batch_kind = ctx.prim_store.get_batch_kind(prim_metadata);

                        let textures = BatchTextures {
                            colors: ctx.prim_store.get_color_textures(prim_metadata),
                        };

                        (AlphaBatchKey::new(batch_kind,
                                            flags,
                                            blend_mode,
                                            textures),
                         ctx.prim_store.cpu_bounding_rects[prim_index.0].as_ref().unwrap())
                    }
                };

                let mut alpha_batch_index = None;
                'outer: for (batch_index, batch) in alpha_batches.iter()
                                                         .enumerate()
                                                         .rev()
                                                         .take(10) {
                    if batch.key.is_compatible_with(&batch_key) {
                        alpha_batch_index = Some(batch_index);
                        break;
                    }

                    // check for intersections
                    for item in &batch.items {
                        let intersects = match *item {
                            PrimitiveBatchItem::StackingContext(stacking_context_index) => {
                                let stacking_context =
                                    &ctx.stacking_context_store[stacking_context_index.0];
                                stacking_context.xf_rect
                                                .as_ref()
                                                .unwrap()
                                                .bounding_rect
                                                .intersects(item_bounding_rect)
                            }
                            PrimitiveBatchItem::Primitive(prim_index) => {
                                let bounding_rect = &ctx.prim_store.cpu_bounding_rects[prim_index.0];
                                bounding_rect.as_ref().unwrap().intersects(item_bounding_rect)
                            }
                        };

                        if intersects {
                            break 'outer;
                        }
                    }
                }

                if alpha_batch_index.is_none() {
                    let new_batch = match item {
                        &AlphaRenderItem::Composite(..) => unreachable!(),
                        &AlphaRenderItem::Blend(..) => {
                            PrimitiveBatch::new_instances(AlphaBatchKind::Blend, batch_key)
                        }
                        &AlphaRenderItem::Primitive(_, prim_index, _) => {
                            let prim_metadata = ctx.prim_store.get_metadata(prim_index);
                            let batch_kind = ctx.prim_store.get_batch_kind(prim_metadata);
                            PrimitiveBatch::new_instances(batch_kind, batch_key)
                        }
                    };
                    alpha_batch_index = Some(alpha_batches.len());
                    alpha_batches.push(new_batch);
                }

                let batch = &mut alpha_batches[alpha_batch_index.unwrap()];
                match item {
                    &AlphaRenderItem::Composite(..) => unreachable!(),
                    &AlphaRenderItem::Blend(stacking_context_index, src_id, info, z) => {
                        ctx.prim_store.add_blend_to_batch(stacking_context_index,
                                                          batch,
                                                          task_index,
                                                          render_tasks.get_static_task_index(&src_id),
                                                          info,
                                                          z);
                    }
                    &AlphaRenderItem::Primitive(stacking_context_index, prim_index, z) => {
                        let stacking_context =
                            &ctx.stacking_context_store[stacking_context_index.0];
                        ctx.prim_store.add_prim_to_batch(prim_index,
                                                         batch,
                                                         stacking_context.packed_layer_index,
                                                         task_index,
                                                         render_tasks,
                                                         child_pass_index,
                                                         z);
                    }
                }
            }

            for item in task.opaque_items.iter().rev() {
                let batch_key = match item {
                    &AlphaRenderItem::Composite(..) => unreachable!(),
                    &AlphaRenderItem::Blend(..) => unreachable!(),
                    &AlphaRenderItem::Primitive(stacking_context_index, prim_index, _) => {
                        let stacking_context = &ctx.stacking_context_store[stacking_context_index.0];
                        let prim_metadata = ctx.prim_store.get_metadata(prim_index);
                        let transform_kind = stacking_context.xf_rect.as_ref().unwrap().kind;
                        let needs_clipping = prim_metadata.clip_task.is_some();
                        let needs_blending = transform_kind == TransformedRectKind::Complex ||
                                             !prim_metadata.is_opaque ||
                                             needs_clipping;
                        let blend_mode = ctx.prim_store.get_blend_mode(needs_blending, prim_metadata);
                        let needs_clipping_flag = if needs_clipping {
                            NEEDS_CLIPPING
                        } else {
                            AlphaBatchKeyFlags::empty()
                        };
                        let flags = match transform_kind {
                            TransformedRectKind::AxisAligned => AXIS_ALIGNED | needs_clipping_flag,
                            _ => needs_clipping_flag,
                        };
                        let batch_kind = ctx.prim_store.get_batch_kind(prim_metadata);

                        let textures = BatchTextures {
                            colors: ctx.prim_store.get_color_textures(prim_metadata),
                        };

                        AlphaBatchKey::new(batch_kind,
                                           flags,
                                           blend_mode,
                                           textures)
                    }
                };

                while existing_opaque_batch_index < opaque_batches.len() &&
                        !opaque_batches[existing_opaque_batch_index].key.is_compatible_with(&batch_key) {
                    existing_opaque_batch_index += 1
                }

                if existing_opaque_batch_index == opaque_batches.len() {
                    let new_batch = match item {
                        &AlphaRenderItem::Composite(..) => unreachable!(),
                        &AlphaRenderItem::Blend(..) => unreachable!(),
                        &AlphaRenderItem::Primitive(_, prim_index, _) => {
                            let prim_metadata = ctx.prim_store.get_metadata(prim_index);
                            let batch_kind = ctx.prim_store.get_batch_kind(prim_metadata);
                            PrimitiveBatch::new_instances(batch_kind, batch_key)
                        }
                    };
                    opaque_batches.push(new_batch)
                }

                let batch = &mut opaque_batches[existing_opaque_batch_index];
                match item {
                    &AlphaRenderItem::Composite(..) => unreachable!(),
                    &AlphaRenderItem::Blend(..) => unreachable!(),
                    &AlphaRenderItem::Primitive(stacking_context_index, prim_index, z) => {
                        let stacking_context =
                            &ctx.stacking_context_store[stacking_context_index.0];
                        ctx.prim_store.add_prim_to_batch(prim_index,
                                                         batch,
                                                         stacking_context.packed_layer_index,
                                                         task_index,
                                                         render_tasks,
                                                         child_pass_index,
                                                         z);
                    }
                }
            }
        }

        self.alpha_batches.extend(alpha_batches.into_iter());
        self.opaque_batches.extend(opaque_batches.into_iter());
    }
}

/// Batcher managing draw calls into the clip mask (in the RT cache).
#[derive(Debug)]
pub struct ClipBatcher {
    /// Rectangle draws fill up the rectangles with rounded corners.
    pub rectangles: Vec<CacheClipInstance>,
    /// Image draws apply the image masking.
    pub images: HashMap<SourceTexture, Vec<CacheClipInstance>>,
}

impl ClipBatcher {
    fn new() -> ClipBatcher {
        ClipBatcher {
            rectangles: Vec::new(),
            images: HashMap::new(),
        }
    }

    fn add<'a>(&mut self,
               task_index: RenderTaskIndex,
               clips: &[(StackingContextIndex, MaskCacheInfo)],
               resource_cache: &ResourceCache,
               stacking_context_store: &'a [StackingContext],
               geometry_kind: MaskGeometryKind) {

        for &(stacking_context_index, ref info) in clips.iter() {
            let stacking_context = &stacking_context_store[stacking_context_index.0];
            let instance = CacheClipInstance {
                task_id: task_index.0 as i32,
                layer_index: stacking_context.packed_layer_index.0 as i32,
                address: GpuStoreAddress(0),
                segment: 0,
            };

            for clip_index in 0..info.effective_clip_count as usize {
                let offset = info.clip_range.start.0 + ((CLIP_DATA_GPU_SIZE * clip_index) as i32);
                match geometry_kind {
                    MaskGeometryKind::Default => {
                        self.rectangles.push(CacheClipInstance {
                            address: GpuStoreAddress(offset),
                            segment: MaskSegment::All as i32,
                            ..instance
                        });
                    }
                    MaskGeometryKind::CornersOnly => {
                        self.rectangles.extend(&[
                            CacheClipInstance {
                                address: GpuStoreAddress(offset),
                                segment: MaskSegment::Corner_TopLeft as i32,
                                ..instance
                            },
                            CacheClipInstance {
                                address: GpuStoreAddress(offset),
                                segment: MaskSegment::Corner_TopRight as i32,
                                ..instance
                            },
                            CacheClipInstance {
                                address: GpuStoreAddress(offset),
                                segment: MaskSegment::Corner_BottomLeft as i32,
                                ..instance
                            },
                            CacheClipInstance {
                                address: GpuStoreAddress(offset),
                                segment: MaskSegment::Corner_BottomRight as i32,
                                ..instance
                            },
                        ]);
                    }
                }
            }

            if let Some((ref mask, address)) = info.image {
                let cache_item = resource_cache.get_cached_image(mask.image, ImageRendering::Auto);
                self.images.entry(cache_item.texture_id)
                           .or_insert(Vec::new())
                           .push(CacheClipInstance {
                    address: address,
                    ..instance
                })
            }
        }
    }
}

struct RenderTargetContext<'a> {
    stacking_context_store: &'a [StackingContext],
    prim_store: &'a PrimitiveStore,
    resource_cache: &'a ResourceCache,
}

/// A render target represents a number of rendering operations on a surface.
pub struct RenderTarget {
    pub alpha_batcher: AlphaBatcher,
    pub clip_batcher: ClipBatcher,
    pub box_shadow_cache_prims: Vec<PrimitiveInstance>,
    // List of text runs to be cached to this render target.
    // TODO(gw): For now, assume that these all come from
    //           the same source texture id. This is almost
    //           always true except for pathological test
    //           cases with more than 4k x 4k of unique
    //           glyphs visible. Once the future glyph / texture
    //           cache changes land, this restriction will
    //           be removed anyway.
    pub text_run_cache_prims: Vec<PrimitiveInstance>,
    pub text_run_textures: BatchTextures,
    // List of blur operations to apply for this render target.
    pub vertical_blurs: Vec<BlurCommand>,
    pub horizontal_blurs: Vec<BlurCommand>,
    pub readbacks: Vec<DeviceIntRect>,
    page_allocator: TexturePage,
}

impl RenderTarget {
    fn new(size: DeviceUintSize) -> RenderTarget {
        RenderTarget {
            alpha_batcher: AlphaBatcher::new(),
            clip_batcher: ClipBatcher::new(),
            box_shadow_cache_prims: Vec::new(),
            text_run_cache_prims: Vec::new(),
            text_run_textures: BatchTextures::no_texture(),
            vertical_blurs: Vec::new(),
            horizontal_blurs: Vec::new(),
            readbacks: Vec::new(),
            page_allocator: TexturePage::new(CacheTextureId(0), size),
        }
    }

    fn build(&mut self,
             ctx: &RenderTargetContext,
             render_tasks: &mut RenderTaskCollection,
             child_pass_index: RenderPassIndex) {
        self.alpha_batcher.build(ctx,
                                 render_tasks,
                                 child_pass_index);
    }

    fn add_task(&mut self,
                task: RenderTask,
                ctx: &RenderTargetContext,
                render_tasks: &RenderTaskCollection,
                pass_index: RenderPassIndex) {
        match task.kind {
            RenderTaskKind::Alpha(info) => {
                self.alpha_batcher.add_task(AlphaBatchTask {
                    task_id: task.id,
                    opaque_items: info.opaque_items,
                    alpha_items: info.alpha_items,
                });
            }
            RenderTaskKind::VerticalBlur(_, prim_index) => {
                // Find the child render task that we are applying
                // a vertical blur on.
                // TODO(gw): Consider a simpler way for render tasks to find
                //           their child tasks than having to construct the
                //           correct id here.
                let child_pass_index = RenderPassIndex(pass_index.0 - 1);
                let task_key = RenderTaskKey::CachePrimitive(PrimitiveCacheKey::TextShadow(prim_index));
                let src_id = RenderTaskId::Dynamic(task_key);
                self.vertical_blurs.push(BlurCommand {
                    task_id: render_tasks.get_task_index(&task.id, pass_index).0 as i32,
                    src_task_id: render_tasks.get_task_index(&src_id, child_pass_index).0 as i32,
                    blur_direction: BlurDirection::Vertical as i32,
                    padding: 0,
                });
            }
            RenderTaskKind::HorizontalBlur(blur_radius, prim_index) => {
                // Find the child render task that we are applying
                // a horizontal blur on.
                let child_pass_index = RenderPassIndex(pass_index.0 - 1);
                let src_id = RenderTaskId::Dynamic(RenderTaskKey::VerticalBlur(blur_radius.0, prim_index));
                self.horizontal_blurs.push(BlurCommand {
                    task_id: render_tasks.get_task_index(&task.id, pass_index).0 as i32,
                    src_task_id: render_tasks.get_task_index(&src_id, child_pass_index).0 as i32,
                    blur_direction: BlurDirection::Horizontal as i32,
                    padding: 0,
                });
            }
            RenderTaskKind::CachePrimitive(prim_index) => {
                let prim_metadata = ctx.prim_store.get_metadata(prim_index);

                match prim_metadata.prim_kind {
                    PrimitiveKind::BoxShadow => {
                        self.box_shadow_cache_prims.push(PrimitiveInstance {
                            global_prim_id: prim_index.0 as i32,
                            prim_address: prim_metadata.gpu_prim_index,
                            task_index: render_tasks.get_task_index(&task.id, pass_index).0 as i32,
                            clip_task_index: 0,
                            layer_index: 0,
                            sub_index: 0,
                            user_data: [0; 2],
                            z_sort_index: 0,        // z is disabled for rendering cache primitives
                        });
                    }
                    PrimitiveKind::TextRun => {
                        let text = &ctx.prim_store.cpu_text_runs[prim_metadata.cpu_prim_index.0];
                        // We only cache text runs with a text-shadow (for now).
                        debug_assert!(text.blur_radius.0 != 0);

                        // TODO(gw): This should always be fine for now, since the texture
                        // atlas grows to 4k. However, it won't be a problem soon, once
                        // we switch the texture atlas to use texture layers!
                        let textures = BatchTextures {
                            colors: ctx.prim_store.get_color_textures(prim_metadata),
                        };

                        debug_assert!(textures.colors[0] != SourceTexture::Invalid);
                        debug_assert!(self.text_run_textures.colors[0] == SourceTexture::Invalid ||
                                      self.text_run_textures.colors[0] == textures.colors[0]);
                        self.text_run_textures = textures;

                        for glyph_index in 0..prim_metadata.gpu_data_count {
                            self.text_run_cache_prims.push(PrimitiveInstance {
                                global_prim_id: prim_index.0 as i32,
                                prim_address: prim_metadata.gpu_prim_index,
                                task_index: render_tasks.get_task_index(&task.id, pass_index).0 as i32,
                                clip_task_index: 0,
                                layer_index: 0,
                                sub_index: prim_metadata.gpu_data_address.0 + glyph_index,
                                user_data: [ text.resource_address.0 + glyph_index, 0],
                                z_sort_index: 0,        // z is disabled for rendering cache primitives
                            });
                        }
                    }
                    _ => {
                        // No other primitives make use of primitive caching yet!
                        unreachable!()
                    }
                }
            }
            RenderTaskKind::CacheMask(ref task_info) => {
                let task_index = render_tasks.get_task_index(&task.id, pass_index);
                self.clip_batcher.add(task_index,
                                      &task_info.clips,
                                      &ctx.resource_cache,
                                      &ctx.stacking_context_store,
                                      task_info.geometry_kind);
            }
            RenderTaskKind::Readback(device_rect) => {
                self.readbacks.push(device_rect);
            }
        }
    }
}

/// A render pass represents a set of rendering operations that don't depend on one
/// another.
///
/// A render pass can have several render targets if there wasn't enough space in one
/// target to do all of the rendering for that pass.
pub struct RenderPass {
    pass_index: RenderPassIndex,
    pub is_framebuffer: bool,
    tasks: Vec<RenderTask>,
    pub targets: Vec<RenderTarget>,
    size: DeviceUintSize,
}

impl RenderPass {
    fn new(pass_index: isize,
           is_framebuffer: bool,
           size: DeviceUintSize) -> RenderPass {
        RenderPass {
            pass_index: RenderPassIndex(pass_index),
            is_framebuffer: is_framebuffer,
            targets: vec![ RenderTarget::new(size) ],
            tasks: vec![],
            size: size,
        }
    }

    fn add_render_task(&mut self, task: RenderTask) {
        self.tasks.push(task);
    }

    fn allocate_target(&mut self, alloc_size: DeviceUintSize) -> DeviceUintPoint {
        let existing_origin = self.targets
                                  .last_mut()
                                  .unwrap()
                                  .page_allocator
                                  .allocate(&alloc_size);
        match existing_origin {
            Some(origin) => origin,
            None => {
                let mut new_target = RenderTarget::new(self.size);
                let origin = new_target.page_allocator
                                       .allocate(&alloc_size)
                                       .expect(&format!("Each render task must allocate <= size of one target! ({:?})", alloc_size));
                self.targets.push(new_target);
                origin
            }
        }
    }


    fn build(&mut self,
             ctx: &RenderTargetContext,
             render_tasks: &mut RenderTaskCollection) {
        // Step through each task, adding to batches as appropriate.
        let tasks = mem::replace(&mut self.tasks, Vec::new());
        for mut task in tasks {
            // Find a target to assign this task to, or create a new
            // one if required.
            match task.location {
                RenderTaskLocation::Fixed => {}
                RenderTaskLocation::Dynamic(ref mut origin, ref size) => {
                    // See if this task is a duplicate.
                    // If so, just skip adding it!
                    match task.id {
                        RenderTaskId::Static(..) => {}
                        RenderTaskId::Dynamic(key) => {
                            // Look up cache primitive key in the render
                            // task data array. If a matching key exists
                            // (that is in this pass) there is no need
                            // to draw it again!
                            if let Some(rect) = render_tasks.get_dynamic_allocation(self.pass_index, key) {
                                debug_assert_eq!(rect.size, *size);
                                continue;
                            }
                        }
                    }

                    let alloc_size = DeviceUintSize::new(size.width as u32, size.height as u32);
                    let alloc_origin = self.allocate_target(alloc_size);

                    *origin = Some((DeviceIntPoint::new(alloc_origin.x as i32,
                                                     alloc_origin.y as i32),
                                    RenderTargetIndex(self.targets.len() - 1)));
                }
            }

            render_tasks.add(&task, self.pass_index);
            self.targets.last_mut().unwrap().add_task(task,
                                                      ctx,
                                                      render_tasks,
                                                      self.pass_index);
        }

        for target in &mut self.targets {
            let child_pass_index = RenderPassIndex(self.pass_index.0 - 1);
            target.build(ctx, render_tasks, child_pass_index);
        }
    }
}

#[derive(Debug, Clone)]
pub enum RenderTaskLocation {
    Fixed,
    Dynamic(Option<(DeviceIntPoint, RenderTargetIndex)>, DeviceIntSize),
}

#[derive(Debug, Clone)]
enum AlphaRenderItem {
    Primitive(StackingContextIndex, PrimitiveIndex, i32),
    Blend(StackingContextIndex, RenderTaskId, LowLevelFilterOp, i32),
    Composite(StackingContextIndex, RenderTaskId, RenderTaskId, MixBlendMode, i32),
}

#[derive(Debug, Clone)]
pub struct AlphaRenderTask {
    screen_origin: DeviceIntPoint,
    opaque_items: Vec<AlphaRenderItem>,
    alpha_items: Vec<AlphaRenderItem>,
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
enum MaskSegment {
    // This must match the SEGMENT_ values
    // in clip_shared.glsl!
    All = 0,
    Corner_TopLeft,
    Corner_TopRight,
    Corner_BottomLeft,
    Corner_BottomRight,
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
enum MaskGeometryKind {
    Default,        // Draw the entire rect
    CornersOnly,    // Draw the corners (simple axis aligned mask)
    // TODO(gw): Add more types here (e.g. 4 rectangles outside the inner rect)
}

#[derive(Debug, Clone)]
pub struct CacheMaskTask {
    actual_rect: DeviceIntRect,
    inner_rect: DeviceIntRect,
    clips: Vec<(StackingContextIndex, MaskCacheInfo)>,
    geometry_kind: MaskGeometryKind,
}

#[derive(Debug)]
enum MaskResult {
    /// The mask is completely outside the region
    Outside,
    /// The mask is inside and needs to be processed
    Inside(RenderTask),
}

#[derive(Debug, Clone)]
pub enum RenderTaskKind {
    Alpha(AlphaRenderTask),
    CachePrimitive(PrimitiveIndex),
    CacheMask(CacheMaskTask),
    VerticalBlur(DeviceIntLength, PrimitiveIndex),
    HorizontalBlur(DeviceIntLength, PrimitiveIndex),
    Readback(DeviceIntRect),
}

// TODO(gw): Consider storing these in a separate array and having
//           primitives hold indices - this could avoid cloning
//           when adding them as child tasks to tiles.
#[derive(Debug, Clone)]
pub struct RenderTask {
    pub id: RenderTaskId,
    pub location: RenderTaskLocation,
    pub children: Vec<RenderTask>,
    pub kind: RenderTaskKind,
}

impl RenderTask {
    fn new_alpha_batch(task_index: RenderTaskIndex,
                       screen_origin: DeviceIntPoint,
                       location: RenderTaskLocation) -> RenderTask {
        RenderTask {
            id: RenderTaskId::Static(task_index),
            children: Vec::new(),
            location: location,
            kind: RenderTaskKind::Alpha(AlphaRenderTask {
                screen_origin: screen_origin,
                alpha_items: Vec::new(),
                opaque_items: Vec::new(),
            }),
        }
    }

    pub fn new_prim_cache(key: PrimitiveCacheKey,
                          size: DeviceIntSize,
                          prim_index: PrimitiveIndex) -> RenderTask {
        RenderTask {
            id: RenderTaskId::Dynamic(RenderTaskKey::CachePrimitive(key)),
            children: Vec::new(),
            location: RenderTaskLocation::Dynamic(None, size),
            kind: RenderTaskKind::CachePrimitive(prim_index),
        }
    }

    fn new_readback(key: StackingContextIndex,
                    screen_rect: DeviceIntRect) -> RenderTask {
        RenderTask {
            id: RenderTaskId::Dynamic(RenderTaskKey::CopyFramebuffer(key)),
            children: Vec::new(),
            location: RenderTaskLocation::Dynamic(None, screen_rect.size),
            kind: RenderTaskKind::Readback(screen_rect),
        }
    }

    fn new_mask(actual_rect: DeviceIntRect,
                mask_key: MaskCacheKey,
                clips: &[(StackingContextIndex, MaskCacheInfo)],
                stacking_context_store: &[StackingContext])
                -> MaskResult {
        if clips.is_empty() {
            return MaskResult::Outside;
        }

        // We scan through the clip stack and detect if our actual rectangle
        // is in the intersection of all of all the outer bounds,
        // and if it's completely inside the intersection of all of the inner bounds.
        let result = clips.iter()
                          .fold(Some(actual_rect), |current, clip| {
            current.and_then(|rect| rect.intersection(&clip.1.outer_rect))
        });

        let task_rect = match result {
            None => return MaskResult::Outside,
            Some(rect) => rect,
        };

        let inner_rect = clips.iter()
                              .fold(Some(task_rect), |current, clip| {
            current.and_then(|rect| rect.intersection(&clip.1.inner_rect))
        });

        // TODO(gw): This optimization is very conservative for now.
        //           For now, only draw optimized geometry if it is
        //           a single aligned rect mask with rounded corners.
        //           In the future, we'll expand this to handle the
        //           more complex types of clip mask geometry.
        let mut geometry_kind = MaskGeometryKind::Default;

        if inner_rect.is_some() && clips.len() == 1 {
            let (stacking_context_index, ref clip_info) = clips[0];
            let stacking_context = &stacking_context_store[stacking_context_index.0];

            if clip_info.image.is_none() && clip_info.effective_clip_count == 1 &&
               stacking_context.xf_rect.as_ref().unwrap().kind == TransformedRectKind::AxisAligned {
                geometry_kind = MaskGeometryKind::CornersOnly;
            }
        }

        let inner_rect = inner_rect.unwrap_or(DeviceIntRect::zero());

        MaskResult::Inside(RenderTask {
            id: RenderTaskId::Dynamic(RenderTaskKey::CacheMask(mask_key)),
            children: Vec::new(),
            location: RenderTaskLocation::Dynamic(None, task_rect.size),
            kind: RenderTaskKind::CacheMask(CacheMaskTask {
                actual_rect: task_rect,
                inner_rect: inner_rect,
                clips: clips.to_vec(),
                geometry_kind: geometry_kind,
            }),
        })
    }

    // Construct a render task to apply a blur to a primitive. For now,
    // this is only used for text runs, but we can probably extend this
    // to handle general blurs to any render task in the future.
    // The render task chain that is constructed looks like:
    //
    //    PrimitiveCacheTask: Draw the text run.
    //           ^
    //           |
    //    VerticalBlurTask: Apply the separable vertical blur to the primitive.
    //           ^
    //           |
    //    HorizontalBlurTask: Apply the separable horizontal blur to the vertical blur.
    //           |
    //           +---- This is stored as the input task to the primitive shader.
    //
    pub fn new_blur(key: PrimitiveCacheKey,
                    size: DeviceIntSize,
                    blur_radius: DeviceIntLength,
                    prim_index: PrimitiveIndex) -> RenderTask {
        let prim_cache_task = RenderTask::new_prim_cache(key,
                                                         size,
                                                         prim_index);

        let blur_target_size = size + DeviceIntSize::new(2 * blur_radius.0,
                                                         2 * blur_radius.0);

        let blur_task_v = RenderTask {
            id: RenderTaskId::Dynamic(RenderTaskKey::VerticalBlur(blur_radius.0, prim_index)),
            children: vec![prim_cache_task],
            location: RenderTaskLocation::Dynamic(None, blur_target_size),
            kind: RenderTaskKind::VerticalBlur(blur_radius, prim_index),
        };

        let blur_task_h = RenderTask {
            id: RenderTaskId::Dynamic(RenderTaskKey::HorizontalBlur(blur_radius.0, prim_index)),
            children: vec![blur_task_v],
            location: RenderTaskLocation::Dynamic(None, blur_target_size),
            kind: RenderTaskKind::HorizontalBlur(blur_radius, prim_index),
        };

        blur_task_h
    }

    fn as_alpha_batch<'a>(&'a mut self) -> &'a mut AlphaRenderTask {
        match self.kind {
            RenderTaskKind::Alpha(ref mut task) => task,
            RenderTaskKind::CachePrimitive(..) |
            RenderTaskKind::CacheMask(..) |
            RenderTaskKind::VerticalBlur(..) |
            RenderTaskKind::Readback(..) |
            RenderTaskKind::HorizontalBlur(..) => unreachable!(),
        }
    }

    // Write (up to) 8 floats of data specific to the type
    // of render task that is provided to the GPU shaders
    // via a vertex texture.
    fn write_task_data(&self) -> RenderTaskData {
        let (target_rect, target_index) = self.get_target_rect();

        // NOTE: The ordering and layout of these structures are
        //       required to match both the GPU structures declared
        //       in prim_shared.glsl, and also the uses in submit_batch()
        //       in renderer.rs.
        // TODO(gw): Maybe there's a way to make this stuff a bit
        //           more type-safe. Although, it will always need
        //           to be kept in sync with the GLSL code anyway.

        match self.kind {
            RenderTaskKind::Alpha(ref task) => {
                RenderTaskData {
                    data: [
                        target_rect.origin.x as f32,
                        target_rect.origin.y as f32,
                        target_rect.size.width as f32,
                        target_rect.size.height as f32,
                        task.screen_origin.x as f32,
                        task.screen_origin.y as f32,
                        target_index.0 as f32,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                }
            }
            RenderTaskKind::CachePrimitive(..) => {
                RenderTaskData {
                    data: [
                        target_rect.origin.x as f32,
                        target_rect.origin.y as f32,
                        target_rect.size.width as f32,
                        target_rect.size.height as f32,
                        target_index.0 as f32,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                }
            }
            RenderTaskKind::CacheMask(ref task) => {
                RenderTaskData {
                    data: [
                        target_rect.origin.x as f32,
                        target_rect.origin.y as f32,
                        (target_rect.origin.x + target_rect.size.width) as f32,
                        (target_rect.origin.y + target_rect.size.height) as f32,
                        task.actual_rect.origin.x as f32,
                        task.actual_rect.origin.y as f32,
                        target_index.0 as f32,
                        0.0,
                        task.inner_rect.origin.x as f32,
                        task.inner_rect.origin.y as f32,
                        (task.inner_rect.origin.x + task.inner_rect.size.width) as f32,
                        (task.inner_rect.origin.y + task.inner_rect.size.height) as f32,
                    ],
                }
            }
            RenderTaskKind::VerticalBlur(blur_radius, _) |
            RenderTaskKind::HorizontalBlur(blur_radius, _) => {
                RenderTaskData {
                    data: [
                        target_rect.origin.x as f32,
                        target_rect.origin.y as f32,
                        target_rect.size.width as f32,
                        target_rect.size.height as f32,
                        target_index.0 as f32,
                        blur_radius.0 as f32,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                }
            }
            RenderTaskKind::Readback(..) => {
                RenderTaskData {
                    data: [
                        target_rect.origin.x as f32,
                        target_rect.origin.y as f32,
                        target_rect.size.width as f32,
                        target_rect.size.height as f32,
                        target_index.0 as f32,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                }
            }
        }
    }

    fn get_target_rect(&self) -> (DeviceIntRect, RenderTargetIndex) {
        match self.location {
            RenderTaskLocation::Fixed => {
                (DeviceIntRect::zero(), RenderTargetIndex(0))
            },
            RenderTaskLocation::Dynamic(origin_and_target_index, size) => {
                let (origin, target_index) = origin_and_target_index.expect("Should have been allocated by now!");
                (DeviceIntRect::new(origin, size), target_index)
            }
        }
    }

    fn assign_to_passes(mut self,
                        pass_index: usize,
                        passes: &mut Vec<RenderPass>) {
        for child in self.children.drain(..) {
            child.assign_to_passes(pass_index - 1,
                                   passes);
        }

        // Sanity check - can be relaxed if needed
        match self.location {
            RenderTaskLocation::Fixed => {
                debug_assert!(pass_index == passes.len() - 1);
            }
            RenderTaskLocation::Dynamic(..) => {
                debug_assert!(pass_index < passes.len() - 1);
            }
        }

        let pass = &mut passes[pass_index];
        pass.add_render_task(self);
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

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum AlphaBatchKind {
    Composite = 0,
    Blend,
    Rectangle,
    TextRun,
    Image,
    YuvImage,
    Border,
    AlignedGradient,
    AngleGradient,
    RadialGradient,
    BoxShadow,
    CacheImage,
}

bitflags! {
    pub flags AlphaBatchKeyFlags: u8 {
        const NEEDS_CLIPPING  = 0b00000001,
        const AXIS_ALIGNED    = 0b00000010,
    }
}

impl AlphaBatchKeyFlags {
    pub fn transform_kind(&self) -> TransformedRectKind {
        if self.contains(AXIS_ALIGNED) {
            TransformedRectKind::AxisAligned
        } else {
            TransformedRectKind::Complex
        }
    }

    pub fn needs_clipping(&self) -> bool {
        self.contains(NEEDS_CLIPPING)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AlphaBatchKey {
    pub kind: AlphaBatchKind,
    pub flags: AlphaBatchKeyFlags,
    pub blend_mode: BlendMode,
    pub textures: BatchTextures,
}

impl AlphaBatchKey {
    fn new(kind: AlphaBatchKind,
           flags: AlphaBatchKeyFlags,
           blend_mode: BlendMode,
           textures: BatchTextures) -> AlphaBatchKey {
        AlphaBatchKey {
            kind: kind,
            flags: flags,
            blend_mode: blend_mode,
            textures: textures,
        }
    }

    fn is_compatible_with(&self, other: &AlphaBatchKey) -> bool {
        self.kind == other.kind &&
            self.flags == other.flags &&
            self.blend_mode == other.blend_mode &&
            textures_compatible(self.textures.colors[0], other.textures.colors[0]) &&
            textures_compatible(self.textures.colors[1], other.textures.colors[1]) &&
            textures_compatible(self.textures.colors[2], other.textures.colors[2])
    }
}

#[repr(C)]
#[derive(Debug)]
pub enum BlurDirection {
    Horizontal = 0,
    Vertical,
}

#[inline]
fn textures_compatible(t1: SourceTexture, t2: SourceTexture) -> bool {
    t1 == SourceTexture::Invalid || t2 == SourceTexture::Invalid || t1 == t2
}

// All Packed Primitives below must be 16 byte aligned.
#[derive(Debug)]
pub struct BlurCommand {
    task_id: i32,
    src_task_id: i32,
    blur_direction: i32,
    padding: i32,
}

/// A clipping primitive drawn into the clipping mask.
/// Could be an image or a rectangle, which defines the
/// way `address` is treated.
#[derive(Clone, Copy, Debug)]
pub struct CacheClipInstance {
    task_id: i32,
    layer_index: i32,
    address: GpuStoreAddress,
    segment: i32,
}

#[derive(Debug, Clone)]
pub struct PrimitiveInstance {
    global_prim_id: i32,
    prim_address: GpuStoreAddress,
    pub task_index: i32,
    clip_task_index: i32,
    layer_index: i32,
    sub_index: i32,
    z_sort_index: i32,
    pub user_data: [i32; 2],
}

#[derive(Debug)]
pub enum PrimitiveBatchData {
    Instances(Vec<PrimitiveInstance>),
    Composite(PrimitiveInstance),
}

#[derive(Debug)]
pub enum PrimitiveBatchItem {
    Primitive(PrimitiveIndex),
    StackingContext(StackingContextIndex),
}

#[derive(Debug)]
pub struct PrimitiveBatch {
    pub key: AlphaBatchKey,
    pub data: PrimitiveBatchData,
    pub items: Vec<PrimitiveBatchItem>,
}

impl PrimitiveBatch {
    fn new_instances(batch_kind: AlphaBatchKind, key: AlphaBatchKey) -> PrimitiveBatch {
        let data = match batch_kind {
            AlphaBatchKind::Rectangle |
            AlphaBatchKind::TextRun |
            AlphaBatchKind::Image |
            AlphaBatchKind::YuvImage |
            AlphaBatchKind::Border |
            AlphaBatchKind::AlignedGradient |
            AlphaBatchKind::AngleGradient |
            AlphaBatchKind::RadialGradient |
            AlphaBatchKind::BoxShadow |
            AlphaBatchKind::Blend |
            AlphaBatchKind::CacheImage => {
                PrimitiveBatchData::Instances(Vec::new())
            }
            AlphaBatchKind::Composite => unreachable!(),
        };

        PrimitiveBatch {
            key: key,
            data: data,
            items: Vec::new(),
        }
    }

    fn new_composite(stacking_context_index: StackingContextIndex,
                     task_index: RenderTaskIndex,
                     backdrop_task: RenderTaskIndex,
                     src_task_index: RenderTaskIndex,
                     mode: MixBlendMode,
                     z_sort_index: i32) -> PrimitiveBatch {
        let data = PrimitiveBatchData::Composite(PrimitiveInstance {
            global_prim_id: -1,
            prim_address: GpuStoreAddress(0),
            task_index: task_index.0 as i32,
            clip_task_index: -1,
            layer_index: -1,
            sub_index: mode as u32 as i32,
            user_data: [ backdrop_task.0 as i32,
                         src_task_index.0 as i32 ],
            z_sort_index: z_sort_index,
        });
        let key = AlphaBatchKey::new(AlphaBatchKind::Composite,
                                     AlphaBatchKeyFlags::empty(),
                                     BlendMode::Alpha,
                                     BatchTextures::no_texture());

        PrimitiveBatch {
            key: key,
            data: data,
            items: vec![PrimitiveBatchItem::StackingContext(stacking_context_index)],
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PackedLayerIndex(pub usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct StackingContextIndex(pub usize);

pub struct StackingContext {
    pub pipeline_id: PipelineId,
    local_transform: LayerToScrollTransform,
    local_rect: LayerRect,
    scroll_layer_id: ScrollLayerId,
    xf_rect: Option<TransformedRect>,
    composite_ops: CompositeOps,
    clip_source: ClipSource,
    clip_cache_info: Option<MaskCacheInfo>,
    packed_layer_index: PackedLayerIndex,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct PackedLayer {
    transform: LayerToWorldTransform,
    inv_transform: WorldToLayerTransform,
    local_clip_rect: LayerRect,
    screen_vertices: [WorldPoint4D; 4],
}

impl Default for PackedLayer {
    fn default() -> PackedLayer {
        PackedLayer {
            transform: LayerToWorldTransform::identity(),
            inv_transform: WorldToLayerTransform::identity(),
            local_clip_rect: LayerRect::zero(),
            screen_vertices: [WorldPoint4D::zero(); 4],
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompositeOps {
    // Requires only a single texture as input (e.g. most filters)
    filters: Vec<LowLevelFilterOp>,
    // Requires two source textures (e.g. mix-blend-mode)
    mix_blend_mode: Option<MixBlendMode>,
}

impl CompositeOps {
    pub fn new(filters: Vec<LowLevelFilterOp>, mix_blend_mode: Option<MixBlendMode>) -> CompositeOps {
        CompositeOps {
            filters: filters,
            mix_blend_mode: mix_blend_mode
        }
    }

    pub fn empty() -> CompositeOps {
        CompositeOps {
            filters: Vec::new(),
            mix_blend_mode: None,
        }
    }

    pub fn count(&self) -> usize {
        self.filters.len() + if self.mix_blend_mode.is_some() { 1 } else { 0 }
    }

    pub fn will_make_invisible(&self) -> bool {
        for op in &self.filters {
            match op {
                &LowLevelFilterOp::Opacity(Au(0)) => return true,
                _ => {}
            }
        }
        false
    }
}

impl StackingContext {
    fn is_visible(&self) -> bool {
        self.xf_rect.is_some()
    }

    fn can_contribute_to_scene(&self) -> bool {
        !self.composite_ops.will_make_invisible()
    }
}

#[derive(Clone, Copy)]
pub struct FrameBuilderConfig {
    pub enable_scrollbars: bool,
    pub enable_subpixel_aa: bool,
}

impl FrameBuilderConfig {
    pub fn new(enable_scrollbars: bool,
               enable_subpixel_aa: bool) -> FrameBuilderConfig {
        FrameBuilderConfig {
            enable_scrollbars: enable_scrollbars,
            enable_subpixel_aa: enable_subpixel_aa,
        }
    }
}

pub struct FrameBuilder {
    screen_rect: LayerRect,
    background_color: Option<ColorF>,
    prim_store: PrimitiveStore,
    cmds: Vec<PrimitiveRunCmd>,
    _debug: bool,                   // Unused for now, but handy to keep around
    config: FrameBuilderConfig,

    stacking_context_store: Vec<StackingContext>,
    packed_layers: Vec<PackedLayer>,

    scrollbar_prims: Vec<ScrollbarPrimitive>,
}

/// A rendering-oriented representation of frame::Frame built by the render backend
/// and presented to the renderer.
pub struct Frame {
    pub viewport_size: LayerSize,
    pub background_color: Option<ColorF>,
    pub device_pixel_ratio: f32,
    pub cache_size: DeviceUintSize,
    pub passes: Vec<RenderPass>,
    pub profile_counters: FrameProfileCounters,

    pub layer_texture_data: Vec<PackedLayer>,
    pub render_task_data: Vec<RenderTaskData>,
    pub gpu_data16: Vec<GpuBlock16>,
    pub gpu_data32: Vec<GpuBlock32>,
    pub gpu_data64: Vec<GpuBlock64>,
    pub gpu_data128: Vec<GpuBlock128>,
    pub gpu_geometry: Vec<PrimitiveGeometry>,
    pub gpu_gradient_data: Vec<GradientData>,
    pub gpu_resource_rects: Vec<TexelRect>,

    // List of textures that we don't know about yet
    // from the backend thread. The render thread
    // will use a callback to resolve these and
    // patch the data structures.
    pub deferred_resolves: Vec<DeferredResolve>,
}

impl FrameBuilder {
    pub fn new(viewport_size: LayerSize,
               background_color: Option<ColorF>,
               debug: bool,
               config: FrameBuilderConfig) -> FrameBuilder {
        FrameBuilder {
            screen_rect: LayerRect::new(LayerPoint::zero(), viewport_size),
            background_color: background_color,
            stacking_context_store: Vec::new(),
            prim_store: PrimitiveStore::new(),
            cmds: Vec::new(),
            _debug: debug,
            packed_layers: Vec::new(),
            scrollbar_prims: Vec::new(),
            config: config,
        }
    }

    fn add_primitive(&mut self,
                     rect: &LayerRect,
                     clip_region: &ClipRegion,
                     container: PrimitiveContainer) -> PrimitiveIndex {

        let geometry = PrimitiveGeometry {
            local_rect: *rect,
            local_clip_rect: clip_region.main,
        };
        let clip_source = if clip_region.is_complex() {
            ClipSource::Region(clip_region.clone())
        } else {
            ClipSource::NoClip
        };
        let clip_info = MaskCacheInfo::new(&clip_source,
                                           false,
                                           &mut self.prim_store.gpu_data32);

        let prim_index = self.prim_store.add_primitive(geometry,
                                                       Box::new(clip_source),
                                                       clip_info,
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

    pub fn push_stacking_context(&mut self,
                                 rect: LayerRect,
                                 clip_region: &ClipRegion,
                                 transform: LayerToScrollTransform,
                                 pipeline_id: PipelineId,
                                 scroll_layer_id: ScrollLayerId,
                                 composite_ops: CompositeOps) {
        let stacking_context_index = StackingContextIndex(self.stacking_context_store.len());
        let packed_layer_index = PackedLayerIndex(self.packed_layers.len());

        let clip_source = ClipSource::Region(clip_region.clone());
        let clip_info = MaskCacheInfo::new(&clip_source,
                                           true, // needs an extra clip for the clip rectangle
                                           &mut self.prim_store.gpu_data32);

        self.stacking_context_store.push(StackingContext {
            local_rect: rect,
            local_transform: transform,
            scroll_layer_id: scroll_layer_id,
            pipeline_id: pipeline_id,
            xf_rect: None,
            composite_ops: composite_ops,
            clip_source: clip_source,
            clip_cache_info: clip_info,
            packed_layer_index: packed_layer_index,
        });

        self.packed_layers.push(Default::default());
        self.cmds.push(PrimitiveRunCmd::PushStackingContext(stacking_context_index));
    }

    pub fn pop_stacking_context(&mut self) {
        self.cmds.push(PrimitiveRunCmd::PopStackingContext);
    }

    pub fn add_solid_rectangle(&mut self,
                               rect: &LayerRect,
                               clip_region: &ClipRegion,
                               color: &ColorF,
                               flags: PrimitiveFlags) {
        if color.a == 0.0 {
            return;
        }

        let prim = RectanglePrimitive {
            color: *color,
        };

        let prim_index = self.add_primitive(rect,
                                            clip_region,
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
            BorderStyle::Ridge |
            BorderStyle::Groove |
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
                      rect: LayerRect,
                      clip_region: &ClipRegion,
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

        let tl_outer = LayerPoint::new(rect.origin.x, rect.origin.y);
        let tl_inner = tl_outer + LayerPoint::new(radius.top_left.width.max(left.width),
                                                  radius.top_left.height.max(top.width));

        let tr_outer = LayerPoint::new(rect.origin.x + rect.size.width, rect.origin.y);
        let tr_inner = tr_outer + LayerPoint::new(-radius.top_right.width.max(right.width),
                                                  radius.top_right.height.max(top.width));

        let bl_outer = LayerPoint::new(rect.origin.x, rect.origin.y + rect.size.height);
        let bl_inner = bl_outer + LayerPoint::new(radius.bottom_left.width.max(left.width),
                                                  -radius.bottom_left.height.max(bottom.width));

        let br_outer = LayerPoint::new(rect.origin.x + rect.size.width,
                                       rect.origin.y + rect.size.height);
        let br_inner = br_outer - LayerPoint::new(radius.bottom_right.width.max(right.width),
                                                  radius.bottom_right.height.max(bottom.width));

        // The border shader is quite expensive. For simple borders, we can just draw
        // the border with a few rectangles. This generally gives better batching, and
        // a GPU win in fragment shader time.
        // More importantly, the software (OSMesa) implementation we run tests on is
        // particularly slow at running our complex border shader, compared to the
        // rectangle shader. This has the effect of making some of our tests time
        // out more often on CI (the actual cause is simply too many Servo processes and
        // threads being run on CI at once).
        // TODO(gw): Detect some more simple cases and handle those with simpler shaders too.
        // TODO(gw): Consider whether it's only worth doing this for large rectangles (since
        //           it takes a little more CPU time to handle multiple rectangles compared
        //           to a single border primitive).
        if left.style == BorderStyle::Solid {
            let same_color = left_color == top_color &&
                             left_color == right_color &&
                             left_color == bottom_color;
            let same_style = left.style == top.style &&
                             left.style == right.style &&
                             left.style == bottom.style;

            if same_color && same_style && radius.is_zero() {
                let rects = [
                    LayerRect::new(rect.origin,
                                   LayerSize::new(rect.size.width, top.width)),
                    LayerRect::new(LayerPoint::new(tl_outer.x, tl_inner.y),
                                   LayerSize::new(left.width,
                                                  rect.size.height - top.width - bottom.width)),
                    LayerRect::new(tr_inner,
                                   LayerSize::new(right.width,
                                                  rect.size.height - top.width - bottom.width)),
                    LayerRect::new(LayerPoint::new(bl_outer.x, bl_inner.y),
                                   LayerSize::new(rect.size.width, bottom.width))
                ];

                for rect in &rects {
                    self.add_solid_rectangle(rect,
                                             clip_region,
                                             &top_color,
                                             PrimitiveFlags::None);
                }

                return;
            }
        }

        //Note: while similar to `ComplexClipRegion::get_inner_rect()` in spirit,
        // this code is a bit more complex and can not there for be merged.
        let inner_rect = rect_from_points_f(tl_inner.x.max(bl_inner.x),
                                            tl_inner.y.max(tr_inner.y),
                                            tr_inner.x.min(br_inner.x),
                                            bl_inner.y.min(br_inner.y));

        let prim_cpu = BorderPrimitiveCpu {
            inner_rect: LayerRect::from_untyped(&inner_rect),
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
                           clip_region,
                           PrimitiveContainer::Border(prim_cpu, prim_gpu));
    }

    pub fn add_gradient(&mut self,
                        rect: LayerRect,
                        clip_region: &ClipRegion,
                        start_point: LayerPoint,
                        end_point: LayerPoint,
                        stops: ItemRange,
                        extend_mode: ExtendMode) {
        // Fast path for clamped, axis-aligned gradients:
        let aligned = extend_mode == ExtendMode::Clamp &&
                      (start_point.x == end_point.x ||
                       start_point.y == end_point.y);
        // Try to ensure that if the gradient is specified in reverse, then so long as the stops
        // are also supplied in reverse that the rendered result will be equivalent. To do this,
        // a reference orientation for the gradient line must be chosen, somewhat arbitrarily, so
        // just designate the reference orientation as start < end. Aligned gradient rendering
        // manages to produce the same result regardless of orientation, so don't worry about
        // reversing in that case.
        let reverse_stops = !aligned &&
                            (start_point.x > end_point.x ||
                             (start_point.x == end_point.x &&
                              start_point.y > end_point.y));

        let gradient_cpu = GradientPrimitiveCpu {
            stops_range: stops,
            extend_mode: extend_mode,
            reverse_stops: reverse_stops,
            cache_dirty: true,
        };

        // To get reftests exactly matching with reverse start/end
        // points, it's necessary to reverse the gradient
        // line in some cases.
        let (sp, ep) = if reverse_stops {
            (end_point, start_point)
        } else {
            (start_point, end_point)
        };

        let gradient_gpu = GradientPrimitiveGpu {
            start_point: sp,
            end_point: ep,
            extend_mode: pack_as_float(extend_mode as u32),
            padding: [0.0, 0.0, 0.0],
        };

        let prim = if aligned {
            PrimitiveContainer::AlignedGradient(gradient_cpu, gradient_gpu)
        } else {
            PrimitiveContainer::AngleGradient(gradient_cpu, gradient_gpu)
        };

        self.add_primitive(&rect, clip_region, prim);
    }

    pub fn add_radial_gradient(&mut self,
                               rect: LayerRect,
                               clip_region: &ClipRegion,
                               start_center: LayerPoint,
                               start_radius: f32,
                               end_center: LayerPoint,
                               end_radius: f32,
                               stops: ItemRange,
                               extend_mode: ExtendMode) {
        let radial_gradient_cpu = RadialGradientPrimitiveCpu {
            stops_range: stops,
            extend_mode: extend_mode,
            cache_dirty: true,
        };

        let radial_gradient_gpu = RadialGradientPrimitiveGpu {
            start_center: start_center,
            end_center: end_center,
            start_radius: start_radius,
            end_radius: end_radius,
            extend_mode: pack_as_float(extend_mode as u32),
            padding: [0.0],
        };

        self.add_primitive(&rect,
                           clip_region,
                           PrimitiveContainer::RadialGradient(radial_gradient_cpu, radial_gradient_gpu));
    }

    pub fn add_text(&mut self,
                    rect: LayerRect,
                    clip_region: &ClipRegion,
                    font_key: FontKey,
                    size: Au,
                    blur_radius: Au,
                    color: &ColorF,
                    glyph_range: ItemRange,
                    glyph_options: Option<GlyphOptions>) {
        if color.a == 0.0 {
            return
        }

        if size.0 <= 0 {
            return
        }

        let (render_mode, glyphs_per_run) = if blur_radius == Au(0) {
            // TODO(gw): Use a proper algorithm to select
            // whether this item should be rendered with
            // subpixel AA!
            let render_mode = if self.config.enable_subpixel_aa {
                FontRenderMode::Subpixel
            } else {
                FontRenderMode::Alpha
            };

            (render_mode, 8)
        } else {
            // TODO(gw): Support breaking up text shadow when
            // the size of the text run exceeds the dimensions
            // of the render target texture.
            (FontRenderMode::Alpha, glyph_range.length)
        };

        let text_run_count = (glyph_range.length + glyphs_per_run - 1) / glyphs_per_run;
        for run_index in 0..text_run_count {
            let start = run_index * glyphs_per_run;
            let end = cmp::min(start + glyphs_per_run, glyph_range.length);
            let sub_range = ItemRange {
                start: glyph_range.start + start,
                length: end - start,
            };

            let prim_cpu = TextRunPrimitiveCpu {
                font_key: font_key,
                logical_font_size: size,
                blur_radius: blur_radius,
                glyph_range: sub_range,
                cache_dirty: true,
                glyph_instances: Vec::new(),
                color_texture_id: SourceTexture::Invalid,
                color: *color,
                render_mode: render_mode,
                glyph_options: glyph_options,
                resource_address: GpuStoreAddress(0),
            };

            let prim_gpu = TextRunPrimitiveGpu {
                color: *color,
            };

            self.add_primitive(&rect,
                               clip_region,
                               PrimitiveContainer::TextRun(prim_cpu, prim_gpu));
        }
    }

    pub fn add_box_shadow(&mut self,
                          box_bounds: &LayerRect,
                          clip_region: &ClipRegion,
                          box_offset: &LayerPoint,
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
                                     clip_region,
                                     color,
                                     PrimitiveFlags::None);
            return;
        }

        let bs_rect = box_bounds.translate(box_offset)
                                .inflate(spread_radius, spread_radius);

        let outside_edge_size = 2.0 * blur_radius;
        let inside_edge_size = outside_edge_size.max(border_radius);
        let edge_size = outside_edge_size + inside_edge_size;
        let outer_rect = bs_rect.inflate(outside_edge_size, outside_edge_size);
        let mut instance_rects = Vec::new();
        let (prim_rect, inverted) = match clip_mode {
            BoxShadowClipMode::Outset | BoxShadowClipMode::None => {
                subtract_rect(&outer_rect, box_bounds, &mut instance_rects);
                (outer_rect, 0.0)
            }
            BoxShadowClipMode::Inset => {
                subtract_rect(box_bounds, &bs_rect, &mut instance_rects);
                (*box_bounds, 1.0)
            }
        };

        if edge_size == 0.0 {
            for rect in &instance_rects {
                self.add_solid_rectangle(rect,
                                         clip_region,
                                         color,
                                         PrimitiveFlags::None)
            }
        } else {
            let prim_gpu = BoxShadowPrimitiveGpu {
                src_rect: *box_bounds,
                bs_rect: bs_rect,
                color: *color,
                blur_radius: blur_radius,
                border_radius: border_radius,
                edge_size: edge_size,
                inverted: inverted,
            };

            self.add_primitive(&prim_rect,
                               clip_region,
                               PrimitiveContainer::BoxShadow(prim_gpu, instance_rects));
        }
    }

    pub fn add_webgl_rectangle(&mut self,
                               rect: LayerRect,
                               clip_region: &ClipRegion,
                               context_id: WebGLContextId) {
        let prim_cpu = ImagePrimitiveCpu {
            kind: ImagePrimitiveKind::WebGL(context_id),
            color_texture_id: SourceTexture::Invalid,
            resource_address: GpuStoreAddress(0),
        };

        let prim_gpu = ImagePrimitiveGpu {
            stretch_size: rect.size,
            tile_spacing: LayerSize::zero(),
        };

        self.add_primitive(&rect,
                           clip_region,
                           PrimitiveContainer::Image(prim_cpu, prim_gpu));
    }

    pub fn add_image(&mut self,
                     rect: LayerRect,
                     clip_region: &ClipRegion,
                     stretch_size: &LayerSize,
                     tile_spacing: &LayerSize,
                     image_key: ImageKey,
                     image_rendering: ImageRendering) {
        let prim_cpu = ImagePrimitiveCpu {
            kind: ImagePrimitiveKind::Image(image_key,
                                            image_rendering,
                                            *tile_spacing),
            color_texture_id: SourceTexture::Invalid,
            resource_address: GpuStoreAddress(0),
        };

        let prim_gpu = ImagePrimitiveGpu {
            stretch_size: *stretch_size,
            tile_spacing: *tile_spacing,
        };

        self.add_primitive(&rect,
                           clip_region,
                           PrimitiveContainer::Image(prim_cpu, prim_gpu));
    }

    pub fn add_yuv_image(&mut self,
                         rect: LayerRect,
                         clip_region: &ClipRegion,
                         y_image_key: ImageKey,
                         u_image_key: ImageKey,
                         v_image_key: ImageKey,
                         color_space: YuvColorSpace) {

        let prim_cpu = YuvImagePrimitiveCpu {
            y_key: y_image_key,
            u_key: u_image_key,
            v_key: v_image_key,
            y_texture_id: SourceTexture::Invalid,
            u_texture_id: SourceTexture::Invalid,
            v_texture_id: SourceTexture::Invalid,
        };

        let prim_gpu = YuvImagePrimitiveGpu::new(rect.size, color_space);

        self.add_primitive(&rect,
                           clip_region,
                           PrimitiveContainer::YuvImage(prim_cpu, prim_gpu));
    }

    /// Compute the contribution (bounding rectangles, and resources) of layers and their
    /// primitives in screen space.
    fn cull_layers(&mut self,
                   screen_rect: &DeviceIntRect,
                   scroll_tree: &ScrollTree,
                   auxiliary_lists_map: &AuxiliaryListsMap,
                   resource_cache: &mut ResourceCache,
                   profile_counters: &mut FrameProfileCounters,
                   device_pixel_ratio: f32) {
        // Build layer screen rects.
        // TODO(gw): This can be done earlier once update_layer_transforms() is fixed.

        // TODO(gw): Remove this stack once the layers refactor is done!
        let mut stacking_context_stack: Vec<StackingContextIndex> = Vec::new();
        let mut clip_info_stack = Vec::new();

        for cmd in &self.cmds {
            match cmd {
                &PrimitiveRunCmd::PushStackingContext(stacking_context_index) => {
                    stacking_context_stack.push(stacking_context_index);
                    let stacking_context =
                        &mut self.stacking_context_store[stacking_context_index.0];
                    let packed_layer =
                        &mut self.packed_layers[stacking_context.packed_layer_index.0];

                    stacking_context.xf_rect = None;

                    let scroll_layer = &scroll_tree.layers[&stacking_context.scroll_layer_id];
                    packed_layer.transform = scroll_layer.world_content_transform
                                                         .with_source::<ScrollLayerPixel>() // the scroll layer is considered a parent of layer
                                                         .pre_mul(&stacking_context.local_transform);
                    packed_layer.inv_transform = packed_layer.transform.inverse().unwrap();

                    if !stacking_context.can_contribute_to_scene() {
                        continue;
                    }

                    let inv_layer_transform = stacking_context.local_transform.inverse().unwrap();
                    let local_viewport_rect =
                        as_scroll_parent_rect(&scroll_layer.combined_local_viewport_rect);
                    let viewport_rect = inv_layer_transform.transform_rect(&local_viewport_rect);
                    let local_clip_rect =
                        stacking_context.clip_source.to_rect().unwrap_or(stacking_context.local_rect);
                    let layer_local_rect =
                         stacking_context.local_rect
                                         .intersection(&viewport_rect)
                                         .and_then(|rect| rect.intersection(&local_clip_rect));

                    if let Some(layer_local_rect) = layer_local_rect {
                        let layer_xf_rect = TransformedRect::new(&layer_local_rect,
                                                                 &packed_layer.transform,
                                                                 device_pixel_ratio);

                        if layer_xf_rect.bounding_rect.intersects(&screen_rect) {
                            packed_layer.screen_vertices = layer_xf_rect.vertices.clone();
                            packed_layer.local_clip_rect = layer_local_rect;
                            stacking_context.xf_rect = Some(layer_xf_rect);
                        }
                    }

                    if let Some(ref mut clip_info) = stacking_context.clip_cache_info {
                        let auxiliary_lists = auxiliary_lists_map.get(&stacking_context.pipeline_id)
                                                                 .expect("No auxiliary lists?");
                        clip_info.update(&stacking_context.clip_source,
                                         &packed_layer.transform,
                                         &mut self.prim_store.gpu_data32,
                                         device_pixel_ratio,
                                         auxiliary_lists);
                        if let ClipSource::Region(ClipRegion{ image_mask: Some(ref mask), .. }) = stacking_context.clip_source {
                            resource_cache.request_image(mask.image, ImageRendering::Auto);
                            //Note: no need to add the stacking context for resolve, all layers get resolved
                        }

                        // Create a task for the stacking context mask, if needed, i.e. if there
                        // are rounded corners or image masks for the stacking context.
                        clip_info_stack.push((stacking_context_index, clip_info.clone()));
                    }

                }
                &PrimitiveRunCmd::PrimitiveRun(prim_index, prim_count) => {
                    let stacking_context_index = stacking_context_stack.last().unwrap();
                    let stacking_context = &self.stacking_context_store[stacking_context_index.0];
                    if !stacking_context.is_visible() {
                        continue;
                    }

                    let packed_layer = &self.packed_layers[stacking_context.packed_layer_index.0];
                    let auxiliary_lists = auxiliary_lists_map.get(&stacking_context.pipeline_id)
                                                             .expect("No auxiliary lists?");

                    for i in 0..prim_count {
                        let prim_index = PrimitiveIndex(prim_index.0 + i);
                        if self.prim_store.build_bounding_rect(prim_index,
                                                               screen_rect,
                                                               &packed_layer.transform,
                                                               &packed_layer.local_clip_rect,
                                                               device_pixel_ratio) {
                            if self.prim_store.prepare_prim_for_render(prim_index,
                                                                       resource_cache,
                                                                       &packed_layer.transform,
                                                                       device_pixel_ratio,
                                                                       auxiliary_lists) {
                                self.prim_store.build_bounding_rect(prim_index,
                                                                    screen_rect,
                                                                    &packed_layer.transform,
                                                                    &packed_layer.local_clip_rect,
                                                                    device_pixel_ratio);
                            }

                            // If the primitive is visible, consider culling it via clip rect(s).
                            // If it is visible but has clips, create the clip task for it.
                            if let Some(prim_bounding_rect) = self.prim_store
                                                                  .cpu_bounding_rects[prim_index.0] {
                                let prim_metadata = &mut self.prim_store.cpu_metadata[prim_index.0];
                                let prim_clip_info = prim_metadata.clip_cache_info.as_ref();
                                let mut visible = true;

                                if let Some(info) = prim_clip_info {
                                    clip_info_stack.push((*stacking_context_index, info.clone()));
                                }

                                // Try to create a mask if we may need to.
                                if !clip_info_stack.is_empty() {
                                    // If the primitive doesn't have a specific clip,
                                    // key the task ID off the stacking context. This means
                                    // that two primitives which are only clipped by the
                                    // stacking context stack can share clip masks during
                                    // render task assignment to targets.
                                    let (mask_key, mask_rect) = match prim_clip_info {
                                        Some(..) => {
                                            (MaskCacheKey::Primitive(prim_index), prim_bounding_rect)
                                        }
                                        None => {
                                            (MaskCacheKey::StackingContext(*stacking_context_index),
                                             stacking_context.xf_rect.as_ref().unwrap().bounding_rect)
                                        }
                                    };
                                    let mask_opt =
                                        RenderTask::new_mask(mask_rect,
                                                             mask_key,
                                                             &clip_info_stack,
                                                             &self.stacking_context_store);
                                    match mask_opt {
                                        MaskResult::Outside => {
                                            // Primitive is completely clipped out.
                                            prim_metadata.clip_task = None;
                                            self.prim_store.cpu_bounding_rects[prim_index.0] = None;
                                            visible = false;
                                        }
                                        MaskResult::Inside(task) => {
                                            // Got a valid clip task, so store it for this primitive.
                                            prim_metadata.clip_task = Some(task);
                                        }
                                    }
                                }

                                if let Some(..) = prim_clip_info {
                                    clip_info_stack.pop();
                                }

                                if visible {
                                    profile_counters.visible_primitives.inc();
                                }
                            }
                        }
                    }
                }
                &PrimitiveRunCmd::PopStackingContext => {
                    let stacking_context_index = *stacking_context_stack.last().unwrap();
                    let stacking_context = &self.stacking_context_store[stacking_context_index.0];
                    if stacking_context.can_contribute_to_scene() {
                        if stacking_context.clip_cache_info.is_some() {
                            clip_info_stack.pop().unwrap();
                        }
                    }

                    stacking_context_stack.pop().unwrap();
                }
            }
        }
    }

    fn update_scroll_bars(&mut self, scroll_tree: &ScrollTree) {
        let distance_from_edge = 8.0;

        for scrollbar_prim in &self.scrollbar_prims {
            let mut geom = (*self.prim_store.gpu_geometry.get(GpuStoreAddress(scrollbar_prim.prim_index.0 as i32))).clone();
            let scroll_layer = &scroll_tree.layers[&scrollbar_prim.scroll_layer_id];

            let scrollable_distance = scroll_layer.scrollable_height();

            if scrollable_distance <= 0.0 {
                geom.local_clip_rect.size = LayerSize::zero();
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

            let clip_source = if scrollbar_prim.border_radius == 0.0 {
                ClipSource::NoClip
            } else {
                ClipSource::Complex(geom.local_rect, scrollbar_prim.border_radius)
            };
            self.prim_store.set_clip_source(scrollbar_prim.prim_index, clip_source);
            *self.prim_store.gpu_geometry.get_mut(GpuStoreAddress(scrollbar_prim.prim_index.0 as i32)) = geom;
        }
    }

    fn build_render_task(&self) -> (RenderTask, usize) {
        let mut next_z = 0;
        let mut next_task_index = RenderTaskIndex(0);

        let mut sc_stack = Vec::new();
        let mut current_task = RenderTask::new_alpha_batch(next_task_index,
                                                           DeviceIntPoint::zero(),
                                                           RenderTaskLocation::Fixed);
        next_task_index.0 += 1;
        let mut alpha_task_stack = Vec::new();

        for cmd in &self.cmds {
            match *cmd {
                PrimitiveRunCmd::PushStackingContext(stacking_context_index) => {
                    let stacking_context = &self.stacking_context_store[stacking_context_index.0];
                    sc_stack.push(stacking_context_index);

                    if !stacking_context.is_visible() {
                        continue;
                    }

                    let composite_count = stacking_context.composite_ops.count();
                    for _ in 0..composite_count {
                        let stacking_context_rect =
                            stacking_context.xf_rect.as_ref().unwrap().bounding_rect;
                        let location = RenderTaskLocation::Dynamic(None, stacking_context_rect.size);
                        let new_task = RenderTask::new_alpha_batch(next_task_index,
                                                                   stacking_context_rect.origin,
                                                                   location);
                        next_task_index.0 += 1;
                        let prev_task = mem::replace(&mut current_task, new_task);
                        alpha_task_stack.push(prev_task);
                    }
                }
                PrimitiveRunCmd::PopStackingContext => {
                    let stacking_context_index = sc_stack.pop().unwrap();
                    let stacking_context = &self.stacking_context_store[stacking_context_index.0];

                    if !stacking_context.is_visible() {
                        continue;
                    }

                    for filter in &stacking_context.composite_ops.filters {
                        let mut prev_task = alpha_task_stack.pop().unwrap();
                        let item = AlphaRenderItem::Blend(stacking_context_index,
                                                          current_task.id,
                                                          *filter,
                                                          next_z);
                        next_z += 1;
                        prev_task.as_alpha_batch().alpha_items.push(item);
                        prev_task.children.push(current_task);
                        current_task = prev_task;
                    }
                    if let Some(mix_blend_mode) = stacking_context.composite_ops.mix_blend_mode {
                        let stacking_context_rect =
                            stacking_context.xf_rect.as_ref().unwrap().bounding_rect;
                        let readback_task =
                            RenderTask::new_readback(stacking_context_index, stacking_context_rect);

                        let mut prev_task = alpha_task_stack.pop().unwrap();
                        let item = AlphaRenderItem::Composite(stacking_context_index,
                                                              readback_task.id,
                                                              current_task.id,
                                                              mix_blend_mode,
                                                              next_z);
                        next_z += 1;
                        prev_task.as_alpha_batch().alpha_items.push(item);
                        prev_task.children.push(current_task);
                        prev_task.children.push(readback_task);
                        current_task = prev_task;
                    }
                }
                PrimitiveRunCmd::PrimitiveRun(first_prim_index, prim_count) => {
                    let stacking_context_index = *sc_stack.last().unwrap();
                    let stacking_context = &self.stacking_context_store[stacking_context_index.0];

                    if !stacking_context.is_visible() {
                        continue;
                    }

                    for i in 0..prim_count {
                        let prim_index = PrimitiveIndex(first_prim_index.0 + i);

                        if self.prim_store.cpu_bounding_rects[prim_index.0].is_some() {
                            let prim_metadata = self.prim_store.get_metadata(prim_index);

                            // Add any dynamic render tasks needed to render this primitive
                            if let Some(ref render_task) = prim_metadata.render_task {
                                current_task.children.push(render_task.clone());
                            }
                            if let Some(ref clip_task) = prim_metadata.clip_task {
                                current_task.children.push(clip_task.clone());
                            }

                            let transform_kind = stacking_context.xf_rect.as_ref().unwrap().kind;
                            let needs_clipping = prim_metadata.clip_task.is_some();
                            let needs_blending = transform_kind == TransformedRectKind::Complex ||
                                                 !prim_metadata.is_opaque ||
                                                 needs_clipping;

                            let items = if needs_blending {
                                &mut current_task.as_alpha_batch().alpha_items
                            } else {
                                &mut current_task.as_alpha_batch().opaque_items
                            };
                            items.push(AlphaRenderItem::Primitive(stacking_context_index,
                                                                  prim_index,
                                                                  next_z));
                            next_z += 1;
                        }
                    }
                }
            }
        }

        debug_assert!(alpha_task_stack.is_empty());
        (current_task, next_task_index.0)
    }

    pub fn build(&mut self,
                 resource_cache: &mut ResourceCache,
                 frame_id: FrameId,
                 scroll_tree: &ScrollTree,
                 auxiliary_lists_map: &AuxiliaryListsMap,
                 device_pixel_ratio: f32) -> Frame {
        let mut profile_counters = FrameProfileCounters::new();
        profile_counters.total_primitives.set(self.prim_store.prim_count());

        resource_cache.begin_frame(frame_id);

        let screen_rect = DeviceIntRect::new(
            DeviceIntPoint::zero(),
            DeviceIntSize::from_lengths(device_length(self.screen_rect.size.width as f32,
                                                      device_pixel_ratio),
                                        device_length(self.screen_rect.size.height as f32,
                                                      device_pixel_ratio)));

        // Pick a size for the cache render targets to be. The main requirement is that it
        // has to be at least as large as the framebuffer size. This ensures that it will
        // always be able to allocate the worst case render task (such as a clip mask that
        // covers the entire screen).
        let cache_size = DeviceUintSize::new(cmp::max(1024, screen_rect.size.width as u32),
                                             cmp::max(1024, screen_rect.size.height as u32));

        self.update_scroll_bars(scroll_tree);

        self.cull_layers(&screen_rect,
                         scroll_tree,
                         auxiliary_lists_map,
                         resource_cache,
                         &mut profile_counters,
                         device_pixel_ratio);

        let (main_render_task, static_render_task_count) = self.build_render_task();
        let mut render_tasks = RenderTaskCollection::new(static_render_task_count);

        let mut required_pass_count = 0;
        main_render_task.max_depth(0, &mut required_pass_count);

        resource_cache.block_until_all_resources_added();

        for stacking_context in self.stacking_context_store.iter() {
            if let Some(ref clip_info) = stacking_context.clip_cache_info {
                self.prim_store.resolve_clip_cache(clip_info, resource_cache);
            }
        }

        let deferred_resolves = self.prim_store.resolve_primitives(resource_cache,
                                                                   device_pixel_ratio);

        let mut passes = Vec::new();

        // Do the allocations now, assigning each tile's tasks to a render
        // pass and target as required.
        for index in 0..required_pass_count {
            passes.push(RenderPass::new(index as isize,
                                        index == required_pass_count-1,
                                        cache_size));
        }

        main_render_task.assign_to_passes(passes.len() - 1, &mut passes);

        for pass in &mut passes {
            let ctx = RenderTargetContext {
                stacking_context_store: &self.stacking_context_store,
                prim_store: &self.prim_store,
                resource_cache: resource_cache,
            };

            pass.build(&ctx, &mut render_tasks);

            profile_counters.passes.inc();
            profile_counters.targets.add(pass.targets.len());
        }

        resource_cache.end_frame();

        Frame {
            device_pixel_ratio: device_pixel_ratio,
            background_color: self.background_color,
            viewport_size: self.screen_rect.size,
            profile_counters: profile_counters,
            passes: passes,
            cache_size: cache_size,
            layer_texture_data: self.packed_layers.clone(),
            render_task_data: render_tasks.render_task_data,
            gpu_data16: self.prim_store.gpu_data16.build(),
            gpu_data32: self.prim_store.gpu_data32.build(),
            gpu_data64: self.prim_store.gpu_data64.build(),
            gpu_data128: self.prim_store.gpu_data128.build(),
            gpu_geometry: self.prim_store.gpu_geometry.build(),
            gpu_gradient_data: self.prim_store.gpu_gradient_data.build(),
            gpu_resource_rects: self.prim_store.gpu_resource_rects.build(),
            deferred_resolves: deferred_resolves,
        }
    }
}
