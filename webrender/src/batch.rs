/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{AlphaType, ImageBufferKind};
use api::{FontInstanceFlags, YuvColorSpace, YuvFormat, ColorDepth, ColorRange};
use api::units::*;
use crate::command_buffer::PrimitiveCommand;
use crate::pattern::PatternKind;
use crate::spatial_tree::SpatialNodeIndex;
use glyph_rasterizer::{GlyphFormat, SubpixelDirection};
use crate::gpu_types::{BrushFlags, BrushInstance, PrimitiveHeaders, ZBufferId, ZBufferIdGenerator};
use crate::gpu_types::SplitCompositeInstance;
use crate::gpu_types::{PrimitiveInstanceData, RasterizationSpace, GlyphInstance};
use crate::gpu_types::{PrimitiveHeader, PrimitiveHeaderIndex};
use crate::gpu_types::{ImageBrushUserData, get_shader_opacity, MaskInstance};
use crate::internal_types::{FastHashMap, Filter, FrameAllocator, FrameMemory, FrameVec, Swizzle, TextureSource};
use crate::picture::{Picture3DContext, PictureCompositeMode};
use crate::prim_store::PrimitiveKind;
use crate::prim_store::{PrimitiveInstance, PrimitiveOpacity, SegmentInstanceIndex};
use crate::prim_store::{BrushSegment, ClipMaskKind, ClipTaskIndex};
use crate::quad;
use crate::render_target::RenderTargetContext;
use crate::render_task_graph::{RenderTaskId, RenderTaskGraph};
use crate::render_task::RenderTaskAddress;
use crate::renderer::{BlendMode, GpuBufferAddress, GpuBufferBuilder, ShaderColorMode};
use crate::resource_cache::GlyphFetchResult;
use crate::space::SpaceMapper;
use crate::transform::{TransformPalette, TransformMetadata};
use crate::visibility::{PrimitiveVisibilityFlags, DrawState};
use std::{f32, i32, usize};
use crate::util::{MaxRect, ScaleOffset};
use crate::segment::EdgeMask;


// Special sentinel value recognized by the shader. It is considered to be
// a dummy task that doesn't mask out anything.
const OPAQUE_TASK_ADDRESS: RenderTaskAddress = RenderTaskAddress(0x7fffffff);

/// Used to signal there are no segments provided with this primitive.
pub const INVALID_SEGMENT_INDEX: i32 = 0xffff;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum BrushBatchKind {
    Solid,
    Image(ImageBufferKind),
    Blend,
    MixBlend {
        task_id: RenderTaskId,
        backdrop_id: RenderTaskId,
    },
    YuvImage(ImageBufferKind, YuvFormat, ColorDepth, YuvColorSpace, ColorRange),
    Opacity,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum BatchKind {
    SplitComposite,
    TextRun(GlyphFormat),
    Brush(BrushBatchKind),
    Quad(PatternKind),
}

/// Input textures for a primitive, without consideration of clip mask
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct TextureSet {
    pub colors: [TextureSource; 3],
}

impl TextureSet {
    const UNTEXTURED: TextureSet = TextureSet {
        colors: [
            TextureSource::Invalid,
            TextureSource::Invalid,
            TextureSource::Invalid,
        ],
    };

    /// A textured primitive
    fn prim_textured(
        color: TextureSource,
    ) -> Self {
        TextureSet {
            colors: [
                color,
                TextureSource::Invalid,
                TextureSource::Invalid,
            ],
        }
    }

    fn is_compatible_with(&self, other: &TextureSet) -> bool {
        self.colors[0].is_compatible(&other.colors[0]) &&
        self.colors[1].is_compatible(&other.colors[1]) &&
        self.colors[2].is_compatible(&other.colors[2])
    }
}

impl TextureSource {
    fn combine(&self, other: TextureSource) -> TextureSource {
        if other == TextureSource::Invalid {
            *self
        } else {
            other
        }
    }
}

/// Optional textures that can be used as a source in the shaders.
/// Textures that are not used by the batch are equal to TextureId::invalid().
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct BatchTextures {
    pub input: TextureSet,
    pub clip_mask: TextureSource,
}

impl BatchTextures {
    /// An empty batch textures (no binding slots set)
    pub fn empty() -> BatchTextures {
        BatchTextures {
            input: TextureSet::UNTEXTURED,
            clip_mask: TextureSource::Invalid,
        }
    }

    /// A textured primitive with optional clip mask
    pub fn prim_textured(
        color: TextureSource,
        clip_mask: TextureSource,
    ) -> BatchTextures {
        BatchTextures {
            input: TextureSet::prim_textured(color),
            clip_mask,
        }
    }

    /// An untextured primitive with optional clip mask
    pub fn prim_untextured(
        clip_mask: TextureSource,
    ) -> BatchTextures {
        BatchTextures {
            input: TextureSet::UNTEXTURED,
            clip_mask,
        }
    }

    /// A composite style effect with single input texture
    pub fn composite_rgb(
        texture: TextureSource,
    ) -> BatchTextures {
        BatchTextures {
            input: TextureSet {
                colors: [
                    texture,
                    TextureSource::Invalid,
                    TextureSource::Invalid,
                ],
            },
            clip_mask: TextureSource::Invalid,
        }
    }

    /// A composite style effect with up to 3 input textures
    pub fn composite_yuv(
        color0: TextureSource,
        color1: TextureSource,
        color2: TextureSource,
    ) -> BatchTextures {
        BatchTextures {
            input: TextureSet {
                colors: [color0, color1, color2],
            },
            clip_mask: TextureSource::Invalid,
        }
    }

    pub fn is_compatible_with(&self, other: &BatchTextures) -> bool {
        if !self.clip_mask.is_compatible(&other.clip_mask) {
            return false;
        }

        self.input.is_compatible_with(&other.input)
    }

    pub fn combine_textures(&self, other: BatchTextures) -> Option<BatchTextures> {
        if !self.is_compatible_with(&other) {
            return None;
        }

        let mut new_textures = BatchTextures::empty();

        new_textures.clip_mask = self.clip_mask.combine(other.clip_mask);

        for i in 0 .. 3 {
            new_textures.input.colors[i] = self.input.colors[i].combine(other.input.colors[i]);
        }

        Some(new_textures)
    }

    fn merge(&mut self, other: &BatchTextures) {
        self.clip_mask = self.clip_mask.combine(other.clip_mask);

        for (s, o) in self.input.colors.iter_mut().zip(other.input.colors.iter()) {
            *s = s.combine(*o);
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct BatchKey {
    pub kind: BatchKind,
    pub blend_mode: BlendMode,
    pub textures: BatchTextures,
}

impl BatchKey {
    pub fn new(kind: BatchKind, blend_mode: BlendMode, textures: BatchTextures) -> Self {
        BatchKey {
            kind,
            blend_mode,
            textures,
        }
    }

    pub fn is_compatible_with(&self, other: &BatchKey) -> bool {
        self.kind == other.kind && self.blend_mode == other.blend_mode && self.textures.is_compatible_with(&other.textures)
    }
}

pub struct BatchRects {
    /// Union of all of the batch's item rects.
    ///
    /// Very often we can skip iterating over item rects by testing against
    /// this one first.
    batch: PictureRect,
    /// When the batch rectangle above isn't a good enough approximation, we
    /// store per item rects.
    items: Option<FrameVec<PictureRect>>,
    // TODO: batch rects don't need to be part of the frame but they currently
    // are. It may be cleaner to remove them from the frame's final data structure
    // and not use the frame's allocator.
    allocator: FrameAllocator,
}

impl BatchRects {
    fn new(allocator: FrameAllocator) -> Self {
        BatchRects {
            batch: PictureRect::zero(),
            items: None,
            allocator,
        }
    }

    #[inline]
    fn add_rect(&mut self, rect: &PictureRect) {
        let union = self.batch.union(rect);
        // If we have already started storing per-item rects, continue doing so.
        // Otherwise, check whether only storing the batch rect is a good enough
        // approximation.
        if let Some(items) = &mut self.items {
            items.push(*rect);
        } else if self.batch.area() + rect.area() < union.area() {
            let mut items = self.allocator.clone().new_vec_with_capacity(16);
            items.push(self.batch);
            items.push(*rect);
            self.items = Some(items);
        }

        self.batch = union;
    }

    #[inline]
    fn intersects(&mut self, rect: &PictureRect) -> bool {
        if !self.batch.intersects(rect) {
            return false;
        }

        if let Some(items) = &self.items {
            items.iter().any(|item| item.intersects(rect))
        } else {
            // If we don't have per-item rects it means the batch rect is a good
            // enough approximation and we didn't bother storing per-rect items.
            true
        }
    }
}


pub struct AlphaBatchList {
    pub batches: FrameVec<PrimitiveBatch>,
    pub batch_rects: FrameVec<BatchRects>,
    current_batch_index: usize,
    current_z_id: ZBufferId,
    break_advanced_blend_batches: bool,
}

impl AlphaBatchList {
    fn new(break_advanced_blend_batches: bool, preallocate: usize, memory: &FrameMemory) -> Self {
        AlphaBatchList {
            batches: memory.new_vec_with_capacity(preallocate),
            batch_rects: memory.new_vec_with_capacity(preallocate),
            current_z_id: ZBufferId::invalid(),
            current_batch_index: usize::MAX,
            break_advanced_blend_batches,
        }
    }

    /// Clear all current batches in this list. This is typically used
    /// when a primitive is encountered that occludes all previous
    /// content in this batch list.
    fn clear(&mut self) {
        self.current_batch_index = usize::MAX;
        self.current_z_id = ZBufferId::invalid();
        self.batches.clear();
        self.batch_rects.clear();
    }

    pub fn set_params_and_get_batch(
        &mut self,
        key: BatchKey,
        features: BatchFeatures,
        // The bounding box of everything at this Z plane. We expect potentially
        // multiple primitive segments coming with the same `z_id`.
        z_bounding_rect: &PictureRect,
        z_id: ZBufferId,
    ) -> &mut FrameVec<PrimitiveInstanceData> {
        if z_id != self.current_z_id ||
           self.current_batch_index == usize::MAX ||
           !self.batches[self.current_batch_index].key.is_compatible_with(&key)
        {
            let mut selected_batch_index = None;

            match key.blend_mode {
                BlendMode::Advanced(_) if self.break_advanced_blend_batches => {
                    // don't try to find a batch
                }
                _ => {
                    for (batch_index, batch) in self.batches.iter().enumerate().rev() {
                        // For normal batches, we only need to check for overlaps for batches
                        // other than the first batch we consider. If the first batch
                        // is compatible, then we know there isn't any potential overlap
                        // issues to worry about.
                        if batch.key.is_compatible_with(&key) {
                            selected_batch_index = Some(batch_index);
                            break;
                        }

                        // check for intersections
                        if self.batch_rects[batch_index].intersects(z_bounding_rect) {
                            break;
                        }
                    }
                }
            }

            if selected_batch_index.is_none() {
                // Text runs tend to have a lot of instances per batch, causing a lot of reallocation
                // churn as items are added one by one, so we give it a head start. Ideally we'd start
                // with a larger number, closer to 1k but in some bad cases with lots of batch break
                // we would be wasting a lot of memory.
                // Generally it is safe to preallocate small-ish values for other batch kinds because
                // the items are small and there are no zero-sized batches so there will always be
                // at least one allocation.
                let prealloc = match key.kind {
                    BatchKind::TextRun(..) => 128,
                    _ => 16,
                };
                let mut new_batch = PrimitiveBatch::new(key, self.batches.allocator().clone());
                new_batch.instances.reserve(prealloc);
                selected_batch_index = Some(self.batches.len());
                self.batches.push(new_batch);
                self.batch_rects.push(BatchRects::new(self.batches.allocator().clone()));
            }

            self.current_batch_index = selected_batch_index.unwrap();
            self.batch_rects[self.current_batch_index].add_rect(z_bounding_rect);
            self.current_z_id = z_id;
        }

        let batch = &mut self.batches[self.current_batch_index];
        batch.features |= features;
        batch.key.textures.merge(&key.textures);

        &mut batch.instances
    }
}

pub struct OpaqueBatchList {
    pub pixel_area_threshold_for_new_batch: f32,
    pub batches: FrameVec<PrimitiveBatch>,
    pub current_batch_index: usize,
    lookback_count: usize,
}

impl OpaqueBatchList {
    fn new(pixel_area_threshold_for_new_batch: f32, lookback_count: usize, memory: &FrameMemory) -> Self {
        OpaqueBatchList {
            batches: memory.new_vec(),
            pixel_area_threshold_for_new_batch,
            current_batch_index: usize::MAX,
            lookback_count,
        }
    }

    /// Clear all current batches in this list. This is typically used
    /// when a primitive is encountered that occludes all previous
    /// content in this batch list.
    fn clear(&mut self) {
        self.current_batch_index = usize::MAX;
        self.batches.clear();
    }

    pub fn set_params_and_get_batch(
        &mut self,
        key: BatchKey,
        features: BatchFeatures,
        // The bounding box of everything at the current Z, whatever it is. We expect potentially
        // multiple primitive segments produced by a primitive, which we allow to check
        // `current_batch_index` instead of iterating the batches.
        z_bounding_rect: &PictureRect,
    ) -> &mut FrameVec<PrimitiveInstanceData> {
        // If the area of this primitive is larger than the given threshold,
        // then it is large enough to warrant breaking a batch for. In this
        // case we just see if it can be added to the existing batch or
        // create a new one.
        let is_large_occluder = z_bounding_rect.area() > self.pixel_area_threshold_for_new_batch;
        // Since primitives of the same kind tend to come in succession, we keep track
        // of the current batch index to skip the search in some cases. We ignore the
        // current batch index in the case of large occluders to make sure they get added
        // at the top of the bach list.
        if is_large_occluder || self.current_batch_index == usize::MAX ||
           !self.batches[self.current_batch_index].key.is_compatible_with(&key) {
            let mut selected_batch_index = None;
            if is_large_occluder {
                if let Some(batch) = self.batches.last() {
                    if batch.key.is_compatible_with(&key) {
                        selected_batch_index = Some(self.batches.len() - 1);
                    }
                }
            } else {
                // Otherwise, look back through a reasonable number of batches.
                for (batch_index, batch) in self.batches.iter().enumerate().rev().take(self.lookback_count) {
                    if batch.key.is_compatible_with(&key) {
                        selected_batch_index = Some(batch_index);
                        break;
                    }
                }
            }

            if selected_batch_index.is_none() {
                let new_batch = PrimitiveBatch::new(key, self.batches.allocator().clone());
                selected_batch_index = Some(self.batches.len());
                self.batches.push(new_batch);
            }

            self.current_batch_index = selected_batch_index.unwrap();
        }

        let batch = &mut self.batches[self.current_batch_index];
        batch.features |= features;
        batch.key.textures.merge(&key.textures);

        &mut batch.instances
    }

    fn finalize(&mut self) {
        // Reverse the instance arrays in the opaque batches
        // to get maximum z-buffer efficiency by drawing
        // front-to-back.
        // TODO(gw): Maybe we can change the batch code to
        //           build these in reverse and avoid having
        //           to reverse the instance array here.
        for batch in &mut self.batches {
            batch.instances.reverse();
        }
    }
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct PrimitiveBatch {
    pub key: BatchKey,
    pub instances: FrameVec<PrimitiveInstanceData>,
    pub features: BatchFeatures,
}

bitflags! {
    /// Features of the batch that, if not requested, may allow a fast-path.
    ///
    /// Rather than breaking batches when primitives request different features,
    /// we always request the minimum amount of features to satisfy all items in
    /// the batch.
    /// The goal is to let the renderer be optionally select more specialized
    /// versions of a shader if the batch doesn't require code certain code paths.
    /// Not all shaders necessarily implement all of these features.
    #[cfg_attr(feature = "capture", derive(Serialize))]
    #[cfg_attr(feature = "replay", derive(Deserialize))]
    #[derive(Debug, Copy, PartialEq, Eq, Clone, PartialOrd, Ord, Hash)]
    pub struct BatchFeatures: u8 {
        const ALPHA_PASS = 1 << 0;
        const ANTIALIASING = 1 << 1;
        const REPETITION = 1 << 2;
        /// Indicates a primitive in this batch may use a clip mask.
        const CLIP_MASK = 1 << 3;
    }
}

impl PrimitiveBatch {
    fn new(key: BatchKey, allocator: FrameAllocator) -> PrimitiveBatch {
        PrimitiveBatch {
            key,
            instances: FrameVec::new_in(allocator),
            features: BatchFeatures::empty(),
        }
    }

    fn merge(&mut self, other: PrimitiveBatch) {
        self.instances.extend(other.instances);
        self.features |= other.features;
        self.key.textures.merge(&other.key.textures);
    }
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct AlphaBatchContainer {
    pub opaque_batches: FrameVec<PrimitiveBatch>,
    pub alpha_batches: FrameVec<PrimitiveBatch>,
    /// The overall scissor rect for this render task, if one
    /// is required.
    pub task_scissor_rect: Option<DeviceIntRect>,
    /// The rectangle of the owning render target that this
    /// set of batches affects.
    pub task_rect: DeviceIntRect,
}

impl AlphaBatchContainer {
    pub fn new(
        task_scissor_rect: Option<DeviceIntRect>,
        memory: &FrameMemory,
    ) -> AlphaBatchContainer {
        AlphaBatchContainer {
            opaque_batches: memory.new_vec(),
            alpha_batches: memory.new_vec(),
            task_scissor_rect,
            task_rect: DeviceIntRect::zero(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.opaque_batches.is_empty() &&
        self.alpha_batches.is_empty()
    }

    fn merge(&mut self, builder: AlphaBatchBuilder, task_rect: &DeviceIntRect) {
        self.task_rect = self.task_rect.union(task_rect);

        for other_batch in builder.opaque_batch_list.batches {
            let batch_index = self.opaque_batches.iter().position(|batch| {
                batch.key.is_compatible_with(&other_batch.key)
            });

            match batch_index {
                Some(batch_index) => {
                    self.opaque_batches[batch_index].merge(other_batch);
                }
                None => {
                    self.opaque_batches.push(other_batch);
                }
            }
        }

        let mut min_batch_index = 0;

        for other_batch in builder.alpha_batch_list.batches {
            let batch_index = self.alpha_batches.iter().skip(min_batch_index).position(|batch| {
                batch.key.is_compatible_with(&other_batch.key)
            });

            match batch_index {
                Some(batch_index) => {
                    let index = batch_index + min_batch_index;
                    self.alpha_batches[index].merge(other_batch);
                    min_batch_index = index;
                }
                None => {
                    self.alpha_batches.push(other_batch);
                    min_batch_index = self.alpha_batches.len();
                }
            }
        }
    }
}

/// Each segment can optionally specify a per-segment
/// texture set and one user data field.
#[derive(Debug, Copy, Clone)]
struct SegmentInstanceData {
    textures: TextureSet,
    specific_resource_address: i32,
}

/// Encapsulates the logic of building batches for items that are blended.
pub struct AlphaBatchBuilder {
    pub alpha_batch_list: AlphaBatchList,
    pub opaque_batch_list: OpaqueBatchList,
    pub render_task_id: RenderTaskId,
    render_task_address: RenderTaskAddress,
}

impl AlphaBatchBuilder {
    pub fn new(
        screen_size: DeviceIntSize,
        break_advanced_blend_batches: bool,
        lookback_count: usize,
        render_task_id: RenderTaskId,
        render_task_address: RenderTaskAddress,
        memory: &FrameMemory,
    ) -> Self {
        // The threshold for creating a new batch is
        // one quarter the screen size.
        let batch_area_threshold = (screen_size.width * screen_size.height) as f32 / 4.0;

        AlphaBatchBuilder {
            alpha_batch_list: AlphaBatchList::new(break_advanced_blend_batches, 128, memory),
            opaque_batch_list: OpaqueBatchList::new(batch_area_threshold, lookback_count, memory),
            render_task_id,
            render_task_address,
        }
    }

    /// Clear all current batches in this builder. This is typically used
    /// when a primitive is encountered that occludes all previous
    /// content in this batch list.
    fn clear(&mut self) {
        self.alpha_batch_list.clear();
        self.opaque_batch_list.clear();
    }

    pub fn build(
        mut self,
        batch_containers: &mut FrameVec<AlphaBatchContainer>,
        merged_batches: &mut AlphaBatchContainer,
        task_rect: DeviceIntRect,
        task_scissor_rect: Option<DeviceIntRect>,
    ) {
        self.opaque_batch_list.finalize();

        if task_scissor_rect.is_none() {
            merged_batches.merge(self, &task_rect);
        } else {
            batch_containers.push(AlphaBatchContainer {
                alpha_batches: self.alpha_batch_list.batches,
                opaque_batches: self.opaque_batch_list.batches,
                task_scissor_rect,
                task_rect,
            });
        }
    }

    pub fn push_single_instance(
        &mut self,
        key: BatchKey,
        features: BatchFeatures,
        bounding_rect: &PictureRect,
        z_id: ZBufferId,
        instance: PrimitiveInstanceData,
    ) {
        self.set_params_and_get_batch(key, features, bounding_rect, z_id)
            .push(instance);
    }

    pub fn set_params_and_get_batch(
        &mut self,
        key: BatchKey,
        features: BatchFeatures,
        bounding_rect: &PictureRect,
        z_id: ZBufferId,
    ) -> &mut FrameVec<PrimitiveInstanceData> {
        match key.blend_mode {
            BlendMode::None => {
                self.opaque_batch_list
                    .set_params_and_get_batch(key, features, bounding_rect)
            }
            BlendMode::Alpha |
            BlendMode::PremultipliedAlpha |
            BlendMode::PremultipliedDestOut |
            BlendMode::SubpixelDualSource |
            BlendMode::Advanced(_) |
            BlendMode::MultiplyDualSource |
            BlendMode::Screen |
            BlendMode::Exclusion |
            BlendMode::PlusLighter => {
                self.alpha_batch_list
                    .set_params_and_get_batch(key, features, bounding_rect, z_id)
            }
        }
    }
}

/// Supports (recursively) adding a list of primitives and pictures to an alpha batch
/// builder. In future, it will support multiple dirty regions / slices, allowing the
/// contents of a picture to be spliced into multiple batch builders.
pub struct BatchBuilder {
    /// A temporary buffer that is used during glyph fetching, stored here
    /// to reduce memory allocations.
    glyph_fetch_buffer: Vec<GlyphFetchResult>,

    batcher: AlphaBatchBuilder,
}

impl BatchBuilder {
    pub fn new(batcher: AlphaBatchBuilder) -> Self {
        BatchBuilder {
            glyph_fetch_buffer: Vec::new(),
            batcher,
        }
    }

    pub fn finalize(self) -> AlphaBatchBuilder {
        self.batcher
    }

    fn add_brush_instance_to_batches(
        &mut self,
        batch_key: BatchKey,
        features: BatchFeatures,
        bounding_rect: &PictureRect,
        z_id: ZBufferId,
        segment_index: i32,
        edge_flags: EdgeMask,
        clip_task_address: RenderTaskAddress,
        brush_flags: BrushFlags,
        prim_header_index: PrimitiveHeaderIndex,
        resource_address: i32,
    ) {
        assert!(
            !(brush_flags.contains(BrushFlags::NORMALIZED_UVS)
                && features.contains(BatchFeatures::REPETITION)),
            "Normalized UVs are not supported with repetition."
        );
        let instance = BrushInstance {
            segment_index,
            edge_flags,
            clip_task_address,
            brush_flags,
            prim_header_index,
            resource_address,
        };

        self.batcher.push_single_instance(
            batch_key,
            features,
            bounding_rect,
            z_id,
            PrimitiveInstanceData::from(instance),
        );
    }

    fn add_split_composite_instance_to_batches(
        &mut self,
        batch_key: BatchKey,
        features: BatchFeatures,
        bounding_rect: &PictureRect,
        z_id: ZBufferId,
        prim_header_index: PrimitiveHeaderIndex,
        polygons_address: i32,
    ) {
        let render_task_address = self.batcher.render_task_address;

        self.batcher.push_single_instance(
            batch_key,
            features,
            bounding_rect,
            z_id,
            PrimitiveInstanceData::from(SplitCompositeInstance {
                prim_header_index,
                render_task_address,
                polygons_address,
                z: z_id,
            }),
        );
    }

    /// Clear all current batchers. This is typically used when a primitive
    /// is encountered that occludes all previous content in this batch list.
    fn clear_batches(&mut self) {
        self.batcher.clear();
    }

    // Adds a primitive to a batch.
    // It can recursively call itself in some situations, for
    // example if it encounters a picture where the items
    // in that picture are being drawn into the same target.
    pub fn add_prim_to_batch(
        &mut self,
        cmd: &PrimitiveCommand,
        prim_spatial_node_index: SpatialNodeIndex,
        ctx: &RenderTargetContext,
        render_tasks: &RenderTaskGraph,
        prim_headers: &mut PrimitiveHeaders,
        transforms: &mut TransformPalette,
        root_spatial_node_index: SpatialNodeIndex,
        surface_spatial_node_index: SpatialNodeIndex,
        z_generator: &mut ZBufferIdGenerator,
        prim_instances: &[PrimitiveInstance],
        gpu_buffer_builder: &mut GpuBufferBuilder,
        segments: &[RenderTaskId],
    ) {
        let (draw_index, extra_prim_gpu_address) = match cmd {
            PrimitiveCommand::Simple { draw_index } => {
                (draw_index, None)
            }
            PrimitiveCommand::Complex { draw_index, gpu_address } => {
                (draw_index, Some(gpu_address.as_int()))
            }
            PrimitiveCommand::Instance { draw_index, gpu_buffer_address } => {
                (draw_index, Some(gpu_buffer_address.as_int()))
            }
            PrimitiveCommand::Quad { pattern, pattern_input, draw_index, gpu_buffer_address, quad_flags, edge_flags, transform_id, src_color_task_ids, blend_mode } => {
                let prim_info = &ctx.scratch.frame.draws[draw_index.0 as usize];
                let bounding_rect = &prim_info.clip_chain.pic_coverage_rect;
                let render_task_address = self.batcher.render_task_address;

                if segments.is_empty() {
                    let z_id = z_generator.next();

                    quad::add_to_batch(
                        *pattern,
                        *pattern_input,
                        render_task_address,
                        *transform_id,
                        *gpu_buffer_address,
                        *quad_flags,
                        *edge_flags,
                        INVALID_SEGMENT_INDEX as u8,
                        *src_color_task_ids,
                        z_id,
                        *blend_mode,
                        render_tasks,
                        gpu_buffer_builder,
                        |key, instance| {
                            let batch = self.batcher.set_params_and_get_batch(
                                key,
                                BatchFeatures::empty(),
                                bounding_rect,
                                z_id,
                            );
                            batch.push(instance);
                        },
                    );
                } else {
                    for (i, task_id) in segments.iter().enumerate() {
                        // TODO(gw): edge_flags should be per-segment, when used for more than composites
                        debug_assert!(edge_flags.is_empty());

                        let z_id = z_generator.next();

                        quad::add_to_batch(
                            *pattern,
                            *pattern_input,
                            render_task_address,
                            *transform_id,
                            *gpu_buffer_address,
                            *quad_flags,
                            *edge_flags,
                            i as u8,
                            [*task_id, src_color_task_ids[1], src_color_task_ids[2]],
                            z_id,
                            *blend_mode,
                            render_tasks,
                            gpu_buffer_builder,
                            |key, instance| {
                                let batch = self.batcher.set_params_and_get_batch(
                                    key,
                                    BatchFeatures::empty(),
                                    bounding_rect,
                                    z_id,
                                );
                                batch.push(instance);
                            },
                        );
                    }
                }

                return;
            }
        };

        let prim_instance = &prim_instances[draw_index.0 as usize];
        let is_anti_aliased = ctx.data_stores.prim_has_anti_aliasing(prim_instance);

        let brush_flags = if is_anti_aliased {
            BrushFlags::FORCE_AA
        } else {
            BrushFlags::empty()
        };

        let vis_flags = match ctx.scratch.frame.draws[draw_index.0 as usize].state {
            DrawState::Culled => {
                return;
            }
            DrawState::PassThrough |
            DrawState::Unset => {
                panic!("bug: invalid visibility state");
            }
            DrawState::Visible { vis_flags, .. } => {
                vis_flags
            }
        };

        // If this primitive is a backdrop, that means that it is known to cover
        // the entire picture cache background. In that case, the renderer will
        // use the backdrop color as a clear color, and so we can drop this
        // primitive and any prior primitives from the batch lists for this
        // picture cache slice.
        if vis_flags.contains(PrimitiveVisibilityFlags::IS_BACKDROP) {
            self.clear_batches();
            return;
        }

        let transform_id = transforms.gpu.get_id(
            prim_spatial_node_index,
            root_spatial_node_index,
            ctx.spatial_tree,
        );

        // TODO(gw): Calculating this for every primitive is a bit
        //           wasteful. We should probably cache this in
        //           the scroll node...
        let transform_metadata = transform_id.metadata();
        let prim_info = &ctx.scratch.frame.draws[draw_index.0 as usize];
        let bounding_rect = &prim_info.clip_chain.pic_coverage_rect;

        let mut z_id = z_generator.next();

        let prim_rect = ctx.data_stores.get_local_prim_rect(
            prim_instance,
            prim_info.snapped_local_rect,
            &ctx.prim_store.pictures,
            ctx.surfaces,
        );

        let mut batch_features = BatchFeatures::empty();
        let may_need_repetition = match prim_instance.kind {
            PrimitiveKind::Image { .. } => {
                let idx = prim_info.kind_scratch.unwrap_image();
                ctx.scratch.frame.images[idx].may_need_repetition
            }
            // Image borders always go through brush_image and may tile
            // their mid sections, so request the repetition-capable
            // shader.
            PrimitiveKind::ImageBorder { .. } => true,
            // Patterned line decorations (Dashed / Dotted / Wavy) batch
            // as `BrushBatchKind::Image` over a cached pattern tile and
            // rely on shader-level repetition to span the segment.
            // Solid lines batch as `BrushBatchKind::Solid`, where the
            // REPETITION flag is harmless.
            PrimitiveKind::LineDecoration { .. } => true,
            // Other prim kinds don't reach the brush_image consumer of
            // BatchFeatures::REPETITION; the flag is dead state for
            // them.
            _ => false,
        };
        if may_need_repetition {
            batch_features |= BatchFeatures::REPETITION;
        }

        if !transform_id.is_2d_axis_aligned() || is_anti_aliased {
            batch_features |= BatchFeatures::ANTIALIASING;
        }

        // Check if the primitive might require a clip mask.
        if prim_info.clip_task_index != ClipTaskIndex::INVALID {
            batch_features |= BatchFeatures::CLIP_MASK;
        }

        if !bounding_rect.is_empty() {
            debug_assert_eq!(prim_info.clip_chain.pic_spatial_node_index, surface_spatial_node_index,
                "The primitive's bounding box is specified in a different coordinate system from the current batch!");
        }

        if let PrimitiveKind::Picture { pic_index, .. } = prim_instance.kind {
            let pic_scratch_handle = ctx.scratch.frame.draws[draw_index.0 as usize].kind_scratch.unwrap_picture();
            let picture = &ctx.prim_store.pictures[pic_index.0];
            let picture_scratch = &ctx.scratch.frame.pictures[pic_scratch_handle];
            if let Some(snapshot) = picture.snapshot {
                if snapshot.detached {
                    return;
                }
            }

            let blend_mode = BlendMode::PremultipliedAlpha;
            let prim_cache_address = ctx.globals.default_image_data;

            match picture.raster_config {
                Some(ref raster_config) => {
                    // If the child picture was rendered in local space, we can safely
                    // interpolate the UV coordinates with perspective correction.
                    let brush_flags = brush_flags | BrushFlags::PERSPECTIVE_INTERPOLATION;

                    let surface = &ctx.surfaces[raster_config.surface_index.0];
                    let mut local_clip_rect = prim_info.clip_chain.local_clip_rect;

                    // If we are drawing with snapping enabled, form a simple transform that just applies
                    // the scale / translation from the raster transform. Otherwise, in edge cases where the
                    // intermediate surface has a non-identity but axis-aligned transform (e.g. a 180 degree
                    // rotation) it can be applied twice.
                    let transform_id = if surface.surface_spatial_node_index == surface.raster_spatial_node_index {
                        transform_id
                    } else {
                        let map_local_to_raster = SpaceMapper::new_with_target(
                            root_spatial_node_index,
                            surface.surface_spatial_node_index,
                            LayoutRect::max_rect(),
                            ctx.spatial_tree,
                        );

                        let raster_rect = map_local_to_raster
                            .map(&prim_rect)
                            .unwrap();

                        let sx = (raster_rect.max.x - raster_rect.min.x) / (prim_rect.max.x - prim_rect.min.x);
                        let sy = (raster_rect.max.y - raster_rect.min.y) / (prim_rect.max.y - prim_rect.min.y);

                        let tx = raster_rect.min.x - sx * prim_rect.min.x;
                        let ty = raster_rect.min.y - sy * prim_rect.min.y;

                        let transform = ScaleOffset::new(sx, sy, tx, ty);

                        let raster_clip_rect = map_local_to_raster
                            .map(&prim_info.clip_chain.local_clip_rect)
                            .unwrap();
                        local_clip_rect = transform.unmap_rect(&raster_clip_rect);

                        transforms.gpu.get_custom(transform.to_transform())
                    };

                    let picture_prim_header = PrimitiveHeader {
                        local_rect: prim_rect,
                        local_clip_rect,
                        specific_prim_address: prim_cache_address.as_int(),
                        transform_id,
                        z: z_id,
                        render_task_address: self.batcher.render_task_address,
                        user_data: [0; 4], // Will be overridden by most uses
                    };

                    let mut is_opaque = prim_info.clip_task_index == ClipTaskIndex::INVALID
                        && surface.is_opaque
                        && transform_id.is_2d_axis_aligned()
                        && !is_anti_aliased
                        && !prim_info.clip_chain.needs_mask;

                    match raster_config.composite_mode {
                        PictureCompositeMode::TileCache { .. } => {
                            // TODO(gw): For now, TileCache is still a composite mode, even though
                            //           it will only exist as a top level primitive and never
                            //           be encountered during batching. Consider making TileCache
                            //           a standalone type, not a picture.
                            return;
                        }
                        PictureCompositeMode::IntermediateSurface { .. } => {
                            // TODO(gw): As an optimization, support making this a pass-through
                            //           and/or drawing directly from here when possible
                            //           (e.g. if not wrapped by filters / different spatial node).
                            return;
                        }
                        _=>{}
                    }

                    let (clip_task_address, clip_mask_texture_id) = ctx.get_prim_clip_task_and_texture(
                        prim_info.clip_task_index,
                        render_tasks,
                    ).unwrap();

                    let pic_task_id = picture_scratch.primary_render_task_id.unwrap();

                    let (uv_rect_address, texture) = render_tasks.resolve_location(
                        pic_task_id,

                    ).unwrap();

                    // The set of input textures that most composite modes use,
                    // howevr some override it.
                    let textures = BatchTextures::prim_textured(
                        texture,
                        clip_mask_texture_id,
                    );

                    let (key, prim_user_data, resource_address) = match raster_config.composite_mode {
                        PictureCompositeMode::TileCache { .. }
                        | PictureCompositeMode::IntermediateSurface { .. }
                        => return,
                        PictureCompositeMode::Filter(ref filter) => {
                            assert!(filter.is_visible());
                            match filter {
                                Filter::Blur { .. } => {
                                    let kind = BatchKind::Brush(
                                        BrushBatchKind::Image(ImageBufferKind::Texture2D)
                                    );

                                    let key = BatchKey::new(
                                        kind,
                                        blend_mode,
                                        textures,
                                    );

                                    let prim_user_data = ImageBrushUserData {
                                        color_mode: ShaderColorMode::Image,
                                        alpha_type: AlphaType::PremultipliedAlpha,
                                        raster_space: RasterizationSpace::Screen,
                                        opacity: 1.0,
                                    }.encode();

                                    (key, prim_user_data, uv_rect_address.as_int())
                                }
                                Filter::DropShadows(shadows) => {
                                    // Draw an instance per shadow first, following by the content.

                                    // The shadows and the content get drawn as a brush image.
                                    let kind = BatchKind::Brush(
                                        BrushBatchKind::Image(ImageBufferKind::Texture2D),
                                    );

                                    // Gets the saved render task ID of the content, which is
                                    // deeper in the render task graph than the direct child.
                                    let secondary_id = picture_scratch.secondary_render_task_id.expect("no secondary!?");
                                    let content_source = {
                                        let secondary_task = &render_tasks[secondary_id];
                                        let texture_id = secondary_task.get_target_texture();
                                        TextureSource::TextureCache(
                                            texture_id,
                                            Swizzle::default(),
                                        )
                                    };

                                    // Retrieve the UV rect addresses for shadow/content.
                                    let shadow_uv_rect_address = uv_rect_address;
                                    let shadow_textures = textures;

                                    let content_uv_rect_address = render_tasks[secondary_id]
                                        .get_texture_address()
                                        .as_int();

                                    // Build BatchTextures for shadow/content
                                    let content_textures = BatchTextures::prim_textured(
                                        content_source,
                                        clip_mask_texture_id,
                                    );

                                    // Build batch keys for shadow/content
                                    let shadow_key = BatchKey::new(kind, blend_mode, shadow_textures);
                                    let content_key = BatchKey::new(kind, blend_mode, content_textures);

                                    for (shadow, shadow_prim_address) in shadows.iter().zip(picture_scratch.extra_gpu_data.iter()) {
                                        let shadow_rect = picture_prim_header.local_rect.translate(shadow.offset);

                                        let shadow_prim_header = PrimitiveHeader {
                                            local_rect: shadow_rect,
                                            specific_prim_address: shadow_prim_address.as_int(),
                                            z: z_id,
                                            user_data: ImageBrushUserData {
                                                color_mode: ShaderColorMode::Alpha,
                                                alpha_type: AlphaType::PremultipliedAlpha,
                                                raster_space: RasterizationSpace::Screen,
                                                opacity: 1.0,
                                            }.encode(),
                                            ..picture_prim_header
                                        };
                                        let shadow_prim_header_index = prim_headers.push(&shadow_prim_header);

                                        self.add_brush_instance_to_batches(
                                            shadow_key,
                                            batch_features,
                                            bounding_rect,
                                            z_id,
                                            INVALID_SEGMENT_INDEX,
                                            EdgeMask::all(),
                                            clip_task_address,
                                            brush_flags,
                                            shadow_prim_header_index,
                                            shadow_uv_rect_address.as_int(),
                                        );
                                    }

                                    // Update z_id for the content
                                    z_id = z_generator.next();

                                    let prim_user_data = ImageBrushUserData {
                                        color_mode: ShaderColorMode::Image,
                                        alpha_type: AlphaType::PremultipliedAlpha,
                                        raster_space: RasterizationSpace::Screen,
                                        opacity: 1.0,
                                    }.encode();

                                    (content_key, prim_user_data, content_uv_rect_address)
                                }
                                Filter::Opacity(_, amount) => {
                                    let amount = (amount * 65536.0) as i32;

                                    let key = BatchKey::new(
                                        BatchKind::Brush(BrushBatchKind::Opacity),
                                        BlendMode::PremultipliedAlpha,
                                        textures,
                                    );

                                    let prim_user_data = [
                                        uv_rect_address.as_int(),
                                        amount,
                                        0,
                                        0,
                                    ];

                                    (key, prim_user_data, 0)
                                }
                                _ => {
                                    // Must be kept in sync with brush_blend.glsl
                                    let filter_mode = filter.as_int();

                                    let user_data = match filter {
                                        Filter::Identity => 0x10000i32, // matches `Contrast(1)`
                                        Filter::Contrast(amount) |
                                        Filter::Grayscale(amount) |
                                        Filter::Invert(amount) |
                                        Filter::Saturate(amount) |
                                        Filter::Sepia(amount) |
                                        Filter::Brightness(amount) => {
                                            (amount * 65536.0) as i32
                                        }
                                        Filter::SrgbToLinear | Filter::LinearToSrgb => 0,
                                        Filter::HueRotate(angle) => {
                                            (0.01745329251 * angle * 65536.0) as i32
                                        }
                                        Filter::ColorMatrix(_) => {
                                            picture_scratch.extra_gpu_data[0].as_int()
                                        }
                                        Filter::Flood(_) => {
                                            picture_scratch.extra_gpu_data[0].as_int()
                                        }

                                        // These filters are handled via different paths.
                                        Filter::ComponentTransfer |
                                        Filter::Blur { .. } |
                                        Filter::DropShadows(..) |
                                        Filter::Opacity(..) |
                                        Filter::SVGGraphNode(..) => unreachable!(),
                                    };

                                    // Other filters that may introduce opacity are handled via different
                                    // paths.
                                    if let Filter::ColorMatrix(..) = filter {
                                        is_opaque = false;
                                    }

                                    let blend_mode = if is_opaque {
                                        BlendMode::None
                                    } else {
                                        BlendMode::PremultipliedAlpha
                                    };

                                    let key = BatchKey::new(
                                        BatchKind::Brush(BrushBatchKind::Blend),
                                        blend_mode,
                                        textures,
                                    );

                                    let prim_user_data = [
                                        uv_rect_address.as_int(),
                                        filter_mode,
                                        user_data,
                                        0,
                                    ];

                                    (key, prim_user_data, 0)
                                }
                            }
                        }
                        PictureCompositeMode::ComponentTransferFilter(handle) => {
                            // This is basically the same as the general filter case above
                            // except we store a little more data in the filter mode and
                            // a gpu cache handle in the user data.
                            let filter_data = &ctx.data_stores.filter_data[handle];
                            let filter_mode : i32 = Filter::ComponentTransfer.as_int() |
                                ((filter_data.data.r_func.to_int() << 28 |
                                  filter_data.data.g_func.to_int() << 24 |
                                  filter_data.data.b_func.to_int() << 20 |
                                  filter_data.data.a_func.to_int() << 16) as i32);

                            let user_data = picture_scratch.extra_gpu_data[0].as_int();

                            let key = BatchKey::new(
                                BatchKind::Brush(BrushBatchKind::Blend),
                                BlendMode::PremultipliedAlpha,
                                textures,
                            );

                            let prim_user_data = [
                                uv_rect_address.as_int(),
                                filter_mode,
                                user_data,
                                0,
                            ];

                            (key, prim_user_data, 0)
                        }
                        PictureCompositeMode::MixBlend(mode) if BlendMode::from_mix_blend_mode(
                            mode,
                            ctx.use_advanced_blending,
                            !ctx.break_advanced_blend_batches,
                            ctx.use_dual_source_blending,
                        ).is_some() => {
                            let key = BatchKey::new(
                                BatchKind::Brush(
                                    BrushBatchKind::Image(ImageBufferKind::Texture2D),
                                ),
                                BlendMode::from_mix_blend_mode(
                                    mode,
                                    ctx.use_advanced_blending,
                                    !ctx.break_advanced_blend_batches,
                                    ctx.use_dual_source_blending,
                                ).unwrap(),
                                textures,
                            );

                            let prim_user_data = ImageBrushUserData {
                                color_mode: match key.blend_mode {
                                    BlendMode::MultiplyDualSource => ShaderColorMode::MultiplyDualSource,
                                    _ => ShaderColorMode::Image,
                                },
                                alpha_type: AlphaType::PremultipliedAlpha,
                                raster_space: RasterizationSpace::Screen,
                                opacity: 1.0,
                            }.encode();

                            (key, prim_user_data, uv_rect_address.as_int())
                        }
                        PictureCompositeMode::MixBlend(mode) => {
                            let backdrop_id = picture_scratch.secondary_render_task_id.expect("no backdrop!?");

                            let color0 = render_tasks[backdrop_id].get_target_texture();
                            let color1 = render_tasks[pic_task_id].get_target_texture();

                            // Create a separate brush instance for each batcher. For most cases,
                            // there is only one batcher. However, in the case of drawing onto
                            // a picture cache, there is one batcher per tile. Although not
                            // currently used, the implementation of mix-blend-mode now supports
                            // doing partial readbacks per-tile. In future, this will be enabled
                            // and allow mix-blends to operate on picture cache surfaces without
                            // a separate isolated intermediate surface.

                            let batch_key = BatchKey::new(
                                BatchKind::Brush(
                                    BrushBatchKind::MixBlend {
                                        task_id: self.batcher.render_task_id,
                                        backdrop_id,
                                    },
                                ),
                                BlendMode::PremultipliedAlpha,
                                BatchTextures {
                                    input: TextureSet {
                                        colors: [
                                            TextureSource::TextureCache(
                                                color0,
                                                Swizzle::default(),
                                            ),
                                            TextureSource::TextureCache(
                                                color1,
                                                Swizzle::default(),
                                            ),
                                            TextureSource::Invalid,
                                        ],
                                    },
                                    clip_mask: clip_mask_texture_id,
                                },
                            );
                            let src_uv_address = render_tasks[pic_task_id].get_texture_address();
                            let readback_uv_address = render_tasks[backdrop_id].get_texture_address();
                            let prim_header = PrimitiveHeader {
                                user_data: [
                                    mode as u32 as i32,
                                    readback_uv_address.as_int(),
                                    src_uv_address.as_int(),
                                    0,
                                ],
                                ..picture_prim_header
                            };
                            let prim_header_index = prim_headers.push(&prim_header);

                            let instance = BrushInstance {
                                segment_index: INVALID_SEGMENT_INDEX,
                                edge_flags: EdgeMask::all(),
                                clip_task_address,
                                brush_flags,
                                prim_header_index,
                                resource_address: 0,
                            };

                            self.batcher.push_single_instance(
                                batch_key,
                                batch_features,
                                bounding_rect,
                                z_id,
                                PrimitiveInstanceData::from(instance),
                            );

                            return;
                        }
                        PictureCompositeMode::Blit(_) => {
                            match picture.context_3d {
                                Picture3DContext::In { root_data: Some(_), .. } => {
                                    unreachable!("bug: should not have a raster_config");
                                }
                                Picture3DContext::In { root_data: None, .. } => {
                                    // TODO(gw): Store this inside the split picture so that we
                                    //           don't need to pass in extra_prim_gpu_address for
                                    //           every prim instance.
                                    // TODO(gw): Ideally we'd skip adding 3d child prims to batches
                                    //           without gpu cache address but it's currently
                                    //           used by the prepare pass. Refactor this!
                                    let extra_prim_gpu_address = match extra_prim_gpu_address {
                                        Some(prim_address) => prim_address,
                                        None => return,
                                    };

                                    // Need a new z-id for each child preserve-3d context added
                                    // by this inner loop.
                                    let z_id = z_generator.next();

                                    let prim_header = PrimitiveHeader {
                                        z: z_id,
                                        transform_id: transforms.gpu.get_id(
                                            prim_spatial_node_index,
                                            root_spatial_node_index,
                                            ctx.spatial_tree,
                                        ),
                                        user_data: [
                                            uv_rect_address.as_int(),
                                            BrushFlags::PERSPECTIVE_INTERPOLATION.bits() as i32,
                                            0,
                                            clip_task_address.0 as i32,
                                        ],
                                        ..picture_prim_header
                                    };
                                    let prim_header_index = prim_headers.push(&prim_header);

                                    let key = BatchKey::new(
                                        BatchKind::SplitComposite,
                                        BlendMode::PremultipliedAlpha,
                                        textures,
                                    );

                                    self.add_split_composite_instance_to_batches(
                                        key,
                                        BatchFeatures::CLIP_MASK,
                                        &prim_info.clip_chain.pic_coverage_rect,
                                        z_id,
                                        prim_header_index,
                                        extra_prim_gpu_address,
                                    );

                                    return;
                                }
                                Picture3DContext::Out { .. } => {
                                    let textures = TextureSet {
                                        colors: [
                                            texture,
                                            TextureSource::Invalid,
                                            TextureSource::Invalid,
                                        ],
                                    };
                                    let batch_params = BrushBatchParameters::shared(
                                        BrushBatchKind::Image(ImageBufferKind::Texture2D),
                                        textures,
                                        ImageBrushUserData {
                                            color_mode: ShaderColorMode::Image,
                                            alpha_type: AlphaType::PremultipliedAlpha,
                                            raster_space: RasterizationSpace::Screen,
                                            opacity: 1.0,
                                        }.encode(),
                                        uv_rect_address.as_int(),
                                    );

                                    let prim_header = PrimitiveHeader {
                                        specific_prim_address: prim_cache_address.as_int(),
                                        user_data: batch_params.prim_user_data,
                                        ..picture_prim_header
                                    };
                                    let prim_header_index = prim_headers.push(&prim_header);

                                    let (opacity, blend_mode) = if is_opaque {
                                        (PrimitiveOpacity::opaque(), BlendMode::None)
                                    } else {
                                        (PrimitiveOpacity::translucent(), BlendMode::PremultipliedAlpha)
                                    };

                                    self.add_segmented_prim_to_batch(
                                        None,
                                        opacity,
                                        &batch_params,
                                        blend_mode,
                                        batch_features,
                                        brush_flags,
                                        EdgeMask::all(),
                                        prim_header_index,
                                        bounding_rect,
                                        transform_metadata,
                                        z_id,
                                        prim_info.clip_task_index,
                                        ctx,
                                        render_tasks,
                                    );

                                    return;
                                }
                            }
                        }
                        PictureCompositeMode::SVGFEGraph(..) => {
                            let kind = BatchKind::Brush(
                                BrushBatchKind::Image(ImageBufferKind::Texture2D)
                            );
                            let key = BatchKey::new(
                                kind,
                                blend_mode,
                                textures,
                            );

                            let prim_user_data = ImageBrushUserData {
                                color_mode: ShaderColorMode::Image,
                                alpha_type: AlphaType::PremultipliedAlpha,
                                raster_space: RasterizationSpace::Screen,
                                opacity: 1.0,
                            }.encode();

                            (key, prim_user_data, uv_rect_address.as_int())
                        }
                    };

                    let prim_header = PrimitiveHeader {
                        z: z_id,
                        user_data: prim_user_data,
                        ..picture_prim_header
                    };
                    let prim_header_index = prim_headers.push(&prim_header);

                    self.add_brush_instance_to_batches(
                        key,
                        batch_features,
                        bounding_rect,
                        z_id,
                        INVALID_SEGMENT_INDEX,
                        EdgeMask::all(),
                        clip_task_address,
                        brush_flags,
                        prim_header_index,
                        resource_address,
                    );
                }
                None => {
                    unreachable!();
                }
            }

            return;
        }

        let base_prim_header = PrimitiveHeader {
            local_rect: prim_rect,
            local_clip_rect: prim_info.clip_chain.local_clip_rect,
            transform_id,
            z: z_id,
            render_task_address: self.batcher.render_task_address,
            specific_prim_address: GpuBufferAddress::INVALID.as_int(), // Will be overridden by most uses
            user_data: [0; 4], // Will be overridden by most uses
        };

        let common_data = ctx.data_stores.as_common_data(prim_instance);

        // Per-instance opacity. Previously cached on the (now immutable)
        // prim template; sourced per kind from its per-frame scratch
        // (Rectangle/Image) or derived directly (constant for YuvImage /
        // NormalBorder, template flag for ImageBorder).
        let opacity = match prim_instance.kind {
            PrimitiveKind::Rectangle { .. } => {
                ctx.scratch.frame.rectangle[prim_info.kind_scratch.unwrap_rectangle()].opacity
            }
            PrimitiveKind::Image { .. } => {
                ctx.scratch.frame.images[prim_info.kind_scratch.unwrap_image()].opacity
            }
            PrimitiveKind::YuvImage { .. } => PrimitiveOpacity::opaque(),
            PrimitiveKind::NormalBorder { .. } => PrimitiveOpacity::translucent(),
            PrimitiveKind::ImageBorder { .. } => {
                PrimitiveOpacity { is_opaque: ctx.scratch.frame.image_border[prim_info.kind_scratch.unwrap_image_border()].is_opaque }
            }
            _ => PrimitiveOpacity::translucent(),
        };

        let needs_blending = !opacity.is_opaque ||
            prim_info.clip_task_index != ClipTaskIndex::INVALID ||
            !transform_metadata.is_2d_axis_aligned ||
            is_anti_aliased;

        let blend_mode = if needs_blending {
            BlendMode::PremultipliedAlpha
        } else {
            BlendMode::None
        };

        match prim_instance.kind {
            // Handled above.
            PrimitiveKind::Picture { .. } => {}
            PrimitiveKind::RadialGradient { .. } => { }
            PrimitiveKind::ConicGradient { .. } => { }
            PrimitiveKind::ImageBorder { .. } => {}
            PrimitiveKind::LineDecoration { .. } => {}
            PrimitiveKind::YuvImage { .. } => {}
            PrimitiveKind::BoxShadow { .. } => {
                unreachable!("BUG: Should not hit box-shadow here as they are handled by quad infra");
            }
            PrimitiveKind::NormalBorder { .. } => {}
            PrimitiveKind::TextRun { data_handle, .. } => {
                let text_run_scratch_handle = prim_info.kind_scratch.unwrap_text_run();
                let run_scratch = &ctx.scratch.frame.text_runs[text_run_scratch_handle];
                let subpx_dir = run_scratch.used_font.get_subpx_dir();
                let prim_data = &ctx.data_stores.text_run[data_handle];

                let glyph_keys = &ctx.scratch.frame.glyph_keys[run_scratch.glyph_keys_range];

                // `local_rect.p0` is the run anchor (the normalized prim rect
                // origin). In device mode the shader transforms it to device
                // space and adds the per-glyph device offsets stored at
                // `gpu_address`. `user_data` carries the raster scale (for
                // local-raster mode's raster -> local mapping) and the mode flag
                // (0 = device, 1 = local raster).
                let prim_header = PrimitiveHeader {
                    local_rect: run_scratch.local_rect,
                    specific_prim_address: run_scratch.gpu_address.as_int(),
                    user_data: [
                        (run_scratch.raster_scale * 65535.0).round() as i32,
                        run_scratch.local_raster as i32,
                        0,
                        0,
                    ],
                    ..base_prim_header
                };
                let prim_header_index = prim_headers.push(&prim_header);
                let base_instance = GlyphInstance::new(
                    prim_header_index,
                );
                let batcher = &mut self.batcher;

                let (clip_task_address, clip_mask_texture_id) = ctx.get_prim_clip_task_and_texture(
                    prim_info.clip_task_index,
                    render_tasks,
                ).unwrap();

                // The run_scratch.used_font.clone() is here instead of inline in the `fetch_glyph`
                // function call to work around a miscompilation.
                // https://github.com/rust-lang/rust/issues/80111
                let font = run_scratch.used_font.clone();
                ctx.resource_cache.fetch_glyphs(
                    font,
                    &glyph_keys,
                    &gpu_buffer_builder.f32,
                    &mut self.glyph_fetch_buffer,
                    |texture_id, glyph_format, glyphs| {
                        debug_assert_ne!(texture_id, TextureSource::Invalid);

                        let subpx_dir = subpx_dir.limit_by(glyph_format);

                        let textures = BatchTextures::prim_textured(
                            texture_id,
                            clip_mask_texture_id,
                        );

                        let kind = BatchKind::TextRun(glyph_format);

                        let (blend_mode, color_mode) = match glyph_format {
                            GlyphFormat::Subpixel |
                            GlyphFormat::TransformedSubpixel => {
                                debug_assert!(ctx.use_dual_source_blending);
                                (
                                    BlendMode::SubpixelDualSource,
                                    ShaderColorMode::SubpixelDualSource,
                                )
                            }
                            GlyphFormat::Alpha |
                            GlyphFormat::TransformedAlpha |
                            GlyphFormat::Bitmap => {
                                (
                                    BlendMode::PremultipliedAlpha,
                                    ShaderColorMode::Alpha,
                                )
                            }
                            GlyphFormat::ColorBitmap => {
                                (
                                    BlendMode::PremultipliedAlpha,
                                    if prim_data.shadow {
                                        // Ignore color and only sample alpha when shadowing.
                                        ShaderColorMode::BitmapShadow
                                    } else {
                                        ShaderColorMode::ColorBitmap
                                    },
                                )
                            }
                        };

                        // Calculate a tighter bounding rect of just the glyphs passed to this
                        // callback from request_glyphs(), rather than using the bounds of the
                        // entire text run. This improves batching when glyphs are fragmented
                        // over multiple textures in the texture cache.
                        // This mirrors the glyph positioning in the ps_text_run shader. The
                        // TRANSFORM_GLYPHS branch covers device mode for 2D rotated/skewed
                        // glyphs; the other branch covers device-mode axis-aligned and
                        // local-raster mode (distinguished by `run_scratch.raster_scale`).
                        // `text_offset` is zero because glyph positions are stored absolutely
                        // (relative to the prim origin via `local_rect.min`), not relative to
                        // a separate snapped reference-frame offset; the TRANSFORM_GLYPHS
                        // branch's `raster_text_offset` then reduces to the reference-frame
                        // device snap that `request_resources` applies.
                        let tight_bounding_rect = {
                            let snap_bias = match subpx_dir {
                                SubpixelDirection::None => DeviceVector2D::new(0.5, 0.5),
                                SubpixelDirection::Horizontal => DeviceVector2D::new(0.125, 0.5),
                                SubpixelDirection::Vertical => DeviceVector2D::new(0.5, 0.125),
                            };
                            let text_offset = LayoutVector2D::zero();

                            let pic_bounding_rect = if run_scratch.used_font.flags.contains(FontInstanceFlags::TRANSFORM_GLYPHS) {
                                let mut device_bounding_rect = DeviceRect::default();

                                let glyph_transform = ctx.spatial_tree.get_relative_transform(
                                    prim_spatial_node_index,
                                    root_spatial_node_index,
                                ).into_transform()
                                    .with_destination::<WorldPixel>()
                                    .then(&euclid::Transform3D::from_scale(ctx.global_device_pixel_scale));

                                let glyph_translation = DeviceVector2D::new(glyph_transform.m41, glyph_transform.m42);

                                let mut use_tight_bounding_rect = true;
                                for glyph in glyphs {
                                    let glyph_offset = prim_data.glyphs[glyph.index_in_text_run as usize].point + prim_header.local_rect.min.to_vector();

                                    let transformed_offset = match glyph_transform.transform_point2d(glyph_offset) {
                                        Some(transformed_offset) => transformed_offset,
                                        None => {
                                            use_tight_bounding_rect = false;
                                            break;
                                        }
                                    };
                                    let raster_glyph_offset = (transformed_offset + snap_bias).floor();
                                    let raster_text_offset = (
                                        glyph_transform.transform_vector2d(text_offset) +
                                        glyph_translation +
                                        DeviceVector2D::new(0.5, 0.5)
                                    ).floor() - glyph_translation;

                                    let device_glyph_rect = DeviceRect::from_origin_and_size(
                                        glyph.offset + raster_glyph_offset.to_vector() + raster_text_offset,
                                        glyph.size.to_f32(),
                                    );

                                    device_bounding_rect = device_bounding_rect.union(&device_glyph_rect);
                                }

                                if use_tight_bounding_rect {
                                    let map_device_to_surface: SpaceMapper<PicturePixel, DevicePixel> = SpaceMapper::new_with_target(
                                        root_spatial_node_index,
                                        surface_spatial_node_index,
                                        device_bounding_rect,
                                        ctx.spatial_tree,
                                    );

                                    match map_device_to_surface.unmap(&device_bounding_rect) {
                                        Some(r) => r.intersection(bounding_rect),
                                        None => Some(*bounding_rect),
                                    }
                                } else {
                                    Some(*bounding_rect)
                                }
                            } else {
                                let mut local_bounding_rect = LayoutRect::default();

                                let glyph_raster_scale = run_scratch.raster_scale * ctx.global_device_pixel_scale.get();

                                for glyph in glyphs {
                                    let glyph_offset = prim_data.glyphs[glyph.index_in_text_run as usize].point + prim_header.local_rect.min.to_vector();
                                    let glyph_scale = LayoutToDeviceScale::new(glyph_raster_scale / glyph.scale);
                                    let raster_glyph_offset = (glyph_offset * LayoutToDeviceScale::new(glyph_raster_scale) + snap_bias).floor() / glyph.scale;
                                    let local_glyph_rect = LayoutRect::from_origin_and_size(
                                        (glyph.offset + raster_glyph_offset.to_vector()) / glyph_scale + text_offset,
                                        glyph.size.to_f32() / glyph_scale,
                                    );

                                    local_bounding_rect = local_bounding_rect.union(&local_glyph_rect);
                                }

                                let map_prim_to_surface: SpaceMapper<LayoutPixel, PicturePixel> = SpaceMapper::new_with_target(
                                    surface_spatial_node_index,
                                    prim_spatial_node_index,
                                    *bounding_rect,
                                    ctx.spatial_tree,
                                );
                                map_prim_to_surface.map(&local_bounding_rect)
                            };

                            let intersected = match pic_bounding_rect {
                                // The text run may have been clipped, for example if part of it is offscreen.
                                // So intersect our result with the original bounding rect.
                                Some(rect) => rect.intersection(bounding_rect).unwrap_or_else(PictureRect::zero),
                                // If space mapping went off the rails, fall back to the old behavior.
                                //TODO: consider skipping the glyph run completely in this case.
                                None => *bounding_rect,
                            };

                            intersected
                        };

                        let key = BatchKey::new(kind, blend_mode, textures);

                        let batch = batcher.alpha_batch_list.set_params_and_get_batch(
                            key,
                            batch_features,
                            &tight_bounding_rect,
                            z_id,
                        );

                        batch.reserve(glyphs.len());
                        for glyph in glyphs {
                            batch.push(base_instance.build(
                                clip_task_address,
                                subpx_dir,
                                glyph.index_in_text_run,
                                glyph.uv_rect_address,
                                color_mode,
                                glyph.subpx_offset_x,
                                glyph.subpx_offset_y,
                                glyph.is_packed_glyph,
                            ));
                        }
                    },
                );
            }
            PrimitiveKind::Rectangle { .. } => {
                let (prim_cache_address, segments) = if prim_info.segment_instance_index == SegmentInstanceIndex::UNUSED {
                    let rect_scratch = prim_info.kind_scratch.unwrap_rectangle();
                    (ctx.scratch.frame.rectangle[rect_scratch].gpu_address, None)
                } else {
                    let segment_instance = &ctx.scratch.frame.segment_instances[prim_info.segment_instance_index];
                    let segments = Some(&ctx.scratch.frame.segments[segment_instance.segments_range]);
                    (segment_instance.gpu_data, segments)
                };

                let batch_params = BrushBatchParameters::shared(
                    BrushBatchKind::Solid,
                    TextureSet::UNTEXTURED,
                    [get_shader_opacity(1.0), 0, 0, 0],
                    0,
                );

                let prim_header = PrimitiveHeader {
                    specific_prim_address: prim_cache_address.as_int(),
                    user_data: batch_params.prim_user_data,
                    ..base_prim_header
                };
                let prim_header_index = prim_headers.push(&prim_header);

                self.add_segmented_prim_to_batch(
                    segments,
                    opacity,
                    &batch_params,
                    blend_mode,
                    batch_features,
                    brush_flags,
                    common_data.transformed_aa_edges,
                    prim_header_index,
                    bounding_rect,
                    transform_metadata,
                    z_id,
                    prim_info.clip_task_index,
                    ctx,
                    render_tasks,
                );
            }
            PrimitiveKind::Image { .. } => {
                unreachable!("BUG: images should always use quad path");
            }
            PrimitiveKind::LinearGradient { .. } => {
                unreachable!("BUG: linear gradients should always use quad path");
            }
            PrimitiveKind::BackdropCapture { .. } => {}
            PrimitiveKind::BackdropRender { .. } => {}
        }
    }

    /// Add a single segment instance to a batch.
    ///
    /// `edge_aa_mask` Specifies the edges that are *allowed* to have anti-aliasing, if and only
    /// if the segments enable it.
    /// In other words passing EdgeAaSegmentFlags::all() does not necessarily mean all edges will
    /// be anti-aliased, only that they could be.
    fn add_segment_to_batch(
        &mut self,
        segment: &BrushSegment,
        segment_data: &SegmentInstanceData,
        segment_index: i32,
        batch_kind: BrushBatchKind,
        prim_header_index: PrimitiveHeaderIndex,
        alpha_blend_mode: BlendMode,
        features: BatchFeatures,
        brush_flags: BrushFlags,
        edge_aa_mask: EdgeMask,
        bounding_rect: &PictureRect,
        transform_metadata: TransformMetadata,
        z_id: ZBufferId,
        prim_opacity: PrimitiveOpacity,
        clip_task_index: ClipTaskIndex,
        ctx: &RenderTargetContext,
        render_tasks: &RenderTaskGraph,
    ) {
        debug_assert!(clip_task_index != ClipTaskIndex::INVALID);

        // Get GPU address of clip task for this segment, or None if
        // the entire segment is clipped out.
        if let Some((clip_task_address, clip_mask)) = ctx.get_clip_task_and_texture(
            clip_task_index,
            segment_index,
            render_tasks,
        ) {
            // If a got a valid (or OPAQUE) clip task address, add the segment.
            let is_inner = segment.edge_flags.is_empty();
            let needs_blending = !prim_opacity.is_opaque ||
                                 clip_task_address != OPAQUE_TASK_ADDRESS ||
                                 (!is_inner && !transform_metadata.is_2d_axis_aligned) ||
                                 brush_flags.contains(BrushFlags::FORCE_AA);

            let textures = BatchTextures {
                input: segment_data.textures,
                clip_mask,
            };

            let batch_key = BatchKey {
                blend_mode: if needs_blending { alpha_blend_mode } else { BlendMode::None },
                kind: BatchKind::Brush(batch_kind),
                textures,
            };

            self.add_brush_instance_to_batches(
                batch_key,
                features,
                bounding_rect,
                z_id,
                segment_index,
                segment.edge_flags & edge_aa_mask,
                clip_task_address,
                brush_flags | BrushFlags::PERSPECTIVE_INTERPOLATION | segment.brush_flags,
                prim_header_index,
                segment_data.specific_resource_address,
            );
        }
    }

    /// Add any segment(s) from a brush to batches.
    ///
    /// `edge_aa_mask` Specifies the edges that are *allowed* to have anti-aliasing, if and only
    /// if the segments enable it.
    /// In other words passing EdgeAaSegmentFlags::all() does not necessarily mean all edges will
    /// be anti-aliased, only that they could be.
    fn add_segmented_prim_to_batch(
        &mut self,
        brush_segments: Option<&[BrushSegment]>,
        prim_opacity: PrimitiveOpacity,
        params: &BrushBatchParameters,
        blend_mode: BlendMode,
        features: BatchFeatures,
        brush_flags: BrushFlags,
        edge_aa_mask: EdgeMask,
        prim_header_index: PrimitiveHeaderIndex,
        bounding_rect: &PictureRect,
        transform_metadata: TransformMetadata,
        z_id: ZBufferId,
        clip_task_index: ClipTaskIndex,
        ctx: &RenderTargetContext,
        render_tasks: &RenderTaskGraph,
    ) {
        match (brush_segments, &params.segment_data) {
            (Some(ref brush_segments), SegmentDataKind::Shared(ref segment_data)) => {
                // A list of segments, but the per-segment data is common
                // between all segments.
                for (segment_index, segment) in brush_segments
                    .iter()
                    .enumerate()
                {
                    self.add_segment_to_batch(
                        segment,
                        segment_data,
                        segment_index as i32,
                        params.batch_kind,
                        prim_header_index,
                        blend_mode,
                        features,
                        brush_flags,
                        edge_aa_mask,
                        bounding_rect,
                        transform_metadata,
                        z_id,
                        prim_opacity,
                        clip_task_index,
                        ctx,
                        render_tasks,
                    );
                }
            }
            (None, SegmentDataKind::Shared(ref segment_data)) => {
                // No segments, and thus no per-segment instance data.
                // Note: the blend mode already takes opacity into account

                let (clip_task_address, clip_mask) = ctx.get_prim_clip_task_and_texture(
                    clip_task_index,
                    render_tasks,
                ).unwrap();

                let textures = BatchTextures {
                    input: segment_data.textures,
                    clip_mask,
                };

                let batch_key = BatchKey {
                    blend_mode,
                    kind: BatchKind::Brush(params.batch_kind),
                    textures,
                };

                self.add_brush_instance_to_batches(
                    batch_key,
                    features,
                    bounding_rect,
                    z_id,
                    INVALID_SEGMENT_INDEX,
                    edge_aa_mask,
                    clip_task_address,
                    brush_flags | BrushFlags::PERSPECTIVE_INTERPOLATION,
                    prim_header_index,
                    segment_data.specific_resource_address,
                );
            }
        }
    }
}

/// Either a single texture / user data for all segments,
/// or a list of one per segment.
enum SegmentDataKind {
    Shared(SegmentInstanceData),
}

/// The parameters that are specific to a kind of brush,
/// used by the common method to add a brush to batches.
struct BrushBatchParameters {
    batch_kind: BrushBatchKind,
    prim_user_data: [i32; 4],
    segment_data: SegmentDataKind,
}

impl BrushBatchParameters {
    /// This brush instance shares the per-segment data
    /// across all segments.
    fn shared(
        batch_kind: BrushBatchKind,
        textures: TextureSet,
        prim_user_data: [i32; 4],
        specific_resource_address: i32,
    ) -> Self {
        BrushBatchParameters {
            batch_kind,
            prim_user_data,
            segment_data: SegmentDataKind::Shared(
                SegmentInstanceData {
                    textures,
                    specific_resource_address,
                }
            ),
        }
    }
}

/// A list of clip instances to be drawn into a target.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct ClipMaskInstanceList {
    pub mask_instances_fast: FrameVec<MaskInstance>,
    pub mask_instances_slow: FrameVec<MaskInstance>,

    pub mask_instances_fast_with_scissor: FastHashMap<DeviceIntRect, FrameVec<MaskInstance>>,
    pub mask_instances_slow_with_scissor: FastHashMap<DeviceIntRect, FrameVec<MaskInstance>>,

    pub image_mask_instances: FastHashMap<TextureSource, FrameVec<PrimitiveInstanceData>>,
    pub image_mask_instances_with_scissor: FastHashMap<(DeviceIntRect, TextureSource), FrameVec<PrimitiveInstanceData>>,
}

impl ClipMaskInstanceList {
    pub fn new(memory: &FrameMemory) -> Self {
        ClipMaskInstanceList {
            mask_instances_fast: memory.new_vec(),
            mask_instances_slow: memory.new_vec(),
            mask_instances_fast_with_scissor: FastHashMap::default(),
            mask_instances_slow_with_scissor: FastHashMap::default(),
            image_mask_instances: FastHashMap::default(),
            image_mask_instances_with_scissor: FastHashMap::default(),
        }
    }

    pub fn is_empty(&self) -> bool {
        // Destructure self to make sure we don't forget to update this method if
        // a new member is added.
        let ClipMaskInstanceList {
            mask_instances_fast,
            mask_instances_slow,
            mask_instances_fast_with_scissor,
            mask_instances_slow_with_scissor,
            image_mask_instances,
            image_mask_instances_with_scissor,
        } = self;

        mask_instances_fast.is_empty()
            && mask_instances_slow.is_empty()
            && mask_instances_fast_with_scissor.is_empty()
            && mask_instances_slow_with_scissor.is_empty()
            && image_mask_instances.is_empty()
            && image_mask_instances_with_scissor.is_empty()
    }
}


impl<'a, 'rc> RenderTargetContext<'a, 'rc> {
    /// Retrieve the GPU task address for a given clip task instance.
    /// Returns None if the segment was completely clipped out.
    /// Returns Some(OPAQUE_TASK_ADDRESS) if no clip mask is needed.
    /// Returns Some(task_address) if there was a valid clip mask.
    fn get_clip_task_and_texture(
        &self,
        clip_task_index: ClipTaskIndex,
        offset: i32,
        render_tasks: &RenderTaskGraph,
    ) -> Option<(RenderTaskAddress, TextureSource)> {
        match self.scratch.frame.clip_mask_instances[clip_task_index.0 as usize + offset as usize] {
            ClipMaskKind::Mask(task_id) => {
                Some((
                    task_id.into(),
                    TextureSource::TextureCache(
                        render_tasks[task_id].get_target_texture(),
                        Swizzle::default(),
                    )
                ))
            }
            ClipMaskKind::None => {
                Some((OPAQUE_TASK_ADDRESS, TextureSource::Invalid))
            }
            ClipMaskKind::Clipped => {
                None
            }
        }
    }

    /// Helper function to get the clip task address for a
    /// non-segmented primitive.
    fn get_prim_clip_task_and_texture(
        &self,
        clip_task_index: ClipTaskIndex,
        render_tasks: &RenderTaskGraph,
    ) -> Option<(RenderTaskAddress, TextureSource)> {
        self.get_clip_task_and_texture(
            clip_task_index,
            0,
            render_tasks,
        )
    }
}
