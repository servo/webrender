/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{AlphaType, ClipMode, ExternalImageType, ImageRendering};
use api::{YuvColorSpace, YuvFormat, ColorDepth, PremultipliedColorF, RasterSpace};
use api::units::*;
use crate::clip::{ClipDataStore, ClipNodeFlags, ClipNodeRange, ClipItem, ClipStore, ClipNodeInstance};
use crate::clip_scroll_tree::{ClipScrollTree, ROOT_SPATIAL_NODE_INDEX, SpatialNodeIndex, CoordinateSystemId};
use crate::glyph_rasterizer::GlyphFormat;
use crate::gpu_cache::{GpuBlockData, GpuCache, GpuCacheHandle, GpuCacheAddress};
use crate::gpu_types::{BrushFlags, BrushInstance, PrimitiveHeaders, ZBufferId, ZBufferIdGenerator};
use crate::gpu_types::{ClipMaskInstance, SplitCompositeInstance, SnapOffsets};
use crate::gpu_types::{PrimitiveInstanceData, RasterizationSpace, GlyphInstance};
use crate::gpu_types::{PrimitiveHeader, PrimitiveHeaderIndex, TransformPaletteId, TransformPalette};
use crate::internal_types::{FastHashMap, SavedTargetIndex, TextureSource, Filter};
use crate::picture::{Picture3DContext, PictureCompositeMode, PicturePrimitive};
use crate::prim_store::{DeferredResolve, EdgeAaSegmentMask, PrimitiveInstanceKind, PrimitiveVisibilityIndex};
use crate::prim_store::{VisibleGradientTile, PrimitiveInstance, PrimitiveOpacity, SegmentInstanceIndex};
use crate::prim_store::{BrushSegment, ClipMaskKind, ClipTaskIndex, VECS_PER_SEGMENT};
use crate::prim_store::{recompute_snap_offsets};
use crate::prim_store::image::ImageSource;
use crate::render_backend::DataStores;
use crate::render_task::{RenderTaskAddress, RenderTaskId, RenderTaskGraph, TileBlit};
use crate::renderer::{BlendMode, ImageBufferKind, ShaderColorMode};
use crate::renderer::{BLOCKS_PER_UV_RECT, MAX_VERTEX_TEXTURE_WIDTH};
use crate::resource_cache::{CacheItem, GlyphFetchResult, ImageRequest, ResourceCache, ImageProperties};
use smallvec::SmallVec;
use std::{f32, i32, usize};
use crate::tiling::{RenderTargetContext};
use crate::util::{project_rect, TransformedRectKind};

// Special sentinel value recognized by the shader. It is considered to be
// a dummy task that doesn't mask out anything.
const OPAQUE_TASK_ADDRESS: RenderTaskAddress = RenderTaskAddress(0x7fff);

/// Used to signal there are no segments provided with this primitive.
const INVALID_SEGMENT_INDEX: i32 = 0xffff;

/// Size in device pixels for tiles that clip masks are drawn in.
const CLIP_RECTANGLE_TILE_SIZE: i32 = 128;

/// The minimum size of a clip mask before trying to draw in tiles.
const CLIP_RECTANGLE_AREA_THRESHOLD: i32 = CLIP_RECTANGLE_TILE_SIZE * CLIP_RECTANGLE_TILE_SIZE * 4;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum BrushBatchKind {
    Solid,
    Image(ImageBufferKind),
    Blend,
    MixBlend {
        task_id: RenderTaskId,
        source_id: RenderTaskId,
        backdrop_id: RenderTaskId,
    },
    YuvImage(ImageBufferKind, YuvFormat, ColorDepth, YuvColorSpace),
    RadialGradient,
    LinearGradient,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum BatchKind {
    SplitComposite,
    TextRun(GlyphFormat),
    Brush(BrushBatchKind),
}

/// Optional textures that can be used as a source in the shaders.
/// Textures that are not used by the batch are equal to TextureId::invalid().
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct BatchTextures {
    pub colors: [TextureSource; 3],
}

impl BatchTextures {
    pub fn no_texture() -> Self {
        BatchTextures {
            colors: [TextureSource::Invalid; 3],
        }
    }

    pub fn render_target_cache() -> Self {
        BatchTextures {
            colors: [
                TextureSource::PrevPassColor,
                TextureSource::PrevPassAlpha,
                TextureSource::Invalid,
            ],
        }
    }

    pub fn color(texture: TextureSource) -> Self {
        BatchTextures {
            colors: [texture, texture, TextureSource::Invalid],
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
        self.kind == other.kind && self.blend_mode == other.blend_mode &&
            textures_compatible(self.textures.colors[0], other.textures.colors[0]) &&
            textures_compatible(self.textures.colors[1], other.textures.colors[1]) &&
            textures_compatible(self.textures.colors[2], other.textures.colors[2])
    }
}

#[inline]
fn textures_compatible(t1: TextureSource, t2: TextureSource) -> bool {
    t1 == TextureSource::Invalid || t2 == TextureSource::Invalid || t1 == t2
}

pub struct AlphaBatchList {
    pub batches: Vec<PrimitiveBatch>,
    pub item_rects: Vec<Vec<PictureRect>>,
    current_batch_index: usize,
    current_z_id: ZBufferId,
    break_advanced_blend_batches: bool,
    lookback_count: usize,
}

impl AlphaBatchList {
    fn new(break_advanced_blend_batches: bool, lookback_count: usize) -> Self {
        AlphaBatchList {
            batches: Vec::new(),
            item_rects: Vec::new(),
            current_z_id: ZBufferId::invalid(),
            current_batch_index: usize::MAX,
            break_advanced_blend_batches,
            lookback_count,
        }
    }

    pub fn set_params_and_get_batch(
        &mut self,
        key: BatchKey,
        // The bounding box of everything at this Z plane. We expect potentially
        // multiple primitive segments coming with the same `z_id`.
        z_bounding_rect: &PictureRect,
        z_id: ZBufferId,
    ) -> &mut Vec<PrimitiveInstanceData> {
        if z_id != self.current_z_id ||
           self.current_batch_index == usize::MAX ||
           !self.batches[self.current_batch_index].key.is_compatible_with(&key)
        {
            let mut selected_batch_index = None;

            match key.blend_mode {
                BlendMode::SubpixelWithBgColor => {
                    'outer_multipass: for (batch_index, batch) in self.batches.iter().enumerate().rev().take(self.lookback_count) {
                        // Some subpixel batches are drawn in two passes. Because of this, we need
                        // to check for overlaps with every batch (which is a bit different
                        // than the normal batching below).
                        for item_rect in &self.item_rects[batch_index] {
                            if item_rect.intersects(z_bounding_rect) {
                                break 'outer_multipass;
                            }
                        }

                        if batch.key.is_compatible_with(&key) {
                            selected_batch_index = Some(batch_index);
                            break;
                        }
                    }
                }
                BlendMode::Advanced(_) if self.break_advanced_blend_batches => {
                    // don't try to find a batch
                }
                _ => {
                    'outer_default: for (batch_index, batch) in self.batches.iter().enumerate().rev().take(self.lookback_count) {
                        // For normal batches, we only need to check for overlaps for batches
                        // other than the first batch we consider. If the first batch
                        // is compatible, then we know there isn't any potential overlap
                        // issues to worry about.
                        if batch.key.is_compatible_with(&key) {
                            selected_batch_index = Some(batch_index);
                            break;
                        }

                        // check for intersections
                        for item_rect in &self.item_rects[batch_index] {
                            if item_rect.intersects(z_bounding_rect) {
                                break 'outer_default;
                            }
                        }
                    }
                }
            }

            if selected_batch_index.is_none() {
                let new_batch = PrimitiveBatch::new(key);
                selected_batch_index = Some(self.batches.len());
                self.batches.push(new_batch);
                self.item_rects.push(Vec::new());
            }

            self.current_batch_index = selected_batch_index.unwrap();
            self.item_rects[self.current_batch_index].push(*z_bounding_rect);
            self.current_z_id = z_id;
        } else if cfg!(debug_assertions) {
            // If it's a different segment of the same (larger) primitive, we expect the bounding box
            // to be the same - coming from the primitive itself, not the segment.
            assert_eq!(self.item_rects[self.current_batch_index].last(), Some(z_bounding_rect));
        }

        &mut self.batches[self.current_batch_index].instances
    }
}

pub struct OpaqueBatchList {
    pub pixel_area_threshold_for_new_batch: f32,
    pub batches: Vec<PrimitiveBatch>,
    pub current_batch_index: usize,
    lookback_count: usize,
}

impl OpaqueBatchList {
    fn new(pixel_area_threshold_for_new_batch: f32, lookback_count: usize) -> Self {
        OpaqueBatchList {
            batches: Vec::new(),
            pixel_area_threshold_for_new_batch,
            current_batch_index: usize::MAX,
            lookback_count,
        }
    }

    pub fn set_params_and_get_batch(
        &mut self,
        key: BatchKey,
        // The bounding box of everything at the current Z, whatever it is. We expect potentially
        // multiple primitive segments produced by a primitive, which we allow to check
        // `current_batch_index` instead of iterating the batches.
        z_bounding_rect: &PictureRect,
    ) -> &mut Vec<PrimitiveInstanceData> {
        if self.current_batch_index == usize::MAX ||
           !self.batches[self.current_batch_index].key.is_compatible_with(&key) {
            let mut selected_batch_index = None;
            let item_area = z_bounding_rect.size.area();

            // If the area of this primitive is larger than the given threshold,
            // then it is large enough to warrant breaking a batch for. In this
            // case we just see if it can be added to the existing batch or
            // create a new one.
            if item_area > self.pixel_area_threshold_for_new_batch {
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
                let new_batch = PrimitiveBatch::new(key);
                selected_batch_index = Some(self.batches.len());
                self.batches.push(new_batch);
            }

            self.current_batch_index = selected_batch_index.unwrap();
        }

        &mut self.batches[self.current_batch_index].instances
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

pub struct BatchList {
    pub alpha_batch_list: AlphaBatchList,
    pub opaque_batch_list: OpaqueBatchList,
    /// A list of rectangle regions this batch should be drawn
    /// in. Each region will have scissor rect set before drawing.
    pub regions: Vec<DeviceIntRect>,
    pub tile_blits: Vec<TileBlit>,
}

impl BatchList {
    pub fn new(
        screen_size: DeviceIntSize,
        regions: Vec<DeviceIntRect>,
        tile_blits: Vec<TileBlit>,
        break_advanced_blend_batches: bool,
        lookback_count: usize,
    ) -> Self {
        // The threshold for creating a new batch is
        // one quarter the screen size.
        let batch_area_threshold = (screen_size.width * screen_size.height) as f32 / 4.0;

        BatchList {
            alpha_batch_list: AlphaBatchList::new(break_advanced_blend_batches, lookback_count),
            opaque_batch_list: OpaqueBatchList::new(batch_area_threshold, lookback_count),
            regions,
            tile_blits,
        }
    }

    pub fn push_single_instance(
        &mut self,
        key: BatchKey,
        bounding_rect: &PictureRect,
        z_id: ZBufferId,
        instance: PrimitiveInstanceData,
    ) {
        self.set_params_and_get_batch(key, bounding_rect, z_id)
            .push(instance);
    }

    pub fn set_params_and_get_batch(
        &mut self,
        key: BatchKey,
        bounding_rect: &PictureRect,
        z_id: ZBufferId,
    ) -> &mut Vec<PrimitiveInstanceData> {
        match key.blend_mode {
            BlendMode::None => {
                self.opaque_batch_list
                    .set_params_and_get_batch(key, bounding_rect)
            }
            BlendMode::Alpha |
            BlendMode::PremultipliedAlpha |
            BlendMode::PremultipliedDestOut |
            BlendMode::SubpixelConstantTextColor(..) |
            BlendMode::SubpixelWithBgColor |
            BlendMode::SubpixelDualSource |
            BlendMode::Advanced(_) => {
                self.alpha_batch_list
                    .set_params_and_get_batch(key, bounding_rect, z_id)
            }
        }
    }

    fn finalize(&mut self) {
        self.opaque_batch_list.finalize()
    }
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct PrimitiveBatch {
    pub key: BatchKey,
    pub instances: Vec<PrimitiveInstanceData>,
}

impl PrimitiveBatch {
    fn new(key: BatchKey) -> PrimitiveBatch {
        PrimitiveBatch {
            key,
            instances: Vec::new(),
        }
    }
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct AlphaBatchContainer {
    pub opaque_batches: Vec<PrimitiveBatch>,
    pub alpha_batches: Vec<PrimitiveBatch>,
    /// The overall scissor rect for this render task, if one
    /// is required.
    pub task_scissor_rect: Option<DeviceIntRect>,
    /// A list of rectangle regions this batch should be drawn
    /// in. Each region will have scissor rect set before drawing.
    pub regions: Vec<DeviceIntRect>,
    pub tile_blits: Vec<TileBlit>,
    /// The rectangle of the owning render target that this
    /// set of batches affects.
    pub task_rect: DeviceIntRect,
}

impl AlphaBatchContainer {
    pub fn new(
        task_scissor_rect: Option<DeviceIntRect>,
        regions: Vec<DeviceIntRect>,
    ) -> AlphaBatchContainer {
        AlphaBatchContainer {
            opaque_batches: Vec::new(),
            alpha_batches: Vec::new(),
            task_scissor_rect,
            regions,
            tile_blits: Vec::new(),
            task_rect: DeviceIntRect::zero(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.opaque_batches.is_empty() &&
        self.alpha_batches.is_empty()
    }

    fn merge(&mut self, batch_list: BatchList, task_rect: &DeviceIntRect) {
        self.task_rect = self.task_rect.union(task_rect);

        for other_batch in batch_list.opaque_batch_list.batches {
            let batch_index = self.opaque_batches.iter().position(|batch| {
                batch.key.is_compatible_with(&other_batch.key)
            });

            match batch_index {
                Some(batch_index) => {
                    self.opaque_batches[batch_index].instances.extend(other_batch.instances);
                }
                None => {
                    self.opaque_batches.push(other_batch);
                }
            }
        }

        let mut min_batch_index = 0;

        for other_batch in batch_list.alpha_batch_list.batches {
            let batch_index = self.alpha_batches.iter().skip(min_batch_index).position(|batch| {
                batch.key.is_compatible_with(&other_batch.key)
            });

            match batch_index {
                Some(batch_index) => {
                    let batch_index = batch_index + min_batch_index;
                    self.alpha_batches[batch_index].instances.extend(other_batch.instances);
                    min_batch_index = batch_index;
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
    textures: BatchTextures,
    user_data: i32,
}

/// Encapsulates the logic of building batches for items that are blended.
pub struct AlphaBatchBuilder {
    pub batch_lists: Vec<BatchList>,
    screen_size: DeviceIntSize,
    break_advanced_blend_batches: bool,
    lookback_count: usize,
    render_task_id: RenderTaskId,
    render_task_address: RenderTaskAddress,
}

impl AlphaBatchBuilder {
    pub fn new(
        screen_size: DeviceIntSize,
        break_advanced_blend_batches: bool,
        lookback_count: usize,
        render_task_id: RenderTaskId,
        render_task_address: RenderTaskAddress,
    ) -> Self {
        let batch_lists = vec![
            BatchList::new(
                screen_size,
                Vec::new(),
                Vec::new(),
                break_advanced_blend_batches,
                lookback_count,
            ),
        ];

        AlphaBatchBuilder {
            batch_lists,
            screen_size,
            break_advanced_blend_batches,
            lookback_count,
            render_task_id,
            render_task_address,
        }
    }

    fn push_new_batch_list(
        &mut self,
        regions: Vec<DeviceIntRect>,
        tile_blits: Vec<TileBlit>,
    ) {
        self.batch_lists.push(BatchList::new(
            self.screen_size,
            regions,
            tile_blits,
            self.break_advanced_blend_batches,
            self.lookback_count,
        ));
    }

    fn current_batch_list(&mut self) -> &mut BatchList {
        self.batch_lists.last_mut().unwrap()
    }

    fn can_merge(&self) -> bool {
        self.batch_lists.len() == 1
    }

    pub fn build(
        mut self,
        batch_containers: &mut Vec<AlphaBatchContainer>,
        merged_batches: &mut AlphaBatchContainer,
        task_rect: DeviceIntRect,
        task_scissor_rect: Option<DeviceIntRect>,
    ) {
        for batch_list in &mut self.batch_lists {
            batch_list.finalize();
        }

        if task_scissor_rect.is_none() && self.can_merge() {
            let batch_list = self.batch_lists.pop().unwrap();
            debug_assert!(batch_list.tile_blits.is_empty());
            merged_batches.merge(batch_list, &task_rect);
        } else {
            for batch_list in self.batch_lists {
                batch_containers.push(AlphaBatchContainer {
                    alpha_batches: batch_list.alpha_batch_list.batches,
                    opaque_batches: batch_list.opaque_batch_list.batches,
                    task_scissor_rect,
                    regions: batch_list.regions,
                    tile_blits: batch_list.tile_blits,
                    task_rect,
                });
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

    /// The batchers that primitives will be added to as the
    /// picture tree is traversed.
    batcher: AlphaBatchBuilder,
}

impl BatchBuilder {
    pub fn new(
        batcher: AlphaBatchBuilder,
    ) -> Self {
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
        bounding_rect: &PictureRect,
        z_id: ZBufferId,
        segment_index: i32,
        edge_flags: EdgeAaSegmentMask,
        clip_task_address: RenderTaskAddress,
        brush_flags: BrushFlags,
        prim_header_index: PrimitiveHeaderIndex,
        user_data: i32,
    ) {
        // TODO(gw): In future, this will be a loop adding the primitive
        //           to multiple batch list(s), depending on the primitive
        //           visibility mask.

        let render_task_address = self.batcher.render_task_address;

        let instance = BrushInstance {
            segment_index,
            edge_flags,
            clip_task_address,
            render_task_address,
            brush_flags,
            prim_header_index,
            user_data,
        };

        self.batcher.current_batch_list().push_single_instance(
            batch_key,
            bounding_rect,
            z_id,
            PrimitiveInstanceData::from(instance),
        );
    }

    fn add_split_composite_instance_to_batches(
        &mut self,
        batch_key: BatchKey,
        bounding_rect: &PictureRect,
        z_id: ZBufferId,
        prim_header_index: PrimitiveHeaderIndex,
        polygons_address: GpuCacheAddress,
    ) {
        // TODO(gw): In future, this will be a loop adding the primitive
        //           to multiple batch list(s), depending on the primitive
        //           visibility mask.

        let render_task_address = self.batcher.render_task_address;

        self.batcher.current_batch_list().push_single_instance(
            batch_key,
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

    /// Add a picture to a given batch builder.
    pub fn add_pic_to_batch(
        &mut self,
        pic: &PicturePrimitive,
        ctx: &RenderTargetContext,
        gpu_cache: &mut GpuCache,
        render_tasks: &RenderTaskGraph,
        deferred_resolves: &mut Vec<DeferredResolve>,
        prim_headers: &mut PrimitiveHeaders,
        transforms: &mut TransformPalette,
        root_spatial_node_index: SpatialNodeIndex,
        surface_spatial_node_index: SpatialNodeIndex,
        z_generator: &mut ZBufferIdGenerator,
    ) {
        // Add each run in this picture to the batch.
        for prim_instance in &pic.prim_list.prim_instances {
            self.add_prim_to_batch(
                prim_instance,
                ctx,
                gpu_cache,
                render_tasks,
                deferred_resolves,
                prim_headers,
                transforms,
                root_spatial_node_index,
                surface_spatial_node_index,
                z_generator,
            );
        }
    }

    // Adds a primitive to a batch.
    // It can recursively call itself in some situations, for
    // example if it encounters a picture where the items
    // in that picture are being drawn into the same target.
    fn add_prim_to_batch(
        &mut self,
        prim_instance: &PrimitiveInstance,
        ctx: &RenderTargetContext,
        gpu_cache: &mut GpuCache,
        render_tasks: &RenderTaskGraph,
        deferred_resolves: &mut Vec<DeferredResolve>,
        prim_headers: &mut PrimitiveHeaders,
        transforms: &mut TransformPalette,
        root_spatial_node_index: SpatialNodeIndex,
        surface_spatial_node_index: SpatialNodeIndex,
        z_generator: &mut ZBufferIdGenerator,
    ) {
        if prim_instance.visibility_info == PrimitiveVisibilityIndex::INVALID {
            return;
        }

        #[cfg(debug_assertions)] //TODO: why is this needed?
        debug_assert_eq!(prim_instance.prepared_frame_id, render_tasks.frame_id());

        let is_chased = prim_instance.is_chased();

        let transform_id = transforms
            .get_id(
                prim_instance.spatial_node_index,
                root_spatial_node_index,
                ctx.clip_scroll_tree,
            );

        // TODO(gw): Calculating this for every primitive is a bit
        //           wasteful. We should probably cache this in
        //           the scroll node...
        let transform_kind = transform_id.transform_kind();
        let prim_info = &ctx.scratch.prim_info[prim_instance.visibility_info.0 as usize];
        let bounding_rect = &prim_info.clip_chain.pic_clip_rect;

        let z_id = z_generator.next();

        let prim_common_data = &ctx.data_stores.as_common_data(&prim_instance);
        let prim_rect = LayoutRect::new(
            prim_instance.prim_origin,
            prim_common_data.prim_size,
        );

        let snap_offsets = prim_info.snap_offsets;

        if is_chased {
            println!("\tbatch {:?} with bound {:?}", prim_rect, bounding_rect);
        }

        if !bounding_rect.is_empty() {
            debug_assert_eq!(prim_info.clip_chain.pic_spatial_node_index, surface_spatial_node_index,
                "The primitive's bounding box is specified in a different coordinate system from the current batch!");
        }

        match prim_instance.kind {
            PrimitiveInstanceKind::Clear { data_handle } => {
                let prim_data = &ctx.data_stores.prim[data_handle];
                let prim_cache_address = gpu_cache.get_address(&prim_data.gpu_cache_handle);

                // TODO(gw): We can abstract some of the common code below into
                //           helper methods, as we port more primitives to make
                //           use of interning.

                let prim_header = PrimitiveHeader {
                    local_rect: prim_rect,
                    local_clip_rect: prim_info.combined_local_clip_rect,
                    snap_offsets,
                    specific_prim_address: prim_cache_address,
                    transform_id,
                };

                let prim_header_index = prim_headers.push(
                    &prim_header,
                    z_id,
                    [get_shader_opacity(1.0), 0, 0, 0],
                );

                let batch_key = BatchKey {
                    blend_mode: BlendMode::PremultipliedDestOut,
                    kind: BatchKind::Brush(BrushBatchKind::Solid),
                    textures: BatchTextures::no_texture(),
                };

                let clip_task_address = ctx.get_prim_clip_task_address(
                    prim_info.clip_task_index,
                    render_tasks,
                ).unwrap_or(OPAQUE_TASK_ADDRESS);

                self.add_brush_instance_to_batches(
                    batch_key,
                    bounding_rect,
                    z_id,
                    INVALID_SEGMENT_INDEX,
                    EdgeAaSegmentMask::all(),
                    clip_task_address,
                    BrushFlags::PERSPECTIVE_INTERPOLATION,
                    prim_header_index,
                    0,
                );
            }
            PrimitiveInstanceKind::NormalBorder { data_handle, ref cache_handles, .. } => {
                let prim_data = &ctx.data_stores.normal_border[data_handle];
                let common_data = &prim_data.common;
                let prim_cache_address = gpu_cache.get_address(&common_data.gpu_cache_handle);
                let cache_handles = &ctx.scratch.border_cache_handles[*cache_handles];
                let specified_blend_mode = BlendMode::PremultipliedAlpha;
                let mut segment_data: SmallVec<[SegmentInstanceData; 8]> = SmallVec::new();

                // Collect the segment instance data from each render
                // task for each valid edge / corner of the border.

                for handle in cache_handles {
                    let rt_cache_entry = ctx.resource_cache
                        .get_cached_render_task(handle);
                    let cache_item = ctx.resource_cache
                        .get_texture_cache_item(&rt_cache_entry.handle);
                    segment_data.push(
                        SegmentInstanceData {
                            textures: BatchTextures::color(cache_item.texture_id),
                            user_data: cache_item.uv_rect_handle.as_int(gpu_cache),
                        }
                    );
                }

                let non_segmented_blend_mode = if !common_data.opacity.is_opaque ||
                    prim_info.clip_task_index != ClipTaskIndex::INVALID ||
                    transform_kind == TransformedRectKind::Complex
                {
                    specified_blend_mode
                } else {
                    BlendMode::None
                };

                let prim_header = PrimitiveHeader {
                    local_rect: prim_rect,
                    local_clip_rect: prim_info.combined_local_clip_rect,
                    snap_offsets,
                    specific_prim_address: prim_cache_address,
                    transform_id,
                };

                let batch_params = BrushBatchParameters::instanced(
                    BrushBatchKind::Image(ImageBufferKind::Texture2DArray),
                    [
                        ShaderColorMode::Image as i32 | ((AlphaType::PremultipliedAlpha as i32) << 16),
                        RasterizationSpace::Local as i32,
                        get_shader_opacity(1.0),
                        0,
                    ],
                    segment_data,
                );

                let prim_header_index = prim_headers.push(
                    &prim_header,
                    z_id,
                    batch_params.prim_user_data,
                );

                let border_data = &prim_data.kind;
                self.add_segmented_prim_to_batch(
                    Some(border_data.brush_segments.as_slice()),
                    common_data.opacity,
                    &batch_params,
                    specified_blend_mode,
                    non_segmented_blend_mode,
                    prim_header_index,
                    bounding_rect,
                    transform_kind,
                    render_tasks,
                    z_id,
                    prim_info.clip_task_index,
                    ctx,
                );
            }
            PrimitiveInstanceKind::TextRun { data_handle, run_index, .. } => {
                let run = &ctx.prim_store.text_runs[run_index];
                let subpx_dir = run.used_font.get_subpx_dir();

                // The GPU cache data is stored in the template and reused across
                // frames and display lists.
                let prim_data = &ctx.data_stores.text_run[data_handle];
                let prim_cache_address = gpu_cache.get_address(&prim_data.gpu_cache_handle);

                let prim_header = PrimitiveHeader {
                    local_rect: prim_rect,
                    local_clip_rect: prim_info.combined_local_clip_rect,
                    snap_offsets,
                    specific_prim_address: prim_cache_address,
                    transform_id,
                };

                let clip_task_address = ctx.get_prim_clip_task_address(
                    prim_info.clip_task_index,
                    render_tasks,
                ).unwrap_or(OPAQUE_TASK_ADDRESS);

                let glyph_keys = &ctx.scratch.glyph_keys[run.glyph_keys_range];
                let rasterization_space = match run.raster_space {
                    RasterSpace::Screen => RasterizationSpace::Screen,
                    RasterSpace::Local(..) => RasterizationSpace::Local,
                };
                let raster_scale = run.raster_space.local_scale().unwrap_or(1.0).max(0.001);
                let prim_header_index = prim_headers.push(
                    &prim_header,
                    z_id,
                    [
                        (run.reference_frame_relative_offset.x * 256.0) as i32,
                        (run.reference_frame_relative_offset.y * 256.0) as i32,
                        (raster_scale * 65535.0).round() as i32,
                        clip_task_address.0 as i32,
                    ],
                );
                let base_instance = GlyphInstance::new(
                    prim_header_index,
                );
                let alpha_batch_list = &mut self.batcher.batch_lists.last_mut().unwrap().alpha_batch_list;
                let render_task_address = render_tasks.get_task_address(
                    self.batcher.render_task_id,
                );

                ctx.resource_cache.fetch_glyphs(
                    run.used_font.clone(),
                    &glyph_keys,
                    &mut self.glyph_fetch_buffer,
                    gpu_cache,
                    |texture_id, mut glyph_format, glyphs| {
                        debug_assert_ne!(texture_id, TextureSource::Invalid);

                        // Ignore color and only sample alpha when shadowing.
                        if run.shadow {
                            glyph_format = glyph_format.ignore_color();
                        }

                        let subpx_dir = subpx_dir.limit_by(glyph_format);

                        let textures = BatchTextures {
                            colors: [
                                texture_id,
                                TextureSource::Invalid,
                                TextureSource::Invalid,
                            ],
                        };

                        let kind = BatchKind::TextRun(glyph_format);

                        let (blend_mode, color_mode) = match glyph_format {
                            GlyphFormat::Subpixel |
                            GlyphFormat::TransformedSubpixel => {
                                if run.used_font.bg_color.a != 0 {
                                    (
                                        BlendMode::SubpixelWithBgColor,
                                        ShaderColorMode::FromRenderPassMode,
                                    )
                                } else if ctx.use_dual_source_blending {
                                    (
                                        BlendMode::SubpixelDualSource,
                                        ShaderColorMode::SubpixelDualSource,
                                    )
                                } else {
                                    (
                                        BlendMode::SubpixelConstantTextColor(run.used_font.color.into()),
                                        ShaderColorMode::SubpixelConstantTextColor,
                                    )
                                }
                            }
                            GlyphFormat::Alpha |
                            GlyphFormat::TransformedAlpha => {
                                (
                                    BlendMode::PremultipliedAlpha,
                                    ShaderColorMode::Alpha,
                                )
                            }
                            GlyphFormat::Bitmap => {
                                (
                                    BlendMode::PremultipliedAlpha,
                                    ShaderColorMode::Bitmap,
                                )
                            }
                            GlyphFormat::ColorBitmap => {
                                (
                                    BlendMode::PremultipliedAlpha,
                                    ShaderColorMode::ColorBitmap,
                                )
                            }
                        };

                        let key = BatchKey::new(kind, blend_mode, textures);
                        let batch = alpha_batch_list.set_params_and_get_batch(
                            key,
                            bounding_rect,
                            z_id,
                        );

                        for glyph in glyphs {
                            batch.push(base_instance.build(
                                glyph.index_in_text_run | ((render_task_address.0 as i32) << 16),
                                glyph.uv_rect_address.as_int(),
                                (rasterization_space as i32) << 16 |
                                (subpx_dir as u32 as i32) << 8 |
                                (color_mode as u32 as i32),
                            ));
                        }
                    },
                );
            }
            PrimitiveInstanceKind::LineDecoration { data_handle, ref cache_handle, .. } => {
                // The GPU cache data is stored in the template and reused across
                // frames and display lists.
                let common_data = &ctx.data_stores.line_decoration[data_handle].common;
                let prim_cache_address = gpu_cache.get_address(&common_data.gpu_cache_handle);

                let (batch_kind, textures, prim_user_data, segment_user_data) = match cache_handle {
                    Some(cache_handle) => {
                        let rt_cache_entry = ctx
                            .resource_cache
                            .get_cached_render_task(cache_handle);
                        let cache_item = ctx
                            .resource_cache
                            .get_texture_cache_item(&rt_cache_entry.handle);
                        let textures = BatchTextures::color(cache_item.texture_id);
                        (
                            BrushBatchKind::Image(get_buffer_kind(cache_item.texture_id)),
                            textures,
                            [
                                ShaderColorMode::Image as i32 | ((AlphaType::PremultipliedAlpha as i32) << 16),
                                RasterizationSpace::Local as i32,
                                get_shader_opacity(1.0),
                                0,
                            ],
                            cache_item.uv_rect_handle.as_int(gpu_cache),
                        )
                    }
                    None => {
                        (
                            BrushBatchKind::Solid,
                            BatchTextures::no_texture(),
                            [get_shader_opacity(1.0), 0, 0, 0],
                            0,
                        )
                    }
                };

                // TODO(gw): We can abstract some of the common code below into
                //           helper methods, as we port more primitives to make
                //           use of interning.
                let blend_mode = if !common_data.opacity.is_opaque ||
                    prim_info.clip_task_index != ClipTaskIndex::INVALID ||
                    transform_kind == TransformedRectKind::Complex
                {
                    BlendMode::PremultipliedAlpha
                } else {
                    BlendMode::None
                };

                let prim_header = PrimitiveHeader {
                    local_rect: prim_rect,
                    local_clip_rect: prim_info.combined_local_clip_rect,
                    snap_offsets,
                    specific_prim_address: prim_cache_address,
                    transform_id,
                };

                let prim_header_index = prim_headers.push(
                    &prim_header,
                    z_id,
                    prim_user_data,
                );

                let batch_key = BatchKey {
                    blend_mode,
                    kind: BatchKind::Brush(batch_kind),
                    textures,
                };

                let clip_task_address = ctx.get_prim_clip_task_address(
                    prim_info.clip_task_index,
                    render_tasks,
                ).unwrap_or(OPAQUE_TASK_ADDRESS);

                self.add_brush_instance_to_batches(
                    batch_key,
                    bounding_rect,
                    z_id,
                    INVALID_SEGMENT_INDEX,
                    EdgeAaSegmentMask::all(),
                    clip_task_address,
                    BrushFlags::PERSPECTIVE_INTERPOLATION,
                    prim_header_index,
                    segment_user_data,
                );
            }
            PrimitiveInstanceKind::Picture { pic_index, segment_instance_index, .. } => {
                let picture = &ctx.prim_store.pictures[pic_index.0];
                let non_segmented_blend_mode = BlendMode::PremultipliedAlpha;
                let prim_cache_address = gpu_cache.get_address(&ctx.globals.default_image_handle);

                let prim_header = PrimitiveHeader {
                    local_rect: picture.snapped_local_rect,
                    local_clip_rect: prim_info.combined_local_clip_rect,
                    snap_offsets,
                    specific_prim_address: prim_cache_address,
                    transform_id,
                };

                match picture.context_3d {
                    // Convert all children of the 3D hierarchy root into batches.
                    Picture3DContext::In { root_data: Some(ref list), .. } => {
                        for child in list {
                            let child_prim_instance = &picture.prim_list.prim_instances[child.anchor];
                            let child_prim_info = &ctx.scratch.prim_info[child_prim_instance.visibility_info.0 as usize];

                            let child_pic_index = match child_prim_instance.kind {
                                PrimitiveInstanceKind::Picture { pic_index, .. } => pic_index,
                                PrimitiveInstanceKind::LineDecoration { .. } |
                                PrimitiveInstanceKind::TextRun { .. } |
                                PrimitiveInstanceKind::NormalBorder { .. } |
                                PrimitiveInstanceKind::ImageBorder { .. } |
                                PrimitiveInstanceKind::Rectangle { .. } |
                                PrimitiveInstanceKind::YuvImage { .. } |
                                PrimitiveInstanceKind::Image { .. } |
                                PrimitiveInstanceKind::LinearGradient { .. } |
                                PrimitiveInstanceKind::RadialGradient { .. } |
                                PrimitiveInstanceKind::Clear { .. } => {
                                    unreachable!();
                                }
                            };
                            let pic = &ctx.prim_store.pictures[child_pic_index.0];

                            // Get clip task, if set, for the picture primitive.
                            let clip_task_address = ctx.get_prim_clip_task_address(
                                child_prim_info.clip_task_index,
                                render_tasks,
                            ).unwrap_or(OPAQUE_TASK_ADDRESS);

                            let prim_header = PrimitiveHeader {
                                local_rect: pic.snapped_local_rect,
                                local_clip_rect: child_prim_info.combined_local_clip_rect,
                                snap_offsets,
                                specific_prim_address: GpuCacheAddress::INVALID,
                                transform_id: transforms
                                    .get_id(
                                        child.spatial_node_index,
                                        root_spatial_node_index,
                                        ctx.clip_scroll_tree,
                                    ),
                            };

                            let raster_config = pic
                                .raster_config
                                .as_ref()
                                .expect("BUG: 3d primitive was not assigned a surface");
                            let (uv_rect_address, _) = render_tasks.resolve_surface(
                                ctx.surfaces[raster_config.surface_index.0]
                                    .render_tasks
                                    .expect("BUG: no surface")
                                    .root,
                                gpu_cache,
                            );

                            let prim_header_index = prim_headers.push(&prim_header, z_id, [
                                uv_rect_address.as_int(),
                                if raster_config.establishes_raster_root { 1 } else { 0 },
                                0,
                                clip_task_address.0 as i32,
                            ]);

                            let key = BatchKey::new(
                                BatchKind::SplitComposite,
                                BlendMode::PremultipliedAlpha,
                                BatchTextures::no_texture(),
                            );

                            self.add_split_composite_instance_to_batches(
                                key,
                                &prim_info.clip_chain.pic_clip_rect,
                                z_id,
                                prim_header_index,
                                child.gpu_address,
                            );
                        }
                    }
                    // Ignore the 3D pictures that are not in the root of preserve-3D
                    // hierarchy, since we process them with the root.
                    Picture3DContext::In { root_data: None, .. } => return,
                    // Proceed for non-3D pictures.
                    Picture3DContext::Out => ()
                }

                match picture.raster_config {
                    Some(ref raster_config) => {
                        // If the child picture was rendered in local space, we can safely
                        // interpolate the UV coordinates with perspective correction.
                        let brush_flags = if raster_config.establishes_raster_root {
                            BrushFlags::PERSPECTIVE_INTERPOLATION
                        } else {
                            BrushFlags::empty()
                        };

                        let clip_task_address = ctx.get_prim_clip_task_address(
                            prim_info.clip_task_index,
                            render_tasks,
                        ).unwrap_or(OPAQUE_TASK_ADDRESS);

                        let surface = &ctx.surfaces[raster_config.surface_index.0];
                        let surface_task = surface.render_tasks.map(|s| s.root);

                        match raster_config.composite_mode {
                            PictureCompositeMode::TileCache { .. } => {
                                let tile_cache = picture.tile_cache.as_ref().unwrap();

                                // If the tile cache is disabled, just recurse into the
                                // picture like a normal pass-through picture, adding
                                // any child primitives into the parent surface batches.
                                if !tile_cache.is_enabled {
                                    // Forcefully break the batches if the
                                    if surface.surface_spatial_node_index != surface_spatial_node_index {
                                        self.batcher.push_new_batch_list(
                                            Vec::new(),
                                            Vec::new(),
                                        );
                                    }

                                    self.add_pic_to_batch(
                                        picture,
                                        ctx,
                                        gpu_cache,
                                        render_tasks,
                                        deferred_resolves,
                                        prim_headers,
                                        transforms,
                                        root_spatial_node_index,
                                        surface.surface_spatial_node_index,
                                        z_generator,
                                    );

                                    if surface.surface_spatial_node_index != surface_spatial_node_index {
                                        self.batcher.push_new_batch_list(
                                            Vec::new(),
                                            Vec::new(),
                                        );
                                    }

                                    return;
                                }

                                // Construct a local clip rect that ensures we only draw pixels where
                                // the local bounds of the picture extend to within the edge tiles.
                                let local_clip_rect = prim_info
                                    .combined_local_clip_rect
                                    .intersection(&picture.snapped_local_rect)
                                    .and_then(|rect| {
                                        rect.intersection(&tile_cache.local_clip_rect)
                                    });

                                if let Some(local_clip_rect) = local_clip_rect {
                                    // Step through each tile in the cache, and draw it with an image
                                    // brush primitive if visible.

                                    let kind = BatchKind::Brush(
                                        BrushBatchKind::Image(ImageBufferKind::Texture2DArray)
                                    );

                                    for tile_index in &tile_cache.tiles_to_draw {
                                        let tile = &tile_cache.tiles[tile_index.0];

                                        // Get the local rect of the tile.
                                        let tile_rect = tile.local_rect;

                                        // Adjust the snap offsets for the tile.
                                        let snap_offsets = recompute_snap_offsets(
                                            tile_rect,
                                            prim_rect,
                                            snap_offsets,
                                        );

                                        let prim_header = PrimitiveHeader {
                                            local_rect: tile_rect,
                                            local_clip_rect,
                                            snap_offsets,
                                            specific_prim_address: prim_cache_address,
                                            transform_id,
                                        };

                                        let prim_header_index = prim_headers.push(&prim_header, z_id, [
                                            ShaderColorMode::Image as i32 | ((AlphaType::PremultipliedAlpha as i32) << 16),
                                            RasterizationSpace::Local as i32,
                                            get_shader_opacity(1.0),
                                            0,
                                        ]);

                                        let cache_item = ctx
                                            .resource_cache
                                            .get_texture_cache_item(&tile.handle);

                                        let key = BatchKey::new(
                                            kind,
                                            BlendMode::None,
                                            BatchTextures::color(cache_item.texture_id),
                                        );

                                        let uv_rect_address = gpu_cache
                                            .get_address(&cache_item.uv_rect_handle)
                                            .as_int();

                                        self.add_brush_instance_to_batches(
                                            key,
                                            bounding_rect,
                                            z_id,
                                            INVALID_SEGMENT_INDEX,
                                            EdgeAaSegmentMask::empty(),
                                            clip_task_address,
                                            brush_flags,
                                            prim_header_index,
                                            uv_rect_address,
                                        );
                                    }

                                    // If there is a dirty rect for the tile cache, recurse into the
                                    // main picture primitive list, and draw them first.
                                    if !tile_cache.dirty_region.is_empty() {
                                        let mut tile_blits = Vec::new();

                                        let (target_rect, _) = render_tasks[self.batcher.render_task_id]
                                            .get_target_rect();

                                        for blit in &tile_cache.pending_blits {
                                            tile_blits.push(TileBlit {
                                                dest_offset: blit.dest_offset,
                                                size: blit.size,
                                                target: blit.target.clone(),
                                                src_offset: DeviceIntPoint::new(
                                                    blit.src_offset.x + target_rect.origin.x,
                                                    blit.src_offset.y + target_rect.origin.y,
                                                ),
                                            })
                                        }

                                        // Collect the list of regions to scissor and repeat
                                        // the draw calls into, based on dirty rects.
                                        let batch_regions = tile_cache
                                            .dirty_region
                                            .dirty_rects
                                            .iter()
                                            .map(|dirty_rect| {
                                                (dirty_rect.world_rect * ctx.global_device_pixel_scale).round().to_i32()
                                            })
                                            .collect();

                                        self.batcher.push_new_batch_list(
                                            batch_regions,
                                            tile_blits,
                                        );

                                        self.add_pic_to_batch(
                                            picture,
                                            ctx,
                                            gpu_cache,
                                            render_tasks,
                                            deferred_resolves,
                                            prim_headers,
                                            transforms,
                                            root_spatial_node_index,
                                            surface.surface_spatial_node_index,
                                            z_generator,
                                        );

                                        self.batcher.push_new_batch_list(
                                            Vec::new(),
                                            Vec::new(),
                                        );
                                    }
                                }
                            }
                            PictureCompositeMode::Filter(ref filter) => {
                                assert!(filter.is_visible());
                                match filter {
                                    Filter::Blur(..) => {
                                        let kind = BatchKind::Brush(
                                            BrushBatchKind::Image(ImageBufferKind::Texture2DArray)
                                        );
                                        let (uv_rect_address, textures) = render_tasks.resolve_surface(
                                            surface_task.expect("bug: surface must be allocated by now"),
                                            gpu_cache,
                                        );
                                        let key = BatchKey::new(
                                            kind,
                                            non_segmented_blend_mode,
                                            textures,
                                        );
                                        let prim_header_index = prim_headers.push(&prim_header, z_id, [
                                            ShaderColorMode::Image as i32 | ((AlphaType::PremultipliedAlpha as i32) << 16),
                                            RasterizationSpace::Screen as i32,
                                            get_shader_opacity(1.0),
                                            0,
                                        ]);

                                        self.add_brush_instance_to_batches(
                                            key,
                                            bounding_rect,
                                            z_id,
                                            INVALID_SEGMENT_INDEX,
                                            EdgeAaSegmentMask::empty(),
                                            clip_task_address,
                                            brush_flags,
                                            prim_header_index,
                                            uv_rect_address.as_int(),
                                        );
                                    }
                                    Filter::DropShadows(shadows) => {
                                        // Draw an instance per shadow first, following by the content.

                                        // The shadows and the content get drawn as a brush image.
                                        let kind = BatchKind::Brush(
                                            BrushBatchKind::Image(ImageBufferKind::Texture2DArray),
                                        );

                                        // Gets the saved render task ID of the content, which is
                                        // deeper in the render task graph than the direct child.
                                        let secondary_id = picture.secondary_render_task_id.expect("no secondary!?");
                                        let saved_index = render_tasks[secondary_id].saved_index.expect("no saved index!?");
                                        debug_assert_ne!(saved_index, SavedTargetIndex::PENDING);

                                        // Build BatchTextures for shadow/content
                                        let shadow_textures = BatchTextures::render_target_cache();
                                        let content_textures = BatchTextures {
                                            colors: [
                                                TextureSource::RenderTaskCache(saved_index),
                                                TextureSource::Invalid,
                                                TextureSource::Invalid,
                                            ],
                                        };

                                        // Build batch keys for shadow/content
                                        let shadow_key = BatchKey::new(kind, non_segmented_blend_mode, shadow_textures);
                                        let content_key = BatchKey::new(kind, non_segmented_blend_mode, content_textures);

                                        // Retrieve the UV rect addresses for shadow/content.
                                        let cache_task_id = surface_task
                                            .expect("bug: surface must be allocated by now");
                                        let shadow_uv_rect_address = render_tasks[cache_task_id]
                                            .get_texture_address(gpu_cache)
                                            .as_int();
                                        let content_uv_rect_address = render_tasks[secondary_id]
                                            .get_texture_address(gpu_cache)
                                            .as_int();

                                        for (shadow, shadow_gpu_data) in shadows.iter().zip(picture.extra_gpu_data_handles.iter()) {
                                            // Get the GPU cache address of the extra data handle.
                                            let shadow_prim_address = gpu_cache.get_address(shadow_gpu_data);

                                            let shadow_rect = prim_header.local_rect.translate(&shadow.offset);

                                            let shadow_prim_header = PrimitiveHeader {
                                                local_rect: shadow_rect,
                                                snap_offsets: prim_info.shadow_snap_offsets,
                                                specific_prim_address: shadow_prim_address,
                                                ..prim_header
                                            };

                                            let shadow_prim_header_index = prim_headers.push(&shadow_prim_header, z_id, [
                                                ShaderColorMode::Alpha as i32 | ((AlphaType::PremultipliedAlpha as i32) << 16),
                                                RasterizationSpace::Screen as i32,
                                                get_shader_opacity(1.0),
                                                0,
                                            ]);

                                            self.add_brush_instance_to_batches(
                                                shadow_key,
                                                bounding_rect,
                                                z_id,
                                                INVALID_SEGMENT_INDEX,
                                                EdgeAaSegmentMask::empty(),
                                                clip_task_address,
                                                brush_flags,
                                                shadow_prim_header_index,
                                                shadow_uv_rect_address,
                                            );
                                        }
                                        let z_id_content = z_generator.next();

                                        let content_prim_header_index = prim_headers.push(&prim_header, z_id_content, [
                                            ShaderColorMode::Image as i32 | ((AlphaType::PremultipliedAlpha as i32) << 16),
                                            RasterizationSpace::Screen as i32,
                                            get_shader_opacity(1.0),
                                            0,
                                        ]);

                                        self.add_brush_instance_to_batches(
                                            content_key,
                                            bounding_rect,
                                            z_id_content,
                                            INVALID_SEGMENT_INDEX,
                                            EdgeAaSegmentMask::empty(),
                                            clip_task_address,
                                            brush_flags,
                                            content_prim_header_index,
                                            content_uv_rect_address,
                                        );
                                    }
                                    _ => {
                                        let filter_mode = match filter {
                                            Filter::Identity => 1, // matches `Contrast(1)`
                                            Filter::Blur(..) => 0,
                                            Filter::Contrast(..) => 1,
                                            Filter::Grayscale(..) => 2,
                                            Filter::HueRotate(..) => 3,
                                            Filter::Invert(..) => 4,
                                            Filter::Saturate(..) => 5,
                                            Filter::Sepia(..) => 6,
                                            Filter::Brightness(..) => 7,
                                            Filter::Opacity(..) => 8,
                                            Filter::DropShadows(..) => 9,
                                            Filter::ColorMatrix(..) => 10,
                                            Filter::SrgbToLinear => 11,
                                            Filter::LinearToSrgb => 12,
                                            Filter::ComponentTransfer => unreachable!(),
                                            Filter::Flood(..) => 14,
                                        };

                                        let user_data = match filter {
                                            Filter::Identity => 0x10000i32, // matches `Contrast(1)`
                                            Filter::Contrast(amount) |
                                            Filter::Grayscale(amount) |
                                            Filter::Invert(amount) |
                                            Filter::Saturate(amount) |
                                            Filter::Sepia(amount) |
                                            Filter::Brightness(amount) |
                                            Filter::Opacity(_, amount) => {
                                                (amount * 65536.0) as i32
                                            }
                                            Filter::SrgbToLinear | Filter::LinearToSrgb => 0,
                                            Filter::HueRotate(angle) => {
                                                (0.01745329251 * angle * 65536.0) as i32
                                            }
                                            // Go through different paths
                                            Filter::Blur(..) |
                                            Filter::DropShadows(..) => {
                                                unreachable!();
                                            }
                                            Filter::ColorMatrix(_) => {
                                                picture.extra_gpu_data_handles[0].as_int(gpu_cache)
                                            }
                                            Filter::ComponentTransfer => unreachable!(),
                                            Filter::Flood(_) => {
                                                picture.extra_gpu_data_handles[0].as_int(gpu_cache)
                                            }
                                        };

                                        let (uv_rect_address, textures) = render_tasks.resolve_surface(
                                            surface_task.expect("bug: surface must be allocated by now"),
                                            gpu_cache,
                                        );

                                        let key = BatchKey::new(
                                            BatchKind::Brush(BrushBatchKind::Blend),
                                            BlendMode::PremultipliedAlpha,
                                            textures,
                                        );

                                        let prim_header_index = prim_headers.push(&prim_header, z_id, [
                                            uv_rect_address.as_int(),
                                            filter_mode,
                                            user_data,
                                            0,
                                        ]);

                                        self.add_brush_instance_to_batches(
                                            key,
                                            bounding_rect,
                                            z_id,
                                            INVALID_SEGMENT_INDEX,
                                            EdgeAaSegmentMask::empty(),
                                            clip_task_address,
                                            brush_flags,
                                            prim_header_index,
                                            0,
                                        );
                                    }
                                }
                            }
                            PictureCompositeMode::ComponentTransferFilter(handle) => {
                                // This is basically the same as the general filter case above
                                // except we store a little more data in the filter mode and
                                // a gpu cache handle in the user data.
                                let filter_data = &ctx.data_stores.filter_data[handle];
                                let filter_mode : i32 = 13 |
                                    ((filter_data.data.r_func.to_int() << 28 |
                                      filter_data.data.g_func.to_int() << 24 |
                                      filter_data.data.b_func.to_int() << 20 |
                                      filter_data.data.a_func.to_int() << 16) as i32);

                                let user_data = filter_data.gpu_cache_handle.as_int(gpu_cache);

                                let (uv_rect_address, textures) = render_tasks.resolve_surface(
                                    surface_task.expect("bug: surface must be allocated by now"),
                                    gpu_cache,
                                );

                                let key = BatchKey::new(
                                    BatchKind::Brush(BrushBatchKind::Blend),
                                    BlendMode::PremultipliedAlpha,
                                    textures,
                                );

                                let prim_header_index = prim_headers.push(&prim_header, z_id, [
                                    uv_rect_address.as_int(),
                                    filter_mode,
                                    user_data,
                                    0,
                                ]);

                                self.add_brush_instance_to_batches(
                                    key,
                                    bounding_rect,
                                    z_id,
                                    INVALID_SEGMENT_INDEX,
                                    EdgeAaSegmentMask::empty(),
                                    clip_task_address,
                                    brush_flags,
                                    prim_header_index,
                                    0,
                                );
                            }
                            PictureCompositeMode::MixBlend(mode) if ctx.use_advanced_blending => {
                                let (uv_rect_address, textures) = render_tasks.resolve_surface(
                                    surface_task.expect("bug: surface must be allocated by now"),
                                    gpu_cache,
                                );
                                let key = BatchKey::new(
                                    BatchKind::Brush(
                                        BrushBatchKind::Image(ImageBufferKind::Texture2DArray),
                                    ),
                                    BlendMode::Advanced(mode),
                                    textures,
                                );
                                let prim_header_index = prim_headers.push(&prim_header, z_id, [
                                    ShaderColorMode::Image as i32 | ((AlphaType::PremultipliedAlpha as i32) << 16),
                                    RasterizationSpace::Local as i32,
                                    get_shader_opacity(1.0),
                                    0,
                                ]);

                                self.add_brush_instance_to_batches(
                                    key,
                                    bounding_rect,
                                    z_id,
                                    INVALID_SEGMENT_INDEX,
                                    EdgeAaSegmentMask::empty(),
                                    clip_task_address,
                                    brush_flags,
                                    prim_header_index,
                                    uv_rect_address.as_int(),
                                );
                            }
                            PictureCompositeMode::MixBlend(mode) => {
                                let cache_task_id = surface_task.expect("bug: surface must be allocated by now");
                                let backdrop_id = picture.secondary_render_task_id.expect("no backdrop!?");

                                let key = BatchKey::new(
                                    BatchKind::Brush(
                                        BrushBatchKind::MixBlend {
                                            task_id: self.batcher.render_task_id,
                                            source_id: cache_task_id,
                                            backdrop_id,
                                        },
                                    ),
                                    BlendMode::PremultipliedAlpha,
                                    BatchTextures::no_texture(),
                                );
                                let backdrop_task_address = render_tasks.get_task_address(backdrop_id);
                                let source_task_address = render_tasks.get_task_address(cache_task_id);
                                let prim_header_index = prim_headers.push(&prim_header, z_id, [
                                    mode as u32 as i32,
                                    backdrop_task_address.0 as i32,
                                    source_task_address.0 as i32,
                                    0,
                                ]);

                                self.add_brush_instance_to_batches(
                                    key,
                                    bounding_rect,
                                    z_id,
                                    INVALID_SEGMENT_INDEX,
                                    EdgeAaSegmentMask::empty(),
                                    clip_task_address,
                                    brush_flags,
                                    prim_header_index,
                                    0,
                                );
                            }
                            PictureCompositeMode::Blit(_) => {
                                let cache_task_id = surface_task.expect("bug: surface must be allocated by now");
                                let uv_rect_address = render_tasks[cache_task_id]
                                    .get_texture_address(gpu_cache)
                                    .as_int();
                                let batch_params = BrushBatchParameters::shared(
                                    BrushBatchKind::Image(ImageBufferKind::Texture2DArray),
                                    BatchTextures::render_target_cache(),
                                    [
                                        ShaderColorMode::Image as i32 | ((AlphaType::PremultipliedAlpha as i32) << 16),
                                        RasterizationSpace::Screen as i32,
                                        get_shader_opacity(1.0),
                                        0,
                                    ],
                                    uv_rect_address,
                                );

                                let is_segmented =
                                    segment_instance_index != SegmentInstanceIndex::INVALID &&
                                    segment_instance_index != SegmentInstanceIndex::UNUSED;

                                let (prim_cache_address, segments) = if is_segmented {
                                    let segment_instance = &ctx.scratch.segment_instances[segment_instance_index];
                                    let segments = Some(&ctx.scratch.segments[segment_instance.segments_range]);
                                    (gpu_cache.get_address(&segment_instance.gpu_cache_handle), segments)
                                } else {
                                    (prim_cache_address, None)
                                };

                                let prim_header = PrimitiveHeader {
                                    local_rect: picture.snapped_local_rect,
                                    local_clip_rect: prim_info.combined_local_clip_rect,
                                    snap_offsets,
                                    specific_prim_address: prim_cache_address,
                                    transform_id,
                                };

                                let prim_header_index = prim_headers.push(
                                    &prim_header,
                                    z_id,
                                    batch_params.prim_user_data,
                                );

                                // TODO(gw): As before, all pictures that get blitted are assumed
                                //           to have alpha. However, we could determine (at least for
                                //           simple, common cases) if the picture content is opaque.
                                //           That would allow inner segments of pictures to be drawn
                                //           with blend disabled, which is a big performance win on
                                //           integrated GPUs.
                                let opacity = PrimitiveOpacity::translucent();
                                let specified_blend_mode = BlendMode::PremultipliedAlpha;

                                self.add_segmented_prim_to_batch(
                                    segments,
                                    opacity,
                                    &batch_params,
                                    specified_blend_mode,
                                    non_segmented_blend_mode,
                                    prim_header_index,
                                    bounding_rect,
                                    transform_kind,
                                    render_tasks,
                                    z_id,
                                    prim_info.clip_task_index,
                                    ctx,
                                );
                            }
                        }
                    }
                    None => {
                        // If this picture is being drawn into an existing target (i.e. with
                        // no composition operation), recurse and add to the current batch list.
                        self.add_pic_to_batch(
                            picture,
                            ctx,
                            gpu_cache,
                            render_tasks,
                            deferred_resolves,
                            prim_headers,
                            transforms,
                            root_spatial_node_index,
                            surface_spatial_node_index,
                            z_generator,
                        );
                    }
                }
            }
            PrimitiveInstanceKind::ImageBorder { data_handle, .. } => {
                let prim_data = &ctx.data_stores.image_border[data_handle];
                let common_data = &prim_data.common;
                let border_data = &prim_data.kind;

                let cache_item = resolve_image(
                    border_data.request,
                    ctx.resource_cache,
                    gpu_cache,
                    deferred_resolves,
                );
                if cache_item.texture_id == TextureSource::Invalid {
                    return;
                }

                let textures = BatchTextures::color(cache_item.texture_id);
                let prim_cache_address = gpu_cache.get_address(&common_data.gpu_cache_handle);
                let specified_blend_mode = BlendMode::PremultipliedAlpha;
                let non_segmented_blend_mode = if !common_data.opacity.is_opaque ||
                    prim_info.clip_task_index != ClipTaskIndex::INVALID ||
                    transform_kind == TransformedRectKind::Complex
                {
                    specified_blend_mode
                } else {
                    BlendMode::None
                };

                let prim_header = PrimitiveHeader {
                    local_rect: prim_rect,
                    local_clip_rect: prim_info.combined_local_clip_rect,
                    snap_offsets,
                    specific_prim_address: prim_cache_address,
                    transform_id,
                };

                let batch_params = BrushBatchParameters::shared(
                    BrushBatchKind::Image(get_buffer_kind(cache_item.texture_id)),
                    textures,
                    [
                        ShaderColorMode::Image as i32 | ((AlphaType::PremultipliedAlpha as i32) << 16),
                        RasterizationSpace::Local as i32,
                        get_shader_opacity(1.0),
                        0,
                    ],
                    cache_item.uv_rect_handle.as_int(gpu_cache),
                );

                let prim_header_index = prim_headers.push(
                    &prim_header,
                    z_id,
                    batch_params.prim_user_data,
                );

                self.add_segmented_prim_to_batch(
                    Some(border_data.brush_segments.as_slice()),
                    common_data.opacity,
                    &batch_params,
                    specified_blend_mode,
                    non_segmented_blend_mode,
                    prim_header_index,
                    bounding_rect,
                    transform_kind,
                    render_tasks,
                    z_id,
                    prim_info.clip_task_index,
                    ctx,
                );
            }
            PrimitiveInstanceKind::Rectangle { data_handle, segment_instance_index, opacity_binding_index, .. } => {
                let prim_data = &ctx.data_stores.prim[data_handle];
                let specified_blend_mode = BlendMode::PremultipliedAlpha;
                let opacity_binding = ctx.prim_store.get_opacity_binding(opacity_binding_index);

                let opacity = PrimitiveOpacity::from_alpha(opacity_binding);
                let opacity = opacity.combine(prim_data.opacity);

                let non_segmented_blend_mode = if !opacity.is_opaque ||
                    prim_info.clip_task_index != ClipTaskIndex::INVALID ||
                    transform_kind == TransformedRectKind::Complex
                {
                    specified_blend_mode
                } else {
                    BlendMode::None
                };

                let batch_params = BrushBatchParameters::shared(
                    BrushBatchKind::Solid,
                    BatchTextures::no_texture(),
                    [get_shader_opacity(opacity_binding), 0, 0, 0],
                    0,
                );

                let (prim_cache_address, segments) = if segment_instance_index == SegmentInstanceIndex::UNUSED {
                    (gpu_cache.get_address(&prim_data.gpu_cache_handle), None)
                } else {
                    let segment_instance = &ctx.scratch.segment_instances[segment_instance_index];
                    let segments = Some(&ctx.scratch.segments[segment_instance.segments_range]);
                    (gpu_cache.get_address(&segment_instance.gpu_cache_handle), segments)
                };

                let prim_header = PrimitiveHeader {
                    local_rect: prim_rect,
                    local_clip_rect: prim_info.combined_local_clip_rect,
                    snap_offsets,
                    specific_prim_address: prim_cache_address,
                    transform_id,
                };

                let prim_header_index = prim_headers.push(
                    &prim_header,
                    z_id,
                    batch_params.prim_user_data,
                );

                self.add_segmented_prim_to_batch(
                    segments,
                    opacity,
                    &batch_params,
                    specified_blend_mode,
                    non_segmented_blend_mode,
                    prim_header_index,
                    bounding_rect,
                    transform_kind,
                    render_tasks,
                    z_id,
                    prim_info.clip_task_index,
                    ctx,
                );
            }
            PrimitiveInstanceKind::YuvImage { data_handle, segment_instance_index, .. } => {
                let yuv_image_data = &ctx.data_stores.yuv_image[data_handle].kind;
                let mut textures = BatchTextures::no_texture();
                let mut uv_rect_addresses = [0; 3];

                //yuv channel
                let channel_count = yuv_image_data.format.get_plane_num();
                debug_assert!(channel_count <= 3);
                for channel in 0 .. channel_count {
                    let image_key = yuv_image_data.yuv_key[channel];

                    let cache_item = resolve_image(
                        ImageRequest {
                            key: image_key,
                            rendering: yuv_image_data.image_rendering,
                            tile: None,
                        },
                        ctx.resource_cache,
                        gpu_cache,
                        deferred_resolves,
                    );

                    if cache_item.texture_id == TextureSource::Invalid {
                        warn!("Warnings: skip a PrimitiveKind::YuvImage");
                        return;
                    }

                    textures.colors[channel] = cache_item.texture_id;
                    uv_rect_addresses[channel] = cache_item.uv_rect_handle.as_int(gpu_cache);
                }

                // All yuv textures should be the same type.
                let buffer_kind = get_buffer_kind(textures.colors[0]);
                assert!(
                    textures.colors[1 .. yuv_image_data.format.get_plane_num()]
                        .iter()
                        .all(|&tid| buffer_kind == get_buffer_kind(tid))
                );

                let kind = BrushBatchKind::YuvImage(
                    buffer_kind,
                    yuv_image_data.format,
                    yuv_image_data.color_depth,
                    yuv_image_data.color_space,
                );

                let batch_params = BrushBatchParameters::shared(
                    kind,
                    textures,
                    [
                        uv_rect_addresses[0],
                        uv_rect_addresses[1],
                        uv_rect_addresses[2],
                        0,
                    ],
                    0,
                );

                let specified_blend_mode = BlendMode::PremultipliedAlpha;

                let non_segmented_blend_mode = if !prim_common_data.opacity.is_opaque ||
                    prim_info.clip_task_index != ClipTaskIndex::INVALID ||
                    transform_kind == TransformedRectKind::Complex
                {
                    specified_blend_mode
                } else {
                    BlendMode::None
                };

                debug_assert_ne!(segment_instance_index, SegmentInstanceIndex::INVALID);
                let (prim_cache_address, segments) = if segment_instance_index == SegmentInstanceIndex::UNUSED {
                    (gpu_cache.get_address(&prim_common_data.gpu_cache_handle), None)
                } else {
                    let segment_instance = &ctx.scratch.segment_instances[segment_instance_index];
                    let segments = Some(&ctx.scratch.segments[segment_instance.segments_range]);
                    (gpu_cache.get_address(&segment_instance.gpu_cache_handle), segments)
                };

                let prim_header = PrimitiveHeader {
                    local_rect: prim_rect,
                    local_clip_rect: prim_info.combined_local_clip_rect,
                    snap_offsets,
                    specific_prim_address: prim_cache_address,
                    transform_id,
                };

                let prim_header_index = prim_headers.push(
                    &prim_header,
                    z_id,
                    batch_params.prim_user_data,
                );

                self.add_segmented_prim_to_batch(
                    segments,
                    prim_common_data.opacity,
                    &batch_params,
                    specified_blend_mode,
                    non_segmented_blend_mode,
                    prim_header_index,
                    bounding_rect,
                    transform_kind,
                    render_tasks,
                    z_id,
                    prim_info.clip_task_index,
                    ctx,
                );
            }
            PrimitiveInstanceKind::Image { data_handle, image_instance_index, .. } => {
                let image_data = &ctx.data_stores.image[data_handle].kind;
                let common_data = &ctx.data_stores.image[data_handle].common;
                let image_instance = &ctx.prim_store.images[image_instance_index];
                let opacity_binding = ctx.prim_store.get_opacity_binding(image_instance.opacity_binding_index);
                let specified_blend_mode = match image_data.alpha_type {
                    AlphaType::PremultipliedAlpha => BlendMode::PremultipliedAlpha,
                    AlphaType::Alpha => BlendMode::Alpha,
                };
                let request = ImageRequest {
                    key: image_data.key,
                    rendering: image_data.image_rendering,
                    tile: None,
                };
                let prim_user_data = [
                    ShaderColorMode::Image as i32 | ((image_data.alpha_type as i32) << 16),
                    RasterizationSpace::Local as i32,
                    get_shader_opacity(opacity_binding),
                    0,
                ];

                if image_instance.visible_tiles.is_empty() {
                    let cache_item = match image_data.source {
                        ImageSource::Default => {
                            resolve_image(
                                request,
                                ctx.resource_cache,
                                gpu_cache,
                                deferred_resolves,
                            )
                        }
                        ImageSource::Cache { ref handle, .. } => {
                            let rt_handle = handle
                                .as_ref()
                                .expect("bug: render task handle not allocated");
                            let rt_cache_entry = ctx.resource_cache
                                .get_cached_render_task(rt_handle);
                            ctx.resource_cache.get_texture_cache_item(&rt_cache_entry.handle)
                        }
                    };

                    if cache_item.texture_id == TextureSource::Invalid {
                        return;
                    }

                    let textures = BatchTextures::color(cache_item.texture_id);

                    let opacity = PrimitiveOpacity::from_alpha(opacity_binding);
                    let opacity = opacity.combine(common_data.opacity);

                    let non_segmented_blend_mode = if !opacity.is_opaque ||
                        prim_info.clip_task_index != ClipTaskIndex::INVALID ||
                        transform_kind == TransformedRectKind::Complex
                    {
                        specified_blend_mode
                    } else {
                        BlendMode::None
                    };

                    let batch_params = BrushBatchParameters::shared(
                        BrushBatchKind::Image(get_buffer_kind(cache_item.texture_id)),
                        textures,
                        prim_user_data,
                        cache_item.uv_rect_handle.as_int(gpu_cache),
                    );

                    debug_assert_ne!(image_instance.segment_instance_index, SegmentInstanceIndex::INVALID);
                    let (prim_cache_address, segments) = if image_instance.segment_instance_index == SegmentInstanceIndex::UNUSED {
                        (gpu_cache.get_address(&common_data.gpu_cache_handle), None)
                    } else {
                        let segment_instance = &ctx.scratch.segment_instances[image_instance.segment_instance_index];
                        let segments = Some(&ctx.scratch.segments[segment_instance.segments_range]);
                        (gpu_cache.get_address(&segment_instance.gpu_cache_handle), segments)
                    };

                    let prim_header = PrimitiveHeader {
                        local_rect: prim_rect,
                        local_clip_rect: prim_info.combined_local_clip_rect,
                        snap_offsets,
                        specific_prim_address: prim_cache_address,
                        transform_id,
                    };

                    let prim_header_index = prim_headers.push(
                        &prim_header,
                        z_id,
                        batch_params.prim_user_data,
                    );

                    self.add_segmented_prim_to_batch(
                        segments,
                        opacity,
                        &batch_params,
                        specified_blend_mode,
                        non_segmented_blend_mode,
                        prim_header_index,
                        bounding_rect,
                        transform_kind,
                        render_tasks,
                        z_id,
                        prim_info.clip_task_index,
                        ctx,
                    );
                } else {
                    const VECS_PER_SPECIFIC_BRUSH: usize = 3;
                    let max_tiles_per_header = (MAX_VERTEX_TEXTURE_WIDTH - VECS_PER_SPECIFIC_BRUSH) / VECS_PER_SEGMENT;

                    let clip_task_address = ctx.get_prim_clip_task_address(
                        prim_info.clip_task_index,
                        render_tasks,
                    ).unwrap_or(OPAQUE_TASK_ADDRESS);

                    // use temporary block storage since we don't know the number of visible tiles beforehand
                    let mut gpu_blocks = Vec::<GpuBlockData>::new();
                    for chunk in image_instance.visible_tiles.chunks(max_tiles_per_header) {
                        gpu_blocks.clear();
                        gpu_blocks.push(PremultipliedColorF::WHITE.into()); //color
                        gpu_blocks.push(PremultipliedColorF::WHITE.into()); //bg color
                        gpu_blocks.push([-1.0, 0.0, 0.0, 0.0].into()); //stretch size
                        // negative first value makes the shader code ignore it and use the local size instead
                        for tile in chunk {
                            let tile_rect = tile.local_rect.translate(&-prim_rect.origin.to_vector());
                            gpu_blocks.push(tile_rect.into());
                            gpu_blocks.push(GpuBlockData::EMPTY);
                        }

                        let gpu_handle = gpu_cache.push_per_frame_blocks(&gpu_blocks);
                        let prim_header = PrimitiveHeader {
                            local_rect: prim_rect,
                            local_clip_rect: image_instance.tight_local_clip_rect,
                            snap_offsets,
                            specific_prim_address: gpu_cache.get_address(&gpu_handle),
                            transform_id,
                        };
                        let prim_header_index = prim_headers.push(&prim_header, z_id, prim_user_data);

                        for (i, tile) in chunk.iter().enumerate() {
                            if let Some((batch_kind, textures, uv_rect_address)) = get_image_tile_params(
                                ctx.resource_cache,
                                gpu_cache,
                                deferred_resolves,
                                request.with_tile(tile.tile_offset),
                            ) {
                                let batch_key = BatchKey {
                                    blend_mode: specified_blend_mode,
                                    kind: BatchKind::Brush(batch_kind),
                                    textures,
                                };
                                self.add_brush_instance_to_batches(
                                    batch_key,
                                    bounding_rect,
                                    z_id,
                                    i as i32,
                                    tile.edge_flags,
                                    clip_task_address,
                                    BrushFlags::SEGMENT_RELATIVE | BrushFlags::PERSPECTIVE_INTERPOLATION,
                                    prim_header_index,
                                    uv_rect_address.as_int(),
                                );
                            }
                        }
                    }
                }
            }
            PrimitiveInstanceKind::LinearGradient { data_handle, gradient_index, .. } => {
                let gradient = &ctx.prim_store.linear_gradients[gradient_index];
                let prim_data = &ctx.data_stores.linear_grad[data_handle];
                let specified_blend_mode = BlendMode::PremultipliedAlpha;

                let mut prim_header = PrimitiveHeader {
                    local_rect: prim_rect,
                    local_clip_rect: prim_info.combined_local_clip_rect,
                    snap_offsets,
                    specific_prim_address: GpuCacheAddress::INVALID,
                    transform_id,
                };

                let non_segmented_blend_mode = if !prim_data.opacity.is_opaque ||
                    prim_info.clip_task_index != ClipTaskIndex::INVALID ||
                    transform_kind == TransformedRectKind::Complex
                {
                    specified_blend_mode
                } else {
                    BlendMode::None
                };

                if let Some(ref cache_handle) = gradient.cache_handle {
                    let rt_cache_entry = ctx.resource_cache
                        .get_cached_render_task(cache_handle);
                    let cache_item = ctx.resource_cache
                        .get_texture_cache_item(&rt_cache_entry.handle);

                    if cache_item.texture_id == TextureSource::Invalid {
                        return;
                    }

                    let textures = BatchTextures::color(cache_item.texture_id);
                    let batch_kind = BrushBatchKind::Image(get_buffer_kind(cache_item.texture_id));
                    let prim_user_data = [
                        ShaderColorMode::Image as i32 | ((AlphaType::PremultipliedAlpha as i32) << 16),
                        RasterizationSpace::Local as i32,
                        get_shader_opacity(1.0),
                        0,
                    ];
                    let segment_user_data = cache_item.uv_rect_handle.as_int(gpu_cache);
                    prim_header.specific_prim_address = gpu_cache.get_address(&ctx.globals.default_image_handle);

                    let prim_header_index = prim_headers.push(
                        &prim_header,
                        z_id,
                        prim_user_data,
                    );

                    let batch_key = BatchKey {
                        blend_mode: non_segmented_blend_mode,
                        kind: BatchKind::Brush(batch_kind),
                        textures,
                    };

                    let clip_task_address = ctx.get_prim_clip_task_address(
                        prim_info.clip_task_index,
                        render_tasks,
                    ).unwrap_or(OPAQUE_TASK_ADDRESS);

                    self.add_brush_instance_to_batches(
                        batch_key,
                        bounding_rect,
                        z_id,
                        INVALID_SEGMENT_INDEX,
                        EdgeAaSegmentMask::all(),
                        clip_task_address,
                        BrushFlags::PERSPECTIVE_INTERPOLATION,
                        prim_header_index,
                        segment_user_data,
                    );
                } else if gradient.visible_tiles_range.is_empty() {
                    let batch_params = BrushBatchParameters::shared(
                        BrushBatchKind::LinearGradient,
                        BatchTextures::no_texture(),
                        [
                            prim_data.stops_handle.as_int(gpu_cache),
                            0,
                            0,
                            0,
                        ],
                        0,
                    );

                    prim_header.specific_prim_address = gpu_cache.get_address(&prim_data.gpu_cache_handle);

                    let prim_header_index = prim_headers.push(
                        &prim_header,
                        z_id,
                        batch_params.prim_user_data,
                    );

                    let segments = if prim_data.brush_segments.is_empty() {
                        None
                    } else {
                        Some(prim_data.brush_segments.as_slice())
                    };

                    self.add_segmented_prim_to_batch(
                        segments,
                        prim_data.opacity,
                        &batch_params,
                        specified_blend_mode,
                        non_segmented_blend_mode,
                        prim_header_index,
                        bounding_rect,
                        transform_kind,
                        render_tasks,
                        z_id,
                        prim_info.clip_task_index,
                        ctx,
                    );
                } else {
                    let visible_tiles = &ctx.scratch.gradient_tiles[gradient.visible_tiles_range];

                    let clip_task_address = ctx.get_prim_clip_task_address(
                        prim_info.clip_task_index,
                        render_tasks,
                    ).unwrap_or(OPAQUE_TASK_ADDRESS);

                    self.add_gradient_tiles(
                        visible_tiles,
                        &prim_data.stops_handle,
                        BrushBatchKind::LinearGradient,
                        specified_blend_mode,
                        bounding_rect,
                        clip_task_address,
                        gpu_cache,
                        &prim_header,
                        prim_headers,
                        z_id,
                    );
                }
            }
            PrimitiveInstanceKind::RadialGradient { data_handle, ref visible_tiles_range, .. } => {
                let prim_data = &ctx.data_stores.radial_grad[data_handle];
                let specified_blend_mode = BlendMode::PremultipliedAlpha;

                let mut prim_header = PrimitiveHeader {
                    local_rect: prim_rect,
                    local_clip_rect: prim_info.combined_local_clip_rect,
                    snap_offsets,
                    specific_prim_address: GpuCacheAddress::INVALID,
                    transform_id,
                };

                if visible_tiles_range.is_empty() {
                    let non_segmented_blend_mode = if !prim_data.opacity.is_opaque ||
                        prim_info.clip_task_index != ClipTaskIndex::INVALID ||
                        transform_kind == TransformedRectKind::Complex
                    {
                        specified_blend_mode
                    } else {
                        BlendMode::None
                    };

                    let batch_params = BrushBatchParameters::shared(
                        BrushBatchKind::RadialGradient,
                        BatchTextures::no_texture(),
                        [
                            prim_data.stops_handle.as_int(gpu_cache),
                            0,
                            0,
                            0,
                        ],
                        0,
                    );

                    prim_header.specific_prim_address = gpu_cache.get_address(&prim_data.gpu_cache_handle);

                    let prim_header_index = prim_headers.push(
                        &prim_header,
                        z_id,
                        batch_params.prim_user_data,
                    );

                    let segments = if prim_data.brush_segments.is_empty() {
                        None
                    } else {
                        Some(prim_data.brush_segments.as_slice())
                    };

                    self.add_segmented_prim_to_batch(
                        segments,
                        prim_data.opacity,
                        &batch_params,
                        specified_blend_mode,
                        non_segmented_blend_mode,
                        prim_header_index,
                        bounding_rect,
                        transform_kind,
                        render_tasks,
                        z_id,
                        prim_info.clip_task_index,
                        ctx,
                    );
                } else {
                    let visible_tiles = &ctx.scratch.gradient_tiles[*visible_tiles_range];

                    let clip_task_address = ctx.get_prim_clip_task_address(
                        prim_info.clip_task_index,
                        render_tasks,
                    ).unwrap_or(OPAQUE_TASK_ADDRESS);

                    self.add_gradient_tiles(
                        visible_tiles,
                        &prim_data.stops_handle,
                        BrushBatchKind::RadialGradient,
                        specified_blend_mode,
                        bounding_rect,
                        clip_task_address,
                        gpu_cache,
                        &prim_header,
                        prim_headers,
                        z_id,
                    );
                }
            }
        }
    }

    /// Add a single segment instance to a batch.
    fn add_segment_to_batch(
        &mut self,
        segment: &BrushSegment,
        segment_data: &SegmentInstanceData,
        segment_index: i32,
        batch_kind: BrushBatchKind,
        prim_header_index: PrimitiveHeaderIndex,
        alpha_blend_mode: BlendMode,
        bounding_rect: &PictureRect,
        transform_kind: TransformedRectKind,
        render_tasks: &RenderTaskGraph,
        z_id: ZBufferId,
        prim_opacity: PrimitiveOpacity,
        clip_task_index: ClipTaskIndex,
        ctx: &RenderTargetContext,
    ) {
        debug_assert!(clip_task_index != ClipTaskIndex::INVALID);

        // Get GPU address of clip task for this segment, or None if
        // the entire segment is clipped out.
        let clip_task_address = match ctx.get_clip_task_address(
            clip_task_index,
            segment_index,
            render_tasks,
        ) {
            Some(clip_task_address) => clip_task_address,
            None => return,
        };

        // If a got a valid (or OPAQUE) clip task address, add the segment.
        let is_inner = segment.edge_flags.is_empty();
        let needs_blending = !prim_opacity.is_opaque ||
                             clip_task_address != OPAQUE_TASK_ADDRESS ||
                             (!is_inner && transform_kind == TransformedRectKind::Complex);

        let batch_key = BatchKey {
            blend_mode: if needs_blending { alpha_blend_mode } else { BlendMode::None },
            kind: BatchKind::Brush(batch_kind),
            textures: segment_data.textures,
        };

        self.add_brush_instance_to_batches(
            batch_key,
            bounding_rect,
            z_id,
            segment_index,
            segment.edge_flags,
            clip_task_address,
            BrushFlags::PERSPECTIVE_INTERPOLATION | segment.brush_flags,
            prim_header_index,
            segment_data.user_data,
        );
    }

    /// Add any segment(s) from a brush to batches.
    fn add_segmented_prim_to_batch(
        &mut self,
        brush_segments: Option<&[BrushSegment]>,
        prim_opacity: PrimitiveOpacity,
        params: &BrushBatchParameters,
        alpha_blend_mode: BlendMode,
        non_segmented_blend_mode: BlendMode,
        prim_header_index: PrimitiveHeaderIndex,
        bounding_rect: &PictureRect,
        transform_kind: TransformedRectKind,
        render_tasks: &RenderTaskGraph,
        z_id: ZBufferId,
        clip_task_index: ClipTaskIndex,
        ctx: &RenderTargetContext,
    ) {
        match (brush_segments, &params.segment_data) {
            (Some(ref brush_segments), SegmentDataKind::Instanced(ref segment_data)) => {
                // In this case, we have both a list of segments, and a list of
                // per-segment instance data. Zip them together to build batches.
                debug_assert_eq!(brush_segments.len(), segment_data.len());
                for (segment_index, (segment, segment_data)) in brush_segments
                    .iter()
                    .zip(segment_data.iter())
                    .enumerate()
                {
                    self.add_segment_to_batch(
                        segment,
                        segment_data,
                        segment_index as i32,
                        params.batch_kind,
                        prim_header_index,
                        alpha_blend_mode,
                        bounding_rect,
                        transform_kind,
                        render_tasks,
                        z_id,
                        prim_opacity,
                        clip_task_index,
                        ctx,
                    );
                }
            }
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
                        alpha_blend_mode,
                        bounding_rect,
                        transform_kind,
                        render_tasks,
                        z_id,
                        prim_opacity,
                        clip_task_index,
                        ctx,
                    );
                }
            }
            (None, SegmentDataKind::Shared(ref segment_data)) => {
                // No segments, and thus no per-segment instance data.
                // Note: the blend mode already takes opacity into account
                let batch_key = BatchKey {
                    blend_mode: non_segmented_blend_mode,
                    kind: BatchKind::Brush(params.batch_kind),
                    textures: segment_data.textures,
                };
                let clip_task_address = ctx.get_prim_clip_task_address(
                    clip_task_index,
                    render_tasks,
                ).unwrap_or(OPAQUE_TASK_ADDRESS);
                self.add_brush_instance_to_batches(
                    batch_key,
                    bounding_rect,
                    z_id,
                    INVALID_SEGMENT_INDEX,
                    EdgeAaSegmentMask::all(),
                    clip_task_address,
                    BrushFlags::PERSPECTIVE_INTERPOLATION,
                    prim_header_index,
                    segment_data.user_data,
                );
            }
            (None, SegmentDataKind::Instanced(..)) => {
                // We should never hit the case where there are no segments,
                // but a list of segment instance data.
                unreachable!();
            }
        }
    }

    fn add_gradient_tiles(
        &mut self,
        visible_tiles: &[VisibleGradientTile],
        stops_handle: &GpuCacheHandle,
        kind: BrushBatchKind,
        blend_mode: BlendMode,
        bounding_rect: &PictureRect,
        clip_task_address: RenderTaskAddress,
        gpu_cache: &GpuCache,
        base_prim_header: &PrimitiveHeader,
        prim_headers: &mut PrimitiveHeaders,
        z_id: ZBufferId,
    ) {
        let key = BatchKey {
            blend_mode,
            kind: BatchKind::Brush(kind),
            textures: BatchTextures::no_texture(),
        };

        let user_data = [stops_handle.as_int(gpu_cache), 0, 0, 0];

        for tile in visible_tiles {
            // Adjust the snap offsets for the tile.
            let snap_offsets = recompute_snap_offsets(
                tile.local_rect,
                base_prim_header.local_rect,
                base_prim_header.snap_offsets,
            );

            let prim_header = PrimitiveHeader {
                specific_prim_address: gpu_cache.get_address(&tile.handle),
                local_rect: tile.local_rect,
                local_clip_rect: tile.local_clip_rect,
                snap_offsets,
                ..*base_prim_header
            };
            let prim_header_index = prim_headers.push(&prim_header, z_id, user_data);

            self.add_brush_instance_to_batches(
                key,
                bounding_rect,
                z_id,
                INVALID_SEGMENT_INDEX,
                EdgeAaSegmentMask::all(),
                clip_task_address,
                BrushFlags::PERSPECTIVE_INTERPOLATION,
                prim_header_index,
                0,
            );
        }
    }
}

fn get_image_tile_params(
    resource_cache: &ResourceCache,
    gpu_cache: &mut GpuCache,
    deferred_resolves: &mut Vec<DeferredResolve>,
    request: ImageRequest,
) -> Option<(BrushBatchKind, BatchTextures, GpuCacheAddress)> {

    let cache_item = resolve_image(
        request,
        resource_cache,
        gpu_cache,
        deferred_resolves,
    );

    if cache_item.texture_id == TextureSource::Invalid {
        None
    } else {
        let textures = BatchTextures::color(cache_item.texture_id);
        Some((
            BrushBatchKind::Image(get_buffer_kind(cache_item.texture_id)),
            textures,
            gpu_cache.get_address(&cache_item.uv_rect_handle),
        ))
    }
}

/// Either a single texture / user data for all segments,
/// or a list of one per segment.
enum SegmentDataKind {
    Shared(SegmentInstanceData),
    Instanced(SmallVec<[SegmentInstanceData; 8]>),
}

/// The parameters that are specific to a kind of brush,
/// used by the common method to add a brush to batches.
struct BrushBatchParameters {
    batch_kind: BrushBatchKind,
    prim_user_data: [i32; 4],
    segment_data: SegmentDataKind,
}

impl BrushBatchParameters {
    /// This brush instance has a list of per-segment
    /// instance data.
    fn instanced(
        batch_kind: BrushBatchKind,
        prim_user_data: [i32; 4],
        segment_data: SmallVec<[SegmentInstanceData; 8]>,
    ) -> Self {
        BrushBatchParameters {
            batch_kind,
            prim_user_data,
            segment_data: SegmentDataKind::Instanced(segment_data),
        }
    }

    /// This brush instance shares the per-segment data
    /// across all segments.
    fn shared(
        batch_kind: BrushBatchKind,
        textures: BatchTextures,
        prim_user_data: [i32; 4],
        segment_user_data: i32,
    ) -> Self {
        BrushBatchParameters {
            batch_kind,
            prim_user_data,
            segment_data: SegmentDataKind::Shared(
                SegmentInstanceData {
                    textures,
                    user_data: segment_user_data,
                }
            ),
        }
    }
}

impl PrimitiveInstance {
    pub fn is_cacheable(
        &self,
        data_stores: &DataStores,
        resource_cache: &ResourceCache,
    ) -> bool {
        let image_key = match self.kind {
            PrimitiveInstanceKind::Image { data_handle, .. } => {
                let image_data = &data_stores.image[data_handle].kind;
                image_data.key
            }
            PrimitiveInstanceKind::YuvImage { data_handle, .. } => {
                let yuv_image_data =
                    &data_stores.yuv_image[data_handle].kind;
                yuv_image_data.yuv_key[0]
            }
            PrimitiveInstanceKind::Picture { .. } |
            PrimitiveInstanceKind::TextRun { .. } |
            PrimitiveInstanceKind::LineDecoration { .. } |
            PrimitiveInstanceKind::NormalBorder { .. } |
            PrimitiveInstanceKind::ImageBorder { .. } |
            PrimitiveInstanceKind::Rectangle { .. } |
            PrimitiveInstanceKind::LinearGradient { .. } |
            PrimitiveInstanceKind::RadialGradient { .. } |
            PrimitiveInstanceKind::Clear { .. } => {
                return true;
            }
        };
        match resource_cache.get_image_properties(image_key) {
            Some(ImageProperties { external_image: Some(_), .. }) => {
                false
            }
            _ => true
        }
    }
}

impl RenderTaskGraph {
    fn resolve_surface(
        &self,
        task_id: RenderTaskId,
        gpu_cache: &GpuCache,
    ) -> (GpuCacheAddress, BatchTextures) {
        (
            self[task_id].get_texture_address(gpu_cache),
            BatchTextures::render_target_cache(),
        )
    }
}

pub fn resolve_image(
    request: ImageRequest,
    resource_cache: &ResourceCache,
    gpu_cache: &mut GpuCache,
    deferred_resolves: &mut Vec<DeferredResolve>,
) -> CacheItem {
    match resource_cache.get_image_properties(request.key) {
        Some(image_properties) => {
            // Check if an external image that needs to be resolved
            // by the render thread.
            match image_properties.external_image {
                Some(external_image) => {
                    // This is an external texture - we will add it to
                    // the deferred resolves list to be patched by
                    // the render thread...
                    let cache_handle = gpu_cache.push_deferred_per_frame_blocks(BLOCKS_PER_UV_RECT);
                    let cache_item = CacheItem {
                        texture_id: TextureSource::External(external_image),
                        uv_rect_handle: cache_handle,
                        uv_rect: DeviceIntRect::new(
                            DeviceIntPoint::zero(),
                            image_properties.descriptor.size,
                        ),
                        texture_layer: 0,
                    };

                    deferred_resolves.push(DeferredResolve {
                        image_properties,
                        address: gpu_cache.get_address(&cache_handle),
                        rendering: request.rendering,
                    });

                    cache_item
                }
                None => {
                    if let Ok(cache_item) = resource_cache.get_cached_image(request) {
                        cache_item
                    } else {
                        // There is no usable texture entry for the image key. Just return an invalid texture here.
                        CacheItem::invalid()
                    }
                }
            }
        }
        None => {
            CacheItem::invalid()
        }
    }
}

/// A list of clip instances to be drawn into a target.
#[derive(Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct ClipBatchList {
    /// Rectangle draws fill up the rectangles with rounded corners.
    pub slow_rectangles: Vec<ClipMaskInstance>,
    pub fast_rectangles: Vec<ClipMaskInstance>,
    /// Image draws apply the image masking.
    pub images: FastHashMap<TextureSource, Vec<ClipMaskInstance>>,
    pub box_shadows: FastHashMap<TextureSource, Vec<ClipMaskInstance>>,
}

impl ClipBatchList {
    fn new() -> Self {
        ClipBatchList {
            slow_rectangles: Vec::new(),
            fast_rectangles: Vec::new(),
            images: FastHashMap::default(),
            box_shadows: FastHashMap::default(),
        }
    }
}

/// Batcher managing draw calls into the clip mask (in the RT cache).
#[derive(Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct ClipBatcher {
    /// The first clip in each clip task. This will overwrite all pixels
    /// in the clip region, so we can skip doing a clear and write with
    /// blending disabled, which is a big performance win on Intel GPUs.
    pub primary_clips: ClipBatchList,
    /// Any subsequent clip masks (rare) for a clip task get drawn in
    /// a second pass with multiplicative blending enabled.
    pub secondary_clips: ClipBatchList,

    gpu_supports_fast_clears: bool,
}

impl ClipBatcher {
    pub fn new(
        gpu_supports_fast_clears: bool,
    ) -> Self {
        ClipBatcher {
            primary_clips: ClipBatchList::new(),
            secondary_clips: ClipBatchList::new(),
            gpu_supports_fast_clears,
        }
    }

    pub fn add_clip_region(
        &mut self,
        clip_data_address: GpuCacheAddress,
        local_pos: LayoutPoint,
        sub_rect: DeviceRect,
        task_origin: DevicePoint,
        screen_origin: DevicePoint,
        device_pixel_scale: f32,
    ) {
        let instance = ClipMaskInstance {
            clip_transform_id: TransformPaletteId::IDENTITY,
            prim_transform_id: TransformPaletteId::IDENTITY,
            clip_data_address,
            resource_address: GpuCacheAddress::INVALID,
            local_pos,
            tile_rect: LayoutRect::zero(),
            sub_rect,
            snap_offsets: SnapOffsets::empty(),
            task_origin,
            screen_origin,
            device_pixel_scale,
        };

        self.primary_clips.slow_rectangles.push(instance);
    }

    /// Where appropriate, draw a clip rectangle as a small series of tiles,
    /// instead of one large rectangle.
    fn add_tiled_clip_mask(
        &mut self,
        mask_screen_rect: DeviceIntRect,
        clip_rect_size: LayoutSize,
        clip_instance: &ClipNodeInstance,
        clip_scroll_tree: &ClipScrollTree,
        world_rect: &WorldRect,
        device_pixel_scale: DevicePixelScale,
        gpu_address: GpuCacheAddress,
        instance: &ClipMaskInstance,
        is_first_clip: bool,
    ) -> bool {
        // Only try to draw in tiles if the clip mark is big enough.
        if mask_screen_rect.area() < CLIP_RECTANGLE_AREA_THRESHOLD {
            return false;
        }

        let clip_spatial_node = &clip_scroll_tree
            .spatial_nodes[clip_instance.spatial_node_index.0 as usize];

        // Only support clips that are axis-aligned to the root coordinate space,
        // for now, to simplify the logic below. This handles the vast majority
        // of real world cases, but could be expanded in future if needed.
        if clip_spatial_node.coordinate_system_id != CoordinateSystemId::root() {
            return false;
        }

        // Get the world rect of the clip rectangle. If we can't transform it due
        // to the matrix, just fall back to drawing the entire clip mask.
        let local_clip_rect = LayoutRect::new(
            clip_instance.local_pos,
            clip_rect_size,
        );
        let transform = clip_scroll_tree.get_world_transform(
            clip_instance.spatial_node_index,
        );
        let world_clip_rect = match project_rect(
            &transform.into_transform(),
            &local_clip_rect,
            world_rect,
        ) {
            Some(rect) => rect,
            None => return false,
        };

        // Work out how many tiles to draw this clip mask in, stretched across the
        // device rect of the primitive clip mask.
        let world_device_rect = world_clip_rect * device_pixel_scale;
        let x_tiles = (mask_screen_rect.size.width + CLIP_RECTANGLE_TILE_SIZE-1) / CLIP_RECTANGLE_TILE_SIZE;
        let y_tiles = (mask_screen_rect.size.height + CLIP_RECTANGLE_TILE_SIZE-1) / CLIP_RECTANGLE_TILE_SIZE;

        // Because we only run this code path for axis-aligned rects (the root coord system check above),
        // and only for rectangles (not rounded etc), the world_device_rect is not conservative - we know
        // that there is no inner_rect, and the world_device_rect should be the real, axis-aligned clip rect.
        let mask_origin = mask_screen_rect.origin.to_f32().to_vector();
        let clip_list = self.get_batch_list(is_first_clip);

        for y in 0 .. y_tiles {
            for x in 0 .. x_tiles {
                let p0 = DeviceIntPoint::new(
                    x * CLIP_RECTANGLE_TILE_SIZE,
                    y * CLIP_RECTANGLE_TILE_SIZE,
                );
                let p1 = DeviceIntPoint::new(
                    (p0.x + CLIP_RECTANGLE_TILE_SIZE).min(mask_screen_rect.size.width),
                    (p0.y + CLIP_RECTANGLE_TILE_SIZE).min(mask_screen_rect.size.height),
                );
                let normalized_sub_rect = DeviceIntRect::new(
                    p0,
                    DeviceIntSize::new(
                        p1.x - p0.x,
                        p1.y - p0.y,
                    ),
                ).to_f32();
                let world_sub_rect = normalized_sub_rect.translate(&mask_origin);

                // If the clip rect completely contains this tile rect, then drawing
                // these pixels would be redundant - since this clip can't possibly
                // affect the pixels in this tile, skip them!
                if !world_device_rect.contains_rect(&world_sub_rect) {
                    clip_list.slow_rectangles.push(ClipMaskInstance {
                        clip_data_address: gpu_address,
                        sub_rect: normalized_sub_rect,
                        ..*instance
                    });
                }
            }
        }

        true
    }

    /// Retrieve the correct clip batch list to append to, depending
    /// on whether this is the first clip mask for a clip task.
    fn get_batch_list(
        &mut self,
        is_first_clip: bool,
    ) -> &mut ClipBatchList {
        if is_first_clip && !self.gpu_supports_fast_clears {
            &mut self.primary_clips
        } else {
            &mut self.secondary_clips
        }
    }

    pub fn add(
        &mut self,
        clip_node_range: ClipNodeRange,
        root_spatial_node_index: SpatialNodeIndex,
        resource_cache: &ResourceCache,
        gpu_cache: &GpuCache,
        clip_store: &ClipStore,
        clip_scroll_tree: &ClipScrollTree,
        transforms: &mut TransformPalette,
        clip_data_store: &ClipDataStore,
        actual_rect: DeviceIntRect,
        world_rect: &WorldRect,
        device_pixel_scale: DevicePixelScale,
        snap_offsets: SnapOffsets,
        task_origin: DevicePoint,
        screen_origin: DevicePoint,
    ) {
        let mut is_first_clip = true;

        for i in 0 .. clip_node_range.count {
            let clip_instance = clip_store.get_instance_from_range(&clip_node_range, i);
            let clip_node = &clip_data_store[clip_instance.handle];

            let clip_transform_id = transforms.get_id(
                clip_instance.spatial_node_index,
                ROOT_SPATIAL_NODE_INDEX,
                clip_scroll_tree,
            );

            let prim_transform_id = transforms.get_id(
                root_spatial_node_index,
                ROOT_SPATIAL_NODE_INDEX,
                clip_scroll_tree,
            );

            let instance = ClipMaskInstance {
                clip_transform_id,
                prim_transform_id,
                clip_data_address: GpuCacheAddress::INVALID,
                resource_address: GpuCacheAddress::INVALID,
                local_pos: clip_instance.local_pos,
                tile_rect: LayoutRect::zero(),
                sub_rect: DeviceRect::new(
                    DevicePoint::zero(),
                    actual_rect.size.to_f32(),
                ),
                snap_offsets,
                task_origin,
                screen_origin,
                device_pixel_scale: device_pixel_scale.0,
            };

            let added_clip = match clip_node.item {
                ClipItem::Image { image, size, .. } => {
                    let request = ImageRequest {
                        key: image,
                        rendering: ImageRendering::Auto,
                        tile: None,
                    };

                    let clip_data_address =
                        gpu_cache.get_address(&clip_node.gpu_cache_handle);

                    let mut add_image = |request: ImageRequest, local_tile_rect: LayoutRect| {
                        let cache_item = match resource_cache.get_cached_image(request) {
                            Ok(item) => item,
                            Err(..) => {
                                warn!("Warnings: skip a image mask");
                                debug!("request: {:?}", request);
                                return;
                            }
                        };

                        self.get_batch_list(is_first_clip)
                            .images
                            .entry(cache_item.texture_id)
                            .or_insert_with(Vec::new)
                            .push(ClipMaskInstance {
                                clip_data_address,
                                resource_address: gpu_cache.get_address(&cache_item.uv_rect_handle),
                                tile_rect: local_tile_rect,
                                ..instance
                            });
                    };

                    match clip_instance.visible_tiles {
                        Some(ref tiles) => {
                            for tile in tiles {
                                add_image(
                                    request.with_tile(tile.tile_offset),
                                    tile.tile_rect,
                                )
                            }
                        }
                        None => {
                            let mask_rect = LayoutRect::new(clip_instance.local_pos, size);
                            add_image(request, mask_rect)
                        }
                    }

                    true
                }
                ClipItem::BoxShadow(ref info) => {
                    let gpu_address =
                        gpu_cache.get_address(&clip_node.gpu_cache_handle);
                    let rt_handle = info
                        .cache_handle
                        .as_ref()
                        .expect("bug: render task handle not allocated");
                    let rt_cache_entry = resource_cache
                        .get_cached_render_task(rt_handle);
                    let cache_item = resource_cache
                        .get_texture_cache_item(&rt_cache_entry.handle);
                    debug_assert_ne!(cache_item.texture_id, TextureSource::Invalid);

                    self.get_batch_list(is_first_clip)
                        .box_shadows
                        .entry(cache_item.texture_id)
                        .or_insert_with(Vec::new)
                        .push(ClipMaskInstance {
                            clip_data_address: gpu_address,
                            resource_address: gpu_cache.get_address(&cache_item.uv_rect_handle),
                            ..instance
                        });

                    true
                }
                ClipItem::Rectangle(_, ClipMode::ClipOut) => {
                    let gpu_address =
                        gpu_cache.get_address(&clip_node.gpu_cache_handle);
                    self.get_batch_list(is_first_clip)
                        .slow_rectangles
                        .push(ClipMaskInstance {
                            clip_data_address: gpu_address,
                            ..instance
                        });

                    true
                }
                ClipItem::Rectangle(clip_rect_size, ClipMode::Clip) => {
                    if clip_instance.flags.contains(ClipNodeFlags::SAME_COORD_SYSTEM) {
                        false
                    } else {
                        let gpu_address = gpu_cache.get_address(&clip_node.gpu_cache_handle);

                        if !self.add_tiled_clip_mask(
                            actual_rect,
                            clip_rect_size,
                            clip_instance,
                            clip_scroll_tree,
                            world_rect,
                            device_pixel_scale,
                            gpu_address,
                            &instance,
                            is_first_clip,
                        ) {
                            self.get_batch_list(is_first_clip)
                                .slow_rectangles
                                .push(ClipMaskInstance {
                                    clip_data_address: gpu_address,
                                    ..instance
                                });
                        }

                        true
                    }
                }
                ClipItem::RoundedRectangle(..) => {
                    let gpu_address =
                        gpu_cache.get_address(&clip_node.gpu_cache_handle);
                    let batch_list = self.get_batch_list(is_first_clip);
                    let instance = ClipMaskInstance {
                        clip_data_address: gpu_address,
                        ..instance
                    };
                    if clip_instance.flags.contains(ClipNodeFlags::USE_FAST_PATH) {
                        batch_list.fast_rectangles.push(instance);
                    } else {
                        batch_list.slow_rectangles.push(instance);
                    }

                    true
                }
            };

            is_first_clip &= !added_clip;
        }
    }
}

fn get_buffer_kind(texture: TextureSource) -> ImageBufferKind {
    match texture {
        TextureSource::External(ext_image) => {
            match ext_image.image_type {
                ExternalImageType::TextureHandle(target) => {
                    target.into()
                }
                ExternalImageType::Buffer => {
                    // The ExternalImageType::Buffer should be handled by resource_cache.
                    // It should go through the non-external case.
                    panic!("Unexpected non-texture handle type");
                }
            }
        }
        _ => ImageBufferKind::Texture2DArray,
    }
}

fn get_shader_opacity(opacity: f32) -> i32 {
    (opacity * 65535.0).round() as i32
}

impl<'a, 'rc> RenderTargetContext<'a, 'rc> {
    /// Retrieve the GPU task address for a given clip task instance.
    /// Returns None if the segment was completely clipped out.
    /// Returns Some(OPAQUE_TASK_ADDRESS) if no clip mask is needed.
    /// Returns Some(task_address) if there was a valid clip mask.
    fn get_clip_task_address(
        &self,
        clip_task_index: ClipTaskIndex,
        offset: i32,
        render_tasks: &RenderTaskGraph,
    ) -> Option<RenderTaskAddress> {
        let address = match self.scratch.clip_mask_instances[clip_task_index.0 as usize + offset as usize] {
            ClipMaskKind::Mask(task_id) => {
                render_tasks.get_task_address(task_id)
            }
            ClipMaskKind::None => {
                OPAQUE_TASK_ADDRESS
            }
            ClipMaskKind::Clipped => {
                return None;
            }
        };

        Some(address)
    }

    /// Helper function to get the clip task address for a
    /// non-segmented primitive.
    fn get_prim_clip_task_address(
        &self,
        clip_task_index: ClipTaskIndex,
        render_tasks: &RenderTaskGraph,
    ) -> Option<RenderTaskAddress> {
        self.get_clip_task_address(
            clip_task_index,
            0,
            render_tasks,
        )
    }
}
