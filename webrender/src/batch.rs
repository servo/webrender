/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{AlphaType, ClipMode, DeviceIntRect, DeviceIntSize, DeviceIntPoint};
use api::{DeviceUintRect, DeviceUintPoint, ExternalImageType, FilterOp, ImageRendering};
use api::{YuvColorSpace, YuvFormat, WorldPixel};
use clip::{ClipNodeFlags, ClipNodeRange, ClipItem, ClipStore};
use euclid::vec3;
use glyph_rasterizer::GlyphFormat;
use gpu_cache::{GpuCache, GpuCacheHandle, GpuCacheAddress};
use gpu_types::{BrushFlags, BrushInstance, PrimitiveHeaders};
use gpu_types::{ClipMaskInstance, SplitCompositeInstance};
use gpu_types::{PrimitiveInstance, RasterizationSpace, GlyphInstance};
use gpu_types::{PrimitiveHeader, PrimitiveHeaderIndex, TransformPaletteId, TransformPalette};
use internal_types::{FastHashMap, SavedTargetIndex, SourceTexture};
use picture::{PictureCompositeMode, PicturePrimitive, PictureSurface};
use plane_split::{BspSplitter, Clipper, Polygon, Splitter};
use prim_store::{BrushKind, BrushPrimitive, BrushSegmentTaskId, DeferredResolve};
use prim_store::{EdgeAaSegmentMask, ImageSource, PrimitiveIndex};
use prim_store::{PrimitiveMetadata, PrimitiveRun, VisibleGradientTile};
use prim_store::{BorderSource, Primitive, PrimitiveDetails};
use render_task::{RenderTaskAddress, RenderTaskId, RenderTaskKind, RenderTaskTree};
use renderer::{BlendMode, ImageBufferKind, ShaderColorMode};
use renderer::BLOCKS_PER_UV_RECT;
use resource_cache::{CacheItem, GlyphFetchResult, ImageRequest, ResourceCache};
use scene::FilterOpHelpers;
use std::{f32, i32};
use tiling::{RenderTargetContext};
use util::{TransformedRectKind};

// Special sentinel value recognized by the shader. It is considered to be
// a dummy task that doesn't mask out anything.
const OPAQUE_TASK_ADDRESS: RenderTaskAddress = RenderTaskAddress(0x7fff);

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
    YuvImage(ImageBufferKind, YuvFormat, YuvColorSpace),
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
    pub colors: [SourceTexture; 3],
}

impl BatchTextures {
    pub fn no_texture() -> Self {
        BatchTextures {
            colors: [SourceTexture::Invalid; 3],
        }
    }

    pub fn render_target_cache() -> Self {
        BatchTextures {
            colors: [
                SourceTexture::CacheRGBA8,
                SourceTexture::CacheA8,
                SourceTexture::Invalid,
            ],
        }
    }

    pub fn color(texture: SourceTexture) -> Self {
        BatchTextures {
            colors: [texture, texture, SourceTexture::Invalid],
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
fn textures_compatible(t1: SourceTexture, t2: SourceTexture) -> bool {
    t1 == SourceTexture::Invalid || t2 == SourceTexture::Invalid || t1 == t2
}

pub struct AlphaBatchList {
    pub batches: Vec<PrimitiveBatch>,
    pub item_rects: Vec<Vec<DeviceIntRect>>,
}

impl AlphaBatchList {
    fn new() -> Self {
        AlphaBatchList {
            batches: Vec::new(),
            item_rects: Vec::new(),
        }
    }

    pub fn get_suitable_batch(
        &mut self,
        key: BatchKey,
        task_relative_bounding_rect: &DeviceIntRect,
    ) -> &mut Vec<PrimitiveInstance> {
        let mut selected_batch_index = None;

        match key.blend_mode {
            BlendMode::SubpixelWithBgColor => {
                'outer_multipass: for (batch_index, batch) in self.batches.iter().enumerate().rev().take(10) {
                    // Some subpixel batches are drawn in two passes. Because of this, we need
                    // to check for overlaps with every batch (which is a bit different
                    // than the normal batching below).
                    for item_rect in &self.item_rects[batch_index] {
                        if item_rect.intersects(task_relative_bounding_rect) {
                            break 'outer_multipass;
                        }
                    }

                    if batch.key.is_compatible_with(&key) {
                        selected_batch_index = Some(batch_index);
                        break;
                    }
                }
            }
            _ => {
                'outer_default: for (batch_index, batch) in self.batches.iter().enumerate().rev().take(10) {
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
                        if item_rect.intersects(task_relative_bounding_rect) {
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

        let selected_batch_index = selected_batch_index.unwrap();
        self.item_rects[selected_batch_index].push(*task_relative_bounding_rect);
        &mut self.batches[selected_batch_index].instances
    }
}

pub struct OpaqueBatchList {
    pub pixel_area_threshold_for_new_batch: f32,
    pub batches: Vec<PrimitiveBatch>,
}

impl OpaqueBatchList {
    fn new(pixel_area_threshold_for_new_batch: f32) -> Self {
        OpaqueBatchList {
            batches: Vec::new(),
            pixel_area_threshold_for_new_batch,
        }
    }

    pub fn get_suitable_batch(
        &mut self,
        key: BatchKey,
        task_relative_bounding_rect: &DeviceIntRect
    ) -> &mut Vec<PrimitiveInstance> {
        let mut selected_batch_index = None;
        let item_area = task_relative_bounding_rect.size.to_f32().area();

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
            for (batch_index, batch) in self.batches.iter().enumerate().rev().take(10) {
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

        let batch = &mut self.batches[selected_batch_index.unwrap()];

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

pub struct BatchList {
    pub alpha_batch_list: AlphaBatchList,
    pub opaque_batch_list: OpaqueBatchList,
    pub combined_bounding_rect: DeviceIntRect,
}

impl BatchList {
    pub fn new(screen_size: DeviceIntSize) -> Self {
        // The threshold for creating a new batch is
        // one quarter the screen size.
        let batch_area_threshold = (screen_size.width * screen_size.height) as f32 / 4.0;

        BatchList {
            alpha_batch_list: AlphaBatchList::new(),
            opaque_batch_list: OpaqueBatchList::new(batch_area_threshold),
            combined_bounding_rect: DeviceIntRect::zero(),
        }
    }

    fn add_bounding_rect(
        &mut self,
        task_relative_bounding_rect: &DeviceIntRect,
    ) {
        self.combined_bounding_rect = self.combined_bounding_rect.union(task_relative_bounding_rect);
    }

    pub fn get_suitable_batch(
        &mut self,
        key: BatchKey,
        task_relative_bounding_rect: &DeviceIntRect,
    ) -> &mut Vec<PrimitiveInstance> {
        self.add_bounding_rect(task_relative_bounding_rect);

        match key.blend_mode {
            BlendMode::None => {
                self.opaque_batch_list
                    .get_suitable_batch(key, task_relative_bounding_rect)
            }
            BlendMode::Alpha |
            BlendMode::PremultipliedAlpha |
            BlendMode::PremultipliedDestOut |
            BlendMode::SubpixelConstantTextColor(..) |
            BlendMode::SubpixelWithBgColor |
            BlendMode::SubpixelDualSource => {
                self.alpha_batch_list
                    .get_suitable_batch(key, task_relative_bounding_rect)
            }
        }
    }

    // Remove any batches that were added but didn't get any instances
    // added to them.
    fn remove_unused_batches(&mut self) {
        if self.opaque_batch_list
               .batches
               .last()
               .map_or(false, |batch| batch.instances.is_empty()) {
            self.opaque_batch_list.batches.pop().unwrap();
        }

        if self.alpha_batch_list
               .batches
               .last()
               .map_or(false, |batch| batch.instances.is_empty()) {
            self.alpha_batch_list.batches.pop().unwrap();
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
    pub instances: Vec<PrimitiveInstance>,
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
    pub target_rect: Option<DeviceIntRect>,
}

impl AlphaBatchContainer {
    pub fn new(target_rect: Option<DeviceIntRect>) -> AlphaBatchContainer {
        AlphaBatchContainer {
            opaque_batches: Vec::new(),
            alpha_batches: Vec::new(),
            target_rect,
        }
    }

    fn merge(&mut self, builder: AlphaBatchBuilder) {
        for other_batch in builder.batch_list.opaque_batch_list.batches {
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

        for other_batch in builder.batch_list.alpha_batch_list.batches {
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

/// Encapsulates the logic of building batches for items that are blended.
pub struct AlphaBatchBuilder {
    pub batch_list: BatchList,
    glyph_fetch_buffer: Vec<GlyphFetchResult>,
    target_rect: DeviceIntRect,
}

impl AlphaBatchBuilder {
    pub fn new(
        screen_size: DeviceIntSize,
        target_rect: DeviceIntRect,
    ) -> Self {
        AlphaBatchBuilder {
            batch_list: BatchList::new(screen_size),
            glyph_fetch_buffer: Vec::new(),
            target_rect,
        }
    }

    pub fn build(mut self, merged_batches: &mut AlphaBatchContainer) -> Option<AlphaBatchContainer> {
        self.batch_list.finalize();

        let task_relative_target_rect = DeviceIntRect::new(
            DeviceIntPoint::zero(),
            self.target_rect.size,
        );

        let can_merge = task_relative_target_rect.contains_rect(&self.batch_list.combined_bounding_rect);

        if can_merge {
            merged_batches.merge(self);
            None
        } else {
            Some(AlphaBatchContainer {
                alpha_batches: self.batch_list.alpha_batch_list.batches,
                opaque_batches: self.batch_list.opaque_batch_list.batches,
                target_rect: Some(self.target_rect),
            })
        }
    }

    pub fn add_pic_to_batch(
        &mut self,
        pic: &PicturePrimitive,
        task_id: RenderTaskId,
        ctx: &RenderTargetContext,
        gpu_cache: &mut GpuCache,
        render_tasks: &RenderTaskTree,
        deferred_resolves: &mut Vec<DeferredResolve>,
        prim_headers: &mut PrimitiveHeaders,
    ) {
        let task_address = render_tasks.get_task_address(task_id);

        let task = &render_tasks[task_id];
        let content_origin = match task.kind {
            RenderTaskKind::Picture(ref pic_task) => {
                pic_task.content_origin
            }
            _ => {
                panic!("todo: tidy this up");
            }
        };

        // Even though most of the time a splitter isn't used or needed,
        // they are cheap to construct so we will always pass one down.
        let mut splitter = BspSplitter::new();

        // Add each run in this picture to the batch.
        for run in &pic.runs {
            let transform_id = ctx
                .transforms
                .get_id(run.spatial_node_index);
            self.add_run_to_batch(
                run,
                transform_id,
                ctx,
                gpu_cache,
                render_tasks,
                task_id,
                task_address,
                deferred_resolves,
                &mut splitter,
                content_origin,
                prim_headers,
            );
        }

        // Flush the accumulated plane splits onto the task tree.
        // Z axis is directed at the screen, `sort` is ascending, and we need back-to-front order.
        for poly in splitter.sort(vec3(0.0, 0.0, 1.0)) {
            let prim_index = PrimitiveIndex(poly.anchor);
            if cfg!(debug_assertions) && ctx.prim_store.chase_id == Some(prim_index) {
                println!("\t\tsplit polygon {:?}", poly.points);
            }
            debug!("process sorted poly {:?} {:?}", prim_index, poly.points);
            let pp = &poly.points;
            let gpu_blocks = [
                [pp[0].x as f32, pp[0].y as f32, pp[0].z as f32, pp[1].x as f32].into(),
                [pp[1].y as f32, pp[1].z as f32, pp[2].x as f32, pp[2].y as f32].into(),
                [pp[2].z as f32, pp[3].x as f32, pp[3].y as f32, pp[3].z as f32].into(),
            ];
            let gpu_handle = gpu_cache.push_per_frame_blocks(&gpu_blocks);
            let key = BatchKey::new(
                BatchKind::SplitComposite,
                BlendMode::PremultipliedAlpha,
                BatchTextures::no_texture(),
            );
            let pic_metadata = &ctx.prim_store.primitives[prim_index.0].metadata;
            let pic = ctx.prim_store.get_pic(prim_index);
            let batch = self.batch_list.get_suitable_batch(key, &pic_metadata.screen_rect.as_ref().expect("bug").clipped);

            let source_task_id = pic
                .surface
                .as_ref()
                .expect("BUG: unexpected surface in splitting")
                .resolve_render_task_id();
            let source_task_address = render_tasks.get_task_address(source_task_id);
            let gpu_address = gpu_cache.get_address(&gpu_handle);

            let instance = SplitCompositeInstance::new(
                task_address,
                source_task_address,
                gpu_address,
                prim_headers.z_generator.next(),
            );

            batch.push(PrimitiveInstance::from(instance));
        }
    }

    // Helper to add an entire primitive run to a batch list.
    // TODO(gw): Restructure this so the param list isn't quite
    //           so daunting!
    fn add_run_to_batch(
        &mut self,
        run: &PrimitiveRun,
        transform_id: TransformPaletteId,
        ctx: &RenderTargetContext,
        gpu_cache: &mut GpuCache,
        render_tasks: &RenderTaskTree,
        task_id: RenderTaskId,
        task_address: RenderTaskAddress,
        deferred_resolves: &mut Vec<DeferredResolve>,
        splitter: &mut BspSplitter<f64, WorldPixel>,
        content_origin: DeviceIntPoint,
        prim_headers: &mut PrimitiveHeaders,
    ) {
        for i in 0 .. run.count {
            let prim_index = PrimitiveIndex(run.base_prim_index.0 + i);
            let metadata = &ctx.prim_store.primitives[prim_index.0].metadata;

            if metadata.screen_rect.is_some() {
                self.add_prim_to_batch(
                    transform_id,
                    prim_index,
                    ctx,
                    gpu_cache,
                    render_tasks,
                    task_id,
                    task_address,
                    deferred_resolves,
                    splitter,
                    content_origin,
                    prim_headers,
                );
            }
        }
    }

    // Adds a primitive to a batch.
    // It can recursively call itself in some situations, for
    // example if it encounters a picture where the items
    // in that picture are being drawn into the same target.
    fn add_prim_to_batch(
        &mut self,
        transform_id: TransformPaletteId,
        prim_index: PrimitiveIndex,
        ctx: &RenderTargetContext,
        gpu_cache: &mut GpuCache,
        render_tasks: &RenderTaskTree,
        task_id: RenderTaskId,
        task_address: RenderTaskAddress,
        deferred_resolves: &mut Vec<DeferredResolve>,
        splitter: &mut BspSplitter<f64, WorldPixel>,
        content_origin: DeviceIntPoint,
        prim_headers: &mut PrimitiveHeaders,
    ) {
        let prim = &ctx.prim_store.primitives[prim_index.0];
        let prim_metadata = &prim.metadata;
        #[cfg(debug_assertions)] //TODO: why is this needed?
        debug_assert_eq!(prim_metadata.prepared_frame_id, render_tasks.frame_id());

        // TODO(gw): Calculating this for every primitive is a bit
        //           wasteful. We should probably cache this in
        //           the scroll node...
        let transform_kind = transform_id.transform_kind();

        let screen_rect = prim_metadata.screen_rect.expect("bug");
        let task_relative_bounding_rect = DeviceIntRect::new(
            DeviceIntPoint::new(
                screen_rect.unclipped.origin.x - content_origin.x,
                screen_rect.unclipped.origin.y - content_origin.y,
            ),
            screen_rect.unclipped.size,
        );

        // If the primitive is internally decomposed into multiple sub-primitives we may not
        // use some of the per-primitive data typically stored in PrimitiveMetadata and get
        // it from each sub-primitive instead.
        let is_multiple_primitives = match prim.details {
            PrimitiveDetails::Brush(ref brush) => {
                match brush.kind {
                    BrushKind::Image { ref visible_tiles, .. } => !visible_tiles.is_empty(),
                    BrushKind::LinearGradient { ref visible_tiles, .. } => !visible_tiles.is_empty(),
                    BrushKind::RadialGradient { ref visible_tiles, .. } => !visible_tiles.is_empty(),
                    _ => false,
                }
            }
            _ => false,
        };

        let prim_cache_address = if is_multiple_primitives {
            GpuCacheAddress::invalid()
        } else {
            gpu_cache.get_address(&prim_metadata.gpu_location)
        };

        let clip_task_address = prim_metadata
            .clip_task_id
            .map_or(OPAQUE_TASK_ADDRESS, |id| render_tasks.get_task_address(id));

        let specified_blend_mode = prim.get_blend_mode();

        let non_segmented_blend_mode = if !prim_metadata.opacity.is_opaque ||
            prim_metadata.clip_task_id.is_some() ||
            transform_kind == TransformedRectKind::Complex {
            specified_blend_mode
        } else {
            BlendMode::None
        };

        let prim_header = PrimitiveHeader {
            local_rect: prim_metadata.local_rect,
            local_clip_rect: prim_metadata.combined_local_clip_rect,
            task_address,
            specific_prim_address: prim_cache_address,
            clip_task_address,
            transform_id,
        };

        if cfg!(debug_assertions) && ctx.prim_store.chase_id == Some(prim_index) {
            println!("\ttask target {:?}", self.target_rect);
            println!("\t{:?}", prim_header);
        }

        match prim.details {
            PrimitiveDetails::Brush(ref brush) => {
                match brush.kind {
                    BrushKind::Picture(ref picture) => {
                        // If this picture is participating in a 3D rendering context,
                        // then don't add it to any batches here. Instead, create a polygon
                        // for it and add it to the current plane splitter.
                        if picture.is_in_3d_context {
                            // Push into parent plane splitter.
                            debug_assert!(picture.surface.is_some());
                            let transform = &ctx.transforms
                                .get_transform(picture.reference_frame_index);

                            match transform.transform_kind {
                                TransformedRectKind::AxisAligned => {
                                    let polygon = Polygon::from_transformed_rect(
                                        picture.real_local_rect.cast(),
                                        transform.m.cast(),
                                        prim_index.0,
                                    ).unwrap();
                                    splitter.add(polygon);
                                }
                                TransformedRectKind::Complex => {
                                    let mut clipper = Clipper::new();
                                    let bounds = (screen_rect.clipped.to_f32() / ctx.device_pixel_scale).to_f64();
                                    let matrix = transform.m.cast();
                                    let results = clipper.clip_transformed(
                                        Polygon::from_rect(picture.real_local_rect.cast(), prim_index.0),
                                        &matrix,
                                        Some(bounds),
                                    );
                                    for poly in results {
                                        splitter.add(poly);
                                    }
                                }
                            }

                            return;
                        }

                        let add_to_parent_pic = match picture.composite_mode {
                            Some(PictureCompositeMode::Filter(filter)) => {
                                assert!(filter.is_visible());
                                match filter {
                                    FilterOp::Blur(..) => {
                                        match picture.surface {
                                            Some(ref surface) => {
                                                let kind = BatchKind::Brush(
                                                    BrushBatchKind::Image(ImageBufferKind::Texture2DArray)
                                                );
                                                let (uv_rect_address, textures) = surface
                                                    .resolve(
                                                        render_tasks,
                                                        ctx.resource_cache,
                                                        gpu_cache,
                                                    );
                                                let key = BatchKey::new(
                                                    kind,
                                                    non_segmented_blend_mode,
                                                    textures,
                                                );
                                                let batch = self.batch_list.get_suitable_batch(key, &task_relative_bounding_rect);
                                                let prim_header_index = prim_headers.push(&prim_header, [
                                                    uv_rect_address.as_int(),
                                                    (ShaderColorMode::ColorBitmap as i32) << 16 |
                                                    RasterizationSpace::Screen as i32,
                                                    0,
                                                ]);

                                                let instance = BrushInstance {
                                                    prim_header_index,
                                                    segment_index: 0,
                                                    edge_flags: EdgeAaSegmentMask::empty(),
                                                    brush_flags: BrushFlags::empty(),
                                                    clip_task_address,
                                                };
                                                batch.push(PrimitiveInstance::from(instance));
                                                false
                                            }
                                            None => {
                                                true
                                            }
                                        }
                                    }
                                    FilterOp::DropShadow(offset, ..) => {
                                        // Draw an instance of the shadow first, following by the content.

                                        // Both the shadow and the content get drawn as a brush image.
                                        if let Some(ref surface) = picture.surface {
                                            let kind = BatchKind::Brush(
                                                BrushBatchKind::Image(ImageBufferKind::Texture2DArray),
                                            );

                                            // Gets the saved render task ID of the content, which is
                                            // deeper in the render task tree than the direct child.
                                            let secondary_id = picture.secondary_render_task_id.expect("no secondary!?");
                                            let saved_index = render_tasks[secondary_id].saved_index.expect("no saved index!?");
                                            debug_assert_ne!(saved_index, SavedTargetIndex::PENDING);

                                            // Build BatchTextures for shadow/content
                                            let shadow_textures = BatchTextures::render_target_cache();
                                            let content_textures = BatchTextures {
                                                colors: [
                                                    SourceTexture::RenderTaskCache(saved_index),
                                                    SourceTexture::Invalid,
                                                    SourceTexture::Invalid,
                                                ],
                                            };

                                            // Build batch keys for shadow/content
                                            let shadow_key = BatchKey::new(kind, non_segmented_blend_mode, shadow_textures);
                                            let content_key = BatchKey::new(kind, non_segmented_blend_mode, content_textures);

                                            // Retrieve the UV rect addresses for shadow/content.
                                            let cache_task_id = surface.resolve_render_task_id();
                                            let shadow_uv_rect_address = render_tasks[cache_task_id]
                                                .get_texture_address(gpu_cache)
                                                .as_int();
                                            let content_uv_rect_address = render_tasks[secondary_id]
                                                .get_texture_address(gpu_cache)
                                                .as_int();

                                            // Get the GPU cache address of the extra data handle.
                                            let shadow_prim_address = gpu_cache.get_address(&picture.extra_gpu_data_handle);

                                            let content_prim_header_index = prim_headers.push(&prim_header, [
                                                content_uv_rect_address,
                                                (ShaderColorMode::ColorBitmap as i32) << 16 |
                                                RasterizationSpace::Screen as i32,
                                                0,
                                            ]);

                                            let shadow_rect = prim_metadata.local_rect.translate(&offset);
                                            let shadow_clip_rect = prim_metadata.local_clip_rect.translate(&offset);

                                            let shadow_prim_header = PrimitiveHeader {
                                                local_rect: shadow_rect,
                                                local_clip_rect: shadow_clip_rect,
                                                specific_prim_address: shadow_prim_address,
                                                ..prim_header
                                            };

                                            let shadow_prim_header_index = prim_headers.push(&shadow_prim_header, [
                                                shadow_uv_rect_address,
                                                (ShaderColorMode::Alpha as i32) << 16 |
                                                RasterizationSpace::Screen as i32,
                                                0,
                                            ]);

                                            let shadow_instance = BrushInstance {
                                                prim_header_index: shadow_prim_header_index,
                                                clip_task_address,
                                                segment_index: 0,
                                                edge_flags: EdgeAaSegmentMask::empty(),
                                                brush_flags: BrushFlags::empty(),
                                            };

                                            let content_instance = BrushInstance {
                                                prim_header_index: content_prim_header_index,
                                                clip_task_address,
                                                segment_index: 0,
                                                edge_flags: EdgeAaSegmentMask::empty(),
                                                brush_flags: BrushFlags::empty(),
                                            };

                                            self.batch_list
                                                .get_suitable_batch(shadow_key, &task_relative_bounding_rect)
                                                .push(PrimitiveInstance::from(shadow_instance));

                                            self.batch_list
                                                .get_suitable_batch(content_key, &task_relative_bounding_rect)
                                                .push(PrimitiveInstance::from(content_instance));
                                        }

                                        false
                                    }
                                    _ => {
                                        match picture.surface {
                                            Some(ref surface) => {
                                                let key = BatchKey::new(
                                                    BatchKind::Brush(BrushBatchKind::Blend),
                                                    BlendMode::PremultipliedAlpha,
                                                    BatchTextures::render_target_cache(),
                                                );

                                                let filter_mode = match filter {
                                                    FilterOp::Blur(..) => 0,
                                                    FilterOp::Contrast(..) => 1,
                                                    FilterOp::Grayscale(..) => 2,
                                                    FilterOp::HueRotate(..) => 3,
                                                    FilterOp::Invert(..) => 4,
                                                    FilterOp::Saturate(..) => 5,
                                                    FilterOp::Sepia(..) => 6,
                                                    FilterOp::Brightness(..) => 7,
                                                    FilterOp::Opacity(..) => 8,
                                                    FilterOp::DropShadow(..) => 9,
                                                    FilterOp::ColorMatrix(..) => 10,
                                                };

                                                let user_data = match filter {
                                                    FilterOp::Contrast(amount) |
                                                    FilterOp::Grayscale(amount) |
                                                    FilterOp::Invert(amount) |
                                                    FilterOp::Saturate(amount) |
                                                    FilterOp::Sepia(amount) |
                                                    FilterOp::Brightness(amount) |
                                                    FilterOp::Opacity(_, amount) => {
                                                        (amount * 65536.0) as i32
                                                    }
                                                    FilterOp::HueRotate(angle) => {
                                                        (0.01745329251 * angle * 65536.0) as i32
                                                    }
                                                    // Go through different paths
                                                    FilterOp::Blur(..) |
                                                    FilterOp::DropShadow(..) => {
                                                        unreachable!();
                                                    }
                                                    FilterOp::ColorMatrix(_) => {
                                                        picture.extra_gpu_data_handle.as_int(gpu_cache)
                                                    }
                                                };

                                                let cache_task_id = surface.resolve_render_task_id();
                                                let cache_task_address = render_tasks.get_task_address(cache_task_id);
                                                let prim_header_index = prim_headers.push(&prim_header, [
                                                    cache_task_address.0 as i32,
                                                    filter_mode,
                                                    user_data,
                                                ]);

                                                let instance = BrushInstance {
                                                    prim_header_index,
                                                    clip_task_address,
                                                    segment_index: 0,
                                                    edge_flags: EdgeAaSegmentMask::empty(),
                                                    brush_flags: BrushFlags::empty(),
                                                };

                                                let batch = self.batch_list.get_suitable_batch(key, &task_relative_bounding_rect);
                                                batch.push(PrimitiveInstance::from(instance));
                                                false
                                            }
                                            None => {
                                                true
                                            }
                                        }
                                    }
                                }
                            }
                            Some(PictureCompositeMode::MixBlend(mode)) => {
                                let cache_task_id = picture
                                    .surface
                                    .as_ref()
                                    .expect("bug: no surface allocated")
                                    .resolve_render_task_id();
                                let backdrop_id = picture.secondary_render_task_id.expect("no backdrop!?");

                                let key = BatchKey::new(
                                    BatchKind::Brush(
                                        BrushBatchKind::MixBlend {
                                            task_id,
                                            source_id: cache_task_id,
                                            backdrop_id,
                                        },
                                    ),
                                    BlendMode::PremultipliedAlpha,
                                    BatchTextures::no_texture(),
                                );
                                let batch = self.batch_list.get_suitable_batch(key, &task_relative_bounding_rect);
                                let backdrop_task_address = render_tasks.get_task_address(backdrop_id);
                                let source_task_address = render_tasks.get_task_address(cache_task_id);
                                let prim_header_index = prim_headers.push(&prim_header, [
                                    mode as u32 as i32,
                                    backdrop_task_address.0 as i32,
                                    source_task_address.0 as i32,
                                ]);

                                let instance = BrushInstance {
                                    prim_header_index,
                                    clip_task_address,
                                    segment_index: 0,
                                    edge_flags: EdgeAaSegmentMask::empty(),
                                    brush_flags: BrushFlags::empty(),
                                };

                                batch.push(PrimitiveInstance::from(instance));
                                false
                            }
                            Some(PictureCompositeMode::Blit) => {
                                let cache_task_id = picture
                                    .surface
                                    .as_ref()
                                    .expect("bug: no surface allocated")
                                    .resolve_render_task_id();
                                let kind = BatchKind::Brush(
                                    BrushBatchKind::Image(ImageBufferKind::Texture2DArray)
                                );
                                let key = BatchKey::new(
                                    kind,
                                    non_segmented_blend_mode,
                                    BatchTextures::render_target_cache(),
                                );
                                let batch = self.batch_list.get_suitable_batch(
                                    key,
                                    &task_relative_bounding_rect
                                );

                                let uv_rect_address = render_tasks[cache_task_id]
                                    .get_texture_address(gpu_cache)
                                    .as_int();
                                let prim_header_index = prim_headers.push(&prim_header, [
                                    uv_rect_address,
                                    (ShaderColorMode::ColorBitmap as i32) << 16 |
                                    RasterizationSpace::Screen as i32,
                                    0,
                                ]);

                                let instance = BrushInstance {
                                    prim_header_index,
                                    clip_task_address,
                                    segment_index: 0,
                                    edge_flags: EdgeAaSegmentMask::empty(),
                                    brush_flags: BrushFlags::empty(),
                                };
                                batch.push(PrimitiveInstance::from(instance));
                                false
                            }
                            None => {
                                true
                            }
                        };

                        // If this picture is being drawn into an existing target (i.e. with
                        // no composition operation), recurse and add to the current batch list.
                        if add_to_parent_pic {
                            self.add_pic_to_batch(
                                picture,
                                task_id,
                                ctx,
                                gpu_cache,
                                render_tasks,
                                deferred_resolves,
                                prim_headers,
                            );
                        }
                    }
                    BrushKind::Image { request, ref visible_tiles, .. } if !visible_tiles.is_empty() => {
                        for tile in visible_tiles {
                            if let Some((batch_kind, textures, user_data)) = get_image_tile_params(
                                    ctx.resource_cache,
                                    gpu_cache,
                                    deferred_resolves,
                                    request.with_tile(tile.tile_offset),
                            ) {
                                let prim_cache_address = gpu_cache.get_address(&tile.handle);
                                let prim_header = PrimitiveHeader {
                                    specific_prim_address: prim_cache_address,
                                    local_rect: tile.local_rect,
                                    local_clip_rect: tile.local_clip_rect,
                                    ..prim_header
                                };
                                let prim_header_index = prim_headers.push(&prim_header, user_data);

                                self.add_image_tile_to_batch(
                                    batch_kind,
                                    specified_blend_mode,
                                    textures,
                                    prim_header_index,
                                    clip_task_address,
                                    &task_relative_bounding_rect,
                                    tile.edge_flags
                                );
                            }
                        }
                    }
                    BrushKind::LinearGradient { ref stops_handle, ref visible_tiles, .. } if !visible_tiles.is_empty() => {
                        add_gradient_tiles(
                            visible_tiles,
                            stops_handle,
                            BrushBatchKind::LinearGradient,
                            specified_blend_mode,
                            &task_relative_bounding_rect,
                            clip_task_address,
                            gpu_cache,
                            &mut self.batch_list,
                            &prim_header,
                            prim_headers,
                        );
                    }
                    BrushKind::RadialGradient { ref stops_handle, ref visible_tiles, .. } if !visible_tiles.is_empty() => {
                        add_gradient_tiles(
                            visible_tiles,
                            stops_handle,
                            BrushBatchKind::RadialGradient,
                            specified_blend_mode,
                            &task_relative_bounding_rect,
                            clip_task_address,
                            gpu_cache,
                            &mut self.batch_list,
                            &prim_header,
                            prim_headers,
                        );
                    }
                    _ => {
                        if let Some((batch_kind, textures, user_data)) = brush.get_batch_params(
                                ctx.resource_cache,
                                gpu_cache,
                                deferred_resolves,
                                ctx.prim_store.chase_id == Some(prim_index),
                        ) {
                            let prim_header_index = prim_headers.push(&prim_header, user_data);
                            if cfg!(debug_assertions) && ctx.prim_store.chase_id == Some(prim_index) {
                                println!("\t{:?} {:?}, task relative bounds {:?}",
                                    batch_kind, prim_header_index, task_relative_bounding_rect);
                            }

                            self.add_brush_to_batch(
                                brush,
                                prim_metadata,
                                batch_kind,
                                specified_blend_mode,
                                non_segmented_blend_mode,
                                textures,
                                prim_header_index,
                                clip_task_address,
                                &task_relative_bounding_rect,
                                transform_kind,
                                render_tasks,
                            );
                        }
                    }
                }
            }
            PrimitiveDetails::TextRun(ref text_cpu) => {
                let subpx_dir = text_cpu.used_font.get_subpx_dir();

                let glyph_fetch_buffer = &mut self.glyph_fetch_buffer;
                let batch_list = &mut self.batch_list;

                ctx.resource_cache.fetch_glyphs(
                    text_cpu.used_font.clone(),
                    &text_cpu.glyph_keys,
                    glyph_fetch_buffer,
                    gpu_cache,
                    |texture_id, mut glyph_format, glyphs| {
                        debug_assert_ne!(texture_id, SourceTexture::Invalid);

                        // Ignore color and only sample alpha when shadowing.
                        if text_cpu.shadow {
                            glyph_format = glyph_format.ignore_color();
                        }

                        let subpx_dir = subpx_dir.limit_by(glyph_format);

                        let textures = BatchTextures {
                            colors: [
                                texture_id,
                                SourceTexture::Invalid,
                                SourceTexture::Invalid,
                            ],
                        };

                        let kind = BatchKind::TextRun(glyph_format);

                        let (blend_mode, color_mode) = match glyph_format {
                            GlyphFormat::Subpixel |
                            GlyphFormat::TransformedSubpixel => {
                                if text_cpu.used_font.bg_color.a != 0 {
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
                                        BlendMode::SubpixelConstantTextColor(text_cpu.used_font.color.into()),
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

                        let prim_header_index = prim_headers.push(&prim_header, [0; 3]);
                        let key = BatchKey::new(kind, blend_mode, textures);
                        let batch = batch_list.get_suitable_batch(key, &task_relative_bounding_rect);
                        let base_instance = GlyphInstance::new(
                            prim_header_index,
                        );

                        for glyph in glyphs {
                            batch.push(base_instance.build(
                                glyph.index_in_text_run,
                                glyph.uv_rect_address.as_int(),
                                (subpx_dir as u32 as i32) << 16 |
                                (color_mode as u32 as i32),
                            ));
                        }
                    },
                );
            }
        }
    }

    fn add_image_tile_to_batch(
        &mut self,
        batch_kind: BrushBatchKind,
        blend_mode: BlendMode,
        textures: BatchTextures,
        prim_header_index: PrimitiveHeaderIndex,
        clip_task_address: RenderTaskAddress,
        task_relative_bounding_rect: &DeviceIntRect,
        edge_flags: EdgeAaSegmentMask,
    ) {
        let base_instance = BrushInstance {
            prim_header_index,
            clip_task_address,
            segment_index: 0,
            edge_flags,
            brush_flags: BrushFlags::PERSPECTIVE_INTERPOLATION,
        };

        self.batch_list.add_bounding_rect(task_relative_bounding_rect);

        let batch_key = BatchKey {
            blend_mode,
            kind: BatchKind::Brush(batch_kind),
            textures,
        };
        let batch = self.batch_list.get_suitable_batch(batch_key, task_relative_bounding_rect);
        batch.push(PrimitiveInstance::from(base_instance));
    }

    fn add_brush_to_batch(
        &mut self,
        brush: &BrushPrimitive,
        prim_metadata: &PrimitiveMetadata,
        batch_kind: BrushBatchKind,
        alpha_blend_mode: BlendMode,
        non_segmented_blend_mode: BlendMode,
        textures: BatchTextures,
        prim_header_index: PrimitiveHeaderIndex,
        clip_task_address: RenderTaskAddress,
        task_relative_bounding_rect: &DeviceIntRect,
        transform_kind: TransformedRectKind,
        render_tasks: &RenderTaskTree,
    ) {
        let base_instance = BrushInstance {
            prim_header_index,
            clip_task_address,
            segment_index: 0,
            edge_flags: EdgeAaSegmentMask::all(),
            brush_flags: BrushFlags::PERSPECTIVE_INTERPOLATION,
        };

        self.batch_list.add_bounding_rect(task_relative_bounding_rect);

        match brush.segment_desc {
            Some(ref segment_desc) => {
                let alpha_batch_key = BatchKey {
                    blend_mode: alpha_blend_mode,
                    kind: BatchKind::Brush(batch_kind),
                    textures,
                };

                let alpha_batch = self.batch_list.alpha_batch_list.get_suitable_batch(
                    alpha_batch_key,
                    task_relative_bounding_rect
                );

                let opaque_batch_key = BatchKey {
                    blend_mode: BlendMode::None,
                    kind: BatchKind::Brush(batch_kind),
                    textures,
                };

                let opaque_batch = self.batch_list.opaque_batch_list.get_suitable_batch(
                    opaque_batch_key,
                    task_relative_bounding_rect
                );

                for (i, segment) in segment_desc.segments.iter().enumerate() {
                    let is_inner = segment.edge_flags.is_empty();
                    let needs_blending = !prim_metadata.opacity.is_opaque ||
                                         segment.clip_task_id.needs_blending() ||
                                         (!is_inner && transform_kind == TransformedRectKind::Complex);

                    let clip_task_address = match segment.clip_task_id {
                        BrushSegmentTaskId::RenderTaskId(id) =>
                            render_tasks.get_task_address(id),
                        BrushSegmentTaskId::Opaque => OPAQUE_TASK_ADDRESS,
                        BrushSegmentTaskId::Empty => continue,
                    };

                    let instance = PrimitiveInstance::from(BrushInstance {
                        segment_index: i as i32,
                        edge_flags: segment.edge_flags,
                        clip_task_address,
                        brush_flags: base_instance.brush_flags | segment.brush_flags,
                        ..base_instance
                    });

                    if needs_blending {
                        alpha_batch.push(instance);
                    } else {
                        opaque_batch.push(instance);
                    }
                }
            }
            None => {
                let batch_key = BatchKey {
                    blend_mode: non_segmented_blend_mode,
                    kind: BatchKind::Brush(batch_kind),
                    textures,
                };
                let batch = self.batch_list.get_suitable_batch(batch_key, task_relative_bounding_rect);
                batch.push(PrimitiveInstance::from(base_instance));
            }
        }

        self.batch_list.remove_unused_batches();
    }
}

fn add_gradient_tiles(
    visible_tiles: &[VisibleGradientTile],
    stops_handle: &GpuCacheHandle,
    kind: BrushBatchKind,
    blend_mode: BlendMode,
    task_relative_bounding_rect: &DeviceIntRect,
    clip_task_address: RenderTaskAddress,
    gpu_cache: &GpuCache,
    batch_list: &mut BatchList,
    base_prim_header: &PrimitiveHeader,
    prim_headers: &mut PrimitiveHeaders,
) {
    batch_list.add_bounding_rect(task_relative_bounding_rect);
    let batch = batch_list.get_suitable_batch(
        BatchKey {
            blend_mode: blend_mode,
            kind: BatchKind::Brush(kind),
            textures: BatchTextures::no_texture(),
        },
        task_relative_bounding_rect
    );

    let user_data = [stops_handle.as_int(gpu_cache), 0, 0];

    for tile in visible_tiles {
        let prim_header = PrimitiveHeader {
            specific_prim_address: gpu_cache.get_address(&tile.handle),
            local_rect: tile.local_rect,
            local_clip_rect: tile.local_clip_rect,
            ..*base_prim_header
        };
        let prim_header_index = prim_headers.push(&prim_header, user_data);

        batch.push(PrimitiveInstance::from(
            BrushInstance {
                prim_header_index,
                clip_task_address,
                segment_index: 0,
                edge_flags: EdgeAaSegmentMask::all(),
                brush_flags: BrushFlags::PERSPECTIVE_INTERPOLATION,
            }
        ));
    }
}

fn get_image_tile_params(
    resource_cache: &ResourceCache,
    gpu_cache: &mut GpuCache,
    deferred_resolves: &mut Vec<DeferredResolve>,
    request: ImageRequest,
) -> Option<(BrushBatchKind, BatchTextures, [i32; 3])> {

    let cache_item = resolve_image(
        request,
        resource_cache,
        gpu_cache,
        deferred_resolves,
    );

    if cache_item.texture_id == SourceTexture::Invalid {
        None
    } else {
        let textures = BatchTextures::color(cache_item.texture_id);
        Some((
            BrushBatchKind::Image(get_buffer_kind(cache_item.texture_id)),
            textures,
            [
                cache_item.uv_rect_handle.as_int(gpu_cache),
                (ShaderColorMode::ColorBitmap as i32) << 16 |
                     RasterizationSpace::Local as i32,
                0,
            ],
        ))
    }
}

impl BrushPrimitive {
    fn get_batch_params(
        &self,
        resource_cache: &ResourceCache,
        gpu_cache: &mut GpuCache,
        deferred_resolves: &mut Vec<DeferredResolve>,
        is_chased: bool,
    ) -> Option<(BrushBatchKind, BatchTextures, [i32; 3])> {
        match self.kind {
            BrushKind::Image { request, ref source, .. } => {
                let cache_item = match *source {
                    ImageSource::Default => {
                        resolve_image(
                            request,
                            resource_cache,
                            gpu_cache,
                            deferred_resolves,
                        )
                    }
                    ImageSource::Cache { ref handle, .. } => {
                        let rt_handle = handle
                            .as_ref()
                            .expect("bug: render task handle not allocated");
                        let rt_cache_entry = resource_cache
                            .get_cached_render_task(rt_handle);
                        resource_cache.get_texture_cache_item(&rt_cache_entry.handle)
                    }
                };
                if cfg!(debug_assertions) && is_chased {
                    println!("\tsource {:?}", cache_item);
                }

                if cache_item.texture_id == SourceTexture::Invalid {
                    None
                } else {
                    let textures = BatchTextures::color(cache_item.texture_id);

                    Some((
                        BrushBatchKind::Image(get_buffer_kind(cache_item.texture_id)),
                        textures,
                        [
                            cache_item.uv_rect_handle.as_int(gpu_cache),
                            (ShaderColorMode::ColorBitmap as i32) << 16|
                             RasterizationSpace::Local as i32,
                            0,
                        ],
                    ))
                }
            }
            BrushKind::Border { ref source, .. } => {
                let cache_item = match *source {
                    BorderSource::Image(request) => {
                        resolve_image(
                            request,
                            resource_cache,
                            gpu_cache,
                            deferred_resolves,
                        )
                    }
                    BorderSource::Border { ref handle, .. } => {
                        let rt_handle = match *handle {
                            Some(ref handle) => handle,
                            None => return None,
                        };
                        let rt_cache_entry = resource_cache
                            .get_cached_render_task(rt_handle);
                        resource_cache.get_texture_cache_item(&rt_cache_entry.handle)
                    }
                };

                if cache_item.texture_id == SourceTexture::Invalid {
                    None
                } else {
                    let textures = BatchTextures::color(cache_item.texture_id);

                    Some((
                        BrushBatchKind::Image(get_buffer_kind(cache_item.texture_id)),
                        textures,
                        [
                            cache_item.uv_rect_handle.as_int(gpu_cache),
                            (ShaderColorMode::ColorBitmap as i32) << 16|
                             RasterizationSpace::Local as i32,
                            0,
                        ],
                    ))
                }
            }
            BrushKind::Picture { .. } => {
                panic!("bug: get_batch_key is handled at higher level for pictures");
            }
            BrushKind::Solid { .. } => {
                Some((
                    BrushBatchKind::Solid,
                    BatchTextures::no_texture(),
                    [0; 3],
                ))
            }
            BrushKind::Clear => {
                Some((
                    BrushBatchKind::Solid,
                    BatchTextures::no_texture(),
                    [0; 3],
                ))
            }
            BrushKind::RadialGradient { ref stops_handle, .. } => {
                Some((
                    BrushBatchKind::RadialGradient,
                    BatchTextures::no_texture(),
                    [
                        stops_handle.as_int(gpu_cache),
                        0,
                        0,
                    ],
                ))
            }
            BrushKind::LinearGradient { ref stops_handle, .. } => {
                Some((
                    BrushBatchKind::LinearGradient,
                    BatchTextures::no_texture(),
                    [
                        stops_handle.as_int(gpu_cache),
                        0,
                        0,
                    ],
                ))
            }
            BrushKind::YuvImage { format, yuv_key, image_rendering, color_space } => {
                let mut textures = BatchTextures::no_texture();
                let mut uv_rect_addresses = [0; 3];

                //yuv channel
                let channel_count = format.get_plane_num();
                debug_assert!(channel_count <= 3);
                for channel in 0 .. channel_count {
                    let image_key = yuv_key[channel];

                    let cache_item = resolve_image(
                        ImageRequest {
                            key: image_key,
                            rendering: image_rendering,
                            tile: None,
                        },
                        resource_cache,
                        gpu_cache,
                        deferred_resolves,
                    );

                    if cache_item.texture_id == SourceTexture::Invalid {
                        warn!("Warnings: skip a PrimitiveKind::YuvImage");
                        return None;
                    }

                    textures.colors[channel] = cache_item.texture_id;
                    uv_rect_addresses[channel] = cache_item.uv_rect_handle.as_int(gpu_cache);
                }

                // All yuv textures should be the same type.
                let buffer_kind = get_buffer_kind(textures.colors[0]);
                assert!(
                    textures.colors[1 .. format.get_plane_num()]
                        .iter()
                        .all(|&tid| buffer_kind == get_buffer_kind(tid))
                );

                let kind = BrushBatchKind::YuvImage(
                    buffer_kind,
                    format,
                    color_space,
                );

                Some((
                    kind,
                    textures,
                    [
                        uv_rect_addresses[0],
                        uv_rect_addresses[1],
                        uv_rect_addresses[2],
                    ],
                ))
            }
        }
    }
}

impl Primitive {
    fn get_blend_mode(&self) -> BlendMode {
        match self.details {
            // Can only resolve the TextRun's blend mode once glyphs are fetched.
            PrimitiveDetails::TextRun(..) => {
                BlendMode::PremultipliedAlpha
            }

            PrimitiveDetails::Brush(ref brush) => {
                match brush.kind {
                    BrushKind::Clear => {
                        BlendMode::PremultipliedDestOut
                    }
                    BrushKind::Image { alpha_type, .. } => {
                        match alpha_type {
                            AlphaType::PremultipliedAlpha => BlendMode::PremultipliedAlpha,
                            AlphaType::Alpha => BlendMode::Alpha,
                        }
                    }
                    BrushKind::Solid { .. } |
                    BrushKind::YuvImage { .. } |
                    BrushKind::RadialGradient { .. } |
                    BrushKind::LinearGradient { .. } |
                    BrushKind::Border { .. } |
                    BrushKind::Picture { .. } => {
                        BlendMode::PremultipliedAlpha
                    }
                }
            }
        }
    }
}

impl PictureSurface {
    // Retrieve the uv rect handle, and texture for a picture surface.
    fn resolve(
        &self,
        render_tasks: &RenderTaskTree,
        resource_cache: &ResourceCache,
        gpu_cache: &GpuCache,
    ) -> (GpuCacheAddress, BatchTextures) {
        match *self {
            PictureSurface::TextureCache(ref handle) => {
                let rt_cache_entry = resource_cache
                    .get_cached_render_task(handle);
                let cache_item = resource_cache
                    .get_texture_cache_item(&rt_cache_entry.handle);

                (
                    gpu_cache.get_address(&cache_item.uv_rect_handle),
                    BatchTextures::color(cache_item.texture_id),
                )
            }
            PictureSurface::RenderTask(task_id) => {
                (
                    render_tasks[task_id].get_texture_address(gpu_cache),
                    BatchTextures::render_target_cache(),
                )
            }
        }
    }

    // Retrieve the render task id for a picture surface. Should only
    // be used where it's known that this picture surface will never
    // be persisted in the texture cache.
    fn resolve_render_task_id(&self) -> RenderTaskId {
        match *self {
            PictureSurface::TextureCache(..) => {
                panic!("BUG: unexpectedly cached render task");
            }
            PictureSurface::RenderTask(task_id) => {
                task_id
            }
        }
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
                        texture_id: SourceTexture::External(external_image),
                        uv_rect_handle: cache_handle,
                        uv_rect: DeviceUintRect::new(
                            DeviceUintPoint::zero(),
                            image_properties.descriptor.size,
                        ),
                        texture_layer: 0,
                    };

                    deferred_resolves.push(DeferredResolve {
                        image_properties,
                        address: gpu_cache.get_address(&cache_handle),
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


/// Batcher managing draw calls into the clip mask (in the RT cache).
#[derive(Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct ClipBatcher {
    /// Rectangle draws fill up the rectangles with rounded corners.
    pub rectangles: Vec<ClipMaskInstance>,
    /// Image draws apply the image masking.
    pub images: FastHashMap<SourceTexture, Vec<ClipMaskInstance>>,
    pub box_shadows: FastHashMap<SourceTexture, Vec<ClipMaskInstance>>,
    pub line_decorations: Vec<ClipMaskInstance>,
}

impl ClipBatcher {
    pub fn new() -> Self {
        ClipBatcher {
            rectangles: Vec::new(),
            images: FastHashMap::default(),
            box_shadows: FastHashMap::default(),
            line_decorations: Vec::new(),
        }
    }

    pub fn add_clip_region(
        &mut self,
        task_address: RenderTaskAddress,
        clip_data_address: GpuCacheAddress,
    ) {
        let instance = ClipMaskInstance {
            render_task_address: task_address,
            transform_id: TransformPaletteId::IDENTITY,
            segment: 0,
            clip_data_address,
            resource_address: GpuCacheAddress::invalid(),
        };

        self.rectangles.push(instance);
    }

    pub fn add(
        &mut self,
        task_address: RenderTaskAddress,
        clip_node_range: ClipNodeRange,
        resource_cache: &ResourceCache,
        gpu_cache: &GpuCache,
        clip_store: &ClipStore,
        transforms: &TransformPalette,
    ) {
        for i in 0 .. clip_node_range.count {
            let (clip_node, flags) = clip_store.get_node_from_range(&clip_node_range, i);

            let instance = ClipMaskInstance {
                render_task_address: task_address,
                transform_id: transforms.get_id(clip_node.spatial_node_index),
                segment: 0,
                clip_data_address: GpuCacheAddress::invalid(),
                resource_address: GpuCacheAddress::invalid(),
            };

            let gpu_address = gpu_cache.get_address(&clip_node.gpu_cache_handle);

            match clip_node.item {
                ClipItem::Image(ref mask) => {
                    if let Ok(cache_item) = resource_cache.get_cached_image(
                        ImageRequest {
                            key: mask.image,
                            rendering: ImageRendering::Auto,
                            tile: None,
                        }
                    ) {
                        self.images
                            .entry(cache_item.texture_id)
                            .or_insert(Vec::new())
                            .push(ClipMaskInstance {
                                clip_data_address: gpu_address,
                                resource_address: gpu_cache.get_address(&cache_item.uv_rect_handle),
                                ..instance
                            });
                    } else {
                        warn!("Warnings: skip a image mask");
                        debug!("Key:{:?} Rect::{:?}", mask.image, mask.rect);
                        continue;
                    }
                }
                ClipItem::LineDecoration(..) => {
                    self.line_decorations.push(ClipMaskInstance {
                        clip_data_address: gpu_address,
                        ..instance
                    });
                }
                ClipItem::BoxShadow(ref info) => {
                    let rt_handle = info
                        .cache_handle
                        .as_ref()
                        .expect("bug: render task handle not allocated");
                    let rt_cache_entry = resource_cache
                        .get_cached_render_task(rt_handle);
                    let cache_item = resource_cache
                        .get_texture_cache_item(&rt_cache_entry.handle);
                    debug_assert_ne!(cache_item.texture_id, SourceTexture::Invalid);

                    self.box_shadows
                        .entry(cache_item.texture_id)
                        .or_insert(Vec::new())
                        .push(ClipMaskInstance {
                            clip_data_address: gpu_address,
                            resource_address: gpu_cache.get_address(&cache_item.uv_rect_handle),
                            ..instance
                        });
                }
                ClipItem::Rectangle(_, mode) => {
                    if !flags.contains(ClipNodeFlags::SAME_COORD_SYSTEM) ||
                        mode == ClipMode::ClipOut {
                        self.rectangles.push(ClipMaskInstance {
                            clip_data_address: gpu_address,
                            ..instance
                        });
                    }
                }
                ClipItem::RoundedRectangle(..) => {
                    self.rectangles.push(ClipMaskInstance {
                        clip_data_address: gpu_address,
                        ..instance
                    });
                }
            }
        }
    }
}

fn get_buffer_kind(texture: SourceTexture) -> ImageBufferKind {
    match texture {
        SourceTexture::External(ext_image) => {
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
