/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{BorderStyle, NormalBorder, PremultipliedColorF, RasterSpace, Shadow};
use api::units::*;
use crate::border::{self, build_border_instances, get_max_scale_for_border};
use crate::border::NormalBorderAu;
use crate::gpu_types::ImageBrushPrimitiveData;
use crate::render_backend::DataStores;
use crate::render_task_cache::{RenderTaskCacheKey, RenderTaskCacheKeyKind, RenderTaskParent, to_cache_size};
use crate::renderer::GpuBufferWriterF;
use crate::scene_building::{CreateShadow, IsVisible};
use crate::frame_builder::{FrameBuildingContext, FrameBuildingState};
use crate::intern;
use crate::internal_types::{LayoutPrimitiveInfo, FrameId};
use crate::prim_store::{
    BorderSegmentInfo, BrushSegment, InternablePrimitive, NinePatchDescriptor, PrimKey, PrimTemplate, PrimTemplateCommonData, PrimitiveInstanceIndex, PrimitiveKind, PrimitiveOpacity, PrimitiveScratchBuffer, PrimitiveStore, VECS_PER_SEGMENT
};
use crate::resource_cache::ImageRequest;
use crate::render_task::{RenderTask, RenderTaskKind};
use crate::render_task_graph::RenderTaskId;
use crate::spatial_tree::SpatialNodeIndex;
use crate::util::clamp_to_scale_factor;
use crate::visibility::KindScratchHandle;

use crate::prim_store::storage;

/// Per-frame scratch data for a NormalBorder primitive.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
pub struct NormalBorderScratch {
    /// Range into `PrimitiveScratchBuffer::border_task_ids` holding the
    /// cached render-task ids for this border's segments.
    pub task_ids: storage::Range<RenderTaskId>,
    /// Range into `PrimitiveFrameScratch::segments` holding the per-
    /// frame brush segments for this border. Built fresh each frame
    /// against the prim's current size in `prepare_prim_for_render`,
    /// so the segmentation matches the rendered rect.
    pub brush_segments_range: storage::Range<BrushSegment>,
    /// Range into `PrimitiveFrameScratch::border_segments` holding the
    /// per-frame edge/corner cache-key + task-size records for this
    /// border. Parallel to `brush_segments_range` and built alongside.
    pub border_segments_range: storage::Range<BorderSegmentInfo>,
    /// True if any side uses a Dotted or Dashed style. Read by batch
    /// to set `BatchFeatures::REPETITION` so the cached dot/dash tile
    /// repeats across the rendered segment via brush_image.
    pub may_need_repetition: bool,
}

impl NormalBorderScratch {
    /// Build the per-frame brush + border segments and the parallel
    /// task-id slot for a NormalBorder prim, push the resulting
    /// `NormalBorderScratch` entry, and wire it up to the prim's
    /// `PrimitiveDrawHeader.kind_scratch`.
    ///
    /// Called from the prep-pass before `update_clip_task` runs, since
    /// `update_clip_task_for_brush` reads the brush segments via the
    /// `NormalBorderScratch` allocated here. The segment list is built
    /// against `common.prim_size`, with the two arenas
    /// (`scratch.frame.segments` and `scratch.frame.border_segments`)
    /// receiving direct pushes through `data_mut` to avoid intermediate
    /// `Vec` allocations.
    pub fn build_for_prim(
        data_handle: NormalBorderDataHandle,
        prim_instance_index: PrimitiveInstanceIndex,
        data_stores: &DataStores,
        scratch: &mut PrimitiveScratchBuffer,
    ) {
        let prim_data = &data_stores.normal_border[data_handle];
        let prim_size = prim_data.common.prim_size;
        let border = &prim_data.kind.border;
        let widths = &prim_data.kind.widths;

        let brush_open = scratch.frame.segments.open_range();
        let border_open = scratch.frame.border_segments.open_range();
        border::create_border_segments(
            prim_size,
            border,
            widths,
            scratch.frame.border_segments.data_mut(),
            scratch.frame.segments.data_mut(),
        );
        let brush_segments_range = scratch.frame.segments.close_range(brush_open);
        let border_segments_range = scratch.frame.border_segments.close_range(border_open);

        let may_need_repetition =
            matches!(border.top.style, BorderStyle::Dotted | BorderStyle::Dashed)
                || matches!(border.right.style, BorderStyle::Dotted | BorderStyle::Dashed)
                || matches!(border.bottom.style, BorderStyle::Dotted | BorderStyle::Dashed)
                || matches!(border.left.style, BorderStyle::Dotted | BorderStyle::Dashed);

        let segment_count = border_segments_range.end.0
            .saturating_sub(border_segments_range.start.0) as usize;
        let task_ids = scratch.frame.border_task_ids.extend(
            std::iter::repeat(RenderTaskId::INVALID).take(segment_count),
        );
        let handle = scratch.frame.normal_border.push(NormalBorderScratch {
            task_ids,
            brush_segments_range,
            border_segments_range,
            may_need_repetition,
        });
        scratch.frame.draws[prim_instance_index.0 as usize].kind_scratch =
            KindScratchHandle::NormalBorder(handle);
    }
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Clone, Eq, MallocSizeOf, PartialEq, Hash)]
pub struct NormalBorderPrim {
    pub border: NormalBorderAu,
    pub widths: LayoutSideOffsetsAu,
}

pub type NormalBorderKey = PrimKey<NormalBorderPrim>;

impl NormalBorderKey {
    pub fn new(
        info: &LayoutPrimitiveInfo,
        normal_border: NormalBorderPrim,
    ) -> Self {
        NormalBorderKey {
            common: info.into(),
            kind: normal_border,
        }
    }
}

impl intern::InternDebug for NormalBorderKey {}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(MallocSizeOf)]
pub struct NormalBorderData {
    pub border: NormalBorder,
    pub widths: LayoutSideOffsets,
}

impl NormalBorderData {
    /// Update the GPU cache for a given primitive template. This may be called multiple
    /// times per frame, by each primitive reference that refers to this interned
    /// template. The initial request call to the GPU cache ensures that work is only
    /// done if the cache entry is invalid (due to first use or eviction).
    pub fn write_brush_gpu_blocks(
        &mut self,
        common: &mut PrimTemplateCommonData,
        brush_segments: &[BrushSegment],
        frame_state: &mut FrameBuildingState,
    ) {
        let mut writer = frame_state.frame_gpu_data.f32.write_blocks(3 + brush_segments.len() * VECS_PER_SEGMENT);

        // Border primitives currently used for
        // image borders, and run through the
        // normal brush_image shader.
        writer.push(&ImageBrushPrimitiveData {
            color: PremultipliedColorF::WHITE,
            background_color: PremultipliedColorF::WHITE,
            stretch_size: common.prim_size,
        });

        for segment in brush_segments {
            segment.write_gpu_blocks(&mut writer);
        }

        common.gpu_buffer_address = writer.finish();
        common.opacity = PrimitiveOpacity::translucent();
    }

    pub fn update(
        &mut self,
        border_segments: &[BorderSegmentInfo],
        prim_spatial_node_index: SpatialNodeIndex,
        device_pixel_scale: DevicePixelScale,
        frame_context: &FrameBuildingContext,
        frame_state: &mut FrameBuildingState,
        task_ids: &mut [RenderTaskId],
    ) {
        // TODO(gw): For now, the scale factors to rasterize borders at are
        //           based on the true world transform of the primitive. When
        //           raster roots with local scale are supported in future,
        //           that will need to be accounted for here.
        let scale = frame_context
            .spatial_tree
            .get_world_transform(prim_spatial_node_index)
            .scale_factors();

        // Scale factors are normalized to a power of 2 to reduce the number of
        // resolution changes.
        // For frames with a changing scale transform round scale factors up to
        // nearest power-of-2 boundary so that we don't keep having to redraw
        // the content as it scales up and down. Rounding up to nearest
        // power-of-2 boundary ensures we never scale up, only down --- avoiding
        // jaggies. It also ensures we never scale down by more than a factor of
        // 2, avoiding bad downscaling quality.
        let scale_width = clamp_to_scale_factor(scale.0, false);
        let scale_height = clamp_to_scale_factor(scale.1, false);
        // Pick the maximum dimension as scale
        let world_scale = LayoutToWorldScale::new(scale_width.max(scale_height));
        let mut scale = world_scale * device_pixel_scale;
        let max_scale = get_max_scale_for_border(border_segments);
        scale.0 = scale.0.min(max_scale.0);

        // For each edge and corner, request the render task by content key
        // from the render task cache. This ensures that the render task for
        // this segment will be available for batching later in the frame.
        // TODO: this does not ensure that segments will be in the same cache
        // texture, though? The brush code path relies on that.

        for (i, segment) in border_segments.iter().enumerate() {
            // Update the cache key device size based on requested scale.
            let cache_size = to_cache_size(segment.local_task_size, &mut scale);
            let cache_key = RenderTaskCacheKey {
                kind: RenderTaskCacheKeyKind::BorderSegment(segment.cache_key.clone()),
                origin: DeviceIntPoint::zero(),
                size: cache_size,
            };

            let task_id = frame_state.resource_cache.request_render_task(
                Some(cache_key),
                false,          // TODO(gw): We don't calculate opacity for borders yet!
                RenderTaskParent::Surface,
                &mut frame_state.frame_gpu_data.f32,
                frame_state.rg_builder,
                &mut frame_state.surface_builder,
                &mut |rg_builder, _| {
                    rg_builder.add().init(RenderTask::new_dynamic(
                        cache_size,
                        RenderTaskKind::new_border_segment(
                            build_border_instances(
                                &segment.cache_key,
                                cache_size,
                                &self.border,
                                scale,
                            )
                        ),
                    ))
                }
            );

            task_ids[i] = task_id;
        }
    }
}

pub type NormalBorderTemplate = PrimTemplate<NormalBorderData>;

impl From<NormalBorderKey> for NormalBorderTemplate {
    fn from(key: NormalBorderKey) -> Self {
        let common = PrimTemplateCommonData::with_key_common(key.common);

        let mut border: NormalBorder = key.kind.border.into();
        let widths = LayoutSideOffsets::from_au(key.kind.widths);

        // FIXME(emilio): Is this the best place to do this?
        border.normalize(&widths);

        NormalBorderTemplate {
            common,
            kind: NormalBorderData {
                border,
                widths,
            }
        }
    }
}

pub type NormalBorderDataHandle = intern::Handle<NormalBorderPrim>;

impl intern::Internable for NormalBorderPrim {
    type Key = NormalBorderKey;
    type StoreData = NormalBorderTemplate;
    type InternData = ();
    const PROFILE_COUNTER: usize = crate::profiler::INTERNED_NORMAL_BORDERS;
}

impl InternablePrimitive for NormalBorderPrim {
    fn into_key(
        self,
        info: &LayoutPrimitiveInfo,
    ) -> NormalBorderKey {
        NormalBorderKey::new(
            info,
            self,
        )
    }

    fn make_instance_kind(
        _key: NormalBorderKey,
        data_handle: NormalBorderDataHandle,
        _: &mut PrimitiveStore,
    ) -> PrimitiveKind {
        PrimitiveKind::NormalBorder {
            data_handle,
        }
    }
}

impl CreateShadow for NormalBorderPrim {
    fn create_shadow(
        &self,
        shadow: &Shadow,
        _: bool,
        _: RasterSpace,
    ) -> Self {
        let border = self.border.with_color(shadow.color.into());
        NormalBorderPrim {
            border,
            widths: self.widths,
        }
    }
}

impl IsVisible for NormalBorderPrim {
    fn is_visible(&self) -> bool {
        true
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Clone, Eq, MallocSizeOf, PartialEq, Hash)]
pub struct ImageBorder {
    #[ignore_malloc_size_of = "Arc"]
    pub request: ImageRequest,
    pub nine_patch: NinePatchDescriptor,
}

pub type ImageBorderKey = PrimKey<ImageBorder>;

impl ImageBorderKey {
    pub fn new(
        info: &LayoutPrimitiveInfo,
        image_border: ImageBorder,
    ) -> Self {
        ImageBorderKey {
            common: info.into(),
            kind: image_border,
        }
    }
}

impl intern::InternDebug for ImageBorderKey {}


/// Per-frame scratch data for an ImageBorder primitive.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
pub struct ImageBorderScratch {
    /// Range into `PrimitiveFrameScratch::segments` holding the per-
    /// frame nine-patch brush segments for this border. Built fresh
    /// each frame against the prim's current size in
    /// `prepare_prim_for_render`.
    pub brush_segments_range: storage::Range<BrushSegment>,
}

impl ImageBorderScratch {
    /// Build the per-frame nine-patch brush segments for an ImageBorder
    /// prim, push the resulting `ImageBorderScratch` entry, and wire it
    /// up to the prim's `PrimitiveDrawHeader.kind_scratch`.
    ///
    /// Called from the prep early pass before `update_clip_task` runs,
    /// since `update_clip_task_for_brush` reads the brush segments via
    /// the scratch entry allocated here.
    pub fn build_for_prim(
        data_handle: ImageBorderDataHandle,
        prim_instance_index: PrimitiveInstanceIndex,
        data_stores: &DataStores,
        scratch: &mut PrimitiveScratchBuffer,
    ) {
        let prim_data = &data_stores.image_border[data_handle];
        let prim_size = prim_data.common.prim_size;
        let nine_patch = &prim_data.kind.nine_patch;

        let brush_open = scratch.frame.segments.open_range();
        scratch.frame.segments.data_mut().extend(
            nine_patch.create_brush_segments(prim_size),
        );
        let brush_segments_range = scratch.frame.segments.close_range(brush_open);

        let handle = scratch.frame.image_border.push(ImageBorderScratch {
            brush_segments_range,
        });
        scratch.frame.draws[prim_instance_index.0 as usize].kind_scratch =
            KindScratchHandle::ImageBorder(handle);
    }
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(MallocSizeOf)]
pub struct ImageBorderData {
    #[ignore_malloc_size_of = "Arc"]
    pub request: ImageRequest,
    pub nine_patch: NinePatchDescriptor,
    pub src_color: Option<RenderTaskId>,
    pub frame_id: FrameId,
    pub is_opaque: bool,
}

impl ImageBorderData {
    /// Update the GPU cache for a given primitive template. This may be called multiple
    /// times per frame, by each primitive reference that refers to this interned
    /// template. The initial request call to the GPU cache ensures that work is only
    /// done if the cache entry is invalid (due to first use or eviction).
    pub fn update(
        &mut self,
        common: &mut PrimTemplateCommonData,
        brush_segments: &[BrushSegment],
        frame_state: &mut FrameBuildingState,
    ) {
        let mut writer = frame_state.frame_gpu_data.f32.write_blocks(3 + brush_segments.len() * VECS_PER_SEGMENT);
        self.write_prim_gpu_blocks(&mut writer, &common.prim_size);
        Self::write_segment_gpu_blocks(&mut writer, brush_segments);
        common.gpu_buffer_address = writer.finish();

        let frame_id = frame_state.rg_builder.frame_id();
        if self.frame_id != frame_id {
            self.frame_id = frame_id;

            let size = frame_state.resource_cache.request_image(
                self.request,
                &mut frame_state.frame_gpu_data.f32,
            );

            let task_id = frame_state.rg_builder.add().init(
                RenderTask::new_image(size, self.request, false)
            );

            self.src_color = Some(task_id);

            let image_properties = frame_state
                .resource_cache
                .get_image_properties(self.request.key);

            self.is_opaque = image_properties
                .map(|properties| properties.descriptor.is_opaque())
                .unwrap_or(true);
        }

        common.opacity = PrimitiveOpacity { is_opaque: self.is_opaque };
    }

    fn write_prim_gpu_blocks(
        &self,
        writer: &mut GpuBufferWriterF,
        prim_size: &LayoutSize,
    ) {
        // Border primitives currently used for
        // image borders, and run through the
        // normal brush_image shader.
        writer.push(&ImageBrushPrimitiveData {
            color: PremultipliedColorF::WHITE,
            background_color: PremultipliedColorF::WHITE,
            stretch_size: *prim_size,
        });
    }

    fn write_segment_gpu_blocks(
        writer: &mut GpuBufferWriterF,
        brush_segments: &[BrushSegment],
    ) {
        for segment in brush_segments {
            segment.write_gpu_blocks(writer);
        }
    }
}

pub type ImageBorderTemplate = PrimTemplate<ImageBorderData>;

impl From<ImageBorderKey> for ImageBorderTemplate {
    fn from(key: ImageBorderKey) -> Self {
        let common = PrimTemplateCommonData::with_key_common(key.common);

        ImageBorderTemplate {
            common,
            kind: ImageBorderData {
                request: key.kind.request,
                nine_patch: key.kind.nine_patch,
                src_color: None,
                frame_id: FrameId::INVALID,
                is_opaque: false,
            }
        }
    }
}

pub type ImageBorderDataHandle = intern::Handle<ImageBorder>;

impl intern::Internable for ImageBorder {
    type Key = ImageBorderKey;
    type StoreData = ImageBorderTemplate;
    type InternData = ();
    const PROFILE_COUNTER: usize = crate::profiler::INTERNED_IMAGE_BORDERS;
}

impl InternablePrimitive for ImageBorder {
    fn into_key(
        self,
        info: &LayoutPrimitiveInfo,
    ) -> ImageBorderKey {
        ImageBorderKey::new(
            info,
            self,
        )
    }

    fn make_instance_kind(
        _key: ImageBorderKey,
        data_handle: ImageBorderDataHandle,
        _: &mut PrimitiveStore,
    ) -> PrimitiveKind {
        PrimitiveKind::ImageBorder {
            data_handle
        }
    }
}

impl IsVisible for ImageBorder {
    fn is_visible(&self) -> bool {
        true
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
fn test_struct_sizes() {
    use std::mem;
    // The sizes of these structures are critical for performance on a number of
    // talos stress tests. If you get a failure here on CI, there's two possibilities:
    // (a) You made a structure smaller than it currently is. Great work! Update the
    //     test expectations and move on.
    // (b) You made a structure larger. This is not necessarily a problem, but should only
    //     be done with care, and after checking if talos performance regresses badly.
    assert_eq!(mem::size_of::<NormalBorderPrim>(), 84, "NormalBorderPrim size changed");
    assert_eq!(mem::size_of::<NormalBorderTemplate>(), 148, "NormalBorderTemplate size changed");
    assert_eq!(mem::size_of::<NormalBorderKey>(), 96, "NormalBorderKey size changed");
    assert_eq!(mem::size_of::<ImageBorder>(), 68, "ImageBorder size changed");
    assert_eq!(mem::size_of::<ImageBorderTemplate>(), 104, "ImageBorderTemplate size changed");
    assert_eq!(mem::size_of::<ImageBorderKey>(), 80, "ImageBorderKey size changed");
}
