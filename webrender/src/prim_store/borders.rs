/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{BorderStyle, NormalBorder, PremultipliedColorF, RasterSpace, Shadow};
use api::units::*;
use crate::border::{build_border_instances, create_border_segments, get_max_scale_for_border};
use crate::border::NormalBorderAu;
use crate::gpu_types::ImageBrushPrimitiveData;
use crate::render_task_cache::{RenderTaskCacheKey, RenderTaskCacheKeyKind, RenderTaskParent, to_cache_size};
use crate::renderer::GpuBufferWriterF;
use crate::scene_building::{CreateShadow, IsVisible};
use crate::frame_builder::{FrameBuildingContext, FrameBuildingState};
use crate::intern;
use crate::internal_types::{LayoutPrimitiveInfo, FrameId};
use crate::prim_store::{
    BorderSegmentInfo, BrushSegment, InternablePrimitive, NinePatchDescriptor, PrimKey, PrimTemplate, PrimTemplateCommonData, PrimitiveKind, PrimitiveOpacity, PrimitiveStore, VECS_PER_SEGMENT
};
use crate::resource_cache::ImageRequest;
use crate::render_task::{RenderTask, RenderTaskKind};
use crate::render_task_graph::RenderTaskId;
use crate::spatial_tree::SpatialNodeIndex;
use crate::util::clamp_to_scale_factor;

use crate::prim_store::storage;

/// Per-frame scratch data for a NormalBorder primitive.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
pub struct NormalBorderScratch {
    /// Range into `PrimitiveScratchBuffer::border_task_ids` holding the
    /// cached render-task ids for this border's segments.
    pub task_ids: storage::Range<RenderTaskId>,
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
    pub brush_segments: Vec<BrushSegment>,
    pub border_segments: Vec<BorderSegmentInfo>,
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
        frame_state: &mut FrameBuildingState,
    ) {
        let mut writer = frame_state.frame_gpu_data.f32.write_blocks(3 + self.brush_segments.len() * VECS_PER_SEGMENT);

        // Border primitives currently used for
        // image borders, and run through the
        // normal brush_image shader.
        writer.push(&ImageBrushPrimitiveData {
            color: PremultipliedColorF::WHITE,
            background_color: PremultipliedColorF::WHITE,
            stretch_size: common.prim_size,
        });

        for segment in &self.brush_segments {
            segment.write_gpu_blocks(&mut writer);
        }

        common.gpu_buffer_address = writer.finish();
        common.opacity = PrimitiveOpacity::translucent();
    }

    pub fn update(
        &mut self,
        common_data: &mut PrimTemplateCommonData,
        prim_spatial_node_index: SpatialNodeIndex,
        device_pixel_scale: DevicePixelScale,
        frame_context: &FrameBuildingContext,
        frame_state: &mut FrameBuildingState,
        task_ids: &mut [RenderTaskId],
    ) {
        common_data.may_need_repetition =
            matches!(self.border.top.style, BorderStyle::Dotted | BorderStyle::Dashed) ||
            matches!(self.border.right.style, BorderStyle::Dotted | BorderStyle::Dashed) ||
            matches!(self.border.bottom.style, BorderStyle::Dotted | BorderStyle::Dashed) ||
            matches!(self.border.left.style, BorderStyle::Dotted | BorderStyle::Dashed);

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
        let max_scale = get_max_scale_for_border(self);
        scale.0 = scale.0.min(max_scale.0);

        // For each edge and corner, request the render task by content key
        // from the render task cache. This ensures that the render task for
        // this segment will be available for batching later in the frame.
        // TODO: this does not ensure that segments will be in the same cache
        // texture, though? The brush code path relies on that.

        for (i, segment) in self.border_segments.iter().enumerate() {
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

        let mut brush_segments = Vec::new();
        let mut border_segments = Vec::new();

        create_border_segments(
            common.prim_size,
            &border,
            &widths,
            &mut border_segments,
            &mut brush_segments,
        );

        NormalBorderTemplate {
            common,
            kind: NormalBorderData {
                brush_segments,
                border_segments,
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
            scratch_handle: storage::Index::INVALID,
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


#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(MallocSizeOf)]
pub struct ImageBorderData {
    #[ignore_malloc_size_of = "Arc"]
    pub request: ImageRequest,
    pub brush_segments: Vec<BrushSegment>,
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
        frame_state: &mut FrameBuildingState,
    ) {
        let mut writer = frame_state.frame_gpu_data.f32.write_blocks(3 + self.brush_segments.len() * VECS_PER_SEGMENT);
        self.write_prim_gpu_blocks(&mut writer, &common.prim_size);
        self.write_segment_gpu_blocks(&mut writer);
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
        &self,
        writer: &mut GpuBufferWriterF,
    ) {
        for segment in &self.brush_segments {
            segment.write_gpu_blocks(writer);
        }
    }
}

pub type ImageBorderTemplate = PrimTemplate<ImageBorderData>;

impl From<ImageBorderKey> for ImageBorderTemplate {
    fn from(key: ImageBorderKey) -> Self {
        let common = PrimTemplateCommonData::with_key_common(key.common);

        let brush_segments = key.kind.nine_patch.create_brush_segments(common.prim_size);
        ImageBorderTemplate {
            common,
            kind: ImageBorderData {
                request: key.kind.request,
                brush_segments,
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
    assert_eq!(mem::size_of::<NormalBorderTemplate>(), 208, "NormalBorderTemplate size changed");
    assert_eq!(mem::size_of::<NormalBorderKey>(), 96, "NormalBorderKey size changed");
    assert_eq!(mem::size_of::<ImageBorder>(), 68, "ImageBorder size changed");
    assert_eq!(mem::size_of::<ImageBorderTemplate>(), 96, "ImageBorderTemplate size changed");
    assert_eq!(mem::size_of::<ImageBorderKey>(), 80, "ImageBorderKey size changed");
}
