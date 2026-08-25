/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{ColorF, NormalBorder, RepeatMode};
use api::units::*;
use smallvec::SmallVec;
use crate::border::{build_border_instances, NormalBorderSegment, MAX_BORDER_RESOLUTION};
use crate::clip::{ClipChainInstance, ClipIntern};
use crate::command_buffer::CommandBufferIndex;
use crate::pattern::image::ImagePattern;
use crate::quad::{self, QuadDescriptor, QuadTransformState};
use crate::visibility::PrimitiveDrawIndex;
use crate::render_task_cache::{RenderTaskCacheKey, RenderTaskCacheKeyKind, RenderTaskParent, to_cache_size};
use crate::scene_building::{IsVisible};
use crate::frame_builder::{FrameBuildingContext, FrameBuildingState, PictureContext};
use crate::intern::{self, DataStore};
use crate::internal_types::LayoutPrimitiveInfo;
use crate::prim_store::{
    InternablePrimitive, NinePatchDescriptor, PrimKey, PrimTemplate, PrimTemplateCommonData, PrimitiveKind, PrimitiveScratchBuffer, PrimitiveStore
};
use crate::resource_cache::ImageRequest;
use crate::render_task::{RenderTask, RenderTaskKind};
use crate::render_task_graph::RenderTaskId;
use crate::spatial_tree::SpatialNodeIndex;
use crate::util::clamp_to_scale_factor;

// `NormalBorderPrim` now lives in `webrender_api::interned_prims` so content-process
// interning can hold it. Re-exported to keep existing references working.
pub use api::interned_prims::NormalBorderPrim;

pub type NormalBorderKey = PrimKey<NormalBorderPrim>;

impl intern::InternDebug for NormalBorderKey {}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(MallocSizeOf)]
pub struct NormalBorderData {
    pub border: NormalBorder,
    pub widths: LayoutSideOffsets,
}

impl NormalBorderData {
    pub fn update(
        &self,
        desc: &QuadDescriptor,
        clip_chain: &ClipChainInstance,
        prim_spatial_node_index: SpatialNodeIndex,
        device_pixel_scale: DevicePixelScale,
        draw_index: PrimitiveDrawIndex,
        quad_transform: &mut QuadTransformState,
        frame_context: &FrameBuildingContext,
        pic_context: &PictureContext,
        targets: &[CommandBufferIndex],
        interned_clips: &DataStore<ClipIntern>,
        frame_state: &mut FrameBuildingState,
        scratch: &mut PrimitiveScratchBuffer,
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
        // Snap the thickness of a border the author declared at >= 1 CSS pixel
        // to a whole device pixel. Border edges are composited onto their
        // layout-space rects, so a transform makes a 1px edge a fractional
        // device thickness: under a downscale it can shrink below a device
        // pixel and be antialiased away entirely, until whole sides of the
        // border vanish (bug 1258112); at other scales the four sides land at
        // different sub-pixel phases and render with visibly uneven thickness
        // (bug 1950029). Rounding the device thickness to the nearest pixel
        // (floored at 1 so a real border can't disappear) makes every side a
        // consistent whole-pixel width. Genuinely sub-CSS-pixel edges are left
        // untouched. Uses the unclamped world scale factors, since that is the
        // transform the edge is actually composited with, not the power-of-2
        // rasterization scale.
        let snap_width = |w: f32, s: f32| {
            if w >= 1.0 && s > 0.0 { (w * s).round().max(1.0) / s } else { w }
        };
        let device_scale_x = scale.0 * device_pixel_scale.0;
        let device_scale_y = scale.1 * device_pixel_scale.0;
        let mut widths = self.widths;
        widths.left = snap_width(widths.left, device_scale_x);
        widths.right = snap_width(widths.right, device_scale_x);
        widths.top = snap_width(widths.top, device_scale_y);
        widths.bottom = snap_width(widths.bottom, device_scale_y);

        let scale_width = clamp_to_scale_factor(scale.0, false);
        let scale_height = clamp_to_scale_factor(scale.1, false);
        // Pick the maximum dimension as scale
        let world_scale = LayoutToWorldScale::new(scale_width.max(scale_height));
        let mut scale = world_scale * device_pixel_scale;

        // Build the per-frame border segments up front so we can clamp the
        // rasterization scale against the largest segment before requesting
        // any render tasks. Capping the scale renders very large corners at a
        // lower resolution and stretches them: the right shape, but blurrier.
        let mut segments: SmallVec<[NormalBorderSegment; 8]> = SmallVec::new();
        crate::border::create_border_segments(
            desc.pattern_rect,
            &self.border,
            &widths,
            &mut |segment| segments.push(segment.clone()),
        );

        let mut max_dim = 1.0;
        for segment in &segments {
            if segment.is_solid.is_none() {
                max_dim = segment.task_size.width.max(segment.task_size.height.max(max_dim));
            }
        }
        let max_scale = LayoutToDeviceScale::new(MAX_BORDER_RESOLUTION as f32 / max_dim);
        scale.0 = scale.0.min(max_scale.0);

        for segment in &segments {
            let segment_bounds = |extent: &LayoutRect| {
                let mut bounds = desc.bounds.intersection_unchecked(extent);
                if let Some(clip_rect) = segment.clip_rect {
                    bounds = bounds.intersection_unchecked(&clip_rect);
                }
                bounds
            };

            if let Some(color) = &segment.is_solid {
                quad::prepare_quad(
                    color,
                    &QuadDescriptor {
                        pattern_rect: segment.pattern_rect,
                        bounds: segment_bounds(&segment.pattern_rect),
                        aligned_aa_edges: desc.aligned_aa_edges & segment.edge_flags,
                        transformed_aa_edges: desc.transformed_aa_edges & segment.edge_flags,
                    },
                    draw_index,
                    &None,
                    clip_chain,
                    quad_transform,
                    frame_context,
                    pic_context,
                    targets,
                    interned_clips,
                    frame_state,
                    scratch,
                );

                continue;
            }

            // Update the cache key device size based on requested scale.
            let cache_size = to_cache_size(segment.task_size, &mut scale);
            let cache_key = RenderTaskCacheKey {
                kind: RenderTaskCacheKeyKind::BorderSegment(segment.cache_key.clone()),
                origin: DeviceIntPoint::zero(),
                size: cache_size,
            };

            // TODO(gw): We don't calculate opacity for borders yet!
            let is_opaque = false;

            let task_id = frame_state.resource_cache.request_render_task(
                Some(cache_key),
                is_opaque,
                RenderTaskParent::Surface,
                &mut frame_state.frame_gpu_data.f32,
                frame_state.rg_builder,
                &mut frame_state.surface_builder,
                &mut |rg_builder, gpu_buffer_builder| {
                    rg_builder.add().init(RenderTask::new_dynamic(
                        cache_size,
                        RenderTaskKind::new_border_segment(
                            build_border_instances(
                                &segment.cache_key,
                                cache_size,
                                &self.border,
                                scale,
                                gpu_buffer_builder,
                            )
                        ),
                    ))
                }
            );

            let pattern = ImagePattern {
                src_task_id: task_id,
                src_is_opaque: is_opaque,
                premultiplied: true,
                sampler_kind: api::ImageBufferKind::Texture2D,
                color: ColorF::WHITE,
            };

            // The texture is drawn across the full segment rect (for
            // corners that is the natural corner-image size, which may
            // extend past the visible area). `clip_rect` crops it back to
            // the visible part for corners whose adjacent corner overlaps.
            let segment_pattern_rect = segment.pattern_rect;

            let mut stretch_size = segment_pattern_rect.size();
            let mut spacing = LayoutSize::zero();
            let mut _repeat_offset = LayoutVector2D::zero();
            crate::border::compute_border_repetition(
                segment_pattern_rect.size(),
                cache_size.to_f32(),
                segment.repeat_x,
                segment.repeat_y,
                &mut stretch_size,
                &mut spacing,
                &mut _repeat_offset,
            );

            // The positioning and size of the dashes and dots is not specified
            // but browsers are encouraged to make the pattern symetrical.
            // One way to do this is to apply the repeat offset computed
            // by compute_border_repetition. However the pattern that we
            // are repeating is meant to be instead stretched to so that
            // an integer number of repetitions fills the space.

            if segment.repeat_x == RepeatMode::Repeat {
                let w = segment_pattern_rect.width();
                let sw = stretch_size.width;
                let scale = w / ((w / sw).round() * sw);

                stretch_size.width *= scale;
            }

            if segment.repeat_y == RepeatMode::Repeat {
                let h = segment_pattern_rect.height();
                let sh = stretch_size.height;
                let scale = h / ((h / sh).round() * sh);

                stretch_size.height *= scale;
            }

            quad::prepare_repeatable_quad(
                &pattern,
                &QuadDescriptor {
                    pattern_rect: segment_pattern_rect,
                    bounds: segment_bounds(&segment_pattern_rect),
                    aligned_aa_edges: desc.aligned_aa_edges & segment.edge_flags,
                    transformed_aa_edges: desc.transformed_aa_edges & segment.edge_flags,
                },
                stretch_size,
                spacing,
                draw_index,
                &None,
                clip_chain,
                quad_transform,
                frame_context,
                pic_context,
                targets,
                interned_clips,
                frame_state,
                scratch,
            );
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
            info.into(),
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


impl IsVisible for NormalBorderPrim {
    fn is_visible(&self) -> bool {
        true
    }
}

////////////////////////////////////////////////////////////////////////////////

// `ImageBorder` now lives in `webrender_api::interned_prims` (with the image
// request inlined as key/rendering/tile so the value is api-resident). The
// frame-time `ImageBorderData` below rebuilds the `ImageRequest`.
pub use api::interned_prims::ImageBorder;

pub type ImageBorderKey = PrimKey<ImageBorder>;

impl intern::InternDebug for ImageBorderKey {}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(MallocSizeOf)]
pub struct ImageBorderData {
    #[ignore_malloc_size_of = "Arc"]
    pub request: ImageRequest,
    pub nine_patch: NinePatchDescriptor,
}

impl ImageBorderData {
    pub fn update(
        &self,
        frame_state: &mut FrameBuildingState,
    ) -> (RenderTaskId, DeviceIntSize, bool) {
        let size = frame_state.resource_cache.request_image(
            self.request,
            &mut frame_state.frame_gpu_data.f32,
        );

        let task_id = frame_state.rg_builder.add().init(
            RenderTask::new_image(size, self.request, false)
        );

        let is_opaque = frame_state
            .resource_cache
            .get_image_properties(self.request.key)
            .map(|properties| properties.descriptor.is_opaque())
            .unwrap_or(true);

        (task_id, size, is_opaque)
    }
}

pub type ImageBorderTemplate = PrimTemplate<ImageBorderData>;

impl From<ImageBorderKey> for ImageBorderTemplate {
    fn from(key: ImageBorderKey) -> Self {
        let common = PrimTemplateCommonData::with_key_common(key.common);

        ImageBorderTemplate {
            common,
            kind: ImageBorderData {
                request: ImageRequest {
                    key: key.kind.key,
                    rendering: key.kind.rendering,
                    tile: key.kind.tile,
                },
                nine_patch: key.kind.nine_patch,
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
            info.into(),
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
    assert_eq!(mem::size_of::<NormalBorderPrim>(), 116, "NormalBorderPrim size changed");
    assert_eq!(mem::size_of::<NormalBorderTemplate>(), 168, "NormalBorderTemplate size changed");
    assert_eq!(mem::size_of::<NormalBorderKey>(), 120, "NormalBorderKey size changed");
    assert_eq!(mem::size_of::<ImageBorder>(), 68, "ImageBorder size changed");
    assert_eq!(mem::size_of::<ImageBorderTemplate>(), 72, "ImageBorderTemplate size changed");
    assert_eq!(mem::size_of::<ImageBorderKey>(), 72, "ImageBorderKey size changed");
}
