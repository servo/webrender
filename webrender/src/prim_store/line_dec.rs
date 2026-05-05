/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{
    ColorF, ColorU, RasterSpace,
    LineOrientation, LineStyle, PremultipliedColorF, Shadow,
};
use api::units::*;
use euclid::Scale;
use crate::gpu_types::ImageBrushPrimitiveData;
use crate::render_task::{RenderTask, RenderTaskKind};
use crate::render_task_cache::{RenderTaskCacheKey, RenderTaskCacheKeyKind, RenderTaskParent};
use crate::render_task_graph::RenderTaskId;
use crate::renderer::GpuBufferAddress;
use crate::scene_building::{CreateShadow, IsVisible};
use crate::frame_builder::{FrameBuildingContext, FrameBuildingState};
use crate::intern;
use crate::internal_types::LayoutPrimitiveInfo;
use crate::prim_store::{
    PrimKey, PrimTemplate, PrimTemplateCommonData,
    InternablePrimitive, PrimitiveStore,
};
use crate::prim_store::PrimitiveKind;
use crate::spatial_tree::SpatialNodeIndex;
use crate::util::clamp_to_scale_factor;

/// Maximum resolution in device pixels at which line decorations are rasterized.
pub const MAX_LINE_DECORATION_RESOLUTION: u32 = 4096;

/// Per-frame scratch data for a LineDecoration primitive.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
pub struct LineDecorationScratch {
    pub task_id: RenderTaskId,
    pub gpu_address: GpuBufferAddress,
}

#[derive(Clone, Debug, Hash, MallocSizeOf, PartialEq, Eq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct LineDecorationCacheKey {
    pub style: LineStyle,
    pub orientation: LineOrientation,
    pub wavy_line_thickness: Au,
    pub size: LayoutSizeAu,
}

/// Identifying key for a line decoration. The mask tile size
/// (`LineDecorationCacheKey`) is intentionally not part of the intern
/// key — it is a deterministic function of `style`, `orientation`,
/// `wavy_line_thickness`, and the prim's per-frame local rect, and is
/// rebuilt during frame building.
#[derive(Clone, Debug, Hash, MallocSizeOf, PartialEq, Eq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct LineDecoration {
    pub style: LineStyle,
    pub orientation: LineOrientation,
    pub wavy_line_thickness: Au,
    pub color: ColorU,
}

pub type LineDecorationKey = PrimKey<LineDecoration>;

impl LineDecorationKey {
    pub fn new(
        info: &LayoutPrimitiveInfo,
        line_dec: LineDecoration,
    ) -> Self {
        LineDecorationKey {
            common: info.into(),
            kind: line_dec,
        }
    }
}

impl intern::InternDebug for LineDecorationKey {}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(MallocSizeOf)]
pub struct LineDecorationData {
    pub style: LineStyle,
    pub orientation: LineOrientation,
    pub wavy_line_thickness: Au,
    pub color: ColorF,
}

impl LineDecorationData {
    /// Per-frame preparation: derive the mask tile size from the prim's
    /// current local size, write the GPU block, and (for non-solid styles)
    /// allocate a cached render task. Returns the task id and the per-
    /// instance GPU buffer address consumed by `batch.rs`.
    pub fn prepare(
        &self,
        prim_size: LayoutSize,
        prim_spatial_node_index: SpatialNodeIndex,
        frame_context: &FrameBuildingContext,
        frame_state: &mut FrameBuildingState,
    ) -> (RenderTaskId, GpuBufferAddress) {
        let cache_key = get_line_decoration_size(
            &prim_size,
            self.orientation,
            self.style,
            self.wavy_line_thickness.to_f32_px(),
        ).map(|size| LineDecorationCacheKey {
            style: self.style,
            orientation: self.orientation,
            wavy_line_thickness: self.wavy_line_thickness,
            size: size.to_au(),
        });

        let mut writer = frame_state.frame_gpu_data.f32.write_blocks(3);
        match cache_key.as_ref() {
            Some(cache_key) => {
                writer.push(&ImageBrushPrimitiveData {
                    color: self.color.premultiplied(),
                    background_color: PremultipliedColorF::WHITE,
                    stretch_size: LayoutSize::new(
                        cache_key.size.width.to_f32_px(),
                        cache_key.size.height.to_f32_px(),
                    ),
                });
            }
            None => {
                writer.push_one(self.color.premultiplied());
            }
        }
        let gpu_address = writer.finish();

        let task_id = match cache_key {
            Some(cache_key) => self.allocate_render_task(
                cache_key,
                prim_spatial_node_index,
                frame_context,
                frame_state,
            ),
            None => RenderTaskId::INVALID,
        };

        (task_id, gpu_address)
    }

    fn allocate_render_task(
        &self,
        cache_key: LineDecorationCacheKey,
        prim_spatial_node_index: SpatialNodeIndex,
        frame_context: &FrameBuildingContext,
        frame_state: &mut FrameBuildingState,
    ) -> RenderTaskId {
        // TODO(gw): These scale factors don't do a great job if the world transform
        //           contains perspective
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
        let scale_factor = LayoutToDeviceScale::new(scale_width.max(scale_height));

        let task_size_f = (LayoutSize::from_au(cache_key.size) * scale_factor).ceil();
        let mut task_size = if task_size_f.width > MAX_LINE_DECORATION_RESOLUTION as f32 ||
            task_size_f.height > MAX_LINE_DECORATION_RESOLUTION as f32 {
                let max_extent = task_size_f.width.max(task_size_f.height);
                let task_scale_factor = Scale::new(MAX_LINE_DECORATION_RESOLUTION as f32 / max_extent);
                let task_size = (LayoutSize::from_au(cache_key.size) * scale_factor * task_scale_factor)
                            .ceil().to_i32();
            task_size
        } else {
            task_size_f.to_i32()
        };

        // It's plausible, due to float accuracy issues that the line decoration may be considered
        // visible even if the scale factors are ~0. However, the render task allocation below requires
        // that the size of the task is > 0. To work around this, ensure that the task size is at least
        // 1x1 pixels
        task_size.width = task_size.width.max(1);
        task_size.height = task_size.height.max(1);

        // Request a pre-rendered image task.
        frame_state.resource_cache.request_render_task(
            Some(RenderTaskCacheKey {
                origin: DeviceIntPoint::zero(),
                size: task_size,
                kind: RenderTaskCacheKeyKind::LineDecoration(cache_key.clone()),
            }),
            false,
            RenderTaskParent::Surface,
            &mut frame_state.frame_gpu_data.f32,
            frame_state.rg_builder,
            &mut frame_state.surface_builder,
            &mut |rg_builder, _| {
                rg_builder.add().init(RenderTask::new_dynamic(
                    task_size,
                    RenderTaskKind::new_line_decoration(
                        cache_key.style,
                        cache_key.orientation,
                        cache_key.wavy_line_thickness.to_f32_px(),
                        LayoutSize::from_au(cache_key.size),
                    ),
                ))
            }
        )
    }
}

pub type LineDecorationTemplate = PrimTemplate<LineDecorationData>;

impl From<LineDecorationKey> for LineDecorationTemplate {
    fn from(line_dec: LineDecorationKey) -> Self {
        let common = PrimTemplateCommonData::with_key_common(line_dec.common);
        LineDecorationTemplate {
            common,
            kind: LineDecorationData {
                style: line_dec.kind.style,
                orientation: line_dec.kind.orientation,
                wavy_line_thickness: line_dec.kind.wavy_line_thickness,
                color: line_dec.kind.color.into(),
            }
        }
    }
}

pub type LineDecorationDataHandle = intern::Handle<LineDecoration>;

impl intern::Internable for LineDecoration {
    type Key = LineDecorationKey;
    type StoreData = LineDecorationTemplate;
    type InternData = ();
    const PROFILE_COUNTER: usize = crate::profiler::INTERNED_LINE_DECORATIONS;
}

impl InternablePrimitive for LineDecoration {
    fn into_key(
        self,
        info: &LayoutPrimitiveInfo,
    ) -> LineDecorationKey {
        LineDecorationKey::new(
            info,
            self,
        )
    }

    fn make_instance_kind(
        _key: LineDecorationKey,
        data_handle: LineDecorationDataHandle,
        _: &mut PrimitiveStore,
    ) -> PrimitiveKind {
        PrimitiveKind::LineDecoration {
            data_handle,
        }
    }
}

impl CreateShadow for LineDecoration {
    fn create_shadow(
        &self,
        shadow: &Shadow,
        _: bool,
        _: RasterSpace,
    ) -> Self {
        LineDecoration {
            style: self.style,
            orientation: self.orientation,
            wavy_line_thickness: self.wavy_line_thickness,
            color: shadow.color.into(),
        }
    }
}

impl IsVisible for LineDecoration {
    fn is_visible(&self) -> bool {
        self.color.a > 0
    }
}

/// Choose the decoration mask tile size for a given line.
///
/// Given a line with overall size `rect_size` and the given `orientation`,
/// return the dimensions of a single mask tile for the decoration pattern
/// described by `style` and `wavy_line_thickness`.
///
/// If `style` is `Solid`, no mask tile is necessary; return `None`. The other
/// styles each have their own characteristic periods of repetition, so for each
/// one, this function returns a `LayoutSize` with the right aspect ratio and
/// whose specific size is convenient for the `cs_line_decoration.glsl` fragment
/// shader to work with. The shader uses a local coordinate space in which the
/// tile fills a rectangle with one corner at the origin, and with the size this
/// function returns.
///
/// The returned size is not necessarily in pixels; device scaling and other
/// concerns can still affect the actual task size.
///
/// Regardless of whether `orientation` is `Vertical` or `Horizontal`, the
/// `width` and `height` of the returned size are always horizontal and
/// vertical, respectively.
pub fn get_line_decoration_size(
    rect_size: &LayoutSize,
    orientation: LineOrientation,
    style: LineStyle,
    wavy_line_thickness: f32,
) -> Option<LayoutSize> {
    let h = match orientation {
        LineOrientation::Horizontal => rect_size.height,
        LineOrientation::Vertical => rect_size.width,
    };

    // TODO(gw): The formulae below are based on the existing gecko and line
    //           shader code. They give reasonable results for most inputs,
    //           but could definitely do with a detailed pass to get better
    //           quality on a wider range of inputs!
    //           See nsCSSRendering::PaintDecorationLine in Gecko.

    let (parallel, perpendicular) = match style {
        LineStyle::Solid => {
            return None;
        }
        LineStyle::Dashed => {
            let dash_length = (3.0 * h).min(64.0).max(1.0);

            (2.0 * dash_length, 4.0)
        }
        LineStyle::Dotted => {
            let diameter = h.min(64.0).max(1.0);
            let period = 2.0 * diameter;

            (period, diameter)
        }
        LineStyle::Wavy => {
            let line_thickness = wavy_line_thickness.max(1.0);
            let slope_length = h - line_thickness;
            let flat_length = ((line_thickness - 1.0) * 2.0).max(1.0);
            let approx_period = 2.0 * (slope_length + flat_length);

            (approx_period, h)
        }
    };

    Some(match orientation {
        LineOrientation::Horizontal => LayoutSize::new(parallel, perpendicular),
        LineOrientation::Vertical => LayoutSize::new(perpendicular, parallel),
    })
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
    assert_eq!(mem::size_of::<LineDecoration>(), 12, "LineDecoration size changed");
    assert_eq!(mem::size_of::<LineDecorationTemplate>(), 32, "LineDecorationTemplate size changed");
    assert_eq!(mem::size_of::<LineDecorationKey>(), 16, "LineDecorationKey size changed");
}
