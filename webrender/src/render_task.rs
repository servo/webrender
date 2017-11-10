/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{ClipId, DeviceIntPoint, DeviceIntRect, DeviceIntSize};
use api::{FilterOp, LayerPoint, LayerRect, MixBlendMode};
use api::{PipelineId, PremultipliedColorF};
use clip::{ClipSource, ClipSourcesWeakHandle, ClipStore};
use clip_scroll_tree::CoordinateSystemId;
use gpu_cache::GpuCacheHandle;
use gpu_types::{ClipScrollNodeIndex};
use internal_types::HardwareCompositeOp;
use prim_store::PrimitiveIndex;
use std::{cmp, usize, f32, i32};
use std::rc::Rc;
use tiling::{RenderPass, RenderTargetIndex};
use tiling::{RenderTargetKind, StackingContextIndex};

const FLOATS_PER_RENDER_TASK_INFO: usize = 12;
pub const MAX_BLUR_STD_DEVIATION: f32 = 4.0;
pub const MIN_DOWNSCALING_RT_SIZE: i32 = 128;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct RenderTaskId(pub u32); // TODO(gw): Make private when using GPU cache!

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct RenderTaskAddress(pub u32);

#[derive(Debug)]
pub struct RenderTaskTree {
    pub tasks: Vec<RenderTask>,
    pub task_data: Vec<RenderTaskData>,
}

pub type ClipChain = Option<Rc<ClipChainNode>>;

#[derive(Debug)]
pub struct ClipChainNode {
    pub work_item: ClipWorkItem,
    pub prev: ClipChain,
}

struct ClipChainNodeIter {
    current: ClipChain,
}

impl Iterator for ClipChainNodeIter {
    type Item = Rc<ClipChainNode>;

    fn next(&mut self) -> ClipChain {
        let previous = self.current.clone();
        self.current = match self.current {
            Some(ref item) => item.prev.clone(),
            None => return None,
        };
        previous
    }
}

impl RenderTaskTree {
    pub fn new() -> RenderTaskTree {
        RenderTaskTree {
            tasks: Vec::new(),
            task_data: Vec::new(),
        }
    }

    pub fn add(&mut self, task: RenderTask) -> RenderTaskId {
        let id = RenderTaskId(self.tasks.len() as u32);
        self.tasks.push(task);
        id
    }

    pub fn max_depth(&self, id: RenderTaskId, depth: usize, max_depth: &mut usize) {
        let depth = depth + 1;
        *max_depth = cmp::max(*max_depth, depth);
        let task = &self.tasks[id.0 as usize];
        for child in &task.children {
            self.max_depth(*child, depth, max_depth);
        }
    }

    pub fn assign_to_passes(
        &self,
        id: RenderTaskId,
        pass_index: usize,
        passes: &mut Vec<RenderPass>,
    ) {
        let task = &self.tasks[id.0 as usize];

        for child in &task.children {
            self.assign_to_passes(*child, pass_index - 1, passes);
        }

        // Sanity check - can be relaxed if needed
        match task.location {
            RenderTaskLocation::Fixed => {
                debug_assert!(pass_index == passes.len() - 1);
            }
            RenderTaskLocation::Dynamic(..) => {
                debug_assert!(pass_index < passes.len() - 1);
            }
        }

        // If this task can be shared between multiple
        // passes, render it in the first pass so that
        // it is available to all subsequent passes.
        let pass_index = if task.is_shared() {
            debug_assert!(task.children.is_empty());
            0
        } else {
            pass_index
        };

        let pass = &mut passes[pass_index];
        pass.add_render_task(id, task.get_dynamic_size(), task.target_kind());
    }

    pub fn get(&self, id: RenderTaskId) -> &RenderTask {
        &self.tasks[id.0 as usize]
    }

    pub fn get_mut(&mut self, id: RenderTaskId) -> &mut RenderTask {
        &mut self.tasks[id.0 as usize]
    }

    pub fn get_task_address(&self, id: RenderTaskId) -> RenderTaskAddress {
        let task = &self.tasks[id.0 as usize];
        match task.kind {
            RenderTaskKind::Alias(alias_id) => RenderTaskAddress(alias_id.0),
            _ => RenderTaskAddress(id.0),
        }
    }

    pub fn build(&mut self) {
        for task in &mut self.tasks {
            self.task_data.push(task.write_task_data());
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum RenderTaskKey {
    /// Draw the alpha mask for a shared clip.
    CacheMask(ClipId),
}

#[derive(Debug)]
pub enum RenderTaskLocation {
    Fixed,
    Dynamic(Option<(DeviceIntPoint, RenderTargetIndex)>, DeviceIntSize),
}

#[derive(Debug)]
pub enum AlphaRenderItem {
    Primitive(ClipScrollNodeIndex, ClipScrollNodeIndex, PrimitiveIndex, i32),
    Blend(StackingContextIndex, RenderTaskId, FilterOp, i32),
    Composite(
        StackingContextIndex,
        RenderTaskId,
        RenderTaskId,
        MixBlendMode,
        i32,
    ),
    SplitComposite(StackingContextIndex, RenderTaskId, GpuCacheHandle, i32),
    HardwareComposite(
        StackingContextIndex,
        RenderTaskId,
        HardwareCompositeOp,
        DeviceIntPoint,
        i32,
        DeviceIntSize,
    ),
}

#[derive(Debug)]
pub struct AlphaRenderTask {
    pub screen_origin: DeviceIntPoint,
    pub items: Vec<AlphaRenderItem>,
    // If this render task is a registered frame output, this
    // contains the pipeline ID it maps to.
    pub frame_output_pipeline_id: Option<PipelineId>,
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub enum MaskSegment {
    // This must match the SEGMENT_ values in clip_shared.glsl!
    All = 0,
    TopLeftCorner,
    TopRightCorner,
    BottomLeftCorner,
    BottomRightCorner,
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub enum MaskGeometryKind {
    Default, // Draw the entire rect
    CornersOnly, // Draw the corners (simple axis aligned mask)
             // TODO(gw): Add more types here (e.g. 4 rectangles outside the inner rect)
}

#[derive(Debug, Clone)]
pub struct ClipWorkItem {
    pub scroll_node_id: ClipScrollNodeIndex,
    pub clip_sources: ClipSourcesWeakHandle,
    pub coordinate_system_id: CoordinateSystemId,
}

impl ClipWorkItem {
    fn get_geometry_kind(
        &self,
        clip_store: &ClipStore,
        prim_coordinate_system_id: CoordinateSystemId
    ) -> MaskGeometryKind {
        let clips = clip_store
            .get_opt(&self.clip_sources)
            .expect("bug: clip handle should be valid")
            .clips();
        let mut rounded_rect_count = 0;

        for &(ref clip, _) in clips {
            match *clip {
                ClipSource::Rectangle(..) => {
                    if self.has_compatible_coordinate_system(prim_coordinate_system_id) {
                        return MaskGeometryKind::Default;
                    }
                },
                ClipSource::RoundedRectangle(..) => {
                    rounded_rect_count += 1;
                }
                ClipSource::Image(..) | ClipSource::BorderCorner(..) => {
                    return MaskGeometryKind::Default;
                }
            }
        }

        if rounded_rect_count == 1 {
            MaskGeometryKind::CornersOnly
        } else {
            MaskGeometryKind::Default
        }
    }

    fn has_compatible_coordinate_system(&self, other_id: CoordinateSystemId) -> bool {
        self.coordinate_system_id == other_id
    }
}

#[derive(Debug)]
pub struct CacheMaskTask {
    actual_rect: DeviceIntRect,
    inner_rect: DeviceIntRect,
    pub clips: Vec<ClipWorkItem>,
    pub geometry_kind: MaskGeometryKind,
    pub coordinate_system_id: CoordinateSystemId,
}

#[derive(Debug)]
pub struct PictureTask {
    pub prim_index: PrimitiveIndex,
    pub target_kind: RenderTargetKind,
    pub content_origin: LayerPoint,
    pub color: PremultipliedColorF,
}

#[derive(Debug)]
pub struct BlurTask {
    pub blur_std_deviation: f32,
    pub target_kind: RenderTargetKind,
    pub regions: Vec<LayerRect>,
    pub color: PremultipliedColorF,
    pub scale_factor: f32,
}

#[derive(Debug)]
pub struct RenderTaskData {
    pub data: [f32; FLOATS_PER_RENDER_TASK_INFO],
}

#[derive(Debug)]
pub enum RenderTaskKind {
    Alpha(AlphaRenderTask),
    Picture(PictureTask),
    CacheMask(CacheMaskTask),
    VerticalBlur(BlurTask),
    HorizontalBlur(BlurTask),
    Readback(DeviceIntRect),
    Alias(RenderTaskId),
    Scaling(RenderTargetKind),
}

#[derive(Debug, Copy, Clone)]
pub enum ClearMode {
    // Applicable to color and alpha targets.
    Zero,
    One,

    // Applicable to color targets only.
    Transparent,
}

#[derive(Debug)]
pub struct RenderTask {
    pub cache_key: Option<RenderTaskKey>,
    pub location: RenderTaskLocation,
    pub children: Vec<RenderTaskId>,
    pub kind: RenderTaskKind,
    pub clear_mode: ClearMode,
}

impl RenderTask {
    pub fn new_alpha_batch(
        screen_origin: DeviceIntPoint,
        location: RenderTaskLocation,
        frame_output_pipeline_id: Option<PipelineId>,
    ) -> Self {
        RenderTask {
            cache_key: None,
            children: Vec::new(),
            location,
            kind: RenderTaskKind::Alpha(AlphaRenderTask {
                screen_origin,
                items: Vec::new(),
                frame_output_pipeline_id,
            }),
            clear_mode: ClearMode::Transparent,
        }
    }

    pub fn new_dynamic_alpha_batch(
        rect: &DeviceIntRect,
        frame_output_pipeline_id: Option<PipelineId>,
    ) -> Self {
        let location = RenderTaskLocation::Dynamic(None, rect.size);
        Self::new_alpha_batch(rect.origin, location, frame_output_pipeline_id)
    }

    pub fn new_picture(
        size: DeviceIntSize,
        prim_index: PrimitiveIndex,
        target_kind: RenderTargetKind,
        content_origin: LayerPoint,
        color: PremultipliedColorF,
        clear_mode: ClearMode,
    ) -> Self {
        RenderTask {
            cache_key: None,
            children: Vec::new(),
            location: RenderTaskLocation::Dynamic(None, size),
            kind: RenderTaskKind::Picture(PictureTask {
                prim_index,
                target_kind,
                content_origin,
                color,
            }),
            clear_mode,
        }
    }

    pub fn new_readback(screen_rect: DeviceIntRect) -> Self {
        RenderTask {
            cache_key: None,
            children: Vec::new(),
            location: RenderTaskLocation::Dynamic(None, screen_rect.size),
            kind: RenderTaskKind::Readback(screen_rect),
            clear_mode: ClearMode::Transparent,
        }
    }

    pub fn new_mask(
        key: Option<ClipId>,
        task_rect: DeviceIntRect,
        raw_clips: ClipChain,
        extra_clip: ClipChain,
        prim_rect: DeviceIntRect,
        clip_store: &ClipStore,
        is_axis_aligned: bool,
        prim_coordinate_system_id: CoordinateSystemId,
    ) -> Option<Self> {
        // Filter out all the clip instances that don't contribute to the result
        let mut current_coordinate_system_id = prim_coordinate_system_id;
        let mut inner_rect = Some(task_rect);
        let clips: Vec<_> = ClipChainNodeIter { current: raw_clips }
            .chain(ClipChainNodeIter { current: extra_clip })
            .filter_map(|node| {
                let work_item = node.work_item.clone();

                // FIXME(1828): This is a workaround until we can fix the inconsistency between
                // the shader and the CPU code around how inner_rects are handled.
                if !node.work_item.has_compatible_coordinate_system(current_coordinate_system_id) {
                    current_coordinate_system_id = node.work_item.coordinate_system_id;
                    inner_rect = None;
                    return Some(work_item)
                }

                let clip_info = clip_store
                    .get_opt(&node.work_item.clip_sources)
                    .expect("bug: clip item should exist");
                debug_assert!(clip_info.has_clips());

                match clip_info.bounds.inner {
                    Some(ref inner) if !inner.device_rect.is_empty() => {
                        inner_rect = inner_rect.and_then(|r| r.intersection(&inner.device_rect));
                        if inner.device_rect.contains_rect(&task_rect) {
                            return None;
                        }
                    }
                    _ => inner_rect = None,
                }

                Some(work_item)
            })
            .collect();

        // Nothing to do, all clips are irrelevant for this case
        if clips.is_empty() {
            return None;
        }


        // TODO(gw): This optimization is very conservative for now.
        //           For now, only draw optimized geometry if it is
        //           a single aligned rect mask with rounded corners.
        //           In the future, we'll expand this to handle the
        //           more complex types of clip mask geometry.
        let mut geometry_kind = MaskGeometryKind::Default;
        if let Some(inner_rect) = inner_rect {
            // If the inner rect completely contains the primitive
            // rect, then this mask can't affect the primitive.
            if inner_rect.contains_rect(&prim_rect) {
                return None;
            }
            if is_axis_aligned && clips.len() == 1 {
                geometry_kind = clips[0].get_geometry_kind(clip_store, prim_coordinate_system_id);
            }
        }

        Some(RenderTask {
            cache_key: key.map(RenderTaskKey::CacheMask),
            children: Vec::new(),
            location: RenderTaskLocation::Dynamic(None, task_rect.size),
            kind: RenderTaskKind::CacheMask(CacheMaskTask {
                actual_rect: task_rect,
                inner_rect: inner_rect.unwrap_or(DeviceIntRect::zero()),
                clips,
                geometry_kind,
                coordinate_system_id: prim_coordinate_system_id,
            }),
            clear_mode: ClearMode::One,
        })
    }

    // Construct a render task to apply a blur to a primitive.
    // The render task chain that is constructed looks like:
    //
    //    PrimitiveCacheTask: Draw the primitives.
    //           ^
    //           |
    //    DownscalingTask(s): Each downscaling task reduces the size of render target to
    //           ^            half. Also reduce the std deviation to half until the std
    //           |            deviation less than 4.0.
    //           |
    //           |
    //    VerticalBlurTask: Apply the separable vertical blur to the primitive.
    //           ^
    //           |
    //    HorizontalBlurTask: Apply the separable horizontal blur to the vertical blur.
    //           |
    //           +---- This is stored as the input task to the primitive shader.
    //
    pub fn new_blur(
        blur_std_deviation: f32,
        src_task_id: RenderTaskId,
        render_tasks: &mut RenderTaskTree,
        target_kind: RenderTargetKind,
        regions: &[LayerRect],
        clear_mode: ClearMode,
        color: PremultipliedColorF,
    ) -> Self {
        // Adjust large std deviation value.
        let mut adjusted_blur_std_deviation = blur_std_deviation;
        let blur_target_size = render_tasks.get(src_task_id).get_dynamic_size();
        let mut adjusted_blur_target_size = blur_target_size;
        let mut downscaling_src_task_id = src_task_id;
        let mut scale_factor = 1.0;
        while adjusted_blur_std_deviation > MAX_BLUR_STD_DEVIATION {
            if adjusted_blur_target_size.width < MIN_DOWNSCALING_RT_SIZE ||
               adjusted_blur_target_size.height < MIN_DOWNSCALING_RT_SIZE {
                break;
            }
            adjusted_blur_std_deviation *= 0.5;
            scale_factor *= 2.0;
            adjusted_blur_target_size = (blur_target_size.to_f32() / scale_factor).to_i32();
            let downscaling_task = RenderTask::new_scaling(
                target_kind,
                downscaling_src_task_id,
                adjusted_blur_target_size
            );
            downscaling_src_task_id = render_tasks.add(downscaling_task);
        }
        scale_factor = blur_target_size.width as f32 / adjusted_blur_target_size.width as f32;

        let blur_task_v = RenderTask {
            cache_key: None,
            children: vec![downscaling_src_task_id],
            location: RenderTaskLocation::Dynamic(None, adjusted_blur_target_size),
            kind: RenderTaskKind::VerticalBlur(BlurTask {
                blur_std_deviation: adjusted_blur_std_deviation,
                target_kind,
                regions: regions.to_vec(),
                color,
                scale_factor,
            }),
            clear_mode,
        };

        let blur_task_v_id = render_tasks.add(blur_task_v);

        let blur_task_h = RenderTask {
            cache_key: None,
            children: vec![blur_task_v_id],
            location: RenderTaskLocation::Dynamic(None, adjusted_blur_target_size),
            kind: RenderTaskKind::HorizontalBlur(BlurTask {
                blur_std_deviation: adjusted_blur_std_deviation,
                target_kind,
                regions: regions.to_vec(),
                color,
                scale_factor,
            }),
            clear_mode,
        };

        blur_task_h
    }

    pub fn new_scaling(
        target_kind: RenderTargetKind,
        src_task_id: RenderTaskId,
        target_size: DeviceIntSize,
    ) -> Self {
        RenderTask {
            cache_key: None,
            children: vec![src_task_id],
            location: RenderTaskLocation::Dynamic(None, target_size),
            kind: RenderTaskKind::Scaling(target_kind),
            clear_mode: match target_kind {
                RenderTargetKind::Color => ClearMode::Transparent,
                RenderTargetKind::Alpha => ClearMode::One,
            },
        }
    }

    pub fn as_alpha_batch_mut<'a>(&'a mut self) -> &'a mut AlphaRenderTask {
        match self.kind {
            RenderTaskKind::Alpha(ref mut task) => task,
            RenderTaskKind::Picture(..) |
            RenderTaskKind::CacheMask(..) |
            RenderTaskKind::VerticalBlur(..) |
            RenderTaskKind::Readback(..) |
            RenderTaskKind::HorizontalBlur(..) |
            RenderTaskKind::Alias(..) |
            RenderTaskKind::Scaling(..) => unreachable!(),
        }
    }

    pub fn as_alpha_batch<'a>(&'a self) -> &'a AlphaRenderTask {
        match self.kind {
            RenderTaskKind::Alpha(ref task) => task,
            RenderTaskKind::Picture(..) |
            RenderTaskKind::CacheMask(..) |
            RenderTaskKind::VerticalBlur(..) |
            RenderTaskKind::Readback(..) |
            RenderTaskKind::HorizontalBlur(..) |
            RenderTaskKind::Alias(..) |
            RenderTaskKind::Scaling(..) => unreachable!(),
        }
    }

    // Write (up to) 8 floats of data specific to the type
    // of render task that is provided to the GPU shaders
    // via a vertex texture.
    pub fn write_task_data(&self) -> RenderTaskData {
        // NOTE: The ordering and layout of these structures are
        //       required to match both the GPU structures declared
        //       in prim_shared.glsl, and also the uses in submit_batch()
        //       in renderer.rs.
        // TODO(gw): Maybe there's a way to make this stuff a bit
        //           more type-safe. Although, it will always need
        //           to be kept in sync with the GLSL code anyway.

        match self.kind {
            RenderTaskKind::Alpha(ref task) => {
                let (target_rect, target_index) = self.get_target_rect();
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
            RenderTaskKind::Picture(ref task) => {
                let (target_rect, target_index) = self.get_target_rect();
                RenderTaskData {
                    data: [
                        target_rect.origin.x as f32,
                        target_rect.origin.y as f32,
                        target_rect.size.width as f32,
                        target_rect.size.height as f32,
                        target_index.0 as f32,
                        task.content_origin.x,
                        task.content_origin.y,
                        0.0,
                        task.color.r,
                        task.color.g,
                        task.color.b,
                        task.color.a,
                    ],
                }
            }
            RenderTaskKind::CacheMask(ref task) => {
                let (target_rect, target_index) = self.get_target_rect();
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
            RenderTaskKind::VerticalBlur(ref task) |
            RenderTaskKind::HorizontalBlur(ref task) => {
                let (target_rect, target_index) = self.get_target_rect();
                RenderTaskData {
                    data: [
                        target_rect.origin.x as f32,
                        target_rect.origin.y as f32,
                        target_rect.size.width as f32,
                        target_rect.size.height as f32,
                        target_index.0 as f32,
                        task.blur_std_deviation,
                        task.scale_factor,
                        0.0,
                        task.color.r,
                        task.color.g,
                        task.color.b,
                        task.color.a,
                    ],
                }
            }
            RenderTaskKind::Readback(..) |
            RenderTaskKind::Scaling(..) => {
                let (target_rect, target_index) = self.get_target_rect();
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
            RenderTaskKind::Alias(..) => RenderTaskData { data: [0.0; 12] },
        }
    }

    pub fn inflate(&mut self, device_radius: i32) {
        match self.kind {
            RenderTaskKind::Alpha(ref mut info) => {
                match self.location {
                    RenderTaskLocation::Fixed => {
                        panic!("bug: inflate only supported for dynamic tasks");
                    }
                    RenderTaskLocation::Dynamic(_, ref mut size) => {
                        size.width += device_radius * 2;
                        size.height += device_radius * 2;
                        info.screen_origin.x -= device_radius;
                        info.screen_origin.y -= device_radius;
                    }
                }
            }

            RenderTaskKind::Readback(..) |
            RenderTaskKind::CacheMask(..) |
            RenderTaskKind::VerticalBlur(..) |
            RenderTaskKind::HorizontalBlur(..) |
            RenderTaskKind::Picture(..) |
            RenderTaskKind::Alias(..) |
            RenderTaskKind::Scaling(..) => {
                panic!("bug: inflate only supported for alpha tasks");
            }
        }
    }

    pub fn get_dynamic_size(&self) -> DeviceIntSize {
        match self.location {
            RenderTaskLocation::Fixed => DeviceIntSize::zero(),
            RenderTaskLocation::Dynamic(_, size) => size,
        }
    }

    pub fn get_target_rect(&self) -> (DeviceIntRect, RenderTargetIndex) {
        match self.location {
            RenderTaskLocation::Fixed => (DeviceIntRect::zero(), RenderTargetIndex(0)),
            RenderTaskLocation::Dynamic(origin_and_target_index, size) => {
                let (origin, target_index) =
                    origin_and_target_index.expect("Should have been allocated by now!");
                (DeviceIntRect::new(origin, size), target_index)
            }
        }
    }

    pub fn target_kind(&self) -> RenderTargetKind {
        match self.kind {
            RenderTaskKind::Alpha(..) |
            RenderTaskKind::Readback(..) => RenderTargetKind::Color,

            RenderTaskKind::CacheMask(..) => {
                RenderTargetKind::Alpha
            }

            RenderTaskKind::VerticalBlur(ref task_info) |
            RenderTaskKind::HorizontalBlur(ref task_info) => {
                task_info.target_kind
            }

            RenderTaskKind::Scaling(target_kind) => {
                target_kind
            }

            RenderTaskKind::Picture(ref task_info) => {
                task_info.target_kind
            }

            RenderTaskKind::Alias(..) => {
                panic!("BUG: target_kind() called on invalidated task");
            }
        }
    }

    // Check if this task wants to be made available as an input
    // to all passes (except the first) in the render task tree.
    // To qualify for this, the task needs to have no children / dependencies.
    // Currently, this is only supported for A8 targets, but it can be
    // trivially extended to also support RGBA8 targets in the future
    // if we decide that is useful.
    pub fn is_shared(&self) -> bool {
        match self.kind {
            RenderTaskKind::Alpha(..) |
            RenderTaskKind::Picture(..) |
            RenderTaskKind::VerticalBlur(..) |
            RenderTaskKind::Readback(..) |
            RenderTaskKind::HorizontalBlur(..) |
            RenderTaskKind::Scaling(..) => false,

            RenderTaskKind::CacheMask(..) => true,

            RenderTaskKind::Alias(..) => {
                panic!("BUG: is_shared() called on aliased task");
            }
        }
    }

    pub fn set_alias(&mut self, id: RenderTaskId) {
        debug_assert!(self.cache_key.is_some());
        // TODO(gw): We can easily handle invalidation of tasks that
        //           contain children in the future. Since we don't
        //           have any cases of that yet, just assert to simplify
        //           the current implementation.
        debug_assert!(self.children.is_empty());
        self.kind = RenderTaskKind::Alias(id);
    }
}
