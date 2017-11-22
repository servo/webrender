/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{BorderRadius, BorderRadiusKind, ColorF, ClipAndScrollInfo, FilterOp, MixBlendMode};
use api::{device_length, DeviceIntRect, DeviceIntSize, PipelineId};
use api::{BoxShadowClipMode, LayerPoint, LayerRect, LayerSize, LayerVector2D, Shadow};
use api::{ClipId, PremultipliedColorF};
use box_shadow::BLUR_SAMPLE_SCALE;
use frame_builder::PrimitiveContext;
use gpu_cache::GpuDataRequest;
use ordered_float::{OrderedFloat};
use prim_store::{PrimitiveIndex, PrimitiveRun, PrimitiveRunLocalRect};
use render_task::{ClearMode, RenderTask, RenderTaskId, RenderTaskTree};
use scene::{FilterOpHelpers, SceneProperties};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tiling::RenderTargetKind;

/*
 A picture represents a dynamically rendered image. It consists of:

 * A number of primitives that are drawn onto the picture.
 * A composite operation describing how to composite this
   picture into its parent.
 * A configuration describing how to draw the primitives on
   this picture (e.g. in screen space or local space).
 */

/// Specifies how this Picture should be composited
/// onto the target it belongs to.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum PictureCompositeMode {
    /// Apply CSS mix-blend-mode effect.
    MixBlend(MixBlendMode),
    /// Apply a CSS filter.
    Filter(FilterOp),
    /// Draw to intermediate surface, copy straight across. This
    /// is used for CSS isolation, and plane splitting.
    Blit,
}

/// Configure whether the primitives on this picture
/// should be rasterized in screen space or local space.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum RasterizationSpace {
    Local = 0,
    Screen = 1,
}

#[derive(Debug)]
pub enum PictureKind {
    TextShadow {
        offset: LayerVector2D,
        color: ColorF,
        blur_radius: f32,
        content_rect: LayerRect,
    },
    BoxShadow {
        blur_radius: f32,
        color: ColorF,
        blur_regions: Vec<LayerRect>,
        clip_mode: BoxShadowClipMode,
        radii_kind: BorderRadiusKind,
        content_rect: LayerRect,
        box_offset: LayerVector2D,
        border_radius: BorderRadius,
        spread_radius: f32,
    },
    Image {
        // If a mix-blend-mode, contains the render task for
        // the readback of the framebuffer that we use to sample
        // from in the mix-blend-mode shader.
        readback_render_task_id: Option<RenderTaskId>,
        /// How this picture should be composited.
        /// If None, don't composite - just draw directly on parent surface.
        composite_mode: Option<PictureCompositeMode>,
        // If true, this picture is part of a 3D context.
        is_in_3d_context: bool,
        // If requested as a frame output (for rendering
        // pages to a texture), this is the pipeline this
        // picture is the root of.
        frame_output_pipeline_id: Option<PipelineId>,
        // The original reference frame ID for this picture.
        // It is only different if this is part of a 3D
        // rendering context.
        reference_frame_id: ClipId,
        real_local_rect: LayerRect,
    },
}

#[derive(Debug)]
pub struct PicturePrimitive {
    // If this picture is drawn to an intermediate surface,
    // the associated render task.
    pub render_task_id: Option<RenderTaskId>,

    // Details specific to this type of picture.
    pub kind: PictureKind,

    // List of primitive runs that make up this picture.
    pub runs: Vec<PrimitiveRun>,

    // The pipeline that the primitives on this picture belong to.
    pub pipeline_id: PipelineId,

    // If true, apply visibility culling to primitives on this
    // picture. For text shadows and box shadows, we want to
    // unconditionally draw them.
    pub cull_children: bool,

    /// Configure whether the primitives on this picture
    /// should be rasterized in screen space or local space.
    pub rasterization_kind: RasterizationSpace,
}

impl PicturePrimitive {
    pub fn new_text_shadow(shadow: Shadow, pipeline_id: PipelineId) -> Self {
        PicturePrimitive {
            runs: Vec::new(),
            render_task_id: None,
            kind: PictureKind::TextShadow {
                offset: shadow.offset,
                color: shadow.color,
                blur_radius: shadow.blur_radius,
                content_rect: LayerRect::zero(),
            },
            pipeline_id,
            cull_children: false,
            rasterization_kind: RasterizationSpace::Local,
        }
    }

    pub fn resolve_scene_properties(&mut self, properties: &SceneProperties) -> bool {
        match self.kind {
            PictureKind::Image { ref mut composite_mode, .. } => {
                match composite_mode {
                    &mut Some(PictureCompositeMode::Filter(ref mut filter)) => {
                        match filter {
                            &mut FilterOp::Opacity(ref binding, ref mut value) => {
                                *value = properties.resolve_float(binding, *value);
                            }
                            _ => {}
                        }

                        filter.is_visible()
                    }
                    _ => true,
                }
            }
            _ => true
        }
    }

    pub fn new_box_shadow(
        blur_radius: f32,
        color: ColorF,
        blur_regions: Vec<LayerRect>,
        clip_mode: BoxShadowClipMode,
        radii_kind: BorderRadiusKind,
        box_offset: &LayerVector2D,
        border_radius: BorderRadius,
        spread_radius: f32,
        pipeline_id: PipelineId,
    ) -> Self {
        PicturePrimitive {
            runs: Vec::new(),
            render_task_id: None,
            kind: PictureKind::BoxShadow {
                blur_radius,
                color,
                blur_regions,
                clip_mode,
                radii_kind,
                content_rect: LayerRect::zero(),
                box_offset: box_offset.clone(),
                border_radius,
                spread_radius,
            },
            pipeline_id,
            cull_children: false,
            rasterization_kind: RasterizationSpace::Local,
        }
    }

    pub fn new_image(
        composite_mode: Option<PictureCompositeMode>,
        is_in_3d_context: bool,
        pipeline_id: PipelineId,
        reference_frame_id: ClipId,
        frame_output_pipeline_id: Option<PipelineId>,
    ) -> PicturePrimitive {
        PicturePrimitive {
            runs: Vec::new(),
            render_task_id: None,
            kind: PictureKind::Image {
                readback_render_task_id: None,
                composite_mode,
                is_in_3d_context,
                frame_output_pipeline_id,
                reference_frame_id,
                real_local_rect: LayerRect::zero(),
            },
            pipeline_id,
            cull_children: true,
            // TODO(gw): Make this configurable based on an
            //           exposed API parameter in StackingContext.
            rasterization_kind: RasterizationSpace::Screen,
        }
    }

    pub fn add_primitive(
        &mut self,
        prim_index: PrimitiveIndex,
        clip_and_scroll: ClipAndScrollInfo
    ) {
        if let Some(ref mut run) = self.runs.last_mut() {
            if run.clip_and_scroll == clip_and_scroll &&
               run.base_prim_index.0 + run.count == prim_index.0 {
                run.count += 1;
                return;
            }
        }

        self.runs.push(PrimitiveRun {
            base_prim_index: prim_index,
            count: 1,
            clip_and_scroll,
        });
    }

    pub fn update_local_rect(&mut self,
        prim_local_rect: LayerRect,
        prim_run_rect: PrimitiveRunLocalRect,
    ) -> LayerRect {
        let local_content_rect = prim_run_rect.local_rect_in_actual_parent_space;

        match self.kind {
            PictureKind::Image { composite_mode, ref mut real_local_rect, .. } => {
                *real_local_rect = prim_run_rect.local_rect_in_original_parent_space;

                match composite_mode {
                    Some(PictureCompositeMode::Filter(FilterOp::Blur(blur_radius))) => {
                        let inflate_size = blur_radius * BLUR_SAMPLE_SCALE;
                        local_content_rect.inflate(inflate_size, inflate_size)
                    }
                    _ => {
                        local_content_rect
                    }
                }
            }
            PictureKind::TextShadow { offset, blur_radius, ref mut content_rect, .. } => {
                let blur_offset = blur_radius * BLUR_SAMPLE_SCALE;

                *content_rect = local_content_rect.inflate(
                    blur_offset,
                    blur_offset,
                );

                content_rect.translate(&offset)
            }
            PictureKind::BoxShadow { blur_radius, clip_mode, radii_kind, ref mut content_rect, .. } => {
                // We need to inflate the content rect if outset.
                match clip_mode {
                    BoxShadowClipMode::Outset => {
                        let blur_offset = blur_radius * BLUR_SAMPLE_SCALE;

                        // If the radii are uniform, we can render just the top
                        // left corner and mirror it across the primitive. In
                        // this case, shift the content rect to leave room
                        // for the blur to take effect.
                        match radii_kind {
                            BorderRadiusKind::Uniform => {
                                let origin = LayerPoint::new(
                                    local_content_rect.origin.x - blur_offset,
                                    local_content_rect.origin.y - blur_offset,
                                );
                                let size = LayerSize::new(
                                    local_content_rect.size.width + blur_offset,
                                    local_content_rect.size.height + blur_offset,
                                );
                                *content_rect = LayerRect::new(origin, size);
                            }
                            BorderRadiusKind::NonUniform => {
                                // For a non-uniform radii, we need to expand
                                // the content rect on all sides for the blur.
                                *content_rect = local_content_rect.inflate(
                                    blur_offset,
                                    blur_offset,
                                );
                            }
                        }
                    }
                    BoxShadowClipMode::Inset => {
                        *content_rect = local_content_rect;
                    }
                }

                prim_local_rect
            }
        }
    }

    pub fn prepare_for_render(
        &mut self,
        prim_index: PrimitiveIndex,
        prim_context: &PrimitiveContext,
        render_tasks: &mut RenderTaskTree,
        prim_screen_rect: &DeviceIntRect,
        child_tasks: Vec<RenderTaskId>,
        parent_tasks: &mut Vec<RenderTaskId>,
    ) {
        match self.kind {
            PictureKind::Image {
                ref mut readback_render_task_id,
                composite_mode,
                ..
            } => {
                match composite_mode {
                    Some(PictureCompositeMode::Filter(FilterOp::Blur(blur_radius))) => {
                        let picture_task = RenderTask::new_picture(
                            Some(prim_screen_rect.size),
                            prim_index,
                            RenderTargetKind::Color,
                            prim_screen_rect.origin.x as f32,
                            prim_screen_rect.origin.y as f32,
                            PremultipliedColorF::TRANSPARENT,
                            ClearMode::Transparent,
                            self.rasterization_kind,
                            child_tasks,
                            None,
                        );

                        let blur_radius = device_length(blur_radius, prim_context.device_pixel_ratio);
                        let blur_std_deviation = blur_radius.0 as f32;
                        let picture_task_id = render_tasks.add(picture_task);

                        let blur_render_task = RenderTask::new_blur(
                            blur_std_deviation,
                            picture_task_id,
                            render_tasks,
                            RenderTargetKind::Color,
                            &[],
                            ClearMode::Transparent,
                            PremultipliedColorF::TRANSPARENT,
                            None,
                        );

                        let blur_render_task_id = render_tasks.add(blur_render_task);
                        self.render_task_id = Some(blur_render_task_id);
                    }
                    Some(PictureCompositeMode::MixBlend(..)) => {
                        let picture_task = RenderTask::new_picture(
                            Some(prim_screen_rect.size),
                            prim_index,
                            RenderTargetKind::Color,
                            prim_screen_rect.origin.x as f32,
                            prim_screen_rect.origin.y as f32,
                            PremultipliedColorF::TRANSPARENT,
                            ClearMode::Transparent,
                            self.rasterization_kind,
                            child_tasks,
                            None,
                        );

                        let readback_task_id = render_tasks.add(RenderTask::new_readback(*prim_screen_rect));

                        *readback_render_task_id = Some(readback_task_id);
                        parent_tasks.push(readback_task_id);

                        self.render_task_id = Some(render_tasks.add(picture_task));
                    }
                    Some(PictureCompositeMode::Filter(filter)) => {
                        // If this filter is not currently going to affect
                        // the picture, just collapse this picture into the
                        // current render task. This most commonly occurs
                        // when opacity == 1.0, but can also occur on other
                        // filters and be a significant performance win.
                        if filter.is_noop() {
                            parent_tasks.extend(child_tasks);
                            self.render_task_id = None;
                        } else {
                            let picture_task = RenderTask::new_picture(
                                Some(prim_screen_rect.size),
                                prim_index,
                                RenderTargetKind::Color,
                                prim_screen_rect.origin.x as f32,
                                prim_screen_rect.origin.y as f32,
                                PremultipliedColorF::TRANSPARENT,
                                ClearMode::Transparent,
                                self.rasterization_kind,
                                child_tasks,
                                None,
                            );

                            self.render_task_id = Some(render_tasks.add(picture_task));
                        }
                    }
                    Some(PictureCompositeMode::Blit) => {
                        let picture_task = RenderTask::new_picture(
                            Some(prim_screen_rect.size),
                            prim_index,
                            RenderTargetKind::Color,
                            prim_screen_rect.origin.x as f32,
                            prim_screen_rect.origin.y as f32,
                            PremultipliedColorF::TRANSPARENT,
                            ClearMode::Transparent,
                            self.rasterization_kind,
                            child_tasks,
                            None,
                        );

                        self.render_task_id = Some(render_tasks.add(picture_task));
                    }
                    None => {
                        parent_tasks.extend(child_tasks);
                        self.render_task_id = None;
                    }
                }
            }
            PictureKind::TextShadow { blur_radius, color, content_rect, .. } => {
                // This is a shadow element. Create a render task that will
                // render the text run to a target, and then apply a gaussian
                // blur to that text run in order to build the actual primitive
                // which will be blitted to the framebuffer.

                let blur_radius = device_length(blur_radius, prim_context.device_pixel_ratio);

                // TODO(gw): Rounding the content rect here to device pixels is not
                // technically correct. Ideally we should ceil() here, and ensure that
                // the extra part pixel in the case of fractional sizes is correctly
                // handled. For now, just use rounding which passes the existing
                // Gecko tests.
                let cache_width =
                    (content_rect.size.width * prim_context.device_pixel_ratio).round() as i32;
                let cache_height =
                    (content_rect.size.height * prim_context.device_pixel_ratio).round() as i32;
                let cache_size = DeviceIntSize::new(cache_width, cache_height);

                // Quote from https://drafts.csswg.org/css-backgrounds-3/#shadow-blur
                // "the image that would be generated by applying to the shadow a
                // Gaussian blur with a standard deviation equal to half the blur radius."
                let blur_std_deviation = blur_radius.0 as f32 * 0.5;

                let picture_task = RenderTask::new_picture(
                    Some(cache_size),
                    prim_index,
                    RenderTargetKind::Color,
                    content_rect.origin.x,
                    content_rect.origin.y,
                    color.premultiplied(),
                    ClearMode::Transparent,
                    self.rasterization_kind,
                    Vec::new(),
                    None,
                );

                let picture_task_id = render_tasks.add(picture_task);

                let render_task = RenderTask::new_blur(
                    blur_std_deviation,
                    picture_task_id,
                    render_tasks,
                    RenderTargetKind::Color,
                    &[],
                    ClearMode::Transparent,
                    color.premultiplied(),
                    None,
                );

                self.render_task_id = Some(render_tasks.add(render_task));
            }
            PictureKind::BoxShadow { blur_radius, clip_mode, ref blur_regions, color, content_rect,
                                     box_offset, border_radius, spread_radius, .. } => {
                let blur_radius = device_length(blur_radius, prim_context.device_pixel_ratio);

                // TODO(gw): Rounding the content rect here to device pixels is not
                // technically correct. Ideally we should ceil() here, and ensure that
                // the extra part pixel in the case of fractional sizes is correctly
                // handled. For now, just use rounding which passes the existing
                // Gecko tests.
                let cache_width =
                    (content_rect.size.width * prim_context.device_pixel_ratio).round() as i32;
                let cache_height =
                    (content_rect.size.height * prim_context.device_pixel_ratio).round() as i32;
                let cache_size = DeviceIntSize::new(cache_width, cache_height);

                // Quote from https://drafts.csswg.org/css-backgrounds-3/#shadow-blur
                // "the image that would be generated by applying to the shadow a
                // Gaussian blur with a standard deviation equal to half the blur radius."
                let blur_std_deviation = blur_radius.0 as f32 * 0.5;

                let blur_clear_mode = match clip_mode {
                    BoxShadowClipMode::Outset => {
                        ClearMode::One
                    }
                    BoxShadowClipMode::Inset => {
                        ClearMode::Zero
                    }
                };

                // hash box shadow properties
                let mut hasher = DefaultHasher::new();
                cache_size.hash(&mut hasher);
                OrderedFloat(content_rect.origin.x).hash(&mut hasher);
                OrderedFloat(content_rect.origin.y).hash(&mut hasher);
                color.premultiplied().hash(&mut hasher);
                blur_clear_mode.hash(&mut hasher);
                OrderedFloat(box_offset.x).hash(&mut hasher);
                OrderedFloat(box_offset.y).hash(&mut hasher);
                border_radius.hash(&mut hasher);
                OrderedFloat(blur_std_deviation).hash(&mut hasher);
                OrderedFloat(spread_radius).hash(&mut hasher);
                let hash_value = hasher.finish();

                let picture_task = RenderTask::new_picture(
                    Some(cache_size),
                    prim_index,
                    RenderTargetKind::Alpha,
                    content_rect.origin.x,
                    content_rect.origin.y,
                    color.premultiplied(),
                    ClearMode::Zero,
                    self.rasterization_kind,
                    Vec::new(),
                    Some(hash_value),
                );

                let picture_task_id = render_tasks.add(picture_task);

                let render_task = RenderTask::new_blur(
                    blur_std_deviation,
                    picture_task_id,
                    render_tasks,
                    RenderTargetKind::Alpha,
                    blur_regions,
                    blur_clear_mode,
                    color.premultiplied(),
                    Some(hash_value),
                );

                self.render_task_id = Some(render_tasks.add(render_task));
            }
        }

        if let Some(render_task_id) = self.render_task_id {
            parent_tasks.push(render_task_id);
        }
    }

    pub fn write_gpu_blocks(&self, mut _request: GpuDataRequest) {
        // TODO(gw): We'll need to write the GPU blocks
        //           here specific to a brush primitive
        //           once we start drawing pictures as brushes!
    }

    pub fn target_kind(&self) -> RenderTargetKind {
        match self.kind {
            PictureKind::TextShadow { .. } => RenderTargetKind::Color,
            PictureKind::BoxShadow { .. } => RenderTargetKind::Alpha,
            PictureKind::Image { .. } => RenderTargetKind::Color,
        }
    }
}
