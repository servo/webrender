/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{DeviceIntPoint, DeviceIntRect};
use api::{LayerPoint, LayerRect, LayerToWorldScale, LayerVector2D};
use api::{ColorF, FilterOp, MixBlendMode, PipelineId};
use api::{PremultipliedColorF, Shadow};
use box_shadow::{BLUR_SAMPLE_SCALE};
use clip_scroll_tree::ClipScrollNodeIndex;
use frame_builder::{FrameBuildingContext, FrameBuildingState, PictureState};
use gpu_cache::{GpuCacheHandle, GpuDataRequest};
use gpu_types::{PictureType};
use prim_store::{BrushKind, BrushPrimitive, PrimitiveIndex, PrimitiveRun, PrimitiveRunLocalRect};
use prim_store::{PrimitiveMetadata, ScrollNodeAndClipChain};
use render_task::{ClearMode, RenderTask};
use render_task::{RenderTaskId, RenderTaskLocation, to_cache_size};
use scene::{FilterOpHelpers, SceneProperties};
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

/// Configure whether the content to be drawn by a picture
/// in local space rasterization or the screen space.
#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum ContentOrigin {
    Local(LayerPoint),
    Screen(DeviceIntPoint),
}

#[derive(Debug)]
pub enum PictureKind {
    TextShadow {
        offset: LayerVector2D,
        color: ColorF,
        blur_radius: f32,
        content_rect: LayerRect,
    },
    Image {
        // If a mix-blend-mode, contains the render task for
        // the readback of the framebuffer that we use to sample
        // from in the mix-blend-mode shader.
        // For drop-shadow filter, this will store the original
        // picture task which would be rendered on screen after
        // blur pass.
        secondary_render_task_id: Option<RenderTaskId>,
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
        reference_frame_index: ClipScrollNodeIndex,
        real_local_rect: LayerRect,
        // An optional cache handle for storing extra data
        // in the GPU cache, depending on the type of
        // picture.
        extra_gpu_data_handle: GpuCacheHandle,
        // The current screen-space rect of the rendered
        // portion of this picture.
        task_rect: DeviceIntRect,
    },
}

#[derive(Debug)]
pub struct PicturePrimitive {
    // If this picture is drawn to an intermediate surface,
    // the associated target information.
    pub surface: Option<RenderTaskId>,

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

    // The brush primitive that will be used to draw this
    // picture.
    // TODO(gw): Having a brush primitive embedded here
    //           makes the code complex in a few places.
    //           Consider a better way to structure this.
    //           Maybe embed the PicturePrimitive inside
    //           the BrushKind enum instead?
    pub brush: BrushPrimitive,
}

impl PicturePrimitive {
    pub fn new_text_shadow(shadow: Shadow, pipeline_id: PipelineId) -> Self {
        PicturePrimitive {
            runs: Vec::new(),
            surface: None,
            kind: PictureKind::TextShadow {
                offset: shadow.offset,
                color: shadow.color,
                blur_radius: shadow.blur_radius,
                content_rect: LayerRect::zero(),
            },
            pipeline_id,
            cull_children: false,
            brush: BrushPrimitive::new(
                BrushKind::Picture,
                None,
            ),
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

    pub fn new_image(
        composite_mode: Option<PictureCompositeMode>,
        is_in_3d_context: bool,
        pipeline_id: PipelineId,
        reference_frame_index: ClipScrollNodeIndex,
        frame_output_pipeline_id: Option<PipelineId>,
    ) -> Self {
        PicturePrimitive {
            runs: Vec::new(),
            surface: None,
            kind: PictureKind::Image {
                secondary_render_task_id: None,
                composite_mode,
                is_in_3d_context,
                frame_output_pipeline_id,
                reference_frame_index,
                real_local_rect: LayerRect::zero(),
                extra_gpu_data_handle: GpuCacheHandle::new(),
                task_rect: DeviceIntRect::zero(),
            },
            pipeline_id,
            cull_children: true,
            brush: BrushPrimitive::new(
                BrushKind::Picture,
                None,
            ),
        }
    }

    pub fn add_primitive(
        &mut self,
        prim_index: PrimitiveIndex,
        clip_and_scroll: ScrollNodeAndClipChain
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

    pub fn update_local_rect(
        &mut self,
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
                    Some(PictureCompositeMode::Filter(FilterOp::DropShadow(offset, blur_radius, _))) => {
                        let inflate_size = blur_radius * BLUR_SAMPLE_SCALE;
                        local_content_rect.inflate(inflate_size, inflate_size)
                                          .translate(&offset)
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
        }
    }

    pub fn prepare_for_render(
        &mut self,
        prim_index: PrimitiveIndex,
        prim_metadata: &mut PrimitiveMetadata,
        pic_state_for_children: PictureState,
        pic_state: &mut PictureState,
        frame_context: &FrameBuildingContext,
        frame_state: &mut FrameBuildingState,
    ) {
        let content_scale = LayerToWorldScale::new(1.0) * frame_context.device_pixel_scale;
        let prim_screen_rect = prim_metadata
                                .screen_rect
                                .as_ref()
                                .expect("bug: trying to draw an off-screen picture!?");

        match self.kind {
            PictureKind::Image {
                ref mut secondary_render_task_id,
                ref mut extra_gpu_data_handle,
                ref mut task_rect,
                composite_mode,
                ..
            } => {
                match composite_mode {
                    Some(PictureCompositeMode::Filter(FilterOp::Blur(blur_radius))) => {
                        // If blur radius is 0, we can skip drawing this an an
                        // intermediate surface.
                        if blur_radius == 0.0 {
                            pic_state.tasks.extend(pic_state_for_children.tasks);
                            self.surface = None;
                        } else {
                            let blur_std_deviation = blur_radius * frame_context.device_pixel_scale.0;
                            let blur_range = (blur_std_deviation * BLUR_SAMPLE_SCALE).ceil() as i32;

                            // The clipped field is the part of the picture that is visible
                            // on screen. The unclipped field is the screen-space rect of
                            // the complete picture, if no screen / clip-chain was applied
                            // (this includes the extra space for blur region). To ensure
                            // that we draw a large enough part of the picture to get correct
                            // blur results, inflate that clipped area by the blur range, and
                            // then intersect with the total screen rect, to minimize the
                            // allocation size.
                            let device_rect = prim_screen_rect
                                .clipped
                                .inflate(blur_range, blur_range)
                                .intersection(&prim_screen_rect.unclipped)
                                .unwrap();

                            // If scrolling or property animation has resulted in the task
                            // rect being different than last time, invalidate the GPU
                            // cache entry for this picture to ensure that the correct
                            // task rect is provided to the image shader.
                            if *task_rect != device_rect {
                                frame_state.gpu_cache.invalidate(&prim_metadata.gpu_location);
                                *task_rect = device_rect;
                            }

                            let content_origin = ContentOrigin::Screen(device_rect.origin);

                            let picture_task = RenderTask::new_picture(
                                RenderTaskLocation::Dynamic(None, device_rect.size),
                                prim_index,
                                RenderTargetKind::Color,
                                content_origin,
                                PremultipliedColorF::TRANSPARENT,
                                ClearMode::Transparent,
                                pic_state_for_children.tasks,
                                PictureType::Image,
                            );

                            let picture_task_id = frame_state.render_tasks.add(picture_task);

                            let blur_render_task = RenderTask::new_blur(
                                blur_std_deviation,
                                picture_task_id,
                                frame_state.render_tasks,
                                RenderTargetKind::Color,
                                ClearMode::Transparent,
                                PremultipliedColorF::TRANSPARENT,
                            );

                            let render_task_id = frame_state.render_tasks.add(blur_render_task);
                            pic_state.tasks.push(render_task_id);
                            self.surface = Some(render_task_id);
                        }
                    }
                    Some(PictureCompositeMode::Filter(FilterOp::DropShadow(offset, blur_radius, color))) => {
                        // TODO(gw): This is totally wrong and can never work with
                        //           transformed drop-shadow elements. Fix me!
                        let rect = (prim_metadata.local_rect.translate(&-offset) * content_scale).round().to_i32();
                        let mut picture_task = RenderTask::new_picture(
                            RenderTaskLocation::Dynamic(None, rect.size),
                            prim_index,
                            RenderTargetKind::Color,
                            ContentOrigin::Screen(rect.origin),
                            PremultipliedColorF::TRANSPARENT,
                            ClearMode::Transparent,
                            pic_state_for_children.tasks,
                            PictureType::Image,
                        );
                        picture_task.mark_for_saving();

                        let blur_std_deviation = blur_radius * frame_context.device_pixel_scale.0;
                        let picture_task_id = frame_state.render_tasks.add(picture_task);

                        let blur_render_task = RenderTask::new_blur(
                            blur_std_deviation.round(),
                            picture_task_id,
                            frame_state.render_tasks,
                            RenderTargetKind::Color,
                            ClearMode::Transparent,
                            color.premultiplied(),
                        );

                        *secondary_render_task_id = Some(picture_task_id);

                        let render_task_id = frame_state.render_tasks.add(blur_render_task);
                        pic_state.tasks.push(render_task_id);
                        self.surface = Some(render_task_id);
                    }
                    Some(PictureCompositeMode::MixBlend(..)) => {
                        let content_origin = ContentOrigin::Screen(prim_screen_rect.clipped.origin);

                        let picture_task = RenderTask::new_picture(
                            RenderTaskLocation::Dynamic(None, prim_screen_rect.clipped.size),
                            prim_index,
                            RenderTargetKind::Color,
                            content_origin,
                            PremultipliedColorF::TRANSPARENT,
                            ClearMode::Transparent,
                            pic_state_for_children.tasks,
                            PictureType::Image,
                        );

                        let readback_task_id = frame_state.render_tasks.add(
                            RenderTask::new_readback(prim_screen_rect.clipped)
                        );

                        *secondary_render_task_id = Some(readback_task_id);
                        pic_state.tasks.push(readback_task_id);

                        let render_task_id = frame_state.render_tasks.add(picture_task);
                        pic_state.tasks.push(render_task_id);
                        self.surface = Some(render_task_id);
                    }
                    Some(PictureCompositeMode::Filter(filter)) => {
                        let content_origin = ContentOrigin::Screen(prim_screen_rect.clipped.origin);

                        // If this filter is not currently going to affect
                        // the picture, just collapse this picture into the
                        // current render task. This most commonly occurs
                        // when opacity == 1.0, but can also occur on other
                        // filters and be a significant performance win.
                        if filter.is_noop() {
                            pic_state.tasks.extend(pic_state_for_children.tasks);
                            self.surface = None;
                        } else {

                            if let FilterOp::ColorMatrix(m) = filter {
                                if let Some(mut request) = frame_state.gpu_cache.request(extra_gpu_data_handle) {
                                    for i in 0..5 {
                                        request.push([m[i*4], m[i*4+1], m[i*4+2], m[i*4+3]]);
                                    }
                                }
                            }

                            let picture_task = RenderTask::new_picture(
                                RenderTaskLocation::Dynamic(None, prim_screen_rect.clipped.size),
                                prim_index,
                                RenderTargetKind::Color,
                                content_origin,
                                PremultipliedColorF::TRANSPARENT,
                                ClearMode::Transparent,
                                pic_state_for_children.tasks,
                                PictureType::Image,
                            );

                            let render_task_id = frame_state.render_tasks.add(picture_task);
                            pic_state.tasks.push(render_task_id);
                            self.surface = Some(render_task_id);
                        }
                    }
                    Some(PictureCompositeMode::Blit) => {
                        let content_origin = ContentOrigin::Screen(prim_screen_rect.clipped.origin);

                        let picture_task = RenderTask::new_picture(
                            RenderTaskLocation::Dynamic(None, prim_screen_rect.clipped.size),
                            prim_index,
                            RenderTargetKind::Color,
                            content_origin,
                            PremultipliedColorF::TRANSPARENT,
                            ClearMode::Transparent,
                            pic_state_for_children.tasks,
                            PictureType::Image,
                        );

                        let render_task_id = frame_state.render_tasks.add(picture_task);
                        pic_state.tasks.push(render_task_id);
                        self.surface = Some(render_task_id);
                    }
                    None => {
                        pic_state.tasks.extend(pic_state_for_children.tasks);
                        self.surface = None;
                    }
                }
            }
            PictureKind::TextShadow { blur_radius, color, content_rect, .. } => {
                // This is a shadow element. Create a render task that will
                // render the text run to a target, and then apply a gaussian
                // blur to that text run in order to build the actual primitive
                // which will be blitted to the framebuffer.
                let cache_size = to_cache_size(content_rect.size * content_scale);

                // Quote from https://drafts.csswg.org/css-backgrounds-3/#shadow-blur
                // "the image that would be generated by applying to the shadow a
                // Gaussian blur with a standard deviation equal to half the blur radius."
                let device_radius = (blur_radius * frame_context.device_pixel_scale.0).round();
                let blur_std_deviation = device_radius * 0.5;

                let picture_task = RenderTask::new_picture(
                    RenderTaskLocation::Dynamic(None, cache_size),
                    prim_index,
                    RenderTargetKind::Color,
                    ContentOrigin::Local(content_rect.origin),
                    color.premultiplied(),
                    ClearMode::Transparent,
                    Vec::new(),
                    PictureType::TextShadow,
                );

                let picture_task_id = frame_state.render_tasks.add(picture_task);

                let blur_render_task = RenderTask::new_blur(
                    blur_std_deviation,
                    picture_task_id,
                    frame_state.render_tasks,
                    RenderTargetKind::Color,
                    ClearMode::Transparent,
                    color.premultiplied(),
                );

                let render_task_id = frame_state.render_tasks.add(blur_render_task);
                pic_state.tasks.push(render_task_id);
                self.surface = Some(render_task_id);
            }
        }
    }

    pub fn write_gpu_blocks(&self, request: &mut GpuDataRequest) {
        // TODO(gw): It's unfortunate that we pay a fixed cost
        //           of 5 GPU blocks / picture, just due to the size
        //           of the color matrix. There aren't typically very
        //           many pictures in a scene, but we should consider
        //           making this more efficient for the common case.
        match self.kind {
            PictureKind::TextShadow { .. } => {
                request.push([0.0; 4]);
            }
            PictureKind::Image { composite_mode, task_rect, .. } => {
                match composite_mode {
                    Some(PictureCompositeMode::Filter(filter)) => {
                        let amount = match filter {
                            FilterOp::Contrast(amount) => amount,
                            FilterOp::Grayscale(amount) => amount,
                            FilterOp::HueRotate(angle) => 0.01745329251 * angle,
                            FilterOp::Invert(amount) => amount,
                            FilterOp::Saturate(amount) => amount,
                            FilterOp::Sepia(amount) => amount,
                            FilterOp::Brightness(amount) => amount,
                            FilterOp::Opacity(_, amount) => amount,

                            // Go through different paths
                            FilterOp::Blur(..) |
                            FilterOp::DropShadow(..) |
                            FilterOp::ColorMatrix(_) => {
                                // TODO(gw): The data for blur (and drop-shadows in the future)
                                //           doesn't match how the brush_blend shader uses this
                                //           data for other filter types (see below). We should
                                //           update the brush_blend and brush_mix_blend shaders
                                //           to do screen-space UV calculation the same way that
                                //           the brush_image shader does, and move the amount
                                //           storage into the extra gpu data or instance data.
                                request.push(task_rect.to_f32());
                                return;
                            }
                        };

                        request.push([amount, 1.0 - amount, 0.0, 0.0]);
                    }
                    _ => {
                        request.push([0.0; 4]);
                    }
                }
            }
        }
    }
}
