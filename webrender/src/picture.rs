/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{ColorF, ClipAndScrollInfo, device_length, DeviceIntSize};
use api::{LayerRect, LayerSize, Shadow};
use frame_builder::PrimitiveContext;
use gpu_cache::GpuDataRequest;
use prim_store::{PrimitiveIndex, PrimitiveMetadata};
use render_task::{RenderTask, RenderTaskId, RenderTaskTree};
use tiling::RenderTargetKind;

/*
 A picture represents a dynamically rendered image. It consists of:

 * A number of primitives that are drawn onto the picture.
 * A composite operation describing how to composite this
   picture into its parent.
 * A configuration describing how to draw the primitives on
   this picture (e.g. in screen space or local space).
 */

#[derive(Clone, Debug)]
pub struct PrimitiveRun {
    pub prim_index: PrimitiveIndex,
    pub count: usize,
    pub clip_and_scroll: ClipAndScrollInfo,
}

#[derive(Debug)]
pub struct BoxShadowPicture {
    content_rect: LayerRect,
    color: ColorF,
    blur_radius: f32,
    slice_top_left: LayerSize,
    slice_bottom_right: LayerSize,
}

#[derive(Debug)]
pub enum PictureKind {
    TextShadow(Shadow),
    BoxShadow(BoxShadowPicture),
}

#[derive(Debug)]
pub struct PicturePrimitive {
    pub prim_runs: Vec<PrimitiveRun>,
    pub render_task_id: Option<RenderTaskId>,
    pub kind: PictureKind,
    pub target_kind: RenderTargetKind,

    // TODO(gw): Add a mode that specifies if this
    //           picture should be rasterized in
    //           screen-space or local-space.
}

impl PicturePrimitive {
    pub fn new_text_shadow(shadow: Shadow) -> PicturePrimitive {
        PicturePrimitive {
            prim_runs: Vec::new(),
            kind: PictureKind::TextShadow(shadow),
            render_task_id: None,
            target_kind: RenderTargetKind::Color,
        }
    }

    pub fn new_box_shadow(
        content_rect: LayerRect,
        color: ColorF,
        blur_radius: f32,
        slice_top_left: LayerSize,
        slice_bottom_right: LayerSize,
    ) -> PicturePrimitive {
        PicturePrimitive {
            prim_runs: Vec::new(),
            kind: PictureKind::BoxShadow(BoxShadowPicture {
                content_rect,
                blur_radius,
                color,
                slice_top_left,
                slice_bottom_right,
            }),
            render_task_id: None,
            target_kind: RenderTargetKind::Alpha,
        }
    }

    pub fn as_text_shadow(&self) -> &Shadow {
        match self.kind {
            PictureKind::TextShadow(ref shadow) => shadow,
            PictureKind::BoxShadow(..) => panic!("bug: not a text shadow"),
        }
    }

    pub fn add_primitive(
        &mut self,
        prim_index: PrimitiveIndex,
        clip_and_scroll: ClipAndScrollInfo
    ) {
        if let Some(ref mut run) = self.prim_runs.last_mut() {
            if run.clip_and_scroll == clip_and_scroll &&
               run.prim_index.0 + run.count == prim_index.0 {
                run.count += 1;
                return;
            }
        }

        self.prim_runs.push(PrimitiveRun {
            prim_index,
            count: 1,
            clip_and_scroll,
        });
    }

    pub fn write_gpu_blocks(&self, mut request: GpuDataRequest) {
        match self.kind {
            PictureKind::TextShadow(ref shadow) => {
                request.push(shadow.color);
                request.push([
                    shadow.offset.x,
                    shadow.offset.y,
                    shadow.blur_radius,
                    0.0,
                ]);
                request.push([0.0; 4]);
            }
            PictureKind::BoxShadow(ref box_shadow) => {
                request.push(box_shadow.color);
                request.push([
                    0.0,
                    0.0,
                    box_shadow.blur_radius,
                    0.0,
                ]);
                request.push([
                    box_shadow.slice_top_left.width,
                    box_shadow.slice_top_left.height,
                    box_shadow.slice_bottom_right.width,
                    box_shadow.slice_bottom_right.height,
                ]);
            }
        }
    }

    pub fn prepare_for_render(
        &mut self,
        prim_index: PrimitiveIndex,
        prim_context: &PrimitiveContext,
        prim_metadata: &PrimitiveMetadata,
        render_tasks: &mut RenderTaskTree,
    ) {
        match self.kind {
            PictureKind::TextShadow(ref shadow) => {
                // This is a shadow element. Create a render task that will
                // render the text run to a target, and then apply a gaussian
                // blur to that text run in order to build the actual primitive
                // which will be blitted to the framebuffer.
                let cache_width =
                    (prim_metadata.local_rect.size.width * prim_context.device_pixel_ratio).ceil() as i32;
                let cache_height =
                    (prim_metadata.local_rect.size.height * prim_context.device_pixel_ratio).ceil() as i32;
                let cache_size = DeviceIntSize::new(cache_width, cache_height);
                let blur_radius = device_length(shadow.blur_radius, prim_context.device_pixel_ratio);

                let picture_task = RenderTask::new_picture(
                    cache_size,
                    prim_index,
                    self.target_kind,
                );
                let picture_task_id = render_tasks.add(picture_task);
                let render_task = RenderTask::new_blur(
                    blur_radius,
                    picture_task_id,
                    render_tasks,
                    self.target_kind,
                );
                self.render_task_id = Some(render_tasks.add(render_task));
            }
            PictureKind::BoxShadow(ref box_shadow) => {
                let cache_width =
                    (box_shadow.content_rect.size.width * prim_context.device_pixel_ratio).ceil() as i32;
                let cache_height =
                    (box_shadow.content_rect.size.height * prim_context.device_pixel_ratio).ceil() as i32;
                let cache_size = DeviceIntSize::new(cache_width, cache_height);
                let blur_radius = device_length(box_shadow.blur_radius, prim_context.device_pixel_ratio);

                let picture_task = RenderTask::new_picture(
                    cache_size,
                    prim_index,
                    self.target_kind,
                );
                let picture_task_id = render_tasks.add(picture_task);
                let render_task = RenderTask::new_blur(
                    blur_radius,
                    picture_task_id,
                    render_tasks,
                    self.target_kind,
                );
                self.render_task_id = Some(render_tasks.add(render_task));
            }
        }
    }
}
