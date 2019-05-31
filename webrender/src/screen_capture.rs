/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Screen capture infrastructure for the Gecko Profiler.

use std::collections::HashMap;

use api::{ImageFormat, TextureTarget};
use api::units::*;

use crate::device::{Device, PBO, DrawTarget, ReadTarget, Texture, TextureFilter};
use crate::internal_types::RenderTargetInfo;
use crate::renderer::Renderer;

/// A handle to a screenshot that is being asynchronously captured and scaled.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AsyncScreenshotHandle(usize);

/// An asynchronously captured screenshot bound to a PBO which has not yet been mapped for copying.
struct AsyncScreenshot {
    /// The PBO that will contain the screenshot data.
    pbo: PBO,
    /// The size of the screenshot.
    screenshot_size: DeviceIntSize,
    /// Thge image format of the screenshot.
    image_format: ImageFormat,
}

/// Renderer infrastructure for capturing screenshots and scaling them asynchronously.
pub(in crate) struct AsyncScreenshotGrabber {
    /// The textures used to scale screenshots.
    scaling_textures: Vec<Texture>,
    /// PBOs available to be used for screenshot readback.
    available_pbos: Vec<PBO>,
    /// PBOs containing screenshots that are awaiting readback.
    awaiting_readback: HashMap<AsyncScreenshotHandle, AsyncScreenshot>,
    /// The handle for the net PBO that will be inserted into `in_use_pbos`.
    next_pbo_handle: usize,
}

impl Default for AsyncScreenshotGrabber {
    fn default() -> Self {
        return AsyncScreenshotGrabber {
            scaling_textures: Vec::new(),
            available_pbos: Vec::new(),
            awaiting_readback: HashMap::new(),
            next_pbo_handle: 1,
        };
    }
}

impl AsyncScreenshotGrabber {
    /// Deinitialize the allocated textures and PBOs.
    pub fn deinit(self, device: &mut Device) {
        for texture in self.scaling_textures {
            device.delete_texture(texture);
        }

        for pbo in self.available_pbos {
            device.delete_pbo(pbo);
        }

        for (_, async_screenshot) in self.awaiting_readback {
            device.delete_pbo(async_screenshot.pbo);
        }
    }

    /// Take a screenshot and scale it asynchronously.
    ///
    /// The returned handle can be used to access the mapped screenshot data via
    /// `map_and_recycle_screenshot`.
    /// The returned size is the size of the screenshot.
    pub fn get_screenshot(
        &mut self,
        device: &mut Device,
        window_rect: DeviceIntRect,
        buffer_size: DeviceIntSize,
        image_format: ImageFormat,
    ) -> (AsyncScreenshotHandle, DeviceIntSize) {
        let scale = (buffer_size.width as f32 / window_rect.size.width as f32)
            .min(buffer_size.height as f32 / window_rect.size.height as f32);
        let screenshot_size = (window_rect.size.to_f32() * scale).round().to_i32();
        let required_size = buffer_size.area() as usize * image_format.bytes_per_pixel() as usize;

        assert!(screenshot_size.width <= buffer_size.width);
        assert!(screenshot_size.height <= buffer_size.height);

        let pbo = match self.available_pbos.pop() {
            Some(pbo) => {
                assert_eq!(pbo.get_reserved_size(), required_size);
                pbo
            }

            None => device.create_pbo_with_size(required_size),
        };

        self.scale_screenshot(
            device,
            ReadTarget::Default,
            window_rect,
            buffer_size,
            screenshot_size,
            image_format,
            0,
        );

        device.read_pixels_into_pbo(
            ReadTarget::from_texture(&self.scaling_textures[0], 0),
            DeviceIntRect::new(DeviceIntPoint::new(0, 0), screenshot_size),
            image_format,
            &pbo,
        );

        let handle = AsyncScreenshotHandle(self.next_pbo_handle);
        self.next_pbo_handle += 1;

        self.awaiting_readback.insert(
            handle,
            AsyncScreenshot {
                pbo,
                screenshot_size,
                image_format,
            },
        );

        (handle, screenshot_size)
    }

    /// Take the screenshot in the given `ReadTarget` and scale it to `dest_size` recursively.
    ///
    /// Each scaling operation scales only by a factor of two to preserve quality.
    ///
    /// Textures are scaled such that `scaling_textures[n]` is half the size of
    /// `scaling_textures[n+1]`.
    ///
    /// After the scaling completes, the final screenshot will be in
    /// `scaling_textures[0]`.
    fn scale_screenshot(
        &mut self,
        device: &mut Device,
        read_target: ReadTarget,
        read_target_rect: DeviceIntRect,
        buffer_size: DeviceIntSize,
        dest_size: DeviceIntSize,
        image_format: ImageFormat,
        level: usize,
    ) {
        let texture_size = buffer_size * (1 << level);
        if level == self.scaling_textures.len() {
            let texture = device.create_texture(
                TextureTarget::Default,
                image_format,
                texture_size.width,
                texture_size.height,
                TextureFilter::Linear,
                Some(RenderTargetInfo { has_depth: false }),
                1,
            );
            self.scaling_textures.push(texture);
        } else {
            let current_texture_size = self.scaling_textures[level].get_dimensions();
            assert_eq!(current_texture_size.width, texture_size.width);
            assert_eq!(current_texture_size.height, texture_size.height);
        }

        let (read_target, read_target_rect) = if read_target_rect.size.width > 2 * dest_size.width {
            self.scale_screenshot(
                device,
                read_target,
                read_target_rect,
                buffer_size,
                dest_size * 2,
                image_format,
                level + 1,
            );

            (
                ReadTarget::from_texture(&self.scaling_textures[level + 1], 0),
                DeviceIntRect::new(DeviceIntPoint::new(0, 0), dest_size * 2),
            )
        } else {
            (read_target, read_target_rect)
        };

        let draw_target = DrawTarget::from_texture(&self.scaling_textures[level], 0 as _, false);

        let draw_target_rect = draw_target
            .to_framebuffer_rect(DeviceIntRect::new(DeviceIntPoint::new(0, 0), dest_size));

        let read_target_rect = FramebufferIntRect::from_untyped(&read_target_rect.to_untyped());

        if level == 0 {
            device.blit_render_target_invert_y(
                read_target,
                read_target_rect,
                draw_target,
                draw_target_rect,
            );
        } else {
            device.blit_render_target(
                read_target,
                read_target_rect,
                draw_target,
                draw_target_rect,
                TextureFilter::Linear,
            );
        }
    }

    /// Map the contents of the screenshot given by the handle and copy it into
    /// the given buffer.
    pub fn map_and_recycle_screenshot(
        &mut self,
        device: &mut Device,
        handle: AsyncScreenshotHandle,
        dst_buffer: &mut [u8],
        dst_stride: usize,
    ) -> bool {
        let AsyncScreenshot {
            pbo,
            screenshot_size,
            image_format,
        } = match self.awaiting_readback.remove(&handle) {
            Some(screenshot) => screenshot,
            None => return false,
        };

        let success = if let Some(bound_pbo) = device.map_pbo_for_readback(&pbo) {
            let src_buffer = &bound_pbo.data;
            let src_stride =
                screenshot_size.width as usize * image_format.bytes_per_pixel() as usize;

            for (src_slice, dst_slice) in src_buffer
                .chunks(src_stride)
                .zip(dst_buffer.chunks_mut(dst_stride))
                .take(screenshot_size.height as usize)
            {
                dst_slice[.. src_stride].copy_from_slice(src_slice);
            }

            true
        } else {
            false
        };

        self.available_pbos.push(pbo);
        success
    }
}

// Screen-capture specific Renderer impls.
impl Renderer {
    /// Take a screenshot and scale it asynchronously.
    ///
    /// The returned handle can be used to access the mapped screenshot data via
    /// `map_and_recycle_screenshot`.
    ///
    /// The returned size is the size of the screenshot.
    pub fn get_screenshot_async(
        &mut self,
        window_rect: DeviceIntRect,
        buffer_size: DeviceIntSize,
        image_format: ImageFormat,
    ) -> (AsyncScreenshotHandle, DeviceIntSize) {
        self.device.begin_frame();

        let handle = self
            .async_screenshots
            .get_or_insert_with(AsyncScreenshotGrabber::default)
            .get_screenshot(&mut self.device, window_rect, buffer_size, image_format);

        self.device.end_frame();

        handle
    }

    /// Map the contents of the screenshot given by the handle and copy it into
    /// the given buffer.
    pub fn map_and_recycle_screenshot(
        &mut self,
        handle: AsyncScreenshotHandle,
        dst_buffer: &mut [u8],
        dst_stride: usize,
    ) -> bool {
        if let Some(async_screenshots) = self.async_screenshots.as_mut() {
            async_screenshots.map_and_recycle_screenshot(
                &mut self.device,
                handle,
                dst_buffer,
                dst_stride,
            )
        } else {
            false
        }
    }

    /// Release the screenshot grabbing structures that the profiler was using.
    pub fn release_profiler_structures(&mut self) {
        if let Some(async_screenshots) = self.async_screenshots.take() {
            self.device.begin_frame();
            async_screenshots.deinit(&mut self.device);
            self.device.end_frame();
        }
    }
}
