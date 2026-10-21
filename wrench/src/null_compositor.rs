/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use webrender::api::ImageRendering;
use webrender::api::units::DeviceIntRect;
use webrender::{ClipRadius, CompositorInputConfig, CompositorSurfaceTransform, LayerCompositor, WindowProperties};

/// A layer compositor that draws every layer to the window and drops
/// compositor surfaces. It lets platforms without an OS compositor
/// integration run frame building with `CompositorKind::Layer`, so only
/// scenes without promoted surfaces render correctly. Layers are drawn with
/// a top-left origin, so the output is vertically flipped.
pub struct NullLayerCompositor;

impl LayerCompositor for NullLayerCompositor {
    fn begin_frame(&mut self, _input: &CompositorInputConfig) -> bool {
        true
    }

    fn bind_layer(&mut self, _index: usize, _dirty_rects: &[DeviceIntRect]) {}

    fn present_layer(&mut self, _index: usize, _dirty_rects: &[DeviceIntRect]) {}

    fn add_surface(
        &mut self,
        _index: usize,
        _transform: CompositorSurfaceTransform,
        _clip_rect: DeviceIntRect,
        _image_rendering: ImageRendering,
        _rounded_clip_rect: DeviceIntRect,
        _rounded_clip_radii: ClipRadius,
    ) {}

    fn end_frame(&mut self) {}

    fn get_window_properties(&self) -> WindowProperties {
        WindowProperties {
            is_opaque: true,
            enable_screenshot: false,
        }
    }
}
