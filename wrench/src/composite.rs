/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::collections::HashMap;
use std::os::raw::c_void;
use std::ptr;

use mozangle::egl::ffi::types::EGLContext;
use webrender::{Compositor, CompositorCapabilities, CompositorSurfaceTransform, NativeSurfaceId, WindowProperties, WindowVisibility};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

use crate::WindowWrapper;

use mozangle::egl::ffi::types::{EGLDisplay, EGLSurface, EGLint};
use mozangle::egl;

// A simplistic implementation of the swapchain part of the `Compositor` trait
// to allow wrench to composite via DirectComposition.

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CompositorHandle(usize);

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Debug)]
struct LayerId(usize);

impl LayerId {
    const INVALID: LayerId = LayerId(0);
}

// FFI bindings to `composite.cpp`
#[link(name="wr_composite", kind="static")]
extern "C" {
    fn wrc_new(d3d11_device: *const c_void, hwnd: *const c_void) -> CompositorHandle;
    fn wrc_delete(compositor: CompositorHandle);

    fn wrc_create_layer(compositor: CompositorHandle, width: i32, height: i32, is_opaque: bool) -> LayerId;
    fn wrc_destroy_layer(layer_id: LayerId);
    fn wrc_get_layer_backbuffer(layer_id: LayerId) -> *mut c_void;
    fn wrc_set_layer_position(layer_id: LayerId, x: f32, y: f32);
    fn wrc_present_layer(layer_id: LayerId);
    fn wrc_add_layer(compositor: CompositorHandle, layer_id: LayerId);

    fn wrc_begin_frame(compositor: CompositorHandle);
    fn wrc_end_frame(compositor: CompositorHandle);
}

// A basic layer - we only create one primary layer in the initial commit
struct WrLayer {
    // Layer dimensions
    width: i32,
    height: i32,
    // EGL surface that gets bound for webrender to draw to
    surface: EGLSurface,
    // Handle to the FFI layer
    layer_id: LayerId,
}

impl WrLayer {
    fn empty() -> Self {
        WrLayer {
            width: 0,
            height: 0,
            layer_id: LayerId::INVALID,
            surface: ptr::null(),
        }
    }
}

// A basic `Compositor` implementation for wrench
pub struct WrCompositor {
    // EGL display and content, provided by winit
    context: EGLContext,
    display: EGLDisplay,
    // FFI compositor handle
    compositor: CompositorHandle,
    // The swapchain layers needed for current scene
    layers: HashMap<NativeSurfaceId, WrLayer>,
}

impl WrCompositor {
    pub fn new(window: &WindowWrapper) -> Self {
        // Retrieve the D3D11 device from winit - this was created when ANGLE was initialized
        let d3d11_device = window.get_d3d11_device();

        // Get the win32 and EGL information needed from the window
        let (hwnd, display, context) = match window {
            WindowWrapper::Angle { window, context: angle, .. } => {
                let hwnd = match window.window_handle().unwrap().as_raw() {
                    RawWindowHandle::Win32(h) => h.hwnd.get() as *const c_void,
                    _ => unreachable!(),
                };
                (hwnd, angle.get_display(), angle.get_context())
            }
            _ => unreachable!(),
        };

        // Construct the FFI part of the compositor impl
        let compositor = unsafe {
            wrc_new(d3d11_device, hwnd)
        };

        WrCompositor {
            display,
            context,
            compositor,
            layers: HashMap::new(),
        }
    }

    fn destroy_layer(&self, layer: WrLayer) {
        unsafe {
            egl::ffi::DestroySurface(self.display, layer.surface);
            wrc_destroy_layer(layer.layer_id);
        }
    }
}

impl Drop for WrCompositor {
    fn drop(&mut self) {
        for (_, layer) in std::mem::take(&mut self.layers) {
            self.destroy_layer(layer);
        }
        unsafe {
            wrc_delete(self.compositor);
        }
    }
}

impl Compositor for WrCompositor {
    fn create_surface(
        &mut self,
        _id: NativeSurfaceId,
        _tile_size: webrender::api::units::DeviceIntSize,
        _is_opaque: bool,
    ) {
        unreachable!();
    }

    fn create_external_surface(&mut self, _id: NativeSurfaceId, _is_opaque: bool) {
        unreachable!();
    }

    fn create_backdrop_surface(&mut self, _id: NativeSurfaceId, _color: webrender::api::ColorF) {
        unreachable!();
    }

    fn destroy_surface(&mut self, id: NativeSurfaceId) {
        if let Some(layer) = self.layers.remove(&id) {
            self.destroy_layer(layer);
        }
    }

    fn create_tile(&mut self, _id: webrender::NativeTileId) {
        unreachable!();
    }

    fn destroy_tile(&mut self, _id: webrender::NativeTileId) {
        unreachable!();
    }

    fn attach_external_image(
        &mut self,
        _id: NativeSurfaceId,
        _external_image: webrender::api::ExternalImageId,
    ) {
        unreachable!();
    }

    fn bind(
        &mut self,
        _id: webrender::NativeTileId,
        _dirty_rect: webrender::api::units::DeviceIntRect,
        _valid_rect: webrender::api::units::DeviceIntRect,
    ) -> webrender::NativeSurfaceInfo {
        unreachable!();
    }

    fn unbind(&mut self) {
        unreachable!();
    }

    fn create_swapchain_surface(
        &mut self,
        id: NativeSurfaceId,
        size: webrender::api::units::DeviceIntSize,
        is_opaque: bool,
        _needs_sync_commit: bool,
    ) {
        let mut layer = WrLayer::empty();

        layer.width = size.width;
        layer.height = size.height;
        layer.layer_id = unsafe {
            wrc_create_layer(self.compositor, layer.width, layer.height, is_opaque)
        };

        let pbuffer_attribs: [EGLint; 5] = [
            egl::ffi::WIDTH as EGLint, layer.width,
            egl::ffi::HEIGHT as EGLint, layer.height,
            egl::ffi::NONE as EGLint,
        ];

        let attribs: [EGLint; 18] = [
            egl::ffi::SURFACE_TYPE as EGLint, egl::ffi::WINDOW_BIT as EGLint,
            egl::ffi::RED_SIZE as EGLint, 8,
            egl::ffi::GREEN_SIZE as EGLint, 8,
            egl::ffi::BLUE_SIZE as EGLint, 8,
            egl::ffi::ALPHA_SIZE as EGLint, 8,
            egl::ffi::RENDERABLE_TYPE as EGLint, egl::ffi::OPENGL_ES2_BIT as EGLint,
            // TODO(gw): Can we disable z-buffer for compositing always?
            egl::ffi::DEPTH_SIZE as EGLint, 24,
            egl::ffi::STENCIL_SIZE as EGLint, 8,
            egl::ffi::NONE as EGLint, egl::ffi::NONE as EGLint
        ];

        // Get the D3D backbuffer texture for the swap-chain
        let back_buffer = unsafe {
            wrc_get_layer_backbuffer(layer.layer_id)
        };

        // Create an EGL surface <-> binding to the D3D backbuffer texture
        let mut egl_config = ptr::null();
        let mut cfg_count = 0;

        let ok = unsafe {
            egl::ffi::ChooseConfig(self.display, attribs.as_ptr(), &mut egl_config, 1, &mut cfg_count)
        };

        assert_ne!(ok, 0);
        assert_eq!(cfg_count, 1);
        assert_ne!(egl_config, ptr::null());

        let surface = unsafe {
            egl::ffi::CreatePbufferFromClientBuffer(
                self.display,
                egl::ffi::D3D_TEXTURE_ANGLE,
                back_buffer,
                egl_config,
                pbuffer_attribs.as_ptr(),
            )
        };
        assert_ne!(surface, ptr::null());

        layer.surface = surface;

        self.layers.insert(id, layer);
    }

    fn resize_swapchain(&mut self, _id: NativeSurfaceId, _size: webrender::api::units::DeviceIntSize) {
        // TODO: Handle resize of layer (not needed for initial commit,
        // but will need to support resizing wrench window)
        todo!();
    }

    fn bind_swapchain(&mut self, id: NativeSurfaceId, _dirty_rects: &[webrender::api::units::DeviceIntRect]) {
        // Bind the DC surface to EGL so that WR can composite to the layer
        let layer = &self.layers[&id];

        let ok = unsafe {
            egl::ffi::MakeCurrent(
                self.display,
                layer.surface,
                layer.surface,
                self.context,
            )
        };
        assert!(ok != 0);
    }

    fn present_swapchain(&mut self, id: NativeSurfaceId, _dirty_rects: &[webrender::api::units::DeviceIntRect]) {
        let layer = &self.layers[&id];

        unsafe {
            wrc_present_layer(layer.layer_id);
        }
    }

    fn begin_frame(&mut self) {
        unsafe {
            // Reset DC visual tree
            wrc_begin_frame(self.compositor);
        }
    }

    fn add_surface(
        &mut self,
        id: NativeSurfaceId,
        transform: CompositorSurfaceTransform,
        _clip_rect: webrender::api::units::DeviceIntRect,
        _image_rendering: webrender::api::ImageRendering,
        _rounded_clip_rect: webrender::api::units::DeviceIntRect,
        _rounded_clip_radii: webrender::ClipRadius,
    ) {
        let layer = &self.layers[&id];

        unsafe {
            wrc_set_layer_position(layer.layer_id, transform.offset.x, transform.offset.y);
            wrc_add_layer(self.compositor, layer.layer_id);
        }
    }

    fn end_frame(&mut self) {
        unsafe {
            // Do any final commits to DC
            wrc_end_frame(self.compositor);
        }
    }

    fn deinit(&mut self) {}

    fn get_capabilities(&self) -> CompositorCapabilities {
        Default::default()
    }

    fn get_window_visibility(&self) -> WindowVisibility {
        Default::default()
    }

    fn get_window_properties(&self) -> WindowProperties {
        WindowProperties {
            is_opaque: true,
            enable_screenshot: false,
        }
    }
}
