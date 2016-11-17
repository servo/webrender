/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::{Size2D, TypedRect, TypedPoint2D, TypedSize2D, Length, UnknownUnit};

// TODO: specify which of these coordinate spaces include dpi scaling factors, etc.

/// Geometry in screen-space space in pixels.
#[derive(Hash, Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct DevicePixel;

// TODO: in gecko the convention is to be explicit in the name integer-based coordinates,
// for example DeviceIntRect. It wouldn't hurt to do the same here.
pub type DeviceRect = TypedRect<i32, DevicePixel>;
pub type DevicePoint = TypedPoint2D<i32, DevicePixel>;
pub type DeviceSize = TypedSize2D<i32, DevicePixel>;
pub type DeviceLength = Length<i32, DevicePixel>;

/// Geometry in a stacking context's local coordinate space (in pixels).
#[derive(Hash, Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct LayerPixel;

pub type LayerRect = TypedRect<f32, LayerPixel>;
pub type LayerPoint = TypedPoint2D<f32, LayerPixel>;
pub type LayerSize = TypedSize2D<f32, LayerPixel>;

/// Geometry in a stacking context's parent coordinate space (in pixels).
#[derive(Hash, Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct ParentLayerPixel;

pub type ParentLayerRect = TypedRect<f32, ParentLayerPixel>;
pub type ParentLayerPoint = TypedPoint2D<f32, ParentLayerPixel>;
pub type ParentLayerSize = TypedSize2D<f32, ParentLayerPixel>;

/// Geometry in the document's coordinate space.
/// TODO: should this be LayoutPixel or CssPixel?
#[derive(Hash, Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct WorldPixel;

pub type WorldRect = TypedRect<f32, WorldPixel>;
pub type WorldPoint = TypedPoint2D<f32, WorldPixel>;
pub type WorldSize = TypedSize2D<f32, WorldPixel>;


pub fn device_pixel(value: f32, device_pixel_ratio: f32) -> DeviceLength {
    DeviceLength::new((value * device_pixel_ratio).round() as i32)
}

