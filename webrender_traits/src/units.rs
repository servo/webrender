/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! A collection of coordinate spaces and their corresponding Point, Size and Rect types.
//!
//! Physical pixels take into account the device pixel ratio and their dimensions tend
//! to correspond to the allocated size of resources in memory, while logical pixels
//! don't have the device pixel ratio applied which means they are agnostic to the usage
//! of hidpi screens and the like.

use euclid::{TypedRect, TypedPoint2D, TypedSize2D, Length};

/// Geometry in screen-space in physical pixels.
#[derive(Hash, Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct DevicePixel;

pub type DeviceIntRect = TypedRect<i32, DevicePixel>;
pub type DeviceIntPoint = TypedPoint2D<i32, DevicePixel>;
pub type DeviceIntSize = TypedSize2D<i32, DevicePixel>;
pub type DeviceIntLength = Length<i32, DevicePixel>;

pub type DeviceUintRect = TypedRect<u32, DevicePixel>;
pub type DeviceUintPoint = TypedPoint2D<u32, DevicePixel>;
pub type DeviceUintSize = TypedSize2D<u32, DevicePixel>;

pub type DeviceRect = TypedRect<f32, DevicePixel>;
pub type DevicePoint = TypedPoint2D<f32, DevicePixel>;
pub type DeviceSize = TypedSize2D<f32, DevicePixel>;

/// Geometry in a stacking context's local coordinate space (logical pixels).
#[derive(Hash, Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct LayerPixel;

pub type LayerRect = TypedRect<f32, LayerPixel>;
pub type LayerPoint = TypedPoint2D<f32, LayerPixel>;
pub type LayerSize = TypedSize2D<f32, LayerPixel>;

/// Geometry in a stacking context's parent coordinate space (logical pixels).
#[derive(Hash, Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct ParentLayerPixel;

pub type ParentLayerRect = TypedRect<f32, ParentLayerPixel>;
pub type ParentLayerPoint = TypedPoint2D<f32, ParentLayerPixel>;
pub type ParentLayerSize = TypedSize2D<f32, ParentLayerPixel>;

/// Geometry in the document's coordinate space (logical pixels).
/// TODO: should this be LayoutPixel or CssPixel?
#[derive(Hash, Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct WorldPixel;

pub type WorldRect = TypedRect<f32, WorldPixel>;
pub type WorldPoint = TypedPoint2D<f32, WorldPixel>;
pub type WorldSize = TypedSize2D<f32, WorldPixel>;


pub fn device_pixel(value: f32, device_pixel_ratio: f32) -> DeviceIntLength {
    DeviceIntLength::new((value * device_pixel_ratio).round() as i32)
}

