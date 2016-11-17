/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::{Size2D, TypedRect, TypedPoint2D, TypedSize2D, Length, UnknownUnit};

pub type DeviceRect = TypedRect<i32, DevicePixel>;
pub type DevicePoint = TypedPoint2D<i32, DevicePixel>;
pub type DeviceSize = TypedSize2D<i32, DevicePixel>;
pub type DeviceLength = Length<i32, DevicePixel>;

#[derive(Hash, Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct DevicePixel;

pub fn device_pixel(value: f32, device_pixel_ratio: f32) -> DeviceLength {
    DeviceLength::new((value * device_pixel_ratio).round() as i32)
}

