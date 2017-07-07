/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::Point2D;
use ColorU;

#[repr(C)]
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct GeometryKey(pub u32, pub u32);

impl GeometryKey {
    pub fn new(key0: u32, key1: u32) -> GeometryKey {
        GeometryKey(key0, key1)
    }
}

pub type Geometry = Vec<GeometryItem>;

#[derive(Clone, Deserialize, Serialize)]
pub enum GeometryItem {
    Shape(Shape)
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Shape {
    pub path: Vec<Command>,
    pub fill: ColorU
}

#[derive(Clone, Deserialize, Serialize)]
pub enum Command {
    MoveTo(Point2D<f32>),
    LineTo(Point2D<f32>),
    Arc(Point2D<f32>, f32, f32, f32),
}
