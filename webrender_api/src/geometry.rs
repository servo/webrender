/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::Point2D;
use {BlobImageDescriptor, BlobImageResources, BlobImageResult};
use ColorU;
use FontKey;
use {IdNamespace};

#[repr(C)]
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct GeometryKey(pub IdNamespace, pub u32);

impl GeometryKey {
    pub fn new(namespace: IdNamespace, key1: u32) -> GeometryKey {
        GeometryKey(namespace, key1)
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

pub trait SvgRenderer: Send {
    fn update(&mut self, key: GeometryKey, data: Geometry);

    fn delete(&mut self, key: GeometryKey);

    fn request(&mut self,
               services: &BlobImageResources,
               key: GeometryKey,
               descriptor: &BlobImageDescriptor);

    fn resolve(&mut self, key: GeometryKey) -> BlobImageResult;

    fn delete_font(&mut self, key: FontKey);
}

