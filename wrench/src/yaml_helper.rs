/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::{Size2D, Point2D, Rect, Matrix4D};
use std::str::FromStr;
use app_units::Au;

use yaml_rust::Yaml;

use webrender_traits::*;

pub trait YamlHelper {
    fn as_force_f32(&self) -> Option<f32>;
    fn as_vec_f32(&self) -> Option<Vec<f32>>;
    fn as_vec_u32(&self) -> Option<Vec<u32>>;
    fn as_rect(&self) -> Option<Rect<f32>>;
    fn as_size(&self) -> Option<Size2D<f32>>;
    fn as_point(&self) -> Option<Point2D<f32>>;
    fn as_matrix4d(&self) -> Option<Matrix4D<f32>>;
    fn as_colorf(&self) -> Option<ColorF>;
    fn as_complex_clip_rect(&self) -> Option<ComplexClipRegion>;
    fn as_clip_region(&self, &mut AuxiliaryListsBuilder) -> Option<ClipRegion>;
    fn as_px_to_au(&self) -> Option<Au>;
    fn as_pt_to_au(&self) -> Option<Au>;
}

impl YamlHelper for Yaml {
    fn as_force_f32(&self) -> Option<f32> {
        match *self {
            Yaml::Integer(iv) => Some(iv as f32),
            Yaml::String(ref sv) => match f32::from_str(sv.as_str()) {
                Ok(v) => Some(v),
                Err(_) => None
            },
            _ => None
        }
    }

    fn as_vec_f32(&self) -> Option<Vec<f32>> {
        if let Some(v) = self.as_str() {
            Some(v.split_whitespace()
                 .map(|v| f32::from_str(v).expect(&format!("expected float value, got '{:?}'", v)))
                 .collect())
        } else if let Some(v) = self.as_vec() {
            Some(v.iter().map(|v| {
                match *v {
                    Yaml::Integer(k) => k as f32,
                    Yaml::String(ref k) | Yaml::Real(ref k) => f32::from_str(&k).expect(&format!("expected float value, got '{:?}'", v)),
                    _ => panic!("expected float value, got '{:?}'", v),
                }
            }).collect())
        } else {
            None
        }
    }

    fn as_vec_u32(&self) -> Option<Vec<u32>> {
        if let Some(v) = self.as_vec() {
            Some(v.iter().map(|v| v.as_i64().unwrap() as u32).collect())
        } else {
            None
        }
    }

    fn as_px_to_au(&self) -> Option<Au> {
        match self.as_force_f32() {
            Some(fv) => Some(Au::from_f32_px(fv)),
            None => None
        }
    }

    fn as_pt_to_au(&self) -> Option<Au> {
        match self.as_force_f32() {
            Some(fv) => Some(Au::from_f32_px(fv * 16. / 12.)),
            None => None
        }
    }

    fn as_rect(&self) -> Option<Rect<f32>> {
        if self.is_badvalue() {
            return None;
        }

        let nums = self.as_vec_f32().unwrap();
        if nums.len() != 4 {
            panic!("rect expected 4 float values, got {} instead ('{:?}')", nums.len(), self);
        }
        Some(Rect::new(Point2D::new(nums[0], nums[1]), Size2D::new(nums[2], nums[3])))
    }

    fn as_size(&self) -> Option<Size2D<f32>> {
        if self.is_badvalue() {
            return None;
        }

        let nums = self.as_vec_f32().unwrap();
        if nums.len() != 2 {
            panic!("size expected 2 float values, got {} instead ('{:?}')", nums.len(), self);
        }
        Some(Size2D::new(nums[0], nums[1]))
    }

    fn as_point(&self) -> Option<Point2D<f32>> {
        if self.is_badvalue() {
            return None;
        }

        let nums = self.as_vec_f32().unwrap();
        if nums.len() != 2 {
            panic!("point expected 2 float values, got {} instead ('{:?}')", nums.len(), self);
        }
        Some(Point2D::new(nums[0], nums[1]))
    }

    fn as_matrix4d(&self) -> Option<Matrix4D<f32>> {
        None
    }

    fn as_colorf(&self) -> Option<ColorF> {
        match self.as_str() {
            None => None,
            Some("red") => Some(ColorF::new(1.0, 0.0, 0.0, 1.0)),
            Some("green") => Some(ColorF::new(0.0, 1.0, 0.0, 1.0)),
            Some("blue") => Some(ColorF::new(0.0, 0.0, 1.0, 1.0)),
            Some("white") => Some(ColorF::new(1.0, 1.0, 1.0, 1.0)),
            Some("black") => Some(ColorF::new(0.0, 0.0, 0.0, 1.0)),
            _ => {
                let mut nums = self.as_vec_f32().unwrap();
                if nums.len() != 3 && nums.len() != 4 {
                    panic!("color expected a color name, or 3-4 floats; got '{:?}'", self);
                }

                if nums.len() == 3 {
                    nums.push(1.0);
                }
                Some(ColorF::new(nums[0] / 255.0, nums[1] / 255.0, nums[2] / 255.0, nums[3]))
            }
        }
    }

    fn as_complex_clip_rect(&self) -> Option<ComplexClipRegion> {
        if self.is_badvalue() {
            return None;
        }

        let nums = self.as_vec_f32().unwrap();
        match nums.len() {
            4 => Some(ComplexClipRegion::new(Rect::new(Point2D::new(nums[0], nums[1]), Size2D::new(nums[2], nums[3])),
                                             BorderRadius::zero())),
            5 => Some(ComplexClipRegion::new(Rect::new(Point2D::new(nums[0], nums[1]), Size2D::new(nums[2], nums[3])),
                                             BorderRadius::uniform(nums[4]))),
            8 => Some(ComplexClipRegion::new(Rect::new(Point2D::new(nums[0], nums[1]), Size2D::new(nums[2], nums[3])),
                                             BorderRadius {
                                                 top_left: Size2D::new(nums[4], nums[4]),
                                                 top_right: Size2D::new(nums[5], nums[5]),
                                                 bottom_left: Size2D::new(nums[6], nums[6]),
                                                 bottom_right: Size2D::new(nums[7], nums[7]),
                                             })),
            12 => Some(ComplexClipRegion::new(Rect::new(Point2D::new(nums[0], nums[1]), Size2D::new(nums[2], nums[3])),
                                              BorderRadius {
                                                  top_left: Size2D::new(nums[4], nums[5]),
                                                  top_right: Size2D::new(nums[6], nums[7]),
                                                  bottom_left: Size2D::new(nums[8], nums[9]),
                                                  bottom_right: Size2D::new(nums[10], nums[11]),
                                              })),
            n => panic!("complex clip rect expected 4, 5, 8, or 12 floats; got {} instead at '{:?}'", n, self),
        }
    }

    fn as_clip_region(&self, mut auxiliary_lists_builder: &mut AuxiliaryListsBuilder) -> Option<ClipRegion> {
        if self.is_badvalue() {
            return None;
        }

        // TODO add support for clip masks
        // TODO add support for rounded rect clips

        // if it's not a vec, then assume it's a single rect
        if self.as_vec().is_none() {
            let rect = self.as_rect().expect(&format!("clip region '{:?}', thought it was a rect but it's not?", self));
            return Some(ClipRegion::new(&rect, Vec::new(), None, &mut auxiliary_lists_builder));
        }

        // otherwise it's an array of complex clip rects
        let mut bounds = Rect::<f32>::zero();
        let mut clips = Vec::<ComplexClipRegion>::new();

        for item in self.as_vec().unwrap() {
            let c = item.as_complex_clip_rect().unwrap();
            bounds = bounds.union(&c.rect);
            clips.push(c);
        }

        Some(ClipRegion::new(&bounds, clips, None, &mut auxiliary_lists_builder))
    }
}
