/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::str::FromStr;
use app_units::Au;
use euclid::TypedSize2D;

use yaml_rust::Yaml;

use webrender_traits::*;

pub trait YamlHelper {
    fn as_force_f32(&self) -> Option<f32>;
    fn as_vec_f32(&self) -> Option<Vec<f32>>;
    fn as_vec_u32(&self) -> Option<Vec<u32>>;
    fn as_pipeline_id(&self) -> Option<PipelineId>;
    fn as_rect(&self) -> Option<LayoutRect>;
    fn as_size(&self) -> Option<LayoutSize>;
    fn as_point(&self) -> Option<LayoutPoint>;
    fn as_matrix4d(&self) -> Option<LayoutTransform>;
    fn as_colorf(&self) -> Option<ColorF>;
    fn as_vec_colorf(&self) -> Option<Vec<ColorF>>;
    fn as_px_to_au(&self) -> Option<Au>;
    fn as_pt_to_au(&self) -> Option<Au>;
    fn as_vec_string(&self) -> Option<Vec<String>>;
    fn as_border_radius(&self) -> Option<BorderRadius>;
}

fn string_to_color(color: &str) -> Option<ColorF> {
    match color {
        "red" => Some(ColorF::new(1.0, 0.0, 0.0, 1.0)),
        "green" => Some(ColorF::new(0.0, 1.0, 0.0, 1.0)),
        "blue" => Some(ColorF::new(0.0, 0.0, 1.0, 1.0)),
        "white" => Some(ColorF::new(1.0, 1.0, 1.0, 1.0)),
        "black" => Some(ColorF::new(0.0, 0.0, 0.0, 1.0)),
        s => {
            let items: Vec<f32> = s.split_whitespace().map(|s| f32::from_str(s).unwrap()).collect();
            if items.len() == 3 {
                Some(ColorF::new(items[0] / 255.0, items[1] / 255.0, items[2] / 255.0, 1.0))
            } else if items.len() == 4 {
                Some(ColorF::new(items[0] / 255.0, items[1] / 255.0, items[2] / 255.0, items[3]))
            } else {
                None
            }
        }
    }
}

impl YamlHelper for Yaml {
    fn as_force_f32(&self) -> Option<f32> {
        match *self {
            Yaml::Integer(iv) => Some(iv as f32),
            Yaml::String(ref sv) | Yaml::Real(ref sv) => match f32::from_str(sv.as_str()) {
                Ok(v) => Some(v),
                Err(_) => None
            },
            _ => None
        }
    }

    fn as_vec_f32(&self) -> Option<Vec<f32>> {
        match *self {
            Yaml::String(ref s) | Yaml::Real(ref s) => {
                s.split_whitespace()
                 .map(|v| f32::from_str(v))
                 .collect::<Result<Vec<_>,_>>()
                 .ok()
            }
            Yaml::Array(ref v) => {
                v.iter().map(|v| {
                    match *v {
                        Yaml::Integer(k) => Ok(k as f32),
                        Yaml::String(ref k) | Yaml::Real(ref k) => {
                            f32::from_str(&k).map_err(|_| false)
                        },
                        _ => Err(false),
                    }
                }).collect::<Result<Vec<_>,_>>().ok()
            }
            Yaml::Integer(k) => {
                Some(vec![k as f32])
            }
            _ => None
        }
    }

    fn as_vec_u32(&self) -> Option<Vec<u32>> {
        if let Some(v) = self.as_vec() {
            Some(v.iter().map(|v| v.as_i64().unwrap() as u32).collect())
        } else {
            None
        }
    }

    fn as_pipeline_id(&self) -> Option<PipelineId> {
        if let Some(mut v) = self.as_vec() {
            let a = v.get(0).and_then(|v| v.as_i64()).map(|v| v as u32);
            let b = v.get(1).and_then(|v| v.as_i64()).map(|v| v as u32);
            match (a, b) {
                (Some(a), Some(b)) if v.len() == 2 => Some(PipelineId(a, b)),
                _ => None,
            }
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

    fn as_rect(&self) -> Option<LayoutRect> {
        if self.is_badvalue() {
            return None;
        }

        if let Some(nums) = self.as_vec_f32() {
            if nums.len() == 4 {
                return Some(LayoutRect::new(LayoutPoint::new(nums[0], nums[1]), LayoutSize::new(nums[2], nums[3])));
            }
        }

        None
    }

    fn as_size(&self) -> Option<LayoutSize> {
        if self.is_badvalue() {
            return None;
        }

        let nums = self.as_vec_f32().unwrap();
        if nums.len() != 2 {
            panic!("size expected 2 float values, got {} instead ('{:?}')", nums.len(), self);
        }
        Some(LayoutSize::new(nums[0], nums[1]))
    }

    fn as_point(&self) -> Option<LayoutPoint> {
        if self.is_badvalue() {
            return None;
        }

        let nums = self.as_vec_f32().unwrap();
        if nums.len() != 2 {
            panic!("point expected 2 float values, got {} instead ('{:?}')", nums.len(), self);
        }
        Some(LayoutPoint::new(nums[0], nums[1]))
    }

    fn as_matrix4d(&self) -> Option<LayoutTransform> {
        None
    }

    fn as_colorf(&self) -> Option<ColorF> {
        if let Some(mut nums) = self.as_vec_f32() {
            if nums.len() != 3 && nums.len() != 4 {
                panic!("color expected a color name, or 3-4 floats; got '{:?}'", self);
            }

            if nums.len() == 3 {
                nums.push(1.0);
            }
            return Some(ColorF::new(nums[0] / 255.0, nums[1] / 255.0, nums[2] / 255.0, nums[3]));
        }

        if let Some(s) = self.as_str() {
            string_to_color(s)
        } else {
            None
        }
    }

    fn as_vec_colorf(&self) -> Option<Vec<ColorF>> {
        if let Some(v) = self.as_vec() {
            Some(v.iter().map(|v| v.as_colorf().unwrap()).collect())
        } else {
            if let Some(color) = self.as_colorf() {
                Some(vec![color])
            } else {
                None
            }
        }
    }

    fn as_vec_string(&self) -> Option<Vec<String>> {
        if let Some(v) = self.as_vec() {
            Some(v.iter().map(|v| v.as_str().unwrap().to_owned()).collect())
        } else if let Some(s) = self.as_str() {
            Some(vec![s.to_owned()])
        } else {
            None
        }
    }

    fn as_border_radius(&self) -> Option<BorderRadius> {
        match *self {
            Yaml::BadValue => { None }
            Yaml::String(ref s) | Yaml::Real(ref s) => {
                let fv = f32::from_str(s).unwrap();
                Some(BorderRadius::uniform(fv))
            }
            Yaml::Integer(v) => {
                Some(BorderRadius::uniform(v as f32))
            }
            Yaml::Hash(_) => {
                let top_left = self["top_left"].as_size().unwrap_or(TypedSize2D::zero());
                let top_right = self["top_right"].as_size().unwrap_or(TypedSize2D::zero());
                let bottom_left = self["bottom_left"].as_size().unwrap_or(TypedSize2D::zero());
                let bottom_right = self["bottom_right"].as_size().unwrap_or(TypedSize2D::zero());
                Some(BorderRadius {
                    top_left: top_left,
                    top_right: top_right,
                    bottom_left: bottom_left,
                    bottom_right: bottom_right,
                })
            }
            _ => {
                panic!("Invalid border radius specified: {:?}", self);
            }
        }
    }
}
