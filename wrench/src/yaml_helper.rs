/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use euclid::{Radians, TypedSize2D};
use parse_function::parse_function;
use std::f32;
use std::str::FromStr;
use webrender_traits::*;
use yaml_rust::Yaml;

pub trait YamlHelper {
    fn as_force_f32(&self) -> Option<f32>;
    fn as_vec_f32(&self) -> Option<Vec<f32>>;
    fn as_vec_u32(&self) -> Option<Vec<u32>>;
    fn as_pipeline_id(&self) -> Option<PipelineId>;
    fn as_rect(&self) -> Option<LayoutRect>;
    fn as_size(&self) -> Option<LayoutSize>;
    fn as_point(&self) -> Option<LayoutPoint>;
    fn as_matrix4d(&self, transform_origin: &LayoutPoint) -> Option<LayoutTransform>;
    fn as_colorf(&self) -> Option<ColorF>;
    fn as_vec_colorf(&self) -> Option<Vec<ColorF>>;
    fn as_px_to_au(&self) -> Option<Au>;
    fn as_pt_to_au(&self) -> Option<Au>;
    fn as_vec_string(&self) -> Option<Vec<String>>;
    fn as_border_radius(&self) -> Option<BorderRadius>;
    fn as_mix_blend_mode(&self) -> Option<MixBlendMode>;
    fn as_scroll_policy(&self) -> Option<ScrollPolicy>;
    fn as_filter_op(&self) -> Option<FilterOp>;
    fn as_vec_filter_op(&self) -> Option<Vec<FilterOp>>;
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

macro_rules! define_enum_conversion {
    ($read_func:ident, $write_func:ident, $T:ty, [ $( ( $x:expr, $y:path ) ),* ]) => {
        pub fn $read_func(text: &str) -> Option<$T> {
            match text {
            $( $x => Some($y), )*
                _ => None
            }
        }
        pub fn $write_func(val: $T) -> &'static str {
            match val {
            $( $y => $x, )*
            }
        }
    }
}

define_enum_conversion!(string_to_mix_blend_mode, mix_blend_mode_to_string, MixBlendMode, [
    ("normal", MixBlendMode::Normal),
    ("multiply", MixBlendMode::Multiply),
    ("screen", MixBlendMode::Screen),
    ("overlay", MixBlendMode::Overlay),
    ("darken", MixBlendMode::Darken),
    ("lighten", MixBlendMode::Lighten),
    ("color-dodge", MixBlendMode::ColorDodge),
    ("color-burn", MixBlendMode::ColorBurn),
    ("hard-light", MixBlendMode::HardLight),
    ("soft-light", MixBlendMode::SoftLight),
    ("difference", MixBlendMode::Difference),
    ("exclusion", MixBlendMode::Exclusion),
    ("hue", MixBlendMode::Hue),
    ("saturation", MixBlendMode::Saturation),
    ("color", MixBlendMode::Color),
    ("luminosity", MixBlendMode::Luminosity)
]);

define_enum_conversion!(string_to_scroll_policy, scroll_policy_to_string, ScrollPolicy, [
    ("scrollable", ScrollPolicy::Scrollable),
    ("fixed", ScrollPolicy::Fixed)
]);

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
                 .collect::<Result<Vec<_>, _>>()
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
                }).collect::<Result<Vec<_>, _>>().ok()
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
        if let Some(v) = self.as_vec() {
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

        if let Some(nums) = self.as_vec_f32() {
            if nums.len() == 2 {
                return Some(LayoutSize::new(nums[0], nums[1]));
            }
        }

        None
    }

    fn as_point(&self) -> Option<LayoutPoint> {
        if self.is_badvalue() {
            return None;
        }

        if let Some(nums) = self.as_vec_f32() {
            if nums.len() == 2 {
                return Some(LayoutPoint::new(nums[0], nums[1]));
            }
        }

        None
    }

    fn as_matrix4d(&self, transform_origin: &LayoutPoint) -> Option<LayoutTransform> {
        if let Some(nums) = self.as_vec_f32() {
            assert_eq!(nums.len(), 16, "expected 16 floats, got '{:?}'", self);
            return Some(LayoutTransform::row_major(nums[0], nums[1], nums[2], nums[3],
                                                   nums[4], nums[5], nums[6], nums[7],
                                                   nums[8], nums[9], nums[10], nums[11],
                                                   nums[12], nums[13], nums[14], nums[15]))
        }
        match self {
            &Yaml::String(ref string) => match parse_function(string) {
                ("translate", ref args) if args.len() == 2 => {
                    return Some(LayoutTransform::create_translation(args[0].parse().unwrap(),
                                                                    args[1].parse().unwrap(),
                                                                    0.))
                }
                ("rotate", ref args) if args.len() == 1 => {
                    // rotate takes a single parameter of degrees and rotates in X-Y plane
                    let pre_transform = LayoutTransform::create_translation(transform_origin.x,
                                                                            transform_origin.y,
                                                                            0.0);
                    let post_transform = LayoutTransform::create_translation(-transform_origin.x,
                                                                             -transform_origin.y,
                                                                             -0.0);

                    let angle = args[0].parse::<f32>().unwrap().to_radians();
                    let theta = 2.0f32 * f32::consts::PI - angle;
                    let transform = LayoutTransform::identity().pre_rotated(0.0,
                                                                            0.0,
                                                                            1.0,
                                                                            Radians::new(theta));

                    Some(pre_transform.pre_mul(&transform).pre_mul(&post_transform))
                }
                (name, _) => {
                    println!("unknown function {}", name);
                    None
                }
            },
            &Yaml::BadValue => None,
            _ => {
                println!("unknown transform {:?}", self);
                None
            }
        }
    }

    fn as_colorf(&self) -> Option<ColorF> {
        if let Some(mut nums) = self.as_vec_f32() {
            assert!(nums.len() == 3 || nums.len() == 4,
                    "color expected a color name, or 3-4 floats; got '{:?}'", self);

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
        if let Some(size) = self.as_size() {
            return Some(BorderRadius::uniform_size(size));
        }

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
                let top_left = self["top-left"].as_size().unwrap_or(TypedSize2D::zero());
                let top_right = self["top-right"].as_size().unwrap_or(TypedSize2D::zero());
                let bottom_left = self["bottom-left"].as_size().unwrap_or(TypedSize2D::zero());
                let bottom_right = self["bottom-right"].as_size().unwrap_or(TypedSize2D::zero());
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

    fn as_mix_blend_mode(&self) -> Option<MixBlendMode> {
        return self.as_str().and_then(|x| string_to_mix_blend_mode(x));
    }

    fn as_scroll_policy(&self) -> Option<ScrollPolicy> {
        return self.as_str().and_then(|string| string_to_scroll_policy(string))
    }

    fn as_filter_op(&self) -> Option<FilterOp> {
        if let Some(s) = self.as_str() {
            match parse_function(s) {
                ("blur", ref args) if args.len() == 1 => {
                    Some(FilterOp::Blur(Au(args[0].parse().unwrap())))
                }
                ("brightness", ref args) if args.len() == 1 => {
                    Some(FilterOp::Brightness(args[0].parse().unwrap()))
                }
                ("contrast", ref args) if args.len() == 1 => {
                    Some(FilterOp::Contrast(args[0].parse().unwrap()))
                }
                ("grayscale", ref args) if args.len() == 1 => {
                    Some(FilterOp::Grayscale(args[0].parse().unwrap()))
                }
                ("hue-rotate", ref args) if args.len() == 1 => {
                    Some(FilterOp::HueRotate(args[0].parse().unwrap()))
                }
                ("invert", ref args) if args.len() == 1 => {
                    Some(FilterOp::Invert(args[0].parse().unwrap()))
                }
                ("opacity", ref args) if args.len() == 1 => {
                    let amount: f32 = args[0].parse().unwrap();
                    Some(FilterOp::Opacity(amount.into()))
                }
                ("saturate", ref args) if args.len() == 1 => {
                    Some(FilterOp::Saturate(args[0].parse().unwrap()))
                }
                ("sepia", ref args) if args.len() == 1 => {
                    Some(FilterOp::Sepia(args[0].parse().unwrap()))
                }
                (_, _) => { None }
            }
        } else {
            None
        }
    }

    fn as_vec_filter_op(&self) -> Option<Vec<FilterOp>> {
        if let Some(v) = self.as_vec() {
            Some(v.iter().map(|x| x.as_filter_op().unwrap()).collect())
        } else {
            self.as_filter_op().map(|op| vec![op])
        }
    }
}

