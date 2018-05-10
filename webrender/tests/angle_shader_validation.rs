/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate mozangle;
extern crate webrender;

use mozangle::shaders::{BuiltInResources, Output, ShaderSpec, ShaderValidator};

// from glslang
const FRAGMENT_SHADER: u32 = 0x8B30;
const VERTEX_SHADER: u32 = 0x8B31;

struct Shader {
    name: &'static str,
    features: &'static [&'static str],
}

const SHADER_PREFIX: &str = "#define WR_MAX_VERTEX_TEXTURE_WIDTH 1024\n";

const CLIP_FEATURES: &[&str] = &["TRANSFORM"];
const CACHE_FEATURES: &[&str] = &[""];
const PRIM_FEATURES: &[&str] = &["", "TRANSFORM"];

const SHADERS: &[Shader] = &[
    // Clip mask shaders
    Shader {
        name: "cs_clip_rectangle",
        features: CLIP_FEATURES,
    },
    Shader {
        name: "cs_clip_image",
        features: CLIP_FEATURES,
    },
    Shader {
        name: "cs_clip_box_shadow",
        features: CLIP_FEATURES,
    },
    Shader {
        name: "cs_clip_border",
        features: CLIP_FEATURES,
    },
    Shader {
        name: "cs_clip_line",
        features: CLIP_FEATURES,
    },
    // Cache shaders
    Shader {
        name: "cs_blur",
        features: CACHE_FEATURES,
    },
    // Prim shaders
    Shader {
        name: "ps_border_corner",
        features: PRIM_FEATURES,
    },
    Shader {
        name: "ps_border_edge",
        features: PRIM_FEATURES,
    },
    Shader {
        name: "ps_split_composite",
        features: PRIM_FEATURES,
    },
    Shader {
        name: "ps_text_run",
        features: PRIM_FEATURES,
    },
    // Brush shaders
    Shader {
        name: "brush_yuv_image",
        features: &["", "YUV_NV12", "YUV_PLANAR", "YUV_INTERLEAVED", "YUV_NV12,TEXTURE_RECT"],
    },
    Shader {
        name: "brush_solid",
        features: &[],
    },
    Shader {
        name: "brush_image",
        features: &["", "ALPHA_PASS"],
    },
    Shader {
        name: "brush_blend",
        features: &[],
    },
    Shader {
        name: "brush_mix_blend",
        features: &[],
    },
    Shader {
        name: "brush_radial_gradient",
        features: &[ "DITHERING" ],
    },
    Shader {
        name: "brush_linear_gradient",
        features: &[],
    },
];

const VERSION_STRING: &str = "#version 300 es\n";

#[test]
fn validate_shaders() {
    mozangle::shaders::initialize().unwrap();

    let resources = BuiltInResources::default();
    let vs_validator =
        ShaderValidator::new(VERTEX_SHADER, ShaderSpec::Gles3, Output::Essl, &resources).unwrap();

    let fs_validator =
        ShaderValidator::new(FRAGMENT_SHADER, ShaderSpec::Gles3, Output::Essl, &resources).unwrap();

    for shader in SHADERS {
        for config in shader.features {
            let mut features = String::new();
            features.push_str(SHADER_PREFIX);

            for feature in config.split(",") {
                features.push_str(&format!("#define WR_FEATURE_{}", feature));
            }

            let (vs, fs) =
                webrender::build_shader_strings(VERSION_STRING, &features, shader.name, &None);

            validate(&vs_validator, shader.name, vs);
            validate(&fs_validator, shader.name, fs);
        }
    }
}

fn validate(validator: &ShaderValidator, name: &str, source: String) {
    // Check for each `switch` to have a `default`, see
    // https://github.com/servo/webrender/wiki/Driver-issues#lack-of-default-case-in-a-switch
    assert_eq!(source.matches("switch").count(), source.matches("default:").count(),
        "Shader '{}' doesn't have all `switch` covered with `default` cases", name);
    // Run Angle validator
    match validator.compile_and_translate(&[&source]) {
        Ok(_) => {
            println!("Shader translated succesfully: {}", name);
        }
        Err(_) => {
            panic!(
                "Shader compilation failed: {}\n{}",
                name,
                validator.info_log()
            );
        }
    }
}
