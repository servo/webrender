/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate mozangle;
extern crate ron;
#[macro_use]
extern crate serde;
extern crate webrender;

use mozangle::shaders::{BuiltInResources, Output, ShaderSpec, ShaderValidator};
use ron::de;
use std::fs::File;
use std::path::PathBuf;

// from glslang
const FRAGMENT_SHADER: u32 = 0x8B30;
const VERTEX_SHADER: u32 = 0x8B31;

const VERSION_STRING: &str = "#version 300 es\n";

// Extensions required by these features are not supported by the angle shader validator.
const EXCLUDED_FEATURES: &'static[&'static str] = &["DUAL_SOURCE_BLENDING", "TEXTURE_EXTERNAL", "TEXTURE_RECT"];

#[derive(Deserialize)]
struct Shader {
    name: String,
    feature_sets: Vec<String>,
}

#[test]
fn validate_shaders() {
    mozangle::shaders::initialize().unwrap();

    let resources = BuiltInResources::default();
    let vs_validator =
        ShaderValidator::new(VERTEX_SHADER, ShaderSpec::Gles3, Output::Essl, &resources).unwrap();

    let fs_validator =
        ShaderValidator::new(FRAGMENT_SHADER, ShaderSpec::Gles3, Output::Essl, &resources).unwrap();

    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("res");
    path.push("shaders.ron");

    let file = File::open(&path).expect("Unable to open shaders.ron");
    let shaders: Vec<Shader> = de::from_reader(file).expect("Unable to deserialize shaders.ron");

    for shader in shaders {
        let mut feature_variants: Vec<String> = Vec::new();

        // Building up possible permutations of features
        for feature_set in shader.feature_sets {
            if feature_variants.is_empty() {
                feature_variants = feature_set.split(',').map(|s| s.to_owned()).collect();
                feature_variants.retain(|f| !EXCLUDED_FEATURES.contains(&f.as_str()));
            } else {
                let prev_variants: Vec<String> = feature_variants.drain(..).collect();
                for variant in prev_variants.iter() {
                    for feature in feature_set.split(',') {
                        if !EXCLUDED_FEATURES.contains(&feature) {
                            feature_variants.push(format!("{},{}", variant, feature));
                        }
                    }
                }
            }
        }

        for variant in feature_variants {
            let features = variant.split(",").collect::<Vec<_>>();
            let (vs, fs) =
                webrender::load_shader_sources(VERSION_STRING,
                                               &features,
                                               &shader.name,
                                               &None);

            validate(&vs_validator, &shader.name, &features, vs);
            validate(&fs_validator, &shader.name, &features, fs);

        }
    }
}

fn validate(validator: &ShaderValidator, name: &str, features: &[&str], source: String) {
    // Check for each `switch` to have a `default`, see
    // https://github.com/servo/webrender/wiki/Driver-issues#lack-of-default-case-in-a-switch
    assert_eq!(source.matches("switch").count(), source.matches("default:").count(),
        "Shader '{}' doesn't have all `switch` covered with `default` cases", name);
    // Run Angle validator
    match validator.compile_and_translate(&[&source]) {
        Ok(_) => {
            println!("Shader translated succesfully: {}, features: {:?}", name, features);
        }
        Err(_) => {
            panic!(
                "Shader compilation failed: {}, features: {:?}\n{}",
                name,
                features,
                validator.info_log()
            );
        }
    }
}
