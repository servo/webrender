/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate ron;
#[macro_use]
extern crate serde;

use ron::de;
use std::env;
use std::fs::{canonicalize, read_dir, File};
use std::io::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const SHADER_IMPORT: &str = "#include ";
const SHADER_KIND_FRAGMENT: &str = "#define WR_FRAGMENT_SHADER\n";
const SHADER_KIND_VERTEX: &str = "#define WR_VERTEX_SHADER\n";
const SHADER_PREFIX: &str = "#define WR_MAX_VERTEX_TEXTURE_WIDTH 1024\n";

const SHADER_VERSION: &str = "#version 150\n";

#[derive(Deserialize)]
struct Shader {
    name: String,
    feature_sets: Vec<String>,
}

fn create_map(glsl_files: Vec<PathBuf>) -> HashMap<String, String> {
    let mut shader_map: HashMap<String, String> = HashMap::new();

    for glsl in glsl_files {
        let shader_name = glsl.file_name().unwrap().to_str().unwrap();
        // strip .glsl
        let shader_name = shader_name.replace(".glsl", "");
        let full_path = canonicalize(&glsl).unwrap();
        let full_name = full_path.as_os_str().to_str().unwrap();
        // if someone is building on a network share, I'm sorry.
        let full_name = full_name.replace("\\\\?\\", "");
        let full_name = full_name.replace("\\", "/");
        shader_map.insert(shader_name.clone(), full_name.clone());
    }
    shader_map
}

// Get a shader string by name
fn get_shader_source(shader_name: &str, shaders: &HashMap<String, String>) -> Option<String> {
    if let Some(shader_file) = shaders.get(shader_name) {
        let shader_file_path = Path::new(shader_file);
        if let Ok(mut shader_source_file) = File::open(shader_file_path) {
            let mut source = String::new();
            shader_source_file.read_to_string(&mut source).unwrap();
            return Some(source)
        }
    }
    None
}

// Parse a shader string for imports. Imports are recursively processed, and
// prepended to the list of outputs.
fn parse_shader_source(source: String, shaders: &HashMap<String, String>, output: &mut String) {
    for line in source.lines() {
        if line.starts_with(SHADER_IMPORT) {
            let imports = line[SHADER_IMPORT.len() ..].split(',');

            // For each import, get the source, and recurse.
            for import in imports {
                if let Some(include) = get_shader_source(import, shaders) {
                    parse_shader_source(include, shaders, output);
                }
            }
        } else {
            output.push_str(line);
            output.push_str("\n");
        }
    }
}

fn build_shader_strings(
    gl_version_string: &str,
    features: &str,
    base_filename: &str,
    shaders: &HashMap<String, String>,
) -> (String, String) {
    // Construct a list of strings to be passed to the shader compiler.
    let mut vs_source = String::new();
    let mut fs_source = String::new();

    // GLSL requires that the version number comes first.
    vs_source.push_str(gl_version_string);
    fs_source.push_str(gl_version_string);

    // Insert the shader name to make debugging easier.
    let name_string = format!("// {}\n", base_filename);
    vs_source.push_str(&name_string);
    fs_source.push_str(&name_string);

    // Define a constant depending on whether we are compiling VS or FS.
    vs_source.push_str(SHADER_KIND_VERTEX);
    fs_source.push_str(SHADER_KIND_FRAGMENT);

    // Add any defines that were passed by the caller.
    vs_source.push_str(features);
    fs_source.push_str(features);

    // Parse the main .glsl file, including any imports
    // and append them to the list of sources.
    let mut shared_result = String::new();
    if let Some(shared_source) = get_shader_source(base_filename, shaders) {
        parse_shader_source(shared_source, shaders, &mut shared_result);
    }

    vs_source.push_str(&shared_result);
    fs_source.push_str(&shared_result);

    (vs_source, fs_source)
}

fn generate_shaders(out_dir: &str, shaders: HashMap<String, String>) {
    let mut file = File::open("res/shaders.ron").expect("Unable to open shaders.ron");
    let mut source = String::new();
    file.read_to_string(&mut source).expect("Unable to read shaders.ron");
    let shader_configs: Vec<Shader> = de::from_str(&source).expect("Unable to deserialize shaders.ron");

    for shader in shader_configs {
        let mut feature_variants: Vec<String> = Vec::new();

        // Building up possible permutations of features
        for feature_set in shader.feature_sets {
            if feature_variants.is_empty() {
                feature_variants = feature_set.split(',').map(|s| s.to_owned()).collect();
            } else {
                let prev_variants: Vec<String> = feature_variants.drain(..).collect();
                for variant in prev_variants.iter() {
                    for feature in feature_set.split(',') {
                        feature_variants.push(format!("{},{}", variant, feature));
                    }
                }
            }
        }

        // Creating a shader file for each permutation
        for variant in feature_variants {
            let mut features = String::new();

            features.push_str(SHADER_PREFIX);
            features.push_str(format!("//Source: {}.glsl\n", shader.name).as_str());

            let mut file_name_postfix = String::new();
            for feature in variant.split(',') {
                if !feature.is_empty() {
                    features.push_str(&format!("#define WR_FEATURE_{}\n", feature));
                    file_name_postfix
                        .push_str(&format!("_{}", feature.to_lowercase().as_str()));
                }
            }

            let (mut vs_source, mut fs_source) =
                build_shader_strings(SHADER_VERSION, &features, &shader.name, &shaders);

            let (vs_file_path, fs_file_path) = (
                Path::new(out_dir).join(format!("{}{}.vert", &shader.name, &file_name_postfix)),
                Path::new(out_dir).join(format!("{}{}.frag", &shader.name, &file_name_postfix)),
            );
            let (mut vs_file, mut fs_file) = (
                File::create(vs_file_path).unwrap(),
                File::create(fs_file_path).unwrap(),
            );
            write!(vs_file, "{}", vs_source).unwrap();
            write!(fs_file, "{}", fs_source).unwrap();
        }
    }
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap_or("out".to_owned());

    let mut glsl_files = vec![];

    println!("cargo:rerun-if-changed=res");
    let res_dir = Path::new("res");
    for entry in read_dir(res_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if entry.file_name().to_str().unwrap().ends_with(".glsl") {
            println!("cargo:rerun-if-changed={}", path.display());
            glsl_files.push(path.to_owned());
        }
    }
    let shaders_map = create_map(glsl_files);
    generate_shaders(&out_dir, shaders_map);
}
