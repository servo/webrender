/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate webrender_build;

use std::env;
use std::fs::{canonicalize, read_dir, File};
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use webrender_build::shader::*;

static UNIQUE_SHADER_ID: AtomicUsize = AtomicUsize::new(0);

fn write_shaders(glsl_files: Vec<PathBuf>, shader_file_path: &Path) {

    use std::collections::HashMap;

    /// Given the imports of a shader, returns a unique shader ID, given the
    fn compute_unique_shader_id(
        shader_source_paths: &Vec<String>,
        all_shaders: &mut HashMap<Vec<String>, usize>
    ) -> usize {
        *all_shaders
            .entry(shader_source_paths.clone())
            .or_insert_with(|| UNIQUE_SHADER_ID.fetch_add(1, Ordering::SeqCst))
    }

    // Compute the source code block, i.e. `shader_hashmap_str` will contain the lines:
    //
    // h.insert("foo_shader", SourceWithId { ... });
    // h.insert("bar_shader", SourceWithId { ... });
    //
    // .. which then get inserted into the `./combined_shader.rs` template at build time.
    let mut shader_hashmap_str = Vec::new();
    // Stores a "file path -> unique ID" mapping for each file path
    let mut all_shader_files = HashMap::new();

    for glsl_file_path in glsl_files {

        // Compute the shader name.
        assert!(glsl_file_path.is_file());
        let shader_name = glsl_file_path.file_name().unwrap().to_str().unwrap();
        let shader_name = shader_name.replace(".glsl", "");

        let base = glsl_file_path.parent().unwrap();
        assert!(base.is_dir());

        // Imports used in this shader (i.e. what sub-shaders the file is made of)
        let mut shader_source_paths = Vec::new();

        let full_name = canonicalize_path(&glsl_file_path);

        parse_shader_source(
            &mut shader_source_paths,
            &full_name,
            &|f| shader_source_from_file(&base.join(&format!("{}.glsl", f))),
        );

        let shader_id = compute_unique_shader_id(&shader_source_paths, &mut all_shader_files);

        let source_code = format!(
            "h.insert(\"{}\", SourceWithId {{ source: include_str!(\"{}\"), id: ShaderId({}) }});",
            shader_name, full_name, shader_id
        );

        shader_hashmap_str.push(source_code);
    }

    let shader_source_code = shader_hashmap_str.join("\n");

    // Format and write the file to a combined shader
    let mut shader_file = File::create(shader_file_path).unwrap();
    write!(shader_file, include_str!("./combined_shaders.rs"), shader_source_code);

}

/// Compute the shader path for insertion into the include_str!() macro.
/// This makes for more compact generated code than inserting the literal
/// shader source into the generated file.
///
/// If someone is building on a network share, I'm sorry.
fn canonicalize_path(input: &PathBuf) -> String {
    let full_path = canonicalize(&input).unwrap();
    let full_name = full_path.as_os_str().to_str().unwrap();
    let full_name = full_name.replace("\\\\?\\", "");
    let full_name = full_name.replace("\\", "/");
    full_name
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap_or("out".to_owned());

    let shaders_file = Path::new(&out_dir).join("shaders.rs");
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

    // Sort the file list so that the shaders.rs file is filled
    // deterministically.
    glsl_files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    write_shaders(glsl_files, &shaders_file);
}
