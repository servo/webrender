/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Functionality for managing source code for shaders.
//!
//! This module is used during precompilation (build.rs) and regular compilation,
//! so it has minimal dependencies.

use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize_program", derive(Deserialize, Serialize))]
pub struct ShaderId(usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize_program", derive(Deserialize, Serialize))]
pub struct ProgramSourceDigest(usize);

impl ::std::fmt::Display for ShaderId {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

const SHADER_IMPORT: &str = "#include ";

/// Parses a shader string for imports. Imports are recursively processed, and
/// prepended to the output stream.
///
/// NOTE: `imports` tracks the `#include ` imports, in order to generate a unique ID
/// for each shader (without having to hash the source code).
pub fn parse_shader_source<G: Fn(&str) -> String>(
    shader_imports: &mut Vec<String>,
    file_path: &str,
    get_source: &G,
) {
    // Load the current shader file and append it to the
    let current_shader_source = get_source(file_path);

    for line in current_shader_source.lines() {
        if line.starts_with(SHADER_IMPORT) {
            let imports = line[SHADER_IMPORT.len() ..].split(',');
            // For each import, get the source, and recurse.
            for import in imports {
                parse_shader_source(shader_imports, import, get_source);
            }
        }
    }

    // Append the source to the output
    shader_imports.push(file_path.to_string());
}

/// Reads a shader source file from disk into a String.
pub fn shader_source_from_file(shader_path: &Path) -> String {
    assert!(shader_path.exists(), "Shader not found");
    let mut source = String::new();
    File::open(&shader_path)
        .expect("Shader not found")
        .read_to_string(&mut source)
        .unwrap();
    source
}
