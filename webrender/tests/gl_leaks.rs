/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Guards against OpenGL knowledge leaking out of the GL backend.
//!
//! Scans the crate for references to `gleam` or the `gl::` module outside
//! `src/device/gl.rs`. Files that still have them must be listed in
//! `ALLOWLIST`. A stale entry fails the test too, so the list only shrinks.

use std::fs;
use std::path::{Path, PathBuf};

/// The GL backend, the only file allowed to use GL.
const GL_BACKEND: &str = "device/gl.rs";

/// Files outside the GL backend that still reference GL. Remove entries as
/// the leaks are fixed. Do not add new entries.
const ALLOWLIST: &[&str] = &[];

/// Files that are GL-specific by design and are never expected to be cleaned up.
const PERMANENT_ALLOWLIST: &[&str] = &[
    // The SWGL compositor drives the software GL implementation directly.
    "compositor/sw_compositor.rs",
    // Selects the backend, so it names the native context type of each one.
    "device/mod.rs",
];

fn is_ident_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

fn line_has_gl_reference(line: &str) -> bool {
    if line.contains("gleam") {
        return true;
    }
    let bytes = line.as_bytes();
    let mut start = 0;
    while let Some(pos) = line[start..].find("gl::") {
        let idx = start + pos;
        let preceded_by_ident = idx > 0 && is_ident_char(bytes[idx - 1]);
        if !preceded_by_ident {
            return true;
        }
        start = idx + 4;
    }
    false
}

fn collect_rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("failed to read source directory") {
        let path = entry.expect("failed to read directory entry").path();
        if path.is_dir() {
            collect_rust_files(&path, out);
        } else if path.extension().map_or(false, |ext| ext == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn gl_does_not_leak_outside_device_layer() {
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    collect_rust_files(&src_dir, &mut files);

    let mut unexpected = Vec::new();
    let mut seen_allowlisted = Vec::new();

    for path in files {
        let rel = path
            .strip_prefix(&src_dir)
            .unwrap()
            .to_str()
            .unwrap()
            .replace('\\', "/");

        if rel == GL_BACKEND {
            continue;
        }

        let source = fs::read_to_string(&path).expect("failed to read source file");
        let leaks: Vec<(usize, &str)> = source
            .lines()
            .enumerate()
            .filter(|(_, line)| line_has_gl_reference(line))
            .map(|(i, line)| (i + 1, line.trim()))
            .collect();

        if leaks.is_empty() {
            continue;
        }

        if PERMANENT_ALLOWLIST.contains(&rel.as_str()) {
            continue;
        }

        if ALLOWLIST.contains(&rel.as_str()) {
            seen_allowlisted.push(rel);
            continue;
        }

        for (line_no, line) in leaks {
            unexpected.push(format!("{}:{}: {}", rel, line_no, line));
        }
    }

    let stale: Vec<&&str> = ALLOWLIST
        .iter()
        .filter(|entry| !seen_allowlisted.iter().any(|seen| seen == *entry))
        .collect();

    assert!(
        unexpected.is_empty(),
        "GL references found outside src/{}. Route them through the GpuBackend \
         trait instead of adding to the allowlist:\n{}",
        GL_BACKEND,
        unexpected.join("\n"),
    );

    assert!(
        stale.is_empty(),
        "Allowlist entries no longer contain GL references; remove them from \
         ALLOWLIST in tests/gl_leaks.rs: {:?}",
        stale,
    );
}
