/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Locating the RenderDoc UI (`qrenderdoc`) and opening captures from wrshell.

use std::path::PathBuf;
use std::process::Command;

/// Locate the `qrenderdoc` binary. Search order:
/// 1. `$WR_RENDERDOC_DIR/bin/qrenderdoc` (set by `mach wrshell`).
/// 2. `qrenderdoc` on `$PATH`.
/// 3. The mozbuild cache: `~/.mozbuild/renderdoc/<version>/bin/qrenderdoc`.
pub fn find_qrenderdoc() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("WR_RENDERDOC_DIR") {
        let candidate = PathBuf::from(dir).join("bin").join("qrenderdoc");
        if candidate.exists() {
            return Some(candidate);
        }
    }

    if let Ok(path) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path) {
            let candidate = dir.join("qrenderdoc");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        let base = home.join(".mozbuild").join("renderdoc");
        if let Ok(entries) = std::fs::read_dir(&base) {
            for entry in entries.flatten() {
                let candidate = entry.path().join("bin").join("qrenderdoc");
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }
    }

    None
}

/// Open a `.rdc` capture in the RenderDoc UI, if `qrenderdoc` can be found.
pub fn open_capture(rdc_path: &str) -> Result<PathBuf, String> {
    let qrenderdoc = find_qrenderdoc().ok_or_else(|| {
        "qrenderdoc not found (set WR_RENDERDOC_DIR or add it to PATH)".to_string()
    })?;
    Command::new(&qrenderdoc)
        .arg(rdc_path)
        .spawn()
        .map(|_| qrenderdoc.clone())
        .map_err(|e| format!("failed to launch {}: {}", qrenderdoc.display(), e))
}
