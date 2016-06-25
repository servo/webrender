/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate serde_codegen;

use std::env;
use std::path::Path;

pub fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();

    let src = Path::new("src/types.rs");
    let dst = Path::new(&out_dir).join("types.rs");

    serde_codegen::expand(&src, &dst).unwrap();
    println!("cargo:rerun-if-changed=src/types.rs");
}
