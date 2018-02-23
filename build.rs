extern crate gl_generator;

use gl_generator::{Registry, Api, Profile, Fallbacks, StaticStructGenerator};
use std::env;
use std::fs::File;
use std::path::Path;

fn main() {
    let dest = env::var("OUT_DIR").unwrap();
    let mut file = File::create(&Path::new(&dest).join("egl_bindings.rs")).unwrap();

    Registry::new(Api::Egl, (1, 5), Profile::Core, Fallbacks::All, [])
        .write_bindings(StaticStructGenerator, &mut file)
        .unwrap();
}
