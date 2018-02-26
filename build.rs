extern crate gl_generator;

use gl_generator::{Registry, Api, Profile, Fallbacks};
use std::env;
use std::fs::File;
use std::path::PathBuf;

fn main() {
    // Building ANGLE is left as an exercise for the reader:
    // https://chromium.googlesource.com/angle/angle/+/HEAD/doc/DevSetup.md
    let relative_angle_dir = PathBuf::from("..").join("angle");

    let angle_build_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .join(relative_angle_dir)
        // Assume `gn gen out/Debug` or `gn gen out/Release` like in build instructions.
        .join("out")
        .join(if &env::var_os("PROFILE").unwrap() == "release" { "Release" } else { "Debug" });
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    // Assuming that OUT_DIR is something like `target/debug/build/directcomposition-*/out`,
    // bin_dir is `target/debug` (where the final executable goes).
    let bin_dir = out_dir.join("..").join("..").join("..");
    for name in &[
        "libEGL.dll.lib",
        "libEGL.dll",
        "libGLESv2.dll",
    ] {
        std::fs::copy(angle_build_dir.join(name), bin_dir.join(name)).unwrap();
    }
    println!("cargo:rustc-link-search=native={}", bin_dir.display());
    println!("cargo:rustc-link-lib=dylib=libEGL.dll");

    let bindings = "egl_bindings.rs";
    Registry::new(Api::Egl, (1, 5), Profile::Core, Fallbacks::All, [
        "EGL_ANGLE_device_d3d",
        "EGL_EXT_platform_base",
        "EGL_EXT_platform_device",
    ]).write_bindings(
        gl_generator::StaticStructGenerator,
        &mut File::create(&out_dir.join(bindings)).unwrap()
    )
    .unwrap();

}
