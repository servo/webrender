#[cfg(not(windows))]
fn main() {
    println!("This demo only runs on Windows.");
}

#[cfg(windows)]
include!("main_windows.rs");
