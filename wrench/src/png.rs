/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use WindowWrapper;
use image::png::PNGEncoder;
use image::{self, ColorType, GenericImage};
use std::fs::File;
use std::path::Path;
use std::sync::mpsc::Receiver;
use webrender::api::*;
use wrench::{Wrench, WrenchThing};
use yaml_frame_reader::YamlFrameReader;

pub fn save_flipped<P: Clone + AsRef<Path>>(
    path: P,
    orig_pixels: Vec<u8>,
    mut size: DeviceUintSize
) {
    let mut buffer = image::RgbaImage::from_raw(
        size.width,
        size.height,
        orig_pixels,
    ).expect("bug: unable to construct image buffer");

    // flip image vertically (texture is upside down)
    buffer = image::imageops::flip_vertical(&buffer);

    if let Ok(existing_image) = image::open(path.clone()) {
        let old_dims = existing_image.dimensions();
        println!("Crop from {:?} to {:?}", size, old_dims);
        size.width = old_dims.0;
        size.height = old_dims.1;
        buffer = image::imageops::crop(
            &mut buffer,
            0,
            0,
            size.width,
            size.height
        ).to_image();
    }

    let encoder = PNGEncoder::new(File::create(path).unwrap());
    encoder
        .encode(&buffer.into_vec(), size.width, size.height, ColorType::RGBA(8))
        .expect("Unable to encode PNG!");
}

pub fn png(wrench: &mut Wrench, window: &mut WindowWrapper, mut reader: YamlFrameReader, rx: Receiver<()>) {
    reader.do_frame(wrench);

    // wait for the frame
    rx.recv().unwrap();
    wrench.render();

    let size = window.get_inner_size_pixels();
    let device_size = DeviceUintSize::new(size.0, size.1);
    let data = wrench
        .renderer
        .read_pixels_rgba8(DeviceUintRect::new(DeviceUintPoint::zero(), device_size));

    let mut out_path = reader.yaml_path().clone();
    out_path.set_extension("png");

    save_flipped(out_path, data, device_size);
}
