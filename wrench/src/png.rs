/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{WindowWrapper, NotifierEvent};
use image::png::PNGEncoder;
use image::{self, ColorType, GenericImageView};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use webrender::api::units::*;
use crate::wrench::{Wrench, WrenchThing};
use crate::yaml_frame_reader::YamlFrameReader;

#[derive(Copy, Clone)]
pub enum ReadSurface {
    Screen,
}

pub struct SaveSettings {
    pub flip_vertical: bool,
    pub try_crop: bool,
}

pub fn save<P: Clone + AsRef<Path>>(
    path: P,
    orig_pixels: Vec<u8>,
    size: DeviceIntSize,
    settings: SaveSettings
) {
    let mut width = size.width as u32;
    let mut height = size.height as u32;
    let mut buffer = image::RgbaImage::from_raw(
        width,
        height,
        orig_pixels,
    ).expect("bug: unable to construct image buffer");

    if settings.flip_vertical {
        // flip image vertically (texture is upside down)
        buffer = image::imageops::flip_vertical(&buffer);
    }

    if settings.try_crop {
        if let Ok(existing_image) = image::open(path.clone()) {
            let old_dims = existing_image.dimensions();
            println!("Crop from {:?} to {:?}", size, old_dims);
            // Cropping can only shrink: image::imageops::crop clamps the crop
            // rect to the source buffer, so trusting a larger existing image
            // would make the dimensions disagree with the pixel data.
            width = old_dims.0.min(width);
            height = old_dims.1.min(height);
            buffer = image::imageops::crop(
                &mut buffer,
                0,
                0,
                width,
                height
            ).to_image();
        }
    }

    let encoder = PNGEncoder::new(File::create(path).unwrap());
    encoder
        .encode(&buffer, width, height, ColorType::Rgba8)
        .expect("Unable to encode PNG!");
}

pub fn save_flipped<P: Clone + AsRef<Path>>(
    path: P,
    orig_pixels: Vec<u8>,
    size: DeviceIntSize,
) {
    save(path, orig_pixels, size, SaveSettings {
        flip_vertical: true,
        try_crop: true,
    })
}

pub fn png(
    wrench: &mut Wrench,
    surface: ReadSurface,
    window: &mut WindowWrapper,
    mut reader: YamlFrameReader,
    rx: &Receiver<NotifierEvent>,
    out_path: Option<PathBuf>,
) {
    reader.do_frame(wrench);

    // wait for the frame
    rx.recv().unwrap();
    wrench.render();

    let (fb_size, data, settings) = match surface {
        ReadSurface::Screen => {
            let dim = window.get_inner_size();
            let rect = FramebufferIntSize::new(dim.width, dim.height).into();
            let data = wrench.renderer.read_pixels_rgba8(rect);
            (dim, data, SaveSettings {
                flip_vertical: true,
                try_crop: true,
            })
        }
    };

    let out_path = out_path.unwrap_or_else(|| {
        let mut path = reader.yaml_path().clone();
        path.set_extension("png");
        path
    });

    save(out_path, data, fb_size, settings);
}
