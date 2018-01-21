/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use api::{CaptureBits, ExternalImageData, ExternalImageId, ImageDescriptor, TexelRect};
#[cfg(feature = "png")]
use api::ImageFormat;
use ron::{de, ser};
use serde::{Deserialize, Serialize};


pub struct CaptureConfig {
    pub root: PathBuf,
    pub bits: CaptureBits,
    pretty: ser::PrettyConfig,
}

impl CaptureConfig {
    pub fn new(root: PathBuf, bits: CaptureBits) -> Self {
        CaptureConfig {
            root,
            bits,
            pretty: ser::PrettyConfig::default(),
        }
    }

    pub fn serialize<T, P>(&self, data: &T, name: P)
    where
        T: Serialize,
        P: AsRef<Path>,
    {
        let ron = ser::to_string_pretty(data, self.pretty.clone())
            .unwrap();
        let path = self.root
            .join(name)
            .with_extension("ron");
        let mut file = File::create(path)
            .unwrap();
        write!(file, "{}\n", ron)
            .unwrap();
    }

    pub fn deserialize<T, P>(root: &PathBuf, name: P) -> Option<T>
    where
        T: for<'a> Deserialize<'a>,
        P: AsRef<Path>,
    {
        let mut string = String::new();
        let path = root
            .join(name)
            .with_extension("ron");
        File::open(path)
            .ok()?
            .read_to_string(&mut string)
            .unwrap();
        Some(de::from_str(&string)
            .unwrap())
    }

    #[cfg(feature = "png")]
    pub fn save_png(
        path: PathBuf, size: (u32, u32), format: ImageFormat, data: &[u8],
    ) {
        use png::{BitDepth, ColorType, Encoder, HasParameters};
        use std::io::BufWriter;

        let color_type = match format {
            ImageFormat::BGRA8 => ColorType::RGBA, //TODO?
            ImageFormat::R8 => ColorType::Grayscale,
            ImageFormat::RG8 => ColorType::GrayscaleAlpha,
            _ => {
                warn!("Unable to save PNG of {:?}", format);
                return;
            }
        };
        let w = BufWriter::new(File::create(path).unwrap());
        let mut enc = Encoder::new(w, size.0, size.1);
        enc
            .set(color_type)
            .set(BitDepth::Eight);
        enc
            .write_header()
            .unwrap()
            .write_image_data(&data)
            .unwrap();
    }
}

#[derive(Deserialize, Serialize)]
pub struct ExternalCaptureImage {
    pub short_path: String,
    pub descriptor: ImageDescriptor,
    pub external: ExternalImageData,
}

#[derive(Deserialize, Serialize)]
pub struct PlainExternalImage {
    pub data: String,
    pub id: ExternalImageId,
    pub channel_index: u8,
    pub uv: TexelRect,
}
