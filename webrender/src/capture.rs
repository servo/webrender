/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use api::{ColorF, ExternalImageData, ImageDescriptor, LayerRect};
use ron::{de, ser};
use serde::{Deserialize, Serialize};

use gpu_types::{ClipScrollNodeData};
use render_task::RenderTaskTree;
use tiling::{RenderPass};

bitflags!{
    pub struct CaptureBits: u8 {
        const SCENE = 0x1;
        const FRAME = 0x2;
    }
}

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

    pub fn deserialize<T, P>(&self, name: P) -> T
    where
        T: for<'a> Deserialize<'a>,
        P: AsRef<Path>,
    {
        let mut string = String::new();
        let path = self.root
            .join(name)
            .with_extension("ron");
        File::open(path)
            .unwrap()
            .read_to_string(&mut string)
            .unwrap();
        de::from_str(&string)
            .unwrap()
    }
}

pub struct ExternalCaptureImage {
    pub short_path: String,
    pub descriptor: ImageDescriptor,
    pub external: ExternalImageData,
}

#[derive(Serialize)]
pub struct PlainFrame {
    background_color: Option<ColorF>,
    passes: Vec<RenderPass>,
    node_data: Vec<ClipScrollNodeData>,
    clip_chain_local_clip_rects: Vec<LayerRect>,
    render_tasks: RenderTaskTree,
}
