/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImageKey(pub u32, pub u32);

impl ImageKey {
    pub fn new(key0: u32, key1: u32) -> ImageKey {
        ImageKey(key0, key1)
    }
}

/// An arbitrary identifier for an external image provided by the
/// application. It must be a unique identifier for each external
/// image.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ExternalImageId(pub u64);

#[repr(u32)]
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum ImageFormat {
    Invalid  = 0,
    A8       = 1,
    RGB8     = 2,
    RGBA8    = 3,
    RGBAF32  = 4,
    RGBA16   = 5,
}

impl ImageFormat {
    pub fn bytes_per_pixel(self) -> Option<u32> {
        match self {
            ImageFormat::A8 => Some(1),
            ImageFormat::RGB8 => Some(3),
            ImageFormat::RGBA8 => Some(4),
            ImageFormat::RGBA16 => Some(8),
            ImageFormat::RGBAF32 => Some(16),
            ImageFormat::Invalid => None,
        }
    }
}

#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ImageDescriptor {
    pub format: ImageFormat,
    pub width: u32,
    pub height: u32,
    pub stride: Option<u32>,
    pub offset: u32,
    pub is_opaque: bool,
}

impl ImageDescriptor {
    pub fn new(width: u32, height: u32, format: ImageFormat, is_opaque: bool) -> Self {
        ImageDescriptor {
            width: width,
            height: height,
            format: format,
            stride: None,
            offset: 0,
            is_opaque: is_opaque,
        }
    }

    pub fn compute_stride(&self) -> u32 {
        self.stride.unwrap_or(self.width * self.format.bytes_per_pixel().unwrap())
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub enum ImageData {
    Raw(Arc<Vec<u8>>),
    Blob(Arc<BlobImageData>),
    ExternalHandle(ExternalImageId),
    ExternalBuffer(ExternalImageId),
}

impl ImageData {
    pub fn new(bytes: Vec<u8>) -> ImageData {
        ImageData::Raw(Arc::new(bytes))
    }

    pub fn new_shared(bytes: Arc<Vec<u8>>) -> ImageData {
        ImageData::Raw(bytes)
    }

    pub fn new_blob_image(commands: Vec<u8>) -> ImageData {
        ImageData::Blob(Arc::new(commands))
    }

    pub fn new_shared_blob_image(commands: Arc<Vec<u8>>) -> ImageData {
        ImageData::Blob(commands)
    }
}

pub trait BlobImageRenderer: Send {
    fn request_blob_image(&mut self,
                            key: ImageKey,
                            data: Arc<BlobImageData>,
                            descriptor: &BlobImageDescriptor);
    fn resolve_blob_image(&mut self, key: ImageKey) -> BlobImageResult;
}

pub type BlobImageData = Vec<u8>;

pub type BlobImageResult = Result<RasterizedBlobImage, BlobImageError>;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BlobImageDescriptor {
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub scale_factor: f32,
}

pub struct RasterizedBlobImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
pub enum BlobImageError {
    Oom,
    InvalidKey,
    InvalidData,
    Other(String),
}
