/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#![deny(missing_docs)]

extern crate serde_bytes;

use font::{FontInstanceKey, FontInstanceData, FontKey, FontTemplate};
use std::sync::Arc;
use {DevicePoint, DeviceUintPoint, DeviceUintRect, DeviceUintSize};
use {IdNamespace, TileOffset, TileSize};
use euclid::size2;

/// An opaque identifier describing an image registered with WebRender.
/// This is used as a handle to reference images, and is used as the
/// hash map key for the actual image storage in the `ResourceCache`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ImageKey(pub IdNamespace, pub u32);

impl ImageKey {
    /// Placeholder Image key, used to represent None.
    pub const DUMMY: Self = ImageKey(IdNamespace(0), 0);

    /// Mints a new ImageKey. The given ID must be unique.
    pub fn new(namespace: IdNamespace, key: u32) -> Self {
        ImageKey(namespace, key)
    }
}

/// An arbitrary identifier for an external image provided by the
/// application. It must be a unique identifier for each external
/// image.
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ExternalImageId(pub u64);

/// Specifies the type of texture target in driver terms.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum TextureTarget {
    /// Standard texture. This maps to GL_TEXTURE_2D in OpenGL.
    Default = 0,
    /// Array texture. This maps to GL_TEXTURE_2D_ARRAY in OpenGL. See
    /// https://www.khronos.org/opengl/wiki/Array_Texture for background
    /// on Array textures.
    Array = 1,
    /// Rectange texture. This maps to GL_TEXTURE_RECTANGLE in OpenGL. This
    /// is similar to as standard texture, with a few subtle differences
    /// (no mipmaps, non-power-of-two dimensions, different coordinate space)
    /// that make it useful for representing the kinds of textures we use
    /// in WebRender. See https://www.khronos.org/opengl/wiki/Rectangle_Texture
    /// for background on Rectangle textures.
    Rect = 2,
    /// External texture. This maps to GL_TEXTURE_EXTERNAL_OES in OpenGL, which
    /// is an extension. This is used for image formats that OpenGL doesn't
    /// understand, particularly YUV. See
    /// https://www.khronos.org/registry/OpenGL/extensions/OES/OES_EGL_image_external.txt
    External = 3,
}

/// Storage format identifier for externally-managed images.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum ExternalImageType {
    /// The image is texture-backed.
    TextureHandle(TextureTarget),
    /// The image is heap-allocated by the embedding.
    Buffer,
}

/// Descriptor for external image resources. See `ImageData`.
#[repr(C)]
#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ExternalImageData {
    /// The identifier of this external image, provided by the embedding.
    pub id: ExternalImageId,
    /// For multi-plane images (i.e. YUV), indicates the plane of the
    /// original image that this struct represents. 0 for single-plane images.
    pub channel_index: u8,
    /// Storage format identifier.
    pub image_type: ExternalImageType,
}

/// Specifies the format of a series of pixels, in driver terms.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub enum ImageFormat {
    /// One-channel, byte storage. The "red" doesn't map to the color
    /// red per se, and is just the way that OpenGL has historically referred
    /// to single-channel buffers.
    R8 = 1,
    /// Four channels, byte storage.
    BGRA8 = 3,
    /// Four channels, float storage.
    RGBAF32 = 4,
    /// Two-channels, byte storage. Similar to `R8`, this just means
    /// "two channels" rather than "red and green".
    RG8 = 5,
    /// Four channels, signed integer storage.
    RGBAI32 = 6,
}

impl ImageFormat {
    /// Returns the number of bytes per pixel for the given format.
    pub fn bytes_per_pixel(self) -> u32 {
        match self {
            ImageFormat::R8 => 1,
            ImageFormat::BGRA8 => 4,
            ImageFormat::RGBAF32 => 16,
            ImageFormat::RG8 => 2,
            ImageFormat::RGBAI32 => 16,
        }
    }
}

/// Metadata (but not storage) describing an image In WebRender.
#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ImageDescriptor {
    /// Format of the image data.
    pub format: ImageFormat,
    /// Width and length of the image data, in pixels.
    pub size: DeviceUintSize,
    /// The number of bytes from the start of one row to the next. Lazily
    /// computed via compute_stride() and cached here when the image is
    /// tile.
    ///
    /// FIXME(bholley): The computation of this looks pretty cheap. Do we have
    /// evidence that this is hot enough that it needs caching?
    pub stride: Option<u32>,
    /// Byte offset used for tiling. FIXME: Better documentation.
    pub offset: u32,
    /// Whether this image is opaque, or has an alpha channel. Avoiding blending
    /// for opaque surfaces is an important optimization.
    pub is_opaque: bool,
    /// Whether to allow the driver to automatically generate mipmaps. If images
    /// are already downscaled appropriately, mipmap generation can be wasted
    /// work, and cause performance problems on some cards/drivers.
    ///
    /// See https://github.com/servo/webrender/pull/2555/
    pub allow_mipmaps: bool,
}

impl ImageDescriptor {
    /// Mints a new ImageDescriptor.
    pub fn new(
        width: u32,
        height: u32,
        format: ImageFormat,
        is_opaque: bool,
        allow_mipmaps: bool,
    ) -> Self {
        ImageDescriptor {
            size: size2(width, height),
            format,
            stride: None,
            offset: 0,
            is_opaque,
            allow_mipmaps,
        }
    }

    /// Returns the cached stride or computes it.
    pub fn compute_stride(&self) -> u32 {
        self.stride.unwrap_or(self.size.width * self.format.bytes_per_pixel())
    }

    /// Computes the total size of the image, in bytes.
    pub fn compute_total_size(&self) -> u32 {
        self.compute_stride() * self.size.height
    }

    /// Computes the bounding rectangle for the image, rooted at (0, 0).
    pub fn full_rect(&self) -> DeviceUintRect {
        DeviceUintRect::new(
            DeviceUintPoint::zero(),
            self.size,
        )
    }
}

/// Represents the backing store of an arbitrary series of pixels for display by
/// WebRender. This storage can take several forms.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ImageData {
    /// A simple series of bytes, provided by the embedding and owned by WebRender.
    /// The format is stored out-of-band, currently in ImageDescriptor.
    Raw(#[serde(with = "serde_image_data_raw")] Arc<Vec<u8>>),
    /// An series of commands that can be rasterized into an image via an
    /// embedding-provided callback.
    Blob(#[serde(with = "serde_image_data_raw")] Arc<BlobImageData>),
    /// An image owned by the embedding, and referenced by WebRender. This may
    /// take the form of a texture or a heap-allocated buffer.
    External(ExternalImageData),
}

mod serde_image_data_raw {
    extern crate serde_bytes;

    use std::sync::Arc;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &Arc<Vec<u8>>, serializer: S) -> Result<S::Ok, S::Error> {
        serde_bytes::serialize(bytes.as_slice(), serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Arc<Vec<u8>>, D::Error> {
        serde_bytes::deserialize(deserializer).map(Arc::new)
    }
}

impl ImageData {
    /// Mints a new raw ImageData, taking ownership of the bytes.
    pub fn new(bytes: Vec<u8>) -> Self {
        ImageData::Raw(Arc::new(bytes))
    }

    /// Mints a new raw ImageData from Arc-ed bytes.
    pub fn new_shared(bytes: Arc<Vec<u8>>) -> Self {
        ImageData::Raw(bytes)
    }

    /// Mints a new Blob ImageData.
    pub fn new_blob_image(commands: BlobImageData) -> Self {
        ImageData::Blob(Arc::new(commands))
    }

    /// Returns true if this ImageData represents a blob.
    #[inline]
    pub fn is_blob(&self) -> bool {
        match *self {
            ImageData::Blob(_) => true,
            _ => false,
        }
    }

    /// Returns true if this variant of ImageData should go through the texture
    /// cache.
    #[inline]
    pub fn uses_texture_cache(&self) -> bool {
        match *self {
            ImageData::External(ref ext_data) => match ext_data.image_type {
                ExternalImageType::TextureHandle(_) => false,
                ExternalImageType::Buffer => true,
            },
            ImageData::Blob(_) => true,
            ImageData::Raw(_) => true,
        }
    }
}

/// The resources exposed by the resource cache available for use by the blob rasterizer.
#[allow(missing_docs)]
pub trait BlobImageResources {
    fn get_font_data(&self, key: FontKey) -> &FontTemplate;
    fn get_font_instance_data(&self, key: FontInstanceKey) -> Option<FontInstanceData>;
    fn get_image(&self, key: ImageKey) -> Option<(&ImageData, &ImageDescriptor)>;
}

/// A handler on the render backend that can create rasterizer objects which will
/// be sent to the scene builder thread to execute the rasterization.
///
/// The handler is responsible for collecting resources, managing/updating blob commands
/// and creating the rasterizer objects, but isn't expected to do any rasterization itself.
pub trait BlobImageHandler: Send {
    /// Creates a snapshot of the current state of blob images in the handler.
    fn create_blob_rasterizer(&mut self) -> Box<AsyncBlobImageRasterizer>;

    /// A hook to let the blob image handler update any state related to resources that
    /// are not bundled in the blob recording itself.
    fn prepare_resources(
        &mut self,
        services: &BlobImageResources,
        requests: &[BlobImageParams],
    );

    /// Register a blob image.
    fn add(&mut self, key: ImageKey, data: Arc<BlobImageData>, tiling: Option<TileSize>);

    /// Update an already registered blob image.
    fn update(&mut self, key: ImageKey, data: Arc<BlobImageData>, dirty_rect: Option<DeviceUintRect>);

    /// Delete an already registered blob image.
    fn delete(&mut self, key: ImageKey);

    /// A hook to let the handler clean up any state related to a font which the resource
    /// cache is about to delete.
    fn delete_font(&mut self, key: FontKey);

    /// A hook to let the handler clean up any state related to a font instance which the
    /// resource cache is about to delete.
    fn delete_font_instance(&mut self, key: FontInstanceKey);

    /// A hook to let the handler clean up any state related a given namespace before the
    /// resource cache deletes them.
    fn clear_namespace(&mut self, namespace: IdNamespace);
}

/// A group of rasterization requests to execute synchronously on the scene builder thread.
pub trait AsyncBlobImageRasterizer : Send {
    /// Rasterize the requests.
    fn rasterize(&mut self, requests: &[BlobImageParams]) -> Vec<(BlobImageRequest, BlobImageResult)>;
}


/// FIXME: document
#[derive(Copy, Clone, Debug)]
pub struct BlobImageParams {
    /// FIXME: document
    pub request: BlobImageRequest,
    /// FIXME: document
    pub descriptor: BlobImageDescriptor,
    /// FIXME: document
    pub dirty_rect: Option<DeviceUintRect>,
}

/// Backing store for blob image command streams.
pub type BlobImageData = Vec<u8>;

/// Result type for blob raserization.
pub type BlobImageResult = Result<RasterizedBlobImage, BlobImageError>;

/// Metadata (but not storage) for a blob image.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BlobImageDescriptor {
    /// Bounding rectangle of the image, in device pixels.
    pub size: DeviceUintSize,
    /// FIXME - Document me.
    pub offset: DevicePoint,
    /// Format for the data in the backing store.
    pub format: ImageFormat,
}

/// Representation of a rasterized blob image. This is obtained by passing
/// `BlobImageData` to the embedding via the rasterization callback.
pub struct RasterizedBlobImage {
    /// The bounding rectangle for this bob image.
    pub rasterized_rect: DeviceUintRect,
    /// Backing store. The format is stored out of band in `BlobImageDescriptor`.
    pub data: Arc<Vec<u8>>,
}

/// Error code for when blob rasterization failed.
#[derive(Clone, Debug)]
pub enum BlobImageError {
    /// Out of memory.
    Oom,
    /// Other failure, embedding-specified.
    Other(String),
}

/// FIXME: document
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlobImageRequest {
    /// Unique handle to the image.
    pub key: ImageKey,
    /// Tiling offset.
    /// FIXME - document tiling better.
    pub tile: Option<TileOffset>,
}
