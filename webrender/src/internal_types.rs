/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{DebugCommand, DeviceUintRect, DocumentId, ExternalImageData, ExternalImageId};
use api::{ImageFormat, LayoutPixel, NotificationRequest};
use device::TextureFilter;
use renderer::PipelineInfo;
use gpu_cache::GpuCacheUpdateList;
use fxhash::FxHasher;
use plane_split::BspSplitter;
use profiler::BackendProfileCounters;
use std::{usize, i32};
use std::collections::{HashMap, HashSet};
use std::f32;
use std::hash::BuildHasherDefault;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "capture")]
use capture::{CaptureConfig, ExternalCaptureImage};
#[cfg(feature = "replay")]
use capture::PlainExternalImage;
use tiling;

pub type FastHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;
pub type FastHashSet<K> = HashSet<K, BuildHasherDefault<FxHasher>>;

/// A concret plane splitter type used in WebRender.
pub type PlaneSplitter = BspSplitter<f32, LayoutPixel>;

/// An ID for a texture that is owned by the `texture_cache` module.
///
/// This can include atlases or standalone textures allocated via the texture
/// cache (e.g.  if an image is too large to be added to an atlas). The texture
/// cache manages the allocation and freeing of these IDs, and the rendering
/// thread maintains a map from cache texture ID to native texture.
///
/// We never reuse IDs, so we use a u64 here to be safe.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct CacheTextureId(pub u64);

/// Canonical type for texture layer indices.
///
/// WebRender is currently not very consistent about layer index types. Some
/// places use i32 (since that's the type used in various OpenGL APIs), some
/// places use u32 (since having it be signed is non-sensical, but the
/// underlying graphics APIs generally operate on 32-bit integers) and some
/// places use usize (since that's most natural in Rust).
///
/// Going forward, we aim to us usize throughout the codebase, since that allows
/// operations like indexing without a cast, and convert to the required type in
/// the device module when making calls into the platform layer.
pub type LayerIndex = usize;

/// Identifies a render pass target that is persisted until the end of the frame.
///
/// By default, only the targets of the immediately-preceding pass are bound as
/// inputs to the next pass. However, tasks can opt into having their target
/// preserved in a list until the end of the frame, and this type specifies the
/// index in that list.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct SavedTargetIndex(pub usize);

impl SavedTargetIndex {
    pub const PENDING: Self = SavedTargetIndex(!0);
}

/// Identifies the source of an input texture to a shader.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum TextureSource {
    /// Equivalent to `None`, allowing us to avoid using `Option`s everywhere.
    Invalid,
    /// An entry in the texture cache.
    TextureCache(CacheTextureId),
    /// An external image texture, mananged by the embedding.
    External(ExternalImageData),
    /// The alpha target of the immediately-preceding pass.
    PrevPassAlpha,
    /// The color target of the immediately-preceding pass.
    PrevPassColor,
    /// A render target from an earlier pass. Unlike the immediately-preceding
    /// passes, these are not made available automatically, but are instead
    /// opt-in by the `RenderTask` (see `mark_for_saving()`).
    RenderTaskCache(SavedTargetIndex),
}

pub const ORTHO_NEAR_PLANE: f32 = -100000.0;
pub const ORTHO_FAR_PLANE: f32 = 100000.0;

#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct RenderTargetInfo {
    pub has_depth: bool,
}

#[derive(Debug)]
pub enum TextureUpdateSource {
    External {
        id: ExternalImageId,
        channel_index: u8,
    },
    Bytes { data: Arc<Vec<u8>> },
}

#[derive(Debug)]
pub enum TextureUpdateOp {
    Create {
        width: u32,
        height: u32,
        format: ImageFormat,
        filter: TextureFilter,
        render_target: Option<RenderTargetInfo>,
        layer_count: i32,
    },
    Update {
        rect: DeviceUintRect,
        stride: Option<u32>,
        offset: u32,
        layer_index: i32,
        source: TextureUpdateSource,
    },
    Free,
}

#[derive(Debug)]
pub struct TextureUpdate {
    pub id: CacheTextureId,
    pub op: TextureUpdateOp,
}

#[derive(Default)]
pub struct TextureUpdateList {
    pub updates: Vec<TextureUpdate>,
}

impl TextureUpdateList {
    pub fn new() -> Self {
        TextureUpdateList {
            updates: Vec::new(),
        }
    }

    #[inline]
    pub fn push(&mut self, update: TextureUpdate) {
        self.updates.push(update);
    }
}

/// Wraps a tiling::Frame, but conceptually could hold more information
pub struct RenderedDocument {
    pub frame: tiling::Frame,
    pub is_new_scene: bool,
}

pub enum DebugOutput {
    FetchDocuments(String),
    FetchClipScrollTree(String),
    #[cfg(feature = "capture")]
    SaveCapture(CaptureConfig, Vec<ExternalCaptureImage>),
    #[cfg(feature = "replay")]
    LoadCapture(PathBuf, Vec<PlainExternalImage>),
}

#[allow(dead_code)]
pub enum ResultMsg {
    DebugCommand(DebugCommand),
    DebugOutput(DebugOutput),
    RefreshShader(PathBuf),
    UpdateGpuCache(GpuCacheUpdateList),
    UpdateResources {
        updates: TextureUpdateList,
        memory_pressure: bool,
    },
    PublishPipelineInfo(PipelineInfo),
    PublishDocument(
        DocumentId,
        RenderedDocument,
        TextureUpdateList,
        BackendProfileCounters,
    ),
    AppendNotificationRequests(Vec<NotificationRequest>),
}

#[derive(Clone, Debug)]
pub struct ResourceCacheError {
    description: String,
}

impl ResourceCacheError {
    pub fn new(description: String) -> ResourceCacheError {
        ResourceCacheError {
            description,
        }
    }
}
