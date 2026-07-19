/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::ColorF;
use api::{ImageRendering, LineOrientation, PrimitiveFlags};
use api::units::*;
use crate::clip::ClipLeafId;
use crate::render_backend::DataStores;
use crate::space::SnapRounding;
use crate::quad::QuadTileClassifier;
use crate::renderer::{GpuBufferAddress, GpuBufferHandle, GpuBufferWriterF};
use crate::segment::EdgeMask;
use crate::debug_item::{DebugItem, DebugMessage};
use crate::debug_colors;
use glyph_rasterizer::{GlyphKey, SubpixelDirection};
use crate::gpu_types::{BrushFlags, BrushSegmentGpuData, QuadSegment};
use crate::intern;
use crate::picture::{PictureInstance, PictureScratch};
use crate::render_task_graph::RenderTaskId;
use crate::resource_cache::ImageProperties;
use std::{hash, u32, usize};
use crate::util::Recycler;
use crate::internal_types::{FastHashSet, LayoutPrimitiveInfo};
use crate::visibility::PrimitiveDrawHeader;

pub mod backdrop;
pub mod borders;
pub mod gradient;
pub mod image;
pub mod line_dec;
pub mod picture;
pub mod rectangle;
pub mod text_run;
pub mod interned;

pub mod storage;

use backdrop::{BackdropCaptureDataHandle, BackdropRenderDataHandle};
use borders::{ImageBorderDataHandle, ImageBorderScratch, NormalBorderDataHandle};
use gradient::{LinearGradientDataHandle, RadialGradientDataHandle, ConicGradientDataHandle};
use image::{ImageDataHandle, ImageScratch, VisibleImageTile, YuvImageDataHandle};
use line_dec::LineDecorationDataHandle;
use picture::PictureDataHandle;
use rectangle::RectangleDataHandle;
use text_run::{TextRunDataHandle, TextRunScratch};
use crate::box_shadow::BoxShadowDataHandle;

pub const VECS_PER_SEGMENT: usize = 2;

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Copy, Clone, MallocSizeOf)]
pub struct PrimitiveOpacity {
    pub is_opaque: bool,
}

impl PrimitiveOpacity {
    pub fn opaque() -> PrimitiveOpacity {
        PrimitiveOpacity { is_opaque: true }
    }

    pub fn translucent() -> PrimitiveOpacity {
        PrimitiveOpacity { is_opaque: false }
    }

    pub fn from_alpha(alpha: f32) -> PrimitiveOpacity {
        PrimitiveOpacity {
            is_opaque: alpha >= 1.0,
        }
    }
}

/// For external images, it's not possible to know the
/// UV coords of the image (or the image data itself)
/// until the render thread receives the frame and issues
/// callbacks to the client application. For external
/// images that are visible, a DeferredResolve is created
/// that is stored in the frame. This allows the render
/// thread to iterate this list and update any changed
/// texture data and update the UV rect. Any filtering
/// is handled externally for NativeTexture external
/// images.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct DeferredResolve {
    pub handle: GpuBufferHandle,
    pub image_properties: ImageProperties,
    pub rendering: ImageRendering,
    pub is_composited: bool,
}

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
pub struct ClipTaskIndex(pub u32);

impl ClipTaskIndex {
    pub const INVALID: ClipTaskIndex = ClipTaskIndex(0);
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, MallocSizeOf, Ord, PartialOrd)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct PictureIndex(pub usize);

impl PictureIndex {
    pub const INVALID: PictureIndex = PictureIndex(!0);
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Copy, Debug, Clone, MallocSizeOf, PartialEq)]
pub struct RectKey {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl RectKey {
    pub fn intersects(&self, other: &Self) -> bool {
        self.x0 < other.x1
            && other.x0 < self.x1
            && self.y0 < other.y1
            && other.y0 < self.y1
    }
}

impl Eq for RectKey {}

impl hash::Hash for RectKey {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.x0.to_bits().hash(state);
        self.y0.to_bits().hash(state);
        self.x1.to_bits().hash(state);
        self.y1.to_bits().hash(state);
    }
}

impl From<RectKey> for LayoutRect {
    fn from(key: RectKey) -> LayoutRect {
        LayoutRect {
            min: LayoutPoint::new(key.x0, key.y0),
            max: LayoutPoint::new(key.x1, key.y1),
        }
    }
}

impl From<RectKey> for WorldRect {
    fn from(key: RectKey) -> WorldRect {
        WorldRect {
            min: WorldPoint::new(key.x0, key.y0),
            max: WorldPoint::new(key.x1, key.y1),
        }
    }
}

impl From<LayoutRect> for RectKey {
    fn from(rect: LayoutRect) -> RectKey {
        RectKey {
            x0: rect.min.x,
            y0: rect.min.y,
            x1: rect.max.x,
            y1: rect.max.y,
        }
    }
}

impl From<PictureRect> for RectKey {
    fn from(rect: PictureRect) -> RectKey {
        RectKey {
            x0: rect.min.x,
            y0: rect.min.y,
            x1: rect.max.x,
            y1: rect.max.y,
        }
    }
}

impl From<WorldRect> for RectKey {
    fn from(rect: WorldRect) -> RectKey {
        RectKey {
            x0: rect.min.x,
            y0: rect.min.y,
            x1: rect.max.x,
            y1: rect.max.y,
        }
    }
}

// `PolygonKey` now lives in `webrender_api` so builder-side interning keys can
// reference it. Re-exported here to keep existing references working.
pub use api::key_types::PolygonKey;

// `SideOffsetsKey`, `SizeKey`, `PointKey` and `VectorKey` now live in
// `webrender_api` so builder-side interning keys can reference them. Re-exported
// here to keep existing references working.
pub use api::key_types::VectorKey;

// `PrimKeyCommonData` now lives in `webrender_api` so interned keys reference
// only api-resident types. Re-exported here to keep existing references working.
pub use api::key_types::PrimKeyCommonData;

impl From<&LayoutPrimitiveInfo> for PrimKeyCommonData {
    fn from(info: &LayoutPrimitiveInfo) -> Self {
        PrimKeyCommonData {
            flags: info.flags,
            aligned_aa_edges: info.aligned_aa_edges,
            transformed_aa_edges: info.transformed_aa_edges,
        }
    }
}

// `PrimKey<T>` now lives in `webrender_api::interned_prims` so builder-side
// interning can construct the alias-based keys. Re-exported here to keep
// existing references working.
pub use api::interned_prims::PrimKey;

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(MallocSizeOf)]
#[derive(Debug)]
pub struct PrimTemplateCommonData {
    pub flags: PrimitiveFlags,
    pub aligned_aa_edges: EdgeMask,
    pub transformed_aa_edges: EdgeMask,
}

impl PrimTemplateCommonData {
    pub fn with_key_common(common: PrimKeyCommonData) -> Self {
        PrimTemplateCommonData {
            flags: common.flags,
            aligned_aa_edges: common.aligned_aa_edges,
            transformed_aa_edges: common.transformed_aa_edges,
        }
    }
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(MallocSizeOf)]
pub struct PrimTemplate<T> {
    pub common: PrimTemplateCommonData,
    pub kind: T,
}

#[derive(Debug, MallocSizeOf)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct VisibleMaskImageTile {
    pub tile_offset: TileOffset,
    pub tile_rect: LayoutRect,
    pub task_id: RenderTaskId,
}

/// Represents the visibility state of a segment (wrt clip masks).
#[cfg_attr(feature = "capture", derive(Serialize))]
#[derive(Debug, Clone)]
pub enum ClipMaskKind {
    /// The segment has a clip mask, specified by the render task.
    Mask(RenderTaskId),
    /// The segment has no clip mask.
    None,
    /// The segment is made invisible / clipped completely.
    Clipped,
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Clone, MallocSizeOf)]
pub struct BrushSegment {
    pub local_rect: LayoutRect,
    pub may_need_clip_mask: bool,
    pub edge_flags: EdgeMask,
    pub extra_data: [f32; 4],
    pub brush_flags: BrushFlags,
}

impl BrushSegment {
    pub fn new(
        local_rect: LayoutRect,
        may_need_clip_mask: bool,
        edge_flags: EdgeMask,
        extra_data: [f32; 4],
        brush_flags: BrushFlags,
    ) -> Self {
        Self {
            local_rect,
            may_need_clip_mask,
            edge_flags,
            extra_data,
            brush_flags,
        }
    }

    pub fn gpu_data(&self) -> BrushSegmentGpuData {
        BrushSegmentGpuData {
            local_rect: self.local_rect,
            extra_data: self.extra_data,
        }
    }

    pub fn write_gpu_blocks(&self, writer: &mut GpuBufferWriterF) {
        writer.push(&self.gpu_data());
    }
}

// `NinePatchDescriptor` now lives in `webrender_api` so builder-side interning
// keys can reference it. Re-exported here to keep existing references working.
pub use api::key_types::NinePatchDescriptor;

#[derive(Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
pub enum PrimitiveKind {
    /// Direct reference to a Picture
    Picture {
        /// Handle to the common interned data for this primitive.
        data_handle: PictureDataHandle,
        pic_index: PictureIndex,
    },
    /// A run of glyphs, with associated font parameters.
    TextRun {
        /// Handle to the common interned data for this primitive.
        data_handle: TextRunDataHandle,
    },
    /// A line decoration. cache_handle refers to a cached render
    /// task handle, if this line decoration is not a simple solid.
    LineDecoration {
        /// Handle to the common interned data for this primitive.
        data_handle: LineDecorationDataHandle,
    },
    NormalBorder {
        /// Handle to the common interned data for this primitive.
        data_handle: NormalBorderDataHandle,
    },
    ImageBorder {
        /// Handle to the common interned data for this primitive.
        data_handle: ImageBorderDataHandle,
    },
    Rectangle {
        /// Handle to the common interned data for this primitive.
        data_handle: RectangleDataHandle,
    },
    YuvImage {
        /// Handle to the common interned data for this primitive.
        data_handle: YuvImageDataHandle,
    },
    Image {
        /// Handle to the common interned data for this primitive.
        data_handle: ImageDataHandle,
    },
    LinearGradient {
        /// Handle to the common interned data for this primitive.
        data_handle: LinearGradientDataHandle,
    },
    RadialGradient {
        /// Handle to the common interned data for this primitive.
        data_handle: RadialGradientDataHandle,
    },
    ConicGradient {
        /// Handle to the common interned data for this primitive.
        data_handle: ConicGradientDataHandle,
    },
    /// Render a portion of a specified backdrop.
    BackdropCapture {
        data_handle: BackdropCaptureDataHandle,
    },
    BackdropRender {
        data_handle: BackdropRenderDataHandle,
        pic_index: PictureIndex,
    },
    BoxShadow {
        data_handle: BoxShadowDataHandle,
    },
}

impl PrimitiveKind {
    pub fn as_pic(&self) -> PictureIndex {
        match self {
            PrimitiveKind::Picture { pic_index, .. } => *pic_index,
            _ => panic!("bug: as_pic called on a prim that is not a picture"),
        }
    }
}

#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct PrimitiveInstanceIndex(pub u32);

impl PrimitiveInstanceIndex {
    pub const INVALID: PrimitiveInstanceIndex = PrimitiveInstanceIndex(!0);
}

#[derive(Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
pub struct PrimitiveInstance {
    /// Identifies the kind of primitive this
    /// instance is, and references to where
    /// the relevant information for the primitive
    /// can be found.
    pub kind: PrimitiveKind,

    /// All information and state related to clip(s) for this primitive
    pub clip_leaf_id: ClipLeafId,

    /// Local-space rect of the primitive (origin + size), as authored by the
    /// display list (not snapped to the device pixel grid). Carries both the
    /// position and the per-instance size; the latter used to live on
    /// `PrimTemplateCommonData.prim_size` but is per-instance now so that the
    /// intern key can deduplicate across differently-sized instances of the
    /// same prim shape.
    pub unsnapped_prim_rect: LayoutRect,
}

/// How a primitive's clips round to the device pixel grid. Distinct from how
/// the prim's own rect rounds (see `SnapPolicy::rect`): a text run rounds its
/// rect out on both axes but its clips out only on the non-sub-pixel axis, and
/// a surface rounds its rect out but leaves its clips exact.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ClipSnap {
    /// Snap every clip edge to the nearest device pixel. Used by prims that
    /// snap their whole geometry to the grid (`snaps`).
    Nearest,
    /// Leave clip edges exact. Used by device-space surfaces, whose clips must
    /// stay at the sub-pixel position matching their contents (bug 2050692).
    Exact,
    /// Device-space text run: round out on the non-sub-pixel axis, keep the
    /// sub-pixel axis exact (bug 2055145 / bug 2050692). `RoundOut` when the run
    /// has no sub-pixel positioning.
    Text(SnapRounding),
}

/// The device-grid snapping policy for one primitive: how its own bounding rect
/// rounds, and how its clips round. These are separate axes - e.g. a line
/// decoration is `{ rect: Line, clip: Nearest }`, a text run is
/// `{ rect: RoundOut, clip: Text(..) }`, a surface is `{ rect: RoundOut, clip:
/// Exact }`.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct SnapPolicy {
    pub rect: SnapRounding,
    pub clip: ClipSnap,
}

impl PrimitiveInstance {
    pub fn new(
        kind: PrimitiveKind,
        clip_leaf_id: ClipLeafId,
        unsnapped_prim_rect: LayoutRect,
    ) -> Self {
        PrimitiveInstance {
            kind,
            clip_leaf_id,
            unsnapped_prim_rect,
        }
    }

    /// How this prim rounds to the device pixel grid: its own rect and its
    /// clips (see `SnapPolicy`).
    ///
    /// `snaps` is the prim's snap policy, taken from its clip leaf: `false` for
    /// a device-space prim (text run or surface) that stays at exact sub-pixel
    /// positions and only needs a conservative, grid-aligned footprint. A
    /// decoration line snaps its thickness specially so it can't vanish or
    /// double with scale (bug 1783779); everything else snaps to the nearest
    /// pixel.
    ///
    /// The two rounding axes differ for device-space prims: a text run rounds
    /// its clip *out* on the non-sub-pixel axis so a grid-snapped glyph row is
    /// never shaved by a fractional clip edge (bug 2055145), while keeping the
    /// sub-pixel axis exact so the clip keeps matching the glyph's exact
    /// sub-pixel position (bug 2050692); a surface leaves its clips exact. Both
    /// keep a `RoundOut` bounding rect. The sub-pixel axis comes from the run's
    /// own font, so clip code stays agnostic to `subpx_dir`.
    pub fn snap_policy(&self, snaps: bool, data_stores: &DataStores) -> SnapPolicy {
        if !snaps {
            let clip = if let PrimitiveKind::TextRun { data_handle, .. } = self.kind {
                ClipSnap::Text(match data_stores.text_run[data_handle].font.get_subpx_dir() {
                    SubpixelDirection::Horizontal =>
                        SnapRounding::RoundOutNonSubpx { subpx_horizontal: true },
                    SubpixelDirection::Vertical =>
                        SnapRounding::RoundOutNonSubpx { subpx_horizontal: false },
                    SubpixelDirection::None => SnapRounding::RoundOut,
                })
            } else {
                ClipSnap::Exact
            };
            return SnapPolicy { rect: SnapRounding::RoundOut, clip };
        }
        let rect = match self.kind {
            PrimitiveKind::LineDecoration { data_handle, .. } => SnapRounding::Line {
                horizontal: data_stores.line_decoration[data_handle].kind.orientation
                    == LineOrientation::Horizontal,
            },
            _ => SnapRounding::Nearest,
        };
        SnapPolicy { rect, clip: ClipSnap::Nearest }
    }

    pub fn uid(&self) -> intern::ItemUid {
        match &self.kind {
            PrimitiveKind::Rectangle { data_handle, .. } => {
                data_handle.uid()
            }
            PrimitiveKind::Image { data_handle, .. } => {
                data_handle.uid()
            }
            PrimitiveKind::ImageBorder { data_handle, .. } => {
                data_handle.uid()
            }
            PrimitiveKind::LineDecoration { data_handle, .. } => {
                data_handle.uid()
            }
            PrimitiveKind::LinearGradient { data_handle, .. } => {
                data_handle.uid()
            }
            PrimitiveKind::NormalBorder { data_handle, .. } => {
                data_handle.uid()
            }
            PrimitiveKind::Picture { data_handle, .. } => {
                data_handle.uid()
            }
            PrimitiveKind::RadialGradient { data_handle, .. } => {
                data_handle.uid()
            }
            PrimitiveKind::ConicGradient { data_handle, .. } => {
                data_handle.uid()
            }
            PrimitiveKind::TextRun { data_handle, .. } => {
                data_handle.uid()
            }
            PrimitiveKind::YuvImage { data_handle, .. } => {
                data_handle.uid()
            }
            PrimitiveKind::BackdropCapture { data_handle, .. } => {
                data_handle.uid()
            }
            PrimitiveKind::BackdropRender { data_handle, .. } => {
                data_handle.uid()
            }
            PrimitiveKind::BoxShadow { data_handle, .. } => {
                data_handle.uid()
            }

        }
    }
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[derive(Debug)]
pub struct BrushSegmentation {
    pub gpu_data: GpuBufferAddress,
    pub segments_range: SegmentsRange,
}

pub type GlyphKeyStorage = storage::Storage<GlyphKey>;
pub type SegmentStorage = storage::Storage<BrushSegment>;
pub type SegmentsRange = storage::Range<BrushSegment>;
pub type SegmentInstanceStorage = storage::Storage<BrushSegmentation>;
pub type SegmentInstanceIndex = storage::Index<BrushSegmentation>;
/// Per-frame scratch storage. All fields are cleared every frame in
/// `begin_frame`. Anything written here lives only for the current frame.
#[cfg_attr(feature = "capture", derive(Serialize))]
pub struct PrimitiveFrameScratch {
    /// Per-frame draw headers, one entry per `PrimitiveInstance`.
    /// Resized to `prim_instances.len()` at frame start and identity-
    /// indexed by `PrimitiveInstanceIndex.0` (a follow-up will switch
    /// this to push-per-draw with `Index<PrimitiveDrawHeader>`). Holds
    /// visibility state, clip chain and clip-task index for each
    /// visible primitive.
    pub draws: Vec<PrimitiveDrawHeader>,

    /// Per-frame scratch for Picture primitives. Holds the picture's
    /// primary/secondary render task ids and any per-composite-mode
    /// extra GPU buffer addresses. Indexed by `scratch_handle` on
    /// `PrimitiveKind::Picture`.
    pub pictures: storage::Storage<PictureScratch>,

    /// Per-frame scratch for Image primitives. Holds the source render
    /// task (or a Range of per-tile tasks for tiled images), normalized-
    /// uvs flag, and image adjustment.
    pub images: storage::Storage<ImageScratch>,

    /// Per-tile entries for tiled Image primitives. Each `ImageScratch`
    /// holds a `Range` into this storage.
    pub visible_image_tiles: storage::Storage<VisibleImageTile>,

    /// Per-frame scratch for TextRun primitives. Holds the per-frame
    /// font snapshot, glyph-key range, snapping offset, and raster
    /// scale for each visible text run.
    pub text_runs: storage::Storage<TextRunScratch>,

    /// Per-frame storage for glyph keys allocated by visible text
    /// runs. Each `TextRunScratch` holds a `Range` into this storage.
    /// Used to be on `PrimitiveSceneCache` (memoized across frames);
    /// graduated to per-frame here so the scene buffer cannot grow
    /// unbounded between scene rebuilds.
    pub glyph_keys: GlyphKeyStorage,

    /// A list of brush segments built each frame for the segmented
    /// brush primitives (Rectangle, YuvImage, non-tiled Image). The
    /// segment builder runs every frame for every visible segmented
    /// prim.
    pub segments: SegmentStorage,

    /// A list of per-prim brush segmentation records (segments range
    /// + GPU buffer address). Each PrimitiveDrawHeader.segment_instance_index
    /// holds an index into this storage, or UNUSED for non-segmented
    /// prims.
    pub segment_instances: SegmentInstanceStorage,

    /// Per-frame scratch for ImageBorder primitives. Holds the range
    /// into `segments` for the nine-patch brush segments built each
    /// frame against the prim's size.
    pub image_border: storage::Storage<ImageBorderScratch>,

    /// Contains a list of clip mask instance parameters
    /// per segment generated.
    pub clip_mask_instances: Vec<ClipMaskKind>,

    /// List of debug display items for rendering. Cleared in `begin_frame`
    /// and refilled in `end_frame` (where retained `messages` are flushed
    /// into it for on-screen display).
    pub debug_items: Vec<DebugItem>,

    /// Set of sub-graphs that are required, determined during visibility pass
    pub required_sub_graphs: FastHashSet<PictureIndex>,

    /// Temporary buffers for building segments in to during prepare pass
    pub quad_direct_segments: Vec<QuadSegment>,
    pub quad_indirect_segments: Vec<QuadSegment>,
}

impl Default for PrimitiveFrameScratch {
    fn default() -> Self {
        PrimitiveFrameScratch {
            draws: Vec::new(),
            pictures: storage::Storage::new(0),
            images: storage::Storage::new(0),
            visible_image_tiles: storage::Storage::new(0),
            text_runs: storage::Storage::new(0),
            glyph_keys: GlyphKeyStorage::new(0),
            segments: SegmentStorage::new(0),
            segment_instances: SegmentInstanceStorage::new(0),
            image_border: storage::Storage::new(0),
            clip_mask_instances: Vec::new(),
            debug_items: Vec::new(),
            required_sub_graphs: FastHashSet::default(),
            quad_direct_segments: Vec::new(),
            quad_indirect_segments: Vec::new(),
        }
    }
}

impl PrimitiveFrameScratch {
    pub fn recycle(&mut self, recycler: &mut Recycler) {
        recycler.recycle_vec(&mut self.draws);
        self.pictures.recycle(recycler);
        self.images.recycle(recycler);
        self.visible_image_tiles.recycle(recycler);
        self.text_runs.recycle(recycler);
        self.glyph_keys.recycle(recycler);
        self.segments.recycle(recycler);
        self.segment_instances.recycle(recycler);
        self.image_border.recycle(recycler);
        recycler.recycle_vec(&mut self.clip_mask_instances);
        recycler.recycle_vec(&mut self.debug_items);
        recycler.recycle_vec(&mut self.quad_direct_segments);
        recycler.recycle_vec(&mut self.quad_indirect_segments);
    }

    pub fn begin_frame(&mut self) {
        self.pictures.clear();
        self.images.clear();
        self.visible_image_tiles.clear();
        self.text_runs.clear();
        self.glyph_keys.clear();
        self.segments.clear();
        self.segment_instances.clear();
        self.image_border.clear();

        // Clear the clip mask tasks for the beginning of the frame. Append
        // a single kind representing no clip mask, at the ClipTaskIndex::INVALID
        // location.
        self.clip_mask_instances.clear();
        self.clip_mask_instances.push(ClipMaskKind::None);
        self.quad_direct_segments.clear();
        self.quad_indirect_segments.clear();

        self.required_sub_graphs.clear();

        self.debug_items.clear();
    }
}

/// Per-scene cache. Now empty — the originally memoized fields have
/// migrated to per-frame storage. Kept as a placeholder for any future
/// scene-stable state and so the lifetime invariant on
/// PrimitiveScratchBuffer (frame / scene / retained) remains visible
/// at the type level; a follow-up may drop it entirely.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[derive(Default)]
pub struct PrimitiveSceneCache {}

impl PrimitiveSceneCache {
    pub fn recycle(&mut self, _recycler: &mut Recycler) {}
}

/// State that lives strictly longer than a single frame *and* is not tied
/// to scene lifetime. These fields manage their own trim/eviction policy
/// rather than being cleared by `begin_frame` or `recycle`.
#[cfg_attr(feature = "capture", derive(Serialize))]
pub struct PrimitiveRetained {
    /// Debug log of recent messages. Trimmed by time/count in
    /// `PrimitiveScratchBuffer::end_frame` and flushed into
    /// `PrimitiveFrameScratch::debug_items` for display.
    messages: Vec<DebugMessage>,

    /// A retained classifier for checking which segments of a tiled
    /// primitive need a mask / are clipped / can be rendered directly.
    pub quad_tile_classifier: QuadTileClassifier,
}

impl Default for PrimitiveRetained {
    fn default() -> Self {
        PrimitiveRetained {
            messages: Vec::new(),
            quad_tile_classifier: QuadTileClassifier::new(),
        }
    }
}

/// Contains various vecs of data that is used only during frame building,
/// where we want to recycle the memory each new display list, to avoid
/// constantly re-allocating and moving memory around. Written during
/// primitive preparation, and read during batching.
///
/// Storage is partitioned by lifetime: `frame` is per-frame (cleared in
/// `begin_frame`), `scene` is per-scene (recycled on scene rebuild), and
/// `retained` lives across both with its own trim policy.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[derive(Default)]
pub struct PrimitiveScratchBuffer {
    pub frame: PrimitiveFrameScratch,
    pub scene: PrimitiveSceneCache,
    pub retained: PrimitiveRetained,
}

impl PrimitiveScratchBuffer {
    pub fn recycle(&mut self, recycler: &mut Recycler) {
        self.frame.recycle(recycler);
        self.scene.recycle(recycler);
    }

    pub fn begin_frame(&mut self) {
        self.frame.begin_frame();
    }

    pub fn end_frame(&mut self) {
        const MSGS_TO_RETAIN: usize = 32;
        const TIME_TO_RETAIN: u64 = 2000000000;
        const LINE_HEIGHT: f32 = 20.0;
        const X0: f32 = 32.0;
        const Y0: f32 = 32.0;
        let now = zeitstempel::now();

        let messages = &mut self.retained.messages;
        let msgs_to_remove = messages.len().max(MSGS_TO_RETAIN) - MSGS_TO_RETAIN;
        let mut msgs_removed = 0;

        messages.retain(|msg| {
            if msgs_removed < msgs_to_remove {
                msgs_removed += 1;
                return false;
            }

            if msg.timestamp + TIME_TO_RETAIN < now {
                return false;
            }

            true
        });

        let mut y = Y0 + messages.len() as f32 * LINE_HEIGHT;
        let shadow_offset = 1.0;
        let debug_items = &mut self.frame.debug_items;

        for msg in messages.iter() {
            debug_items.push(DebugItem::Text {
                position: DevicePoint::new(X0 + shadow_offset, y + shadow_offset),
                color: debug_colors::BLACK,
                msg: msg.msg.clone(),
            });

            debug_items.push(DebugItem::Text {
                position: DevicePoint::new(X0, y),
                color: debug_colors::RED,
                msg: msg.msg.clone(),
            });

            y -= LINE_HEIGHT;
        }
    }

    pub fn push_debug_rect_with_stroke_width(
        &mut self,
        rect: WorldRect,
        border: ColorF,
        stroke_width: f32
    ) {
        let top_edge = WorldRect::new(
            WorldPoint::new(rect.min.x + stroke_width, rect.min.y),
            WorldPoint::new(rect.max.x - stroke_width, rect.min.y + stroke_width)
        );
        self.push_debug_rect(top_edge * DevicePixelScale::new(1.0), 1, border, border);

        let bottom_edge = WorldRect::new(
            WorldPoint::new(rect.min.x + stroke_width, rect.max.y - stroke_width),
            WorldPoint::new(rect.max.x - stroke_width, rect.max.y)
        );
        self.push_debug_rect(bottom_edge * DevicePixelScale::new(1.0), 1, border, border);

        let right_edge = WorldRect::new(
            WorldPoint::new(rect.max.x - stroke_width, rect.min.y),
            rect.max
        );
        self.push_debug_rect(right_edge * DevicePixelScale::new(1.0), 1, border, border);

        let left_edge = WorldRect::new(
            rect.min,
            WorldPoint::new(rect.min.x + stroke_width, rect.max.y)
        );
        self.push_debug_rect(left_edge * DevicePixelScale::new(1.0), 1, border, border);
    }

    #[allow(dead_code)]
    pub fn push_debug_rect(
        &mut self,
        rect: DeviceRect,
        thickness: i32,
        outer_color: ColorF,
        inner_color: ColorF,
    ) {
        self.frame.debug_items.push(DebugItem::Rect {
            rect,
            outer_color,
            inner_color,
            thickness,
        });
    }

    #[allow(dead_code)]
    pub fn push_debug_string(
        &mut self,
        position: DevicePoint,
        color: ColorF,
        msg: String,
    ) {
        self.frame.debug_items.push(DebugItem::Text {
            position,
            color,
            msg,
        });
    }

    #[allow(dead_code)]
    pub fn log(
        &mut self,
        msg: String,
    ) {
        self.retained.messages.push(DebugMessage {
            msg,
            timestamp: zeitstempel::now(),
        })
    }
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Clone, Debug)]
pub struct PrimitiveStoreStats {
    picture_count: usize,
}

impl PrimitiveStoreStats {
    pub fn empty() -> Self {
        PrimitiveStoreStats {
            picture_count: 0,
        }
    }
}

#[cfg_attr(feature = "capture", derive(Serialize))]
pub struct PrimitiveStore {
    pub pictures: Vec<PictureInstance>,
}

impl PrimitiveStore {
    pub fn new(stats: &PrimitiveStoreStats) -> PrimitiveStore {
        PrimitiveStore {
            pictures: Vec::with_capacity(stats.picture_count),
        }
    }

    pub fn reset(&mut self) {
        self.pictures.clear();
    }

    pub fn get_stats(&self) -> PrimitiveStoreStats {
        PrimitiveStoreStats {
            picture_count: self.pictures.len(),
        }
    }

    #[allow(unused)]
    pub fn print_picture_tree(&self, root: PictureIndex) {
        use crate::print_tree::PrintTree;
        let mut pt = PrintTree::new("picture tree");
        self.pictures[root.0].print(&self.pictures, root, &mut pt);
    }
}

impl Default for PrimitiveStore {
    fn default() -> Self {
        PrimitiveStore::new(&PrimitiveStoreStats::empty())
    }
}

/// Trait for primitives that are directly internable.
/// see SceneBuilder::add_primitive<P>
pub trait InternablePrimitive: intern::Internable<InternData = ()> + Sized {
    /// Whether this primitive snaps its geometry and clips to the device pixel
    /// grid. Overridden to `false` for device-space content (text runs), whose
    /// clips must stay at their exact sub-pixel position (bug 2050692). Used
    /// when building the primitive's clip leaf.
    const SNAP_CLIPS: bool = true;

    /// Build a new key from self with `info`.
    fn into_key(
        self,
        info: &LayoutPrimitiveInfo,
    ) -> Self::Key;

    fn make_instance_kind(
        key: Self::Key,
        data_handle: intern::Handle<Self>,
        prim_store: &mut PrimitiveStore,
    ) -> PrimitiveKind;
}


#[test]
#[cfg(target_pointer_width = "64")]
fn test_struct_sizes() {
    use std::mem;
    // The sizes of these structures are critical for performance on a number of
    // talos stress tests. If you get a failure here on CI, there's two possibilities:
    // (a) You made a structure smaller than it currently is. Great work! Update the
    //     test expectations and move on.
    // (b) You made a structure larger. This is not necessarily a problem, but should only
    //     be done with care, and after checking if talos performance regresses badly.
    assert_eq!(mem::size_of::<PrimitiveInstance>(), 48, "PrimitiveInstance size changed");
    assert_eq!(mem::size_of::<PrimitiveKind>(), 24, "PrimitiveKind size changed");
}
