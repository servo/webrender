/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! The high-level module responsible for interfacing with the GPU.
//!
//! Much of WebRender's design is driven by separating work into different
//! threads. To avoid the complexities of multi-threaded GPU access, we restrict
//! all communication with the GPU to one thread, the render thread. But since
//! issuing GPU commands is often a bottleneck, we move everything else (i.e.
//! the computation of what commands to issue) to another thread, the
//! RenderBackend thread. The RenderBackend, in turn, may delegate work to other
//! thread (like the SceneBuilder threads or Rayon workers), but the
//! Render-vs-RenderBackend distinction is the most important.
//!
//! The consumer is responsible for initializing the render thread before
//! calling into WebRender, which means that this module also serves as the
//! initial entry point into WebRender, and is responsible for spawning the
//! various other threads discussed above. That said, WebRender initialization
//! returns both the `Renderer` instance as well as a channel for communicating
//! directly with the `RenderBackend`. Aside from a few high-level operations
//! like 'render now', most of interesting commands from the consumer go over
//! that channel and operate on the `RenderBackend`.
//!
//! ## Space conversion guidelines
//! At this stage, we shuld be operating with `DevicePixel` and `FramebufferPixel` only.
//! "Framebuffer" space represents the final destination of our rendeing,
//! and it happens to be Y-flipped on OpenGL. The conversion is done as follows:
//!   - for rasterized primitives, the orthographics projection transforms
//! the content rectangle to -1 to 1
//!   - the viewport transformation is setup to map the whole range to
//! the framebuffer rectangle provided by the document view, stored in `DrawTarget`
//!   - all the direct framebuffer operations, like blitting, reading pixels, and setting
//! up the scissor, are accepting already transformed coordinates, which we can get by
//! calling `DrawTarget::to_framebuffer_rect`

use api::{BlobImageHandler, ColorF, ColorU, MixBlendMode};
use api::{DocumentId, Epoch, ExternalImageHandler};
use api::CrashAnnotator;
#[cfg(feature = "replay")]
use api::ExternalImageId;
use api::{ExternalImageSource, ExternalImageType, FontRenderMode, ImageFormat};
use api::{PipelineId, ImageRendering, Checkpoint, NotificationRequest};
use api::{VoidPtrToSizeFn, PremultipliedColorF};
use api::{RenderNotifier, ImageBufferKind, SharedFontInstanceMap};
#[cfg(feature = "replay")]
use api::ExternalImage;
use api::units::*;
use api::channel::{unbounded_channel, Sender, Receiver};
pub use api::DebugFlags;
use core::time::Duration;
use crate::render_api::{RenderApiSender, DebugCommand, ApiMsg, FrameMsg, MemoryReport};
use crate::batch::{AlphaBatchContainer, BatchKind, BatchFeatures, BatchTextures, BrushBatchKind, ClipBatchList};
#[cfg(any(feature = "capture", feature = "replay"))]
use crate::capture::{CaptureConfig, ExternalCaptureImage, PlainExternalImage};
use crate::composite::{CompositeState, CompositeTileSurface, CompositeTile, ResolvedExternalSurface, CompositorSurfaceTransform};
use crate::composite::{CompositorKind, Compositor, NativeTileId, CompositeSurfaceFormat, ResolvedExternalSurfaceColorData};
use crate::composite::{CompositorConfig, NativeSurfaceOperationDetails, NativeSurfaceId, NativeSurfaceOperation};
use crate::c_str;
use crate::debug_colors;
use crate::debug_render::{DebugItem, DebugRenderer};
use crate::device::{CustomVAO, DepthFunction, Device, DrawTarget, ExternalTexture, GpuFrameId, Program};
use crate::device::{ProgramCache, ReadTarget, ShaderError, Texture, TextureFilter, TextureFlags, TextureSlot};
use crate::device::{TextureUploader, UploadMethod, UploadPBOPool, UploadStagingBuffer, VAO, VBO, VertexUsageHint};
use crate::device::query::{GpuSampler, GpuTimer};
#[cfg(feature = "capture")]
use crate::device::FBOId;
use euclid::{rect, Transform3D, Scale, default};
use crate::frame_builder::{Frame, ChasePrimitive, FrameBuilderConfig};
use gleam::gl;
use crate::glyph_cache::GlyphCache;
use crate::glyph_rasterizer::{GlyphFormat, GlyphRasterizer};
use crate::gpu_cache::{GpuBlockData, GpuCacheUpdate, GpuCacheUpdateList};
use crate::gpu_cache::{GpuCacheDebugChunk, GpuCacheDebugCmd};
use crate::gpu_types::{PrimitiveHeaderI, PrimitiveHeaderF, PrimitiveInstanceData, ScalingInstance, SvgFilterInstance};
use crate::gpu_types::{ClearInstance, CompositeInstance, TransformData, ZBufferId, BlurInstance};
use crate::internal_types::{TextureSource, ResourceCacheError};
use crate::internal_types::{CacheTextureId, DebugOutput, FastHashMap, FastHashSet, LayerIndex, RenderedDocument, ResultMsg};
use crate::internal_types::{TextureCacheAllocationKind, TextureCacheUpdate, TextureUpdateList, TextureUpdateSource};
use crate::internal_types::{RenderTargetInfo, Swizzle, DeferredResolveIndex};
use malloc_size_of::MallocSizeOfOps;
use crate::picture::{self, ResolvedSurfaceTexture};
use crate::prim_store::DeferredResolve;
use crate::profiler::{self, GpuProfileTag, TransactionProfile};
use crate::profiler::{Profiler, add_event_marker, add_text_marker, thread_is_being_profiled};
use crate::device::query::{GpuProfiler, GpuDebugMethod};
use rayon::{ThreadPool, ThreadPoolBuilder};
use crate::render_backend::{FrameId, RenderBackend};
use crate::render_task_graph::RenderTaskGraph;
use crate::render_task::{RenderTask, RenderTaskData, RenderTaskKind};
use crate::resource_cache::ResourceCache;
use crate::scene_builder_thread::{SceneBuilderThread, SceneBuilderThreadChannels, LowPrioritySceneBuilderThread};
use crate::screen_capture::AsyncScreenshotGrabber;
use crate::shade::{Shaders, WrShaders};
use crate::guillotine_allocator::{GuillotineAllocator, FreeRectSlice};
use crate::texture_cache::{TextureCache, TextureCacheConfig};
use crate::render_target::{AlphaRenderTarget, ColorRenderTarget, PictureCacheTarget};
use crate::render_target::{RenderTarget, TextureCacheRenderTarget};
use crate::render_target::{RenderTargetKind, BlitJob, BlitJobSource};
use crate::tile_cache::PictureCacheDebugInfo;
use crate::util::drain_filter;

use std;
use std::cmp;
use std::collections::VecDeque;
#[cfg(any(feature = "capture", feature = "replay"))]
use std::collections::hash_map::Entry;
use std::f32;
use std::marker::PhantomData;
use std::mem;
use std::os::raw::c_void;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::cell::RefCell;
use tracy_rs::register_thread_with_profiler;
use time::precise_time_ns;

cfg_if! {
    if #[cfg(feature = "debugger")] {
        use serde_json;
        use crate::debug_server;
    }
}

const VERTEX_TEXTURE_EXTRA_ROWS: i32 = 10;

/// The size of the array of each type of vertex data texture that
/// is round-robin-ed each frame during bind_frame_data. Doing this
/// helps avoid driver stalls while updating the texture in some
/// drivers. The size of these textures are typically very small
/// (e.g. < 16 kB) so it's not a huge waste of memory. Despite that,
/// this is a short-term solution - we want to find a better way
/// to provide this frame data, which will likely involve some
/// combination of UBO/SSBO usage. Although this only affects some
/// platforms, it's enabled on all platforms to reduce testing
/// differences between platforms.
const VERTEX_DATA_TEXTURE_COUNT: usize = 3;

/// Use this hint for all vertex data re-initialization. This allows
/// the driver to better re-use RBOs internally.
pub const ONE_TIME_USAGE_HINT: VertexUsageHint = VertexUsageHint::Stream;

/// Is only false if no WR instances have ever been created.
static HAS_BEEN_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Returns true if a WR instance has ever been initialized in this process.
pub fn wr_has_been_initialized() -> bool {
    HAS_BEEN_INITIALIZED.load(Ordering::SeqCst)
}

pub const MAX_VERTEX_TEXTURE_WIDTH: usize = webrender_build::MAX_VERTEX_TEXTURE_WIDTH;
/// Enabling this toggle would force the GPU cache scattered texture to
/// be resized every frame, which enables GPU debuggers to see if this
/// is performed correctly.
const GPU_CACHE_RESIZE_TEST: bool = false;

/// Number of GPU blocks per UV rectangle provided for an image.
pub const BLOCKS_PER_UV_RECT: usize = 2;

const GPU_TAG_BRUSH_OPACITY: GpuProfileTag = GpuProfileTag {
    label: "B_Opacity",
    color: debug_colors::DARKMAGENTA,
};
const GPU_TAG_BRUSH_LINEAR_GRADIENT: GpuProfileTag = GpuProfileTag {
    label: "B_LinearGradient",
    color: debug_colors::POWDERBLUE,
};
const GPU_TAG_BRUSH_RADIAL_GRADIENT: GpuProfileTag = GpuProfileTag {
    label: "B_RadialGradient",
    color: debug_colors::LIGHTPINK,
};
const GPU_TAG_BRUSH_CONIC_GRADIENT: GpuProfileTag = GpuProfileTag {
    label: "B_ConicGradient",
    color: debug_colors::GREEN,
};
const GPU_TAG_BRUSH_YUV_IMAGE: GpuProfileTag = GpuProfileTag {
    label: "B_YuvImage",
    color: debug_colors::DARKGREEN,
};
const GPU_TAG_BRUSH_MIXBLEND: GpuProfileTag = GpuProfileTag {
    label: "B_MixBlend",
    color: debug_colors::MAGENTA,
};
const GPU_TAG_BRUSH_BLEND: GpuProfileTag = GpuProfileTag {
    label: "B_Blend",
    color: debug_colors::ORANGE,
};
const GPU_TAG_BRUSH_IMAGE: GpuProfileTag = GpuProfileTag {
    label: "B_Image",
    color: debug_colors::SPRINGGREEN,
};
const GPU_TAG_BRUSH_SOLID: GpuProfileTag = GpuProfileTag {
    label: "B_Solid",
    color: debug_colors::RED,
};
const GPU_TAG_CACHE_CLIP: GpuProfileTag = GpuProfileTag {
    label: "C_Clip",
    color: debug_colors::PURPLE,
};
const GPU_TAG_CACHE_BORDER: GpuProfileTag = GpuProfileTag {
    label: "C_Border",
    color: debug_colors::CORNSILK,
};
const GPU_TAG_CACHE_LINE_DECORATION: GpuProfileTag = GpuProfileTag {
    label: "C_LineDecoration",
    color: debug_colors::YELLOWGREEN,
};
const GPU_TAG_CACHE_GRADIENT: GpuProfileTag = GpuProfileTag {
    label: "C_Gradient",
    color: debug_colors::BROWN,
};
const GPU_TAG_SETUP_TARGET: GpuProfileTag = GpuProfileTag {
    label: "target init",
    color: debug_colors::SLATEGREY,
};
const GPU_TAG_SETUP_DATA: GpuProfileTag = GpuProfileTag {
    label: "data init",
    color: debug_colors::LIGHTGREY,
};
const GPU_TAG_PRIM_SPLIT_COMPOSITE: GpuProfileTag = GpuProfileTag {
    label: "SplitComposite",
    color: debug_colors::DARKBLUE,
};
const GPU_TAG_PRIM_TEXT_RUN: GpuProfileTag = GpuProfileTag {
    label: "TextRun",
    color: debug_colors::BLUE,
};
const GPU_TAG_BLUR: GpuProfileTag = GpuProfileTag {
    label: "Blur",
    color: debug_colors::VIOLET,
};
const GPU_TAG_BLIT: GpuProfileTag = GpuProfileTag {
    label: "Blit",
    color: debug_colors::LIME,
};
const GPU_TAG_SCALE: GpuProfileTag = GpuProfileTag {
    label: "Scale",
    color: debug_colors::GHOSTWHITE,
};
const GPU_SAMPLER_TAG_ALPHA: GpuProfileTag = GpuProfileTag {
    label: "Alpha targets",
    color: debug_colors::BLACK,
};
const GPU_SAMPLER_TAG_OPAQUE: GpuProfileTag = GpuProfileTag {
    label: "Opaque pass",
    color: debug_colors::BLACK,
};
const GPU_SAMPLER_TAG_TRANSPARENT: GpuProfileTag = GpuProfileTag {
    label: "Transparent pass",
    color: debug_colors::BLACK,
};
const GPU_TAG_SVG_FILTER: GpuProfileTag = GpuProfileTag {
    label: "SvgFilter",
    color: debug_colors::LEMONCHIFFON,
};
const GPU_TAG_COMPOSITE: GpuProfileTag = GpuProfileTag {
    label: "Composite",
    color: debug_colors::TOMATO,
};
const GPU_TAG_CLEAR: GpuProfileTag = GpuProfileTag {
    label: "Clear",
    color: debug_colors::CHOCOLATE,
};

/// The clear color used for the texture cache when the debug display is enabled.
/// We use a shade of blue so that we can still identify completely blue items in
/// the texture cache.
const TEXTURE_CACHE_DBG_CLEAR_COLOR: [f32; 4] = [0.0, 0.0, 0.8, 1.0];

impl BatchKind {
    #[cfg(feature = "debugger")]
    fn debug_name(&self) -> &'static str {
        match *self {
            BatchKind::SplitComposite => "SplitComposite",
            BatchKind::Brush(kind) => {
                match kind {
                    BrushBatchKind::Solid => "Brush (Solid)",
                    BrushBatchKind::Image(..) => "Brush (Image)",
                    BrushBatchKind::Blend => "Brush (Blend)",
                    BrushBatchKind::MixBlend { .. } => "Brush (Composite)",
                    BrushBatchKind::YuvImage(..) => "Brush (YuvImage)",
                    BrushBatchKind::ConicGradient => "Brush (ConicGradient)",
                    BrushBatchKind::RadialGradient => "Brush (RadialGradient)",
                    BrushBatchKind::LinearGradient => "Brush (LinearGradient)",
                    BrushBatchKind::Opacity => "Brush (Opacity)",
                }
            }
            BatchKind::TextRun(_) => "TextRun",
        }
    }

    fn sampler_tag(&self) -> GpuProfileTag {
        match *self {
            BatchKind::SplitComposite => GPU_TAG_PRIM_SPLIT_COMPOSITE,
            BatchKind::Brush(kind) => {
                match kind {
                    BrushBatchKind::Solid => GPU_TAG_BRUSH_SOLID,
                    BrushBatchKind::Image(..) => GPU_TAG_BRUSH_IMAGE,
                    BrushBatchKind::Blend => GPU_TAG_BRUSH_BLEND,
                    BrushBatchKind::MixBlend { .. } => GPU_TAG_BRUSH_MIXBLEND,
                    BrushBatchKind::YuvImage(..) => GPU_TAG_BRUSH_YUV_IMAGE,
                    BrushBatchKind::ConicGradient => GPU_TAG_BRUSH_CONIC_GRADIENT,
                    BrushBatchKind::RadialGradient => GPU_TAG_BRUSH_RADIAL_GRADIENT,
                    BrushBatchKind::LinearGradient => GPU_TAG_BRUSH_LINEAR_GRADIENT,
                    BrushBatchKind::Opacity => GPU_TAG_BRUSH_OPACITY,
                }
            }
            BatchKind::TextRun(_) => GPU_TAG_PRIM_TEXT_RUN,
        }
    }
}

fn flag_changed(before: DebugFlags, after: DebugFlags, select: DebugFlags) -> Option<bool> {
    if before & select != after & select {
        Some(after.contains(select))
    } else {
        None
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub enum ShaderColorMode {
    FromRenderPassMode = 0,
    Alpha = 1,
    SubpixelConstantTextColor = 2,
    SubpixelWithBgColorPass0 = 3,
    SubpixelWithBgColorPass1 = 4,
    SubpixelWithBgColorPass2 = 5,
    SubpixelDualSource = 6,
    Bitmap = 7,
    ColorBitmap = 8,
    Image = 9,
}

impl From<GlyphFormat> for ShaderColorMode {
    fn from(format: GlyphFormat) -> ShaderColorMode {
        match format {
            GlyphFormat::Alpha | GlyphFormat::TransformedAlpha => ShaderColorMode::Alpha,
            GlyphFormat::Subpixel | GlyphFormat::TransformedSubpixel => {
                panic!("Subpixel glyph formats must be handled separately.");
            }
            GlyphFormat::Bitmap => ShaderColorMode::Bitmap,
            GlyphFormat::ColorBitmap => ShaderColorMode::ColorBitmap,
        }
    }
}

/// Enumeration of the texture samplers used across the various WebRender shaders.
///
/// Each variant corresponds to a uniform declared in shader source. We only bind
/// the variants we need for a given shader, so not every variant is bound for every
/// batch.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum TextureSampler {
    Color0,
    Color1,
    Color2,
    GpuCache,
    TransformPalette,
    RenderTasks,
    Dither,
    PrimitiveHeadersF,
    PrimitiveHeadersI,
    ClipMask,
}

impl TextureSampler {
    pub(crate) fn color(n: usize) -> TextureSampler {
        match n {
            0 => TextureSampler::Color0,
            1 => TextureSampler::Color1,
            2 => TextureSampler::Color2,
            _ => {
                panic!("There are only 3 color samplers.");
            }
        }
    }
}

impl Into<TextureSlot> for TextureSampler {
    fn into(self) -> TextureSlot {
        match self {
            TextureSampler::Color0 => TextureSlot(0),
            TextureSampler::Color1 => TextureSlot(1),
            TextureSampler::Color2 => TextureSlot(2),
            TextureSampler::GpuCache => TextureSlot(3),
            TextureSampler::TransformPalette => TextureSlot(4),
            TextureSampler::RenderTasks => TextureSlot(5),
            TextureSampler::Dither => TextureSlot(6),
            TextureSampler::PrimitiveHeadersF => TextureSlot(7),
            TextureSampler::PrimitiveHeadersI => TextureSlot(8),
            TextureSampler::ClipMask => TextureSlot(9),
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PackedVertex {
    pub pos: [f32; 2],
}

pub(crate) mod desc {
    use crate::device::{VertexAttribute, VertexAttributeKind, VertexDescriptor};

    pub const PRIM_INSTANCES: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aData",
                count: 4,
                kind: VertexAttributeKind::I32,
            },
        ],
    };

    pub const BLUR: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aBlurRenderTaskAddress",
                count: 1,
                kind: VertexAttributeKind::U16,
            },
            VertexAttribute {
                name: "aBlurSourceTaskAddress",
                count: 1,
                kind: VertexAttributeKind::U16,
            },
            VertexAttribute {
                name: "aBlurDirection",
                count: 1,
                kind: VertexAttributeKind::I32,
            },
        ],
    };

    pub const LINE: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aTaskRect",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aLocalSize",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aWavyLineThickness",
                count: 1,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aStyle",
                count: 1,
                kind: VertexAttributeKind::I32,
            },
            VertexAttribute {
                name: "aAxisSelect",
                count: 1,
                kind: VertexAttributeKind::F32,
            },
        ],
    };

    pub const GRADIENT: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aTaskRect",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aStops",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            // TODO(gw): We should probably pack these as u32 colors instead
            //           of passing as full float vec4 here. It won't make much
            //           difference in real world, since these are only invoked
            //           rarely, when creating the cache.
            VertexAttribute {
                name: "aColor0",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aColor1",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aColor2",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aColor3",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aAxisSelect",
                count: 1,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aStartStop",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
    };

    pub const BORDER: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aTaskOrigin",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aRect",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aColor0",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aColor1",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aFlags",
                count: 1,
                kind: VertexAttributeKind::I32,
            },
            VertexAttribute {
                name: "aWidths",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aRadii",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aClipParams1",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aClipParams2",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
        ],
    };

    pub const SCALE: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aScaleTargetRect",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aScaleSourceRect",
                count: 4,
                kind: VertexAttributeKind::I32,
            },
            VertexAttribute {
                name: "aScaleSourceLayer",
                count: 1,
                kind: VertexAttributeKind::I32,
            },
        ],
    };

    pub const CLIP_RECT: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            // common clip attributes
            VertexAttribute {
                name: "aClipDeviceArea",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aClipOrigins",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aDevicePixelScale",
                count: 1,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aTransformIds",
                count: 2,
                kind: VertexAttributeKind::I32,
            },
            // specific clip attributes
            VertexAttribute {
                name: "aClipLocalPos",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aClipLocalRect",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aClipMode",
                count: 1,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aClipRect_TL",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aClipRadii_TL",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aClipRect_TR",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aClipRadii_TR",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aClipRect_BL",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aClipRadii_BL",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aClipRect_BR",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aClipRadii_BR",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
        ],
    };

    pub const CLIP_BOX_SHADOW: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            // common clip attributes
            VertexAttribute {
                name: "aClipDeviceArea",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aClipOrigins",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aDevicePixelScale",
                count: 1,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aTransformIds",
                count: 2,
                kind: VertexAttributeKind::I32,
            },
            // specific clip attributes
            VertexAttribute {
                name: "aClipDataResourceAddress",
                count: 2,
                kind: VertexAttributeKind::U16,
            },
            VertexAttribute {
                name: "aClipSrcRectSize",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aClipMode",
                count: 1,
                kind: VertexAttributeKind::I32,
            },
            VertexAttribute {
                name: "aStretchMode",
                count: 2,
                kind: VertexAttributeKind::I32,
            },
            VertexAttribute {
                name: "aClipDestRect",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
        ],
    };

    pub const CLIP_IMAGE: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            // common clip attributes
            VertexAttribute {
                name: "aClipDeviceArea",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aClipOrigins",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aDevicePixelScale",
                count: 1,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aTransformIds",
                count: 2,
                kind: VertexAttributeKind::I32,
            },
            // specific clip attributes
            VertexAttribute {
                name: "aClipTileRect",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aClipDataResourceAddress",
                count: 2,
                kind: VertexAttributeKind::U16,
            },
            VertexAttribute {
                name: "aClipLocalRect",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
        ],
    };

    pub const GPU_CACHE_UPDATE: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::U16Norm,
            },
            VertexAttribute {
                name: "aValue",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[],
    };

    pub const RESOLVE: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aRect",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
        ],
    };

    pub const SVG_FILTER: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aFilterRenderTaskAddress",
                count: 1,
                kind: VertexAttributeKind::U16,
            },
            VertexAttribute {
                name: "aFilterInput1TaskAddress",
                count: 1,
                kind: VertexAttributeKind::U16,
            },
            VertexAttribute {
                name: "aFilterInput2TaskAddress",
                count: 1,
                kind: VertexAttributeKind::U16,
            },
            VertexAttribute {
                name: "aFilterKind",
                count: 1,
                kind: VertexAttributeKind::U16,
            },
            VertexAttribute {
                name: "aFilterInputCount",
                count: 1,
                kind: VertexAttributeKind::U16,
            },
            VertexAttribute {
                name: "aFilterGenericInt",
                count: 1,
                kind: VertexAttributeKind::U16,
            },
            VertexAttribute {
                name: "aFilterExtraDataAddress",
                count: 2,
                kind: VertexAttributeKind::U16,
            },
        ],
    };

    pub const VECTOR_STENCIL: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aFromPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aCtrlPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aToPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aFromNormal",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aCtrlNormal",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aToNormal",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aPathID",
                count: 1,
                kind: VertexAttributeKind::U16,
            },
            VertexAttribute {
                name: "aPad",
                count: 1,
                kind: VertexAttributeKind::U16,
            },
        ],
    };

    pub const VECTOR_COVER: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aTargetRect",
                count: 4,
                kind: VertexAttributeKind::I32,
            },
            VertexAttribute {
                name: "aStencilOrigin",
                count: 2,
                kind: VertexAttributeKind::I32,
            },
            VertexAttribute {
                name: "aSubpixel",
                count: 1,
                kind: VertexAttributeKind::U16,
            },
            VertexAttribute {
                name: "aPad",
                count: 1,
                kind: VertexAttributeKind::U16,
            },
        ],
    };

    pub const COMPOSITE: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aDeviceRect",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aDeviceClipRect",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aColor",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aParams",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aUvRect0",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aUvRect1",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aUvRect2",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aTextureLayers",
                count: 3,
                kind: VertexAttributeKind::F32,
            },
        ],
    };

    pub const CLEAR: VertexDescriptor = VertexDescriptor {
        vertex_attributes: &[
            VertexAttribute {
                name: "aPosition",
                count: 2,
                kind: VertexAttributeKind::F32,
            },
        ],
        instance_attributes: &[
            VertexAttribute {
                name: "aRect",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
            VertexAttribute {
                name: "aColor",
                count: 4,
                kind: VertexAttributeKind::F32,
            },
        ],
    };
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) enum VertexArrayKind {
    Primitive,
    Blur,
    ClipImage,
    ClipRect,
    ClipBoxShadow,
    VectorStencil,
    VectorCover,
    Border,
    Scale,
    LineDecoration,
    Gradient,
    Resolve,
    SvgFilter,
    Composite,
    Clear,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GraphicsApi {
    OpenGL,
}

#[derive(Clone, Debug)]
pub struct GraphicsApiInfo {
    pub kind: GraphicsApi,
    pub renderer: String,
    pub version: String,
}

#[derive(Debug)]
pub struct GpuProfile {
    pub frame_id: GpuFrameId,
    pub paint_time_ns: u64,
}

impl GpuProfile {
    fn new(frame_id: GpuFrameId, timers: &[GpuTimer]) -> GpuProfile {
        let mut paint_time_ns = 0;
        for timer in timers {
            paint_time_ns += timer.time_ns;
        }
        GpuProfile {
            frame_id,
            paint_time_ns,
        }
    }
}

#[derive(Debug)]
pub struct CpuProfile {
    pub frame_id: GpuFrameId,
    pub backend_time_ns: u64,
    pub composite_time_ns: u64,
    pub draw_calls: usize,
}

impl CpuProfile {
    fn new(
        frame_id: GpuFrameId,
        backend_time_ns: u64,
        composite_time_ns: u64,
        draw_calls: usize,
    ) -> CpuProfile {
        CpuProfile {
            frame_id,
            backend_time_ns,
            composite_time_ns,
            draw_calls,
        }
    }
}

/// The selected partial present mode for a given frame.
#[derive(Debug, Copy, Clone)]
enum PartialPresentMode {
    /// The device supports fewer dirty rects than the number of dirty rects
    /// that WR produced. In this case, the WR dirty rects are union'ed into
    /// a single dirty rect, that is provided to the caller.
    Single {
        dirty_rect: DeviceRect,
    },
}

/// Helper struct for resolving device Textures for use during rendering passes.
///
/// Manages the mapping between the at-a-distance texture handles used by the
/// `RenderBackend` (which does not directly interface with the GPU) and actual
/// device texture handles.
struct TextureResolver {
    /// A map to resolve texture cache IDs to native textures.
    texture_cache_map: FastHashMap<CacheTextureId, Texture>,

    /// Map of external image IDs to native textures.
    external_images: FastHashMap<DeferredResolveIndex, ExternalTexture>,

    /// A special 1x1 dummy texture used for shaders that expect to work with
    /// the output of the previous pass but are actually running in the first
    /// pass.
    dummy_cache_texture: Texture,
}

impl TextureResolver {
    fn new(device: &mut Device) -> TextureResolver {
        let dummy_cache_texture = device
            .create_texture(
                ImageBufferKind::Texture2D,
                ImageFormat::RGBA8,
                1,
                1,
                TextureFilter::Linear,
                None,
                1,
            );
        device.upload_texture_immediate(
            &dummy_cache_texture,
            &[0xff, 0xff, 0xff, 0xff],
        );

        TextureResolver {
            texture_cache_map: FastHashMap::default(),
            external_images: FastHashMap::default(),
            dummy_cache_texture,
        }
    }

    fn deinit(self, device: &mut Device) {
        device.delete_texture(self.dummy_cache_texture);

        for (_id, texture) in self.texture_cache_map {
            device.delete_texture(texture);
        }
    }

    fn begin_frame(&mut self) {
    }

    fn end_pass(
        &mut self,
        device: &mut Device,
        textures_to_invalidate: &[CacheTextureId],
    ) {
        // For any texture that is no longer needed, immediately
        // invalidate it so that tiled GPUs don't need to resolve it
        // back to memory.
        for texture_id in textures_to_invalidate {
            let render_target = &self.texture_cache_map[texture_id];
            device.invalidate_render_target(render_target);
        }
    }

    // Bind a source texture to the device.
    fn bind(&self, texture_id: &TextureSource, sampler: TextureSampler, device: &mut Device) -> Swizzle {
        match *texture_id {
            TextureSource::Invalid => {
                Swizzle::default()
            }
            TextureSource::Dummy => {
                let swizzle = Swizzle::default();
                device.bind_texture(sampler, &self.dummy_cache_texture, swizzle);
                swizzle
            }
            TextureSource::External(ref index, _) => {
                let texture = self.external_images
                    .get(index)
                    .expect("BUG: External image should be resolved by now");
                device.bind_external_texture(sampler, texture);
                Swizzle::default()
            }
            TextureSource::TextureCache(index, swizzle) => {
                let texture = &self.texture_cache_map[&index];
                device.bind_texture(sampler, texture, swizzle);
                swizzle
            }
        }
    }

    // Get the real (OpenGL) texture ID for a given source texture.
    // For a texture cache texture, the IDs are stored in a vector
    // map for fast access.
    fn resolve(&self, texture_id: &TextureSource) -> Option<(&Texture, Swizzle)> {
        match *texture_id {
            TextureSource::Invalid => None,
            TextureSource::Dummy => {
                Some((&self.dummy_cache_texture, Swizzle::default()))
            }
            TextureSource::External(..) => {
                panic!("BUG: External textures cannot be resolved, they can only be bound.");
            }
            TextureSource::TextureCache(index, swizzle) => {
                Some((&self.texture_cache_map[&index], swizzle))
            }
        }
    }

    // Retrieve the deferred / resolved UV rect if an external texture, otherwise
    // return the default supplied UV rect.
    fn get_uv_rect(
        &self,
        source: &TextureSource,
        default_value: TexelRect,
    ) -> TexelRect {
        match source {
            TextureSource::External(ref index, _) => {
                let texture = self.external_images
                    .get(index)
                    .expect("BUG: External image should be resolved by now");
                texture.get_uv_rect()
            }
            _ => {
                default_value
            }
        }
    }

    fn report_memory(&self) -> MemoryReport {
        let mut report = MemoryReport::default();

        // We're reporting GPU memory rather than heap-allocations, so we don't
        // use size_of_op.
        for t in self.texture_cache_map.values() {
            report.texture_cache_textures += t.size_in_bytes();
        }

        report
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum BlendMode {
    None,
    Alpha,
    PremultipliedAlpha,
    PremultipliedDestOut,
    SubpixelDualSource,
    SubpixelConstantTextColor(ColorF),
    SubpixelWithBgColor,
    Advanced(MixBlendMode),
}

/// Tracks the state of each row in the GPU cache texture.
struct CacheRow {
    /// Mirrored block data on CPU for this row. We store a copy of
    /// the data on the CPU side to improve upload batching.
    cpu_blocks: Box<[GpuBlockData; MAX_VERTEX_TEXTURE_WIDTH]>,
    /// The first offset in this row that is dirty.
    min_dirty: u16,
    /// The last offset in this row that is dirty.
    max_dirty: u16,
}

impl CacheRow {
    fn new() -> Self {
        CacheRow {
            cpu_blocks: Box::new([GpuBlockData::EMPTY; MAX_VERTEX_TEXTURE_WIDTH]),
            min_dirty: MAX_VERTEX_TEXTURE_WIDTH as _,
            max_dirty: 0,
        }
    }

    fn is_dirty(&self) -> bool {
        return self.min_dirty < self.max_dirty;
    }

    fn clear_dirty(&mut self) {
        self.min_dirty = MAX_VERTEX_TEXTURE_WIDTH as _;
        self.max_dirty = 0;
    }

    fn add_dirty(&mut self, block_offset: usize, block_count: usize) {
        self.min_dirty = self.min_dirty.min(block_offset as _);
        self.max_dirty = self.max_dirty.max((block_offset + block_count) as _);
    }

    fn dirty_blocks(&self) -> &[GpuBlockData] {
        return &self.cpu_blocks[self.min_dirty as usize .. self.max_dirty as usize];
    }
}

/// The bus over which CPU and GPU versions of the GPU cache
/// get synchronized.
enum GpuCacheBus {
    /// PBO-based updates, currently operate on a row granularity.
    /// Therefore, are subject to fragmentation issues.
    PixelBuffer {
        /// Per-row data.
        rows: Vec<CacheRow>,
    },
    /// Shader-based scattering updates. Currently rendered by a set
    /// of points into the GPU texture, each carrying a `GpuBlockData`.
    Scatter {
        /// Special program to run the scattered update.
        program: Program,
        /// VAO containing the source vertex buffers.
        vao: CustomVAO,
        /// VBO for positional data, supplied as normalized `u16`.
        buf_position: VBO<[u16; 2]>,
        /// VBO for gpu block data.
        buf_value: VBO<GpuBlockData>,
        /// Currently stored block count.
        count: usize,
    },
}

/// The device-specific representation of the cache texture in gpu_cache.rs
struct GpuCacheTexture {
    texture: Option<Texture>,
    bus: GpuCacheBus,
}

impl GpuCacheTexture {

    /// Ensures that we have an appropriately-sized texture. Returns true if a
    /// new texture was created.
    fn ensure_texture(&mut self, device: &mut Device, height: i32) {
        // If we already have a texture that works, we're done.
        if self.texture.as_ref().map_or(false, |t| t.get_dimensions().height >= height) {
            if GPU_CACHE_RESIZE_TEST {
                // Special debug mode - resize the texture even though it's fine.
            } else {
                return;
            }
        }

        // Take the old texture, if any.
        let blit_source = self.texture.take();

        // Create the new texture.
        assert!(height >= 2, "Height is too small for ANGLE");
        let new_size = DeviceIntSize::new(MAX_VERTEX_TEXTURE_WIDTH as _, height);
        // If glCopyImageSubData is supported, this texture doesn't need
        // to be a render target. This prevents GL errors due to framebuffer
        // incompleteness on devices that don't support RGBAF32 render targets.
        // TODO(gw): We still need a proper solution for the subset of devices
        //           that don't support glCopyImageSubData *OR* rendering to a
        //           RGBAF32 render target. These devices will currently fail
        //           to resize the GPU cache texture.
        let supports_copy_image_sub_data = device.get_capabilities().supports_copy_image_sub_data;
        let rt_info =  match self.bus {
            GpuCacheBus::PixelBuffer { .. } if supports_copy_image_sub_data => None,
            _ => Some(RenderTargetInfo { has_depth: false }),
        };
        let mut texture = device.create_texture(
            ImageBufferKind::Texture2D,
            ImageFormat::RGBAF32,
            new_size.width,
            new_size.height,
            TextureFilter::Nearest,
            rt_info,
            1,
        );

        // Copy the contents of the previous texture, if applicable.
        if let Some(blit_source) = blit_source {
            device.copy_entire_texture(&mut texture, &blit_source);
            device.delete_texture(blit_source);
        }

        self.texture = Some(texture);
    }

    fn new(device: &mut Device, use_scatter: bool) -> Result<Self, RendererError> {
        let bus = if use_scatter {
            let program = device.create_program_linked(
                "gpu_cache_update",
                &[],
                &desc::GPU_CACHE_UPDATE,
            )?;
            let buf_position = device.create_vbo();
            let buf_value = device.create_vbo();
            //Note: the vertex attributes have to be supplied in the same order
            // as for program creation, but each assigned to a different stream.
            let vao = device.create_custom_vao(&[
                buf_position.stream_with(&desc::GPU_CACHE_UPDATE.vertex_attributes[0..1]),
                buf_value   .stream_with(&desc::GPU_CACHE_UPDATE.vertex_attributes[1..2]),
            ]);
            GpuCacheBus::Scatter {
                program,
                vao,
                buf_position,
                buf_value,
                count: 0,
            }
        } else {
            GpuCacheBus::PixelBuffer {
                rows: Vec::new(),
            }
        };

        Ok(GpuCacheTexture {
            texture: None,
            bus,
        })
    }

    fn deinit(mut self, device: &mut Device) {
        if let Some(t) = self.texture.take() {
            device.delete_texture(t);
        }
        if let GpuCacheBus::Scatter { program, vao, buf_position, buf_value, .. } = self.bus {
            device.delete_program(program);
            device.delete_custom_vao(vao);
            device.delete_vbo(buf_position);
            device.delete_vbo(buf_value);
        }
    }

    fn get_height(&self) -> i32 {
        self.texture.as_ref().map_or(0, |t| t.get_dimensions().height)
    }

    fn prepare_for_updates(
        &mut self,
        device: &mut Device,
        total_block_count: usize,
        max_height: i32,
    ) {
        self.ensure_texture(device, max_height);
        match self.bus {
            GpuCacheBus::PixelBuffer { .. } => {},
            GpuCacheBus::Scatter {
                ref mut buf_position,
                ref mut buf_value,
                ref mut count,
                ..
            } => {
                *count = 0;
                if total_block_count > buf_value.allocated_count() {
                    device.allocate_vbo(buf_position, total_block_count, ONE_TIME_USAGE_HINT);
                    device.allocate_vbo(buf_value,    total_block_count, ONE_TIME_USAGE_HINT);
                }
            }
        }
    }

    fn update(&mut self, device: &mut Device, updates: &GpuCacheUpdateList) {
        match self.bus {
            GpuCacheBus::PixelBuffer { ref mut rows, .. } => {
                for update in &updates.updates {
                    match *update {
                        GpuCacheUpdate::Copy {
                            block_index,
                            block_count,
                            address,
                        } => {
                            let row = address.v as usize;

                            // Ensure that the CPU-side shadow copy of the GPU cache data has enough
                            // rows to apply this patch.
                            while rows.len() <= row {
                                // Add a new row.
                                rows.push(CacheRow::new());
                            }

                            // Copy the blocks from the patch array in the shadow CPU copy.
                            let block_offset = address.u as usize;
                            let data = &mut rows[row].cpu_blocks;
                            for i in 0 .. block_count {
                                data[block_offset + i] = updates.blocks[block_index + i];
                            }

                            // This row is dirty (needs to be updated in GPU texture).
                            rows[row].add_dirty(block_offset, block_count);
                        }
                    }
                }
            }
            GpuCacheBus::Scatter {
                ref buf_position,
                ref buf_value,
                ref mut count,
                ..
            } => {
                //TODO: re-use this heap allocation
                // Unused positions will be left as 0xFFFF, which translates to
                // (1.0, 1.0) in the vertex output position and gets culled out
                let mut position_data = vec![[!0u16; 2]; updates.blocks.len()];
                let size = self.texture.as_ref().unwrap().get_dimensions().to_usize();

                for update in &updates.updates {
                    match *update {
                        GpuCacheUpdate::Copy {
                            block_index,
                            block_count,
                            address,
                        } => {
                            // Convert the absolute texel position into normalized
                            let y = ((2*address.v as usize + 1) << 15) / size.height;
                            for i in 0 .. block_count {
                                let x = ((2*address.u as usize + 2*i + 1) << 15) / size.width;
                                position_data[block_index + i] = [x as _, y as _];
                            }
                        }
                    }
                }

                device.fill_vbo(buf_value, &updates.blocks, *count);
                device.fill_vbo(buf_position, &position_data, *count);
                *count += position_data.len();
            }
        }
    }

    fn flush(&mut self, device: &mut Device, pbo_pool: &mut UploadPBOPool) -> usize {
        let texture = self.texture.as_ref().unwrap();
        match self.bus {
            GpuCacheBus::PixelBuffer { ref mut rows } => {
                let rows_dirty = rows
                    .iter()
                    .filter(|row| row.is_dirty())
                    .count();
                if rows_dirty == 0 {
                    return 0
                }

                let mut uploader = device.upload_texture(pbo_pool);

                for (row_index, row) in rows.iter_mut().enumerate() {
                    if !row.is_dirty() {
                        continue;
                    }

                    let blocks = row.dirty_blocks();
                    let rect = DeviceIntRect::new(
                        DeviceIntPoint::new(row.min_dirty as i32, row_index as i32),
                        DeviceIntSize::new(blocks.len() as i32, 1),
                    );

                    uploader.upload(device, texture, rect, 0, None, None, blocks.as_ptr(), blocks.len());

                    row.clear_dirty();
                }

                uploader.flush(device);

                rows_dirty
            }
            GpuCacheBus::Scatter { ref program, ref vao, count, .. } => {
                device.disable_depth();
                device.set_blend(false);
                device.bind_program(program);
                device.bind_custom_vao(vao);
                device.bind_draw_target(
                    DrawTarget::from_texture(
                        texture,
                        0,
                        false,
                    ),
                );
                device.draw_nonindexed_points(0, count as _);
                0
            }
        }
    }
}

struct VertexDataTexture<T> {
    texture: Option<Texture>,
    format: ImageFormat,
    _marker: PhantomData<T>,
}

impl<T> VertexDataTexture<T> {
    fn new(
        format: ImageFormat,
    ) -> Self {
        VertexDataTexture {
            texture: None,
            format,
            _marker: PhantomData,
        }
    }

    /// Returns a borrow of the GPU texture. Panics if it hasn't been initialized.
    fn texture(&self) -> &Texture {
        self.texture.as_ref().unwrap()
    }

    /// Returns an estimate of the GPU memory consumed by this VertexDataTexture.
    fn size_in_bytes(&self) -> usize {
        self.texture.as_ref().map_or(0, |t| t.size_in_bytes())
    }

    fn update<'a>(
        &'a mut self,
        device: &mut Device,
        texture_uploader: &mut TextureUploader<'a>,
        data: &mut Vec<T>
    ) {
        debug_assert!(mem::size_of::<T>() % 16 == 0);
        let texels_per_item = mem::size_of::<T>() / 16;
        let items_per_row = MAX_VERTEX_TEXTURE_WIDTH / texels_per_item;
        debug_assert_ne!(items_per_row, 0);

        // Ensure we always end up with a texture when leaving this method.
        let mut len = data.len();
        if len == 0 {
            if self.texture.is_some() {
                return;
            }
            data.reserve(items_per_row);
            len = items_per_row;
        } else {
            // Extend the data array to have enough capacity to upload at least
            // a multiple of the row size.  This ensures memory safety when the
            // array is passed to OpenGL to upload to the GPU.
            let extra = len % items_per_row;
            if extra != 0 {
                let padding = items_per_row - extra;
                data.reserve(padding);
                len += padding;
            }
        }

        let needed_height = (len / items_per_row) as i32;
        let existing_height = self.texture.as_ref().map_or(0, |t| t.get_dimensions().height);

        // Create a new texture if needed.
        //
        // These textures are generally very small, which is why we don't bother
        // with incremental updates and just re-upload every frame. For most pages
        // they're one row each, and on stress tests like css-francine they end up
        // in the 6-14 range. So we size the texture tightly to what we need (usually
        // 1), and shrink it if the waste would be more than `VERTEX_TEXTURE_EXTRA_ROWS`
        // rows. This helps with memory overhead, especially because there are several
        // instances of these textures per Renderer.
        if needed_height > existing_height || needed_height + VERTEX_TEXTURE_EXTRA_ROWS < existing_height {
            // Drop the existing texture, if any.
            if let Some(t) = self.texture.take() {
                device.delete_texture(t);
            }

            let texture = device.create_texture(
                ImageBufferKind::Texture2D,
                self.format,
                MAX_VERTEX_TEXTURE_WIDTH as i32,
                // Ensure height is at least two to work around
                // https://bugs.chromium.org/p/angleproject/issues/detail?id=3039
                needed_height.max(2),
                TextureFilter::Nearest,
                None,
                1,
            );
            self.texture = Some(texture);
        }

        // Note: the actual width can be larger than the logical one, with a few texels
        // of each row unused at the tail. This is needed because there is still hardware
        // (like Intel iGPUs) that prefers power-of-two sizes of textures ([1]).
        //
        // [1] https://software.intel.com/en-us/articles/opengl-performance-tips-power-of-two-textures-have-better-performance
        let logical_width = if needed_height == 1 {
            data.len() * texels_per_item
        } else {
            MAX_VERTEX_TEXTURE_WIDTH - (MAX_VERTEX_TEXTURE_WIDTH % texels_per_item)
        };

        let rect = DeviceIntRect::new(
            DeviceIntPoint::zero(),
            DeviceIntSize::new(logical_width as i32, needed_height),
        );

        debug_assert!(len <= data.capacity(), "CPU copy will read out of bounds");
        texture_uploader.upload(device, self.texture(), rect, 0, None, None, data.as_ptr(), len);
    }

    fn deinit(mut self, device: &mut Device) {
        if let Some(t) = self.texture.take() {
            device.delete_texture(t);
        }
    }
}

#[derive(PartialEq)]
struct TargetSelector {
    size: DeviceIntSize,
    num_layers: usize,
    format: ImageFormat,
}

struct LazyInitializedDebugRenderer {
    debug_renderer: Option<DebugRenderer>,
    failed: bool,
}

impl LazyInitializedDebugRenderer {
    pub fn new() -> Self {
        Self {
            debug_renderer: None,
            failed: false,
        }
    }

    pub fn get_mut<'a>(&'a mut self, device: &mut Device) -> Option<&'a mut DebugRenderer> {
        if self.failed {
            return None;
        }
        if self.debug_renderer.is_none() {
            match DebugRenderer::new(device) {
                Ok(renderer) => { self.debug_renderer = Some(renderer); }
                Err(_) => {
                    // The shader compilation code already logs errors.
                    self.failed = true;
                }
            }
        }

        self.debug_renderer.as_mut()
    }

    /// Returns mut ref to `DebugRenderer` if one already exists, otherwise returns `None`.
    pub fn try_get_mut<'a>(&'a mut self) -> Option<&'a mut DebugRenderer> {
        self.debug_renderer.as_mut()
    }

    pub fn deinit(self, device: &mut Device) {
        if let Some(debug_renderer) = self.debug_renderer {
            debug_renderer.deinit(device);
        }
    }
}

// NB: If you add more VAOs here, be sure to deinitialize them in
// `Renderer::deinit()` below.
pub struct RendererVAOs {
    prim_vao: VAO,
    blur_vao: VAO,
    clip_rect_vao: VAO,
    clip_box_shadow_vao: VAO,
    clip_image_vao: VAO,
    border_vao: VAO,
    line_vao: VAO,
    scale_vao: VAO,
    gradient_vao: VAO,
    resolve_vao: VAO,
    svg_filter_vao: VAO,
    composite_vao: VAO,
    clear_vao: VAO,
}

/// Information about the state of the debugging / profiler overlay in native compositing mode.
struct DebugOverlayState {
    /// True if any of the current debug flags will result in drawing a debug overlay.
    is_enabled: bool,

    /// The current size of the debug overlay surface. None implies that the
    /// debug surface isn't currently allocated.
    current_size: Option<DeviceIntSize>,
}

impl DebugOverlayState {
    fn new() -> Self {
        DebugOverlayState {
            is_enabled: false,
            current_size: None,
        }
    }
}

pub struct VertexDataTextures {
    prim_header_f_texture: VertexDataTexture<PrimitiveHeaderF>,
    prim_header_i_texture: VertexDataTexture<PrimitiveHeaderI>,
    transforms_texture: VertexDataTexture<TransformData>,
    render_task_texture: VertexDataTexture<RenderTaskData>,
}

impl VertexDataTextures {
    fn new() -> Self {
        VertexDataTextures {
            prim_header_f_texture: VertexDataTexture::new(ImageFormat::RGBAF32),
            prim_header_i_texture: VertexDataTexture::new(ImageFormat::RGBAI32),
            transforms_texture: VertexDataTexture::new(ImageFormat::RGBAF32),
            render_task_texture: VertexDataTexture::new(ImageFormat::RGBAF32),
        }
    }

    fn update(
        &mut self,
        device: &mut Device,
        pbo_pool: &mut UploadPBOPool,
        frame: &mut Frame,
    ) {
        let mut texture_uploader = device.upload_texture(pbo_pool);
        self.prim_header_f_texture.update(
            device,
            &mut texture_uploader,
            &mut frame.prim_headers.headers_float,
        );
        self.prim_header_i_texture.update(
            device,
            &mut texture_uploader,
            &mut frame.prim_headers.headers_int,
        );
        self.transforms_texture.update(
            device,
            &mut texture_uploader,
            &mut frame.transform_palette,
        );
        self.render_task_texture.update(
            device,
            &mut texture_uploader,
            &mut frame.render_tasks.task_data,
        );

        // Flush and drop the texture uploader now, so that
        // we can borrow the textures to bind them.
        texture_uploader.flush(device);

        device.bind_texture(
            TextureSampler::PrimitiveHeadersF,
            &self.prim_header_f_texture.texture(),
            Swizzle::default(),
        );
        device.bind_texture(
            TextureSampler::PrimitiveHeadersI,
            &self.prim_header_i_texture.texture(),
            Swizzle::default(),
        );
        device.bind_texture(
            TextureSampler::TransformPalette,
            &self.transforms_texture.texture(),
            Swizzle::default(),
        );
        device.bind_texture(
            TextureSampler::RenderTasks,
            &self.render_task_texture.texture(),
            Swizzle::default(),
        );
    }

    fn size_in_bytes(&self) -> usize {
        self.prim_header_f_texture.size_in_bytes() +
        self.prim_header_i_texture.size_in_bytes() +
        self.transforms_texture.size_in_bytes() +
        self.render_task_texture.size_in_bytes()
    }

    fn deinit(
        self,
        device: &mut Device,
    ) {
        self.transforms_texture.deinit(device);
        self.prim_header_f_texture.deinit(device);
        self.prim_header_i_texture.deinit(device);
        self.render_task_texture.deinit(device);
    }
}

/// Tracks buffer damage rects over a series of frames.
#[derive(Debug)]
struct BufferDamageTracker {
    damage_rects: [DeviceRect; 2],
    current_offset: usize,
}

impl Default for BufferDamageTracker {
    fn default() -> Self {
        Self {
            damage_rects: [DeviceRect::default(); 2],
            current_offset: 0,
        }
     }
}

impl BufferDamageTracker {
    /// Sets the damage rect for the current frame. Should only be called *after*
    /// get_damage_rect() has been called to get the current backbuffer's damage rect.
    fn push_dirty_rect(&mut self, rect: &DeviceRect) {
        self.damage_rects[self.current_offset] = rect.clone();
        self.current_offset = match self.current_offset {
            0 => self.damage_rects.len() - 1,
            n => n - 1,
        }
    }

    /// Gets the damage rect for the current backbuffer, given the backbuffer's age.
    /// (The number of frames since it was previously the backbuffer.)
    /// Returns an empty rect if the buffer is valid, and None if the entire buffer is invalid.
    fn get_damage_rect(&self, buffer_age: usize) -> Option<DeviceRect> {
        match buffer_age {
            // 0 means this is a new buffer, so is completely invalid.
            0 => None,
            // 1 means this backbuffer was also the previous frame's backbuffer
            // (so must have been copied to the frontbuffer). It is therefore entirely valid.
            1 => Some(DeviceRect::zero()),
            // We must calculate the union of the damage rects since this buffer was previously
            // the backbuffer.
            n if n <= self.damage_rects.len() + 1 => {
                Some(
                    self.damage_rects.iter()
                        .cycle()
                        .skip(self.current_offset + 1)
                        .take(n - 1)
                        .fold(DeviceRect::zero(), |acc, r| acc.union(r))
                )
            }
            // The backbuffer is older than the number of frames for which we track,
            // so we treat it as entirely invalid.
            _ => None,
        }
    }
}

/// The renderer is responsible for submitting to the GPU the work prepared by the
/// RenderBackend.
///
/// We have a separate `Renderer` instance for each instance of WebRender (generally
/// one per OS window), and all instances share the same thread.
pub struct Renderer {
    result_rx: Receiver<ResultMsg>,
    debug_server: Box<dyn DebugServer>,
    pub device: Device,
    pending_texture_updates: Vec<TextureUpdateList>,
    /// True if there are any TextureCacheUpdate pending.
    pending_texture_cache_updates: bool,
    pending_native_surface_updates: Vec<NativeSurfaceOperation>,
    pending_gpu_cache_updates: Vec<GpuCacheUpdateList>,
    pending_gpu_cache_clear: bool,
    pending_shader_updates: Vec<PathBuf>,
    active_documents: FastHashMap<DocumentId, RenderedDocument>,

    shaders: Rc<RefCell<Shaders>>,

    max_recorded_profiles: usize,

    clear_color: Option<ColorF>,
    enable_clear_scissor: bool,
    enable_advanced_blend_barriers: bool,
    clear_caches_with_quads: bool,

    debug: LazyInitializedDebugRenderer,
    debug_flags: DebugFlags,
    profile: TransactionProfile,
    frame_counter: u64,
    resource_upload_time: f64,
    gpu_cache_upload_time: f64,
    profiler: Profiler,

    last_time: u64,

    pub gpu_profiler: GpuProfiler,
    vaos: RendererVAOs,

    gpu_cache_texture: GpuCacheTexture,
    vertex_data_textures: Vec<VertexDataTextures>,
    current_vertex_data_textures: usize,

    /// When the GPU cache debugger is enabled, we keep track of the live blocks
    /// in the GPU cache so that we can use them for the debug display. This
    /// member stores those live blocks, indexed by row.
    gpu_cache_debug_chunks: Vec<Vec<GpuCacheDebugChunk>>,

    gpu_cache_frame_id: FrameId,
    gpu_cache_overflow: bool,

    pipeline_info: PipelineInfo,

    // Manages and resolves source textures IDs to real texture IDs.
    texture_resolver: TextureResolver,

    texture_upload_pbo_pool: UploadPBOPool,

    dither_matrix_texture: Option<Texture>,

    /// Optional trait object that allows the client
    /// application to provide external buffers for image data.
    external_image_handler: Option<Box<dyn ExternalImageHandler>>,

    /// Optional function pointers for measuring memory used by a given
    /// heap-allocated pointer.
    size_of_ops: Option<MallocSizeOfOps>,

    pub renderer_errors: Vec<RendererError>,

    pub(in crate) async_frame_recorder: Option<AsyncScreenshotGrabber>,
    pub(in crate) async_screenshots: Option<AsyncScreenshotGrabber>,

    /// List of profile results from previous frames. Can be retrieved
    /// via get_frame_profiles().
    cpu_profiles: VecDeque<CpuProfile>,
    gpu_profiles: VecDeque<GpuProfile>,

    /// Notification requests to be fulfilled after rendering.
    notifications: Vec<NotificationRequest>,

    device_size: Option<DeviceIntSize>,

    /// A lazily created texture for the zoom debugging widget.
    zoom_debug_texture: Option<Texture>,

    /// The current mouse position. This is used for debugging
    /// functionality only, such as the debug zoom widget.
    cursor_position: DeviceIntPoint,

    /// Guards to check if we might be rendering a frame with expired texture
    /// cache entries.
    shared_texture_cache_cleared: bool,

    /// The set of documents which we've seen a publish for since last render.
    documents_seen: FastHashSet<DocumentId>,

    #[cfg(feature = "capture")]
    read_fbo: FBOId,
    #[cfg(feature = "replay")]
    owned_external_images: FastHashMap<(ExternalImageId, u8), ExternalTexture>,

    /// The compositing config, affecting how WR composites into the final scene.
    compositor_config: CompositorConfig,

    current_compositor_kind: CompositorKind,

    /// Maintains a set of allocated native composite surfaces. This allows any
    /// currently allocated surfaces to be cleaned up as soon as deinit() is
    /// called (the normal bookkeeping for native surfaces exists in the
    /// render backend thread).
    allocated_native_surfaces: FastHashSet<NativeSurfaceId>,

    /// If true, partial present state has been reset and everything needs to
    /// be drawn on the next render.
    force_redraw: bool,

    /// State related to the debug / profiling overlays
    debug_overlay_state: DebugOverlayState,

    /// Tracks the dirty rectangles from previous frames. Used on platforms
    /// that require keeping the front buffer fully correct when doing
    /// partial present (e.g. unix desktop with EGL_EXT_buffer_age).
    buffer_damage_tracker: BufferDamageTracker,

    max_primitive_instance_count: usize,
}

#[derive(Debug)]
pub enum RendererError {
    Shader(ShaderError),
    Thread(std::io::Error),
    Resource(ResourceCacheError),
    MaxTextureSize,
}

impl From<ShaderError> for RendererError {
    fn from(err: ShaderError) -> Self {
        RendererError::Shader(err)
    }
}

impl From<std::io::Error> for RendererError {
    fn from(err: std::io::Error) -> Self {
        RendererError::Thread(err)
    }
}

impl From<ResourceCacheError> for RendererError {
    fn from(err: ResourceCacheError) -> Self {
        RendererError::Resource(err)
    }
}

impl Renderer {
    /// Initializes WebRender and creates a `Renderer` and `RenderApiSender`.
    ///
    /// # Examples
    /// Initializes a `Renderer` with some reasonable values. For more information see
    /// [`RendererOptions`][rendereroptions].
    ///
    /// ```rust,ignore
    /// # use webrender::renderer::Renderer;
    /// # use std::path::PathBuf;
    /// let opts = webrender::RendererOptions {
    ///    device_pixel_ratio: 1.0,
    ///    resource_override_path: None,
    ///    enable_aa: false,
    /// };
    /// let (renderer, sender) = Renderer::new(opts);
    /// ```
    /// [rendereroptions]: struct.RendererOptions.html
    pub fn new(
        gl: Rc<dyn gl::Gl>,
        notifier: Box<dyn RenderNotifier>,
        mut options: RendererOptions,
        shaders: Option<&mut WrShaders>,
    ) -> Result<(Self, RenderApiSender), RendererError> {
        if !wr_has_been_initialized() {
            // If the profiler feature is enabled, try to load the profiler shared library
            // if the path was provided.
            #[cfg(feature = "profiler")]
            unsafe {
                if let Ok(ref tracy_path) = std::env::var("WR_TRACY_PATH") {
                    let ok = tracy_rs::load(tracy_path);
                    println!("Load tracy from {} -> {}", tracy_path, ok);
                }
            }

            register_thread_with_profiler("Compositor".to_owned());
        }

        HAS_BEEN_INITIALIZED.store(true, Ordering::SeqCst);

        let (api_tx, api_rx) = unbounded_channel();
        let (result_tx, result_rx) = unbounded_channel();
        let gl_type = gl.get_type();

        let debug_server = new_debug_server(options.start_debug_server, api_tx.clone());

        let mut device = Device::new(
            gl,
            options.crash_annotator.clone(),
            options.resource_override_path.clone(),
            options.use_optimized_shaders,
            options.upload_method.clone(),
            options.cached_programs.take(),
            options.allow_texture_storage_support,
            options.allow_texture_swizzling,
            options.dump_shader_source.take(),
            options.surface_origin_is_top_left,
            options.panic_on_gl_error,
        );

        let color_cache_formats = device.preferred_color_formats();
        let swizzle_settings = device.swizzle_settings();
        let use_dual_source_blending =
            device.get_capabilities().supports_dual_source_blending &&
            options.allow_dual_source_blending;
        let ext_blend_equation_advanced =
            options.allow_advanced_blend_equation &&
            device.get_capabilities().supports_advanced_blend_equation;
        let ext_blend_equation_advanced_coherent =
            device.supports_extension("GL_KHR_blend_equation_advanced_coherent");

        // 512 is the minimum that the texture cache can work with.
        const MIN_TEXTURE_SIZE: i32 = 512;
        if let Some(user_limit) = options.max_texture_size {
            assert!(user_limit >= MIN_TEXTURE_SIZE);
            device.clamp_max_texture_size(user_limit);
        }
        if device.max_texture_size() < MIN_TEXTURE_SIZE {
            // Broken GL contexts can return a max texture size of zero (See #1260).
            // Better to gracefully fail now than panic as soon as a texture is allocated.
            error!(
                "Device reporting insufficient max texture size ({})",
                device.max_texture_size()
            );
            return Err(RendererError::MaxTextureSize);
        }
        let max_texture_size = device.max_texture_size();

        device.begin_frame();

        let shaders = match shaders {
            Some(shaders) => Rc::clone(&shaders.shaders),
            None => Rc::new(RefCell::new(Shaders::new(&mut device, gl_type, &options)?)),
        };

        let dither_matrix_texture = if options.enable_dithering {
            let dither_matrix: [u8; 64] = [
                0,
                48,
                12,
                60,
                3,
                51,
                15,
                63,
                32,
                16,
                44,
                28,
                35,
                19,
                47,
                31,
                8,
                56,
                4,
                52,
                11,
                59,
                7,
                55,
                40,
                24,
                36,
                20,
                43,
                27,
                39,
                23,
                2,
                50,
                14,
                62,
                1,
                49,
                13,
                61,
                34,
                18,
                46,
                30,
                33,
                17,
                45,
                29,
                10,
                58,
                6,
                54,
                9,
                57,
                5,
                53,
                42,
                26,
                38,
                22,
                41,
                25,
                37,
                21,
            ];

            let texture = device.create_texture(
                ImageBufferKind::Texture2D,
                ImageFormat::R8,
                8,
                8,
                TextureFilter::Nearest,
                None,
                1,
            );
            device.upload_texture_immediate(&texture, &dither_matrix);

            Some(texture)
        } else {
            None
        };

        let x0 = 0.0;
        let y0 = 0.0;
        let x1 = 1.0;
        let y1 = 1.0;

        let quad_indices: [u16; 6] = [0, 1, 2, 2, 1, 3];
        let quad_vertices = [
            PackedVertex { pos: [x0, y0] },
            PackedVertex { pos: [x1, y0] },
            PackedVertex { pos: [x0, y1] },
            PackedVertex { pos: [x1, y1] },
        ];

        let prim_vao = device.create_vao(&desc::PRIM_INSTANCES);
        device.bind_vao(&prim_vao);
        device.update_vao_indices(&prim_vao, &quad_indices, VertexUsageHint::Static);
        device.update_vao_main_vertices(&prim_vao, &quad_vertices, VertexUsageHint::Static);

        let blur_vao = device.create_vao_with_new_instances(&desc::BLUR, &prim_vao);
        let clip_rect_vao = device.create_vao_with_new_instances(&desc::CLIP_RECT, &prim_vao);
        let clip_box_shadow_vao = device.create_vao_with_new_instances(&desc::CLIP_BOX_SHADOW, &prim_vao);
        let clip_image_vao = device.create_vao_with_new_instances(&desc::CLIP_IMAGE, &prim_vao);
        let border_vao = device.create_vao_with_new_instances(&desc::BORDER, &prim_vao);
        let scale_vao = device.create_vao_with_new_instances(&desc::SCALE, &prim_vao);
        let line_vao = device.create_vao_with_new_instances(&desc::LINE, &prim_vao);
        let gradient_vao = device.create_vao_with_new_instances(&desc::GRADIENT, &prim_vao);
        let resolve_vao = device.create_vao_with_new_instances(&desc::RESOLVE, &prim_vao);
        let svg_filter_vao = device.create_vao_with_new_instances(&desc::SVG_FILTER, &prim_vao);
        let composite_vao = device.create_vao_with_new_instances(&desc::COMPOSITE, &prim_vao);
        let clear_vao = device.create_vao_with_new_instances(&desc::CLEAR, &prim_vao);
        let texture_upload_pbo_pool = UploadPBOPool::new(&mut device, options.upload_pbo_default_size);

        let texture_resolver = TextureResolver::new(&mut device);

        let mut vertex_data_textures = Vec::new();
        for _ in 0 .. VERTEX_DATA_TEXTURE_COUNT {
            vertex_data_textures.push(VertexDataTextures::new());
        }

        // On some (mostly older, integrated) GPUs, the normal GPU texture cache update path
        // doesn't work well when running on ANGLE, causing CPU stalls inside D3D and/or the
        // GPU driver. See https://bugzilla.mozilla.org/show_bug.cgi?id=1576637 for much
        // more detail. To reduce the number of code paths we have active that require testing,
        // we will enable the GPU cache scatter update path on all devices running with ANGLE.
        // We want a better solution long-term, but for now this is a significant performance
        // improvement on HD4600 era GPUs, and shouldn't hurt performance in a noticeable
        // way on other systems running under ANGLE.
        let is_software = device.get_capabilities().renderer_name.starts_with("Software");

        // On other GL platforms, like macOS or Android, creating many PBOs is very inefficient.
        // This is what happens in GPU cache updates in PBO path. Instead, we switch everything
        // except software GL to use the GPU scattered updates.
        let supports_scatter = match gl_type {
            gl::GlType::Gl => true,
            gl::GlType::Gles => device.supports_extension("GL_EXT_color_buffer_float"),
        };

        let gpu_cache_texture = GpuCacheTexture::new(
            &mut device,
            supports_scatter && !is_software,
        )?;

        device.end_frame();

        let backend_notifier = notifier.clone();

        let prefer_subpixel_aa = options.force_subpixel_aa || (options.enable_subpixel_aa && use_dual_source_blending);
        let default_font_render_mode = match (options.enable_aa, prefer_subpixel_aa) {
            (true, true) => FontRenderMode::Subpixel,
            (true, false) => FontRenderMode::Alpha,
            (false, _) => FontRenderMode::Mono,
        };

        let compositor_kind = match options.compositor_config {
            CompositorConfig::Draw { max_partial_present_rects, draw_previous_partial_present_regions, .. } => {
                CompositorKind::Draw { max_partial_present_rects, draw_previous_partial_present_regions }
            }
            CompositorConfig::Native { ref compositor, max_update_rects, .. } => {
                let capabilities = compositor.get_capabilities();

                CompositorKind::Native {
                    max_update_rects,
                    virtual_surface_size: capabilities.virtual_surface_size,
                }
            }
        };

        let config = FrameBuilderConfig {
            default_font_render_mode,
            dual_source_blending_is_enabled: true,
            dual_source_blending_is_supported: use_dual_source_blending,
            chase_primitive: options.chase_primitive,
            testing: options.testing,
            gpu_supports_fast_clears: options.gpu_supports_fast_clears,
            gpu_supports_advanced_blend: ext_blend_equation_advanced,
            advanced_blend_is_coherent: ext_blend_equation_advanced_coherent,
            gpu_supports_render_target_partial_update: device.get_capabilities().supports_render_target_partial_update,
            batch_lookback_count: RendererOptions::BATCH_LOOKBACK_COUNT,
            background_color: options.clear_color,
            compositor_kind,
            tile_size_override: None,
            max_depth_ids: device.max_depth_ids(),
            max_target_size: max_texture_size,
        };
        info!("WR {:?}", config);

        let device_pixel_ratio = options.device_pixel_ratio;
        let debug_flags = options.debug_flags;
        let size_of_op = options.size_of_op;
        let enclosing_size_of_op = options.enclosing_size_of_op;
        let make_size_of_ops =
            move || size_of_op.map(|o| MallocSizeOfOps::new(o, enclosing_size_of_op));
        let thread_listener = Arc::new(options.thread_listener);
        let thread_listener_for_rayon_start = thread_listener.clone();
        let thread_listener_for_rayon_end = thread_listener.clone();
        let workers = options
            .workers
            .take()
            .unwrap_or_else(|| {
                let worker = ThreadPoolBuilder::new()
                    .thread_name(|idx|{ format!("WRWorker#{}", idx) })
                    .start_handler(move |idx| {
                        register_thread_with_profiler(format!("WRWorker#{}", idx));
                        if let Some(ref thread_listener) = *thread_listener_for_rayon_start {
                            thread_listener.thread_started(&format!("WRWorker#{}", idx));
                        }
                    })
                    .exit_handler(move |idx| {
                        if let Some(ref thread_listener) = *thread_listener_for_rayon_end {
                            thread_listener.thread_stopped(&format!("WRWorker#{}", idx));
                        }
                    })
                    .build();
                Arc::new(worker.unwrap())
            });
        let sampler = options.sampler;
        let namespace_alloc_by_client = options.namespace_alloc_by_client;

        let font_instances = SharedFontInstanceMap::new();

        let blob_image_handler = options.blob_image_handler.take();
        let thread_listener_for_render_backend = thread_listener.clone();
        let thread_listener_for_scene_builder = thread_listener.clone();
        let thread_listener_for_lp_scene_builder = thread_listener.clone();
        let scene_builder_hooks = options.scene_builder_hooks;
        let rb_thread_name = format!("WRRenderBackend#{}", options.renderer_id.unwrap_or(0));
        let scene_thread_name = format!("WRSceneBuilder#{}", options.renderer_id.unwrap_or(0));
        let lp_scene_thread_name = format!("WRSceneBuilderLP#{}", options.renderer_id.unwrap_or(0));
        let glyph_rasterizer = GlyphRasterizer::new(workers)?;

        let (scene_builder_channels, scene_tx) =
            SceneBuilderThreadChannels::new(api_tx.clone());

        let sb_font_instances = font_instances.clone();

        thread::Builder::new().name(scene_thread_name.clone()).spawn(move || {
            register_thread_with_profiler(scene_thread_name.clone());
            if let Some(ref thread_listener) = *thread_listener_for_scene_builder {
                thread_listener.thread_started(&scene_thread_name);
            }

            let mut scene_builder = SceneBuilderThread::new(
                config,
                device_pixel_ratio,
                sb_font_instances,
                make_size_of_ops(),
                scene_builder_hooks,
                scene_builder_channels,
            );
            scene_builder.run();

            if let Some(ref thread_listener) = *thread_listener_for_scene_builder {
                thread_listener.thread_stopped(&scene_thread_name);
            }
        })?;

        let low_priority_scene_tx = if options.support_low_priority_transactions {
            let (low_priority_scene_tx, low_priority_scene_rx) = unbounded_channel();
            let lp_builder = LowPrioritySceneBuilderThread {
                rx: low_priority_scene_rx,
                tx: scene_tx.clone(),
                simulate_slow_ms: 0,
            };

            thread::Builder::new().name(lp_scene_thread_name.clone()).spawn(move || {
                register_thread_with_profiler(lp_scene_thread_name.clone());
                if let Some(ref thread_listener) = *thread_listener_for_lp_scene_builder {
                    thread_listener.thread_started(&lp_scene_thread_name);
                }

                let mut scene_builder = lp_builder;
                scene_builder.run();

                if let Some(ref thread_listener) = *thread_listener_for_lp_scene_builder {
                    thread_listener.thread_stopped(&lp_scene_thread_name);
                }
            })?;

            low_priority_scene_tx
        } else {
            scene_tx.clone()
        };

        let backend_blob_handler = blob_image_handler
            .as_ref()
            .map(|handler| handler.create_similar());

        let texture_cache_config = options.texture_cache_config.clone();
        let mut picture_tile_size = options.picture_tile_size.unwrap_or(picture::TILE_SIZE_DEFAULT);
        // Clamp the picture tile size to reasonable values.
        picture_tile_size.width = picture_tile_size.width.max(128).min(4096);
        picture_tile_size.height = picture_tile_size.height.max(128).min(4096);

        let rb_scene_tx = scene_tx.clone();
        let rb_low_priority_scene_tx = scene_tx.clone();
        let rb_font_instances = font_instances.clone();
        let enable_multithreading = options.enable_multithreading;
        thread::Builder::new().name(rb_thread_name.clone()).spawn(move || {
            register_thread_with_profiler(rb_thread_name.clone());
            if let Some(ref thread_listener) = *thread_listener_for_render_backend {
                thread_listener.thread_started(&rb_thread_name);
            }

            let texture_cache = TextureCache::new(
                max_texture_size,
                picture_tile_size,
                color_cache_formats,
                swizzle_settings,
                &texture_cache_config,
            );

            let glyph_cache = GlyphCache::new();

            let mut resource_cache = ResourceCache::new(
                texture_cache,
                glyph_rasterizer,
                glyph_cache,
                rb_font_instances,
            );

            resource_cache.enable_multithreading(enable_multithreading);

            let mut backend = RenderBackend::new(
                api_rx,
                result_tx,
                rb_scene_tx,
                rb_low_priority_scene_tx,
                device_pixel_ratio,
                resource_cache,
                backend_notifier,
                backend_blob_handler,
                config,
                sampler,
                make_size_of_ops(),
                debug_flags,
                namespace_alloc_by_client,
            );
            backend.run();
            if let Some(ref thread_listener) = *thread_listener_for_render_backend {
                thread_listener.thread_stopped(&rb_thread_name);
            }
        })?;

        let debug_method = if !options.enable_gpu_markers {
            // The GPU markers are disabled.
            GpuDebugMethod::None
        } else if device.supports_extension("GL_KHR_debug") {
            GpuDebugMethod::KHR
        } else if device.supports_extension("GL_EXT_debug_marker") {
            GpuDebugMethod::MarkerEXT
        } else {
            println!("Warning: asking to enable_gpu_markers but no supporting extension was found");
            GpuDebugMethod::None
        };

        info!("using {:?}", debug_method);

        let gpu_profiler = GpuProfiler::new(Rc::clone(device.rc_gl()), debug_method);
        #[cfg(feature = "capture")]
        let read_fbo = device.create_fbo();

        let mut renderer = Renderer {
            result_rx,
            debug_server,
            device,
            active_documents: FastHashMap::default(),
            pending_texture_updates: Vec::new(),
            pending_texture_cache_updates: false,
            pending_native_surface_updates: Vec::new(),
            pending_gpu_cache_updates: Vec::new(),
            pending_gpu_cache_clear: false,
            pending_shader_updates: Vec::new(),
            shaders,
            debug: LazyInitializedDebugRenderer::new(),
            debug_flags: DebugFlags::empty(),
            profile: TransactionProfile::new(),
            frame_counter: 0,
            resource_upload_time: 0.0,
            gpu_cache_upload_time: 0.0,
            profiler: Profiler::new(),
            max_recorded_profiles: options.max_recorded_profiles,
            clear_color: options.clear_color,
            enable_clear_scissor: options.enable_clear_scissor,
            enable_advanced_blend_barriers: !ext_blend_equation_advanced_coherent,
            clear_caches_with_quads: options.clear_caches_with_quads,
            last_time: 0,
            gpu_profiler,
            vaos: RendererVAOs {
                prim_vao,
                blur_vao,
                clip_rect_vao,
                clip_box_shadow_vao,
                clip_image_vao,
                border_vao,
                scale_vao,
                gradient_vao,
                resolve_vao,
                line_vao,
                svg_filter_vao,
                composite_vao,
                clear_vao,
            },
            vertex_data_textures,
            current_vertex_data_textures: 0,
            pipeline_info: PipelineInfo::default(),
            dither_matrix_texture,
            external_image_handler: None,
            size_of_ops: make_size_of_ops(),
            cpu_profiles: VecDeque::new(),
            gpu_profiles: VecDeque::new(),
            gpu_cache_texture,
            gpu_cache_debug_chunks: Vec::new(),
            gpu_cache_frame_id: FrameId::INVALID,
            gpu_cache_overflow: false,
            texture_upload_pbo_pool,
            texture_resolver,
            renderer_errors: Vec::new(),
            async_frame_recorder: None,
            async_screenshots: None,
            #[cfg(feature = "capture")]
            read_fbo,
            #[cfg(feature = "replay")]
            owned_external_images: FastHashMap::default(),
            notifications: Vec::new(),
            device_size: None,
            zoom_debug_texture: None,
            cursor_position: DeviceIntPoint::zero(),
            shared_texture_cache_cleared: false,
            documents_seen: FastHashSet::default(),
            force_redraw: true,
            compositor_config: options.compositor_config,
            current_compositor_kind: compositor_kind,
            allocated_native_surfaces: FastHashSet::default(),
            debug_overlay_state: DebugOverlayState::new(),
            buffer_damage_tracker: BufferDamageTracker::default(),
            max_primitive_instance_count:
                RendererOptions::MAX_INSTANCE_BUFFER_SIZE / mem::size_of::<PrimitiveInstanceData>(),
        };

        // We initially set the flags to default and then now call set_debug_flags
        // to ensure any potential transition when enabling a flag is run.
        renderer.set_debug_flags(debug_flags);

        let sender = RenderApiSender::new(
            api_tx,
            scene_tx,
            low_priority_scene_tx,
            blob_image_handler,
            font_instances,
        );
        Ok((renderer, sender))
    }

    pub fn device_size(&self) -> Option<DeviceIntSize> {
        self.device_size
    }

    /// Update the current position of the debug cursor.
    pub fn set_cursor_position(
        &mut self,
        position: DeviceIntPoint,
    ) {
        self.cursor_position = position;
    }

    pub fn get_max_texture_size(&self) -> i32 {
        self.device.max_texture_size()
    }

    pub fn get_graphics_api_info(&self) -> GraphicsApiInfo {
        GraphicsApiInfo {
            kind: GraphicsApi::OpenGL,
            version: self.device.gl().get_string(gl::VERSION),
            renderer: self.device.gl().get_string(gl::RENDERER),
        }
    }

    pub fn preferred_color_format(&self) -> ImageFormat {
        self.device.preferred_color_formats().external
    }

    pub fn optimal_texture_stride_alignment(&self, format: ImageFormat) -> usize {
        self.device.optimal_pbo_stride().num_bytes(format).get()
    }

    pub fn set_clear_color(&mut self, color: Option<ColorF>) {
        self.clear_color = color;
    }

    pub fn flush_pipeline_info(&mut self) -> PipelineInfo {
        mem::replace(&mut self.pipeline_info, PipelineInfo::default())
    }

    /// Returns the Epoch of the current frame in a pipeline.
    pub fn current_epoch(&self, document_id: DocumentId, pipeline_id: PipelineId) -> Option<Epoch> {
        self.pipeline_info.epochs.get(&(pipeline_id, document_id)).cloned()
    }

    /// Processes the result queue.
    ///
    /// Should be called before `render()`, as texture cache updates are done here.
    pub fn update(&mut self) {
        profile_scope!("update");

        // Pull any pending results and return the most recent.
        while let Ok(msg) = self.result_rx.try_recv() {
            match msg {
                ResultMsg::PublishPipelineInfo(mut pipeline_info) => {
                    for ((pipeline_id, document_id), epoch) in pipeline_info.epochs {
                        self.pipeline_info.epochs.insert((pipeline_id, document_id), epoch);
                    }
                    self.pipeline_info.removed_pipelines.extend(pipeline_info.removed_pipelines.drain(..));
                }
                ResultMsg::PublishDocument(
                    document_id,
                    mut doc,
                    resource_update_list,
                ) => {
                    // Add a new document to the active set

                    // If the document we are replacing must be drawn (in order to
                    // update the texture cache), issue a render just to
                    // off-screen targets, ie pass None to render_impl. We do this
                    // because a) we don't need to render to the main framebuffer
                    // so it is cheaper not to, and b) doing so without a
                    // subsequent present would break partial present.
                    if let Some(mut prev_doc) = self.active_documents.remove(&document_id) {
                        doc.profile.merge(&mut prev_doc.profile);

                        if prev_doc.frame.must_be_drawn() {
                            self.render_impl(
                                document_id,
                                &mut prev_doc,
                                None,
                                0,
                            ).ok();
                        }
                    }

                    self.active_documents.insert(document_id, doc);

                    // IMPORTANT: The pending texture cache updates must be applied
                    //            *after* the previous frame has been rendered above
                    //            (if neceessary for a texture cache update). For
                    //            an example of why this is required:
                    //            1) Previous frame contains a render task that
                    //               targets Texture X.
                    //            2) New frame contains a texture cache update which
                    //               frees Texture X.
                    //            3) bad stuff happens.

                    //TODO: associate `document_id` with target window
                    self.pending_texture_cache_updates |= !resource_update_list.texture_updates.updates.is_empty();
                    self.pending_texture_updates.push(resource_update_list.texture_updates);
                    self.pending_native_surface_updates.extend(resource_update_list.native_surface_updates);
                    self.documents_seen.insert(document_id);
                }
                ResultMsg::UpdateGpuCache(mut list) => {
                    if list.clear {
                        self.pending_gpu_cache_clear = true;
                    }
                    if list.clear {
                        self.gpu_cache_debug_chunks = Vec::new();
                    }
                    for cmd in mem::replace(&mut list.debug_commands, Vec::new()) {
                        match cmd {
                            GpuCacheDebugCmd::Alloc(chunk) => {
                                let row = chunk.address.v as usize;
                                if row >= self.gpu_cache_debug_chunks.len() {
                                    self.gpu_cache_debug_chunks.resize(row + 1, Vec::new());
                                }
                                self.gpu_cache_debug_chunks[row].push(chunk);
                            },
                            GpuCacheDebugCmd::Free(address) => {
                                let chunks = &mut self.gpu_cache_debug_chunks[address.v as usize];
                                let pos = chunks.iter()
                                    .position(|x| x.address == address).unwrap();
                                chunks.remove(pos);
                            },
                        }
                    }
                    self.pending_gpu_cache_updates.push(list);
                }
                ResultMsg::UpdateResources {
                    resource_updates,
                    memory_pressure,
                } => {
                    if memory_pressure {
                        // If a memory pressure event arrives _after_ a new scene has
                        // been published that writes persistent targets (i.e. cached
                        // render tasks to the texture cache, or picture cache tiles)
                        // but _before_ the next update/render loop, those targets
                        // will not be updated due to the active_documents list being
                        // cleared at the end of this message. To work around that,
                        // if any of the existing documents have not rendered yet, and
                        // have picture/texture cache targets, force a render so that
                        // those targets are updated.
                        let active_documents = mem::replace(
                            &mut self.active_documents,
                            FastHashMap::default(),
                        );
                        for (doc_id, mut doc) in active_documents {
                            if doc.frame.must_be_drawn() {
                                // As this render will not be presented, we must pass None to
                                // render_impl. This avoids interfering with partial present
                                // logic, as well as being more efficient.
                                self.render_impl(
                                    doc_id,
                                    &mut doc,
                                    None,
                                    0,
                                ).ok();
                            }
                        }
                    }

                    self.pending_texture_cache_updates |= !resource_updates.texture_updates.updates.is_empty();
                    self.pending_texture_updates.push(resource_updates.texture_updates);
                    self.pending_native_surface_updates.extend(resource_updates.native_surface_updates);
                    self.device.begin_frame();

                    self.update_texture_cache();
                    self.update_native_surfaces();

                    // Flush the render target pool on memory pressure.
                    //
                    // This needs to be separate from the block below because
                    // the device module asserts if we delete textures while
                    // not in a frame.
                    if memory_pressure {
                        self.texture_upload_pbo_pool.on_memory_pressure(&mut self.device);
                    }

                    self.device.end_frame();
                }
                ResultMsg::AppendNotificationRequests(mut notifications) => {
                    // We need to know specifically if there are any pending
                    // TextureCacheUpdate updates in any of the entries in
                    // pending_texture_updates. They may simply be nops, which do not
                    // need to prevent issuing the notification, and if so, may not
                    // cause a timely frame render to occur to wake up any listeners.
                    if !self.pending_texture_cache_updates {
                        drain_filter(
                            &mut notifications,
                            |n| { n.when() == Checkpoint::FrameTexturesUpdated },
                            |n| { n.notify(); },
                        );
                    }
                    self.notifications.append(&mut notifications);
                }
                ResultMsg::ForceRedraw => {
                    self.force_redraw = true;
                }
                ResultMsg::RefreshShader(path) => {
                    self.pending_shader_updates.push(path);
                }
                ResultMsg::DebugOutput(output) => match output {
                    DebugOutput::FetchDocuments(string) |
                    DebugOutput::FetchClipScrollTree(string) => {
                        self.debug_server.send(string);
                    }
                    #[cfg(feature = "capture")]
                    DebugOutput::SaveCapture(config, deferred) => {
                        self.save_capture(config, deferred);
                    }
                    #[cfg(feature = "replay")]
                    DebugOutput::LoadCapture(config, plain_externals) => {
                        self.active_documents.clear();
                        self.load_capture(config, plain_externals);
                    }
                },
                ResultMsg::DebugCommand(command) => {
                    self.handle_debug_command(command);
                }
            }
        }
    }

    #[cfg(not(feature = "debugger"))]
    fn get_screenshot_for_debugger(&mut self) -> String {
        // Avoid unused param warning.
        let _ = &self.debug_server;
        String::new()
    }

    #[cfg(feature = "debugger")]
    fn get_screenshot_for_debugger(&mut self) -> String {
        use api::{ImageDescriptor, ImageDescriptorFlags};

        let desc = ImageDescriptor::new(1024, 768, ImageFormat::BGRA8, ImageDescriptorFlags::IS_OPAQUE);
        let data = self.device.read_pixels(&desc);
        let screenshot = debug_server::Screenshot::new(desc.size, data);

        serde_json::to_string(&screenshot).unwrap()
    }

    #[cfg(not(feature = "debugger"))]
    fn get_passes_for_debugger(&self) -> String {
        // Avoid unused param warning.
        let _ = &self.debug_server;
        String::new()
    }

    #[cfg(feature = "debugger")]
    fn debug_alpha_target(target: &AlphaRenderTarget) -> debug_server::Target {
        let mut debug_target = debug_server::Target::new("A8");

        debug_target.add(
            debug_server::BatchKind::Cache,
            "Scalings",
            target.scalings.len(),
        );
        debug_target.add(
            debug_server::BatchKind::Cache,
            "Zero Clears",
            target.zero_clears.len(),
        );
        debug_target.add(
            debug_server::BatchKind::Cache,
            "One Clears",
            target.one_clears.len(),
        );
        debug_target.add(
            debug_server::BatchKind::Clip,
            "BoxShadows [p]",
            target.clip_batcher.primary_clips.box_shadows.len(),
        );
        debug_target.add(
            debug_server::BatchKind::Clip,
            "BoxShadows [s]",
            target.clip_batcher.secondary_clips.box_shadows.len(),
        );
        debug_target.add(
            debug_server::BatchKind::Cache,
            "Vertical Blur",
            target.vertical_blurs.len(),
        );
        debug_target.add(
            debug_server::BatchKind::Cache,
            "Horizontal Blur",
            target.horizontal_blurs.len(),
        );
        debug_target.add(
            debug_server::BatchKind::Clip,
            "Slow Rectangles [p]",
            target.clip_batcher.primary_clips.slow_rectangles.len(),
        );
        debug_target.add(
            debug_server::BatchKind::Clip,
            "Fast Rectangles [p]",
            target.clip_batcher.primary_clips.fast_rectangles.len(),
        );
        debug_target.add(
            debug_server::BatchKind::Clip,
            "Slow Rectangles [s]",
            target.clip_batcher.secondary_clips.slow_rectangles.len(),
        );
        debug_target.add(
            debug_server::BatchKind::Clip,
            "Fast Rectangles [s]",
            target.clip_batcher.secondary_clips.fast_rectangles.len(),
        );
        for (_, items) in target.clip_batcher.primary_clips.images.iter() {
            debug_target.add(debug_server::BatchKind::Clip, "Image mask [p]", items.len());
        }
        for (_, items) in target.clip_batcher.secondary_clips.images.iter() {
            debug_target.add(debug_server::BatchKind::Clip, "Image mask [s]", items.len());
        }

        debug_target
    }

    #[cfg(feature = "debugger")]
    fn debug_color_target(target: &ColorRenderTarget) -> debug_server::Target {
        let mut debug_target = debug_server::Target::new("RGBA8");

        debug_target.add(
            debug_server::BatchKind::Cache,
            "Scalings",
            target.scalings.len(),
        );
        debug_target.add(
            debug_server::BatchKind::Cache,
            "Vertical Blur",
            target.vertical_blurs.len(),
        );
        debug_target.add(
            debug_server::BatchKind::Cache,
            "Horizontal Blur",
            target.horizontal_blurs.len(),
        );
        debug_target.add(
            debug_server::BatchKind::Cache,
            "SVG Filters",
            target.svg_filters.iter().map(|(_, batch)| batch.len()).sum(),
        );

        for alpha_batch_container in &target.alpha_batch_containers {
            for batch in alpha_batch_container.opaque_batches.iter().rev() {
                debug_target.add(
                    debug_server::BatchKind::Opaque,
                    batch.key.kind.debug_name(),
                    batch.instances.len(),
                );
            }

            for batch in &alpha_batch_container.alpha_batches {
                debug_target.add(
                    debug_server::BatchKind::Alpha,
                    batch.key.kind.debug_name(),
                    batch.instances.len(),
                );
            }
        }

        debug_target
    }

    #[cfg(feature = "debugger")]
    fn debug_texture_cache_target(target: &TextureCacheRenderTarget) -> debug_server::Target {
        let mut debug_target = debug_server::Target::new("Texture Cache");

        debug_target.add(
            debug_server::BatchKind::Cache,
            "Horizontal Blur",
            target.horizontal_blurs.len(),
        );

        debug_target
    }

    #[cfg(feature = "debugger")]
    fn get_passes_for_debugger(&self) -> String {
        let mut debug_passes = debug_server::PassList::new();

        for (_, render_doc) in &self.active_documents {
            for pass in &render_doc.frame.passes {
                let mut debug_targets = Vec::new();
                debug_targets.extend(pass.alpha.targets.iter().map(Self::debug_alpha_target));
                debug_targets.extend(pass.color.targets.iter().map(Self::debug_color_target));
                debug_targets.extend(pass.texture_cache.iter().map(|(_, target)| Self::debug_texture_cache_target(target)));

                debug_passes.add(debug_server::Pass { targets: debug_targets });
            }
        }

        serde_json::to_string(&debug_passes).unwrap()
    }

    #[cfg(not(feature = "debugger"))]
    fn get_render_tasks_for_debugger(&self) -> String {
        String::new()
    }

    #[cfg(feature = "debugger")]
    fn get_render_tasks_for_debugger(&self) -> String {
        let mut debug_root = debug_server::RenderTaskList::new();

        for (_, render_doc) in &self.active_documents {
            let debug_node = debug_server::TreeNode::new("document render tasks");
            let mut builder = debug_server::TreeNodeBuilder::new(debug_node);

            let render_tasks = &render_doc.frame.render_tasks;
            match render_tasks.tasks.first() {
                Some(main_task) => main_task.print_with(&mut builder, render_tasks),
                None => continue,
            };

            debug_root.add(builder.build());
        }

        serde_json::to_string(&debug_root).unwrap()
    }

    fn handle_debug_command(&mut self, command: DebugCommand) {
        match command {
            DebugCommand::EnableDualSourceBlending(_) |
            DebugCommand::SetPictureTileSize(_) => {
                panic!("Should be handled by render backend");
            }
            DebugCommand::FetchDocuments |
            DebugCommand::FetchClipScrollTree => {}
            DebugCommand::FetchRenderTasks => {
                let json = self.get_render_tasks_for_debugger();
                self.debug_server.send(json);
            }
            DebugCommand::FetchPasses => {
                let json = self.get_passes_for_debugger();
                self.debug_server.send(json);
            }
            DebugCommand::FetchScreenshot => {
                let json = self.get_screenshot_for_debugger();
                self.debug_server.send(json);
            }
            DebugCommand::SaveCapture(..) |
            DebugCommand::LoadCapture(..) |
            DebugCommand::StartCaptureSequence(..) |
            DebugCommand::StopCaptureSequence => {
                panic!("Capture commands are not welcome here! Did you build with 'capture' feature?")
            }
            DebugCommand::ClearCaches(_)
            | DebugCommand::SimulateLongSceneBuild(_)
            | DebugCommand::SimulateLongLowPrioritySceneBuild(_)
            | DebugCommand::EnableNativeCompositor(_)
            | DebugCommand::SetBatchingLookback(_)
            | DebugCommand::EnableMultithreading(_) => {}
            DebugCommand::InvalidateGpuCache => {
                match self.gpu_cache_texture.bus {
                    GpuCacheBus::PixelBuffer { ref mut rows, .. } => {
                        info!("Invalidating GPU caches");
                        for row in rows {
                            row.add_dirty(0, MAX_VERTEX_TEXTURE_WIDTH);
                        }
                    }
                    GpuCacheBus::Scatter { .. } => {
                        warn!("Unable to invalidate scattered GPU cache");
                    }
                }
            }
            DebugCommand::SetFlags(flags) => {
                self.set_debug_flags(flags);
            }
        }
    }

    /// Set a callback for handling external images.
    pub fn set_external_image_handler(&mut self, handler: Box<dyn ExternalImageHandler>) {
        self.external_image_handler = Some(handler);
    }

    /// Retrieve (and clear) the current list of recorded frame profiles.
    pub fn get_frame_profiles(&mut self) -> (Vec<CpuProfile>, Vec<GpuProfile>) {
        let cpu_profiles = self.cpu_profiles.drain(..).collect();
        let gpu_profiles = self.gpu_profiles.drain(..).collect();
        (cpu_profiles, gpu_profiles)
    }

    /// Reset the current partial present state. This forces the entire framebuffer
    /// to be refreshed next time `render` is called.
    pub fn force_redraw(&mut self) {
        self.force_redraw = true;
    }

    /// Renders the current frame.
    ///
    /// A Frame is supplied by calling [`generate_frame()`][webrender_api::Transaction::generate_frame].
    /// buffer_age is the age of the current backbuffer. It is only relevant if partial present
    /// is active, otherwise 0 should be passed here.
    pub fn render(
        &mut self,
        device_size: DeviceIntSize,
        buffer_age: usize,
    ) -> Result<RenderResults, Vec<RendererError>> {
        self.device_size = Some(device_size);

        // TODO(gw): We want to make the active document that is
        //           being rendered configurable via the public
        //           API in future. For now, just select the last
        //           added document as the active one to render
        //           (Gecko only ever creates a single document
        //           per renderer right now).
        let doc_id = self.active_documents.keys().last().cloned();

        let result = match doc_id {
            Some(doc_id) => {
                // Remove the doc from the map to appease the borrow checker
                let mut doc = self.active_documents
                    .remove(&doc_id)
                    .unwrap();

                let result = self.render_impl(
                    doc_id,
                    &mut doc,
                    Some(device_size),
                    buffer_age,
                );

                self.active_documents.insert(doc_id, doc);

                result
            }
            None => {
                self.last_time = precise_time_ns();
                Ok(RenderResults::default())
            }
        };

        drain_filter(
            &mut self.notifications,
            |n| { n.when() == Checkpoint::FrameRendered },
            |n| { n.notify(); },
        );

        // This is the end of the rendering pipeline. If some notifications are is still there,
        // just clear them and they will autimatically fire the Checkpoint::TransactionDropped
        // event. Otherwise they would just pile up in this vector forever.
        self.notifications.clear();

        tracy_frame_marker!();

        result
    }

    /// Update the state of any debug / profiler overlays. This is currently only needed
    /// when running with the native compositor enabled.
    fn update_debug_overlay(&mut self, framebuffer_size: DeviceIntSize) {
        // If any of the following debug flags are set, something will be drawn on the debug overlay.
        self.debug_overlay_state.is_enabled = self.debug_flags.intersects(
            DebugFlags::PROFILER_DBG |
            DebugFlags::RENDER_TARGET_DBG |
            DebugFlags::TEXTURE_CACHE_DBG |
            DebugFlags::EPOCHS |
            DebugFlags::GPU_CACHE_DBG |
            DebugFlags::PICTURE_CACHING_DBG |
            DebugFlags::PRIMITIVE_DBG |
            DebugFlags::ZOOM_DBG
        );

        // Update the debug overlay surface, if we are running in native compositor mode.
        if let CompositorKind::Native { .. } = self.current_compositor_kind {
            let compositor = self.compositor_config.compositor().unwrap();

            // If there is a current surface, destroy it if we don't need it for this frame, or if
            // the size has changed.
            if let Some(current_size) = self.debug_overlay_state.current_size {
                if !self.debug_overlay_state.is_enabled || current_size != framebuffer_size {
                    compositor.destroy_surface(NativeSurfaceId::DEBUG_OVERLAY);
                    self.debug_overlay_state.current_size = None;
                }
            }

            // Allocate a new surface, if we need it and there isn't one.
            if self.debug_overlay_state.is_enabled && self.debug_overlay_state.current_size.is_none() {
                compositor.create_surface(
                    NativeSurfaceId::DEBUG_OVERLAY,
                    DeviceIntPoint::zero(),
                    framebuffer_size,
                    false,
                );
                compositor.create_tile(
                    NativeTileId::DEBUG_OVERLAY,
                );
                self.debug_overlay_state.current_size = Some(framebuffer_size);
            }
        }
    }

    /// Bind a draw target for the debug / profiler overlays, if required.
    fn bind_debug_overlay(&mut self) {
        // Debug overlay setup are only required in native compositing mode
        if self.debug_overlay_state.is_enabled {
            if let CompositorKind::Native { .. } = self.current_compositor_kind {
                let compositor = self.compositor_config.compositor().unwrap();
                let surface_size = self.debug_overlay_state.current_size.unwrap();

                // Bind the native surface
                let surface_info = compositor.bind(
                    NativeTileId::DEBUG_OVERLAY,
                    DeviceIntRect::new(
                        DeviceIntPoint::zero(),
                        surface_size,
                    ),
                    DeviceIntRect::new(
                        DeviceIntPoint::zero(),
                        surface_size,
                    ),
                );

                // Bind the native surface to current FBO target
                let draw_target = DrawTarget::NativeSurface {
                    offset: surface_info.origin,
                    external_fbo_id: surface_info.fbo_id,
                    dimensions: surface_size,
                };
                self.device.bind_draw_target(draw_target);

                // When native compositing, clear the debug overlay each frame.
                self.device.clear_target(
                    Some([0.0, 0.0, 0.0, 0.0]),
                    None, // debug renderer does not use depth
                    None,
                );
            }
        }
    }

    /// Unbind the draw target for debug / profiler overlays, if required.
    fn unbind_debug_overlay(&mut self) {
        // Debug overlay setup are only required in native compositing mode
        if self.debug_overlay_state.is_enabled {
            if let CompositorKind::Native { .. } = self.current_compositor_kind {
                let compositor = self.compositor_config.compositor().unwrap();
                // Unbind the draw target and add it to the visual tree to be composited
                compositor.unbind();

                compositor.add_surface(
                    NativeSurfaceId::DEBUG_OVERLAY,
                    CompositorSurfaceTransform::identity(),
                    DeviceIntRect::new(
                        DeviceIntPoint::zero(),
                        self.debug_overlay_state.current_size.unwrap(),
                    ),
                    ImageRendering::Auto,
                );
            }
        }
    }

    // If device_size is None, don't render to the main frame buffer. This is useful to
    // update texture cache render tasks but avoid doing a full frame render. If the
    // render is not going to be presented, then this must be set to None, as performing a
    // composite without a present will confuse partial present.
    fn render_impl(
        &mut self,
        doc_id: DocumentId,
        active_doc: &mut RenderedDocument,
        device_size: Option<DeviceIntSize>,
        buffer_age: usize,
    ) -> Result<RenderResults, Vec<RendererError>> {
        profile_scope!("render");
        let mut results = RenderResults::default();
        self.profile.start_time(profiler::RENDERER_TIME);

        let compositor_kind = active_doc.frame.composite_state.compositor_kind;
        // CompositorKind is updated
        if self.current_compositor_kind != compositor_kind {
            let enable = match (self.current_compositor_kind, compositor_kind) {
                (CompositorKind::Native { .. }, CompositorKind::Draw { .. }) => {
                    if self.debug_overlay_state.current_size.is_some() {
                        self.compositor_config
                            .compositor()
                            .unwrap()
                            .destroy_surface(NativeSurfaceId::DEBUG_OVERLAY);
                        self.debug_overlay_state.current_size = None;
                    }
                    false
                }
                (CompositorKind::Draw { .. }, CompositorKind::Native { .. }) => {
                    true
                }
                (current_compositor_kind, active_doc_compositor_kind) => {
                    dbg!(current_compositor_kind, active_doc_compositor_kind);
                    unreachable!();
                }
            };

            self.compositor_config
                .compositor()
                .unwrap()
                .enable_native_compositor(enable);
            self.current_compositor_kind = compositor_kind;
        }

        // The texture resolver scope should be outside of any rendering, including
        // debug rendering. This ensures that when we return render targets to the
        // pool via glInvalidateFramebuffer, we don't do any debug rendering after
        // that point. Otherwise, the bind / invalidate / bind logic trips up the
        // render pass logic in tiled / mobile GPUs, resulting in an extra copy /
        // resolve step when the debug overlay is enabled.
        self.texture_resolver.begin_frame();

        if let Some(device_size) = device_size {
            self.update_gpu_profile(device_size);
        }

        let cpu_frame_id = {
            let _gm = self.gpu_profiler.start_marker("begin frame");
            let frame_id = self.device.begin_frame();
            self.gpu_profiler.begin_frame(frame_id);

            self.device.disable_scissor();
            self.device.disable_depth();
            self.set_blend(false, FramebufferKind::Main);
            //self.update_shaders();

            self.update_texture_cache();
            self.update_native_surfaces();

            frame_id
        };

        // Inform the client that we are starting a composition transaction if native
        // compositing is enabled. This needs to be done early in the frame, so that
        // we can create debug overlays after drawing the main surfaces.
        if let CompositorKind::Native { .. } = self.current_compositor_kind {
            let compositor = self.compositor_config.compositor().unwrap();
            compositor.begin_frame();
        }

        if let Some(device_size) = device_size {
            // Update the state of the debug overlay surface, ensuring that
            // the compositor mode has a suitable surface to draw to, if required.
            self.update_debug_overlay(device_size);
        }

        let frame = &mut active_doc.frame;
        let profile = &mut active_doc.profile;
        assert!(self.current_compositor_kind == frame.composite_state.compositor_kind);

        if self.shared_texture_cache_cleared {
            assert!(self.documents_seen.contains(&doc_id),
                    "Cleared texture cache without sending new document frame.");
        }

        match self.prepare_gpu_cache(&frame.deferred_resolves) {
            Ok(..) => {
                assert!(frame.gpu_cache_frame_id <= self.gpu_cache_frame_id,
                    "Received frame depends on a later GPU cache epoch ({:?}) than one we received last via `UpdateGpuCache` ({:?})",
                    frame.gpu_cache_frame_id, self.gpu_cache_frame_id);

                {
                    profile_scope!("gl.flush");
                    self.device.gl().flush();  // early start on gpu cache updates
                }

                self.draw_frame(
                    frame,
                    device_size,
                    buffer_age,
                    &mut results,
                );

                // TODO(nical): do this automatically by selecting counters in the wr profiler
                // Profile marker for the number of invalidated picture cache
                if thread_is_being_profiled() {
                    let duration = Duration::new(0,0);
                    if let Some(n) = self.profiler.get(profiler::RENDERED_PICTURE_TILES) {
                        let message = (n as usize).to_string();
                        add_text_marker(cstr!("NumPictureCacheInvalidated"), &message, duration);
                    }
                }

                if device_size.is_some() {
                    self.draw_frame_debug_items(&frame.debug_items);
                }

                self.profile.merge(profile);
            }
            Err(e) => {
                self.renderer_errors.push(e);
            }
        }

        self.unlock_external_images(&frame.deferred_resolves);

        let _gm = self.gpu_profiler.start_marker("end frame");
        self.gpu_profiler.end_frame();

        if let Some(device_size) = device_size {
            // Bind a surface to draw the debug / profiler information to.
            self.bind_debug_overlay();

            self.draw_render_target_debug(device_size);
            self.draw_texture_cache_debug(device_size);
            self.draw_gpu_cache_debug(device_size);
            self.draw_zoom_debug(device_size);
            self.draw_epoch_debug();
        }

        self.profile.end_time(profiler::RENDERER_TIME);
        self.profile.end_time_if_started(profiler::TOTAL_FRAME_CPU_TIME);

        let current_time = precise_time_ns();
        if device_size.is_some() {
            let time = profiler::ns_to_ms(current_time - self.last_time);
            self.profile.set(profiler::FRAME_TIME, time);
        }

        if self.max_recorded_profiles > 0 {
            while self.cpu_profiles.len() >= self.max_recorded_profiles {
                self.cpu_profiles.pop_front();
            }
            let cpu_profile = CpuProfile::new(
                cpu_frame_id,
                (self.profile.get_or(profiler::FRAME_BUILDING_TIME, 0.0) * 1000000.0) as u64,
                (self.profile.get_or(profiler::RENDERER_TIME, 0.0) * 1000000.0) as u64,
                self.profile.get_or(profiler::DRAW_CALLS, 0.0) as usize,
            );
            self.cpu_profiles.push_back(cpu_profile);
        }

        if thread_is_being_profiled() {
            let duration = Duration::new(0,0);
            let message = (self.profile.get_or(profiler::DRAW_CALLS, 0.0) as usize).to_string();
            add_text_marker(cstr!("NumDrawCalls"), &message, duration);
        }

        results.stats.texture_upload_mb = self.profile.get_or(profiler::TEXTURE_UPLOADS_MEM, 0.0);
        self.frame_counter += 1;
        results.stats.resource_upload_time = self.resource_upload_time;
        self.resource_upload_time = 0.0;
        results.stats.gpu_cache_upload_time = self.gpu_cache_upload_time;
        self.gpu_cache_upload_time = 0.0;

        // Note: this clears the values in self.profile.
        self.profiler.set_counters(&mut self.profile);

        // Note: profile counters must be set before this or they will count for next frame.
        self.profiler.update();

        if self.debug_flags.intersects(DebugFlags::PROFILER_DBG | DebugFlags::PROFILER_CAPTURE) {
            if let Some(device_size) = device_size {
                //TODO: take device/pixel ratio into equation?
                if let Some(debug_renderer) = self.debug.get_mut(&mut self.device) {
                    self.profiler.draw_profile(
                        self.frame_counter,
                        debug_renderer,
                        device_size,
                    );
                }
            }
        }

        if self.debug_flags.contains(DebugFlags::ECHO_DRIVER_MESSAGES) {
            self.device.echo_driver_messages();
        }

        if let Some(debug_renderer) = self.debug.try_get_mut() {
            let small_screen = self.debug_flags.contains(DebugFlags::SMALL_SCREEN);
            let scale = if small_screen { 1.6 } else { 1.0 };
            // TODO(gw): Tidy this up so that compositor config integrates better
            //           with the (non-compositor) surface y-flip options.
            let surface_origin_is_top_left = match self.current_compositor_kind {
                CompositorKind::Native { .. } => true,
                CompositorKind::Draw { .. } => self.device.surface_origin_is_top_left(),
            };
            debug_renderer.render(
                &mut self.device,
                device_size,
                scale,
                surface_origin_is_top_left,
            );
        }
        self.texture_upload_pbo_pool.end_frame(&mut self.device);
        self.device.end_frame();

        if device_size.is_some() {
            self.last_time = current_time;

            // Unbind the target for the debug overlay. No debug or profiler drawing
            // can occur afer this point.
            self.unbind_debug_overlay();
        }

        // Inform the client that we are finished this composition transaction if native
        // compositing is enabled. This must be called after any debug / profiling compositor
        // surfaces have been drawn and added to the visual tree.
        if let CompositorKind::Native { .. } = self.current_compositor_kind {
            profile_scope!("compositor.end_frame");
            let compositor = self.compositor_config.compositor().unwrap();
            compositor.end_frame();
        }

        self.documents_seen.clear();
        self.shared_texture_cache_cleared = false;

        if self.renderer_errors.is_empty() {
            Ok(results)
        } else {
            Err(mem::replace(&mut self.renderer_errors, Vec::new()))
        }
    }

    fn update_gpu_profile(&mut self, device_size: DeviceIntSize) {
        let _gm = self.gpu_profiler.start_marker("build samples");
        // Block CPU waiting for last frame's GPU profiles to arrive.
        // In general this shouldn't block unless heavily GPU limited.
        let (gpu_frame_id, timers, samplers) = self.gpu_profiler.build_samples();

        if self.max_recorded_profiles > 0 {
            while self.gpu_profiles.len() >= self.max_recorded_profiles {
                self.gpu_profiles.pop_front();
            }

            self.gpu_profiles.push_back(GpuProfile::new(gpu_frame_id, &timers));
        }

        self.profiler.set_gpu_time_queries(timers);

        if !samplers.is_empty() {
            let screen_fraction = 1.0 / device_size.to_f32().area();

            fn accumulate_sampler_value(description: &str, samplers: &[GpuSampler]) -> f32 {
                let mut accum = 0.0;
                for sampler in samplers {
                    if sampler.tag.label != description {
                        continue;
                    }

                    accum += sampler.count as f32;
                }

                accum
            }

            let alpha_targets = accumulate_sampler_value(&"Alpha targets", &samplers) * screen_fraction;
            let transparent_pass = accumulate_sampler_value(&"Transparent pass", &samplers) * screen_fraction;
            let opaque_pass = accumulate_sampler_value(&"Opaque pass", &samplers) * screen_fraction;
            self.profile.set(profiler::ALPHA_TARGETS_SAMPLERS, alpha_targets);
            self.profile.set(profiler::TRANSPARENT_PASS_SAMPLERS, transparent_pass);
            self.profile.set(profiler::OPAQUE_PASS_SAMPLERS, opaque_pass);
            self.profile.set(profiler::TOTAL_SAMPLERS, alpha_targets + transparent_pass + opaque_pass);
        }
    }

    fn update_gpu_cache(&mut self) {
        let _gm = self.gpu_profiler.start_marker("gpu cache update");

        // For an artificial stress test of GPU cache resizing,
        // always pass an extra update list with at least one block in it.
        let gpu_cache_height = self.gpu_cache_texture.get_height();
        if gpu_cache_height != 0 && GPU_CACHE_RESIZE_TEST {
            self.pending_gpu_cache_updates.push(GpuCacheUpdateList {
                frame_id: FrameId::INVALID,
                clear: false,
                height: gpu_cache_height,
                blocks: vec![[1f32; 4].into()],
                updates: Vec::new(),
                debug_commands: Vec::new(),
            });
        }

        let (updated_blocks, max_requested_height) = self
            .pending_gpu_cache_updates
            .iter()
            .fold((0, gpu_cache_height), |(count, height), list| {
                (count + list.blocks.len(), cmp::max(height, list.height))
            });

        if max_requested_height > self.get_max_texture_size() && !self.gpu_cache_overflow {
            self.gpu_cache_overflow = true;
            self.renderer_errors.push(RendererError::MaxTextureSize);
        }

        // Note: if we decide to switch to scatter-style GPU cache update
        // permanently, we can have this code nicer with `BufferUploader` kind
        // of helper, similarly to how `TextureUploader` API is used.
        self.gpu_cache_texture.prepare_for_updates(
            &mut self.device,
            updated_blocks,
            max_requested_height,
        );

        for update_list in self.pending_gpu_cache_updates.drain(..) {
            assert!(update_list.height <= max_requested_height);
            if update_list.frame_id > self.gpu_cache_frame_id {
                self.gpu_cache_frame_id = update_list.frame_id
            }
            self.gpu_cache_texture
                .update(&mut self.device, &update_list);
        }

        self.profile.start_time(profiler::GPU_CACHE_UPLOAD_TIME);
        let updated_rows = self.gpu_cache_texture.flush(
            &mut self.device,
            &mut self.texture_upload_pbo_pool
        );
        self.gpu_cache_upload_time += self.profile.end_time(profiler::GPU_CACHE_UPLOAD_TIME);

        self.profile.set(profiler::GPU_CACHE_ROWS_UPDATED, updated_rows);
        self.profile.set(profiler::GPU_CACHE_BLOCKS_UPDATED, updated_blocks);
    }

    fn prepare_gpu_cache(
        &mut self,
        deferred_resolves: &[DeferredResolve],
    ) -> Result<(), RendererError> {
        if self.pending_gpu_cache_clear {
            let use_scatter =
                matches!(self.gpu_cache_texture.bus, GpuCacheBus::Scatter { .. });
            let new_cache = GpuCacheTexture::new(&mut self.device, use_scatter)?;
            let old_cache = mem::replace(&mut self.gpu_cache_texture, new_cache);
            old_cache.deinit(&mut self.device);
            self.pending_gpu_cache_clear = false;
        }

        let deferred_update_list = self.update_deferred_resolves(deferred_resolves);
        self.pending_gpu_cache_updates.extend(deferred_update_list);

        self.update_gpu_cache();

        // Note: the texture might have changed during the `update`,
        // so we need to bind it here.
        self.device.bind_texture(
            TextureSampler::GpuCache,
            self.gpu_cache_texture.texture.as_ref().unwrap(),
            Swizzle::default(),
        );

        Ok(())
    }

    fn update_texture_cache(&mut self) {
        profile_scope!("update_texture_cache");

        let _gm = self.gpu_profiler.start_marker("texture cache update");
        let mut pending_texture_updates = mem::replace(&mut self.pending_texture_updates, vec![]);
        self.pending_texture_cache_updates = false;

        self.profile.start_time(profiler::TEXTURE_CACHE_UPLOAD_TIME);

        for update_list in pending_texture_updates.drain(..) {
            for allocation in update_list.allocations {
                match allocation.kind {
                    TextureCacheAllocationKind::Alloc(_) => add_event_marker(c_str!("TextureCacheAlloc")),
                    TextureCacheAllocationKind::Reset(_) => add_event_marker(c_str!("TextureCacheReset")),
                    TextureCacheAllocationKind::Free => add_event_marker(c_str!("TextureCacheFree")),
                };
                let old = match allocation.kind {
                    TextureCacheAllocationKind::Alloc(ref info) |
                    TextureCacheAllocationKind::Reset(ref info) => {
                        // Create a new native texture, as requested by the texture cache.
                        //
                        // Ensure no PBO is bound when creating the texture storage,
                        // or GL will attempt to read data from there.
                        let mut texture = self.device.create_texture(
                            info.target,
                            info.format,
                            info.width,
                            info.height,
                            info.filter,
                            // This needs to be a render target because some render
                            // tasks get rendered into the texture cache.
                            Some(RenderTargetInfo { has_depth: info.has_depth }),
                            info.layer_count,
                        );

                        if info.is_shared_cache {
                            texture.flags_mut()
                                .insert(TextureFlags::IS_SHARED_TEXTURE_CACHE);

                            // On Mali-Gxx devices we use batched texture uploads as it performs much better.
                            // However, due to another driver bug we must ensure the textures are fully cleared,
                            // otherwise we get visual artefacts when blitting to the texture cache.
                            if self.device.get_capabilities().prefers_batched_texture_uploads &&
                                !self.device.get_capabilities().supports_render_target_partial_update
                            {
                                self.clear_texture(&texture, [0.0; 4]);
                            }

                            // Textures in the cache generally don't need to be cleared,
                            // but we do so if the debug display is active to make it
                            // easier to identify unallocated regions.
                            if self.debug_flags.contains(DebugFlags::TEXTURE_CACHE_DBG) {
                                self.clear_texture(&texture, TEXTURE_CACHE_DBG_CLEAR_COLOR);
                            }
                        }

                        self.texture_resolver.texture_cache_map.insert(allocation.id, texture)
                    }
                    TextureCacheAllocationKind::Free => {
                        self.texture_resolver.texture_cache_map.remove(&allocation.id)
                    }
                };

                match allocation.kind {
                    TextureCacheAllocationKind::Alloc(_) => {
                        assert!(old.is_none(), "Renderer and backend disagree!");
                    }
                    TextureCacheAllocationKind::Reset(_) |
                    TextureCacheAllocationKind::Free => {
                        assert!(old.is_some(), "Renderer and backend disagree!");
                    }
                }

                if let Some(old) = old {
                    self.device.delete_texture(old);
                }
            }

            let mut bytes_uploaded = 0;

            // On some devices performing many small texture uploads is slow, so instead we batch
            // updates in to a small number of uploads to temporary textures, then copy from those
            // textures to the correct place in the texture cache.
            // A list of temporary textures that batches of updates are uploaded to.
            let mut batch_upload_textures = Vec::new();
            #[derive(Debug)]
            pub struct BatchUploadCopy<'a> {
                // Index within batch_upload_textures
                src_texture_index: usize,
                src_offset: DeviceIntPoint,
                // We store the texture ID as well as a reference to the texture so that we
                // can easily sort the copies to group them by destination target.
                dest_texture_id: CacheTextureId,
                dest_texture: &'a Texture,
                dest_layer_index: LayerIndex,
                dest_offset: DeviceIntPoint,
                size: DeviceIntSize,
            }
            // A list of copies that must be performed from the temporary textures to the texture cache.
            let mut batch_upload_copies = Vec::new();
            #[derive(Debug)]
            pub struct BatchUploadBuffer<'a> {
                staging_buffer: UploadStagingBuffer<'a>,
                texture_index: usize,
            }
            // For each texture format, this stores a list of staging buffers
            // and a texture allocator for packing the buffers.
            let mut batch_upload_buffers = FastHashMap::default();

            // For best performance we use a single TextureUploader for all uploads.
            // This allows us to fill PBOs more efficiently and therefore allocate fewer PBOs.
            let mut uploader = self.device.upload_texture(
                &mut self.texture_upload_pbo_pool,
            );
            let device = &mut self.device;

            for (texture_id, updates) in update_list.updates {
                let texture = &self.texture_resolver.texture_cache_map[&texture_id];

                for update in updates {
                    let TextureCacheUpdate { rect, stride, offset, layer_index, format_override, source } = update;

                    let dummy_data;
                    let data = match source {
                        TextureUpdateSource::Bytes { ref data } => {
                            &data[offset as usize ..]
                        }
                        TextureUpdateSource::External { id, channel_index } => {
                            let handler = self.external_image_handler
                                .as_mut()
                                .expect("Found external image, but no handler set!");
                            // The filter is only relevant for NativeTexture external images.
                            match handler.lock(id, channel_index, ImageRendering::Auto).source {
                                ExternalImageSource::RawData(data) => {
                                    &data[offset as usize ..]
                                }
                                ExternalImageSource::Invalid => {
                                    // Create a local buffer to fill the pbo.
                                    let bpp = texture.get_format().bytes_per_pixel();
                                    let width = stride.unwrap_or(rect.size.width * bpp);
                                    let total_size = width * rect.size.height;
                                    // WR haven't support RGBAF32 format in texture_cache, so
                                    // we use u8 type here.
                                    dummy_data = vec![0xFFu8; total_size as usize];
                                    &dummy_data
                                }
                                ExternalImageSource::NativeTexture(eid) => {
                                    panic!("Unexpected external texture {:?} for the texture cache update of {:?}", eid, id);
                                }
                            }
                        }
                        TextureUpdateSource::DebugClear => {
                            let draw_target = DrawTarget::from_texture(
                                texture,
                                layer_index as usize,
                                false,
                            );
                            device.bind_draw_target(draw_target);
                            device.clear_target(
                                Some(TEXTURE_CACHE_DBG_CLEAR_COLOR),
                                None,
                                Some(draw_target.to_framebuffer_rect(update.rect.to_i32()))
                            );

                            continue;
                        }
                    };

                    const BATCH_UPLOAD_TEXTURE_SIZE: DeviceIntSize = DeviceIntSize::new(512, 512);
                    let use_batch_upload = device.get_capabilities().prefers_batched_texture_uploads &&
                        texture.flags().contains(TextureFlags::IS_SHARED_TEXTURE_CACHE) &&
                        rect.size.width <= BATCH_UPLOAD_TEXTURE_SIZE.width &&
                        rect.size.height <= BATCH_UPLOAD_TEXTURE_SIZE.height;

                    if use_batch_upload {
                        let (allocator, buffers) = batch_upload_buffers.entry(texture.get_format())
                            .or_insert_with(|| (GuillotineAllocator::new(None), Vec::new()));

                        // Allocate a region within the staging buffer for this update. If there is
                        // no room in an existing buffer then allocate another texture and buffer.
                        let (slice, origin) = match allocator.allocate(&rect.size) {
                            Some((slice, origin)) => (slice, origin),
                            None => {
                                let new_slice = FreeRectSlice(buffers.len() as u32);
                                allocator.extend(new_slice, BATCH_UPLOAD_TEXTURE_SIZE, rect.size);

                                let staging_texture = device.create_texture(
                                    ImageBufferKind::Texture2D,
                                    texture.get_format(),
                                    BATCH_UPLOAD_TEXTURE_SIZE.width,
                                    BATCH_UPLOAD_TEXTURE_SIZE.height,
                                    TextureFilter::Nearest,
                                    // Currently we need render target support as we always use glBlitFramebuffer
                                    // to copy the texture data. Instead, we should use glCopyImageSubData on some
                                    // platforms, and avoid creating the FBOs in that case.
                                    Some(RenderTargetInfo { has_depth: false }),
                                    1,
                                );

                                let staging_buffer = uploader.stage(
                                    device,
                                    staging_texture.get_format(),
                                    staging_texture.get_dimensions(),
                                ).unwrap();

                                let texture_index = batch_upload_textures.len();
                                batch_upload_textures.push(staging_texture);

                                buffers.push(BatchUploadBuffer {
                                    staging_buffer: staging_buffer,
                                    texture_index,
                                });
                                (new_slice, DeviceIntPoint::zero())
                            }
                        };
                        let buffer = &mut buffers[slice.0 as usize];
                        let allocated_rect = DeviceIntRect::new(origin, rect.size);

                        batch_upload_copies.push(BatchUploadCopy {
                            src_texture_index: buffer.texture_index,
                            src_offset: allocated_rect.origin,
                            dest_texture_id: texture_id,
                            dest_texture: texture,
                            dest_layer_index: layer_index as LayerIndex,
                            dest_offset: rect.origin,
                            size: rect.size,
                        });

                        unsafe {
                            let bytes_pp = texture.get_format().bytes_per_pixel() as usize;
                            let width_bytes = rect.size.width as usize * bytes_pp;
                            let src_stride = stride.map_or(width_bytes, |stride| {
                                assert!(stride >= 0);
                                stride as usize
                            });
                            let src_size = (rect.size.height as usize - 1) * src_stride + width_bytes;
                            assert!(src_size <= data.len());

                            let src: &[mem::MaybeUninit<u8>] = std::slice::from_raw_parts(data.as_ptr() as *const _, src_size);
                            let dst_stride = buffer.staging_buffer.get_stride();
                            let dst = buffer.staging_buffer.get_mapping();

                            // copy the data line-by-line in to the buffer so that we do not overwrite
                            // any other region of the buffer.
                            for y in 0..allocated_rect.size.height as usize {
                                let src_start = y * src_stride;
                                let src_end = src_start + width_bytes;
                                let dst_start = (allocated_rect.origin.y as usize + y as usize) * dst_stride +
                                    allocated_rect.origin.x as usize * bytes_pp;
                                let dst_end = dst_start + width_bytes;

                                dst[dst_start..dst_end].copy_from_slice(&src[src_start..src_end])
                            }
                        }
                    } else {
                        bytes_uploaded += uploader.upload(
                            device,
                            texture,
                            rect,
                            layer_index,
                            stride,
                            format_override,
                            data.as_ptr(),
                            data.len()
                        );
                    }

                    if let TextureUpdateSource::External { id, channel_index } = source {
                        let handler = self.external_image_handler
                            .as_mut()
                            .expect("Found external image, but no handler set!");
                        handler.unlock(id, channel_index);
                    }
                }
            }

            // Upload batched texture updates to their temporary textures.
            for batch_buffer in batch_upload_buffers.into_iter().map(|(_, (_, buffers))| buffers).flatten() {
                let texture = &batch_upload_textures[batch_buffer.texture_index];
                bytes_uploaded += uploader.upload_staged(
                    device,
                    texture,
                    DeviceIntRect::from_size(texture.get_dimensions()),
                    0,
                    None,
                    batch_buffer.staging_buffer,
                );
            }

            // Flush all uploads, batched or otherwise.
            uploader.flush(device);

            // Copy updates that were batch uploaded to their correct destination in the texture cache.
            // Sort them by destination and source to minimize framebuffer binding changes.
            batch_upload_copies.sort_unstable_by_key(|b| (b.dest_texture_id.0, b.dest_layer_index, b.src_texture_index));
            for copy in batch_upload_copies {
                device.copy_texture_sub_region(
                    &batch_upload_textures[copy.src_texture_index],
                    copy.src_offset.x as _,
                    copy.src_offset.y as _,
                    0,
                    copy.dest_texture,
                    copy.dest_offset.x as _,
                    copy.dest_offset.y as _,
                    copy.dest_layer_index,
                    copy.size.width as _,
                    copy.size.height as _,
                    1,
                );
            }

            for texture in batch_upload_textures.drain(..) {
                device.delete_texture(texture);
            }

            if update_list.clears_shared_cache {
                self.shared_texture_cache_cleared = true;
            }

            self.profile.inc(profiler::TEXTURE_UPLOADS);
            self.profile.add(profiler::TEXTURE_UPLOADS_MEM, profiler::bytes_to_mb(bytes_uploaded));
        }

        drain_filter(
            &mut self.notifications,
            |n| { n.when() == Checkpoint::FrameTexturesUpdated },
            |n| { n.notify(); },
        );

        let t = self.profile.end_time(profiler::TEXTURE_CACHE_UPLOAD_TIME);
        self.resource_upload_time += t;
    }

    fn bind_textures(&mut self, textures: &BatchTextures) {
        for i in 0 .. 3 {
            self.texture_resolver.bind(
                &textures.input.colors[i],
                TextureSampler::color(i),
                &mut self.device,
            );
        }

        self.texture_resolver.bind(
            &textures.clip_mask,
            TextureSampler::ClipMask,
            &mut self.device,
        );

        // TODO: this probably isn't the best place for this.
        if let Some(ref texture) = self.dither_matrix_texture {
            self.device.bind_texture(TextureSampler::Dither, texture, Swizzle::default());
        }
    }

    fn draw_instanced_batch<T>(
        &mut self,
        data: &[T],
        vertex_array_kind: VertexArrayKind,
        textures: &BatchTextures,
        stats: &mut RendererStats,
    ) {
        self.bind_textures(textures);

        // If we end up with an empty draw call here, that means we have
        // probably introduced unnecessary batch breaks during frame
        // building - so we should be catching this earlier and removing
        // the batch.
        debug_assert!(!data.is_empty());

        let vao = get_vao(vertex_array_kind, &self.vaos);
        self.device.bind_vao(vao);

        let chunk_size = if self.debug_flags.contains(DebugFlags::DISABLE_BATCHING) {
            1
        } else if vertex_array_kind == VertexArrayKind::Primitive {
            self.max_primitive_instance_count
        } else {
            data.len()
        };

        for chunk in data.chunks(chunk_size) {
            self.device
                .update_vao_instances(vao, chunk, ONE_TIME_USAGE_HINT);
            self.device
                .draw_indexed_triangles_instanced_u16(6, chunk.len() as i32);
            self.profile.inc(profiler::DRAW_CALLS);
            stats.total_draw_calls += 1;
        }

        self.profile.add(profiler::VERTICES, 6 * data.len());
    }

    fn handle_readback_composite(
        &mut self,
        draw_target: DrawTarget,
        uses_scissor: bool,
        source: &RenderTask,
        backdrop: &RenderTask,
        readback: &RenderTask,
    ) {
        if uses_scissor {
            self.device.disable_scissor();
        }

        let texture_source = TextureSource::TextureCache(
            readback.get_target_texture(),
            Swizzle::default(),
        );
        let (cache_texture, _) = self.texture_resolver
            .resolve(&texture_source).expect("bug: no source texture");

        // Before submitting the composite batch, do the
        // framebuffer readbacks that are needed for each
        // composite operation in this batch.
        let (readback_rect, readback_layer) = readback.get_target_rect();
        let (backdrop_rect, _) = backdrop.get_target_rect();
        let (backdrop_screen_origin, backdrop_scale) = match backdrop.kind {
            RenderTaskKind::Picture(ref task_info) => (task_info.content_origin, task_info.device_pixel_scale),
            _ => panic!("bug: composite on non-picture?"),
        };
        let (source_screen_origin, source_scale) = match source.kind {
            RenderTaskKind::Picture(ref task_info) => (task_info.content_origin, task_info.device_pixel_scale),
            _ => panic!("bug: composite on non-picture?"),
        };

        // Bind the FBO to blit the backdrop to.
        // Called per-instance in case the layer (and therefore FBO)
        // changes. The device will skip the GL call if the requested
        // target is already bound.
        let cache_draw_target = DrawTarget::from_texture(
            cache_texture,
            readback_layer.0 as usize,
            false,
        );

        let source_in_backdrop_space = source_screen_origin * (backdrop_scale.0 / source_scale.0);

        let mut src = DeviceIntRect::new(
            (source_in_backdrop_space + (backdrop_rect.origin.to_f32() - backdrop_screen_origin)).to_i32(),
            readback_rect.size,
        );
        let mut dest = readback_rect.to_i32();
        let device_to_framebuffer = Scale::new(1i32);

        // Need to invert the y coordinates and flip the image vertically when
        // reading back from the framebuffer.
        if draw_target.is_default() {
            src.origin.y = draw_target.dimensions().height as i32 - src.size.height - src.origin.y;
            dest.origin.y += dest.size.height;
            dest.size.height = -dest.size.height;
        }

        self.device.blit_render_target(
            draw_target.into(),
            src * device_to_framebuffer,
            cache_draw_target,
            dest * device_to_framebuffer,
            TextureFilter::Linear,
        );

        // Restore draw target to current pass render target + layer, and reset
        // the read target.
        self.device.bind_draw_target(draw_target);
        self.device.reset_read_target();

        if uses_scissor {
            self.device.enable_scissor();
        }
    }

    fn handle_blits(
        &mut self,
        blits: &[BlitJob],
        render_tasks: &RenderTaskGraph,
        draw_target: DrawTarget,
    ) {
        if blits.is_empty() {
            return;
        }

        let _timer = self.gpu_profiler.start_timer(GPU_TAG_BLIT);

        // TODO(gw): For now, we don't bother batching these by source texture.
        //           If if ever shows up as an issue, we can easily batch them.
        for blit in blits {
            let (source, layer, source_rect) = match blit.source {
                BlitJobSource::Texture(texture_id, layer, source_rect) => {
                    // A blit from a texture into this target.
                    (texture_id, layer as usize, source_rect)
                }
                BlitJobSource::RenderTask(task_id) => {
                    // A blit from the child render task into this target.
                    // TODO(gw): Support R8 format here once we start
                    //           creating mips for alpha masks.
                    let source = &render_tasks[task_id];
                    let (source_rect, layer) = source.get_target_rect();
                    let source_texture = TextureSource::TextureCache(
                        source.get_target_texture(),
                        Swizzle::default(),
                    );
                    (source_texture, layer.0, source_rect)
                }
            };

            debug_assert_eq!(source_rect.size, blit.target_rect.size);
            let (texture, swizzle) = self.texture_resolver
                .resolve(&source)
                .expect("BUG: invalid source texture");

            if swizzle != Swizzle::default() {
                error!("Swizzle {:?} can't be handled by a blit", swizzle);
            }

            let read_target = DrawTarget::from_texture(
                texture,
                layer,
                false,
            );

            self.device.blit_render_target(
                read_target.into(),
                read_target.to_framebuffer_rect(source_rect),
                draw_target,
                draw_target.to_framebuffer_rect(blit.target_rect),
                TextureFilter::Linear,
            );
        }
    }

    fn handle_scaling(
        &mut self,
        scalings: &FastHashMap<TextureSource, Vec<ScalingInstance>>,
        projection: &default::Transform3D<f32>,
        stats: &mut RendererStats,
    ) {
        if scalings.is_empty() {
            return
        }

        let _timer = self.gpu_profiler.start_timer(GPU_TAG_SCALE);

        for (source, instances) in scalings {
            let buffer_kind = source.image_buffer_kind();

            self.shaders
                .borrow_mut()
                .get_scale_shader(buffer_kind)
                .bind(
                    &mut self.device,
                    &projection,
                    &mut self.renderer_errors,
                );

            self.draw_instanced_batch(
                instances,
                VertexArrayKind::Scale,
                &BatchTextures::composite_rgb(*source),
                stats,
            );
        }
    }

    fn handle_svg_filters(
        &mut self,
        textures: &BatchTextures,
        svg_filters: &[SvgFilterInstance],
        projection: &default::Transform3D<f32>,
        stats: &mut RendererStats,
    ) {
        if svg_filters.is_empty() {
            return;
        }

        let _timer = self.gpu_profiler.start_timer(GPU_TAG_SVG_FILTER);

        self.shaders.borrow_mut().cs_svg_filter.bind(
            &mut self.device,
            &projection,
            &mut self.renderer_errors
        );

        self.draw_instanced_batch(
            &svg_filters,
            VertexArrayKind::SvgFilter,
            textures,
            stats,
        );
    }

    fn draw_picture_cache_target(
        &mut self,
        target: &PictureCacheTarget,
        draw_target: DrawTarget,
        projection: &default::Transform3D<f32>,
        render_tasks: &RenderTaskGraph,
        stats: &mut RendererStats,
    ) {
        profile_scope!("draw_picture_cache_target");

        self.profile.inc(profiler::RENDERED_PICTURE_TILES);
        let _gm = self.gpu_profiler.start_marker("picture cache target");
        let framebuffer_kind = FramebufferKind::Other;

        {
            let _timer = self.gpu_profiler.start_timer(GPU_TAG_SETUP_TARGET);
            self.device.bind_draw_target(draw_target);
            self.device.enable_depth_write();
            self.set_blend(false, framebuffer_kind);

            let clear_color = target.clear_color.map(|c| c.to_array());
            let scissor_rect = if self.device.get_capabilities().supports_render_target_partial_update {
                target.alpha_batch_container.task_scissor_rect
            } else {
                None
            };
            match scissor_rect {
                // If updating only a dirty rect within a picture cache target, the
                // clear must also be scissored to that dirty region.
                Some(r) if self.clear_caches_with_quads => {
                    self.device.enable_depth(DepthFunction::Always);
                    // Save the draw call count so that our reftests don't get confused...
                    let old_draw_call_count = stats.total_draw_calls;
                    if clear_color.is_none() {
                        self.device.disable_color_write();
                    }
                    let instance = ClearInstance {
                        rect: [
                            r.origin.x as f32, r.origin.y as f32,
                            r.size.width as f32, r.size.height as f32,
                        ],
                        color: clear_color.unwrap_or([0.0; 4]),
                    };
                    self.shaders.borrow_mut().ps_clear.bind(
                        &mut self.device,
                        &projection,
                        &mut self.renderer_errors,
                    );
                    self.draw_instanced_batch(
                        &[instance],
                        VertexArrayKind::Clear,
                        &BatchTextures::empty(),
                        stats,
                    );
                    if clear_color.is_none() {
                        self.device.enable_color_write();
                    }
                    stats.total_draw_calls = old_draw_call_count;
                    self.device.disable_depth();
                }
                other => {
                    let scissor_rect = other.map(|rect| {
                        draw_target.build_scissor_rect(Some(rect))
                    });
                    self.device.clear_target(clear_color, Some(1.0), scissor_rect);
                }
            };
            self.device.disable_depth_write();
        }

        self.draw_alpha_batch_container(
            &target.alpha_batch_container,
            draw_target,
            framebuffer_kind,
            projection,
            render_tasks,
            stats,
        );

        self.device.invalidate_depth_target();
    }

    /// Draw an alpha batch container into a given draw target. This is used
    /// by both color and picture cache target kinds.
    fn draw_alpha_batch_container(
        &mut self,
        alpha_batch_container: &AlphaBatchContainer,
        draw_target: DrawTarget,
        framebuffer_kind: FramebufferKind,
        projection: &default::Transform3D<f32>,
        render_tasks: &RenderTaskGraph,
        stats: &mut RendererStats,
    ) {
        let uses_scissor = alpha_batch_container.task_scissor_rect.is_some();

        if uses_scissor {
            self.device.enable_scissor();
            let scissor_rect = draw_target.build_scissor_rect(
                alpha_batch_container.task_scissor_rect,
            );
            self.device.set_scissor_rect(scissor_rect)
        }

        if !alpha_batch_container.opaque_batches.is_empty()
            && !self.debug_flags.contains(DebugFlags::DISABLE_OPAQUE_PASS) {
            let _gl = self.gpu_profiler.start_marker("opaque batches");
            let opaque_sampler = self.gpu_profiler.start_sampler(GPU_SAMPLER_TAG_OPAQUE);
            self.set_blend(false, framebuffer_kind);
            //Note: depth equality is needed for split planes
            self.device.enable_depth(DepthFunction::LessEqual);
            self.device.enable_depth_write();

            // Draw opaque batches front-to-back for maximum
            // z-buffer efficiency!
            for batch in alpha_batch_container
                .opaque_batches
                .iter()
                .rev()
                {
                    if should_skip_batch(&batch.key.kind, self.debug_flags) {
                        continue;
                    }

                    self.shaders.borrow_mut()
                        .get(&batch.key, batch.features, self.debug_flags)
                        .bind(
                            &mut self.device, projection,
                            &mut self.renderer_errors,
                        );

                    let _timer = self.gpu_profiler.start_timer(batch.key.kind.sampler_tag());
                    self.draw_instanced_batch(
                        &batch.instances,
                        VertexArrayKind::Primitive,
                        &batch.key.textures,
                        stats
                    );
                }

            self.device.disable_depth_write();
            self.gpu_profiler.finish_sampler(opaque_sampler);
        } else {
            self.device.disable_depth();
        }

        if !alpha_batch_container.alpha_batches.is_empty()
            && !self.debug_flags.contains(DebugFlags::DISABLE_ALPHA_PASS) {
            let _gl = self.gpu_profiler.start_marker("alpha batches");
            let transparent_sampler = self.gpu_profiler.start_sampler(GPU_SAMPLER_TAG_TRANSPARENT);
            self.set_blend(true, framebuffer_kind);

            let mut prev_blend_mode = BlendMode::None;
            let shaders_rc = self.shaders.clone();

            for batch in &alpha_batch_container.alpha_batches {
                if should_skip_batch(&batch.key.kind, self.debug_flags) {
                    continue;
                }

                let mut shaders = shaders_rc.borrow_mut();
                let shader = shaders.get(
                    &batch.key,
                    batch.features | BatchFeatures::ALPHA_PASS,
                    self.debug_flags,
                );

                if batch.key.blend_mode != prev_blend_mode {
                    match batch.key.blend_mode {
                        _ if self.debug_flags.contains(DebugFlags::SHOW_OVERDRAW) &&
                            framebuffer_kind == FramebufferKind::Main => {
                            self.device.set_blend_mode_show_overdraw();
                        }
                        BlendMode::None => {
                            unreachable!("bug: opaque blend in alpha pass");
                        }
                        BlendMode::Alpha => {
                            self.device.set_blend_mode_alpha();
                        }
                        BlendMode::PremultipliedAlpha => {
                            self.device.set_blend_mode_premultiplied_alpha();
                        }
                        BlendMode::PremultipliedDestOut => {
                            self.device.set_blend_mode_premultiplied_dest_out();
                        }
                        BlendMode::SubpixelDualSource => {
                            self.device.set_blend_mode_subpixel_dual_source();
                        }
                        BlendMode::SubpixelConstantTextColor(color) => {
                            self.device.set_blend_mode_subpixel_constant_text_color(color);
                        }
                        BlendMode::SubpixelWithBgColor => {
                            // Using the three pass "component alpha with font smoothing
                            // background color" rendering technique:
                            //
                            // /webrender/doc/text-rendering.md
                            //
                            self.device.set_blend_mode_subpixel_with_bg_color_pass0();
                            // need to make sure the shader is bound
                            shader.bind(
                                &mut self.device,
                                projection,
                                &mut self.renderer_errors,
                            );
                            self.device.switch_mode(ShaderColorMode::SubpixelWithBgColorPass0 as _);
                        }
                        BlendMode::Advanced(mode) => {
                            if self.enable_advanced_blend_barriers {
                                self.device.gl().blend_barrier_khr();
                            }
                            self.device.set_blend_mode_advanced(mode);
                        }
                    }
                    prev_blend_mode = batch.key.blend_mode;
                }

                // Handle special case readback for composites.
                if let BatchKind::Brush(BrushBatchKind::MixBlend { task_id, source_id, backdrop_id }) = batch.key.kind {
                    // composites can't be grouped together because
                    // they may overlap and affect each other.
                    debug_assert_eq!(batch.instances.len(), 1);
                    self.handle_readback_composite(
                        draw_target,
                        uses_scissor,
                        &render_tasks[source_id],
                        &render_tasks[task_id],
                        &render_tasks[backdrop_id],
                    );
                }

                let _timer = self.gpu_profiler.start_timer(batch.key.kind.sampler_tag());
                shader.bind(
                    &mut self.device,
                    projection,
                    &mut self.renderer_errors,
                );

                self.draw_instanced_batch(
                    &batch.instances,
                    VertexArrayKind::Primitive,
                    &batch.key.textures,
                    stats
                );

                if batch.key.blend_mode == BlendMode::SubpixelWithBgColor {
                    self.set_blend_mode_subpixel_with_bg_color_pass1(framebuffer_kind);
                    // re-binding the shader after the blend mode change
                    shader.bind(
                        &mut self.device,
                        projection,
                        &mut self.renderer_errors,
                    );
                    self.device.switch_mode(ShaderColorMode::SubpixelWithBgColorPass1 as _);

                    // When drawing the 2nd and 3rd passes, we know that the VAO, textures etc
                    // are all set up from the previous draw_instanced_batch call,
                    // so just issue a draw call here to avoid re-uploading the
                    // instances and re-binding textures etc.
                    self.device
                        .draw_indexed_triangles_instanced_u16(6, batch.instances.len() as i32);

                    self.set_blend_mode_subpixel_with_bg_color_pass2(framebuffer_kind);
                    // re-binding the shader after the blend mode change
                    shader.bind(
                        &mut self.device,
                        projection,
                        &mut self.renderer_errors,
                    );
                    self.device.switch_mode(ShaderColorMode::SubpixelWithBgColorPass2 as _);

                    self.device
                        .draw_indexed_triangles_instanced_u16(6, batch.instances.len() as i32);
                }

                if batch.key.blend_mode == BlendMode::SubpixelWithBgColor {
                    prev_blend_mode = BlendMode::None;
                }
            }

            self.set_blend(false, framebuffer_kind);
            self.gpu_profiler.finish_sampler(transparent_sampler);
        }

        self.device.disable_depth();
        if uses_scissor {
            self.device.disable_scissor();
        }
    }

    /// Rasterize any external compositor surfaces that require updating
    fn update_external_native_surfaces(
        &mut self,
        external_surfaces: &[ResolvedExternalSurface],
        results: &mut RenderResults,
    ) {
        if external_surfaces.is_empty() {
            return;
        }

        let opaque_sampler = self.gpu_profiler.start_sampler(GPU_SAMPLER_TAG_OPAQUE);

        self.device.disable_depth();
        self.set_blend(false, FramebufferKind::Main);

        for surface in external_surfaces {
            // See if this surface needs to be updated
            let (native_surface_id, surface_size) = match surface.update_params {
                Some(params) => params,
                None => continue,
            };

            // When updating an external surface, the entire surface rect is used
            // for all of the draw, dirty, valid and clip rect parameters.
            let surface_rect = surface_size.into();

            // Bind the native compositor surface to update
            let surface_info = self.compositor_config
                .compositor()
                .unwrap()
                .bind(
                    NativeTileId {
                        surface_id: native_surface_id,
                        x: 0,
                        y: 0,
                    },
                    surface_rect,
                    surface_rect,
                );

            // Bind the native surface to current FBO target
            let draw_target = DrawTarget::NativeSurface {
                offset: surface_info.origin,
                external_fbo_id: surface_info.fbo_id,
                dimensions: surface_size,
            };
            self.device.bind_draw_target(draw_target);

            let projection = Transform3D::ortho(
                0.0,
                surface_size.width as f32,
                0.0,
                surface_size.height as f32,
                self.device.ortho_near_plane(),
                self.device.ortho_far_plane(),
            );

            let ( textures, instance ) = match surface.color_data {
                ResolvedExternalSurfaceColorData::Yuv{
                        ref planes, color_space, format, rescale, .. } => {

                    // Bind an appropriate YUV shader for the texture format kind
                    self.shaders
                        .borrow_mut()
                        .get_composite_shader(
                            CompositeSurfaceFormat::Yuv,
                            surface.image_buffer_kind,
                        ).bind(
                            &mut self.device,
                            &projection,
                            &mut self.renderer_errors
                        );

                    let textures = BatchTextures::composite_yuv(
                        planes[0].texture,
                        planes[1].texture,
                        planes[2].texture,
                    );

                    // When the texture is an external texture, the UV rect is not known when
                    // the external surface descriptor is created, because external textures
                    // are not resolved until the lock() callback is invoked at the start of
                    // the frame render. To handle this, query the texture resolver for the
                    // UV rect if it's an external texture, otherwise use the default UV rect.
                    let uv_rects = [
                        self.texture_resolver.get_uv_rect(&textures.input.colors[0], planes[0].uv_rect),
                        self.texture_resolver.get_uv_rect(&textures.input.colors[1], planes[1].uv_rect),
                        self.texture_resolver.get_uv_rect(&textures.input.colors[2], planes[2].uv_rect),
                    ];

                    let instance = CompositeInstance::new_yuv(
                        surface_rect.to_f32(),
                        surface_rect.to_f32(),
                        // z-id is not relevant when updating a native compositor surface.
                        // TODO(gw): Support compositor surfaces without z-buffer, for memory / perf win here.
                        ZBufferId(0),
                        color_space,
                        format,
                        rescale,
                        [
                            planes[0].texture_layer as f32,
                            planes[1].texture_layer as f32,
                            planes[2].texture_layer as f32,
                        ],
                        uv_rects,
                    );

                    ( textures, instance )
                },
                ResolvedExternalSurfaceColorData::Rgb{ ref plane, flip_y, .. } => {
                    self.shaders
                        .borrow_mut()
                        .get_composite_shader(
                            CompositeSurfaceFormat::Rgba,
                            surface.image_buffer_kind,
                        ).bind(
                            &mut self.device,
                            &projection,
                            &mut self.renderer_errors
                        );

                    let textures = BatchTextures::composite_rgb(plane.texture);
                    let mut uv_rect = self.texture_resolver.get_uv_rect(&textures.input.colors[0], plane.uv_rect);
                    if flip_y {
                        let y = uv_rect.uv0.y;
                        uv_rect.uv0.y = uv_rect.uv1.y;
                        uv_rect.uv1.y = y;
                    }
                    let instance = CompositeInstance::new_rgb(
                        surface_rect.to_f32(),
                        surface_rect.to_f32(),
                        PremultipliedColorF::WHITE,
                        plane.texture_layer as f32,
                        ZBufferId(0),
                        uv_rect,
                    );

                    ( textures, instance )
                },
            };

            self.draw_instanced_batch(
                &[instance],
                VertexArrayKind::Composite,
                &textures,
                &mut results.stats,
            );

            self.compositor_config
                .compositor()
                .unwrap()
                .unbind();
        }

        self.gpu_profiler.finish_sampler(opaque_sampler);
    }

    /// Draw a list of tiles to the framebuffer
    fn draw_tile_list<'a, I: Iterator<Item = &'a CompositeTile>>(
        &mut self,
        tiles_iter: I,
        external_surfaces: &[ResolvedExternalSurface],
        projection: &default::Transform3D<f32>,
        partial_present_mode: Option<PartialPresentMode>,
        stats: &mut RendererStats,
    ) {
        self.shaders
            .borrow_mut()
            .get_composite_shader(
                CompositeSurfaceFormat::Rgba,
                ImageBufferKind::Texture2D,
            ).bind(
                &mut self.device,
                projection,
                &mut self.renderer_errors
            );

        let mut current_shader_params = (CompositeSurfaceFormat::Rgba, ImageBufferKind::Texture2D);
        let mut current_textures = BatchTextures::empty();
        let mut instances = Vec::new();

        for tile in tiles_iter {
            // Determine a clip rect to apply to this tile, depending on what
            // the partial present mode is.
            let partial_clip_rect = match partial_present_mode {
                Some(PartialPresentMode::Single { dirty_rect }) => dirty_rect,
                None => tile.rect,
            };

            let clip_rect = match partial_clip_rect.intersection(&tile.clip_rect) {
                Some(rect) => rect,
                None => continue,
            };

            // Simple compositor needs the valid rect in device space to match clip rect
            let valid_device_rect = tile.valid_rect.translate(
                tile.rect.origin.to_vector()
            );

            // Only composite the part of the tile that contains valid pixels
            let clip_rect = match clip_rect.intersection(&valid_device_rect) {
                Some(rect) => rect,
                None => continue,
            };

            // Work out the draw params based on the tile surface
            let (instance, textures, shader_params) = match tile.surface {
                CompositeTileSurface::Color { color } => {
                    let dummy = TextureSource::Dummy;
                    let image_buffer_kind = dummy.image_buffer_kind();
                    (
                        CompositeInstance::new(
                            tile.rect,
                            clip_rect,
                            color.premultiplied(),
                            0.0,
                            tile.z_id,
                        ),
                        BatchTextures::composite_rgb(dummy),
                        (CompositeSurfaceFormat::Rgba, image_buffer_kind),
                    )
                }
                CompositeTileSurface::Clear => {
                    let dummy = TextureSource::Dummy;
                    let image_buffer_kind = dummy.image_buffer_kind();
                    (
                        CompositeInstance::new(
                            tile.rect,
                            clip_rect,
                            PremultipliedColorF::BLACK,
                            0.0,
                            tile.z_id,
                        ),
                        BatchTextures::composite_rgb(dummy),
                        (CompositeSurfaceFormat::Rgba, image_buffer_kind),
                    )
                }
                CompositeTileSurface::Texture { surface: ResolvedSurfaceTexture::TextureCache { texture, layer } } => {
                    (
                        CompositeInstance::new(
                            tile.rect,
                            clip_rect,
                            PremultipliedColorF::WHITE,
                            layer as f32,
                            tile.z_id,
                        ),
                        BatchTextures::composite_rgb(texture),
                        (CompositeSurfaceFormat::Rgba, ImageBufferKind::Texture2D),
                    )
                }
                CompositeTileSurface::ExternalSurface { external_surface_index } => {
                    let surface = &external_surfaces[external_surface_index.0];

                    match surface.color_data {
                        ResolvedExternalSurfaceColorData::Yuv{ ref planes, color_space, format, rescale, .. } => {
                            let textures = BatchTextures::composite_yuv(
                                planes[0].texture,
                                planes[1].texture,
                                planes[2].texture,
                            );

                            // When the texture is an external texture, the UV rect is not known when
                            // the external surface descriptor is created, because external textures
                            // are not resolved until the lock() callback is invoked at the start of
                            // the frame render. To handle this, query the texture resolver for the
                            // UV rect if it's an external texture, otherwise use the default UV rect.
                            let uv_rects = [
                                self.texture_resolver.get_uv_rect(&textures.input.colors[0], planes[0].uv_rect),
                                self.texture_resolver.get_uv_rect(&textures.input.colors[1], planes[1].uv_rect),
                                self.texture_resolver.get_uv_rect(&textures.input.colors[2], planes[2].uv_rect),
                            ];

                            (
                                CompositeInstance::new_yuv(
                                    tile.rect,
                                    clip_rect,
                                    tile.z_id,
                                    color_space,
                                    format,
                                    rescale,
                                    [
                                        planes[0].texture_layer as f32,
                                        planes[1].texture_layer as f32,
                                        planes[2].texture_layer as f32,
                                    ],
                                    uv_rects,
                                ),
                                textures,
                                (CompositeSurfaceFormat::Yuv, surface.image_buffer_kind),
                            )
                        },
                        ResolvedExternalSurfaceColorData::Rgb{ ref plane, flip_y, .. } => {

                            let mut uv_rect = self.texture_resolver.get_uv_rect(&plane.texture, plane.uv_rect);
                            if flip_y {
                                let y = uv_rect.uv0.y;
                                uv_rect.uv0.y = uv_rect.uv1.y;
                                uv_rect.uv1.y = y;
                            }

                            (
                                CompositeInstance::new_rgb(
                                    tile.rect,
                                    clip_rect,
                                    PremultipliedColorF::WHITE,
                                    plane.texture_layer as f32,
                                    tile.z_id,
                                    uv_rect,
                                ),
                                BatchTextures::composite_rgb(plane.texture),
                                (CompositeSurfaceFormat::Rgba, surface.image_buffer_kind),
                            )
                        },
                    }
                }
                CompositeTileSurface::Texture { surface: ResolvedSurfaceTexture::Native { .. } } => {
                    unreachable!("bug: found native surface in simple composite path");
                }
            };

            // Flush batch if shader params or textures changed
            let flush_batch = !current_textures.is_compatible_with(&textures) ||
                shader_params != current_shader_params;

            if flush_batch {
                if !instances.is_empty() {
                    self.draw_instanced_batch(
                        &instances,
                        VertexArrayKind::Composite,
                        &current_textures,
                        stats,
                    );
                    instances.clear();
                }
            }

            if shader_params != current_shader_params {
                self.shaders
                    .borrow_mut()
                    .get_composite_shader(shader_params.0, shader_params.1)
                    .bind(
                        &mut self.device,
                        projection,
                        &mut self.renderer_errors
                    );

                current_shader_params = shader_params;
            }

            current_textures = textures;

            // Add instance to current batch
            instances.push(instance);
        }

        // Flush the last batch
        if !instances.is_empty() {
            self.draw_instanced_batch(
                &instances,
                VertexArrayKind::Composite,
                &current_textures,
                stats,
            );
        }
    }

    /// Composite picture cache tiles into the framebuffer. This is currently
    /// the only way that picture cache tiles get drawn. In future, the tiles
    /// will often be handed to the OS compositor, and this method will be
    /// rarely used.
    fn composite_simple(
        &mut self,
        composite_state: &CompositeState,
        draw_target: DrawTarget,
        projection: &default::Transform3D<f32>,
        results: &mut RenderResults,
        partial_present_mode: Option<PartialPresentMode>,
    ) {
        let _gm = self.gpu_profiler.start_marker("framebuffer");
        let _timer = self.gpu_profiler.start_timer(GPU_TAG_COMPOSITE);

        self.device.bind_draw_target(draw_target);
        self.device.enable_depth(DepthFunction::LessEqual);
        self.device.enable_depth_write();

        // Clear the framebuffer
        let clear_color = self.clear_color.map(|color| color.to_array());

        match partial_present_mode {
            Some(PartialPresentMode::Single { dirty_rect }) => {
                // We have a single dirty rect, so clear only that
                self.device.clear_target(clear_color,
                                         Some(1.0),
                                         Some(draw_target.to_framebuffer_rect(dirty_rect.to_i32())));
            }
            None => {
                // Partial present is disabled, so clear the entire framebuffer
                self.device.clear_target(clear_color,
                                         Some(1.0),
                                         None);
            }
        }

        // We are only interested in tiles backed with actual cached pixels so we don't
        // count clear tiles here.
        let num_tiles = composite_state.opaque_tiles.len()
            + composite_state.alpha_tiles.len();
        self.profile.set(profiler::PICTURE_TILES, num_tiles);

        // Draw opaque tiles first, front-to-back to get maxmum
        // z-reject efficiency.
        if !composite_state.opaque_tiles.is_empty() {
            let opaque_sampler = self.gpu_profiler.start_sampler(GPU_SAMPLER_TAG_OPAQUE);
            self.device.enable_depth_write();
            self.set_blend(false, FramebufferKind::Main);
            self.draw_tile_list(
                composite_state.opaque_tiles.iter().rev(),
                &composite_state.external_surfaces,
                projection,
                partial_present_mode,
                &mut results.stats,
            );
            self.gpu_profiler.finish_sampler(opaque_sampler);
        }

        if !composite_state.clear_tiles.is_empty() {
            let transparent_sampler = self.gpu_profiler.start_sampler(GPU_SAMPLER_TAG_TRANSPARENT);
            self.device.disable_depth_write();
            self.set_blend(true, FramebufferKind::Main);
            self.device.set_blend_mode_premultiplied_dest_out();
            self.draw_tile_list(
                composite_state.clear_tiles.iter(),
                &composite_state.external_surfaces,
                projection,
                partial_present_mode,
                &mut results.stats,
            );
            self.gpu_profiler.finish_sampler(transparent_sampler);
        }

        // Draw alpha tiles
        if !composite_state.alpha_tiles.is_empty() {
            let transparent_sampler = self.gpu_profiler.start_sampler(GPU_SAMPLER_TAG_TRANSPARENT);
            self.device.disable_depth_write();
            self.set_blend(true, FramebufferKind::Main);
            self.set_blend_mode_premultiplied_alpha(FramebufferKind::Main);
            self.draw_tile_list(
                composite_state.alpha_tiles.iter(),
                &composite_state.external_surfaces,
                projection,
                partial_present_mode,
                &mut results.stats,
            );
            self.gpu_profiler.finish_sampler(transparent_sampler);
        }
    }

    fn draw_color_target(
        &mut self,
        draw_target: DrawTarget,
        target: &ColorRenderTarget,
        clear_color: Option<[f32; 4]>,
        clear_depth: Option<f32>,
        render_tasks: &RenderTaskGraph,
        projection: &default::Transform3D<f32>,
        stats: &mut RendererStats,
    ) {
        profile_scope!("draw_color_target");

        self.profile.inc(profiler::COLOR_PASSES);
        let _gm = self.gpu_profiler.start_marker("color target");

        // sanity check for the depth buffer
        if let DrawTarget::Texture { with_depth, .. } = draw_target {
            assert!(with_depth >= target.needs_depth());
        }

        let framebuffer_kind = if draw_target.is_default() {
            FramebufferKind::Main
        } else {
            FramebufferKind::Other
        };

        {
            let _timer = self.gpu_profiler.start_timer(GPU_TAG_SETUP_TARGET);
            self.device.bind_draw_target(draw_target);
            self.device.disable_depth();
            self.set_blend(false, framebuffer_kind);

            if clear_depth.is_some() {
                self.device.enable_depth_write();
            }

            let clear_rect = match draw_target {
                DrawTarget::NativeSurface { .. } => {
                    unreachable!("bug: native compositor surface in child target");
                }
                DrawTarget::Default { rect, total_size, .. } if rect.origin == FramebufferIntPoint::zero() && rect.size == total_size => {
                    // whole screen is covered, no need for scissor
                    None
                }
                DrawTarget::Default { rect, .. } => {
                    Some(rect)
                }
                DrawTarget::Texture { .. } if self.enable_clear_scissor => {
                    // TODO(gw): Applying a scissor rect and minimal clear here
                    // is a very large performance win on the Intel and nVidia
                    // GPUs that I have tested with. It's possible it may be a
                    // performance penalty on other GPU types - we should test this
                    // and consider different code paths.
                    //
                    // Note: The above measurements were taken when render
                    // target slices were minimum 2048x2048. Now that we size
                    // them adaptively, this may be less of a win (except perhaps
                    // on a mostly-unused last slice of a large texture array).
                    Some(draw_target.to_framebuffer_rect(target.used_rect))
                }
                DrawTarget::Texture { .. } | DrawTarget::External { .. } => {
                    None
                }
            };

            self.device.clear_target(
                clear_color,
                clear_depth,
                clear_rect,
            );

            if clear_depth.is_some() {
                self.device.disable_depth_write();
            }
        }

        // Handle any blits from the texture cache to this target.
        self.handle_blits(
            &target.blits,
            render_tasks,
            draw_target,
        );

        // Draw any blurs for this target.
        // Blurs are rendered as a standard 2-pass
        // separable implementation.
        // TODO(gw): In the future, consider having
        //           fast path blur shaders for common
        //           blur radii with fixed weights.
        if !target.vertical_blurs.is_empty() || !target.horizontal_blurs.is_empty() {
            let _timer = self.gpu_profiler.start_timer(GPU_TAG_BLUR);

            self.set_blend(false, framebuffer_kind);
            self.shaders.borrow_mut().cs_blur_rgba8
                .bind(&mut self.device, projection, &mut self.renderer_errors);

            if !target.vertical_blurs.is_empty() {
                self.draw_blurs(
                    &target.vertical_blurs,
                    stats,
                );
            }

            if !target.horizontal_blurs.is_empty() {
                self.draw_blurs(
                    &target.horizontal_blurs,
                    stats,
                );
            }
        }

        self.handle_scaling(
            &target.scalings,
            projection,
            stats,
        );

        for (ref textures, ref filters) in &target.svg_filters {
            self.handle_svg_filters(
                textures,
                filters,
                projection,
                stats,
            );
        }

        for alpha_batch_container in &target.alpha_batch_containers {
            self.draw_alpha_batch_container(
                alpha_batch_container,
                draw_target,
                framebuffer_kind,
                projection,
                render_tasks,
                stats,
            );
        }

        if clear_depth.is_some() {
            self.device.invalidate_depth_target();
        }
    }

    fn draw_blurs(
        &mut self,
        blurs: &FastHashMap<TextureSource, Vec<BlurInstance>>,
        stats: &mut RendererStats,
    ) {
        for (texture, blurs) in blurs {
            let textures = BatchTextures::composite_rgb(
                *texture,
            );

            self.draw_instanced_batch(
                blurs,
                VertexArrayKind::Blur,
                &textures,
                stats,
            );
        }
    }

    /// Draw all the instances in a clip batcher list to the current target.
    fn draw_clip_batch_list(
        &mut self,
        list: &ClipBatchList,
        projection: &default::Transform3D<f32>,
        stats: &mut RendererStats,
    ) {
        if self.debug_flags.contains(DebugFlags::DISABLE_CLIP_MASKS) {
            return;
        }

        // draw rounded cornered rectangles
        if !list.slow_rectangles.is_empty() {
            let _gm2 = self.gpu_profiler.start_marker("slow clip rectangles");
            self.shaders.borrow_mut().cs_clip_rectangle_slow.bind(
                &mut self.device,
                projection,
                &mut self.renderer_errors,
            );
            self.draw_instanced_batch(
                &list.slow_rectangles,
                VertexArrayKind::ClipRect,
                &BatchTextures::empty(),
                stats,
            );
        }
        if !list.fast_rectangles.is_empty() {
            let _gm2 = self.gpu_profiler.start_marker("fast clip rectangles");
            self.shaders.borrow_mut().cs_clip_rectangle_fast.bind(
                &mut self.device,
                projection,
                &mut self.renderer_errors,
            );
            self.draw_instanced_batch(
                &list.fast_rectangles,
                VertexArrayKind::ClipRect,
                &BatchTextures::empty(),
                stats,
            );
        }

        // draw box-shadow clips
        for (mask_texture_id, items) in list.box_shadows.iter() {
            let _gm2 = self.gpu_profiler.start_marker("box-shadows");
            let textures = BatchTextures::composite_rgb(*mask_texture_id);
            self.shaders.borrow_mut().cs_clip_box_shadow
                .bind(&mut self.device, projection, &mut self.renderer_errors);
            self.draw_instanced_batch(
                items,
                VertexArrayKind::ClipBoxShadow,
                &textures,
                stats,
            );
        }

        // draw image masks
        for (mask_texture_id, items) in list.images.iter() {
            let _gm2 = self.gpu_profiler.start_marker("clip images");
            let textures = BatchTextures::composite_rgb(*mask_texture_id);
            self.shaders.borrow_mut().cs_clip_image
                .bind(&mut self.device, projection, &mut self.renderer_errors);
            self.draw_instanced_batch(
                items,
                VertexArrayKind::ClipImage,
                &textures,
                stats,
            );
        }
    }

    fn draw_alpha_target(
        &mut self,
        draw_target: DrawTarget,
        target: &AlphaRenderTarget,
        projection: &default::Transform3D<f32>,
        render_tasks: &RenderTaskGraph,
        stats: &mut RendererStats,
    ) {
        profile_scope!("draw_alpha_target");

        self.profile.inc(profiler::ALPHA_PASSES);
        let _gm = self.gpu_profiler.start_marker("alpha target");
        let alpha_sampler = self.gpu_profiler.start_sampler(GPU_SAMPLER_TAG_ALPHA);

        {
            let _timer = self.gpu_profiler.start_timer(GPU_TAG_SETUP_TARGET);
            self.device.bind_draw_target(draw_target);
            self.device.disable_depth();
            self.device.disable_depth_write();
            self.set_blend(false, FramebufferKind::Other);

            // TODO(gw): Applying a scissor rect and minimal clear here
            // is a very large performance win on the Intel and nVidia
            // GPUs that I have tested with. It's possible it may be a
            // performance penalty on other GPU types - we should test this
            // and consider different code paths.

            let zero_color = [0.0, 0.0, 0.0, 0.0];
            for &task_id in &target.zero_clears {
                let (rect, _) = render_tasks[task_id].get_target_rect();
                self.device.clear_target(
                    Some(zero_color),
                    None,
                    Some(draw_target.to_framebuffer_rect(rect)),
                );
            }

            let one_color = [1.0, 1.0, 1.0, 1.0];
            for &task_id in &target.one_clears {
                let (rect, _) = render_tasks[task_id].get_target_rect();
                self.device.clear_target(
                    Some(one_color),
                    None,
                    Some(draw_target.to_framebuffer_rect(rect)),
                );
            }
        }

        // Draw any blurs for this target.
        // Blurs are rendered as a standard 2-pass
        // separable implementation.
        // TODO(gw): In the future, consider having
        //           fast path blur shaders for common
        //           blur radii with fixed weights.
        if !target.vertical_blurs.is_empty() || !target.horizontal_blurs.is_empty() {
            let _timer = self.gpu_profiler.start_timer(GPU_TAG_BLUR);

            self.shaders.borrow_mut().cs_blur_a8
                .bind(&mut self.device, projection, &mut self.renderer_errors);

            if !target.vertical_blurs.is_empty() {
                self.draw_blurs(
                    &target.vertical_blurs,
                    stats,
                );
            }

            if !target.horizontal_blurs.is_empty() {
                self.draw_blurs(
                    &target.horizontal_blurs,
                    stats,
                );
            }
        }

        self.handle_scaling(
            &target.scalings,
            projection,
            stats,
        );

        // Draw the clip items into the tiled alpha mask.
        {
            let _timer = self.gpu_profiler.start_timer(GPU_TAG_CACHE_CLIP);

            // TODO(gw): Consider grouping multiple clip masks per shader
            //           invocation here to reduce memory bandwith further?

            // Draw the primary clip mask - since this is the first mask
            // for the task, we can disable blending, knowing that it will
            // overwrite every pixel in the mask area.
            self.set_blend(false, FramebufferKind::Other);
            self.draw_clip_batch_list(
                &target.clip_batcher.primary_clips,
                projection,
                stats,
            );

            // switch to multiplicative blending for secondary masks, using
            // multiplicative blending to accumulate clips into the mask.
            self.set_blend(true, FramebufferKind::Other);
            self.set_blend_mode_multiply(FramebufferKind::Other);
            self.draw_clip_batch_list(
                &target.clip_batcher.secondary_clips,
                projection,
                stats,
            );
        }

        self.gpu_profiler.finish_sampler(alpha_sampler);
    }

    fn draw_texture_cache_target(
        &mut self,
        texture: &CacheTextureId,
        layer: LayerIndex,
        target: &TextureCacheRenderTarget,
        render_tasks: &RenderTaskGraph,
        stats: &mut RendererStats,
    ) {
        profile_scope!("draw_texture_cache_target");

        self.device.disable_depth();
        self.device.disable_depth_write();

        self.set_blend(false, FramebufferKind::Other);

        let texture = &self.texture_resolver.texture_cache_map[texture];
        let target_size = texture.get_dimensions();

        let projection = Transform3D::ortho(
            0.0,
            target_size.width as f32,
            0.0,
            target_size.height as f32,
            self.device.ortho_near_plane(),
            self.device.ortho_far_plane(),
        );

        let draw_target = DrawTarget::from_texture(
            texture,
            layer,
            false,
        );
        self.device.bind_draw_target(draw_target);

        {
            let _timer = self.gpu_profiler.start_timer(GPU_TAG_CLEAR);

            self.device.disable_depth();
            self.device.disable_depth_write();
            self.set_blend(false, FramebufferKind::Other);

            let color = [0.0, 0.0, 0.0, 0.0];
            if self.clear_caches_with_quads && !target.clears.is_empty() {
                let instances = target.clears
                    .iter()
                    .map(|r| ClearInstance {
                        rect: [
                            r.origin.x as f32, r.origin.y as f32,
                            r.size.width as f32, r.size.height as f32,
                        ],
                        color,
                    })
                    .collect::<Vec<_>>();
                self.shaders.borrow_mut().ps_clear.bind(
                    &mut self.device,
                    &projection,
                    &mut self.renderer_errors,
                );
                self.draw_instanced_batch(
                    &instances,
                    VertexArrayKind::Clear,
                    &BatchTextures::empty(),
                    stats,
                );
            } else {
                for rect in &target.clears {
                    self.device.clear_target(
                        Some(color),
                        None,
                        Some(draw_target.to_framebuffer_rect(*rect)),
                    );
                }
            }

            // Handle any blits to this texture from child tasks.
            self.handle_blits(
                &target.blits,
                render_tasks,
                draw_target,
            );
        }

        // Draw any borders for this target.
        if !target.border_segments_solid.is_empty() ||
           !target.border_segments_complex.is_empty()
        {
            let _timer = self.gpu_profiler.start_timer(GPU_TAG_CACHE_BORDER);

            self.set_blend(true, FramebufferKind::Other);
            self.set_blend_mode_premultiplied_alpha(FramebufferKind::Other);

            if !target.border_segments_solid.is_empty() {
                self.shaders.borrow_mut().cs_border_solid.bind(
                    &mut self.device,
                    &projection,
                    &mut self.renderer_errors,
                );

                self.draw_instanced_batch(
                    &target.border_segments_solid,
                    VertexArrayKind::Border,
                    &BatchTextures::empty(),
                    stats,
                );
            }

            if !target.border_segments_complex.is_empty() {
                self.shaders.borrow_mut().cs_border_segment.bind(
                    &mut self.device,
                    &projection,
                    &mut self.renderer_errors,
                );

                self.draw_instanced_batch(
                    &target.border_segments_complex,
                    VertexArrayKind::Border,
                    &BatchTextures::empty(),
                    stats,
                );
            }

            self.set_blend(false, FramebufferKind::Other);
        }

        // Draw any line decorations for this target.
        if !target.line_decorations.is_empty() {
            let _timer = self.gpu_profiler.start_timer(GPU_TAG_CACHE_LINE_DECORATION);

            self.set_blend(true, FramebufferKind::Other);
            self.set_blend_mode_premultiplied_alpha(FramebufferKind::Other);

            self.shaders.borrow_mut().cs_line_decoration.bind(
                &mut self.device,
                &projection,
                &mut self.renderer_errors,
            );

            self.draw_instanced_batch(
                &target.line_decorations,
                VertexArrayKind::LineDecoration,
                &BatchTextures::empty(),
                stats,
            );

            self.set_blend(false, FramebufferKind::Other);
        }

        // Draw any gradients for this target.
        if !target.gradients.is_empty() {
            let _timer = self.gpu_profiler.start_timer(GPU_TAG_CACHE_GRADIENT);

            self.set_blend(false, FramebufferKind::Other);

            self.shaders.borrow_mut().cs_gradient.bind(
                &mut self.device,
                &projection,
                &mut self.renderer_errors,
            );

            self.draw_instanced_batch(
                &target.gradients,
                VertexArrayKind::Gradient,
                &BatchTextures::empty(),
                stats,
            );
        }

        // Draw any blurs for this target.
        if !target.horizontal_blurs.is_empty() {
            let _timer = self.gpu_profiler.start_timer(GPU_TAG_BLUR);

            {
                let mut shaders = self.shaders.borrow_mut();
                match target.target_kind {
                    RenderTargetKind::Alpha => &mut shaders.cs_blur_a8,
                    RenderTargetKind::Color => &mut shaders.cs_blur_rgba8,
                }.bind(&mut self.device, &projection, &mut self.renderer_errors);
            }

            self.draw_blurs(
                &target.horizontal_blurs,
                stats,
            );
        }
    }

    fn update_deferred_resolves(&mut self, deferred_resolves: &[DeferredResolve]) -> Option<GpuCacheUpdateList> {
        // The first thing we do is run through any pending deferred
        // resolves, and use a callback to get the UV rect for this
        // custom item. Then we patch the resource_rects structure
        // here before it's uploaded to the GPU.
        if deferred_resolves.is_empty() {
            return None;
        }

        let handler = self.external_image_handler
            .as_mut()
            .expect("Found external image, but no handler set!");

        let mut list = GpuCacheUpdateList {
            frame_id: FrameId::INVALID,
            clear: false,
            height: self.gpu_cache_texture.get_height(),
            blocks: Vec::new(),
            updates: Vec::new(),
            debug_commands: Vec::new(),
        };

        for (i, deferred_resolve) in deferred_resolves.iter().enumerate() {
            self.gpu_profiler.place_marker("deferred resolve");
            let props = &deferred_resolve.image_properties;
            let ext_image = props
                .external_image
                .expect("BUG: Deferred resolves must be external images!");
            // Provide rendering information for NativeTexture external images.
            let image = handler.lock(ext_image.id, ext_image.channel_index, deferred_resolve.rendering);
            let texture_target = match ext_image.image_type {
                ExternalImageType::TextureHandle(target) => target,
                ExternalImageType::Buffer => {
                    panic!("not a suitable image type in update_deferred_resolves()");
                }
            };

            // In order to produce the handle, the external image handler may call into
            // the GL context and change some states.
            self.device.reset_state();

            let texture = match image.source {
                ExternalImageSource::NativeTexture(texture_id) => {
                    ExternalTexture::new(
                        texture_id,
                        texture_target,
                        Swizzle::default(),
                        image.uv,
                    )
                }
                ExternalImageSource::Invalid => {
                    warn!("Invalid ext-image");
                    debug!(
                        "For ext_id:{:?}, channel:{}.",
                        ext_image.id,
                        ext_image.channel_index
                    );
                    // Just use 0 as the gl handle for this failed case.
                    ExternalTexture::new(
                        0,
                        texture_target,
                        Swizzle::default(),
                        image.uv,
                    )
                }
                ExternalImageSource::RawData(_) => {
                    panic!("Raw external data is not expected for deferred resolves!");
                }
            };

            self.texture_resolver
                .external_images
                .insert(DeferredResolveIndex(i as u32), texture);

            list.updates.push(GpuCacheUpdate::Copy {
                block_index: list.blocks.len(),
                block_count: BLOCKS_PER_UV_RECT,
                address: deferred_resolve.address,
            });
            list.blocks.push(image.uv.into());
            list.blocks.push([0f32; 4].into());
        }

        Some(list)
    }

    fn unlock_external_images(
        &mut self,
        deferred_resolves: &[DeferredResolve],
    ) {
        if !self.texture_resolver.external_images.is_empty() {
            let handler = self.external_image_handler
                .as_mut()
                .expect("Found external image, but no handler set!");

            for (index, _) in self.texture_resolver.external_images.drain() {
                let props = &deferred_resolves[index.0 as usize].image_properties;
                let ext_image = props
                    .external_image
                    .expect("BUG: Deferred resolves must be external images!");
                handler.unlock(ext_image.id, ext_image.channel_index);
            }
        }
    }

    /// Update the dirty rects based on current compositing mode and config
    // TODO(gw): This can be tidied up significantly once the Draw compositor
    //           is implemented in terms of the compositor trait.
    fn calculate_dirty_rects(
        &mut self,
        buffer_age: usize,
        composite_state: &CompositeState,
        draw_target_dimensions: DeviceIntSize,
        results: &mut RenderResults,
    ) -> Option<PartialPresentMode> {
        let mut partial_present_mode = None;

        let (max_partial_present_rects, draw_previous_partial_present_regions) = match self.current_compositor_kind {
            CompositorKind::Native { .. } => {
                // Assume that we can return a single dirty rect for native
                // compositor for now, and that there is no buffer-age functionality.
                // These params can be exposed by the compositor capabilities struct
                // as the Draw compositor is ported to use it.
                (1, false)
            }
            CompositorKind::Draw { draw_previous_partial_present_regions, max_partial_present_rects } => {
                (max_partial_present_rects, draw_previous_partial_present_regions)
            }
        };

        if max_partial_present_rects > 0 {
            let prev_frames_damage_rect = if let Some(..) = self.compositor_config.partial_present() {
                self.buffer_damage_tracker
                    .get_damage_rect(buffer_age)
                    .or_else(|| Some(DeviceRect::from_size(draw_target_dimensions.to_f32())))
            } else {
                None
            };

            let can_use_partial_present =
                composite_state.dirty_rects_are_valid &&
                !self.force_redraw &&
                !(prev_frames_damage_rect.is_none() && draw_previous_partial_present_regions) &&
                !self.debug_overlay_state.is_enabled;

            if can_use_partial_present {
                let mut combined_dirty_rect = DeviceRect::zero();

                // Work out how many dirty rects WR produced, and if that's more than
                // what the device supports.
                for tile in composite_state.opaque_tiles.iter().chain(composite_state.alpha_tiles.iter()) {
                    let tile_dirty_rect = tile.dirty_rect.translate(tile.rect.origin.to_vector());
                    let transformed_dirty_rect = if let Some(transform) = tile.transform {
                        transform.outer_transformed_rect(&tile_dirty_rect)
                    } else {
                        Some(tile_dirty_rect)
                    };

                    if let Some(dirty_rect) = transformed_dirty_rect {
                        combined_dirty_rect = combined_dirty_rect.union(&dirty_rect);
                    }
                }

                let combined_dirty_rect = combined_dirty_rect.round();
                let combined_dirty_rect_i32 = combined_dirty_rect.to_i32();
                // Return this frame's dirty region. If nothing has changed, don't return any dirty
                // rects at all (the client can use this as a signal to skip present completely).
                if !combined_dirty_rect.is_empty() {
                    results.dirty_rects.push(combined_dirty_rect_i32);
                }

                // Track this frame's dirty region, for calculating subsequent frames' damage.
                if draw_previous_partial_present_regions {
                    self.buffer_damage_tracker.push_dirty_rect(&combined_dirty_rect);
                }

                // If the implementation requires manually keeping the buffer consistent,
                // then we must combine this frame's dirty region with that of previous frames
                // to determine the total_dirty_rect. The is used to determine what region we
                // render to, and is what we send to the compositor as the buffer damage region
                // (eg for KHR_partial_update).
                let total_dirty_rect = if draw_previous_partial_present_regions {
                    combined_dirty_rect.union(&prev_frames_damage_rect.unwrap())
                } else {
                    combined_dirty_rect
                };

                partial_present_mode = Some(PartialPresentMode::Single {
                    dirty_rect: total_dirty_rect,
                });

                if let Some(partial_present) = self.compositor_config.partial_present() {
                    partial_present.set_buffer_damage_region(&[total_dirty_rect.to_i32()]);
                }
            } else {
                // If we don't have a valid partial present scenario, return a single
                // dirty rect to the client that covers the entire framebuffer.
                let fb_rect = DeviceIntRect::new(
                    DeviceIntPoint::zero(),
                    draw_target_dimensions,
                );
                results.dirty_rects.push(fb_rect);

                if draw_previous_partial_present_regions {
                    self.buffer_damage_tracker.push_dirty_rect(&fb_rect.to_f32());
                }
            }

            self.force_redraw = false;
        }

        partial_present_mode
    }

    fn bind_frame_data(&mut self, frame: &mut Frame) {
        profile_scope!("bind_frame_data");

        let _timer = self.gpu_profiler.start_timer(GPU_TAG_SETUP_DATA);

        self.vertex_data_textures[self.current_vertex_data_textures].update(
            &mut self.device,
            &mut self.texture_upload_pbo_pool,
            frame,
        );
        self.current_vertex_data_textures =
            (self.current_vertex_data_textures + 1) % VERTEX_DATA_TEXTURE_COUNT;
    }

    fn update_native_surfaces(&mut self) {
        profile_scope!("update_native_surfaces");

        match self.compositor_config {
            CompositorConfig::Native { ref mut compositor, .. } => {
                for op in self.pending_native_surface_updates.drain(..) {
                    match op.details {
                        NativeSurfaceOperationDetails::CreateSurface { id, virtual_offset, tile_size, is_opaque } => {
                            let _inserted = self.allocated_native_surfaces.insert(id);
                            debug_assert!(_inserted, "bug: creating existing surface");
                            compositor.create_surface(
                                    id,
                                    virtual_offset,
                                    tile_size,
                                    is_opaque,
                            );
                        }
                        NativeSurfaceOperationDetails::CreateExternalSurface { id, is_opaque } => {
                            let _inserted = self.allocated_native_surfaces.insert(id);
                            debug_assert!(_inserted, "bug: creating existing surface");
                            compositor.create_external_surface(
                                id,
                                is_opaque,
                            );
                        }
                        NativeSurfaceOperationDetails::DestroySurface { id } => {
                            let _existed = self.allocated_native_surfaces.remove(&id);
                            debug_assert!(_existed, "bug: removing unknown surface");
                            compositor.destroy_surface(id);
                        }
                        NativeSurfaceOperationDetails::CreateTile { id } => {
                            compositor.create_tile(id);
                        }
                        NativeSurfaceOperationDetails::DestroyTile { id } => {
                            compositor.destroy_tile(id);
                        }
                        NativeSurfaceOperationDetails::AttachExternalImage { id, external_image } => {
                            compositor.attach_external_image(id, external_image);
                        }
                    }
                }
            }
            CompositorConfig::Draw { .. } => {
                // Ensure nothing is added in simple composite mode, since otherwise
                // memory will leak as this doesn't get drained
                debug_assert!(self.pending_native_surface_updates.is_empty());
            }
        }
    }

    fn draw_frame(
        &mut self,
        frame: &mut Frame,
        device_size: Option<DeviceIntSize>,
        buffer_age: usize,
        results: &mut RenderResults,
    ) {
        profile_scope!("draw_frame");

        // These markers seem to crash a lot on Android, see bug 1559834
        #[cfg(not(target_os = "android"))]
        let _gm = self.gpu_profiler.start_marker("draw frame");

        if frame.passes.is_empty() {
            frame.has_been_rendered = true;
            return;
        }

        self.device.disable_depth_write();
        self.set_blend(false, FramebufferKind::Other);
        self.device.disable_stencil();

        self.bind_frame_data(frame);

        // Determine the present mode and dirty rects, if device_size
        // is Some(..). If it's None, no composite will occur and only
        // picture cache and texture cache targets will be updated.
        // TODO(gw): Split Frame so that it's clearer when a composite
        //           is occurring.
        let present_mode = device_size.and_then(|device_size| {
            self.calculate_dirty_rects(
                buffer_age,
                &frame.composite_state,
                device_size,
                results,
            )
        });

        // If we have a native OS compositor, then make use of that interface to
        // specify how to composite each of the picture cache surfaces. First, we
        // need to find each tile that may be bound and updated later in the frame
        // and invalidate it so that the native render compositor knows that these
        // tiles can't be composited early. Next, after all such tiles have been
        // invalidated, then we queue surfaces for native composition by the render
        // compositor before we actually update the tiles. This allows the render
        // compositor to start early composition while the tiles are updating.
        if let CompositorKind::Native { .. } = self.current_compositor_kind {
            let compositor = self.compositor_config.compositor().unwrap();
            // Invalidate any native surface tiles that might be updated by passes.
            if !frame.has_been_rendered {
                for tile in frame.composite_state.opaque_tiles.iter().chain(frame.composite_state.alpha_tiles.iter()) {
                    if !tile.dirty_rect.is_empty() {
                        if let CompositeTileSurface::Texture { surface: ResolvedSurfaceTexture::Native { id, .. } } =
                            tile.surface {
                            compositor.invalidate_tile(id);
                        }
                    }
                }
            }
            // Ensure any external surfaces that might be used during early composition
            // are invalidated first so that the native compositor can properly schedule
            // composition to happen only when the external surface is updated.
            // See update_external_native_surfaces for more details.
            for surface in &frame.composite_state.external_surfaces {
                if let Some((native_surface_id, _)) = surface.update_params {
                    compositor.invalidate_tile(NativeTileId { surface_id: native_surface_id, x: 0, y: 0 });
                }
            }
            // Finally queue native surfaces for early composition, if applicable. By now,
            // we have already invalidated any tiles that such surfaces may depend upon, so
            // the native render compositor can keep track of when to actually schedule
            // composition as surfaces are updated.
            frame.composite_state.composite_native(
                &results.dirty_rects,
                &mut **compositor,
            );
        }

        for (_pass_index, pass) in frame.passes.iter_mut().enumerate() {
            #[cfg(not(target_os = "android"))]
            let _gm = self.gpu_profiler.start_marker(&format!("pass {}", _pass_index));

            profile_scope!("offscreen target");

            // If this frame has already been drawn, then any texture
            // cache targets have already been updated and can be
            // skipped this time.
            if !frame.has_been_rendered {
                for (&(texture_id, target_index), target) in &pass.texture_cache {
                    self.draw_texture_cache_target(
                        &texture_id,
                        target_index,
                        target,
                        &frame.render_tasks,
                        &mut results.stats,
                    );
                }

                if !pass.picture_cache.is_empty() {
                    self.profile.inc(profiler::COLOR_PASSES);
                }

                // Draw picture caching tiles for this pass.
                for picture_target in &pass.picture_cache {
                    results.stats.color_target_count += 1;

                    let draw_target = match picture_target.surface {
                        ResolvedSurfaceTexture::TextureCache { ref texture, layer } => {
                            let (texture, _) = self.texture_resolver
                                .resolve(texture)
                                .expect("bug");

                            DrawTarget::from_texture(
                                texture,
                                layer as usize,
                                true,
                            )
                        }
                        ResolvedSurfaceTexture::Native { id, size } => {
                            let surface_info = match self.current_compositor_kind {
                                CompositorKind::Native { .. } => {
                                    let compositor = self.compositor_config.compositor().unwrap();
                                    compositor.bind(
                                        id,
                                        picture_target.dirty_rect,
                                        picture_target.valid_rect,
                                    )
                                }
                                CompositorKind::Draw { .. } => {
                                    unreachable!();
                                }
                            };

                            DrawTarget::NativeSurface {
                                offset: surface_info.origin,
                                external_fbo_id: surface_info.fbo_id,
                                dimensions: size,
                            }
                        }
                    };

                    let projection = Transform3D::ortho(
                        0.0,
                        draw_target.dimensions().width as f32,
                        0.0,
                        draw_target.dimensions().height as f32,
                        self.device.ortho_near_plane(),
                        self.device.ortho_far_plane(),
                    );

                    self.draw_picture_cache_target(
                        picture_target,
                        draw_target,
                        &projection,
                        &frame.render_tasks,
                        &mut results.stats,
                    );

                    // Native OS surfaces must be unbound at the end of drawing to them
                    if let ResolvedSurfaceTexture::Native { .. } = picture_target.surface {
                        match self.current_compositor_kind {
                            CompositorKind::Native { .. } => {
                                let compositor = self.compositor_config.compositor().unwrap();
                                compositor.unbind();
                            }
                            CompositorKind::Draw { .. } => {
                                unreachable!();
                            }
                        }
                    }
                }
            }

            for target in &pass.alpha.targets {
                results.stats.alpha_target_count += 1;

                let texture_id = target.texture_id();

                let alpha_tex = self.texture_resolver
                    .texture_cache_map
                    .get_mut(&texture_id)
                    .expect("bug: texture not allocated");

                let draw_target = DrawTarget::from_texture(
                    alpha_tex,
                    0,
                    false,
                );

                let projection = Transform3D::ortho(
                    0.0,
                    draw_target.dimensions().width as f32,
                    0.0,
                    draw_target.dimensions().height as f32,
                    self.device.ortho_near_plane(),
                    self.device.ortho_far_plane(),
                );

                self.draw_alpha_target(
                    draw_target,
                    target,
                    &projection,
                    &frame.render_tasks,
                    &mut results.stats,
                );
            }

            let color_rt_info = RenderTargetInfo { has_depth: pass.color.needs_depth() };

            for target in &pass.color.targets {
                results.stats.color_target_count += 1;

                let texture_id = target.texture_id();

                let color_tex = self.texture_resolver
                    .texture_cache_map
                    .get_mut(&texture_id)
                    .expect("bug: texture not allocated");

                self.device.reuse_render_target::<u8>(
                    color_tex,
                    color_rt_info,
                );

                let draw_target = DrawTarget::from_texture(
                    color_tex,
                    0,
                    target.needs_depth(),
                );

                let projection = Transform3D::ortho(
                    0.0,
                    draw_target.dimensions().width as f32,
                    0.0,
                    draw_target.dimensions().height as f32,
                    self.device.ortho_near_plane(),
                    self.device.ortho_far_plane(),
                );

                let clear_depth = if target.needs_depth() {
                    Some(1.0)
                } else {
                    None
                };

                self.draw_color_target(
                    draw_target,
                    target,
                    Some([0.0, 0.0, 0.0, 0.0]),
                    clear_depth,
                    &frame.render_tasks,
                    &projection,
                    &mut results.stats,
                );
            }

            // Only end the pass here and invalidate previous textures for
            // off-screen targets. Deferring return of the inputs to the
            // frame buffer until the implicit end_pass in end_frame allows
            // debug draw overlays to be added without triggering a copy
            // resolve stage in mobile / tiled GPUs.
            self.texture_resolver.end_pass(
                &mut self.device,
                &pass.textures_to_invalidate,
            );
            {
                profile_scope!("gl.flush");
                self.device.gl().flush();
            }
        }

        self.composite_frame(
            frame,
            device_size,
            results,
            present_mode,
        );

        if let Some(device_size) = device_size {
            self.draw_frame_debug_items(&frame.debug_items);
            self.draw_render_target_debug(device_size);
            self.draw_texture_cache_debug(device_size);
            self.draw_gpu_cache_debug(device_size);
            self.draw_zoom_debug(device_size);
        }
        self.draw_epoch_debug();

        frame.has_been_rendered = true;
    }

    fn composite_frame(
        &mut self,
        frame: &mut Frame,
        device_size: Option<DeviceIntSize>,
        results: &mut RenderResults,
        present_mode: Option<PartialPresentMode>,
    ) {
        profile_scope!("main target");

        if let Some(device_size) = device_size {
            results.stats.color_target_count += 1;
            results.picture_cache_debug = mem::replace(
                &mut frame.composite_state.picture_cache_debug,
                PictureCacheDebugInfo::new(),
            );

            let size = frame.device_rect.size.to_f32();
            let surface_origin_is_top_left = self.device.surface_origin_is_top_left();
            let (bottom, top) = if surface_origin_is_top_left {
              (0.0, size.height)
            } else {
              (size.height, 0.0)
            };

            let projection = Transform3D::ortho(
                0.0,
                size.width,
                bottom,
                top,
                self.device.ortho_near_plane(),
                self.device.ortho_far_plane(),
            );

            let fb_scale = Scale::<_, _, FramebufferPixel>::new(1i32);
            let mut fb_rect = frame.device_rect * fb_scale;

            if !surface_origin_is_top_left {
                fb_rect.origin.y = device_size.height - fb_rect.origin.y - fb_rect.size.height;
            }

            let draw_target = DrawTarget::Default {
                rect: fb_rect,
                total_size: device_size * fb_scale,
                surface_origin_is_top_left,
            };

            // If we have a native OS compositor, then make use of that interface
            // to specify how to composite each of the picture cache surfaces.
            match self.current_compositor_kind {
                CompositorKind::Native { .. } => {
                    // We have already queued surfaces for early native composition by this point.
                    // All that is left is to finally update any external native surfaces that were
                    // invalidated so that composition can complete.
                    self.update_external_native_surfaces(
                        &frame.composite_state.external_surfaces,
                        results,
                    );
                }
                CompositorKind::Draw { .. } => {
                    self.composite_simple(
                        &frame.composite_state,
                        draw_target,
                        &projection,
                        results,
                        present_mode,
                    );
                }
            }
        } else {
            // Rendering a frame without presenting it will confuse the partial
            // present logic, so force a full present for the next frame.
            self.force_redraw();
        }
    }

    pub fn debug_renderer(&mut self) -> Option<&mut DebugRenderer> {
        self.debug.get_mut(&mut self.device)
    }

    pub fn get_debug_flags(&self) -> DebugFlags {
        self.debug_flags
    }

    pub fn set_debug_flags(&mut self, flags: DebugFlags) {
        if let Some(enabled) = flag_changed(self.debug_flags, flags, DebugFlags::GPU_TIME_QUERIES) {
            if enabled {
                self.gpu_profiler.enable_timers();
            } else {
                self.gpu_profiler.disable_timers();
            }
        }
        if let Some(enabled) = flag_changed(self.debug_flags, flags, DebugFlags::GPU_SAMPLE_QUERIES) {
            if enabled {
                self.gpu_profiler.enable_samplers();
            } else {
                self.gpu_profiler.disable_samplers();
            }
        }

        self.debug_flags = flags;
    }

    pub fn set_profiler_ui(&mut self, ui_str: &str) {
        self.profiler.set_ui(ui_str);
    }

    fn draw_frame_debug_items(&mut self, items: &[DebugItem]) {
        if items.is_empty() {
            return;
        }

        let debug_renderer = match self.debug.get_mut(&mut self.device) {
            Some(render) => render,
            None => return,
        };

        for item in items {
            match item {
                DebugItem::Rect { rect, outer_color, inner_color } => {
                    debug_renderer.add_quad(
                        rect.origin.x,
                        rect.origin.y,
                        rect.origin.x + rect.size.width,
                        rect.origin.y + rect.size.height,
                        (*inner_color).into(),
                        (*inner_color).into(),
                    );

                    debug_renderer.add_rect(
                        &rect.to_i32(),
                        (*outer_color).into(),
                    );
                }
                DebugItem::Text { ref msg, position, color } => {
                    debug_renderer.add_text(
                        position.x,
                        position.y,
                        msg,
                        (*color).into(),
                        None,
                    );
                }
            }
        }
    }

    fn draw_render_target_debug(&mut self, device_size: DeviceIntSize) {
        if !self.debug_flags.contains(DebugFlags::RENDER_TARGET_DBG) {
            return;
        }

        let debug_renderer = match self.debug.get_mut(&mut self.device) {
            Some(render) => render,
            None => return,
        };

        let textures = self.texture_resolver
            .texture_cache_map
            .values()
            .filter(|texture| { texture.is_render_target() })
            .collect::<Vec<&Texture>>();

        Self::do_debug_blit(
            &mut self.device,
            debug_renderer,
            textures,
            device_size,
            0,
            &|_| [0.0, 1.0, 0.0, 1.0], // Use green for all RTs.
        );
    }

    fn draw_zoom_debug(
        &mut self,
        device_size: DeviceIntSize,
    ) {
        if !self.debug_flags.contains(DebugFlags::ZOOM_DBG) {
            return;
        }

        let debug_renderer = match self.debug.get_mut(&mut self.device) {
            Some(render) => render,
            None => return,
        };

        let source_size = DeviceIntSize::new(64, 64);
        let target_size = DeviceIntSize::new(1024, 1024);

        let source_origin = DeviceIntPoint::new(
            (self.cursor_position.x - source_size.width / 2)
                .min(device_size.width - source_size.width)
                .max(0),
            (self.cursor_position.y - source_size.height / 2)
                .min(device_size.height - source_size.height)
                .max(0),
        );

        let source_rect = DeviceIntRect::new(
            source_origin,
            source_size,
        );

        let target_rect = DeviceIntRect::new(
            DeviceIntPoint::new(
                device_size.width - target_size.width - 64,
                device_size.height - target_size.height - 64,
            ),
            target_size,
        );

        let texture_rect = FramebufferIntRect::new(
            FramebufferIntPoint::zero(),
            source_rect.size.cast_unit(),
        );

        debug_renderer.add_rect(
            &target_rect.inflate(1, 1),
            debug_colors::RED.into(),
        );

        if self.zoom_debug_texture.is_none() {
            let texture = self.device.create_texture(
                ImageBufferKind::Texture2D,
                ImageFormat::BGRA8,
                source_rect.size.width,
                source_rect.size.height,
                TextureFilter::Nearest,
                Some(RenderTargetInfo { has_depth: false }),
                1,
            );

            self.zoom_debug_texture = Some(texture);
        }

        // Copy frame buffer into the zoom texture
        let read_target = DrawTarget::new_default(device_size, self.device.surface_origin_is_top_left());
        self.device.blit_render_target(
            read_target.into(),
            read_target.to_framebuffer_rect(source_rect),
            DrawTarget::from_texture(
                self.zoom_debug_texture.as_ref().unwrap(),
                0,
                false,
            ),
            texture_rect,
            TextureFilter::Nearest,
        );

        // Draw the zoom texture back to the framebuffer
        self.device.blit_render_target(
            ReadTarget::from_texture(
                self.zoom_debug_texture.as_ref().unwrap(),
                0,
            ),
            texture_rect,
            read_target,
            read_target.to_framebuffer_rect(target_rect),
            TextureFilter::Nearest,
        );
    }

    fn draw_texture_cache_debug(&mut self, device_size: DeviceIntSize) {
        if !self.debug_flags.contains(DebugFlags::TEXTURE_CACHE_DBG) {
            return;
        }

        let debug_renderer = match self.debug.get_mut(&mut self.device) {
            Some(render) => render,
            None => return,
        };

        let textures =
            self.texture_resolver.texture_cache_map.values().collect::<Vec<&Texture>>();

        fn select_color(texture: &Texture) -> [f32; 4] {
            if texture.flags().contains(TextureFlags::IS_SHARED_TEXTURE_CACHE) {
                [1.0, 0.5, 0.0, 1.0] // Orange for shared.
            } else {
                [1.0, 0.0, 1.0, 1.0] // Fuchsia for standalone.
            }
        }

        Self::do_debug_blit(
            &mut self.device,
            debug_renderer,
            textures,
            device_size,
            if self.debug_flags.contains(DebugFlags::RENDER_TARGET_DBG) { 544 } else { 0 },
            &select_color,
        );
    }

    fn do_debug_blit(
        device: &mut Device,
        debug_renderer: &mut DebugRenderer,
        mut textures: Vec<&Texture>,
        device_size: DeviceIntSize,
        bottom: i32,
        select_color: &dyn Fn(&Texture) -> [f32; 4],
    ) {
        let mut spacing = 16;
        let mut size = 512;

        let fb_width = device_size.width;
        let fb_height = device_size.height;
        let num_layers: i32 = textures.iter()
            .map(|texture| texture.get_layer_count())
            .sum();

        if num_layers * (size + spacing) > fb_width {
            let factor = fb_width as f32 / (num_layers * (size + spacing)) as f32;
            size = (size as f32 * factor) as i32;
            spacing = (spacing as f32 * factor) as i32;
        }

        // Sort the display by layer size (in bytes), so that left-to-right is
        // largest-to-smallest.
        //
        // Note that the vec here is in increasing order, because the elements
        // get drawn right-to-left.
        textures.sort_by_key(|t| t.layer_size_in_bytes());

        let mut i = 0;
        for texture in textures.iter() {
            let y = spacing + bottom;
            let dimensions = texture.get_dimensions();
            let src_rect = FramebufferIntRect::new(
                FramebufferIntPoint::zero(),
                FramebufferIntSize::new(dimensions.width as i32, dimensions.height as i32),
            );

            let layer_count = texture.get_layer_count() as usize;
            for layer in 0 .. layer_count {
                let x = fb_width - (spacing + size) * (i as i32 + 1);

                // If we have more targets than fit on one row in screen, just early exit.
                if x > fb_width {
                    return;
                }

                //TODO: properly use FramebufferPixel coordinates

                // Draw the info tag.
                let text_margin = 1;
                let text_height = 14; // Visually aproximated.
                let tag_height = text_height + text_margin * 2;
                let tag_rect = rect(x, y, size, tag_height);
                let tag_color = select_color(texture);
                device.clear_target(
                    Some(tag_color),
                    None,
                    Some(tag_rect.cast_unit()),
                );

                // Draw the dimensions onto the tag.
                let dim = texture.get_dimensions();
                let mut text_rect = tag_rect;
                text_rect.origin.y =
                    fb_height - text_rect.origin.y - text_rect.size.height; // Top-relative.
                debug_renderer.add_text(
                    (x + text_margin) as f32,
                    (fb_height - y - text_margin) as f32, // Top-relative.
                    &format!("{}x{}", dim.width, dim.height),
                    ColorU::new(0, 0, 0, 255),
                    Some(text_rect.to_f32())
                );

                // Blit the contents of the layer. We need to invert Y because
                // we're blitting from a texture to the main framebuffer, which
                // use different conventions.
                let dest_rect = rect(x, y + tag_height, size, size);
                if !device.surface_origin_is_top_left() {
                    device.blit_render_target_invert_y(
                        ReadTarget::from_texture(texture, layer),
                        src_rect,
                        DrawTarget::new_default(device_size, device.surface_origin_is_top_left()),
                        FramebufferIntRect::from_untyped(&dest_rect),
                    );
                } else {
                    device.blit_render_target(
                        ReadTarget::from_texture(texture, layer),
                        src_rect,
                        DrawTarget::new_default(device_size, device.surface_origin_is_top_left()),
                        FramebufferIntRect::from_untyped(&dest_rect),
                        TextureFilter::Linear,
                    );
                }
                i += 1;
            }
        }
    }

    fn draw_epoch_debug(&mut self) {
        if !self.debug_flags.contains(DebugFlags::EPOCHS) {
            return;
        }

        let debug_renderer = match self.debug.get_mut(&mut self.device) {
            Some(render) => render,
            None => return,
        };

        let dy = debug_renderer.line_height();
        let x0: f32 = 30.0;
        let y0: f32 = 30.0;
        let mut y = y0;
        let mut text_width = 0.0;
        for ((pipeline, document_id), epoch) in  &self.pipeline_info.epochs {
            y += dy;
            let w = debug_renderer.add_text(
                x0, y,
                &format!("({:?}, {:?}): {:?}", pipeline, document_id, epoch),
                ColorU::new(255, 255, 0, 255),
                None,
            ).size.width;
            text_width = f32::max(text_width, w);
        }

        let margin = 10.0;
        debug_renderer.add_quad(
            x0 - margin,
            y0 - margin,
            x0 + text_width + margin,
            y + margin,
            ColorU::new(25, 25, 25, 200),
            ColorU::new(51, 51, 51, 200),
        );
    }

    fn draw_gpu_cache_debug(&mut self, device_size: DeviceIntSize) {
        if !self.debug_flags.contains(DebugFlags::GPU_CACHE_DBG) {
            return;
        }

        let debug_renderer = match self.debug.get_mut(&mut self.device) {
            Some(render) => render,
            None => return,
        };

        let (x_off, y_off) = (30f32, 30f32);
        let height = self.gpu_cache_texture.texture
            .as_ref().map_or(0, |t| t.get_dimensions().height)
            .min(device_size.height - (y_off as i32) * 2) as usize;
        debug_renderer.add_quad(
            x_off,
            y_off,
            x_off + MAX_VERTEX_TEXTURE_WIDTH as f32,
            y_off + height as f32,
            ColorU::new(80, 80, 80, 80),
            ColorU::new(80, 80, 80, 80),
        );

        let upper = self.gpu_cache_debug_chunks.len().min(height);
        for chunk in self.gpu_cache_debug_chunks[0..upper].iter().flatten() {
            let color = ColorU::new(250, 0, 0, 200);
            debug_renderer.add_quad(
                x_off + chunk.address.u as f32,
                y_off + chunk.address.v as f32,
                x_off + chunk.address.u as f32 + chunk.size as f32,
                y_off + chunk.address.v as f32 + 1.0,
                color,
                color,
            );
        }
    }

    /// Pass-through to `Device::read_pixels_into`, used by Gecko's WR bindings.
    pub fn read_pixels_into(&mut self, rect: FramebufferIntRect, format: ImageFormat, output: &mut [u8]) {
        self.device.read_pixels_into(rect, format, output);
    }

    pub fn read_pixels_rgba8(&mut self, rect: FramebufferIntRect) -> Vec<u8> {
        let mut pixels = vec![0; (rect.size.width * rect.size.height * 4) as usize];
        self.device.read_pixels_into(rect, ImageFormat::RGBA8, &mut pixels);
        pixels
    }

    pub fn read_gpu_cache(&mut self) -> (DeviceIntSize, Vec<u8>) {
        let texture = self.gpu_cache_texture.texture.as_ref().unwrap();
        let size = device_size_as_framebuffer_size(texture.get_dimensions());
        let mut texels = vec![0; (size.width * size.height * 16) as usize];
        self.device.begin_frame();
        self.device.bind_read_target(ReadTarget::from_texture(texture, 0));
        self.device.read_pixels_into(
            size.into(),
            ImageFormat::RGBAF32,
            &mut texels,
        );
        self.device.reset_read_target();
        self.device.end_frame();
        (texture.get_dimensions(), texels)
    }

    // De-initialize the Renderer safely, assuming the GL is still alive and active.
    pub fn deinit(mut self) {
        //Note: this is a fake frame, only needed because texture deletion is require to happen inside a frame
        self.device.begin_frame();
        // If we are using a native compositor, ensure that any remaining native
        // surfaces are freed.
        if let CompositorConfig::Native { mut compositor, .. } = self.compositor_config {
            for id in self.allocated_native_surfaces.drain() {
                compositor.destroy_surface(id);
            }
            // Destroy the debug overlay surface, if currently allocated.
            if self.debug_overlay_state.current_size.is_some() {
                compositor.destroy_surface(NativeSurfaceId::DEBUG_OVERLAY);
            }
            compositor.deinit();
        }
        self.gpu_cache_texture.deinit(&mut self.device);
        if let Some(dither_matrix_texture) = self.dither_matrix_texture {
            self.device.delete_texture(dither_matrix_texture);
        }
        if let Some(zoom_debug_texture) = self.zoom_debug_texture {
            self.device.delete_texture(zoom_debug_texture);
        }
        for textures in self.vertex_data_textures.drain(..) {
            textures.deinit(&mut self.device);
        }
        self.texture_upload_pbo_pool.deinit(&mut self.device);
        self.texture_resolver.deinit(&mut self.device);
        self.device.delete_vao(self.vaos.prim_vao);
        self.device.delete_vao(self.vaos.resolve_vao);
        self.device.delete_vao(self.vaos.clip_rect_vao);
        self.device.delete_vao(self.vaos.clip_box_shadow_vao);
        self.device.delete_vao(self.vaos.clip_image_vao);
        self.device.delete_vao(self.vaos.gradient_vao);
        self.device.delete_vao(self.vaos.blur_vao);
        self.device.delete_vao(self.vaos.line_vao);
        self.device.delete_vao(self.vaos.border_vao);
        self.device.delete_vao(self.vaos.scale_vao);
        self.device.delete_vao(self.vaos.svg_filter_vao);
        self.device.delete_vao(self.vaos.composite_vao);
        self.device.delete_vao(self.vaos.clear_vao);

        self.debug.deinit(&mut self.device);

        if let Ok(shaders) = Rc::try_unwrap(self.shaders) {
            shaders.into_inner().deinit(&mut self.device);
        }

        if let Some(async_screenshots) = self.async_screenshots.take() {
            async_screenshots.deinit(&mut self.device);
        }

        if let Some(async_frame_recorder) = self.async_frame_recorder.take() {
            async_frame_recorder.deinit(&mut self.device);
        }

        #[cfg(feature = "capture")]
        self.device.delete_fbo(self.read_fbo);
        #[cfg(feature = "replay")]
        for (_, ext) in self.owned_external_images {
            self.device.delete_external_texture(ext);
        }
        self.device.end_frame();
    }

    fn size_of<T>(&self, ptr: *const T) -> usize {
        let op = self.size_of_ops.as_ref().unwrap().size_of_op;
        unsafe { op(ptr as *const c_void) }
    }

    /// Collects a memory report.
    pub fn report_memory(&self) -> MemoryReport {
        let mut report = MemoryReport::default();

        // GPU cache CPU memory.
        if let GpuCacheBus::PixelBuffer{ref rows, ..} = self.gpu_cache_texture.bus {
            for row in rows.iter() {
                report.gpu_cache_cpu_mirror += self.size_of(&*row.cpu_blocks as *const _);
            }
        }

        // GPU cache GPU memory.
        report.gpu_cache_textures +=
            self.gpu_cache_texture.texture.as_ref().map_or(0, |t| t.size_in_bytes());

        // Render task CPU memory.
        for (_id, doc) in &self.active_documents {
            report.render_tasks += self.size_of(doc.frame.render_tasks.tasks.as_ptr());
            report.render_tasks += self.size_of(doc.frame.render_tasks.task_data.as_ptr());
        }

        // Vertex data GPU memory.
        for textures in &self.vertex_data_textures {
            report.vertex_data_textures += textures.size_in_bytes();
        }

        // Texture cache and render target GPU memory.
        report += self.texture_resolver.report_memory();

        // Texture upload PBO memory.
        report += self.texture_upload_pbo_pool.report_memory();

        // Textures held internally within the device layer.
        report += self.device.report_memory();

        report
    }

    // Sets the blend mode. Blend is unconditionally set if the "show overdraw" debugging mode is
    // enabled.
    fn set_blend(&mut self, mut blend: bool, framebuffer_kind: FramebufferKind) {
        if framebuffer_kind == FramebufferKind::Main &&
                self.debug_flags.contains(DebugFlags::SHOW_OVERDRAW) {
            blend = true
        }
        self.device.set_blend(blend)
    }

    fn set_blend_mode_multiply(&mut self, framebuffer_kind: FramebufferKind) {
        if framebuffer_kind == FramebufferKind::Main &&
                self.debug_flags.contains(DebugFlags::SHOW_OVERDRAW) {
            self.device.set_blend_mode_show_overdraw();
        } else {
            self.device.set_blend_mode_multiply();
        }
    }

    fn set_blend_mode_premultiplied_alpha(&mut self, framebuffer_kind: FramebufferKind) {
        if framebuffer_kind == FramebufferKind::Main &&
                self.debug_flags.contains(DebugFlags::SHOW_OVERDRAW) {
            self.device.set_blend_mode_show_overdraw();
        } else {
            self.device.set_blend_mode_premultiplied_alpha();
        }
    }

    fn set_blend_mode_subpixel_with_bg_color_pass1(&mut self, framebuffer_kind: FramebufferKind) {
        if framebuffer_kind == FramebufferKind::Main &&
                self.debug_flags.contains(DebugFlags::SHOW_OVERDRAW) {
            self.device.set_blend_mode_show_overdraw();
        } else {
            self.device.set_blend_mode_subpixel_with_bg_color_pass1();
        }
    }

    fn set_blend_mode_subpixel_with_bg_color_pass2(&mut self, framebuffer_kind: FramebufferKind) {
        if framebuffer_kind == FramebufferKind::Main &&
                self.debug_flags.contains(DebugFlags::SHOW_OVERDRAW) {
            self.device.set_blend_mode_show_overdraw();
        } else {
            self.device.set_blend_mode_subpixel_with_bg_color_pass2();
        }
    }

    /// Clears all the layers of a texture with a given color.
    fn clear_texture(&mut self, texture: &Texture, color: [f32; 4]) {
        for i in 0..texture.get_layer_count() {
            self.device.bind_draw_target(DrawTarget::from_texture(
                &texture,
                i as usize,
                false,
            ));
            self.device.clear_target(Some(color), None, None);
        }
    }
}

pub trait ThreadListener {
    fn thread_started(&self, thread_name: &str);
    fn thread_stopped(&self, thread_name: &str);
}

/// Allows callers to hook in at certain points of the async scene build. These
/// functions are all called from the scene builder thread.
pub trait SceneBuilderHooks {
    /// This is called exactly once, when the scene builder thread is started
    /// and before it processes anything.
    fn register(&self);
    /// This is called before each scene build starts.
    fn pre_scene_build(&self);
    /// This is called before each scene swap occurs.
    fn pre_scene_swap(&self, scenebuild_time: u64);
    /// This is called after each scene swap occurs. The PipelineInfo contains
    /// the updated epochs and pipelines removed in the new scene compared to
    /// the old scene.
    fn post_scene_swap(&self, document_id: &Vec<DocumentId>, info: PipelineInfo, sceneswap_time: u64);
    /// This is called after a resource update operation on the scene builder
    /// thread, in the case where resource updates were applied without a scene
    /// build.
    fn post_resource_update(&self, document_ids: &Vec<DocumentId>);
    /// This is called after a scene build completes without any changes being
    /// made. We guarantee that each pre_scene_build call will be matched with
    /// exactly one of post_scene_swap, post_resource_update or
    /// post_empty_scene_build.
    fn post_empty_scene_build(&self);
    /// This is a generic callback which provides an opportunity to run code
    /// on the scene builder thread. This is called as part of the main message
    /// loop of the scene builder thread, but outside of any specific message
    /// handler.
    fn poke(&self);
    /// This is called exactly once, when the scene builder thread is about to
    /// terminate.
    fn deregister(&self);
}

/// Allows callers to hook into the main render_backend loop and provide
/// additional frame ops for generate_frame transactions. These functions
/// are all called from the render backend thread.
pub trait AsyncPropertySampler {
    /// This is called exactly once, when the render backend thread is started
    /// and before it processes anything.
    fn register(&self);
    /// This is called for each transaction with the generate_frame flag set
    /// (i.e. that will trigger a render). The list of frame messages returned
    /// are processed as though they were part of the original transaction.
    fn sample(&self, document_id: DocumentId, generated_frame_id: Option<u64>) -> Vec<FrameMsg>;
    /// This is called exactly once, when the render backend thread is about to
    /// terminate.
    fn deregister(&self);
}

bitflags! {
    /// Flags that control how shaders are pre-cached, if at all.
    #[derive(Default)]
    pub struct ShaderPrecacheFlags: u32 {
        /// Needed for const initialization
        const EMPTY                 = 0;

        /// Only start async compile
        const ASYNC_COMPILE         = 1 << 2;

        /// Do a full compile/link during startup
        const FULL_COMPILE          = 1 << 3;
    }
}

pub struct RendererOptions {
    pub device_pixel_ratio: f32,
    pub resource_override_path: Option<PathBuf>,
    /// Whether to use shaders that have been optimized at build time.
    pub use_optimized_shaders: bool,
    pub enable_aa: bool,
    pub enable_dithering: bool,
    pub max_recorded_profiles: usize,
    pub precache_flags: ShaderPrecacheFlags,
    /// Enable sub-pixel anti-aliasing if a fast implementation is available.
    pub enable_subpixel_aa: bool,
    /// Enable sub-pixel anti-aliasing if it requires a slow implementation.
    pub force_subpixel_aa: bool,
    pub clear_color: Option<ColorF>,
    pub enable_clear_scissor: bool,
    pub max_texture_size: Option<i32>,
    pub upload_method: UploadMethod,
    /// The default size in bytes for PBOs used to upload texture data.
    pub upload_pbo_default_size: usize,
    pub workers: Option<Arc<ThreadPool>>,
    pub enable_multithreading: bool,
    pub blob_image_handler: Option<Box<dyn BlobImageHandler>>,
    pub crash_annotator: Option<Box<dyn CrashAnnotator>>,
    pub thread_listener: Option<Box<dyn ThreadListener + Send + Sync>>,
    pub size_of_op: Option<VoidPtrToSizeFn>,
    pub enclosing_size_of_op: Option<VoidPtrToSizeFn>,
    pub cached_programs: Option<Rc<ProgramCache>>,
    pub debug_flags: DebugFlags,
    pub renderer_id: Option<u64>,
    pub scene_builder_hooks: Option<Box<dyn SceneBuilderHooks + Send>>,
    pub sampler: Option<Box<dyn AsyncPropertySampler + Send>>,
    pub chase_primitive: ChasePrimitive,
    pub support_low_priority_transactions: bool,
    pub namespace_alloc_by_client: bool,
    pub testing: bool,
    /// Set to true if this GPU supports hardware fast clears as a performance
    /// optimization. Likely requires benchmarking on various GPUs to see if
    /// it is a performance win. The default is false, which tends to be best
    /// performance on lower end / integrated GPUs.
    pub gpu_supports_fast_clears: bool,
    pub allow_dual_source_blending: bool,
    pub allow_advanced_blend_equation: bool,
    /// If true, allow textures to be initialized with glTexStorage.
    /// This affects VRAM consumption and data upload paths.
    pub allow_texture_storage_support: bool,
    /// If true, we allow the data uploaded in a different format from the
    /// one expected by the driver, pretending the format is matching, and
    /// swizzling the components on all the shader sampling.
    pub allow_texture_swizzling: bool,
    /// Use `ps_clear` shader with batched quad rendering to clear the rects
    /// in texture cache and picture cache tasks.
    /// This helps to work around some Intel drivers
    /// that incorrectly synchronize clears to following draws.
    pub clear_caches_with_quads: bool,
    /// Start the debug server for this renderer.
    pub start_debug_server: bool,
    /// Output the source of the shader with the given name.
    pub dump_shader_source: Option<String>,
    pub surface_origin_is_top_left: bool,
    /// The configuration options defining how WR composites the final scene.
    pub compositor_config: CompositorConfig,
    pub enable_gpu_markers: bool,
    /// If true, panic whenever a GL error occurs. This has a significant
    /// performance impact, so only use when debugging specific problems!
    pub panic_on_gl_error: bool,
    pub picture_tile_size: Option<DeviceIntSize>,
    pub texture_cache_config: TextureCacheConfig,
}

impl RendererOptions {
    /// Number of batches to look back in history for adding the current
    /// transparent instance into.
    const BATCH_LOOKBACK_COUNT: usize = 10;

    /// Since we are re-initializing the instance buffers on every draw call,
    /// the driver has to internally manage PBOs in flight.
    /// It's typically done by bucketing up to a specific limit, and then
    /// just individually managing the largest buffers.
    /// Having a limit here allows the drivers to more easily manage
    /// the PBOs for us.
    const MAX_INSTANCE_BUFFER_SIZE: usize = 0x20000; // actual threshold in macOS GL drivers
}

impl Default for RendererOptions {
    fn default() -> Self {
        RendererOptions {
            device_pixel_ratio: 1.0,
            resource_override_path: None,
            use_optimized_shaders: false,
            enable_aa: true,
            enable_dithering: false,
            debug_flags: DebugFlags::empty(),
            max_recorded_profiles: 0,
            precache_flags: ShaderPrecacheFlags::empty(),
            enable_subpixel_aa: false,
            force_subpixel_aa: false,
            clear_color: Some(ColorF::new(1.0, 1.0, 1.0, 1.0)),
            enable_clear_scissor: true,
            max_texture_size: None,
            // This is best as `Immediate` on Angle, or `Pixelbuffer(Dynamic)` on GL,
            // but we are unable to make this decision here, so picking the reasonable medium.
            upload_method: UploadMethod::PixelBuffer(ONE_TIME_USAGE_HINT),
            upload_pbo_default_size: 512 * 512 * 4,
            workers: None,
            enable_multithreading: true,
            blob_image_handler: None,
            crash_annotator: None,
            thread_listener: None,
            size_of_op: None,
            enclosing_size_of_op: None,
            renderer_id: None,
            cached_programs: None,
            scene_builder_hooks: None,
            sampler: None,
            chase_primitive: ChasePrimitive::Nothing,
            support_low_priority_transactions: false,
            namespace_alloc_by_client: false,
            testing: false,
            gpu_supports_fast_clears: false,
            allow_dual_source_blending: true,
            allow_advanced_blend_equation: false,
            allow_texture_storage_support: true,
            allow_texture_swizzling: true,
            clear_caches_with_quads: true,
            // For backwards compatibility we set this to true by default, so
            // that if the debugger feature is enabled, the debug server will
            // be started automatically. Users can explicitly disable this as
            // needed.
            start_debug_server: true,
            dump_shader_source: None,
            surface_origin_is_top_left: false,
            compositor_config: CompositorConfig::default(),
            enable_gpu_markers: true,
            panic_on_gl_error: false,
            picture_tile_size: None,
            texture_cache_config: TextureCacheConfig::DEFAULT,
        }
    }
}

pub trait DebugServer {
    fn send(&mut self, _message: String);
}

struct NoopDebugServer;

impl NoopDebugServer {
    fn new(_: Sender<ApiMsg>) -> Self {
        NoopDebugServer
    }
}

impl DebugServer for NoopDebugServer {
    fn send(&mut self, _: String) {}
}

#[cfg(feature = "debugger")]
fn new_debug_server(enable: bool, api_tx: Sender<ApiMsg>) -> Box<dyn DebugServer> {
    if enable {
        Box::new(debug_server::DebugServerImpl::new(api_tx))
    } else {
        Box::new(NoopDebugServer::new(api_tx))
    }
}

#[cfg(not(feature = "debugger"))]
fn new_debug_server(_enable: bool, api_tx: Sender<ApiMsg>) -> Box<dyn DebugServer> {
    Box::new(NoopDebugServer::new(api_tx))
}

/// Some basic statistics about the rendered scene, used in Gecko, as
/// well as in wrench reftests to ensure that tests are batching and/or
/// allocating on render targets as we expect them to.
#[repr(C)]
#[derive(Debug, Default)]
pub struct RendererStats {
    pub total_draw_calls: usize,
    pub alpha_target_count: usize,
    pub color_target_count: usize,
    pub texture_upload_mb: f64,
    pub resource_upload_time: f64,
    pub gpu_cache_upload_time: f64,
}

/// Return type from render(), which contains some repr(C) statistics as well as
/// some non-repr(C) data.
#[derive(Debug, Default)]
pub struct RenderResults {
    /// Statistics about the frame that was rendered.
    pub stats: RendererStats,

    /// A list of the device dirty rects that were updated
    /// this frame.
    /// TODO(gw): This is an initial interface, likely to change in future.
    /// TODO(gw): The dirty rects here are currently only useful when scrolling
    ///           is not occurring. They are still correct in the case of
    ///           scrolling, but will be very large (until we expose proper
    ///           OS compositor support where the dirty rects apply to a
    ///           specific picture cache slice / OS compositor surface).
    pub dirty_rects: Vec<DeviceIntRect>,

    /// Information about the state of picture cache tiles. This is only
    /// allocated and stored if config.testing is true (such as wrench)
    pub picture_cache_debug: PictureCacheDebugInfo,
}

#[cfg(any(feature = "capture", feature = "replay"))]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
struct PlainTexture {
    data: String,
    size: (DeviceIntSize, i32),
    format: ImageFormat,
    filter: TextureFilter,
    has_depth: bool,
    is_array: bool,
}


#[cfg(any(feature = "capture", feature = "replay"))]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
struct PlainRenderer {
    device_size: Option<DeviceIntSize>,
    gpu_cache: PlainTexture,
    gpu_cache_frame_id: FrameId,
    textures: FastHashMap<CacheTextureId, PlainTexture>,
}

#[cfg(any(feature = "capture", feature = "replay"))]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
struct PlainExternalResources {
    images: Vec<ExternalCaptureImage>
}

#[cfg(feature = "replay")]
enum CapturedExternalImageData {
    NativeTexture(gl::GLuint),
    Buffer(Arc<Vec<u8>>),
}

#[cfg(feature = "replay")]
struct DummyExternalImageHandler {
    data: FastHashMap<(ExternalImageId, u8), (CapturedExternalImageData, TexelRect)>,
}

#[cfg(feature = "replay")]
impl ExternalImageHandler for DummyExternalImageHandler {
    fn lock(&mut self, key: ExternalImageId, channel_index: u8, _rendering: ImageRendering) -> ExternalImage {
        let (ref captured_data, ref uv) = self.data[&(key, channel_index)];
        ExternalImage {
            uv: *uv,
            source: match *captured_data {
                CapturedExternalImageData::NativeTexture(tid) => ExternalImageSource::NativeTexture(tid),
                CapturedExternalImageData::Buffer(ref arc) => ExternalImageSource::RawData(&*arc),
            }
        }
    }
    fn unlock(&mut self, _key: ExternalImageId, _channel_index: u8) {}
}

#[derive(Default)]
pub struct PipelineInfo {
    pub epochs: FastHashMap<(PipelineId, DocumentId), Epoch>,
    pub removed_pipelines: Vec<(PipelineId, DocumentId)>,
}

impl Renderer {
    #[cfg(feature = "capture")]
    fn save_texture(
        texture: &Texture, name: &str, root: &PathBuf, device: &mut Device
    ) -> PlainTexture {
        use std::fs;
        use std::io::Write;

        let short_path = format!("textures/{}.raw", name);

        let bytes_per_pixel = texture.get_format().bytes_per_pixel();
        let read_format = texture.get_format();
        let rect_size = texture.get_dimensions();

        let mut file = fs::File::create(root.join(&short_path))
            .expect(&format!("Unable to create {}", short_path));
        let bytes_per_layer = (rect_size.width * rect_size.height * bytes_per_pixel) as usize;
        let mut data = vec![0; bytes_per_layer];

        //TODO: instead of reading from an FBO with `read_pixels*`, we could
        // read from textures directly with `get_tex_image*`.

        for layer_id in 0 .. texture.get_layer_count() {
            let rect = device_size_as_framebuffer_size(rect_size).into();

            device.attach_read_texture(texture, layer_id);
            #[cfg(feature = "png")]
            {
                let mut png_data;
                let (data_ref, format) = match texture.get_format() {
                    ImageFormat::RGBAF32 => {
                        png_data = vec![0; (rect_size.width * rect_size.height * 4) as usize];
                        device.read_pixels_into(rect, ImageFormat::RGBA8, &mut png_data);
                        (&png_data, ImageFormat::RGBA8)
                    }
                    fm => (&data, fm),
                };
                CaptureConfig::save_png(
                    root.join(format!("textures/{}-{}.png", name, layer_id)),
                    rect_size, format,
                    None,
                    data_ref,
                );
            }
            device.read_pixels_into(rect, read_format, &mut data);
            file.write_all(&data)
                .unwrap();
        }

        PlainTexture {
            data: short_path,
            size: (rect_size, texture.get_layer_count()),
            format: texture.get_format(),
            filter: texture.get_filter(),
            has_depth: texture.supports_depth(),
            is_array: texture.is_array(),
        }
    }

    #[cfg(feature = "replay")]
    fn load_texture(
        target: ImageBufferKind,
        plain: &PlainTexture,
        rt_info: Option<RenderTargetInfo>,
        root: &PathBuf,
        device: &mut Device
    ) -> (Texture, Vec<u8>)
    {
        use std::fs::File;
        use std::io::Read;

        let mut texels = Vec::new();
        File::open(root.join(&plain.data))
            .expect(&format!("Unable to open texture at {}", plain.data))
            .read_to_end(&mut texels)
            .unwrap();

        let texture = device.create_texture(
            target,
            plain.format,
            plain.size.0.width,
            plain.size.0.height,
            plain.filter,
            rt_info,
            plain.size.1,
        );
        device.upload_texture_immediate(&texture, &texels);

        (texture, texels)
    }

    #[cfg(feature = "capture")]
    fn save_capture(
        &mut self,
        config: CaptureConfig,
        deferred_images: Vec<ExternalCaptureImage>,
    ) {
        use std::fs;
        use std::io::Write;
        use api::ExternalImageData;
        use crate::render_api::CaptureBits;

        let root = config.resource_root();

        self.device.begin_frame();
        let _gm = self.gpu_profiler.start_marker("read GPU data");
        self.device.bind_read_target_impl(self.read_fbo);

        if config.bits.contains(CaptureBits::EXTERNAL_RESOURCES) && !deferred_images.is_empty() {
            info!("saving external images");
            let mut arc_map = FastHashMap::<*const u8, String>::default();
            let mut tex_map = FastHashMap::<u32, String>::default();
            let handler = self.external_image_handler
                .as_mut()
                .expect("Unable to lock the external image handler!");
            for def in &deferred_images {
                info!("\t{}", def.short_path);
                let ExternalImageData { id, channel_index, image_type } = def.external;
                // The image rendering parameter is irrelevant because no filtering happens during capturing.
                let ext_image = handler.lock(id, channel_index, ImageRendering::Auto);
                let (data, short_path) = match ext_image.source {
                    ExternalImageSource::RawData(data) => {
                        let arc_id = arc_map.len() + 1;
                        match arc_map.entry(data.as_ptr()) {
                            Entry::Occupied(e) => {
                                (None, e.get().clone())
                            }
                            Entry::Vacant(e) => {
                                let short_path = format!("externals/d{}.raw", arc_id);
                                (Some(data.to_vec()), e.insert(short_path).clone())
                            }
                        }
                    }
                    ExternalImageSource::NativeTexture(gl_id) => {
                        let tex_id = tex_map.len() + 1;
                        match tex_map.entry(gl_id) {
                            Entry::Occupied(e) => {
                                (None, e.get().clone())
                            }
                            Entry::Vacant(e) => {
                                let target = match image_type {
                                    ExternalImageType::TextureHandle(target) => target,
                                    ExternalImageType::Buffer => unreachable!(),
                                };
                                info!("\t\tnative texture of target {:?}", target);
                                let layer_index = 0; //TODO: what about layered textures?
                                self.device.attach_read_texture_external(gl_id, target, layer_index);
                                let data = self.device.read_pixels(&def.descriptor);
                                let short_path = format!("externals/t{}.raw", tex_id);
                                (Some(data), e.insert(short_path).clone())
                            }
                        }
                    }
                    ExternalImageSource::Invalid => {
                        info!("\t\tinvalid source!");
                        (None, String::new())
                    }
                };
                if let Some(bytes) = data {
                    fs::File::create(root.join(&short_path))
                        .expect(&format!("Unable to create {}", short_path))
                        .write_all(&bytes)
                        .unwrap();
                    #[cfg(feature = "png")]
                    CaptureConfig::save_png(
                        root.join(&short_path).with_extension("png"),
                        def.descriptor.size,
                        def.descriptor.format,
                        def.descriptor.stride,
                        &bytes,
                    );
                }
                let plain = PlainExternalImage {
                    data: short_path,
                    external: def.external,
                    uv: ext_image.uv,
                };
                config.serialize_for_resource(&plain, &def.short_path);
            }
            for def in &deferred_images {
                handler.unlock(def.external.id, def.external.channel_index);
            }
            let plain_external = PlainExternalResources {
                images: deferred_images,
            };
            config.serialize_for_resource(&plain_external, "external_resources");
        }

        if config.bits.contains(CaptureBits::FRAME) {
            let path_textures = root.join("textures");
            if !path_textures.is_dir() {
                fs::create_dir(&path_textures).unwrap();
            }

            info!("saving GPU cache");
            self.update_gpu_cache(); // flush pending updates
            let mut plain_self = PlainRenderer {
                device_size: self.device_size,
                gpu_cache: Self::save_texture(
                    &self.gpu_cache_texture.texture.as_ref().unwrap(),
                    "gpu", &root, &mut self.device,
                ),
                gpu_cache_frame_id: self.gpu_cache_frame_id,
                textures: FastHashMap::default(),
            };

            info!("saving cached textures");
            for (id, texture) in &self.texture_resolver.texture_cache_map {
                let file_name = format!("cache-{}", plain_self.textures.len() + 1);
                info!("\t{}", file_name);
                let plain = Self::save_texture(texture, &file_name, &root, &mut self.device);
                plain_self.textures.insert(*id, plain);
            }

            config.serialize_for_resource(&plain_self, "renderer");
        }

        self.device.reset_read_target();
        self.device.end_frame();

        let mut stats_file = fs::File::create(config.root.join("profiler-stats.txt"))
            .expect(&format!("Unable to create profiler-stats.txt"));
        if self.debug_flags.intersects(DebugFlags::PROFILER_DBG | DebugFlags::PROFILER_CAPTURE) {
            self.profiler.dump_stats(&mut stats_file).unwrap();
        } else {
            writeln!(stats_file, "Turn on PROFILER_DBG or PROFILER_CAPTURE to get stats here!").unwrap();
        }

        info!("done.");
    }

    #[cfg(feature = "replay")]
    fn load_capture(
        &mut self,
        config: CaptureConfig,
        plain_externals: Vec<PlainExternalImage>,
    ) {
        use std::fs::File;
        use std::io::Read;
        use std::slice;

        info!("loading external buffer-backed images");
        assert!(self.texture_resolver.external_images.is_empty());
        let mut raw_map = FastHashMap::<String, Arc<Vec<u8>>>::default();
        let mut image_handler = DummyExternalImageHandler {
            data: FastHashMap::default(),
        };

        let root = config.resource_root();

        // Note: this is a `SCENE` level population of the external image handlers
        // It would put both external buffers and texture into the map.
        // But latter are going to be overwritten later in this function
        // if we are in the `FRAME` level.
        for plain_ext in plain_externals {
            let data = match raw_map.entry(plain_ext.data) {
                Entry::Occupied(e) => e.get().clone(),
                Entry::Vacant(e) => {
                    let mut buffer = Vec::new();
                    File::open(root.join(e.key()))
                        .expect(&format!("Unable to open {}", e.key()))
                        .read_to_end(&mut buffer)
                        .unwrap();
                    e.insert(Arc::new(buffer)).clone()
                }
            };
            let ext = plain_ext.external;
            let value = (CapturedExternalImageData::Buffer(data), plain_ext.uv);
            image_handler.data.insert((ext.id, ext.channel_index), value);
        }

        if let Some(external_resources) = config.deserialize_for_resource::<PlainExternalResources, _>("external_resources") {
            info!("loading external texture-backed images");
            let mut native_map = FastHashMap::<String, gl::GLuint>::default();
            for ExternalCaptureImage { short_path, external, descriptor } in external_resources.images {
                let target = match external.image_type {
                    ExternalImageType::TextureHandle(target) => target,
                    ExternalImageType::Buffer => continue,
                };
                let plain_ext = config.deserialize_for_resource::<PlainExternalImage, _>(&short_path)
                    .expect(&format!("Unable to read {}.ron", short_path));
                let key = (external.id, external.channel_index);

                let tid = match native_map.entry(plain_ext.data) {
                    Entry::Occupied(e) => e.get().clone(),
                    Entry::Vacant(e) => {
                        //TODO: provide a way to query both the layer count and the filter from external images
                        let (layer_count, filter) = (1, TextureFilter::Linear);
                        let plain_tex = PlainTexture {
                            data: e.key().clone(),
                            size: (descriptor.size, layer_count),
                            format: descriptor.format,
                            filter,
                            has_depth: false,
                            is_array: false,
                        };
                        let t = Self::load_texture(
                            target,
                            &plain_tex,
                            None,
                            &root,
                            &mut self.device
                        );
                        let extex = t.0.into_external();
                        self.owned_external_images.insert(key, extex.clone());
                        e.insert(extex.internal_id()).clone()
                    }
                };

                let value = (CapturedExternalImageData::NativeTexture(tid), plain_ext.uv);
                image_handler.data.insert(key, value);
            }
        }

        if let Some(renderer) = config.deserialize_for_resource::<PlainRenderer, _>("renderer") {
            info!("loading cached textures");
            self.device_size = renderer.device_size;
            self.device.begin_frame();

            for (_id, texture) in self.texture_resolver.texture_cache_map.drain() {
                self.device.delete_texture(texture);
            }
            for (id, texture) in renderer.textures {
                info!("\t{}", texture.data);
                let target = if texture.is_array {
                    ImageBufferKind::Texture2DArray
                } else {
                    ImageBufferKind::Texture2D
                };
                let t = Self::load_texture(
                    target,
                    &texture,
                    Some(RenderTargetInfo { has_depth: texture.has_depth }),
                    &root,
                    &mut self.device
                );
                self.texture_resolver.texture_cache_map.insert(id, t.0);
            }

            info!("loading gpu cache");
            if let Some(t) = self.gpu_cache_texture.texture.take() {
                self.device.delete_texture(t);
            }
            let (t, gpu_cache_data) = Self::load_texture(
                ImageBufferKind::Texture2D,
                &renderer.gpu_cache,
                Some(RenderTargetInfo { has_depth: false }),
                &root,
                &mut self.device,
            );
            self.gpu_cache_texture.texture = Some(t);
            match self.gpu_cache_texture.bus {
                GpuCacheBus::PixelBuffer { ref mut rows, .. } => {
                    let dim = self.gpu_cache_texture.texture.as_ref().unwrap().get_dimensions();
                    let blocks = unsafe {
                        slice::from_raw_parts(
                            gpu_cache_data.as_ptr() as *const GpuBlockData,
                            gpu_cache_data.len() / mem::size_of::<GpuBlockData>(),
                        )
                    };
                    // fill up the CPU cache from the contents we just loaded
                    rows.clear();
                    rows.extend((0 .. dim.height).map(|_| CacheRow::new()));
                    let chunks = blocks.chunks(MAX_VERTEX_TEXTURE_WIDTH);
                    debug_assert_eq!(chunks.len(), rows.len());
                    for (row, chunk) in rows.iter_mut().zip(chunks) {
                        row.cpu_blocks.copy_from_slice(chunk);
                    }
                }
                GpuCacheBus::Scatter { .. } => {}
            }
            self.gpu_cache_frame_id = renderer.gpu_cache_frame_id;

            self.device.end_frame();
        } else {
            info!("loading cached textures");
            self.device.begin_frame();
            for (_id, texture) in self.texture_resolver.texture_cache_map.drain() {
                self.device.delete_texture(texture);
            }

            info!("loading gpu cache");
            if let Some(t) = self.gpu_cache_texture.texture.take() {
                self.device.delete_texture(t);
            }
            self.device.end_frame();
        }

        self.external_image_handler = Some(Box::new(image_handler) as Box<_>);
        info!("done.");
    }
}

fn get_vao(vertex_array_kind: VertexArrayKind, vaos: &RendererVAOs) -> &VAO {
    match vertex_array_kind {
        VertexArrayKind::Primitive => &vaos.prim_vao,
        VertexArrayKind::ClipImage => &vaos.clip_image_vao,
        VertexArrayKind::ClipRect => &vaos.clip_rect_vao,
        VertexArrayKind::ClipBoxShadow => &vaos.clip_box_shadow_vao,
        VertexArrayKind::Blur => &vaos.blur_vao,
        VertexArrayKind::VectorStencil | VertexArrayKind::VectorCover => unreachable!(),
        VertexArrayKind::Border => &vaos.border_vao,
        VertexArrayKind::Scale => &vaos.scale_vao,
        VertexArrayKind::LineDecoration => &vaos.line_vao,
        VertexArrayKind::Gradient => &vaos.gradient_vao,
        VertexArrayKind::Resolve => &vaos.resolve_vao,
        VertexArrayKind::SvgFilter => &vaos.svg_filter_vao,
        VertexArrayKind::Composite => &vaos.composite_vao,
        VertexArrayKind::Clear => &vaos.clear_vao,
    }
}
#[derive(Clone, Copy, PartialEq)]
enum FramebufferKind {
    Main,
    Other,
}

fn should_skip_batch(kind: &BatchKind, flags: DebugFlags) -> bool {
    match kind {
        BatchKind::TextRun(_) => {
            flags.contains(DebugFlags::DISABLE_TEXT_PRIMS)
        }
        BatchKind::Brush(BrushBatchKind::ConicGradient) |
        BatchKind::Brush(BrushBatchKind::RadialGradient) |
        BatchKind::Brush(BrushBatchKind::LinearGradient) => {
            flags.contains(DebugFlags::DISABLE_GRADIENT_PRIMS)
        }
        _ => false,
    }
}

impl CompositeState {
    /// Use the client provided native compositor interface to add all picture
    /// cache tiles to the OS compositor
    fn composite_native(
        &self,
        dirty_rects: &[DeviceIntRect],
        compositor: &mut dyn Compositor,
    ) {
        // Add each surface to the visual tree. z-order is implicit based on
        // order added. Offset and clip rect apply to all tiles within this
        // surface.
        for surface in &self.descriptor.surfaces {
            compositor.add_surface(
                surface.surface_id.expect("bug: no native surface allocated"),
                surface.transform,
                surface.clip_rect.to_i32(),
                surface.image_rendering,
            );
        }
        compositor.start_compositing(dirty_rects);
    }
}

mod tests {
    #[test]
    fn test_buffer_damage_tracker() {
        use super::BufferDamageTracker;
        use api::units::{DevicePoint, DeviceRect, DeviceSize};

        let mut tracker = BufferDamageTracker::default();
        assert_eq!(tracker.get_damage_rect(0), None);
        assert_eq!(tracker.get_damage_rect(1), Some(DeviceRect::zero()));
        assert_eq!(tracker.get_damage_rect(2), Some(DeviceRect::zero()));
        assert_eq!(tracker.get_damage_rect(3), Some(DeviceRect::zero()));
        assert_eq!(tracker.get_damage_rect(4), None);

        let damage1 = DeviceRect::new(DevicePoint::new(10.0, 10.0), DeviceSize::new(10.0, 10.0));
        let damage2 = DeviceRect::new(DevicePoint::new(20.0, 20.0), DeviceSize::new(10.0, 10.0));
        let combined = damage1.union(&damage2);

        tracker.push_dirty_rect(&damage1);
        assert_eq!(tracker.get_damage_rect(0), None);
        assert_eq!(tracker.get_damage_rect(1), Some(DeviceRect::zero()));
        assert_eq!(tracker.get_damage_rect(2), Some(damage1));
        assert_eq!(tracker.get_damage_rect(3), Some(damage1));
        assert_eq!(tracker.get_damage_rect(4), None);

        tracker.push_dirty_rect(&damage2);
        assert_eq!(tracker.get_damage_rect(0), None);
        assert_eq!(tracker.get_damage_rect(1), Some(DeviceRect::zero()));
        assert_eq!(tracker.get_damage_rect(2), Some(damage2));
        assert_eq!(tracker.get_damage_rect(3), Some(combined));
        assert_eq!(tracker.get_damage_rect(4), None);
    }
}
