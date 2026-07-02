/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

pub mod gradient;
pub mod box_shadow;
pub mod repeat;
pub mod image;
pub mod cutout;
pub mod yuv;
pub mod backdrop;
pub mod filter;

use api::units::*;
use api::ColorF;

use crate::frame_builder::FrameBuilderConfig;
use crate::render_task_graph::RenderTaskId;
use crate::renderer::{BlendMode, GpuBufferBuilder};
use crate::spatial_tree::SpatialTree;
use crate::transform::TransformPalette;

#[repr(u32)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PatternKind {
    ColorOrTexture = 0,
    Gradient = 1,
    Repeat = 2,

    Mask = 3,
    BoxShadow = 4,
    // Variants of ColorOrTexture that use a non-default sampler type
    // (samplerExternalOES / __samplerExternal2DY2YEXT / sampler2DRect). The
    // quad shader is compiled in matching per-kind variants; see
    // ps_quad_textured.glsl.
    TextureExternal = 5,
    TextureExternalBT709 = 6,
    TextureRect = 7,
    // Samples up to three planes (sColor0/1/2) and converts from YUV to RGB.
    // Like ColorOrTexture, the YUV pattern comes in sampler-type-specific
    // variants so that the planes are sampled with the matching sColor
    // declaration; see ps_quad_yuv.glsl. `Yuv` is the default (TEXTURE_2D).
    Yuv = 8,
    YuvTextureExternal = 9,
    YuvTextureExternalBT709 = 10,
    YuvTextureRect = 11,
    // Samples a captured backdrop texture using a (bilerp) 4-corner uv quad.
    Backdrop = 12,
    // Applies a filter to a sampled source texture; see ps_quad_blend.glsl.
    Blend = 13,
    // When adding patterns, don't forget to update the NUM_PATTERNS constant.
}

pub const NUM_PATTERNS: u32 = 14;

impl PatternKind {
    pub fn from_u32(val: u32) -> Self {
        assert!(val < NUM_PATTERNS);
        unsafe { std::mem::transmute(val) }
    }

    pub fn num_src_textures(&self) -> usize {
        match self {
            PatternKind::Gradient
            | PatternKind::Mask
            => 0,
            PatternKind::Yuv
            | PatternKind::YuvTextureExternal
            | PatternKind::YuvTextureExternalBT709
            | PatternKind::YuvTextureRect
            => 3,
            _ => 1,
        }
    }
}

/// A 32bit payload used as input for the pattern-specific logic in the shader.
///
/// Patterns typically use it as a GpuBuffer offset to fetch their data.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PatternShaderInput(pub i32, pub i32);

impl Default for PatternShaderInput {
    fn default() -> Self {
        PatternShaderInput(0, 0)
    }
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PatternTextureInput {
    /// The render tasks providing the source texture(s) sampled by the pattern.
    ///
    /// Most patterns only sample from a single texture (slot 0). Multi-plane
    /// patterns such as YUV use the additional slots (bound to sColor1/sColor2).
    pub task_ids: [RenderTaskId; 3],
}

impl Default for PatternTextureInput {
    fn default() -> Self {
        PatternTextureInput {
            task_ids: [RenderTaskId::INVALID; 3],
        }
    }
}

impl PatternTextureInput {
    pub fn new(task_id: RenderTaskId) -> Self {
        PatternTextureInput {
            task_ids: [task_id, RenderTaskId::INVALID, RenderTaskId::INVALID],
        }
    }

    pub fn yuv(task_ids: [RenderTaskId; 3]) -> Self {
        PatternTextureInput {
            task_ids,
        }
    }

    /// The primary (plane 0) source texture.
    pub fn task_id(&self) -> RenderTaskId {
        self.task_ids[0]
    }
}

pub struct PatternBuilderContext<'a> {
    pub spatial_tree: &'a SpatialTree,
    pub fb_config: &'a FrameBuilderConfig,
    pub prim_origin: LayoutPoint,
}

pub struct PatternBuilderState<'a> {
    pub frame_gpu_data: &'a mut GpuBufferBuilder,
    #[allow(unused)]
    pub transforms: &'a mut TransformPalette,
}

pub trait PatternBuilder {
    fn build(
        &self,
        sub_rect: Option<DeviceRect>,
        offset: LayoutVector2D,
        ctx: &PatternBuilderContext,
        state: &mut PatternBuilderState,
    ) -> Pattern;
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[derive(Clone, Debug)]
pub struct Pattern {
    pub kind: PatternKind,
    pub shader_input: PatternShaderInput,
    pub texture_input: PatternTextureInput,
    pub base_color: ColorF,
    pub is_opaque: bool,
    pub blend_mode: BlendMode,
}

impl Pattern {
    pub fn color(color: ColorF) -> Self {
        Pattern {
            kind: PatternKind::ColorOrTexture,
            shader_input: PatternShaderInput(
                TEXTURED_SHADER_MODE_COLOR,
                0,
            ),
            texture_input: PatternTextureInput::default(),
            base_color: color,
            is_opaque: color.a >= 1.0,
            blend_mode: BlendMode::PremultipliedAlpha,
        }
    }

    pub fn texture(src_task: RenderTaskId, is_opaque: bool) -> Self {
        Pattern {
            kind: PatternKind::ColorOrTexture,
            shader_input: PatternShaderInput(
                TEXTURED_SHADER_MODE_TEXTURE,
                TEXTURED_SHADER_MAP_TO_PRIMITIVE,
            ),
            texture_input: PatternTextureInput::new(src_task),
            base_color: ColorF::WHITE,
            is_opaque,
            blend_mode: BlendMode::PremultipliedAlpha,
        }
    }

    pub fn with_blend_mode(mut self, blend_mode: BlendMode) -> Self {
        self.blend_mode = blend_mode;

        self
    }

    pub fn with_base_color(mut self, color: ColorF) -> Self {
        self.base_color = color;

        self
    }

    pub fn as_render_task(&self) -> Option<RenderTaskId> {
        if self.kind != PatternKind::ColorOrTexture || self.texture_input.task_id() == RenderTaskId::INVALID {
            return None;
        }

        Some(self.texture_input.task_id())
    }
}

pub const TEXTURED_SHADER_MODE_COLOR: i32 = 0;
pub const TEXTURED_SHADER_MODE_TEXTURE: i32 = 1;
// Only read the input texture's alpha.
pub const TEXTURED_SHADER_MODE_TEXTURE_ALPHA: i32 = 2;

// In the texture mode, whether to map the texture to the primitive's local rect
// or segment rect.
pub const TEXTURED_SHADER_MAP_TO_PRIMITIVE: i32 = 0;
pub const TEXTURED_SHADER_MAP_TO_SEGMENT: i32 = 1;

impl PatternBuilder for ColorF {
    fn build(
        &self,
        _sub_rect: Option<DeviceRect>,
        _offset: LayoutVector2D,
        _ctx: &PatternBuilderContext,
        _state: &mut PatternBuilderState,
    ) -> Pattern {
        Pattern::color(*self)
    }
}
