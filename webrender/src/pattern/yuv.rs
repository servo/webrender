/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{ColorF, ImageBufferKind, YuvFormat, YuvRangedColorSpace};
use api::units::*;

use crate::pattern::{Pattern, PatternBuilder, PatternBuilderContext, PatternBuilderState, PatternKind, PatternShaderInput, PatternTextureInput};
use crate::render_task_graph::RenderTaskId;
use crate::renderer::BlendMode;
use crate::util::pack_as_float;

// Column-major 3x3 matrices converting YUV to RGB, matching the constants in
// res/yuv.glsl. Each inner array is a column.
const RGB_FROM_YUV_REC601: [[f32; 3]; 3] = [
    [1.0, 1.0, 1.0],
    [0.0, -0.17207, 0.886],
    [0.701, -0.35707, 0.0],
];
const RGB_FROM_YUV_REC709: [[f32; 3]; 3] = [
    [1.0, 1.0, 1.0],
    [0.0, -0.09366, 0.9278],
    [0.7874, -0.23406, 0.0],
];
const RGB_FROM_YUV_REC2020: [[f32; 3]; 3] = [
    [1.0, 1.0, 1.0],
    [0.0, -0.08228, 0.9407],
    [0.7373, -0.28568, 0.0],
];
const RGB_FROM_YUV_GBR_IDENTITY: [[f32; 3]; 3] = [
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
];

fn channel_max(bit_depth: u32, format: YuvFormat) -> f32 {
    if bit_depth > 8 {
        if format == YuvFormat::P010 {
            // This is an MSB format.
            ((1u32 << bit_depth) - 1) as f32
        } else {
            // For >8bpc, we get the low bits, not the high bits.
            65535.0
        }
    } else {
        255.0
    }
}

fn zero_one_identity(bit_depth: u32, channel_max: f32) -> [f32; 4] {
    let all_ones_normalized = ((1u32 << bit_depth) - 1) as f32 / channel_max;
    [0.0, 0.0, all_ones_normalized, all_ones_normalized]
}

fn zero_one_narrow_range(bit_depth: u32, channel_max: f32) -> [f32; 4] {
    let shift = bit_depth - 8;
    [
        ((16i32 << shift) as f32) / channel_max,
        ((128i32 << shift) as f32) / channel_max,
        ((235i32 << shift) as f32) / channel_max,
        ((240i32 << shift) as f32) / channel_max,
    ]
}

fn zero_one_full_range(bit_depth: u32, channel_max: f32) -> [f32; 4] {
    let narrow = zero_one_narrow_range(bit_depth, channel_max);
    let identity = zero_one_identity(bit_depth, channel_max);
    [0.0, narrow[1], identity[2], identity[3]]
}

/// The YUV-to-RGB conversion parameters, computed on the CPU.
///
/// This mirrors `get_rgb_from_ycbcr_info` in res/yuv.glsl. We compute it on the
/// CPU rather than in the shader because the quad vertex shader runs in SWGL's
/// vectorized path, where the integer bit-shifts used by the range computation
/// are not available.
pub struct YuvColorMatrix {
    pub ycbcr_bias: [f32; 3],
    /// Column-major 3x3 matrix.
    pub rgb_from_debiased_ycbcr: [[f32; 3]; 3],
    pub rescale_factor: i32,
}

impl YuvColorMatrix {
    pub fn new(bit_depth: u32, color_space: YuvRangedColorSpace, format: YuvFormat) -> Self {
        let channel_max = channel_max(bit_depth, format);

        let (rgb_from_yuv, packed_zero_one) = match color_space {
            YuvRangedColorSpace::Rec601Narrow =>
                (RGB_FROM_YUV_REC601, zero_one_narrow_range(bit_depth, channel_max)),
            YuvRangedColorSpace::Rec601Full =>
                (RGB_FROM_YUV_REC601, zero_one_full_range(bit_depth, channel_max)),
            YuvRangedColorSpace::Rec709Narrow =>
                (RGB_FROM_YUV_REC709, zero_one_narrow_range(bit_depth, channel_max)),
            YuvRangedColorSpace::Rec709Full =>
                (RGB_FROM_YUV_REC709, zero_one_full_range(bit_depth, channel_max)),
            YuvRangedColorSpace::Rec2020Narrow =>
                (RGB_FROM_YUV_REC2020, zero_one_narrow_range(bit_depth, channel_max)),
            YuvRangedColorSpace::Rec2020Full =>
                (RGB_FROM_YUV_REC2020, zero_one_full_range(bit_depth, channel_max)),
            YuvRangedColorSpace::GbrIdentity =>
                (RGB_FROM_YUV_GBR_IDENTITY, zero_one_identity(bit_depth, channel_max)),
        };

        let zero = [packed_zero_one[0], packed_zero_one[1]];
        let one = [packed_zero_one[2], packed_zero_one[3]];
        // Such that yuv_value = (ycbcr_sample - zero) / (one - zero).
        let scale = [1.0 / (one[0] - zero[0]), 1.0 / (one[1] - zero[1])];

        // rgb_from_yuv * diag(scale.x, scale.y, scale.y), which scales each
        // column of the YUV matrix.
        let rgb_from_debiased_ycbcr = [
            [rgb_from_yuv[0][0] * scale[0], rgb_from_yuv[0][1] * scale[0], rgb_from_yuv[0][2] * scale[0]],
            [rgb_from_yuv[1][0] * scale[1], rgb_from_yuv[1][1] * scale[1], rgb_from_yuv[1][2] * scale[1]],
            [rgb_from_yuv[2][0] * scale[1], rgb_from_yuv[2][1] * scale[1], rgb_from_yuv[2][2] * scale[1]],
        ];

        // swgl_commitTextureLinearYUV needs to know how many bits of scaling are
        // required to normalize HDR textures. MSB HDR formats don't need it.
        let rescale_factor = if bit_depth > 8 && format != YuvFormat::P010 {
            16 - bit_depth as i32
        } else {
            0
        };

        YuvColorMatrix {
            ycbcr_bias: [zero[0], zero[1], zero[1]],
            rgb_from_debiased_ycbcr,
            rescale_factor,
        }
    }
}

/// Pattern that samples up to three planes and converts from YUV to RGB.
///
/// The Y plane is sampled from slot 0 (its uv rect travels through the standard
/// quad primitive block), while the U and V planes are sampled from slots 1 and
/// 2 and their uv rects are written to a dedicated gpu block addressed by the
/// pattern shader input, alongside the precomputed color conversion parameters.
pub struct YuvPattern {
    /// Render tasks for the (up to three) source planes. Unused planes are
    /// `RenderTaskId::INVALID`.
    pub planes: [RenderTaskId; 3],
    pub format: YuvFormat,
    pub color_space: YuvRangedColorSpace,
    pub channel_bit_depth: u32,
    /// Texture target the planes are sampled from. Selects the matching
    /// ps_quad_yuv shader variant (TEXTURE_2D / RECT / EXTERNAL / EXTERNAL_BT709).
    pub sampler_kind: ImageBufferKind,
}

impl PatternBuilder for YuvPattern {
    fn build(
        &self,
        _sub_rect: Option<DeviceRect>,
        _offset: LayoutVector2D,
        _ctx: &PatternBuilderContext,
        state: &mut PatternBuilderState,
    ) -> Pattern {
        let mat = YuvColorMatrix::new(self.channel_bit_depth, self.color_space, self.format);
        let m = &mat.rgb_from_debiased_ycbcr;

        // See fetch_yuv_quad_data in ps_quad_yuv.glsl.
        let mut writer = state.frame_gpu_data.f32.write_blocks(6);
        writer.push_render_task(self.planes[1]);
        writer.push_render_task(self.planes[2]);
        writer.push_one([
            mat.ycbcr_bias[0],
            mat.ycbcr_bias[1],
            mat.ycbcr_bias[2],
            pack_as_float(self.format as u32),
        ]);
        writer.push_one([m[0][0], m[0][1], m[0][2], pack_as_float(mat.rescale_factor as u32)]);
        writer.push_one([m[1][0], m[1][1], m[1][2], 0.0]);
        writer.push_one([m[2][0], m[2][1], m[2][2], 0.0]);
        let addr = writer.finish();

        Pattern {
            kind: match self.sampler_kind {
                ImageBufferKind::Texture2D => PatternKind::Yuv,
                ImageBufferKind::TextureExternal => PatternKind::YuvTextureExternal,
                ImageBufferKind::TextureExternalBT709 => PatternKind::YuvTextureExternalBT709,
                ImageBufferKind::TextureRect => PatternKind::YuvTextureRect,
            },
            shader_input: PatternShaderInput(addr.as_int(), 0),
            texture_input: PatternTextureInput::yuv(self.planes),
            // YUV images are always opaque.
            base_color: ColorF::WHITE,
            is_opaque: true,
            blend_mode: BlendMode::PremultipliedAlpha,
        }
    }
}
