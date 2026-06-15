/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/// This shader renders YUV images in a color target.

#include ps_quad,yuv

varying highp vec2 vUv_Y;
flat varying highp vec4 vUvBounds_Y;

varying highp vec2 vUv_U;
flat varying highp vec4 vUvBounds_U;

varying highp vec2 vUv_V;
flat varying highp vec4 vUvBounds_V;

flat varying YUV_PRECISION vec3 vYcbcrBias;
flat varying YUV_PRECISION mat3 vRgbFromDebiasedYcbcr;

// YUV format. Packed in to vector to work around bug 1630356.
flat varying mediump ivec2 vFormat;

#ifdef SWGL_DRAW_SPAN
flat varying mediump int vRescaleFactor;
#endif

#ifdef WR_VERTEX_SHADER

struct YuvQuadData {
    RectWithEndpoint uv_rect_u;
    RectWithEndpoint uv_rect_v;
    YUV_PRECISION vec3 ycbcr_bias;
    YUV_PRECISION mat3 rgb_from_debiased_ycbcr;
    int format;
    int rescale_factor;
};

// See YuvPattern::build in src/pattern/yuv.rs.
//
// The YUV-to-RGB conversion parameters are computed on the CPU and fetched here.
YuvQuadData fetch_yuv_quad_data(int address) {
    vec4 d[3] = fetch_from_gpu_buffer_3f(address);
    vec4 e[3] = fetch_from_gpu_buffer_3f(address + 3);

    YuvQuadData yuv;
    yuv.uv_rect_u = RectWithEndpoint(d[0].xy, d[0].zw);
    yuv.uv_rect_v = RectWithEndpoint(d[1].xy, d[1].zw);
    yuv.ycbcr_bias = d[2].xyz;
    yuv.format = int(d[2].w);
    yuv.rgb_from_debiased_ycbcr = mat3(e[0].xyz, e[1].xyz, e[2].xyz);
    yuv.rescale_factor = int(e[0].w);

    return yuv;
}

void pattern_vertex(PrimitiveInfo info) {
    YuvQuadData yuv = fetch_yuv_quad_data(info.pattern_input.x);

#ifdef SWGL_DRAW_SPAN
    vRescaleFactor = yuv.rescale_factor;
#endif

    vYcbcrBias = yuv.ycbcr_bias;
    vRgbFromDebiasedYcbcr = yuv.rgb_from_debiased_ycbcr;

    vFormat.x = yuv.format;

    // Normalized position within the primitive rect.
    RectWithEndpoint rect = info.local_prim_rect;
    vec2 f = (info.local_pos - rect.p0) / rect_size(rect);

    // The Y plane uv rect travels through the standard quad primitive block (as
    // the segment uv rect); the U and V plane uv rects come from the pattern's
    // own gpu block.
    RectWithEndpoint uv_rect_y = info.segment.uv_rect;

    // The additional test for 99 works around a gen6 shader compiler bug: 1708937
    if (vFormat.x == YUV_FORMAT_PLANAR || vFormat.x == 99) {
        write_uv_rect(uv_rect_y.p0, uv_rect_y.p1, f, TEX_SIZE_YUV(sColor0), vUv_Y, vUvBounds_Y);
        write_uv_rect(yuv.uv_rect_u.p0, yuv.uv_rect_u.p1, f, TEX_SIZE_YUV(sColor1), vUv_U, vUvBounds_U);
        write_uv_rect(yuv.uv_rect_v.p0, yuv.uv_rect_v.p1, f, TEX_SIZE_YUV(sColor2), vUv_V, vUvBounds_V);
    } else if (vFormat.x == YUV_FORMAT_NV12 || vFormat.x == YUV_FORMAT_P010 || vFormat.x == YUV_FORMAT_NV16) {
        write_uv_rect(uv_rect_y.p0, uv_rect_y.p1, f, TEX_SIZE_YUV(sColor0), vUv_Y, vUvBounds_Y);
        write_uv_rect(yuv.uv_rect_u.p0, yuv.uv_rect_u.p1, f, TEX_SIZE_YUV(sColor1), vUv_U, vUvBounds_U);
    } else if (vFormat.x == YUV_FORMAT_INTERLEAVED) {
        write_uv_rect(uv_rect_y.p0, uv_rect_y.p1, f, TEX_SIZE_YUV(sColor0), vUv_Y, vUvBounds_Y);
    }
}

#endif

#ifdef WR_FRAGMENT_SHADER

vec4 pattern_fragment(vec4 color) {
    vec4 yuv_color = sample_yuv(
        vFormat.x,
        vYcbcrBias,
        vRgbFromDebiasedYcbcr,
        vUv_Y,
        vUv_U,
        vUv_V,
        vUvBounds_Y,
        vUvBounds_U,
        vUvBounds_V
    );

    return color * yuv_color;
}

#if defined(SWGL_DRAW_SPAN)
void swgl_drawSpanRGBA8() {
    if (vFormat.x == YUV_FORMAT_PLANAR) {
        swgl_commitTextureLinearYUV(sColor0, vUv_Y, vUvBounds_Y,
                                    sColor1, vUv_U, vUvBounds_U,
                                    sColor2, vUv_V, vUvBounds_V,
                                    vYcbcrBias,
                                    vRgbFromDebiasedYcbcr,
                                    vRescaleFactor);
    } else if (vFormat.x == YUV_FORMAT_NV12 || vFormat.x == YUV_FORMAT_P010 || vFormat.x == YUV_FORMAT_NV16) {
        swgl_commitTextureLinearYUV(sColor0, vUv_Y, vUvBounds_Y,
                                    sColor1, vUv_U, vUvBounds_U,
                                    vYcbcrBias,
                                    vRgbFromDebiasedYcbcr,
                                    vRescaleFactor);
    } else if (vFormat.x == YUV_FORMAT_INTERLEAVED) {
        swgl_commitTextureLinearYUV(sColor0, vUv_Y, vUvBounds_Y,
                                    vYcbcrBias,
                                    vRgbFromDebiasedYcbcr,
                                    vRescaleFactor);
    }
}
#endif

#endif
