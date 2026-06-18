/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/// This shader renders a captured backdrop (e.g. for backdrop-filter) through
/// the quad infrastructure. It is the quad-based equivalent of the brush path
/// that draws BackdropRender primitives.
///
/// The backdrop texture may be in a different coordinate space than the
/// primitive, so the sampling position is given by four homogeneous screen
/// space uv corners which are bilinearly interpolated across the primitive rect
/// and then mapped into the backdrop's texture-cache rect (the standard quad uv
/// rect, provided via the source render task).
///
/// The backdrop uv must vary linearly in screen space (it samples by screen
/// position), so we use the same trick as brush_image's screen-space raster
/// path: premultiply the uv by the clip-space w in the vertex shader and divide
/// it back out per fragment via gl_FragCoord.w. For axis-aligned (non
/// perspective) primitives w is 1 and this is a no-op.

#include ps_quad

varying highp vec2 v_uv;
flat varying highp vec4 v_uv_bounds;

#ifdef WR_VERTEX_SHADER

void pattern_vertex(PrimitiveInfo info) {
    // Homogeneous screen-space uv corners. See BackdropPattern::build.
    vec4 uvs[4] = fetch_from_gpu_buffer_4f(info.pattern_input.x);

    // Normalized position within the primitive rect.
    RectWithEndpoint rect = info.local_prim_rect;
    vec2 f = (info.local_pos - rect.p0) / rect_size(rect);

    // Bilinearly interpolate the homogeneous corners, then do the perspective
    // divide to get the screen-space uv.
    vec4 x = mix(uvs[0], uvs[1], f.x);
    vec4 y = mix(uvs[2], uvs[3], f.x);
    vec4 z = mix(x, y, f.y);
    vec2 screen_uv = z.xy / z.w;

    // Map the screen-space uv into the backdrop's texture-cache rect.
    RectWithEndpoint uv_rect = info.segment.uv_rect;
    vec2 uv = mix(uv_rect.p0, uv_rect.p1, screen_uv);

    vec2 texture_size = vec2(TEX_SIZE(sColor0));
    v_uv = uv / texture_size;
    v_uv_bounds = vec4(uv_rect.p0 + vec2(0.5), uv_rect.p1 - vec2(0.5)) / texture_size.xyxy;

    // Interpolate the uv linearly in screen space (see file comment).
    v_uv *= gl_Position.w;
}

#endif

#ifdef WR_FRAGMENT_SHADER

vec4 pattern_fragment(vec4 color) {
    vec2 uv = v_uv * gl_FragCoord.w;
    uv = clamp(uv, v_uv_bounds.xy, v_uv_bounds.zw);
    vec4 texel = TEX_SAMPLE(sColor0, uv);
    return color * texel;
}

#endif
