/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/// Composites a picture with its backdrop using a csS mix-blend-mode.
///
/// Two textures are sampled:
///  - sColor0: the captured backdrop (a readback render task). Its texture coordinates
///    arrives as the standard quad segment uv rect.
///  - sColor1: the picture's own rendered content (the source). Its texture coordinates
///    is passed via the pattern's gpu block.

#include ps_quad

// Backdrop (sColor0) sampling.
varying highp vec2 v_backdrop_uv;
flat varying highp vec4 v_backdrop_uv_bounds;

// Source (sColor1) sampling.
varying highp vec2 v_src_uv;
flat varying highp vec4 v_src_uv_bounds;

// mix-blend op. Packed in to a vector to work around bug 1630356.
flat varying mediump ivec2 v_op;

#ifdef WR_VERTEX_SHADER

void write_uv(vec2 f, RectWithEndpoint uv_rect, vec2 texture_size, out vec2 out_uv, out vec4 out_bounds) {
    vec2 uv = mix(uv_rect.p0, uv_rect.p1, f);
    out_uv = uv / texture_size;
    out_bounds = vec4(uv_rect.p0 + vec2(0.5), uv_rect.p1 - vec2(0.5)) / texture_size.xyxy;
}

void pattern_vertex(PrimitiveInfo info) {
    // Source texture-cache rect (resolved from the source render task).
    vec4 src_uv_rect_raw = fetch_from_gpu_buffer_1f(info.pattern_input.x);
    RectWithEndpoint src_uv_rect = RectWithEndpoint(src_uv_rect_raw.xy, src_uv_rect_raw.zw);

    v_op.x = info.pattern_input.y;

    // Normalized position within the primitive rect.
    RectWithEndpoint rect = info.local_prim_rect;
    vec2 f = (info.local_pos - rect.p0) / rect_size(rect);

    write_uv(f, info.segment.uv_rect, vec2(TEX_SIZE(sColor0)), v_backdrop_uv, v_backdrop_uv_bounds);
    write_uv(f, src_uv_rect, vec2(TEX_SIZE(sColor1)), v_src_uv, v_src_uv_bounds);
}

#endif

#ifdef WR_FRAGMENT_SHADER

vec3 multiply(vec3 cb, vec3 cs) {
    return cb * cs;
}

vec3 screen(vec3 cb, vec3 cs) {
    return cb + cs - (cb * cs);
}

vec3 hard_light(vec3 cb, vec3 cs) {
    vec3 m = multiply(cb, 2.0 * cs);
    vec3 s = screen(cb, 2.0 * cs - 1.0);
    vec3 edge = vec3(0.5, 0.5, 0.5);
    return mix(m, s, step(edge, cs));
}

float color_dodge(float cb, float cs) {
    if (cb == 0.0)
        return 0.0;
    else if (cs == 1.0)
        return 1.0;
    else
        return min(1.0, cb / (1.0 - cs));
}

float color_burn(float cb, float cs) {
    if (cb == 1.0)
        return 1.0;
    else if (cs == 0.0)
        return 0.0;
    else
        return 1.0 - min(1.0, (1.0 - cb) / cs);
}

float soft_light(float cb, float cs) {
    if (cs <= 0.5) {
        return cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb);
    } else {
        float D;

        if (cb <= 0.25)
            D = ((16.0 * cb - 12.0) * cb + 4.0) * cb;
        else
            D = sqrt(cb);

        return cb + (2.0 * cs - 1.0) * (D - cb);
    }
}

vec3 difference(vec3 cb, vec3 cs) {
    return abs(cb - cs);
}

// These functions below are taken from the spec.
float sat(vec3 c) {
    return max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
}

float lum(vec3 c) {
    vec3 f = vec3(0.3, 0.59, 0.11);
    return dot(c, f);
}

vec3 clip_color(vec3 C) {
    float L = lum(C);
    float n = min(C.r, min(C.g, C.b));
    float x = max(C.r, max(C.g, C.b));

    if (n < 0.0 && L != n)
        C = L + (((C - L) * L) / (L - n));

    if (x > 1.0 && x != L)
        C = L + (((C - L) * (1.0 - L)) / (x - L));

    return C;
}

vec3 set_lum(vec3 C, float l) {
    float d = l - lum(C);
    return clip_color(C + d);
}

void set_sat_inner(inout float Cmin, inout float Cmid, inout float Cmax, float s) {
    if (Cmax > Cmin) {
        Cmid = (((Cmid - Cmin) * s) / (Cmax - Cmin));
        Cmax = s;
    } else {
        Cmid = 0.0;
        Cmax = 0.0;
    }
    Cmin = 0.0;
}

vec3 set_sat(vec3 C, float s) {
    if (C.r <= C.g) {
        if (C.g <= C.b) {
            set_sat_inner(C.r, C.g, C.b, s);
        } else {
            if (C.r <= C.b) {
                set_sat_inner(C.r, C.b, C.g, s);
            } else {
                set_sat_inner(C.b, C.r, C.g, s);
            }
        }
    } else {
        if (C.r <= C.b) {
            set_sat_inner(C.g, C.r, C.b, s);
        } else {
            if (C.g <= C.b) {
                set_sat_inner(C.g, C.b, C.r, s);
            } else {
                set_sat_inner(C.b, C.g, C.r, s);
            }
        }
    }
    return C;
}

vec3 hue(vec3 cb, vec3 cs) {
    return set_lum(set_sat(cs, sat(cb)), lum(cb));
}

vec3 saturation(vec3 cb, vec3 cs) {
    return set_lum(set_sat(cb, sat(cs)), lum(cb));
}

vec3 luminosity(vec3 cb, vec3 cs) {
    return set_lum(cb, lum(cs));
}

const int MIX_BLEND_MULTIPLY     = 1;
const int MIX_BLEND_SCREEN       = 2;
const int MIX_BLEND_OVERLAY      = 3;
const int MIX_BLEND_DARKEN       = 4;
const int MIX_BLEND_LIGHTEN      = 5;
const int MIX_BLEND_COLOR_DODGE  = 6;
const int MIX_BLEND_COLOR_BURN   = 7;
const int MIX_BLEND_HARD_LIGHT   = 8;
const int MIX_BLEND_SOFT_LIGHT   = 9;
const int MIX_BLEND_DIFFERENCE   = 10;
const int MIX_BLEND_EXCLUSION    = 11;
const int MIX_BLEND_HUE          = 12;
const int MIX_BLEND_SAUTURATION  = 13;
const int MIX_BLEND_COLOR        = 14;
const int MIX_BLEND_lumINOSITY   = 15;
const int MIX_BLEND_PLUS_LIGHTER = 16;

vec4 pattern_fragment(vec4 base_color) {
    vec2 backdrop_uv = clamp(v_backdrop_uv, v_backdrop_uv_bounds.xy, v_backdrop_uv_bounds.zw);
    vec2 src_uv = clamp(v_src_uv, v_src_uv_bounds.xy, v_src_uv_bounds.zw);

    vec4 cb = texture(sColor0, backdrop_uv);
    vec4 cs = texture(sColor1, src_uv);

    // The mix-blend-mode functions assume no premultiplied alpha.
    if (cb.a != 0.0) {
        cb.rgb /= cb.a;
    }

    if (cs.a != 0.0) {
        cs.rgb /= cs.a;
    }

    // Return yellow if none of the branches match (shouldn't happen).
    vec4 result = vec4(1.0, 1.0, 0.0, 1.0);

    // On Android v_op has been packed in to a vector to avoid a driver bug
    // on Adreno 3xx. However, this runs in to another Adreno 3xx driver bug
    // where the switch doesn't match any cases. Unpacking the value from the
    // vec in to a local variable prior to the switch works around this, but
    // gets optimized away by glslopt. Adding a bitwise AND prevents that.
    // See bug 1726755.
    // default: default: to appease angle_shader_validation
    switch (v_op.x & 0xFF) {
        case MIX_BLEND_MULTIPLY:
            result.rgb = multiply(cb.rgb, cs.rgb);
            break;
        case MIX_BLEND_OVERLAY:
            // Overlay is inverse of hard_light
            result.rgb = hard_light(cs.rgb, cb.rgb);
            break;
        case MIX_BLEND_DARKEN:
            result.rgb = min(cs.rgb, cb.rgb);
            break;
        case MIX_BLEND_LIGHTEN:
            result.rgb = max(cs.rgb, cb.rgb);
            break;
        case MIX_BLEND_COLOR_DODGE:
            result.r = color_dodge(cb.r, cs.r);
            result.g = color_dodge(cb.g, cs.g);
            result.b = color_dodge(cb.b, cs.b);
            break;
        case MIX_BLEND_COLOR_BURN:
            result.r = color_burn(cb.r, cs.r);
            result.g = color_burn(cb.g, cs.g);
            result.b = color_burn(cb.b, cs.b);
            break;
        case MIX_BLEND_HARD_LIGHT:
            result.rgb = hard_light(cb.rgb, cs.rgb);
            break;
        case MIX_BLEND_SOFT_LIGHT:
            result.r = soft_light(cb.r, cs.r);
            result.g = soft_light(cb.g, cs.g);
            result.b = soft_light(cb.b, cs.b);
            break;
        case MIX_BLEND_DIFFERENCE:
            result.rgb = difference(cb.rgb, cs.rgb);
            break;
        case MIX_BLEND_HUE:
            result.rgb = hue(cb.rgb, cs.rgb);
            break;
        case MIX_BLEND_SAUTURATION:
            result.rgb = saturation(cb.rgb, cs.rgb);
            break;
        case MIX_BLEND_COLOR:
            result.rgb = set_lum(cs.rgb, lum(cb.rgb));
            break;
        case MIX_BLEND_lumINOSITY:
            result.rgb = luminosity(cb.rgb, cs.rgb);
            break;
        case MIX_BLEND_SCREEN:
        case MIX_BLEND_EXCLUSION:
        case MIX_BLEND_PLUS_LIGHTER:
            // This should be unreachable, since we implement
            // MixBlendMode::screen, MixBlendMode::exclusion and
            // MixBlendMode::PlusLighter using glBlendFuncseparate.
            break;
        default:
            break;
    }

    result.rgb = (1.0 - cb.a) * cs.rgb + cb.a * result.rgb;
    result.a = cs.a;
    result.rgb *= result.a;

    return base_color * result;
}

#endif
