/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/// Box-shadow blur rendering via the quad infrastructure.
///
/// GPU buffer layout at pattern_input.x (5 blocks):
///   [0] alloc_size.x, alloc_size.y, dest_rect_size.x, dest_rect_size.y
///   [1] dest_rect_offset.x, dest_rect_offset.y, clip_mode (0=outset, 1=inset), 0
///   [2] element_offset_rel_prim.x, element_offset_rel_prim.y, element_size.x, element_size.y
///   [3] element_radius.tl.w, element_radius.tl.h, element_radius.tr.w, element_radius.tr.h
///   [4] element_radius.br.w, element_radius.br.h, element_radius.bl.w, element_radius.bl.h
///
/// For outset: prim_rect == dest_rect, element_offset_rel_prim is typically negative
///             (element sits inside the inflated shadow rect).
/// For inset:  prim_rect is the element rect; dest_rect is the shadow area
///             (offset and potentially smaller), and blur alpha is inverted.
///
/// Element clipping (clip-out for outset, clip-in for inset) is handled analytically
/// in this shader via a rounded-rect SDF, enabling Direct rendering for the common case.

#include ps_quad,ellipse,shared

// xy: position relative to dest_rect.p0, for nine-patch UV sampling.
// zw: position in local (primitive) space, for element clip SDF evaluation.
varying highp vec4 v_shadow_pos_local_pos;

// xy: 1 / alloc_size — UV denominator.
// z: 1.0 for inset (blur alpha inverted, element clip-in), 0.0 for outset (clip-out).
//    Packed in to a vector to work around bug 1630356.
// w: unused.
flat varying highp vec4 v_uv_scale_inset;

// Nine-patch edges: .xy = 0.5 (near), .zw = dest_rect_size/alloc_size - 0.5 (far).
flat varying highp vec4 v_edge;

// Atlas UV rect (normalized) and sample bounds.
flat varying highp vec4 v_uv_rect;
flat varying highp vec4 v_uv_bounds;

// Element clip SDF data: corner ellipse centers (xy) and radii (zw).
// Plane normals and constants, as well as the element rect bounds, are all
// reconstructed in the fragment shader from these centers and radii. Keeping
// the varying count low matters here: this shader sits close to the GLES3
// minimum of 15 varying vectors, and older GPUs fail to link a program that
// exceeds their varying limit, falling back to software (bug 2043249).
flat varying highp vec4 vElemCenter_Radius_TL;
flat varying highp vec4 vElemCenter_Radius_TR;
flat varying highp vec4 vElemCenter_Radius_BR;
flat varying highp vec4 vElemCenter_Radius_BL;

#ifdef WR_VERTEX_SHADER

void pattern_vertex(PrimitiveInfo info) {
    vec4 data0 = fetch_from_gpu_buffer_1f(info.pattern_input.x);
    vec4 data1 = fetch_from_gpu_buffer_1f(info.pattern_input.x + 1);
    vec4 data2 = fetch_from_gpu_buffer_1f(info.pattern_input.x + 2);
    vec4 data3 = fetch_from_gpu_buffer_1f(info.pattern_input.x + 3);
    vec4 data4 = fetch_from_gpu_buffer_1f(info.pattern_input.x + 4);

    vec2 alloc_size     = data0.xy;
    vec2 dest_rect_size = data0.zw;
    vec2 dest_rect_off  = data1.xy;
    v_uv_scale_inset    = vec4(vec2(1.0) / alloc_size, data1.z, 0.0);

    v_shadow_pos_local_pos = vec4(
        info.local_pos - info.local_prim_rect.p0 - dest_rect_off,
        info.local_pos
    );

    v_edge = vec4(
        0.5,
        0.5,
        dest_rect_size.x / alloc_size.x - 0.5,
        dest_rect_size.y / alloc_size.y - 0.5
    );

    vec2 texture_size = vec2(TEX_SIZE(sColor0));
    v_uv_rect = vec4(info.segment.uv_rect.p0, info.segment.uv_rect.p1) / texture_size.xyxy;
    v_uv_bounds = vec4(
        info.segment.uv_rect.p0 + vec2(0.5),
        info.segment.uv_rect.p1 - vec2(0.5)
    ) / texture_size.xyxy;

    // Element clip: compute corner centers and radii. The half-space plane
    // constants and the element rect bounds are reconstructed from these in the
    // fragment shader, to keep the varying count low (see bug 2043249).
    vec2 elem_p0 = info.local_prim_rect.p0 + data2.xy;
    vec2 elem_p1 = elem_p0 + data2.zw;

    vec2 r_tl = data3.xy;
    vec2 r_tr = data3.zw;
    vec2 r_br = data4.xy;
    vec2 r_bl = data4.zw;

    vElemCenter_Radius_TL = vec4(elem_p0 + r_tl, r_tl);
    vElemCenter_Radius_TR = vec4(elem_p1.x - r_tr.x, elem_p0.y + r_tr.y, r_tr);
    vElemCenter_Radius_BR = vec4(elem_p1 - r_br, r_br);
    vElemCenter_Radius_BL = vec4(elem_p0.x + r_bl.x, elem_p1.y - r_bl.y, r_bl);
}

#endif

#ifdef WR_FRAGMENT_SHADER

vec4 pattern_fragment(vec4 base_color) {
    vec2 shadow_pos = v_shadow_pos_local_pos.xy;
    vec2 local_pos  = v_shadow_pos_local_pos.zw;
    vec2 uv_scale   = v_uv_scale_inset.xy;
    float inset     = v_uv_scale_inset.z;

    vec2 uv_linear = shadow_pos * uv_scale;

    vec2 uv = clamp(uv_linear, vec2(0.0), v_edge.xy);
    uv += max(vec2(0.0), uv_linear - v_edge.zw);

    uv = mix(v_uv_rect.xy, v_uv_rect.zw, uv);
    uv = clamp(uv, v_uv_bounds.xy, v_uv_bounds.zw);

    float alpha = TEX_SAMPLE(sColor0, uv).r;

    // Inset shadows: the blur texture encodes the shadow shape interior
    // (alpha=1 inside shadow_rect). We invert to get alpha=1 at the element
    // boundary fading toward zero at the shadow_rect center.
    alpha = mix(alpha, 1.0 - alpha, inset);

    // Element clip: clip-out for outset (inset=0), clip-in for inset (inset=1).
    // distance_to_rounded_rect returns negative inside the element rect, positive outside.
    // distance_aa returns 1 when dist < 0 (inside) and 0 when dist > 0 (outside).
    float aa_range = compute_aa_range(local_pos);

    vec2 c_tl = vElemCenter_Radius_TL.xy;
    vec2 c_tr = vElemCenter_Radius_TR.xy;
    vec2 c_br = vElemCenter_Radius_BR.xy;
    vec2 c_bl = vElemCenter_Radius_BL.xy;

    vec2 r_tl = vElemCenter_Radius_TL.zw;
    vec2 r_tr = vElemCenter_Radius_TR.zw;
    vec2 r_br = vElemCenter_Radius_BR.zw;
    vec2 r_bl = vElemCenter_Radius_BL.zw;

    // Reconstruct plane normals from the stored radii.
    vec2 n_tl = -r_tl.yx;
    vec2 n_tr = vec2(r_tr.y, -r_tr.x);
    vec2 n_br = r_br.yx;
    vec2 n_bl = vec2(-r_bl.y, r_bl.x);

    // Reconstruct the corner half-space plane constants from the centers and
    // radii. Each plane passes through the point where the corner arc meets the
    // adjacent straight edge (e.g. the TL plane through (elem_p0.x, center.y)).
    vec3 elem_plane_tl = vec3(n_tl, dot(n_tl, vec2(c_tl.x - r_tl.x, c_tl.y)));
    vec3 elem_plane_tr = vec3(n_tr, dot(n_tr, vec2(c_tr.x,          c_tr.y - r_tr.y)));
    vec3 elem_plane_br = vec3(n_br, dot(n_br, vec2(c_br.x + r_br.x, c_br.y)));
    vec3 elem_plane_bl = vec3(n_bl, dot(n_bl, vec2(c_bl.x,          c_bl.y + r_bl.y)));

    // Reconstruct the element rect bounds from the TL and BR corner data.
    vec4 elem_bounds = vec4(c_tl - r_tl, c_br + r_br);

    float elem_dist = distance_to_rounded_rect(
        local_pos,
        elem_plane_tl, vec4(c_tl, inverse_radii_squared(r_tl)),
        elem_plane_tr, vec4(c_tr, inverse_radii_squared(r_tr)),
        elem_plane_br, vec4(c_br, inverse_radii_squared(r_br)),
        elem_plane_bl, vec4(c_bl, inverse_radii_squared(r_bl)),
        elem_bounds
    );

    // Outset (inset=0): dist < 0 = inside element → should be clipped out → use -elem_dist.
    // Inset (inset=1): dist < 0 = inside element → should be kept → use elem_dist.
    float element_clip = distance_aa(aa_range, mix(-elem_dist, elem_dist, inset));

    return base_color * alpha * element_clip;
}

#endif
