/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include ps_quad,sample_color0,blend

// x: filter op, y: component-transfer lookup table GPU buffer address.
// Must be highp: the GPU buffer address can exceed the mediump range.
flat varying highp ivec2 v_op_table_address_vec;
#define v_op v_op_table_address_vec.x
#define v_table_address v_op_table_address_vec.y

// Single scalar packed into a vec2 to work around a driver bug.
flat varying mediump vec2 v_amount;

// Color matrix and offset for matrix-based filters (grayscale, hue-rotate,
// saturate, sepia, color-matrix).
flat varying highp mat4 v_color_mat;
flat varying mediump vec4 v_color_offset;

// The component-transfer function to use for each channel, stored as floats to
// work around driver bugs with integer varyings (see brush_blend.glsl).
flat varying mediump vec4 v_funcs;

#ifdef WR_VERTEX_SHADER

void pattern_vertex(PrimitiveInfo info) {
    // The source maps to the primitive rect; the uv rect is provided via the
    // segment (resolved from the source render task).
    vec2 f = (info.local_pos - info.local_prim_rect.p0) / rect_size(info.local_prim_rect);
    vs_init_sample_color0(f, info.segment.uv_rect);

    int filter_mode = info.pattern_input.x;
    float amount = float(info.pattern_input.y) / 65536.0;

    v_op = filter_mode & 0xffff;
    v_amount.x = amount;

    v_funcs.r = float((filter_mode >> 28) & 0xf);
    v_funcs.g = float((filter_mode >> 24) & 0xf);
    v_funcs.b = float((filter_mode >> 20) & 0xf);
    v_funcs.a = float((filter_mode >> 16) & 0xf);

    SetupFilterParams(
        v_op,
        amount,
        info.pattern_input.y,
        v_color_offset,
        v_color_mat,
        v_table_address
    );
}

#endif

#ifdef WR_FRAGMENT_SHADER

vec4 pattern_fragment(vec4 base_color) {
    vec4 Cs = fs_sample_color0();

    float alpha;
    vec3 rgb;
    CalculateFilter(
        Cs,
        v_op,
        v_amount.x,
        v_table_address,
        v_color_offset,
        v_color_mat,
        v_funcs,
        rgb,
        alpha
    );

    return base_color * vec4(rgb, 1.0) * alpha;
}

#endif
