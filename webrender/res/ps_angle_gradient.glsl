/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared

flat varying int vGradientAddress;
flat varying float vGradientRepeat;

flat varying vec2 vScaledDir;
flat varying vec2 vStartPoint;

varying vec2 vPos;

#ifdef WR_VERTEX_SHADER

#define VECS_PER_GRADIENT           2

struct Gradient {
    vec4 start_end_point;
    vec4 extend_mode;
};

Gradient fetch_gradient(int address) {
    vec4 data[2] = fetch_from_resource_cache_2(address);
    return Gradient(data[0], data[1]);
}

void main(void) {
    Primitive prim = load_primitive();
    Gradient gradient = fetch_gradient(prim.specific_prim_address);

    VertexInfo vi = write_vertex(prim.local_rect,
                                 prim.local_clip_rect,
                                 prim.z,
                                 prim.scroll_node,
                                 prim.task,
                                 prim.local_rect);

    vPos = vi.local_pos - prim.local_rect.p0;

    vec2 start_point = gradient.start_end_point.xy;
    vec2 end_point = gradient.start_end_point.zw;
    vec2 dir = end_point - start_point;

    vStartPoint = start_point;
    vScaledDir = dir / dot(dir, dir);

    vGradientAddress = prim.specific_prim_address + VECS_PER_GRADIENT;

    // Whether to repeat the gradient instead of clamping.
    vGradientRepeat = float(int(gradient.extend_mode.x) != EXTEND_MODE_CLAMP);

    write_clip(vi.screen_pos, prim.clip_area);
}
#endif

#ifdef WR_FRAGMENT_SHADER
void main(void) {
    float offset = dot(vPos - vStartPoint, vScaledDir);

    vec4 color = sample_gradient(vGradientAddress,
                                 offset,
                                 vGradientRepeat);

    oFragColor = color * do_clip();
}
#endif
