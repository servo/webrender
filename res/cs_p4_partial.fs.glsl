#line 1

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

void main(void) {
    vec4 result = fetch_initial_color();

    if (is_prim_valid(vPartialRects0.x)) {
        vec4 prim_color = texture(sLayer0, vUv0);
        result = mix(result, prim_color, prim_color.a * vLayerValues0.x);
    }
    if (is_prim_valid(vPartialRects0.y)) {
        vec4 prim_color = texture(sLayer1, vUv1);
        result = mix(result, prim_color, prim_color.a * vLayerValues0.y);
    }
    if (is_prim_valid(vPartialRects0.z)) {
        vec4 prim_color = texture(sLayer2, vUv2);
        result = mix(result, prim_color, prim_color.a * vLayerValues0.z);
    }
    if (is_prim_valid(vPartialRects0.w)) {
        vec4 prim_color = texture(sLayer3, vUv3);
        result = mix(result, prim_color, prim_color.a * vLayerValues0.w);
    }

    oFragColor = result;
}
