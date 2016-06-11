#line 1

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

void main(void) {
    vec4 p0 = texture(sLayer0, vUv0);
    vec4 p1 = texture(sLayer1, vUv1);

    vec4 result = fetch_initial_color();

    result = mix(result, p0, p0.a * vLayerValues0.x);
    result = mix(result, p1, p1.a * vLayerValues0.y);

    oFragColor = result;

    //oFragColor = vec4(1, 0, 0, 1);
    /*
    vec4 result = fetch_initial_color();

    if (is_prim_valid(vPartialRects0.x)) {
        vec4 prim_color = texture(sLayer0, vUv0);
        result = mix(result, prim_color, prim_color.a * vLayerValues0.x);
    }
    if (is_prim_valid(vPartialRects0.y)) {
        vec4 prim_color = texture(sLayer1, vUv1);
        result = mix(result, prim_color, prim_color.a * vLayerValues0.y);
    }

    oFragColor = result;*/
}
