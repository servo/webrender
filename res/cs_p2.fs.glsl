#line 1

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

void main(void) {
    vec4 prim_colors[2];
    prim_colors[0] = texture(sLayer0, vUv0);
    prim_colors[1] = texture(sLayer1, vUv1);

    vec4 result = fetch_initial_color();
    result = mix(result, prim_colors[0], prim_colors[0].a * vLayerValues.x);
    result = mix(result, prim_colors[1], prim_colors[1].a * vLayerValues.y);

    oFragColor = result;
}
