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

    oFragColor = result;
}
