#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

void main() {
    switch (gl_VertexID) {
        case 0: vUv = vec2(0.0, 0.0); break;
        case 1: vUv = vec2(1.0, 0.0); break;
        case 2: vUv = vec2(1.0, 1.0); break;
        case 3: vUv = vec2(0.0, 1.0); break;
        default: vUv = vec2(0.0, 0.0);
    };
    gl_Position = vec4(aPosition.xy * 2.0 - 1.0, 0.0, 1.0);
}
