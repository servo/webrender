/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

uniform usampler2D sGradients;

void main(void) {
    vec2 texture_size = vec2(textureSize(sGradients, 0));

    // Either saturate or modulo the offset depending on repeat mode, then scale to number of
    // gradient color entries (texture width / 2).
    float x = mix(clamp(vOffset, 0.0, 1.0), fract(vOffset), vGradientRepeat) * 0.5 * texture_size.x;

    // Grab the colors from the two color entry and interpolate between them using x's
    // fractional remainder.
    int x0 = int(2.0 * floor(x));
    int x1 = x0 + 1;

    uvec4 color0 = texelFetch(sGradients, ivec2(x0, vGradientIndex), 0);
    uvec4 color1 = texelFetch(sGradients, ivec2(x1, vGradientIndex), 0);

    vec4 color = mix(color0, color1, fract(x));

    oFragColor = dither(color * (1.0 / 65535.0));
}
