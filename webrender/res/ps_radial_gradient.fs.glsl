/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

uniform sampler2D sGradients;

void main(void) {
    vec2 texture_size = vec2(textureSize(sGradients, 0));

    // Find where we intersect the gradient line
    float x = distance(vPos, vCenter) * vInvRadius;

    // Either saturate or modulo the offset depending on repeat mode, then scale to number of
    // gradient color entries (texture width / 2).
    x = mix(clamp(x, 0.0, 1.0), fract(x), vGradientRepeat) * 0.5 * texture_size.x;

    // Start at the center of first color in the nearest 2-color entry, then offset with the
    // fractional remainder to interpolate between the colors. Rely on texture clamping when
    // outside of valid range.
    x = 2.0 * floor(x) + 0.5 + fract(x);

    // Use linear filtering to mix in the low bits (vGradientIndex + 1) with the high
    // bits (vGradientIndex)
    float y = vGradientIndex + 1.0 / 256.0;
    oFragColor = dither(texture(sGradients, vec2(x, y) / texture_size));
}
