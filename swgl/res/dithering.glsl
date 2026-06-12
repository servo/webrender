/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef WR_FRAGMENT_SHADER

#ifdef WR_FEATURE_DITHERING

// A 8x8 texture producing values in the range [0, 64/255].
uniform sampler2D sDither;

vec4 dither(vec4 color) {
    const int matrix_mask = 7;

    ivec2 pos = ivec2(gl_FragCoord.xy) & ivec2(matrix_mask);
    float noise_normalized = texelFetch(sDither, pos, 0).r * 255.0 / 64.0;
    // Center the range around zero and scale it down by some amount to
    // avoid hardware-specific rounding differences.
    float noise = (noise_normalized - 0.5) / 256.0 * 0.87;

    return color + vec4(noise, noise, noise, 0);
}

#else

vec4 dither(vec4 color) {
    return color;
}

#endif //WR_FEATURE_DITHERING

#endif //WR_FRAGMENT_SHADER
