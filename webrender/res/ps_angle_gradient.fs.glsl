/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

uniform sampler2D sGradients;

void main(void) {
    vec2 pos = mod(vPos, vTileRepeat);

    if (pos.x >= vTileSize.x ||
        pos.y >= vTileSize.y) {
        discard;
    }

    // Normalized offset of this vertex within the gradient, before clamp/repeat.
    float offset = dot(pos - vStartPoint, vScaledDir);

    vec2 texture_size = vec2(textureSize(sGradients, 0));

    // Either saturate or modulo the offset depending on repeat mode, then scale to number of
    // gradient color entries (texture width / 2).
    float x = mix(clamp(offset, 0.0, 1.0), fract(offset), vGradientRepeat) * 0.5 * texture_size.x;

    x = 2.0 * floor(x) + 0.5 + fract(x);

    // Use linear filtering to mix in the low bits (vGradientIndex + 1) with the high
    // bits (vGradientIndex)
    float y = vGradientIndex * 2.0 + 0.5 + 1.0 / 256.0;

#ifdef WR_FEATURE_DITHERING
    oFragColor = dither(texture(sGradients, vec2(x, y) / texture_size));
#else
    oFragColor = texture(sGradients, vec2(x, y) / texture_size);
#endif
}
