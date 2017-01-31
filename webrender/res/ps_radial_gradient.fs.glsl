/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

uniform sampler2D sGradients;

void main(void) {
    vec2 cd = vEndCenter - vStartCenter;
    vec2 pd = vPos - vStartCenter;
    float rd = vEndRadius - vStartRadius;

    // Solve for t in length(t * cd - pd) = vStartRadius + t * rd
    // using a quadratic equation in form of At^2 - 2Bt + C = 0
    float A = dot(cd, cd) - rd * rd;
    float B = dot(pd, cd) + vStartRadius * rd;
    float C = dot(pd, pd) - vStartRadius * vStartRadius;

    float x;
    if (A == 0.0) {
        // Since A is 0, just solve for -2Bt + C = 0
        if (B == 0.0) {
            discard;
        }
        float t = 0.5 * C / B;
        if (vStartRadius + rd * t >= 0.0) {
            x = t;
        } else {
            discard;
        }
    } else {
        float discr = B * B - A * C;
        if (discr < 0.0) {
            discard;
        }
        discr = sqrt(discr);
        float t0 = (B + discr) / A;
        float t1 = (B - discr) / A;
        if (vStartRadius + rd * t0 >= 0.0) {
            x = t0;
        } else if (vStartRadius + rd * t1 >= 0.0) {
            x = t1;
        } else {
            discard;
        }
    }

    vec2 texture_size = vec2(textureSize(sGradients, 0));

    // Either saturate or modulo the offset depending on repeat mode, then scale to number of
    // gradient color entries (texture width / 2).
    x = mix(clamp(x, 0.0, 1.0), fract(x), vGradientRepeat) * 0.5 * texture_size.x;

    // Start at the center of first color in the nearest 2-color entry, then offset with the
    // fractional remainder to interpolate between the colors. Rely on texture clamping when
    // outside of valid range.
    x = 2.0 * floor(x) + 0.5 + fract(x);

    // Normalize the texture coordates so we can use texture() for bilinear filtering.
    oFragColor = texture(sGradients, vec2(x, vGradientIndex) / texture_size);
}
