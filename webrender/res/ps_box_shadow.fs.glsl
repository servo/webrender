/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

void main(void) {
    vec2 uv = min(vec2(1.0), vMirrorPoint - abs(vUv.xy - vMirrorPoint));
    uv = mix(vCacheUvRectCoords.xy, vCacheUvRectCoords.zw, uv);
    oFragColor = vColor * texture(sCache, vec3(uv, vUv.z));

#ifdef WR_FEATURE_CLIP
    float alpha = 1.0;
    alpha = min(alpha, do_clip());

    // Need to do an inverted clip here iff we have a complex clip
    // and we're an outer box shadow.
    alpha = vIsInset == 1.0 ? alpha : 1.0 - alpha;
    oFragColor = oFragColor * vec4(1.0, 1.0, 1.0, alpha);
#endif
}
