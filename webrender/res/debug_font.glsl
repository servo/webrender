//import shared,shared_other

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

varying vec2 vColorTexCoord;
varying vec4 vColor;

#ifdef WR_VERTEX_SHADER
in vec4 aColor;
in vec4 aColorTexCoord;

void main(void) {
    vColor = aColor;
    vColorTexCoord = aColorTexCoord.xy;
    vec4 pos = vec4(aPosition, 1.0);
    pos.xy = floor(pos.xy * uDevicePixelRatio + 0.5) / uDevicePixelRatio;
    gl_Position = uTransform * pos;
}
#endif

#ifdef WR_FRAGMENT_SHADER
void main(void) {
    float alpha = texture(sColor0, vec3(vColorTexCoord.xy, 0.0)).r;
    oFragColor = vec4(vColor.xyz, vColor.w * alpha);
}
#endif
