/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

float rounded_rect(vec2 pos) {
    float pixels_per_fragment = length(fwidth(pos.xy));
    float nudge = 0.5 * pixels_per_fragment;

    // TODO(gw): Support ellipse clip!
    float d = max(0.0, distance(pos, vClipRefPoint_Radius.xy));
    d = (d - vClipRefPoint_Radius.z + nudge) / pixels_per_fragment;

    return 1.0 - smoothstep(0.0, 1.0, d);
}

void main(void) {
    float alpha = 1.f;
    vec2 local_pos = init_transform_fs(vPos, vLocalRect, alpha);

    float clip_alpha = rounded_rect(local_pos);

    oFragColor = vec4(min(alpha, clip_alpha), 0.0, 0.0, 1.0);
}
