/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

void main(void) {
    float alpha = 1.f;
    vec2 local_pos = init_transform_fs(vPos, vLocalRect, alpha);
    float clip_alpha = do_clip(local_pos);
    float inv_alpha = 1.0 - min(alpha, clip_alpha);
    oFragColor = vec4(1, 1, 1, inv_alpha);
}
