#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

void main(void) {
    int offset = gl_InstanceID * 1;
    ivec4 data0 = int_data[offset + 0];
    Tile tile = fetch_tile(data0.x);

    vec2 final_pos = tile.screen_origin_task_origin.zw +
                     tile.size_target_index.xy * aPosition.xy;

    gl_Position = uTransform * vec4(final_pos, 0, 1);
}
