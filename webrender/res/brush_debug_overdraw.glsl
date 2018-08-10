/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared,brush

#ifdef WR_VERTEX_SHADER

void brush_vs(
    VertexInfo vi,
    int prim_address,
    RectWithSize local_rect,
    RectWithSize segment_rect,
    ivec3 user_data,
    mat4 transform,
    PictureTask pic_task,
    int brush_flags,
    vec4 unused
) {
}
#endif

#ifdef WR_FRAGMENT_SHADER
Fragment brush_fs() {
    vec4 color = WR_DEBUG_OVERDRAW_COLOR;
    return Fragment(color);
}
#endif

