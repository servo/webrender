#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

struct CacheClipInstance {
    int render_task_index;
    int layer_index;
    int clip_index;
    int pad;
};

CacheClipInstance fetch_clip_item(int index) {
    CacheClipInstance cci;

    int offset = index * 1;

    ivec4 data0 = int_data[offset + 0];

    cci.render_task_index = data0.x;
    cci.layer_index = data0.y;
    cci.clip_index = data0.z;
    cci.pad = 0;

    return cci;
}

void main(void) {
    CacheClipInstance cci = fetch_clip_item(gl_InstanceID);
    Tile tile = fetch_tile(cci.render_task_index);
    Layer layer = fetch_layer(cci.layer_index);
    ClipData clip = fetch_clip(cci.clip_index);
    vec4 local_rect = clip.rect.rect;

    TransformVertexInfo vi = write_transform_vertex(local_rect,
                                                    local_rect,
                                                    layer,
                                                    tile);
    vLocalRect = vi.clipped_local_rect;
    vPos = vi.local_pos;

    vClipRect = vec4(local_rect.xy, local_rect.xy + local_rect.zw);
    vClipRadius = vec4(clip.top_left.outer_inner_radius.x,
                       clip.top_right.outer_inner_radius.x,
                       clip.bottom_right.outer_inner_radius.x,
                       clip.bottom_left.outer_inner_radius.x);
}
