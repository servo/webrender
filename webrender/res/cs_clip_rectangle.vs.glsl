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

struct ClipRect {
    vec4 rect;
    vec4 dummy;
};

ClipRect fetch_clip_rect(int index) {
    ClipRect rect;

    ivec2 uv = get_fetch_uv_2(index);

    rect.rect = texelFetchOffset(sData32, uv, 0, ivec2(0, 0));
    //rect.dummy = texelFetchOffset(sData32, uv, 0, ivec2(1, 0));
    rect.dummy = vec4(0.0, 0.0, 0.0, 0.0);

    return rect;
}

struct ClipCorner {
    vec4 rect;
    vec4 outer_inner_radius;
};

ClipCorner fetch_clip_corner(int index) {
    ClipCorner corner;

    ivec2 uv = get_fetch_uv_2(index);

    corner.rect = texelFetchOffset(sData32, uv, 0, ivec2(0, 0));
    corner.outer_inner_radius = texelFetchOffset(sData32, uv, 0, ivec2(1, 0));

    return corner;
}

struct ClipData {
    ClipRect rect;
    ClipCorner top_left;
    ClipCorner top_right;
    ClipCorner bottom_left;
    ClipCorner bottom_right;
};

ClipData fetch_clip(int index) {
    ClipData clip;

    clip.rect = fetch_clip_rect(index + 0);
    clip.top_left = fetch_clip_corner(index + 1);
    clip.top_right = fetch_clip_corner(index + 2);
    clip.bottom_left = fetch_clip_corner(index + 3);
    clip.bottom_right = fetch_clip_corner(index + 4);

    return clip;
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
