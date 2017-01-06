#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

struct ClipCorner {
    vec4 rect;
    vec4 clip_ref_radius;
};

ClipCorner fetch_clip_corner(int index) {
    ClipCorner corner;

    ivec2 uv = get_fetch_uv_2(index);

    corner.rect = texelFetchOffset(sData32, uv, 0, ivec2(0, 0));
    corner.clip_ref_radius = texelFetchOffset(sData32, uv, 0, ivec2(1, 0));

    return corner;
}

void main(void) {
    CacheClipInstance cci = fetch_clip_item(gl_InstanceID);
    ClipArea area = fetch_clip_area(cci.render_task_index);
    Layer layer = fetch_layer(cci.layer_index);
    ClipCorner clip = fetch_clip_corner(cci.data_index);
    vec4 local_rect = clip.rect;

    TransformVertexInfo vi = write_clip_tile_vertex(local_rect,
                                                    layer,
                                                    area);
    vLocalRect = vi.clipped_local_rect;
    vPos = vi.local_pos;

    vClipRect = vec4(local_rect.xy, local_rect.xy + local_rect.zw);
    vClipRadius = clip.clip_ref_radius.zw;
    vClipRef = clip.clip_ref_radius.xy;
}
