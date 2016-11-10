#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef WR_VERTEX_SHADER

struct CacheClipInstance {
    int render_task_index;
    int layer_index;
    int data_index;
    int pad;
};

CacheClipInstance fetch_clip_item(int index) {
    CacheClipInstance cci;

    int offset = index * 1;

    ivec4 data0 = int_data[offset + 0];

    cci.render_task_index = data0.x;
    cci.layer_index = data0.y;
    cci.data_index = data0.z;
    cci.pad = 0;

    return cci;
}

// The transformed vertex function that always covers the whole tile with the primitive
TransformVertexInfo write_clip_tile_vertex(vec4 local_clip_rect,
                                           Layer layer,
                                           Tile tile) {
    vec2 lp0_base = local_clip_rect.xy;
    vec2 lp1_base = local_clip_rect.xy + local_clip_rect.zw;

    vec2 lp0 = clamp_rect(lp0_base, layer.local_clip_rect);
    vec2 lp1 = clamp_rect(lp1_base, layer.local_clip_rect);
    vec4 clipped_local_rect = vec4(lp0, lp1 - lp0);

    // always cover the whole tile
    // compute the device space position of this vertex
    vec2 clamped_pos = tile.screen_origin_task_origin.xy +
                       tile.size_target_index.xy * aPosition.xy;

    // compute the point position in side the layer, in CSS space
    vec4 layer_pos = get_layer_pos(clamped_pos / uDevicePixelRatio, layer);

    // apply the task offset
    vec2 final_pos = clamped_pos + tile.screen_origin_task_origin.zw - tile.screen_origin_task_origin.xy;

    gl_Position = uTransform * vec4(final_pos, 0, 1);

    return TransformVertexInfo(layer_pos.xyw, clamped_pos, clipped_local_rect);
}

#endif //WR_VERTEX_SHADER
