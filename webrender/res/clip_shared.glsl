#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef WR_VERTEX_SHADER

in int aClipRenderTaskIndex;
in int aClipLayerIndex;
in int aClipDataIndex;

struct CacheClipInstance {
    int render_task_index;
    int layer_index;
    int data_index;
};

CacheClipInstance fetch_clip_item(int index) {
    CacheClipInstance cci;

    cci.render_task_index = aClipRenderTaskIndex;
    cci.layer_index = aClipLayerIndex;
    cci.data_index = aClipDataIndex;

    return cci;
}

// The transformed vertex function that always covers the whole clip area,
// which is the intersection of all clip instances of a given primitive
TransformVertexInfo write_clip_tile_vertex(vec4 local_rect,
                                           Layer layer,
                                           ClipArea area) {
    vec2 lp0_base = local_rect.xy;
    vec2 lp1_base = local_rect.xy + local_rect.zw;

    vec2 lp0 = clamp_rect(clamp_rect(lp0_base, local_rect),
                          layer.local_clip_rect);
    vec2 lp1 = clamp_rect(clamp_rect(lp1_base, local_rect),
                          layer.local_clip_rect);

    vec4 clipped_local_rect = vec4(lp0, lp1 - lp0);

    vec2 p0 = lp0;
    vec2 p1 = vec2(lp1.x, lp0.y);
    vec2 p2 = vec2(lp0.x, lp1.y);
    vec2 p3 = lp1;

    vec4 t0 = layer.transform * vec4(p0, 0, 1);
    vec4 t1 = layer.transform * vec4(p1, 0, 1);
    vec4 t2 = layer.transform * vec4(p2, 0, 1);
    vec4 t3 = layer.transform * vec4(p3, 0, 1);

    vec2 tp0 = t0.xy / t0.w;
    vec2 tp1 = t1.xy / t1.w;
    vec2 tp2 = t2.xy / t2.w;
    vec2 tp3 = t3.xy / t3.w;

    // compute a CSS space aligned bounding box
    vec2 min_pos = min(min(tp0.xy, tp1.xy), min(tp2.xy, tp3.xy));
    vec2 max_pos = max(max(tp0.xy, tp1.xy), max(tp2.xy, tp3.xy));

    // clamp to the tile boundaries, in device space
    vec2 min_pos_clamped = clamp(min_pos * uDevicePixelRatio,
                                 area.screen_origin_target_index.xy,
                                 area.screen_origin_target_index.xy + area.task_bounds.zw - area.task_bounds.xy);

    vec2 max_pos_clamped = clamp(max_pos * uDevicePixelRatio,
                                 area.screen_origin_target_index.xy,
                                 area.screen_origin_target_index.xy + area.task_bounds.zw - area.task_bounds.xy);

    // compute the device space position of this vertex
    vec2 clamped_pos = mix(min_pos_clamped,
                           max_pos_clamped,
                           aPosition.xy);

    // compute the point position in side the layer, in CSS space
    vec4 layer_pos = get_layer_pos(clamped_pos / uDevicePixelRatio, layer);

    // apply the task offset
    vec2 final_pos = clamped_pos - area.screen_origin_target_index.xy + area.task_bounds.xy;

    gl_Position = uTransform * vec4(final_pos, 0.0, 1.0);

    return TransformVertexInfo(layer_pos.xyw, clamped_pos, clipped_local_rect);
}

#endif //WR_VERTEX_SHADER
