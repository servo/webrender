#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

uniform sampler2D sSplitGeometry;

struct SplitGeometry {
    vec2 points[4];
};

SplitGeometry fetch_split_geometry(int index) {
    ivec2 uv = get_fetch_uv(index, VECS_PER_SPLIT_GEOM);
    vec4 data0 = texelFetchOffset(sSplitGeometry, uv, 0, ivec2(0, 0));
    vec4 data1 = texelFetchOffset(sSplitGeometry, uv, 0, ivec2(1, 0));
    return SplitGeometry(vec2[4](data0.xy, data0.zw, data1.xy, data1.zw));
}

void main(void) {
    PrimitiveInstance pi = fetch_prim_instance();
    SplitGeometry geometry = fetch_split_geometry(pi.specific_prim_index);
    AlphaBatchTask src_task = fetch_alpha_batch_task(pi.user_data.x);
    Layer layer = fetch_layer(pi.layer_index);

    vec2 normalized_pos = mix(
        mix(geometry.points[0], geometry.points[1], aPosition.x),
        mix(geometry.points[3], geometry.points[2], aPosition.x),
        aPosition.y);
    vec2 local_pos = normalized_pos * src_task.size; // + layer.local_clip_rect.p0
    vec4 world_pos_homogen = layer.transform * vec4(local_pos, 0.0, 1.0);
    vec2 world_pos = world_pos_homogen.xy / world_pos_homogen.w;
    vec4 final_pos = vec4(world_pos * uDevicePixelRatio, pi.z, 1.0);

    gl_Position = uTransform * final_pos;

    vec2 uv_pos = src_task.render_target_origin + normalized_pos * src_task.size;
    vec2 texture_size = vec2(textureSize(sCacheRGBA8, 0));
    vUv = vec3(uv_pos / texture_size, src_task.render_target_layer_index);
}
