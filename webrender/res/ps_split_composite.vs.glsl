#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

uniform sampler2D sSplitGeometry;

struct SplitGeometry {
    vec4 points[4];
};

SplitGeometry fetch_split_geometry(int index) {
    ivec2 uv = get_fetch_uv(index, VECS_PER_SPLIT_GEOM);
    return SplitGeometry(vec4[4](
        texelFetchOffset(sSplitGeometry, uv, 0, ivec2(0, 0)),
        texelFetchOffset(sSplitGeometry, uv, 0, ivec2(1, 0)),
        texelFetchOffset(sSplitGeometry, uv, 0, ivec2(2, 0)),
        texelFetchOffset(sSplitGeometry, uv, 0, ivec2(3, 0))
    ));
}

void main(void) {
    PrimitiveInstance pi = fetch_prim_instance();

    SplitGeometry geometry = fetch_split_geometry(pi.specific_prim_index);

    gl_Position = mix(
        mix(geometry.points[0], geometry.points[1], aPosition.x),
        mix(geometry.points[2], geometry.points[3], aPosition.x),
        aPosition.y);

    AlphaBatchTask src_task = fetch_alpha_batch_task(pi.user_data.x);

    vec2 texture_size = vec2(textureSize(sCacheRGBA8, 0));
    vec2 st0 = src_task.render_target_origin / texture_size;
    vec2 st1 = (src_task.render_target_origin + src_task.size) / texture_size;
    vUv = vec3(mix(st0, st1, aPosition.xy), src_task.render_target_layer_index);
}
