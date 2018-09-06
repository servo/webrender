/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared

varying vec3 vUv;
flat varying vec4 vUvSampleBounds;

#ifdef WR_VERTEX_SHADER
struct SplitGeometry {
    vec2 local[4];
    RectWithSize local_rect;
};

SplitGeometry fetch_split_geometry(int address) {
    ivec2 uv = get_resource_cache_uv(address);

    vec4 data0 = TEXEL_FETCH(sResourceCache, uv, 0, ivec2(0, 0));
    vec4 data1 = TEXEL_FETCH(sResourceCache, uv, 0, ivec2(1, 0));
    vec4 data2 = TEXEL_FETCH(sResourceCache, uv, 0, ivec2(2, 0));

    SplitGeometry geo;
    geo.local = vec2[4](
        data0.xy,
        data0.zw,
        data1.xy,
        data1.zw
    );
    geo.local_rect = RectWithSize(data2.xy, data2.zw);

    return geo;
}

vec2 bilerp(vec2 a, vec2 b, vec2 c, vec2 d, float s, float t) {
    vec2 x = mix(a, b, t);
    vec2 y = mix(c, d, t);
    return mix(x, y, s);
}

struct SplitCompositeInstance {
    int render_task_index;
    int polygons_address;
    float z;
    int uv_address;
    int transform_id;
};

SplitCompositeInstance fetch_composite_instance() {
    SplitCompositeInstance ci;

    ci.render_task_index = aData.x & 0xffff;
    ci.z = float(aData.x >> 16);
    ci.polygons_address = aData.y;
    ci.uv_address = aData.z;
    ci.transform_id = aData.w;

    return ci;
}

void main(void) {
    SplitCompositeInstance ci = fetch_composite_instance();
    SplitGeometry geometry = fetch_split_geometry(ci.polygons_address);
    PictureTask dest_task = fetch_picture_task(ci.render_task_index);
    Transform transform = fetch_transform(ci.transform_id);
    ImageResource res = fetch_image_resource(ci.uv_address);
    ImageResourceExtra extra_data = fetch_image_resource_extra(ci.uv_address);

    vec2 dest_origin = dest_task.common_data.task_rect.p0 -
                       dest_task.content_origin;

    vec2 local_pos = bilerp(geometry.local[0], geometry.local[1],
                            geometry.local[3], geometry.local[2],
                            aPosition.y, aPosition.x);
    vec4 world_pos = transform.m * vec4(local_pos, 0.0, 1.0);

    vec4 final_pos = vec4(
        dest_origin + world_pos.xy * uDevicePixelRatio,
        world_pos.w * ci.z,
        world_pos.w
    );

    gl_Position = uTransform * final_pos;

    vec2 texture_size = vec2(textureSize(sCacheRGBA8, 0));
    vec2 uv0 = res.uv_rect.p0;
    vec2 uv1 = res.uv_rect.p1;

    vec2 min_uv = min(uv0, uv1);
    vec2 max_uv = max(uv0, uv1);

    vUvSampleBounds = vec4(
        min_uv + vec2(0.5),
        max_uv - vec2(0.5)
    ) / texture_size.xyxy;

    vec2 f = (local_pos - geometry.local_rect.p0) / geometry.local_rect.size;

    vec2 x = mix(extra_data.st_tl, extra_data.st_tr, f.x);
    vec2 y = mix(extra_data.st_bl, extra_data.st_br, f.x);
    f = mix(x, y, f.y);
    vec2 uv = mix(uv0, uv1, f);

    vUv = vec3(uv / texture_size, res.layer);
}
#endif

#ifdef WR_FRAGMENT_SHADER
void main(void) {
    vec2 uv = clamp(vUv.xy, vUvSampleBounds.xy, vUvSampleBounds.zw);
    oFragColor = textureLod(sCacheRGBA8, vec3(uv, vUv.z), 0.0);
}
#endif
