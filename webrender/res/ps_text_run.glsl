/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared

flat varying vec4 vColor;
varying vec3 vUv;
flat varying vec4 vUvBorder;

#ifdef WR_VERTEX_SHADER

#define MODE_ALPHA              0
#define MODE_SUBPX_CONST_COLOR  1
#define MODE_SUBPX_PASS0        2
#define MODE_SUBPX_PASS1        3
#define MODE_SUBPX_BG_PASS0     4
#define MODE_SUBPX_BG_PASS1     5
#define MODE_SUBPX_BG_PASS2     6
#define MODE_COLOR_BITMAP       7

VertexInfo write_text_vertex(vec2 local_pos,
                             RectWithSize local_clip_rect,
                             float z,
                             Layer layer,
                             AlphaBatchTask task) {
    // Clamp to the two local clip rects.
    vec2 clamped_local_pos = clamp_rect(clamp_rect(local_pos, local_clip_rect), layer.local_clip_rect);

    // Transform the current vertex to the world cpace.
    vec4 world_pos = layer.transform * vec4(clamped_local_pos, 0.0, 1.0);

    // Convert the world positions to device pixel space.
    vec2 device_pos = world_pos.xy / world_pos.w * uDevicePixelRatio;

    // Apply offsets for the render task to get correct screen location.
    vec2 final_pos = device_pos -
                     task.screen_space_origin +
                     task.render_target_origin;

    gl_Position = uTransform * vec4(final_pos, z, 1.0);

    VertexInfo vi = VertexInfo(clamped_local_pos, device_pos);
    return vi;
}

void main(void) {
    Primitive prim = load_primitive();
    TextRun text = fetch_text_run(prim.specific_prim_address);

    int glyph_index = prim.user_data0;
    int resource_address = prim.user_data1;

    Glyph glyph = fetch_glyph(prim.specific_prim_address,
                              glyph_index,
                              text.subpx_dir);
    GlyphResource res = fetch_glyph_resource(resource_address);

#ifdef WR_FEATURE_GLYPH_TRANSFORM
    mat2 transform = mat2(prim.layer.transform) * (uDevicePixelRatio / res.scale);
    mat2 inv_transform = inverse(transform);
#else
    float transform = uDevicePixelRatio / res.scale;
    float inv_transform = res.scale / uDevicePixelRatio;
#endif

    vec2 glyph_pos = res.offset + transform * (text.offset + glyph.offset);
    vec2 glyph_size = res.uv_rect.zw - res.uv_rect.xy;

    // Select the corner of the glyph rect that we are processing.
    // Transform it into local space for vertex processing.
    vec2 local_pos = inv_transform * (glyph_pos + glyph_size * aPosition.xy);

    VertexInfo vi = write_text_vertex(local_pos,
                                      prim.local_clip_rect,
                                      prim.z,
                                      prim.layer,
                                      prim.task);

    vec2 f = (transform * vi.local_pos - glyph_pos) / glyph_size;

    write_clip(vi.screen_pos, prim.clip_area);

#ifdef WR_FEATURE_SUBPX_BG_PASS1
    vColor = vec4(text.color.a) * text.bg_color;
#else
    switch (uMode) {
        case MODE_ALPHA:
        case MODE_SUBPX_PASS1:
        case MODE_SUBPX_BG_PASS2:
            vColor = text.color;
            break;
        case MODE_SUBPX_CONST_COLOR:
        case MODE_SUBPX_PASS0:
        case MODE_SUBPX_BG_PASS0:
        case MODE_COLOR_BITMAP:
            vColor = vec4(text.color.a);
            break;
        case MODE_SUBPX_BG_PASS1:
            // This should never be reached.
            break;
    }
#endif

    vec2 texture_size = vec2(textureSize(sColor0, 0));
    vec2 st0 = res.uv_rect.xy / texture_size;
    vec2 st1 = res.uv_rect.zw / texture_size;

    vUv = vec3(mix(st0, st1, f), res.layer);
    vUvBorder = (res.uv_rect + vec4(0.5, 0.5, -0.5, -0.5)) / texture_size.xyxy;
}
#endif

#ifdef WR_FRAGMENT_SHADER
void main(void) {
    vec3 tc = vec3(clamp(vUv.xy, vUvBorder.xy, vUvBorder.zw), vUv.z);
    vec4 mask = texture(sColor0, tc);

    float alpha = do_clip();

#ifdef WR_FEATURE_SUBPX_BG_PASS1
    mask.rgb = vec3(mask.a) - mask.rgb;
#endif

    oFragColor = vColor * mask * alpha;
}
#endif
