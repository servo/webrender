/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared

flat varying vec4 vColor;
varying vec3 vUv;
flat varying vec4 vUvBorder;

#ifdef WR_FEATURE_TRANSFORM
varying vec3 vLocalPos;
#endif

#ifdef WR_VERTEX_SHADER

#define MODE_ALPHA          0
#define MODE_SUBPX_OPAQUE   1
#define MODE_SUBPX_PASS0    2
#define MODE_SUBPX_PASS1    3
#define MODE_SUBPX_BG_PASS0 4
#define MODE_SUBPX_BG_PASS1 5
#define MODE_SUBPX_BG_PASS2 6
#define MODE_COLOR_BITMAP   7

void main(void) {
    Primitive prim = load_primitive();
    TextRun text = fetch_text_run(prim.specific_prim_address);

    int glyph_index = prim.user_data0;
    int resource_address = prim.user_data1;

    Glyph glyph = fetch_glyph(prim.specific_prim_address,
                              glyph_index,
                              text.subpx_dir);
    GlyphResource res = fetch_glyph_resource(resource_address);

    vec2 local_pos = glyph.offset +
                     text.offset +
                     vec2(res.offset.x, -res.offset.y) / uDevicePixelRatio;

    RectWithSize local_rect = RectWithSize(local_pos,
                                           (res.uv_rect.zw - res.uv_rect.xy) * res.scale / uDevicePixelRatio);

#ifdef WR_FEATURE_TRANSFORM
    TransformVertexInfo vi = write_transform_vertex(local_rect,
                                                    prim.local_clip_rect,
                                                    prim.z,
                                                    prim.layer,
                                                    prim.task,
                                                    local_rect);
    vLocalPos = vi.local_pos;
    vec2 f = (vi.local_pos.xy / vi.local_pos.z - local_rect.p0) / local_rect.size;
#else
    VertexInfo vi = write_vertex(local_rect,
                                 prim.local_clip_rect,
                                 prim.z,
                                 prim.layer,
                                 prim.task,
                                 local_rect);
    vec2 f = (vi.local_pos - local_rect.p0) / local_rect.size;
#endif

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
        case MODE_SUBPX_PASS0:
        case MODE_SUBPX_BG_PASS0:
        case MODE_COLOR_BITMAP:
            vColor = vec4(text.color.a);
            break;
        case MODE_SUBPX_OPAQUE:
            // The text foreground color is handled by the constant
            // color blend mode.
            vColor = vec4(1.0);
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

    float alpha = 1.0;
#ifdef WR_FEATURE_TRANSFORM
    init_transform_fs(vLocalPos, alpha);
#endif
    alpha *= do_clip();

#ifdef WR_FEATURE_SUBPX_BG_PASS1
    mask.rgb = vec3(mask.a) - mask.rgb;
#endif

    oFragColor = vColor * mask * alpha;
}
#endif
