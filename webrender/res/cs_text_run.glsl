/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared

varying vec3 vUv;
flat varying vec4 vColor;

#ifdef WR_VERTEX_SHADER
// Draw a text run to a cache target. These are always
// drawn un-transformed. These are used for effects such
// as text-shadow.

void main(void) {
    Primitive prim = load_primitive();
    TextRun text = fetch_text_run(prim.specific_prim_address);

    int glyph_index = prim.user_data0;
    int resource_address = prim.user_data1;

    Glyph glyph = fetch_glyph(prim.specific_prim_address,
                              glyph_index,
                              text.subpx_dir);

    GlyphResource res = fetch_glyph_resource(resource_address);

    // Glyphs size is already in device-pixels.
    // The render task origin is in device-pixels. Offset that by
    // the glyph offset, relative to its primitive bounding rect.
    vec2 size = (res.uv_rect.zw - res.uv_rect.xy) * res.scale;
    vec2 local_pos = glyph.offset + vec2(res.offset.x, -res.offset.y) / uDevicePixelRatio;
    vec2 origin = prim.task.common_data.task_rect.p0 +
                  uDevicePixelRatio * (local_pos - prim.task.content_origin);
    vec4 local_rect = vec4(origin, size);

    vec2 texture_size = vec2(textureSize(sColor0, 0));
    vec2 st0 = res.uv_rect.xy / texture_size;
    vec2 st1 = res.uv_rect.zw / texture_size;

    vec2 pos = mix(local_rect.xy,
                   local_rect.xy + local_rect.zw,
                   aPosition.xy);

    vUv = vec3(mix(st0, st1, aPosition.xy), res.layer);
    vColor = prim.task.color;

    gl_Position = uTransform * vec4(pos, 0.0, 1.0);
}
#endif

#ifdef WR_FRAGMENT_SHADER
void main(void) {
    float a = texture(sColor0, vUv).a;
    oFragColor = vColor * a;
}
#endif
