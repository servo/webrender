#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

struct Text {
    PrimitiveInfo info;
    ivec2 glyph_range;
};

struct Glyph {
    vec4 local_rect;
    vec4 color;
    vec4 st_rect;
};

layout(std140) uniform Items {
    Text texts[WR_MAX_PRIM_ITEMS];
};

layout(std140) uniform Glyphs {
    Glyph glyphs[WR_MAX_GLYPHS];
};

void main(void) {
    Text text = texts[gl_InstanceID / 32];
    int glyphIndex = text.glyph_range.x + gl_InstanceID % 32;
    if (glyphIndex < text.glyph_range.y) {
        Glyph glyph = glyphs[glyphIndex];
        text.info.local_rect = glyph.local_rect;
        VertexInfo vi = write_vertex(text.info);

        vec2 f = (vi.local_clamped_pos - vi.local_rect.p0) / (vi.local_rect.p1 - vi.local_rect.p0);

        vColor = glyph.color;
        vUv = mix(glyph.st_rect.xy,
                  glyph.st_rect.zw,
                  f);
    } else {
        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
        vColor = vec4(0.0);
        vUv = vec2(0.0);
    }
}
