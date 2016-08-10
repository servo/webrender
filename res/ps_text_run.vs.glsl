#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

struct TextRun {
    PrimitiveInfo info;
    int first_glyph_index;
};

struct Glyph {
    vec4 local_rect;
    vec4 color;
    vec4 st_rect;
};

layout(std140) uniform Items {
    TextRun texts[WR_MAX_PRIM_ITEMS];
};

layout(std140) uniform Glyphs {
    Glyph glyphs[WR_MAX_GLYPHS];
};

void main(void) {
    TextRun text_run = texts[gl_InstanceID / 8];
    int glyphIndex = text_run.first_glyph_index + gl_InstanceID % 8;
    Glyph glyph = glyphs[glyphIndex];
    text_run.info.local_rect = glyph.local_rect;
    VertexInfo vi = write_vertex(text_run.info);

    vec2 f = (vi.local_clamped_pos - vi.local_rect.p0) / (vi.local_rect.p1 - vi.local_rect.p0);

    vColor = glyph.color;
    vUv = mix(glyph.st_rect.xy,
              glyph.st_rect.zw,
              f);
}
