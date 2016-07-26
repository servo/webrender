#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

struct Glyph {
    PrimitiveInfo info;
    vec4 color;
    vec4 st_rect;
};

layout(std140) uniform Items {
    Glyph glyphs[WR_MAX_PRIM_ITEMS];
};

void main(void) {
    Glyph glyph = glyphs[gl_InstanceID];
    Layer layer = layers[glyph.info.layer_tile_part.x];
    Tile tile = tiles[glyph.info.layer_tile_part.y];

    vColor = glyph.color;

    vec2 p0 = floor(0.5 + glyph.info.local_rect.xy * uDevicePixelRatio) / uDevicePixelRatio;
    vec2 p1 = p0 + glyph.info.local_rect.zw;

    vec2 local_pos = mix(p0, p1, aPosition.xy);

    vec4 world_pos = layer.transform * vec4(local_pos, 0, 1);

    vec2 device_pos = world_pos.xy * uDevicePixelRatio;

    vec2 clamped_pos = clamp(device_pos,
                             tile.actual_rect.xy,
                             tile.actual_rect.xy + tile.actual_rect.zw);

    vec4 local_clamped_pos = layer.inv_transform * vec4(clamped_pos / uDevicePixelRatio, 0, 1);

    vec2 f = (local_clamped_pos.xy - p0) / glyph.info.local_rect.zw;

    vUv = mix(glyph.st_rect.xy,
              glyph.st_rect.zw,
              f);

    vec2 final_pos = clamped_pos + tile.target_rect.xy - tile.actual_rect.xy;

    gl_Position = uTransform * vec4(final_pos, 0, 1);
}
