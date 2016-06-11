#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

struct Rectangle {
	PrimitiveInfo info;
    vec4 local_rect;
	vec4 color;
};

layout(std140) uniform Items {
    Rectangle rects[WR_MAX_PRIM_ITEMS];
};

void main(void) {
    Rectangle rect = rects[gl_InstanceID];
    Layer layer = layers[rect.info.layer_tile_part.x];
    Tile tile = tiles[rect.info.layer_tile_part.y];

    vColor = rect.color;

    vec2 local_pos = mix(rect.local_rect.xy,
                         rect.local_rect.xy + rect.local_rect.zw,
                         aPosition.xy);

    local_pos = clamp(local_pos,
                      rect.info.local_clip_rect.xy,
                      rect.info.local_clip_rect.xy + rect.info.local_clip_rect.zw);

    vec4 world_pos = layer.transform * vec4(local_pos, 0, 1);

    vec2 device_pos = world_pos.xy * uDevicePixelRatio;

    vec2 clamped_pos = clamp(device_pos,
                             tile.actual_rect.xy,
                             tile.actual_rect.xy + tile.actual_rect.zw);

    vec2 final_pos = clamped_pos + tile.target_rect.xy - tile.actual_rect.xy;

    gl_Position = uTransform * vec4(final_pos, 0, 1);
}
