#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define DIR_HORIZONTAL      uint(0)
#define DIR_VERTICAL        uint(1)

struct Gradient {
	PrimitiveInfo info;
    vec4 local_rect;
    vec4 color0;
    vec4 color1;
    uvec4 dir;
};

layout(std140) uniform Items {
    Gradient gradients[WR_MAX_PRIM_ITEMS];
};

void main(void) {
    Gradient gradient = gradients[gl_InstanceID];
    Layer layer = layers[gradient.info.layer_tile_part.x];
    Tile tile = tiles[gradient.info.layer_tile_part.y];

    vec2 local_pos = mix(gradient.local_rect.xy,
                         gradient.local_rect.xy + gradient.local_rect.zw,
                         aPosition.xy);

    vec4 world_pos = layer.transform * vec4(local_pos, 0, 1);

    vec2 device_pos = world_pos.xy * uDevicePixelRatio;

    vec2 clamped_pos = clamp(device_pos,
                             tile.actual_rect.xy,
                             tile.actual_rect.xy + tile.actual_rect.zw);

    vec4 local_clamped_pos = layer.inv_transform * vec4(clamped_pos / uDevicePixelRatio, 0, 1);

    vec2 f = (local_clamped_pos.xy - gradient.local_rect.xy) / gradient.local_rect.zw;

    vec2 final_pos = clamped_pos + tile.target_rect.xy - tile.actual_rect.xy;

    gl_Position = uTransform * vec4(final_pos, 0, 1);

    switch (gradient.dir.x) {
        case DIR_HORIZONTAL:
            vF = f.x;
            break;
        case DIR_VERTICAL:
            vF = f.y;
            break;
    }

    vColor0 = gradient.color0;
    vColor1 = gradient.color1;
}
