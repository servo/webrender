#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

struct AngleGradient {
	PrimitiveInfo info;
    vec4 start_end_point;
    uvec4 stop_count;
    vec4 colors[MAX_STOPS_PER_ANGLE_GRADIENT];
    vec4 offsets[MAX_STOPS_PER_ANGLE_GRADIENT/4];
};

layout(std140) uniform Items {
    AngleGradient gradients[WR_MAX_PRIM_ITEMS];
};

void main(void) {
    AngleGradient gradient = gradients[gl_InstanceID];
    Layer layer = layers[gradient.info.layer_tile_part.x];
    Tile tile = tiles[gradient.info.layer_tile_part.y];

    vec2 local_pos = mix(gradient.info.local_rect.xy,
                         gradient.info.local_rect.xy + gradient.info.local_rect.zw,
                         aPosition.xy);

    vec4 world_pos = layer.transform * vec4(local_pos, 0, 1);

    vec2 device_pos = world_pos.xy * uDevicePixelRatio;

    vec2 clamped_pos = clamp(device_pos,
                             tile.actual_rect.xy,
                             tile.actual_rect.xy + tile.actual_rect.zw);

    vec4 local_clamped_pos = layer.inv_transform * vec4(clamped_pos / uDevicePixelRatio, 0, 1);

    vec2 final_pos = clamped_pos + tile.target_rect.xy - tile.actual_rect.xy;

    vStopCount = int(gradient.stop_count.x);
    vPos = local_clamped_pos.xy;
    vStartPoint = gradient.start_end_point.xy;
    vEndPoint = gradient.start_end_point.zw;

    for (int i=0 ; i < int(gradient.stop_count.x) ; ++i) {
        vColors[i] = gradient.colors[i];
        vOffsets[i] = gradient.offsets[i];
    }

    gl_Position = uTransform * vec4(final_pos, 0, 1);
}
