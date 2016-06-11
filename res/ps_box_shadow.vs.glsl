#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

struct BoxShadow {
	PrimitiveInfo info;
	vec4 local_rect;
	vec4 color;
    vec4 border_radii_blur_radius_inverted;
    vec4 bs_rect;
    vec4 src_rect;
};

layout(std140) uniform Items {
    BoxShadow boxshadows[WR_MAX_PRIM_ITEMS];
};

void main(void) {
    BoxShadow bs = boxshadows[gl_InstanceID];
    Layer layer = layers[bs.info.layer_tile_part.x];
    Tile tile = tiles[bs.info.layer_tile_part.y];

    vColor = bs.color;

    vec2 local_pos = mix(bs.local_rect.xy,
                         bs.local_rect.xy + bs.local_rect.zw,
                         aPosition.xy);

    vec4 world_pos = layer.transform * vec4(local_pos, 0, 1);

    vec2 device_pos = world_pos.xy * uDevicePixelRatio;

    vec2 clamped_pos = clamp(device_pos,
                             tile.actual_rect.xy,
                             tile.actual_rect.xy + tile.actual_rect.zw);

    vec2 final_pos = clamped_pos + tile.target_rect.xy - tile.actual_rect.xy;

    vec4 local_clamped_pos = layer.inv_transform * vec4(clamped_pos / uDevicePixelRatio, 0, 1);

    vPos = local_clamped_pos.xy;
    vColor = bs.color;
    vBorderRadii = bs.border_radii_blur_radius_inverted.xy;
    vBlurRadius = bs.border_radii_blur_radius_inverted.z;
    vBoxShadowRect = vec4(bs.bs_rect.xy, bs.bs_rect.xy + bs.bs_rect.zw);
    vSrcRect = vec4(bs.src_rect.xy, bs.src_rect.xy + bs.src_rect.zw);
    vInverted = bs.border_radii_blur_radius_inverted.w;

    gl_Position = uTransform * vec4(final_pos, 0, 1);
}
