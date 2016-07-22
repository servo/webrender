#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

struct Image {
    PrimitiveInfo info;
    vec4 local_rect;    // Size of the box we need to fill with this image.
    vec4 st_rect;       // Location of the image texture in the texture atlas.
    vec2 stretch_size;  // Size of the actual image.
    Clip clip;
};

layout(std140) uniform Items {
    Image images[WR_MAX_PRIM_ITEMS];
};

void main(void) {
    Image image = images[gl_InstanceID];
    Layer layer = layers[image.info.layer_tile_part.x];
    Tile tile = tiles[image.info.layer_tile_part.y];

    // Our location within the image
    vec2 local_pos = mix(image.local_rect.xy,
                         image.local_rect.xy + image.local_rect.zw,
                         aPosition.xy);

    local_pos = clamp(local_pos,
                      image.info.local_clip_rect.xy,
                      image.info.local_clip_rect.xy + image.info.local_clip_rect.zw);

    vClipRect = vec4(image.clip.rect.xy, image.clip.rect.xy + image.clip.rect.zw);
    vClipRadius = image.clip.top_left.outer_inner_radius.x;

    vec4 world_pos = layer.transform * vec4(local_pos, 0, 1);

    vec2 device_pos = world_pos.xy * uDevicePixelRatio;

    vec2 clamped_pos = clamp(device_pos,
                             tile.actual_rect.xy,
                             tile.actual_rect.xy + tile.actual_rect.zw);

    vec4 local_clamped_pos = layer.inv_transform * vec4(clamped_pos / uDevicePixelRatio, 0, 1);
    vPos = local_clamped_pos.xy;

    // vUv will contain how many times this image has wrapped around the image size.
    vUv = (local_clamped_pos.xy - image.local_rect.xy) / image.stretch_size.xy;
    vTextureSize = image.st_rect.zw - image.st_rect.xy;
    vTextureOffset = image.st_rect.xy;

    vec2 final_pos = clamped_pos + tile.target_rect.xy - tile.actual_rect.xy;

    gl_Position = uTransform * vec4(final_pos, 0, 1);
}
