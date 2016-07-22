#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

struct Border {
    PrimitiveInfo info;
    vec4 local_rect;
    vec4 color0;
    vec4 color1;
    vec4 radii;
    uvec4 border_style_trbl;
};

layout(std140) uniform Items {
    Border borders[WR_MAX_PRIM_ITEMS];
};

uint get_border_style(Border a_border, uint a_edge) {
  switch (a_edge) {
    case PST_TOP:
    case PST_TOP_LEFT:
      return a_border.border_style_trbl.x;
    case PST_BOTTOM_LEFT:
    case PST_LEFT:
      return a_border.border_style_trbl.z;
    case PST_BOTTOM_RIGHT:
    case PST_BOTTOM:
      return a_border.border_style_trbl.w;
    case PST_TOP_RIGHT:
    case PST_RIGHT:
      return a_border.border_style_trbl.y;
  }
}

void main(void) {
    Border border = borders[gl_InstanceID];
    Layer layer = layers[border.info.layer_tile_part.x];
    Tile tile = tiles[border.info.layer_tile_part.y];

    vec2 local_pos = mix(border.local_rect.xy,
                         border.local_rect.xy + border.local_rect.zw,
                         aPosition.xy);

    vec4 world_pos = layer.transform * vec4(local_pos, 0, 1);

    vec2 device_pos = world_pos.xy * uDevicePixelRatio;

    vec2 clamped_pos = clamp(device_pos,
                             tile.actual_rect.xy,
                             tile.actual_rect.xy + tile.actual_rect.zw);

    vec4 local_clamped_pos = layer.inv_transform * vec4(clamped_pos / uDevicePixelRatio, 0, 1);

    vec2 final_pos = clamped_pos + tile.target_rect.xy - tile.actual_rect.xy;

    gl_Position = uTransform * vec4(final_pos, 0, 1);

    // Just our boring radius position.
    vRadii = border.radii;

    float x0, y0, x1, y1;
    vBorderPart = border.info.layer_tile_part.z;
    switch (vBorderPart) {
        // These are the layer tile part PrimitivePart as uploaded by the tiling.rs
        case PST_TOP_LEFT:
            x0 = border.local_rect.x;
            y0 = border.local_rect.y;
            // These are width / heights
            x1 = border.local_rect.x + border.local_rect.z;
            y1 = border.local_rect.y + border.local_rect.w;

            // The radius here is the border-radius. This is 0, so vRefPoint will
            // just be the top left (x,y) corner.
            vRefPoint = vec2(x0, y0) + vRadii.xy;
            break;
        case PST_TOP_RIGHT:
            x0 = border.local_rect.x + border.local_rect.z;
            y0 = border.local_rect.y;
            x1 = border.local_rect.x;
            y1 = border.local_rect.y + border.local_rect.w;
            vRefPoint = vec2(x0, y0) + vec2(-vRadii.x, vRadii.y);
            break;
        case PST_BOTTOM_LEFT:
            x0 = border.local_rect.x;
            y0 = border.local_rect.y + border.local_rect.w;
            x1 = border.local_rect.x + border.local_rect.z;
            y1 = border.local_rect.y;
            vRefPoint = vec2(x0, y0) + vec2(vRadii.x, -vRadii.y);
            break;
        case PST_BOTTOM_RIGHT:
            x0 = border.local_rect.x;
            y0 = border.local_rect.y;
            x1 = border.local_rect.x + border.local_rect.z;
            y1 = border.local_rect.y + border.local_rect.w;
            vRefPoint = vec2(x1, y1) + vec2(-vRadii.x, -vRadii.y);
            break;
        case PST_TOP:
        case PST_LEFT:
        case PST_BOTTOM:
        case PST_RIGHT:
            vRefPoint = border.local_rect.xy;
            x0 = border.local_rect.x;
            y0 = border.local_rect.y;
            x1 = border.local_rect.x + border.local_rect.z;
            y1 = border.local_rect.y + border.local_rect.w;
            break;
    }

    vBorderStyle = get_border_style(border, vBorderPart);

    // y1 - y0 is the height of the corner / line
    // x1 - x0 is the width of the corner / line.
    float width = x1 - x0;
    float height = y1 - y0;
    // This is just a weighting of the pixel colors it seems?
    vF = (local_clamped_pos.x - x0) * height - (local_clamped_pos.y - y0) * width;

    // This is what was currently sent.
    vColor0 = border.color0;
    vColor1 = border.color1;

    // These are in device space
    vPos = clamped_pos;
    vBorders = vec4(border.local_rect.x, border.local_rect.y,
                    border.local_rect.z,
                    border.local_rect.w) * uDevicePixelRatio;
}
