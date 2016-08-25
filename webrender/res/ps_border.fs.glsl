#line 1

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

void discard_pixels_in_rounded_borders(vec2 local_pos) {
  float distanceFromRef = distance(vRefPoint, local_pos);
  if (vRadii.x > 0.0 && (distanceFromRef > vRadii.x || distanceFromRef < vRadii.z)) {
      discard;
  }
}

vec4 get_fragment_color(float distanceFromMixLine, float pixelsPerFragment) {
  // Here we are mixing between the two border colors. We need to convert
  // distanceFromMixLine it to pixel space to properly anti-alias and them push
  // it between the limits accepted by `mix`.
  float colorMix = min(max(distanceFromMixLine / pixelsPerFragment, -0.5), 0.5) + 0.5;
  return mix(vHorizontalColor, vVerticalColor, colorMix);
}

float alpha_for_solid_border(float distance_from_ref,
                             float inner_radius,
                             float outer_radius,
                             float pixels_per_fragment) {
  // We want to start anti-aliasing one pixel in from the border.
  float nudge = 1 * pixels_per_fragment;
  inner_radius += nudge;
  outer_radius -= nudge;

  if ((distance_from_ref < outer_radius && distance_from_ref > inner_radius)) {
    return 1.0;
  }

  float distance_from_border = max(distance_from_ref - outer_radius,
                                   inner_radius - distance_from_ref);

  // Move the distance back into pixels.
  distance_from_border /= pixels_per_fragment;

  // Apply a more gradual fade out to transparent.
  distance_from_border -= 0.5;

  return smoothstep(1.0, 0, distance_from_border);
}

float alpha_for_solid_border_corner(vec2 local_pos,
                                    float inner_radius,
                                    float outer_radius,
                                    float pixels_per_fragment) {
  float distance_from_ref = distance(vRefPoint, local_pos);
  return alpha_for_solid_border(distance_from_ref, inner_radius, outer_radius, pixels_per_fragment);
}

#ifdef WR_FEATURE_TRANSFORM

#else
// draw a circle at position aDesiredPos with a aRadius
vec4 drawCircle(vec2 aPixel, vec2 aDesiredPos, float aRadius, vec3 aColor) {
  float farFromCenter = length(aDesiredPos - aPixel) - aRadius;
  float pixelInCircle = 1.00 - clamp(farFromCenter, 0.0, 1.0);
  return vec4(aColor, pixelInCircle);
}

// Draw a rectangle at aRect fill it with aColor. Only works on non-rotated
// rects.
vec4 drawRect(vec2 aPixel, vec4 aRect, vec3 aColor) {
   // GLSL origin is bottom left, positive Y is up
   bool inRect = (aRect.x <= aPixel.x) && (aPixel.x <= aRect.x + aRect.z) &&
            (aPixel.y >= aRect.y) && (aPixel.y <= aRect.y + aRect.w);
   return vec4(aColor, float(inRect));
}

vec4 draw_dotted_edge() {
  // Everything here should be in device pixels.
  // We want the dot to be roughly the size of the whole border spacing
  float border_spacing = min(vBorders.w, vBorders.z);
  float radius = floor(border_spacing / 2.0);
  float diameter = radius * 2.0;
  // The amount of space between dots. 2.2 was chosen because it looks kind of
  // like firefox.
  float circleSpacing = diameter * 2.2;

  vec2 size = vBorders.zw;
  // Get our position within this specific segment
  vec2 position = vDevicePos - vBorders.xy;

  // Break our position into square tiles with circles in them.
  vec2 circleCount = floor(size / circleSpacing);
  circleCount = max(circleCount, 1.0);

  vec2 distBetweenCircles = size / circleCount;
  vec2 circleCenter = distBetweenCircles / 2.0;

  // Find out which tile this pixel belongs to.
  vec2 destTile = floor(position / distBetweenCircles);
  destTile = destTile * distBetweenCircles;

  // Where we want to draw the actual circle.
  vec2 tileCenter = destTile + circleCenter;

  // Find the position within the tile
  vec2 positionInTile = mod(position, distBetweenCircles);
  vec2 finalPosition = positionInTile + destTile;

  vec4 white = vec4(1.0, 1.0, 1.0, 1.0);
  // See if we should draw a circle or not
  vec4 circleColor = drawCircle(finalPosition, tileCenter, radius, vVerticalColor.xyz);
  return mix(white, circleColor, circleColor.a);
}

vec4 draw_double_edge(float pos, float len, float pixelsPerFragment) {
  float total_border_width = len;
  float one_third_width = total_border_width / 3.0;

  // Contribution of the outer border segment.
  float alpha = alpha_for_solid_border(pos,
                                       total_border_width - one_third_width,
                                       total_border_width,
                                       pixelsPerFragment);

  // Contribution of the inner border segment.
  alpha += alpha_for_solid_border(pos, 0, one_third_width, pixelsPerFragment);
  return get_fragment_color(vDistanceFromMixLine, pixelsPerFragment) * vec4(1, 1, 1, alpha);
}

vec4 draw_double_edge_vertical(float pixelsPerFragment) {
  // Get our position within this specific segment
  float position = vLocalPos.x - vLocalRect.x;
  return draw_double_edge(position, vLocalRect.z, pixelsPerFragment);
}

vec4 draw_double_edge_horizontal(float pixelsPerFragment) {
  // Get our position within this specific segment
  float position = vLocalPos.y - vLocalRect.y;
  return draw_double_edge(position, vLocalRect.w, pixelsPerFragment);
}

vec4 draw_double_edge_corner_with_radius(float pixelsPerFragment) {
  float total_border_width = vRadii.x - vRadii.z;
  float one_third_width = total_border_width / 3.0;

  // Contribution of the outer border segment.
  float alpha = alpha_for_solid_border_corner(vLocalPos,
                                              vRadii.x - one_third_width,
                                              vRadii.x,
                                              pixelsPerFragment);

  // Contribution of the inner border segment.
  alpha += alpha_for_solid_border_corner(vLocalPos,
                                         vRadii.z,
                                         vRadii.z + one_third_width,
                                         pixelsPerFragment);
  return get_fragment_color(vDistanceFromMixLine, pixelsPerFragment) * vec4(1, 1, 1, alpha);
}

vec4 draw_double_edge_corner(float pixelsPerFragment) {
  if (vRadii.x > 0) {
    return draw_double_edge_corner_with_radius(pixelsPerFragment);
  }

  bool is_vertical = (vBorderPart == PST_TOP_LEFT) ? vDistanceFromMixLine < 0 :
                                                     vDistanceFromMixLine >= 0;
  if (is_vertical) {
    return draw_double_edge_vertical(pixelsPerFragment);
  } else {
    return draw_double_edge_horizontal(pixelsPerFragment);
  }
}

// Our current edge calculation is based only on
// the size of the border-size, but we need to draw
// the dashes in the center of the segment we're drawing.
// This calculates how much to nudge and which axis to nudge on.
vec2 get_dashed_nudge_factor(vec2 dash_size, bool is_corner) {
  if (is_corner) {
    return vec2(0.0, 0.0);
  }

  bool xAxisFudge = vBorders.z > vBorders.w;
  if (xAxisFudge) {
    return vec2(dash_size.x / 2.0, 0);
  }

  return vec2(0.0, dash_size.y / 2.0);
}

vec4 draw_dashed_edge(bool is_corner) {
  // Everything here should be in device pixels.
  // We want the dot to be roughly the size of the whole border spacing
  // 5.5 here isn't a magic number, it's just what mostly looks like FF/Chrome
  // TODO: Investigate exactly what FF does.
  float dash_interval = min(vBorders.w, vBorders.z) * 5.5;
  vec2 edge_size = vec2(vBorders.z, vBorders.w);
  vec2 dash_size = vec2(dash_interval / 2.0, dash_interval / 2.0);
  vec2 position = vDevicePos - vBorders.xy;

  vec2 dash_count = floor(edge_size/ dash_interval);
  vec2 dist_between_dashes = edge_size / dash_count;

  vec2 target_rect_index = floor(position / dist_between_dashes);
  vec2 target_rect_loc = target_rect_index * dist_between_dashes;
  target_rect_loc += get_dashed_nudge_factor(dash_size, is_corner);
  vec4 target_rect = vec4(target_rect_loc, dash_size);

  vec4 white = vec4(1.0, 1.0, 1.0, 1.0);
  vec4 target_colored_rect = drawRect(position, target_rect, vVerticalColor.xyz);
  return mix(white, target_colored_rect, target_colored_rect.a);
}

void draw_dotted_border(void) {
  switch (vBorderPart) {
    // These are the layer tile part PrimitivePart as uploaded by the tiling.rs
    case PST_TOP_LEFT:
    case PST_TOP_RIGHT:
    case PST_BOTTOM_LEFT:
    case PST_BOTTOM_RIGHT:
    {
      // TODO: Fix for corners with a border-radius
      oFragColor = draw_dotted_edge();
      break;
    }
    case PST_BOTTOM:
    case PST_TOP:
    case PST_LEFT:
    case PST_RIGHT:
    {
      oFragColor = draw_dotted_edge();
      break;
    }
  }
}

void draw_dashed_border(void) {
  switch (vBorderPart) {
    // These are the layer tile part PrimitivePart as uploaded by the tiling.rs
    case PST_TOP_LEFT:
    case PST_TOP_RIGHT:
    case PST_BOTTOM_LEFT:
    case PST_BOTTOM_RIGHT:
    {
      // TODO: Fix for corners with a border-radius
      bool is_corner = true;
      oFragColor = draw_dashed_edge(is_corner);
      break;
    }
    case PST_BOTTOM:
    case PST_TOP:
    case PST_LEFT:
    case PST_RIGHT:
    {
      bool is_corner = false;
      oFragColor = draw_dashed_edge(is_corner);
      break;
    }
  }
}

void draw_double_border(vec2 localPos) {
  float pixelsPerFragment = length(fwidth(localPos.xy));
  switch (vBorderPart) {
    // These are the layer tile part PrimitivePart as uploaded by the tiling.rs
    case PST_TOP_LEFT:
    case PST_TOP_RIGHT:
    case PST_BOTTOM_LEFT:
    case PST_BOTTOM_RIGHT:
    {
      oFragColor = draw_double_edge_corner(pixelsPerFragment);
      break;
    }
    case PST_BOTTOM:
    case PST_TOP:
    {
      oFragColor = draw_double_edge_horizontal(pixelsPerFragment);
      break;
    }
    case PST_LEFT:
    case PST_RIGHT:
    {
      oFragColor = draw_double_edge_vertical(pixelsPerFragment);
      break;
    }
  }
}

#endif

void draw_solid_border(float distanceFromMixLine, vec2 localPos) {
  switch (vBorderPart) {
    case PST_TOP_LEFT:
    case PST_TOP_RIGHT:
    case PST_BOTTOM_LEFT:
    case PST_BOTTOM_RIGHT: {
      // This is the conversion factor for transformations and device pixel scaling.
      float pixelsPerFragment = length(fwidth(localPos.xy));
      oFragColor = get_fragment_color(distanceFromMixLine, pixelsPerFragment);

      if (vRadii.x > 0.0) {
        float alpha = alpha_for_solid_border_corner(localPos, vRadii.z, vRadii.x, pixelsPerFragment);
        oFragColor *= vec4(1, 1, 1, alpha);
      }

      break;
    }
    default:
      oFragColor = vHorizontalColor;
      discard_pixels_in_rounded_borders(localPos);
  }
}


// TODO: Investigate performance of this shader and see
//       if it's worthwhile splitting it / removing branches etc.
void main(void) {
#ifdef WR_FEATURE_TRANSFORM
    float alpha = 0;
    vec2 local_pos = init_transform_fs(vLocalPos, vLocalRect, alpha);
#else
    vec2 local_pos = vLocalPos;
#endif

#ifdef WR_FEATURE_TRANSFORM
    // TODO(gw): Support other border styles for transformed elements.
    float distance_from_mix_line = (local_pos.x - vPieceRect.x) * vPieceRect.w -
                                   (local_pos.y - vPieceRect.y) * vPieceRect.z;
    distance_from_mix_line /= vPieceRectHypotenuseLength;
    draw_solid_border(distance_from_mix_line, local_pos);
    oFragColor *= vec4(1, 1, 1, alpha);

#else
    switch (vBorderStyle) {
        case BORDER_STYLE_DASHED:
            discard_pixels_in_rounded_borders(local_pos);
            draw_dashed_border();
            break;
        case BORDER_STYLE_DOTTED:
            discard_pixels_in_rounded_borders(local_pos);
            draw_dotted_border();
            break;
        case BORDER_STYLE_OUTSET:
        case BORDER_STYLE_INSET:
        case BORDER_STYLE_SOLID:
        case BORDER_STYLE_NONE:
            draw_solid_border(vDistanceFromMixLine, local_pos);
            break;
        case BORDER_STYLE_DOUBLE:
            draw_double_border(local_pos);
            break;
        default:
            discard;
    }
#endif
}
