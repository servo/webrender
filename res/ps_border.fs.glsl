/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// draw a circle at position aDesiredPos with a aRadius
vec4 drawCircle(vec2 aPixel, vec2 aDesiredPos, float aRadius, vec3 aColor) {
  float farFromCenter = length(aDesiredPos - aPixel) - aRadius;
  float pixelInCircle = 1.00 - clamp(farFromCenter, 0.0, 1.0);
  return vec4(aColor, pixelInCircle);
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

  vec2 size = vec2(vBorders.z, vBorders.w);
  // Get our position within this specific segment
  vec2 position = vPos - vBorders.xy;

  // Break our position into square tiles with circles in them.
  vec2 circleCount = floor(size / circleSpacing);
  circleCount = max(circleCount, 1);

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
  vec3 black = vec3(0.0, 0.0, 0.0);
  // See if we should draw a circle or not
  vec4 circleColor = drawCircle(finalPosition, tileCenter, radius, black);

  return mix(white, circleColor, circleColor.a);
}

void draw_dotted_border(void) {
  switch (vBorderPart) {
    // These are the layer tile part PrimitivePart as uploaded by the tiling.rs
    case PST_TOP_LEFT:
    case PST_TOP_RIGHT:
    case PST_BOTTOM_LEFT:
    case PST_BOTTOM_RIGHT:
      // TODO: Fix for corners with a border-radius
      oFragColor = draw_dotted_edge();
      break;
    case PST_BOTTOM:
    case PST_TOP:
    case PST_LEFT:
    case PST_RIGHT:
      oFragColor = draw_dotted_edge();
      break;
  }
}

void main(void) {
	if (vRadii.x > 0.0 &&
		(distance(vRefPoint, vPos) > vRadii.x ||
		 distance(vRefPoint, vPos) < vRadii.z)) {
		discard;
	}

  switch (vBorderStyle) {
    case BORDER_STYLE_DOTTED:
    {
      draw_dotted_border();
      break;
    }
    case BORDER_STYLE_NONE:
    case BORDER_STYLE_SOLID:
    {
      float color = step(0.0, vF);
      oFragColor = mix(vColor1, vColor0, color);
      break;
    }
    default:
    {
      discard;
      break;
    }
  }
}
