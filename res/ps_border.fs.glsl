/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// draw a circle at position aDesiredPos with a aRadius
vec4 drawCircle(vec2 aPixel, vec2 aDesiredPos, float aRadius, vec3 aColor) {
  float farFromCenter = length(aDesiredPos - aPixel) - aRadius;
  float pixelInCircle = 1.00 - clamp(farFromCenter, 0.0, 1.0);
  return vec4(aColor, pixelInCircle);
}

void draw_dotted_border(void) {
  // Everything here should be in device pixels.
  float radius = 3;   // Diameter of 20
  float diameter = radius * 2.0;
  float circleSpacing = diameter * 2.0;

  vec2 size = vec2(vBorders.z - vBorders.x, vBorders.w - vBorders.y);
  // Get our position within this specific segment
  vec2 position = vPos - vBorders.xy;

  // Break our position into square tiles with circles in them.
  vec2 circleCount = size / circleSpacing;
  vec2 distBetweenCircles = size / circleCount;
  vec2 circleCenter = distBetweenCircles / 2.0;

  // Find out which tile this pixel belongs to.
  vec2 destTile = floor(position / distBetweenCircles);
  destTile = destTile * distBetweenCircles;
  vec2 tileCenter = destTile + circleCenter;

  // Find the position within the tile
  vec2 positionInTile = mod(position, distBetweenCircles);
  vec2 finalPosition = positionInTile + destTile;

  vec4 white = vec4(1.0, 1.0, 1.0, 1.0);
  vec3 black = vec3(0.0, 0.0, 0.0);
  // See if we should draw a circle or not
  vec4 circleColor = drawCircle(finalPosition, tileCenter, radius, black);

  oFragColor = mix(white, circleColor, circleColor.a);
}

void main(void) {
	if (vRadii.x > 0.0 &&
		(distance(vRefPoint, vPos) > vRadii.x ||
		 distance(vRefPoint, vPos) < vRadii.z)) {
		discard;
	}

  switch (vBorderStyle) {
    case BORDER_STYLE_DOTTED:
      draw_dotted_border();
      break;
    case BORDER_STYLE_NONE:
    case BORDER_STYLE_SOLID:
    {
      float color = step(0.0, vF);
      vec4 red = vec4(1, 0, 0, 1);
      vec4 green = vec4(0, 1, 0, 1);
      oFragColor = mix(green, red, color);
      break;
    }
  }
}
