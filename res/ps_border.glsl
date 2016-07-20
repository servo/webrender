#line 1

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// These two are interpolated
varying float vF;   // This is a weighting as we get closer to the bottom right corner?

// These are not changing.
flat varying vec4 vColor0;  // The border color
flat varying vec4 vColor1;  // The border color
flat varying vec4 vRadii;   // The border radius

// These are in device space
varying vec2 vPos;  // This is the clamped position of the current position.
flat varying vec4 vBorders; // The borders

// for corners, this is the beginning of the corner.
// For the lines, this is the top left of the line.
flat varying vec2 vRefPoint;
flat varying int vBorderStyle;
