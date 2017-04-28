/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

flat varying vec4 vColor0;
flat varying vec4 vColor1;
flat varying vec2 vEdgeDistance;
flat varying float vAxisSelect;
flat varying float vAlphaSelect;
flat varying vec3 vDashParams;

#ifdef WR_FEATURE_TRANSFORM
varying vec3 vLocalPos;
flat varying RectWithSize vLocalRect;
#else
varying vec2 vLocalPos;
#endif
