/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

varying vec4 vData;

#ifdef WR_VERTEX_SHADER
attribute vec4 aValue;
attribute vec2 aPosition;

void main() {
	vData = aValue;
	gl_Position = vec4(aPosition * 2.0 - 1.0, 0.0, 1.0);
}

#endif //WR_VERTEX_SHADER

#ifdef WR_FRAGMENT_SHADER
void main() {
	gl_FragColor = vData;
}
#endif //WR_FRAGMENT_SHADER
