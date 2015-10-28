#version 110

attribute vec3 aPosition;
attribute vec4 aColor;
attribute vec4 aBorderPosition;
attribute vec4 aBorderRadii;

uniform mat4 uTransform;

varying vec4 vColor;
varying vec2 vPosition;
varying vec4 vBorderPosition;
varying vec4 vBorderRadii;

void main(void)
{
	vColor = aColor;
	vPosition = aPosition.xy;
    vBorderPosition = aBorderPosition;
    vBorderRadii = aBorderRadii;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}
