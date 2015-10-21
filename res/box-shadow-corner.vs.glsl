#version 110

attribute vec3 aPosition;
attribute vec4 aColor;
attribute vec4 aBorderPosition;
attribute vec2 aDestTextureSize;
attribute vec4 aBorderRadii;
attribute float aBlurRadius;

uniform mat4 uTransform;

varying vec2 vPosition;
varying vec4 vColor;
varying vec4 vBorderPosition;
varying vec2 vDestTextureSize;
varying vec4 vBorderRadii;
varying float vBlurRadius;

void main(void)
{
	vPosition = aPosition.xy;
	vColor = aColor;
    vBorderPosition = aBorderPosition;
    vDestTextureSize = aDestTextureSize;
    vBorderRadii = aBorderRadii;
    vBlurRadius = aBlurRadius;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}
