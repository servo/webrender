#version 110

attribute vec3 aPosition;
attribute vec2 aColorTexCoord;
attribute float aBlurRadius;
attribute vec2 aDestTextureSize;
attribute vec2 aSourceTextureSize;

uniform mat4 uTransform;

varying vec2 vColorTexCoord;
varying float vBlurRadius;
varying vec2 vDestTextureSize;
varying vec2 vSourceTextureSize;

void main(void)
{
	vColorTexCoord = aColorTexCoord;
    vBlurRadius = aBlurRadius;
    vDestTextureSize = aDestTextureSize;
    vSourceTextureSize = aSourceTextureSize;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}

