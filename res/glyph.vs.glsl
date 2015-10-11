#version 110

attribute vec3 aPosition;
attribute vec2 aTexCoord;
attribute vec4 aColor;
attribute vec4 aMatrixIndex;

uniform mat4 uTransform;
uniform mat4 uMatrixPalette[32];

varying vec4 vColor;
varying vec2 vTexCoord;

void main(void)
{
	vColor = aColor;
	vTexCoord = aTexCoord;

	mat4 matrix = uMatrixPalette[int(aMatrixIndex.x)];
    gl_Position = uTransform * matrix * vec4(aPosition, 1.0);
}
