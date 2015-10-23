#version 110

attribute vec2 aPosition;
attribute vec4 aColor;

uniform mat4 uTransform;

varying vec4 vColor;

void main(void)
{
	vColor = aColor / 255.0;
    gl_Position = uTransform * vec4(aPosition, 0.0, 1.0);
}
