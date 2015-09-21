#version 130

in vec2 aPosition;
in vec4 aColor;

uniform mat4 uTransform;

out vec4 vColor;

void main(void)
{
	vColor = aColor;
    gl_Position = uTransform * vec4(aPosition, 0.0, 1.0);
}
