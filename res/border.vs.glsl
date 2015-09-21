#version 130

in vec3 aPosition;
in vec4 aColor;

uniform mat4 uTransform;

out vec4 vColor;
out vec2 vPosition;

void main(void)
{
	vColor = aColor;
	vPosition = aPosition.xy;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}
