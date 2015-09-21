#version 130

in vec3 aPosition;
in vec2 aTexCoord;
in vec4 aColor;

uniform mat4 uTransform;

out vec4 vColor;
out vec2 vTexCoord;

void main(void)
{
	vColor = aColor;
	vTexCoord = aTexCoord;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}
