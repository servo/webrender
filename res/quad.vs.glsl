#version 130

in vec3 aPosition;
in vec2 aColorTexCoord;
in vec2 aMaskTexCoord;
in vec4 aColor;

uniform mat4 uTransform;

out vec4 vColor;
out vec2 vColorTexCoord;
out vec2 vMaskTexCoord;

void main(void)
{
	vColor = aColor;
	vColorTexCoord = aColorTexCoord;
	vMaskTexCoord = aMaskTexCoord;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}
