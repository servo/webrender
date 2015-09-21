#version 130

#ifdef GL_ES
    precision mediump float;
#endif

uniform sampler2D sDiffuse;
uniform sampler2D sMask;

in vec2 vColorTexCoord;
in vec2 vMaskTexCoord;
in vec4 vColor;

out vec4 outColor;

void main(void)
{
	vec4 diffuse = texture2D(sDiffuse, vColorTexCoord);
	vec4 mask = vec4(1.0, 1.0, 1.0, texture2D(sMask, vMaskTexCoord).r);
	outColor = diffuse * vColor * mask;
}
