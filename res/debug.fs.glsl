#version 130

#ifdef GL_ES
    precision mediump float;
#endif

in vec4 vColor;

out vec4 outColor;

void main(void)
{
	outColor = vColor;
}
