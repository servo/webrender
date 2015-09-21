#version 130

#ifdef GL_ES
    precision mediump float;
#endif

uniform sampler2D sDiffuse;

in vec2 vTexCoord;
in vec4 vColor;

out vec4 outColor;

void main(void)
{
	float alpha = texture2D(sDiffuse, vTexCoord).r;
	outColor = vec4(vColor.xyz, alpha);
}
