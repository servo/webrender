#version 110

#ifdef GL_ES
    precision mediump float;
#endif

uniform sampler2D sDiffuse;

varying vec2 vTexCoord;
varying vec4 vColor;

void main(void)
{
	#ifdef PLATFORM_ANDROID
		float alpha = texture2D(sDiffuse, vTexCoord).a;
	#else
		float alpha = texture2D(sDiffuse, vTexCoord).r;
	#endif
	gl_FragColor = vec4(vColor.xyz, alpha * vColor.w);
}
