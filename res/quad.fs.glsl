#ifdef GL_ES
    precision mediump float;
#endif

uniform sampler2D sDiffuse;
uniform sampler2D sMask;

varying vec2 vColorTexCoord;
varying vec2 vMaskTexCoord;
varying vec4 vColor;

void main(void)
{
	vec4 diffuse = texture2D(sDiffuse, vColorTexCoord);

	#ifdef PLATFORM_ANDROID
		vec4 mask = vec4(1.0, 1.0, 1.0, texture2D(sMask, vMaskTexCoord).a);
	#else
		vec4 mask = vec4(1.0, 1.0, 1.0, texture2D(sMask, vMaskTexCoord).r);
	#endif

	gl_FragColor = diffuse * vColor * mask;
}
