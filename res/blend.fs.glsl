#version 110

#ifdef GL_ES
    precision mediump float;
#endif

uniform sampler2D sDiffuse;
uniform sampler2D sMask;

varying vec2 vColorTexCoord;
varying vec2 vMaskTexCoord;

void main(void)
{
	vec4 render_target = texture2D(sDiffuse, vColorTexCoord);
	vec4 frame_buffer = texture2D(sMask, vMaskTexCoord);

	gl_FragColor = vec4(abs(frame_buffer.xyz - render_target.xyz), 1);
}
