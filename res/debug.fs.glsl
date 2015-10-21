#version 110

#ifdef GL_ES
    precision mediump float;
#endif

varying vec4 vColor;

void main(void)
{
	gl_FragColor = vColor;
}
