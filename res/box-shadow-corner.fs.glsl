#version 110

#ifdef GL_ES
    precision mediump float;
#endif

uniform sampler2D sDiffuse;

uniform vec4 uPosition;
uniform float uBlurRadius;
uniform float uArcRadius;

varying vec4 vColor;
varying vec2 vPosition;

void main(void)
{
    vec2 lPosition = vPosition - uPosition.xy;
    vec2 lArcCenter = uPosition.zw;
    float lDistance = distance(lPosition, vec2(lArcCenter));
    float lValue = clamp(lDistance, uArcRadius - uBlurRadius, uArcRadius + uBlurRadius);
    lValue = ((lValue - uArcRadius) / uBlurRadius + 1.0) / 2.0;
    gl_FragColor = vec4(1.0 - lValue);
}

