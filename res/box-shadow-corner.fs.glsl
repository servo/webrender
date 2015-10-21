#version 110

#ifdef GL_ES
    precision mediump float;
#endif

uniform sampler2D sDiffuse;

varying vec4 vColor;
varying vec2 vPosition;
varying vec4 vBorderPosition;
varying vec2 vDestTextureSize;
varying vec4 vBorderRadii;
varying float vBlurRadius;

void main(void)
{
    vec2 lPosition = vPosition - vBorderPosition.zw;
    vec2 lArcCenter = vDestTextureSize;
    float lArcRadius = vBorderRadii.x;
    float lDistance = distance(lPosition, vec2(lArcCenter));
    float lValue = clamp(lDistance, lArcRadius - vBlurRadius, lArcRadius + vBlurRadius);
    lValue = ((lValue - lArcRadius) / vBlurRadius + 1.0) / 2.0;
    gl_FragColor = vColor - vec4(lValue);
}

