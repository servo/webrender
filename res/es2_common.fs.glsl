#version 110

#define SERVO_ES2

precision highp float;

uniform sampler2D sDiffuse;
uniform sampler2D sMask;
uniform vec4 uBlendParams;
uniform vec4 uAtlasParams;
uniform vec2 uDirection;
uniform vec4 uFilterParams;

varying vec2 vPosition;
varying vec4 vColor;
varying vec2 vColorTexCoord;
varying vec2 vMaskTexCoord;
varying vec4 vBorderPosition;
varying vec4 vBorderRadii;
varying vec2 vDestTextureSize;
varying vec2 vSourceTextureSize;
varying float vBlurRadius;
varying vec4 vTileParams;
varying vec4 vClipInRect;
varying vec4 vClipOutRect;

vec4 Texture(sampler2D sampler, vec2 texCoord) {
    return texture2D(sampler, texCoord);
}

float GetAlphaFromMask(vec4 mask) {
    return mask.a;
}

void SetFragColor(vec4 color) {
    gl_FragColor = color;
}

