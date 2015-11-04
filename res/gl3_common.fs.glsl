#version 150

uniform sampler2DArray sDiffuse;
uniform sampler2DArray sMask;
uniform sampler2D sDiffuse2D;
uniform sampler2D sMask2D;
uniform vec4 uBlendParams;
uniform vec2 uDirection;
uniform vec4 uFilterParams;
uniform vec2 uTextureSize;

in vec2 vPosition;
in vec4 vColor;
in vec3 vColorTexCoord;
in vec3 vMaskTexCoord;
in vec4 vBorderPosition;
in vec4 vBorderRadii;
in vec2 vDestTextureSize;
in vec2 vSourceTextureSize;
in float vBlurRadius;

out vec4 oFragColor;

vec4 Texture(sampler2DArray sampler, vec3 texCoord) {
    return texture(sampler, texCoord);
}

vec4 Texture2D(sampler2D sampler, vec2 texCoord) {
    return texture(sampler, texCoord);
}

float GetAlphaFromMask(vec4 mask) {
    return mask.r;
}

void SetFragColor(vec4 color) {
    oFragColor = color;
}

