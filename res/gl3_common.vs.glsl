#version 150

uniform mat4 uTransform;
uniform mat4 uMatrixPalette[32];
uniform vec2 uDirection;
uniform vec2 uTextureSize;
uniform vec4 uBlendParams;
uniform vec4 uFilterParams;
uniform float uDevicePixelRatio;
uniform vec4 uTileParams[256];

in vec3 aPosition;
in vec4 aColor;
in vec2 aColorTexCoord;
in vec2 aMaskTexCoord;
in vec4 aBorderPosition;
in vec4 aBorderRadii;
in vec2 aSourceTextureSize;
in vec2 aDestTextureSize;
in float aBlurRadius;
in vec4 aMisc;  // x = matrix index; y = color tex index; z = mask tex index; w=tile params index

out vec2 vPosition;
out vec4 vColor;
out vec3 vColorTexCoord;
out vec3 vMaskTexCoord;
out vec4 vBorderPosition;
out vec4 vBorderRadii;
out vec2 vDestTextureSize;
out vec2 vSourceTextureSize;
out float vBlurRadius;
out vec4 vTileParams;
