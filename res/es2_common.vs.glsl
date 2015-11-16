#version 110

uniform mat4 uTransform;
uniform mat4 uMatrixPalette[32];
uniform vec2 uDirection;
uniform vec2 uTextureSize;
uniform vec4 uBlendParams;
uniform vec4 uFilterParams;
uniform float uDevicePixelRatio;
uniform vec4 uTileParams[256];

attribute vec3 aPosition;
attribute vec4 aColor;
attribute vec2 aColorTexCoord;
attribute vec2 aMaskTexCoord;
attribute vec4 aBorderPosition;
attribute vec4 aBorderRadii;
attribute vec2 aSourceTextureSize;
attribute vec2 aDestTextureSize;
attribute float aBlurRadius;
attribute vec4 aMisc;   // x = matrix index; y = color tex index; z = mask tex index; w=tile params index

varying vec2 vPosition;
varying vec4 vColor;
varying vec3 vColorTexCoord;
varying vec3 vMaskTexCoord;
varying vec4 vBorderPosition;
varying vec4 vBorderRadii;
varying vec2 vDestTextureSize;
varying vec2 vSourceTextureSize;
varying float vBlurRadius;
varying vec4 vTileParams;
