#version 110

#define SERVO_ES2

uniform mat4 uTransform;
uniform vec4 uOffsets[32];
uniform mat4 uMatrixPalette[32];
uniform vec2 uDirection;
uniform vec4 uBlendParams;
uniform vec4 uFilterParams;
uniform float uDevicePixelRatio;
uniform vec4 uTileParams[64];

attribute vec3 aPosition;
attribute vec4 aColor;
attribute vec2 aColorTexCoord;
attribute vec2 aMaskTexCoord;
attribute vec4 aBorderPosition;
attribute vec4 aBorderRadii;
attribute vec2 aSourceTextureSize;
attribute vec2 aDestTextureSize;
attribute float aBlurRadius;
attribute vec4 aMisc;   // x = matrix index; w = tile params index

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
