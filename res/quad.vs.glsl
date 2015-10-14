#version 110

attribute vec3 aPosition;
attribute vec2 aColorTexCoord;
attribute vec2 aMaskTexCoord;
attribute vec4 aColor;
attribute vec4 aMatrixIndex;

uniform mat4 uTransform;
uniform mat4 uMatrixPalette[32];

varying vec4 vColor;
varying vec2 vColorTexCoord;
varying vec2 vMaskTexCoord;

void main(void)
{
    vColor = aColor;
    vColorTexCoord = aColorTexCoord;
    vMaskTexCoord = aMaskTexCoord;

    mat4 matrix = uMatrixPalette[int(aMatrixIndex.x)];
    vec4 pos = matrix * vec4(aPosition, 1.0);
    pos.xy = floor(pos.xy + 0.5);
    gl_Position = uTransform * pos;
}
