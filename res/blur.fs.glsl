#version 110

#ifdef GL_ES
    precision mediump float;
#endif

uniform sampler2D sDiffuse;
uniform vec2 uDirection;

varying vec2 vColorTexCoord;
varying float vBlurRadius;
varying vec2 vDestTextureSize;
varying vec2 vSourceTextureSize;

float gauss(float x, float sigma) {
    return (1.0 / sqrt(6.283185307179586 * sigma * sigma)) * exp(-(x * x) / (2.0 * sigma * sigma));
}

void main(void) {
    vec2 sideOffsets = (vDestTextureSize - vSourceTextureSize) / 2.0;
    int range = int(vBlurRadius) * 3;
    float sigma = vBlurRadius / 2.0;
    float value = 0.0;
    for (int offset = -range; offset <= range; offset++) {
        float offsetF = float(offset);
        vec2 lColorTexCoord = (vColorTexCoord * vDestTextureSize - sideOffsets) /
            vSourceTextureSize;
        lColorTexCoord += vec2(offsetF) / vSourceTextureSize * uDirection;
        float x = lColorTexCoord.x >= 0.0 &&
            lColorTexCoord.x <= 1.0 &&
            lColorTexCoord.y >= 0.0 &&
            lColorTexCoord.y <= 1.0 ?
            texture2D(sDiffuse, lColorTexCoord).r :
            0.0;
        value += x * gauss(offsetF, sigma);
    }
    gl_FragColor = vec4(value, value, value, 1.0);
}

