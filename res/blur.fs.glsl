#version 110

#ifdef GL_ES
    precision mediump float;
#endif

uniform sampler2D sMask;
uniform float uBlurRadius;
uniform vec2 uDirection;
uniform vec2 uDestTextureSize;
uniform vec2 uSourceTextureSize;

varying vec2 vMaskTexCoord;

float gauss(float x, float sigma) {
    return (1.0 / sqrt(6.283185307179586 * sigma * sigma)) * exp(-(x * x) / (2.0 * sigma * sigma));
}

void main(void) {
    vec2 sideOffsets = (uDestTextureSize - uSourceTextureSize) / 2.0;
    int range = int(uBlurRadius) * 3;
    float sigma = uBlurRadius / 2.0;
    float value = 0.0;
    for (int offset = -range; offset <= range; offset++) {
        float offsetF = float(offset);
        vec2 lMaskTexCoord = (vMaskTexCoord * uDestTextureSize - sideOffsets) / uSourceTextureSize;
        lMaskTexCoord += vec2(offsetF) / uSourceTextureSize * uDirection;
        float x = lMaskTexCoord.x >= 0.0 &&
            lMaskTexCoord.x <= 1.0 &&
            lMaskTexCoord.y >= 0.0 &&
            lMaskTexCoord.y <= 1.0 ?
            texture2D(sMask, lMaskTexCoord).r :
            0.0;
        value += x * gauss(offsetF, sigma);
    }
    gl_FragColor = vec4(value, value, value, 1.0);
}

