// `vBorderPosition` is the position of the source texture in the atlas.

float gauss(float x, float sigma) {
    return (1.0 / sqrt(6.283185307179586 * sigma * sigma)) * exp(-(x * x) / (2.0 * sigma * sigma));
}

void main(void) {
#ifdef SERVO_ES2
    // TODO(gw): for loops have to be unrollable on es2.
    SetFragColor(vec4(1.0, 0.0, 0.0, 1.0));
#else
    vec2 sideOffsets = (vDestTextureSize - vSourceTextureSize) / 2.0;
    int range = int(vBlurRadius) * 3;
    float sigma = vBlurRadius / 2.0;
    float value = 0.0;
    vec2 sourceTextureUvOrigin = vBorderPosition.xy;
    vec2 sourceTextureUvSize = vBorderPosition.zw - sourceTextureUvOrigin;
    for (int offset = -range; offset <= range; offset++) {
        float offsetF = float(offset);
        vec2 lColorTexCoord = (vColorTexCoord.xy * vDestTextureSize - sideOffsets) /
            vSourceTextureSize;
        lColorTexCoord += vec2(offsetF) / vSourceTextureSize * uDirection;
        float x = lColorTexCoord.x >= 0.0 &&
            lColorTexCoord.x <= 1.0 &&
            lColorTexCoord.y >= 0.0 &&
            lColorTexCoord.y <= 1.0 ?
            Texture(sDiffuse, lColorTexCoord * sourceTextureUvSize + sourceTextureUvOrigin).r :
            0.0;
        value += x * gauss(offsetF, sigma);
    }
    SetFragColor(vec4(value, value, value, 1.0));
#endif
}

