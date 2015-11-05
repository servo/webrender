float erf(float x) {
    bool negative = x < 0.0;
    if (negative)
        x = -x;
    float x2 = x * x;
    float x3 = x2 * x;
    float x4 = x2 * x2;
    float denom = 1.0 + 0.278393 * x + 0.230389 * x2 + 0.000972 * x3 + 0.078108 * x4;
    float result = 1.0 - 1.0 / (denom * denom * denom * denom);
    return negative ? -result : result;
}

void main(void)
{
    float range = int(vBlurRadius) * 3.0;
    float sigma = vBlurRadius / 2.0;
    float sigmaSqrt2 = sigma * 1.41421356237;

    vec2 position = vPosition - vBorderPosition.zw;
    vec2 arcCenter = vDestTextureSize;
    float arcRadius = vBorderRadii.x;
    float distance = distance(position, vec2(arcCenter));
    float value = clamp(distance, arcRadius - vBlurRadius, arcRadius + vBlurRadius);
    float minValue = min(value - range, arcRadius) - value;
    float maxValue = min(value + range, arcRadius) - value;
    if (minValue < maxValue) {
        value = 1.0 - 0.5 * (erf(maxValue / sigmaSqrt2) - erf(minValue / sigmaSqrt2));
    } else {
        value = 0.0;
    }
    SetFragColor(vColor - vec4(value));
}

