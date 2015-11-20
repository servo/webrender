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
    float range = floor(vBlurRadius) * 3.0;
    float sigma = vBlurRadius / 2.0;
    float sigmaSqrt2 = sigma * 1.41421356237;

    float length;
    float value;
    vec2 position = vPosition - vBorderPosition.zw;
    if (vBorderRadii.z == 0.0) {
        length = range;
        value = position.x;
    } else {
        length = vBorderRadii.x;
        vec2 center = vec2(max(position.x - range, length),
                           max(position.y - range, length));
        value = distance(position - range, center);
    }

    float minValue = min(value - range, length) - value;
    float maxValue = min(value + range, length) - value;
    if (minValue < maxValue) {
        value = 1.0 - 0.5 * (erf(maxValue / sigmaSqrt2) - erf(minValue / sigmaSqrt2));
    } else {
        value = 1.0;
    }
    SetFragColor(vec4(vColor.rgb, vColor.a == 0.0 ? value : 1.0 - value));
}

