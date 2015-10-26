void main(void)
{
    vec2 lPosition = vPosition - vBorderPosition.zw;
    vec2 lArcCenter = vDestTextureSize;
    float lArcRadius = vBorderRadii.x;
    float lDistance = distance(lPosition, vec2(lArcCenter));
    float lValue = clamp(lDistance, lArcRadius - vBlurRadius, lArcRadius + vBlurRadius);
    lValue = ((lValue - lArcRadius) / vBlurRadius + 1.0) / 2.0;
    SetFragColor(vColor - vec4(lValue));
}

