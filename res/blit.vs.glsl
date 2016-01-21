void main(void)
{
    vColor = aColorRectTL / 255.0;
    vColorTexCoord = aColorTexCoordRectTop.xy;
    vec4 pos = vec4(aPosition, 1.0);
    pos.xy = floor(pos.xy * uDevicePixelRatio + 0.5) / uDevicePixelRatio;
    gl_Position = uTransform * pos;
}
