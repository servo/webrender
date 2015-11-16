void main(void)
{
    vec2 colorTexCoord = vTileParams.xy + fract(vColorTexCoord.xy) * vTileParams.zw;
    vec4 diffuse = Texture(sDiffuse, vec3(colorTexCoord, vColorTexCoord.z));
    vec4 mask = Texture(sMask, vMaskTexCoord);
    float alpha = GetAlphaFromMask(mask);
    SetFragColor(diffuse * vec4(vColor.rgb, vColor.a * alpha));
}

