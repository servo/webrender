void main(void)
{
    vec4 diffuse = Texture(sDiffuse, vColorTexCoord);
    vec4 mask = Texture(sMask, vMaskTexCoord);
    float alpha = GetAlphaFromMask(mask);
	SetFragColor(diffuse * vec4(vColor.rgb, vColor.a * alpha));
}

