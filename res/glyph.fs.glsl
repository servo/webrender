void main(void)
{
    vec4 diffuse = Texture(sDiffuse, vColorTexCoord);
    float alpha = GetAlphaFromMask(diffuse);
	SetFragColor(vec4(vColor.xyz, alpha * vColor.w));
}

