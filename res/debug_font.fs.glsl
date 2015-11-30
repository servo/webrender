void main(void)
{
    float alpha = Texture(sDiffuse, vColorTexCoord.xy).r;
    SetFragColor(vec4(vColor.xyz, vColor.w * alpha));
}
