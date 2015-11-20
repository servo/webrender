void main(void)
{
    vec4 diffuse = Texture(sDiffuse, vColorTexCoord.xy);
    SetFragColor(diffuse * vColor);
}
