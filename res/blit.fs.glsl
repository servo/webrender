void main(void)
{
    vec4 diffuse = Texture2D(sDiffuse2D, vColorTexCoord.xy);
    SetFragColor(diffuse * vColor);
}
