void main(void)
{
    vec4 diffuse = Texture(sDiffuse, vColorTexCoord);
    SetFragColor(diffuse * vColor);
}
