void main(void)
{
    vColor = aColor / 255.0;
    vColorTexCoord = vec3(aColorTexCoord / 65535.0, aMisc.y);
    mat4 matrix = uMatrixPalette[int(aMisc.x)];
    vec4 pos = matrix * vec4(aPosition, 1.0);
    pos.xy = floor(pos.xy + 0.5);
    gl_Position = uTransform * pos;
}
