void main(void)
{
    vColor = aColor / 255.0;
    vColorTexCoord = vec3(aColorTexCoord.xy, aMisc.y);
    vMaskTexCoord = vec3(aMaskTexCoord.xy / 65535.0, aMisc.z);
    mat4 matrix = uMatrixPalette[int(aMisc.x)];
    vec4 pos = matrix * vec4(aPosition, 1.0);
    pos.xy = floor(pos.xy * uDevicePixelRatio + 0.5) / uDevicePixelRatio;
    gl_Position = uTransform * pos;
}

