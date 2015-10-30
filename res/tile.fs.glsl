void main(void) {
    vec2 textureSize = vBorderPosition.zw - vBorderPosition.xy;
    vec3 colorTexCoord = vec3(vBorderPosition.xy + mod(vColorTexCoord.xy, 1.0) * textureSize,
                              vColorTexCoord.z);
    vec4 diffuse = Texture(sDiffuse, colorTexCoord);
    SetFragColor(diffuse);
}

