void main(void) {
    vec2 textureSize = vBorderPosition.zw - vBorderPosition.xy;
    vec2 colorTexCoord = vBorderPosition.xy + mod(vColorTexCoord.xy, 1.0) * textureSize;
    vec4 diffuse = Texture(sDiffuse, colorTexCoord);
    SetFragColor(diffuse);
}

