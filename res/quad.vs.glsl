void main(void)
{
    // Normalize the vertex color
    vColor = aColor / 255.0;

    // Extract the image tiling parameters.
    // These are passed to the fragment shader, since
    // the uv interpolation must be done per-fragment.
    vTileParams = uTileParams[int(aMisc.w)];

    // Normalize the mask texture coordinates.
    vec2 maskTexCoord = aMaskTexCoord.xy / 65535.0;
    vec2 colorTexCoord = aColorTexCoord.xy;

    // Pass through the color and mask texture coordinates to fragment shader
    vColorTexCoord = vec3(colorTexCoord, aMisc.y);
    vMaskTexCoord = vec3(maskTexCoord, aMisc.z);

    // Extract the complete (stacking context + css transform) transform
    // for this vertex. Transform the position by it.
    mat4 matrix = uMatrixPalette[int(aMisc.x)];
    vec4 pos = matrix * vec4(aPosition, 1.0);

    // Snap the vertex to pixel position to guarantee correct texture
    // sampling when using bilinear filtering.
    pos.xy = round(pos.xy * uDevicePixelRatio) / uDevicePixelRatio;

    // Transform by the orthographic projection into clip space.
    gl_Position = uTransform * pos;
}

