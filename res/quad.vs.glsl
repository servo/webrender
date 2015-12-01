void main(void)
{
    // Normalize the vertex color
    vColor = aColor / 255.0;

    // Extract the image tiling parameters.
    // These are passed to the fragment shader, since
    // the uv interpolation must be done per-fragment.
    vTileParams = uTileParams[int(aMisc.w)];

    // Normalize the mask texture coordinates.
    vec2 maskTexCoord = aMaskTexCoord / 65535.0;
    vec2 colorTexCoord = aColorTexCoord;

    // Pass through the color and mask texture coordinates to fragment shader
    vColorTexCoord = colorTexCoord;
    vMaskTexCoord = maskTexCoord;

    // Extract the complete (stacking context + css transform) transform
    // for this vertex. Transform the position by it.
    mat4 matrix = uMatrixPalette[int(aMisc.x)];
    vec4 pos = matrix * vec4(aPosition, 1.0);

    // Snap the vertex to pixel position to guarantee correct texture
    // sampling when using bilinear filtering.
#ifdef SERVO_ES2
    // TODO(gw): ES2 doesn't have round(). Do we ever get negative coords here?
    pos.xy = floor(0.5 + pos.xy * uDevicePixelRatio) / uDevicePixelRatio;
#else
    pos.xy = round(pos.xy * uDevicePixelRatio) / uDevicePixelRatio;
#endif

    // Transform by the orthographic projection into clip space.
    gl_Position = uTransform * pos;
}

