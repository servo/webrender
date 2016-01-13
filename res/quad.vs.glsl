vec2 SnapToPixels(vec2 pos)
{
    // Snap the vertex to pixel position to guarantee correct texture
    // sampling when using bilinear filtering.
#ifdef SERVO_ES2
    // TODO(gw): ES2 doesn't have round(). Do we ever get negative coords here?
    return floor(0.5 + pos * uDevicePixelRatio) / uDevicePixelRatio;
#else
    return round(pos * uDevicePixelRatio) / uDevicePixelRatio;
#endif
}

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
    vec4 offsetParams = uOffsets[int(aMisc.x)];
    mat4 matrix = uMatrixPalette[int(aMisc.x)];

    vec4 localPos = vec4(aPosition.xy, 0.0, 1.0);

    vClipInRect = uClipRects[int(aMisc.y)];
    vClipOutRect = uClipRects[int(aMisc.z)];
    vPosition = localPos.xy;

    vec4 worldPos = matrix * localPos;
    worldPos.xy += offsetParams.xy;
    worldPos.xy += offsetParams.zw;
    worldPos.xy = SnapToPixels(worldPos.xy);

    // Transform by the orthographic projection into clip space.
    gl_Position = uTransform * worldPos;
}

