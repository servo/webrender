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
    // Extract the image tiling parameters.
    // These are passed to the fragment shader, since
    // the uv interpolation must be done per-fragment.
    vTileParams = uTileParams[Bottom7Bits(int(aMisc.w))];

    // Determine the position, color, and mask texture coordinates of this vertex.
    vec4 localPos = vec4(0.0, 0.0, 0.0, 1.0);
    bool isBorderCorner = int(aMisc.w) >= 0x80;
    bool isBottomTriangle = IsBottomTriangle();
    if (aPosition.y == 0.0) {
        localPos.y = aPositionRect.y;
        if (aPosition.x == 0.0) {
            localPos.x = aPositionRect.x;
            vColorTexCoord = aColorTexCoordRectTop.xy;
            vMaskTexCoord = aMaskTexCoordRectTop.xy;
            if (!isBorderCorner) {
                vColor = aColorRectTL;
            } else {
                vColor = !isBottomTriangle ? aColorRectTR : aColorRectBL;
            }
        } else {
            localPos.x = aPositionRect.x + aPositionRect.z;
            vColorTexCoord = aColorTexCoordRectTop.zw;
            vMaskTexCoord = aMaskTexCoordRectTop.zw;
            vColor = aColorRectTR;
        }
    } else {
        localPos.y = aPositionRect.y + aPositionRect.w;
        if (aPosition.x == 0.0) {
            localPos.x = aPositionRect.x;
            vColorTexCoord = aColorTexCoordRectBottom.zw;
            vMaskTexCoord = aMaskTexCoordRectBottom.zw;
            vColor = aColorRectBL;
        } else {
            localPos.x = aPositionRect.x + aPositionRect.z;
            vColorTexCoord = aColorTexCoordRectBottom.xy;
            vMaskTexCoord = aMaskTexCoordRectBottom.xy;
            if (!isBorderCorner) {
                vColor = aColorRectBR;
            } else {
                vColor = !isBottomTriangle ? aColorRectTR : aColorRectBL;
            }
        }
    }

    // Normalize the vertex color and mask texture coordinates.
    vColor /= 255.0;
    vMaskTexCoord /= 65535.0;

    // Extract the complete (stacking context + css transform) transform
    // for this vertex. Transform the position by it.
    vec4 offsetParams = uOffsets[Bottom7Bits(int(aMisc.x))];
    mat4 matrix = uMatrixPalette[Bottom7Bits(int(aMisc.x))];

    localPos.xy += offsetParams.xy;

    vClipInRect = uClipRects[int(aMisc.y)];
    vClipOutRect = uClipRects[int(aMisc.z)];
    vPosition = localPos.xy;

    vec4 worldPos = matrix * localPos;
    worldPos.xy = SnapToPixels(worldPos.xy);

    // Transform by the orthographic projection into clip space.
    gl_Position = uTransform * worldPos;
}

