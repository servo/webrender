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

vec2 Bilerp2(vec2 tl, vec2 tr, vec2 br, vec2 bl, vec2 st) {
    return mix(mix(tl, bl, st.y), mix(tr, br, st.y), st.x);
}

vec4 Bilerp4(vec4 tl, vec4 tr, vec4 br, vec4 bl, vec2 st) {
    return mix(mix(tl, bl, st.y), mix(tr, br, st.y), st.x);
}

void main(void)
{
    // Extract the image tiling parameters.
    // These are passed to the fragment shader, since
    // the uv interpolation must be done per-fragment.
    vTileParams = uTileParams[Bottom7Bits(int(aMisc.w))];

    // Determine clip rects.
    vClipOutRect = uClipRects[int(aMisc.z)];
    vec4 clipInRect = uClipRects[int(aMisc.y)];

    // Determine the position, color, and mask texture coordinates of this vertex.
    vec4 localPos = vec4(0.0, 0.0, 0.0, 1.0);
    bool isBorderCorner = int(aMisc.w) >= 0x80;
    bool isBottomTriangle = IsBottomTriangle();
    if (aPosition.y == 0.0) {
        localPos.y = aPositionRect.y;
        if (aPosition.x == 0.0) {
            localPos.x = aPositionRect.x;
            if (isBorderCorner) {
                vColor = !isBottomTriangle ? aColorRectTR : aColorRectBL;
            }
        } else {
            localPos.x = aPositionRect.x + aPositionRect.z;
            if (isBorderCorner) {
                vColor = aColorRectTR;
            }
        }
    } else {
        localPos.y = aPositionRect.y + aPositionRect.w;
        if (aPosition.x == 0.0) {
            localPos.x = aPositionRect.x;
            if (isBorderCorner) {
                vColor = aColorRectBL;
            }
        } else {
            localPos.x = aPositionRect.x + aPositionRect.z;
            if (isBorderCorner) {
                vColor = !isBottomTriangle ? aColorRectTR : aColorRectBL;
            }
        }
    }

    // Extract the complete (stacking context + css transform) transform
    // for this vertex. Transform the position by it.
    vec4 offsetParams = uOffsets[Bottom7Bits(int(aMisc.x))];
    mat4 matrix = uMatrixPalette[Bottom7Bits(int(aMisc.x))];

    localPos.xy += offsetParams.xy;

    // Clip and compute varyings.
    localPos.xy = clamp(localPos.xy, clipInRect.xy, clipInRect.zw);
    vec2 localST = (localPos.xy - (aPositionRect.xy + offsetParams.xy)) / aPositionRect.zw;
    vColorTexCoord = Bilerp2(aColorTexCoordRectTop.xy, aColorTexCoordRectTop.zw,
                             aColorTexCoordRectBottom.xy, aColorTexCoordRectBottom.zw,
                             localST);
    vMaskTexCoord = Bilerp2(aMaskTexCoordRectTop.xy, aMaskTexCoordRectTop.zw,
                            aMaskTexCoordRectBottom.xy, aMaskTexCoordRectBottom.zw,
                            localST);
    if (!isBorderCorner) {
        vColor = Bilerp4(aColorRectTL, aColorRectTR, aColorRectBR, aColorRectBL, localST);
    }

    // Normalize the vertex color and mask texture coordinates.
    vColor /= 255.0;
    vMaskTexCoord /= uAtlasParams.zw;

    vPosition = localPos.xy;

    vec4 worldPos = matrix * localPos;

    // Transform by the orthographic projection into clip space.
    gl_Position = uTransform * worldPos;
}

