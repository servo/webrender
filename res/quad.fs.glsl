// GLSL point in rect test.
// See: https://stackoverflow.com/questions/12751080/glsl-point-inside-box-test
bool PointInRect(vec2 p, vec2 p0, vec2 p1)
{
    vec2 s = step(p0, p) - step(p1, p);
    return s.x * s.y != 0.0;
}

void main(void)
{
    // Clip in rect
    if (!PointInRect(vPosition, vClipInRect.xy, vClipInRect.zw)) {
        discard;
    }

    // Clip out rect
    if (PointInRect(vPosition, vClipOutRect.xy, vClipOutRect.zw)) {
        discard;
    }

    // Apply image tiling parameters (offset and scale) to color UVs.
    vec2 colorTexCoord = vTileParams.xy + fract(vColorTexCoord.xy) * vTileParams.zw;
    vec2 maskTexCoord = vMaskTexCoord.xy;

    // Snap the texture coordinates to the nearest texel.
    // This is important to avoid linear filtering
    // artifacts when the texture coordinates have been
    // passed through the clipper, and may not be aligned
    // to texel boundaries.
    vec2 dColor = 0.5 / uAtlasParams.xy;
    vec2 dMask = 0.5 / uAtlasParams.zw;
    vec2 snappedColorTexCoord = dColor + floor(colorTexCoord * uAtlasParams.xy) / uAtlasParams.xy;
    vec2 snappedMaskTexCoord = dMask + floor(maskTexCoord * uAtlasParams.zw) / uAtlasParams.zw;

    // Fetch the diffuse and mask texels.
    vec4 diffuse = Texture(sDiffuse, snappedColorTexCoord);
    vec4 mask = Texture(sMask, snappedMaskTexCoord);

    // Extract alpha from the mask (component depends on platform)
    float alpha = GetAlphaFromMask(mask);

    // Write the final fragment color.
    SetFragColor(diffuse * vec4(vColor.rgb, vColor.a * alpha));
}
