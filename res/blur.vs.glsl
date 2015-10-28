void main(void)
{
	vColorTexCoord = vec3(aColorTexCoord, aMisc.y);
    vBorderPosition = aBorderPosition;
    vBlurRadius = aBlurRadius;
    vDestTextureSize = aDestTextureSize;
    vSourceTextureSize = aSourceTextureSize;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}

