void main(void)
{
	vColorTexCoord = aColorTexCoord;
    vBorderPosition = aBorderPosition;
    vBlurRadius = aBlurRadius;
    vDestTextureSize = aDestTextureSize;
    vSourceTextureSize = aSourceTextureSize;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}

