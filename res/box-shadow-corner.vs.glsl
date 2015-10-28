void main(void)
{
	vPosition = aPosition.xy;
	vColor = aColor;
    vBorderPosition = aBorderPosition;
    vDestTextureSize = aDestTextureSize;
    vBorderRadii = aBorderRadii;
    vBlurRadius = aBlurRadius;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}

