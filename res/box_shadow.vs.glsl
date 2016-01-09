void main(void)
{
	vPosition = aPosition.xy;
	vColor = aColorRectTL;
    vBorderPosition = aBorderPosition;
    vBorderRadii = aBorderRadii;
    vBlurRadius = aBlurRadius;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}

