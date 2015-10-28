void main(void)
{
	vColor = aColor;
	vPosition = aPosition.xy;
    vBorderPosition = aBorderPosition;
    vBorderRadii = aBorderRadii;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}
