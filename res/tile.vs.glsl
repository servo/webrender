void main(void)
{
	vColorTexCoord = aBorderRadii.xy;
    vBorderPosition = aBorderPosition;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}

