void main(void)
{
	vColorTexCoord = aColorTexCoordRectTop.xy;
	vMaskTexCoord = aMaskTexCoordRectTop.xy;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}

