void main(void)
{
	vColorTexCoord = aColorTexCoordRectTop.xy;
	vMaskTexCoord = aMaskTexCoordRectTop.xy / 65535.0;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}
