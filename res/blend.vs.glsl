void main(void)
{
	vColorTexCoord = aColorTexCoord;
	vMaskTexCoord = aMaskTexCoord / 65535.0;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}
