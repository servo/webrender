void main(void)
{
	vColorTexCoord = aColorTexCoord;
	vMaskTexCoord = aMaskTexCoord;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}

