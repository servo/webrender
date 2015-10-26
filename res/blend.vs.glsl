void main(void)
{
	vColorTexCoord = vec3(aColorTexCoord / 65535.0, 0.0);
	vMaskTexCoord = vec3(aMaskTexCoord / 65535.0, 0.0);
    gl_Position = uTransform * vec4(aPosition, 1.0);
}
