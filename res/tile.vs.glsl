void main(void)
{
	vColorTexCoord = vec3(aBorderRadii.xy, aMisc.y);
    vBorderPosition = aBorderPosition;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}

