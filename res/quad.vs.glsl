attribute vec3 aPosition;
attribute vec2 aColorTexCoord;
attribute vec2 aMaskTexCoord;
attribute vec4 aColor;

uniform mat4 uTransform;

varying vec4 vColor;
varying vec2 vColorTexCoord;
varying vec2 vMaskTexCoord;

void main(void)
{
	vColor = aColor;
	vColorTexCoord = aColorTexCoord;
	vMaskTexCoord = aMaskTexCoord;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}
