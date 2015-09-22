attribute vec3 aPosition;
attribute vec4 aColor;

uniform mat4 uTransform;

varying vec4 vColor;
varying vec2 vPosition;

void main(void)
{
	vColor = aColor;
	vPosition = aPosition.xy;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}
