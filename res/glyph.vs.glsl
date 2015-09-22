attribute vec3 aPosition;
attribute vec2 aTexCoord;
attribute vec4 aColor;

uniform mat4 uTransform;

varying vec4 vColor;
varying vec2 vTexCoord;

void main(void)
{
	vColor = aColor;
	vTexCoord = aTexCoord;
    gl_Position = uTransform * vec4(aPosition, 1.0);
}
