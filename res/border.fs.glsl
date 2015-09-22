#ifdef GL_ES
    precision mediump float;
#endif

uniform sampler2D sDiffuse;

uniform vec4 uPosition;
uniform vec4 uRadii;

varying vec4 vColor;
varying vec2 vPosition;

/*
    Ellipse equation:

    (x-h)^2     (y-k)^2
    -------  +  -------   <=  1
      rx^2        ry^2

 */

void main(void)
{
    float h = uPosition.x;
    float k = uPosition.y;
    float outer_rx = uRadii.x;
    float outer_ry = uRadii.y;
    float inner_rx = uRadii.z;
    float inner_ry = uRadii.w;

    float outer_dx = ((vPosition.x - h) * (vPosition.x - h)) / (outer_rx * outer_rx);
    float outer_dy = ((vPosition.y - k) * (vPosition.y - k)) / (outer_ry * outer_ry);

    float inner_dx = ((vPosition.x - h) * (vPosition.x - h)) / (inner_rx * inner_rx);
    float inner_dy = ((vPosition.y - k) * (vPosition.y - k)) / (inner_ry * inner_ry);

    if ((outer_dx + outer_dy <= 1.0) &&
        (inner_dx + inner_dy >= 1.0)) {
        gl_FragColor = vec4(1.0);
    } else {
        gl_FragColor = vec4(0.0);
    }
}
