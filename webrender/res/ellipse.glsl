/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef WR_FRAGMENT_SHADER

//
// Signed distance to an ellipse.
// Taken from http://www.iquilezles.org/www/articles/ellipsedist/ellipsedist.htm
// Note that this fails for exact circles.
//
float sdEllipse( vec2 p, in vec2 ab ) {
    p = abs( p ); if( p.x > p.y ){ p=p.yx; ab=ab.yx; }
    float l = ab.y*ab.y - ab.x*ab.x;

    float m = ab.x*p.x/l;
    float n = ab.y*p.y/l;
    float m2 = m*m;
    float n2 = n*n;

    float c = (m2 + n2 - 1.0)/3.0;
    float c3 = c*c*c;

    float q = c3 + m2*n2*2.0;
    float d = c3 + m2*n2;
    float g = m + m*n2;

    float co;

    if( d<0.0 )
    {
        float p = acos(q/c3)/3.0;
        float s = cos(p);
        float t = sin(p)*sqrt(3.0);
        float rx = sqrt( -c*(s + t + 2.0) + m2 );
        float ry = sqrt( -c*(s - t + 2.0) + m2 );
        co = ( ry + sign(l)*rx + abs(g)/(rx*ry) - m)/2.0;
    }
    else
    {
        float h = 2.0*m*n*sqrt( d );
        float s = sign(q+h)*pow( abs(q+h), 1.0/3.0 );
        float u = sign(q-h)*pow( abs(q-h), 1.0/3.0 );
        float rx = -s - u - c*4.0 + 2.0*m2;
        float ry = (s - u)*sqrt(3.0);
        float rm = sqrt( rx*rx + ry*ry );
        float p = ry/sqrt(rm-rx);
        co = (p + 2.0*g/rm - m)/2.0;
    }

    float si = sqrt( 1.0 - co*co );

    vec2 r = vec2( ab.x*co, ab.y*si );

    return length(r - p ) * sign(p.y-r.y);
}

float distance_to_ellipse(vec2 p, vec2 radii) {
    // sdEllipse fails on exact circles, so handle equal
    // radii here. The branch coherency should make this
    // a performance win for the circle case too.
    if (radii.x == radii.y) {
        return length(p) - radii.x;
    } else {
        return sdEllipse(p, radii);
    }
}

float clip_against_ellipse_if_needed(
    vec2 pos,
    float current_distance,
    vec4 ellipse_center_radius,
    vec2 sign_modifier,
    float afwidth
) {
    float ellipse_distance = distance_to_ellipse(pos - ellipse_center_radius.xy,
                                                 ellipse_center_radius.zw);

    return mix(current_distance,
               ellipse_distance + afwidth,
               all(lessThan(sign_modifier * pos, sign_modifier * ellipse_center_radius.xy)));
}

float rounded_rect(vec2 pos,
                   vec4 clip_center_radius_tl,
                   vec4 clip_center_radius_tr,
                   vec4 clip_center_radius_br,
                   vec4 clip_center_radius_bl) {
    float current_distance = 0.0;

    // Apply AA
    float afwidth = 0.5 * length(fwidth(pos));

    // Clip against each ellipse.
    current_distance = clip_against_ellipse_if_needed(pos,
                                                      current_distance,
                                                      clip_center_radius_tl,
                                                      vec2(1.0),
                                                      afwidth);

    current_distance = clip_against_ellipse_if_needed(pos,
                                                      current_distance,
                                                      clip_center_radius_tr,
                                                      vec2(-1.0, 1.0),
                                                      afwidth);

    current_distance = clip_against_ellipse_if_needed(pos,
                                                      current_distance,
                                                      clip_center_radius_br,
                                                      vec2(-1.0),
                                                      afwidth);

    current_distance = clip_against_ellipse_if_needed(pos,
                                                      current_distance,
                                                      clip_center_radius_bl,
                                                      vec2(1.0, -1.0),
                                                      afwidth);

    return smoothstep(0.0, afwidth, 1.0 - current_distance);
}
#endif
