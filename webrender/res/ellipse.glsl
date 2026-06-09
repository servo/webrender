/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// Preprocess the radii for computing the distance approximation. This should
// be used in the vertex shader if possible to avoid doing expensive division
// in the fragment shader. When dealing with a point (zero radii), approximate
// it as an ellipse with very small radii so that we don't need to branch.
vec2 inverse_radii_squared(vec2 radii) {
    return 1.0 / max(radii * radii, 1.0e-6);
}

// Same as above, but the exponentiation will come later,
// in the superellipse calculation. We sill want to precompute
// the inverse though.
vec2 inverse_radii(vec2 radii) {
    return 1.0 / max(radii, 1.0e-3);
}

#ifdef WR_FRAGMENT_SHADER

// One iteration of Newton's method on the 2D equation of an ellipse:
//
//     E(x, y) = x^2/a^2 + y^2/b^2 - 1
//
// The Jacobian of this equation is:
//
//     J(E(x, y)) = [ 2*x/a^2 2*y/b^2 ]
//
// We approximate the distance with:
//
//     E(x, y) / ||J(E(x, y))||
//
// See G. Taubin, "Distance Approximations for Rasterizing Implicit
// Curves", section 3.
//
// A scale relative to the unit scale of the ellipse may be passed in to cause
// the math to degenerate to length(p) when scale is 0, or otherwise give the
// normal distance approximation if scale is 1.
float distance_to_ellipse_approx(vec2 p, vec2 inv_radii_sq, float scale) {
    vec2 p_r = p * inv_radii_sq;
    float g = dot(p, p_r) - scale;
    vec2 dG = (1.0 + scale) * p_r;
    return g * inversesqrt(dot(dG, dG));
}

// Slower but more accurate version that uses the exact distance when dealing
// with a 0-radius point distance and otherwise uses the faster approximation
// when dealing with non-zero radii.
float distance_to_ellipse(vec2 p, vec2 radii) {
    return distance_to_ellipse_approx(p, inverse_radii_squared(radii),
                                      float(all(greaterThan(radii, vec2(0.0)))));
}

float distance_to_rounded_rect(
    vec2 pos,
    vec3 plane_tl,
    vec4 center_radius_tl,
    vec3 plane_tr,
    vec4 center_radius_tr,
    vec3 plane_br,
    vec4 center_radius_br,
    vec3 plane_bl,
    vec4 center_radius_bl,
    vec4 rect_bounds
) {
    // Clip against each ellipse. If the fragment is in a corner, one of the
    // branches below will select it as the corner to calculate the distance
    // to. We use half-space planes to detect which corner's ellipse the
    // fragment is inside, where the plane is defined by a normal and offset.
    // If outside any ellipse, default to a small offset so a negative distance
    // is returned for it.
    vec4 corner = vec4(vec2(1.0e-6), vec2(1.0));

    // Calculate the ellipse parameters for each corner.
    center_radius_tl.xy = center_radius_tl.xy - pos;
    center_radius_tr.xy = (center_radius_tr.xy - pos) * vec2(-1.0, 1.0);
    center_radius_br.xy = pos - center_radius_br.xy;
    center_radius_bl.xy = (center_radius_bl.xy - pos) * vec2(1.0, -1.0);

    // Evaluate each half-space plane in turn to select a corner.
    if (dot(pos, plane_tl.xy) > plane_tl.z) {
      corner = center_radius_tl;
    }
    if (dot(pos, plane_tr.xy) > plane_tr.z) {
      corner = center_radius_tr;
    }
    if (dot(pos, plane_br.xy) > plane_br.z) {
      corner = center_radius_br;
    }
    if (dot(pos, plane_bl.xy) > plane_bl.z) {
      corner = center_radius_bl;
    }

    // Calculate the distance of the selected corner and the rectangle bounds,
    // whichever is greater.
    return max(distance_to_ellipse_approx(corner.xy, corner.zw, 1.0),
               signed_distance_rect(pos, rect_bounds.xy, rect_bounds.zw));
}

// `superellipse(K)` defines the corner via |x/a|^n + |y/b|^n = 1 with
// n = 2^K. The shader stores K (not n). Special cases by spec:
//   round    K =  1   n = 2  (the standard ellipse — fast path)
//   squircle K =  2   n = 4
//   bevel    K =  0   n = 1  (straight cut)
//   scoop    K = -1   n = 0.5 (concave bite into the corner)
//   square   K = +inf
//   notch    K = -inf
//
// `p` is the offset of the fragment from the ellipse center (matching
// distance_to_ellipse's frame); the box corner is at (radii.x, radii.y)
// in this frame, the box interior at one or both components negative.
// The curve lives in the +x +y quadrant; F<0 marks the inside of the
// shape, F>0 the carved-away corner region.
float distance_to_superellipse_approx(vec2 p, vec2 inv_radii, float k) {
    // Square: corner clings to the box (n=+inf). Inside the box's
    // L-corner = inside the shape; treat the corner of the rect as the
    // SDF reference. From the ellipse center, the box corner sits at
    // +radii.{x,y}; if p is past either, we're outside the box.
    if (k > 10.0) {
        vec2 radii = 1.0 / inv_radii;
        return max(p.x - radii.x, p.y - radii.y);
    }
    // Notch: K=-inf. The curve degenerates to the inner L (x=0 or y=0
    // within the corner box), so the whole corner box is cut away.
    // Inside the corner box (p > 0), distance to the curve is the
    // perpendicular to the nearer axis, positive because the box-corner
    // side is outside the shape.
    if (k < -10.0) {
        return min(p.x, p.y);
    }

    // Bevel: K=0, n=1 → straight chamfer x/a + y/b = 1. Use only the
    // +x,+y projection so the line distance matches geometry inside
    // the box (where p has negative components).
    if (k == 0.0) {
        vec2 pp = max(p, vec2(0.0));
        float g = dot(pp, inv_radii) - 1.0;
        return g * inversesqrt(dot(inv_radii, inv_radii));
    }

    // Convex case (K>0): standard superellipse |p.x/a|^n + |p.y/b|^n = 1, n = 2^k,
    // centred at the ellipse center. F<0 inside shape, F>0 outside.
    //
    // Concave case (K<0): mirrored superellipse |s/r|^n = 1, n = 2^|k| > 1,
    // centered at the box-outer corner. p = radii - p re-centres the
    // frame. The implicit ellipse's interior (F<0) is the lens around
    // box-outer i.e. OUTSIDE the shape; flip sign so "inside shape"
    // stays negative.
    if (k < 0.0) {
        vec2 radii = 1.0 / inv_radii;
        p = radii - p;
    }

    // Convert the CSS superellipse parameter (k) to the actual
    // exponent: 2^k
    float n = exp2(abs(k));

    // Divide by radii: normalize the position to [0, 1]
    vec2 q = p * inv_radii;

    // Compute the superellipse function
    vec2 qn = pow(q, vec2(n));
    qn = clamp(qn, 0.0, 1.0e3); // Clamp to avoid numerical overflow
    float f = qn.x + qn.y - 1.0;

    // Compute the gradient of the superellipse function
    vec2 qn1 = pow(q, vec2(n - 1.0));
    qn1 = clamp(qn1, 0.0, 1.0e2); // Clamp to avoid numerical overflow
    vec2 grad = n * qn1 * inv_radii;

    // Clamp the gradient to avoid numerical issues, it creates
    // a discontinuity, but the estimated SDF is already way
    // off at that point, and it avoids the distance going to
    // infinity for larger exponents.
    grad = max(grad, inv_radii);

    // SDF estimation; will be okay-ish close to the borders (around 0),
    // but will quickly over/underestimate further away. Should be good
    // enough for distance anti-aliasing, but not for general purpose
    // distance estimation.
    return sign(k) * f * inversesqrt(dot(grad, grad));
}

float distance_to_superellipse(vec2 p, vec2 radii, float k) {
    return distance_to_superellipse_approx(p, inverse_radii(radii), k);
}

// Same as distance_to_rounded_rect but with per-corner shape values.
float distance_to_shaped_rect(
    vec2 pos,
    vec4 center_radius_tl,
    vec4 center_radius_tr,
    vec4 center_radius_br,
    vec4 center_radius_bl,
    vec4 rect_bounds,
    vec4 corner_shapes
) {
    vec2 corner_p = vec2(1.0e-6);
    vec2 corner_inv = vec2(1.0);
    float corner_k = 1.0;

    float in_corner = 0.0;

    vec2 p_tl = center_radius_tl.xy - pos;
    vec2 p_tr = (center_radius_tr.xy - pos) * vec2(-1.0, 1.0);
    vec2 p_br = pos - center_radius_br.xy;
    vec2 p_bl = (center_radius_bl.xy - pos) * vec2(1.0, -1.0);

    if (p_tl.x >= 0.0 && p_tl.y >= 0.0) {
        corner_p = p_tl;
        corner_inv = center_radius_tl.zw;
        corner_k = corner_shapes.x;
        in_corner = 1.0;
    }
    if (p_tr.x >= 0.0 && p_tr.y >= 0.0) {
        corner_p = p_tr;
        corner_inv = center_radius_tr.zw;
        corner_k = corner_shapes.y;
        in_corner = 1.0;
    }
    if (p_br.x >= 0.0 && p_br.y >= 0.0) {
        corner_p = p_br;
        corner_inv = center_radius_br.zw;
        corner_k = corner_shapes.z;
        in_corner = 1.0;
    }
    if (p_bl.x >= 0.0 && p_bl.y >= 0.0) {
        corner_p = p_bl;
        corner_inv = center_radius_bl.zw;
        corner_k = corner_shapes.w;
        in_corner = 1.0;
    }

    float d_corner;
    if (in_corner == 0.0) {
        // Not in any corner. let the rect SDF decide. Use a strongly
        // negative value so max() defers to the rect.
        d_corner = -1.0e6;
    } else if (corner_k == 1.0) {
        d_corner = distance_to_ellipse_approx(corner_p, corner_inv, 1.0);
    } else {
        d_corner = distance_to_superellipse_approx(corner_p, corner_inv, corner_k);
    }

    return max(d_corner,
               signed_distance_rect(pos, rect_bounds.xy, rect_bounds.zw));
}
#endif
