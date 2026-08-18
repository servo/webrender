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

// Spec: https://drafts.csswg.org/css-borders/#normalized-superellipse-half-corner
//
// Contrary to the spec, the symmetric case (shape < 0) is handled in
// compute_contoured_superellipse() instead of in this function.
float compute_superellipse_half_corner(float shape) {
    shape = min(1.0, abs(shape));

    float n = exp2(shape);
    float convex_half_corner = pow(0.5, 1.0 / n);

    return convex_half_corner;
}

#if 0

// Alternate method to find the normals of a superellipse
// at the start and end points or the curve.
//
// Using implicit differentiation [1, 2], we find that for the superellipse
// defined with this implicit formula:
//
// S(x, y) = (x/a)^n + (y/a)^n - 1 = 0
//
// where (a, b) are the radii and n is the order,
// we find that the gradient at the point (x0, y0) is:
//
// dy/dx = -b^n * x0^(n-1) / a^n * y0^(n-1)
//
// Introducing an intermediate variable q, this simplifies to:
//
// let q = [(x0 / a)^(n-1); (y0 / b)^(n-1)]
//
// dy/dx = -b*qx / a*qy
//
// And, symmetrically:
//
// dx/dy = -a*qy / b*qx
//
// Using both derivatives is useful for numerical stabillty: as the two
// normals (at start and end) tend towards orthogonality when the exponent
// approches 2 (= circle), one of them is likely to go towards infinity.
//
// The exact start and end of the curve are discontinuities, so we pick
// points that are technically not on the curve but very close to start
// and end points, at a distance of 5% of the radii:
//
// Start gradient evaluated at: (0.05*a, b)
// End gradient evaluated at: (a, 0.05*b)
//
// Using ratios of the radii simplifies the computation further:
//
// let x0 = 0.05*a
// let y0 = b
//
// qx = (x0 / a)^(n-1)
//    = (0.05*a / a)^(n-1)
//    = 0.05^(n-1)
//
// qy = (b / b)^(n-1)
//    = 1
//
// The same simplification applies symmetrically for the dx/dy calculation
// at the end point.
//
// In the final form, we can compute a single q value (instead of a vector),
// and derive the gradients from this and the radii ratios (see code below):
//
// let q = 0.05^(n-1)
// dy/dx = -q * b / a
// dx/dy = -q * a / b
//
// The tangents are then defined from the gradients:
// let t1 = (1, dy/dx)
// let t2 = (dx/dy, 1)
//
// And the normals are obtained by a simple 90° rotation of the tangents:
// let n1 = (t1.y, -t1.x) = (dy/dx, -1)
// let n2 = (-t2.y, t2.x) = (-1, dx/dy)
//
// Returns start and end normals in (x, y) and (z, w) respectively
//
// [1] https://www.kristakingmath.com/blog/equation-of-the-tangent-line-with-implicit-differentiation
// [2] https://www.khanacademy.org/math/ap-calculus-bc/bc-differentiation-2-new/bc-3-2/v/implicit-differentiation-1
//
// Note: the points where we compute the gradients are technically not
// on the curve. In practice, we found that it is not an issue unless the
// curve is concave (n < 1). As the spec defines this case to be implemented
// as a geometrical symmetry from the convex case (n > 1), it is never encountered
// in this implementation.
//
// For reference, here is the full form that reprojects the point on the curve,
// given a fixed x0:
//
// qx = (x0 / a)^(n-1)                  # Same as before
// qy = (1 - x0 / a * qx)^((n-1) / n)   # Project the y coord on the curve
//
// Which yields:
//
// dy/dx = -b / a * qx * (1 - x0 / a * qx)^((1-n) / n) # Careful, exponent flipped
//
vec4 compute_superellipse_normals(vec2 radii, float shape) {
    float n = exp2(abs(shape));
    float q = pow(0.05, n - 1.0);

    // x: dy/dx at (0.05 * radii.x, radii.y)
    // y: dx/dy at (radii.x, 0.05 * radii.y)
    vec2 grad = -q * radii.yx / max(radii.xy, 0.1);

    vec2 n1 = normalize(vec2(grad.x, -1.0));
    vec2 n2 = normalize(vec2(-1.0, grad.y));

    return vec4(n1, n2);
}

#else

// Spec: https://drafts.csswg.org/css-borders/#contour-path
//
// Returns start and end normals in (x, y) and (z, w) respectively
vec4 compute_superellipse_normals(vec2 radii, float shape) {

    float convex_half_corner = compute_superellipse_half_corner(shape);
    vec2 tangent = vec2(0.5 - 2.0 * convex_half_corner, 2.0 * convex_half_corner - 1.5);

    // Avoid dividing by zero
    radii = max(radii, 0.1);

    vec2 n1 = normalize(radii.yx * tangent.yx);
    vec2 n2 = normalize(radii.yx * tangent.xy);

    return vec4(n1, n2);
}

#endif

// Spec: https://drafts.csswg.org/css-borders/#contour-path
//
// Returns an offset (x,y), and new radii (zw)
vec4 compute_contoured_superellipse(vec2 radii, float shape, vec2 inset) {
    vec4 normals = compute_superellipse_normals(radii, shape);

    // Flip x/y for symmetry
    if (shape < 0.0) {
        normals = normals.zwxy;
    }

    // Compute delta by extruding inset along the normals
    vec4 delta = normals * inset.yyxx;

    vec2 offset = vec2(delta.x, delta.w);
    vec2 contourRadii = max(radii + vec2(delta.z, delta.y) - offset, 0.1);
    return vec4(offset, contourRadii);
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
        float g = dot(p, inv_radii) - 1.0;
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

    // If position is outside the shaped region, extend along the curve tangents
    if (any(lessThanEqual(q, vec2(0.0)))) {
        vec2 radii = 1.0 / inv_radii;

        vec4 normals = compute_superellipse_normals(radii, k);

        float d1 = dot(p, normals.xy) - normals.y * radii.y;
        float d2 = dot(p, normals.zw) - normals.z * radii.x;
        if (k >= 0.0) {
            return max(-d1, -d2);
        } else {
            return min(d1, d2);
        }
    }

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
    if (any(lessThan(radii, vec2(1.0)))) {
        return -1.0;
    }

    return distance_to_superellipse_approx(p, inverse_radii(radii), k);
}

float distance_to_shaped_corner(vec2 pos, vec2 inv_radii, float shape) {
    if (shape == 1.0) {
        return distance_to_ellipse_approx(pos, inv_radii, 1.0);
    } else {
        return distance_to_superellipse_approx(pos, inv_radii, shape);
    }
}

// Same as distance_to_rounded_rect but with per-corner shape values.
float distance_to_shaped_rect(
    vec2 pos,
    vec4 center_radius_tl,
    vec4 center_radius_tr,
    vec4 center_radius_br,
    vec4 center_radius_bl,
    vec4 rect_bounds,
    vec4 corner_shapes,
    vec4 clamped_inset // always negative (i.e outset-only)
) {
    float d = signed_distance_rect(pos, rect_bounds.xy, rect_bounds.zw);

    vec2 p_tl = center_radius_tl.xy - pos;
    vec2 p_tr = (center_radius_tr.xy - pos) * vec2(-1.0, 1.0);
    vec2 p_br = pos - center_radius_br.xy;
    vec2 p_bl = (center_radius_bl.xy - pos) * vec2(1.0, -1.0);

    vec2 i_tl = (corner_shapes.x == 1.0) ? vec2(0.0) : clamped_inset.xw;
    vec2 i_tr = (corner_shapes.y == 1.0) ? vec2(0.0) : clamped_inset.xy;
    vec2 i_br = (corner_shapes.z == 1.0) ? vec2(0.0) : clamped_inset.zy;
    vec2 i_bl = (corner_shapes.w == 1.0) ? vec2(0.0) : clamped_inset.zw;

    if (p_tl.x >= i_tl.x && p_tl.y >= i_tl.y) {
        d = max(d, distance_to_shaped_corner(p_tl, center_radius_tl.zw, corner_shapes.x));
    }
    if (p_tr.x >= i_tr.x && p_tr.y >= i_tr.y) {
        d = max(d, distance_to_shaped_corner(p_tr, center_radius_tr.zw, corner_shapes.y));
    }
    if (p_br.x >= i_br.x && p_br.y >= i_br.y) {
        d = max(d, distance_to_shaped_corner(p_br, center_radius_br.zw, corner_shapes.z));
    }
    if (p_bl.x >= i_bl.x && p_bl.y >= i_bl.y) {
        d = max(d, distance_to_shaped_corner(p_bl, center_radius_bl.zw, corner_shapes.w));
    }

    return d;
}
#endif
