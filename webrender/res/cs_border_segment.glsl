/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared,ellipse,shared_border

// For edges, the colors are the same. For corners, these
// are the colors of each edge making up the corner.
flat varying vec4 vColor0;
flat varying vec4 vColor1;

// A point + tangent defining the line where the edge
// transition occurs. Used for corners only.
flat varying vec4 vColorLine;

// Pass through the border style directly. Style
// is the same for edges, may be different for corners.
flat varying ivec2 vStyle;

// x = true if corner, y and z specify if edge is
// horizontal or vertical.
flat varying ivec3 vConfig;

// Local space position of the clip center.
flat varying vec2 vClipCenter;

// An outer and inner elliptical radii for border
// corner clipping.
flat varying vec4 vClipRadii;

// Scale the rect origin by this to get the outer
// corner from the segment rectangle.
flat varying vec2 vClipSign;

// Widths of the edges for this edge or corner.
flat varying vec2 vWidths;

// Vertical / horizontal clip distances for double style.
flat varying vec4 vEdgeClip;

// Stores widths/2 and widths/3 to save doing this in FS.
flat varying vec4 vPartialWidths;

// Local space position
varying vec2 vPos;

#define SEGMENT_TOP_LEFT        0
#define SEGMENT_TOP_RIGHT       1
#define SEGMENT_BOTTOM_RIGHT    2
#define SEGMENT_BOTTOM_LEFT     3
#define SEGMENT_LEFT            4
#define SEGMENT_TOP             5
#define SEGMENT_RIGHT           6
#define SEGMENT_BOTTOM          7

#define EDGE_VERTICAL           0
#define EDGE_HORIZONTAL         1

#ifdef WR_VERTEX_SHADER

in vec2 aTaskOrigin;
in vec4 aRect;
in vec4 aColor0;
in vec4 aColor1;
in int aFlags;
in vec2 aWidths;
in vec2 aRadii;

vec2 get_outer_corner_scale(int segment) {
    vec2 p;

    switch (segment) {
        case SEGMENT_TOP_LEFT:
            p = vec2(0.0, 0.0);
            break;
        case SEGMENT_TOP_RIGHT:
            p = vec2(1.0, 0.0);
            break;
        case SEGMENT_BOTTOM_RIGHT:
            p = vec2(1.0, 1.0);
            break;
        case SEGMENT_BOTTOM_LEFT:
            p = vec2(0.0, 1.0);
            break;
        default:
            // Should never get hit
            p = vec2(0.0);
            break;
    }

    return p;
}

void main(void) {
    int segment = aFlags & 0xff;
    int style0 = (aFlags >> 8) & 0xff;
    int style1 = (aFlags >> 16) & 0xff;

    vec2 outer_scale = get_outer_corner_scale(segment);
    vec2 outer = outer_scale * aRect.zw;
    vec2 clip_sign = 1.0 - 2.0 * outer_scale;

    ivec3 cfg;
    switch (segment) {
        case SEGMENT_TOP_LEFT:
            cfg = ivec3(true, EDGE_HORIZONTAL, EDGE_VERTICAL);
            break;
        case SEGMENT_TOP_RIGHT:
            cfg = ivec3(true, EDGE_VERTICAL, EDGE_HORIZONTAL);
            break;
        case SEGMENT_BOTTOM_RIGHT:
            cfg = ivec3(true, EDGE_HORIZONTAL, EDGE_VERTICAL);
            break;
        case SEGMENT_BOTTOM_LEFT:
            cfg = ivec3(true, EDGE_VERTICAL, EDGE_HORIZONTAL);
            break;
        case SEGMENT_LEFT:
        case SEGMENT_RIGHT:
            cfg = ivec3(false, EDGE_HORIZONTAL, EDGE_HORIZONTAL);
            break;
        case SEGMENT_TOP:
        case SEGMENT_BOTTOM:
            cfg = ivec3(false, EDGE_VERTICAL, EDGE_VERTICAL);
            break;
        default:
            break;
    }

    vConfig = cfg;
    vStyle = ivec2(style0, style1);
    vColor0 = aColor0;
    vColor1 = aColor1;
    vWidths = aWidths;
    vPartialWidths = vec4(vWidths / 3.0, vWidths / 2.0);
    vPos = aRect.zw * aPosition.xy;

    vClipSign = clip_sign;
    vClipCenter = outer + clip_sign * aRadii;
    vClipRadii = vec4(aRadii, max(aRadii - aWidths, 0.0));
    vColorLine = vec4(outer, aWidths.y * -clip_sign.y, aWidths.x * clip_sign.x);

    // Derive the positions for the edge clips, which must be handled
    // differently between corners and edges.
    if (cfg.x != 0) {
        vec2 p0 = outer + clip_sign * vPartialWidths.xy;
        vec2 p1 = outer + clip_sign * (vWidths - vPartialWidths.xy);

        vEdgeClip = vec4(
            min(p0, p1),
            max(p0, p1)
        );
    } else {
        vEdgeClip = vec4(vPartialWidths.xy, aRect.zw - vPartialWidths.xy);
    }

    gl_Position = uTransform * vec4(aTaskOrigin + aRect.xy + vPos, 0.0, 1.0);
}
#endif

#ifdef WR_FRAGMENT_SHADER
vec4 evaluate_color_for_style_in_corner(
    vec2 clip_relative_pos,
    int style,
    vec4 color,
    vec4 clip_radii,
    vec2 widths,
    float aa_range
) {
    switch (style) {
        case BORDER_STYLE_DOUBLE: {
            float d_radii_a = distance_to_ellipse(
                clip_relative_pos,
                clip_radii.xy - vPartialWidths.xy,
                aa_range
            );
            float d_radii_b = distance_to_ellipse(
                clip_relative_pos,
                clip_radii.xy - 2.0 * vPartialWidths.xy,
                aa_range
            );
            float d = min(-d_radii_a, d_radii_b);
            float alpha = distance_aa(aa_range, d);
            return alpha * color;
        }
        default:
            break;
    }

    return color;
}

vec4 evaluate_color_for_style_in_edge(
    vec2 pos,
    int style,
    vec4 color,
    vec2 widths,
    float aa_range,
    int edge_kind
) {
    switch (style) {
        case BORDER_STYLE_DOUBLE: {
            float d0 = -1.0;
            float d1 = -1.0;
            if (edge_kind == EDGE_VERTICAL) {
                // To pass reftests, we need to ensure that
                // we don't apply edge clips for double style
                // when there are < 3 device pixels.
                if (widths.y > 3.0) {
                    d0 = pos.y - vEdgeClip.y;
                    d1 = vEdgeClip.w - pos.y;
                }
            } else {
                if (widths.x > 3.0) {
                    d0 = pos.x - vEdgeClip.x;
                    d1 = vEdgeClip.z - pos.x;
                }
            }
            float d = min(d0, d1);
            float alpha = distance_aa(aa_range, d);
            return alpha * color;
        }
        default:
            break;
    }

    return color;
}

void main(void) {
    float aa_range = compute_aa_range(vPos);
    float d = -1.0;
    vec4 color0, color1;

    // Determine which side of the edge transition this
    // fragment belongs to.
    float mix_factor = 0.0;
    if (vConfig.x != 0) {
        float d_line = distance_to_line(vColorLine.xy, vColorLine.zw, vPos);
        mix_factor = distance_aa(aa_range, -d_line);
    }

    // Check if inside corner clip-region
    vec2 clip_relative_pos = vPos - vClipCenter;
    bool in_clip_region = all(lessThan(vClipSign * clip_relative_pos, vec2(0.0)));

    if (in_clip_region) {
        float d_radii_a = distance_to_ellipse(clip_relative_pos, vClipRadii.xy, aa_range);
        float d_radii_b = distance_to_ellipse(clip_relative_pos, vClipRadii.zw, aa_range);
        float d_radii = max(d_radii_a, -d_radii_b);
        d = max(d, d_radii);

        color0 = evaluate_color_for_style_in_corner(
            clip_relative_pos,
            vStyle.x,
            vColor0,
            vClipRadii,
            vWidths,
            aa_range
        );
        color1 = evaluate_color_for_style_in_corner(
            clip_relative_pos,
            vStyle.y,
            vColor1,
            vClipRadii,
            vWidths,
            aa_range
        );
    } else {
        color0 = evaluate_color_for_style_in_edge(
            vPos,
            vStyle.x,
            vColor0,
            vWidths,
            aa_range,
            vConfig.y
        );
        color1 = evaluate_color_for_style_in_edge(
            vPos,
            vStyle.y,
            vColor1,
            vWidths,
            aa_range,
            vConfig.z
        );
    }

    float alpha = distance_aa(aa_range, d);
    vec4 color = mix(color0, color1, mix_factor);
    oFragColor = color * alpha;
}
#endif
