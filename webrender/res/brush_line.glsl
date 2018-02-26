/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define VECS_PER_SPECIFIC_BRUSH 2

#include shared,prim_shared,brush

varying vec2 vLocalPos;

flat varying vec4 vColor;
flat varying int vStyle;
flat varying float vAxisSelect;
flat varying vec4 vParams;
flat varying vec2 vLocalOrigin;

#ifdef WR_VERTEX_SHADER

#define LINE_ORIENTATION_VERTICAL       0
#define LINE_ORIENTATION_HORIZONTAL     1

struct Line {
    vec4 color;
    float wavyLineThickness;
    float style;
    float orientation;
};

Line fetch_line(int address) {
    vec4 data[2] = fetch_from_resource_cache_2(address);
    return Line(data[0], data[1].x, data[1].y, data[1].z);
}

void brush_vs(
    VertexInfo vi,
    int prim_address,
    RectWithSize local_rect,
    ivec3 user_data,
    PictureTask pic_task
) {
    vLocalPos = vi.local_pos;

    // Note: `line` name is reserved in HLSL
    Line line_prim = fetch_line(prim_address);

    switch (int(abs(pic_task.pic_kind_and_raster_mode))) {
        case PIC_TYPE_TEXT_SHADOW:
            vColor = pic_task.color;
            break;
        default:
            vColor = line_prim.color;
            break;
    }

    vec2 pos, size;

    switch (int(line_prim.orientation)) {
        case LINE_ORIENTATION_HORIZONTAL:
            vAxisSelect = 0.0;
            pos = local_rect.p0;
            size = local_rect.size;
            break;
        case LINE_ORIENTATION_VERTICAL:
            vAxisSelect = 1.0;
            pos = local_rect.p0.yx;
            size = local_rect.size.yx;
            break;
        default:
            vAxisSelect = 0.0;
            pos = size = vec2(0.0);
    }

    vLocalOrigin = pos;
    vStyle = int(line_prim.style);

    switch (vStyle) {
        case LINE_STYLE_SOLID: {
            break;
        }
        case LINE_STYLE_DASHED: {
            float dash_length = size.y * 3.0;
            vParams = vec4(2.0 * dash_length, // period
                           dash_length,       // dash length
                           0.0,
                           0.0);
            break;
        }
        case LINE_STYLE_DOTTED: {
            float diameter = size.y;
            float period = diameter * 2.0;
            float center_line = pos.y + 0.5 * size.y;
            float max_x = floor(size.x / period) * period;
            vParams = vec4(period,
                           diameter / 2.0, // radius
                           center_line,
                           max_x);
            break;
        }
        case LINE_STYLE_WAVY: {
            // This logic copied from gecko to get the same results
            float line_thickness = max(line_prim.wavyLineThickness, 1.0);
            // Difference in height between peaks and troughs
            // (and since slopes are 45 degrees, the length of each slope)
            float slope_length = size.y - line_thickness;
            // Length of flat runs
            float flat_length = max((line_thickness - 1.0) * 2.0, 1.0);

            vParams = vec4(line_thickness / 2.0,
                           slope_length,
                           flat_length,
                           size.y);
            break;
        }
        default:
            vParams = vec4(0.0);
    }
}
#endif

#ifdef WR_FRAGMENT_SHADER

#define MAGIC_WAVY_LINE_AA_SNAP         0.5

vec4 brush_fs() {
    // Find the appropriate distance to apply the step over.
    vec2 local_pos = vLocalPos;
    float aa_range = compute_aa_range(local_pos);
    float alpha = 1.0;

    // Select the x/y coord, depending on which axis this edge is.
    vec2 pos = mix(local_pos.xy, local_pos.yx, vAxisSelect);

    switch (vStyle) {
        case LINE_STYLE_SOLID: {
            break;
        }
        case LINE_STYLE_DASHED: {
            // Get the main-axis position relative to closest dot or dash.
            float x = mod(pos.x - vLocalOrigin.x, vParams.x);

            // Calculate dash alpha (on/off) based on dash length
            alpha = step(x, vParams.y);
            break;
        }
        case LINE_STYLE_DOTTED: {
            // Get the main-axis position relative to closest dot or dash.
            float x = mod(pos.x - vLocalOrigin.x, vParams.x);

            // Get the dot alpha
            vec2 dot_relative_pos = vec2(x, pos.y) - vParams.yz;
            float dot_distance = length(dot_relative_pos) - vParams.y;
            alpha = distance_aa(aa_range, dot_distance);
            // Clip off partial dots
            alpha *= step(pos.x - vLocalOrigin.x, vParams.w);
            break;
        }
        case LINE_STYLE_WAVY: {
            vec2 normalized_local_pos = pos - vLocalOrigin.xy;

            float half_line_thickness = vParams.x;
            float slope_length = vParams.y;
            float flat_length = vParams.z;
            float vertical_bounds = vParams.w;
            // Our pattern is just two slopes and two flats
            float half_period = slope_length + flat_length;

            float mid_height = vertical_bounds / 2.0;
            float peak_offset = mid_height - half_line_thickness;
            // Flip the wave every half period
            float flip = -2.0 * (step(mod(normalized_local_pos.x, 2.0 * half_period), half_period) - 0.5);
            // float flip = -1.0;
            peak_offset *= flip;
            float peak_height = mid_height + peak_offset;

            // Convert pos to a local position within one half period
            normalized_local_pos.x = mod(normalized_local_pos.x, half_period);

            // Compute signed distance to the 3 lines that make up an arc
            float dist1 = distance_to_line(vec2(0.0, peak_height),
                                           vec2(1.0, -flip),
                                           normalized_local_pos);
            float dist2 = distance_to_line(vec2(0.0, peak_height),
                                           vec2(0, -flip),
                                           normalized_local_pos);
            float dist3 = distance_to_line(vec2(flat_length, peak_height),
                                           vec2(-1.0, -flip),
                                           normalized_local_pos);
            float dist = abs(max(max(dist1, dist2), dist3));

            // Apply AA based on the thickness of the wave
            alpha = distance_aa(aa_range, dist - half_line_thickness);

            // Disable AA for thin lines
            if (half_line_thickness <= 1.0) {
                alpha = 1.0 - step(alpha, MAGIC_WAVY_LINE_AA_SNAP);
            }

            break;
        }
        default: break;
    }

    return vColor * alpha;
}
#endif
