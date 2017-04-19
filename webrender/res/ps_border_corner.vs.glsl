#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

void set_radii(vec2 border_radius,
               vec2 border_width,
               vec2 invalid_radii,
               out vec4 radii) {
    if (border_radius.x > 0.0 && border_radius.y > 0.0) {
        // Set inner/outer radius on valid border radius.
        radii.xy = border_radius;
    } else {
        // No border radius - ensure clip has no effect.
        radii.xy = invalid_radii;
    }

    if (all(greaterThan(border_radius, border_width))) {
        radii.zw = border_radius - border_width;
    } else {
        radii.zw = vec2(0.0);
    }
}

void set_edge_line(vec2 border_width,
                   vec2 outer_corner,
                   vec2 gradient_sign) {
    vec2 gradient = border_width * gradient_sign;
    vColorEdgeLine = vec4(outer_corner, vec2(-gradient.y, gradient.x));
}

void main(void) {
    Primitive prim = load_primitive();
    Border border = fetch_border(prim.prim_index);
    int sub_part = prim.sub_index;
    BorderCorners corners = get_border_corners(border, prim.local_rect);

    vec4 adjusted_widths = get_effective_border_widths(border);
    vec2 p0, p1;

    // TODO(gw): We'll need to pass through multiple styles
    //           once we support style transitions per corner.
    int style;
    vec4 edge_distance;

    switch (sub_part) {
        case 0: {
            p0 = corners.tl_outer;
            p1 = corners.tl_inner;
            vColor0 = border.colors[0];
            vColor1 = border.colors[1];
            vClipCenter = corners.tl_outer + border.radii[0].xy;
            vClipSign = vec2(1.0);
            set_radii(border.radii[0].xy,
                      adjusted_widths.xy,
                      vec2(2.0 * border.widths.xy),
                      vRadii0);
            set_radii(border.radii[0].xy - (border.widths.xy - adjusted_widths.xy),
                      adjusted_widths.xy,
                      vec2(0.0),
                      vRadii1);
            set_edge_line(border.widths.xy,
                          corners.tl_outer,
                          vec2(1.0, 1.0));
            style = int(border.style.x);
            edge_distance = vec4(p0 + adjusted_widths.xy,
                                 p0 + border.widths.xy - adjusted_widths.xy);
            break;
        }
        case 1: {
            p0 = vec2(corners.tr_inner.x, corners.tr_outer.y);
            p1 = vec2(corners.tr_outer.x, corners.tr_inner.y);
            vColor0 = border.colors[1];
            vColor1 = border.colors[2];
            vClipCenter = corners.tr_outer + vec2(-border.radii[0].z, border.radii[0].w);
            vClipSign = vec2(-1.0, 1.0);
            set_radii(border.radii[0].zw,
                      adjusted_widths.zy,
                      vec2(2.0 * border.widths.zy),
                      vRadii0);
            set_radii(border.radii[0].zw - (border.widths.zy - adjusted_widths.zy),
                      adjusted_widths.zy,
                      vec2(0.0),
                      vRadii1);
            set_edge_line(border.widths.zy,
                          corners.tr_outer,
                          vec2(-1.0, 1.0));
            style = int(border.style.y);
            edge_distance = vec4(p1.x - adjusted_widths.z,
                                 p0.y + adjusted_widths.y,
                                 p1.x - border.widths.z + adjusted_widths.z,
                                 p0.y + border.widths.y - adjusted_widths.y);
            break;
        }
        case 2: {
            p0 = corners.br_inner;
            p1 = corners.br_outer;
            vColor0 = border.colors[2];
            vColor1 = border.colors[3];
            vClipCenter = corners.br_outer - border.radii[1].xy;
            vClipSign = vec2(-1.0, -1.0);
            set_radii(border.radii[1].xy,
                      adjusted_widths.zw,
                      vec2(2.0 * border.widths.zw),
                      vRadii0);
            set_radii(border.radii[1].xy - (border.widths.zw - adjusted_widths.zw),
                      adjusted_widths.zw,
                      vec2(0.0),
                      vRadii1);
            set_edge_line(border.widths.zw,
                          corners.br_outer,
                          vec2(-1.0, -1.0));
            style = int(border.style.z);
            edge_distance = vec4(p1.x - adjusted_widths.z,
                                 p1.y - adjusted_widths.w,
                                 p1.x - border.widths.z + adjusted_widths.z,
                                 p1.y - border.widths.w + adjusted_widths.w);
            break;
        }
        case 3: {
            p0 = vec2(corners.bl_outer.x, corners.bl_inner.y);
            p1 = vec2(corners.bl_inner.x, corners.bl_outer.y);
            vColor0 = border.colors[3];
            vColor1 = border.colors[0];
            vClipCenter = corners.bl_outer + vec2(border.radii[1].z, -border.radii[1].w);
            vClipSign = vec2(1.0, -1.0);
            set_radii(border.radii[1].zw,
                      adjusted_widths.xw,
                      vec2(2.0 * border.widths.xw),
                      vRadii0);
            set_radii(border.radii[1].zw - (border.widths.xw - adjusted_widths.xw),
                      adjusted_widths.xw,
                      vec2(0.0),
                      vRadii1);
            set_edge_line(border.widths.xw,
                          corners.bl_outer,
                          vec2(1.0, -1.0));
            style = int(border.style.w);
            edge_distance = vec4(p0.x + adjusted_widths.x,
                                 p1.y - adjusted_widths.w,
                                 p0.x + border.widths.x - adjusted_widths.x,
                                 p1.y - border.widths.w + adjusted_widths.w);
            break;
        }
    }

    switch (int(style)) {
        case BORDER_STYLE_DOUBLE: {
            vEdgeDistance = edge_distance;
            break;
        }
        default: {
            vEdgeDistance = vec4(0.0);
            break;
        }
    }

    RectWithSize segment_rect;
    segment_rect.p0 = p0;
    segment_rect.size = p1 - p0;

#ifdef WR_FEATURE_TRANSFORM
    TransformVertexInfo vi = write_transform_vertex(segment_rect,
                                                    prim.local_clip_rect,
                                                    prim.z,
                                                    prim.layer,
                                                    prim.task,
                                                    prim.local_rect.p0);
    vLocalPos = vi.local_pos;
    vLocalRect = segment_rect;
#else
    VertexInfo vi = write_vertex(segment_rect,
                                 prim.local_clip_rect,
                                 prim.z,
                                 prim.layer,
                                 prim.task,
                                 prim.local_rect.p0);
    vLocalPos = vi.local_pos.xy;
#endif

    write_clip(vi.screen_pos, prim.clip_area);
}
