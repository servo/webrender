/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared

varying vec3 vUv;
flat varying vec4 vUvBounds;

#if defined WR_FEATURE_ALPHA
flat varying vec4 vColor;
#endif

#ifdef WR_VERTEX_SHADER
// Draw a cached primitive (e.g. a blurred text run) from the
// target cache to the framebuffer, applying tile clip boundaries.

void main(void) {
    Primitive prim = load_primitive();

    RectWithSize local_rect;

    RenderTaskData child_task = fetch_render_task(prim.user_data1);
    vUv.z = child_task.data1.x;

    vec2 uv0, uv1;

#if defined WR_FEATURE_COLOR
    vec2 texture_size = vec2(textureSize(sColor0, 0).xy);
    local_rect = prim.local_rect;
    uv0 = child_task.data0.xy;
    uv1 = child_task.data0.xy + child_task.data0.zw;
#else
    Picture pic = fetch_picture(prim.specific_prim_address);

    vec2 texture_size = vec2(textureSize(sColor1, 0).xy);
    vColor = pic.color;

    vec2 size_tl = 2.0 * pic.blur_radius + pic.slice_top_left;
    vec2 size_br = 2.0 * pic.blur_radius + pic.slice_bottom_right;

    vec2 p0 = prim.local_rect.p0;
    vec2 p3 = p0 + prim.local_rect.size;

    vec2 p1 = p0 + size_tl;
    vec2 p2 = p3 - size_br;

    vec2 image_uv0 = child_task.data0.xy;
    vec2 image_uv1 = image_uv0 + child_task.data0.zw;

    switch (prim.user_data0) {
        case 0: {
            local_rect = RectWithSize(p0, size_tl);
            uv0 = child_task.data0.xy;
            uv1 = uv0 + size_tl;
            break;
        }
        case 1: {
            vec2 segment_size = vec2(size_br.x, size_tl.y);
            local_rect = RectWithSize(vec2(p2.x, p0.y), segment_size);
            uv0 = vec2(image_uv1.x - segment_size.x, image_uv0.y);
            uv1 = uv0 + segment_size;
            break;
        }
        case 2: {
            local_rect = RectWithSize(p2, size_br);
            uv0 = image_uv1 - size_br;
            uv1 = uv0 + size_br;
            break;
        }
        case 3: {
            vec2 segment_size = vec2(size_tl.x, size_br.y);
            local_rect = RectWithSize(vec2(p0.x, p2.y), segment_size);
            uv0 = vec2(image_uv0.x, image_uv1.y - segment_size.y);
            uv1 = uv0 + segment_size;
            break;
        }

        case 4: {
            local_rect = RectWithSize(vec2(p1.x, p0.y), vec2(p2.x - p1.x, p1.y - p0.y));
            uv0 = vec2(image_uv0.x + size_tl.x, image_uv0.y);
            uv1 = vec2(uv0.x + 1.0, image_uv0.y + size_tl.y);
            break;
        }
        case 5: {
            local_rect = RectWithSize(vec2(p1.x, p2.y), vec2(p2.x - p1.x, p3.y - p2.y));
            uv0 = vec2(image_uv0.x + size_tl.x, image_uv1.y - size_br.y);
            uv1 = vec2(uv0.x + 1.0, image_uv1.y);
            break;
        }

        case 6: {
            local_rect = RectWithSize(vec2(p0.x, p1.y), vec2(p1.x - p0.x, p2.y - p1.y));
            uv0 = vec2(image_uv0.x, image_uv0.y + size_tl.y);
            uv1 = vec2(image_uv0.x + size_tl.x, uv0.y + 1.0);
            break;
        }
        case 7: {
            local_rect = RectWithSize(vec2(p2.x, p1.y), vec2(p3.x - p2.x, p2.y - p1.y));
            uv0 = vec2(image_uv1.x - size_br.x, image_uv0.y + size_tl.y);
            uv1 = vec2(image_uv1.x, uv0.y + 1.0);
            break;
        }

        case 8: {
            local_rect = RectWithSize(p1, p2 - p1);
            uv0 = image_uv0 + size_tl;
            uv1 = image_uv1 - size_br;
            break;
        }
    }
#endif

    VertexInfo vi = write_vertex(local_rect,
                                 prim.local_clip_rect,
                                 prim.z,
                                 prim.layer,
                                 prim.task,
                                 prim.local_rect);

    vec2 f = (vi.local_pos - local_rect.p0) / local_rect.size;

    vUv.xy = mix(uv0 / texture_size,
                 uv1 / texture_size,
                 f);
    vUvBounds = vec4(uv0 + vec2(0.5), uv1 - vec2(0.5)) / texture_size.xyxy;

    write_clip(vi.screen_pos, prim.clip_area);
}
#endif

#ifdef WR_FRAGMENT_SHADER
void main(void) {
    vec2 uv = clamp(vUv.xy, vUvBounds.xy, vUvBounds.zw);

#if defined WR_FEATURE_COLOR
    vec4 color = texture(sColor0, vec3(uv, vUv.z));
#else
    vec4 color = vColor * texture(sColor1, vec3(uv, vUv.z)).r;
#endif

    // Un-premultiply the color from sampling the gradient.
    if (color.a > 0.0) {
        color.rgb /= color.a;

        // Apply the clip mask
        color.a = min(color.a, do_clip());

        // Pre-multiply the result.
        color.rgb *= color.a;
    }

    oFragColor = color;
}
#endif
