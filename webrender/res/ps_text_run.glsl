/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared

flat varying vec4 vColor;
varying vec3 vUv;
flat varying vec4 vUvBorder;
flat varying vec2 vMaskSwizzle;

#ifdef WR_FEATURE_GLYPH_TRANSFORM
varying vec4 vUvClip;
#endif

#ifdef WR_VERTEX_SHADER

#define VECS_PER_TEXT_RUN           2
#define GLYPHS_PER_GPU_BLOCK        2U

#ifdef WR_FEATURE_GLYPH_TRANSFORM
RectWithSize transform_rect(RectWithSize rect, mat2 transform) {
    vec2 center = transform * (rect.p0 + rect.size * 0.5);
    vec2 radius = mat2(abs(transform[0]), abs(transform[1])) * (rect.size * 0.5);
    return RectWithSize(center - radius, radius * 2.0);
}

bool rect_inside_rect(RectWithSize little, RectWithSize big) {
    return all(lessThanEqual(vec4(big.p0, little.p0 + little.size),
                             vec4(little.p0, big.p0 + big.size)));
}
#endif //WR_FEATURE_GLYPH_TRANSFORM

struct Glyph {
    vec2 offset;
};

Glyph fetch_glyph(int specific_prim_address,
                  int glyph_index) {
    // Two glyphs are packed in each texel in the GPU cache.
    int glyph_address = specific_prim_address +
                        VECS_PER_TEXT_RUN +
                        int(uint(glyph_index) / GLYPHS_PER_GPU_BLOCK);
    vec4 data = fetch_from_gpu_cache_1(glyph_address);
    // Select XY or ZW based on glyph index.
    // We use "!= 0" instead of "== 1" here in order to work around a driver
    // bug with equality comparisons on integers.
    vec2 glyph = mix(data.xy, data.zw,
                     bvec2(uint(glyph_index) % GLYPHS_PER_GPU_BLOCK != 0U));

    return Glyph(glyph);
}

struct GlyphResource {
    vec4 uv_rect;
    float layer;
    vec2 offset;
    float scale;
};

GlyphResource fetch_glyph_resource(int address) {
    vec4 data[2] = fetch_from_gpu_cache_2(address);
    return GlyphResource(data[0], data[1].x, data[1].yz, data[1].w);
}

struct TextRun {
    vec4 color;
    vec4 bg_color;
};

TextRun fetch_text_run(int address) {
    vec4 data[2] = fetch_from_gpu_cache_2(address);
    return TextRun(data[0], data[1]);
}

VertexInfo write_text_vertex(RectWithSize local_clip_rect,
                             float z,
#ifdef WR_FEATURE_GLYPH_TRANSFORM
                             mat2 glyph_transform,
#else
                             float glyph_scale,
#endif
                             Transform transform,
                             PictureTask task,
                             vec2 text_offset,
                             vec2 glyph_offset,
                             RectWithSize glyph_rect,
                             vec2 snap_bias) {
    // Glyph space refers to the pixel space used by glyph rasterization during frame
    // building. If a non-identity transform was used, WR_FEATURE_GLYPH_TRANSFORM will
    // be set. Otherwise, regardless of whether the raster space is LOCAL or SCREEN,
    // we ignored the transform during glyph rasterization, and need to snap just using
    // the device pixel scale and the raster scale.
#ifdef WR_FEATURE_GLYPH_TRANSFORM
    // Transform from glyph space back to local space.
    mat2 glyph_transform_inv = inverse(glyph_transform);

    // Glyph raster pixels include the impact of the transform. This path can only be
    // entered for 3d transforms that can be coerced into a 2d transform; they have no
    // perspective, and have a 2d inverse. This is a looser condition than axis aligned
    // transforms because it also allows 2d rotations.
    vec2 raster_glyph_offset = glyph_transform * glyph_offset;
    vec2 raster_snap_offset = floor(raster_glyph_offset + snap_bias) - raster_glyph_offset;
    vec2 local_snap_offset = glyph_transform_inv * raster_snap_offset;

    // We want to eliminate any subpixel translation in device space to ensure glyph
    // snapping is stable for equivalent glyph subpixel positions. Note that we must use
    // device pixels, and not glyph raster pixels for this purpose.
    vec2 device_text_pos = (transform.m * vec4(text_offset, 0.0, 1.0)).xy * task.device_pixel_scale;
    vec2 device_snap_offset = floor(device_text_pos + 0.5) - device_text_pos;

    // The glyph rect is in device space, so transform it back to local space.
    RectWithSize local_rect = transform_rect(glyph_rect, glyph_transform_inv);

    // Select the corner of the glyph's local space rect that we are processing.
    vec2 local_pos = local_rect.p0 + local_rect.size * aPosition.xy;

    // If the glyph's local rect would fit inside the local clip rect, then select a corner from
    // the device space glyph rect to reduce overdraw of clipped pixels in the fragment shader.
    // Otherwise, fall back to clamping the glyph's local rect to the local clip rect.
    if (rect_inside_rect(local_rect, local_clip_rect)) {
        local_pos = glyph_transform_inv * (glyph_rect.p0 + glyph_rect.size * aPosition.xy);
    }
#else
    // Glyph raster pixels do not include the impact of the transform. Instead it was
    // replaced with an identity transform during glyph rasterization. As such only the
    // impact of the raster scale (if in local space) and the device pixel scale (for both
    // local and screen space) are included.
    //
    // This implies one or more of the following conditions:
    // - The transform is an identity. In that case, setting WR_FEATURE_GLYPH_TRANSFORM
    //   should have the same output result as not. We just distingush which path to use
    //   based on the transform used during glyph rasterization. (Screen space).
    // - The transform contains an animation. We will imply local raster space in such
    //   cases to avoid constantly rerasterizing the glyphs.
    // - The transform has perspective or does not have a 2d inverse (Screen or local space).
    // - The transform's scale will result in result in very large rasterized glyphs and
    //   we clamped the size. This will imply local raster space.
    vec2 raster_glyph_offset = glyph_offset * glyph_scale;
    vec2 raster_snap_offset = floor(raster_glyph_offset + snap_bias) - raster_glyph_offset;
    vec2 local_snap_offset = raster_snap_offset / glyph_scale;

    // The transform may be animated, so we don't want to do any snapping here for the
    // text offset to avoid glyphs wiggling. The text offset should have been snapped
    // already for axis aligned transforms excluding any animations during frame building.
    vec2 device_snap_offset = vec2(0.0);

    // Select the corner of the glyph rect that we are processing.
    vec2 local_pos = glyph_rect.p0 + glyph_rect.size * aPosition.xy;
#endif

    // Clamp to the local clip rect.
    local_pos = clamp_rect(local_pos, local_clip_rect);

    // Map the clamped local space corner into device space.
    vec4 world_pos = transform.m * vec4(local_pos, 0.0, 1.0);
    vec2 device_pos = world_pos.xy * task.device_pixel_scale;
    vec4 snapped_world_pos = transform.m * vec4(local_pos + local_snap_offset, 0.0, 1.0);
    vec2 snapped_device_pos = snapped_world_pos.xy * task.device_pixel_scale + device_snap_offset * snapped_world_pos.w;

    // Apply offsets for the render task to get correct screen location.
    vec2 final_offset = -task.content_origin + task.common_data.task_rect.p0;

    gl_Position = uTransform * vec4(snapped_device_pos + final_offset * world_pos.w, z * world_pos.w, world_pos.w);

    VertexInfo vi = VertexInfo(
        local_pos,
        snapped_device_pos - device_pos,
        world_pos
    );

    return vi;
}

void main(void) {
    Instance instance = decode_instance_attributes();

    int glyph_index = instance.segment_index;
    int subpx_dir = (instance.flags >> 24) & 0xff;
    int color_mode = (instance.flags >> 16) & 0xff;

    PrimitiveHeader ph = fetch_prim_header(instance.prim_header_address);
    Transform transform = fetch_transform(ph.transform_id);
    ClipArea clip_area = fetch_clip_area(instance.clip_address);
    PictureTask task = fetch_picture_task(instance.picture_task_address);

    TextRun text = fetch_text_run(ph.specific_prim_address);
    vec2 text_offset = vec2(ph.user_data.xy) / 256.0;

    if (color_mode == COLOR_MODE_FROM_PASS) {
        color_mode = uMode;
    }

    Glyph glyph = fetch_glyph(ph.specific_prim_address, glyph_index);
    glyph.offset += ph.local_rect.p0 - text_offset;

    GlyphResource res = fetch_glyph_resource(instance.resource_address);

#ifdef WR_FEATURE_GLYPH_TRANSFORM
    // Transform from local space to glyph space.
    mat2 glyph_transform = mat2(transform.m) * task.device_pixel_scale;

    // Compute the glyph rect in glyph space.
    RectWithSize glyph_rect = RectWithSize(res.offset + glyph_transform * (text_offset + glyph.offset),
                                           res.uv_rect.zw - res.uv_rect.xy);
#else
    float raster_scale = float(ph.user_data.z) / 65535.0;

    // Scale from glyph space to local space.
    float glyph_scale_inv = res.scale / (raster_scale * task.device_pixel_scale);
    float glyph_scale = 1.0 / glyph_scale_inv;

    // Compute the glyph rect in local space.
    RectWithSize glyph_rect = RectWithSize(glyph_scale_inv * res.offset + text_offset + glyph.offset,
                                           glyph_scale_inv * (res.uv_rect.zw - res.uv_rect.xy));
#endif

    vec2 snap_bias;
    // In subpixel mode, the subpixel offset has already been
    // accounted for while rasterizing the glyph. However, we
    // must still round with a subpixel bias rather than rounding
    // to the nearest whole pixel, depending on subpixel direciton.
    switch (subpx_dir) {
        case SUBPX_DIR_NONE:
        default:
            snap_bias = vec2(0.5);
            break;
        case SUBPX_DIR_HORIZONTAL:
            // Glyphs positioned [-0.125, 0.125] get a
            // subpx position of zero. So include that
            // offset in the glyph position to ensure
            // we round to the correct whole position.
            snap_bias = vec2(0.125, 0.5);
            break;
        case SUBPX_DIR_VERTICAL:
            snap_bias = vec2(0.5, 0.125);
            break;
        case SUBPX_DIR_MIXED:
            snap_bias = vec2(0.125);
            break;
    }

    VertexInfo vi = write_text_vertex(ph.local_clip_rect,
                                      ph.z,
#ifdef WR_FEATURE_GLYPH_TRANSFORM
                                      glyph_transform,
#else
                                      glyph_scale,
#endif
                                      transform,
                                      task,
                                      text_offset,
                                      glyph.offset,
                                      glyph_rect,
                                      snap_bias);

#ifdef WR_FEATURE_GLYPH_TRANSFORM
    vec2 f = (glyph_transform * vi.local_pos - glyph_rect.p0) / glyph_rect.size;
    vUvClip = vec4(f, 1.0 - f);
#else
    vec2 f = (vi.local_pos - glyph_rect.p0) / glyph_rect.size;
#endif

    write_clip(vi.world_pos, vi.snap_offset, clip_area);

    switch (color_mode) {
        case COLOR_MODE_ALPHA:
        case COLOR_MODE_BITMAP:
            vMaskSwizzle = vec2(0.0, 1.0);
            vColor = text.color;
            break;
        case COLOR_MODE_SUBPX_BG_PASS2:
        case COLOR_MODE_SUBPX_DUAL_SOURCE:
            vMaskSwizzle = vec2(1.0, 0.0);
            vColor = text.color;
            break;
        case COLOR_MODE_SUBPX_CONST_COLOR:
        case COLOR_MODE_SUBPX_BG_PASS0:
        case COLOR_MODE_COLOR_BITMAP:
            vMaskSwizzle = vec2(1.0, 0.0);
            vColor = vec4(text.color.a);
            break;
        case COLOR_MODE_SUBPX_BG_PASS1:
            vMaskSwizzle = vec2(-1.0, 1.0);
            vColor = vec4(text.color.a) * text.bg_color;
            break;
        default:
            vMaskSwizzle = vec2(0.0);
            vColor = vec4(1.0);
    }

    vec2 texture_size = vec2(textureSize(sColor0, 0));
    vec2 st0 = res.uv_rect.xy / texture_size;
    vec2 st1 = res.uv_rect.zw / texture_size;

    vUv = vec3(mix(st0, st1, f), res.layer);
    vUvBorder = (res.uv_rect + vec4(0.5, 0.5, -0.5, -0.5)) / texture_size.xyxy;
}
#endif

#ifdef WR_FRAGMENT_SHADER
void main(void) {
    vec3 tc = vec3(clamp(vUv.xy, vUvBorder.xy, vUvBorder.zw), vUv.z);
    vec4 mask = texture(sColor0, tc);
    mask.rgb = mask.rgb * vMaskSwizzle.x + mask.aaa * vMaskSwizzle.y;

    float alpha = do_clip();
#ifdef WR_FEATURE_GLYPH_TRANSFORM
    alpha *= float(all(greaterThanEqual(vUvClip, vec4(0.0))));
#endif

#if defined(WR_FEATURE_DEBUG_OVERDRAW)
    oFragColor = WR_DEBUG_OVERDRAW_COLOR;
#elif defined(WR_FEATURE_DUAL_SOURCE_BLENDING)
    vec4 alpha_mask = mask * alpha;
    oFragColor = vColor * alpha_mask;
    oFragBlend = alpha_mask * vColor.a;
#else
    write_output(vColor * mask * alpha);
#endif
}
#endif
