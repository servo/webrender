#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#if defined(GL_ES)
    #if GL_ES == 1
        #ifdef GL_FRAGMENT_PRECISION_HIGH
        precision highp sampler2DArray;
        #else
        precision mediump sampler2DArray;
        #endif
    #endif
#endif

#define PST_TOP_LEFT     0
#define PST_TOP          1
#define PST_TOP_RIGHT    2
#define PST_RIGHT        3
#define PST_BOTTOM_RIGHT 4
#define PST_BOTTOM       5
#define PST_BOTTOM_LEFT  6
#define PST_LEFT         7

#define BORDER_LEFT      0
#define BORDER_TOP       1
#define BORDER_RIGHT     2
#define BORDER_BOTTOM    3

#define UV_NORMALIZED    uint(0)
#define UV_PIXEL         uint(1)

#define EXTEND_MODE_CLAMP  0
#define EXTEND_MODE_REPEAT 1

uniform sampler2DArray sCache;

flat varying vec4 vClipMaskUvBounds;
varying vec3 vClipMaskUv;

#ifdef WR_VERTEX_SHADER

#define VECS_PER_LAYER             13
#define VECS_PER_RENDER_TASK        3
#define VECS_PER_PRIM_GEOM          2

uniform sampler2D sLayers;
uniform sampler2D sRenderTasks;
uniform sampler2D sPrimGeometry;

uniform sampler2D sData16;
uniform sampler2D sData32;
uniform sampler2D sData64;
uniform sampler2D sData128;
uniform sampler2D sResourceRects;

// Instanced attributes
in int aGlobalPrimId;
in int aPrimitiveAddress;
in int aTaskIndex;
in int aClipTaskIndex;
in int aLayerIndex;
in int aElementIndex;
in ivec2 aUserData;
in int aZIndex;

// get_fetch_uv is a macro to work around a macOS Intel driver parsing bug.
// TODO: convert back to a function once the driver issues are resolved, if ever.
// https://github.com/servo/webrender/pull/623
// https://github.com/servo/servo/issues/13953
#define get_fetch_uv(i, vpi)  ivec2(vpi * (i % (WR_MAX_VERTEX_TEXTURE_WIDTH/vpi)), i / (WR_MAX_VERTEX_TEXTURE_WIDTH/vpi))

ivec2 get_fetch_uv_1(int index) {
    return get_fetch_uv(index, 1);
}

ivec2 get_fetch_uv_2(int index) {
    return get_fetch_uv(index, 2);
}

ivec2 get_fetch_uv_4(int index) {
    return get_fetch_uv(index, 4);
}

ivec2 get_fetch_uv_8(int index) {
    return get_fetch_uv(index, 8);
}

struct RectWithSize {
    vec2 p0;
    vec2 size;
};

struct RectWithEndpoint {
    vec2 p0;
    vec2 p1;
};

RectWithEndpoint to_rect_with_endpoint(RectWithSize rect) {
    RectWithEndpoint result;
    result.p0 = rect.p0;
    result.p1 = rect.p0 + rect.size;

    return result;
}

RectWithSize to_rect_with_size(RectWithEndpoint rect) {
    RectWithSize result;
    result.p0 = rect.p0;
    result.size = rect.p1 - rect.p0;

    return result;
}

vec2 clamp_rect(vec2 point, RectWithSize rect) {
    return clamp(point, rect.p0, rect.p0 + rect.size);
}

vec2 clamp_rect(vec2 point, RectWithEndpoint rect) {
    return clamp(point, rect.p0, rect.p1);
}

// Clamp 2 points at once.
vec4 clamp_rect(vec4 points, RectWithSize rect) {
    return clamp(points, rect.p0.xyxy, rect.p0.xyxy + rect.size.xyxy);
}

vec4 clamp_rect(vec4 points, RectWithEndpoint rect) {
    return clamp(points, rect.p0.xyxy, rect.p1.xyxy);
}

struct Layer {
    mat4 transform;
    mat4 inv_transform;
    RectWithSize local_clip_rect;
    vec4 screen_vertices[4];
};

Layer fetch_layer(int index) {
    Layer layer;

    // Create a UV base coord for each 8 texels.
    // This is required because trying to use an offset
    // of more than 8 texels doesn't work on some versions
    // of OSX.
    ivec2 uv = get_fetch_uv(index, VECS_PER_LAYER);
    ivec2 uv0 = ivec2(uv.x + 0, uv.y);
    ivec2 uv1 = ivec2(uv.x + 8, uv.y);

    layer.transform[0] = texelFetchOffset(sLayers, uv0, 0, ivec2(0, 0));
    layer.transform[1] = texelFetchOffset(sLayers, uv0, 0, ivec2(1, 0));
    layer.transform[2] = texelFetchOffset(sLayers, uv0, 0, ivec2(2, 0));
    layer.transform[3] = texelFetchOffset(sLayers, uv0, 0, ivec2(3, 0));

    layer.inv_transform[0] = texelFetchOffset(sLayers, uv0, 0, ivec2(4, 0));
    layer.inv_transform[1] = texelFetchOffset(sLayers, uv0, 0, ivec2(5, 0));
    layer.inv_transform[2] = texelFetchOffset(sLayers, uv0, 0, ivec2(6, 0));
    layer.inv_transform[3] = texelFetchOffset(sLayers, uv0, 0, ivec2(7, 0));

    vec4 clip_rect = texelFetchOffset(sLayers, uv1, 0, ivec2(0, 0));
    layer.local_clip_rect = RectWithSize(clip_rect.xy, clip_rect.zw);

    layer.screen_vertices[0] = texelFetchOffset(sLayers, uv1, 0, ivec2(1, 0));
    layer.screen_vertices[1] = texelFetchOffset(sLayers, uv1, 0, ivec2(2, 0));
    layer.screen_vertices[2] = texelFetchOffset(sLayers, uv1, 0, ivec2(3, 0));
    layer.screen_vertices[3] = texelFetchOffset(sLayers, uv1, 0, ivec2(4, 0));

    return layer;
}

struct RenderTaskData {
    vec4 data0;
    vec4 data1;
    vec4 data2;
};

RenderTaskData fetch_render_task(int index) {
    RenderTaskData task;

    ivec2 uv = get_fetch_uv(index, VECS_PER_RENDER_TASK);

    task.data0 = texelFetchOffset(sRenderTasks, uv, 0, ivec2(0, 0));
    task.data1 = texelFetchOffset(sRenderTasks, uv, 0, ivec2(1, 0));
    task.data2 = texelFetchOffset(sRenderTasks, uv, 0, ivec2(2, 0));

    return task;
}

struct AlphaBatchTask {
    vec2 screen_space_origin;
    vec2 render_target_origin;
    vec2 size;
    float render_target_layer_index;
};

AlphaBatchTask fetch_alpha_batch_task(int index) {
    RenderTaskData data = fetch_render_task(index);

    AlphaBatchTask task;
    task.render_target_origin = data.data0.xy;
    task.size = data.data0.zw;
    task.screen_space_origin = data.data1.xy;
    task.render_target_layer_index = data.data1.z;

    return task;
}

struct ClipArea {
    vec4 task_bounds;
    vec4 screen_origin_target_index;
    vec4 inner_rect;
};

ClipArea fetch_clip_area(int index) {
    ClipArea area;

    if (index == 0x7FFFFFFF) { //special sentinel task index
        area.task_bounds = vec4(0.0, 0.0, 0.0, 0.0);
        area.screen_origin_target_index = vec4(0.0, 0.0, 0.0, 0.0);
        area.inner_rect = vec4(0.0);
    } else {
        RenderTaskData task = fetch_render_task(index);
        area.task_bounds = task.data0;
        area.screen_origin_target_index = task.data1;
        area.inner_rect = task.data2;
    }

    return area;
}

struct Gradient {
    vec4 start_end_point;
    vec4 extend_mode;
};

Gradient fetch_gradient(int index) {
    Gradient gradient;

    ivec2 uv = get_fetch_uv_2(index);

    gradient.start_end_point = texelFetchOffset(sData32, uv, 0, ivec2(0, 0));
    gradient.extend_mode = texelFetchOffset(sData32, uv, 0, ivec2(1, 0));

    return gradient;
}

struct GradientStop {
    vec4 color;
    vec4 offset;
};

GradientStop fetch_gradient_stop(int index) {
    GradientStop stop;

    ivec2 uv = get_fetch_uv_2(index);

    stop.color = texelFetchOffset(sData32, uv, 0, ivec2(0, 0));
    stop.offset = texelFetchOffset(sData32, uv, 0, ivec2(1, 0));

    return stop;
}

struct RadialGradient {
    vec4 start_end_center;
    vec4 start_end_radius_extend_mode;
};

RadialGradient fetch_radial_gradient(int index) {
    RadialGradient gradient;

    ivec2 uv = get_fetch_uv_2(index);

    gradient.start_end_center = texelFetchOffset(sData32, uv, 0, ivec2(0, 0));
    gradient.start_end_radius_extend_mode = texelFetchOffset(sData32, uv, 0, ivec2(1, 0));

    return gradient;
}

struct Glyph {
    vec4 offset;
};

Glyph fetch_glyph(int index) {
    Glyph glyph;

    ivec2 uv = get_fetch_uv_1(index);

    glyph.offset = texelFetchOffset(sData16, uv, 0, ivec2(0, 0));

    return glyph;
}

RectWithSize fetch_instance_geometry(int index) {
    ivec2 uv = get_fetch_uv_1(index);

    vec4 rect = texelFetchOffset(sData16, uv, 0, ivec2(0, 0));

    return RectWithSize(rect.xy, rect.zw);
}

struct PrimitiveGeometry {
    RectWithSize local_rect;
    RectWithSize local_clip_rect;
};

PrimitiveGeometry fetch_prim_geometry(int index) {
    PrimitiveGeometry pg;

    ivec2 uv = get_fetch_uv(index, VECS_PER_PRIM_GEOM);

    vec4 local_rect = texelFetchOffset(sPrimGeometry, uv, 0, ivec2(0, 0));
    pg.local_rect = RectWithSize(local_rect.xy, local_rect.zw);
    vec4 local_clip_rect = texelFetchOffset(sPrimGeometry, uv, 0, ivec2(1, 0));
    pg.local_clip_rect = RectWithSize(local_clip_rect.xy, local_clip_rect.zw);

    return pg;
}

struct PrimitiveInstance {
    int global_prim_index;
    int specific_prim_index;
    int render_task_index;
    int clip_task_index;
    int layer_index;
    int sub_index;
    int z;
    ivec2 user_data;
};

PrimitiveInstance fetch_prim_instance() {
    PrimitiveInstance pi;

    pi.global_prim_index = aGlobalPrimId;
    pi.specific_prim_index = aPrimitiveAddress;
    pi.render_task_index = aTaskIndex;
    pi.clip_task_index = aClipTaskIndex;
    pi.layer_index = aLayerIndex;
    pi.sub_index = aElementIndex;
    pi.user_data = aUserData;
    pi.z = aZIndex;

    return pi;
}

struct CachePrimitiveInstance {
    int global_prim_index;
    int specific_prim_index;
    int render_task_index;
    int sub_index;
    ivec2 user_data;
};

CachePrimitiveInstance fetch_cache_instance() {
    CachePrimitiveInstance cpi;

    PrimitiveInstance pi = fetch_prim_instance();

    cpi.global_prim_index = pi.global_prim_index;
    cpi.specific_prim_index = pi.specific_prim_index;
    cpi.render_task_index = pi.render_task_index;
    cpi.sub_index = pi.sub_index;
    cpi.user_data = pi.user_data;

    return cpi;
}

struct Primitive {
    Layer layer;
    ClipArea clip_area;
    AlphaBatchTask task;
    RectWithSize local_rect;
    RectWithSize local_clip_rect;
    int prim_index;
    // when sending multiple primitives of the same type (e.g. border segments)
    // this index allows the vertex shader to recognize the difference
    int sub_index;
    ivec2 user_data;
    float z;
};

Primitive load_primitive_custom(PrimitiveInstance pi) {
    Primitive prim;

    prim.layer = fetch_layer(pi.layer_index);
    prim.clip_area = fetch_clip_area(pi.clip_task_index);
    prim.task = fetch_alpha_batch_task(pi.render_task_index);

    PrimitiveGeometry pg = fetch_prim_geometry(pi.global_prim_index);
    prim.local_rect = pg.local_rect;
    prim.local_clip_rect = pg.local_clip_rect;

    prim.prim_index = pi.specific_prim_index;
    prim.sub_index = pi.sub_index;
    prim.user_data = pi.user_data;
    prim.z = float(pi.z);

    return prim;
}

Primitive load_primitive() {
    PrimitiveInstance pi = fetch_prim_instance();

    return load_primitive_custom(pi);
}


// Return the intersection of the plane (set up by "normal" and "point")
// with the ray (set up by "ray_origin" and "ray_dir"),
// writing the resulting scaler into "t".
bool ray_plane(vec3 normal, vec3 point, vec3 ray_origin, vec3 ray_dir, out float t)
{
    float denom = dot(normal, ray_dir);
    if (denom > 1e-6) {
        vec3 d = point - ray_origin;
        t = dot(d, normal) / denom;
        return t >= 0.0;
    }

    return false;
}

// Apply the inverse transform "inv_transform"
// to the reference point "ref" in CSS space,
// producing a local point on a layer plane,
// set by a base point "a" and a normal "n".
vec4 untransform(vec2 ref, vec3 n, vec3 a, mat4 inv_transform) {
    vec3 p = vec3(ref, -10000.0);
    vec3 d = vec3(0, 0, 1.0);

    float t = 0.0;
    // get an intersection of the layer plane with Z axis vector,
    // originated from the "ref" point
    ray_plane(n, a, p, d, t);
    float z = p.z + d.z * t; // Z of the visible point on the layer

    vec4 r = inv_transform * vec4(ref, z, 1.0);
    return r;
}

// Given a CSS space position, transform it back into the layer space.
vec4 get_layer_pos(vec2 pos, Layer layer) {
    // get 3 of the layer corners in CSS space
    vec3 a = layer.screen_vertices[0].xyz / layer.screen_vertices[0].w;
    vec3 b = layer.screen_vertices[3].xyz / layer.screen_vertices[3].w;
    vec3 c = layer.screen_vertices[2].xyz / layer.screen_vertices[2].w;
    // get the normal to the layer plane
    vec3 n = normalize(cross(b-a, c-a));
    return untransform(pos, n, a, layer.inv_transform);
}

struct VertexInfo {
    RectWithEndpoint local_rect;
    vec2 local_pos;
    vec2 screen_pos;
};

VertexInfo write_vertex(RectWithSize instance_rect,
                        RectWithSize local_clip_rect,
                        float z,
                        Layer layer,
                        AlphaBatchTask task) {
    RectWithEndpoint local_rect = to_rect_with_endpoint(instance_rect);

    // Select the corner of the local rect that we are processing.
    vec2 local_pos = mix(local_rect.p0, local_rect.p1, aPosition.xy);

    // xy = top left corner of the local rect, zw = position of current vertex.
    vec4 local_p0_pos = vec4(local_rect.p0, local_pos);

    // Clamp to the two local clip rects.
    local_p0_pos = clamp_rect(local_p0_pos, local_clip_rect);
    local_p0_pos = clamp_rect(local_p0_pos, layer.local_clip_rect);

    // Transform the top corner and current vertex to world space.
    vec4 world_p0 = layer.transform * vec4(local_p0_pos.xy, 0.0, 1.0);
    world_p0.xyz /= world_p0.w;
    vec4 world_pos = layer.transform * vec4(local_p0_pos.zw, 0.0, 1.0);
    world_pos.xyz /= world_pos.w;

    // Convert the world positions to device pixel space. xy=top left corner. zw=current vertex.
    vec4 device_p0_pos = vec4(world_p0.xy, world_pos.xy) * uDevicePixelRatio;

    // Calculate the distance to snap the vertex by (snap top left corner).
    vec2 snap_delta = device_p0_pos.xy - floor(device_p0_pos.xy + 0.5);

    // Apply offsets for the render task to get correct screen location.
    vec2 final_pos = device_p0_pos.zw -
                     snap_delta -
                     task.screen_space_origin +
                     task.render_target_origin;

    gl_Position = uTransform * vec4(final_pos, z, 1.0);

    VertexInfo vi = VertexInfo(local_rect, local_p0_pos.zw, device_p0_pos.zw);
    return vi;
}

#ifdef WR_FEATURE_TRANSFORM

struct TransformVertexInfo {
    vec3 local_pos;
    vec2 screen_pos;
    vec4 clipped_local_rect;
};

TransformVertexInfo write_transform_vertex(RectWithSize instance_rect,
                                           RectWithSize local_clip_rect,
                                           float z,
                                           Layer layer,
                                           AlphaBatchTask task) {
    vec2 lp0_base = instance_rect.p0;
    vec2 lp1_base = instance_rect.p0 + instance_rect.size;

    vec2 lp0 = clamp_rect(clamp_rect(lp0_base, local_clip_rect),
                          layer.local_clip_rect);
    vec2 lp1 = clamp_rect(clamp_rect(lp1_base, local_clip_rect),
                          layer.local_clip_rect);

    vec4 clipped_local_rect = vec4(lp0, lp1 - lp0);

    vec2 p0 = lp0;
    vec2 p1 = vec2(lp1.x, lp0.y);
    vec2 p2 = vec2(lp0.x, lp1.y);
    vec2 p3 = lp1;

    vec4 t0 = layer.transform * vec4(p0, 0, 1);
    vec4 t1 = layer.transform * vec4(p1, 0, 1);
    vec4 t2 = layer.transform * vec4(p2, 0, 1);
    vec4 t3 = layer.transform * vec4(p3, 0, 1);

    vec2 tp0 = t0.xy / t0.w;
    vec2 tp1 = t1.xy / t1.w;
    vec2 tp2 = t2.xy / t2.w;
    vec2 tp3 = t3.xy / t3.w;

    // compute a CSS space aligned bounding box
    vec2 min_pos = uDevicePixelRatio * min(min(tp0.xy, tp1.xy), min(tp2.xy, tp3.xy));
    vec2 max_pos = uDevicePixelRatio * max(max(tp0.xy, tp1.xy), max(tp2.xy, tp3.xy));

    // compute the device space position of this vertex
    vec2 device_pos = mix(min_pos, max_pos, aPosition.xy);

    // compute the point position in side the layer, in CSS space
    vec4 layer_pos = get_layer_pos(device_pos / uDevicePixelRatio, layer);

    // apply the task offset
    vec2 final_pos = device_pos - task.screen_space_origin + task.render_target_origin;

    gl_Position = uTransform * vec4(final_pos, z, 1.0);

    return TransformVertexInfo(layer_pos.xyw, device_pos, clipped_local_rect);
}

#endif //WR_FEATURE_TRANSFORM

struct ResourceRect {
    vec4 uv_rect;
};

ResourceRect fetch_resource_rect(int index) {
    ResourceRect rect;

    ivec2 uv = get_fetch_uv_1(index);

    rect.uv_rect = texelFetchOffset(sResourceRects, uv, 0, ivec2(0, 0));

    return rect;
}

struct Rectangle {
    vec4 color;
};

Rectangle fetch_rectangle(int index) {
    Rectangle rect;

    ivec2 uv = get_fetch_uv_1(index);

    rect.color = texelFetchOffset(sData16, uv, 0, ivec2(0, 0));

    return rect;
}

struct TextRun {
    vec4 color;
};

TextRun fetch_text_run(int index) {
    TextRun text;

    ivec2 uv = get_fetch_uv_1(index);

    text.color = texelFetchOffset(sData16, uv, 0, ivec2(0, 0));

    return text;
}

struct Image {
    vec4 stretch_size_and_tile_spacing;  // Size of the actual image and amount of space between
                                         //     tiled instances of this image.
};

Image fetch_image(int index) {
    Image image;

    ivec2 uv = get_fetch_uv_1(index);

    image.stretch_size_and_tile_spacing = texelFetchOffset(sData16, uv, 0, ivec2(0, 0));

    return image;
}

// YUV color spaces
#define YUV_REC601 1
#define YUV_REC709 2

struct YuvImage {
    vec4 y_st_rect;
    vec4 u_st_rect;
    vec4 v_st_rect;
    vec2 size;
    int color_space;
};

YuvImage fetch_yuv_image(int index) {
    YuvImage image;

    ivec2 uv = get_fetch_uv_4(index);

    image.y_st_rect = texelFetchOffset(sData64, uv, 0, ivec2(0, 0));
    image.u_st_rect = texelFetchOffset(sData64, uv, 0, ivec2(1, 0));
    image.v_st_rect = texelFetchOffset(sData64, uv, 0, ivec2(2, 0));
    vec4 size_color_space = texelFetchOffset(sData64, uv, 0, ivec2(3, 0));
    image.size = size_color_space.xy;
    image.color_space = int(size_color_space.z);

    return image;
}

struct BoxShadow {
    vec4 src_rect;
    vec4 bs_rect;
    vec4 color;
    vec4 border_radius_edge_size_blur_radius_inverted;
};

BoxShadow fetch_boxshadow(int index) {
    BoxShadow bs;

    ivec2 uv = get_fetch_uv_4(index);

    bs.src_rect = texelFetchOffset(sData64, uv, 0, ivec2(0, 0));
    bs.bs_rect = texelFetchOffset(sData64, uv, 0, ivec2(1, 0));
    bs.color = texelFetchOffset(sData64, uv, 0, ivec2(2, 0));
    bs.border_radius_edge_size_blur_radius_inverted = texelFetchOffset(sData64, uv, 0, ivec2(3, 0));

    return bs;
}

void write_clip(vec2 global_pos, ClipArea area) {
    vec2 texture_size = vec2(textureSize(sCache, 0).xy);
    vec2 uv = global_pos + area.task_bounds.xy - area.screen_origin_target_index.xy;
    vClipMaskUvBounds = area.task_bounds / texture_size.xyxy;
    vClipMaskUv = vec3(uv / texture_size, area.screen_origin_target_index.z);
}
#endif //WR_VERTEX_SHADER

#ifdef WR_FRAGMENT_SHADER
float distance_from_rect(vec2 p, vec2 origin, vec2 size) {
    vec2 clamped = clamp(p, origin, origin + size);
    return distance(clamped, p);
}

vec2 init_transform_fs(vec3 local_pos, vec4 local_rect, out float fragment_alpha) {
    fragment_alpha = 1.0;
    vec2 pos = local_pos.xy / local_pos.z;

    float border_distance = distance_from_rect(pos, local_rect.xy, local_rect.zw);
    if (border_distance != 0.0) {
        float delta = length(fwidth(local_pos.xy));
        fragment_alpha = 1.0 - smoothstep(0.0, 1.0, border_distance / delta * 2.0);
    }

    return pos;
}

float do_clip() {
    // anything outside of the mask is considered transparent
    bvec4 inside = lessThanEqual(
        vec4(vClipMaskUvBounds.xy, vClipMaskUv.xy),
        vec4(vClipMaskUv.xy, vClipMaskUvBounds.zw));
    // check for the dummy bounds, which are given to the opaque objects
    return vClipMaskUvBounds.xy == vClipMaskUvBounds.zw ? 1.0:
        all(inside) ? textureLod(sCache, vClipMaskUv, 0.0).r : 0.0;
}
#endif //WR_FRAGMENT_SHADER
