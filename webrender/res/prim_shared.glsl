/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include rect

#define EXTEND_MODE_CLAMP  0
#define EXTEND_MODE_REPEAT 1

#define LINE_STYLE_SOLID        0
#define LINE_STYLE_DOTTED       1
#define LINE_STYLE_DASHED       2
#define LINE_STYLE_WAVY         3

#define SUBPX_DIR_NONE        0
#define SUBPX_DIR_HORIZONTAL  1
#define SUBPX_DIR_VERTICAL    2

#define RASTER_LOCAL            0
#define RASTER_SCREEN           1

#define EPSILON     0.0001

uniform sampler2DArray sCacheA8;
uniform sampler2DArray sCacheRGBA8;

// An A8 target for standalone tasks that is available to all passes.
uniform sampler2DArray sSharedCacheA8;

uniform sampler2D sGradients;

vec2 clamp_rect(vec2 pt, RectWithSize rect) {
    return clamp(pt, rect.p0, rect.p0 + rect.size);
}

float distance_to_line(vec2 p0, vec2 perp_dir, vec2 p) {
    vec2 dir_to_p0 = p0 - p;
    return dot(normalize(perp_dir), dir_to_p0);
}

// TODO: convert back to RectWithEndPoint if driver issues are resolved, if ever.
flat varying vec4 vClipMaskUvBounds;
varying vec3 vClipMaskUv;
flat varying vec4 vLocalBounds;

// TODO(gw): This is here temporarily while we have
//           both GPU store and cache. When the GPU
//           store code is removed, we can change the
//           PrimitiveInstance instance structure to
//           use 2x unsigned shorts as vertex attributes
//           instead of an int, and encode the UV directly
//           in the vertices.
ivec2 get_resource_cache_uv(int address) {
    return ivec2(address % WR_MAX_VERTEX_TEXTURE_WIDTH,
                 address / WR_MAX_VERTEX_TEXTURE_WIDTH);
}

uniform HIGHP_SAMPLER_FLOAT sampler2D sResourceCache;

vec4[2] fetch_from_resource_cache_2_direct(ivec2 address) {
    return vec4[2](
        TEXEL_FETCH(sResourceCache, address, 0, ivec2(0, 0)),
        TEXEL_FETCH(sResourceCache, address, 0, ivec2(1, 0))
    );
}

vec4[2] fetch_from_resource_cache_2(int address) {
    ivec2 uv = get_resource_cache_uv(address);
    return vec4[2](
        TEXEL_FETCH(sResourceCache, uv, 0, ivec2(0, 0)),
        TEXEL_FETCH(sResourceCache, uv, 0, ivec2(1, 0))
    );
}

#ifdef WR_VERTEX_SHADER

#define VECS_PER_CLIP_SCROLL_NODE   9
#define VECS_PER_LOCAL_CLIP_RECT    1
#define VECS_PER_RENDER_TASK        2
#define VECS_PER_PRIM_HEADER        2
#define VECS_PER_TEXT_RUN           3
#define VECS_PER_GRADIENT_STOP      2

uniform HIGHP_SAMPLER_FLOAT sampler2D sClipScrollNodes;
uniform HIGHP_SAMPLER_FLOAT sampler2D sLocalClipRects;
uniform HIGHP_SAMPLER_FLOAT sampler2D sRenderTasks;

// Instanced attributes
in ivec4 aData0;
in ivec4 aData1;

// get_fetch_uv is a macro to work around a macOS Intel driver parsing bug.
// TODO: convert back to a function once the driver issues are resolved, if ever.
// https://github.com/servo/webrender/pull/623
// https://github.com/servo/servo/issues/13953
// Do the division with unsigned ints because that's more efficient with D3D
#define get_fetch_uv(i, vpi)  ivec2(int(uint(vpi) * (uint(i) % uint(WR_MAX_VERTEX_TEXTURE_WIDTH/vpi))), int(uint(i) / uint(WR_MAX_VERTEX_TEXTURE_WIDTH/vpi)))


vec4[8] fetch_from_resource_cache_8(int address) {
    ivec2 uv = get_resource_cache_uv(address);
    return vec4[8](
        TEXEL_FETCH(sResourceCache, uv, 0, ivec2(0, 0)),
        TEXEL_FETCH(sResourceCache, uv, 0, ivec2(1, 0)),
        TEXEL_FETCH(sResourceCache, uv, 0, ivec2(2, 0)),
        TEXEL_FETCH(sResourceCache, uv, 0, ivec2(3, 0)),
        TEXEL_FETCH(sResourceCache, uv, 0, ivec2(4, 0)),
        TEXEL_FETCH(sResourceCache, uv, 0, ivec2(5, 0)),
        TEXEL_FETCH(sResourceCache, uv, 0, ivec2(6, 0)),
        TEXEL_FETCH(sResourceCache, uv, 0, ivec2(7, 0))
    );
}

vec4[3] fetch_from_resource_cache_3(int address) {
    ivec2 uv = get_resource_cache_uv(address);
    return vec4[3](
        TEXEL_FETCH(sResourceCache, uv, 0, ivec2(0, 0)),
        TEXEL_FETCH(sResourceCache, uv, 0, ivec2(1, 0)),
        TEXEL_FETCH(sResourceCache, uv, 0, ivec2(2, 0))
    );
}

vec4[3] fetch_from_resource_cache_3_direct(ivec2 address) {
    return vec4[3](
        TEXEL_FETCH(sResourceCache, address, 0, ivec2(0, 0)),
        TEXEL_FETCH(sResourceCache, address, 0, ivec2(1, 0)),
        TEXEL_FETCH(sResourceCache, address, 0, ivec2(2, 0))
    );
}

vec4[4] fetch_from_resource_cache_4_direct(ivec2 address) {
    return vec4[4](
        TEXEL_FETCH(sResourceCache, address, 0, ivec2(0, 0)),
        TEXEL_FETCH(sResourceCache, address, 0, ivec2(1, 0)),
        TEXEL_FETCH(sResourceCache, address, 0, ivec2(2, 0)),
        TEXEL_FETCH(sResourceCache, address, 0, ivec2(3, 0))
    );
}

vec4[4] fetch_from_resource_cache_4(int address) {
    ivec2 uv = get_resource_cache_uv(address);
    return vec4[4](
        TEXEL_FETCH(sResourceCache, uv, 0, ivec2(0, 0)),
        TEXEL_FETCH(sResourceCache, uv, 0, ivec2(1, 0)),
        TEXEL_FETCH(sResourceCache, uv, 0, ivec2(2, 0)),
        TEXEL_FETCH(sResourceCache, uv, 0, ivec2(3, 0))
    );
}

vec4 fetch_from_resource_cache_1_direct(ivec2 address) {
    return texelFetch(sResourceCache, address, 0);
}

vec4 fetch_from_resource_cache_1(int address) {
    ivec2 uv = get_resource_cache_uv(address);
    return texelFetch(sResourceCache, uv, 0);
}

struct ClipScrollNode {
    mat4 transform;
    mat4 inv_transform;
    bool is_axis_aligned;
};

ClipScrollNode fetch_clip_scroll_node(int index) {
    ClipScrollNode node;

    // Create a UV base coord for each 8 texels.
    // This is required because trying to use an offset
    // of more than 8 texels doesn't work on some versions
    // of OSX.
    ivec2 uv = get_fetch_uv(index, VECS_PER_CLIP_SCROLL_NODE);
    ivec2 uv0 = ivec2(uv.x + 0, uv.y);
    ivec2 uv1 = ivec2(uv.x + 8, uv.y);

    node.transform[0] = TEXEL_FETCH(sClipScrollNodes, uv0, 0, ivec2(0, 0));
    node.transform[1] = TEXEL_FETCH(sClipScrollNodes, uv0, 0, ivec2(1, 0));
    node.transform[2] = TEXEL_FETCH(sClipScrollNodes, uv0, 0, ivec2(2, 0));
    node.transform[3] = TEXEL_FETCH(sClipScrollNodes, uv0, 0, ivec2(3, 0));

    node.inv_transform[0] = TEXEL_FETCH(sClipScrollNodes, uv0, 0, ivec2(4, 0));
    node.inv_transform[1] = TEXEL_FETCH(sClipScrollNodes, uv0, 0, ivec2(5, 0));
    node.inv_transform[2] = TEXEL_FETCH(sClipScrollNodes, uv0, 0, ivec2(6, 0));
    node.inv_transform[3] = TEXEL_FETCH(sClipScrollNodes, uv0, 0, ivec2(7, 0));

    vec4 misc = TEXEL_FETCH(sClipScrollNodes, uv1, 0, ivec2(0, 0));
    node.is_axis_aligned = misc.x == 0.0;

    return node;
}

RectWithSize fetch_clip_chain_rect(int index) {
    ivec2 uv = get_fetch_uv(index, VECS_PER_LOCAL_CLIP_RECT);
    vec4 rect = TEXEL_FETCH(sLocalClipRects, uv, 0, ivec2(0, 0));
    return RectWithSize(rect.xy, rect.zw);
}

struct RenderTaskCommonData {
    RectWithSize task_rect;
    float texture_layer_index;
};

struct RenderTaskData {
    RenderTaskCommonData common_data;
    vec3 data1;
};

RenderTaskData fetch_render_task_data(int index) {
    ivec2 uv = get_fetch_uv(index, VECS_PER_RENDER_TASK);

    vec4 texel0 = TEXEL_FETCH(sRenderTasks, uv, 0, ivec2(0, 0));
    vec4 texel1 = TEXEL_FETCH(sRenderTasks, uv, 0, ivec2(1, 0));

    RectWithSize task_rect = RectWithSize(
        texel0.xy,
        texel0.zw
    );

    RenderTaskCommonData common_data = RenderTaskCommonData(
        task_rect,
        texel1.x
    );

    RenderTaskData data = RenderTaskData(
        common_data,
        texel1.yzw
    );

    return data;
}

RenderTaskCommonData fetch_render_task_common_data(int index) {
    ivec2 uv = get_fetch_uv(index, VECS_PER_RENDER_TASK);

    vec4 texel0 = TEXEL_FETCH(sRenderTasks, uv, 0, ivec2(0, 0));
    vec4 texel1 = TEXEL_FETCH(sRenderTasks, uv, 0, ivec2(1, 0));

    RectWithSize task_rect = RectWithSize(
        texel0.xy,
        texel0.zw
    );

    RenderTaskCommonData data = RenderTaskCommonData(
        task_rect,
        texel1.x
    );

    return data;
}

#define PIC_TYPE_IMAGE          1
#define PIC_TYPE_TEXT_SHADOW    2

/*
 The dynamic picture that this brush exists on. Right now, it
 contains minimal information. In the future, it will describe
 the transform mode of primitives on this picture, among other things.
 */
struct PictureTask {
    RenderTaskCommonData common_data;
    vec2 content_origin;
};

PictureTask fetch_picture_task(int address) {
    RenderTaskData task_data = fetch_render_task_data(address);

    PictureTask task = PictureTask(
        task_data.common_data,
        task_data.data1.xy
    );

    return task;
}

struct ClipArea {
    RenderTaskCommonData common_data;
    vec2 screen_origin;
    bool local_space;
};

ClipArea fetch_clip_area(int index) {
    ClipArea area;

    if (index == 0x7FFF) { //special sentinel task index
        area.common_data = RenderTaskCommonData(
            RectWithSize(vec2(0.0), vec2(0.0)),
            0.0
        );
        area.screen_origin = vec2(0.0);
        area.local_space = false;
    } else {
        RenderTaskData task_data = fetch_render_task_data(index);

        area.common_data = task_data.common_data;
        area.screen_origin = task_data.data1.xy;
        area.local_space = task_data.data1.z == 0.0;
    }

    return area;
}

struct Glyph {
    vec2 offset;
};

Glyph fetch_glyph(int specific_prim_address,
                  int glyph_index,
                  int subpx_dir) {
    // Two glyphs are packed in each texel in the GPU cache.
    int glyph_address = specific_prim_address +
                        VECS_PER_TEXT_RUN +
                        glyph_index / 2;
    vec4 data = fetch_from_resource_cache_1(glyph_address);
    // Select XY or ZW based on glyph index.
    // We use "!= 0" instead of "== 1" here in order to work around a driver
    // bug with equality comparisons on integers.
    vec2 glyph = mix(data.xy, data.zw, bvec2(glyph_index % 2 != 0));

    // In subpixel mode, the subpixel offset has already been
    // accounted for while rasterizing the glyph.
    switch (subpx_dir) {
        case SUBPX_DIR_NONE:
            break;
        case SUBPX_DIR_HORIZONTAL:
            // Glyphs positioned [-0.125, 0.125] get a
            // subpx position of zero. So include that
            // offset in the glyph position to ensure
            // we round to the correct whole position.
            glyph.x = floor(glyph.x + 0.125);
            break;
        case SUBPX_DIR_VERTICAL:
            glyph.y = floor(glyph.y + 0.125);
            break;
        default: break;
    }

    return Glyph(glyph);
}

struct PrimitiveInstance {
    int prim_address;
    int specific_prim_address;
    int render_task_index;
    int clip_task_index;
    int scroll_node_id;
    int clip_chain_rect_index;
    int z;
    int user_data0;
    int user_data1;
    int user_data2;
};

PrimitiveInstance fetch_prim_instance() {
    PrimitiveInstance pi;

    pi.prim_address = aData0.x;
    pi.specific_prim_address = pi.prim_address + VECS_PER_PRIM_HEADER;
    pi.render_task_index = aData0.y;
    pi.clip_task_index = aData0.z;
    pi.clip_chain_rect_index = aData0.w / 65536;
    pi.scroll_node_id = aData0.w % 65536;
    pi.z = aData1.x;
    pi.user_data0 = aData1.y;
    pi.user_data1 = aData1.z;
    pi.user_data2 = aData1.w;

    return pi;
}

struct CompositeInstance {
    int render_task_index;
    int src_task_index;
    int backdrop_task_index;
    int user_data0;
    int user_data1;
    float z;
    int user_data2;
    int user_data3;
};

CompositeInstance fetch_composite_instance() {
    CompositeInstance ci;

    ci.render_task_index = aData0.x;
    ci.src_task_index = aData0.y;
    ci.backdrop_task_index = aData0.z;
    ci.z = float(aData0.w);

    ci.user_data0 = aData1.x;
    ci.user_data1 = aData1.y;
    ci.user_data2 = aData1.z;
    ci.user_data3 = aData1.w;

    return ci;
}

struct Primitive {
    ClipScrollNode scroll_node;
    ClipArea clip_area;
    PictureTask task;
    RectWithSize local_rect;
    RectWithSize local_clip_rect;
    int specific_prim_address;
    int user_data0;
    int user_data1;
    int user_data2;
    float z;
};

struct PrimitiveGeometry {
    RectWithSize local_rect;
    RectWithSize local_clip_rect;
};

PrimitiveGeometry fetch_primitive_geometry(int address) {
    vec4 geom[2] = fetch_from_resource_cache_2(address);
    return PrimitiveGeometry(RectWithSize(geom[0].xy, geom[0].zw),
                             RectWithSize(geom[1].xy, geom[1].zw));
}

Primitive load_primitive() {
    PrimitiveInstance pi = fetch_prim_instance();

    Primitive prim;

    prim.scroll_node = fetch_clip_scroll_node(pi.scroll_node_id);
    prim.clip_area = fetch_clip_area(pi.clip_task_index);
    prim.task = fetch_picture_task(pi.render_task_index);

    RectWithSize clip_chain_rect = fetch_clip_chain_rect(pi.clip_chain_rect_index);

    PrimitiveGeometry geom = fetch_primitive_geometry(pi.prim_address);
    prim.local_rect = geom.local_rect;
    prim.local_clip_rect = intersect_rects(clip_chain_rect, geom.local_clip_rect);

    prim.specific_prim_address = pi.specific_prim_address;
    prim.user_data0 = pi.user_data0;
    prim.user_data1 = pi.user_data1;
    prim.user_data2 = pi.user_data2;
    prim.z = float(pi.z);

    return prim;
}

// Return the intersection of the plane (set up by "normal" and "point")
// with the ray (set up by "ray_origin" and "ray_dir"),
// writing the resulting scaler into "t".
bool ray_plane(vec3 normal, vec3 pt, vec3 ray_origin, vec3 ray_dir, out float t)
{
    float denom = dot(normal, ray_dir);
    if (abs(denom) > 1e-6) {
        vec3 d = pt - ray_origin;
        t = dot(d, normal) / denom;
        return t >= 0.0;
    }

    return false;
}

// Apply the inverse transform "inv_transform"
// to the reference point "ref" in CSS space,
// producing a local point on a ClipScrollNode plane,
// set by a base point "a" and a normal "n".
vec4 untransform(vec2 ref, vec3 n, vec3 a, mat4 inv_transform) {
    vec3 p = vec3(ref, -10000.0);
    vec3 d = vec3(0, 0, 1.0);

    float t = 0.0;
    // get an intersection of the ClipScrollNode plane with Z axis vector,
    // originated from the "ref" point
    ray_plane(n, a, p, d, t);
    float z = p.z + d.z * t; // Z of the visible point on the ClipScrollNode

    vec4 r = inv_transform * vec4(ref, z, 1.0);
    return r;
}

// Given a CSS space position, transform it back into the ClipScrollNode space.
vec4 get_node_pos(vec2 pos, ClipScrollNode node) {
    // get a point on the scroll node plane
    vec4 ah = node.transform * vec4(0.0, 0.0, 0.0, 1.0);
    vec3 a = ah.xyz / ah.w;

    // get the normal to the scroll node plane
    vec3 n = transpose(mat3(node.inv_transform)) * vec3(0.0, 0.0, 1.0);
    return untransform(pos, n, a, node.inv_transform);
}

// Compute a snapping offset in world space (adjusted to pixel ratio),
// given local position on the scroll_node and a snap rectangle.
vec2 compute_snap_offset(vec2 local_pos,
                         mat4 transform,
                         RectWithSize snap_rect) {
    // Ensure that the snap rect is at *least* one device pixel in size.
    // TODO(gw): It's not clear to me that this is "correct". Specifically,
    //           how should it interact with sub-pixel snap rects when there
    //           is a scroll_node transform with scale present? But it does fix
    //           the test cases we have in Servo that are failing without it
    //           and seem better than not having this at all.
    snap_rect.size = max(snap_rect.size, vec2(1.0 / uDevicePixelRatio));

    // Transform the snap corners to the world space.
    vec4 world_snap_p0 = transform * vec4(snap_rect.p0, 0.0, 1.0);
    vec4 world_snap_p1 = transform * vec4(snap_rect.p0 + snap_rect.size, 0.0, 1.0);
    // Snap bounds in world coordinates, adjusted for pixel ratio. XY = top left, ZW = bottom right
    vec4 world_snap = uDevicePixelRatio * vec4(world_snap_p0.xy, world_snap_p1.xy) /
                                          vec4(world_snap_p0.ww, world_snap_p1.ww);
    /// World offsets applied to the corners of the snap rectangle.
    vec4 snap_offsets = floor(world_snap + 0.5) - world_snap;

    /// Compute the position of this vertex inside the snap rectangle.
    vec2 normalized_snap_pos = (local_pos - snap_rect.p0) / snap_rect.size;
    /// Compute the actual world offset for this vertex needed to make it snap.
    return mix(snap_offsets.xy, snap_offsets.zw, normalized_snap_pos);
}

struct VertexInfo {
    vec2 local_pos;
    vec2 screen_pos;
    float w;
    vec2 snapped_device_pos;
};

VertexInfo write_vertex(RectWithSize instance_rect,
                        RectWithSize local_clip_rect,
                        float z,
                        ClipScrollNode scroll_node,
                        PictureTask task,
                        RectWithSize snap_rect) {

    // Select the corner of the local rect that we are processing.
    vec2 local_pos = instance_rect.p0 + instance_rect.size * aPosition.xy;

    // Clamp to the two local clip rects.
    vec2 clamped_local_pos = clamp_rect(local_pos, local_clip_rect);

    /// Compute the snapping offset.
    vec2 snap_offset = compute_snap_offset(
        clamped_local_pos,
        scroll_node.transform,
        snap_rect
    );

    // Transform the current vertex to world space.
    vec4 world_pos = scroll_node.transform * vec4(clamped_local_pos, 0.0, 1.0);

    // Convert the world positions to device pixel space.
    vec2 device_pos = world_pos.xy / world_pos.w * uDevicePixelRatio;

    // Apply offsets for the render task to get correct screen location.
    vec2 snapped_device_pos = device_pos + snap_offset;
    vec2 final_pos = snapped_device_pos -
                     task.content_origin +
                     task.common_data.task_rect.p0;

    gl_Position = uTransform * vec4(final_pos, z, 1.0);

    VertexInfo vi = VertexInfo(
        clamped_local_pos,
        device_pos,
        world_pos.w,
        snapped_device_pos
    );

    return vi;
}

float cross2(vec2 v0, vec2 v1) {
    return v0.x * v1.y - v0.y * v1.x;
}

// Return intersection of line (p0,p1) and line (p2,p3)
vec2 intersect_lines(vec2 p0, vec2 p1, vec2 p2, vec2 p3) {
    vec2 d0 = p0 - p1;
    vec2 d1 = p2 - p3;

    float s0 = cross2(p0, p1);
    float s1 = cross2(p2, p3);

    float d = cross2(d0, d1);
    float nx = s0 * d1.x - d0.x * s1;
    float ny = s0 * d1.y - d0.y * s1;

    return vec2(nx / d, ny / d);
}

VertexInfo write_transform_vertex(RectWithSize local_segment_rect,
                                  RectWithSize local_prim_rect,
                                  RectWithSize local_clip_rect,
                                  vec4 clip_edge_mask,
                                  float z,
                                  ClipScrollNode scroll_node,
                                  PictureTask task,
                                  bool do_perspective_interpolation) {
    // Calculate a clip rect from local_rect + local clip
    RectWithEndpoint clip_rect = to_rect_with_endpoint(local_clip_rect);
    RectWithEndpoint segment_rect = to_rect_with_endpoint(local_segment_rect);
    segment_rect.p0 = clamp(segment_rect.p0, clip_rect.p0, clip_rect.p1);
    segment_rect.p1 = clamp(segment_rect.p1, clip_rect.p0, clip_rect.p1);

    // Calculate a clip rect from local_rect + local clip
    RectWithEndpoint prim_rect = to_rect_with_endpoint(local_prim_rect);
    prim_rect.p0 = clamp(prim_rect.p0, clip_rect.p0, clip_rect.p1);
    prim_rect.p1 = clamp(prim_rect.p1, clip_rect.p0, clip_rect.p1);

    // As this is a transform shader, extrude by 2 (local space) pixels
    // in each direction. This gives enough space around the edge to
    // apply distance anti-aliasing. Technically, it:
    // (a) slightly over-estimates the number of required pixels in the simple case.
    // (b) might not provide enough edge in edge case perspective projections.
    // However, it's fast and simple. If / when we ever run into issues, we
    // can do some math on the projection matrix to work out a variable
    // amount to extrude.

    // Only extrude along edges where we are going to apply AA.
    float extrude_amount = 2.0;
    vec4 extrude_distance = vec4(extrude_amount) * clip_edge_mask;
    local_segment_rect.p0 -= extrude_distance.xy;
    local_segment_rect.size += extrude_distance.xy + extrude_distance.zw;

    // Select the corner of the local rect that we are processing.
    vec2 local_pos = local_segment_rect.p0 + local_segment_rect.size * aPosition.xy;

    // Transform the current vertex to the world cpace.
    vec4 world_pos = scroll_node.transform * vec4(local_pos, 0.0, 1.0);

    // Convert the world positions to device pixel space.
    vec2 device_pos = world_pos.xy / world_pos.w * uDevicePixelRatio;
    vec2 task_offset = task.common_data.task_rect.p0 - task.content_origin;

    // Force w = 1, if we don't want perspective interpolation (for
    // example, drawing a screen-space quad on an element with a
    // perspective transform).
    world_pos.w = mix(1.0, world_pos.w, do_perspective_interpolation);

    // We want the world space coords to be perspective divided by W.
    // We also want that to apply to any interpolators. However, we
    // want a constant Z across the primitive, since we're using it
    // for draw ordering - so scale by the W coord to ensure this.
    vec4 final_pos = vec4(device_pos + task_offset, z, 1.0) * world_pos.w;
    gl_Position = uTransform * final_pos;

    vLocalBounds = mix(
        vec4(prim_rect.p0, prim_rect.p1),
        vec4(segment_rect.p0, segment_rect.p1),
        clip_edge_mask
    );

    VertexInfo vi = VertexInfo(
        local_pos,
        device_pos,
        world_pos.w,
        device_pos
    );

    return vi;
}

VertexInfo write_transform_vertex_primitive(Primitive prim) {
    return write_transform_vertex(
        prim.local_rect,
        prim.local_rect,
        prim.local_clip_rect,
        vec4(1.0),
        prim.z,
        prim.scroll_node,
        prim.task,
        true
    );
}

struct GlyphResource {
    vec4 uv_rect;
    float layer;
    vec2 offset;
    float scale;
};

GlyphResource fetch_glyph_resource(int address) {
    vec4 data[2] = fetch_from_resource_cache_2(address);
    return GlyphResource(data[0], data[1].x, data[1].yz, data[1].w);
}

struct ImageResource {
    RectWithEndpoint uv_rect;
    float layer;
    vec3 user_data;
};

ImageResource fetch_image_resource(int address) {
    //Note: number of blocks has to match `renderer::BLOCKS_PER_UV_RECT`
    vec4 data[2] = fetch_from_resource_cache_2(address);
    RectWithEndpoint uv_rect = RectWithEndpoint(data[0].xy, data[0].zw);
    return ImageResource(uv_rect, data[1].x, data[1].yzw);
}

ImageResource fetch_image_resource_direct(ivec2 address) {
    vec4 data[2] = fetch_from_resource_cache_2_direct(address);
    RectWithEndpoint uv_rect = RectWithEndpoint(data[0].xy, data[0].zw);
    return ImageResource(uv_rect, data[1].x, data[1].yzw);
}

struct TextRun {
    vec4 color;
    vec4 bg_color;
    vec2 offset;
};

TextRun fetch_text_run(int address) {
    vec4 data[3] = fetch_from_resource_cache_3(address);
    return TextRun(data[0], data[1], data[2].xy);
}

struct Image {
    vec4 stretch_size_and_tile_spacing;  // Size of the actual image and amount of space between
                                         //     tiled instances of this image.
};

Image fetch_image(int address) {
    vec4 data = fetch_from_resource_cache_1(address);
    return Image(data);
}

void write_clip(vec2 global_pos, ClipArea area) {
    vec2 uv = global_pos +
              area.common_data.task_rect.p0 -
              area.screen_origin;
    vClipMaskUvBounds = vec4(
        area.common_data.task_rect.p0,
        area.common_data.task_rect.p0 + area.common_data.task_rect.size
    );
    vClipMaskUv = vec3(uv, area.common_data.texture_layer_index);
}
#endif //WR_VERTEX_SHADER

#ifdef WR_FRAGMENT_SHADER

/// Find the appropriate half range to apply the AA approximation over.
/// This range represents a coefficient to go from one CSS pixel to half a device pixel.
float compute_aa_range(vec2 position) {
    // The constant factor is chosen to compensate for the fact that length(fw) is equal
    // to sqrt(2) times the device pixel ratio in the typical case. 0.5/sqrt(2) = 0.35355.
    //
    // This coefficient is chosen to ensure that any sample 0.5 pixels or more inside of
    // the shape has no anti-aliasing applied to it (since pixels are sampled at their center,
    // such a pixel (axis aligned) is fully inside the border). We need this so that antialiased
    // curves properly connect with non-antialiased vertical or horizontal lines, among other things.
    //
    // Lines over a half-pixel away from the pixel center *can* intersect with the pixel square;
    // indeed, unless they are horizontal or vertical, they are guaranteed to. However, choosing
    // a nonzero area for such pixels causes noticeable artifacts at the junction between an anti-
    // aliased corner and a straight edge.
    //
    // We may want to adjust this constant in specific scenarios (for example keep the principled
    // value for straight edges where we want pixel-perfect equivalence with non antialiased lines
    // when axis aligned, while selecting a larger and smoother aa range on curves).
    return 0.35355 * length(fwidth(position));
}

/// Return the blending coefficient for distance antialiasing.
///
/// 0.0 means inside the shape, 1.0 means outside.
///
/// This cubic polynomial approximates the area of a 1x1 pixel square under a
/// line, given the signed Euclidean distance from the center of the square to
/// that line. Calculating the *exact* area would require taking into account
/// not only this distance but also the angle of the line. However, in
/// practice, this complexity is not required, as the area is roughly the same
/// regardless of the angle.
///
/// The coefficients of this polynomial were determined through least-squares
/// regression and are accurate to within 2.16% of the total area of the pixel
/// square 95% of the time, with a maximum error of 3.53%.
///
/// See the comments in `compute_aa_range()` for more information on the
/// cutoff values of -0.5 and 0.5.
float distance_aa(float aa_range, float signed_distance) {
    float dist = 0.5 * signed_distance / aa_range;
    if (dist <= -0.5 + EPSILON)
        return 1.0;
    if (dist >= 0.5 - EPSILON)
        return 0.0;
    return 0.5 + dist * (0.8431027 * dist * dist - 1.14453603);
}

float signed_distance_rect(vec2 pos, vec2 p0, vec2 p1) {
    vec2 d = max(p0 - pos, pos - p1);
    return length(max(vec2(0.0), d)) + min(0.0, max(d.x, d.y));
}

float init_transform_fs(vec2 local_pos) {
    // Get signed distance from local rect bounds.
    float d = signed_distance_rect(
        local_pos,
        vLocalBounds.xy,
        vLocalBounds.zw
    );

    // Find the appropriate distance to apply the AA smoothstep over.
    float aa_range = compute_aa_range(local_pos);

    // Only apply AA to fragments outside the signed distance field.
    return distance_aa(aa_range, d);
}

float do_clip() {
    // anything outside of the mask is considered transparent
    bvec4 inside = lessThanEqual(
        vec4(vClipMaskUvBounds.xy, vClipMaskUv.xy),
        vec4(vClipMaskUv.xy, vClipMaskUvBounds.zw));
    // check for the dummy bounds, which are given to the opaque objects
    return vClipMaskUvBounds.xy == vClipMaskUvBounds.zw ? 1.0:
        all(inside) ? texelFetch(sCacheA8, ivec3(vClipMaskUv), 0).r : 0.0;
}

#ifdef WR_FEATURE_DITHERING
vec4 dither(vec4 color) {
    const int matrix_mask = 7;

    ivec2 pos = ivec2(gl_FragCoord.xy) & ivec2(matrix_mask);
    float noise_normalized = (texelFetch(sDither, pos, 0).r * 255.0 + 0.5) / 64.0;
    float noise = (noise_normalized - 0.5) / 256.0; // scale down to the unit length

    return color + vec4(noise, noise, noise, 0);
}
#else
vec4 dither(vec4 color) {
    return color;
}
#endif //WR_FEATURE_DITHERING

vec4 sample_gradient(int address, float offset, float gradient_repeat) {
    // Modulo the offset if the gradient repeats.
    float x = mix(offset, fract(offset), gradient_repeat);

    // Calculate the color entry index to use for this offset:
    //     offsets < 0 use the first color entry, 0
    //     offsets from [0, 1) use the color entries in the range of [1, N-1)
    //     offsets >= 1 use the last color entry, N-1
    //     so transform the range [0, 1) -> [1, N-1)

    // TODO(gw): In the future we might consider making the size of the
    // LUT vary based on number / distribution of stops in the gradient.
    const int GRADIENT_ENTRIES = 128;
    x = 1.0 + x * float(GRADIENT_ENTRIES);

    // Calculate the texel to index into the gradient color entries:
    //     floor(x) is the gradient color entry index
    //     fract(x) is the linear filtering factor between start and end
    int lut_offset = 2 * int(floor(x));     // There is a [start, end] color per entry.

    // Ensure we don't fetch outside the valid range of the LUT.
    lut_offset = clamp(lut_offset, 0, 2 * (GRADIENT_ENTRIES + 1));

    // Fetch the start and end color.
    vec4 texels[2] = fetch_from_resource_cache_2(address + lut_offset);

    // Finally interpolate and apply dithering
    return dither(mix(texels[0], texels[1], fract(x)));
}

#endif //WR_FRAGMENT_SHADER
