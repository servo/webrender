#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define PST_INVALID      uint(0)
#define PST_TOP_LEFT     uint(1)
#define PST_TOP_RIGHT    uint(2)
#define PST_BOTTOM_LEFT  uint(3)
#define PST_BOTTOM_RIGHT uint(4)
#define PST_TOP          uint(5)
#define PST_LEFT         uint(6)
#define PST_BOTTOM       uint(7)
#define PST_RIGHT        uint(8)

// Border styles as defined in webrender_traits/types.rs
#define BORDER_STYLE_NONE         uint(0)
#define BORDER_STYLE_SOLID        uint(1)
#define BORDER_STYLE_DOUBLE       uint(2)
#define BORDER_STYLE_DOTTED       uint(3)
#define BORDER_STYLE_DASHED       uint(4)
#define BORDER_STYLE_HIDDEN       uint(5)
#define BORDER_STYLE_GROOVE       uint(6)
#define BORDER_STYLE_RIDGE        uint(7)
#define BORDER_STYLE_INSET        uint(8)
#define BORDER_STYLE_OUTSET       uint(9)

struct Layer {
    mat4 transform;
    mat4 inv_transform;
    ivec4 world_clip_rect;
    vec4 screen_vertices[4];
};

layout(std140) uniform Layers {
    Layer layers[WR_MAX_PRIM_LAYERS];
};

struct Tile {
    uvec4 actual_rect;
    uvec4 target_rect;
};

layout(std140) uniform Tiles {
    Tile tiles[WR_MAX_PRIM_TILES];
};

struct PrimitiveInfo {
    uvec4 layer_tile_part;
    vec4 local_clip_rect;
};

struct ClipCorner {
    vec4 rect;
    vec4 outer_inner_radius;
};

struct Clip {
    vec4 rect;
    ClipCorner top_left;
    ClipCorner top_right;
    ClipCorner bottom_left;
    ClipCorner bottom_right;
};

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

vec4 untransform(vec2 ref, vec3 n, vec3 a, mat4 inv_transform) {
    vec3 p = vec3(ref, -10000.0);
    vec3 d = vec3(0, 0, 1.0);

    float t;
    ray_plane(n, a, p, d, t);
    vec3 c = p + d * t;

    vec4 r = inv_transform * vec4(c, 1.0);
    return r;
}

vec3 get_layer_pos(vec2 pos, uint layer_index) {
    Layer layer = layers[layer_index];
    vec3 a = layer.screen_vertices[0].xyz / layer.screen_vertices[0].w;
    vec3 b = layer.screen_vertices[3].xyz / layer.screen_vertices[3].w;
    vec3 c = layer.screen_vertices[2].xyz / layer.screen_vertices[2].w;
    vec3 n = normalize(cross(b-a, c-a));
    vec4 local_pos = untransform(pos, n, a, layer.inv_transform);
    return local_pos.xyw;
}

#ifdef WR_FRAGMENT_SHADER
void do_clip(vec2 pos, vec4 clip_rect, float radius) {
    vec2 ref_tl = clip_rect.xy + vec2( radius,  radius);
    vec2 ref_tr = clip_rect.zy + vec2(-radius,  radius);
    vec2 ref_bl = clip_rect.xw + vec2( radius, -radius);
    vec2 ref_br = clip_rect.zw + vec2(-radius, -radius);

    float d_tl = distance(pos, ref_tl);
    float d_tr = distance(pos, ref_tr);
    float d_bl = distance(pos, ref_bl);
    float d_br = distance(pos, ref_br);

    bool out0 = pos.x < ref_tl.x && pos.y < ref_tl.y && d_tl > radius;
    bool out1 = pos.x > ref_tr.x && pos.y < ref_tr.y && d_tr > radius;
    bool out2 = pos.x < ref_bl.x && pos.y > ref_bl.y && d_bl > radius;
    bool out3 = pos.x > ref_br.x && pos.y > ref_br.y && d_br > radius;

    // TODO(gw): Alpha anti-aliasing based on edge distance!
    if (out0 || out1 || out2 || out3) {
        discard;
    }
}

bool point_in_rect(vec2 p, vec2 p0, vec2 p1) {
    return p.x >= p0.x &&
           p.y >= p0.y &&
           p.x <= p1.x &&
           p.y <= p1.y;
}
#endif
