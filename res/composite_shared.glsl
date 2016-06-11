#line 1

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define MAX_PRIMS_PER_COMPOSITE         (8)

#define INVALID_LAYER_INDEX             uint(0xffffffff)

uniform sampler2D sLayer0;
uniform sampler2D sLayer1;
uniform sampler2D sLayer2;
uniform sampler2D sLayer3;
uniform sampler2D sLayer4;
uniform sampler2D sLayer5;
uniform sampler2D sLayer6;
uniform sampler2D sLayer7;
uniform sampler2D sCache;

struct CompositeTile {
    ivec4 rect;
    ivec4 src_rects[MAX_PRIMS_PER_COMPOSITE];
    vec4 blend_info[MAX_PRIMS_PER_COMPOSITE/4];
};

layout(std140) uniform Tiles {
    CompositeTile tiles[WR_MAX_COMPOSITE_TILES];
};

#ifdef WR_VERTEX_SHADER

/*
uint pack_rect(ivec4 rect, ivec2 ref_point) {
    int x0 = max(0, rect.x - ref_point.x);
    int y0 = max(0, rect.y - ref_point.y);
    int x1 = min(rect.x + rect.z - ref_point.x, 255);       // should never hit with right preconditions in bsp tree!
    int y1 = min(rect.y + rect.w - ref_point.y, 255);

    uint x = uint(x0) <<  0;
    uint y = uint(y0) <<  8;
    uint z = uint(x1) << 16;
    uint w = uint(y1) << 24;
    return x | y | z | w;
}
*/

void write_prim(CompositeTile tile,
                int index,
                out vec2 uv,
                out float blend_info) {
    uv = mix(tile.src_rects[index].xy / 2048.0,
             (tile.src_rects[index].xy + tile.src_rects[index].zw) / 2048.0,
             aPosition.xy);
    blend_info = tile.blend_info[index/4][index % 4];
}

/*
void write_partial_prim(vec2 pos,
                        uint prim_index,
                        CompositeTile tile,
                        out vec2 uv,
                        out uint partial_rect,
                        out float blend_info) {
    Renderable ren = renderables[prim_index];
    vec4 prim_rect = ren.screen_rect;
    vec2 f = (pos - prim_rect.xy) / prim_rect.zw;
    uv = mix(ren.st_rect.xy, ren.st_rect.zw, f);

    partial_rect = pack_rect(ren.screen_rect, tile.rect.xy);
    blend_info = ren.local_offset_blend_info.z;
}
*/

void write_vertex(CompositeTile tile) {
    vec4 pos = vec4(mix(tile.rect.xy,
                        tile.rect.xy + tile.rect.zw,
                        aPosition.xy),
                    0.0,
                    1.0);

    //vTilePos = pos.xy - tile.rect.xy;

    gl_Position = uTransform * pos;
    //return pos.xy;
}

#endif

#ifdef WR_FRAGMENT_SHADER

/*
ivec4 unpack_rect(uint rect) {
    int x = int(rect & uint(0x000000ff));
    int y = int((rect & uint(0x0000ff00)) >> 8);
    int z = int((rect & uint(0x00ff0000)) >> 16);
    int w = int((rect & uint(0xff000000)) >> 24);
    return ivec4(x, y, z, w);
}

bool is_prim_valid(uint packed_rect) {
    ivec4 rect = unpack_rect(packed_rect);

    return vTilePos.x > rect.x &&
           vTilePos.x < rect.z &&
           vTilePos.y > rect.y &&
           vTilePos.y < rect.w;
}
*/

vec4 fetch_initial_color() {
    return vec4(1, 1, 1, 0);
}

#endif
