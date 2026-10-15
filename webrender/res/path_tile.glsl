/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/// Rasterization of the path tiles produced by path_tiler.rs.

#include gpu_buffer

#define PATH_TILE_SIZE 16.0
// Edge coordinates have 15 steps per pixel (see EDGE_STEPS_PER_PIXEL in
// path_tiler.rs), and the unorm texels map 255 to 1.0.
#define PATH_EDGE_SCALE (255.0 / 15.0)
#define PATH_TILE_COORD_MASK 0x3FF

#define PATH_FILL_RULE_NONZERO 1
#define PATH_FILL_INVERTED 2

// The edges of all path tiles, one per texel. See GpuBufferBlockEdge.
uniform sampler2D sPathEdges;

#ifdef WR_VERTEX_SHADER

struct PathTile {
    // In pixels, relative to the origin of the render task.
    vec4 rect;
    // The range of the tile's edges in sPathEdges.
    ivec2 edges;
    int backdrop;
    int path_address;
};

// See PathTileInstance.
PathTile decode_path_tile(ivec4 data) {
    PathTile tile;

    vec2 offset = vec2(
        float((data.x >> 10) & PATH_TILE_COORD_MASK),
        float(data.x & PATH_TILE_COORD_MASK)
    );
    float extend_x = float((data.x >> 20) & PATH_TILE_COORD_MASK);
    tile.rect = vec4(
        offset.x,
        offset.y,
        offset.x + 1.0 + extend_x,
        offset.y + 1.0
    ) * PATH_TILE_SIZE;

    int edge_count = (data.z >> 16) & 0xFFFF;
    tile.edges = ivec2(data.y, data.y + edge_count);
    tile.backdrop = (data.z & 0xFFFF) - 128;
    tile.path_address = data.w;

    return tile;
}

struct PathInfo {
    int render_task_address;
    int fill_rule;
    float opacity;
};

// See PathInfo in path_tiler.rs.
PathInfo fetch_path_info(int address) {
    ivec4 data = fetch_from_gpu_buffer_1i(address);

    PathInfo path;
    path.render_task_address = data.x;
    path.fill_rule = data.y;
    path.opacity = float(data.z) / 65535.0;

    return path;
}

#endif

#ifdef WR_FRAGMENT_SHADER

float rasterize_edge_analytical(vec2 p0, vec2 p1) {
    // The overlap range on the y axis between the current row of pixels and the segment.
    // It can be a negative range (negative edge winding).
    float y0 = clamp(p0.y, 0.0, 1.0);
    float y1 = clamp(p1.y, 0.0, 1.0);

    if (y0 == y1) {
        return 0.0;
    }

    float inv_dy = 1.0 / (p1.y - p0.y);
    // The interpolation factors at the start and end of the intersection between the edge
    // and the row of pixels.
    float t0 = (y0 - p0.y) * inv_dy;
    float t1 = (y1 - p0.y) * inv_dy;
    // X positions at t0 and t1
    float x0 = p0.x * (1.0 - t0) + p1.x * t0;
    float x1 = p0.x * (1.0 - t1) + p1.x * t1;

    // Jitter to avoid NaN when dividing by xmin-xmax (for example vertical edges).
    float jitter = 1e-5;

    float xmin = min(min(x1, x0), 1.0) - jitter;
    float xmax = max(x1, x0);
    float b = min(xmax, 1.0);
    float c = max(b, 0.0);
    float d = max(xmin, 0.0);
    float area = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);

    return area * (y1 - y0);
}

float path_resolve_mask(float winding_number, int fill_rule) {
    float mask;
    if ((fill_rule & PATH_FILL_RULE_NONZERO) == 0) {
        // Even-odd.
        mask = 1.0 - abs(mod(abs(winding_number), 2.0) - 1.0);
    } else {
        mask = min(abs(winding_number), 1.0);
    }

    if ((fill_rule & PATH_FILL_INVERTED) != 0) {
        mask = 1.0 - mask;
    }

    return mask;
}

// The coverage of a pixel of a path tile.
//
// uv is the position of the pixel center relative to the tile's origin, in pixels.
float path_tile_coverage(
    vec2 uv,
    HIGHP_FS_ADDRESS ivec2 edges,
    int backdrop,
    int fill_rule
) {
    float winding_number = float(backdrop);

    // rasterize_edge_analytical works in coordinates local to the pixel, with
    // the pixel covering the unit square, so shift by half a pixel to get the
    // top-left corner of the pixel.
    vec2 pixel_offset = uv - vec2(0.5);

    // This isn't necessary but to be on the safe side and make sure we don't accidentally
    // hang from of some corrupted data, restrict the loop to a large-ish number of segments.
    HIGHP_FS_ADDRESS int end = min(edges.y, edges.x + 512);
    for (HIGHP_FS_ADDRESS int edge_idx = edges.x; edge_idx < end; edge_idx++) {
        ivec2 edge_uv = ivec2(
            int(uint(edge_idx) % WR_MAX_VERTEX_TEXTURE_WIDTH),
            int(uint(edge_idx) / WR_MAX_VERTEX_TEXTURE_WIDTH)
        );
        vec4 edge = texelFetch(sPathEdges, edge_uv, 0) * PATH_EDGE_SCALE;

        // Move to coordinates local to the current pixel.
        winding_number += rasterize_edge_analytical(
            edge.xy - pixel_offset,
            edge.zw - pixel_offset
        );
    }

    return path_resolve_mask(winding_number, fill_rule);
}

#endif
