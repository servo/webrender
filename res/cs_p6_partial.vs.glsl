#line 1

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

void main() {
    CompositeTile tile = tiles[gl_InstanceID];
    vec2 pos = write_vertex(tile);

    write_partial_prim(pos, tile.prim_indices[0].x, tile, vUv0, vPartialRects0.x, vLayerValues0.x);
    write_partial_prim(pos, tile.prim_indices[0].y, tile, vUv1, vPartialRects0.y, vLayerValues0.y);
    write_partial_prim(pos, tile.prim_indices[0].z, tile, vUv2, vPartialRects0.z, vLayerValues0.z);
    write_partial_prim(pos, tile.prim_indices[0].w, tile, vUv3, vPartialRects0.w, vLayerValues0.w);
    write_partial_prim(pos, tile.prim_indices[1].x, tile, vUv4, vPartialRects1.x, vLayerValues1.x);
    write_partial_prim(pos, tile.prim_indices[1].y, tile, vUv5, vPartialRects1.y, vLayerValues1.y);
}
