#line 1

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

void main() {
    CompositeTile tile = tiles[gl_InstanceID];

    vUv0 = mix(tile.src_rects[0].xy / 4096.0,
    		   (tile.src_rects[0].xy + tile.src_rects[0].zw) / 4096.0,
    		   aPosition.xy);

    vUv1 = mix(tile.src_rects[1].xy / 4096.0,
    		   (tile.src_rects[1].xy + tile.src_rects[1].zw) / 4096.0,
    		   aPosition.xy);

    vLayerValues0 = vec4(1, 1, 1, 1);

    write_vertex(tile);

    /*

    write_partial_prim(pos, tile.prim_indices[0].x, tile, vUv0, vPartialRects0.x, vLayerValues0.x);
    write_partial_prim(pos, tile.prim_indices[0].y, tile, vUv1, vPartialRects0.y, vLayerValues0.y);
    */
}
