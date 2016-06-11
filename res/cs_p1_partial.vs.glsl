#line 1

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

void main() {
    CompositeTile tile = tiles[gl_InstanceID];
    vec2 pos = write_vertex(tile);

    write_partial_prim(pos, tile.prim_indices[0].x, tile, vUv0, vPartialRects0.x, vLayerValues0.x);
}
