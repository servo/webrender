#line 1

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

void main() {
    CompositeTile tile = tiles[gl_InstanceID];
    write_vertex(tile);

    write_prim(tile, 0, vUv0, vLayerValues0.x);
    write_prim(tile, 1, vUv1, vLayerValues0.y);
    write_prim(tile, 2, vUv2, vLayerValues0.z);
    write_prim(tile, 3, vUv3, vLayerValues0.w);
    write_prim(tile, 4, vUv4, vLayerValues1.x);
    write_prim(tile, 5, vUv5, vLayerValues1.y);
    write_prim(tile, 6, vUv6, vLayerValues1.z);
    write_prim(tile, 7, vUv7, vLayerValues1.w);
}
