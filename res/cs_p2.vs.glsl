#line 1

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

void main() {
    CompositeTile tile = tiles[gl_InstanceID];
    write_vertex(tile);

    write_prim(tile, 0, vUv0, vLayerValues.x);
    write_prim(tile, 1, vUv1, vLayerValues.y);
}
