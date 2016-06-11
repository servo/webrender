#line 1

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

void main() {
    CompositeTile tile = tiles[gl_InstanceID];
    vec2 pos = write_vertex(tile);

    vUv0 = write_prim(pos, tile.prim_indices[0].x);
    vUv1 = write_prim(pos, tile.prim_indices[0].y);
    vUv2 = write_prim(pos, tile.prim_indices[0].z);
    vUv3 = write_prim(pos, tile.prim_indices[0].w);

    uint li0 = tile.layer_indices[0].x;
    uint li1 = tile.layer_indices[0].y;
    uint li2 = tile.layer_indices[0].z;
    uint li3 = tile.layer_indices[0].w;

    if (li0 == INVALID_LAYER_INDEX || li0 == li1) {
        vLayerValues.x = 0.0;
    } else {
        vLayerValues.x = layers[li0].blend_info.x;
    }

    if (li1 == INVALID_LAYER_INDEX || li1 == li2) {
        vLayerValues.y = 0.0;
    } else {
        vLayerValues.y = layers[li1].blend_info.x;
    }

    if (li2 == INVALID_LAYER_INDEX || li2 == li3) {
        vLayerValues.z = 0.0;
    } else {
        vLayerValues.z = layers[li2].blend_info.x;
    }

    if (li3 == INVALID_LAYER_INDEX) {
        vLayerValues.w = 0.0;
    } else {
        vLayerValues.w = layers[li3].blend_info.x;
    }
}
