#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

void main(void) {
    CacheClipInstance cci = fetch_clip_item(gl_InstanceID);
    Tile tile = fetch_tile(cci.render_task_index);
    Layer layer = fetch_layer(cci.layer_index);
    ImageMaskData mask = fetch_mask_data(cci.address);
    vec4 local_rect = mask.local_rect;

    TransformVertexInfo vi = write_transform_vertex(local_rect,
                                                    local_rect,
                                                    layer,
                                                    tile);
    vLocalRect = vi.clipped_local_rect;
    vPos = vi.local_pos;

    vClipMaskUv = vec3((vPos.xy / vPos.z - local_rect.xy) / local_rect.zw, 0.0);
    vec2 texture_size = textureSize(sMask, 0);
    vClipMaskUvRect = mask.uv_rect / texture_size.xyxy;
}
