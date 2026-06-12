/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include gpu_buffer

#define VECS_PER_IMAGE_RESOURCE     2

#ifdef WR_VERTEX_SHADER

#include rect

struct ImageSource {
    RectWithEndpoint uv_rect;
    vec4 user_data;
};

ImageSource fetch_image_source(int address) {
    //Note: number of blocks has to match `renderer::BLOCKS_PER_UV_RECT`
    vec4 data[2] = fetch_from_gpu_buffer_2f(address);
    RectWithEndpoint uv_rect = RectWithEndpoint(data[0].xy, data[0].zw);
    return ImageSource(uv_rect, data[1]);
}

ImageSource fetch_image_source_direct(ivec2 address) {
    vec4 data[2] = fetch_from_gpu_buffer_2f_direct(address);
    RectWithEndpoint uv_rect = RectWithEndpoint(data[0].xy, data[0].zw);
    return ImageSource(uv_rect, data[1]);
}

// Fetch optional extra data for a texture cache resource. This can contain
// a polygon defining a UV rect within the texture cache resource.
// Note: the polygon coordinates are in homogeneous space.
struct ImageSourceExtra {
    vec4 st_tl;
    vec4 st_tr;
    vec4 st_bl;
    vec4 st_br;
};

ImageSourceExtra fetch_image_source_extra(int address) {
    vec4 data[4] = fetch_from_gpu_buffer_4f(address + VECS_PER_IMAGE_RESOURCE);
    return ImageSourceExtra(
        data[0],
        data[1],
        data[2],
        data[3]
    );
}

#endif // WR_VERTEX_SHADER
