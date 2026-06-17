/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include gpu_buffer

#define SEGMENT_TOP_LEFT        0
#define SEGMENT_TOP_RIGHT       1
#define SEGMENT_BOTTOM_RIGHT    2
#define SEGMENT_BOTTOM_LEFT     3
#define SEGMENT_LEFT            4
#define SEGMENT_TOP             5
#define SEGMENT_RIGHT           6
#define SEGMENT_BOTTOM          7

#ifdef WR_VERTEX_SHADER

PER_INSTANCE in vec2 aTaskOrigin;
PER_INSTANCE in int aFlags;
PER_INSTANCE in int aGpuDataAddress;
PER_INSTANCE in vec4 aClipParams1;
PER_INSTANCE in vec4 aClipParams2;

struct BorderInstanceGpuData {
    vec4 rect;
    vec4 color0;
    vec4 color1;
    vec2 widths;
    vec2 radii;
    float shape;
};

BorderInstanceGpuData fetch_gpu_data(int index) {
    BorderInstanceGpuData data;

    vec4 texels[5] = fetch_from_gpu_buffer_5f(index);
    data.rect = texels[0];
    data.color0 = texels[1];
    data.color1 = texels[2];
    data.widths = texels[3].xy;
    data.radii = texels[3].zw;
    data.shape = texels[4].x;

    return data;
}

#endif
