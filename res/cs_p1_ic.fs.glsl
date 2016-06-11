#line 1

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

void main(void) {
    vec4 prim_colors[4];
    prim_colors[0] = texture(sLayer0, vUv0);
    prim_colors[1] = texture(sLayer1, vUv1);
    prim_colors[2] = texture(sLayer2, vUv2);
    prim_colors[3] = texture(sLayer3, vUv3);

    vec4 result = vec4(1, 1, 1, 1);
    vec4 layer_color = vec4(0, 0, 0, 0);

    layer_color = mix(layer_color, prim_colors[0], prim_colors[0].a);
    result = mix(result, layer_color, layer_color.a * vLayerValues.x);
    layer_color = mix(layer_color, vec4(0, 0, 0, 0), vec4(vLayerValues.x > 0.0));

    layer_color = mix(layer_color, prim_colors[1], prim_colors[1].a);
    result = mix(result, layer_color, layer_color.a * vLayerValues.y);
    layer_color = mix(layer_color, vec4(0, 0, 0, 0), vec4(vLayerValues.y > 0.0));

    layer_color = mix(layer_color, prim_colors[2], prim_colors[2].a);
    result = mix(result, layer_color, layer_color.a * vLayerValues.z);
    layer_color = mix(layer_color, vec4(0, 0, 0, 0), vec4(vLayerValues.z > 0.0));

    layer_color = mix(layer_color, prim_colors[3], prim_colors[3].a);
    result = mix(result, layer_color, layer_color.a * vLayerValues.w);
    layer_color = mix(layer_color, vec4(0, 0, 0, 0), vec4(vLayerValues.w > 0.0));

    oFragColor = result;
}
