#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

void main(void) {
    Primitive prim = load_primitive();
    RadialGradient gradient = fetch_radial_gradient(prim.prim_index);

    float radius_x = gradient.center_radius_x_ratio_xy.z;
    float xy_ratio = gradient.center_radius_x_ratio_xy.w;

    vInvRadius = 1.0 / radius_x;

    VertexInfo vi = write_vertex(prim.local_rect,
                                 prim.local_clip_rect,
                                 prim.z,
                                 prim.layer,
                                 prim.task);
    vPos = vi.local_pos;
    vPos.y *= xy_ratio;

    // Snap the center point to device pixel units.
    // I'm not sure this is entirely correct, but the
    // old render path does this, and it is needed to
    // make the angle gradient ref tests pass. It might
    // be better to fix this higher up in DL construction
    // and not snap here?
    vCenter = floor(0.5 + gradient.center_radius_x_ratio_xy.xy * uDevicePixelRatio) / uDevicePixelRatio;
    vCenter.y *= xy_ratio;

    // V coordinate of gradient row in lookup texture.
    vGradientIndex = float(prim.sub_index) * 2.0 + 0.5;

    // Whether to repeat the gradient instead of clamping.
    vGradientRepeat = float(int(gradient.extend_mode.y) == EXTEND_MODE_REPEAT);
}
