/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// This code is inspired by the premultiply code in Gecko's gfx/2d/Swizzle.cpp
pub fn premultiply(data: &mut [u32]) {
    let alpha_shift; let rb_shift; let rb_dst_shift; let g_dst_shift; let g_shift;
    if cfg!(target_endian = "little") {
        alpha_shift = 24;
        rb_shift = 0;
        rb_dst_shift = 8;
        g_shift = 0;
        g_dst_shift = 8;
    } else {
        // big endian
        alpha_shift = 0;
        rb_shift = 8;
        rb_dst_shift = 0;
        g_shift = 8;
        g_dst_shift = 8;
    }

    for pixel in data.iter_mut() {
        let a = (*pixel >> alpha_shift) & 0xff;
        // Isolate the R and B components
        let mut rb = (*pixel >> rb_shift) & 0xff00ff;
        // Approximate the multiply by alpha and divide by 255 which is essentially:
        // c = c*a + 255; c = (c + (c >> 8)) >> 8;
        // However, we omit the final >> 8 to fold it with the final shift into place
        // depending on desired output format.
        rb = rb * a + 0xff00ff;
        rb = (rb + ((rb >> 8) & 0xff00ff)) & 0xff00ff00;

        // Use same approximation as above, but G is shifted 8 bits left.
        // Alpha is left out and handled separately.
        let mut g = *pixel & (0xff00 << g_shift);
        g = g * a + (0xff00 << g_shift);
        g = (g + (g >> 8)) & (0xff0000 << g_shift);

        *pixel = (a << alpha_shift) | (rb >> rb_dst_shift) | (g >> g_dst_shift);
    }
}

pub fn unpremultiply(data: &mut [u32]) {
    // this could be much better but I'm lazy
    for pixel in data.iter_mut() {
        let p = (*pixel).to_le();
        let a = (p >> 24) & 0xff;
        let mut r = (p >> 16) & 0xff;
        let mut g = (p >> 8)  & 0xff;
        let mut b = (p >> 0)  & 0xff;

        r = r * 255 / a;
        g = g * 255 / a;
        b = b * 255 / a;

        let result = (a << 24) | (r << 16) | (g << 8) | b;
        *pixel = result.to_le();
    }
}

#[test]
fn it_works() {
    let mut f = [0x80ffffff, 0x8000ff00];
    premultiply(&mut f);
    assert!(f[0] == 0x80808080 && f[1] == 0x80008000);
    unpremultiply(&mut f);
    assert!(f[0] == 0x80ffffff && f[1] == 0x8000ff00);
}
