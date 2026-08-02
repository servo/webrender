/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::RawtestHarness;
use crate::euclid::point2;
use webrender::api::*;
use webrender::api::units::*;
use webrender::Transaction;

struct SnapTestContext {
    root_spatial_id: SpatialId,
    test_size: FramebufferIntSize,
    font_size: f32,
    ahem_font_key: FontInstanceKey,
    variant: usize,
}

enum SnapTestExpectation {
    Rect {
        expected_color: ColorU,
        expected_rect: DeviceIntRect,
        expected_offset: i32,
    }
}

struct ScrollRequest {
    external_scroll_id: ExternalScrollId,
    amount: f32,
}

struct SnapTestResult {
    scrolls: Vec<ScrollRequest>,
    expected: SnapTestExpectation,
}

type SnapTestFunction = fn(&mut DisplayListBuilder, &mut SnapTestContext) -> SnapTestResult;

struct SnapTest {
    name: &'static str,
    f: SnapTestFunction,
    variations: usize,
}

struct SnapVariation {
    offset: f32,
    expected: i32,
}

const MAGENTA_RECT: SnapTest = SnapTest {
    name: "clear",
    f: dl_clear,
    variations: 1,
};

// Types of snap tests
const TESTS: &[SnapTest; 4] = &[
    // Rectangle, no transform/scroll
    SnapTest {
        name: "rect",
        f: dl_simple_rect,
        variations: SIMPLE_FRACTIONAL_VARIANTS.len(),
    },

    // Glyph, no transform/scroll
    SnapTest {
        name: "glyph",
        f: dl_simple_glyph,
        variations: SIMPLE_FRACTIONAL_VARIANTS.len(),
    },

    // Rect, APZ
    SnapTest {
        name: "scroll1",
        f: dl_scrolling1,
        variations: SCROLL_VARIANTS.len(),
    },

    // Rect, APZ + external scroll offset
    SnapTest {
        name: "rect-apz-ext-1",
        f: dl_scrolling_ext1,
        variations: EXTERNAL_SCROLL_VARIANTS.len(),
    }
];

// Variants we will run for each snap test with expected float offset and raster difference
const SIMPLE_FRACTIONAL_VARIANTS: [SnapVariation; 13] = [
    SnapVariation {
        offset: 0.0,
        expected: 0,
    },
    SnapVariation {
        offset: 0.1,
        expected: 0,
    },
    SnapVariation {
        offset: 0.25,
        expected: 0,
    },
    SnapVariation {
        offset: 0.33,
        expected: 0,
    },
    SnapVariation {
        offset: 0.49,
        expected: 0,
    },
    SnapVariation {
        offset: 0.5,
        expected: 1,
    },
    SnapVariation {
        offset: 0.51,
        expected: 1,
    },
    SnapVariation {
        offset: -0.1,
        expected: 0,
    },
    SnapVariation {
        offset: -0.25,
        expected: 0,
    },
    SnapVariation {
        offset: -0.33,
        expected: 0,
    },
    SnapVariation {
        offset: -0.49,
        expected: 0,
    },
    SnapVariation {
        offset: -0.5,
        expected: 0,
    },
    SnapVariation {
        offset: -0.51,
        expected: -1,
    },
];

struct ScrollVariation{
    apz_scroll: f32,
    prim_offset: f32,
    expected: i32,
}

const SCROLL_VARIANTS: [ScrollVariation; 3] = [
    ScrollVariation {
        apz_scroll: 0.0,
        prim_offset: 0.0,
        expected: 0,
    },
    ScrollVariation {
        apz_scroll: -1.0,
        prim_offset: 0.0,
        expected: 1,
    },
    ScrollVariation {
        apz_scroll: -1.5,
        prim_offset: 0.0,
        expected: 2,
    },
];

struct ExternalScrollVariation{
    external_offset: f32,
    apz_scroll: f32,
    prim_offset: f32,
    expected: i32,
}

const EXTERNAL_SCROLL_VARIANTS: [ExternalScrollVariation; 1] = [
    ExternalScrollVariation {
        external_offset: 100.0,
        apz_scroll: -101.0,
        prim_offset: -100.0,
        expected: 1,
    },
];

impl<'a> RawtestHarness<'a> {
    pub fn test_snapping(&mut self) {
        println!("\tsnapping test...");

        // Test size needs to be:
        // (a) as small as possible for performance
        // (b) a factor of 5 (ahem font baseline requirement)
        // (c) an even number (center placement of test render)
        let test_size = FramebufferIntSize::new(20, 20);
        let mut any_fails = false;

        // Load the ahem.css test font
        let font_bytes = include_bytes!("../../reftests/text/Ahem.ttf").into();
        let font_key = self.wrench.font_key_from_bytes(font_bytes, 0);
        let font_size = 0.5 * test_size.width as f32;
        let ahem_font_key = self.wrench.add_font_instance(
            font_key,
            font_size,
            FontInstanceFlags::empty(),
            Some(FontRenderMode::Alpha),
            SyntheticItalics::disabled(),
        );

        // Run each test
        for test in TESTS {
            for i in 0 .. test.variations {
                let mut ctx = SnapTestContext {
                    ahem_font_key,
                    font_size,
                    root_spatial_id: SpatialId::root_scroll_node(self.wrench.root_pipeline_id),
                    test_size,
                    variant: i,
                };

                any_fails = !self.run_snap_test(test, &mut ctx);

                // Each test clears to a magenta rect before running the next test. This
                // ensures that if WR's invalidation logic would skip rendering a test due
                // to detection that it's the same output, we will still render it to test
                // the pixel snapping is actually correct
                assert!(self.run_snap_test(&MAGENTA_RECT, &mut ctx));
            }
        }

        assert!(!any_fails);
    }

    fn run_snap_test(
        &mut self,
        test: &SnapTest,
        ctx: &mut SnapTestContext,
    ) -> bool {
        let mut builder = DisplayListBuilder::new(self.wrench.root_pipeline_id);
        builder.begin();

        let result = (test.f)(&mut builder, ctx);

        let window_size = self.window.get_inner_size();
        let window_rect = FramebufferIntRect::from_origin_and_size(
            point2(0, window_size.height - ctx.test_size.height),
            ctx.test_size,
        );

        let txn = Transaction::new();
        self.submit_dl(&mut Epoch(0), builder, txn);

        for scroll in result.scrolls {
            let mut txn = Transaction::new();
            txn.set_scroll_offsets(
                scroll.external_scroll_id,
                vec![SampledScrollOffset {
                    offset: LayoutVector2D::new(0.0, scroll.amount),
                    generation: APZScrollGeneration::default(),
                }],
            );
            txn.generate_frame(0, true, false, RenderReasons::TESTING);
            self.wrench.api.send_transaction(self.wrench.document_id, txn);

            self.render_and_get_pixels(window_rect);
        }

        let pixels = self.render_and_get_pixels(window_rect);

        let ok = validate_output(
            &pixels,
            result.expected,
            ctx.test_size,
        );

        if !ok {
            println!("FAIL {} [{}]", test.name, ctx.variant);

            // enable to save output as png for debugging
            // use crate::png;
            // png::save(
            //     format!("snap_test_{}.png", test.name),
            //     pixels.clone(),
            //     ctx.test_size.cast_unit(),
            //     png::SaveSettings {
            //         flip_vertical: true,
            //         try_crop: false,
            //     },
            // );

            // enable to log output to console for debugging
            // for y in 0 .. ctx.test_size.height {
            //     for x in 0 .. ctx.test_size.width {
            //         let i = ((ctx.test_size.height - y - 1) * ctx.test_size.width + x) as usize * 4;
            //         let r = pixels[i+0];
            //         let g = pixels[i+1];
            //         let b = pixels[i+2];
            //         let a = pixels[i+3];
            //         print!("[{:2x},{:2x},{:2x},{:2x}], ", r, g, b, a);
            //     }
            //     print!("\n");
            // }
        }

        ok
    }
}

fn validate_output(
    pixels: &[u8],
    expected: SnapTestExpectation,
    frame_buffer_size: FramebufferIntSize,
) -> bool {
    match expected {
        SnapTestExpectation::Rect { expected_color, expected_rect, expected_offset } => {
            let expected_rect = expected_rect.translate(
                DeviceIntVector2D::new(0, expected_offset)
            );

            for y in 0 .. frame_buffer_size.height {
                for x in 0 .. frame_buffer_size.width {
                    let i = ((frame_buffer_size.height - y - 1) * frame_buffer_size.width + x) as usize * 4;
                    let actual = ColorU::new(
                        pixels[i+0],
                        pixels[i+1],
                        pixels[i+2],
                        pixels[i+3],
                    );

                    let expected = if expected_rect.contains(DeviceIntPoint::new(x, y)) {
                        expected_color
                    } else {
                        ColorU::new(255, 255, 255, 255)
                    };

                    if expected != actual {
                        println!("FAILED at ({}, {}):", x, y);
                        println!("\tExpected [{:2x},{:2x},{:2x},{:2x}]",
                            expected.r,
                            expected.g,
                            expected.b,
                            expected.a,
                        );
                        println!("\tGot      [{:2x},{:2x},{:2x},{:2x}]",
                            actual.r,
                            actual.g,
                            actual.b,
                            actual.a,
                        );
                        return false;
                    }
                }
            }

            true
        }
    }
}

fn dl_clear(
    builder: &mut DisplayListBuilder,
    ctx: &mut SnapTestContext,
) -> SnapTestResult {
    let color = ColorF::new(1.0, 0.0, 1.0, 1.0);

    let bounds = ctx.test_size
        .to_f32()
        .cast_unit()
        .into();

    builder.push_rect(
        &CommonItemProperties {
            clip_rect: bounds,
            clip_chain_id: ClipChainId::INVALID,
            spatial_id: ctx.root_spatial_id,
            flags: PrimitiveFlags::default(),
        },
        bounds,
        color,
    );

    SnapTestResult {
        scrolls: vec![],
        expected: SnapTestExpectation::Rect {
            expected_color: color.into(),
            expected_rect: ctx.test_size.cast_unit().into(),
            expected_offset: 0,
        }
    }
}

// Draw a centered rect
fn dl_simple_rect(
    builder: &mut DisplayListBuilder,
    ctx: &mut SnapTestContext
) -> SnapTestResult {
    let color = ColorF::BLACK;
    let variant = &SIMPLE_FRACTIONAL_VARIANTS[ctx.variant];

    let prim_size = DeviceIntSize::new(
        ctx.test_size.width / 2,
        ctx.test_size.height / 2
    );

    let rect = DeviceIntRect::from_origin_and_size(
        DeviceIntPoint::new(
            (ctx.test_size.width - prim_size.width) / 2,
            (ctx.test_size.height - prim_size.height) / 2,
        ),
        prim_size,
    );

    let bounds = rect
        .to_f32()
        .cast_unit()
        .translate(
            LayoutVector2D::new(0.0, variant.offset)
        );

    builder.push_rect(
        &CommonItemProperties {
            clip_rect: bounds,
            clip_chain_id: ClipChainId::INVALID,
            spatial_id: ctx.root_spatial_id,
            flags: PrimitiveFlags::default(),
        },
        bounds,
        color,
    );

    SnapTestResult {
        scrolls: vec![],
        expected: SnapTestExpectation::Rect {
            expected_color: color.into(),
            expected_rect: rect,
            expected_offset: variant.expected,
        }
    }
}

// Draw a centered glyph with ahem.css font
fn dl_simple_glyph(
    builder: &mut DisplayListBuilder,
    ctx: &mut SnapTestContext,
) -> SnapTestResult {
    let color = ColorF::BLACK;
    let variant = &SIMPLE_FRACTIONAL_VARIANTS[ctx.variant];

    let prim_size = DeviceIntSize::new(
        ctx.test_size.width / 2,
        ctx.test_size.height / 2
    );

    let rect = DeviceIntRect::from_origin_and_size(
        DeviceIntPoint::new(
            (ctx.test_size.width - prim_size.width) / 2,
            (ctx.test_size.height - prim_size.height) / 2,
        ),
        prim_size,
    );

    let bounds = rect
        .to_f32()
        .cast_unit()
        .translate(
            LayoutVector2D::new(0.0, variant.offset)
    );

    builder.push_text(
        &CommonItemProperties {
            clip_rect: bounds,
            clip_chain_id: ClipChainId::INVALID,
            spatial_id: ctx.root_spatial_id,
            flags: PrimitiveFlags::default(),
        },
        bounds,
        &[
            GlyphInstance {
                index: 0x41,
                point: LayoutPoint::new(
                    bounds.min.x,
                    // ahem.css font has baseline at 0.8em
                    bounds.min.y + ctx.font_size * 0.8,
                ),
            }
        ],
        ctx.ahem_font_key,
        color,
        None,
    );

    SnapTestResult {
        scrolls: vec![],
        expected: SnapTestExpectation::Rect {
            expected_color: color.into(),
            expected_rect: rect,
            expected_offset: variant.expected,
        }
    }
}

fn dl_scrolling1(
    builder: &mut DisplayListBuilder,
    ctx: &mut SnapTestContext,
) -> SnapTestResult {
    let color = ColorF::BLACK;
    let external_scroll_id = ExternalScrollId(1, PipelineId::dummy());
    let variant = &SCROLL_VARIANTS[ctx.variant];

    let scroll_id = builder.define_scroll_frame(
        ctx.root_spatial_id,
        external_scroll_id,
        LayoutRect::from_size(LayoutSize::new(100.0, 1000.0)),
        LayoutRect::from_size(LayoutSize::new(100.0, 100.0)),
        LayoutVector2D::zero(),
        APZScrollGeneration::default(),
        HasScrollLinkedEffect::No,
    );

    let prim_size = DeviceIntSize::new(
        ctx.test_size.width / 2,
        ctx.test_size.height / 2
    );

    let rect = DeviceIntRect::from_origin_and_size(
        DeviceIntPoint::new(
            (ctx.test_size.width - prim_size.width) / 2,
            (ctx.test_size.height - prim_size.height) / 2,
        ),
        prim_size,
    );

    let bounds = rect
        .to_f32()
        .cast_unit()
        .translate(
            LayoutVector2D::new(0.0, variant.prim_offset)
        );

    builder.push_rect(
        &CommonItemProperties {
            clip_rect: bounds,
            clip_chain_id: ClipChainId::INVALID,
            spatial_id: scroll_id,
            flags: PrimitiveFlags::default(),
        },
        bounds,
        color,
    );

    SnapTestResult {
        scrolls: vec![
            ScrollRequest {
                external_scroll_id,
                amount: variant.apz_scroll,
            }
        ],
        expected: SnapTestExpectation::Rect {
            expected_color: color.into(),
            expected_rect: rect,
            expected_offset: variant.expected,
        }
    }
}

fn dl_scrolling_ext1(
    builder: &mut DisplayListBuilder,
    ctx: &mut SnapTestContext,
) -> SnapTestResult {
    let color = ColorF::BLACK;
    let external_scroll_id = ExternalScrollId(1, PipelineId::dummy());
    let variant = &EXTERNAL_SCROLL_VARIANTS[ctx.variant];

    let scroll_id = builder.define_scroll_frame(
        ctx.root_spatial_id,
        external_scroll_id,
        LayoutRect::from_size(LayoutSize::new(100.0, 1000.0)),
        LayoutRect::from_size(LayoutSize::new(100.0, 100.0)),
        LayoutVector2D::new(0.0, variant.external_offset),
        APZScrollGeneration::default(),
        HasScrollLinkedEffect::No,
    );

    let prim_size = DeviceIntSize::new(
        ctx.test_size.width / 2,
        ctx.test_size.height / 2
    );

    let rect = DeviceIntRect::from_origin_and_size(
        DeviceIntPoint::new(
            (ctx.test_size.width - prim_size.width) / 2,
            (ctx.test_size.height - prim_size.height) / 2,
        ),
        prim_size,
    );

    let bounds = rect
        .to_f32()
        .cast_unit()
        .translate(
            LayoutVector2D::new(0.0, variant.prim_offset)
        );

    builder.push_rect(
        &CommonItemProperties {
            clip_rect: bounds,
            clip_chain_id: ClipChainId::INVALID,
            spatial_id: scroll_id,
            flags: PrimitiveFlags::default(),
        },
        bounds,
        color,
    );

    SnapTestResult {
        scrolls: vec![
            ScrollRequest {
                external_scroll_id,
                amount: variant.apz_scroll,
            }
        ],
        expected: SnapTestExpectation::Rect {
            expected_color: color.into(),
            expected_rect: rect,
            expected_offset: variant.expected,
        }
    }
}
