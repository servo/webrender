/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::NotifierEvent;
use crate::WindowWrapper;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use crate::wrench::{Wrench, WrenchThing};
use crate::yaml_frame_reader::YamlFrameReader;
use webrender::{PictureCacheDebugInfo, TileDebugInfo};
use webrender::api::units::*;
use webrender::api::BorderRadius;

#[derive(Debug, Clone, Copy, PartialEq)]
enum InvalidationOp {
    Equal,
    NotEqual,
}

#[derive(Debug)]
struct InvalidationTest {
    op: InvalidationOp,
    file1: PathBuf,
    file2: PathBuf,
    /// Build the second display list with a fresh `DisplayListBuilder`, the way
    /// a replaced content process would. Its interning slot numbering restarts,
    /// so the receiver has to drop what the first builder interned before
    /// applying anything (`DeltaAction::Reset`).
    new_builder: bool,
}

fn parse_manifest(path: &Path) -> Vec<InvalidationTest> {
    let file = File::open(path)
        .unwrap_or_else(|e| panic!("Failed to open manifest {}: {}", path.display(), e));
    let reader = BufReader::new(file);
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut tests = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let line = match line.find('#') {
            Some(pos) => &line[..pos],
            None => &line,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 3 || tokens.len() > 4 {
            panic!(
                "{}:{}: expected 'OP file1 file2 [new-builder]', got: {}",
                path.display(),
                line_num + 1,
                line,
            );
        }

        let new_builder = match tokens.get(3) {
            None => false,
            Some(&"new-builder") => true,
            Some(other) => panic!(
                "{}:{}: unknown option '{}', expected new-builder",
                path.display(),
                line_num + 1,
                other,
            ),
        };

        let op = match tokens[0] {
            "==" => InvalidationOp::Equal,
            "!=" => InvalidationOp::NotEqual,
            other => panic!(
                "{}:{}: unknown operator '{}', expected == or !=",
                path.display(),
                line_num + 1,
                other,
            ),
        };

        tests.push(InvalidationTest {
            op,
            file1: dir.join(tokens[1]),
            file2: dir.join(tokens[2]),
            new_builder,
        });
    }

    tests
}

pub struct TestHarness<'a> {
    wrench: &'a mut Wrench,
    window: &'a mut WindowWrapper,
    rx: &'a Receiver<NotifierEvent>,
}

struct RenderResult {
    pc_debug: PictureCacheDebugInfo,
    composite_needed: bool,
    /// The pixel read back at `probe`, if one was requested.
    probe_pixel: Option<[u8; 4]>,
}

// Convenience method to build a picture rect
fn pr(x: f32, y: f32, w: f32, h: f32) -> PictureRect {
    PictureRect::from_origin_and_size(
        PicturePoint::new(x, y),
        PictureSize::new(w, h),
    )
}

impl<'a> TestHarness<'a> {
    pub fn new(
        wrench: &'a mut Wrench,
        window: &'a mut WindowWrapper,
        rx: &'a Receiver<NotifierEvent>
    ) -> Self {
        TestHarness {
            wrench,
            window,
            rx,
        }
    }

    /// Main entry point for invalidation tests
    pub fn run(
        mut self,
    ) -> usize {
        // Run hardcoded tests
        self.test_basic();
        self.test_composite_nop();
        self.test_scroll_subpic();
        self.test_clip_promotion();
        self.test_redundant_scroll_root();
        self.test_rounded_rect_intersection();
        self.test_promotion_shapes();
        self.test_surface_sub_slice();
        self.test_backdrop_sampled_margin();
        self.test_backdrop_scroll_under_clip();
        self.test_backdrop_cross_slice_scroll();
        self.test_backdrop_cross_slice_scroll_under_clip();
        self.test_backdrop_cross_slice_rounded();
        self.test_backdrop_cross_slice_compositor_surfaces();
        self.test_backdrop_cross_slice_changes();
        self.test_backdrop_cross_slice_uncovered();
        self.test_backdrop_cross_slice_move_together();
        self.test_backdrop_own_slice_compositor_surfaces();

        // Run manifest-based tests
        let manifest_path = PathBuf::from("invalidation/invalidation.list");
        if manifest_path.exists() {
            self.run_list_tests(&manifest_path)
        } else {
            0
        }
    }

    fn has_any_dirty_tile(pc_debug: &PictureCacheDebugInfo) -> bool {
        for slice_info in pc_debug.slices.values() {
            for tiles in &slice_info.sub_slices {
                for tile_info in tiles.values() {
                    if matches!(tile_info, TileDebugInfo::Dirty(..)) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn run_list_tests(
        &mut self,
        manifest_path: &Path,
    ) -> usize {
        let tests = parse_manifest(manifest_path);
        let mut failures = 0;

        for test in &tests {
            let file1_str = test.file1.to_string_lossy();
            let file2_str = test.file2.to_string_lossy();

            // Render file1 (baseline)
            self.render_yaml_path(&test.file1);
            if test.new_builder {
                self.wrench.drop_dl_builders();
            }
            // Render file2 (the change)
            let results = self.render_yaml_path(&test.file2);

            let has_dirty = Self::has_any_dirty_tile(&results.pc_debug);

            let pass = match test.op {
                InvalidationOp::Equal => !has_dirty,
                InvalidationOp::NotEqual => has_dirty,
            };

            let op_str = match test.op {
                InvalidationOp::Equal => "==",
                InvalidationOp::NotEqual => "!=",
            };

            let opts = if test.new_builder { " new-builder" } else { "" };
            if pass {
                println!("PASS {} {} {}{}", op_str, file1_str, file2_str, opts);
            } else {
                println!("FAIL {} {} {}{}", op_str, file1_str, file2_str, opts);
                failures += 1;
            }
        }

        failures
    }

    /// Simple validation / proof of concept of invalidation testing
    fn test_basic(
        &mut self,
    ) {
        // Render basic.yaml, ensure that the valid/dirty rects are as expected
        let results = self.render_yaml("basic");
        let tile_info = results.pc_debug.slice(0).tile(0, 0).as_dirty();
        assert_eq!(
            tile_info.local_valid_rect,
            pr(100.0, 100.0, 500.0, 100.0),
        );
        assert_eq!(
            tile_info.local_dirty_rect,
            pr(100.0, 100.0, 500.0, 100.0),
        );

        // Render it again and ensure the tile was considered valid (no rasterization was done)
        let results = self.render_yaml("basic");
        assert_eq!(*results.pc_debug.slice(0).tile(0, 0), TileDebugInfo::Valid);
    }

    /// Ensure WR detects composites are needed for position changes within a single tile.
    fn test_composite_nop(
        &mut self,
    ) {
        // Render composite_nop_1.yaml, ensure that the valid/dirty rects are as expected
        let results = self.render_yaml("composite_nop_1");
        let tile_info = results.pc_debug.slice(0).tile(0, 0).as_dirty();
        assert_eq!(
            tile_info.local_valid_rect,
            pr(100.0, 100.0, 100.0, 100.0),
        );
        assert_eq!(
            tile_info.local_dirty_rect,
            pr(100.0, 100.0, 100.0, 100.0),
        );

        // Render composite_nop_2.yaml, ensure that the valid/dirty rects are as expected
        let results = self.render_yaml("composite_nop_2");
        let tile_info = results.pc_debug.slice(0).tile(0, 0).as_dirty();
        assert_eq!(
            tile_info.local_valid_rect,
            pr(100.0, 120.0, 100.0, 100.0),
        );
        assert_eq!(
            tile_info.local_dirty_rect,
            pr(100.0, 120.0, 100.0, 100.0),
        );

        // Main part of this test - ensure WR detects a composite is required in this case
        assert!(results.composite_needed);
    }

    /// Ensure that tile cache pictures are not invalidated upon scrolling
    fn test_scroll_subpic(
        &mut self,
    ) {
        // First frame at scroll-offset 0
        let results = self.render_yaml("scroll_subpic_1");

        // Ensure we actually rendered something
        assert!(
            matches!(results.pc_debug.slice(0).tile(0, 0), TileDebugInfo::Dirty(..)),
            "Ensure the first test frame actually rendered something",
        );

        // Second frame just scrolls to scroll-offset 50
        let results = self.render_yaml("scroll_subpic_2");

        // Ensure the cache tile was not invalidated
        assert!(
            results.pc_debug.slice(0).tile(0, 0).is_valid(),
            "Ensure the cache tile was not invalidated after scrolling",
        );
    }

    /// Ensure that a root-level stacking context with a rounded-rect clip
    /// allows tile cache barriers to fire, producing multiple slices.
    fn test_clip_promotion(&mut self) {
        let results = self.render_yaml("clip_promotion");

        let slices = results.pc_debug.slices.len();
        assert!(slices > 1, "Expected multiple slices");
    }

    /// Ensure that a scroll frame which is only a scroll root by way of the
    /// outermost-scroll-root fallback (no scrollable range, or below the
    /// minimum scroll root size) does not start a picture cache slice, while
    /// a real scroll root still does.
    fn test_redundant_scroll_root(&mut self) {
        let results = self.render_yaml("real_scroll_root");
        assert!(
            results.pc_debug.slices.len() > 1,
            "Expected a real scroll root to get its own slice",
        );

        for name in ["redundant_scroll_root_small", "redundant_scroll_root_zero_range"] {
            let results = self.render_yaml(name);
            assert_eq!(
                results.pc_debug.slices.len(), 1,
                "Expected a single slice for {}", name,
            );
        }
    }

    /// Ensure that two rounded-rect clips in the shared clip chain are combined
    /// into a single compositor clip with the correct intersected radii.
    fn test_rounded_rect_intersection(
        &mut self,
    ) {
        let results = self.render_yaml("rounded_rect_intersect");

        let slice = results.pc_debug.slice(0);

        // The two clips (outer: 400x400 r=20, inner: 400x350 r=20 on the top
        // corners only, both at origin) should be combined into a single compositor
        // clip on the tile cache slice. The combined clip rect is the intersection
        // (400x350), with radius 20 on the top corners and none on the bottom.
        let clip = slice.compositor_clip.as_ref().expect(
            "Expected a compositor clip on the tile cache slice after combining two rounded rects"
        );

        let expected_rect = DeviceRect::from_origin_and_size(
            DevicePoint::new(0.0, 0.0),
            DeviceSize::new(400.0, 350.0),
        );
        assert_eq!(clip.rect, expected_rect, "Combined clip rect");

        let expected_radius = BorderRadius {
            top_left: LayoutSize::new(20.0, 20.0),
            top_right: LayoutSize::new(20.0, 20.0),
            bottom_left: LayoutSize::zero(),
            bottom_right: LayoutSize::zero(),
            shape_top_left: 1.0,
            shape_top_right: 1.0,
            shape_bottom_left: 1.0,
            shape_bottom_right: 1.0,
        };
        assert_eq!(clip.radius, expected_radius, "Combined clip radii");

        // Combining these two clips would give radius 20 on the top corners and 15
        // on the bottom. Native compositors can only apply a single radius
        // (bug 2079096), so only one of the clips may become the compositor clip.
        let results = self.render_yaml("rounded_rect_intersect_mixed");
        let clip = results.pc_debug.slice(0).compositor_clip.as_ref().expect(
            "Expected one of the rounded rects to be the compositor clip"
        );
        assert_eq!(clip.radius, BorderRadius::uniform(clip.radius.top_left.width), "Uniform clip radii");
    }

    /// Check which rounded-rect clip shapes are accepted as compositor clips.
    ///
    /// The reftests in reftests/compositor/ check that both paths produce the
    /// same pixels, but they cannot tell which path ran. This pins that: only
    /// shapes the compositing shader can represent (round corner shapes, circular
    /// radii, radii fitting in their quadrant, clip mode, root coordinate system)
    /// may become a compositor clip. Anything else must fall back to the regular
    /// per-primitive clip path, where it is still applied correctly.
    fn test_promotion_shapes(&mut self) {
        // (reftest yaml, whether the slice should end up with a compositor clip)
        let cases = [
            ("rounded-clip", true),
            ("single-radius-zero-corners", true),
            ("radius-clamped", true),
            ("two-clips-combined", true),
            ("rect-and-rounded", true),
            ("tile-boundary", true),
            ("larger-than-tile", true),
            // Not representable by the compositing shader: elliptical radii and
            // non-round corner shapes are rejected by can_use_fast_path_in.
            ("elliptical-radius", false),
            ("corner-shape", false),
            // Representable by the compositing shader, but native compositors such
            // as CoreAnimation can only apply a single radius (bug 2079096).
            ("per-corner-radius", false),
            // A scroll frame is an axis-aligned translation, so it stays in the
            // root coordinate system and the clip is still promoted.
            ("scrolled-clip", true),
            // A rotation does leave the root coordinate system, so this one is
            // not promotable. It has no reftest listing: the test side rasterizes
            // into an axis-aligned surface and resamples it while the reference
            // rasterizes the clip rotated, which differs legitimately.
            ("rotated-clip", false),
            // Not at the root of the stacking context stack.
            ("nested-clip", false),
            // Clipped out entirely, so there is nothing to apply. Note this one
            // depends on the window being smaller than the clip's offset.
            ("clipped-out", false),
        ];

        for (name, expect_clip) in cases {
            let path = PathBuf::from(format!("reftests/compositor/{}.yaml", name));
            let results = self.render_yaml_path(&path);

            let has_clip = results
                .pc_debug
                .slices
                .values()
                .any(|slice| slice.compositor_clip.is_some());

            assert_eq!(
                has_clip, expect_clip,
                "{}: expected compositor clip on a slice: {}, got: {}",
                name, expect_clip, has_clip,
            );
        }
    }

    /// Ensure a change inside an off-screen surface invalidates the tiles the
    /// surface's picture is drawn into, when a compositor surface splits the
    /// slice into sub-slices. The nested drop-shadow inflates the outer
    /// picture's coverage into the compositor surface's area, which none of
    /// its children reach on their own (bug 2077958).
    fn test_surface_sub_slice(&mut self) {
        self.render_yaml("surface_sub_slice_1");
        let results = self.render_yaml("surface_sub_slice_2");
        let slice = results.pc_debug.slice(0);

        assert!(
            slice.sub_slices.len() > 1,
            "Expected the compositor surface to split the slice into sub-slices",
        );

        let offset = webrender::TileOffset::new(0, 0);
        let states: Vec<_> = slice.sub_slices
            .iter()
            .filter_map(|tiles| tiles.get(&offset))
            .collect();

        assert!(
            states.iter().any(|state| matches!(state, TileDebugInfo::Dirty(..))),
            "Expected the changed content to dirty a tile: {:?}",
            states,
        );
        assert!(
            !states.iter().any(|state| state.is_valid()),
            "A sub-slice tile drawing the picture stayed valid: {:?}",
            states,
        );
    }

    /// A backdrop-filter whose drop shadow samples content in another tile must
    /// invalidate its own tile when only that content changes.
    fn test_backdrop_sampled_margin(&mut self) {
        self.render_yaml("backdrop_sampled_margin_1");
        let results = self.render_yaml("backdrop_sampled_margin_2");

        assert!(
            matches!(results.pc_debug.slice(0).tile(0, 1), TileDebugInfo::Dirty(..)),
            "Ensure the backdrop's tile is invalidated by a change to content its filter samples",
        );
    }

    fn test_backdrop_scroll_under_clip(&mut self) {
        // Scrolling a backdrop-filter in a clipped scroll frame redraws it, which
        // dirties tiles that the slice's clip then keeps from being drawn. A
        // later frame with no change must not redraw anything because of them.
        self.render_yaml("basic");
        self.render_yaml("backdrop_scroll_under_clip_1");
        self.render_yaml("backdrop_scroll_under_clip_2");
        let results = self.render_yaml("backdrop_scroll_under_clip_2");
        assert!(
            !Self::has_any_dirty_tile(&results.pc_debug),
            "Ensure an unchanged frame after scrolling a backdrop redraws no tiles",
        );
    }

    fn test_backdrop_cross_slice_scroll_under_clip(&mut self) {
        // As test_backdrop_scroll_under_clip, with the backdrop read from a fixed
        // background slice below its scroll frame's slice.
        self.render_yaml("basic");
        self.render_yaml("backdrop_cross_slice_scroll_under_clip_1");
        self.render_yaml("backdrop_cross_slice_scroll_under_clip_2");
        let results = self.render_yaml("backdrop_cross_slice_scroll_under_clip_2");
        assert!(
            !Self::has_any_dirty_tile(&results.pc_debug),
            "Ensure an unchanged frame after scrolling a backdrop over a lower slice redraws no tiles",
        );
    }

    fn test_backdrop_cross_slice_scroll(&mut self) {
        self.render_yaml("backdrop_cross_slice_scroll_1");
        // Behind the header, over the rounded gradient once scrolled.
        let results = self.render_yaml_path_probe(
            &PathBuf::from("invalidation/backdrop_cross_slice_scroll_2.yaml"),
            Some((200, 20)),
        );

        let dirty_tiles = |slice: usize| {
            results.pc_debug.slices[&slice].sub_slices.iter().flat_map(|tiles| tiles.values())
                .filter(|tile| matches!(tile, TileDebugInfo::Dirty(..)))
                .count()
        };

        assert_eq!(results.pc_debug.slices.len(), 3, "Ensure the backdrop's slices are not merged");
        assert_eq!(dirty_tiles(1), 0, "Ensure scrolling the content under the backdrop keeps its tiles");
        assert!(dirty_tiles(2) > 0, "Ensure the backdrop is redrawn when the content under it scrolls");

        // Content with a clip mask, in a slice whose tiles are all clean, is
        // still drawn into the backdrop.
        let [r, g, b, _] = results.probe_pixel.unwrap();
        assert!(
            (r as u32 + g as u32 + b as u32) < 600,
            "Ensure masked content under the backdrop is drawn after scrolling, got {:?}",
            (r, g, b),
        );
    }

    fn test_backdrop_cross_slice_rounded(&mut self) {
        // Content in a lower slice with a rounded compositor clip (an iframe in
        // a rounded root-level stacking context, like browser content under
        // chrome with rounded corners), under a backdrop-filter. It must look as
        // it does when everything is in one slice: clipped at the corner, and
        // visible through the backdrop.
        for &probe in &[(44, 4), (200, 40)] {
            let results = self.assert_probe_matches(
                "invalidation/backdrop_cross_slice_rounded.yaml",
                "invalidation/backdrop_cross_slice_rounded_ref.yaml",
                probe,
                "lower slice content under a backdrop is clipped like the slice",
            );
            assert!(results.pc_debug.slices.len() > 2, "Ensure the iframe content is in a slice of its own");
        }
    }

    fn test_backdrop_cross_slice_compositor_surfaces(&mut self) {
        // An image that would be promoted to an overlay or an underlay in a
        // slice below a backdrop-filter (an iframe, so a separate primary slice)
        // must still be drawn into the backdrop, as when it is in one slice.
        for kind in &["overlay", "underlay"] {
            for &probe in &[(200, 40), (200, 120)] {
                self.assert_probe_matches(
                    &format!("invalidation/backdrop_cross_slice_{}.yaml", kind),
                    &format!("invalidation/backdrop_cross_slice_{}_ref.yaml", kind),
                    probe,
                    &format!("a {} candidate under a backdrop in a lower slice is drawn into it", kind),
                );
            }
        }
    }

    fn test_backdrop_cross_slice_changes(&mut self) {
        // Changes to a slice below a backdrop-filter that don't dirty any of its
        // tiles still change what the backdrop reads from it. The reference and
        // the state before the change are each rendered after a scene of a
        // single slice, so that neither keeps tiles of the backdrop or the slice
        // below it from an earlier scene.
        let cases = [
            ("clip_grow", (100, 40), "content revealed by a lower slice's wider clip shows through the backdrop"),
            ("radius", (70, 20), "the backdrop is clipped like the lower slice after its rounded clip changes"),
            ("unsampled", (200, 40), "the backdrop stops showing a lower slice it no longer samples"),
        ];
        for (name, probe, what) in cases {
            let before = format!("invalidation/backdrop_cross_slice_{}_1.yaml", name);
            let after = format!("invalidation/backdrop_cross_slice_{}_2.yaml", name);
            self.render_yaml("basic");
            let expected = self.render_yaml_path_probe(&PathBuf::from(&after), Some(probe)).probe_pixel.unwrap();
            self.render_yaml("basic");
            self.render_yaml_path(&PathBuf::from(&before));
            let actual = self.render_yaml_path_probe(&PathBuf::from(&after), Some(probe)).probe_pixel.unwrap();
            assert!(
                expected.iter().zip(actual.iter()).all(|(e, a)| (*e as i32 - *a as i32).abs() <= 8),
                "Ensure {} at {:?}: expected {:?}, got {:?}",
                what, probe, expected, actual,
            );
        }
    }

    fn test_backdrop_cross_slice_uncovered(&mut self) {
        // A backdrop-filter's tiles, occluded for a frame, lose their textures
        // and are redrawn once uncovered, though nothing under the backdrop
        // changed. The backdrop must still read the slice below its own.
        let probe = (150, 200);
        self.render_yaml("basic");
        let expected = self.render_yaml_path_probe(
            &PathBuf::from("invalidation/backdrop_cross_slice_uncovered_1.yaml"),
            Some(probe),
        ).probe_pixel.unwrap();
        self.render_yaml("basic");
        self.render_yaml("backdrop_cross_slice_uncovered_1");
        self.render_yaml("backdrop_cross_slice_uncovered_2");
        let actual = self.render_yaml_path_probe(
            &PathBuf::from("invalidation/backdrop_cross_slice_uncovered_1.yaml"),
            Some(probe),
        ).probe_pixel.unwrap();
        assert!(
            expected.iter().zip(actual.iter()).all(|(e, a)| (*e as i32 - *a as i32).abs() <= 8),
            "Ensure the backdrop reads the slice below once uncovered at {:?}: expected {:?}, got {:?}",
            probe, expected, actual,
        );
    }

    fn test_backdrop_cross_slice_move_together(&mut self) {
        // A backdrop-filter and the slice below it, moved by the same amount:
        // nothing under the backdrop changed, so its tiles are kept.
        let probe = (200, 80);
        self.render_yaml("basic");
        let expected = self.render_yaml_path_probe(
            &PathBuf::from("invalidation/backdrop_cross_slice_move_together_2.yaml"),
            Some(probe),
        ).probe_pixel.unwrap();
        self.render_yaml("basic");
        self.render_yaml("backdrop_cross_slice_move_together_1");
        let results = self.render_yaml_path_probe(
            &PathBuf::from("invalidation/backdrop_cross_slice_move_together_2.yaml"),
            Some(probe),
        );
        let actual = results.probe_pixel.unwrap();

        let dirty_tiles = |slice: usize| {
            results.pc_debug.slices[&slice].sub_slices.iter().flat_map(|tiles| tiles.values())
                .filter(|tile| matches!(tile, TileDebugInfo::Dirty(..)))
                .count()
        };

        assert_eq!(results.pc_debug.slices.len(), 2, "Ensure the backdrop's slices are not merged");
        assert_eq!(dirty_tiles(0), 0, "Ensure scrolling the content under the backdrop keeps its tiles");
        assert_eq!(dirty_tiles(1), 0, "Ensure the backdrop isn't redrawn when it moves with the content under it");
        assert!(
            expected.iter().zip(actual.iter()).all(|(e, a)| (*e as i32 - *a as i32).abs() <= 8),
            "Ensure the backdrop still shows the content under it after both move at {:?}: expected {:?}, got {:?}",
            probe, expected, actual,
        );
    }

    fn test_backdrop_own_slice_compositor_surfaces(&mut self) {
        // Compositor surfaces aren't drawn into a backdrop, so nothing in the
        // area a backdrop-filter samples is promoted, but anything elsewhere,
        // in its slice or in a slice above, still is.
        assert_eq!(
            self.render_yaml_overlays("backdrop_own_slice_overlay_control"), 1,
            "Ensure the image is promoted to an overlay with no backdrop-filter",
        );
        assert_eq!(
            self.render_yaml_overlays("backdrop_own_slice_overlay_outside"), 1,
            "Ensure an image outside the area a backdrop-filter in its slice samples is promoted",
        );
        assert_eq!(
            self.render_yaml_overlays("backdrop_above_slice_overlay"), 1,
            "Ensure an image in a slice above a backdrop-filter's is promoted",
        );

        self.assert_probe_matches(
            "invalidation/backdrop_own_slice_overlay_inside.yaml",
            "invalidation/backdrop_own_slice_overlay_inside_ref.yaml",
            (200, 40),
            "an image in the area a backdrop-filter in its slice samples is drawn into it",
        );
        assert_eq!(
            self.render_yaml_overlays("backdrop_own_slice_overlay_inside"), 0,
            "Ensure an image in the area a backdrop-filter in its slice samples isn't promoted",
        );
    }

    /// Render a YAML file by name (relative to invalidation/), and return the
    /// number of compositor surfaces promoted to overlays in the frame.
    fn render_yaml_overlays(&mut self, filename: &str) -> u32 {
        self.wrench.renderer.take_frame_build_profiles();
        self.render_yaml(filename);
        let profiles = self.wrench.renderer.take_frame_build_profiles();
        let counters = profiles.last().expect("bug: no frame was built");
        counters
            .iter()
            .find(|counter| counter.name == "Compositor surface overlays")
            .map_or(0, |counter| counter.value as u32)
    }

    /// Render `test` and `reference`, and assert that their pixels at `probe`
    /// match to within 8 per channel. Returns the results of rendering `test`.
    fn assert_probe_matches(
        &mut self,
        test: &str,
        reference: &str,
        probe: (i32, i32),
        what: &str,
    ) -> RenderResult {
        let expected = self.render_yaml_path_probe(&PathBuf::from(reference), Some(probe)).probe_pixel.unwrap();
        let results = self.render_yaml_path_probe(&PathBuf::from(test), Some(probe));
        let actual = results.probe_pixel.unwrap();
        assert!(
            expected.iter().zip(actual.iter()).all(|(e, a)| (*e as i32 - *a as i32).abs() <= 8),
            "Ensure {} at {:?}: expected {:?}, got {:?}",
            what, probe, expected, actual,
        );
        results
    }

    /// Render a YAML file by name (relative to invalidation/), and return the picture cache debug info
    fn render_yaml(
        &mut self,
        filename: &str,
    ) -> RenderResult {
        let path = PathBuf::from(format!("invalidation/{}.yaml", filename));
        self.render_yaml_path(&path)
    }

    fn render_yaml_path(
        &mut self,
        path: &Path,
    ) -> RenderResult {
        self.render_yaml_path_probe(path, None)
    }

    /// Render a yaml file, reading back the pixel at `probe` (in the yaml's
    /// coordinates) before presenting.
    fn render_yaml_path_probe(
        &mut self,
        path: &Path,
        probe: Option<(i32, i32)>,
    ) -> RenderResult {
        let mut reader = YamlFrameReader::new(path);

        reader.do_frame(self.wrench);
        let composite_needed = match self.rx.recv().unwrap() {
            NotifierEvent::WakeUp { composite_needed } => composite_needed,
            NotifierEvent::ShutDown => unreachable!(),
        };
        let results = self.wrench.render();

        let probe_pixel = probe.map(|(x, y)| {
            let window_size = self.window.get_inner_size();
            let rect = FramebufferIntRect::from_origin_and_size(
                FramebufferIntPoint::new(x, window_size.height - 1 - y),
                FramebufferIntSize::new(1, 1),
            );
            let pixels = self.wrench.renderer.read_pixels_rgba8(rect);
            [pixels[0], pixels[1], pixels[2], pixels[3]]
        });
        self.window.swap_buffers();

        RenderResult {
            pc_debug: results.picture_cache_debug,
            composite_needed,
            probe_pixel,
        }
    }
}
