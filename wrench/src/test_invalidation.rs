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
        if tokens.len() != 3 {
            panic!(
                "{}:{}: expected 'OP file1 file2', got: {}",
                path.display(),
                line_num + 1,
                line,
            );
        }

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
        });
    }

    tests
}

pub struct TestHarness<'a> {
    wrench: &'a mut Wrench,
    window: &'a mut WindowWrapper,
    rx: Receiver<NotifierEvent>,
}

struct RenderResult {
    pc_debug: PictureCacheDebugInfo,
    composite_needed: bool,
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
        rx: Receiver<NotifierEvent>
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
        self.test_rounded_rect_intersection();

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
            for tile_info in slice_info.tiles.values() {
                if matches!(tile_info, TileDebugInfo::Dirty(..)) {
                    return true;
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

            if pass {
                println!("PASS {} {} {}", op_str, file1_str, file2_str);
            } else {
                println!("FAIL {} {} {}", op_str, file1_str, file2_str);
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

    /// Ensure that two rounded-rect clips in the shared clip chain are combined
    /// into a single compositor clip with the correct intersected radii.
    fn test_rounded_rect_intersection(
        &mut self,
    ) {
        let results = self.render_yaml("rounded_rect_intersect");

        let slice = results.pc_debug.slice(0);

        // The two clips (outer: 400x400 r=20, inner: 400x350 r=15, both at origin)
        // should be combined into a single compositor clip on the tile cache slice.
        // The combined clip rect is the intersection (400x350), with the top corners
        // taking the larger radius (20) and bottom corners the smaller (15).
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
            bottom_left: LayoutSize::new(15.0, 15.0),
            bottom_right: LayoutSize::new(15.0, 15.0),
            shape_top_left: 1.0,
            shape_top_right: 1.0,
            shape_bottom_left: 1.0,
            shape_bottom_right: 1.0,
        };
        assert_eq!(clip.radius, expected_radius, "Combined clip radii");
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
        let mut reader = YamlFrameReader::new(path);

        reader.do_frame(self.wrench);
        let composite_needed = match self.rx.recv().unwrap() {
            NotifierEvent::WakeUp { composite_needed } => composite_needed,
            NotifierEvent::ShutDown => unreachable!(),
        };
        let results = self.wrench.render();
        self.window.swap_buffers();

        RenderResult {
            pc_debug: results.picture_cache_debug,
            composite_needed,
        }
    }
}
