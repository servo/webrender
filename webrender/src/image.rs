/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{TileOffset, TileRange, LayoutRect, LayoutSize, LayoutPoint};
use api::{DeviceIntSize, DeviceIntRect, TileSize};
use euclid::{point2, size2};
use prim_store::EdgeAaSegmentMask;

use std::i32;
use std::ops::Range;

/// If repetitions are far enough apart that only one is within
/// the primitive rect, then we can simplify the parameters and
/// treat the primitive as not repeated.
/// This can let us avoid unnecessary work later to handle some
/// of the parameters.
pub fn simplify_repeated_primitive(
    stretch_size: &LayoutSize,
    tile_spacing: &mut LayoutSize,
    prim_rect: &mut LayoutRect,
) {
    let stride = *stretch_size + *tile_spacing;

    if stride.width >= prim_rect.size.width {
        tile_spacing.width = 0.0;
        prim_rect.size.width = f32::min(prim_rect.size.width, stretch_size.width);
    }
    if stride.height >= prim_rect.size.height {
        tile_spacing.height = 0.0;
        prim_rect.size.height = f32::min(prim_rect.size.height, stretch_size.height);
    }
}

pub struct Repetition {
    pub origin: LayoutPoint,
    pub edge_flags: EdgeAaSegmentMask,
}

pub struct RepetitionIterator {
    current_x: i32,
    x_count: i32,
    current_y: i32,
    y_count: i32,
    row_flags: EdgeAaSegmentMask,
    current_origin: LayoutPoint,
    initial_origin: LayoutPoint,
    stride: LayoutSize,
}

impl Iterator for RepetitionIterator {
    type Item = Repetition;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_x == self.x_count {
            self.current_y += 1;
            if self.current_y >= self.y_count {
                return None;
            }
            self.current_x = 0;

            self.row_flags = EdgeAaSegmentMask::empty();
            if self.current_y == self.y_count - 1 {
                self.row_flags |= EdgeAaSegmentMask::BOTTOM;
            }

            self.current_origin.x = self.initial_origin.x;
            self.current_origin.y += self.stride.height;
        }

        let mut edge_flags = self.row_flags;
        if self.current_x == 0 {
            edge_flags |= EdgeAaSegmentMask::LEFT;
        }

        if self.current_x == self.x_count - 1 {
            edge_flags |= EdgeAaSegmentMask::RIGHT;
        }

        let repetition = Repetition {
            origin: self.current_origin,
            edge_flags: edge_flags,
        };

        self.current_origin.x += self.stride.width;
        self.current_x += 1;

        Some(repetition)
    }
}

pub fn repetitions(
    prim_rect: &LayoutRect,
    visible_rect: &LayoutRect,
    stride: LayoutSize,
) -> RepetitionIterator {
    assert!(stride.width > 0.0);
    assert!(stride.height > 0.0);

    let visible_rect = match prim_rect.intersection(&visible_rect) {
        Some(rect) => rect,
        None => {
            return RepetitionIterator {
                current_origin: LayoutPoint::zero(),
                initial_origin: LayoutPoint::zero(),
                current_x: 0,
                current_y: 0,
                x_count: 0,
                y_count: 0,
                stride,
                row_flags: EdgeAaSegmentMask::empty(),
            }
        }
    };

    let nx = if visible_rect.origin.x > prim_rect.origin.x {
        f32::floor((visible_rect.origin.x - prim_rect.origin.x) / stride.width)
    } else {
        0.0
    };

    let ny = if visible_rect.origin.y > prim_rect.origin.y {
        f32::floor((visible_rect.origin.y - prim_rect.origin.y) / stride.height)
    } else {
        0.0
    };

    let x0 = prim_rect.origin.x + nx * stride.width;
    let y0 = prim_rect.origin.y + ny * stride.height;

    let x_most = visible_rect.max_x();
    let y_most = visible_rect.max_y();

    let x_count = f32::ceil((x_most - x0) / stride.width) as i32;
    let y_count = f32::ceil((y_most - y0) / stride.height) as i32;

    let mut row_flags = EdgeAaSegmentMask::TOP;
    if y_count == 1 {
        row_flags |= EdgeAaSegmentMask::BOTTOM;
    }

    RepetitionIterator {
        current_origin: LayoutPoint::new(x0, y0),
        initial_origin: LayoutPoint::new(x0, y0),
        current_x: 0,
        current_y: 0,
        x_count,
        y_count,
        row_flags,
        stride,
    }
}

#[derive(Debug)]
pub struct Tile {
    pub rect: LayoutRect,
    pub offset: TileOffset,
    pub edge_flags: EdgeAaSegmentMask,
}

#[derive(Debug)]
pub struct TileIteratorExtent {
    /// Range of tiles to iterate over in number of tiles.
    tile_range: Range<i32>,
    /// Size of the first tile in layout space.
    first_tile_layout_size: f32,
    /// Size of the last tile in layout space.
    last_tile_layout_size: f32,
}

#[derive(Debug)]
pub struct TileIterator {
    current_tile: TileOffset,
    x: TileIteratorExtent,
    y: TileIteratorExtent,
    regular_tile_size: LayoutSize,
    local_origin: LayoutPoint,
    row_flags: EdgeAaSegmentMask,
}

impl Iterator for TileIterator {
    type Item = Tile;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_tile.x == self.x.tile_range.end {
            self.current_tile.y += 1;
            if self.current_tile.y >= self.y.tile_range.end {
                return None;
            }
            self.current_tile.x = self.x.tile_range.start;
            self.row_flags = EdgeAaSegmentMask::empty();
            if self.current_tile.y == self.y.tile_range.end - 1 {
                self.row_flags |= EdgeAaSegmentMask::BOTTOM;
            }
        }

        let tile_offset = self.current_tile;

        let mut segment_rect = LayoutRect {
            origin: LayoutPoint::new(
                self.local_origin.x + tile_offset.x as f32 * self.regular_tile_size.width,
                self.local_origin.y + tile_offset.y as f32 * self.regular_tile_size.height,
            ),
            size: self.regular_tile_size,
        };

        let mut edge_flags = self.row_flags;

        if tile_offset.x == self.x.tile_range.start {
            edge_flags |= EdgeAaSegmentMask::LEFT;
            segment_rect.size.width = self.x.first_tile_layout_size;
        }
        if tile_offset.x == self.x.tile_range.end - 1 {
            edge_flags |= EdgeAaSegmentMask::RIGHT;
            segment_rect.size.width = self.x.last_tile_layout_size;
        }

        if tile_offset.y == self.y.tile_range.start {
            segment_rect.size.height = self.y.first_tile_layout_size;
        } else if tile_offset.y == self.y.tile_range.end - 1 {
            segment_rect.size.height = self.y.last_tile_layout_size;
        }
        assert!(tile_offset.y < self.y.tile_range.end);
        let tile = Tile {
            rect: segment_rect,
            offset: tile_offset,
            edge_flags,
        };

        self.current_tile.x += 1;

        Some(tile)
    }
}

pub fn tiles(
    prim_rect: &LayoutRect,
    visible_rect: &LayoutRect,
    device_image_size: &DeviceIntSize,
    device_tile_size: i32,
) -> TileIterator {
    // The image resource is tiled. We have to generate an image primitive
    // for each tile.
    // We need to do this because the image is broken up into smaller tiles in the texture
    // cache and the image shader is not able to work with this type of sparse representation.

    // The tiling logic works as follows:
    //
    //  +-#################-+  -+
    //  | #//|    |    |//# |   | image size
    //  | #//|    |    |//# |   |
    //  +-#--+----+----+--#-+   |  -+
    //  | #//|    |    |//# |   |   | regular tile size
    //  | #//|    |    |//# |   |   |
    //  +-#--+----+----+--#-+   |  -+-+
    //  | #//|////|////|//# |   |     | "leftover" height
    //  | ################# |  -+  ---+
    //  +----+----+----+----+
    //
    // In the ascii diagram above, a large image is split into tiles of almost regular size.
    // The tiles on the edges (hatched in the diagram) can be smaller than the regular tiles
    // and are handled separately in the code (we'll call them boundary tiles).
    //
    // Each generated segment corresponds to a tile in the texture cache, with the
    // assumption that the boundary tiles are sized to fit their own irregular size in the
    // texture cache.
    //
    // Because we can have very large virtual images we iterate over the visible portion of
    // the image in layer space intead of iterating over all device tiles.

    let visible_rect = match prim_rect.intersection(&visible_rect) {
        Some(rect) => rect,
        None => {
            return TileIterator {
                current_tile: TileOffset::zero(),
                x: TileIteratorExtent {
                    tile_range: 0..0,
                    first_tile_layout_size: 0.0,
                    last_tile_layout_size: 0.0,
                },
                y: TileIteratorExtent {
                    tile_range: 0..0,
                    first_tile_layout_size: 0.0,
                    last_tile_layout_size: 0.0,
                },
                row_flags: EdgeAaSegmentMask::empty(),
                regular_tile_size: LayoutSize::zero(),
                local_origin: LayoutPoint::zero(),
            }
        }
    };

    // TODO: these values hold for regular images but not necessarily for blobs.
    // the latters can have image bounds with negative values (the blob image's
    // visible area provided by gecko).
    //
    // Likewise, the alyout space tiling origin (layout position of tile offset
    // (0, 0)) for blobs can be different from the top-left corner of the primitive
    // rect.
    //
    // This info needs to be patched through.
    let layout_tiling_origin = prim_rect.origin;
    let device_image_range_x = 0..device_image_size.width;
    let device_image_range_y = 0..device_image_size.height;

    // Size of regular tiles in layout space.
    let layout_tile_size = LayoutSize::new(
        device_tile_size as f32 / device_image_size.width as f32 * prim_rect.size.width,
        device_tile_size as f32 / device_image_size.height as f32 * prim_rect.size.height,
    );

    // The decomposition logic is exactly the same on each axis so we reduce
    // this to a 1-dimmensional problem in an attempt to make the code simpler.

    let x_extent = tiles_1d(
        layout_tile_size.width,
        visible_rect.min_x()..visible_rect.max_x(),
        device_image_range_x,
        device_tile_size,
        layout_tiling_origin.x,
    );

    let y_extent = tiles_1d(
        layout_tile_size.height,
        visible_rect.min_y()..visible_rect.max_y(),
        device_image_range_y,
        device_tile_size,
        layout_tiling_origin.y,
    );

    let mut row_flags = EdgeAaSegmentMask::TOP;
    if y_extent.tile_range.end == y_extent.tile_range.start + 1 {
        // Single row of tiles (both top and bottom edge).
        row_flags |= EdgeAaSegmentMask::BOTTOM;
    }

    TileIterator {
        current_tile: point2(
            x_extent.tile_range.start,
            y_extent.tile_range.start,
        ),
        x: x_extent,
        y: y_extent,
        row_flags,
        regular_tile_size: layout_tile_size,
        local_origin: prim_rect.origin,
    }
}

/// Decompose tiles along an arbitray axis.
///
/// Considering the 2d problem below:
///
///   +----+----+----+----+
///   | ################# |  -+
///   | #//|////|////|//# |   | image size
///   +-#--+----+----+--#-+   |  -+
///   | #//|    |    |//# |   |   | regular tile size
///   | #//|    |    |//# |   |   |
///   +-#--+----+----+--#-+   |  -+-+
///   | #//|////|////|//# |   |     | "leftover" height
///   | ################# |  -+  ---+
///   +----+----+----+----+
///
/// This function only treats the problem in one dimmension,
/// so either:
///
///   +----+----+----+----+
///   | ...|....|....|... |
///   | .  |    |    |  . |
///   +-#--+----+----+--#-+
///___| #//|    |    |//# |____
///   | #//|    |    |//# |
///   +-#--+----+----+--#-+
///   | .  |    |    |  . |
///   | ...|....|....|... |
///   +----+----+----+----+
///
/// Or:
///           |
///   +----+----+----+----+
///   | ...######....|... |
///   | .  |////|    |  . |
///   +----+----+----+----+
///   | .  |    |    |  . |
///   | .  |    |    |  . |
///   +----+----+----+----+
///   | .  |////|    |  . |
///   | ...######....|... |
///   +----+----+----+----+
//            |
fn tiles_1d(
    layout_tile_size: f32,
    layout_visible_range: Range<f32>,
    device_image_range: Range<i32>,
    device_tile_size: i32,
    layout_tiling_origin: f32,
) -> TileIteratorExtent {
    // Sizes of the boundary tiles in pixels.
    let first_tile_device_size = first_tile_size_1d(&device_image_range, device_tile_size);
    let last_tile_device_size = last_tile_size_1d(&device_image_range, device_tile_size);

    // Offsets of first and last tiles of this row/column (in number of tiles) without
    // taking culling into account.
    let (first_image_tile, last_image_tile) = first_and_last_tile_1d(&device_image_range, device_tile_size);

    // The visible tiles (because of culling).
    //
    // Here we don't need to do the off by one dance we did above because f32::floor
    // behaves the way we want.
    let first_visible_tile = f32::floor((layout_visible_range.start - layout_tiling_origin) / layout_tile_size) as i32;
    let last_visible_tile = f32::floor((layout_visible_range.end - layout_tiling_origin) / layout_tile_size) as i32;

    // Combine the above two to get the tiles in the image that are visible this frame.

    let first_tile = i32::max(first_image_tile, first_visible_tile);
    let last_tile = i32::min(last_image_tile, last_visible_tile);

    // The size in layout space of the boundary tiles.
    let first_tile_layout_size = if first_tile == first_image_tile {
        first_tile_device_size as f32 * layout_tile_size / device_tile_size as f32
    } else {
        // boundary tile was culled out, so the new first tile is a regularly sized tile.
        layout_tile_size
    };

    // Idem.
    let last_tile_layout_size = if last_tile == last_image_tile {
        last_tile_device_size as f32 * layout_tile_size / device_tile_size as f32
    } else {
        layout_tile_size
    };

    TileIteratorExtent {
        tile_range: first_tile..(last_tile + 1),
        first_tile_layout_size,
        last_tile_layout_size,
    }
}

/// Compute the tile offsets of teh first and last tiles in an arbitrary dimmension.
///
///        0
///        :
///  #-+---+---+---+---+---+--#
///  # |   |   |   |   |   |  #
///  #-+---+---+---+---+---+--#
///  ^     :               ^
///
///  +------------------------+  image_range
///        +---+  regular_tile_size
///
fn first_and_last_tile_1d(
    image_range: &Range<i32>,
    regular_tile_size: i32,
) -> (i32, i32) {
    // Integer division truncates towards zero so with negative values if the first/last
    // tile isn't a full tile we can get offset by one which we account for here.

    let mut first_image_tile = image_range.start / regular_tile_size;
    if image_range.start % regular_tile_size != 0 && image_range.start < 0 {
        first_image_tile -= 1;
    }

    let mut last_image_tile = image_range.end / regular_tile_size;
    if image_range.end % regular_tile_size == 0 || image_range.end < 0 {
        last_image_tile -= 1;
    }

    (first_image_tile, last_image_tile)
}

// Sizes of the first boundary tile in pixels.
//
// It can be smaller than the regular tile size if the image is not a multiple
// of the regular tile size.
fn first_tile_size_1d(
    image_range: &Range<i32>,
    regular_tile_size: i32,
) -> i32 {
    // We have to account for how the modulo operation behaves for negative values.
    let image_size = image_range.end - image_range.start;
    match image_range.start % regular_tile_size {
        //             .      #------+------+      .
        //             .      #//////|      |      .
        0 => i32::min(regular_tile_size, image_size),
        //   (zero) -> 0      .   #--+------+      .
        //             .      .   #//|      |      .
        // modulo(m):          ~~~
        m if m > 0 => regular_tile_size - m,
        //             .      .   #--+------+      0 <- (zero)
        //             .      .   #//|      |      .
        // modulo(m):             ~~~
        m => m,
    }
}

// Sizes of the last boundary tile in pixels.
//
// It can be smaller than the regular tile size if the image is not a multiple
// of the regular tile size.
fn last_tile_size_1d(
    image_range: &Range<i32>,
    regular_tile_size: i32,
) -> i32 {
    // We have to account for how the modulo operation behaves for negative values.
    let image_size = image_range.end - image_range.start;
    match image_range.end % regular_tile_size {
        //                    +------+------#      .
        // tiles:      .      |      |//////#      .
        0 => i32::min(regular_tile_size, image_size),
        //             .      +------+--#   .      0 <- (zero)
        //             .      |      |//#   .      .
        // modulo (m):                   ~~~
        m if m < 0 => regular_tile_size - m,
        //   (zero) -> 0      +------+--#   .      .
        //             .      |      |//#   .      .
        // modulo (m):                ~~~
        m => m,
    }
}

// Compute the width and height in pixels of a tile depending on its position in the image.
pub fn compute_tile_size(
    image_rect: &DeviceIntRect,
    regular_tile_size: TileSize,
    tile: TileOffset,
) -> DeviceIntSize {
    let regular_tile_size = regular_tile_size as i32;
    let img_range_x = image_rect.min_x()..image_rect.max_x();
    let img_range_y = image_rect.min_y()..image_rect.max_y();
    let (x_first, x_last) = first_and_last_tile_1d(&img_range_x, regular_tile_size);
    let (y_first, y_last) = first_and_last_tile_1d(&img_range_y, regular_tile_size);

    // Most tiles are going to have base_size as width and height,
    // except for tiles around the edges that are shrunk to fit the mage data
    // (See decompose_tiled_image in frame.rs).
    let actual_width = match tile.x as i32 {
        x if x == x_first => first_tile_size_1d(&img_range_x, regular_tile_size),
        x if x == x_last => last_tile_size_1d(&img_range_x, regular_tile_size),
        _ => regular_tile_size,
    };

    let actual_height = match tile.y as i32 {
        y if y == y_first => first_tile_size_1d(&img_range_y, regular_tile_size),
        y if y == y_last => last_tile_size_1d(&img_range_y, regular_tile_size),
        _ => regular_tile_size,
    };

    assert!(actual_width > 0);
    assert!(actual_height > 0);

    size2(actual_width, actual_height)
}


pub fn compute_tile_range(
    visible_area: &DeviceIntRect,
    tile_size: u16,
) -> TileRange {
    // Tile dimensions in normalized coordinates.
    let tw = 1. / (tile_size as f32);
    let th = 1. / (tile_size as f32);

    let t0 = point2(
        f32::floor(visible_area.origin.x as f32 * tw),
        f32::floor(visible_area.origin.y as f32 * th),
    ).try_cast::<i32>().unwrap_or_else(|| panic!("compute_tile_range bad values {:?} {:?}", visible_area, tile_size));

    let t1 = point2(
        f32::ceil(visible_area.max_x() as f32 * tw),
        f32::ceil(visible_area.max_y() as f32 * th),
    ).try_cast::<i32>().unwrap_or_else(|| panic!("compute_tile_range bad values {:?} {:?}", visible_area, tile_size));

    TileRange {
        origin: t0,
        size: (t1 - t0).to_size(),
    }
}

pub fn for_each_tile_in_range(
    range: &TileRange,
    mut callback: impl FnMut(TileOffset),
) {
    for y in range.min_y()..range.max_y() {
        for x in range.min_x()..range.max_x() {
            callback(point2(x, y));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use api::{LayoutRect, DeviceIntSize};
    use euclid::{rect, size2};

    // this checks some additional invariants
    fn checked_for_each_tile(
        prim_rect: &LayoutRect,
        visible_rect: &LayoutRect,
        device_image_size: &DeviceIntSize,
        device_tile_size: i32,
        callback: &mut FnMut(&LayoutRect, TileOffset, EdgeAaSegmentMask),
    ) {
        let mut coverage = LayoutRect::zero();
        let mut seen_tiles = HashSet::new();
        for tile in tiles(
            prim_rect,
            visible_rect,
            device_image_size,
            device_tile_size,
        ) {
            // make sure we don't get sent duplicate tiles
            assert!(!seen_tiles.contains(&tile.offset));
            seen_tiles.insert(tile.offset);
            coverage = coverage.union(&tile.rect);
            assert!(prim_rect.contains_rect(&tile.rect));
            callback(&tile.rect, tile.offset, tile.edge_flags);
        }
        assert!(prim_rect.contains_rect(&coverage));
        assert!(coverage.contains_rect(&visible_rect.intersection(&prim_rect).unwrap_or(LayoutRect::zero())));
    }

    #[test]
    fn basic() {
        let mut count = 0;
        checked_for_each_tile(&rect(0., 0., 1000., 1000.),
            &rect(75., 75., 400., 400.),
            &size2(400, 400),
            36,
            &mut |_tile_rect, _tile_offset, _tile_flags| {
                count += 1;
            },
        );
        assert_eq!(count, 36);
    }

    #[test]
    fn empty() {
        let mut count = 0;
        checked_for_each_tile(&rect(0., 0., 74., 74.),
              &rect(75., 75., 400., 400.),
              &size2(400, 400),
              36,
              &mut |_tile_rect, _tile_offset, _tile_flags| {
                count += 1;
              },
        );
        assert_eq!(count, 0);
    }
}
