/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{TileOffset, LayerRect, LayerSize, LayerPoint, LayerVector2D, DeviceUintSize};
use euclid::{vec2, rect};
use prim_store::EdgeAaSegmentMask;

/// If repetitions are far enough apart that only one is within
/// the primitive rect, then we can simplify the parameters and
/// treat the primitive as not repeated.
/// This can let us avoid unnecessary work later to handle some
/// of the parameters.
pub fn simplify_repeated_primitive(
    stretch_size: &LayerSize,
    tile_spacing: &mut LayerSize,
    prim_rect: &mut LayerRect,
) {
    let stride = *stretch_size + *tile_spacing;

    if stride.width >= prim_rect.size.width {
        tile_spacing.width = 0.0;
        prim_rect.size.width = f32::min(prim_rect.size.width, stretch_size.width);
    }
    if stride.width >= prim_rect.size.height {
        tile_spacing.height = 0.0;
        prim_rect.size.height = f32::min(prim_rect.size.height, stretch_size.height);
    }
}

pub fn for_each_repetition(
    prim_rect: &LayerRect,
    visible_rect: &LayerRect,
    stride: &LayerSize,
    callback: &mut FnMut(&LayerPoint, EdgeAaSegmentMask),
) {
    assert!(stride.width > 0.0);
    assert!(stride.height > 0.0);

    let visible_rect = match prim_rect.intersection(&visible_rect) {
       Some(rect) => rect,
       None => return,
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

    let mut p = LayerPoint::new(x0, y0);

    let x_most = visible_rect.max_x();
    let y_most = visible_rect.max_y();

    let x_count = f32::ceil((x_most - x0) / stride.width) as i32;
    let y_count = f32::ceil((y_most - y0) / stride.height) as i32;

    for y in 0..y_count {
        let mut row_flags = EdgeAaSegmentMask::empty();
        if y == 0 {
            row_flags |= EdgeAaSegmentMask::TOP;
        }
        if y == y_count - 1 {
            row_flags |= EdgeAaSegmentMask::BOTTOM;
        }

        for x in 0..x_count {
            let mut edge_flags = row_flags;
            if x == 0 {
                edge_flags |= EdgeAaSegmentMask::LEFT;
            }
            if x == x_count - 1 {
                edge_flags |= EdgeAaSegmentMask::RIGHT;
            }

            callback(&p, edge_flags);

            p.x += stride.width;
        }

        p.x = x0;
        p.y += stride.height;
    }
}

pub fn for_each_tile(
    prim_rect: &LayerRect,
    visible_rect: &LayerRect,
    device_image_size: &DeviceUintSize,
    device_tile_size: u32,
    callback: &mut FnMut(&LayerRect, TileOffset, EdgeAaSegmentMask),
) {
    // The image resource is tiled. We have to generate an image primitive
    // for each tile.
    // We need to do this because the image is broken up into smaller tiles in the texture
    // cache and the image shader is not able to work with this type of sparse representation.

    // The tiling logic works as follows:
    //
    //  ###################-+  -+
    //  #    |    |    |//# |   | image size
    //  #    |    |    |//# |   |
    //  #----+----+----+--#-+   |  -+
    //  #    |    |    |//# |   |   | regular tile size
    //  #    |    |    |//# |   |   |
    //  #----+----+----+--#-+   |  -+-+
    //  #////|////|////|//# |   |     | "leftover" height
    //  ################### |  -+  ---+
    //  #----+----+----+----+
    //
    // In the ascii diagram above, a large image is split into tiles of almost regular size.
    // The tiles on the right and bottom edges (hatched in the diagram) are smaller than
    // the regular tiles and are handled separately in the code see leftover_width/height.
    // each generated segment corresponds to a tile in the texture cache, with the
    // assumption that the smaller tiles with leftover sizes are sized to fit their own
    // irregular size in the texture cache.

    // Because we can have very large virtual images we iterate over the visible portion of
    // the image in layer space intead of iterating over device tiles.

    let visible_rect = match prim_rect.intersection(&visible_rect) {
       Some(rect) => rect,
       None => return,
    };

    let device_tile_size_f32 = device_tile_size as f32;

    // Ratio between (image space) tile size and image size .
    let tile_dw = device_tile_size_f32 / (device_image_size.width as f32);
    let tile_dh = device_tile_size_f32 / (device_image_size.height as f32);

    // size of regular tiles in layout space.
    let layer_tile_size = LayerSize::new(
        tile_dw * prim_rect.size.width,
        tile_dh * prim_rect.size.height,
    );

    // The size in pixels of the tiles on the right and bottom edges, smaller
    // than the regular tile size if the image is not a multiple of the tile size.
    // Zero means the image size is a multiple of the tile size.
    let leftover_device_size = DeviceUintSize::new(
        device_image_size.width % device_tile_size,
        device_image_size.height % device_tile_size
    );

    // The size in layer space of the tiles on the right and bottom edges.
    let leftover_layer_size = LayerSize::new(
        layer_tile_size.width * leftover_device_size.width as f32 / device_tile_size_f32,
        layer_tile_size.height * leftover_device_size.height as f32 / device_tile_size_f32,
    );

    // Offset of the row and column of tiles with leftover size.
    let leftover_offset = TileOffset::new(
        (device_image_size.width / device_tile_size) as u16,
        (device_image_size.height / device_tile_size) as u16,
    );

    // Number of culled out tiles to skip before the first visible tile.
    let t0 = TileOffset::new(
        if visible_rect.origin.x > prim_rect.origin.x {
            f32::floor((visible_rect.origin.x - prim_rect.origin.x) / layer_tile_size.width) as u16
        } else {
            0
        },
        if visible_rect.origin.y > prim_rect.origin.y {
            f32::floor((visible_rect.origin.y - prim_rect.origin.y) / layer_tile_size.height) as u16
        } else {
            0
        },
    );

    // Position of the first visible tile (top-left) in layer space.
    let x0 = prim_rect.origin.x + t0.x as f32 * layer_tile_size.width;
    let y0 = prim_rect.origin.y + t0.y as f32 * layer_tile_size.height;

    let x_count = f32::ceil((visible_rect.max_x() - x0) / layer_tile_size.width) as u16;
    let y_count = f32::ceil((visible_rect.max_y() - y0) / layer_tile_size.height) as u16;

    for y in 0..y_count {

        let mut row_flags = EdgeAaSegmentMask::empty();
        if y == 0 {
            row_flags |= EdgeAaSegmentMask::TOP;
        }
        if y == y_count - 1 {
            row_flags |= EdgeAaSegmentMask::BOTTOM;
        }

        for x in 0..x_count {
            let tile_offset = t0 + vec2(x, y);


            let mut segment_rect = LayerRect {
                origin: LayerPoint::new(
                    x0 + tile_offset.x as f32 * layer_tile_size.width,
                    y0 + tile_offset.y as f32 * layer_tile_size.height,
                ),
                size: layer_tile_size,
            };

            if tile_offset.x == leftover_offset.x {
                segment_rect.size.width = leftover_layer_size.width;
            }

            if tile_offset.y == leftover_offset.y {
                segment_rect.size.height = leftover_layer_size.height;
            }

            let mut edge_flags = row_flags;
            if x == 0 {
                edge_flags |= EdgeAaSegmentMask::LEFT;
            }
            if x == x_count - 1 {
                edge_flags |= EdgeAaSegmentMask::RIGHT;
            }

            callback(&segment_rect, tile_offset, edge_flags);
        }
    }
}

pub struct DecomposedTile {
    pub rect: LayerRect,
    pub stretch_size: LayerSize,
    pub tile_offset: TileOffset,
}

pub struct TiledImageInfo {
    /// The bounds of the item in layout space.
    pub rect: LayerRect,
    /// The space between each repeated pattern in layout space.
    pub tile_spacing: LayerSize,
    /// The size in layout space of each repetition of the image.
    pub stretch_size: LayerSize,

    /// The size the image occupies in the cache in device space.
    pub device_image_size: DeviceUintSize,
    /// The size of the tiles in the cache in device pixels.
    pub device_tile_size: u32,
}

/// Decomposes an image that is repeated into an image per individual repetition.
/// We need to do this when we are unable to perform the repetition in the shader,
/// for example if the image is tiled.
///
/// In all of the "decompose" methods below, we independently handle horizontal and vertical
/// decomposition. This lets us generate the minimum amount of primitives by, for example,
/// decomposing the repetition horizontally while repeating vertically in the shader (for
/// an image where the width is too bug but the height is not).
///
/// decompose_image and decompose_row handle image repetitions while decompose_cache_tiles
/// takes care of the decomposition required by the internal tiling of the image in the cache.
///
/// Note that the term tiling is overloaded: There is the tiling we get from repeating images
/// in layout space, and the tiling that we do in the texture cache (to avoid hitting texture
/// size limits). The latter is referred to as "device" tiling here to disambiguate.
pub fn decompose_image(info: &TiledImageInfo, callback: &mut FnMut(&DecomposedTile)) {

    let no_vertical_tiling = info.device_image_size.height <= info.device_tile_size;
    let no_vertical_spacing = info.tile_spacing.height == 0.0;

    if no_vertical_tiling && no_vertical_spacing {
        decompose_row(&info.rect, info, callback);
        return;
    }

    // Decompose each vertical repetition into rows.
    let layout_stride = info.stretch_size.height + info.tile_spacing.height;
    let num_repetitions = (info.rect.size.height / layout_stride).ceil() as u32;

    for i in 0 .. num_repetitions {
        let row_rect = rect(
            info.rect.origin.x,
            info.rect.origin.y + (i as f32) * layout_stride,
            info.rect.size.width,
            info.stretch_size.height,
        ).intersection(&info.rect);

        if let Some(row_rect) = row_rect {
            decompose_row(&row_rect, info, callback);
        }
    }
}


fn decompose_row(item_rect: &LayerRect, info: &TiledImageInfo, callback: &mut FnMut(&DecomposedTile)) {

    let no_horizontal_tiling = info.device_image_size.width <= info.device_tile_size;
    let no_horizontal_spacing = info.tile_spacing.width == 0.0;

    if no_horizontal_tiling && no_horizontal_spacing {
        decompose_cache_tiles(item_rect, info, callback);
        return;
    }

    // Decompose each horizontal repetition.
    let layout_stride = info.stretch_size.width + info.tile_spacing.width;
    let num_repetitions = (item_rect.size.width / layout_stride).ceil() as u32;

    for i in 0 .. num_repetitions {
        let decomposed_rect = rect(
            item_rect.origin.x + (i as f32) * layout_stride,
            item_rect.origin.y,
            info.stretch_size.width,
            item_rect.size.height,
        ).intersection(item_rect);

        if let Some(decomposed_rect) = decomposed_rect {
            decompose_cache_tiles(&decomposed_rect, info, callback);
        }
    }
}

fn decompose_cache_tiles(
    item_rect: &LayerRect,
    info: &TiledImageInfo,
    callback: &mut FnMut(&DecomposedTile),
) {
    // The image resource is tiled. We have to generate an image primitive
    // for each tile.
    // We need to do this because the image is broken up into smaller tiles in the texture
    // cache and the image shader is not able to work with this type of sparse representation.

    // The tiling logic works as follows:
    //
    //  ###################-+  -+
    //  #    |    |    |//# |   | image size
    //  #    |    |    |//# |   |
    //  #----+----+----+--#-+   |  -+
    //  #    |    |    |//# |   |   | regular tile size
    //  #    |    |    |//# |   |   |
    //  #----+----+----+--#-+   |  -+-+
    //  #////|////|////|//# |   |     | "leftover" height
    //  ################### |  -+  ---+
    //  #----+----+----+----+
    //
    // In the ascii diagram above, a large image is split into tiles of almost regular size.
    // The tiles on the right and bottom edges (hatched in the diagram) are smaller than
    // the regular tiles and are handled separately in the code see leftover_width/height.
    // each generated image primitive corresponds to a tile in the texture cache, with the
    // assumption that the smaller tiles with leftover sizes are sized to fit their own
    // irregular size in the texture cache.
    //
    // For the case where we don't tile along an axis, we can still perform the repetition in
    // the shader (for this particular axis), and it is worth special-casing for this to avoid
    // generating many primitives.
    // This can happen with very tall and thin images used as a repeating background.
    // Apparently web authors do that...

    let needs_repeat_x = info.stretch_size.width < item_rect.size.width;
    let needs_repeat_y = info.stretch_size.height < item_rect.size.height;

    let tiled_in_x = info.device_image_size.width > info.device_tile_size;
    let tiled_in_y = info.device_image_size.height > info.device_tile_size;

    // If we don't actually tile in this dimension, repeating can be done in the shader.
    let shader_repeat_x = needs_repeat_x && !tiled_in_x;
    let shader_repeat_y = needs_repeat_y && !tiled_in_y;

    let tile_size_f32 = info.device_tile_size as f32;

    // Note: this rounds down so it excludes the partially filled tiles on the right and
    // bottom edges (we handle them separately below).
    let num_tiles_x = (info.device_image_size.width / info.device_tile_size) as u16;
    let num_tiles_y = (info.device_image_size.height / info.device_tile_size) as u16;

    // Ratio between (image space) tile size and image size.
    let img_dw = tile_size_f32 / (info.device_image_size.width as f32);
    let img_dh = tile_size_f32 / (info.device_image_size.height as f32);

    // Stretched size of the tile in layout space.
    let stretched_tile_size = LayerSize::new(
        img_dw * info.stretch_size.width,
        img_dh * info.stretch_size.height,
    );

    // The size in pixels of the tiles on the right and bottom edges, smaller
    // than the regular tile size if the image is not a multiple of the tile size.
    // Zero means the image size is a multiple of the tile size.
    let leftover = DeviceUintSize::new(
        info.device_image_size.width % info.device_tile_size,
        info.device_image_size.height % info.device_tile_size
    );

    for ty in 0 .. num_tiles_y {
        for tx in 0 .. num_tiles_x {
            add_device_tile(
                item_rect,
                stretched_tile_size,
                TileOffset::new(tx, ty),
                1.0,
                1.0,
                shader_repeat_x,
                shader_repeat_y,
                callback,
            );
        }
        if leftover.width != 0 {
            // Tiles on the right edge that are smaller than the tile size.
            add_device_tile(
                item_rect,
                stretched_tile_size,
                TileOffset::new(num_tiles_x, ty),
                (leftover.width as f32) / tile_size_f32,
                1.0,
                shader_repeat_x,
                shader_repeat_y,
                callback,
            );
        }
    }

    if leftover.height != 0 {
        for tx in 0 .. num_tiles_x {
            // Tiles on the bottom edge that are smaller than the tile size.
            add_device_tile(
                item_rect,
                stretched_tile_size,
                TileOffset::new(tx, num_tiles_y),
                1.0,
                (leftover.height as f32) / tile_size_f32,
                shader_repeat_x,
                shader_repeat_y,
                callback,
            );
        }

        if leftover.width != 0 {
            // Finally, the bottom-right tile with a "leftover" size.
            add_device_tile(
                item_rect,
                stretched_tile_size,
                TileOffset::new(num_tiles_x, num_tiles_y),
                (leftover.width as f32) / tile_size_f32,
                (leftover.height as f32) / tile_size_f32,
                shader_repeat_x,
                shader_repeat_y,
                callback,
            );
        }
    }
}

fn add_device_tile(
    item_rect: &LayerRect,
    stretched_tile_size: LayerSize,
    tile_offset: TileOffset,
    tile_ratio_width: f32,
    tile_ratio_height: f32,
    shader_repeat_x: bool,
    shader_repeat_y: bool,
    callback: &mut FnMut(&DecomposedTile),
) {
    // If the image is tiled along a given axis, we can't have the shader compute
    // the image repetition pattern. In this case we base the primitive's rectangle size
    // on the stretched tile size which effectively cancels the repetition (and repetition
    // has to be emulated by generating more primitives).
    // If the image is not tiled along this axis, we can perform the repetition in the
    // shader. In this case we use the item's size in the primitive (on that particular
    // axis).
    // See the shader_repeat_x/y code below.

    let stretch_size = LayerSize::new(
        stretched_tile_size.width * tile_ratio_width,
        stretched_tile_size.height * tile_ratio_height,
    );

    let mut prim_rect = LayerRect::new(
        item_rect.origin + LayerVector2D::new(
            tile_offset.x as f32 * stretched_tile_size.width,
            tile_offset.y as f32 * stretched_tile_size.height,
        ),
        stretch_size,
    );

    if shader_repeat_x {
        assert_eq!(tile_offset.x, 0);
        prim_rect.size.width = item_rect.size.width;
    }

    if shader_repeat_y {
        assert_eq!(tile_offset.y, 0);
        prim_rect.size.height = item_rect.size.height;
    }

    // Fix up the primitive's rect if it overflows the original item rect.
    if let Some(rect) = prim_rect.intersection(item_rect) {
        callback(&DecomposedTile {
            tile_offset,
            rect,
            stretch_size,
        });
    }
}
