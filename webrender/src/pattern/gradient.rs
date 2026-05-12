/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::units::*;
use api::{ColorF, ExtendMode, GradientStop};
use crate::pattern::{Pattern, PatternKind, PatternShaderInput, PatternTextureInput};
use crate::renderer::{GpuBufferBuilder, GpuBufferWriterF};

#[repr(u8)]
#[derive(Copy, Clone, Debug)]
pub enum GradientKind {
    Linear = 0,
    Radial = 1,
    Conic = 2,
}

pub fn linear_gradient_pattern(
    start: LayoutPoint,
    end: LayoutPoint,
    extend_mode: ExtendMode,
    stops: &[GradientStop],
    _is_software: bool,
    gpu_buffer_builder: &mut GpuBufferBuilder
) -> Pattern {
    let num_blocks = 2 + gpu_gradient_stops_blocks(stops.len());
    let mut writer = gpu_buffer_builder.f32.write_blocks(num_blocks);
    writer.push_one([
        start.x,
        start.y,
        end.x,
        end.y,
    ]);
    writer.push_one([
        0.0,
        0.0,
        0.0,
        0.0,
    ]);

    let is_opaque = write_gpu_gradient_stops_tree(stops, GradientKind::Linear, extend_mode, &mut writer);

    let gradient_address = writer.finish();

    Pattern {
        kind: PatternKind::Gradient,
        shader_input: PatternShaderInput(
            gradient_address.as_int(),
            0,
        ),
        texture_input: PatternTextureInput::default(),
        base_color: ColorF::WHITE,
        is_opaque,
    }
}

pub fn radial_gradient_pattern(
    center: LayoutPoint,
    scale: DeviceVector2D,
    start_radius: f32,
    end_radius: f32,
    ratio_xy: f32,
    extend_mode: ExtendMode,
    stops: &[GradientStop],
    _is_software: bool,
    gpu_buffer_builder: &mut GpuBufferBuilder
) -> Pattern {
    let num_blocks = 2 + gpu_gradient_stops_blocks(stops.len());
    let mut writer = gpu_buffer_builder.f32.write_blocks(num_blocks);
    writer.push_one([
        center.x,
        center.y,
        scale.x,
        scale.y,
    ]);
    writer.push_one([
        start_radius,
        end_radius,
        ratio_xy,
        0.0,
    ]);

    let is_opaque = write_gpu_gradient_stops_tree(stops, GradientKind::Radial, extend_mode, &mut writer);

    let gradient_address = writer.finish();

    Pattern {
        kind: PatternKind::Gradient,
        shader_input: PatternShaderInput(
            gradient_address.as_int(),
            0,
        ),
        texture_input: PatternTextureInput::default(),
        base_color: ColorF::WHITE,
        is_opaque,
    }
}

pub fn conic_gradient_pattern(
    center: LayoutPoint,
    scale: DeviceVector2D,
    angle: f32, // in radians
    start_offset: f32,
    end_offset: f32,
    extend_mode: ExtendMode,
    stops: &[GradientStop],
    gpu_buffer_builder: &mut GpuBufferBuilder
) -> Pattern {
    let num_blocks = 2 + gpu_gradient_stops_blocks(stops.len());
    let mut writer = gpu_buffer_builder.f32.write_blocks(num_blocks);
    writer.push_one([
        center.x,
        center.y,
        scale.x,
        scale.y,
    ]);
    writer.push_one([
        start_offset,
        end_offset,
        angle,
        0.0,
    ]);
    let is_opaque = write_gpu_gradient_stops_tree(stops, GradientKind::Conic, extend_mode, &mut writer);
    let gradient_address = writer.finish();

    Pattern {
        kind: PatternKind::Gradient,
        shader_input: PatternShaderInput(
            gradient_address.as_int(),
            0,
        ),
        texture_input: PatternTextureInput::default(),
        base_color: ColorF::WHITE,
        is_opaque,
    }
}


fn write_gpu_gradient_stops_header_and_colors(
    stops: &[GradientStop],
    kind: GradientKind,
    extend_mode: ExtendMode,
    writer: &mut GpuBufferWriterF,
) -> bool {
    // Write the header.
    writer.push_one([
        (kind as u8) as f32,
        stops.len() as f32,
        if extend_mode == ExtendMode::Repeat { 1.0 } else { 0.0 },
        0.0
    ]);

    // Write the stop colors.
    let mut is_opaque = true;
    for stop in stops {
        writer.push_one(stop.color.premultiplied());
        is_opaque &= stop.color.a == 1.0;
    }

    is_opaque
}

// Push stop offsets in rearranged order so that the search can be carried
// out as an implicit tree traversal.
//
// The structure of the tree is:
//  - Each level is plit into 5 partitions.
//  - The root level has one node (4 offsets -> 5 partitions).
//  - Each level has 5 more nodes than the previous one.
//  - Levels are pushed one by one starting from the root
//
// ```ascii
// level : indices
// ------:---------
//   0   :                                                               24     ...
//   1   :          4         9            14             19             |      ...
//   2   :  0,1,2,3,|,5,6,7,8,|10,11,12,13,| ,15,16,17,18,| ,20,21,22,23,| ,25, ...
// ```
//
// In the example above:
// - The first (root) contains a single block containing the stop offsets from
//   indices [24, 49, 74, 99].
// - The second level contains blocks of offsets from indices [4, 9, 14, 19],
//   [29, 34, 39, 44], etc.
// - The third (leaf) level contains blocks from indices [0,1,2,3], [5,6,7,8],
//   [15, 16, 17, 18], etc.
//
// Placeholder offsets (1.0) are used when a level has more capacity than the
// input number of stops.
//
// Conceptually, blocks [0,1,2,3] and [5,6,7,8] are the first two children of
// the node [4,9,14,19], separated by the offset from index 4.
// Links are not explicitly represented via pointers or indices. Instead the
// position in the buffer is sufficient to represent the level and index of the
// stop (at the expense of having to store extra padding to round up each tree
// level to its power-of-5-aligned size).
//
// This scheme is meant to make the traversal efficient loading offsets in
// blocks of 4. The shader can converge to the leaf in very few loads.
pub fn write_gpu_gradient_stops_tree(
    stops: &[GradientStop],
    kind: GradientKind,
    extend_mode: ExtendMode,
    writer: &mut GpuBufferWriterF,
) -> bool {
    let is_opaque = write_gpu_gradient_stops_header_and_colors(
        stops,
        kind,
        extend_mode,
        writer
    );

    let num_stops = stops.len();
    let mut num_levels = 1;
    let mut index_stride = 5;
    let mut next_index_stride = 1;
    // Number of 4-offsets blocks for the current level.
    // The root has 1, then each level has 5 more than the previous one.
    let mut num_blocks_for_level = 1;
    let mut offset_blocks = 1;
    while offset_blocks * 4 < num_stops {
        num_blocks_for_level *= 5;
        offset_blocks += num_blocks_for_level;

        num_levels += 1;
        index_stride *= 5;
        next_index_stride *= 5;
    }

    // Fix offset_blocks up to account for the fact that we don't
    // store the entirety of the last level;
    let num_blocks_for_last_level = num_blocks_for_level.min(num_stops / 5 + 1);

    // Reset num_blocks_for_level for the traversal.
    num_blocks_for_level = 1;

    // Go over each level, starting from the root.
    for level in 0..num_levels {
        // This scheme rounds up the number of offsets to store for each
        // level to the next power of 5, which can represent a lot of wasted
        // space, especially for the last levels. We need each level to start
        // at a specific power-of-5-aligned offset so we can't get around the
        // wasted space for all levels except the last one (which has the most
        // waste).
        let is_last_level = level == num_levels - 1;
        let num_blocks = if is_last_level {
            num_blocks_for_last_level
        } else {
            num_blocks_for_level
        };

        for block_idx in 0..num_blocks {
            let mut block = [1.0; 4];
            for i in 0..4 {
                let linear_idx = block_idx * index_stride
                    + i * next_index_stride
                    + next_index_stride - 1;

                if linear_idx < num_stops {
                    block[i] = stops[linear_idx].offset;
                }
            }
            writer.push_one(block);
        }

        index_stride = next_index_stride;
        next_index_stride /= 5;
        num_blocks_for_level *= 5;
    }

    return is_opaque;
}

fn gpu_gradient_stops_blocks(num_stops: usize) -> usize {
    let header_blocks = 1;
    let color_blocks = num_stops;

    // If this is changed, matching changes should be made to the
    // equivalent code in write_gpu_gradient_stops_tree.
    let mut num_blocks_for_level = 1;
    let mut offset_blocks = 1;
    while offset_blocks * 4 < num_stops {
        num_blocks_for_level *= 5;
        offset_blocks += num_blocks_for_level;
    }

    // Fix the capacity up to account for the fact that we don't
    // store the entirety of the last level;
    let num_blocks_for_last_level = num_blocks_for_level.min(num_stops / 5 + 1);
    offset_blocks -= num_blocks_for_level;
    offset_blocks += num_blocks_for_last_level;

    header_blocks + color_blocks + offset_blocks
}
