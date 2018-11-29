/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{DeviceIntPoint, DeviceIntRect, DeviceIntSize};
use std::ops;
use util;

const NUM_BINS: usize = 3;
/// The minimum number of pixels on each side that we require for rects to be classified as
/// particular bin of freelists.
const MIN_RECT_AXIS_SIZES: [i32; NUM_BINS] = [1, 16, 32];

/// A texture allocator using the guillotine algorithm with the rectangle merge improvement. See
/// sections 2.2 and 2.2.5 in "A Thousand Ways to Pack the Bin - A Practical Approach to Two-
/// Dimensional Rectangle Bin Packing":
///
///    http://clb.demon.fi/files/RectangleBinPack.pdf
///
/// This approach was chosen because of its simplicity, good performance, and easy support for
/// dynamic texture deallocation.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct GuillotineAllocator {
    texture_size: DeviceIntSize,
    free_list: FreeRectList,
    allocations: u32,
}

impl GuillotineAllocator {
    pub fn new(texture_size: DeviceIntSize) -> Self {
        let mut free_list = FreeRectList::new();
        free_list.push(DeviceIntRect::new(
            DeviceIntPoint::zero(),
            texture_size,
        ));
        GuillotineAllocator {
            texture_size,
            free_list,
            allocations: 0,
        }
    }

    fn find_index_of_best_rect_in_bin(
        &self,
        bin: FreeListBin,
        requested_dimensions: &DeviceIntSize,
    ) -> Option<FreeListIndex> {
        let mut smallest_index_and_area = None;
        for (candidate_index, candidate_rect) in self.free_list[bin].iter().enumerate() {
            if requested_dimensions.width > candidate_rect.size.width ||
                requested_dimensions.height > candidate_rect.size.height
            {
                continue;
            }

            let candidate_area = candidate_rect.size.area();
            match smallest_index_and_area {
                Some((_, area)) if candidate_area >= area => continue,
                _ => smallest_index_and_area = Some((candidate_index, candidate_area)),
            }
        }

        smallest_index_and_area.map(|(index, _)| FreeListIndex(index))
    }

    /// Find a suitable rect in the free list. We choose the smallest such rect
    /// in terms of area (Best-Area-Fit, BAF).
    fn find_index_of_best_rect(
        &self,
        requested_dimensions: &DeviceIntSize,
    ) -> Option<(FreeListBin, FreeListIndex)> {
        let start_bin = FreeListBin::for_size(requested_dimensions);
        (start_bin.0 .. NUM_BINS as u8)
            .find_map(|id| {
                self.find_index_of_best_rect_in_bin(FreeListBin(id), requested_dimensions)
                    .map(|index| (FreeListBin(id), index))
            })
    }

    pub fn allocate(&mut self, requested_dimensions: &DeviceIntSize) -> Option<DeviceIntPoint> {
        if requested_dimensions.width == 0 || requested_dimensions.height == 0 {
            return Some(DeviceIntPoint::new(0, 0));
        }
        let (bin, index) = self.find_index_of_best_rect(requested_dimensions)?;

        // Remove the rect from the free list and decide how to guillotine it. We choose the split
        // that results in the single largest area (Min Area Split Rule, MINAS).
        let chosen_rect = self.free_list[bin].swap_remove(index.0);
        let candidate_free_rect_to_right = DeviceIntRect::new(
            DeviceIntPoint::new(
                chosen_rect.origin.x + requested_dimensions.width,
                chosen_rect.origin.y,
            ),
            DeviceIntSize::new(
                chosen_rect.size.width - requested_dimensions.width,
                requested_dimensions.height,
            ),
        );
        let candidate_free_rect_to_bottom = DeviceIntRect::new(
            DeviceIntPoint::new(
                chosen_rect.origin.x,
                chosen_rect.origin.y + requested_dimensions.height,
            ),
            DeviceIntSize::new(
                requested_dimensions.width,
                chosen_rect.size.height - requested_dimensions.height,
            ),
        );

        // Guillotine the rectangle.
        let new_free_rect_to_right;
        let new_free_rect_to_bottom;
        if candidate_free_rect_to_right.size.area() > candidate_free_rect_to_bottom.size.area() {
            new_free_rect_to_right = DeviceIntRect::new(
                candidate_free_rect_to_right.origin,
                DeviceIntSize::new(
                    candidate_free_rect_to_right.size.width,
                    chosen_rect.size.height,
                ),
            );
            new_free_rect_to_bottom = candidate_free_rect_to_bottom
        } else {
            new_free_rect_to_right = candidate_free_rect_to_right;
            new_free_rect_to_bottom = DeviceIntRect::new(
                candidate_free_rect_to_bottom.origin,
                DeviceIntSize::new(
                    chosen_rect.size.width,
                    candidate_free_rect_to_bottom.size.height,
                ),
            )
        }

        // Add the guillotined rects back to the free list.
        if !util::rect_is_empty(&new_free_rect_to_right) {
            self.free_list[bin].push(new_free_rect_to_right);
        }
        if !util::rect_is_empty(&new_free_rect_to_bottom) {
            self.free_list[bin].push(new_free_rect_to_bottom);
        }

        // Bump the allocation counter.
        self.allocations += 1;

        // Return the result.
        Some(chosen_rect.origin)
    }
}

/// A binning free list. Binning is important to avoid sifting through lots of small strips when
/// allocating many texture items.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
struct FreeRectList {
    bins: [Vec<DeviceIntRect>; NUM_BINS],
}

impl ops::Index<FreeListBin> for FreeRectList {
    type Output = Vec<DeviceIntRect>;
    fn index(&self, bin: FreeListBin) -> &Self::Output {
        &self.bins[bin.0 as usize]
    }
}

impl ops::IndexMut<FreeListBin> for FreeRectList {
    fn index_mut(&mut self, bin: FreeListBin) -> &mut Self::Output {
        &mut self.bins[bin.0 as usize]
    }
}

impl FreeRectList {
    fn new() -> Self {
        FreeRectList {
            bins: [Vec::new(), Vec::new(), Vec::new()],
        }
    }

    fn push(&mut self, rect: DeviceIntRect) {
        self[FreeListBin::for_size(&rect.size)].push(rect)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct FreeListBin(u8);

#[derive(Debug, Clone, Copy)]
struct FreeListIndex(usize);

impl FreeListBin {
    fn for_size(size: &DeviceIntSize) -> Self {
        MIN_RECT_AXIS_SIZES
            .iter()
            .enumerate()
            .rev()
            .find(|(_, &min_size)| min_size <= size.width && min_size <= size.height)
            .map(|(id, _)| FreeListBin(id as u8))
            .expect("Unable to find a bin!")
    }
}
