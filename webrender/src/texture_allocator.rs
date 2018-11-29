/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{DeviceIntPoint, DeviceIntRect, DeviceIntSize};
use util;

//TODO: gather real-world statistics on the bin usage in order to assist the decision
// on where to place the size thresholds.

/// This is an optimization tweak to enable looking through all the free rectangles in a bin
/// and choosing the smallest, as opposed to picking the first match.
const FIND_SMALLEST_AREA: bool = false;

const NUM_BINS: usize = 3;
/// The minimum number of pixels on each side that we require for rects to be classified as
/// particular bin of freelists.
const MIN_RECT_AXIS_SIZES: [i32; NUM_BINS] = [1, 16, 32];

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

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct FreeRectSlice(pub u32);

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct FreeRect {
    slice: FreeRectSlice,
    rect: DeviceIntRect,
}

/// A texture allocator using the guillotine algorithm with the rectangle merge improvement. See
/// sections 2.2 and 2.2.5 in "A Thousand Ways to Pack the Bin - A Practical Approach to Two-
/// Dimensional Rectangle Bin Packing":
///
///    http://clb.demon.fi/files/RectangleBinPack.pdf
///
/// This approach was chosen because of its simplicity, good performance, and easy support for
/// dynamic texture deallocation.
///
/// Note: the allocations are spread across multiple textures, and also are binned
/// orthogonally in order to speed up the search.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct ArrayAllocationTracker {
    bins: [Vec<FreeRect>; NUM_BINS],
}

impl ArrayAllocationTracker {
    pub fn new() -> Self {
        ArrayAllocationTracker {
            bins: [
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ],
        }
    }

    fn push(&mut self, slice: FreeRectSlice, rect: DeviceIntRect) {
        let id = FreeListBin::for_size(&rect.size).0 as usize;
        self.bins[id].push(FreeRect {
            slice,
            rect,
        })
    }

    /// Find a suitable rect in the free list. We choose the smallest such rect
    /// in terms of area (Best-Area-Fit, BAF).
    fn find_index_of_best_rect(
        &self,
        requested_dimensions: &DeviceIntSize,
    ) -> Option<(FreeListBin, FreeListIndex)> {
        let start_bin = FreeListBin::for_size(requested_dimensions);
        (start_bin.0 .. NUM_BINS as u8)
            .find_map(|id| if FIND_SMALLEST_AREA {
                let mut smallest_index_and_area = None;
                for (candidate_index, candidate) in self.bins[id as usize].iter().enumerate() {
                    if requested_dimensions.width > candidate.rect.size.width ||
                        requested_dimensions.height > candidate.rect.size.height
                    {
                        continue;
                    }

                    let candidate_area = candidate.rect.size.area();
                    match smallest_index_and_area {
                        Some((_, area)) if candidate_area >= area => continue,
                        _ => smallest_index_and_area = Some((candidate_index, candidate_area)),
                    }
                }

                smallest_index_and_area
                    .map(|(index, _)| (FreeListBin(id), FreeListIndex(index)))
            } else {
                self.bins[id as usize]
                    .iter()
                    .position(|candidate| {
                        requested_dimensions.width <= candidate.rect.size.width &&
                        requested_dimensions.height <= candidate.rect.size.height
                    })
                    .map(|index| (FreeListBin(id), FreeListIndex(index)))
            })
    }

    // Split that results in the single largest area (Min Area Split Rule, MINAS).
    fn split_guillotine(&mut self, chosen: &FreeRect, requested_dimensions: &DeviceIntSize) {
        let candidate_free_rect_to_right = DeviceIntRect::new(
            DeviceIntPoint::new(
                chosen.rect.origin.x + requested_dimensions.width,
                chosen.rect.origin.y,
            ),
            DeviceIntSize::new(
                chosen.rect.size.width - requested_dimensions.width,
                requested_dimensions.height,
            ),
        );
        let candidate_free_rect_to_bottom = DeviceIntRect::new(
            DeviceIntPoint::new(
                chosen.rect.origin.x,
                chosen.rect.origin.y + requested_dimensions.height,
            ),
            DeviceIntSize::new(
                requested_dimensions.width,
                chosen.rect.size.height - requested_dimensions.height,
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
                    chosen.rect.size.height,
                ),
            );
            new_free_rect_to_bottom = candidate_free_rect_to_bottom
        } else {
            new_free_rect_to_right = candidate_free_rect_to_right;
            new_free_rect_to_bottom = DeviceIntRect::new(
                candidate_free_rect_to_bottom.origin,
                DeviceIntSize::new(
                    chosen.rect.size.width,
                    candidate_free_rect_to_bottom.size.height,
                ),
            )
        }

        // Add the guillotined rects back to the free list.
        if !util::rect_is_empty(&new_free_rect_to_right) {
            self.push(chosen.slice, new_free_rect_to_right);
        }
        if !util::rect_is_empty(&new_free_rect_to_bottom) {
            self.push(chosen.slice, new_free_rect_to_bottom);
        }
    }

    pub fn allocate(
        &mut self, requested_dimensions: &DeviceIntSize
    ) -> Option<(FreeRectSlice, DeviceIntPoint)> {
        if requested_dimensions.width == 0 || requested_dimensions.height == 0 {
            return Some((FreeRectSlice(0), DeviceIntPoint::new(0, 0)));
        }
        let (bin, index) = self.find_index_of_best_rect(requested_dimensions)?;

        // Remove the rect from the free list and decide how to guillotine it.
        let chosen = self.bins[bin.0 as usize].swap_remove(index.0);
        self.split_guillotine(&chosen, requested_dimensions);

        // Return the result.
        Some((chosen.slice, chosen.rect.origin))
    }

    /// Add a new slice to the allocator, and immediately allocate a rect from it.
    pub fn extend(
        &mut self,
        slice: FreeRectSlice,
        total_size: DeviceIntSize,
        requested_dimensions: DeviceIntSize,
    ) {
        self.split_guillotine(
            &FreeRect { slice, rect: total_size.into() },
            &requested_dimensions
        );
    }
}
