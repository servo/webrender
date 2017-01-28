/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use device::{MAX_TEXTURE_SIZE, TextureFilter};
use fnv::FnvHasher;
use freelist::{FreeList, FreeListItem, FreeListItemId};
use internal_types::{TextureUpdate, TextureUpdateOp};
use internal_types::{CacheTextureId, RenderTargetMode, TextureUpdateList, RectUv};
use std::cmp::{self, Ordering};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::BuildHasherDefault;
use std::mem;
use std::slice::Iter;
use std::sync::Arc;
use time;
use util;
use webrender_traits::{ImageData, ImageFormat, DevicePixel, DeviceIntPoint};
use webrender_traits::{DeviceUintRect, DeviceUintSize, DeviceUintPoint};
use webrender_traits::ImageDescriptor;

/// The number of bytes we're allowed to use for a texture.
const MAX_BYTES_PER_TEXTURE: u32 = 1024 * 1024 * 256;  // 256MB

/// The number of RGBA pixels we're allowed to use for a texture.
const MAX_RGBA_PIXELS_PER_TEXTURE: u32 = MAX_BYTES_PER_TEXTURE / 4;

/// The desired initial size of each texture, in pixels.
const INITIAL_TEXTURE_SIZE: u32 = 1024;

/// The desired initial area of each texture, in pixels squared.
const INITIAL_TEXTURE_AREA: u32 = INITIAL_TEXTURE_SIZE * INITIAL_TEXTURE_SIZE;

/// The square root of the number of RGBA pixels we're allowed to use for a texture, rounded down.
/// to the next power of two.
const SQRT_MAX_RGBA_PIXELS_PER_TEXTURE: u32 = 8192;

/// The minimum number of pixels on each side that we require for rects to be classified as
/// "medium" within the free list.
const MINIMUM_MEDIUM_RECT_SIZE: u32 = 16;

/// The minimum number of pixels on each side that we require for rects to be classified as
/// "large" within the free list.
const MINIMUM_LARGE_RECT_SIZE: u32 = 32;

/// The amount of time in milliseconds we give ourselves to coalesce rects before giving up.
const COALESCING_TIMEOUT: u64 = 100;

/// The number of items that we process in the coalescing work list before checking whether we hit
/// the timeout.
const COALESCING_TIMEOUT_CHECKING_INTERVAL: usize = 256;

pub type TextureCacheItemId = FreeListItemId;

#[inline]
fn copy_pixels(src: &[u8],
               target: &mut Vec<u8>,
               x: u32,
               y: u32,
               count: u32,
               width: u32,
               stride: Option<u32>,
               bpp: u32) {
    let row_length = match stride {
      Some(value) => value / bpp,
      None => width,
    };

    let pixel_index = (y * row_length + x) * bpp;
    for byte in src.iter().skip(pixel_index as usize).take((count * bpp) as usize) {
        target.push(*byte);
    }
}

/// A texture allocator using the guillotine algorithm with the rectangle merge improvement. See
/// sections 2.2 and 2.2.5 in "A Thousand Ways to Pack the Bin - A Practical Approach to Two-
/// Dimensional Rectangle Bin Packing":
///
///    http://clb.demon.fi/files/RectangleBinPack.pdf
///
/// This approach was chosen because of its simplicity, good performance, and easy support for
/// dynamic texture deallocation.
pub struct TexturePage {
    texture_id: CacheTextureId,
    texture_size: DeviceUintSize,
    free_list: FreeRectList,
    allocations: u32,
    dirty: bool,
}

impl TexturePage {
    pub fn new(texture_id: CacheTextureId, texture_size: DeviceUintSize) -> TexturePage {
        let mut page = TexturePage {
            texture_id: texture_id,
            texture_size: texture_size,
            free_list: FreeRectList::new(),
            allocations: 0,
            dirty: false,
        };
        page.clear();
        page
    }

    fn find_index_of_best_rect_in_bin(&self, bin: FreeListBin, requested_dimensions: &DeviceUintSize)
                                      -> Option<FreeListIndex> {
        let mut smallest_index_and_area = None;
        for (candidate_index, candidate_rect) in self.free_list.iter(bin).enumerate() {
            if !requested_dimensions.fits_inside(&candidate_rect.size) {
                continue
            }

            let candidate_area = candidate_rect.size.width * candidate_rect.size.height;
            smallest_index_and_area = Some((candidate_index, candidate_area));
            break
        }

        smallest_index_and_area.map(|(index, _)| FreeListIndex(bin, index))
    }

    fn find_index_of_best_rect(&self, requested_dimensions: &DeviceUintSize)
                               -> Option<FreeListIndex> {
        match FreeListBin::for_size(requested_dimensions) {
            FreeListBin::Large => {
                self.find_index_of_best_rect_in_bin(FreeListBin::Large, requested_dimensions)
            }
            FreeListBin::Medium => {
                match self.find_index_of_best_rect_in_bin(FreeListBin::Medium,
                                                          requested_dimensions) {
                    Some(index) => Some(index),
                    None => {
                        self.find_index_of_best_rect_in_bin(FreeListBin::Large,
                                                            requested_dimensions)
                    }
                }
            }
            FreeListBin::Small => {
                match self.find_index_of_best_rect_in_bin(FreeListBin::Small,
                                                          requested_dimensions) {
                    Some(index) => Some(index),
                    None => {
                        match self.find_index_of_best_rect_in_bin(FreeListBin::Medium,
                                                                  requested_dimensions) {
                            Some(index) => Some(index),
                            None => {
                                self.find_index_of_best_rect_in_bin(FreeListBin::Large,
                                                                    requested_dimensions)
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn allocate(&mut self,
                    requested_dimensions: &DeviceUintSize) -> Option<DeviceUintPoint> {
        // First, try to find a suitable rect in the free list. We choose the smallest such rect
        // in terms of area (Best-Area-Fit, BAF).
        let mut index = self.find_index_of_best_rect(requested_dimensions);

        // If one couldn't be found and we're dirty, coalesce rects and try again.
        if index.is_none() && self.dirty {
            self.coalesce();
            index = self.find_index_of_best_rect(requested_dimensions)
        }

        // If a rect still can't be found, fail.
        let index = match index {
            None => return None,
            Some(index) => index,
        };

        // Remove the rect from the free list and decide how to guillotine it. We choose the split
        // that results in the single largest area (Min Area Split Rule, MINAS).
        let chosen_rect = self.free_list.remove(index);
        let candidate_free_rect_to_right =
            DeviceUintRect::new(
                DeviceUintPoint::new(chosen_rect.origin.x + requested_dimensions.width, chosen_rect.origin.y),
                DeviceUintSize::new(chosen_rect.size.width - requested_dimensions.width, requested_dimensions.height));
        let candidate_free_rect_to_bottom =
            DeviceUintRect::new(
                DeviceUintPoint::new(chosen_rect.origin.x, chosen_rect.origin.y + requested_dimensions.height),
                DeviceUintSize::new(requested_dimensions.width, chosen_rect.size.height - requested_dimensions.height));
        let candidate_free_rect_to_right_area = candidate_free_rect_to_right.size.width *
            candidate_free_rect_to_right.size.height;
        let candidate_free_rect_to_bottom_area = candidate_free_rect_to_bottom.size.width *
            candidate_free_rect_to_bottom.size.height;

        // Guillotine the rectangle.
        let new_free_rect_to_right;
        let new_free_rect_to_bottom;
        if candidate_free_rect_to_right_area > candidate_free_rect_to_bottom_area {
            new_free_rect_to_right = DeviceUintRect::new(
                candidate_free_rect_to_right.origin,
                DeviceUintSize::new(candidate_free_rect_to_right.size.width,
                                    chosen_rect.size.height));
            new_free_rect_to_bottom = candidate_free_rect_to_bottom
        } else {
            new_free_rect_to_right = candidate_free_rect_to_right;
            new_free_rect_to_bottom =
                DeviceUintRect::new(candidate_free_rect_to_bottom.origin,
                          DeviceUintSize::new(chosen_rect.size.width,
                                              candidate_free_rect_to_bottom.size.height))
        }

        // Add the guillotined rects back to the free list. If any changes were made, we're now
        // dirty since coalescing might be able to defragment.
        if !util::rect_is_empty(&new_free_rect_to_right) {
            self.free_list.push(&new_free_rect_to_right);
            self.dirty = true
        }
        if !util::rect_is_empty(&new_free_rect_to_bottom) {
            self.free_list.push(&new_free_rect_to_bottom);
            self.dirty = true
        }

        // Bump the allocation counter.
        self.allocations += 1;

        // Return the result.
        Some(chosen_rect.origin)
    }

    #[inline(never)]
    fn coalesce(&mut self) {
        // Iterate to a fixed point or until a timeout is reached.
        let deadline = time::precise_time_ns() + COALESCING_TIMEOUT;
        let mut free_list = mem::replace(&mut self.free_list, FreeRectList::new()).into_vec();
        let mut changed = false;

        // Combine rects that have the same width and are adjacent.
        let mut new_free_list = Vec::new();
        free_list.sort_by(|a, b| {
            match a.size.width.cmp(&b.size.width) {
                Ordering::Equal => a.origin.x.cmp(&b.origin.x),
                ordering => ordering,
            }
        });
        for work_index in 0..free_list.len() {
            if work_index % COALESCING_TIMEOUT_CHECKING_INTERVAL == 0 &&
                    time::precise_time_ns() >= deadline {
                self.free_list = FreeRectList::from_slice(&free_list[..]);
                self.dirty = true;
                return
            }

            if free_list[work_index].size.width == 0 {
                continue
            }
            for candidate_index in (work_index + 1)..free_list.len() {
                if free_list[work_index].size.width != free_list[candidate_index].size.width ||
                        free_list[work_index].origin.x != free_list[candidate_index].origin.x {
                    break
                }
                if free_list[work_index].origin.y == free_list[candidate_index].max_y() ||
                        free_list[work_index].max_y() == free_list[candidate_index].origin.y {
                    changed = true;
                    free_list[work_index] =
                        free_list[work_index].union(&free_list[candidate_index]);
                    free_list[candidate_index].size.width = 0
                }
                new_free_list.push(free_list[work_index])
            }
            new_free_list.push(free_list[work_index])
        }
        free_list = new_free_list;

        // Combine rects that have the same height and are adjacent.
        let mut new_free_list = Vec::new();
        free_list.sort_by(|a, b| {
            match a.size.height.cmp(&b.size.height) {
                Ordering::Equal => a.origin.y.cmp(&b.origin.y),
                ordering => ordering,
            }
        });
        for work_index in 0..free_list.len() {
            if work_index % COALESCING_TIMEOUT_CHECKING_INTERVAL == 0 &&
                    time::precise_time_ns() >= deadline {
                self.free_list = FreeRectList::from_slice(&free_list[..]);
                self.dirty = true;
                return
            }

            if free_list[work_index].size.height == 0 {
                continue
            }
            for candidate_index in (work_index + 1)..free_list.len() {
                if free_list[work_index].size.height !=
                        free_list[candidate_index].size.height ||
                        free_list[work_index].origin.y != free_list[candidate_index].origin.y {
                    break
                }
                if free_list[work_index].origin.x == free_list[candidate_index].max_x() ||
                        free_list[work_index].max_x() == free_list[candidate_index].origin.x {
                    changed = true;
                    free_list[work_index] =
                        free_list[work_index].union(&free_list[candidate_index]);
                    free_list[candidate_index].size.height = 0
                }
            }
            new_free_list.push(free_list[work_index])
        }
        free_list = new_free_list;

        self.free_list = FreeRectList::from_slice(&free_list[..]);
        self.dirty = changed
    }

    pub fn clear(&mut self) {
        self.free_list = FreeRectList::new();
        self.free_list.push(&DeviceUintRect::new(
            DeviceUintPoint::zero(),
            self.texture_size));
        self.allocations = 0;
        self.dirty = false;
    }

    fn free(&mut self, rect: &DeviceUintRect) {
        debug_assert!(self.allocations > 0);
        self.allocations -= 1;
        if self.allocations == 0 {
            self.clear();
            return
        }

        self.free_list.push(rect);
        self.dirty = true
    }

    fn grow(&mut self, new_texture_size: DeviceUintSize) {
        assert!(new_texture_size.width >= self.texture_size.width);
        assert!(new_texture_size.height >= self.texture_size.height);

        let new_rects = [
            DeviceUintRect::new(DeviceUintPoint::new(self.texture_size.width, 0),
                                DeviceUintSize::new(new_texture_size.width - self.texture_size.width,
                                                    new_texture_size.height)),

            DeviceUintRect::new(DeviceUintPoint::new(0, self.texture_size.height),
                                DeviceUintSize::new(self.texture_size.width,
                                                    new_texture_size.height - self.texture_size.height)),
        ];

        for rect in &new_rects {
            if rect.size.width > 0 && rect.size.height > 0 {
                self.free_list.push(rect);
            }
        }

        self.texture_size = new_texture_size
    }

    fn can_grow(&self) -> bool {
        self.texture_size.width < max_texture_size() ||
        self.texture_size.height < max_texture_size()
    }
}

/// A binning free list. Binning is important to avoid sifting through lots of small strips when
/// allocating many texture items.
struct FreeRectList {
    small: Vec<DeviceUintRect>,
    medium: Vec<DeviceUintRect>,
    large: Vec<DeviceUintRect>,
}

impl FreeRectList {
    fn new() -> FreeRectList {
        FreeRectList {
            small: vec![],
            medium: vec![],
            large: vec![],
        }
    }

    fn from_slice(vector: &[DeviceUintRect]) -> FreeRectList {
        let mut free_list = FreeRectList::new();
        for rect in vector {
            free_list.push(rect)
        }
        free_list
    }

    fn push(&mut self, rect: &DeviceUintRect) {
        match FreeListBin::for_size(&rect.size) {
            FreeListBin::Small => self.small.push(*rect),
            FreeListBin::Medium => self.medium.push(*rect),
            FreeListBin::Large => self.large.push(*rect),
        }
    }

    fn remove(&mut self, index: FreeListIndex) -> DeviceUintRect {
        match index.0 {
            FreeListBin::Small => self.small.swap_remove(index.1),
            FreeListBin::Medium => self.medium.swap_remove(index.1),
            FreeListBin::Large => self.large.swap_remove(index.1),
        }
    }

    fn iter(&self, bin: FreeListBin) -> Iter<DeviceUintRect> {
        match bin {
            FreeListBin::Small => self.small.iter(),
            FreeListBin::Medium => self.medium.iter(),
            FreeListBin::Large => self.large.iter(),
        }
    }

    fn into_vec(mut self) -> Vec<DeviceUintRect> {
        self.small.extend(self.medium.drain(..));
        self.small.extend(self.large.drain(..));
        self.small
    }
}

#[derive(Debug, Clone, Copy)]
struct FreeListIndex(FreeListBin, usize);

#[derive(Debug, Clone, Copy, PartialEq)]
enum FreeListBin {
    Small,
    Medium,
    Large,
}

impl FreeListBin {
    pub fn for_size(size: &DeviceUintSize) -> FreeListBin {
        if size.width >= MINIMUM_LARGE_RECT_SIZE && size.height >= MINIMUM_LARGE_RECT_SIZE {
            FreeListBin::Large
        } else if size.width >= MINIMUM_MEDIUM_RECT_SIZE &&
                size.height >= MINIMUM_MEDIUM_RECT_SIZE {
            FreeListBin::Medium
        } else {
            FreeListBin::Small
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextureCacheItem {
    // Identifies the texture and array slice
    pub texture_id: CacheTextureId,

    // The texture coordinates for this item
    pub pixel_rect: RectUv<i32, DevicePixel>,

    // The size of the entire texture (not just the allocated rectangle)
    pub texture_size: DeviceUintSize,

    // The size of the actual allocated rectangle,
    // and the requested size. The allocated size
    // is the same as the requested in most cases,
    // unless the item has a border added for
    // bilinear filtering / texture bleeding purposes.
    pub allocated_rect: DeviceUintRect,
    pub requested_rect: DeviceUintRect,
}

// Structure squat the width/height fields to maintain the free list information :)
impl FreeListItem for TextureCacheItem {
    fn next_free_id(&self) -> Option<FreeListItemId> {
        if self.requested_rect.size.width == 0 {
            debug_assert!(self.requested_rect.size.height == 0);
            None
        } else {
            debug_assert!(self.requested_rect.size.width == 1);
            Some(FreeListItemId::new(self.requested_rect.size.height))
        }
    }

    fn set_next_free_id(&mut self, id: Option<FreeListItemId>) {
        match id {
            Some(id) => {
                self.requested_rect.size.width = 1;
                self.requested_rect.size.height = id.value();
            }
            None => {
                self.requested_rect.size.width = 0;
                self.requested_rect.size.height = 0;
            }
        }
    }
}

impl TextureCacheItem {
    fn new(texture_id: CacheTextureId,
           allocated_rect: DeviceUintRect,
           requested_rect: DeviceUintRect,
           texture_size: &DeviceUintSize)
           -> TextureCacheItem {
        TextureCacheItem {
            texture_id: texture_id,
            texture_size: *texture_size,
            pixel_rect: RectUv {
                top_left: DeviceIntPoint::new(requested_rect.origin.x as i32,
                                           requested_rect.origin.y as i32),
                top_right: DeviceIntPoint::new((requested_rect.origin.x + requested_rect.size.width) as i32,
                                            requested_rect.origin.y as i32),
                bottom_left: DeviceIntPoint::new(requested_rect.origin.x as i32,
                                              (requested_rect.origin.y + requested_rect.size.height) as i32),
                bottom_right: DeviceIntPoint::new((requested_rect.origin.x + requested_rect.size.width) as i32,
                                               (requested_rect.origin.y + requested_rect.size.height) as i32)
            },
            allocated_rect: allocated_rect,
            requested_rect: requested_rect,
        }
    }
}

struct TextureCacheArena {
    pages_a8: Vec<TexturePage>,
    pages_rgb8: Vec<TexturePage>,
    pages_rgba8: Vec<TexturePage>,
}

impl TextureCacheArena {
    fn new() -> TextureCacheArena {
        TextureCacheArena {
            pages_a8: Vec::new(),
            pages_rgb8: Vec::new(),
            pages_rgba8: Vec::new(),
        }
    }

    fn texture_page_for_id(&mut self, id: CacheTextureId) -> Option<&mut TexturePage> {
        for page in self.pages_a8.iter_mut().chain(self.pages_rgb8.iter_mut())
                                            .chain(self.pages_rgba8.iter_mut()) {
            if page.texture_id == id {
                return Some(page)
            }
        }
        None
    }
}

pub struct CacheTextureIdList {
    next_id: usize,
    free_list: Vec<usize>,
}

impl CacheTextureIdList {
    fn new() -> CacheTextureIdList {
        CacheTextureIdList {
            next_id: 0,
            free_list: Vec::new(),
        }
    }

    fn allocate(&mut self) -> CacheTextureId {
        // If nothing on the free list of texture IDs,
        // allocate a new one.
        if self.free_list.is_empty() {
            self.free_list.push(self.next_id);
            self.next_id += 1;
        }

        let id = self.free_list.pop().unwrap();
        CacheTextureId(id)
    }

    fn free(&mut self, id: CacheTextureId) {
        self.free_list.push(id.0);
    }
}

pub struct TextureCache {
    cache_id_list: CacheTextureIdList,
    free_texture_levels: HashMap<ImageFormat, Vec<FreeTextureLevel>, BuildHasherDefault<FnvHasher>>,
    items: FreeList<TextureCacheItem>,
    arena: TextureCacheArena,
    pending_updates: TextureUpdateList,
}

#[derive(PartialEq, Eq, Debug)]
pub enum AllocationKind {
    TexturePage,
    Standalone,
}

#[derive(Debug)]
pub struct AllocationResult {
    kind: AllocationKind,
    item: TextureCacheItem,
}

impl TextureCache {
    pub fn new() -> TextureCache {
        TextureCache {
            cache_id_list: CacheTextureIdList::new(),
            free_texture_levels: HashMap::with_hasher(Default::default()),
            items: FreeList::new(),
            pending_updates: TextureUpdateList::new(),
            arena: TextureCacheArena::new(),
        }
    }

    // The fn is a closure(x: u32, y: u32, w: u32, h: u32, src: Arc<Vec<u8>>, stride: Option<u32>).
    pub fn insert_image_border<F>(src: &[u8],
                               allocated_rect: DeviceUintRect,
                               requested_rect: DeviceUintRect,
                               stride: Option<u32>,
                               bpp: u32,
                               mut op: F) where F: FnMut(u32, u32, u32, u32, Arc<Vec<u8>>, Option<u32>){
        let mut top_row_data = Vec::new();
        let mut bottom_row_data = Vec::new();
        let mut left_column_data = Vec::new();
        let mut right_column_data = Vec::new();

        copy_pixels(&src, &mut top_row_data, 0, 0, 1, requested_rect.size.width, stride, bpp);
        copy_pixels(&src, &mut top_row_data, 0, 0, requested_rect.size.width, requested_rect.size.width, stride, bpp);
        copy_pixels(&src, &mut top_row_data, requested_rect.size.width - 1, 0, 1, requested_rect.size.width, stride, bpp);

        copy_pixels(&src, &mut bottom_row_data, 0, requested_rect.size.height - 1, 1, requested_rect.size.width, stride, bpp);
        copy_pixels(&src, &mut bottom_row_data, 0, requested_rect.size.height - 1, requested_rect.size.width, requested_rect.size.width, stride, bpp);
        copy_pixels(&src, &mut bottom_row_data, requested_rect.size.width - 1, requested_rect.size.height - 1, 1, requested_rect.size.width, stride, bpp);

        for y in 0..requested_rect.size.height {
            copy_pixels(&src, &mut left_column_data, 0, y, 1, requested_rect.size.width, stride, bpp);
            copy_pixels(&src, &mut right_column_data, requested_rect.size.width - 1, y, 1, requested_rect.size.width, stride, bpp);
        }

        op(allocated_rect.origin.x, allocated_rect.origin.y, allocated_rect.size.width, 1, Arc::new(top_row_data), None);
        op(allocated_rect.origin.x, allocated_rect.origin.y + requested_rect.size.height + 1, allocated_rect.size.width, 1, Arc::new(bottom_row_data), None);
        op(allocated_rect.origin.x, requested_rect.origin.y, 1, requested_rect.size.height, Arc::new(left_column_data), None);
        op(allocated_rect.origin.x + requested_rect.size.width + 1, requested_rect.origin.y, 1, requested_rect.size.height, Arc::new(right_column_data), None);
    }

    pub fn pending_updates(&mut self) -> TextureUpdateList {
        mem::replace(&mut self.pending_updates, TextureUpdateList::new())
    }

    // TODO(gw): This API is a bit ugly (having to allocate an ID and
    //           then use it). But it has to be that way for now due to
    //           how the raster_jobs code works.
    pub fn new_item_id(&mut self) -> TextureCacheItemId {
        let new_item = TextureCacheItem {
            pixel_rect: RectUv {
                top_left: DeviceIntPoint::zero(),
                top_right: DeviceIntPoint::zero(),
                bottom_left: DeviceIntPoint::zero(),
                bottom_right: DeviceIntPoint::zero(),
            },
            allocated_rect: DeviceUintRect::zero(),
            requested_rect: DeviceUintRect::zero(),
            texture_size: DeviceUintSize::zero(),
            texture_id: CacheTextureId(0),
        };
        self.items.insert(new_item)
    }

    pub fn allocate(&mut self,
                    image_id: TextureCacheItemId,
                    requested_width: u32,
                    requested_height: u32,
                    format: ImageFormat,
                    filter: TextureFilter)
                    -> AllocationResult {
        let requested_size = DeviceUintSize::new(requested_width, requested_height);

        // TODO(gw): For now, anything that requests nearest filtering
        //           just fails to allocate in a texture page, and gets a standalone
        //           texture. This isn't ideal, as it causes lots of batch breaks,
        //           but is probably rare enough that it can be fixed up later (it's also
        //           fairly trivial to implement, just tedious).
        if filter == TextureFilter::Nearest {
            // Fall back to standalone texture allocation.
            let texture_id = self.cache_id_list.allocate();
            let cache_item = TextureCacheItem::new(
                texture_id,
                DeviceUintRect::new(DeviceUintPoint::zero(), requested_size),
                DeviceUintRect::new(DeviceUintPoint::zero(), requested_size),
                &requested_size);
            *self.items.get_mut(image_id) = cache_item;

            return AllocationResult {
                item: self.items.get(image_id).clone(),
                kind: AllocationKind::Standalone,
            }
        }

        let mode = RenderTargetMode::SimpleRenderTarget;
        let page_list = match format {
            ImageFormat::A8 => &mut self.arena.pages_a8,
            ImageFormat::RGBA8 => &mut self.arena.pages_rgba8,
            ImageFormat::RGB8 => &mut self.arena.pages_rgb8,
            ImageFormat::Invalid | ImageFormat::RGBAF32 => unreachable!(),
        };

        let border_size = 1;
        let allocation_size = DeviceUintSize::new(requested_width + border_size * 2,
                                          requested_height + border_size * 2);

        // TODO(gw): Handle this sensibly (support failing to render items that can't fit?)
        assert!(allocation_size.width < max_texture_size());
        assert!(allocation_size.height < max_texture_size());

        // Loop until an allocation succeeds, growing or adding new
        // texture pages as required.
        loop {
            let location = page_list.last_mut().and_then(|last_page| {
                last_page.allocate(&allocation_size)
            });

            if let Some(location) = location {
                let page = page_list.last_mut().unwrap();

                let allocated_rect = DeviceUintRect::new(location, allocation_size);
                let requested_rect = DeviceUintRect::new(
                    DeviceUintPoint::new(location.x + border_size, location.y + border_size),
                    requested_size);

                let cache_item = TextureCacheItem::new(page.texture_id,
                                                       allocated_rect,
                                                       requested_rect,
                                                       &page.texture_size);
                *self.items.get_mut(image_id) = cache_item;

                return AllocationResult {
                    item: self.items.get(image_id).clone(),
                    kind: AllocationKind::TexturePage,
                }
            }

            if !page_list.is_empty() && page_list.last().unwrap().can_grow() {
                let last_page = page_list.last_mut().unwrap();
                // Grow the texture.
                let new_width = cmp::min(last_page.texture_size.width * 2, max_texture_size());
                let new_height = cmp::min(last_page.texture_size.height * 2, max_texture_size());
                let texture_size = DeviceUintSize::new(new_width, new_height);
                self.pending_updates.push(TextureUpdate {
                    id: last_page.texture_id,
                    op: texture_grow_op(texture_size, format, mode),
                });
                last_page.grow(texture_size);

                self.items.for_each_item(|item| {
                    if item.texture_id == last_page.texture_id {
                        item.texture_size = texture_size;
                    }
                });

                continue;
            }

            // We need a new page.
            let texture_size = initial_texture_size();
            let free_texture_levels_entry = self.free_texture_levels.entry(format);
            let mut free_texture_levels = match free_texture_levels_entry {
                Entry::Vacant(entry) => entry.insert(Vec::new()),
                Entry::Occupied(entry) => entry.into_mut(),
            };
            if free_texture_levels.is_empty() {
                let texture_id = self.cache_id_list.allocate();

                let update_op = TextureUpdate {
                    id: texture_id,
                    op: texture_create_op(texture_size, format, mode),
                };
                self.pending_updates.push(update_op);

                free_texture_levels.push(FreeTextureLevel {
                    texture_id: texture_id,
                });
            }
            let free_texture_level = free_texture_levels.pop().unwrap();
            let texture_id = free_texture_level.texture_id;

            let page = TexturePage::new(texture_id, texture_size);
            page_list.push(page);
        }
    }

    pub fn update(&mut self,
                  image_id: TextureCacheItemId,
                  descriptor: ImageDescriptor,
                  data: ImageData) {
        let existing_item = self.items.get(image_id);

        // TODO(gw): Handle updates to size/format!
        debug_assert!(existing_item.requested_rect.size.width == descriptor.width);
        debug_assert!(existing_item.requested_rect.size.height == descriptor.height);

        let op = match data {
            ImageData::ExternalHandle(..) | ImageData::ExternalBuffer(..)=> {
                panic!("Doesn't support Update() for external image.");
            }
            ImageData::Raw(bytes) => {
                TextureUpdateOp::Update {
                    page_pos_x: existing_item.requested_rect.origin.x,
                    page_pos_y: existing_item.requested_rect.origin.y,
                    width: descriptor.width,
                    height: descriptor.height,
                    data: bytes,
                    stride: descriptor.stride,
                }
            }
        };

        let update_op = TextureUpdate {
            id: existing_item.texture_id,
            op: op,
        };

        self.pending_updates.push(update_op);
    }

    pub fn insert(&mut self,
                  image_id: TextureCacheItemId,
                  descriptor: ImageDescriptor,
                  filter: TextureFilter,
                  data: ImageData) {
        let width = descriptor.width;
        let height = descriptor.height;
        let format = descriptor.format;
        let stride = descriptor.stride;

        let result = self.allocate(image_id,
                                   width,
                                   height,
                                   format,
                                   filter);

        let bpp = format.bytes_per_pixel().unwrap();

        match result.kind {
            AllocationKind::TexturePage => {
                match data {
                    ImageData::ExternalHandle(..) => {
                        panic!("External handle should not go through texture_cache.");
                    }
                    ImageData::Raw(bytes) => {
                        let mut op = |x , y , w , h , src , stride| {
                            let update_op = TextureUpdate {
                                id: result.item.texture_id,
                                op: TextureUpdateOp::Update {
                                    page_pos_x: x,
                                    page_pos_y: y,
                                    width: w,
                                    height: h,
                                    data: src,
                                    stride: stride,
                                },
                            };

                            self.pending_updates.push(update_op);
                        };

                        // image's borders
                        TextureCache::insert_image_border(&bytes,
                                                          result.item.allocated_rect,
                                                          result.item.requested_rect,
                                                          stride,
                                                          bpp,
                                                          &mut op);
                        // image itself
                        op(result.item.requested_rect.origin.x, result.item.requested_rect.origin.y,
                           result.item.requested_rect.size.width, result.item.requested_rect.size.height,
                           bytes, stride);
                    }
                    ImageData::ExternalBuffer(id) => {
                        let update_op = TextureUpdate {
                            id: result.item.texture_id,
                            op: TextureUpdateOp::UpdateForExternalBuffer {
                                allocated_rect: result.item.allocated_rect,
                                requested_rect: result.item.requested_rect,
                                id: id,
                                bpp: bpp,
                                stride: stride,
                            },
                        };

                        self.pending_updates.push(update_op);
                    }
                }
            }
            AllocationKind::Standalone => {
                match data {
                    ImageData::ExternalHandle(..) => {
                        panic!("External handle should not go through texture_cache.");
                    }
                    _ => {
                        let update_op = TextureUpdate {
                            id: result.item.texture_id,
                            op: TextureUpdateOp::Create {
                                width: width,
                                height: height,
                                format: format,
                                filter: filter,
                                mode: RenderTargetMode::None,
                                data: Some(data),
                            },
                        };

                        self.pending_updates.push(update_op);
                    }
                }
            }
        }
    }

    pub fn get(&self, id: TextureCacheItemId) -> &TextureCacheItem {
        self.items.get(id)
    }

    pub fn free(&mut self, id: TextureCacheItemId) {
        {
            let item = self.items.get(id);
            match self.arena.texture_page_for_id(item.texture_id) {
                Some(texture_page) => texture_page.free(&item.allocated_rect),
                None => {
                    // This is a standalone texture allocation. Just push it back onto the free
                    // list.
                    self.pending_updates.push(TextureUpdate {
                        id: item.texture_id,
                        op: TextureUpdateOp::Free,
                    });
                    self.cache_id_list.free(item.texture_id);
                }
            }
        }

        self.items.free(id)
    }
}

fn texture_create_op(texture_size: DeviceUintSize, format: ImageFormat, mode: RenderTargetMode)
                     -> TextureUpdateOp {
    TextureUpdateOp::Create {
        width: texture_size.width,
        height: texture_size.height,
        format: format,
        filter: TextureFilter::Linear,
        mode: mode,
        data: None,
    }
}

fn texture_grow_op(texture_size: DeviceUintSize,
                   format: ImageFormat,
                   mode: RenderTargetMode)
                   -> TextureUpdateOp {
    TextureUpdateOp::Grow {
        width: texture_size.width,
        height: texture_size.height,
        format: format,
        filter: TextureFilter::Linear,
        mode: mode,
    }
}

trait FitsInside {
    fn fits_inside(&self, other: &Self) -> bool;
}

impl FitsInside for DeviceUintSize {
    fn fits_inside(&self, other: &DeviceUintSize) -> bool {
        self.width <= other.width && self.height <= other.height
    }
}

/// FIXME(pcwalton): Would probably be more efficient as a bit vector.
#[derive(Clone, Copy)]
pub struct FreeTextureLevel {
    texture_id: CacheTextureId,
}

/// Returns the number of pixels on a side we start out with for our texture atlases.
fn initial_texture_size() -> DeviceUintSize {
    let max_hardware_texture_size = *MAX_TEXTURE_SIZE as u32;
    let initial_size = if max_hardware_texture_size * max_hardware_texture_size > INITIAL_TEXTURE_AREA {
        INITIAL_TEXTURE_SIZE
    } else {
        max_hardware_texture_size
    };
    DeviceUintSize::new(initial_size, initial_size)
}

/// Returns the number of pixels on a side we're allowed to use for our texture atlases.
fn max_texture_size() -> u32 {
    let max_hardware_texture_size = *MAX_TEXTURE_SIZE as u32;
    if max_hardware_texture_size * max_hardware_texture_size > MAX_RGBA_PIXELS_PER_TEXTURE {
        SQRT_MAX_RGBA_PIXELS_PER_TEXTURE
    } else {
        max_hardware_texture_size
    }
}
