/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Overview of the GPU cache.
//!
//! The main goal of the GPU cache is to allow on-demand
//! allocation and construction of GPU resources for the
//! vertex shaders to consume.
//!
//! Every item that wants to be stored in the GPU cache
//! should reserve a "slot" once via ```reserve_slot```.
//! This maps from a user provided unique key to a slot
//! id. Reserving a slot is a cheap operation, that does
//! *not* allocate room in the cache.
//!
//! On any frame when that data is required, the caller
//! must request that slot, via ```request_slot```.
//!
//! When ```end_frame``` is called, a user provided
//! closure is invoked. This closure is responsible
//! for building the GPU resources for the given
//! key. The closure will only be invoked for resources
//! that were not already in the cache.
//!
//! After ```end_frame``` has occurred, callers can
//! use the ```get_address``` API to get the allocated
//! address in the GPU cache of a given resource slot
//! for this frame.

use device::FrameId;
use profiler::GpuCacheProfileCounters;
use renderer::MAX_VERTEX_TEXTURE_WIDTH;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::mem;
use webrender_traits::ColorF;

pub const GPU_CACHE_INITIAL_HEIGHT: u32 = 512;
const FRAMES_BEFORE_EVICTION: usize = 10;

/// A single texel in RGBAF32 texture - 16 bytes.
#[derive(Copy, Clone, Debug)]
pub struct GpuBlockData {
    pub data: [f32; 4],
}

/// Conversion helpers for GpuBlockData
impl Into<GpuBlockData> for ColorF {
    fn into(self) -> GpuBlockData {
        GpuBlockData {
            data: [self.r, self.g, self.b, self.a],
        }
    }
}

impl Into<GpuBlockData> for [f32; 4] {
    fn into(self) -> GpuBlockData {
        GpuBlockData {
            data: self,
        }
    }
}

// Any data type that can be stored in the GPU cache should
// implement this trait.
pub trait ToGpuBlocks {
    // Append an arbitrary number of GPU blocks to the
    // provided array.
    fn write_gpu_blocks(&self, blocks: &mut Vec<GpuBlockData>);
}

/// A reserved resource slot in the cache. Reserving a slot
/// does *not* imply that the data is allocated or present
/// in the cache.
#[derive(Copy, Clone, Debug)]
pub struct GpuCacheSlotId(u32);

// A unique address in the GPU cache. These are uploaded
// as part of the primitive instances, to allow the vertex
// shader to fetch the specific data.
#[derive(Copy, Debug, Clone)]
pub struct GpuCacheAddress {
    pub u: u16,
    pub v: u16,
}

impl GpuCacheAddress {
    fn new(u: usize, v: usize) -> GpuCacheAddress {
        GpuCacheAddress {
            u: u as u16,
            v: v as u16,
        }
    }
}

// An entry in a free-list of blocks in the GPU cache.
struct Block {
    // The location in the cache of this block.
    address: GpuCacheAddress,
    // Index of the next free block in this list.
    next: Option<BlockIndex>,
}

impl Block {
    fn new(address: GpuCacheAddress, next: Option<BlockIndex>) -> Block {
        Block {
            address: address,
            next: next,
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct BlockIndex(usize);

// A row in the cache texture.
struct Row {
    // The fixed size of blocks that this row supports.
    // Each row becomes a slab allocator for a fixed block size.
    // This means no dealing with fragmentation within a cache
    // row as items are allocated and freed.
    block_size: usize,
    // The index in the ```blocks``` array where the free blocks
    // exist for this row. Allows finding the block structure quickly
    // when free'ing a block, from its address only.
    first_block_index: BlockIndex,
}

impl Row {
    fn new(block_size: usize, first_block_index: BlockIndex) -> Row {
        Row {
            block_size: block_size,
            first_block_index: first_block_index,
        }
    }
}

// A list of update operations that can be applied on the cache
// this frame. The list of updates is created by the render backend
// during frame construction. It's passed to the render thread
// where GL commands can be applied.
pub enum GpuCacheUpdate {
    Copy {
        block_index: usize,
        block_count: usize,
        address: GpuCacheAddress,
    }
}

pub struct GpuCacheUpdateList {
    // The current height of the texture. The render thread
    // should resize the texture if required.
    pub height: u32,
    // List of updates to apply.
    pub updates: Vec<GpuCacheUpdate>,
    // A flat list of GPU blocks that are pending upload
    // to GPU memory.
    pub blocks: Vec<GpuBlockData>,
}

// Holds the free lists of fixed size blocks. Mostly
// just serves to work around the borrow checker.
struct FreeBlockLists {
    free_list_1: Option<BlockIndex>,
    free_list_2: Option<BlockIndex>,
    free_list_4: Option<BlockIndex>,
    free_list_8: Option<BlockIndex>,
    free_list_large: Option<BlockIndex>,
}

impl FreeBlockLists {
    fn new() -> FreeBlockLists {
        FreeBlockLists {
            free_list_1: None,
            free_list_2: None,
            free_list_4: None,
            free_list_8: None,
            free_list_large: None,
        }
    }

    fn clear(&mut self) {
        *self = Self::new();
    }

    fn get_block_size_and_free_list(&mut self, block_count: usize) -> (usize, &mut Option<BlockIndex>) {
        // Find the appropriate free list to use
        // based on the block size.
        match block_count {
            0 => panic!("Can't allocate zero sized blocks!"),
            1 => (1, &mut self.free_list_1),
            2 => (2, &mut self.free_list_2),
            3...4 => (4, &mut self.free_list_4),
            5...8 => (8, &mut self.free_list_8),
            9...MAX_VERTEX_TEXTURE_WIDTH => (MAX_VERTEX_TEXTURE_WIDTH, &mut self.free_list_large),
            _ => panic!("Can't allocate > MAX_VERTEX_TEXTURE_WIDTH per resource!"),
        }
    }
}

// CPU-side representation of the GPU resource cache texture.
struct Texture {
    // Current texture height
    height: u32,
    // All blocks that have been created for this texture
    blocks: Vec<Block>,
    // Metadata about each allocated row.
    rows: Vec<Row>,
    // Free lists of available blocks for each supported
    // block size in the texture. These are intrusive
    // linked lists.
    free_lists: FreeBlockLists,
    // Pending blocks that have been written this frame
    // and will need to be sent to the GPU.
    pending_blocks: Vec<GpuBlockData>,
    // Pending update commands.
    updates: Vec<GpuCacheUpdate>,
}

impl Texture {
    fn new() -> Texture {
        Texture {
            height: GPU_CACHE_INITIAL_HEIGHT,
            blocks: Vec::new(),
            rows: Vec::new(),
            free_lists: FreeBlockLists::new(),
            pending_blocks: Vec::new(),
            updates: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.rows.clear();
        self.blocks.clear();
        self.pending_blocks.clear();
        self.updates.clear();
        self.free_lists.clear();

        // TODO(gw): Right now, we never shrink the size of the backing
        // texture. We could consider shrinking the texture here, if
        // you navigate away from a page that required a very large
        // texture backing store.
    }

    // Push new data into the cache. The ```block_index``` field represents
    // where the data was pushed into the texture ```pending_blocks``` array.
    // Return the allocated address for this data.
    fn push_data(&mut self,
                 block_index: usize,
                 block_count: usize,
                 profile_counters: &mut GpuCacheProfileCounters) -> GpuCacheAddress {
        // Find the appropriate free list to use based on the block size.
        let (alloc_size, free_list) = self.free_lists
                                          .get_block_size_and_free_list(block_count);

        // See if we need a new row (if free-list has nothing available)
        if free_list.is_none() {
            // TODO(gw): Handle the case where we need to resize
            //           the cache texture itself!
            if self.rows.len() as u32 == self.height {
                panic!("need to re-alloc texture!!");
            }

            // Create a new row.
            let items_per_row = MAX_VERTEX_TEXTURE_WIDTH / alloc_size;
            let row_index = self.rows.len();
            self.rows.push(Row::new(alloc_size, BlockIndex(self.blocks.len())));
            profile_counters.allocated_rows.inc();

            // Create a ```Block``` for each possible allocation address
            // in this row, and link it in to the free-list for this
            // block size.
            let mut prev_block_index = None;
            for i in 0..items_per_row {
                let address = GpuCacheAddress::new(i * alloc_size, row_index);
                let block_index = BlockIndex(self.blocks.len());
                let block = Block::new(address, prev_block_index);
                self.blocks.push(block);
                prev_block_index = Some(block_index);
            }

            *free_list = prev_block_index;
        }

        // Given the code above, it's now guaranteed that there is a block
        // available in the appropriate free-list. Pull a block from the
        // head of the list.
        let free_block_index = free_list.take().unwrap();
        let block = &self.blocks[free_block_index.0 as usize];
        *free_list = block.next;
        profile_counters.allocated_blocks.add(alloc_size);

        // Add this update to the pending list of blocks that need
        // to be updated on the GPU.
        self.updates.push(GpuCacheUpdate::Copy {
            block_index: block_index,
            block_count: block_count,
            address: block.address,
        });

        block.address
    }

    // Free a currently allocated data block in the cache.
    fn free(&mut self,
            address: GpuCacheAddress,
            profile_counters: &mut GpuCacheProfileCounters) {
        // Get the row metadata from the address.
        let row = &mut self.rows[address.v as usize];

        // Use the row metadata to determine which free-list
        // this block belongs to.
        let (_, free_list) = self.free_lists
                                 .get_block_size_and_free_list(row.block_size);
        profile_counters.allocated_blocks.sub(row.block_size);

        // Find the actual ```Block``` structure and link it in
        // to the correct free-list linked list.
        let block_index = row.first_block_index.0 + address.u as usize;
        let block = &mut self.blocks[block_index];
        block.next = *free_list;
        *free_list = Some(BlockIndex(block_index));
    }
}

/// The states that a reserved slot can exist in.
#[derive(Debug)]
enum SlotDetails {
    /// Slot is reserved but not in use - either
    /// never requested or has been evicted.
    Empty,
    /// Slot has been requested this frame and
    /// is pending upload to the GPU.
    Pending,
    /// Slot is currently allocated with backing
    /// store in the cache.
    Occupied {
        /// The last frame this slot was accessed.
        /// Used for determine which blocks to
        /// evict from the cache.
        last_access_time: FrameId,
        /// The current address of the slot in the
        /// cache backing store.
        address: GpuCacheAddress,
    }
}

/// A reserved cache slot.
#[derive(Debug)]
struct CacheSlot<K> {
    /// Intrusive link. This will either be in
    /// a free-list linked list, or the pending
    /// or occupied linked list.
    next: Option<GpuCacheSlotId>,
    /// The key of the data itself - passed to caller
    /// when the data needs to be built.
    key: K,
    /// Current state of the slot.
    details: SlotDetails,
}

/// The main LRU cache interface.
pub struct GpuCache<K: Hash + Eq> {
    /// Current frame ID.
    frame_id: FrameId,
    /// Free-list of cache slots.
    slots: Vec<CacheSlot<K>>,
    /// Mapping of caller data keys to reserved slots in the cache.
    keys: HashMap<K, GpuCacheSlotId>,
    /// Intrusive linked list of cache slots. Having these makes it
    /// fast to iterate all the pending slots (for GPU upload), and
    /// all the occupied slots (to check for cache eviction).
    pending_list_head: Option<GpuCacheSlotId>,
    occupied_list_head: Option<GpuCacheSlotId>,
    /// CPU-side texture allocator.
    texture: Texture,
    /// Hack to reset profile counters on new display list.
    reset_counters: bool,
}

impl<K> GpuCache<K> where K: Hash + Eq + Copy + fmt::Debug {
    pub fn new() -> GpuCache<K> {
        GpuCache {
            frame_id: FrameId::new(0),
            slots: Vec::new(),
            keys: HashMap::new(),
            pending_list_head: None,
            occupied_list_head: None,
            texture: Texture::new(),
            reset_counters: true,
        }
    }

    /// Begin a new frame.
    pub fn begin_frame(&mut self, profile_counters: &mut GpuCacheProfileCounters) {
        debug_assert!(self.texture.pending_blocks.is_empty());
        self.frame_id = self.frame_id + 1;

        if self.reset_counters {
            profile_counters.allocated_blocks.set(0);
            profile_counters.allocated_rows.set(0);
            self.reset_counters = false;
        }
    }

    /// Reserve a slot for a resource in the cache. This does
    /// not allocate any backing store.
    pub fn reserve_slot(&mut self, key: K) -> GpuCacheSlotId {
        let slot = GpuCacheSlotId(self.slots.len() as u32);
        self.slots.push(CacheSlot {
            next: None,
            key: key,
            details: SlotDetails::Empty,
        });

        // Ensure that the key is unique
        let old_entry = self.keys.insert(key, slot);
        debug_assert!(old_entry.is_none());

        slot
    }

    /// Request that this reserved slot actually be uploaded
    /// to the GPU this frame. This ensures that the resource
    /// will be built, if not already in the cache. This must
    /// be called every frame when a resource is used, so that
    /// the timestamp on the resource can be updated correctly.
    pub fn request_slot(&mut self, id: GpuCacheSlotId) {
        let slot = &mut self.slots[id.0 as usize];

        match slot.details {
            SlotDetails::Empty => {
                // The slot was currently unused. Move
                // it to the pending linked list.
                slot.details = SlotDetails::Pending;
                slot.next = self.pending_list_head;
                self.pending_list_head = Some(id);
            }
            SlotDetails::Pending { .. } => {
                // The slot has already been requested
                // this frame. It will already be moved
                // to the pending list, so do nothing.
            }
            SlotDetails::Occupied { ref mut last_access_time, .. } => {
                // The slot exists in the cache already.
                // Update timestamp so it doesn't get evicted.
                *last_access_time = self.frame_id;
            }
        }
    }

    /// Recycle the GPU cache for a new display list. This allows
    /// re-using existing backing allocations rather than calling
    /// the system memory allocator again.
    pub fn recycle(mut self) -> GpuCache<K> {
        self.slots.clear();
        self.keys.clear();
        self.texture.clear();
        self.pending_list_head = None;
        self.occupied_list_head = None;
        self.reset_counters = true;

        self
    }

    /// End the frame. This will invoke the user provided closure for
    /// any GPU resources that need to be built for this frame.
    pub fn end_frame<F>(&mut self,
                        profile_counters: &mut GpuCacheProfileCounters,
                        f: F) -> GpuCacheUpdateList
        where F: Fn(K, &mut Vec<GpuBlockData>) {

        // Prune any old items from the list to make room.
        // Traverse the occupied linked list and see
        // which items have not been used for a long time.
        let mut current_slot = self.occupied_list_head;
        let mut prev_slot: Option<GpuCacheSlotId> = None;

        while let Some(index) = current_slot {
            let (next_slot, should_unlink) = {
                let slot = &mut self.slots[index.0 as usize];
                let next_slot = slot.next;

                let should_unlink = match slot.details {
                    SlotDetails::Empty |
                    SlotDetails::Pending => panic!("Invalid occupied slot {:?}", index),
                    SlotDetails::Occupied { last_access_time, address } => {
                        // If this resource has not been used in the last
                        // few frames, free it from the texture and mark
                        // as empty.
                        if last_access_time + FRAMES_BEFORE_EVICTION < self.frame_id {
                            self.texture.free(address, profile_counters);
                            slot.details = SlotDetails::Empty;
                            slot.next = None;
                            true
                        } else {
                            false
                        }
                    }
                };

                (next_slot, should_unlink)
            };

            // If the slot was released, we will need to remove it
            // from the occupied linked list.
            if should_unlink {
                match prev_slot {
                    Some(prev_slot) => {
                        self.slots[prev_slot.0 as usize].next = next_slot;
                    }
                    None => {
                        self.occupied_list_head = next_slot;
                    }
                }
            } else {
                prev_slot = current_slot;
            }

            current_slot = next_slot;
        }

        // Run through the pending list and build new items.
        let mut current_slot = self.pending_list_head;
        while let Some(index) = current_slot {
            let slot = &mut self.slots[index.0 as usize];
            current_slot = slot.next;

            match slot.details {
                SlotDetails::Empty |
                SlotDetails::Occupied { .. } => unreachable!(),
                SlotDetails::Pending => {
                    slot.next = self.occupied_list_head;
                    self.occupied_list_head = Some(index);

                    // Build it!
                    let start_index = self.texture.pending_blocks.len();
                    f(slot.key, &mut self.texture.pending_blocks);
                    let block_count = self.texture.pending_blocks.len() - start_index;

                    // Push the data to the texture pending updates list.
                    let address = self.texture.push_data(start_index,
                                                         block_count,
                                                         profile_counters);

                    slot.details = SlotDetails::Occupied {
                        address: address,
                        last_access_time: self.frame_id,
                    };
                }
            }
        }
        self.pending_list_head = None;

        GpuCacheUpdateList {
            height: self.texture.height,
            updates: mem::replace(&mut self.texture.updates, Vec::new()),
            blocks: mem::replace(&mut self.texture.pending_blocks, Vec::new()),
        }
    }

    /// Get the actual GPU address in the texture for a given slot ID.
    /// It's assumed at this point that the given slot has been requested
    /// and built for this frame. Attempting to get the address for a
    /// freed or pending slot will panic!
    pub fn get_address(&self, id: &GpuCacheSlotId) -> GpuCacheAddress {
        match self.slots[id.0 as usize].details {
            SlotDetails::Empty => panic!("Trying to access a vacant gpu cache entry {:?}", id),
            SlotDetails::Pending => panic!("Trying to get address before build {:?}", id),
            SlotDetails::Occupied { last_access_time, address } => {
                // Ensure that it was actually requested this frame, and not
                // just accidentally left over in the cache from previous
                // frames.
                debug_assert_eq!(last_access_time, self.frame_id);
                address
            }
        }
    }
}
