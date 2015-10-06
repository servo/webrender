use bit_vec::BitVec;
use device::TextureId;
use internal_types::{TextureUpdate, TextureUpdateOp, TextureUpdateDetails};
use internal_types::{RenderTargetMode, TextureUpdateList};
use std::collections::HashMap;
use std::mem;
use types::{ImageFormat, ImageID, RasterItem};

struct TexturePage {
    texture_size: u32,
    block_size: u32,
    alloc_bitvec: BitVec,
    blocks_per_row: u32,
    texture_id: TextureId,
}

impl TexturePage {
    fn new(texture_id: TextureId, texture_size: u32, block_size: u32) -> TexturePage {
        debug_assert!(texture_size % block_size == 0);
        let blocks_per_row = texture_size / block_size;

        TexturePage {
            texture_size: texture_size,
            block_size: block_size,
            blocks_per_row: blocks_per_row,
            alloc_bitvec: BitVec::from_elem((blocks_per_row * blocks_per_row) as usize, false),
            texture_id: texture_id,
        }
    }

    fn is_full(&self) -> bool {
        self.alloc_bitvec.all()
    }

    fn allocate(&mut self) -> (u32, u32) {
        assert!(!self.alloc_bitvec.all());
        let mut free_index = None;

        for (index, value) in self.alloc_bitvec.iter().enumerate() {
            if !value {
                free_index = Some(index);
                break;
            }
        }

        let free_index = free_index.unwrap();
        self.alloc_bitvec.set(free_index, true);
        let free_index = free_index as u32;

        (free_index % self.blocks_per_row, free_index / self.blocks_per_row)
    }
}

#[derive(Debug)]
pub struct TextureCacheItem {
    pub u0: f32,        // todo(gw): don't precalc these?
    pub v0: f32,
    pub u1: f32,
    pub v1: f32,
    pub x0: i32,
    pub y0: i32,
    pub width: u32,
    pub height: u32,
    pub texture_id: TextureId,      // todo(gw): can this ever get invalidated? (page defragmentation?)
    pub format: ImageFormat,
}

impl TextureCacheItem {
    fn new(texture_id: TextureId, format: ImageFormat, x0: i32, y0: i32, width: u32, height: u32, u0: f32, v0: f32, u1: f32, v1: f32) -> TextureCacheItem {
        TextureCacheItem {
            texture_id: texture_id,
            u0: u0,
            v0: v0,
            u1: u1,
            v1: v1,
            x0: x0,
            y0: y0,
            width: width,
            height: height,
            format: format,
        }
    }
}

struct TextureCacheLevel {
    pages_a8: Vec<TexturePage>,
    pages_rgb8: Vec<TexturePage>,
    pages_rgba8: Vec<TexturePage>,
    size: u32,
}

impl TextureCacheLevel {
    fn new(size: u32) -> TextureCacheLevel {
        TextureCacheLevel {
            pages_a8: Vec::new(),
            pages_rgb8: Vec::new(),
            pages_rgba8: Vec::new(),
            size: size,
        }
    }
}

pub struct TextureCache {
    free_texture_ids: Vec<TextureId>,
    items: HashMap<ImageID, TextureCacheItem>,
    levels: [TextureCacheLevel; 4],
    pending_updates: TextureUpdateList,
}

#[derive(PartialEq, Eq, Debug)]
pub enum AllocationKind {
    TexturePage,
    Standalone,
}

pub struct AllocationResult {
    texture_id: TextureId,
    x: u32,
    y: u32,
    kind: AllocationKind,
}

impl TextureCache {
    pub fn new(free_texture_ids: Vec<TextureId>) -> TextureCache {
        TextureCache {
            free_texture_ids: free_texture_ids,
            items: HashMap::new(),
            pending_updates: TextureUpdateList::new(),
            levels: [
                TextureCacheLevel::new(32),
                TextureCacheLevel::new(64),
                TextureCacheLevel::new(128),
                TextureCacheLevel::new(256),
            ],
        }
    }

    pub fn pending_updates(&mut self) -> TextureUpdateList {
        mem::replace(&mut self.pending_updates, TextureUpdateList::new())
    }

    pub fn allocate_render_target(&mut self,
                                  width: u32,
                                  height: u32,
                                  format: ImageFormat) -> TextureId {
        let texture_id = self.free_texture_ids.pop().expect("TODO: Handle running out of texture IDs!");
        let update_op = TextureUpdate {
            id: texture_id,
            op: TextureUpdateOp::Create(width, height, format, RenderTargetMode::RenderTarget, None),
        };
        self.pending_updates.push(update_op);
        texture_id
    }

    pub fn free_render_target(&mut self, texture_id: TextureId) {
        self.free_texture_ids.push(texture_id);
        let update_op = TextureUpdate {
            id: texture_id,
            op: TextureUpdateOp::DeinitRenderTarget(texture_id),
        };
        self.pending_updates.push(update_op);
    }

    pub fn allocate(&mut self,
                    image_id: ImageID,
                    x0: i32,
                    y0: i32,
                    width: u32,
                    height: u32,
                    format: ImageFormat) -> AllocationResult {
        for level in &mut self.levels {
            if width <= level.size && height <= level.size {
                let (page_list, mode) = match format {
                    ImageFormat::A8 => (&mut level.pages_a8, RenderTargetMode::RenderTarget),
                    ImageFormat::RGBA8 => (&mut level.pages_rgba8, RenderTargetMode::None),
                    ImageFormat::RGB8 => (&mut level.pages_rgb8, RenderTargetMode::None),
                    ImageFormat::Invalid => unreachable!(),
                };

                let need_new_page = match page_list.last_mut() {
                    Some(page) => page.is_full(),
                    None => true,
                };

                if need_new_page {
                    let texture_size = 1024;
                    let texture_id = self.free_texture_ids.pop().expect("TODO: Handle running out of texture IDs!");
                    let update_op = TextureUpdate {
                        id: texture_id,
                        op: TextureUpdateOp::Create(texture_size, texture_size, format, mode, None),
                    };
                    self.pending_updates.push(update_op);
                    let page = TexturePage::new(texture_id, texture_size, level.size);
                    page_list.push(page);
                }

                let page = page_list.last_mut().unwrap();

                // todo: only a problem until multiple pages supported (as required)
                assert!(width <= page.block_size);
                assert!(height <= page.block_size);
                let (x, y) = page.allocate();

                // todo: take into account padding etc.
                // todo: make page index a separate type
                let tx0 = x * page.block_size;
                let ty0 = y * page.block_size;

                // todo: take into account padding etc.
                let u0 = x as f32 / page.blocks_per_row as f32;
                let v0 = y as f32 / page.blocks_per_row as f32;
                let u1 = u0 + width as f32 / page.texture_size as f32;
                let v1 = v0 + height as f32 / page.texture_size as f32;
                let cache_item = TextureCacheItem::new(page.texture_id, format, x0, y0, width, height, u0, v0, u1, v1);
                self.items.insert(image_id, cache_item);

                return AllocationResult {
                    texture_id: page.texture_id,
                    x: tx0,
                    y: ty0,
                    kind: AllocationKind::TexturePage,
                }
            }
        }

        let texture_id = self.free_texture_ids.pop().expect("TODO: Handle running out of texture ids!");
        let cache_item = TextureCacheItem::new(texture_id, format, 0, 0, width, height, 0.0, 0.0, 1.0, 1.0);
        self.items.insert(image_id, cache_item);

        AllocationResult {
            texture_id: texture_id,
            x: 0,
            y: 0,
            kind: AllocationKind::Standalone,
        }
    }

    pub fn insert_raster_op(&mut self,
                            image_id: ImageID,
                            item: &RasterItem) {
        let update_op = match item {
            &RasterItem::BorderRadius(ref op) => {
                let width = op.outer_radius_x.to_nearest_px() as u32;
                let height = op.outer_radius_y.to_nearest_px() as u32;

                let allocation = self.allocate(image_id,
                                               0,
                                               0,
                                               width,
                                               height,
                                               ImageFormat::A8);

                assert!(allocation.kind == AllocationKind::TexturePage);        // TODO: Handle large border radii not fitting in texture cache page

                TextureUpdate {
                    id: allocation.texture_id,
                    op: TextureUpdateOp::Update(allocation.x,
                                                allocation.y,
                                                width,
                                                height,
                                                TextureUpdateDetails::BorderRadius(op.outer_radius_x,
                                                                                   op.outer_radius_y,
                                                                                   op.inner_radius_x,
                                                                                   op.inner_radius_y)),
                }
            }
        };

        self.pending_updates.push(update_op);
    }

    pub fn insert(&mut self,
                  image_id: ImageID,
                  x0: i32,
                  y0: i32,
                  width: u32,
                  height: u32,
                  format: ImageFormat,
                  bytes: Vec<u8>) {

        let result = self.allocate(image_id, x0, y0, width, height, format);

        let op = match result.kind {
            AllocationKind::TexturePage => {
                TextureUpdateOp::Update(result.x, result.y, width, height, TextureUpdateDetails::Blit(bytes))
            }
            AllocationKind::Standalone => {
                TextureUpdateOp::Create(width, height, format, RenderTargetMode::None, Some(bytes))
            }
        };

        let update_op = TextureUpdate {
            id: result.texture_id,
            op: op,
        };

        self.pending_updates.push(update_op);
    }

    pub fn exists(&self, id: ImageID) -> bool {
        self.items.get(&id).is_some()
    }

    pub fn get(&self, id: ImageID) -> &TextureCacheItem {
        self.items.get(&id).expect(&format!("id {:?} was not cached!", id))
    }
}
