use app_units::Au;
use device::TextureId;
use euclid::{Point2D, Rect, Size2D};
use fnv::FnvHasher;
use internal_types::{TextureUpdate, TextureUpdateOp, TextureUpdateDetails};
use internal_types::{RasterItem, RenderTargetMode, TextureImage, TextureUpdateList};
use std::collections::HashMap;
use std::collections::hash_state::DefaultState;
use std::mem;
use types::{ImageFormat, ImageID};
use util;

const TEXTURE_SIZE: u32 = 1024;

/// A texture allocator using the guillotine algorithm with the rectangle merge improvement. See
/// sections 2.2 and 2.2.5 in "A Thousand Ways to Pack the Bin - A Practical Approach to Two-
/// Dimensional Rectangle Bin Packing":
///
///    http://clb.demon.fi/files/RectangleBinPack.pdf
///
/// This approach was chosen because of its simplicity, good performance, and easy support for
/// dynamic texture deallocation.
struct TexturePage {
    texture_id: TextureId,
    texture_size: u32,
    free_list: Vec<Rect<u32>>,
    dirty: bool,
}

impl TexturePage {
    fn new(texture_id: TextureId, texture_size: u32) -> TexturePage {
        TexturePage {
            texture_id: texture_id,
            texture_size: texture_size,
            free_list: vec![
                Rect::new(Point2D::new(0, 0), Size2D::new(texture_size, texture_size))
            ],
            dirty: false,
        }
    }

    fn allocate(&mut self, requested_dimensions: &Size2D<u32>) -> Option<Point2D<u32>> {
        // First, try to find a suitable rect in the free list.
        let mut index = self.free_list.iter().position(|rect| {
            requested_dimensions.fits_inside(&rect.size)
        });

        // If one couldn't be found and we're dirty, coalesce rects and try again.
        if index.is_none() && self.dirty {
            self.coalesce();
            index = self.free_list.iter().position(|rect| {
                requested_dimensions.fits_inside(&rect.size)
            })
        }

        // If a rect still can't be found, fail.
        let index = match index {
            None => return None,
            Some(index) => index,
        };

        // Remove the rect from the free list and guillotine it.
        let chosen_rect = self.free_list.swap_remove(index);
        let new_free_rect_to_right =
            Rect::new(Point2D::new(chosen_rect.origin.x + requested_dimensions.width,
                                   chosen_rect.origin.y),
                      Size2D::new(chosen_rect.size.width - requested_dimensions.width,
                                  requested_dimensions.height));
        let new_free_rect_to_bottom =
            Rect::new(Point2D::new(chosen_rect.origin.x,
                                   chosen_rect.origin.y + requested_dimensions.height),
                      Size2D::new(chosen_rect.size.width,
                                  chosen_rect.size.height - requested_dimensions.height));

        // Add the guillotined rects back to the free list. If any changes were made, we're now
        // dirty since coalescing might be able to defragment.
        if !util::rect_is_empty(&new_free_rect_to_right) {
            self.free_list.push(new_free_rect_to_right);
            self.dirty = true
        }
        if !util::rect_is_empty(&new_free_rect_to_bottom) {
            self.free_list.push(new_free_rect_to_bottom);
            self.dirty = true
        }

        // Return the result.
        Some(chosen_rect.origin)
    }

    fn coalesce(&mut self) {
        // Iterate to a fixed point.
        loop {
            let mut changed = false;

            // Attempt to merge rects in the free list.
            let mut coalesced_free_rects = Vec::new();
            loop {
                let work_rect = match self.free_list.pop() {
                    None => break,
                    Some(work_rect) => work_rect,
                };

                let index_of_rect_to_merge_with = self.free_list.iter().position(|candidate_rect| {
                    (work_rect.origin.x == candidate_rect.origin.x &&
                        work_rect.size.width == candidate_rect.size.width &&
                        (work_rect.origin.y == candidate_rect.max_y() ||
                         work_rect.max_y() == candidate_rect.origin.y)) ||
                    (work_rect.origin.y == candidate_rect.origin.y &&
                        work_rect.size.height == candidate_rect.size.height &&
                        (work_rect.origin.x == candidate_rect.max_x() ||
                         work_rect.max_x() == candidate_rect.origin.x))
                });

                match index_of_rect_to_merge_with {
                    None => coalesced_free_rects.push(work_rect),
                    Some(index_of_rect_to_merge_with) => {
                        let rect_to_merge_with =
                            self.free_list.swap_remove(index_of_rect_to_merge_with);
                        coalesced_free_rects.push(work_rect.union(&rect_to_merge_with));
                        changed = true;
                    }
                }
            }

            self.free_list = coalesced_free_rects;
            if !changed {
                break
            }
        }

        self.dirty = false
    }

    #[allow(dead_code)]
    fn free(&mut self, rect: &Rect<u32>) {
        self.free_list.push(*rect);
        self.dirty = true
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

    fn to_image(&self) -> TextureImage {
        TextureImage {
            texture_id: self.texture_id,
            texel_uv: Rect::new(Point2D::new(self.u0, self.v0), Size2D::new(self.u1 - self.u0,
                                                                            self.v1 - self.v0)),
            pixel_uv: Point2D::new((self.u0 * (TEXTURE_SIZE as f32)) as u32,
                                   (self.v0 * (TEXTURE_SIZE as f32)) as u32),
        }
    }
}

struct TextureCacheArena {
    pages_a8: Vec<TexturePage>,
    pages_rgb8: Vec<TexturePage>,
    pages_rgba8: Vec<TexturePage>,
    alternate_pages_a8: Vec<TexturePage>,
}

impl TextureCacheArena {
    fn new() -> TextureCacheArena {
        TextureCacheArena {
            pages_a8: Vec::new(),
            pages_rgb8: Vec::new(),
            pages_rgba8: Vec::new(),
            alternate_pages_a8: Vec::new(),
        }
    }
}

pub struct TextureCache {
    free_texture_ids: Vec<TextureId>,
    items: HashMap<ImageID, TextureCacheItem, DefaultState<FnvHasher>>,
    arena: TextureCacheArena,
    pending_updates: TextureUpdateList,
}

#[derive(PartialEq, Eq, Debug)]
pub enum AllocationKind {
    TexturePage,
    Standalone,
}

pub struct AllocationResult {
    texture_id: TextureId,
    uv: Point2D<u32>,
    kind: AllocationKind,
}

impl TextureCache {
    pub fn new(free_texture_ids: Vec<TextureId>) -> TextureCache {
        TextureCache {
            free_texture_ids: free_texture_ids,
            items: HashMap::with_hash_state(Default::default()),
            pending_updates: TextureUpdateList::new(),
            arena: TextureCacheArena::new(),
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
                    format: ImageFormat,
                    alternate: bool)
                    -> AllocationResult {
        let (page_list, mode) = match (format, alternate) {
            (ImageFormat::A8, false) => (&mut self.arena.pages_a8, RenderTargetMode::RenderTarget),
            (ImageFormat::A8, true) => {
                (&mut self.arena.alternate_pages_a8, RenderTargetMode::RenderTarget)
            }
            (ImageFormat::RGBA8, false) => {
                (&mut self.arena.pages_rgba8, RenderTargetMode::RenderTarget)
            }
            (ImageFormat::RGB8, false) => (&mut self.arena.pages_rgb8, RenderTargetMode::None),
            (ImageFormat::Invalid, false) | (_, true) => unreachable!(),
        };

        let size = Size2D::new(width, height);
        let location = page_list.last_mut().and_then(|last_page| last_page.allocate(&size));
        let location = match location {
            Some(location) => location,
            None => {
                // We need a new page.
                let texture_size = 1024;
                let texture_id = self.free_texture_ids
                                     .pop()
                                     .expect("TODO: Handle running out of texture IDs!");
                let update_op = TextureUpdate {
                    id: texture_id,
                    op: TextureUpdateOp::Create(texture_size, texture_size, format, mode, None),
                };
                self.pending_updates.push(update_op);
                let page = TexturePage::new(texture_id, texture_size);
                page_list.push(page);

                match page_list.last_mut().unwrap().allocate(&size) {
                    Some(location) => location,
                    None => {
                        // Fall back to standalone texture allocation.
                        let texture_id = self.free_texture_ids
                                             .pop()
                                             .expect("TODO: Handle running out of texture ids!");
                        let cache_item = TextureCacheItem::new(texture_id,
                                                               format,
                                                               x0, y0,
                                                               width, height,
                                                               0.0, 0.0,
                                                               1.0, 1.0);
                        self.items.insert(image_id, cache_item);

                        return AllocationResult {
                            texture_id: texture_id,
                            uv: Point2D::new(0, 0),
                            kind: AllocationKind::Standalone,
                        }
                    }
                }
            }
        };

        let page = page_list.last_mut().unwrap();

        // todo: take into account padding etc.
        // todo: make page index a separate type
        let tx0 = location.x;
        let ty0 = location.y;

        // todo: take into account padding etc.
        let u0 = location.x as f32 / page.texture_size as f32;
        let v0 = location.y as f32 / page.texture_size as f32;
        let u1 = u0 + width as f32 / page.texture_size as f32;
        let v1 = v0 + height as f32 / page.texture_size as f32;
        let cache_item = TextureCacheItem::new(page.texture_id,
                                               format,
                                               x0, y0,
                                               width, height,
                                               u0, v0,
                                               u1, v1);
        self.items.insert(image_id, cache_item);

        AllocationResult {
            texture_id: page.texture_id,
            uv: Point2D::new(tx0, ty0),
            kind: AllocationKind::TexturePage,
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
                                               op.image_format,
                                               false);

                assert!(allocation.kind == AllocationKind::TexturePage);        // TODO: Handle large border radii not fitting in texture cache page

                TextureUpdate {
                    id: allocation.texture_id,
                    op: TextureUpdateOp::Update(allocation.uv.x,
                                                allocation.uv.y,
                                                width,
                                                height,
                                                TextureUpdateDetails::BorderRadius(
                                                    op.outer_radius_x,
                                                    op.outer_radius_y,
                                                    op.inner_radius_x,
                                                    op.inner_radius_y,
                                                    op.inverted)),
                }
            }
            &RasterItem::BoxShadowCorner(ref op) => {
                let size = op.border_radius + op.blur_radius;
                let allocation = self.allocate(image_id,
                                               0,
                                               0,
                                               size.to_nearest_px() as u32,
                                               size.to_nearest_px() as u32,
                                               ImageFormat::A8,
                                               false);

                // TODO(pcwalton): Handle large box shadows not fitting in texture cache page.
                assert!(allocation.kind == AllocationKind::TexturePage);

                TextureUpdate {
                    id: allocation.texture_id,
                    op: TextureUpdateOp::Update(
                        allocation.uv.x,
                        allocation.uv.y,
                        size.to_nearest_px() as u32,
                        size.to_nearest_px() as u32,
                        TextureUpdateDetails::BoxShadowCorner(op.blur_radius,
                                                              op.border_radius,
                                                              op.inverted)),
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
                  insert_op: TextureInsertOp) {

        let result = self.allocate(image_id, x0, y0, width, height, format, false);

        let op = match (result.kind, insert_op) {
            (AllocationKind::TexturePage, TextureInsertOp::Blit(bytes)) => {
                TextureUpdateOp::Update(result.uv.x,
                                        result.uv.y,
                                        width,
                                        height,
                                        TextureUpdateDetails::Blit(bytes))
            }
            (AllocationKind::TexturePage,
             TextureInsertOp::Blur(bytes, glyph_size, blur_radius)) => {
                let unblurred_glyph_image_id = ImageID::new();
                let horizontal_blur_image_id = ImageID::new();
                // TODO(pcwalton): Destroy these!
                self.allocate(unblurred_glyph_image_id,
                              0, 0,
                              glyph_size.width, glyph_size.height,
                              ImageFormat::A8,
                              false);
                self.allocate(horizontal_blur_image_id,
                              0, 0,
                              width, height,
                              ImageFormat::A8,
                              true);
                let unblurred_glyph_item = self.get(unblurred_glyph_image_id);
                let horizontal_blur_item = self.get(horizontal_blur_image_id);
                TextureUpdateOp::Update(
                    result.uv.x, result.uv.y,
                    width, height,
                    TextureUpdateDetails::Blur(bytes,
                                               glyph_size,
                                               blur_radius,
                                               unblurred_glyph_item.to_image(),
                                               horizontal_blur_item.to_image()))
            }
            (AllocationKind::Standalone, TextureInsertOp::Blit(bytes)) => {
                TextureUpdateOp::Create(width, height, format, RenderTargetMode::None, Some(bytes))
            }
            (AllocationKind::Standalone, TextureInsertOp::Blur(_, _, _)) => {
                println!("ERROR: Can't blur with a standalone texture yet!");
                return
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
        match self.items.get(&id) {
            Some(item) => item,
            None => panic!("id {:?} was not cached!", id),
        }
    }
}

pub enum TextureInsertOp {
    Blit(Vec<u8>),
    Blur(Vec<u8>, Size2D<u32>, Au),
}

trait FitsInside {
    fn fits_inside(&self, other: &Self) -> bool;
}

impl FitsInside for Size2D<u32> {
    fn fits_inside(&self, other: &Size2D<u32>) -> bool {
        self.width <= other.width && self.height <= other.height
    }
}

