use app_units::Au;
use device::{MAX_TEXTURE_SIZE, TextureId, TextureIndex};
use euclid::{Point2D, Rect, Size2D};
use fnv::FnvHasher;
use freelist::{FreeList, FreeListItem, FreeListItemId};
use internal_types::{TextureTarget, TextureUpdate, TextureUpdateOp, TextureUpdateDetails};
use internal_types::{RasterItem, RenderTargetMode, TextureImage, TextureUpdateList};
use internal_types::{BasicRotationAngle, RectUv};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::collections::hash_state::DefaultState;
use std::mem;
use tessellator::BorderCornerTessellation;
use util;
use webrender_traits::ImageFormat;

/// The number of bytes we're allowed to use for a texture.
const MAX_BYTES_PER_TEXTURE: u32 = 64 * 1024 * 1024;

/// The number of RGBA pixels we're allowed to use for a texture.
const MAX_RGBA_PIXELS_PER_TEXTURE: u32 = MAX_BYTES_PER_TEXTURE / 4;

/// The square root of the number of RGBA pixels we're allowed to use for a texture, rounded down.
/// to the next power of two.
const SQRT_MAX_RGBA_PIXELS_PER_TEXTURE: u32 = 4096;

pub type TextureCacheItemId = FreeListItemId;

pub enum BorderType {
    NoBorder,
    SinglePixel,
}

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
    texture_index: TextureIndex,
    dirty: bool,
}

impl TexturePage {
    fn new(texture_id: TextureId, texture_index: TextureIndex, texture_size: u32) -> TexturePage {
        TexturePage {
            texture_id: texture_id,
            texture_index: texture_index,
            texture_size: texture_size,
            free_list: vec![
                Rect::new(Point2D::new(0, 0), Size2D::new(texture_size, texture_size))
            ],
            dirty: false,
        }
    }

    fn find_index_of_best_rect(&self, requested_dimensions: &Size2D<u32>) -> Option<usize> {
        let mut smallest_index_and_area = None;
        for (candidate_index, candidate_rect) in self.free_list.iter().enumerate() {
            if !requested_dimensions.fits_inside(&candidate_rect.size) {
                continue
            }

            let candidate_area = candidate_rect.size.width * candidate_rect.size.height;
            match smallest_index_and_area {
                Some((_, smallest_area_so_far)) if candidate_area > smallest_area_so_far => {}
                _ => smallest_index_and_area = Some((candidate_index, candidate_area)),
            }
        }
        smallest_index_and_area.map(|(index, _)| index)
    }

    fn allocate(&mut self, requested_dimensions: &Size2D<u32>) -> Option<Point2D<u32>> {
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
        let chosen_rect = self.free_list.swap_remove(index);
        let candidate_free_rect_to_right =
            Rect::new(Point2D::new(chosen_rect.origin.x + requested_dimensions.width,
                                   chosen_rect.origin.y),
                      Size2D::new(chosen_rect.size.width - requested_dimensions.width,
                                  requested_dimensions.height));
        let candidate_free_rect_to_bottom =
            Rect::new(Point2D::new(chosen_rect.origin.x,
                                   chosen_rect.origin.y + requested_dimensions.height),
                      Size2D::new(requested_dimensions.width,
                                  chosen_rect.size.height - requested_dimensions.height));
        let candidate_free_rect_to_right_area = candidate_free_rect_to_right.size.width *
            candidate_free_rect_to_right.size.height;
        let candidate_free_rect_to_bottom_area = candidate_free_rect_to_bottom.size.width *
            candidate_free_rect_to_bottom.size.height;

        // Guillotine the rectangle.
        let new_free_rect_to_right;
        let new_free_rect_to_bottom;
        if candidate_free_rect_to_right_area > candidate_free_rect_to_bottom_area {
            new_free_rect_to_right = Rect::new(candidate_free_rect_to_right.origin,
                                               Size2D::new(candidate_free_rect_to_right.size.width,
                                                           chosen_rect.size.height));
            new_free_rect_to_bottom = candidate_free_rect_to_bottom
        } else {
            new_free_rect_to_right = candidate_free_rect_to_right;
            new_free_rect_to_bottom =
                Rect::new(candidate_free_rect_to_bottom.origin,
                          Size2D::new(chosen_rect.size.width,
                                      candidate_free_rect_to_bottom.size.height))
        }

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

// TODO(gw): This is used to store data specific to glyphs.
//           Perhaps find a better place to store it.
#[derive(Debug, Clone)]
pub struct TextureCacheItemUserData {
    pub x0: i32,
    pub y0: i32,
}

#[derive(Debug, Clone)]
pub struct TextureCacheItem {
    // Identifies the texture and array slice
    pub texture_id: TextureId,
    pub texture_index: TextureIndex,

    // Custom data - unused by texture cache itself
    pub user_data: TextureCacheItemUserData,

    // The texture coordinates for this item
    pub uv_rect: RectUv,

    // The size of the actual allocated rectangle,
    // and the requested size. The allocated size
    // is the same as the requested in most cases,
    // unless the item has a border added for
    // bilinear filtering / texture bleeding purposes.
    pub allocated_rect: Rect<u32>,
    pub requested_rect: Rect<u32>,
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
    fn new(texture_id: TextureId,
           texture_index: TextureIndex,
           user_x0: i32, user_y0: i32,
           allocated_rect: Rect<u32>,
           requested_rect: Rect<u32>,
           uv_rect: RectUv)
           -> TextureCacheItem {
        TextureCacheItem {
            texture_id: texture_id,
            texture_index: texture_index,
            uv_rect: uv_rect,
            user_data: TextureCacheItemUserData {
                x0: user_x0,
                y0: user_y0,
            },
            allocated_rect: allocated_rect,
            requested_rect: requested_rect,
        }
    }

    fn to_image(&self) -> TextureImage {
        let texture_size = texture_size() as f32;
        TextureImage {
            texture_id: self.texture_id,
            texture_index: self.texture_index,
            texel_uv: Rect::new(Point2D::new(self.uv_rect.top_left.x,
                                             self.uv_rect.top_left.y),
                                Size2D::new(self.uv_rect.bottom_right.x - self.uv_rect.top_left.x,
                                            self.uv_rect.bottom_right.y - self.uv_rect.top_left.y)),
            pixel_uv: Point2D::new((self.uv_rect.top_left.x * texture_size) as u32,
                                   (self.uv_rect.top_left.y * texture_size) as u32),
        }
    }
}

struct TextureCacheArena {
    pages_a8: Vec<TexturePage>,
    pages_rgb8: Vec<TexturePage>,
    pages_rgba8: Vec<TexturePage>,
    alternate_pages_a8: Vec<TexturePage>,
    alternate_pages_rgba8: Vec<TexturePage>,
}

impl TextureCacheArena {
    fn new() -> TextureCacheArena {
        TextureCacheArena {
            pages_a8: Vec::new(),
            pages_rgb8: Vec::new(),
            pages_rgba8: Vec::new(),
            alternate_pages_a8: Vec::new(),
            alternate_pages_rgba8: Vec::new(),
        }
    }
}

pub struct TextureCache {
    free_texture_ids: Vec<TextureId>,
    free_texture_levels: HashMap<ImageFormat, Vec<FreeTextureLevel>, DefaultState<FnvHasher>>,
    alternate_free_texture_levels: HashMap<ImageFormat,
                                           Vec<FreeTextureLevel>,
                                           DefaultState<FnvHasher>>,
    items: FreeList<TextureCacheItem>,
    arena: TextureCacheArena,
    pending_updates: TextureUpdateList,
}

#[derive(PartialEq, Eq, Debug)]
pub enum AllocationKind {
    TexturePage,
    Standalone,
}

pub struct AllocationResult {
    kind: AllocationKind,
    item: TextureCacheItem,
}

impl TextureCache {
    pub fn new(free_texture_ids: Vec<TextureId>) -> TextureCache {
        TextureCache {
            free_texture_ids: free_texture_ids,
            free_texture_levels: HashMap::with_hash_state(Default::default()),
            alternate_free_texture_levels: HashMap::with_hash_state(Default::default()),
            items: FreeList::new(),
            pending_updates: TextureUpdateList::new(),
            arena: TextureCacheArena::new(),
        }
    }

    pub fn pending_updates(&mut self) -> TextureUpdateList {
        mem::replace(&mut self.pending_updates, TextureUpdateList::new())
    }

    // TODO(gw): This API is a bit ugly (having to allocate an ID and
    //           then use it). But it has to be that way for now due to
    //           how the raster_jobs code works.
    pub fn new_item_id(&mut self) -> TextureCacheItemId {
        let new_item = TextureCacheItem {
            uv_rect: RectUv {
                top_left: Point2D::zero(),
                top_right: Point2D::zero(),
                bottom_left: Point2D::zero(),
                bottom_right: Point2D::zero(),
            },
            user_data: TextureCacheItemUserData {
                x0: 0,
                y0: 0,
            },
            allocated_rect: Rect::zero(),
            requested_rect: Rect::zero(),
            texture_id: TextureId::invalid(),
            texture_index: TextureIndex(0),
        };
        self.items.insert(new_item)
    }

    pub fn allocate_render_target(&mut self,
                                  target: TextureTarget,
                                  width: u32,
                                  height: u32,
                                  levels: u32,
                                  format: ImageFormat)
                                  -> TextureId {
        let texture_id = self.free_texture_ids
                             .pop()
                             .expect("TODO: Handle running out of texture IDs!");
        let update_op = TextureUpdate {
            id: texture_id,
            index: TextureIndex(0),
            op: TextureUpdateOp::Create(target,
                                        width,
                                        height,
                                        levels,
                                        format,
                                        RenderTargetMode::RenderTarget,
                                        None),
        };
        self.pending_updates.push(update_op);
        texture_id
    }

    pub fn free_render_target(&mut self, texture_id: TextureId) {
        self.free_texture_ids.push(texture_id);
        let update_op = TextureUpdate {
            id: texture_id,
            index: TextureIndex(0),
            op: TextureUpdateOp::DeinitRenderTarget(texture_id),
        };
        self.pending_updates.push(update_op);
    }

    pub fn allocate(&mut self,
                    image_id: TextureCacheItemId,
                    user_x0: i32,
                    user_y0: i32,
                    requested_width: u32,
                    requested_height: u32,
                    format: ImageFormat,
                    alternate: bool,
                    border_type: BorderType)
                    -> AllocationResult {
        let (page_list, mode) = match (format, alternate) {
            (ImageFormat::A8, false) => (&mut self.arena.pages_a8, RenderTargetMode::RenderTarget),
            (ImageFormat::A8, true) => {
                (&mut self.arena.alternate_pages_a8, RenderTargetMode::RenderTarget)
            }
            (ImageFormat::RGBA8, false) => {
                (&mut self.arena.pages_rgba8, RenderTargetMode::RenderTarget)
            }
            (ImageFormat::RGBA8, true) => {
                (&mut self.arena.alternate_pages_rgba8, RenderTargetMode::RenderTarget)
            }
            (ImageFormat::RGB8, false) => (&mut self.arena.pages_rgb8, RenderTargetMode::None),
            (ImageFormat::Invalid, false) | (_, true) => unreachable!(),
        };

        let border_size = match border_type {
            BorderType::NoBorder => 0,
            BorderType::SinglePixel => 1,
        };
        let requested_size = Size2D::new(requested_width, requested_height);
        let allocation_size = Size2D::new(requested_width + border_size * 2,
                                          requested_height + border_size * 2);
        let location = page_list.last_mut().and_then(|last_page| last_page.allocate(&allocation_size));
        let location = match location {
            Some(location) => location,
            None => {
                // We need a new page.
                let texture_size = texture_size();
                let (texture_id, texture_index) = {
                    let free_texture_levels_entry = if !alternate {
                        self.free_texture_levels.entry(format)
                    } else {
                        self.alternate_free_texture_levels.entry(format)
                    };
                    let mut free_texture_levels = match free_texture_levels_entry {
                        Entry::Vacant(entry) => entry.insert(Vec::new()),
                        Entry::Occupied(entry) => entry.into_mut(),
                    };
                    if free_texture_levels.is_empty() {
                        create_new_texture_page(&mut self.pending_updates,
                                                &mut self.free_texture_ids,
                                                &mut free_texture_levels,
                                                texture_size,
                                                format,
                                                mode);
                    }
                    let free_texture_level = free_texture_levels.pop().unwrap();
                    (free_texture_level.texture_id, free_texture_level.texture_index)
                };

                let page = TexturePage::new(texture_id, texture_index, texture_size);
                page_list.push(page);

                match page_list.last_mut().unwrap().allocate(&allocation_size) {
                    Some(location) => location,
                    None => {
                        // Fall back to standalone texture allocation.
                        let texture_id = self.free_texture_ids
                                             .pop()
                                             .expect("TODO: Handle running out of texture ids!");
                        let cache_item = TextureCacheItem::new(texture_id,
                                                               TextureIndex(0),
                                                               user_x0, user_y0,
                                                               Rect::new(Point2D::zero(),
                                                                         requested_size),
                                                               Rect::new(Point2D::zero(),
                                                                         requested_size),
                                                               RectUv {
                                                                    top_left: Point2D::new(0.0, 0.0),
                                                                    top_right: Point2D::new(1.0, 0.0),
                                                                    bottom_left: Point2D::new(0.0, 1.0),
                                                                    bottom_right: Point2D::new(1.0, 1.0),
                                                               });
                        *self.items.get_mut(image_id) = cache_item;

                        return AllocationResult {
                            item: self.items.get(image_id).clone(),
                            kind: AllocationKind::Standalone,
                        }
                    }
                }
            }
        };

        let page = page_list.last_mut().unwrap();

        let allocated_rect = Rect::new(location, allocation_size);
        let requested_rect = Rect::new(Point2D::new(location.x + border_size,
                                                    location.y + border_size),
                                       requested_size);

        let u0 = requested_rect.origin.x as f32 / page.texture_size as f32;
        let v0 = requested_rect.origin.y as f32 / page.texture_size as f32;
        let u1 = u0 + requested_size.width as f32 / page.texture_size as f32;
        let v1 = v0 + requested_size.height as f32 / page.texture_size as f32;
        let cache_item = TextureCacheItem::new(page.texture_id,
                                               page.texture_index,
                                               user_x0, user_y0,
                                               allocated_rect,
                                               requested_rect,
                                               RectUv {
                                                    top_left: Point2D::new(u0, v0),
                                                    top_right: Point2D::new(u1, v0),
                                                    bottom_left: Point2D::new(u0, v1),
                                                    bottom_right: Point2D::new(u1, v1)
                                               });
        *self.items.get_mut(image_id) = cache_item;

        // TODO(pcwalton): Select a texture index if we're using texture arrays.
        AllocationResult {
            item: self.items.get(image_id).clone(),
            kind: AllocationKind::TexturePage,
        }
    }

    pub fn insert_raster_op(&mut self,
                            image_id: TextureCacheItemId,
                            item: &RasterItem) {
        let update_op = match item {
            &RasterItem::BorderRadius(ref op) => {
                let rect = Rect::new(Point2D::new(0.0, 0.0),
                                     Size2D::new(op.outer_radius_x.to_f32_px(),
                                                 op.outer_radius_y.to_f32_px()));
                let tessellated_rect = rect.tessellate_border_corner(
                    &Size2D::new(op.outer_radius_x.to_f32_px(), op.outer_radius_y.to_f32_px()),
                    &Size2D::new(op.inner_radius_x.to_f32_px(), op.inner_radius_y.to_f32_px()),
                    BasicRotationAngle::Upright,
                    op.index);
                let width = tessellated_rect.size.width.round() as u32;
                let height = tessellated_rect.size.height.round() as u32;

                let allocation = self.allocate(image_id,
                                               0,
                                               0,
                                               width,
                                               height,
                                               op.image_format,
                                               false,
                                               BorderType::NoBorder);

                assert!(allocation.kind == AllocationKind::TexturePage);        // TODO: Handle large border radii not fitting in texture cache page

                TextureUpdate {
                    id: allocation.item.texture_id,
                    index: allocation.item.texture_index,
                    op: TextureUpdateOp::Update(allocation.item.requested_rect.origin.x,
                                                allocation.item.requested_rect.origin.y,
                                                width,
                                                height,
                                                TextureUpdateDetails::BorderRadius(
                                                    op.outer_radius_x,
                                                    op.outer_radius_y,
                                                    op.inner_radius_x,
                                                    op.inner_radius_y,
                                                    op.index,
                                                    op.inverted)),
                }
            }
            &RasterItem::BoxShadow(ref op) => {
                let allocation = self.allocate(image_id,
                                               0,
                                               0,
                                               op.raster_size.to_nearest_px() as u32,
                                               op.raster_size.to_nearest_px() as u32,
                                               ImageFormat::RGBA8,
                                               false,
                                               BorderType::NoBorder);

                // TODO(pcwalton): Handle large box shadows not fitting in texture cache page.
                assert!(allocation.kind == AllocationKind::TexturePage);

                TextureUpdate {
                    id: allocation.item.texture_id,
                    index: allocation.item.texture_index,
                    op: TextureUpdateOp::Update(
                        allocation.item.requested_rect.origin.x,
                        allocation.item.requested_rect.origin.y,
                        op.raster_size.to_nearest_px() as u32,
                        op.raster_size.to_nearest_px() as u32,
                        TextureUpdateDetails::BoxShadow(op.blur_radius, op.part, op.inverted)),
                }
            }
        };

        self.pending_updates.push(update_op);
    }

    pub fn update(&mut self,
                  image_id: TextureCacheItemId,
                  width: u32,
                  height: u32,
                  _format: ImageFormat,
                  bytes: Vec<u8>) {
        let existing_item = self.items.get(image_id);

        // TODO(gw): Handle updates to size/format!
        debug_assert!(existing_item.requested_rect.size.width == width);
        debug_assert!(existing_item.requested_rect.size.height == height);

        let op = TextureUpdateOp::Update(existing_item.requested_rect.origin.x,
                                         existing_item.requested_rect.origin.y,
                                         width,
                                         height,
                                         TextureUpdateDetails::Blit(bytes));

        let update_op = TextureUpdate {
            id: existing_item.texture_id,
            index: existing_item.texture_index,
            op: op,
        };

        self.pending_updates.push(update_op);
    }

    pub fn insert(&mut self,
                  image_id: TextureCacheItemId,
                  x0: i32,
                  y0: i32,
                  width: u32,
                  height: u32,
                  format: ImageFormat,
                  insert_op: TextureInsertOp) {

        let result = self.allocate(image_id, x0, y0, width, height, format, false, BorderType::NoBorder);

        let op = match (result.kind, insert_op) {
            (AllocationKind::TexturePage, TextureInsertOp::Blit(bytes)) => {
                TextureUpdateOp::Update(result.item.requested_rect.origin.x,
                                        result.item.requested_rect.origin.y,
                                        width,
                                        height,
                                        TextureUpdateDetails::Blit(bytes))
            }
            (AllocationKind::TexturePage,
             TextureInsertOp::Blur(bytes, glyph_size, blur_radius)) => {
                let unblurred_glyph_image_id = self.new_item_id();
                let horizontal_blur_image_id = self.new_item_id();
                // TODO(pcwalton): Destroy these!
                self.allocate(unblurred_glyph_image_id,
                              0, 0,
                              glyph_size.width, glyph_size.height,
                              ImageFormat::A8,
                              false,
                              BorderType::NoBorder);
                self.allocate(horizontal_blur_image_id,
                              0, 0,
                              width, height,
                              ImageFormat::A8,
                              true,
                              BorderType::NoBorder);
                let unblurred_glyph_item = self.get(unblurred_glyph_image_id);
                let horizontal_blur_item = self.get(horizontal_blur_image_id);
                TextureUpdateOp::Update(
                    result.item.requested_rect.origin.x,
                    result.item.requested_rect.origin.y,
                    width,
                    height,
                    TextureUpdateDetails::Blur(bytes,
                                               glyph_size,
                                               blur_radius,
                                               unblurred_glyph_item.to_image(),
                                               horizontal_blur_item.to_image()))
            }
            (AllocationKind::Standalone, TextureInsertOp::Blit(bytes)) => {
                TextureUpdateOp::Create(self.texture_target_for_standalone_texture(),
                                        width,
                                        height,
                                        1,
                                        format,
                                        RenderTargetMode::None,
                                        Some(bytes))
            }
            (AllocationKind::Standalone, TextureInsertOp::Blur(_, _, _)) => {
                println!("ERROR: Can't blur with a standalone texture yet!");
                return
            }
        };

        let update_op = TextureUpdate {
            id: result.item.texture_id,
            index: result.item.texture_index,
            op: op,
        };

        self.pending_updates.push(update_op);
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    pub fn texture_target_for_standalone_texture(&self) -> TextureTarget {
        TextureTarget::Texture2D
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    pub fn texture_target_for_standalone_texture(&self) -> TextureTarget {
        TextureTarget::TextureArray
    }

    pub fn get(&self, id: TextureCacheItemId) -> &TextureCacheItem {
        self.items.get(id)
    }

}

#[cfg(any(target_os = "android", target_os = "gonk"))]
fn texture_create_op(texture_size: u32, levels: u32, format: ImageFormat, mode: RenderTargetMode)
                     -> TextureUpdateOp {
    debug_assert!(levels == 1);
    TextureUpdateOp::Create(TextureTarget::Texture2D,
                            texture_size,
                            texture_size,
                            levels,
                            format,
                            mode,
                            None)
}

#[cfg(not(any(target_os = "android", target_os = "gonk")))]
fn texture_create_op(texture_size: u32, levels: u32, format: ImageFormat, mode: RenderTargetMode)
                     -> TextureUpdateOp {
    TextureUpdateOp::Create(TextureTarget::TextureArray,
                            texture_size,
                            texture_size,
                            levels,
                            format,
                            mode,
                            None)
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

/// FIXME(pcwalton): Would probably be more efficient as a bit vector.
#[derive(Clone, Copy)]
pub struct FreeTextureLevel {
    texture_id: TextureId,
    texture_index: TextureIndex,
}

fn create_new_texture_page(pending_updates: &mut TextureUpdateList,
                           free_texture_ids: &mut Vec<TextureId>,
                           free_texture_levels: &mut Vec<FreeTextureLevel>,
                           texture_size: u32,
                           format: ImageFormat,
                           mode: RenderTargetMode) {
    let texture_id = free_texture_ids.pop().expect("TODO: Handle running out of texture IDs!");
    let update_op = TextureUpdate {
        id: texture_id,
        index: TextureIndex(0),
        op: texture_create_op(texture_size, levels_per_texture() as u32, format, mode),
    };
    pending_updates.push(update_op);

    for i in 0..levels_per_texture() {
        free_texture_levels.push(FreeTextureLevel {
            texture_id: texture_id,
            texture_index: TextureIndex(i),
        })
    }
}

/// Returns the number of pixels on a side we're allowed to use for our texture atlases.
fn texture_size() -> u32 {
    let max_hardware_texture_size = *MAX_TEXTURE_SIZE as u32;
    if max_hardware_texture_size * max_hardware_texture_size > MAX_RGBA_PIXELS_PER_TEXTURE {
        SQRT_MAX_RGBA_PIXELS_PER_TEXTURE
    } else {
        max_hardware_texture_size
    }
}

/// Returns the number of texture levels we're allowed to use.
fn levels_per_texture() -> u8 {
    let texture_size = texture_size();
    (MAX_RGBA_PIXELS_PER_TEXTURE / (texture_size * texture_size)) as u8
}

