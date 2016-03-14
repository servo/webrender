/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use device::{ProgramId, TextureId, TextureFilter};
use euclid::{Point2D, Rect, Size2D};
use internal_types::{MAX_RECT, AxisDirection, PackedVertexColorMode, PackedVertexForQuad};
use internal_types::{PackedVertexForTextureCacheUpdate, RectUv, DevicePixel};
use std::sync::atomic::Ordering::SeqCst;
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT};
use texture_cache::{BorderType, TexturePage};
use webrender_traits::{ColorF, ComplexClipRegion};

pub const MAX_MATRICES_PER_BATCH: usize = 32;
pub const MAX_CLIP_RECTS_PER_BATCH: usize = 64;
pub const MAX_TILE_PARAMS_PER_BATCH: usize = 64;       // TODO(gw): Constrain to max FS uniform vectors...
pub const INVALID_TILE_PARAM: u8 = 0;
pub const INVALID_CLIP_RECT_PARAM: usize = 0;

static ID_COUNTER: AtomicUsize = ATOMIC_USIZE_INIT;

#[inline]
pub fn new_id() -> usize {
    ID_COUNTER.fetch_add(1, SeqCst)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct VertexBufferId(pub usize);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixIndex(pub u8);

#[derive(Clone, Debug)]
pub struct OffsetParams {
    pub stacking_context_x0: f32,
    pub stacking_context_y0: f32,
    pub render_target_x0: f32,
    pub render_target_y0: f32,
}

impl OffsetParams {
    pub fn identity() -> OffsetParams {
        OffsetParams {
            stacking_context_x0: 0.0,
            stacking_context_y0: 0.0,
            render_target_x0: 0.0,
            render_target_y0: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TileParams {
    pub u0: f32,
    pub v0: f32,
    pub u_size: f32,
    pub v_size: f32,
}

impl VertexBufferId {
    fn new() -> VertexBufferId {
        VertexBufferId(new_id())
    }
}

pub struct VertexBuffer {
    pub id: VertexBufferId,
    pub instance_count: u32,
    pub vertices: Vec<PackedVertexForQuad>,
}

impl VertexBuffer {
    pub fn new() -> VertexBuffer {
        VertexBuffer {
            id: VertexBufferId::new(),
            instance_count: 0,
            vertices: vec![],
        }
    }
}

#[derive(Debug)]
pub struct Batch {
    pub color_texture_id: TextureId,
    pub mask_texture_id: TextureId,
    pub first_instance: u32,
    pub instance_count: u32,
    pub tile_params: Vec<TileParams>,
    pub clip_rects: Vec<Rect<f32>>,
}

impl Batch {
    pub fn new(color_texture_id: TextureId, mask_texture_id: TextureId, first_instance: u32)
               -> Batch {
        let default_tile_params = vec![
            TileParams {
                u0: 0.0,
                v0: 0.0,
                u_size: 1.0,
                v_size: 1.0,
            }
        ];

        let default_clip_rects = vec![
            Rect::new(Point2D::new(0.0, 0.0), Size2D::new(0.0, 0.0)),
        ];

        Batch {
            color_texture_id: color_texture_id,
            mask_texture_id: mask_texture_id,
            first_instance: first_instance,
            instance_count: 0,
            tile_params: default_tile_params,
            clip_rects: default_clip_rects,
        }
    }

    // TODO: This is quite inefficient - perhaps have a hashmap in addition to the vec...
    fn clip_rect_index(&self, clip_rect: &Rect<f32>) -> Option<usize> {
        self.clip_rects.iter().rposition(|existing_rect| {
            existing_rect.origin.x == clip_rect.origin.x &&
            existing_rect.origin.y == clip_rect.origin.y &&
            existing_rect.size.width == clip_rect.size.width &&
            existing_rect.size.height == clip_rect.size.height
        })
    }

    pub fn can_add_to_batch(&self,
                            color_texture_id: TextureId,
                            mask_texture_id: TextureId,
                            needs_tile_params: bool,
                            clip_in_rect: &Rect<f32>,
                            clip_out_rect: &Option<Rect<f32>>) -> bool {
        let color_texture_ok = color_texture_id == self.color_texture_id;
        let mask_texture_ok = mask_texture_id == self.mask_texture_id;
        let tile_params_ok = !needs_tile_params ||
                             self.tile_params.len() < MAX_TILE_PARAMS_PER_BATCH;

        let used_clip_count = self.clip_rects.len();

        let clip_rects_ok = if used_clip_count + 2 < MAX_CLIP_RECTS_PER_BATCH {
            true
        } else {
            let mut new_clip_count = 0;

            if self.clip_rect_index(clip_in_rect).is_none() {
                new_clip_count += 1;
            }

            if let &Some(ref clip_out_rect) = clip_out_rect {
                if self.clip_rect_index(clip_out_rect).is_none() {
                    new_clip_count += 1;
                }
            }

            used_clip_count + new_clip_count < MAX_CLIP_RECTS_PER_BATCH
        };

        color_texture_ok &&
        mask_texture_ok &&
        tile_params_ok &&
        clip_rects_ok
    }

    pub fn add_draw_item(&mut self,
                         tile_params: Option<TileParams>,
                         clip_in_rect: &Rect<f32>,
                         clip_out_rect: &Option<Rect<f32>>) -> (u8, u8, u8) {
        self.instance_count += 1;

        let tile_params_index = tile_params.map_or(INVALID_TILE_PARAM, |tile_params| {
            let index = self.tile_params.len();
            debug_assert!(index < MAX_TILE_PARAMS_PER_BATCH);
            self.tile_params.push(tile_params);
            index as u8
        });

        let clip_in_rect_index = match self.clip_rect_index(clip_in_rect) {
            Some(clip_in_rect_index) => {
                clip_in_rect_index
            }
            None => {
                let new_index = self.clip_rects.len();
                debug_assert!(new_index < MAX_CLIP_RECTS_PER_BATCH);
                self.clip_rects.push(*clip_in_rect);
                new_index
            }
        } as u8;

        let clip_out_rect_index = match clip_out_rect {
            &Some(ref clip_out_rect) => {
                match self.clip_rect_index(clip_out_rect) {
                    Some(clip_out_rect_index) => {
                        clip_out_rect_index
                    }
                    None => {
                        let new_index = self.clip_rects.len();
                        debug_assert!(new_index < MAX_CLIP_RECTS_PER_BATCH);
                        self.clip_rects.push(*clip_out_rect);
                        new_index
                    }
                }
            }
            &None => {
                INVALID_CLIP_RECT_PARAM
            }
        } as u8;

        (tile_params_index, clip_in_rect_index, clip_out_rect_index)
    }
}

pub struct BatchBuilder<'a> {
    vertex_buffer: &'a mut VertexBuffer,
    batches: Vec<Batch>,
    current_matrix_index: u8,

    clip_offset: Point2D<f32>,

    clip_in_rect_stack: Vec<Rect<f32>>,
    cached_clip_in_rect: Option<Rect<f32>>,

    clip_out_rect: Option<Rect<f32>>,

    // TODO(gw): Support nested complex clip regions!
    pub complex_clip: Option<ComplexClipRegion>,

    pub device_pixel_ratio: f32,
}

impl<'a> BatchBuilder<'a> {
    pub fn new(vertex_buffer: &mut VertexBuffer,
               device_pixel_ratio: f32) -> BatchBuilder {
        BatchBuilder {
            vertex_buffer: vertex_buffer,
            batches: Vec::new(),
            current_matrix_index: 0,
            clip_in_rect_stack: Vec::new(),
            cached_clip_in_rect: Some(MAX_RECT),
            clip_out_rect: None,
            complex_clip: None,
            clip_offset: Point2D::zero(),
            device_pixel_ratio: device_pixel_ratio,
        }
    }

    pub fn finalize(self) -> Vec<Batch> {
        self.batches
    }

    pub fn set_current_clip_rect_offset(&mut self, offset: Point2D<f32>) {
        self.clip_offset = offset;
    }

    pub fn next_draw_list(&mut self) {
        debug_assert!((self.current_matrix_index as usize) < MAX_MATRICES_PER_BATCH);
        self.current_matrix_index += 1;
    }

    // TODO(gw): This is really inefficient to call this every push/pop...
    fn update_clip_in_rect(&mut self) {
        self.cached_clip_in_rect = Some(MAX_RECT);

        for rect in &self.clip_in_rect_stack {
            self.cached_clip_in_rect = self.cached_clip_in_rect.unwrap().intersection(rect);
            if self.cached_clip_in_rect.is_none() {
                return;
            }
        }
    }

    pub fn push_clip_in_rect(&mut self, rect: &Rect<f32>) {
        let rect = rect.translate(&self.clip_offset);
        self.clip_in_rect_stack.push(rect);
        self.update_clip_in_rect();
    }

    pub fn pop_clip_in_rect(&mut self) {
        self.clip_in_rect_stack.pop();
        self.update_clip_in_rect();
    }

    pub fn set_clip_out_rect(&mut self, rect: Option<Rect<f32>>) -> Option<Rect<f32>> {
        let rect = rect.map(|rect| {
            rect.translate(&self.clip_offset)
        });
        let old_rect = self.clip_out_rect.take();
        self.clip_out_rect = rect;
        old_rect
    }

    pub fn push_complex_clip(&mut self, clip: &[ComplexClipRegion]) {
        // TODO(gw): Handle nested complex clips!
        if clip.len() > 0 {
            self.complex_clip = Some(clip[0]);
        } else {
            self.complex_clip = None;
        }
    }

    pub fn pop_complex_clip(&mut self) {
        self.complex_clip = None;
    }

    // Colors are in the order: top left, top right, bottom right, bottom left.
    pub fn add_rectangle(&mut self,
                         color_texture_id: TextureId,
                         mask_texture_id: TextureId,
                         pos_rect: &Rect<f32>,
                         uv_rect: &RectUv<f32>,
                         muv_rect: &RectUv<DevicePixel>,
                         colors: &[ColorF; 4],
                         color_mode: PackedVertexColorMode,
                         tile_params: Option<TileParams>) {
        let (tile_params_index,
             clip_in_rect_index,
             clip_out_rect_index) = match self.cached_clip_in_rect {
            None => return,
            Some(ref clip_in_rect) => {
                let need_new_batch = match self.batches.last_mut() {
                    Some(batch) => {
                        !batch.can_add_to_batch(color_texture_id,
                                                mask_texture_id,
                                                tile_params.is_some(),
                                                clip_in_rect,
                                                &self.clip_out_rect)
                    }
                    None => {
                        true
                    }
                };

                if need_new_batch {
                    self.batches.push(Batch::new(color_texture_id,
                                                 mask_texture_id,
                                                 self.vertex_buffer.instance_count));
                }

                self.batches.last_mut().unwrap().add_draw_item(tile_params,
                                                               clip_in_rect,
                                                               &self.clip_out_rect)
            }
        };

        let mut vertex = PackedVertexForQuad::new(pos_rect, colors, uv_rect, muv_rect, color_mode);
        vertex.matrix_index = self.current_matrix_index;
        vertex.tile_params_index = vertex.tile_params_index | tile_params_index;
        vertex.clip_in_rect_index = clip_in_rect_index;
        vertex.clip_out_rect_index = clip_out_rect_index;

        self.push_vertex_for_rectangle(vertex);

        self.vertex_buffer.instance_count += 1
    }

    fn push_vertex_for_rectangle(&mut self, vertex: PackedVertexForQuad) {
        self.vertex_buffer.vertices.push(vertex);
    }
}

// Information needed to blit an item from a raster op batch target to final destination.
pub struct BlitJob {
    pub dest_texture_id: TextureId,
    pub size: Size2D<u32>,
    pub src_origin: Point2D<u32>,
    pub dest_origin: Point2D<u32>,
    pub border_type: BorderType,
}

/// A batch for raster jobs.
pub struct RasterBatch {
    pub program_id: ProgramId,
    pub blur_direction: Option<AxisDirection>,
    pub color_texture_id: TextureId,
    pub vertices: Vec<PackedVertexForTextureCacheUpdate>,
    pub indices: Vec<u16>,
    pub page_allocator: TexturePage,
    pub dest_texture_id: TextureId,
    pub blit_jobs: Vec<BlitJob>,
}

impl RasterBatch {
    pub fn new(target_texture_id: TextureId,
               target_texture_size: u32,
               program_id: ProgramId,
               blur_direction: Option<AxisDirection>,
               color_texture_id: TextureId,
               dest_texture_id: TextureId)
               -> RasterBatch {
        RasterBatch {
            program_id: program_id,
            blur_direction: blur_direction,
            color_texture_id: color_texture_id,
            dest_texture_id: dest_texture_id,
            vertices: Vec::new(),
            indices: Vec::new(),
            page_allocator: TexturePage::new(target_texture_id, target_texture_size),
            blit_jobs: Vec::new(),
        }
    }

    pub fn add_rect_if_possible<F>(&mut self,
                                   dest_texture_id: TextureId,
                                   color_texture_id: TextureId,
                                   program_id: ProgramId,
                                   blur_direction: Option<AxisDirection>,
                                   dest_rect: &Rect<u32>,
                                   border_type: BorderType,
                                   f: &F) -> bool
                                   where F: Fn(&Rect<f32>) -> [PackedVertexForTextureCacheUpdate; 4] {
        // TODO(gw): How to detect / handle if border type is single pixel but not in an atlas!?

        let batch_ok = program_id == self.program_id &&
            blur_direction == self.blur_direction &&
            color_texture_id == self.color_texture_id &&
            dest_texture_id == self.dest_texture_id;

        if batch_ok {
            let origin = self.page_allocator.allocate(&dest_rect.size, TextureFilter::Linear);

            if let Some(origin) = origin {
                let vertices = f(&Rect::new(Point2D::new(origin.x as f32,
                                                         origin.y as f32),
                                            Size2D::new(dest_rect.size.width as f32,
                                                        dest_rect.size.height as f32)));
                let mut i = 0;
                let index_offset = self.vertices.len();
                while i < vertices.len() {
                    let index_base = (index_offset + i) as u16;
                    self.indices.push(index_base + 0);
                    self.indices.push(index_base + 1);
                    self.indices.push(index_base + 2);
                    self.indices.push(index_base + 2);
                    self.indices.push(index_base + 3);
                    self.indices.push(index_base + 1);
                    i += 4;
                }

                self.vertices.extend_from_slice(&vertices);

                self.blit_jobs.push(BlitJob {
                    dest_texture_id: dest_texture_id,
                    size: dest_rect.size,
                    dest_origin: dest_rect.origin,
                    src_origin: origin,
                    border_type: border_type,
                });

                return true;
            }
        }

        false
    }
}
