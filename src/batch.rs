use device::{ProgramId, TextureId};
use euclid::{Point2D, Rect, Size2D};
use internal_types::{MAX_RECT, AxisDirection, PackedVertex, PackedVertexForTextureCacheUpdate, Primitive};
use std::sync::atomic::Ordering::SeqCst;
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT};
use std::sync::Arc;
use std::u16;
use webrender_traits::{ComplexClipRegion};

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
    pub vertices: Vec<PackedVertex>,
    pub indices: Vec<u16>,
}

impl VertexBuffer {
    pub fn new() -> VertexBuffer {
        VertexBuffer {
            id: VertexBufferId::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct Batch {
    pub color_texture_id: TextureId,
    pub mask_texture_id: TextureId,
    pub first_vertex: u32,
    pub index_count: u16,
    pub tile_params: Vec<TileParams>,
    pub clip_rects: Vec<Rect<f32>>,
}

impl Batch {
    pub fn new(color_texture_id: TextureId, mask_texture_id: TextureId, first_vertex: u32)
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
            first_vertex: first_vertex,
            index_count: 0,
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
                            index_count: u16,
                            needs_tile_params: bool,
                            clip_in_rect: &Rect<f32>,
                            clip_out_rect: &Option<Rect<f32>>) -> bool {
        let color_texture_ok = color_texture_id == self.color_texture_id;
        let mask_texture_ok = mask_texture_id == self.mask_texture_id;
        let index_count_ok = (self.index_count as u32 + index_count as u32) < u16::MAX as u32;
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
        index_count_ok &&
        tile_params_ok &&
        clip_rects_ok
    }

    pub fn add_draw_item(&mut self,
                         index_count: u16,
                         tile_params: Option<TileParams>,
                         clip_in_rect: &Rect<f32>,
                         clip_out_rect: &Option<Rect<f32>>) -> (u8, u8, u8) {
        self.index_count += index_count;

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

    clip_in_rect_stack: Vec<Rect<f32>>,
    cached_clip_in_rect: Option<Rect<f32>>,

    clip_out_rect: Option<Rect<f32>>,

    // TODO(gw): Support nested complex clip regions!
    pub complex_clip: Option<ComplexClipRegion>,
}

impl<'a> BatchBuilder<'a> {
    pub fn new(vertex_buffer: &mut VertexBuffer) -> BatchBuilder {
        BatchBuilder {
            vertex_buffer: vertex_buffer,
            batches: Vec::new(),
            current_matrix_index: 0,
            clip_in_rect_stack: Vec::new(),
            cached_clip_in_rect: Some(MAX_RECT),
            clip_out_rect: None,
            complex_clip: None,
        }
    }

    pub fn finalize(self) -> Vec<Arc<Batch>> {
        self.batches.into_iter().map(|batch| Arc::new(batch)).collect()
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
        self.clip_in_rect_stack.push(*rect);
        self.update_clip_in_rect();
    }

    pub fn pop_clip_in_rect(&mut self) {
        self.clip_in_rect_stack.pop();
        self.update_clip_in_rect();
    }

    pub fn set_clip_out_rect(&mut self, rect: Option<Rect<f32>>) -> Option<Rect<f32>> {
        let old_rect = self.clip_out_rect.take();
        self.clip_out_rect = rect;
        old_rect
    }

    pub fn push_complex_clip(&mut self, clip: &Vec<ComplexClipRegion>) {
        // TODO(gw): Handle nested complex clips!
        debug_assert!(clip.len() == 0 || clip.len() == 1);
        if clip.len() == 1 {
            self.complex_clip = Some(clip[0]);
        } else {
            self.complex_clip = None;
        }
    }

    pub fn pop_complex_clip(&mut self) {
        self.complex_clip = None;
    }

    pub fn add_draw_item(&mut self,
                         color_texture_id: TextureId,
                         mask_texture_id: TextureId,
                         primitive: Primitive,
                         vertices: &mut [PackedVertex],
                         tile_params: Option<TileParams>,) {
        if let Some(ref clip_in_rect) = self.cached_clip_in_rect {
            let index_count = match primitive {
                Primitive::Rectangles => {
                    (vertices.len() / 4 * 6) as u16
                }
                Primitive::Triangles => vertices.len() as u16,
            };

            let need_new_batch = match self.batches.last_mut() {
                Some(batch) => {
                    !batch.can_add_to_batch(color_texture_id,
                                            mask_texture_id,
                                            index_count,
                                            tile_params.is_some(),
                                            clip_in_rect,
                                            &self.clip_out_rect)
                }
                None => {
                    true
                }
            };

            let index_offset = self.vertex_buffer.vertices.len();

            if need_new_batch {
                self.batches.push(Batch::new(color_texture_id,
                                             mask_texture_id,
                                             self.vertex_buffer.indices.len() as u32));
            }

            match primitive {
                Primitive::Rectangles => {
                    for i in (0..vertices.len()).step_by(4) {
                        let index_base = (index_offset + i) as u16;
                        debug_assert!(index_base as usize == index_offset + i);
                        self.vertex_buffer.indices.push(index_base + 0);
                        self.vertex_buffer.indices.push(index_base + 1);
                        self.vertex_buffer.indices.push(index_base + 2);
                        self.vertex_buffer.indices.push(index_base + 2);
                        self.vertex_buffer.indices.push(index_base + 3);
                        self.vertex_buffer.indices.push(index_base + 1);
                    }
                }
                Primitive::Triangles => {
                    for i in (0..vertices.len()).step_by(3) {
                        let index_base = (index_offset + i) as u16;
                        debug_assert!(index_base as usize == index_offset + i);
                        self.vertex_buffer.indices.push(index_base + 0);
                        self.vertex_buffer.indices.push(index_base + 1);
                        self.vertex_buffer.indices.push(index_base + 2);
                    }
                }
            }

            let (tile_params_index,
                 clip_in_rect_index,
                 clip_out_rect_index) = self.batches.last_mut().unwrap().add_draw_item(index_count,
                                                                                       tile_params,
                                                                                       clip_in_rect,
                                                                                       &self.clip_out_rect);

            for vertex in vertices.iter_mut() {
                vertex.matrix_index = self.current_matrix_index;
                vertex.tile_params_index = tile_params_index;
                vertex.clip_in_rect_index = clip_in_rect_index;
                vertex.clip_out_rect_index = clip_out_rect_index;
            }

            self.vertex_buffer.vertices.extend_from_slice(vertices);

            // TODO(gw): Handle exceeding u16 index buffer!
            debug_assert!(self.vertex_buffer.vertices.len() < 65535);
        }
    }
}

/// A batch for raster jobs.
pub struct RasterBatch {
    pub program_id: ProgramId,
    pub blur_direction: Option<AxisDirection>,
    pub dest_texture_id: TextureId,
    pub color_texture_id: TextureId,
    pub vertices: Vec<PackedVertexForTextureCacheUpdate>,
    pub indices: Vec<u16>,
}

impl RasterBatch {
    pub fn new(program_id: ProgramId,
               blur_direction: Option<AxisDirection>,
               dest_texture_id: TextureId,
               color_texture_id: TextureId)
               -> RasterBatch {
        debug_assert!(dest_texture_id != color_texture_id);
        RasterBatch {
            program_id: program_id,
            blur_direction: blur_direction,
            dest_texture_id: dest_texture_id,
            color_texture_id: color_texture_id,
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn can_add_to_batch(&self,
                            dest_texture_id: TextureId,
                            color_texture_id: TextureId,
                            program_id: ProgramId,
                            blur_direction: Option<AxisDirection>)
                            -> bool {
        let batch_ok = program_id == self.program_id &&
            blur_direction == self.blur_direction &&
            dest_texture_id == self.dest_texture_id &&
            color_texture_id == self.color_texture_id;
/*
        println!("batch ok? {:?} program_id={:?}/{:?} blur_direction={:?}/{:?} \
                  dest_texture_id {:?}/{:?} color_texture_id={:?}/{:?}",
                 batch_ok,
                 program_id, self.program_id,
                 blur_direction, self.blur_direction,
                 dest_texture_id, self.dest_texture_id,
                 color_texture_id, self.color_texture_id);
*/
        batch_ok
    }

    pub fn add_draw_item(&mut self,
                         dest_texture_id: TextureId,
                         color_texture_id: TextureId,
                         vertices: &[PackedVertexForTextureCacheUpdate]) {
        debug_assert!(dest_texture_id == self.dest_texture_id);
        debug_assert!(color_texture_id == self.color_texture_id);

        for i in (0..vertices.len()).step_by(4) {
            let index_offset = self.vertices.len();
            let index_base = (index_offset + i) as u16;
            self.indices.push(index_base + 0);
            self.indices.push(index_base + 1);
            self.indices.push(index_base + 2);
            self.indices.push(index_base + 2);
            self.indices.push(index_base + 3);
            self.indices.push(index_base + 1);
        }

        self.vertices.extend_from_slice(vertices);
    }
}
