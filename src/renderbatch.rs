use device::{ProgramId, TextureId};
use internal_types::{BatchId, DisplayItemKey, DrawListIndex};
use internal_types::{PackedVertex, Primitive};
use std::collections::HashMap;
use std::collections::hash_map::Entry::{Occupied, Vacant};

const MAX_MATRICES_PER_BATCH: usize = 32;

pub struct RenderBatch {
    pub batch_id: BatchId,
    pub sort_key: DisplayItemKey,
    pub program_id: ProgramId,
    pub color_texture_id: TextureId,
    pub mask_texture_id: TextureId,
    pub vertices: Vec<PackedVertex>,
    pub indices: Vec<u16>,
    pub matrix_map: HashMap<DrawListIndex, u8>,
}

impl RenderBatch {
    pub fn new(batch_id: BatchId,
           sort_key: DisplayItemKey,
           program_id: ProgramId,
           color_texture_id: TextureId,
           mask_texture_id: TextureId) -> RenderBatch {
        RenderBatch {
            sort_key: sort_key,
            batch_id: batch_id,
            program_id: program_id,
            color_texture_id: color_texture_id,
            mask_texture_id: mask_texture_id,
            vertices: Vec::new(),
            indices: Vec::new(),
            matrix_map: HashMap::new(),
        }
    }

    pub fn can_add_to_batch(&self,
                        color_texture_id: TextureId,
                        mask_texture_id: TextureId,
                        key: &DisplayItemKey,
                        program_id: ProgramId) -> bool {
        let matrix_ok = self.matrix_map.len() < MAX_MATRICES_PER_BATCH ||
                        self.matrix_map.contains_key(&key.draw_list_index);

        program_id == self.program_id &&
            color_texture_id == self.color_texture_id &&
            mask_texture_id == self.mask_texture_id &&
            self.vertices.len() < 65535 &&                  // to ensure we can use u16 index buffers
            matrix_ok
    }

    pub fn add_draw_item(&mut self,
                     color_texture_id: TextureId,
                     mask_texture_id: TextureId,
                     primitive: Primitive,
                     vertices: &mut [PackedVertex],
                     key: &DisplayItemKey) {
        debug_assert!(color_texture_id == self.color_texture_id);
        debug_assert!(mask_texture_id == self.mask_texture_id);

        let next_matrix_index = self.matrix_map.len() as u8;
        let matrix_index = match self.matrix_map.entry(key.draw_list_index) {
            Vacant(entry) => *entry.insert(next_matrix_index),
            Occupied(entry) => *entry.get(),
        };
        debug_assert!(self.matrix_map.len() <= MAX_MATRICES_PER_BATCH);

        let index_offset = self.vertices.len();

        match primitive {
            Primitive::Rectangles | Primitive::Glyphs => {
                for i in (0..vertices.len()).step_by(4) {
                    let index_base = (index_offset + i) as u16;
                    self.indices.push(index_base + 0);
                    self.indices.push(index_base + 1);
                    self.indices.push(index_base + 2);
                    self.indices.push(index_base + 2);
                    self.indices.push(index_base + 3);
                    self.indices.push(index_base + 1);
                }
            }
            Primitive::TriangleFan => {
                for i in (1..vertices.len() - 1) {
                    self.indices.push(index_offset as u16);        // center vertex
                    self.indices.push((index_offset + i + 0) as u16);
                    self.indices.push((index_offset + i + 1) as u16);
                }
            }
        }

        for vertex in vertices.iter_mut() {
            vertex.matrix_index = matrix_index;
        }

        self.vertices.push_all(vertices);
    }
}
