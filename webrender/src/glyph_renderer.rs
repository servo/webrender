/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! GPU glyph rasterization using Pathfinder.

use api::{DeviceIntPoint, DeviceIntRect, DeviceUintSize, FontRenderMode};
use api::{ImageFormat, TextureTarget};
use batch::BatchTextures;
use debug_colors;
use device::{Device, Texture, TextureFilter, VAO};
use euclid::{Point2D, Size2D, Transform3D, TypedVector2D, Vector2D};
use image_loader;
use internal_types::{RenderTargetInfo, SourceTexture};
use pathfinder_gfx_utils::ShelfBinPacker;
use profiler::GpuProfileTag;
use renderer::{self, ImageBufferKind, Renderer, RendererError, RendererStats, VertexArrayKind};
use shade::{LazilyCompiledShader, ShaderKind};
use tiling::GlyphJob;

static AREA_LUT_BYTES: &'static [u8] = include_bytes!("../res/area-lut.png");

const HORIZONTAL_BIN_PADDING: i32 = 3;

const GPU_TAG_GLYPH_STENCIL: GpuProfileTag = GpuProfileTag {
    label: "Glyph Stencil",
    color: debug_colors::STEELBLUE,
};
const GPU_TAG_GLYPH_COVER: GpuProfileTag = GpuProfileTag {
    label: "Glyph Cover",
    color: debug_colors::LIGHTSTEELBLUE,
};

pub struct GlyphRenderer {
    pub area_lut_texture: Texture,
    pub vector_stencil_vao: VAO,
    pub vector_cover_vao: VAO,

    // These are Pathfinder shaders, used for rendering vector graphics.
    vector_stencil: LazilyCompiledShader,
    vector_cover: LazilyCompiledShader,
}

impl GlyphRenderer {
    pub fn new(device: &mut Device, prim_vao: &VAO, precache_shaders: bool)
               -> Result<GlyphRenderer, RendererError> {
        let area_lut_image = image_loader::load_from_memory(AREA_LUT_BYTES).unwrap().to_luma();
        let mut area_lut_texture = device.create_texture(TextureTarget::Default, ImageFormat::R8);
        device.init_texture(&mut area_lut_texture,
                            area_lut_image.width(),
                            area_lut_image.height(),
                            TextureFilter::Linear,
                            None,
                            1,
                            Some(&area_lut_image));

        let vector_stencil_vao =
            device.create_vao_with_new_instances(&renderer::desc::VECTOR_STENCIL, prim_vao);
        let vector_cover_vao = device.create_vao_with_new_instances(&renderer::desc::VECTOR_COVER,
                                                                    prim_vao);

        // Load Pathfinder vector graphics shaders.
        let vector_stencil = try!{
            LazilyCompiledShader::new(ShaderKind::VectorStencil,
                                      "pf_vector_stencil",
                                      &[ImageBufferKind::Texture2D.get_feature_string()],
                                      device,
                                      precache_shaders)
        };
        let vector_cover = try!{
            LazilyCompiledShader::new(ShaderKind::VectorCover,
                                      "pf_vector_cover",
                                      &[ImageBufferKind::Texture2D.get_feature_string()],
                                      device,
                                      precache_shaders)
        };

        Ok(GlyphRenderer {
            area_lut_texture,
            vector_stencil_vao,
            vector_cover_vao,
            vector_stencil,
            vector_cover,
        })
    }
}

impl Renderer {
    /// Renders glyphs using the vector graphics shaders (Pathfinder).
    pub fn stencil_glyphs(&mut self,
                          glyphs: &[GlyphJob],
                          projection: &Transform3D<f32>,
                          target_size: &DeviceUintSize,
                          stats: &mut RendererStats)
                          -> Option<StenciledGlyphPage> {
        if glyphs.is_empty() {
            return None
        }

        let _timer = self.gpu_profile.start_timer(GPU_TAG_GLYPH_STENCIL);

        // Initialize temporary framebuffer.
        // FIXME(pcwalton): Cache this!
        // FIXME(pcwalton): Use RF32, not RGBAF32!
        let mut current_page = StenciledGlyphPage {
            texture: self.device.create_texture(TextureTarget::Default, ImageFormat::RGBAF32),
            glyphs: vec![],
        };
        self.device.init_texture::<f32>(&mut current_page.texture,
                                        target_size.width,
                                        target_size.height,
                                        TextureFilter::Nearest,
                                        Some(RenderTargetInfo {
                                            has_depth: false,
                                        }),
                                        1,
                                        None);

        // Allocate all target rects.
        let mut packer = ShelfBinPacker::new(&target_size.to_i32().to_untyped(),
                                             &Vector2D::new(HORIZONTAL_BIN_PADDING, 0));
        let mut glyph_indices: Vec<_> = (0..(glyphs.len())).collect();
        glyph_indices.sort_by(|&a, &b| {
            glyphs[b].target_rect.size.height.cmp(&glyphs[a].target_rect.size.height)
        });
        for &glyph_index in &glyph_indices {
            let glyph = &glyphs[glyph_index];
            let x_scale = x_scale_for_render_mode(glyph.render_mode);
            let stencil_size = Size2D::new(glyph.target_rect.size.width * x_scale,
                                           glyph.target_rect.size.height);
            match packer.add(&stencil_size) {
                Err(_) => return None,
                Ok(origin) => {
                    current_page.glyphs.push(VectorCoverInstanceAttrs {
                        target_rect: glyph.target_rect,
                        stencil_origin: DeviceIntPoint::from_untyped(&origin),
                        subpixel: (glyph.render_mode == FontRenderMode::Subpixel) as u16,
                    })
                }
            }
        }

        // Initialize path info.
        // TODO(pcwalton): Cache this texture!
        let mut path_info_texture = self.device.create_texture(TextureTarget::Default,
                                                               ImageFormat::RGBAF32);

        let mut path_info_texels = Vec::with_capacity(glyphs.len() * 12);
        for (stenciled_glyph_index, &glyph_index) in glyph_indices.iter().enumerate() {
            let glyph = &glyphs[glyph_index];
            let stenciled_glyph = &current_page.glyphs[stenciled_glyph_index];
            let x_scale = x_scale_for_render_mode(glyph.render_mode) as f32;
            let glyph_origin = TypedVector2D::new(-glyph.origin.x as f32 * x_scale,
                                                  -glyph.origin.y as f32);
            let subpixel_offset = TypedVector2D::new(glyph.subpixel_offset.x * x_scale,
                                                     glyph.subpixel_offset.y);
            let rect = stenciled_glyph.stencil_rect()
                                      .to_f32()
                                      .translate(&glyph_origin)
                                      .translate(&subpixel_offset);
            path_info_texels.extend_from_slice(&[
                x_scale, 0.0, 0.0, -1.0,
                rect.origin.x, rect.max_y(), 0.0, 0.0,
                rect.size.width, rect.size.height,
                glyph.embolden_amount.x,
                glyph.embolden_amount.y,
            ]);
        }

        self.device.init_texture(&mut path_info_texture,
                                 3,
                                 glyphs.len() as u32,
                                 TextureFilter::Nearest,
                                 None,
                                 1,
                                 Some(&path_info_texels));

        self.glyph_renderer.vector_stencil.bind(&mut self.device,
                                                projection,
                                                &mut self.renderer_errors);

        let area_lut_external_texture = self.glyph_renderer.area_lut_texture.to_external();
        let path_info_external_texture = path_info_texture.to_external();
        let batch_textures = BatchTextures {
            colors: [
                SourceTexture::Custom(area_lut_external_texture),
                SourceTexture::Custom(path_info_external_texture),
                SourceTexture::Invalid,
            ],
        };

        self.device.bind_draw_target(Some((&current_page.texture, 0)), Some(*target_size));
        self.device.clear_target(Some([0.0, 0.0, 0.0, 0.0]), None, None);

        self.device.set_blend(true);
        self.device.set_blend_mode_subpixel_pass1();

        let mut instance_data = vec![];
        for (path_id, &glyph_id) in glyph_indices.iter().enumerate() {
            let glyph = &glyphs[glyph_id];
            instance_data.extend(glyph.mesh
                                      .stencil_segments
                                      .iter()
                                      .zip(glyph.mesh.stencil_normals.iter())
                                      .map(|(segment, normals)| {
                VectorStencilInstanceAttrs {
                    from_position: segment.from,
                    ctrl_position: segment.ctrl,
                    to_position: segment.to,
                    from_normal: normals.from,
                    ctrl_normal: normals.ctrl,
                    to_normal: normals.to,
                    path_id: path_id as u16,
                }
            }));
        }

        self.draw_instanced_batch(&instance_data,
                                  VertexArrayKind::VectorStencil,
                                  &batch_textures,
                                  stats);

        self.device.delete_texture(path_info_texture);

        Some(current_page)
    }

    /// Blits glyphs from the stencil texture to the texture cache.
    ///
    /// Deletes the stencil texture at the end.
    /// FIXME(pcwalton): This is bad. Cache it somehow.
    pub fn cover_glyphs(&mut self,
                        stencil_page: StenciledGlyphPage,
                        projection: &Transform3D<f32>,
                        stats: &mut RendererStats) {
        debug_assert!(!stencil_page.glyphs.is_empty());

        let _timer = self.gpu_profile.start_timer(GPU_TAG_GLYPH_COVER);

        self.glyph_renderer.vector_cover.bind(&mut self.device,
                                              projection,
                                              &mut self.renderer_errors);

        let stencil_external_texture = stencil_page.texture.to_external();
        let batch_textures = BatchTextures::color(SourceTexture::Custom(stencil_external_texture));

        self.draw_instanced_batch(&stencil_page.glyphs,
                                  VertexArrayKind::VectorCover,
                                  &batch_textures,
                                  stats);

        self.device.delete_texture(stencil_page.texture);
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct VectorStencilInstanceAttrs {
    from_position: Point2D<f32>,
    ctrl_position: Point2D<f32>,
    to_position: Point2D<f32>,
    from_normal: Vector2D<f32>,
    ctrl_normal: Vector2D<f32>,
    to_normal: Vector2D<f32>,
    path_id: u16,
}

pub struct StenciledGlyphPage {
    texture: Texture,
    glyphs: Vec<VectorCoverInstanceAttrs>,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct VectorCoverInstanceAttrs {
    target_rect: DeviceIntRect,
    stencil_origin: DeviceIntPoint,
    subpixel: u16,
}

impl VectorCoverInstanceAttrs {
    fn stencil_rect(&self) -> DeviceIntRect {
        DeviceIntRect::new(self.stencil_origin, self.target_rect.size)
    }
}

fn x_scale_for_render_mode(render_mode: FontRenderMode) -> i32 {
    match render_mode {
        FontRenderMode::Subpixel => 3,
        FontRenderMode::Mono | FontRenderMode::Alpha => 1,
    }
}
