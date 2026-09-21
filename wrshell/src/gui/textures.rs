/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::{Gui, Document, DocumentKind};
use webrender_api::debugger::DebuggerTextureContent;
use webrender_api::{ImageFormat, TextureCacheCategory};
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

const MIN_ZOOM: f32 = 1.0 / 64.0;
const MAX_ZOOM: f32 = 64.0;

/// Pan/zoom state of the texture viewer, one per texture document.
pub struct TextureView {
    /// Scale factor between texture pixels and screen points.
    zoom: f32,
    /// Position of the texture's top-left corner relative to the top-left
    /// corner of the viewport, in screen points.
    offset: egui::Vec2,
    /// Recompute zoom and offset to fit the texture in the viewport on the
    /// next frame.
    fit_requested: bool,
    /// Result of the last save, shown next to the save button.
    status: Option<String>,
    /// Coordinates and color of the texel under the pointer during the
    /// previous frame, shown in the toolbar.
    hovered_texel: Option<(u32, u32, [u8; 4])>,
}

impl TextureView {
    pub fn new() -> Self {
        TextureView {
            zoom: 1.0,
            offset: egui::Vec2::ZERO,
            fit_requested: true,
            status: None,
            hovered_texel: None,
        }
    }

    /// Zoom around a fixed point of the viewport, so that the texel under that
    /// point stays there.
    fn zoom_around(&mut self, anchor: egui::Vec2, new_zoom: f32) {
        let new_zoom = new_zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        let texel = (anchor - self.offset) / self.zoom;
        self.offset = anchor - texel * new_zoom;
        self.zoom = new_zoom;
    }
}

pub fn texture_viewer_ui(
    ui: &mut egui::Ui,
    image: &DebuggerTextureContent,
    handle: &egui::TextureHandle,
    view: &mut TextureView,
) {
    ui.horizontal(|ui| {
        ui.label(format!("Size: {}x{}, Format {:?}", image.width, image.height, image.format));

        ui.separator();

        if ui.button("-").on_hover_text("Zoom out").clicked() {
            view.zoom = (view.zoom * 0.5).clamp(MIN_ZOOM, MAX_ZOOM);
        }
        if ui.button("+").on_hover_text("Zoom in").clicked() {
            view.zoom = (view.zoom * 2.0).clamp(MIN_ZOOM, MAX_ZOOM);
        }
        if ui.button("1:1").on_hover_text("Reset the zoom to 100%").clicked() {
            view.zoom = 1.0;
        }
        if ui.button("Fit").on_hover_text("Fit the texture in the view").clicked() {
            view.fit_requested = true;
        }
        ui.label(format!("{:.0}%", view.zoom * 100.0));

        ui.separator();

        // The texel is picked while painting the canvas below, so this shows
        // the value from the previous frame.
        let readout = match view.hovered_texel {
            Some((x, y, [r, g, b, a])) => {
                format!("({x}, {y}) rgba({r:>3}, {g:>3}, {b:>3}, {a:>3})")
            }
            None => "(-, -) rgba(  -,   -,   -,   -)".to_string(),
        };
        ui.label(egui::RichText::new(readout).monospace())
            .on_hover_text("Color of the texel under the pointer");

        ui.separator();

        if ui.button("Save as PNG").clicked() {
            view.status = Some(match save_png(image) {
                Ok(path) => format!("Saved {}", path.display()),
                Err(e) => format!("Failed to save: {e}"),
            });
        }

        if let Some(status) = &view.status {
            ui.label(status.as_str());
        }
    });

    let (response, painter) = ui.allocate_painter(
        ui.available_size(),
        egui::Sense::click_and_drag(),
    );

    let viewport = response.rect;
    let tex_size = egui::vec2(image.width as f32, image.height as f32);

    if view.fit_requested {
        view.fit_requested = false;
        let scale = (viewport.width() / tex_size.x).min(viewport.height() / tex_size.y);
        view.zoom = scale.clamp(MIN_ZOOM, MAX_ZOOM);
        view.offset = (viewport.size() - tex_size * view.zoom) * 0.5;
    }

    if response.dragged() {
        view.offset += response.drag_delta();
    }

    view.hovered_texel = None;

    if let Some(pointer) = response.hover_pos() {
        let anchor = pointer - viewport.min;

        let texel = (anchor - view.offset) / view.zoom;
        if texel.x >= 0.0 && texel.y >= 0.0
            && texel.x < tex_size.x && texel.y < tex_size.y {
            let (x, y) = (texel.x as u32, texel.y as u32);
            view.hovered_texel = texel_at(image, x, y).map(|rgba| (x, y, rgba));
        }

        // Pinch gestures and ctrl+wheel come as a zoom delta, a plain wheel
        // scroll comes as a scroll delta which we also treat as zoom since
        // panning is done by dragging.
        let (zoom_delta, scroll) = ui.input(|i| (i.zoom_delta(), i.smooth_scroll_delta.y));
        let factor = zoom_delta * (scroll * 0.002).exp();
        if factor != 1.0 {
            view.zoom_around(anchor, view.zoom * factor);
        }
    }

    let image_rect = egui::Rect::from_min_size(
        viewport.min + view.offset,
        tex_size * view.zoom,
    );

    painter.rect_filled(viewport, 0.0, egui::Color32::from_rgb(40, 40, 40));

    painter.with_clip_rect(viewport).image(
        handle.id(),
        image_rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );

    painter.rect_stroke(
        image_rect,
        0.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(120, 120, 120)),
        egui::StrokeKind::Outside,
    );
}

/// Write the texture to a PNG file in the current directory, named after the
/// texture. Returns the path of the written file.
fn save_png(texture: &DebuggerTextureContent) -> Result<PathBuf, String> {
    let rgba = texture_to_rgba(texture)
        .ok_or_else(|| format!("unsupported format {:?}", texture.format))?;

    let file_name: String = texture.name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let path = PathBuf::from(format!("{file_name}.png"));

    let file = File::create(&path).map_err(|e| e.to_string())?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), texture.width, texture.height);
    encoder.set_color(png::ColorType::RGBA);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
    writer.write_image_data(&rgba).map_err(|e| e.to_string())?;

    Ok(std::fs::canonicalize(&path).unwrap_or(path))
}

/// Read a single texel as unmultiplied RGBA8, or None if the format is not
/// supported or the coordinates are out of the texture's data.
fn texel_at(texture: &DebuggerTextureContent, x: u32, y: u32) -> Option<[u8; 4]> {
    let index = (y as usize) * (texture.width as usize) + (x as usize);
    match texture.format {
        ImageFormat::RGBA8 | ImageFormat::BGRA8 => {
            let px: &[u8] = texture.data.get(index * 4 .. index * 4 + 4)?;
            if texture.format == ImageFormat::BGRA8 {
                Some([px[2], px[1], px[0], px[3]])
            } else {
                Some([px[0], px[1], px[2], px[3]])
            }
        }
        ImageFormat::R8 => {
            let gray = *texture.data.get(index)?;
            Some([gray, gray, gray, 255])
        }
        _ => None,
    }
}

/// Convert the texture's pixels into unmultiplied RGBA8, or None if the format
/// is not supported.
fn texture_to_rgba(texture: &DebuggerTextureContent) -> Option<Vec<u8>> {
    match texture.format {
        ImageFormat::RGBA8 => Some(texture.data.clone()),
        ImageFormat::BGRA8 => {
            let mut rgba = texture.data.clone();
            for pixel in rgba.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
            Some(rgba)
        }
        ImageFormat::R8 => Some(
            texture.data.iter()
                .flat_map(|&gray| [gray, gray, gray, 255])
                .collect()
        ),
        _ => None,
    }
}

pub fn texture_list_ui(app: &mut Gui, ui: &mut egui::Ui) {
   texture_list_inner(app, ui, TextureCacheCategory::Atlas);
   texture_list_inner(app, ui, TextureCacheCategory::Standalone);
   texture_list_inner(app, ui, TextureCacheCategory::RenderTarget);
   texture_list_inner(app, ui, TextureCacheCategory::PictureTile);
}

fn texture_category_query(category: TextureCacheCategory) -> &'static str {
    match category {
        TextureCacheCategory::Atlas => "atlas-textures",
        TextureCacheCategory::Standalone => "standalone-textures",
        TextureCacheCategory::PictureTile => "tile-textures",
        TextureCacheCategory::RenderTarget => "target-textures",
    }
}

fn texture_category_label(category: TextureCacheCategory) -> &'static str {
    match category {
        TextureCacheCategory::Atlas => "Atlases",
        TextureCacheCategory::Standalone => "Standalone",
        TextureCacheCategory::PictureTile => "Tiles",
        TextureCacheCategory::RenderTarget => "Render targets",
    }
}

fn texture_list_inner(app: &mut Gui, ui: &mut egui::Ui, category: TextureCacheCategory) {
    let width = ui.available_width();

    let cursor = ui.cursor().min;
    let refresh_rect = egui::Rect {
        min: egui::Pos2::new(cursor.x + width - 20.0, cursor.y),
        max: egui::Pos2::new(cursor.x + width, cursor.y + 20.0),
    };
    let refresh_button = egui::widgets::Button::new("↓");
    if ui.place(refresh_rect, refresh_button).clicked() {
        let query_result = app.net.get_with_query(
            "query", &[("type", texture_category_query(category))]
        );

        if let Ok(Some(msg)) = query_result {
            app.data_model.preview_doc_index = None;
            app.data_model.documents.retain(|doc| !doc_is_texture(doc, category));

            // Note: deserializing the textures takes a long time in
            // debug builds.
            let new_textures = serde_json::from_str(msg.as_str()).unwrap();
            add_textures(app, new_textures);
        }
    }

    egui::CollapsingHeader::new(texture_category_label(category)).default_open(true).show(ui, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            for (i, doc) in app.data_model.documents.iter().enumerate() {
                if !doc_is_texture(doc, category) {
                    continue;
                }

                let item = egui::Button::selectable(
                    app.data_model.preview_doc_index == Some(i),
                    &doc.title,
                ).min_size(egui::vec2(width - 20.0, 20.0));

                if ui.add(item).clicked() {
                    app.data_model.preview_doc_index = Some(i);
                }
            }
        });
    });
}

pub fn add_textures(
    app: &mut Gui,
    mut textures: Vec<DebuggerTextureContent>,
) {
    textures.sort_by(|a, b| a.name.cmp(&b.name));
    for texture in textures {
        app.data_model.documents.push(Document {
            title: texture.name.clone(),
            kind: DocumentKind::Texture {
                content: texture,
                handle: None,
                view: TextureView::new(),
            }
        });
    }
}

/// Perform uploads if need be. Happens earlier in the update because it needs
/// access to the egui context.
pub fn prepare(app: &mut super::Gui, ctx: &egui::Context) {
    if let Some(idx) = app.data_model.preview_doc_index {
        if idx >= app.data_model.documents.len() {
            return;
        }

        let DocumentKind::Texture { content, handle, .. } = &mut app.data_model.documents[idx].kind else {
            return;
        };

        if handle.is_some() {
            return;
        }

        if let Some(gpu_texture) = upload_texture(ctx, content) {
            *handle = Some(gpu_texture)
        }
    }
}

fn upload_texture(
    ctx: &egui::Context,
    texture: &DebuggerTextureContent,
) -> Option<egui::TextureHandle> {
    let Some(rgba) = texture_to_rgba(texture) else {
        println!("Unsupported texture format: {:?}", texture.format);
        return None;
    };

    let color_image = egui::ColorImage::from_rgba_unmultiplied(
        [texture.width as usize, texture.height as usize],
        &rgba,
    );

    // Nearest magnification keeps texels crisp when zoomed in, while linear
    // minification avoids aliasing when zoomed out.
    let options = egui::TextureOptions {
        magnification: egui::TextureFilter::Nearest,
        minification: egui::TextureFilter::Linear,
        ..Default::default()
    };

    Some(ctx.load_texture(&texture.name, color_image, options))
}

fn doc_is_texture(doc: &Document, kind: TextureCacheCategory) -> bool {
    match doc.kind {
        DocumentKind::Texture { content: DebuggerTextureContent { category, .. }, .. } => {
            category == kind
        },
        _ => false,
    }
}
