/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use webrender_api::ColorF;
use webrender_api::debugger::CompositorDebugInfo;

pub fn ui(
    ui: &mut egui::Ui,
    info: &mut CompositorDebugInfo,
) {
    let mut valid_z_layers = 0;
    for tile in &info.tiles {
        valid_z_layers |= 1 << tile.z_id;
    }

    let colors = [
        ColorF::new(1.0, 1.0, 1.0, 1.0),
        ColorF::new(1.0, 0.0, 0.0, 1.0),
        ColorF::new(0.0, 1.0, 0.0, 1.0),
        ColorF::new(0.7, 0.7, 1.0, 1.0),
        ColorF::new(1.0, 1.0, 0.0, 1.0),
        ColorF::new(0.0, 1.0, 1.0, 1.0),
        ColorF::new(1.0, 0.0, 1.0, 1.0),
    ];

    ui.horizontal(|ui| {
        for i in 0..32 {
            if valid_z_layers & 1 << i != 0 {
                let mut enabled = info.enabled_z_layers & (1 << i) != 0;
                if ui.checkbox(&mut enabled, format!("z_id {}", i)).changed() {
                    if enabled {
                        info.enabled_z_layers |= 1 << i;
                    } else {
                        info.enabled_z_layers &= !(1 << i);
                    }
                }
            }
        }
    });

    let (response, painter) = ui.allocate_painter(
        ui.available_size(),
        egui::Sense::hover(),
    );

    let rect = response.rect;

    // Background
    painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(77, 77, 77));

    let mut x_max = -f32::INFINITY;
    let mut x_min = f32::INFINITY;
    let mut y_max = -f32::INFINITY;
    let mut y_min = f32::INFINITY;

    for tile in &info.tiles {
        x_min = x_min.min(tile.device_rect.min.x);
        x_max = x_max.max(tile.device_rect.max.x);
        y_min = y_min.min(tile.device_rect.min.y);
        y_max = y_max.max(tile.device_rect.max.y);
    }

    let src_width = x_max - x_min;
    let src_height = y_max - y_min;

    let dest_width = rect.width();
    let dest_height = rect.height();

    let scale_x = dest_width / src_width;
    let scale_y = dest_height / src_height;
    let scale = scale_x.min(scale_y);

    let scaled_width = src_width * scale;
    let scaled_height = src_height * scale;

    let offset_x = 0.5 * (dest_width - scaled_width);
    let offset_y = 0.5 * (dest_height - scaled_height);

    let sx0 = rect.left() + (x_min - x_min) * scale + offset_x;
    let sx1 = rect.left() + (x_max - x_min) * scale + offset_x;
    let sy0 = rect.top() + (y_min - y_min) * scale + offset_y;
    let sy1 = rect.top() + (y_max - y_min) * scale + offset_y;

    painter.rect_stroke(
        egui::Rect::from_min_max(egui::pos2(sx0, sy0), egui::pos2(sx1, sy1)),
        0.0,
        egui::Stroke::new(1.0_f32, egui::Color32::WHITE),
        egui::StrokeKind::Outside,
    );

    for tile in &info.tiles {
        if info.enabled_z_layers & (1 << tile.z_id) == 0 {
            continue;
        }

        let rect_opt = tile.device_rect.intersection(&tile.clip_rect);

        if let Some(tile_rect) = rect_opt {
            let x0 = rect.left() + (tile_rect.min.x - x_min) * scale + offset_x;
            let x1 = rect.left() + (tile_rect.max.x - x_min) * scale + offset_x;
            let y0 = rect.top() + (tile_rect.min.y - y_min) * scale + offset_y;
            let y1 = rect.top() + (tile_rect.max.y - y_min) * scale + offset_y;

            let color = colors[tile.z_id as usize % colors.len()];
            let fill_color = egui::Color32::from_rgba_premultiplied(
                (color.r * 255.0 * 0.3) as u8,
                (color.g * 255.0 * 0.3) as u8,
                (color.b * 255.0 * 0.3) as u8,
                (color.a * 255.0 * 0.3) as u8,
            );
            let stroke_color = egui::Color32::from_rgba_premultiplied(
                (color.r * 255.0) as u8,
                (color.g * 255.0) as u8,
                (color.b * 255.0) as u8,
                (color.a * 255.0) as u8,
            );

            let tile_rect = egui::Rect::from_min_max(
                egui::pos2(x0, y0),
                egui::pos2(x1, y1),
            );

            painter.rect_filled(tile_rect, 0.0, fill_color);
            painter.rect_stroke(tile_rect, 0.0, egui::Stroke::new(1.0_f32, stroke_color), egui::StrokeKind::Outside);
        }
    }
}
