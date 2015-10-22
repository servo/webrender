//! Display list optimization.
//!
//! This just applies a few heuristics to display lists to reduce performance hazards.

use euclid::{Point2D, Rect, Size2D};
use std::mem;
use types::{DisplayItem, DisplayListBuilder, DrawList, ImageDisplayItem, SpecificDisplayItem};

pub const IMAGE_TILE_THRESHOLD: f32 = 5000.0;

fn optimize_draw_list(draw_list: &mut DrawList) {
    let old_draw_list = mem::replace(draw_list, DrawList::new());
    for item in old_draw_list.items.into_iter() {
        match item.item {
            SpecificDisplayItem::Image(ref image) => {
                // Break up large tiled images into smaller ones so that large background images
                // won't result in the construction of a whole bunch of needless vertices.
                let tile_count = (item.rect.size.width / image.stretch_size.width).ceil() *
                    (item.rect.size.height / image.stretch_size.height).ceil();
                if tile_count > IMAGE_TILE_THRESHOLD {
                    let tile_size = (image.stretch_size.width * image.stretch_size.height *
                                     IMAGE_TILE_THRESHOLD).sqrt();
                    let tile_size = Size2D::new((tile_size / image.stretch_size.width).ceil() *
                                                    image.stretch_size.width,
                                                (tile_size / image.stretch_size.height).ceil() *
                                                    image.stretch_size.height);
                    let mut y = item.rect.origin.y;
                    while y < item.rect.max_y() {
                        let mut x = item.rect.origin.x;
                        while x < item.rect.max_x() {
                            draw_list.push(DisplayItem {
                                item: SpecificDisplayItem::Image(ImageDisplayItem {
                                    image_id: image.image_id,
                                    stretch_size: image.stretch_size,
                                }),
                                rect: Rect::new(Point2D::new(x, y), tile_size),
                                clip: item.clip.clone(),
                                node_index: item.node_index,
                            });
                            x += tile_size.width;
                        }
                        y += tile_size.height;
                    }
                    continue
                }
            }
            _ => {}
        }

        draw_list.push(item);
    }
}

pub fn optimize_display_list_builder(display_list_builder: &mut DisplayListBuilder) {
    optimize_draw_list(&mut display_list_builder.background_and_borders);
    optimize_draw_list(&mut display_list_builder.block_backgrounds_and_borders);
    optimize_draw_list(&mut display_list_builder.floats);
    optimize_draw_list(&mut display_list_builder.content);
    optimize_draw_list(&mut display_list_builder.positioned_content);
    optimize_draw_list(&mut display_list_builder.outlines);
}

