use aabbtree::AABBTreeNode;
use app_units::Au;
use batch_builder;
use euclid::{Rect, Size2D};
use fnv::FnvHasher;
use internal_types::{BorderRadiusRasterOp, BoxShadowRasterOp, DrawListItemIndex};
use internal_types::{Glyph, GlyphKey, RasterItem};
use resource_cache::ResourceCache;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::hash_state::DefaultState;
use tessellator;
use webrender_traits::{BorderRadius, BorderStyle, BoxShadowClipMode, ImageRendering};
use webrender_traits::{FontKey, ImageFormat, ImageKey, SpecificDisplayItem};

type RequiredImageSet = HashSet<(ImageKey, ImageRendering), DefaultState<FnvHasher>>;
type RequiredGlyphMap = HashMap<FontKey, HashSet<Glyph>, DefaultState<FnvHasher>>;
type RequiredRasterSet = HashSet<RasterItem, DefaultState<FnvHasher>>;

pub struct ResourceList {
    required_images: RequiredImageSet,
    required_glyphs: RequiredGlyphMap,
    required_rasters: RequiredRasterSet,
}

impl ResourceList {
    pub fn new() -> ResourceList {
        ResourceList {
            required_glyphs: HashMap::with_hash_state(Default::default()),
            required_images: HashSet::with_hash_state(Default::default()),
            required_rasters: HashSet::with_hash_state(Default::default()),
        }
    }

    pub fn add_image(&mut self,
                     key: ImageKey,
                     image_rendering: ImageRendering) {
        self.required_images.insert((key, image_rendering));
    }

    pub fn add_glyph(&mut self, font_key: FontKey, glyph: Glyph) {
        match self.required_glyphs.entry(font_key) {
            Occupied(entry) => {
                entry.into_mut().insert(glyph);
            }
            Vacant(entry) => {
                let mut hash_set = HashSet::new();
                hash_set.insert(glyph);
                entry.insert(hash_set);
            }
        }
    }

    pub fn add_radius_raster(&mut self,
                             outer_radius: &Size2D<f32>,
                             inner_radius: &Size2D<f32>,
                             inverted: bool,
                             index: u32,
                             image_format: ImageFormat) {
        if let Some(raster_item) = BorderRadiusRasterOp::create(outer_radius,
                                                                inner_radius,
                                                                inverted,
                                                                index,
                                                                image_format) {
            self.required_rasters.insert(RasterItem::BorderRadius(raster_item));
        }
    }

    pub fn add_radius_raster_for_border_radii(&mut self, radii: &BorderRadius) {
        let zero_size = Size2D::new(0.0, 0.0);
        self.add_radius_raster(&radii.top_left, &zero_size, false, 0, ImageFormat::A8);
        self.add_radius_raster(&radii.top_right, &zero_size, false, 0, ImageFormat::A8);
        self.add_radius_raster(&radii.bottom_left, &zero_size, false, 0, ImageFormat::A8);
        self.add_radius_raster(&radii.bottom_right, &zero_size, false, 0, ImageFormat::A8);
    }

    pub fn add_box_shadow_corner(&mut self,
                                 blur_radius: f32,
                                 border_radius: f32,
                                 box_rect: &Rect<f32>,
                                 inverted: bool) {
        if let Some(raster_item) = BoxShadowRasterOp::create_corner(blur_radius,
                                                                    border_radius,
                                                                    box_rect,
                                                                    inverted) {
            self.required_rasters.insert(RasterItem::BoxShadow(raster_item));
        }
    }

    pub fn add_box_shadow_edge(&mut self,
                               blur_radius: f32,
                               border_radius: f32,
                               box_rect: &Rect<f32>,
                               inverted: bool) {
        if let Some(raster_item) = BoxShadowRasterOp::create_edge(blur_radius,
                                                                  border_radius,
                                                                  box_rect,
                                                                  inverted) {
            self.required_rasters.insert(RasterItem::BoxShadow(raster_item));
        }
    }

    pub fn for_each_image<F>(&self, mut f: F) where F: FnMut(ImageKey, ImageRendering) {
        for &(image_id, image_rendering) in &self.required_images {
            f(image_id, image_rendering);
        }
    }

    pub fn for_each_raster<F>(&self, mut f: F) where F: FnMut(&RasterItem) {
        for raster_item in &self.required_rasters {
            f(raster_item);
        }
    }

    pub fn for_each_glyph<F>(&self, mut f: F) where F: FnMut(&GlyphKey) {
        for (font_id, glyphs) in &self.required_glyphs {
            let mut glyph_key = GlyphKey::new(font_id.clone(), Au(0), Au(0), 0);

            for glyph in glyphs {
                glyph_key.size = glyph.size;
                glyph_key.index = glyph.index;
                glyph_key.blur_radius = glyph.blur_radius;

                f(&glyph_key);
            }
        }
    }
}

pub trait BuildRequiredResources {
    fn build_resource_list(&mut self, resource_cache: &ResourceCache);
}

impl BuildRequiredResources for AABBTreeNode {
    fn build_resource_list(&mut self, resource_cache: &ResourceCache) {
        //let _pf = util::ProfileScope::new("  build_resource_list");
        let mut resource_list = ResourceList::new();

        for group in &self.draw_list_group_segments {
            for draw_list_index_buffer in &group.index_buffers {
                let draw_list = resource_cache.get_draw_list(draw_list_index_buffer.draw_list_id);

                for index in &draw_list_index_buffer.indices {
                    let DrawListItemIndex(index) = *index;
                    let display_item = &draw_list.items[index as usize];

                    // Handle border radius for complex clipping regions.
                    for complex_clip_region in display_item.clip.complex.iter() {
                        resource_list.add_radius_raster_for_border_radii(&complex_clip_region.radii);
                    }

                    match display_item.item {
                        SpecificDisplayItem::Image(ref info) => {
                            resource_list.add_image(info.image_key, info.image_rendering);
                        }
                        SpecificDisplayItem::Text(ref info) => {
                            for glyph in &info.glyphs {
                                let glyph = Glyph::new(info.size, info.blur_radius, glyph.index);
                                resource_list.add_glyph(info.font_key, glyph);
                            }
                        }
                        SpecificDisplayItem::WebGL(..) => {}
                        SpecificDisplayItem::Rectangle(..) => {}
                        SpecificDisplayItem::Gradient(..) => {}
                        SpecificDisplayItem::BoxShadow(ref info) => {
                            resource_list.add_radius_raster_for_border_radii(
                                &BorderRadius::uniform(info.border_radius));

                            let box_rect = batch_builder::compute_box_shadow_rect(&info.box_bounds,
                                                                                  &info.offset,
                                                                                  info.spread_radius);
                            resource_list.add_box_shadow_corner(info.blur_radius,
                                                                info.border_radius,
                                                                &box_rect,
                                                                false);
                            resource_list.add_box_shadow_edge(info.blur_radius,
                                                              info.border_radius,
                                                              &box_rect,
                                                              false);
                            if info.clip_mode == BoxShadowClipMode::Inset {
                                resource_list.add_box_shadow_corner(info.blur_radius,
                                                                    info.border_radius,
                                                                    &box_rect,
                                                                    true);
                                resource_list.add_box_shadow_edge(info.blur_radius,
                                                                  info.border_radius,
                                                                  &box_rect,
                                                                  true);
                            }
                        }
                        SpecificDisplayItem::Border(ref info) => {
                            for rect_index in 0..tessellator::quad_count_for_border_corner(
                                    &info.radius.top_left) {
                                resource_list.add_radius_raster(&info.radius.top_left,
                                                                &info.top_left_inner_radius(),
                                                                false,
                                                                rect_index,
                                                                ImageFormat::A8);
                            }
                            for rect_index in 0..tessellator::quad_count_for_border_corner(
                                    &info.radius.top_right) {
                                resource_list.add_radius_raster(&info.radius.top_right,
                                                                &info.top_right_inner_radius(),
                                                                false,
                                                                rect_index,
                                                                ImageFormat::A8);
                            }
                            for rect_index in 0..tessellator::quad_count_for_border_corner(
                                    &info.radius.bottom_left) {
                                resource_list.add_radius_raster(&info.radius.bottom_left,
                                                                &info.bottom_left_inner_radius(),
                                                                false,
                                                                rect_index,
                                                                ImageFormat::A8);
                            }
                            for rect_index in 0..tessellator::quad_count_for_border_corner(
                                    &info.radius.bottom_right) {
                                resource_list.add_radius_raster(&info.radius.bottom_right,
                                                                &info.bottom_right_inner_radius(),
                                                                false,
                                                                rect_index,
                                                                ImageFormat::A8);
                            }

                            if info.top.style == BorderStyle::Dotted {
                                resource_list.add_radius_raster(&Size2D::new(info.top.width / 2.0,
                                                                             info.top.width / 2.0),
                                                                &Size2D::new(0.0, 0.0),
                                                                false,
                                                                0,
                                                                ImageFormat::RGBA8);
                            }
                            if info.right.style == BorderStyle::Dotted {
                                resource_list.add_radius_raster(&Size2D::new(info.right.width / 2.0,
                                                                             info.right.width / 2.0),
                                                                &Size2D::new(0.0, 0.0),
                                                                false,
                                                                0,
                                                                ImageFormat::RGBA8);
                            }
                            if info.bottom.style == BorderStyle::Dotted {
                                resource_list.add_radius_raster(&Size2D::new(info.bottom.width / 2.0,
                                                                             info.bottom.width / 2.0),
                                                                &Size2D::new(0.0, 0.0),
                                                                false,
                                                                0,
                                                                ImageFormat::RGBA8);
                            }
                            if info.left.style == BorderStyle::Dotted {
                                resource_list.add_radius_raster(&Size2D::new(info.left.width / 2.0,
                                                                             info.left.width / 2.0),
                                                                &Size2D::new(0.0, 0.0),
                                                                false,
                                                                0,
                                                                ImageFormat::RGBA8);
                            }
                        }
                    }
                }
            }
        }

        self.resource_list = Some(resource_list);
    }
}
