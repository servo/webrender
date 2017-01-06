/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use clap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use webrender_traits::*;
use yaml_helper::YamlHelper;
use yaml_rust::{Yaml, YamlLoader};
use euclid::TypedPoint2D;

use wrench::{Wrench, WrenchThing, layout_simple_ascii};
use {WHITE_COLOR, BLACK_COLOR, PLATFORM_DEFAULT_FACE_NAME};

fn broadcast<T: Clone>(base_vals: &[T], num_items: usize) -> Vec<T>
{
    if base_vals.len() == num_items {
        return base_vals.to_vec();
    }

    if num_items % base_vals.len() != 0 {
        panic!("Cannot broadcast {} elements into {}", base_vals.len(), num_items);
    }

    let mut vals = vec![];
    loop {
        if vals.len() == num_items {
            break;
        }
        vals.extend_from_slice(&base_vals);
    }
    vals
}

pub struct YamlFrameReader {
    frame_built: bool,
    yaml_path: PathBuf,
    aux_dir: PathBuf,
    frame_count: u32,

    builder: Option<DisplayListBuilder>,
    queue_depth: u32,

    include_only: Vec<String>,

    watch_source: bool,
}

impl YamlFrameReader {
    pub fn new(yaml_path: &Path) -> YamlFrameReader {
        YamlFrameReader {
            watch_source: false,
            frame_built: false,
            yaml_path: yaml_path.to_owned(),
            aux_dir: yaml_path.parent().unwrap().to_owned(),
            frame_count: 0,

            builder: None,

            queue_depth: 1,
            include_only: vec![],
        }
    }

    pub fn new_from_args(args: &clap::ArgMatches) -> YamlFrameReader {
        let yaml_file = args.value_of("INPUT").map(|s| PathBuf::from(s)).unwrap();

        let mut y = YamlFrameReader::new(&yaml_file);
        y.watch_source = args.is_present("watch");
        y.queue_depth = args.value_of("queue").map(|s| s.parse::<u32>().unwrap()).unwrap_or(1);
        y.include_only = args.values_of("include").map(|v| v.map(|s| s.to_owned()).collect()).unwrap_or(vec![]);
        y
    }

    pub fn builder<'a>(&'a mut self) -> &'a mut DisplayListBuilder {
        self.builder.as_mut().unwrap()
    }

    pub fn build(&mut self, wrench: &mut Wrench) {
        let mut file = File::open(&self.yaml_path).unwrap();
        let mut src = String::new();
        file.read_to_string(&mut src).unwrap();

        let mut yaml_doc = YamlLoader::load_from_str(&src).expect("Failed to parse YAML file");
        assert!(yaml_doc.len() == 1);

        let yaml = yaml_doc.pop().unwrap();
        if !yaml["pipelines"].is_badvalue() {
            let pipelines = yaml["pipelines"].as_vec().unwrap();
            for pipeline in pipelines {
                let pipeline_id = pipeline["id"].as_pipeline_id().unwrap();
                self.builder = Some(DisplayListBuilder::new(pipeline_id));
                self.add_stacking_context_from_yaml(wrench, pipeline);
                wrench.send_lists(self.frame_count, self.builder.as_ref().unwrap().clone());
            }

        }
        self.builder = Some(DisplayListBuilder::new(wrench.root_pipeline_id));
        if yaml["root"].is_badvalue() {
            panic!("Missing root stacking context");
        }
        self.add_stacking_context_from_yaml(wrench, &yaml["root"]);
    }

    fn to_clip_region(&mut self, item: &Yaml, item_bounds: &LayoutRect, wrench: &mut Wrench) -> Option<ClipRegion> {
        match item {
            &Yaml::String(_) => {
                let rect = item.as_rect().expect(&format!("clip region '{:?}', thought it was a rect string but it's not?", item));
                Some(self.builder().new_clip_region(&rect, vec![], None))
            }
            &Yaml::Array(ref v) => {
                if let Some(rect) = item.as_rect() {
                    // it's a rect (as an array)
                    Some(self.builder().new_clip_region(&rect, vec![], None))
                } else {
                    // it may be an array of simple rects
                    let rects = v.iter().map(|v| {
                         v.as_rect().map(|r| ComplexClipRegion::new(r, BorderRadius::zero())).ok_or(())
                     })
                     .collect::<Result<Vec<_>,_>>()
                     .expect(&format!("clip region array '{:?}', tried to do array of rects but failed", item));
                    Some(self.builder().new_clip_region(item_bounds, rects, None))
                }
            }
            &Yaml::Hash(_) => {
                let bounds = item["rect"].as_rect().unwrap_or(*item_bounds);
                let complex = if let Some(complex) = item["complex"].as_vec() {
                    complex.iter().map(|item|
                        match item {
                            &Yaml::String(_) | &Yaml::Array(_) => {
                                ComplexClipRegion::new(item.as_rect().expect("not a rect"), BorderRadius::zero())
                            }
                            &Yaml::Hash(_) => {
                                let rect = item["rect"].as_rect().expect("complex clip entry must have rect");
                                let radius = item["radius"].as_border_radius().unwrap_or(BorderRadius::zero());
                                ComplexClipRegion::new(rect, radius)
                            }
                            _ => {
                                panic!("Invalid complex clip region item entry");
                            }
                        }
                    ).collect()
                } else {
                    vec![]
                };

                let image_mask = if item["image_mask"].as_hash().is_some() {
                    let image_mask = &item["image_mask"];
                    let (image_key, image_dims) = wrench.add_or_get_image(&self.rsrc_path(&image_mask["image"]));
                    let image_rect = image_mask["rect"].as_rect().unwrap_or(LayoutRect::new(LayoutPoint::zero(), image_dims));
                    let image_repeat = image_mask["repeat"].as_bool().expect("expected boolean");
                    Some(ImageMask { image: image_key, rect: image_rect, repeat: image_repeat })
                } else {
                    None
                };
                println!("R {:?} cx {:?}", bounds, complex);
                Some(self.builder().new_clip_region(&bounds, complex, image_mask))
            }
            _ => {
                None
            }
        }
    }

    fn handle_rect(&mut self, wrench: &mut Wrench, clip_region: &ClipRegion, item: &Yaml) {
        let bounds_key = if item["type"].is_badvalue() { "rect" } else { "bounds" };
        let rect = item[bounds_key].as_rect().expect("rect type must have bounds");
        let color = item["color"].as_colorf().unwrap_or(*WHITE_COLOR);

        let clip = self.to_clip_region(&item["clip"], &rect, wrench).unwrap_or(*clip_region);
        self.builder().push_rect(rect, clip, color);
    }

    fn handle_gradient(&mut self, wrench: &mut Wrench, clip_region: &ClipRegion, item: &Yaml) {
        let bounds_key = if item["type"].is_badvalue() { "gradient" } else { "bounds" };
        let bounds = item[bounds_key].as_rect().expect("gradient must have bounds");
        let start = item["start"].as_point().expect("gradient must have start");
        let end = item["end"].as_point().expect("gradient must have end");
        let stops = item["stops"].as_vec().expect("gradient must have stops")
            .chunks(2).map(|chunk| GradientStop {
                offset: chunk[0].as_force_f32().expect("gradient stop offset is not f32"),
                color: chunk[1].as_colorf().expect("gradient stop color is not color"),
            }).collect::<Vec<_>>();

        let clip = self.to_clip_region(&item["clip"], &bounds, wrench).unwrap_or(*clip_region);
        self.builder().push_gradient(bounds, clip, start, end, stops);
    }

    fn handle_border(&mut self, wrench: &mut Wrench, clip_region: &ClipRegion, item: &Yaml) {
        let bounds_key = if item["type"].is_badvalue() { "border" } else { "bounds" };
        let bounds = item[bounds_key].as_rect().expect("borders must have bounds");
        let widths = item["width"].as_vec_f32().expect("borders must have width(s)");
        let colors = item["color"].as_vec_colorf().expect("borders must have color(s)");
        let styles = item["style"].as_vec_string().expect("borders must have style(s)");
        let styles = styles.iter().map(|s| match s.as_str() {
            "none" => BorderStyle::None,
            "solid" => BorderStyle::Solid,
            "double" => BorderStyle::Double,
            "dotted" => BorderStyle::Dotted,
            "dashed" => BorderStyle::Dashed,
            "hidden" => BorderStyle::Hidden,
            "ridge" => BorderStyle::Ridge,
            "inset" => BorderStyle::Inset,
            "outset" => BorderStyle::Outset,
            "groove" => BorderStyle::Groove,
            s => {
                panic!("Unknown border style '{}'", s);
            }
        }).collect::<Vec<BorderStyle>>();
        let radius = item["radius"].as_border_radius().unwrap_or(BorderRadius::zero());

        let widths = broadcast(&widths, 4);
        let colors = broadcast(&colors, 4);
        let styles = broadcast(&styles, 4);

        let top = BorderSide { width: widths[0], color: colors[0], style: styles[0] };
        let left = BorderSide { width: widths[1], color: colors[1], style: styles[1] };
        let bottom = BorderSide { width: widths[2], color: colors[2], style: styles[2] };
        let right = BorderSide { width: widths[3], color: colors[3], style: styles[3] };

        let clip = self.to_clip_region(&item["clip"], &bounds, wrench).unwrap_or(*clip_region);
        self.builder().push_border(bounds, clip, left, top, right, bottom, radius);
    }

    fn handle_box_shadow(&mut self, wrench: &mut Wrench, clip_region: &ClipRegion, item: &Yaml) {
        let bounds_key = if item["type"].is_badvalue() { "box_shadow" } else { "bounds" };
        let bounds = item[bounds_key].as_rect().expect("box shadow must have bounds");
        let box_bounds = item["box_bounds"].as_rect().unwrap_or(bounds);
        let offset = item["offset"].as_point().unwrap_or(TypedPoint2D::zero());
        let color = item["color"].as_colorf().unwrap_or(ColorF::new(0.0, 0.0, 0.0, 1.0));
        let blur_radius = item["blur_radius"].as_force_f32().unwrap_or(0.0);
        let spread_radius = item["spread_radius"].as_force_f32().unwrap_or(0.0);
        let border_radius = item["border_radius"].as_force_f32().unwrap_or(0.0);
        let clip_mode = if let Some(mode) = item.as_str() {
            match mode {
                "none" => BoxShadowClipMode::None,
                "outset" => BoxShadowClipMode::Outset,
                "inset" => BoxShadowClipMode::Inset,
                s => panic!("Unknown box shadow clip mode {}", s),
            }
        } else {
            BoxShadowClipMode::None
        };

        let clip = self.to_clip_region(&item["clip"], &bounds, wrench).unwrap_or(*clip_region);
        self.builder().push_box_shadow(bounds, clip, box_bounds, offset, color, blur_radius, spread_radius,
                                       border_radius, clip_mode);
    }

    fn rsrc_path(&self, item: &Yaml) -> PathBuf {
        let filename = item.as_str().unwrap();
        let mut file = self.aux_dir.clone();
        file.push(filename);
        file
    }

    fn handle_image(&mut self, wrench: &mut Wrench, clip_region: &ClipRegion, item: &Yaml) {
        let filename = &item[if item["type"].is_badvalue() { "image" } else { "src" }];
        let (image_key, image_dims) = wrench.add_or_get_image(&self.rsrc_path(filename));

        let bounds_raws = item["bounds"].as_vec_f32().unwrap();
        let bounds = if bounds_raws.len() == 2 {
            LayoutRect::new(LayoutPoint::new(bounds_raws[0], bounds_raws[1]),
                            image_dims)
        } else if bounds_raws.len() == 4 {
            LayoutRect::new(LayoutPoint::new(bounds_raws[0], bounds_raws[1]),
                            LayoutSize::new(bounds_raws[2], bounds_raws[3]))
        } else {
            panic!("image expected 2 or 4 values in bounds, got '{:?}'", item["bounds"]);
        };

        let stretch_size = item["stretch_size"].as_size()
            .unwrap_or(image_dims);
        let tile_spacing = item["tile_spacing"].as_size()
            .unwrap_or(LayoutSize::new(0.0, 0.0));
        let rendering = match item["rendering"].as_str() {
            Some("auto") | None => ImageRendering::Auto,
            Some("crisp_edges") => ImageRendering::CrispEdges,
            Some("pixelated") => ImageRendering::Pixelated,
            Some(_) => panic!("ImageRendering can be auto, crisp_edges, or pixelated -- got {:?}", item),
        };
        let clip = self.to_clip_region(&item["clip"], &bounds, wrench).unwrap_or(*clip_region);
        self.builder().push_image(bounds, clip, stretch_size, tile_spacing, rendering, image_key);
    }

    fn handle_text(&mut self, wrench: &mut Wrench, clip_region: &ClipRegion, item: &Yaml) {
        let size = item["size"].as_pt_to_au().unwrap_or(Au::from_f32_px(16.0));
        let color = item["color"].as_colorf().unwrap_or(*BLACK_COLOR);
        let blur_radius = item["blur_radius"].as_px_to_au().unwrap_or(Au::from_f32_px(0.0));

        let (font_key, native_key) = if !item["family"].is_badvalue() {
            wrench.font_key_from_yaml_table(item)
        } else if !item["font"].is_badvalue() {
            let font_file = self.rsrc_path(&item["font"]);
            let mut file = File::open(&font_file).expect("Couldn't open font file");
            let mut bytes = vec![];
            file.read_to_end(&mut bytes).expect("failed to read font file");
            wrench.font_key_from_bytes(bytes)
        } else {
            wrench.font_key_from_name(&*PLATFORM_DEFAULT_FACE_NAME)
        };

        if item["glyphs"].is_badvalue() && item["text"].is_badvalue() {
            panic!("text item had neither text nor glyphs!");
        }

        let (glyphs, rect) = if item["text"].is_badvalue() {
            // if glyphs are specified, then the glyph positions can have the
            // origin baked in.
            let origin = item["origin"].as_point().unwrap_or(LayoutPoint::new(0.0, 0.0));
            let glyph_indices = item["glyphs"].as_vec_u32().unwrap();
            let glyph_offsets = item["offsets"].as_vec_f32().unwrap();
            assert!(glyph_offsets.len() == glyph_indices.len() * 2);

            let glyphs = glyph_indices.iter().enumerate().map(|k| {
                GlyphInstance {
                    index: *k.1,
                    x: origin.x + glyph_offsets[k.0*2],
                    y: origin.y + glyph_offsets[k.0*2+1],
                }
            }).collect();
            // TODO(gw): We could optionally use the WR API to query glyph dimensions
            //           here and calculate the bounding region here if we want to.
            let rect = item["bounds"].as_rect()
                                     .expect("Text items with glyphs require bounds [for now]");
            (glyphs, rect)
        } else {
            if native_key.is_none() {
                panic!("Can't layout simple ascii text with raw font [for now]");
            }
            let native_key = native_key.unwrap();
            let text = item["text"].as_str().unwrap();
            let (glyph_indices, glyph_advances) =
                layout_simple_ascii(native_key, text, size);
            println!("Text layout: {}", text);
            println!(" glyphs  -> {:?}", glyph_indices);
            println!("    adv  -> {:?}", glyph_advances);
            let origin = item["origin"].as_point()
                .expect("origin required for text without glyphs");

            let mut x = origin.x;
            let y = origin.y;
            let glyphs = glyph_indices.iter().zip(glyph_advances).map(|arg| {
                let gi = GlyphInstance { index: *arg.0 as u32, x: x, y: y };
                x = x + arg.1;
                gi
            }).collect();
            // FIXME this is incorrect!
            let rect = LayoutRect::new(LayoutPoint::new(0.0, 0.0), wrench.window_size_f32());
            (glyphs, rect)
        };

        let clip = self.to_clip_region(&item["clip"], &rect, wrench).unwrap_or(*clip_region);
        self.builder().push_text(rect, clip, glyphs, font_key, color, size, blur_radius);
    }

    fn handle_iframe(&mut self, wrench: &mut Wrench, clip_region: &ClipRegion, item: &Yaml)
    {
        let bounds = item["bounds"].as_rect().expect("iframe must have bounds");
        let pipeline_id = item["id"].as_pipeline_id().unwrap();

        let clip = self.to_clip_region(&item["clip"], &bounds, wrench).unwrap_or(*clip_region);
        self.builder().push_iframe(bounds, clip, pipeline_id);
    }

    pub fn add_display_list_items_from_yaml(&mut self, wrench: &mut Wrench, yaml: &Yaml) {
        let full_clip_region = {
            let win_size = wrench.window_size_f32();
            self.builder().new_clip_region(&LayoutRect::new(LayoutPoint::new(0.0, 0.0), win_size),
                                           Vec::new(), None)
        };

        for ref item in yaml.as_vec().unwrap() {
            // an explicit type can be skipped with some shorthand
            let item_type =
                if !item["rect"].is_badvalue() { "rect" }
                else if !item["image"].is_badvalue() { "image" }
                else if !item["text"].is_badvalue() { "text" }
                else if !item["glyphs"].is_badvalue() { "glyphs" }
                else if !item["items"].is_badvalue() { "stacking_context" }
                // note: box_shadow shorthand check has to come before border
                else if !item["box_shadow"].is_badvalue() { "box_shadow" }
                else if !item["border"].is_badvalue() { "border" }
                else if !item["gradient"].is_badvalue() { "gradient" }
                else { item["type"].as_str().unwrap_or("unknown") };

            if item_type != "stacking_context" &&
               !self.include_only.is_empty() &&
               !self.include_only.contains(&item_type.to_owned()) {
                continue;
            }

            match item_type {
                "rect" => self.handle_rect(wrench, &full_clip_region, &item),
                "image" => self.handle_image(wrench, &full_clip_region, &item),
                "text" | "glyphs" => self.handle_text(wrench, &full_clip_region, &item),
                "stacking_context" => self.add_stacking_context_from_yaml(wrench, &item),
                "border" => self.handle_border(wrench, &full_clip_region, &item),
                "gradient" => self.handle_gradient(wrench, &full_clip_region, &item),
                "box_shadow" => self.handle_box_shadow(wrench, &full_clip_region, &item),
                "iframe" => self.handle_iframe(wrench, &full_clip_region, &item),
                _ => {
                    //println!("Skipping {:?}", item);
                }
            }
        }
    }

    pub fn add_stacking_context_from_yaml(&mut self, wrench: &mut Wrench, yaml: &Yaml) {
        let bounds = yaml["bounds"].as_rect().unwrap_or(LayoutRect::new(LayoutPoint::new(0.0, 0.0), wrench.window_size_f32()));
        let z_index = yaml["z_index"].as_i64().unwrap_or(0);
        let transform = yaml["transform"].as_matrix4d().unwrap_or(LayoutTransform::identity());
        let perspective = yaml["perspective"].as_matrix4d().unwrap_or(LayoutTransform::identity());
        let sc_full_rect = LayoutRect::new(LayoutPoint::new(0.0, 0.0), bounds.size);
        let clip = self.to_clip_region(&yaml["clip"], &sc_full_rect, wrench).unwrap_or(ClipRegion::simple(&sc_full_rect));

        // FIXME handle these
        let mix_blend_mode = MixBlendMode::Normal;
        let filters: Vec<FilterOp> = Vec::new();

        self.builder().push_stacking_context(ScrollPolicy::Scrollable,
                                             bounds,
                                             clip,
                                             z_index as i32,
                                             &transform,
                                             &perspective,
                                             mix_blend_mode,
                                             filters);

        if !yaml["items"].is_badvalue() {
            self.add_display_list_items_from_yaml(wrench, &yaml["items"]);
        }

        self.builder().pop_stacking_context();
    }
}

impl WrenchThing for YamlFrameReader {
    fn do_frame(&mut self, wrench: &mut Wrench) -> u32 {
        if !self.frame_built || self.watch_source {
            self.build(wrench);
            self.frame_built = false;
        }

        self.frame_count += 1;

        if !self.frame_built || wrench.should_rebuild_display_lists() {
            wrench.begin_frame();
            wrench.send_lists(self.frame_count, self.builder.as_ref().unwrap().clone());
        } else {
            wrench.refresh();
        }

        self.frame_built = true;
        self.frame_count
    }

    fn next_frame(&mut self) {
    }

    fn prev_frame(&mut self) {
    }

    fn queue_frames(&self) -> u32 {
        self.queue_depth
    }
}
