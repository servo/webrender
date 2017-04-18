/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use clap;
use euclid::{Point2D, TypedPoint2D, SideOffsets2D};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use webrender_traits::*;
use wrench::{Wrench, WrenchThing, layout_simple_ascii};
use yaml_helper::YamlHelper;
use yaml_rust::{Yaml, YamlLoader};
use {WHITE_COLOR, BLACK_COLOR, PLATFORM_DEFAULT_FACE_NAME};

fn broadcast<T: Clone>(base_vals: &[T], num_items: usize) -> Vec<T> {
    if base_vals.len() == num_items {
        return base_vals.to_vec();
    }

    assert_eq!(num_items % base_vals.len(), 0,
           "Cannot broadcast {} elements into {}", base_vals.len(), num_items);

    let mut vals = vec![];
    loop {
        if vals.len() == num_items {
            break;
        }
        vals.extend_from_slice(base_vals);
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

    /// A HashMap of offsets which specify what scroll offsets particular
    /// scroll layers should be initialized with.
    scroll_offsets: HashMap<ClipId, LayerPoint>,
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
            scroll_offsets: HashMap::new(),
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

    pub fn builder(&mut self) -> &mut DisplayListBuilder {
        self.builder.as_mut().unwrap()
    }

    pub fn reset(&mut self) {
        self.scroll_offsets.clear();
    }

    pub fn build(&mut self, wrench: &mut Wrench) {
        let mut file = File::open(&self.yaml_path).unwrap();
        let mut src = String::new();
        file.read_to_string(&mut src).unwrap();

        let mut yaml_doc = YamlLoader::load_from_str(&src).expect("Failed to parse YAML file");
        assert_eq!(yaml_doc.len(), 1);

        let yaml = yaml_doc.pop().unwrap();
        if !yaml["pipelines"].is_badvalue() {
            let pipelines = yaml["pipelines"].as_vec().unwrap();
            for pipeline in pipelines {
                self.reset();

                let pipeline_id = pipeline["id"].as_pipeline_id().unwrap();
                self.builder = Some(DisplayListBuilder::new(pipeline_id));
                self.add_stacking_context_from_yaml(wrench, pipeline, true);
                wrench.send_lists(self.frame_count,
                                  self.builder.as_ref().unwrap().clone(),
                                  &self.scroll_offsets);
            }

        }

        self.reset();
        self.builder = Some(DisplayListBuilder::new(wrench.root_pipeline_id));

        assert!(!yaml["root"].is_badvalue(), "Missing root stacking context");
        self.add_stacking_context_from_yaml(wrench, &yaml["root"], true);
    }

    fn to_clip_region(&mut self,
                      item: &Yaml,
                      item_bounds: &LayoutRect,
                      wrench: &mut Wrench)
                      -> Option<ClipRegion> {
        match *item {
            Yaml::String(_) => {
                let rect = item.as_rect().expect(&format!("Could not parse rect string: '{:?}'",
                                                          item));
                Some(self.builder().new_clip_region(&rect, vec![], None))
            }
            Yaml::Array(ref v) => {
                if let Some(rect) = item.as_rect() {
                    // it's a rect (as an array)
                    Some(self.builder().new_clip_region(&rect, vec![], None))
                } else {
                    // it may be an array of simple rects
                    let rects = v.iter().map(|v| {
                         v.as_rect().map(|r| {
                            ComplexClipRegion::new(r, BorderRadius::zero())
                        }).ok_or(())
                     })
                     .collect::<Result<Vec<_>, _>>()
                     .expect(&format!("Could not parse clip region array: '{:?}'", item));
                    Some(self.builder().new_clip_region(item_bounds, rects, None))
                }
            }
            Yaml::Hash(_) => {
                let bounds = item["rect"].as_rect().unwrap_or(*item_bounds);
                let complex = item["complex"].as_vec().unwrap_or(&Vec::new()).iter().filter_map(|item|
                    match *item {
                        Yaml::String(_) | Yaml::Array(_) => {
                            let rect = item.as_rect().expect("not a rect");
                            Some(ComplexClipRegion::new(rect, BorderRadius::zero()))
                        }
                        Yaml::Hash(_) => {
                            let rect = item["rect"].as_rect()
                                                   .expect("complex clip entry must have rect");
                            let radius = item["radius"].as_border_radius()
                                                       .unwrap_or(BorderRadius::zero());
                            Some(ComplexClipRegion::new(rect, radius))
                        }
                        _ => {
                            println!("Invalid complex clip region item entry {:?}", item);
                            None
                        }
                    }
                ).collect();

                let image_mask = if item["image-mask"].as_hash().is_some() {
                    let image_mask = &item["image-mask"];
                    let (image_key, image_dims) =
                        wrench.add_or_get_image(&self.rsrc_path(&image_mask["image"]), None);
                    let image_rect =
                        image_mask["rect"].as_rect().unwrap_or(LayoutRect::new(LayoutPoint::zero(),
                                                                               image_dims));
                    let image_repeat = image_mask["repeat"].as_bool().expect("expected boolean");
                    Some(ImageMask { image: image_key, rect: image_rect, repeat: image_repeat })
                } else {
                    None
                };
                Some(self.builder().new_clip_region(&bounds, complex, image_mask))
            }
            Yaml::BadValue => {
                None
            }
            _ => {
                println!("Unable to parse clip region {:?}", item);
                None
            }
        }
    }

    fn to_gradient(&mut self, item: &Yaml) -> Gradient {
        let start = item["start"].as_point().expect("gradient must have start");
        let end = item["end"].as_point().expect("gradient must have end");
        let stops = item["stops"].as_vec().expect("gradient must have stops")
            .chunks(2).map(|chunk| GradientStop {
                offset: chunk[0].as_force_f32().expect("gradient stop offset is not f32"),
                color: chunk[1].as_colorf().expect("gradient stop color is not color"),
            }).collect::<Vec<_>>();
        let extend_mode = if item["repeat"].as_bool().unwrap_or(false) {
            ExtendMode::Repeat
        } else {
            ExtendMode::Clamp
        };

        self.builder().create_gradient(start, end, stops, extend_mode)
    }

    fn to_radial_gradient(&mut self, item: &Yaml) -> RadialGradient {
        if item["start-center"].is_badvalue() {
            let center = item["center"].as_point().expect("radial gradient must have start center");
            let radius = item["radius"].as_size().expect("radial gradient must have start radius");
            let stops = item["stops"].as_vec().expect("radial gradient must have stops")
                .chunks(2).map(|chunk| GradientStop {
                    offset: chunk[0].as_force_f32().expect("gradient stop offset is not f32"),
                    color: chunk[1].as_colorf().expect("gradient stop color is not color"),
                }).collect::<Vec<_>>();
            let extend_mode = if item["repeat"].as_bool().unwrap_or(false) {
                ExtendMode::Repeat
            } else {
                ExtendMode::Clamp
            };

            self.builder().create_radial_gradient(center, radius,
                                                  stops, extend_mode)
        } else {
            let start_center = item["start-center"].as_point().expect("radial gradient must have start center");
            let start_radius = item["start-radius"].as_force_f32().expect("radial gradient must have start radius");
            let end_center = item["end-center"].as_point().expect("radial gradient must have end center");
            let end_radius = item["end-radius"].as_force_f32().expect("radial gradient must have end radius");
            let ratio_xy = item["ratio-xy"].as_force_f32().unwrap_or(1.0);
            let stops = item["stops"].as_vec().expect("radial gradient must have stops")
                .chunks(2).map(|chunk| GradientStop {
                    offset: chunk[0].as_force_f32().expect("gradient stop offset is not f32"),
                    color: chunk[1].as_colorf().expect("gradient stop color is not color"),
                }).collect::<Vec<_>>();
            let extend_mode = if item["repeat"].as_bool().unwrap_or(false) {
                ExtendMode::Repeat
            } else {
                ExtendMode::Clamp
            };

            self.builder().create_complex_radial_gradient(start_center, start_radius,
                                                          end_center, end_radius,
                                                          ratio_xy, stops, extend_mode)
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
        let gradient = self.to_gradient(item);
        let tile_size = item["tile-size"].as_size().unwrap_or(bounds.size);
        let tile_spacing = item["tile-spacing"].as_size().unwrap_or(LayoutSize::zero());

        let clip = self.to_clip_region(&item["clip"], &bounds, wrench).unwrap_or(*clip_region);
        self.builder().push_gradient(bounds, clip, gradient, tile_size, tile_spacing);
    }

    fn handle_radial_gradient(&mut self, wrench: &mut Wrench, clip_region: &ClipRegion, item: &Yaml) {
        let bounds_key = if item["type"].is_badvalue() { "radial-gradient" } else { "bounds" };
        let bounds = item[bounds_key].as_rect().expect("radial gradient must have bounds");
        let gradient = self.to_radial_gradient(item);
        let tile_size = item["tile-size"].as_size().unwrap_or(bounds.size);
        let tile_spacing = item["tile-spacing"].as_size().unwrap_or(LayoutSize::zero());

        let clip = self.to_clip_region(&item["clip"], &bounds, wrench).unwrap_or(*clip_region);
        self.builder().push_radial_gradient(bounds, clip, gradient, tile_size, tile_spacing);
    }

    fn handle_border(&mut self, wrench: &mut Wrench, clip_region: &ClipRegion, item: &Yaml) {
        let bounds_key = if item["type"].is_badvalue() { "border" } else { "bounds" };
        let bounds = item[bounds_key].as_rect().expect("borders must have bounds");
        let widths = item["width"].as_vec_f32().expect("borders must have width(s)");
        let widths = broadcast(&widths, 4);
        let widths = BorderWidths { top: widths[0], left: widths[1], bottom: widths[2], right: widths[3] };
        let border_details = if let Some(border_type) = item["border-type"].as_str() {
            match border_type {
                "normal" => {
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

                    let colors = broadcast(&colors, 4);
                    let styles = broadcast(&styles, 4);

                    let top = BorderSide { color: colors[0], style: styles[0] };
                    let left = BorderSide { color: colors[1], style: styles[1] };
                    let bottom = BorderSide { color: colors[2], style: styles[2] };
                    let right = BorderSide { color: colors[3], style: styles[3] };
                    Some(BorderDetails::Normal(NormalBorder {
                        top: top,
                        left: left,
                        bottom: bottom,
                        right: right,
                        radius: radius,
                    }))
                },
                "image" => {
                    let image = &item["image"];
                    let (image_key, _) =
                        wrench.add_or_get_image(&self.rsrc_path(&image["image"]), None);
                    let image_width = item["image-width"].as_i64().expect("border must have image-width");
                    let image_height = item["image-height"].as_i64().expect("border must have image-height");
                    let slice = item["slice"].as_vec_u32().expect("border must have slice");
                    let slice = broadcast(&slice, 4);
                    let outset = item["outset"].as_vec_f32().expect("border must have outset");
                    let outset = broadcast(&outset, 4);
                    let repeat_horizontal =
                        match item["repeat-horizontal"].as_str().expect("border must have repeat-horizontal") {
                            "stretch" => RepeatMode::Stretch,
                            "repeat" => RepeatMode::Repeat,
                            "round" => RepeatMode::Round,
                            "space" => RepeatMode::Space,
                            s => panic!("Unknown box border image repeat mode {}", s),
                        };
                    let repeat_vertical =
                        match item["repeat-vertical"].as_str().expect("border must have repeat-vertical") {
                            "stretch" => RepeatMode::Stretch,
                            "repeat" => RepeatMode::Repeat,
                            "round" => RepeatMode::Round,
                            "space" => RepeatMode::Space,
                            s => panic!("Unknown box border image repeat mode {}", s),
                        };
                    Some(BorderDetails::Image(ImageBorder {
                        image_key: image_key,
                        patch: NinePatchDescriptor {
                            width: image_width as u32,
                            height: image_height as u32,
                            slice: SideOffsets2D::new(slice[0], slice[1], slice[2], slice[3]),
                        },
                        outset: SideOffsets2D::new(outset[0], outset[1], outset[2], outset[3]),
                        repeat_horizontal: repeat_horizontal,
                        repeat_vertical: repeat_vertical,
                    }))
                },
                "gradient" => {
                    let gradient = self.to_gradient(item);
                    let outset = item["outset"].as_vec_f32().expect("borders must have outset");
                    let outset = broadcast(&outset, 4);
                    Some(BorderDetails::Gradient(GradientBorder {
                        gradient: gradient,
                        outset: SideOffsets2D::new(outset[0], outset[1], outset[2], outset[3]),
                    }))
                },
                "radial-gradient" => {
                    let gradient = self.to_radial_gradient(item);
                    let outset = item["outset"].as_vec_f32().expect("borders must have outset");
                    let outset = broadcast(&outset, 4);
                    Some(BorderDetails::RadialGradient(RadialGradientBorder {
                        gradient: gradient,
                        outset: SideOffsets2D::new(outset[0], outset[1], outset[2], outset[3]),
                    }))
                },
                _ => {
                    println!("Unable to parse border {:?}", item);
                    None
                },
            }
        } else {
            println!("Unable to parse border {:?}", item);
            None
        };
        let clip = self.to_clip_region(&item["clip"], &bounds, wrench).unwrap_or(*clip_region);
        if let Some(details) = border_details {
            self.builder().push_border(bounds,
                                       clip,
                                       widths,
                                       details);
        }
    }

    fn handle_box_shadow(&mut self, wrench: &mut Wrench, clip_region: &ClipRegion, item: &Yaml) {
        let bounds_key = if item["type"].is_badvalue() { "box-shadow" } else { "bounds" };
        let bounds = item[bounds_key].as_rect().expect("box shadow must have bounds");
        let box_bounds = item["box-bounds"].as_rect().unwrap_or(bounds);
        let offset = item["offset"].as_point().unwrap_or(TypedPoint2D::zero());
        let color = item["color"].as_colorf().unwrap_or(ColorF::new(0.0, 0.0, 0.0, 1.0));
        let blur_radius = item["blur-radius"].as_force_f32().unwrap_or(0.0);
        let spread_radius = item["spread-radius"].as_force_f32().unwrap_or(0.0);
        let border_radius = item["border-radius"].as_force_f32().unwrap_or(0.0);
        let clip_mode = if let Some(mode) = item["clip-mode"].as_str() {
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
        let tiling = item["tile-size"].as_i64();
        let (image_key, image_dims) = wrench.add_or_get_image(&self.rsrc_path(filename), tiling);

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

        let stretch_size = item["stretch-size"].as_size()
            .unwrap_or(image_dims);
        let tile_spacing = item["tile-spacing"].as_size()
            .unwrap_or(LayoutSize::new(0.0, 0.0));
        let rendering = match item["rendering"].as_str() {
            Some("auto") | None => ImageRendering::Auto,
            Some("crisp-edges") => ImageRendering::CrispEdges,
            Some("pixelated") => ImageRendering::Pixelated,
            Some(_) => panic!("ImageRendering can be auto, crisp-edges, or pixelated -- got {:?}", item),
        };
        let clip = self.to_clip_region(&item["clip"], &bounds, wrench).unwrap_or(*clip_region);
        self.builder().push_image(bounds, clip, stretch_size, tile_spacing, rendering, image_key);
    }

    fn handle_text(&mut self, wrench: &mut Wrench, clip_region: &ClipRegion, item: &Yaml) {
        let size = item["size"].as_pt_to_au().unwrap_or(Au::from_f32_px(16.0));
        let color = item["color"].as_colorf().unwrap_or(*BLACK_COLOR);
        let blur_radius = item["blur-radius"].as_px_to_au().unwrap_or(Au::from_f32_px(0.0));

        let (font_key, native_key) = if !item["family"].is_badvalue() {
            wrench.font_key_from_yaml_table(item)
        } else if !item["font"].is_badvalue() {
            let font_file = self.rsrc_path(&item["font"]);
            let font_index = item["font-index"].as_i64().unwrap_or(0) as u32;
            let mut file = File::open(&font_file).expect("Couldn't open font file");
            let mut bytes = vec![];
            file.read_to_end(&mut bytes).expect("failed to read font file");
            wrench.font_key_from_bytes(bytes, font_index)
        } else {
            wrench.font_key_from_name(&*PLATFORM_DEFAULT_FACE_NAME)
        };

        assert!(!(item["glyphs"].is_badvalue() && item["text"].is_badvalue()),
               "text item had neither text nor glyphs!");

        let (glyphs, rect) = if item["text"].is_badvalue() {
            // if glyphs are specified, then the glyph positions can have the
            // origin baked in.
            let origin = item["origin"].as_point().unwrap_or(LayoutPoint::new(0.0, 0.0));
            let glyph_indices = item["glyphs"].as_vec_u32().unwrap();
            let glyph_offsets = item["offsets"].as_vec_f32().unwrap();
            assert_eq!(glyph_offsets.len(), glyph_indices.len() * 2);

            let glyphs = glyph_indices.iter().enumerate().map(|k| {
                GlyphInstance {
                    index: *k.1,
                    point: Point2D::new(origin.x + glyph_offsets[k.0 * 2],
                                        origin.y + glyph_offsets[k.0 * 2 + 1])
                }
            }).collect::<Vec<_>>();
            // TODO(gw): We could optionally use the WR API to query glyph dimensions
            //           here and calculate the bounding region here if we want to.
            let rect = item["bounds"].as_rect()
                                     .expect("Text items with glyphs require bounds [for now]");
            (glyphs, rect)
        } else {
            assert!(native_key.is_some(), "Can't layout simple ascii text with raw font [for now]");
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
                let gi = GlyphInstance { index: *arg.0 as u32,
                                         point: Point2D::new(x, y), };
                x += arg.1;
                gi
            }).collect::<Vec<_>>();
            // FIXME this is incorrect!
            let rect = LayoutRect::new(LayoutPoint::new(0.0, 0.0), wrench.window_size_f32());
            (glyphs, rect)
        };

        let clip = self.to_clip_region(&item["clip"], &rect, wrench).unwrap_or(*clip_region);
        self.builder().push_text(rect, clip, &glyphs, font_key, color, size, blur_radius, None);
    }

    fn handle_iframe(&mut self, wrench: &mut Wrench, clip_region: &ClipRegion, item: &Yaml) {
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

        for item in yaml.as_vec().unwrap() {
            // an explicit type can be skipped with some shorthand
            let item_type =
                if !item["rect"].is_badvalue() {
                    "rect"
                } else if !item["image"].is_badvalue() {
                    "image"
                } else if !item["text"].is_badvalue() {
                     "text"
                } else if !item["glyphs"].is_badvalue() {
                    "glyphs"
                } else if !item["box-shadow"].is_badvalue() {
                    // Note: box_shadow shorthand check has to come before border.
                    "box-shadow"
                } else if !item["border"].is_badvalue() {
                    "border"
                } else if !item["gradient"].is_badvalue() {
                    "gradient"
                } else if !item["radial-gradient"].is_badvalue() {
                    "radial-gradient"
                } else {
                    item["type"].as_str().unwrap_or("unknown")
                };

            if item_type == "stacking-context" {
                self.add_stacking_context_from_yaml(wrench, item, false);
                continue;
            }


            if !self.include_only.is_empty() && !self.include_only.contains(&item_type.to_owned()) {
                continue;
            }

            let yaml_clip_id = item["clip-id"].as_i64();
            if let Some(yaml_id) = yaml_clip_id {
                let id = ClipId::new(yaml_id as u64, self.builder().pipeline_id);
                self.builder().push_clip_id(id);
            }

            match item_type {
                "rect" => self.handle_rect(wrench, &full_clip_region, item),
                "image" => self.handle_image(wrench, &full_clip_region, item),
                "text" | "glyphs" => self.handle_text(wrench, &full_clip_region, item),
                "scroll-layer" => self.add_scroll_layer_from_yaml(wrench, item),
                "clip" => { self.handle_clip_from_yaml(wrench, item); }
                "border" => self.handle_border(wrench, &full_clip_region, item),
                "gradient" => self.handle_gradient(wrench, &full_clip_region, item),
                "radial-gradient" => self.handle_radial_gradient(wrench, &full_clip_region, item),
                "box-shadow" => self.handle_box_shadow(wrench, &full_clip_region, item),
                "iframe" => self.handle_iframe(wrench, &full_clip_region, item),
                "stacking-context" => { },
                _ => println!("Skipping unknown item type: {:?}", item),
            }

            if yaml_clip_id.is_some() {
                self.builder().pop_clip_id();
            }
        }
    }

    pub fn add_scroll_layer_from_yaml(&mut self, wrench: &mut Wrench, yaml: &Yaml) {
        let id = self.handle_clip_from_yaml(wrench, yaml);

        self.builder().push_clip_id(id);
        if !yaml["items"].is_badvalue() {
            self.add_display_list_items_from_yaml(wrench, &yaml["items"]);
        }
        self.builder().pop_clip_id();
    }

    pub fn handle_clip_from_yaml(&mut self, wrench: &mut Wrench, yaml: &Yaml) -> ClipId {
        let content_rect = yaml["bounds"].as_rect().expect("scroll layer must have content rect");

        let default_clip = LayoutRect::new(LayoutPoint::zero(), content_rect.size);
        let clip = self.to_clip_region(&yaml["clip"], &default_clip, wrench)
                       .unwrap_or(ClipRegion::simple(&default_clip));
        let id = yaml["id"].as_i64().map(|id| ClipId::new(id as u64, self.builder().pipeline_id));

        let id = self.builder().define_clip(content_rect, clip, id);

        if let Some(size) = yaml["scroll-offset"].as_point() {
            self.scroll_offsets.insert(id, LayerPoint::new(size.x, size.y));
        }

        id
    }

    pub fn add_stacking_context_from_yaml(&mut self,
                                          wrench: &mut Wrench,
                                          yaml: &Yaml,
                                          is_root: bool) {
        let default_bounds = LayoutRect::new(LayoutPoint::zero(), wrench.window_size_f32());
        let bounds = yaml["bounds"].as_rect().unwrap_or(default_bounds);
        let z_index = yaml["z-index"].as_i64().unwrap_or(0);

        // TODO(gw): Add support for specifying the transform origin in yaml.
        let transform_origin = LayoutPoint::new(bounds.origin.x + bounds.size.width * 0.5,
                                                bounds.origin.y + bounds.size.height * 0.5);

        let transform = yaml["transform"].as_matrix4d(&transform_origin).map(
            |transform| transform.into());

        // TODO(gw): Support perspective-origin.
        let perspective = match yaml["perspective"].as_force_f32() {
            Some(perspective) if perspective == 0.0 => None,
            Some(perspective) => Some(LayoutTransform::create_perspective(perspective)),
            None => None,
        };

        let transform_style = yaml["transform-style"].as_transform_style()
                                                     .unwrap_or(TransformStyle::Flat);
        let mix_blend_mode = yaml["mix-blend-mode"].as_mix_blend_mode()
                                                   .unwrap_or(MixBlendMode::Normal);
        let scroll_policy = yaml["scroll-policy"].as_scroll_policy()
                                                 .unwrap_or(ScrollPolicy::Scrollable);

        if is_root {
            if let Some(size) = yaml["scroll-offset"].as_point() {
                let id = ClipId::root_scroll_node(self.builder().pipeline_id);
                self.scroll_offsets.insert(id, LayerPoint::new(size.x, size.y));
            }
        }

        let filters = yaml["filters"].as_vec_filter_op().unwrap_or(vec![]);

        self.builder().push_stacking_context(scroll_policy,
                                             bounds,
                                             z_index as i32,
                                             transform.into(),
                                             transform_style,
                                             perspective,
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
            wrench.send_lists(self.frame_count,
                              self.builder.as_ref().unwrap().clone(),
                              &self.scroll_offsets);
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
