/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate yaml_rust;

use euclid::{TypedMatrix4D, TypedPoint2D, TypedRect, TypedSize2D};
use image::{ColorType, save_buffer};
use premultiply::unpremultiply;
use scene::Scene;
use std::borrow::BorrowMut;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::{fmt, fs, slice};
use super::CURRENT_FRAME_NUMBER;
use time;
use webrender;
use webrender_traits::*;
use webrender_traits::SpecificDisplayItem::*;
use yaml_helper::{mix_blend_mode_to_string, scroll_policy_to_string};
use yaml_rust::{Yaml, YamlEmitter};

type Table = yaml_rust::yaml::Hash;

fn array_elements_are_same<T: PartialEq>(v: &[T]) -> bool {
    if v.len() > 0 {
        let first = &v[0];
        for o in v.iter() {
            if *first != *o {
                return false;
            }
        }
    }
    true
}

fn new_table() -> Table {
    Table::new()
}

fn yaml_node(parent: &mut Table, key: &str, value: Yaml) {
    parent.insert(Yaml::String(key.to_owned()), value);
}

fn str_node(parent: &mut Table, key: &str, value: &str) {
    yaml_node(parent, key, Yaml::String(value.to_owned()));
}

fn path_node(parent: &mut Table, key: &str, value: &Path) {
    let pstr = value.to_str().unwrap().to_owned().replace("\\", "/");
    yaml_node(parent, key, Yaml::String(pstr));
}

fn color_to_string(value: ColorF) -> String {
    if value.r == 1.0 && value.g == 1.0 && value.b == 1.0 && value.a == 1.0 {
        "white".to_owned()
    } else if value.r == 0.0 && value.g == 0.0 && value.b == 0.0 && value.a == 1.0 {
        "black".to_owned()
    } else {
        format!("{} {} {} {:.4}", value.r * 255.0, value.g * 255.0, value.b * 255.0, value.a)
    }
}

fn color_node(parent: &mut Table, key: &str, value: ColorF) {
    yaml_node(parent, key, Yaml::String(color_to_string(value)));
}

fn point_node<U>(parent: &mut Table, key: &str, value: &TypedPoint2D<f32, U>) {
    f32_vec_node(parent, key, &[value.x, value.y]);
}

fn size_node<U>(parent: &mut Table, key: &str, value: &TypedSize2D<f32, U>) {
    f32_vec_node(parent, key, &[value.width, value.height]);
}

fn rect_yaml<U>(value: &TypedRect<f32, U>) -> Yaml {
    f32_vec_yaml(&[value.origin.x, value.origin.y, value.size.width, value.size.height], false)
}

fn rect_node<U>(parent: &mut Table, key: &str, value: &TypedRect<f32, U>) {
    yaml_node(parent, key, rect_yaml(value));
}

fn matrix4d_node<U1, U2>(parent: &mut Table, key: &str, value: &TypedMatrix4D<f32, U1, U2>) {
    f32_vec_node(parent, key, &value.to_row_major_array());
}

fn u32_node(parent: &mut Table, key: &str, value: u32) {
    yaml_node(parent, key, Yaml::Integer(value as i64));
}

fn usize_node(parent: &mut Table, key: &str, value: usize) {
    yaml_node(parent, key, Yaml::Integer(value as i64));
}

fn i32_node(parent: &mut Table, key: &str, value: i32) {
    yaml_node(parent, key, Yaml::Integer(value as i64));
}

fn f32_node(parent: &mut Table, key: &str, value: f32) {
    yaml_node(parent, key, Yaml::Real(value.to_string()));
}

fn bool_node(parent: &mut Table, key: &str, value: bool) {
    yaml_node(parent, key, Yaml::Boolean(value));
}

fn table_node(parent: &mut Table, key: &str, value: Table) {
    yaml_node(parent, key, Yaml::Hash(value));
}

fn string_vec_yaml(value: &[String], check_unique: bool) -> Yaml {
    if value.len() > 0 && check_unique && array_elements_are_same(value) {
        Yaml::String(value[0].clone())
    } else {
        Yaml::Array(value.iter().map(|v| Yaml::String(v.clone())).collect())
    }
}

fn u32_vec_yaml(value: &[u32], check_unique: bool) -> Yaml {
    if value.len() > 0 && check_unique && array_elements_are_same(value) {
        Yaml::Integer(value[0] as i64)
    } else {
        Yaml::Array(value.iter().map(|v| Yaml::Integer(*v as i64)).collect())
    }
}

fn u32_vec_node(parent: &mut Table, key: &str, value: &[u32]) {
    yaml_node(parent, key, u32_vec_yaml(value, false));
}

fn f32_vec_yaml(value: &[f32], check_unique: bool) -> Yaml {
    if value.len() > 0 && check_unique && array_elements_are_same(value) {
        Yaml::Real(value[0].to_string())
    } else {
        Yaml::Array(value.iter().map(|v| Yaml::Real(v.to_string())).collect())
    }
}

fn f32_vec_node(parent: &mut Table, key: &str, value: &[f32]) {
    yaml_node(parent, key, f32_vec_yaml(value, false));
}

fn vec_node(parent: &mut Table, key: &str, value: Vec<Yaml>) {
    yaml_node(parent, key, Yaml::Array(value));
}

fn mix_blend_mode_node(parent: &mut Table, key: &str, value: MixBlendMode) {
    yaml_node(parent, key, Yaml::String(mix_blend_mode_to_string(value).to_owned()));
}

fn scroll_policy_node(parent: &mut Table, key: &str, value: ScrollPolicy) {
    yaml_node(parent, key, Yaml::String(scroll_policy_to_string(value).to_owned()));
}

fn maybe_radius_yaml(radius: &BorderRadius) -> Option<Yaml> {
    if let Some(radius) = radius.is_uniform_size() {
        if radius == LayoutSize::zero() {
            None
        } else {
            Some(f32_vec_yaml(&vec![radius.width, radius.height], false))
        }
    } else {
        let mut table = new_table();
        size_node(&mut table, "top-left", &radius.top_left);
        size_node(&mut table, "top-right", &radius.top_right);
        size_node(&mut table, "bottom-left", &radius.bottom_left);
        size_node(&mut table, "bottom-right", &radius.bottom_right);
        Some(Yaml::Hash(table))
    }
}

fn write_sc(parent: &mut Table, sc: &StackingContext) {
    scroll_policy_node(parent, "scroll-policy", sc.scroll_policy);
    i32_node(parent, "z-index", sc.z_index);

    match sc.transform {
        Some(PropertyBinding::Value(transform)) => matrix4d_node(parent, "transform", &transform),
        Some(PropertyBinding::Binding(..)) => panic!("TODO: Handle property bindings in wrench!"),
        None => {}
    };

    if let Some(perspective) = sc.perspective {
        matrix4d_node(parent, "perspective", &perspective);
    }

    // mix_blend_mode
    if sc.mix_blend_mode != MixBlendMode::Normal {
        mix_blend_mode_node(parent, "mix-blend-mode", sc.mix_blend_mode)
    }
    // filters
}

#[cfg(target_os = "windows")]
fn native_font_handle_to_yaml(handle: &NativeFontHandle, parent: &mut yaml_rust::yaml::Hash) {
    str_node(parent, "family", &handle.family_name);
    u32_node(parent, "weight", handle.weight.to_u32());
    u32_node(parent, "style", handle.style.to_u32());
    u32_node(parent, "stretch", handle.stretch.to_u32());
}

#[cfg(not(target_os = "windows"))]
fn native_font_handle_to_yaml(_: &NativeFontHandle, _: &mut yaml_rust::yaml::Hash) {
    panic!("Can't native_handle_to_yaml on this platform");
}

enum CachedFont {
    Native(NativeFontHandle),
    Raw(Option<Vec<u8>>, Option<PathBuf>),
}

struct CachedImage {
    width: u32,
    height: u32,
    stride: u32,
    format: ImageFormat,
    bytes: Option<Vec<u8>>,
    path: Option<PathBuf>,
    tiling: Option<u16>,
}

pub struct YamlFrameWriter {
    frame_base: PathBuf,
    rsrc_base: PathBuf,
    next_rsrc_num: u32,
    rsrc_prefix: String,
    images: HashMap<ImageKey, CachedImage>,
    fonts: HashMap<FontKey, CachedFont>,

    last_frame_written: u32,
    pipeline_id: Option<PipelineId>,

    dl_descriptor: Option<BuiltDisplayListDescriptor>,
    aux_descriptor: Option<AuxiliaryListsDescriptor>,
}

pub struct YamlFrameWriterReceiver {
    frame_writer: YamlFrameWriter,
    scene: Scene,
}

impl YamlFrameWriterReceiver {
    pub fn new(path: &Path) -> YamlFrameWriterReceiver {
        YamlFrameWriterReceiver {
            frame_writer: YamlFrameWriter::new(path),
            scene: Scene::new(),
        }
    }
}

impl fmt::Debug for YamlFrameWriterReceiver {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "YamlFrameWriterReceiver")
    }
}

impl YamlFrameWriter {
    pub fn new(path: &Path) -> YamlFrameWriter {
        let mut rsrc_base = path.to_owned();
        rsrc_base.push("res");
        fs::create_dir_all(&rsrc_base).ok();

        let rsrc_prefix = format!("{}", time::get_time().sec);

        YamlFrameWriter {
            frame_base: path.to_owned(),
            rsrc_base: rsrc_base,
            rsrc_prefix: rsrc_prefix,
            next_rsrc_num: 1,
            images: HashMap::new(),
            fonts: HashMap::new(),

            dl_descriptor: None,
            aux_descriptor: None,

            pipeline_id: None,

            last_frame_written: u32::max_value(),
        }
    }

    pub fn begin_write_root_display_list(&mut self,
                                         scene: &mut Scene,
                                         background_color: &Option<ColorF>,
                                         epoch: &Epoch,
                                         pipeline_id: &PipelineId,
                                         viewport_size: &LayoutSize,
                                         display_list: &BuiltDisplayListDescriptor,
                                         auxiliary_lists: &AuxiliaryListsDescriptor)
    {
        unsafe {
            if CURRENT_FRAME_NUMBER == self.last_frame_written {
                return;
            }
            self.last_frame_written = CURRENT_FRAME_NUMBER;
        }

        self.dl_descriptor = Some(display_list.clone());
        self.aux_descriptor = Some(auxiliary_lists.clone());
        self.pipeline_id = Some(pipeline_id.clone());

        scene.begin_root_display_list(pipeline_id, epoch,
                                      background_color,
                                      viewport_size);
    }

    pub fn finish_write_root_display_list(&mut self, scene: &mut Scene, data: &[u8]) {
        let dl_desc = self.dl_descriptor.take().unwrap();
        let aux_desc = self.aux_descriptor.take().unwrap();

        assert_eq!(data.len(), dl_desc.size() + aux_desc.size() + 4);

        // skip 4-byte epoch header
        let dl_data = data[4..dl_desc.size() + 4].to_vec();
        let aux_data = data[dl_desc.size() + 4..].to_vec();

        let dl = BuiltDisplayList::from_data(dl_data, dl_desc);
        let aux = AuxiliaryLists::from_data(aux_data, aux_desc);

        let mut root_dl_table = new_table();
        {
            let mut iter = dl.all_display_items().iter();
            self.write_display_list(&mut root_dl_table, &mut iter, &aux, &mut ClipIdMapper::new());
        }


        let mut root = new_table();
        if let Some(root_pipeline_id) = scene.root_pipeline_id {
            u32_vec_node(&mut root_dl_table, "id", &vec![root_pipeline_id.0, root_pipeline_id.1]);

            let referenced_pipeline_ids = dl.all_display_items().iter()
                .flat_map(|base| {
                    if let SpecificDisplayItem::Iframe(k) = base.item {
                        Some(k.pipeline_id)
                    } else {
                        None
                    }
                });

            let mut pipelines = vec![];
            for pipeline_id in referenced_pipeline_ids {
                if !scene.display_lists.contains_key(&pipeline_id) {
                    continue;
                }
                let mut pipeline = new_table();
                u32_vec_node(&mut pipeline, "id", &vec![pipeline_id.0, pipeline_id.1]);

                let dl = scene.display_lists.get(&pipeline_id).unwrap();
                let aux = scene.pipeline_auxiliary_lists.get(&pipeline_id).unwrap();
                let mut iter = dl.iter();
                self.write_display_list(&mut pipeline, &mut iter, &aux, &mut ClipIdMapper::new());
                pipelines.push(Yaml::Hash(pipeline));
            }

            table_node(&mut root, "root", root_dl_table);

            root.insert(Yaml::String("pipelines".to_owned()), Yaml::Array(pipelines));

            let mut s = String::new();
            // FIXME YamlEmitter wants a std::fmt::Write, not a io::Write, so we can't pass a file
            // directly.  This seems broken.
            {
                let mut emitter = YamlEmitter::new(&mut s);
                emitter.dump(&Yaml::Hash(root)).unwrap();
            }
            let sb = s.into_bytes();
            let mut frame_file_name = self.frame_base.clone();
            let current_shown_frame = unsafe { CURRENT_FRAME_NUMBER };
            frame_file_name.push(format!("frame-{}.yaml", current_shown_frame));
            let mut file = File::create(&frame_file_name).unwrap();
            file.write_all(&sb).unwrap();

        }

        scene.finish_root_display_list(self.pipeline_id.unwrap(), dl, aux);
    }

    fn next_rsrc_paths(prefix: &str, counter: &mut u32, base_path: &Path, base: &str, ext: &str) -> (PathBuf, PathBuf) {
        let mut path_file = base_path.to_owned();
        let mut path = PathBuf::from("res");

        let fstr = format!("{}-{}-{}.{}", prefix, base, counter, ext);
        path_file.push(&fstr);
        path.push(&fstr);

        *counter += 1;

        (path_file, path)
    }

    fn path_for_image(&mut self, key: &ImageKey) -> Option<PathBuf> {
        if let Some(ref mut data) = self.images.get_mut(&key) {
            if data.path.is_some() {
                return data.path.clone();
            }
        } else {
            return None;
        };

        // Remove the data to munge it
        let mut data = self.images.remove(&key).unwrap();
        let mut bytes = data.bytes.take().unwrap();
        let (path_file, path) = Self::next_rsrc_paths(&self.rsrc_prefix,
                                                      &mut self.next_rsrc_num,
                                                      &self.rsrc_base,
                                                      "img",
                                                      "png");

        assert!(data.stride > 0);
        let (color_type, bpp) = match data.format {
            ImageFormat::RGB8 => {
                (ColorType::RGB(8), 3)
            }
            ImageFormat::RGBA8 => {
                (ColorType::RGBA(8), 4)
            }
            ImageFormat::A8 => {
                (ColorType::Gray(8), 1)
            }
            _ => {
                println!("Failed to write image with format {:?}, dimensions {}x{}, stride {}",
                         data.format, data.width, data.height, data.stride);
                return None;
            }
        };

        if data.stride == data.width * bpp {
            if data.format == ImageFormat::RGBA8 {
                unpremultiply(bytes.as_mut_slice());
            }
            save_buffer(&path_file, &bytes, data.width, data.height, color_type).unwrap();
        } else {
            // takes a buffer with a stride and copies it into a new buffer that has stride == width
            assert!(data.stride > data.width * bpp);
            let mut tmp: Vec<_>  = bytes[..].chunks(data.stride as usize)
                                            .flat_map(|chunk| chunk[..(data.width * bpp) as usize].iter().cloned())
                                            .collect();
            if data.format == ImageFormat::RGBA8 {
                unpremultiply(tmp.as_mut_slice());
            }

            save_buffer(&path_file, &tmp, data.width, data.height, color_type).unwrap();
        }

        data.path = Some(path.clone());
        // put it back
        self.images.insert(*key, data);
        Some(path)
    }

    fn make_clip_node(&mut self, clip: &ClipRegion, aux: &AuxiliaryLists) -> Yaml {
        if clip.is_complex() {
            let complex = aux.complex_clip_regions(&clip.complex);
            let mut complex_table = new_table();
            rect_node(&mut complex_table, "rect", &clip.main);

            if complex.len() > 0 {
                let complex_items = complex.iter().map(|ccx|
                    if ccx.radii.is_zero() {
                        rect_yaml(&ccx.rect)
                    } else {
                        let mut t = new_table();
                        rect_node(&mut t, "rect", &ccx.rect);
                        yaml_node(&mut t, "radius", maybe_radius_yaml(&ccx.radii).unwrap());
                        Yaml::Hash(t)
                    }
                ).collect();
                vec_node(&mut complex_table, "complex", complex_items);
            }

            if let Some(ref mask) = clip.image_mask {
                let mut mask_table = new_table();
                if let Some(path) = self.path_for_image(&mask.image) {
                    path_node(&mut mask_table, "image", &path);
                }
                rect_node(&mut mask_table, "rect", &mask.rect);
                bool_node(&mut mask_table, "repeat", mask.repeat);

                table_node(&mut complex_table, "image-mask", mask_table);
            }

            Yaml::Hash(complex_table)
        } else {
            rect_yaml(&clip.main)
        }
    }

    fn write_display_list_items(&mut self,
                                list: &mut Vec<Yaml>,
                                list_iterator: &mut slice::Iter<DisplayItem>,
                                aux: &AuxiliaryLists,
                                clip_id_mapper: &mut ClipIdMapper) {
        while let Some(ref base) = list_iterator.next() {
            let mut v = new_table();
            rect_node(&mut v, "bounds", &base.rect);
            yaml_node(&mut v, "clip", self.make_clip_node(&base.clip, aux));
            usize_node(&mut v, "clip-id", clip_id_mapper.map(&base.scroll_layer_id));

            match base.item {
                Rectangle(item) => {
                    str_node(&mut v, "type", "rect");
                    color_node(&mut v, "color", item.color);
                },
                Text(item) => {
                    let gi = aux.glyph_instances(&item.glyphs);
                    let mut indices: Vec<u32> = vec![];
                    let mut offsets: Vec<f32> = vec![];
                    for ref g in gi.iter() {
                        indices.push(g.index);
                        offsets.push(g.point.x);
                        offsets.push(g.point.y);
                    }
                    u32_vec_node(&mut v, "glyphs", &indices);
                    f32_vec_node(&mut v, "offsets", &offsets);
                    f32_node(&mut v, "size", item.size.to_f32_px() * 12.0 / 16.0);
                    color_node(&mut v, "color", item.color);

                    let entry = self.fonts.entry(item.font_key).or_insert_with(|| {
                        println!("Warning: font key not found in fonts table!");
                        CachedFont::Raw(Some(vec![]), None)
                    });

                    match entry {
                        &mut CachedFont::Native(ref handle) => {
                            native_font_handle_to_yaml(&handle, &mut v);
                        }
                        &mut CachedFont::Raw(ref mut bytes_opt, ref mut path_opt) => {
                            if let Some(bytes) = bytes_opt.take() {
                                let (path_file, path) =
                                    Self::next_rsrc_paths(&self.rsrc_prefix,
                                                          &mut self.next_rsrc_num,
                                                          &self.rsrc_base,
                                                          "font",
                                                          "ttf");
                                let mut file = File::create(&path_file).unwrap();
                                file.write_all(&bytes).unwrap();
                                *path_opt = Some(path);
                            }

                            path_node(&mut v, "font", path_opt.as_ref().unwrap());
                        }
                    }
                },
                Image(item) => {
                    if let Some(path) = self.path_for_image(&item.image_key) {
                        path_node(&mut v, "image", &path);
                    }
                    if let Some(&CachedImage { tiling: Some(tile_size), .. }) = self.images.get(&item.image_key) {
                        u32_node(&mut v, "tile-size", tile_size as u32);
                    }
                    size_node(&mut v, "stretch-size", &item.stretch_size);
                    size_node(&mut v, "tile-spacing", &item.tile_spacing);
                    match item.image_rendering {
                        ImageRendering::Auto => (),
                        ImageRendering::CrispEdges => str_node(&mut v, "rendering", "crisp-edges"),
                        ImageRendering::Pixelated => str_node(&mut v, "rendering", "pixelated"),
                    };
                },
                YuvImage(_) => {
                    str_node(&mut v, "type", "yuv-image");
                    // TODO
                    println!("TODO YAML YuvImage");
                },
                WebGL(_) => {
                    str_node(&mut v, "type", "webgl");
                    // TODO
                    println!("TODO YAML WebGL");
                },
                Border(item) => {
                    str_node(&mut v, "type", "border");
                    match item.details {
                        BorderDetails::Normal(ref details) => {
                            let trbl = vec![&details.top, &details.right, &details.bottom, &details.left];
                            let widths: Vec<f32> = vec![ item.widths.top,
                                                         item.widths.right,
                                                         item.widths.bottom,
                                                         item.widths.left ];
                            let colors: Vec<String> = trbl.iter().map(|x| color_to_string(x.color)).collect();
                            let styles: Vec<String> = trbl.iter().map(|x| {
                                match x.style {
                                    BorderStyle::None => "none",
                                    BorderStyle::Solid => "solid",
                                    BorderStyle::Double => "double",
                                    BorderStyle::Dotted => "dotted",
                                    BorderStyle::Dashed => "dashed",
                                    BorderStyle::Hidden => "hidden",
                                    BorderStyle::Ridge => "ridge",
                                    BorderStyle::Inset => "inset",
                                    BorderStyle::Outset => "outset",
                                    BorderStyle::Groove => "groove",
                                }.to_owned()
                            }).collect();
                            yaml_node(&mut v, "width", f32_vec_yaml(&widths, true));
                            str_node(&mut v, "border-type", "normal");
                            yaml_node(&mut v, "color", string_vec_yaml(&colors, true));
                            yaml_node(&mut v, "style", string_vec_yaml(&styles, true));
                            if let Some(radius_node) = maybe_radius_yaml(&details.radius) {
                                yaml_node(&mut v, "radius", radius_node);
                            }
                        }
                        BorderDetails::Image(ref details) => {
                            let widths: Vec<f32> = vec![ item.widths.top,
                                                         item.widths.right,
                                                         item.widths.bottom,
                                                         item.widths.left ];
                            let outset: Vec<f32> = vec![ details.outset.top,
                                                         details.outset.right,
                                                         details.outset.bottom,
                                                         details.outset.left];
                            yaml_node(&mut v, "width", f32_vec_yaml(&widths, true));
                            str_node(&mut v, "border-type", "image");
                            if let Some(path) = self.path_for_image(&details.image_key) {
                                path_node(&mut v, "image", &path);
                            }
                            u32_node(&mut v, "image-width", details.patch.width);
                            u32_node(&mut v, "image-height", details.patch.height);
                            let slice: Vec<u32> = vec![ details.patch.slice.top,
                                                        details.patch.slice.right,
                                                        details.patch.slice.bottom,
                                                        details.patch.slice.left ];
                            yaml_node(&mut v, "slice", u32_vec_yaml(&slice, true));
                            yaml_node(&mut v, "outset", f32_vec_yaml(&outset, true));
                            match details.repeat_horizontal {
                                RepeatMode::Stretch => str_node(&mut v, "repeat-horizontal", "stretch"),
                                RepeatMode::Repeat => str_node(&mut v, "repeat-horizontal", "repeat"),
                                RepeatMode::Round => str_node(&mut v, "repeat-horizontal", "round"),
                                RepeatMode::Space => str_node(&mut v, "repeat-horizontal", "space"),
                            };
                            match details.repeat_vertical {
                                RepeatMode::Stretch => str_node(&mut v, "repeat-vertical", "stretch"),
                                RepeatMode::Repeat => str_node(&mut v, "repeat-vertical", "repeat"),
                                RepeatMode::Round => str_node(&mut v, "repeat-vertical", "round"),
                                RepeatMode::Space => str_node(&mut v, "repeat-vertical", "space"),
                            };
                        }
                        BorderDetails::Gradient(ref details) => {
                            let widths: Vec<f32> = vec![ item.widths.top,
                                                         item.widths.right,
                                                         item.widths.bottom,
                                                         item.widths.left ];
                            let outset: Vec<f32> = vec![ details.outset.top,
                                                         details.outset.right,
                                                         details.outset.bottom,
                                                         details.outset.left];
                            yaml_node(&mut v, "width", f32_vec_yaml(&widths, true));
                            str_node(&mut v, "border-type", "gradient");
                            point_node(&mut v, "start", &details.gradient.start_point);
                            point_node(&mut v, "end", &details.gradient.end_point);
                            let mut stops = vec![];
                            for stop in aux.gradient_stops(&details.gradient.stops) {
                                stops.push(Yaml::Real(stop.offset.to_string()));
                                stops.push(Yaml::String(color_to_string(stop.color)));
                            }
                            yaml_node(&mut v, "stops", Yaml::Array(stops));
                            bool_node(&mut v, "repeat", details.gradient.extend_mode == ExtendMode::Repeat);
                            yaml_node(&mut v, "outset", f32_vec_yaml(&outset, true));
                        }
                        BorderDetails::RadialGradient(ref details) => {
                            let widths: Vec<f32> = vec![ item.widths.top,
                                                         item.widths.right,
                                                         item.widths.bottom,
                                                         item.widths.left ];
                            let outset: Vec<f32> = vec![ details.outset.top,
                                                         details.outset.right,
                                                         details.outset.bottom,
                                                         details.outset.left];
                            yaml_node(&mut v, "width", f32_vec_yaml(&widths, true));
                            str_node(&mut v, "border-type", "radial-gradient");
                            point_node(&mut v, "start-center", &details.gradient.start_center);
                            f32_node(&mut v, "start-radius", details.gradient.start_radius);
                            point_node(&mut v, "end-center", &details.gradient.end_center);
                            f32_node(&mut v, "end-radius", details.gradient.end_radius);
                            f32_node(&mut v, "ratio-xy", details.gradient.ratio_xy);
                            let mut stops = vec![];
                            for stop in aux.gradient_stops(&details.gradient.stops) {
                                stops.push(Yaml::Real(stop.offset.to_string()));
                                stops.push(Yaml::String(color_to_string(stop.color)));
                            }
                            yaml_node(&mut v, "stops", Yaml::Array(stops));
                            bool_node(&mut v, "repeat", details.gradient.extend_mode == ExtendMode::Repeat);
                            yaml_node(&mut v, "outset", f32_vec_yaml(&outset, true));
                        }
                    }
                },
                BoxShadow(item) => {
                    str_node(&mut v, "type", "box-shadow");
                    rect_node(&mut v, "box-bounds", &item.box_bounds);
                    point_node(&mut v, "offset", &item.offset);
                    color_node(&mut v, "color", item.color);
                    f32_node(&mut v, "blur-radius", item.blur_radius);
                    f32_node(&mut v, "spread-radius", item.spread_radius);
                    f32_node(&mut v, "border-radius", item.border_radius);
                    let clip_mode = match item.clip_mode {
                        BoxShadowClipMode::None => "none",
                        BoxShadowClipMode::Outset => "outset",
                        BoxShadowClipMode::Inset => "inset"
                    };
                    str_node(&mut v, "clip-mode", clip_mode);
                },
                Gradient(item) => {
                    str_node(&mut v, "type", "gradient");
                    point_node(&mut v, "start", &item.gradient.start_point);
                    point_node(&mut v, "end", &item.gradient.end_point);
                    let mut stops = vec![];
                    for stop in aux.gradient_stops(&item.gradient.stops) {
                        stops.push(Yaml::Real(stop.offset.to_string()));
                        stops.push(Yaml::String(color_to_string(stop.color)));
                    }
                    yaml_node(&mut v, "stops", Yaml::Array(stops));
                    bool_node(&mut v, "repeat", item.gradient.extend_mode == ExtendMode::Repeat);
                },
                RadialGradient(item) => {
                    str_node(&mut v, "type", "radial-gradient");
                    point_node(&mut v, "start-center", &item.gradient.start_center);
                    f32_node(&mut v, "start-radius", item.gradient.start_radius);
                    point_node(&mut v, "end-center", &item.gradient.end_center);
                    f32_node(&mut v, "end-radius", item.gradient.end_radius);
                    f32_node(&mut v, "ratio-xy", item.gradient.ratio_xy);
                    let mut stops = vec![];
                    for stop in aux.gradient_stops(&item.gradient.stops) {
                        stops.push(Yaml::Real(stop.offset.to_string()));
                        stops.push(Yaml::String(color_to_string(stop.color)));
                    }
                    yaml_node(&mut v, "stops", Yaml::Array(stops));
                    bool_node(&mut v, "repeat", item.gradient.extend_mode == ExtendMode::Repeat);
                },
                Iframe(item) => {
                    str_node(&mut v, "type", "iframe");
                    u32_vec_node(&mut v, "id", &vec![item.pipeline_id.0, item.pipeline_id.1]);
                },
                PushStackingContext(item) => {
                    str_node(&mut v, "type", "stacking-context");
                    write_sc(&mut v, &item.stacking_context);
                    self.write_display_list(&mut v, list_iterator, aux, clip_id_mapper);
                },
                Clip(item) => {
                    str_node(&mut v, "type", "clip");
                    size_node(&mut v, "content-size", &item.content_size);
                    usize_node(&mut v, "id", clip_id_mapper.add_id(item.id));
                }
                PopStackingContext => return,
            }
            if !v.is_empty() {
                list.push(Yaml::Hash(v));
            }
        }
    }

    fn write_display_list(&mut self,
                          parent: &mut Table,
                          list_iterator: &mut slice::Iter<DisplayItem>,
                          aux: &AuxiliaryLists,
                          clip_id_mapper: &mut ClipIdMapper) {
        let mut list = vec![];
        self.write_display_list_items(&mut list, list_iterator, aux, clip_id_mapper);
        parent.insert(Yaml::String("items".to_owned()), Yaml::Array(list));
    }
}

impl webrender::ApiRecordingReceiver for YamlFrameWriterReceiver {
    fn write_msg(&mut self, _: u32, msg: &ApiMsg) {
        match msg {
            &ApiMsg::SetRootPipeline(ref pipeline_id) => {
                self.scene.set_root_pipeline_id(pipeline_id.clone());
            }
            &ApiMsg::Scroll(..) |
            &ApiMsg::TickScrollingBounce |
            &ApiMsg::WebGLCommand(..) => {
            }

            &ApiMsg::AddRawFont(ref key, ref bytes) => {
                self.frame_writer.fonts.insert(*key, CachedFont::Raw(Some(bytes.clone()), None));
            }

            &ApiMsg::AddNativeFont(ref key, ref native_font_handle) => {
                self.frame_writer.fonts.insert(*key, CachedFont::Native(native_font_handle.clone()));
            }

            &ApiMsg::AddImage(ref key, ref descriptor, ref data, ref tiling) => {
                let stride = descriptor.stride.unwrap_or(
                    descriptor.width * descriptor.format.bytes_per_pixel().unwrap()
                );
                let bytes = match data {
                    &ImageData::Raw(ref v) => { (**v).clone() }
                    &ImageData::External(_) => { return; }
                    &ImageData::Blob(_) => { return; }
                };
                self.frame_writer.images.insert(*key, CachedImage {
                    width: descriptor.width,
                    height: descriptor.height,
                    stride: stride,
                    format: descriptor.format,
                    bytes: Some(bytes),
                    path: None,
                    tiling: *tiling,
                });
            }

            &ApiMsg::UpdateImage(ref key, ref descriptor, ref img_data, _dirty_rect) => {
                if let Some(ref mut data) = self.frame_writer.images.get_mut(key) {
                    assert!(data.width == descriptor.width);
                    assert!(data.height == descriptor.height);
                    assert!(data.format == descriptor.format);

                    if let &ImageData::Raw(ref bytes) = img_data {
                        *data.path.borrow_mut() = None;
                        *data.bytes.borrow_mut() = Some((**bytes).clone());
                    } else {
                        // Other existing image types only make sense within the gecko integration.
                        println!("Wrench only supports updating buffer images (ignoring update command).");
                    }
                }
            }

            &ApiMsg::DeleteImage(ref key) => {
                self.frame_writer.images.remove(key);
            }

            &ApiMsg::SetRootDisplayList(ref background_color,
                                        ref epoch,
                                        ref pipeline_id,
                                        ref viewport_size,
                                        ref display_list,
                                        ref auxiliary_lists,
                                        _preserve_frame_state) => {
                self.frame_writer.begin_write_root_display_list(&mut self.scene,
                                                                background_color,
                                                                epoch,
                                                                pipeline_id,
                                                                viewport_size,
                                                                display_list,
                                                                auxiliary_lists);
            }
            _ => {}
        }
    }

    fn write_payload(&mut self, _frame: u32, data: &[u8]) {
        if self.frame_writer.dl_descriptor.is_some() {
            self.frame_writer.finish_write_root_display_list(&mut self.scene, data);
        }
    }
}

/// This structure allows mapping both Clip and ClipExternalId ScrollLayerIds
/// onto one set of numeric ids. This prevents ids from clashing in the yaml output.
struct ClipIdMapper {
    hash_map: HashMap<ScrollLayerId, usize>,
    current_clip_id: usize,
}

impl ClipIdMapper {
    fn new() -> ClipIdMapper {
        ClipIdMapper {
            hash_map: HashMap::new(),
            current_clip_id: 1,
        }
    }

    fn add_id(&mut self, id: ScrollLayerId) -> usize {
        self.hash_map.insert(id, self.current_clip_id);
        self.current_clip_id += 1;
        self.current_clip_id - 1
    }

    fn map(&self, id: &ScrollLayerId) -> usize {
        *self.hash_map.get(id).unwrap()
    }
}
