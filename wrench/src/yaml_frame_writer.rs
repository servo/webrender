extern crate yaml_rust;

use app_units::Au;
use euclid::{TypedPoint2D, TypedSize2D, TypedRect, TypedMatrix4D};
use image::{ColorType, save_buffer};
use std::borrow::BorrowMut;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{Cursor, Read, Write};
use std::slice;
use std::path::{Path, PathBuf};
use webrender;
use webrender_traits::*;
use yaml_rust::{Yaml, YamlEmitter};
use scene::Scene;
use time;

use super::CURRENT_FRAME_NUMBER;

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

fn matrix4d_node<U1,U2>(parent: &mut Table, key: &str, value: &TypedMatrix4D<f32, U1, U2>) {
    f32_vec_node(parent, key, &value.to_row_major_array());
}

fn u32_node(parent: &mut Table, key: &str, value: u32) {
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

fn string_vec_node(parent: &mut Table, key: &str, value: &[String]) {
    yaml_node(parent, key, string_vec_yaml(value, false));
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

fn maybe_radius_yaml(radius: &BorderRadius) -> Option<Yaml> {
    if let Some(radius) = radius.is_uniform() {
        if radius == 0.0 {
            None
        } else {
            Some(Yaml::Real(radius.to_string()))
        }
    } else {
        let mut table = new_table();
        size_node(&mut table, "top_left", &radius.top_left);
        size_node(&mut table, "top_right", &radius.top_right);
        size_node(&mut table, "bottom_left", &radius.bottom_left);
        size_node(&mut table, "bottom_right", &radius.bottom_right);
        Some(Yaml::Hash(table))
    }
}

fn write_sc(parent: &mut Table, sc: &StackingContext) {
    // overwrite "bounds" with the proper one
    rect_node(parent, "bounds", &sc.bounds);
    // scroll_policy
    i32_node(parent, "z_index", sc.z_index);
    if sc.transform != LayoutTransform::identity() {
        matrix4d_node(parent, "transform", &sc.transform);
    }
    if sc.perspective != LayoutTransform::identity() {
        matrix4d_node(parent, "perspective", &sc.perspective);
    }
    // mix_blend_mode
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
fn native_font_handle_to_yaml(native_handle: &NativeFontHandle, parent: &mut yaml_rust::yaml::Hash) {
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

    pub fn finish_write_root_display_list(&mut self,
                                          scene: &mut Scene,
                                          frame: u32,
                                          data: &[u8])
    {
        let dl_desc = self.dl_descriptor.take().unwrap();
        let aux_desc = self.aux_descriptor.take().unwrap();

        let mut auxiliary_data = Cursor::new(&data[4..]);

        let mut built_display_list_data = vec![0; dl_desc.size()];
        let mut aux_list_data = vec![0; aux_desc.size()];

        auxiliary_data.read_exact(&mut built_display_list_data[..]).unwrap();
        auxiliary_data.read_exact(&mut aux_list_data[..]).unwrap();

        let dl = BuiltDisplayList::from_data(built_display_list_data, dl_desc);
        let aux = AuxiliaryLists::from_data(aux_list_data, aux_desc);

        let mut root_dl_table = new_table();
        {
            let mut iter = dl.all_display_items().iter();
            self.write_dl(&mut root_dl_table, &mut iter, &aux);
        }

        scene.finish_root_display_list(self.pipeline_id.unwrap(), dl, aux);

        let mut root = new_table();
        if let Some(root_pipeline_id) = scene.root_pipeline_id {
            u32_vec_node(&mut root_dl_table, "id", &vec![root_pipeline_id.0, root_pipeline_id.1]);

            let mut pipelines = vec![];
            for pipeline_id in scene.pipeline_map.keys() {
                // write out all pipelines other than the root one
                if *pipeline_id == root_pipeline_id {
                    continue;
                }

                let mut pipeline = new_table();
                u32_vec_node(&mut pipeline, "id", &vec![pipeline_id.0, pipeline_id.1]);

                let dl = scene.display_lists.get(pipeline_id).unwrap();
                let aux = scene.pipeline_auxiliary_lists.get(pipeline_id).unwrap();
                let mut iter = dl.iter();
                self.write_dl(&mut pipeline, &mut iter, &aux);
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
        let bytes = data.bytes.take().unwrap();
        let (path_file, path) = Self::next_rsrc_paths(&self.rsrc_prefix, &mut self.next_rsrc_num, &self.rsrc_base, "img", "png");

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
            save_buffer(&path_file, &bytes, data.width, data.height, ColorType::RGB(8)).unwrap();
        } else {
            // takes a buffer with a stride and copies it into a new buffer that has stride == width
            assert!(data.stride > data.width * bpp);
            let tmp: Vec<_>  = bytes[..].chunks(data.stride as usize)
                                        .flat_map(|chunk| chunk[..(data.width * bpp) as usize].iter().cloned())
                                        .collect();

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
                let mut complex_items = complex.iter().map(|ccx|
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

                table_node(&mut complex_table, "image_mask", mask_table);
            }

            Yaml::Hash(complex_table)
        } else {
            rect_yaml(&clip.main)
        }
    }

    fn write_dl_items(&mut self, list: &mut Vec<Yaml>, dl_iter: &mut slice::Iter<DisplayItem>, aux: &AuxiliaryLists) {
        use webrender_traits::SpecificDisplayItem::*;
        while let Some(ref base) = dl_iter.next() {
            let mut v = new_table();
            rect_node(&mut v, "bounds", &base.rect);
            yaml_node(&mut v, "clip", self.make_clip_node(&base.clip, aux));

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
                        offsets.push(g.x); offsets.push(g.y);
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
                                let (path_file, path) = Self::next_rsrc_paths(&self.rsrc_prefix, &mut self.next_rsrc_num, &self.rsrc_base, "font", "ttf");
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
                    size_node(&mut v, "strech", &item.stretch_size);
                    size_node(&mut v, "spacing", &item.tile_spacing);
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
                    let trbl = vec![&item.top, &item.right, &item.bottom, &item.left];
                    let widths: Vec<f32> = trbl.iter().map(|x| x.width).collect();
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
                    yaml_node(&mut v, "color", string_vec_yaml(&colors, true));
                    yaml_node(&mut v, "style", string_vec_yaml(&styles, true));
                    if let Some(radius_node) = maybe_radius_yaml(&item.radius) {
                        yaml_node(&mut v, "radius", radius_node);
                    }
                },
                BoxShadow(item) => {
                    str_node(&mut v, "type", "box-shadow");
                    rect_node(&mut v, "box_bounds", &item.box_bounds);
                    point_node(&mut v, "offset", &item.offset);
                    color_node(&mut v, "color", item.color);
                    f32_node(&mut v, "blur_radius", item.blur_radius);
                    f32_node(&mut v, "spread_radius", item.spread_radius);
                    f32_node(&mut v, "border_radius", item.border_radius);
                    let clip_mode = match item.clip_mode {
                        BoxShadowClipMode::None => "none",
                        BoxShadowClipMode::Outset => "outset",
                        BoxShadowClipMode::Inset => "inset"
                    };
                    str_node(&mut v, "clip_mode", clip_mode);
                },
                Gradient(item) => {
                    str_node(&mut v, "type", "gradient");
                    point_node(&mut v, "start", &item.start_point);
                    point_node(&mut v, "end", &item.end_point);
                    let mut stops = vec![];
                    for stop in aux.gradient_stops(&item.stops) {
                        stops.push(Yaml::Real(stop.offset.to_string()));
                        stops.push(Yaml::String(color_to_string(stop.color)));
                    }
                    yaml_node(&mut v, "stops", Yaml::Array(stops));
                },
                RadialGradient(item) => {
                    str_node(&mut v, "type", "radial_gradient");
                    point_node(&mut v, "start_center", &item.start_center);
                    f32_node(&mut v, "start_radius", item.start_radius);
                    point_node(&mut v, "end_center", &item.end_center);
                    f32_node(&mut v, "end_radius", item.end_radius);
                    let mut stops = vec![];
                    for stop in aux.gradient_stops(&item.stops) {
                        stops.push(Yaml::Real(stop.offset.to_string()));
                        stops.push(Yaml::String(color_to_string(stop.color)));
                    }
                    yaml_node(&mut v, "stops", Yaml::Array(stops));
                },
                Iframe(item) => {
                    str_node(&mut v, "type", "iframe");
                    u32_vec_node(&mut v, "id", &vec![item.pipeline_id.0, item.pipeline_id.1]);
                },
                PushStackingContext(item) => {
                    str_node(&mut v, "type", "stacking_context");
                    write_sc(&mut v, &item.stacking_context);
                    self.write_dl(&mut v, dl_iter, aux);
                },
                PopStackingContext => {
                    return;
                },
                PushScrollLayer(item) => {
                    // TODO
                    //println!("TODO PushScrollLayer");
                },
                PopScrollLayer => {
                    //println!("TODO PopScrollLayer");
                    // TODO
                },
            }
            if !v.is_empty() {
                list.push(Yaml::Hash(v));
            }
        }
    }

    fn write_dl(&mut self, parent: &mut Table, dl_iter: &mut slice::Iter<DisplayItem>, aux: &AuxiliaryLists) {
        let mut list = vec![];
        self.write_dl_items(&mut list, dl_iter, aux);
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

            &ApiMsg::AddImage(ref key, width, height, stride,
                              format, ref data) => {
                let stride = if let Some(stride) = stride {
                    stride
                } else {
                    match format {
                        ImageFormat::A8 => width,
                        ImageFormat::RGBA8 | ImageFormat::RGB8 => width*4,
                        ImageFormat::RGBAF32 => width*16,
                        _ => panic!("Invalid image format"),
                    }
                };
                let bytes = match data {
                    &ImageData::Raw(ref v) => { (**v).clone() }
                    &ImageData::External(_) => { return; }
                };
                self.frame_writer.images.insert(*key, CachedImage {
                    width: width, height: height, stride: stride,
                    format: format,
                    bytes: Some(bytes),
                    path: None,
                });
            }

            &ApiMsg::UpdateImage(ref key, width, height,
                                 format, ref bytes) => {
                if let Some(ref mut data) = self.frame_writer.images.get_mut(key) {
                    assert!(data.width == width);
                    assert!(data.height == height);
                    assert!(data.format == format);

                    *data.path.borrow_mut() = None;
                    *data.bytes.borrow_mut() = Some(bytes.clone());
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
                                        ref auxiliary_lists) => {
                self.frame_writer.begin_write_root_display_list(&mut self.scene, background_color, epoch, pipeline_id,
                                                   viewport_size, display_list, auxiliary_lists);
            }
            _ => {}
        }
    }

    fn write_payload(&mut self, frame: u32, data: &[u8]) {
        if self.frame_writer.dl_descriptor.is_some() {
            self.frame_writer.finish_write_root_display_list(&mut self.scene, frame, data);
        }
    }
}
