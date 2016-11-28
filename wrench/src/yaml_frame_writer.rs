extern crate yaml_rust;

use app_units::Au;
use euclid::{Point2D, Size2D, Rect, Matrix4D};
use image::{ColorType, save_buffer};
use std::borrow::BorrowMut;
use std::cell::Cell;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fs;
use std::fs::File;
use std::io::{Cursor, Read, Write};
use std::slice;
use std::path::{Path, PathBuf};
use webrender;
use webrender_traits::*;
use yaml_rust::{Yaml, YamlEmitter};

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

fn size_node(parent: &mut Table, key: &str, value: &Size2D<f32>) {
    yaml_node(parent, key, Yaml::String(format!("{} {}", value.width, value.height)));
}

fn rect_node(parent: &mut Table, key: &str, value: &Rect<f32>) {
    yaml_node(parent, key, Yaml::String(format!("{} {} {} {}", value.origin.x, value.origin.y,
                                                               value.size.width, value.size.height)));
}

fn matrix4d_node(parent: &mut Table, key: &str, value: &Matrix4D<f32>) {
    yaml_node(parent, key, Yaml::String(format!("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
                                                value.m11, value.m12, value.m13, value.m14,
                                                value.m21, value.m22, value.m23, value.m24,
                                                value.m31, value.m32, value.m33, value.m34,
                                                value.m41, value.m42, value.m43, value.m44)));
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
    yaml_node(parent, key,
                  f32_vec_yaml(value, false));
}

fn clip_node(parent: &mut Table, key: &str, value: &ClipRegion) {
    // pass for now
}

fn write_sc(parent: &mut Table, sc: &StackingContext) {
    // scroll_policy
    rect_node(parent, "bounds", &sc.bounds);
    rect_node(parent, "overflow", &sc.overflow);
    i32_node(parent, "z_index", sc.z_index);
    if sc.transform != Matrix4D::<f32>::identity() {
        matrix4d_node(parent, "transform", &sc.transform);
    }
    if sc.perspective != Matrix4D::<f32>::identity() {
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
    images: HashMap<ImageKey, CachedImage>,
    fonts: HashMap<FontKey, CachedFont>,

    last_frame_written: u32,

    dl_descriptor: Option<BuiltDisplayListDescriptor>,
    aux_descriptor: Option<AuxiliaryListsDescriptor>,
}

impl YamlFrameWriter {
    pub fn new(path: &Path) -> YamlFrameWriter {
        let mut rsrc_base = path.to_owned();
        rsrc_base.push("res");
        fs::create_dir_all(&rsrc_base).ok();

        YamlFrameWriter {
            frame_base: path.to_owned(),
            rsrc_base: rsrc_base,
            next_rsrc_num: 1,
            images: HashMap::new(),
            fonts: HashMap::new(),

            dl_descriptor: None,
            aux_descriptor: None,

            last_frame_written: u32::max_value(),
        }
    }

    pub fn begin_write_root_display_list(&mut self,
                                         _: &ColorF,
                                         _: &Epoch,
                                         _: &PipelineId,
                                         _: &Size2D<f32>,
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
    }

    pub fn finish_write_root_display_list(&mut self,
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
        let mut iter = dl.all_display_items().iter();
        self.write_dl(&mut root_dl_table, &mut iter, &aux);

        let mut root = new_table();
        table_node(&mut root, "root", root_dl_table);

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

    fn next_rsrc_paths(counter: &mut u32, base_path: &Path, base: &str, ext: &str) -> (PathBuf, PathBuf) {
        let mut path_file = base_path.to_owned();
        let mut path = PathBuf::from("res");

        let fstr = format!("{}-{}.{}", base, counter, ext);
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
        let (path_file, path) = Self::next_rsrc_paths(&mut self.next_rsrc_num, &self.rsrc_base, "img", "png");

        let ok = match data.format {
            ImageFormat::RGB8 => {
                if data.stride == data.width * 3 {
                    save_buffer(&path_file, &bytes, data.width, data.height, ColorType::RGB(8)).unwrap();
                    true
                } else {
                    false
                }
            }
            ImageFormat::RGBA8 => {
                if data.stride == data.width * 4 {
                    save_buffer(&path_file, &bytes, data.width, data.height, ColorType::RGBA(8)).unwrap();
                    true
                } else {
                    false
                }
            }
            ImageFormat::A8 => {
                if data.stride == data.width {
                    save_buffer(&path_file, &bytes, data.width, data.height, ColorType::Gray(8)).unwrap();
                    true
                } else {
                    false
                }
            }
            _ => { false }
        };

        if !ok {
            println!("Failed to write image with format {:?}, dimensions {}x{}, stride {}",
                     data.format, data.width, data.height, data.stride);
            return None;
        }

        data.path = Some(path.clone());
        // put it back
        self.images.insert(*key, data);
        Some(path)
    }

    fn write_dl_items(&mut self, list: &mut Vec<Yaml>, dl_iter: &mut slice::Iter<DisplayItem>, aux: &AuxiliaryLists) {
        use webrender_traits::SpecificDisplayItem::*;
        while let Some(ref base) = dl_iter.next() {
            let mut v = new_table();
            match base.item {
                Rectangle(item) => {
                    str_node(&mut v, "type", "rect");
                    color_node(&mut v, "color", item.color);
                    rect_node(&mut v, "bounds", &base.rect);
                    clip_node(&mut v, "clip", &base.clip);
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
                    rect_node(&mut v, "bounds", &base.rect);
                    clip_node(&mut v, "clip", &base.clip);

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
                                let (path_file, path) = Self::next_rsrc_paths(&mut self.next_rsrc_num, &self.rsrc_base, "font", "ttf");
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
                    rect_node(&mut v, "bounds", &base.rect);
                    clip_node(&mut v, "clip", &base.clip);
                    size_node(&mut v, "strech", &item.stretch_size);
                    size_node(&mut v, "spacing", &item.tile_spacing);
                    match item.image_rendering {
                        ImageRendering::Auto => (),
                        ImageRendering::CrispEdges => str_node(&mut v, "rendering", "crisp-edges"),
                        ImageRendering::Pixelated => str_node(&mut v, "rendering", "pixelated"),
                    };
                },
                YuvImage(_) => {
                    // TODO
                    println!("TODO YAML YuvImage");
                },
                WebGL(_) => {
                    // TODO
                    println!("TODO YAML WebGL");
                    //rect_node(&mut v, "bounds", &base.rect);
                    //clip_node(&mut v, "clip", &base.clip);
                },
                Border(item) => {
                    rect_node(&mut v, "bounds", &base.rect);
                    clip_node(&mut v, "clip", &base.clip);
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
                },
                BoxShadow(item) => {
                    // TODO
                    println!("TODO YAML BoxShadow");
                    rect_node(&mut v, "bounds", &base.rect);
                    clip_node(&mut v, "clip", &base.clip);
                },
                Gradient(item) => {
                    // TODO
                    println!("TODO YAML Gradient");
                    rect_node(&mut v, "bounds", &base.rect);
                    clip_node(&mut v, "clip", &base.clip);
                },
                Iframe(item) => {
                    // TODO
                    println!("TODO YAML Iframe");
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
                    println!("TODO PushScrollLayer");
                    rect_node(&mut v, "bounds", &base.rect);
                    clip_node(&mut v, "clip", &base.clip);
                },
                PopScrollLayer => {
                    println!("TODO PopScrollLayer");
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

impl webrender::ApiRecordingReceiver for YamlFrameWriter {
    fn write_msg(&mut self, _: u32, msg: &ApiMsg) {
        match msg {
            &ApiMsg::SetRootPipeline(..) |
            &ApiMsg::Scroll(..) |
            &ApiMsg::TickScrollingBounce |
            &ApiMsg::WebGLCommand(..) => {
            }

            &ApiMsg::AddRawFont(ref key, ref bytes) => {
                self.fonts.insert(*key, CachedFont::Raw(Some(bytes.clone()), None));
            }

            &ApiMsg::AddNativeFont(ref key, ref native_font_handle) => {
                self.fonts.insert(*key, CachedFont::Native(native_font_handle.clone()));
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
                self.images.insert(*key, CachedImage {
                    width: width, height: height, stride: stride,
                    format: format,
                    bytes: Some(bytes),
                    path: None,
                });
            }

            &ApiMsg::UpdateImage(ref key, width, height,
                                 format, ref bytes) => {
                if let Some(ref mut data) = self.images.get_mut(key) {
                    assert!(data.width == width);
                    assert!(data.height == height);
                    assert!(data.format == format);

                    *data.path.borrow_mut() = None;
                    *data.bytes.borrow_mut() = Some(bytes.clone());
                }
            }

            &ApiMsg::DeleteImage(ref key) => {
                self.images.remove(key);
            }

            &ApiMsg::SetRootDisplayList(ref background_color,
                                        ref epoch,
                                        ref pipeline_id,
                                        ref viewport_size,
                                        ref display_list,
                                        ref auxiliary_lists) => {
                self.begin_write_root_display_list(background_color, epoch, pipeline_id,
                                                   viewport_size, display_list, auxiliary_lists);
            }
            _ => {}
        }
    }

    fn write_payload(&mut self, frame: u32, data: &[u8]) {
        if self.dl_descriptor.is_some() {
            self.finish_write_root_display_list(frame, data);
        }
    }
}
