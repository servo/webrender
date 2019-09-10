/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate yaml_rust;

use euclid::{Point2D, Rect, Size2D, Transform3D, Vector2D};
use image::{save_buffer, ColorType};
use crate::premultiply::unpremultiply;
use crate::scene::{Scene, SceneProperties};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::{fmt, fs};
use super::CURRENT_FRAME_NUMBER;
use time;
use webrender;
use webrender::api::*;
use webrender::api::channel::Payload;
use webrender::api::units::*;
use crate::yaml_helper::StringEnum;
use yaml_rust::{Yaml, YamlEmitter};

type Table = yaml_rust::yaml::Hash;

fn array_elements_are_same<T: PartialEq>(v: &[T]) -> bool {
    if !v.is_empty() {
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

fn enum_node<E: StringEnum>(parent: &mut Table, key: &str, value: E) {
    yaml_node(parent, key, Yaml::String(value.as_str().to_owned()));
}

fn color_to_string(value: ColorF) -> String {
    if value.r == 1.0 && value.g == 1.0 && value.b == 1.0 && value.a == 1.0 {
        "white".to_owned()
    } else if value.r == 0.0 && value.g == 0.0 && value.b == 0.0 && value.a == 1.0 {
        "black".to_owned()
    } else {
        format!(
            "{} {} {} {:.4}",
            value.r * 255.0,
            value.g * 255.0,
            value.b * 255.0,
            value.a
        )
    }
}

fn filter_input_to_string(input: FilterPrimitiveInput) -> String {
    match input {
        FilterPrimitiveInput::Original => "original".into(),
        FilterPrimitiveInput::Previous => "previous".into(),
        FilterPrimitiveInput::OutputOfPrimitiveIndex(index) => index.to_string(),
    }
}

fn color_node(parent: &mut Table, key: &str, value: ColorF) {
    yaml_node(parent, key, Yaml::String(color_to_string(value)));
}

fn point_node<U>(parent: &mut Table, key: &str, value: &Point2D<f32, U>) {
    f32_vec_node(parent, key, &[value.x, value.y]);
}

fn vector_node<U>(parent: &mut Table, key: &str, value: &Vector2D<f32, U>) {
    f32_vec_node(parent, key, &[value.x, value.y]);
}

fn size_node<U>(parent: &mut Table, key: &str, value: &Size2D<f32, U>) {
    f32_vec_node(parent, key, &[value.width, value.height]);
}

fn rect_yaml<U>(value: &Rect<f32, U>) -> Yaml {
    f32_vec_yaml(
        &[
            value.origin.x,
            value.origin.y,
            value.size.width,
            value.size.height,
        ],
        false,
    )
}

fn rect_node<U>(parent: &mut Table, key: &str, value: &Rect<f32, U>) {
    yaml_node(parent, key, rect_yaml(value));
}

fn matrix4d_node<U1, U2>(parent: &mut Table, key: &str, value: &Transform3D<f32, U1, U2>) {
    f32_vec_node(parent, key, &value.to_row_major_array());
}

fn u32_node(parent: &mut Table, key: &str, value: u32) {
    yaml_node(parent, key, Yaml::Integer(value as i64));
}

fn i32_node(parent: &mut Table, key: &str, value: i32) {
    yaml_node(parent, key, Yaml::Integer(value as i64));
}

fn usize_node(parent: &mut Table, key: &str, value: usize) {
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

fn filter_input_node(parent: &mut Table, key: &str, value: FilterPrimitiveInput) {
    yaml_node(parent, key, Yaml::String(filter_input_to_string(value)));
}

fn string_vec_yaml(value: &[String], check_unique: bool) -> Yaml {
    if !value.is_empty() && check_unique && array_elements_are_same(value) {
        Yaml::String(value[0].clone())
    } else {
        Yaml::Array(value.iter().map(|v| Yaml::String(v.clone())).collect())
    }
}

fn u32_vec_yaml(value: &[u32], check_unique: bool) -> Yaml {
    if !value.is_empty() && check_unique && array_elements_are_same(value) {
        Yaml::Integer(value[0] as i64)
    } else {
        Yaml::Array(value.iter().map(|v| Yaml::Integer(*v as i64)).collect())
    }
}

fn i32_vec_yaml(value: &[i32], check_unique: bool) -> Yaml {
    if !value.is_empty() && check_unique && array_elements_are_same(value) {
        Yaml::Integer(value[0] as i64)
    } else {
        Yaml::Array(value.iter().map(|v| Yaml::Integer(*v as i64)).collect())
    }
}

fn u32_vec_node(parent: &mut Table, key: &str, value: &[u32]) {
    yaml_node(parent, key, u32_vec_yaml(value, false));
}

fn f32_vec_yaml(value: &[f32], check_unique: bool) -> Yaml {
    if !value.is_empty() && check_unique && array_elements_are_same(value) {
        Yaml::Real(value[0].to_string())
    } else {
        Yaml::Array(value.iter().map(|v| Yaml::Real(v.to_string())).collect())
    }
}

fn f32_vec_node(parent: &mut Table, key: &str, value: &[f32]) {
    yaml_node(parent, key, f32_vec_yaml(value, false));
}

fn maybe_radius_yaml(radius: &BorderRadius) -> Option<Yaml> {
    if let Some(radius) = radius.is_uniform_size() {
        if radius == LayoutSize::zero() {
            None
        } else {
            Some(f32_vec_yaml(&[radius.width, radius.height], false))
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

fn common_node(v: &mut Table, clip_id_mapper: &mut ClipIdMapper, info: &CommonItemProperties) {
    rect_node(v, "clip-rect", &info.clip_rect);
    bool_node(v, "backface-visible", info.is_backface_visible);

    clip_and_scroll_node(v, clip_id_mapper, info.clip_id, info.spatial_id);

    if let Some(tag) = info.hit_info {
        yaml_node(
            v,
            "hit-testing-tag",
             Yaml::Array(vec![Yaml::Integer(tag.0 as i64), Yaml::Integer(tag.1 as i64)])
        );
    }
}

fn clip_and_scroll_node(
    v: &mut Table,
    clip_id_mapper: &mut ClipIdMapper,
    clip_id: ClipId,
    spatial_id: SpatialId
) {
    let clip_id = if clip_id.is_root() { None } else { Some(clip_id) };
    yaml_node(v, "clip-and-scroll",
        clip_id_mapper.map_clip_and_scroll_ids(clip_id, spatial_id)
    );
}

fn write_reference_frame(
    parent: &mut Table,
    reference_frame: &ReferenceFrame,
    properties: &SceneProperties,
    clip_id_mapper: &mut ClipIdMapper,
) {
    // FIXME: This ignores the scrolling_relative_to member in
    // ReferenceFrameKind::Perspective, but it's a bit annoying to fix since the
    // frame reader abuses `ExternalScrollId`s.

    matrix4d_node(
        parent,
        match reference_frame.kind {
            ReferenceFrameKind::Transform => "transform",
            ReferenceFrameKind::Perspective { .. } => "perspective",
        },
        &properties.resolve_layout_transform(&reference_frame.transform)
    );

    usize_node(parent, "id", clip_id_mapper.add_spatial_id(reference_frame.id));
}

fn shadow_parameters(shadow: &Shadow) -> String {
    format!(
        "[{},{}],{},[{}]",
        shadow.offset.x, shadow.offset.y,
        shadow.blur_radius,
        color_to_string(shadow.color)
    )
}

fn write_filters(
    parent: &mut Table,
    name: &str,
    filter_iter: impl IntoIterator<Item = FilterOp>,
    properties: &SceneProperties,
) {
    let mut filters = vec![];
    for filter in filter_iter {
        match filter {
            FilterOp::Identity => { filters.push(Yaml::String("identity".into())) }
            FilterOp::Blur(x) => { filters.push(Yaml::String(format!("blur({})", x))) }
            FilterOp::Brightness(x) => { filters.push(Yaml::String(format!("brightness({})", x))) }
            FilterOp::Contrast(x) => { filters.push(Yaml::String(format!("contrast({})", x))) }
            FilterOp::Grayscale(x) => { filters.push(Yaml::String(format!("grayscale({})", x))) }
            FilterOp::HueRotate(x) => { filters.push(Yaml::String(format!("hue-rotate({})", x))) }
            FilterOp::Invert(x) => { filters.push(Yaml::String(format!("invert({})", x))) }
            FilterOp::Opacity(x, _) => {
                filters.push(Yaml::String(format!("opacity({})",
                                                  properties.resolve_float(&x))))
            }
            FilterOp::Saturate(x) => { filters.push(Yaml::String(format!("saturate({})", x))) }
            FilterOp::Sepia(x) => { filters.push(Yaml::String(format!("sepia({})", x))) }
            FilterOp::DropShadow(shadow) => {
                filters.push(Yaml::String(format!(
                    "drop-shadow({})",
                    shadow_parameters(&shadow)
                )))
            }
            FilterOp::ColorMatrix(matrix) => {
                filters.push(Yaml::String(format!("color-matrix({:?})", matrix)))
            }
            FilterOp::SrgbToLinear => {
                filters.push(Yaml::String("srgb-to-linear".to_string()))
            }
            FilterOp::LinearToSrgb => {
                filters.push(Yaml::String("linear-to-srgb".to_string()))
            }
            FilterOp::ComponentTransfer => {
                filters.push(Yaml::String("component-transfer".to_string()))
            }
            FilterOp::Flood(color) => {
                filters.push(Yaml::String(format!("flood({})", color_to_string(color))))
            }
        }
    }

    yaml_node(parent, name, Yaml::Array(filters));
}

fn write_filter_datas(
    parent: &mut Table,
    name: &str,
    filter_data_iter: &[TempFilterData],
) {
    let mut filter_datas = vec![];
    for filter_data in filter_data_iter {
        let func_types = filter_data.func_types.iter().map(|func_type| {
            match func_type {
                ComponentTransferFuncType::Identity => { Yaml::String("Identity".to_string()) }
                ComponentTransferFuncType::Table => { Yaml::String("Table".to_string()) }
                ComponentTransferFuncType::Discrete => { Yaml::String("Discrete".to_string()) }
                ComponentTransferFuncType::Linear => { Yaml::String("Linear".to_string()) }
                ComponentTransferFuncType::Gamma => { Yaml::String("Gamma".to_string()) }
            }
        }).collect();
        let r_values = filter_data.r_values.iter().map(|value| {
            Yaml::String(format!("{}", value))
        }).collect();
        let g_values = filter_data.g_values.iter().map(|value| {
            Yaml::String(format!("{}", value))
        }).collect();
        let b_values = filter_data.b_values.iter().map(|value| {
            Yaml::String(format!("{}", value))
        }).collect();
        let a_values = filter_data.a_values.iter().map(|value| {
            Yaml::String(format!("{}", value))
        }).collect();

        let avec: Vec<Yaml> = [
            Yaml::Array(func_types),
            Yaml::Array(r_values),
            Yaml::Array(g_values),
            Yaml::Array(b_values),
            Yaml::Array(a_values),
        ].to_vec();
        filter_datas.push(Yaml::Array(avec));
    }

    yaml_node(parent, name, Yaml::Array(filter_datas));
}

fn write_filter_primitives(
    parent: &mut Table,
    name: &str,
    filter_primitive_iter: impl IntoIterator<Item = FilterPrimitive>,
) {
    let mut filter_primitives = vec![];
    for filter_primitive in filter_primitive_iter {
        let mut table = new_table();
        match filter_primitive.kind {
            FilterPrimitiveKind::Identity(identity_primitive) => {
                yaml_node(&mut table, "type", Yaml::String("identity".into()));
                filter_input_node(&mut table, "in", identity_primitive.input);
            }
            FilterPrimitiveKind::Blend(blend_primitive) => {
                yaml_node(&mut table, "type", Yaml::String("blend".into()));
                filter_input_node(&mut table, "in1", blend_primitive.input1);
                filter_input_node(&mut table, "in2", blend_primitive.input2);
                enum_node(&mut table, "mode", blend_primitive.mode);
            }
            FilterPrimitiveKind::Flood(flood_primitive) => {
                yaml_node(&mut table, "type", Yaml::String("flood".into()));
                color_node(&mut table, "color", flood_primitive.color);
            }
            FilterPrimitiveKind::Blur(blur_primitive) => {
                yaml_node(&mut table, "type", Yaml::String("blur".into()));
                filter_input_node(&mut table, "in", blur_primitive.input);
                f32_node(&mut table, "radius", blur_primitive.radius);
            }
            FilterPrimitiveKind::Opacity(opacity_primitive) => {
                yaml_node(&mut table, "type", Yaml::String("opacity".into()));
                filter_input_node(&mut table, "in", opacity_primitive.input);
                f32_node(&mut table, "opacity", opacity_primitive.opacity);
            }
            FilterPrimitiveKind::ColorMatrix(color_matrix_primitive) => {
                yaml_node(&mut table, "type", Yaml::String("color-matrix".into()));
                filter_input_node(&mut table, "in", color_matrix_primitive.input);
                f32_vec_node(&mut table, "matrix", &color_matrix_primitive.matrix);
            }
            FilterPrimitiveKind::DropShadow(drop_shadow_primitive) => {
                yaml_node(&mut table, "type", Yaml::String("drop-shadow".into()));
                filter_input_node(&mut table, "in", drop_shadow_primitive.input);
                vector_node(&mut table, "offset", &drop_shadow_primitive.shadow.offset);
                color_node(&mut table, "color", drop_shadow_primitive.shadow.color);
                f32_node(&mut table, "radius", drop_shadow_primitive.shadow.blur_radius);
            }
            FilterPrimitiveKind::ComponentTransfer(component_transfer_primitive) => {
                yaml_node(&mut table, "type", Yaml::String("component-transfer".into()));
                filter_input_node(&mut table, "in", component_transfer_primitive.input);
            }
            FilterPrimitiveKind::Offset(info) => {
                yaml_node(&mut table, "type", Yaml::String("offset".into()));
                filter_input_node(&mut table, "in", info.input);
                vector_node(&mut table, "offset", &info.offset);
            }
            FilterPrimitiveKind::Composite(info) => {
                yaml_node(&mut table, "type", Yaml::String("composite".into()));
                filter_input_node(&mut table, "in1", info.input1);
                filter_input_node(&mut table, "in2", info.input2);

                let operator = match info.operator {
                    CompositeOperator::Over => "over",
                    CompositeOperator::In => "in",
                    CompositeOperator::Out => "out",
                    CompositeOperator::Atop => "atop",
                    CompositeOperator::Xor => "xor",
                    CompositeOperator::Lighter => "lighter",
                    CompositeOperator::Arithmetic(..) => "arithmetic",
                };
                str_node(&mut table, "operator", operator);

                if let CompositeOperator::Arithmetic(k_vals) = info.operator {
                    f32_vec_node(&mut table, "k-values", &k_vals);
                }
            }
        }
        enum_node(&mut table, "color-space", filter_primitive.color_space);
        filter_primitives.push(Yaml::Hash(table));
    }

    yaml_node(parent, name, Yaml::Array(filter_primitives));
}

fn write_stacking_context(
    parent: &mut Table,
    sc: &StackingContext,
    properties: &SceneProperties,
    filter_iter: impl IntoIterator<Item = FilterOp>,
    filter_data_iter: &[TempFilterData],
    filter_primitive_iter: impl IntoIterator<Item = FilterPrimitive>,
) {
    enum_node(parent, "transform-style", sc.transform_style);

    let raster_space = match sc.raster_space {
        RasterSpace::Local(scale) => {
            format!("local({})", scale)
        }
        RasterSpace::Screen => {
            "screen".to_owned()
        }
    };
    str_node(parent, "raster-space", &raster_space);

    // mix_blend_mode
    if sc.mix_blend_mode != MixBlendMode::Normal {
        enum_node(parent, "mix-blend-mode", sc.mix_blend_mode)
    }

    write_filters(parent, "filters", filter_iter, properties);
    write_filter_datas(parent, "filter-datas", filter_data_iter);
    write_filter_primitives(parent, "filter-primitives", filter_primitive_iter);
}

#[cfg(target_os = "macos")]
fn native_font_handle_to_yaml(
    rsrc: &mut ResourceGenerator,
    handle: &NativeFontHandle,
    parent: &mut yaml_rust::yaml::Hash,
    path_opt: &mut Option<PathBuf>,
) {
    let path = match *path_opt {
        Some(ref path) => { path.clone() },
        None => {
            use crate::cgfont_to_data;
            let bytes = cgfont_to_data::font_to_data(handle.0.clone()).unwrap();
            let (path_file, path) = rsrc.next_rsrc_paths(
                "font",
                "ttf",
            );
            let mut file = fs::File::create(&path_file).unwrap();
            file.write_all(&bytes).unwrap();
            *path_opt = Some(path.clone());
            path
        }
    };

    path_node(parent, "font", &path);
}

#[cfg(not(target_os = "macos"))]
fn native_font_handle_to_yaml(
    _rsrc: &mut ResourceGenerator,
    handle: &NativeFontHandle,
    parent: &mut yaml_rust::yaml::Hash,
    _: &mut Option<PathBuf>,
) {
    str_node(parent, "font", handle.path.as_os_str().to_str().unwrap());
    if handle.index != 0 {
        u32_node(parent, "font-index", handle.index);
    }
}

fn radial_gradient_to_yaml(
    table: &mut Table,
    gradient: &webrender::api::RadialGradient,
    stops_range: ItemRange<GradientStop>,
) {
    point_node(table, "center", &gradient.center);
    size_node(table, "radius", &gradient.radius);

    let first_offset = gradient.start_offset;
    let last_offset = gradient.end_offset;
    let stops_delta = last_offset - first_offset;
    assert!(first_offset <= last_offset);

    let mut denormalized_stops = vec![];
    for stop in stops_range {
        let denormalized_stop = (stop.offset * stops_delta) + first_offset;
        denormalized_stops.push(Yaml::Real(denormalized_stop.to_string()));
        denormalized_stops.push(Yaml::String(color_to_string(stop.color)));
    }
    yaml_node(table, "stops", Yaml::Array(denormalized_stops));
    bool_node(table, "repeat", gradient.extend_mode == ExtendMode::Repeat);
}

enum CachedFont {
    Native(NativeFontHandle, Option<PathBuf>),
    Raw(Option<Vec<u8>>, u32, Option<PathBuf>),
}

struct CachedFontInstance {
    font_key: FontKey,
    glyph_size: Au,
}

struct CachedImage {
    width: i32,
    height: i32,
    stride: i32,
    format: ImageFormat,
    bytes: Option<Vec<u8>>,
    path: Option<PathBuf>,
    tiling: Option<u16>,
}

struct ResourceGenerator {
    base: PathBuf,
    next_num: u32,
    prefix: String,
}

impl ResourceGenerator {
    fn next_rsrc_paths(&mut self, base: &str, ext: &str) -> (PathBuf, PathBuf) {
        let mut path_file = self.base.to_owned();
        let mut path = PathBuf::from("res");

        let fstr = format!("{}-{}-{}.{}", self.prefix, base, self.next_num, ext);
        path_file.push(&fstr);
        path.push(&fstr);

        self.next_num += 1;

        (path_file, path)
    }
}

pub struct YamlFrameWriter {
    frame_base: PathBuf,
    rsrc_gen: ResourceGenerator,
    images: HashMap<ImageKey, CachedImage>,
    fonts: HashMap<FontKey, CachedFont>,
    font_instances: HashMap<FontInstanceKey, CachedFontInstance>,

    last_frame_written: u32,
    pipeline_id: Option<PipelineId>,

    dl_descriptor: Option<BuiltDisplayListDescriptor>,
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

    fn update_document(&mut self, txn: &TransactionMsg) {
        self.frame_writer.update_resources(&txn.resource_updates);
        for doc_msg in &txn.scene_ops {
            match *doc_msg {
                SceneMsg::SetDisplayList {
                    ref epoch,
                    ref pipeline_id,
                    ref background,
                    ref viewport_size,
                    ref list_descriptor,
                    ..
                } => {
                    self.frame_writer.begin_write_display_list(
                        &mut self.scene,
                        epoch,
                        pipeline_id,
                        background,
                        viewport_size,
                        list_descriptor,
                    );
                }
                SceneMsg::SetRootPipeline(ref pipeline_id) => {
                    self.scene.set_root_pipeline_id(pipeline_id.clone());
                }
                SceneMsg::RemovePipeline(ref pipeline_id) => {
                    self.scene.remove_pipeline(pipeline_id);
                }
                _ => {}
            }
        }
        for doc_msg in &txn.frame_ops {
            match *doc_msg {
                FrameMsg::UpdateDynamicProperties(ref properties) => {
                    self.scene.properties.set_properties(properties);
                }
                _ => {}
            }
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
            rsrc_gen: ResourceGenerator {
                base: rsrc_base,
                prefix: rsrc_prefix,
                next_num: 1,
            },
            images: HashMap::new(),
            fonts: HashMap::new(),
            font_instances: HashMap::new(),

            dl_descriptor: None,

            pipeline_id: None,

            last_frame_written: u32::max_value(),
        }
    }

    pub fn begin_write_display_list(
        &mut self,
        scene: &mut Scene,
        epoch: &Epoch,
        pipeline_id: &PipelineId,
        background_color: &Option<ColorF>,
        viewport_size: &LayoutSize,
        display_list: &BuiltDisplayListDescriptor,
    ) {
        unsafe {
            if CURRENT_FRAME_NUMBER == self.last_frame_written {
                return;
            }
            self.last_frame_written = CURRENT_FRAME_NUMBER;
        }

        self.dl_descriptor = Some(display_list.clone());
        self.pipeline_id = Some(pipeline_id.clone());

        scene.begin_display_list(pipeline_id, epoch, background_color, viewport_size);
    }

    pub fn finish_write_display_list(&mut self, scene: &mut Scene, data: &[u8]) {
        let dl_desc = self.dl_descriptor.take().unwrap();

        let payload = Payload::from_data(data);

        let dl = BuiltDisplayList::from_data(payload.display_list_data, dl_desc);

        let mut root_dl_table = new_table();
        {
            let mut iter = dl.iter();
            self.write_display_list(&mut root_dl_table, &dl, scene, &mut iter, &mut ClipIdMapper::new());
        }

        let mut root = new_table();
        if let Some(root_pipeline_id) = scene.root_pipeline_id {
            u32_vec_node(
                &mut root_dl_table,
                "id",
                &[root_pipeline_id.0, root_pipeline_id.1],
            );

            let mut referenced_pipeline_ids = vec![];
            let mut traversal = dl.iter();
            while let Some(item) = traversal.next() {
                if let &DisplayItem::Iframe(k) = item.item() {
                    referenced_pipeline_ids.push(k.pipeline_id);
                }
            }

            let mut pipelines = vec![];
            for pipeline_id in referenced_pipeline_ids {
                if !scene.display_lists.contains_key(&pipeline_id) {
                    continue;
                }
                let mut pipeline = new_table();
                u32_vec_node(&mut pipeline, "id", &[pipeline_id.0, pipeline_id.1]);

                let dl = scene.display_lists.get(&pipeline_id).unwrap();
                let mut iter = dl.iter();
                self.write_display_list(&mut pipeline, &dl, scene, &mut iter, &mut ClipIdMapper::new());
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
            let mut file = fs::File::create(&frame_file_name).unwrap();
            file.write_all(&sb).unwrap();
        }

        scene.finish_display_list(self.pipeline_id.unwrap(), dl);
    }

    fn update_resources(&mut self, updates: &[ResourceUpdate]) {
        for update in updates {
            match *update {
                ResourceUpdate::AddImage(ref img) => {
                    if let Some(ref data) = self.images.get(&img.key) {
                          if data.path.is_some() {
                              return;
                          }
                    }

                    let stride = img.descriptor.stride.unwrap_or(
                        img.descriptor.size.width * img.descriptor.format.bytes_per_pixel(),
                    );
                    let bytes = match img.data {
                        ImageData::Raw(ref v) => (**v).clone(),
                        ImageData::External(_) => {
                            return;
                        }
                    };
                    self.images.insert(
                        img.key,
                        CachedImage {
                            width: img.descriptor.size.width,
                            height: img.descriptor.size.height,
                            stride,
                            format: img.descriptor.format,
                            bytes: Some(bytes),
                            tiling: img.tiling,
                            path: None,
                        },
                    );
                }
                ResourceUpdate::UpdateImage(ref img) => {
                    if let Some(ref mut data) = self.images.get_mut(&img.key) {
                        assert_eq!(data.width, img.descriptor.size.width);
                        assert_eq!(data.height, img.descriptor.size.height);
                        assert_eq!(data.format, img.descriptor.format);

                        if let ImageData::Raw(ref bytes) = img.data {
                            data.path = None;
                            data.bytes = Some((**bytes).clone());
                        } else {
                            // Other existing image types only make sense
                            // within the gecko integration.
                            println!(
                                "Wrench only supports updating buffer images ({}).",
                                "ignoring update command"
                            );
                        }
                    }
                }
                ResourceUpdate::AddBlobImage(..)
                | ResourceUpdate::UpdateBlobImage(..) => {
                    println!("Blob images not supported (ignoring command).");
                }
                ResourceUpdate::DeleteImage(img) => {
                    self.images.remove(&img);
                }
                ResourceUpdate::AddFont(ref font) => match font {
                    &AddFont::Raw(key, ref bytes, index) => {
                        self.fonts
                            .insert(key, CachedFont::Raw(Some(bytes.clone()), index, None));
                    }
                    &AddFont::Native(key, ref handle) => {
                        self.fonts.insert(key, CachedFont::Native(handle.clone(), None));
                    }
                },
                ResourceUpdate::DeleteFont(_) => {}
                ResourceUpdate::AddFontInstance(ref instance) => {
                    self.font_instances.insert(
                        instance.key,
                        CachedFontInstance {
                            font_key: instance.font_key,
                            glyph_size: instance.glyph_size,
                        },
                    );
                }
                ResourceUpdate::DeleteFontInstance(_) => {}
                ResourceUpdate::SetBlobImageVisibleArea(..) => {}
            }
        }
    }



    fn path_for_image(&mut self, key: ImageKey) -> Option<PathBuf> {
        let data = match self.images.get_mut(&key) {
            Some(data) => data,
            None => return None,
        };

        if data.path.is_some() {
            return data.path.clone();
        }
        let mut bytes = data.bytes.take().unwrap();
        let (path_file, path) = self.rsrc_gen.next_rsrc_paths(
            "img",
            "png",
        );

        assert!(data.stride > 0);
        let (color_type, bpp, do_unpremultiply) = match data.format {
            ImageFormat::RGBA8 |
            ImageFormat::BGRA8 => (ColorType::RGBA(8), 4, true),
            ImageFormat::R8 => (ColorType::Gray(8), 1, false),
            _ => {
                println!(
                    "Failed to write image with format {:?}, dimensions {}x{}, stride {}",
                    data.format,
                    data.width,
                    data.height,
                    data.stride
                );
                return None;
            }
        };

        if data.stride == data.width * bpp {
            if do_unpremultiply {
                unpremultiply(bytes.as_mut_slice());
            }
            save_buffer(
                &path_file,
                &bytes,
                data.width as u32,
                data.height as u32,
                color_type,
            ).unwrap();
        } else {
            // takes a buffer with a stride and copies it into a new buffer that has stride == width
            assert!(data.stride > data.width * bpp);
            let mut tmp: Vec<_> = bytes[..]
                .chunks(data.stride as usize)
                .flat_map(|chunk| {
                    chunk[.. (data.width * bpp) as usize].iter().cloned()
                })
                .collect();
            if do_unpremultiply {
                unpremultiply(tmp.as_mut_slice());
            }

            save_buffer(
                &path_file,
                &tmp,
                data.width as u32,
                data.height as u32,
                color_type
            ).unwrap();
        }

        data.path = Some(path.clone());
        Some(path)
    }

    fn make_complex_clip_node(&mut self, complex_clip: &ComplexClipRegion) -> Yaml {
        let mut t = new_table();
        rect_node(&mut t, "rect", &complex_clip.rect);
        yaml_node(
            &mut t,
            "radius",
            maybe_radius_yaml(&complex_clip.radii).unwrap(),
        );
        enum_node(&mut t, "clip-mode", complex_clip.mode);
        Yaml::Hash(t)
    }

    fn make_complex_clips_node(
        &mut self,
        complex_clips: ItemRange<ComplexClipRegion>,
    ) -> Option<Yaml> {
        let iter = complex_clips.iter();
        if iter.len() == 0 {
            return None;
        }

        let complex_items = iter
            .map(|ccx| if ccx.radii.is_zero() {
                rect_yaml(&ccx.rect)
            } else {
                self.make_complex_clip_node(&ccx)
            })
            .collect();
        Some(Yaml::Array(complex_items))
    }

    fn make_clip_mask_image_node(&mut self, image_mask: &Option<ImageMask>) -> Option<Yaml> {
        let mask = match image_mask {
            &Some(ref mask) => mask,
            &None => return None,
        };

        let mut mask_table = new_table();
        if let Some(path) = self.path_for_image(mask.image) {
            path_node(&mut mask_table, "image", &path);
        }
        rect_node(&mut mask_table, "rect", &mask.rect);
        bool_node(&mut mask_table, "repeat", mask.repeat);
        Some(Yaml::Hash(mask_table))
    }

    fn write_display_list_items(
        &mut self,
        list: &mut Vec<Yaml>,
        display_list: &BuiltDisplayList,
        scene: &Scene,
        list_iterator: &mut BuiltDisplayListIter,
        clip_id_mapper: &mut ClipIdMapper,
    ) {
        // continue_traversal is a big borrowck hack
        let mut continue_traversal = None;
        loop {
            if let Some(traversal) = continue_traversal.take() {
                *list_iterator = traversal;
            }
            let base = match list_iterator.next() {
                Some(base) => base,
                None => break,
            };

            let mut v = new_table();
            match *base.item() {
                DisplayItem::Rectangle(item) => {
                    str_node(&mut v, "type", "rect");
                    common_node(&mut v, clip_id_mapper, &item.common);
                    color_node(&mut v, "color", item.color);
                }
                DisplayItem::HitTest(item) => {
                    str_node(&mut v, "type", "hit-test");
                    common_node(&mut v, clip_id_mapper, &item.common);
                }
                DisplayItem::ClearRectangle(item) => {
                    str_node(&mut v, "type", "clear-rect");
                    common_node(&mut v, clip_id_mapper, &item.common);
                }
                DisplayItem::Line(item) => {
                    str_node(&mut v, "type", "line");
                    common_node(&mut v, clip_id_mapper, &item.common);
                    rect_node(&mut v, "bounds", &item.area);
                    if let LineStyle::Wavy = item.style {
                        f32_node(&mut v, "thickness", item.wavy_line_thickness);
                    }
                    str_node(&mut v, "orientation", item.orientation.as_str());
                    color_node(&mut v, "color", item.color);
                    str_node(&mut v, "style", item.style.as_str());
                }
                DisplayItem::Text(item) => {
                    let mut indices: Vec<u32> = vec![];
                    let mut offsets: Vec<f32> = vec![];
                    for g in base.glyphs() {
                        indices.push(g.index);
                        offsets.push(g.point.x);
                        offsets.push(g.point.y);
                    }
                    common_node(&mut v, clip_id_mapper, &item.common);
                    rect_node(&mut v, "bounds", &item.bounds);
                    u32_vec_node(&mut v, "glyphs", &indices);
                    f32_vec_node(&mut v, "offsets", &offsets);

                    let instance = self.font_instances.entry(item.font_key).or_insert_with(|| {
                        println!("Warning: font instance key not found in font instances table!");
                        CachedFontInstance {
                            font_key: FontKey::new(IdNamespace(0), 0),
                            glyph_size: Au::from_px(16),
                        }
                    });

                    f32_node(
                        &mut v,
                        "size",
                        instance.glyph_size.to_f32_px() * 12.0 / 16.0,
                    );
                    color_node(&mut v, "color", item.color);

                    let entry = self.fonts.entry(instance.font_key).or_insert_with(|| {
                        println!("Warning: font key not found in fonts table!");
                        CachedFont::Raw(Some(vec![]), 0, None)
                    });

                    match entry {
                        &mut CachedFont::Native(ref handle, ref mut path_opt) => {
                            native_font_handle_to_yaml(&mut self.rsrc_gen, handle, &mut v, path_opt);
                        }
                        &mut CachedFont::Raw(ref mut bytes_opt, index, ref mut path_opt) => {
                            if let Some(bytes) = bytes_opt.take() {
                                let (path_file, path) = self.rsrc_gen.next_rsrc_paths(
                                    "font",
                                    "ttf",
                                );
                                let mut file = fs::File::create(&path_file).unwrap();
                                file.write_all(&bytes).unwrap();
                                *path_opt = Some(path);
                            }

                            path_node(&mut v, "font", path_opt.as_ref().unwrap());
                            if index != 0 {
                                u32_node(&mut v, "font-index", index);
                            }
                        }
                    }
                }
                DisplayItem::Image(item) => {
                    common_node(&mut v, clip_id_mapper, &item.common);
                    rect_node(&mut v, "bounds", &item.bounds);
                    if let Some(path) = self.path_for_image(item.image_key) {
                        path_node(&mut v, "image", &path);
                    }
                    if let Some(&CachedImage {
                        tiling: Some(tile_size),
                        ..
                    }) = self.images.get(&item.image_key)
                    {
                        u32_node(&mut v, "tile-size", tile_size as u32);
                    }
                    size_node(&mut v, "stretch-size", &item.stretch_size);
                    size_node(&mut v, "tile-spacing", &item.tile_spacing);
                    match item.image_rendering {
                        ImageRendering::Auto => (),
                        ImageRendering::CrispEdges => str_node(&mut v, "rendering", "crisp-edges"),
                        ImageRendering::Pixelated => str_node(&mut v, "rendering", "pixelated"),
                    };
                    match item.alpha_type {
                        AlphaType::PremultipliedAlpha => str_node(&mut v, "alpha-type", "premultiplied-alpha"),
                        AlphaType::Alpha => str_node(&mut v, "alpha-type", "alpha"),
                    };
                }
                DisplayItem::YuvImage(item) => {
                    str_node(&mut v, "type", "yuv-image");
                    common_node(&mut v, clip_id_mapper, &item.common);
                    // TODO
                    println!("TODO YAML YuvImage");
                }
                DisplayItem::Border(item) => {
                    str_node(&mut v, "type", "border");
                    rect_node(&mut v, "bounds", &item.bounds);
                    common_node(&mut v, clip_id_mapper, &item.common);
                    match item.details {
                        BorderDetails::Normal(ref details) => {
                            let trbl =
                                vec![&details.top, &details.right, &details.bottom, &details.left];
                            let widths: Vec<f32> = vec![
                                item.widths.top,
                                item.widths.right,
                                item.widths.bottom,
                                item.widths.left,
                            ];
                            let colors: Vec<String> =
                                trbl.iter().map(|x| color_to_string(x.color)).collect();
                            let styles: Vec<String> = trbl.iter()
                                .map(|x| {
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
                                })
                                .collect();
                            yaml_node(&mut v, "width", f32_vec_yaml(&widths, true));
                            str_node(&mut v, "border-type", "normal");
                            yaml_node(&mut v, "color", string_vec_yaml(&colors, true));
                            yaml_node(&mut v, "style", string_vec_yaml(&styles, true));
                            bool_node(&mut v, "do_aa", details.do_aa);
                            if let Some(radius_node) = maybe_radius_yaml(&details.radius) {
                                yaml_node(&mut v, "radius", radius_node);
                            }
                        }
                        BorderDetails::NinePatch(ref details) => {
                            let widths: Vec<f32> = vec![
                                item.widths.top,
                                item.widths.right,
                                item.widths.bottom,
                                item.widths.left,
                            ];
                            let outset: Vec<f32> = vec![
                                details.outset.top,
                                details.outset.right,
                                details.outset.bottom,
                                details.outset.left,
                            ];
                            yaml_node(&mut v, "width", f32_vec_yaml(&widths, true));

                            match details.source {
                                NinePatchBorderSource::Image(image_key) => {
                                    str_node(&mut v, "border-type", "image");
                                    if let Some(path) = self.path_for_image(image_key) {
                                        path_node(&mut v, "image", &path);
                                    }
                                }
                                NinePatchBorderSource::Gradient(gradient) => {
                                    str_node(&mut v, "gradient", "image");
                                    point_node(&mut v, "start", &gradient.start_point);
                                    point_node(&mut v, "end", &gradient.end_point);
                                    let mut stops = vec![];
                                    for stop in base.gradient_stops() {
                                        stops.push(Yaml::Real(stop.offset.to_string()));
                                        stops.push(Yaml::String(color_to_string(stop.color)));
                                    }
                                    yaml_node(&mut v, "stops", Yaml::Array(stops));
                                    bool_node(&mut v, "repeat", gradient.extend_mode == ExtendMode::Repeat);
                                }
                                NinePatchBorderSource::RadialGradient(gradient) => {
                                    str_node(&mut v, "border-type", "radial-gradient");
                                    radial_gradient_to_yaml(
                                        &mut v,
                                        &gradient,
                                        base.gradient_stops(),
                                    );
                                }
                            }

                            i32_node(&mut v, "image-width", details.width);
                            i32_node(&mut v, "image-height", details.height);
                            let slice = [
                                details.slice.top,
                                details.slice.right,
                                details.slice.bottom,
                                details.slice.left,
                            ];
                            yaml_node(&mut v, "slice", i32_vec_yaml(&slice, true));
                            yaml_node(&mut v, "outset", f32_vec_yaml(&outset, true));
                            match details.repeat_horizontal {
                                RepeatMode::Stretch => {
                                    str_node(&mut v, "repeat-horizontal", "stretch")
                                }
                                RepeatMode::Repeat => {
                                    str_node(&mut v, "repeat-horizontal", "repeat")
                                }
                                RepeatMode::Round => str_node(&mut v, "repeat-horizontal", "round"),
                                RepeatMode::Space => str_node(&mut v, "repeat-horizontal", "space"),
                            };
                            match details.repeat_vertical {
                                RepeatMode::Stretch => {
                                    str_node(&mut v, "repeat-vertical", "stretch")
                                }
                                RepeatMode::Repeat => str_node(&mut v, "repeat-vertical", "repeat"),
                                RepeatMode::Round => str_node(&mut v, "repeat-vertical", "round"),
                                RepeatMode::Space => str_node(&mut v, "repeat-vertical", "space"),
                            };
                        }
                    }
                }
                DisplayItem::BoxShadow(item) => {
                    str_node(&mut v, "type", "box-shadow");
                    common_node(&mut v, clip_id_mapper, &item.common);
                    rect_node(&mut v, "box-bounds", &item.box_bounds);
                    vector_node(&mut v, "offset", &item.offset);
                    color_node(&mut v, "color", item.color);
                    f32_node(&mut v, "blur-radius", item.blur_radius);
                    f32_node(&mut v, "spread-radius", item.spread_radius);
                    if let Some(radius_node) = maybe_radius_yaml(&item.border_radius) {
                        yaml_node(&mut v, "border-radius", radius_node);
                    }
                    let clip_mode = match item.clip_mode {
                        BoxShadowClipMode::Outset => "outset",
                        BoxShadowClipMode::Inset => "inset",
                    };
                    str_node(&mut v, "clip-mode", clip_mode);
                }
                DisplayItem::Gradient(item) => {
                    str_node(&mut v, "type", "gradient");
                    rect_node(&mut v, "bounds", &item.bounds);
                    common_node(&mut v, clip_id_mapper, &item.common);
                    point_node(&mut v, "start", &item.gradient.start_point);
                    point_node(&mut v, "end", &item.gradient.end_point);
                    size_node(&mut v, "tile-size", &item.tile_size);
                    size_node(&mut v, "tile-spacing", &item.tile_spacing);
                    let mut stops = vec![];
                    for stop in base.gradient_stops() {
                        stops.push(Yaml::Real(stop.offset.to_string()));
                        stops.push(Yaml::String(color_to_string(stop.color)));
                    }
                    yaml_node(&mut v, "stops", Yaml::Array(stops));
                    bool_node(
                        &mut v,
                        "repeat",
                        item.gradient.extend_mode == ExtendMode::Repeat,
                    );
                }
                DisplayItem::RadialGradient(item) => {
                    str_node(&mut v, "type", "radial-gradient");
                    rect_node(&mut v, "bounds", &item.bounds);
                    common_node(&mut v, clip_id_mapper, &item.common);
                    size_node(&mut v, "tile-size", &item.tile_size);
                    size_node(&mut v, "tile-spacing", &item.tile_spacing);
                    radial_gradient_to_yaml(
                        &mut v,
                        &item.gradient,
                        base.gradient_stops(),
                    );
                }
                DisplayItem::Iframe(item) => {
                    str_node(&mut v, "type", "iframe");
                    rect_node(&mut v, "bounds", &item.bounds);
                    rect_node(&mut v, "clip_rect", &item.clip_rect);
                    clip_and_scroll_node(
                        &mut v,
                        clip_id_mapper,
                        item.space_and_clip.clip_id,
                        item.space_and_clip.spatial_id
                    );
                    u32_vec_node(&mut v, "id", &[item.pipeline_id.0, item.pipeline_id.1]);
                    bool_node(&mut v, "ignore_missing_pipeline", item.ignore_missing_pipeline);
                }
                DisplayItem::PushStackingContext(item) => {
                    str_node(&mut v, "type", "stacking-context");
                    clip_and_scroll_node(
                        &mut v,
                        clip_id_mapper,
                        item.stacking_context.clip_id.unwrap_or(ClipId::invalid()),
                        item.spatial_id
                    );
                    point_node(&mut v, "origin", &item.origin);
                    bool_node(&mut v, "backface-visible", item.is_backface_visible);
                    write_stacking_context(
                        &mut v,
                        &item.stacking_context,
                        &scene.properties,
                        base.filters(),
                        base.filter_datas(),
                        base.filter_primitives(),
                    );

                    let mut sub_iter = base.sub_iter();
                    self.write_display_list(&mut v, display_list, scene, &mut sub_iter, clip_id_mapper);
                    continue_traversal = Some(sub_iter);
                }
                DisplayItem::PushReferenceFrame(item) => {
                    str_node(&mut v, "type", "reference-frame");
                    write_reference_frame(
                        &mut v,
                        &item.reference_frame,
                        &scene.properties,
                        clip_id_mapper,
                    );

                    let mut sub_iter = base.sub_iter();
                    self.write_display_list(&mut v, display_list, scene, &mut sub_iter, clip_id_mapper);
                    continue_traversal = Some(sub_iter);
                }
                DisplayItem::Clip(item) => {
                    str_node(&mut v, "type", "clip");
                    clip_and_scroll_node(
                        &mut v,
                        clip_id_mapper,
                        item.parent_space_and_clip.clip_id,
                        item.parent_space_and_clip.spatial_id);
                    rect_node(&mut v, "clip-rect", &item.clip_rect);
                    usize_node(&mut v, "id", clip_id_mapper.add_clip_id(item.id));

                    if let Some(complex) = self.make_complex_clips_node(
                        base.complex_clip()
                    ) {
                        yaml_node(&mut v, "complex", complex);
                    }

                    if let Some(mask_yaml) = self.make_clip_mask_image_node(&item.image_mask) {
                        yaml_node(&mut v, "image-mask", mask_yaml);
                    }
                }
                DisplayItem::ClipChain(item) => {
                    str_node(&mut v, "type", "clip-chain");

                    let id = ClipId::ClipChain(item.id);
                    u32_node(&mut v, "id", clip_id_mapper.add_clip_id(id) as u32);

                    let clip_ids = base.clip_chain_items().iter().map(|clip_id| {
                        clip_id_mapper.map_clip_id(&clip_id)
                    }).collect();
                    yaml_node(&mut v, "clips", Yaml::Array(clip_ids));

                    if let Some(parent) = item.parent {
                        let parent = ClipId::ClipChain(parent);
                        yaml_node(&mut v, "parent", clip_id_mapper.map_clip_id(&parent));
                    }
                }
                DisplayItem::ScrollFrame(item) => {
                    str_node(&mut v, "type", "scroll-frame");
                    usize_node(&mut v, "id", clip_id_mapper.add_spatial_id(item.scroll_frame_id));
                    size_node(&mut v, "content-size", &item.content_rect.size);
                    rect_node(&mut v, "bounds", &item.clip_rect);
                    vector_node(&mut v, "external-scroll-offset", &item.external_scroll_offset);

                    if let Some(complex) = self.make_complex_clips_node(
                        base.complex_clip()
                    ) {
                        yaml_node(&mut v, "complex", complex);
                    }

                    if let Some(mask_yaml) = self.make_clip_mask_image_node(&item.image_mask) {
                        yaml_node(&mut v, "image-mask", mask_yaml);
                    }
                }
                DisplayItem::StickyFrame(item) => {
                    str_node(&mut v, "type", "sticky-frame");
                    usize_node(&mut v, "id", clip_id_mapper.add_spatial_id(item.id));
                    rect_node(&mut v, "bounds", &item.bounds);

                    if let Some(margin) = item.margins.top {
                        f32_node(&mut v, "margin-top", margin);
                    }
                    if let Some(margin) = item.margins.bottom {
                        f32_node(&mut v, "margin-bottom", margin);
                    }
                    if let Some(margin) = item.margins.left {
                        f32_node(&mut v, "margin-left", margin);
                    }
                    if let Some(margin) = item.margins.right {
                        f32_node(&mut v, "margin-right", margin);
                    }

                    let horizontal = vec![
                        Yaml::Real(item.horizontal_offset_bounds.min.to_string()),
                        Yaml::Real(item.horizontal_offset_bounds.max.to_string()),
                    ];
                    let vertical = vec![
                        Yaml::Real(item.vertical_offset_bounds.min.to_string()),
                        Yaml::Real(item.vertical_offset_bounds.max.to_string()),
                    ];

                    yaml_node(&mut v, "horizontal-offset-bounds", Yaml::Array(horizontal));
                    yaml_node(&mut v, "vertical-offset-bounds", Yaml::Array(vertical));

                    let applied = vec![
                        Yaml::Real(item.previously_applied_offset.x.to_string()),
                        Yaml::Real(item.previously_applied_offset.y.to_string()),
                    ];
                    yaml_node(&mut v, "previously-applied-offset", Yaml::Array(applied));
                }
                DisplayItem::BackdropFilter(item) => {
                    str_node(&mut v, "type", "backdrop-filter");
                    common_node(&mut v, clip_id_mapper, &item.common);

                    write_filters(&mut v, "filters", base.filters(), &scene.properties);
                    write_filter_datas(&mut v, "filter-datas", base.filter_datas());
                    write_filter_primitives(&mut v, "filter-primitives", base.filter_primitives());
                }

                DisplayItem::PopReferenceFrame |
                DisplayItem::PopStackingContext => return,

                DisplayItem::SetGradientStops |
                DisplayItem::SetFilterOps |
                DisplayItem::SetFilterData |
                DisplayItem::SetFilterPrimitives => panic!("dummy item yielded?"),

                DisplayItem::PushShadow(item) => {
                    str_node(&mut v, "type", "shadow");
                    vector_node(&mut v, "offset", &item.shadow.offset);
                    color_node(&mut v, "color", item.shadow.color);
                    f32_node(&mut v, "blur-radius", item.shadow.blur_radius);
                }
                DisplayItem::PopAllShadows => {
                    str_node(&mut v, "type", "pop-all-shadows");
                }
            }
            if !v.is_empty() {
                list.push(Yaml::Hash(v));
            }
        }
    }

    fn write_display_list(
        &mut self,
        parent: &mut Table,
        display_list: &BuiltDisplayList,
        scene: &Scene,
        list_iterator: &mut BuiltDisplayListIter,
        clip_id_mapper: &mut ClipIdMapper,
    ) {
        let mut list = vec![];
        self.write_display_list_items(&mut list, display_list, scene, list_iterator, clip_id_mapper);
        parent.insert(Yaml::String("items".to_owned()), Yaml::Array(list));
    }
}

impl webrender::ApiRecordingReceiver for YamlFrameWriterReceiver {
    fn write_msg(&mut self, _: u32, msg: &ApiMsg) {
        match *msg {
            ApiMsg::UpdateResources(ref updates) => {
                self.frame_writer.update_resources(updates);
            }
            ApiMsg::UpdateDocuments(_, ref txns) => {
                for txn in txns {
                    self.update_document(txn);
                }
            }
            _ => {}
        }
    }

    fn write_payload(&mut self, _frame: u32, data: &[u8]) {
        if self.frame_writer.dl_descriptor.is_some() {
            self.frame_writer
                .finish_write_display_list(&mut self.scene, data);
        }
    }
}

/// This structure allows mapping both `Clip` and `ClipExternalId`
/// `ClipIds` onto one set of numeric ids. It also handles `SpatialId`
/// in a separate map. This prevents ids from clashing in the yaml output.
struct ClipIdMapper {
    clip_map: HashMap<ClipId, usize>,
    spatial_map: HashMap<SpatialId, usize>,
    current_clip_id: usize,
    current_spatial_id: usize,
}

impl ClipIdMapper {
    fn new() -> Self {
        ClipIdMapper {
            clip_map: HashMap::new(),
            spatial_map: HashMap::new(),
            current_clip_id: 1, // see FIRST_CLIP_NODE_INDEX
            current_spatial_id: 2, // see FIRST_SPATIAL_NODE_INDEX
        }
    }

    fn add_clip_id(&mut self, id: ClipId) -> usize {
        self.clip_map.insert(id, self.current_clip_id);
        self.current_clip_id += 1;
        self.current_clip_id - 1
    }

    fn add_spatial_id(&mut self, id: SpatialId) -> usize {
        self.spatial_map.insert(id, self.current_spatial_id);
        self.current_spatial_id += 1;
        self.current_spatial_id - 1
    }

    fn map_spatial_id(&self, id: &SpatialId) -> Yaml {
        if id.is_root_reference_frame() {
            Yaml::String("root-reference-frame".to_owned())
        } else if id.is_root_scroll_node() {
            Yaml::String("root-scroll-node".to_owned())
        } else {
            Yaml::Integer(self.spatial_map[id] as i64)
        }
    }

    fn map_clip_id(&self, id: &ClipId) -> Yaml {
        assert!(id.is_valid());
        if id.is_root() {
            Yaml::String("root_clip".to_owned())
        } else {
            Yaml::Integer(self.clip_map[id] as i64)
        }
    }

    fn map_clip_and_scroll_ids(&self, clip_id: Option<ClipId>, spatial_id: SpatialId) -> Yaml {
        let scroll_node_yaml = self.map_spatial_id(&spatial_id);
        match clip_id {
            Some(ref clip_node_id) => Yaml::Array(vec![
                scroll_node_yaml,
                self.map_clip_id(&clip_node_id)
            ]),
            None => scroll_node_yaml,
        }
    }
}
