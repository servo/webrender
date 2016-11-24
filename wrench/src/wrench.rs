/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
#[cfg(windows)]
use dwrote;
use euclid::Size2D;
use gleam::gl;
use glutin;
use glutin::WindowProxy;
use image;
use image::GenericImage;
use std::collections::HashMap;
use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use webrender;
use webrender_traits::*;
use yaml_rust::Yaml;
use yaml_frame_writer::YamlFrameWriter;
use json_frame_writer::JsonFrameWriter;

use {WHITE_COLOR, BLACK_COLOR};

pub enum SaveType {
    Yaml,
    Json,
}

struct Notifier {
    window_proxy: WindowProxy,
}

impl Notifier {
    fn new(window_proxy: WindowProxy) -> Notifier {
        Notifier {
            window_proxy: window_proxy,
        }
    }
}

impl RenderNotifier for Notifier {
    fn new_frame_ready(&mut self) {
        self.window_proxy.wakeup_event_loop();
    }

    fn new_scroll_frame_ready(&mut self, _composite_needed: bool) {
        self.window_proxy.wakeup_event_loop();
    }

    fn pipeline_size_changed(&mut self,
                             _: PipelineId,
                             _: Option<Size2D<f32>>) {
    }
}

#[cfg(target_os = "windows")]
pub fn layout_simple_ascii(face: NativeFontHandle, text: &str, size: Au) -> (Vec<u16>, Vec<f32>) {
    let system_fc = dwrote::FontCollection::system();
    let font = system_fc.get_font_from_descriptor(&(face as dwrote::FontDescriptor)).unwrap();
    let face = font.create_font_face();

    let chars: Vec<u32> = text.chars().map(|c| c as u32).collect();
    let indices = face.get_glyph_indices(&chars);
    let glyph_metrics = face.get_design_glyph_metrics(&indices, false);

    let device_pixel_ratio: f32 = 1.0;
    let em_size = size.to_f32_px() / 16.;
    let design_units_per_pixel = face.metrics().designUnitsPerEm as f32 / 16. as f32;
    let scaled_design_units_to_pixels = (em_size * device_pixel_ratio) / design_units_per_pixel;

    let advances = glyph_metrics.iter().map(|m| m.advanceWidth as f32 * scaled_design_units_to_pixels).collect();

    (indices, advances)
}

#[cfg(not(target_os = "windows"))]
pub fn layout_simple_ascii(face: NativeFontHandle, text: &str, size: Au) -> (Vec<u16>, Vec<f32>) {
    panic!("Can't layout simple ascii on this platform");
}

pub trait WrenchThing {
    fn next_frame(&mut self);
    fn prev_frame(&mut self);
    fn do_frame(&mut self, &mut Wrench) -> u32;
}

pub struct Wrench {
    pub window: glutin::Window,
    pub window_size: Size2D<u32>,
    pub device_pixel_ratio: f32,

    pub renderer: webrender::renderer::Renderer,
    pub sender: RenderApiSender,
    pub api: RenderApi,

    pub image_map: HashMap<PathBuf, (ImageKey, Size2D<f32>)>,

    // internal housekeeping
    pub root_pipeline_id: PipelineId,
    pub next_scroll_layer_id: usize,

    pub gl_renderer: String,
    pub gl_version: String,
}

impl Wrench {
    pub fn new(shader_override_path: Option<PathBuf>,
               dp_ratio: Option<f32>,
               win_size: Option<&str>,
               save_type: Option<SaveType>,
               subpixel_aa: bool,
               debug: bool)
           -> Wrench
    {
        // First create our GL window
        let window = glutin::WindowBuilder::new()
            .with_gl(glutin::GlRequest::Specific(glutin::Api::OpenGl, (3, 2)))
            .build().unwrap();

        unsafe {
            window.make_current().ok();
            gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);
            gl::clear_color(0.3, 0.0, 0.0, 1.0);
        }

        let gl_version = unsafe {
            let data = CStr::from_ptr(gl::GetString(gl::VERSION) as *const _).to_bytes().to_vec();
            String::from_utf8(data).unwrap()
        };

        let gl_renderer = unsafe {
            let data = CStr::from_ptr(gl::GetString(gl::RENDERER) as *const _).to_bytes().to_vec();
            String::from_utf8(data).unwrap()
        };

        let size = win_size.map(|s| {
            let x = s.find('x').expect("Size must be specified exactly as widthxheight");
            let w = s[0..x].parse::<u32>().expect("Invalid size width");
            let h = s[x+1..].parse::<u32>().expect("Invalid size height");
            Size2D::new(w, h)
        }).unwrap_or(Size2D::<u32>::new(1920, 1080));

        let dp_ratio = dp_ratio.unwrap_or(1.0);
        let win_size_mult = 1.0; //dp_ratio / window.hidpi_factor();

        println!("OpenGL version {}, {}", gl_version, gl_renderer);
        println!("Shader override path: {:?}", shader_override_path);
        println!("hidpi factor: {} (native {})", dp_ratio, window.hidpi_factor());

        if let Some(ref save_type) = save_type {
            let recorder = match save_type {
                &SaveType::Yaml => Box::new(YamlFrameWriter::new(&PathBuf::from("yaml_frames")))
                    as Box<webrender::ApiRecordingReceiver>,
                &SaveType::Json => Box::new(JsonFrameWriter::new(&PathBuf::from("json_frames")))
                    as Box<webrender::ApiRecordingReceiver>,
            };
            webrender::set_recording_detour(Some(recorder));
        }

        window.set_inner_size((size.width as f32 * win_size_mult) as u32,
                              (size.height as f32 * win_size_mult) as u32);

        let opts = webrender::RendererOptions {
            device_pixel_ratio: dp_ratio,
            resource_override_path: shader_override_path,
            enable_aa: false,
            enable_msaa: false,
            enable_profiler: false,
            enable_recording: save_type.is_some(),
            enable_scrollbars: false,
            enable_subpixel_aa: subpixel_aa,
            debug: debug,
            precache_shaders: false,
            renderer_kind: RendererKind::Native,
        };

        let (renderer, sender) = webrender::renderer::Renderer::new(opts);
        let api = sender.create_api();

        let notifier = Box::new(Notifier::new(window.create_window_proxy()));
        renderer.set_render_notifier(notifier);

        let mut wrench = Wrench {
            window: window,
            window_size: size,

            renderer: renderer,
            sender: sender,
            api: api,

            device_pixel_ratio: dp_ratio,

            image_map: HashMap::new(),

            root_pipeline_id: PipelineId(0, 0),
            next_scroll_layer_id: 0,

            gl_renderer: gl_renderer,
            gl_version: gl_version,
        };

        wrench.set_title("start");

        wrench
    }

    pub fn set_title(&mut self, extra: &str) {
        self.window.set_title(&format!("Wrench: {} ({}x) - {} - {}", extra,
            self.device_pixel_ratio, self.gl_renderer, self.gl_version));
    }

    pub fn window_size_f32(&self) -> Size2D<f32> {
        return Size2D::new(self.window_size.width as f32,
                           self.window_size.height as f32)
    }

    pub fn next_scroll_layer_id(&mut self) -> ScrollLayerId {
        let scroll_layer_id = ServoScrollRootId(self.next_scroll_layer_id);
        self.next_scroll_layer_id += 1;
        ScrollLayerId::new(self.root_pipeline_id, 0, scroll_layer_id)
    }

    #[cfg(target_os = "windows")]
    pub fn font_key_from_native_handle(&mut self, descriptor: &NativeFontHandle) -> FontKey {
        self.api.add_native_font(descriptor.clone())
    }

    #[cfg(target_os = "windows")]
    pub fn font_key_from_name(&mut self, font_name: &str) -> (FontKey, Option<NativeFontHandle>) {
        let system_fc = dwrote::FontCollection::system();
        let family = system_fc.get_font_family_by_name(font_name).unwrap();
        let font = family.get_first_matching_font(dwrote::FontWeight::Regular,
                                                  dwrote::FontStretch::Normal,
                                                  dwrote::FontStyle::Normal);
        let descriptor = font.to_descriptor();
        let key = self.api.add_native_font(descriptor.clone());
        (key, Some(descriptor))
    }

    #[cfg(target_os = "windows")]
    pub fn font_key_from_yaml_table(&mut self, item: &Yaml) -> (FontKey, Option<NativeFontHandle>) {
        assert!(!item["family"].is_badvalue());
        let family = item["family"].as_str().unwrap();
        let weight = dwrote::FontWeight::from_u32(item["weight"].as_i64().unwrap_or(400) as u32);
        let style = dwrote::FontStyle::from_u32(item["style"].as_i64().unwrap_or(0) as u32);
        let stretch = dwrote::FontStretch::from_u32(item["stretch"].as_i64().unwrap_or(5) as u32);

        let desc = dwrote::FontDescriptor {
            family_name: family.to_owned(),
            weight: weight,
            style: style,
            stretch: stretch,
        };
        (self.font_key_from_native_handle(&desc), Some(desc))
    }

    #[cfg(not(target_os = "windows"))]
    pub fn font_key_from_native_handle(&mut self, descriptor: &NativeFontHandle) -> FontKey {
        panic!("Can't font_key_from_native_handle on this platform");
    }

    #[cfg(not(target_os = "windows"))]
    pub fn font_key_from_name(&mut self, font_name: &str) -> (FontKey, Option<NativeFontHandle>) {
        panic!("Can't font_key_from_name on this platform");
    }

    #[cfg(not(target_os = "windows"))]
    pub fn font_key_from_yaml_table(&mut self, item: &Yaml) -> (FontKey, Option<NativeFontHandle>) {
        panic!("Can't font_key_from_yaml_table on this platform");
    }

    pub fn font_key_from_bytes(&mut self, bytes: Vec<u8>) -> (FontKey, Option<NativeFontHandle>) {
        let key = self.api.add_raw_font(bytes);
        (key, None)
    }

    pub fn add_or_get_image(&mut self, file: &Path) -> (ImageKey, Size2D<f32>) {
        let key = file.to_owned();
        if let Some(k) = self.image_map.get(&key) {
            return *k
        }

        let image = image::open(file).unwrap();
        let image_dims = image.dimensions();
        let image_key = self.api.add_image(image_dims.0, image_dims.1,
                                           None, // stride
                                           match image {
                                               image::ImageLuma8(_) => ImageFormat::A8,
                                               image::ImageRgb8(_) => ImageFormat::RGB8,
                                               image::ImageRgba8(_) => ImageFormat::RGBA8,
                                               _ => panic!("We don't support whatever your crazy image type is, come on"),
                                           },
                                           ImageData::Raw(Arc::new(image.raw_pixels())));

        let val = (image_key, Size2D::new(image_dims.0 as f32, image_dims.1 as f32));
        self.image_map.insert(key, val);
        val
    }

    pub fn update(&mut self) {
        let (width, height) = self.window.get_inner_size().unwrap();
        let dim = Size2D::new(width, height);
        if dim != self.window_size {
            gl::viewport(0, 0, width as i32, height as i32);
            self.window_size = dim;
        }

        gl::clear(gl::COLOR_BUFFER_BIT);
    }

    pub fn render(&mut self) {
        self.renderer.update();
        self.renderer.render(self.window_size);
        self.window.swap_buffers().ok();
    }

    //pub fn set_recorder<T>(&mut self, r: Box<webrender_traits::AppMsgReceiver>) {
    //}

    pub fn show_onscreen_help(&mut self) {
        let help_lines = [
            "Esc, Q - Quit",
            "H - Toggle help",
            "R - Toggle recreating display items each frame",
            "P - Toggle profiler"
        ];

        let color_and_offset = [ (*BLACK_COLOR, 2.0), (*WHITE_COLOR, 0.0) ];
        let dr = self.renderer.debug_renderer();

        for ref co in color_and_offset.iter() {
            let x = self.device_pixel_ratio * (15.0 + co.1);
            let mut y = self.device_pixel_ratio * (15.0 + co.1 + dr.line_height());
            for ref line in help_lines.iter() {
                dr.add_text(x, y, line, &co.0);
                y += self.device_pixel_ratio * dr.line_height();
            }
        }
    }
}
