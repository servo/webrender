/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


use app_units::Au;
use blob;
use crossbeam::sync::chase_lev;
#[cfg(windows)]
use dwrote;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use font_loader::system_fonts;
use glutin::WindowProxy;
use json_frame_writer::JsonFrameWriter;
use ron_frame_writer::RonFrameWriter;
use std::collections::HashMap;
use std::path::PathBuf;
use time;
use webrender;
use webrender::api::*;
use yaml_frame_writer::YamlFrameWriterReceiver;
use {WindowWrapper, BLACK_COLOR, WHITE_COLOR};

// TODO(gw): This descriptor matches what we currently support for fonts
//           but is quite a mess. We should at least document and
//           use better types for things like the style and stretch.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum FontDescriptor {
    Path { path: PathBuf, font_index: u32 },
    Family { name: String },
    Properties {
        family: String,
        weight: u32,
        style: u32,
        stretch: u32,
    },
}

pub enum SaveType {
    Yaml,
    Json,
    Ron,
    Binary,
}

struct Notifier {
    window_proxy: Option<WindowProxy>,
    frames_notified: u32,
    timing_receiver: chase_lev::Stealer<time::SteadyTime>,
    verbose: bool,
}

impl Notifier {
    fn new(
        window_proxy: Option<WindowProxy>,
        timing_receiver: chase_lev::Stealer<time::SteadyTime>,
        verbose: bool,
    ) -> Notifier {
        Notifier {
            window_proxy,
            frames_notified: 0,
            timing_receiver,
            verbose,
        }
    }
}

impl RenderNotifier for Notifier {
    fn new_frame_ready(&mut self) {
        match self.timing_receiver.steal() {
            chase_lev::Steal::Data(last_timing) => {
                self.frames_notified += 1;
                if self.verbose && self.frames_notified == 600 {
                    let elapsed = time::SteadyTime::now() - last_timing;
                    println!(
                        "frame latency (consider queue depth here): {:3.6} ms",
                        elapsed.num_microseconds().unwrap() as f64 / 1000.
                    );
                    self.frames_notified = 0;
                }
            }
            _ => {
                println!("Notified of frame, but no frame was ready?");
            }
        }
        if let Some(ref window_proxy) = self.window_proxy {
            #[cfg(not(target_os = "android"))]
            window_proxy.wakeup_event_loop();
        }
    }

    fn new_scroll_frame_ready(&mut self, _composite_needed: bool) {
        if let Some(ref window_proxy) = self.window_proxy {
            #[cfg(not(target_os = "android"))]
            window_proxy.wakeup_event_loop();
        }
    }
}

pub trait WrenchThing {
    fn next_frame(&mut self);
    fn prev_frame(&mut self);
    fn do_frame(&mut self, &mut Wrench) -> u32;
    fn queue_frames(&self) -> u32 {
        0
    }
}

pub struct Wrench {
    window_size: DeviceUintSize,
    device_pixel_ratio: f32,

    pub renderer: webrender::Renderer,
    pub api: RenderApi,
    pub document_id: DocumentId,
    pub root_pipeline_id: PipelineId,

    window_title_to_set: Option<String>,

    graphics_api: webrender::GraphicsApiInfo,

    pub rebuild_display_lists: bool,
    pub verbose: bool,

    pub frame_start_sender: chase_lev::Worker<time::SteadyTime>,
}

impl Wrench {
    pub fn new(
        window: &mut WindowWrapper,
        shader_override_path: Option<PathBuf>,
        dp_ratio: f32,
        save_type: Option<SaveType>,
        size: DeviceUintSize,
        do_rebuild: bool,
        no_subpixel_aa: bool,
        debug: bool,
        verbose: bool,
        no_scissor: bool,
        no_batch: bool,
    ) -> Wrench {
        println!("Shader override path: {:?}", shader_override_path);

        let recorder = save_type.map(|save_type| match save_type {
            SaveType::Yaml => Box::new(
                YamlFrameWriterReceiver::new(&PathBuf::from("yaml_frames")),
            ) as Box<webrender::ApiRecordingReceiver>,
            SaveType::Json => Box::new(JsonFrameWriter::new(&PathBuf::from("json_frames"))) as
                Box<webrender::ApiRecordingReceiver>,
            SaveType::Ron => Box::new(RonFrameWriter::new(&PathBuf::from("ron_frames"))) as
                Box<webrender::ApiRecordingReceiver>,
            SaveType::Binary => Box::new(webrender::BinaryRecorder::new(
                &PathBuf::from("wr-record.bin"),
            )) as Box<webrender::ApiRecordingReceiver>,
        });

        let opts = webrender::RendererOptions {
            device_pixel_ratio: dp_ratio,
            resource_override_path: shader_override_path,
            recorder,
            enable_subpixel_aa: !no_subpixel_aa,
            debug,
            enable_clear_scissor: !no_scissor,
            enable_batcher: !no_batch,
            max_recorded_profiles: 16,
            blob_image_renderer: Some(Box::new(blob::CheckerboardRenderer::new())),
            ..Default::default()
        };

        let (renderer, sender) = webrender::Renderer::new(window.clone_gl(), opts).unwrap();
        let api = sender.create_api();
        let document_id = api.add_document(size);

        let proxy = window.create_window_proxy();
        // put an Awakened event into the queue to kick off the first frame
        if let Some(ref wp) = proxy {
            #[cfg(not(target_os = "android"))]
            wp.wakeup_event_loop();
        }

        let (timing_sender, timing_receiver) = chase_lev::deque();
        let notifier = Box::new(Notifier::new(proxy, timing_receiver, verbose));
        renderer.set_render_notifier(notifier);

        let graphics_api = renderer.get_graphics_api_info();

        let mut wrench = Wrench {
            window_size: size,

            renderer,
            api,
            document_id,
            window_title_to_set: None,

            rebuild_display_lists: do_rebuild,
            verbose,
            device_pixel_ratio: dp_ratio,

            root_pipeline_id: PipelineId(0, 0),

            graphics_api,
            frame_start_sender: timing_sender,
        };

        wrench.set_title("start");
        wrench
            .api
            .set_root_pipeline(wrench.document_id, wrench.root_pipeline_id);

        wrench
    }

    pub fn layout_simple_ascii(
        &self,
        font_key: FontKey,
        text: &str,
        size: Au,
        origin: LayerPoint,
    ) -> (Vec<u32>, Vec<LayerPoint>, LayoutRect) {
        // Map the string codepoints to glyph indices in this font.
        // Just drop any glyph that isn't present in this font.
        let indices: Vec<u32> = self.api
            .get_glyph_indices(font_key, text)
            .iter()
            .filter_map(|idx| *idx)
            .collect();

        // Retrieve the metrics for each glyph.
        let font = FontInstance::new(
            font_key,
            size,
            ColorF::new(0.0, 0.0, 0.0, 1.0),
            FontRenderMode::Alpha,
            SubpixelDirection::Horizontal,
            None,
            Vec::new(),
            false,
        );
        let mut keys = Vec::new();
        for glyph_index in &indices {
            keys.push(GlyphKey::new(
                *glyph_index,
                LayerPoint::zero(),
                FontRenderMode::Alpha,
                SubpixelDirection::Horizontal,
            ));
        }
        let metrics = self.api.get_glyph_dimensions(font, keys);
        let mut bounding_rect = LayoutRect::zero();
        let mut positions = Vec::new();

        let mut x = origin.x;
        let y = origin.y;
        for metric in metrics {
            positions.push(LayerPoint::new(x, y));

            match metric {
                Some(metric) => {
                    let glyph_rect = LayoutRect::new(
                        LayoutPoint::new(x + metric.left as f32, y - metric.top as f32),
                        LayoutSize::new(metric.width as f32, metric.height as f32)
                    );
                    bounding_rect = bounding_rect.union(&glyph_rect);
                    x += metric.advance;
                }
                None => {
                    // Extract the advances from the metrics. The get_glyph_dimensions API
                    // has a limitation that it can't currently get dimensions for non-renderable
                    // glyphs (e.g. spaces), so just use a rough estimate in that case.
                    let space_advance = size.to_f32_px() / 3.0;
                    x += space_advance;
                }
            }
        }

        (indices, positions, bounding_rect)
    }

    pub fn set_title(&mut self, extra: &str) {
        self.window_title_to_set = Some(format!(
            "Wrench: {} ({}x) - {} - {}",
            extra,
            self.device_pixel_ratio,
            self.graphics_api.renderer,
            self.graphics_api.version
        ));
    }

    pub fn take_title(&mut self) -> Option<String> {
        self.window_title_to_set.take()
    }

    pub fn should_rebuild_display_lists(&self) -> bool {
        self.rebuild_display_lists
    }

    pub fn window_size_f32(&self) -> LayoutSize {
        LayoutSize::new(
            self.window_size.width as f32,
            self.window_size.height as f32,
        )
    }

    #[cfg(target_os = "windows")]
    pub fn font_key_from_native_handle(&mut self, descriptor: &NativeFontHandle) -> FontKey {
        let key = self.api.generate_font_key();
        let mut resources = ResourceUpdates::new();
        resources.add_native_font(key, descriptor.clone());
        self.api.update_resources(resources);
        key
    }

    #[cfg(target_os = "windows")]
    pub fn font_key_from_name(&mut self, font_name: &str) -> FontKey {
        let system_fc = dwrote::FontCollection::system();
        let family = system_fc.get_font_family_by_name(font_name).unwrap();
        let font = family.get_first_matching_font(
            dwrote::FontWeight::Regular,
            dwrote::FontStretch::Normal,
            dwrote::FontStyle::Normal,
        );
        let descriptor = font.to_descriptor();
        self.font_key_from_native_handle(&descriptor)
    }

    #[cfg(target_os = "windows")]
    pub fn font_key_from_properties(
        &mut self,
        family: &str,
        weight: u32,
        style: u32,
        stretch: u32,
    ) -> FontKey {
        let weight = dwrote::FontWeight::from_u32(weight);
        let style = dwrote::FontStyle::from_u32(style);
        let stretch = dwrote::FontStretch::from_u32(stretch);

        let desc = dwrote::FontDescriptor {
            family_name: family.to_owned(),
            weight,
            style,
            stretch,
        };
        self.font_key_from_native_handle(&desc)
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    pub fn font_key_from_properties(
        &mut self,
        family: &str,
        _weight: u32,
        _style: u32,
        _stretch: u32,
    ) -> FontKey {
        let property = system_fonts::FontPropertyBuilder::new()
            .family(family)
            .build();
        let (font, index) = system_fonts::get(&property).unwrap();
        self.font_key_from_bytes(font, index as u32)
    }

    #[cfg(unix)]
    pub fn font_key_from_name(&mut self, font_name: &str) -> FontKey {
        let property = system_fonts::FontPropertyBuilder::new()
            .family(font_name)
            .build();
        let (font, index) = system_fonts::get(&property).unwrap();
        self.font_key_from_bytes(font, index as u32)
    }

    #[cfg(target_os = "android")]
    pub fn font_key_from_name(&mut self, font_name: &str) -> FontKey {
        unimplemented!()
    }

    pub fn font_key_from_bytes(&mut self, bytes: Vec<u8>, index: u32) -> FontKey {
        let key = self.api.generate_font_key();
        let mut update = ResourceUpdates::new();
        update.add_raw_font(key, bytes, index);
        self.api.update_resources(update);
        key
    }

    pub fn add_font_instance(&mut self,
        font_key: FontKey,
        size: Au,
        synthetic_italics: bool,
        render_mode: Option<FontRenderMode>,
    ) -> FontInstanceKey {
        let key = self.api.generate_font_instance_key();
        let mut update = ResourceUpdates::new();
        let options = FontInstanceOptions {
            render_mode: render_mode.unwrap_or(FontRenderMode::Subpixel),
            synthetic_italics,
            ..Default::default()
        };
        update.add_font_instance(key, font_key, size, Some(options), None, Vec::new());
        self.api.update_resources(update);
        key
    }

    pub fn update(&mut self, dim: DeviceUintSize) {
        if dim != self.window_size {
            self.window_size = dim;
        }
    }

    pub fn begin_frame(&mut self) {
        self.frame_start_sender.push(time::SteadyTime::now());
    }

    pub fn send_lists(
        &mut self,
        frame_number: u32,
        display_lists: Vec<(PipelineId, LayerSize, BuiltDisplayList)>,
        scroll_offsets: &HashMap<ClipId, LayerPoint>,
    ) {
        let root_background_color = Some(ColorF::new(1.0, 1.0, 1.0, 1.0));

        for display_list in display_lists {
            self.api.set_display_list(
                self.document_id,
                Epoch(frame_number),
                root_background_color,
                self.window_size_f32(),
                display_list,
                false,
                ResourceUpdates::new(),
            );
        }

        for (id, offset) in scroll_offsets {
            self.api.scroll_node_with_id(
                self.document_id,
                *offset,
                *id,
                ScrollClamping::NoClamping,
            );
        }

        self.api.generate_frame(self.document_id, None);
    }

    pub fn get_frame_profiles(
        &mut self,
    ) -> (Vec<webrender::CpuProfile>, Vec<webrender::GpuProfile>) {
        self.renderer.get_frame_profiles()
    }

    pub fn render(&mut self) {
        self.renderer.update();
        self.renderer.render(self.window_size).unwrap();
    }

    pub fn refresh(&mut self) {
        self.begin_frame();
        self.api.generate_frame(self.document_id, None);
    }

    pub fn show_onscreen_help(&mut self) {
        let help_lines = [
            "Esc, Q - Quit",
            "H - Toggle help",
            "R - Toggle recreating display items each frame",
            "P - Toggle profiler",
            "O - Toggle showing intermediate targets",
            "I - Toggle showing texture caches",
            "B - Toggle showing alpha primitive rects",
            "M - Trigger memory pressure event",
        ];

        let color_and_offset = [(*BLACK_COLOR, 2.0), (*WHITE_COLOR, 0.0)];
        let dr = self.renderer.debug_renderer();

        for ref co in &color_and_offset {
            let x = self.device_pixel_ratio * (15.0 + co.1);
            let mut y = self.device_pixel_ratio * (15.0 + co.1 + dr.line_height());
            for ref line in &help_lines {
                dr.add_text(x, y, line, co.0.into());
                y += self.device_pixel_ratio * dr.line_height();
            }
        }
    }
}
