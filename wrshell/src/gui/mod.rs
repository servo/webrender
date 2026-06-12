/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

mod debug_flags;
mod profiler;
mod shell;
mod textures;
mod composite_view;
mod draw_calls;
mod timeline;

use eframe::egui;
use webrender_api::{DebugFlags, RenderCommandInfo};
use webrender_api::debugger::{DebuggerMessage, DebuggerTextureContent, ProfileCounterId, CompositorDebugInfo};
use crate::{command, net};
use std::collections::{HashMap, BTreeMap, VecDeque};
use std::fs;
use std::io::Write;
use std::sync::mpsc;

use profiler::Graph;

#[allow(dead_code)]
enum ApplicationEvent {
    RunCommand(String),
    NetworkEvent(net::NetworkEvent),
}

struct LoggedFrame {
    pub render_commands: Option<Vec<RenderCommandInfo>>,
}

struct FrameLog {
    pub frames: VecDeque<LoggedFrame>,
    pub enabled: bool,
    pub frames_end: usize,
}

impl FrameLog {
    pub fn new() -> Self {
        FrameLog {
            frames: VecDeque::with_capacity(100),
            enabled: false,
            frames_end: 0,
        }
    }

    pub fn first_frame_index(&self) -> usize {
        self.frames_end - self.frames.len()
    }

    pub fn last_frame_index(&self) -> usize {
        self.frames_end.max(1) - 1
    }

    pub fn frame(&self, idx: usize) -> Option<&LoggedFrame> {
        let i = idx - self.first_frame_index();
        if i < self.frames.len() {
            return Some(&self.frames[i]);
        }

        None
    }
}

struct DataModel {
    is_connected: bool,
    debug_flags: DebugFlags,
    cmd: String,
    log: Vec<String>,
    documents: Vec<Document>,
    preview_doc_index: Option<usize>,
    profile_graphs: HashMap<ProfileCounterId, Graph>,
    frame_log: FrameLog,
    timeline: timeline::Timeline,
}

impl DataModel {
    fn new() -> Self {
        DataModel {
            is_connected: false,
            debug_flags: DebugFlags::empty(),
            cmd: String::new(),
            log: Vec::new(),
            documents: Vec::new(),
            preview_doc_index: None,
            profile_graphs: HashMap::new(),
            frame_log: FrameLog::new(),
            timeline: timeline::Timeline::new(),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub enum Tool {
    DebugFlags,
    Profiler,
    Shell,
    Documents,
    Preview,
    DrawCalls,
    Timeline,
}

impl egui_tiles::Behavior<Tool> for Gui {
    fn tab_title_for_pane(&mut self, tool: &Tool) -> egui::WidgetText {
        let title = match tool {
            Tool::DebugFlags => { "Debug flags" }
            Tool::Profiler => { "Profiler" }
            Tool::Shell => { "Shell" }
            Tool::Documents => { "Documents" }
            Tool::Preview => { "Preview" }
            Tool::DrawCalls => { "Draw calls" }
            Tool::Timeline => { "Timeline" }
        };

        title.into()
    }

    fn pane_ui(&mut self, ui: &mut egui::Ui, _id: egui_tiles::TileId, tool: &mut Tool) -> egui_tiles::UiResponse {
        // Add a bit of margin around the panes, otherwise their content hugs the
        // border in an awkward way. There may be a setting somewhere in egui_tiles
        // rather than doing it manually.
        egui::Frame::new().inner_margin(egui::Margin::symmetric(5, 5)).show(ui, |ui| {
            match tool {
                Tool::Documents => { do_documents_ui(self, ui); }
                Tool::Preview => { do_preview_ui(self, ui); }
                Tool::DebugFlags => { debug_flags::ui(self, ui); }
                Tool::Profiler => { profiler::ui(self, ui); }
                Tool::Shell => { shell::ui(self, ui); }
                Tool::DrawCalls => { draw_calls::ui(self, ui); }
                Tool::Timeline => { timeline::ui(self, ui); }
            }
        });

        Default::default()
    }

    fn simplification_options(&self) -> egui_tiles::SimplificationOptions {
        let mut options = egui_tiles::SimplificationOptions::default();
        options.all_panes_must_have_tabs = true;
        options
    }

    fn tab_title_spacing(&self, _visuals: &egui::Visuals) -> f32 {
        50.0
    }

    fn gap_width(&self, _style: &egui::Style) -> f32 {
        2.0
    }

    fn tab_outline_stroke(
        &self,
        _visuals: &egui::Visuals,
        _tiles: &egui_tiles::Tiles<Tool>,
        _tile_id: egui_tiles::TileId,
        _state: &egui_tiles::TabState,
    ) -> egui::Stroke {
        egui::Stroke::NONE
    }
}

pub struct Gui {
    data_model: DataModel,
    net: net::HttpConnection,
    cmd_list: command::CommandList,
    cmd_history: Vec<String>,
    cmd_history_index: usize,
    doc_id: usize,
    event_receiver: mpsc::Receiver<ApplicationEvent>,
    ui_tiles: Option<egui_tiles::Tree<Tool>>,
}

impl Gui {
    pub fn new(host: &str, cmd_list: command::CommandList) -> Self {
        let net = net::HttpConnection::new(host);
        let data_model = DataModel::new();

        let (event_sender, event_receiver) = mpsc::channel();

        // Spawn network event thread
        let host_clone = host.to_string();
        std::thread::spawn(move || {
            net::NetworkEventStream::spawn(&host_clone, move |event| {
                let _ = event_sender.send(ApplicationEvent::NetworkEvent(event));
            });
        });

        // Try to load saved ui_tiles state, or create default layout
        let save = fs::read_to_string(config_path())
            .ok()
            .and_then(|content| {
                serde_json::from_str::<GuiSavedState>(&content)
                    .map_err(|e| {
                        eprintln!("Failed to deserialize ui_tiles state: {}", e);
                        e
                    })
                    .ok()
            }).and_then(|save| {
                if save.version == GuiSavedState::VERSION {
                    Some(save)
                } else {
                    None
                }
            }).unwrap_or_else(|| {
                // Create default layout
                let mut tiles = egui_tiles::Tiles::default();
                let tabs = vec![
                    tiles.insert_pane(Tool::Profiler),
                    tiles.insert_pane(Tool::Preview),
                    tiles.insert_pane(Tool::DrawCalls),
                ];
                let side = vec![
                    tiles.insert_pane(Tool::DebugFlags),
                    tiles.insert_pane(Tool::Documents),
                ];
                let side = tiles.insert_vertical_tile(side);
                let main_tile = tiles.insert_tab_tile(tabs);
                let main_and_side = vec![
                    side,
                    main_tile,
                ];
                let shell_and_timeline = vec![
                    tiles.insert_pane(Tool::Shell),
                    tiles.insert_pane(Tool::Timeline),
                ];
                let shell_and_timeline = tiles.insert_tab_tile(shell_and_timeline);
                let main_and_side = tiles.insert_horizontal_tile(main_and_side);
                let root = tiles.insert_vertical_tile(vec![
                    main_and_side,
                    shell_and_timeline,
                ]);

                GuiSavedState {
                    version: GuiSavedState::VERSION,
                    cmd_history: Vec::new(),
                    ui_tiles: egui_tiles::Tree::new("WR debugger", root, tiles),
                }
            });

        Gui {
            data_model,
            net,
            cmd_list,
            cmd_history: save.cmd_history,
            cmd_history_index: 0,
            doc_id: 0,
            event_receiver,
            ui_tiles: Some(save.ui_tiles),
        }
    }

    pub fn run(self) {
        let native_options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1280.0, 720.0])
                .with_title("WebRender Debug UI"),
            ..Default::default()
        };

        let _ = eframe::run_native(
            "WebRender Debug UI",
            native_options,
            Box::new(|cc| {
                // Load fonts
                let mut fonts = egui::FontDefinitions::default();

                fonts.font_data.insert(
                    "FiraSans".to_owned(),
                    egui::FontData::from_static(include_bytes!("../../res/FiraSans-Regular.ttf")).into(),
                );
                fonts.font_data.insert(
                    "FiraCode".to_owned(),
                    egui::FontData::from_static(include_bytes!("../../res/FiraCode-Regular.ttf")).into(),
                );

                fonts.families.get_mut(&egui::FontFamily::Proportional).unwrap()
                    .insert(0, "FiraSans".to_owned());
                fonts.families.get_mut(&egui::FontFamily::Monospace).unwrap()
                    .insert(0, "FiraCode".to_owned());

                cc.egui_ctx.set_fonts(fonts);

                // Load layout settings if available
                if let Ok(_ini) = fs::read_to_string("default-layout.ini") {
                    // egui uses a different state persistence mechanism
                    // This would need to be adapted for egui's state system
                }

                Ok(Box::new(self))
            }),
        );
    }
}

impl eframe::App for Gui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Process pending events
        while let Ok(event) = self.event_receiver.try_recv() {
            match event {
                ApplicationEvent::RunCommand(cmd_name) => {
                    self.handle_command(&cmd_name);
                }
                ApplicationEvent::NetworkEvent(net_event) => {
                    self.handle_network_event(net_event);
                }
            }
        }

        textures::prepare(self, ctx);

        // Main menu bar
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Exit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("Help", |ui| {
                    ui.label("About");
                });

                // Connection status on the right
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (msg, text_color, bg_color) = if self.data_model.is_connected {
                        ("Connected", egui::Color32::from_rgb(48, 48, 48), egui::Color32::from_rgb(0, 255, 0))
                    } else {
                        ("Disconnected", egui::Color32::WHITE, egui::Color32::from_rgb(255, 0, 0))
                    };

                    // TODO: measure the text instead.
                    let mut area = ui.max_rect();
                    area.min.x = area.max.x - 85.0;
                    area.max.x += 10.0;

                    ui.painter().rect_filled(area, 0, bg_color);

                    let label = egui::Label::new(
                        egui::RichText::new(msg).color(text_color)
                    );
                    ui.add(label).on_hover_text("Connection status");
                });
            });
        });

        let mut ui_tiles = self.ui_tiles.take().unwrap();
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.spacing_mut().window_margin = egui::Margin::ZERO;
            ui_tiles.ui(self, ui);
        });
        self.ui_tiles = Some(ui_tiles);
    }
}

impl Gui {
    fn handle_command(&mut self, cmd_name: &str) {
        match self.cmd_list.get_mut(cmd_name) {
            Some(cmd) => {
                let mut ctx = command::CommandContext::new(
                    BTreeMap::new(),
                    &mut self.net,
                );
                let output = cmd.run(&mut ctx);

                match output {
                    command::CommandOutput::Log(msg) => {
                        self.data_model.log.push(msg);
                    }
                    command::CommandOutput::Err(msg) => {
                        self.data_model.log.push(msg);
                    }
                    command::CommandOutput::TextDocument { title, content } => {
                        let title = format!("{} [id {}]", title, self.doc_id);
                        self.doc_id += 1;
                        self.data_model.preview_doc_index = Some(self.data_model.documents.len());
                        self.data_model.documents.push(
                            Document {
                                title,
                                kind: DocumentKind::Text {
                                    content,
                                }
                            }
                        );
                    }
                    command::CommandOutput::SerdeDocument { kind, ref content } => {
                        let title = format!("Compositor [id {}]", self.doc_id);
                        self.doc_id += 1;
                        self.data_model.preview_doc_index = Some(self.data_model.documents.len());

                        let kind = match kind.as_str() {
                            "composite-view" => {
                                let info = serde_json::from_str(content).unwrap();
                                DocumentKind::Compositor {
                                    info,
                                }
                            }
                            _ => {
                                unreachable!("unknown content");
                            }
                        };

                        self.data_model.documents.push(
                            Document {
                                title,
                                kind,
                            }
                        );
                    }
                    command::CommandOutput::Textures(textures) => {
                        textures::add_textures(self, textures);
                    }
                }
            }
            None => {
                self.data_model.log.push(
                    format!("Unknown command '{}'", cmd_name)
                );
            }
        }
    }

    fn handle_network_event(&mut self, event: net::NetworkEvent) {
        match event {
            net::NetworkEvent::Connected => {
                self.data_model.is_connected = true;
            }
            net::NetworkEvent::Disconnected => {
                self.data_model.is_connected = false;
            }
            net::NetworkEvent::Message(msg) => {
                match msg {
                    DebuggerMessage::SetDebugFlags(info) => {
                        self.data_model.debug_flags = info.flags;
                    }
                    DebuggerMessage::InitProfileCounters(info) => {
                        let selected_counters = [
                            "Frame building",
                            "Renderer",
                        ];

                        for counter in info.counters {
                            if selected_counters.contains(&counter.name.as_str()) {
                                println!("Add profile counter {:?}", counter.name);
                                self.data_model.profile_graphs.insert(
                                    counter.id,
                                    Graph::new(&counter.name, 512),
                                );
                            }
                        }
                    }
                    DebuggerMessage::UpdateFrameLog(info) => {
                        if let Some(updates) = info.profile_counters {
                            for counter in &updates {
                                if let Some(graph) = self.data_model
                                    .profile_graphs
                                    .get_mut(&counter.id) {
                                    graph.push(counter.value);
                                }
                            }
                        }
                        let frame_log = &mut self.data_model.frame_log;
                        if frame_log.frames.len() == frame_log.frames.capacity() {
                            frame_log.frames.pop_front();
                        }
                        frame_log.frames.push_back(LoggedFrame {
                            render_commands: info.render_commands.clone(),
                        });
                        frame_log.frames_end += 1;
                        let first = frame_log.first_frame_index();
                        let last = frame_log.last_frame_index();

                        let mut current = self.data_model.timeline.current_frame;
                        current = current.clamp(first, last);

                        // Make the current frame 'stick' to the last frame (that is, we were
                        // already viewing the last frame, and a new frame was just pushed)
                        if last == current + 1 {
                            current = last
                        }

                        self.data_model.timeline.current_frame = current;
                    }
                }
            }
        }
    }
}

impl Drop for Gui {
    fn drop(&mut self) {
        // Serialize and save UI state
        const MAX_HISTORY_SAVED: usize = 50;
        let hist_len = self.cmd_history.len();
        let range = if hist_len > MAX_HISTORY_SAVED {
            hist_len - MAX_HISTORY_SAVED..hist_len
        } else {
            0..hist_len
        };

        let save = GuiSavedState {
            version: GuiSavedState::VERSION,
            ui_tiles: self.ui_tiles.take().unwrap(),
            cmd_history: self.cmd_history.drain(range).collect(),
        };

        match serde_json::to_string(&save) {
            Ok(serialized) => {
                if let Err(e) = fs::File::create(config_path())
                    .and_then(|mut file| file.write_all(serialized.as_bytes()))
                {
                    eprintln!("Failed to save ui_tiles state: {}", e);
                }
            }
            Err(e) => {
                eprintln!("Failed to serialize ui_tiles: {}", e);
            }
        }
    }
}

pub enum DocumentKind {
    Text {
        content: String,
    },
    Compositor {
        info: CompositorDebugInfo,
    },
    Texture {
        content: DebuggerTextureContent,
        handle: Option<egui::TextureHandle>,
    }
}

pub struct Document {
    pub title: String,
    pub kind: DocumentKind,
}

fn do_documents_ui(app: &mut Gui, ui: &mut egui::Ui) {
    let width = ui.available_width();
    for (i, doc) in app.data_model.documents.iter().enumerate() {
        if let DocumentKind::Texture { .. } = doc.kind {
            // Handle textures separately below.
            continue;
        }

        let item = egui::Button::selectable(
            app.data_model.preview_doc_index == Some(i),
            &doc.title,
        ).min_size(egui::vec2(width, 20.0));

        if ui.add(item).clicked() {
            app.data_model.preview_doc_index = Some(i);
        }
    }

    textures::texture_list_ui(app, ui);
}

fn do_preview_ui(app: &mut Gui, ui: &mut egui::Ui) {
    if let Some(idx) = app.data_model.preview_doc_index {
        if idx >= app.data_model.documents.len() {
            app.data_model.preview_doc_index = None;
            return;
        }

        let doc = &app.data_model.documents[idx];

        match &doc.kind {
            DocumentKind::Text { content } => {
                egui::ScrollArea::both().show(ui, |ui| {
                    ui.label(egui::RichText::new(content).monospace());
                });
            }
            DocumentKind::Compositor { .. } => {
                // We need to handle compositor separately due to borrow checker
                if let Some(Document { kind: DocumentKind::Compositor { info }, .. }) =
                    app.data_model.documents.get_mut(idx) {
                    composite_view::ui(ui, info);
                }
            }
            DocumentKind::Texture { content, handle } => {
                if let Some(handle) = handle {
                    textures::texture_viewer_ui(ui, &content, &handle);
                }
            }
        }
    }
}


// TODO: It would be better to use confy::load/store instead of loading
// files manually but for some reason serialization of the ui tiles panics
// in confy.
fn config_path() -> std::path::PathBuf {
    confy::get_configuration_file_path("wr_debugger", None).unwrap()
}

#[derive(serde::Serialize, serde::Deserialize)]
struct GuiSavedState {
    version: u32,
    cmd_history: Vec<String>,
    ui_tiles: egui_tiles::Tree<Tool>,
}

impl GuiSavedState {
    /// Update this number to reset the configuration. This ensures that new
    /// panels are added.
    const VERSION: u32 = 2;
}
