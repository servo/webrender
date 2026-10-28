/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Shader editor panel: fetches the GLSL source of one of the connected WR
//! instance's `.glsl` files, edits it in place, and pushes it back for
//! recompilation. Compile errors come back with a file and line and are
//! highlighted in the editor.

use std::collections::{HashMap, HashSet};
use webrender_api::debugger::{SetShaderSourceRequest, ShaderDiagnostic};
use webrender_api::debugger::{ShaderListReply, ShaderReloadReply, ShaderSourceReply};
use super::{Gui, format_shader_diagnostic};

const ERROR_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 105, 180);

/// One file open in the editor.
struct Buffer {
    /// Source as last fetched from WR, for Revert to go back to.
    fetched: String,
    /// Source as edited locally.
    edited: String,
}

impl Buffer {
    fn is_dirty(&self) -> bool {
        self.fetched != self.edited
    }
}

pub struct ShaderEditorState {
    list: Option<ShaderListReply>,
    /// File shown in the editor.
    selected: Option<String>,
    /// Edit buffers, keyed by file name, so switching files keeps the edits.
    buffers: HashMap<String, Buffer>,
    /// Substring filter applied to the file list.
    filter: String,
    /// Diagnostics from the last push, or streamed from a later frame.
    diagnostics: Vec<ShaderDiagnostic>,
    status: String,
}

impl ShaderEditorState {
    pub fn new() -> Self {
        ShaderEditorState {
            list: None,
            selected: None,
            buffers: HashMap::new(),
            filter: String::new(),
            diagnostics: Vec::new(),
            status: "Not loaded".to_string(),
        }
    }

    /// Record diagnostics streamed from a frame that failed to build a shader
    /// after a push had already been answered.
    pub fn push_streamed_diagnostics(&mut self, diagnostics: &[ShaderDiagnostic]) {
        self.diagnostics.extend_from_slice(diagnostics);
        self.status = format!(
            "{} error(s) reported while rendering",
            diagnostics.len(),
        );
    }
}

/// Fetch the list of shader files and variants.
fn refresh_list(app: &mut Gui) {
    match app.net.get_with_query("query", &[("type", "shaders")]) {
        Ok(Some(body)) => match serde_json::from_str::<ShaderListReply>(&body) {
            Ok(list) => {
                let state = &mut app.data_model.shaders;
                state.status = if list.supported {
                    format!("{} files, {} variants", list.files.len(), list.variants.len())
                } else {
                    "This instance cannot recompile shaders at runtime (SWGL)".to_string()
                };
                state.list = Some(list);
            }
            Err(err) => {
                app.data_model.shaders.status = format!("Malformed reply from WR: {}", err);
            }
        },
        Ok(None) => {
            app.data_model.shaders.status = "Empty reply from WR".to_string();
        }
        Err(err) => {
            app.data_model.shaders.status = err;
        }
    }
}

/// Fetch one file's source and open it in the editor, discarding any local
/// edits to it.
fn fetch_source(app: &mut Gui, name: &str) {
    match app.net.get_with_query("shader-source", &[("name", name)]) {
        Ok(Some(body)) => match serde_json::from_str::<ShaderSourceReply>(&body) {
            Ok(ShaderSourceReply::Source { name, source, is_override }) => {
                let state = &mut app.data_model.shaders;
                state.buffers.insert(
                    name.clone(),
                    Buffer {
                        fetched: source.clone(),
                        edited: source,
                    },
                );
                state.selected = Some(name.clone());
                state.status = if is_override {
                    format!("{}.glsl (overridden)", name)
                } else {
                    format!("{}.glsl", name)
                };
            }
            Ok(ShaderSourceReply::Expanded { .. }) => {
                app.data_model.shaders.status =
                    "Unexpected expanded source in reply".to_string();
            }
            Ok(ShaderSourceReply::Error(err)) => {
                app.data_model.shaders.status = err;
            }
            Err(err) => {
                app.data_model.shaders.status = format!("Malformed reply from WR: {}", err);
            }
        },
        Ok(None) => {
            app.data_model.shaders.status = "Empty reply from WR".to_string();
        }
        Err(err) => {
            app.data_model.shaders.status = err;
        }
    }
}

/// Push a source for `name`, or drop its override when `source` is `None`.
fn push_source(app: &mut Gui, name: String, source: Option<String>) {
    let request = SetShaderSourceRequest { name: name.clone(), source };

    let reply = match app.net.post_with_content("shader-source", &request) {
        Ok(Some(body)) => match serde_json::from_str::<ShaderReloadReply>(&body) {
            Ok(reply) => reply,
            Err(err) => {
                app.data_model.shaders.status = format!("Malformed reply from WR: {}", err);
                return;
            }
        },
        Ok(None) => {
            app.data_model.shaders.status = "Empty reply from WR".to_string();
            return;
        }
        Err(err) => {
            app.data_model.shaders.status = err;
            return;
        }
    };

    let state = &mut app.data_model.shaders;
    state.diagnostics.clear();

    match reply {
        ShaderReloadReply::Ok { recompiled } => {
            // WR now holds what the editor shows, so Revert should come back
            // to this rather than to whatever was fetched before.
            if let Some(buffer) = state.buffers.get_mut(&name) {
                buffer.fetched = buffer.edited.clone();
            }
            state.status = if recompiled == 0 {
                // Nothing was linked yet, which is the normal state for an
                // instance that precaches asynchronously. The override is
                // installed and applies when these shaders are first used.
                "Override installed; no linked variant used this source yet".to_string()
            } else {
                format!("Recompiled {} variant(s)", recompiled)
            };
            // The overridden flag in the file list has changed.
            refresh_list(app);
        }
        ShaderReloadReply::Errors(diagnostics) => {
            state.status = format!(
                "{} diagnostic(s); the instance kept its previous shaders",
                diagnostics.len(),
            );
            state.diagnostics = diagnostics;
        }
        ShaderReloadReply::Unsupported(msg) | ShaderReloadReply::Error(msg) => {
            state.status = msg;
        }
    }
}

/// Open the preprocessed source of one variant as a text document. This is the
/// text a driver log's line numbers refer to when the location could not be
/// resolved back to a `.glsl` file.
fn show_expanded(app: &mut Gui, name: &str, features: &[String]) {
    let features = features.join(",");
    let reply = app.net.get_with_query(
        "shader-source",
        &[("name", name), ("features", features.as_str())],
    );

    match reply {
        Ok(Some(body)) => match serde_json::from_str::<ShaderSourceReply>(&body) {
            Ok(ShaderSourceReply::Expanded { variant, vertex, fragment }) => {
                let content = format!(
                    "// ==== {} vertex shader ====\n{}\n\
                     // ==== {} fragment shader ====\n{}",
                    variant, vertex, variant, fragment,
                );
                app.add_text_document(format!("Expanded {}", variant), content);
            }
            Ok(ShaderSourceReply::Source { .. }) => {
                app.data_model.shaders.status = "Unexpected raw source in reply".to_string();
            }
            Ok(ShaderSourceReply::Error(err)) => {
                app.data_model.shaders.status = err;
            }
            Err(err) => {
                app.data_model.shaders.status = format!("Malformed reply from WR: {}", err);
            }
        },
        Ok(None) => {
            app.data_model.shaders.status = "Empty reply from WR".to_string();
        }
        Err(err) => {
            app.data_model.shaders.status = err;
        }
    }
}

pub fn ui(app: &mut Gui, ui: &mut egui::Ui) {
    if app.data_model.shaders.list.is_none() {
        refresh_list(app);
    }

    let mut pending_fetch = None;
    let mut pending_select = None;
    let mut pending_push = None;
    let mut pending_expanded = None;

    ui.horizontal(|ui| {
        if ui.button("Refresh").clicked() {
            refresh_list(app);
        }

        let state = &app.data_model.shaders;
        let selected = state.selected.clone();
        let dirty = selected
            .as_ref()
            .and_then(|name| state.buffers.get(name))
            .map_or(false, Buffer::is_dirty);
        let overridden = selected.as_ref().map_or(false, |name| {
            state.list.as_ref().map_or(false, |list| {
                list.files.iter().any(|file| &file.name == name && file.overridden)
            })
        });

        ui.separator();

        if ui
            .add_enabled(dirty, egui::Button::new("Apply"))
            .on_hover_text("Send the edited source to WR and recompile (Ctrl+Enter)")
            .clicked()
        {
            if let Some(name) = selected.clone() {
                let source = state.buffers[&name].edited.clone();
                pending_push = Some((name, Some(source)));
            }
        }

        if ui
            .add_enabled(dirty, egui::Button::new("Revert"))
            .on_hover_text("Discard local edits, back to the source WR reported")
            .clicked()
        {
            if let Some(name) = selected.clone() {
                pending_fetch = Some(name);
            }
        }

        if ui
            .add_enabled(overridden, egui::Button::new("Reset override"))
            .on_hover_text("Drop the override in WR and restore the built-in source")
            .clicked()
        {
            if let Some(name) = selected.clone() {
                pending_push = Some((name, None));
            }
        }
    });

    ui.label(&app.data_model.shaders.status);
    ui.separator();

    egui::SidePanel::left(ui.make_persistent_id("shader-files"))
        .resizable(true)
        .default_width(240.0)
        .show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Filter:");
                ui.text_edit_singleline(&mut app.data_model.shaders.filter);
            });

            let state = &app.data_model.shaders;
            let filter = state.filter.to_lowercase();
            let selected = state.selected.clone();

            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .show(ui, |ui| {
                    let Some(list) = &state.list else { return };

                    for file in &list.files {
                        if !filter.is_empty() && !file.name.to_lowercase().contains(&filter) {
                            continue;
                        }

                        let is_selected = selected.as_deref() == Some(file.name.as_str());
                        let dirty = state
                            .buffers
                            .get(&file.name)
                            .map_or(false, Buffer::is_dirty);

                        let mut label = file.name.clone();
                        if file.overridden {
                            label.push_str(" [override]");
                        }
                        if dirty {
                            label.push('*');
                        }

                        let mut text = egui::RichText::new(label);
                        if file.overridden || dirty {
                            text = text.color(ERROR_COLOR);
                        }

                        if ui.selectable_label(is_selected, text).clicked() {
                            // A file that is already buffered keeps its local
                            // edits; only a file seen for the first time is
                            // fetched.
                            if state.buffers.contains_key(&file.name) {
                                pending_select = Some(file.name.clone());
                            } else {
                                pending_fetch = Some(file.name.clone());
                            }
                        }
                    }

                    ui.separator();
                    ui.label(egui::RichText::new("Variants").strong());

                    let Some(name) = &selected else { return };
                    for variant in &list.variants {
                        if &variant.base_filename != name {
                            continue;
                        }

                        let features = if variant.features.is_empty() {
                            "(no features)".to_string()
                        } else {
                            variant.features.join(", ")
                        };
                        let marker = if variant.compiled { "compiled" } else { "not built" };

                        ui.horizontal(|ui| {
                            let mut text = egui::RichText::new(format!("{} - {}", features, marker));
                            if !variant.compiled {
                                text = text.color(ui.visuals().weak_text_color());
                            }
                            ui.label(text);
                            if ui
                                .small_button("expanded")
                                .on_hover_text("Show the preprocessed source of this variant")
                                .clicked()
                            {
                                pending_expanded =
                                    Some((name.clone(), variant.features.clone()));
                            }
                        });
                    }
                });
        });

    egui::TopBottomPanel::bottom(ui.make_persistent_id("shader-diagnostics"))
        .resizable(true)
        .default_height(140.0)
        .show_inside(ui, |ui| {
            let state = &app.data_model.shaders;
            if state.diagnostics.is_empty() {
                ui.label(egui::RichText::new("No diagnostics").weak());
                return;
            }

            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .show(ui, |ui| {
                    for diagnostic in &state.diagnostics {
                        ui.label(
                            egui::RichText::new(format_shader_diagnostic(diagnostic))
                                .monospace()
                                .color(ERROR_COLOR),
                        );

                        // Quote the offending line, so the error can be read
                        // without hunting for it in the editor.
                        if let (Some(file), Some(line)) = (&diagnostic.file, diagnostic.line) {
                            let stem = file.trim_end_matches(".glsl");
                            if let Some(buffer) = state.buffers.get(stem) {
                                if let Some(text) = buffer.edited.lines().nth(line as usize - 1) {
                                    ui.label(
                                        egui::RichText::new(format!("  {:>5} | {}", line, text))
                                            .monospace()
                                            .weak(),
                                    );
                                }
                            }
                        }
                    }
                });
        });

    egui::CentralPanel::default().show_inside(ui, |ui| {
        let state = &mut app.data_model.shaders;
        let Some(name) = state.selected.clone() else {
            ui.label("Select a shader file");
            return;
        };

        // Lines of the open file that a diagnostic points at.
        let error_lines: HashSet<u32> = state
            .diagnostics
            .iter()
            .filter(|diagnostic| {
                diagnostic
                    .file
                    .as_deref()
                    .map_or(false, |file| file.trim_end_matches(".glsl") == name)
            })
            .filter_map(|diagnostic| diagnostic.line)
            .collect();

        let Some(buffer) = state.buffers.get_mut(&name) else {
            ui.label("Select a shader file");
            return;
        };

        let mut layouter = |ui: &egui::Ui, text: &dyn egui::TextBuffer, wrap_width: f32| {
            let font_id = egui::TextStyle::Monospace.resolve(ui.style());
            let mut job = egui::text::LayoutJob::default();
            job.wrap.max_width = wrap_width;

            // The job must reproduce the buffer exactly, or the text cursor
            // would land on the wrong character.
            for (index, line) in text.as_str().split_inclusive('\n').enumerate() {
                let mut format =
                    egui::TextFormat::simple(font_id.clone(), ui.visuals().text_color());
                if error_lines.contains(&(index as u32 + 1)) {
                    format.color = ERROR_COLOR;
                    format.underline = egui::Stroke::new(1.0_f32, ERROR_COLOR);
                }
                job.append(line, 0.0, format);
            }

            ui.fonts_mut(|fonts| fonts.layout_job(job))
        };

        egui::ScrollArea::vertical()
            .auto_shrink(false)
            .show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut buffer.edited)
                        .code_editor()
                        .desired_width(f32::INFINITY)
                        .layouter(&mut layouter),
                );
            });

        if buffer.is_dirty()
            && ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Enter))
        {
            pending_push = Some((name, Some(buffer.edited.clone())));
        }
    });

    if let Some(name) = pending_select {
        app.data_model.shaders.selected = Some(name);
    }
    if let Some(name) = pending_fetch {
        fetch_source(app, &name);
    }
    if let Some((name, features)) = pending_expanded {
        show_expanded(app, &name, &features);
    }
    if let Some((name, source)) = pending_push {
        push_source(app, name, source);
    }
}
