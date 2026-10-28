/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Scene tree panel: shows the picture / primitive tree of the built scene
//! currently rendered by the WR instance. Hovering a primitive highlights it
//! in the rendered frame, clicking it shows its details in a side panel, each
//! primitive has a checkbox to disable it, and right-clicking a row opens a
//! menu of actions on the node.

use std::collections::HashSet;
use webrender_api::ColorF;
use webrender_api::debugger::{SceneDebugHighlightMode, SceneDebugNode, SceneDebugOverride, SceneDebugTree};
use super::Gui;

pub struct SceneTreeState {
    tree: Option<SceneDebugTree>,
    selection: Selection,
    status: String,
}

/// The part of the state that the tree widgets mutate while the tree itself
/// is borrowed.
struct Selection {
    /// The primitive currently pushed to WR as highlighted: the one whose row
    /// was hovered during the last frame of the UI.
    highlighted: Option<u32>,
    /// The primitive whose row is hovered during the current UI frame.
    hovered: Option<u32>,
    /// The primitive whose row was clicked, shown in the details panel.
    selected: Option<u32>,
    highlight_mode: SceneDebugHighlightMode,
    disabled: HashSet<u32>,
    /// Action picked in a row's context menu, applied once the whole tree has
    /// been laid out since it needs the tree while the rows only see nodes.
    pending: Option<Action>,
}

/// A node of the tree, as referred to by context menu actions.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Target {
    Prim(u32),
    Root(usize),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Action {
    /// Disable every primitive except the target, its ancestors (hiding them
    /// would hide the target too) and its descendants.
    Focus(Target),
    /// Disable the target's descendants, keeping the target itself.
    DisableChildren(Target),
    /// Enable the target and all of its descendants.
    EnableSubtree(Target),
    EnableAll,
}

impl SceneTreeState {
    pub fn new() -> Self {
        SceneTreeState {
            tree: None,
            selection: Selection {
                highlighted: None,
                hovered: None,
                selected: None,
                highlight_mode: SceneDebugHighlightMode::Replace,
                disabled: HashSet::new(),
                pending: None,
            },
            status: "Not loaded".to_string(),
        }
    }

    fn to_override(&self) -> SceneDebugOverride {
        let mut disabled: Vec<u32> = self.selection.disabled.iter().cloned().collect();
        disabled.sort_unstable();
        SceneDebugOverride {
            scene_generation: self.tree.as_ref().map_or(0, |tree| tree.scene_generation),
            highlighted: self.selection.highlighted,
            highlight_mode: self.selection.highlight_mode,
            disabled,
        }
    }
}

/// Fetch the scene tree from the WR instance, resetting the selection if the
/// scene generation changed.
fn refresh(app: &mut Gui) {
    let state = &mut app.data_model.scene_tree;
    match app.net.get_with_query("query", &[("type", "scene")]) {
        Ok(Some(body)) => match serde_json::from_str::<SceneDebugTree>(&body) {
            Ok(tree) => {
                let previous_generation = state.tree.as_ref().map(|tree| tree.scene_generation);
                let generation_changed = previous_generation != Some(tree.scene_generation);
                if generation_changed {
                    state.selection.highlighted = None;
                    state.selection.hovered = None;
                    state.selection.selected = None;
                    state.selection.disabled.clear();
                }
                state.status = format!(
                    "Scene generation {}, {} primitives",
                    tree.scene_generation,
                    tree.prim_count,
                );
                state.tree = Some(tree);
            }
            Err(err) => {
                state.status = format!("Malformed reply from WR: {}", err);
            }
        },
        Ok(None) => {
            state.status = "Empty reply from WR".to_string();
        }
        Err(err) => {
            state.status = err;
        }
    }
}

fn push_override(app: &mut Gui) {
    let debug_override = app.data_model.scene_tree.to_override();
    match app.net.post_with_content("scene-override", &debug_override) {
        Ok(Some(reply)) if reply == "ok" => {}
        Ok(Some(reply)) => {
            // WR rejected the override, most likely because the scene changed
            // under us. Reload the tree (which resets a stale selection) but
            // keep the error visible.
            refresh(app);
            app.data_model.scene_tree.status = reply;
        }
        Ok(None) => {
            app.data_model.scene_tree.status = "Empty reply from WR".to_string();
        }
        Err(err) => {
            app.data_model.scene_tree.status = err;
        }
    }
}

pub fn ui(app: &mut Gui, ui: &mut egui::Ui) {
    let mut changed = false;

    ui.horizontal(|ui| {
        if ui.button("Refresh").clicked() {
            // When the generation changed, WR already dropped the previous
            // override along with the old scene, so there is nothing to push.
            refresh(app);
        }

        let state = &mut app.data_model.scene_tree;
        let has_disabled = !state.selection.disabled.is_empty();
        if ui.add_enabled(has_disabled, egui::Button::new("Enable all")).clicked() {
            state.selection.disabled.clear();
            changed = true;
        }

        ui.separator();
        ui.label("Highlight hovered:");
        let mode = &mut state.selection.highlight_mode;
        changed |= ui.radio_value(mode, SceneDebugHighlightMode::Replace, "Replace")
            .on_hover_text("Draw an opaque pink quad in place of the primitive")
            .changed();
        changed |= ui.radio_value(mode, SceneDebugHighlightMode::Overlay, "Overlay")
            .on_hover_text("Outline the primitive in pink on top of the frame, keeping its content visible")
            .changed();
    });

    ui.label(&app.data_model.scene_tree.status);
    ui.separator();

    let state = &mut app.data_model.scene_tree;
    state.selection.hovered = None;
    let SceneTreeState { tree, selection, .. } = state;
    if let Some(tree) = tree {
        egui::SidePanel::right(ui.make_persistent_id("scene-details"))
            .resizable(true)
            .default_width(320.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                    details_ui(ui, tree, selection);
                });
            });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ScrollArea::both().auto_shrink(false).show(ui, |ui| {
                for (i, root) in tree.roots.iter().enumerate() {
                    changed |= node_ui(ui, root, i, selection);
                }
            });
        });

        if let Some(action) = selection.pending.take() {
            changed |= apply_action(tree, selection, action);
        }
    }

    // The highlight follows the hovered row. Only push when it actually moves
    // (including to nothing, when the pointer leaves the rows) since every
    // push makes WR rebuild a frame.
    if state.selection.hovered != state.selection.highlighted {
        state.selection.highlighted = state.selection.hovered;
        changed = true;
    }

    if changed {
        push_override(app);
    }
}

fn node_ui(
    ui: &mut egui::Ui,
    node: &SceneDebugNode,
    salt: usize,
    selection: &mut Selection,
) -> bool {
    let mut changed = false;

    let drawn = node.prim_index.is_none() || matches!(
        node.draw_state.as_str(),
        "Visible" | "PassThrough"
    );
    let disabled = node.prim_index.map_or(false, |index| selection.disabled.contains(&index));
    let highlighted = node.prim_index.is_some() && node.prim_index == selection.highlighted;

    let mut text_format = egui::TextFormat::default();
    text_format.font_id = egui::TextStyle::Body.resolve(ui.style());
    text_format.color = if highlighted {
        egui::Color32::from_rgb(255, 105, 180)
    } else if disabled || !drawn {
        ui.visuals().weak_text_color()
    } else {
        ui.visuals().text_color()
    };
    if disabled {
        text_format.strikethrough = egui::Stroke::new(1.0_f32, text_format.color);
    }

    let mut job = egui::text::LayoutJob::default();
    let mut label = node.kind.clone();
    if node.color.is_some() {
        label.push(' ');
    }
    job.append(&label, 0.0, text_format.clone());
    if let Some(color) = node.color {
        let mut text_format = egui::TextFormat::default();
        text_format.font_id = egui::TextStyle::Monospace.resolve(ui.style());
        append_colored_dot(ui, &mut job, color);
    }
    let text = egui::WidgetText::from(job);

    let target = match node.prim_index {
        Some(index) => Target::Prim(index),
        None => Target::Root(salt),
    };
    let context_menu = |ui: &mut egui::Ui, selection: &mut Selection| {
        if ui.button("Focus on this node")
            .on_hover_text("Disable every other primitive, except this node's ancestors and descendants")
            .clicked()
        {
            selection.pending = Some(Action::Focus(target));
            ui.close();
        }
        if !node.children.is_empty() {
            if ui.button("Disable children").clicked() {
                selection.pending = Some(Action::DisableChildren(target));
                ui.close();
            }
            if ui.button("Enable children").clicked() {
                selection.pending = Some(Action::EnableSubtree(target));
                ui.close();
            }
        }
        if ui.add_enabled(!selection.disabled.is_empty(), egui::Button::new("Enable all")).clicked() {
            selection.pending = Some(Action::EnableAll);
            ui.close();
        }
    };

    let row = |ui: &mut egui::Ui, selection: &mut Selection| -> bool {
        let mut changed = false;
        if let Some(index) = node.prim_index {
            let mut enabled = !disabled;
            if ui.checkbox(&mut enabled, "").on_hover_text("Render this primitive").changed() {
                if enabled {
                    selection.disabled.remove(&index);
                } else {
                    selection.disabled.insert(index);
                }
                changed = true;
            }
            let selected = selection.selected == Some(index);
            let response = ui.selectable_label(selected, text.clone());
            if response.hovered() {
                selection.hovered = Some(index);
            }
            if response.clicked() {
                selection.selected = if selected { None } else { Some(index) };
            }
            response.context_menu(|ui| context_menu(ui, selection));
        } else {
            let response = ui.add(egui::Label::new(text.clone()).sense(egui::Sense::click()));
            response.context_menu(|ui| context_menu(ui, selection));
        }
        changed
    };

    if node.children.is_empty() {
        ui.horizontal(|ui| {
            changed |= row(ui, selection);
        });
    } else {
        let id = ui.make_persistent_id(("scene-tree", node.prim_index, node.picture_index, salt));
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, node.prim_index.is_none())
            .show_header(ui, |ui| {
                changed |= row(ui, selection);
            })
            .body(|ui| {
                for (i, child) in node.children.iter().enumerate() {
                    changed |= node_ui(ui, child, i, selection);
                }
            });
    }

    changed
}

/// Apply a context menu action to the disabled set. Returns whether the set
/// changed.
fn apply_action(tree: &SceneDebugTree, selection: &mut Selection, action: Action) -> bool {
    let target = match action {
        Action::Focus(target) | Action::DisableChildren(target) | Action::EnableSubtree(target) => target,
        Action::EnableAll => {
            let was_empty = selection.disabled.is_empty();
            selection.disabled.clear();
            return !was_empty;
        }
    };

    let mut path = Vec::new();
    let found = match target {
        Target::Prim(index) => tree.roots.iter().any(|root| find_path(root, index, &mut path)),
        Target::Root(i) => tree.roots.get(i).map_or(false, |root| {
            path.push(root);
            true
        }),
    };
    if !found {
        return false;
    }
    let node = *path.last().unwrap();

    let mut subtree = Vec::new();
    for child in &node.children {
        collect_prims(child, &mut subtree);
    }

    let before = selection.disabled.clone();
    match action {
        Action::Focus(_) => {
            let mut all = Vec::new();
            for root in &tree.roots {
                collect_prims(root, &mut all);
            }
            selection.disabled = all.into_iter().collect();
            for ancestor in &path {
                if let Some(index) = ancestor.prim_index {
                    selection.disabled.remove(&index);
                }
            }
            for index in subtree {
                selection.disabled.remove(&index);
            }
        }
        Action::DisableChildren(_) => {
            selection.disabled.extend(subtree);
        }
        Action::EnableSubtree(_) => {
            if let Some(index) = node.prim_index {
                selection.disabled.remove(&index);
            }
            for index in subtree {
                selection.disabled.remove(&index);
            }
        }
        Action::EnableAll => unreachable!(),
    }
    selection.disabled != before
}

fn collect_prims(node: &SceneDebugNode, out: &mut Vec<u32>) {
    if let Some(index) = node.prim_index {
        out.push(index);
    }
    for child in &node.children {
        collect_prims(child, out);
    }
}

/// Locate the node for a primitive index. On success `path` holds the chain
/// of nodes from the root down to (and including) the node.
fn find_path<'a>(
    node: &'a SceneDebugNode,
    prim_index: u32,
    path: &mut Vec<&'a SceneDebugNode>,
) -> bool {
    path.push(node);
    if node.prim_index == Some(prim_index) {
        return true;
    }
    for child in &node.children {
        if find_path(child, prim_index, path) {
            return true;
        }
    }
    path.pop();
    false
}

fn to_color32(color: ColorF) -> egui::Color32 {
    let channel = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    egui::Color32::from_rgba_unmultiplied(
        channel(color.r),
        channel(color.g),
        channel(color.b),
        channel(color.a),
    )
}

/// Append a color swatch to a text layout: a filled square glyph in the color,
/// drawn over a neutral gray background so that dark or translucent colors
/// stay visible, followed by the alpha when it is not opaque.
fn append_colored_dot(
    ui: &egui::Ui,
    job: &mut egui::text::LayoutJob,
    color: ColorF,
) {
    let mut text_format = egui::TextFormat::default();
    text_format.font_id = egui::TextStyle::Monospace.resolve(ui.style());
    text_format.color = to_color32(color);
    text_format.strikethrough = egui::Stroke::NONE;
    job.append("\u{25CF}", 0.0, text_format);
}

fn node_title(node: &SceneDebugNode) -> String {
    match node.prim_index {
        Some(index) => format!("[{}] {}", index, node.kind),
        None => node.kind.clone(),
    }
}

fn format_rect<U>(rect: &webrender_api::euclid::Box2D<f32, U>) -> String {
    format!(
        "({:.2}, {:.2}) {:.2} x {:.2}",
        rect.min.x,
        rect.min.y,
        rect.width(),
        rect.height(),
    )
}

fn details_ui(ui: &mut egui::Ui, tree: &SceneDebugTree, selection: &mut Selection) {
    let Some(prim_index) = selection.selected else {
        ui.heading("Primitive");
        ui.separator();
        ui.weak("Click a primitive in the tree to inspect it.");
        return;
    };

    let mut path = Vec::new();
    if !tree.roots.iter().any(|root| find_path(root, prim_index, &mut path)) {
        ui.heading("Primitive");
        ui.separator();
        ui.weak(format!("Primitive {} is not in the current scene.", prim_index));
        return;
    }
    let node = *path.last().unwrap();

    ui.heading(&node.kind);
    ui.separator();

    let disabled = selection.disabled.contains(&prim_index);

    egui::Grid::new("scene-details-grid")
        .num_columns(2)
        .striped(true)
        .show(ui, |ui| {
            fn row(ui: &mut egui::Ui, name: &str, value: String) {
                ui.strong(name);
                ui.add(egui::Label::new(egui::RichText::new(value).monospace()).wrap());
                ui.end_row();
            }

            row(ui, "Index", prim_index.to_string());
            row(ui, "Local rect", format_rect(&node.local_rect));
            row(ui, "Device rect", match &node.device_rect {
                Some(rect) => format_rect(rect),
                None => "-".to_string(),
            });
            if let Some(color) = node.color {
                ui.strong("Color");
                let mut text_format = egui::TextFormat::default();
                text_format.font_id = egui::TextStyle::Monospace.resolve(ui.style());
                text_format.color = ui.visuals().text_color();
                let mut job = egui::text::LayoutJob::default();
                append_colored_dot(ui, &mut job, color);
                job.append(
                    &format!(" rgba({:.3}, {:.3}, {:.3}, {:.3})", color.r, color.g, color.b, color.a),
                    0.0,
                    text_format,
                );
                ui.add(egui::Label::new(job).wrap());
                ui.end_row();
            }
            if !node.detail.is_empty() {
                row(ui, "Detail", node.detail.clone());
            }
            if let Some(picture_index) = node.picture_index {
                row(ui, "Picture", picture_index.to_string());
            }
            row(ui, "Spatial node", node.spatial_node_index.to_string());
            row(ui, "Last frame", if node.draw_state.is_empty() {
                "-".to_string()
            } else {
                node.draw_state.clone()
            });
            row(ui, "Rendering", if disabled { "disabled".to_string() } else { "enabled".to_string() });
            if !node.children.is_empty() {
                row(ui, "Children", node.children.len().to_string());
            }
        });

    ui.add_space(8.0);
    ui.strong("Ancestors");
    ui.indent("scene-details-ancestors", |ui| {
        for ancestor in &path[..path.len() - 1] {
            match ancestor.prim_index {
                Some(index) => {
                    let response = ui.selectable_label(false, node_title(ancestor));
                    if response.hovered() {
                        selection.hovered = Some(index);
                    }
                    if response.clicked() {
                        selection.selected = Some(index);
                    }
                }
                None => {
                    ui.label(node_title(ancestor));
                }
            }
        }
    });

    if !node.children.is_empty() {
        ui.add_space(8.0);
        ui.strong("Children");
        ui.indent("scene-details-children", |ui| {
            for child in &node.children {
                let Some(index) = child.prim_index else { continue };
                let response = ui.selectable_label(false, node_title(child));
                if response.hovered() {
                    selection.hovered = Some(index);
                }
                if response.clicked() {
                    selection.selected = Some(index);
                }
            }
        });
    }
}
