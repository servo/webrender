/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use webrender_api::DebugFlags;
use super::Gui;

pub fn ui(app: &mut Gui, ui: &mut egui::Ui) {
    fn debug_flag(app: &mut Gui, ui: &mut egui::Ui, flag: DebugFlags, label: &str) -> bool {
        let initial = app.data_model.debug_flags;
        let mut checked = initial.contains(flag);
        let changed = ui.checkbox(&mut checked, label).changed();
        app.data_model.debug_flags.set(flag, checked);

        changed
    }

    let push_flags = debug_flag(app, ui, DebugFlags::FORCE_PICTURE_INVALIDATION, "Force invalidation")
        | debug_flag(app, ui, DebugFlags::PROFILER_DBG, "Pofiler")
        | debug_flag(app, ui, DebugFlags::RENDER_TARGET_DBG, "Render targets")
        | debug_flag(app, ui, DebugFlags::TEXTURE_CACHE_DBG, "Texture cache")
        | debug_flag(app, ui, DebugFlags::PICTURE_CACHING_DBG, "Picture cache")
        | debug_flag(app, ui, DebugFlags::PICTURE_BORDERS, "Picture borders")
        | debug_flag(app, ui, DebugFlags::HIGHLIGHT_BACKDROP_FILTERS, "Highlight backdrop filters")
        | debug_flag(app, ui, DebugFlags::DISABLE_ALPHA_PASS, "Skip alpha pass")
        | debug_flag(app, ui, DebugFlags::DISABLE_OPAQUE_PASS, "Skip opaque pass")
        | debug_flag(app, ui, DebugFlags::SHOW_OVERDRAW, "Show overdraw");

    if push_flags {
        app.net.post_with_content("debug-flags", &app.data_model.debug_flags).ok();
    }
}
