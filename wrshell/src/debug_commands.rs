/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::command::{Command, CommandList, CommandDescriptor};
use crate::command::{CommandContext, CommandOutput};
use webrender_api::DebugFlags;
use webrender_api::debugger::DebuggerTextureContent;

// Implementation of a basic set of debug commands to demonstrate functionality

// Register the debug commands in this source file
pub fn register(cmd_list: &mut CommandList) {
    cmd_list.register_command(Box::new(PingCommand));
    cmd_list.register_command(Box::new(GenerateFrameCommand));
    cmd_list.register_command(Box::new(ToggleProfilerCommand));
    cmd_list.register_command(Box::new(GetSpatialTreeCommand));
    cmd_list.register_command(Box::new(GetCompositeConfigCommand));
    cmd_list.register_command(Box::new(GetCompositeViewCommand));
    cmd_list.register_command(Box::new(GetTexturesCommand { kind: None }));
    cmd_list.register_command(Box::new(GetTexturesCommand { kind: Some("atlas") }));
    cmd_list.register_command(Box::new(GetTexturesCommand { kind: Some("standalone") }));
    cmd_list.register_command(Box::new(GetTexturesCommand { kind: Some("render-target") }));
    cmd_list.register_command(Box::new(GetTexturesCommand { kind: Some("tile") }));
}

struct PingCommand;
struct GenerateFrameCommand;
struct ToggleProfilerCommand;
struct GetSpatialTreeCommand;
struct GetCompositeConfigCommand;
struct GetCompositeViewCommand;
struct GetTexturesCommand { kind: Option<&'static str> }

impl Command for PingCommand {
    fn descriptor(&self) -> CommandDescriptor {
        CommandDescriptor {
            name: "ping",
            help: "Test connection to specified host",
            ..Default::default()
        }
    }

    fn run(
        &mut self,
        ctx: &mut CommandContext,
    ) -> CommandOutput {
        match ctx.net.get("ping") {
            Ok(output) => {
                CommandOutput::Log(output.expect("empty response"))
            }
            Err(err) => {
                CommandOutput::Err(err)
            }
        }
    }
}

impl Command for GenerateFrameCommand {
    fn descriptor(&self) -> CommandDescriptor {
        CommandDescriptor {
            name: "generate-frame",
            help: "Generate and render one frame",
            alias: Some("f"),
            ..Default::default()
        }
    }

    fn run(
        &mut self,
        ctx: &mut CommandContext,
    ) -> CommandOutput {
        match ctx.net.post("generate-frame") {
            Ok(..) => {
                CommandOutput::Log("ok".to_string())
            }
            Err(err) => {
                CommandOutput::Err(err)
            }
        }
    }
}

impl Command for ToggleProfilerCommand {
    fn descriptor(&self) -> CommandDescriptor {
        CommandDescriptor {
            name: "toggle-profiler",
            help: "Toggle the on-screen profiler overlay",
            alias: Some("p"),
            ..Default::default()
        }
    }

    fn run(
        &mut self,
        ctx: &mut CommandContext,
    ) -> CommandOutput {
        match ctx.net.get("debug-flags") {
            Ok(flags_string) => {
                let mut flags: DebugFlags = serde_json::from_str(flags_string.as_ref().unwrap()).unwrap();
                flags ^= DebugFlags::PROFILER_DBG;

                match ctx.net.post_with_content("debug-flags", &flags) {
                    Ok(output) => {
                        CommandOutput::Log(output.expect("empty response"))
                    }
                    Err(err) => {
                        CommandOutput::Err(err)
                    }
                }
            }
            Err(err) => {
                CommandOutput::Err(err)
            }
        }
    }
}

impl Command for GetSpatialTreeCommand {
    fn descriptor(&self) -> CommandDescriptor {
        CommandDescriptor {
            name: "get-spatial-tree",
            help: "Print the current spatial tree to console",
            ..Default::default()
        }
    }

    fn run(
        &mut self,
        ctx: &mut CommandContext,
    ) -> CommandOutput {
        match ctx.net.get_with_query(
            "query",
            &[("type", "spatial-tree")],
        ) {
            Ok(output) => {
                CommandOutput::TextDocument {
                    title: "Spatial Tree".to_string(),
                    content: output.expect("empty response"),
                }
            }
            Err(err) => {
                CommandOutput::Err(err)
            }
        }
    }
}

impl Command for GetCompositeConfigCommand {
    fn descriptor(&self) -> CommandDescriptor {
        CommandDescriptor {
            name: "get-composite-cfg",
            help: "Print the current compositing config to the console",
            ..Default::default()
        }
    }

    fn run(
        &mut self,
        ctx: &mut CommandContext,
    ) -> CommandOutput {
        match ctx.net.get_with_query(
            "query",
            &[("type", "composite-config")],
        ) {
            Ok(output) => {
                CommandOutput::TextDocument {
                    title: "Composite Config".into(),
                    content: output.expect("empty response"),
                }
            }
            Err(err) => {
                CommandOutput::Err(err)
            }
        }
    }
}

impl Command for GetCompositeViewCommand {
    fn descriptor(&self) -> CommandDescriptor {
        CommandDescriptor {
            name: "get-composite-view",
            help: "Print the current compositing config to the console",
            ..Default::default()
        }
    }

    fn run(
        &mut self,
        ctx: &mut CommandContext,
    ) -> CommandOutput {
        match ctx.net.get_with_query(
            "query",
            &[("type", "composite-view")],
        ) {
            Ok(output) => {
                CommandOutput::SerdeDocument {
                    kind: "composite-view".into(),
                    content: output.expect("empty response"),
                }
            }
            Err(err) => {
                CommandOutput::Err(err)
            }
        }
    }
}

impl Command for GetTexturesCommand {
    fn descriptor(&self) -> CommandDescriptor {
        CommandDescriptor {
            name: match self.kind {
                Some("atlas") => "get-atlas-textures",
                Some("standalone") => "get-standalone-textures",
                Some("render-target") => "get-target-textures",
                Some("tile") => "get-tile-textures",
                _ => "get-textures",
            },
            help: "Fetch all gpu textures",
            ..Default::default()
        }
    }

    fn run(
        &mut self,
        ctx: &mut CommandContext,
    ) -> CommandOutput {
        let kind = match self.kind {
            Some("atlas") => "atlas-textures",
            Some("standalone") => "standalone-textures",
            Some("render-target") => "target-textures",
            Some("tile") => "tile-textures",
            _ => "textures",
        };
        match ctx.net.get_with_query(
            "query",
            &[("type", kind)],
        ) {
            Ok(output) => {
                let mut textures: Vec<DebuggerTextureContent> = serde_json::from_str(
                    output.unwrap().as_str()
                ).unwrap();
                textures.sort_by(|a, b| a.name.cmp(&b.name));
                CommandOutput::Textures(textures)
            }
            Err(err) => {
                CommandOutput::Err(err)
            }
        }
    }
}
