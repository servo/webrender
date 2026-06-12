/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

mod cli;
mod command;
mod debug_commands;
mod gui;
mod net;
mod script_commands;
mod wrench;

use argh::FromArgs;
use std::str;

// Main entry point for the WR debugger, which defaults to CLI mode but can also run in GUI mode.

#[derive(Debug, Copy, Clone)]
enum Mode {
    Repl,
    Gui,
}

impl str::FromStr for Mode {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Mode, &'static str> {
        match s {
            "repl" => Ok(Mode::Repl),
            "gui" => Ok(Mode::Gui),
            _ => Err("Invalid mode"),
        }
    }
}

#[derive(FromArgs, Debug)]
/// WebRender Shell arguments
struct Args {
    #[argh(positional)]
    #[argh(default = "Mode::Repl")]
    /// mode to run the shell in
    mode: Mode,

    #[argh(option, short = 'h')]
    #[argh(default = "\"localhost:3583\".into()")]
    /// host to connect to
    host: String,
}

fn main() {
    let args: Args = argh::from_env();

    let mut cmd_list = command::CommandList::new();
    debug_commands::register(&mut cmd_list);
    script_commands::register(&mut cmd_list);

    match args.mode {
        Mode::Repl => {
            let cli = cli::Cli::new(&args.host, cmd_list);
            cli.run();
        }
        Mode::Gui => {
            let gui = gui::Gui::new(&args.host, cmd_list);
            gui.run();
        }
    }
}
