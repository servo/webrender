## WRShell - Remote debugging for WebRender

### Overview

WRShell is a client for accessing the remote debugging functionality in WebRender.

It doesn't contain many features yet, but the intent is to provide a foundation that is easy to add new debugging and profiling functionality to over time, as needed.

WRShell can run in either CLI or GUI mode. In CLI mode, it runs inside a terminal, providing a simple way to query and adjust parameters of a connected WR instance. In GUI mode, a similar interface is provided to run commands, along with a richer way to view and inspect WR state, as well as viewing realtime data streamed from a WR instance, such as profile counters and graphs.

### Architecture

When a WR instance starts up with remote debugger enabled, it opens a HTTP server listener on socket 3583.

WRShell connects to a running WR instance (e.g. Firefox or Wrench) via HTTP for simple operations, and can optionally open a WebSocket stream for streaming realtime updates, such as profile counters and changes to WR state.

WRShell can connect to either a local instance of WR (default), or an instance running on the same network, such as an Android device, by specifying a target IP address.

### Current Features

 * Query connected status
 * Get or set debug flags
 * Query and display the current spatial tree (basic only, needs additional functionality)
 * Display profile counter graphs (basic only, needs to be extended)

### Building

#### WRShell

Ensure you have `libSDL3` development packages installed on your host machine.

```
cd wrshell
cargo build --release
```

#### Enabling remote debugger in WR

* Enable the `debugger` cargo feature (e.g. in `webrender_bindings` or `wrench`).
* Ensure `enable_debugger` is `true` in the `WebrenderOptions` struct when WR is created.
* If building in Firefox, you will need to locally vendor the remote debugger dependencies - `./mach vendor rust`. We may vendor these in m-c in future.
* When running an application, you should see `Start debug server on [host]` in the output log when starting if the debugger is enabled.

### Running

To run in CLI mode (defaults to local host connection if no IP specified):

`cargo run --release [target IP]`

To run in GUI mode (defaults to local host connection if no IP specified):

`cargo run --release -- gui [target IP]`

### Extending

### Adding commands

First, add functionality to WR for a new operation by adding a new endpoint in the `src/webrender/debugger.rs` in the `start` function.

Next, add a new command to the shell in `wrshell/src/debug_commands.rs`, and register it in `debug_commands::register`.
