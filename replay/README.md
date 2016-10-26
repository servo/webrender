# wr-replay
Replay tool for the [WebRender](https://github.com/servo/webrender).

Usage:
	- in the WebRender client (be it [servo]() or [wr-sample]()), enable the recording option: `RendererOptions::enable_recording` when initializing the renderer
	- run the client normally for a few frames, the frame dumps will be stored in `record` folder
	- run the replay with `cargo run <shader_dir> <replay_dir>`
