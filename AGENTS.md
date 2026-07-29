# WebRender module instructions

WebRender is the GPU-based 2D rendering engine used by Firefox, written in Rust.
It is a standalone Cargo workspace vendored into the Firefox repository.

## Layout
- `webrender/` — the renderer crate (scene building, frame building, batching,
  clip/spatial trees, GPU backend).
- `webrender_api/` — the serializable display-list API shared with Gecko.
- `wrench/` — standalone test/replay harness (reftests, captures, YAML/RON).
- `swgl/` — software GL backend.
- `wr_glyph_rasterizer/`, `peek-poke`, `wr_malloc_size_of`, `webrender_build` —
  supporting crates.

## Testing
`wrench` is the standalone test/replay harness. Firefox CI reftests are golden
when they disagree with wrench semantics.

### Running wrench reftests
Reftests must run **headless** for deterministic PNGs (OSMesa + llvmpipe at
scale_factor 1.0) that match CI. This needs the `dev-wrench` distrobox image —
the host glibc/GL stack will not produce matching pixels. Plain host builds
(`cargo build -p wrench`) and hardware `wrench show`/`wrench png` runs do NOT
need the distrobox; use the host for iteration, the distrobox only for the
reftest suite and reference-PNG generation.

Set the box up once from `ci-scripts/` (defined by `ci-scripts/distrobox.ini`):

```
cd gfx/wr/ci-scripts && distrobox assemble create
```

Always go through `script/headless.py` (never `wrench` directly — it panics on
GL init) and through the distrobox. Run from the repo root:

```
distrobox enter dev-wrench -- bash -c \
  "cd gfx/wr/wrench && python3 script/headless.py <subcommand> <args>"
```

The script builds the release `wrench` binary itself, so do not run a separate
`cargo build` first. Useful subcommands (passed through to wrench):
- `reftest [PATH]` — run the reftest list, a directory, or a single yaml. Paths
  are relative to `gfx/wr/wrench/`,
  e.g. `reftests/clip/clip-between-picclip-and-lca.yaml`.
- `png <yaml> <out.png>` — render one yaml to PNG at the reference scale factor;
  use this (not `wrench png`) when generating/comparing reference images.
- `test_invalidation` — asserts tile-cache invalidation and compositor clip
  promotion behaviour that pixel reftests cannot catch (source
  `wrench/src/test_invalidation.rs`). Exit 0 means all pass.

Environment knobs: `OPTIMIZED=0` (debug build), `WRENCH_HEADLESS_TARGET=<path>`
(reuse a prebuilt target dir), `CARGOFLAGS=...`, `DEBUGGER=rr|gdb|rust-gdb|cgdb`.

Redirect output to `artifacts/` rather than piping through `tail`/`grep`, so the
full log stays available:

```
distrobox enter dev-wrench -- bash -c \
  "cd gfx/wr/wrench && python3 script/headless.py reftest" \
  > artifacts/wrench-reftest.log 2>&1
```

## Screenshots of real WebRender output
Automated screenshots do **not** show WebRender output by default. WebDriver,
Marionette and the DevTools/MCP screenshot tools all re-render the document
through the software `drawSnapshot` (`CrossProcessPaint`) path, which bypasses
the compositor — a capture can look correct while the real on-screen output is
wrong. Never conclude a rendering bug is fixed, or fails to reproduce, from a
default screenshot.

Set `remote.screenshot.use_readback` to `true` to capture the composited
framebuffer instead. See `gfx/docs/DebuggingWebRenderScreenshots.md` for the
mechanism and the full list of limitations, and `remote/doc/Prefs.md` for the
pref itself. The limitations that matter most in practice:

- Every capture degrades to the content area of the foreground tab;
  full-document, element and clip-region captures all return the viewport.
- The session must be in **content** scope. In chrome scope there is no content
  area to read back and the capture silently falls back to `drawSnapshot`.
- Unsupported on macOS: the parent process only honours the GPU process's
  readback request in automation, and otherwise `IPC_FAIL`s (killing the GPU
  process, or crashing the parent in a debug build).
- **Headless still does not show hardware WebRender.** Headless force-disables
  `HW_COMPOSITING` (`gfxPlatform::InitCompositorAccelerationPrefs`), so a
  headless readback captures software WebRender. Run on a real display to
  reproduce hardware-specific artifacts.
- Readback flushes scene builds and forces a fresh frame
  (`FlushFrameGeneration(RenderReasons::SNAPSHOT)`) before reading pixels, so it
  is a poor tool for artifacts that only appear through partial present or
  damage tracking — capturing repaints what you are trying to observe.
- The capture reflects whatever is currently composited, so apply zoom, scroll
  or DOM changes *before* capturing.

Because every readback returns the whole viewport, crop to the region of
interest before looking at the image — a full viewport costs far more tokens
than the few hundred pixels that actually matter. The PNG is in device pixels
(content area scaled by the chrome window's `devicePixelRatio`, see
`remote/shared/Capture.sys.mjs`), so coordinates read out of a WR display list
or capture already have DPR baked in and can be used directly; only a
`getBoundingClientRect()` rect is in CSS pixels and needs scaling first. Take
the rect from one of those rather than eyeballing it, then:

```
magick in.png -crop 30x30+30+30 out.png           # WxH+X+Y, out of place
magick mogrify -crop 30x30+30+30 a.png            # in place
```
