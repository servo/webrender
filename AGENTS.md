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
