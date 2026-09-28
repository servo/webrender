/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Repeatedly builds and/or renders the frame described by a yaml file
//! or a capture, as fast as possible, and reports timings.
//!
//! Each iteration asks the render backend to build a new frame with
//! `DebugCommand::GenerateFrame`, which invalidates every picture cache tile,
//! so that iteration N+1 redoes the work of iteration N instead of reusing it.
//! In render-only mode, the frame is built once and each iteration redraws
//! all of it, including picture cache tiles and texture cache targets.

use crate::wrench::{Wrench, WrenchThing};
use crate::yaml_frame_reader::YamlFrameReader;
use crate::{NotifierEvent, WindowWrapper};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};
use webrender::api::DebugFlags;
use webrender::render_api::DebugCommand;
use webrender::ProfileCounterValue;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BenchMode {
    /// Only build frames. The renderer consumes them without presenting.
    FrameBuild,
    /// Resend the display lists every iteration, measuring scene building and
    /// frame building (yaml only).
    SceneBuild,
    /// Build the frame once and render it every iteration.
    RenderOnly,
}

pub struct BenchOptions {
    pub mode: BenchMode,
    /// Also render and present each frame.
    pub render: bool,
    /// Wait for the GPU to finish each rendered frame (glFinish).
    pub gpu_sync: bool,
    /// Measure GPU time with timer queries.
    pub gpu_queries: bool,
    pub iterations: usize,
    pub warmup: usize,
    /// Print statistics for every profiler counter recorded while building frames.
    pub all_counters: bool,
    pub csv: Option<PathBuf>,
    pub save: Option<PathBuf>,
    pub baseline: Option<PathBuf>,
}

pub enum BenchInput {
    Yaml(PathBuf),
    /// A capture directory, and for capture sequences, the (scene, frame) to load.
    Capture(PathBuf, Option<(u32, u32)>),
}

impl BenchInput {
    pub fn from_path(path: PathBuf) -> Self {
        if path.join("scenes").is_dir() {
            BenchInput::Capture(path, Some((1, 1)))
        } else if path.is_dir() {
            BenchInput::Capture(path, None)
        } else {
            BenchInput::Yaml(path)
        }
    }

    fn path(&self) -> &Path {
        match self {
            BenchInput::Yaml(path) | BenchInput::Capture(path, _) => path,
        }
    }
}

const FRAME_BUILD_WALL: &str = "Frame build (wall)";
const SCENE_BUILD_WALL: &str = "Scene + frame build (wall)";
const RENDERER_WALL: &str = "Renderer (wall)";
const CONSUME_WALL: &str = "Renderer update (wall)";
const RENDERER_CPU: &str = "Renderer";
const GPU: &str = "GPU (timer queries)";
const TOTAL_WALL: &str = "Total (wall)";
const DRAW_CALLS: &str = "Draw calls";

/// Frame building profiler counters shown by default. The name must match the
/// counter names in webrender's profiler.
const DEFAULT_COUNTERS: &[&str] = &[
    "Scene building",
    "Frame building",
    "Visibility",
    "Prepare",
    "Batching",
];

/// Samples of a metric, one per measured iteration.
#[derive(Default, Serialize, Deserialize)]
struct Series {
    name: String,
    unit: String,
    samples: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
struct Summary {
    name: String,
    unit: String,
    mean: f64,
    median: f64,
    min: f64,
    max: f64,
    stddev: f64,
}

impl Series {
    fn summarize(&self) -> Summary {
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len().max(1) as f64;
        let mean = sorted.iter().sum::<f64>() / n;
        let variance = sorted.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        let median = if sorted.is_empty() {
            0.0
        } else if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
        } else {
            sorted[sorted.len() / 2]
        };
        Summary {
            name: self.name.clone(),
            unit: self.unit.clone(),
            mean,
            median,
            min: sorted.first().cloned().unwrap_or(0.0),
            max: sorted.last().cloned().unwrap_or(0.0),
            stddev: variance.sqrt(),
        }
    }
}

/// The measured metrics, in the order they were first recorded.
#[derive(Default)]
struct Metrics {
    series: Vec<Series>,
    index: HashMap<String, usize>,
}

impl Metrics {
    fn add(&mut self, name: &str, unit: &str, value: f64) {
        let idx = match self.index.get(name) {
            Some(idx) => *idx,
            None => {
                self.series.push(Series {
                    name: name.to_string(),
                    unit: unit.to_string(),
                    samples: Vec::new(),
                });
                self.index.insert(name.to_string(), self.series.len() - 1);
                self.series.len() - 1
            }
        };
        self.series[idx].samples.push(value);
    }

    fn get(&self, name: &str) -> Option<&Series> {
        self.index.get(name).map(|idx| &self.series[*idx])
    }
}

#[derive(Serialize, Deserialize)]
struct SavedResults {
    input: String,
    iterations: usize,
    summaries: Vec<Summary>,
    series: Vec<Series>,
}

pub struct BenchHarness<'a> {
    wrench: &'a mut Wrench,
    window: &'a mut WindowWrapper,
    rx: &'a Receiver<NotifierEvent>,
    options: BenchOptions,
    yaml_reader: Option<YamlFrameReader>,
    num_documents: usize,
}

impl<'a> BenchHarness<'a> {
    pub fn new(
        wrench: &'a mut Wrench,
        window: &'a mut WindowWrapper,
        rx: &'a Receiver<NotifierEvent>,
        options: BenchOptions,
    ) -> Self {
        BenchHarness {
            wrench,
            window,
            rx,
            options,
            yaml_reader: None,
            num_documents: 1,
        }
    }

    pub fn run(mut self, input: BenchInput) {
        if self.options.mode == BenchMode::SceneBuild && !matches!(input, BenchInput::Yaml(..)) {
            panic!("--scene-build requires a yaml input");
        }

        self.load(&input);
        self.settle();

        let mut flags = self.wrench.renderer.get_debug_flags();
        flags.set(DebugFlags::GPU_TIME_QUERIES, self.options.gpu_queries);
        flags.set(DebugFlags::SKIP_RENDERING, !self.options.render);
        if flags != self.wrench.renderer.get_debug_flags() {
            self.wrench.api.send_debug_cmd(DebugCommand::SetFlags(flags));
            self.settle();
        }

        let mut metrics = Metrics::default();
        let total = self.options.warmup + self.options.iterations;
        let bench_start = Instant::now();
        for i in 0..total {
            let measuring = i >= self.options.warmup;
            if i == self.options.warmup {
                self.wrench.renderer.take_frame_build_profiles();
                self.wrench.get_frame_profiles();
            }
            let mut iteration = Metrics::default();
            self.iteration(&mut iteration);
            if measuring {
                for series in iteration.series {
                    for sample in series.samples {
                        metrics.add(&series.name, &series.unit, sample);
                    }
                }
            }
        }
        let elapsed = bench_start.elapsed();

        self.report(&input, &metrics, elapsed);
    }

    fn load(&mut self, input: &BenchInput) {
        match input {
            BenchInput::Yaml(path) => {
                let mut reader = YamlFrameReader::new(path);
                if self.options.mode == BenchMode::SceneBuild {
                    self.wrench.rebuild_display_lists = true;
                }
                reader.do_frame(self.wrench);
                self.yaml_reader = Some(reader);
            }
            BenchInput::Capture(path, ids) => {
                let documents = self.wrench.api.load_capture(path.clone(), *ids);
                assert!(!documents.is_empty(), "No document in capture {:?}", path);
                self.num_documents = documents.len();
                self.wrench.document_id = documents[0].document_id;
            }
        }
    }

    /// Consume and render everything the backend produces until it goes quiet,
    /// so that the measured iterations start from a steady state.
    fn settle(&mut self) {
        loop {
            match self.rx.recv_timeout(Duration::from_millis(250)) {
                Ok(NotifierEvent::WakeUp { .. }) => {
                    self.wrench.render();
                    self.window.swap_buffers();
                }
                Ok(NotifierEvent::ShutDown) => panic!("Unexpected shutdown"),
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => panic!("Notifier disconnected"),
            }
        }
        self.wrench.render();
        self.wrench.renderer.take_frame_build_profiles();
        self.wrench.get_frame_profiles();
    }

    fn request_frame(&mut self) -> usize {
        match self.options.mode {
            BenchMode::FrameBuild => {
                self.wrench.api.send_debug_cmd(DebugCommand::GenerateFrame);
                self.num_documents
            }
            BenchMode::SceneBuild => {
                let reader = self.yaml_reader.as_mut().unwrap();
                let before = reader.frame_count();
                let after = reader.do_frame(self.wrench);
                (after - before) as usize
            }
            BenchMode::RenderOnly => unreachable!(),
        }
    }

    fn iteration(&mut self, metrics: &mut Metrics) {
        let start = Instant::now();
        if self.options.mode == BenchMode::RenderOnly {
            self.wrench.renderer.invalidate_rendered_frames();
        } else {
            let expected_notifications = self.request_frame();
            for _ in 0..expected_notifications {
                match self.rx.recv() {
                    Ok(NotifierEvent::WakeUp { .. }) => {}
                    Ok(NotifierEvent::ShutDown) => panic!("Unexpected shutdown"),
                    Err(e) => panic!("Notifier error: {:?}", e),
                }
            }
            metrics.add(self.build_wall_name(), "ms", ms(start.elapsed()));
        }

        let renderer_start = Instant::now();
        if self.options.render {
            self.wrench.render();
            if self.options.gpu_sync {
                self.wrench.gl().finish();
            }
            self.window.upload_software_to_native();
            self.window.swap_buffers();
        } else {
            // Consume the frame without drawing it. The persistent targets it
            // writes to (picture cache tiles, cached render tasks) are left
            // stale, see DebugFlags::SKIP_RENDERING.
            self.wrench.renderer.update();
        }
        let end = Instant::now();

        if !self.options.render {
            metrics.add(CONSUME_WALL, "ms", ms(end - renderer_start));
        }
        if self.options.render {
            metrics.add(RENDERER_WALL, "ms", ms(end - renderer_start));
            let (cpu_profiles, gpu_profiles) = self.wrench.renderer.get_frame_profiles();
            if let Some(profile) = cpu_profiles.last() {
                metrics.add(RENDERER_CPU, "ms", profile.composite_time_ns as f64 / 1_000_000.0);
                metrics.add(DRAW_CALLS, "", profile.draw_calls as f64);
            }
            // GPU timer query results arrive a few frames late, so these
            // samples are for earlier iterations.
            for profile in gpu_profiles {
                metrics.add(GPU, "ms", profile.paint_time_ns as f64 / 1_000_000.0);
            }
        }
        metrics.add(TOTAL_WALL, "ms", ms(end - start));

        // Sum the counters of all documents built in this iteration.
        let mut per_iteration: Vec<ProfileCounterValue> = Vec::new();
        let profiles = self.wrench.renderer.take_frame_build_profiles();
        if profiles.is_empty() && self.options.mode != BenchMode::RenderOnly {
            println!("warning: no frame was built in this iteration");
        }
        for profile in profiles {
            for counter in profile {
                match per_iteration.iter_mut().find(|c| c.name == counter.name) {
                    Some(c) => c.value += counter.value,
                    None => per_iteration.push(counter),
                }
            }
        }
        for counter in per_iteration {
            metrics.add(counter.name, counter.unit, counter.value);
        }
    }

    fn build_wall_name(&self) -> &'static str {
        match self.options.mode {
            BenchMode::FrameBuild | BenchMode::RenderOnly => FRAME_BUILD_WALL,
            BenchMode::SceneBuild => SCENE_BUILD_WALL,
        }
    }

    fn report(&self, input: &BenchInput, metrics: &Metrics, elapsed: Duration) {
        let baseline: Option<SavedResults> = self.options.baseline.as_ref().map(|path| {
            let file = File::open(path)
                .unwrap_or_else(|e| panic!("Unable to open baseline {:?}: {}", path, e));
            serde_json::from_reader(BufReader::new(file))
                .unwrap_or_else(|e| panic!("Unable to parse baseline {:?}: {}", path, e))
        });

        let (mode, invalidation) = match self.options.mode {
            BenchMode::FrameBuild => ("frame building", "all tiles invalidated"),
            BenchMode::SceneBuild => ("scene building", "tiles retained across scenes"),
            BenchMode::RenderOnly => ("rendering", "frame built once, all targets redrawn"),
        };
        println!();
        println!(
            "{} ({}{}), {} iterations after {} warmup, {}",
            input.path().display(),
            mode,
            if self.options.render && self.options.mode != BenchMode::RenderOnly { " + rendering" } else { "" },
            self.options.iterations,
            self.options.warmup,
            invalidation,
        );
        if !self.options.render {
            println!("Frames are discarded without being drawn (use --render to draw them).");
        }
        println!(
            "{:<32} {:>10} {:>10} {:>10} {:>10} {:>10}{}",
            "", "mean", "median", "min", "max", "stddev",
            if baseline.is_some() { "   median vs baseline" } else { "" },
        );

        let mut shown: Vec<&str> = vec![self.build_wall_name()];
        shown.extend_from_slice(DEFAULT_COUNTERS);
        if self.options.render {
            shown.extend_from_slice(&[RENDERER_WALL, RENDERER_CPU, DRAW_CALLS]);
        } else {
            shown.push(CONSUME_WALL);
        }
        if self.options.gpu_queries {
            shown.push(GPU);
        }
        shown.push(TOTAL_WALL);

        let print = |series: &Series| {
            let summary = series.summarize();
            let comparison = baseline.as_ref()
                .and_then(|b| b.summaries.iter().find(|s| s.name == summary.name))
                .map(|b| {
                    if b.median == 0.0 {
                        String::from("   n/a")
                    } else {
                        let delta = (summary.median - b.median) / b.median * 100.0;
                        format!("   {:>+7.2}% (was {:.3})", delta, b.median)
                    }
                })
                .unwrap_or_default();
            println!(
                "{:<32} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}{}",
                format!("{} {}", series.name, if series.unit.is_empty() { String::new() } else { format!("({})", series.unit) }),
                summary.mean, summary.median, summary.min, summary.max, summary.stddev,
                comparison,
            );
        };

        for name in &shown {
            if let Some(series) = metrics.get(name) {
                print(series);
            }
        }

        if self.options.all_counters {
            println!();
            println!("All profiler counters recorded while building frames:");
            for series in &metrics.series {
                let all_zero = series.samples.iter().all(|v| *v == 0.0);
                if !shown.contains(&series.name.as_str()) && !all_zero {
                    print(series);
                }
            }
        }

        println!();
        println!(
            "{} iterations in {:.3} s ({:.1} iterations/s)",
            self.options.iterations + self.options.warmup,
            elapsed.as_secs_f64(),
            (self.options.iterations + self.options.warmup) as f64 / elapsed.as_secs_f64(),
        );

        if let Some(ref path) = self.options.csv {
            write_csv(path, metrics);
            println!("Wrote per-iteration samples to {}", path.display());
        }

        if let Some(ref path) = self.options.save {
            let results = SavedResults {
                input: input.path().display().to_string(),
                iterations: self.options.iterations,
                summaries: metrics.series.iter().map(Series::summarize).collect(),
                series: metrics.series.iter().map(|s| Series {
                    name: s.name.clone(),
                    unit: s.unit.clone(),
                    samples: s.samples.clone(),
                }).collect(),
            };
            let file = File::create(path)
                .unwrap_or_else(|e| panic!("Unable to create {:?}: {}", path, e));
            serde_json::to_writer_pretty(BufWriter::new(file), &results).unwrap();
            println!("Saved results to {}", path.display());
        }
    }
}

fn write_csv(path: &Path, metrics: &Metrics) {
    let file = File::create(path).unwrap_or_else(|e| panic!("Unable to create {:?}: {}", path, e));
    let mut out = BufWriter::new(file);
    let header: Vec<String> = metrics.series.iter()
        .map(|s| if s.unit.is_empty() { s.name.clone() } else { format!("{} ({})", s.name, s.unit) })
        .collect();
    writeln!(out, "iteration,{}", header.join(",")).unwrap();
    let rows = metrics.series.iter().map(|s| s.samples.len()).max().unwrap_or(0);
    for row in 0..rows {
        let values: Vec<String> = metrics.series.iter()
            .map(|s| s.samples.get(row).map(|v| v.to_string()).unwrap_or_default())
            .collect();
        writeln!(out, "{},{}", row, values.join(",")).unwrap();
    }
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
