/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use WindowWrapper;
use serde_json;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use webrender::api::*;
use wrench::{Wrench, WrenchThing};
use yaml_frame_reader::YamlFrameReader;

const COLOR_DEFAULT: &str = "\x1b[0m";
const COLOR_RED: &str = "\x1b[31m";
const COLOR_GREEN: &str = "\x1b[32m";
const COLOR_MAGENTA: &str = "\x1b[95m";

pub struct Benchmark {
    pub test: PathBuf,
}

pub struct BenchmarkManifest {
    pub benchmarks: Vec<Benchmark>,
}

impl BenchmarkManifest {
    pub fn new(manifest: &Path) -> BenchmarkManifest {
        let dir = manifest.parent().unwrap();
        let f = File::open(manifest).expect(&format!("couldn't open manifest: {}", manifest.display()));
        let file = BufReader::new(&f);

        let mut benchmarks = Vec::new();

        for line in file.lines() {
            let l = line.unwrap();

            // strip the comments
            let s = &l[0..l.find('#').unwrap_or(l.len())];
            let s = s.trim();
            if s.is_empty() {
                continue;
            }

            let mut items = s.split_whitespace();

            match items.next() {
                Some("include") => {
                    let include = dir.join(items.next().unwrap());

                    benchmarks.append(&mut BenchmarkManifest::new(include.as_path()).benchmarks);
                }
                Some(name) => {
                    let test = dir.join(name);
                    benchmarks.push(Benchmark {
                        test,
                    });
                }
                _ => panic!(),
            };
        }

        BenchmarkManifest {
            benchmarks: benchmarks
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct TestProfile {
    name: String,
    composite_time_ns: u64,
    paint_time_ns: u64,
    draw_calls: usize,
}

#[derive(Serialize, Deserialize)]
struct Profile {
    tests: Vec<TestProfile>,
}

impl Profile {
    fn new() -> Profile {
        Profile {
            tests: Vec::new(),
        }
    }

    fn add(&mut self, profile: TestProfile) {
        self.tests.push(profile);
    }

    fn save(&self, filename: &str) {
        let mut file = File::create(&filename).unwrap();
        let s = serde_json::to_string_pretty(self).unwrap();
        file.write_all(&s.into_bytes()).unwrap();
        file.write_all(b"\n").unwrap();
    }

    fn load(filename: &str) -> Profile {
        let mut file = File::open(&filename).unwrap();
        let mut string = String::new();
        file.read_to_string(&mut string).unwrap();
        serde_json::from_str(&string).expect("Unable to load profile!")
    }

    fn build_set_and_map_of_tests(&self) -> (HashSet<String>, HashMap<String, TestProfile>) {
        let mut hash_set = HashSet::new();
        let mut hash_map = HashMap::new();

        for test in &self.tests {
            hash_set.insert(test.name.clone());
            hash_map.insert(test.name.clone(), test.clone());
        }

        (hash_set, hash_map)
    }
}

pub struct PerfHarness<'a> {
    wrench: &'a mut Wrench,
    window: &'a mut WindowWrapper,
    rx: Receiver<()>,
}

impl<'a> PerfHarness<'a> {
    pub fn new(wrench: &'a mut Wrench,
               window: &'a mut WindowWrapper) -> PerfHarness<'a>
    {
        // setup a notifier so we can wait for frames to be finished
        struct Notifier {
            tx: Sender<()>,
        };
        impl RenderNotifier for Notifier {
            fn new_frame_ready(&mut self) {
                self.tx.send(()).unwrap();
            }
            fn new_scroll_frame_ready(&mut self, _composite_needed: bool) {}
        }
        let (tx, rx) = channel();
        wrench.renderer.set_render_notifier(Box::new(Notifier { tx: tx }));

        PerfHarness {
            wrench,
            window,
            rx,
        }
    }

    pub fn run(mut self, base_manifest: &Path, filename: &str) {
        let manifest = BenchmarkManifest::new(base_manifest);

        let mut profile = Profile::new();

        for t in manifest.benchmarks {
            let stats = self.render_yaml(t.test.as_path());
            profile.add(stats);
        }

        profile.save(filename);
    }

    fn render_yaml(&mut self, filename: &Path) -> TestProfile {
        let mut reader = YamlFrameReader::new(filename);
        reader.do_frame(self.wrench);

        // wait for the frame
        self.rx.recv().unwrap();

        // Loop until we get a reasonable number of CPU and GPU
        // frame profiles. Then take the mean.
        let mut cpu_frame_profiles = Vec::new();
        let mut gpu_frame_profiles = Vec::new();

        while cpu_frame_profiles.len() < 10 ||
              gpu_frame_profiles.len() < 10 {
            self.wrench.render();
            self.window.swap_buffers();
            let (cpu_profiles, gpu_profiles) = self.wrench.get_frame_profiles();
            cpu_frame_profiles.extend(cpu_profiles);
            gpu_frame_profiles.extend(gpu_profiles);
        }

        // Ensure the draw calls match in every sample.
        let draw_calls = cpu_frame_profiles[0].draw_calls;
        assert!(cpu_frame_profiles.iter().all(|s| s.draw_calls == draw_calls));

        cpu_frame_profiles.sort_by_key(|a| a.composite_time_ns);
        gpu_frame_profiles.sort_by_key(|a| a.paint_time_ns);

        // Remove the two slowest and fastest frames.
        let cpu_samples = &cpu_frame_profiles[2..cpu_frame_profiles.len() - 2];
        let gpu_samples = &gpu_frame_profiles[2..gpu_frame_profiles.len() - 2];

        // Use the mean value from the remaining frames.
        let composite_time: u64 = cpu_samples.iter()
                                             .map(|s| s.composite_time_ns)
                                             .sum();
        let paint_time: u64 = gpu_samples.iter()
                                         .map(|s| s.paint_time_ns)
                                         .sum();

        TestProfile {
            name: filename.to_str().unwrap().to_string(),
            composite_time_ns: composite_time / cpu_samples.len() as u64,
            paint_time_ns: paint_time / gpu_samples.len() as u64,
            draw_calls,
        }
    }
}

fn select_color(base: f32, value: f32) -> &'static str {
    let tolerance = base * 0.1;
    if (value - base).abs() < tolerance {
        COLOR_DEFAULT
    } else if value > base {
        COLOR_RED
    } else {
        COLOR_GREEN
    }
}

pub fn compare(first_filename: &str, second_filename: &str) {
    let profile0 = Profile::load(first_filename);
    let profile1 = Profile::load(second_filename);

    let (set0, map0) = profile0.build_set_and_map_of_tests();
    let (set1, map1) = profile1.build_set_and_map_of_tests();

    println!("+------------------------------------------------+--------------+------------------+------------------+");
    println!("|  Test name                                     | Draw Calls   | Composite (ms)   | Paint (ms)       |");
    println!("+------------------------------------------------+--------------+------------------+------------------+");

    for test_name in set0.symmetric_difference(&set1) {
        println!("| {}{:47}{}|{:14}|{:18}|{:18}|", COLOR_MAGENTA, test_name, COLOR_DEFAULT, " -", " -", " -");
    }

    for test_name in set0.intersection(&set1) {
        let test0 = &map0[test_name];
        let test1 = &map1[test_name];

        let composite_time0 = test0.composite_time_ns as f32 / 1000000.0;
        let composite_time1 = test1.composite_time_ns as f32 / 1000000.0;

        let paint_time0 = test0.paint_time_ns as f32 / 1000000.0;
        let paint_time1 = test1.paint_time_ns as f32 / 1000000.0;

        let draw_calls_color = if test0.draw_calls == test1.draw_calls {
            COLOR_DEFAULT
        } else if test0.draw_calls > test1.draw_calls {
            COLOR_GREEN
        } else {
            COLOR_RED
        };

        let composite_time_color = select_color(composite_time0, composite_time1);
        let paint_time_color = select_color(paint_time0, paint_time1);

        let draw_call_string = format!(" {} -> {}", test0.draw_calls, test1.draw_calls);
        let composite_time_string = format!(" {:.2} -> {:.2}", composite_time0,
                                                               composite_time1);
        let paint_time_string = format!(" {:.2} -> {:.2}", paint_time0,
                                                           paint_time1);

        println!("| {:47}|{}{:14}{}|{}{:18}{}|{}{:18}{}|", test_name,
                                                           draw_calls_color,
                                                           draw_call_string,
                                                           COLOR_DEFAULT,
                                                           composite_time_color,
                                                           composite_time_string,
                                                           COLOR_DEFAULT,
                                                           paint_time_color,
                                                           paint_time_string,
                                                           COLOR_DEFAULT);
    }

    println!("+------------------------------------------------+--------------+------------------+------------------+");
}
