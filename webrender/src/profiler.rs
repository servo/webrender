/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use debug_render::DebugRenderer;
use euclid::{Point2D, Size2D, Rect};
use std::collections::vec_deque::VecDeque;
use std::f32;
use webrender_traits::ColorF;
use time::precise_time_ns;

trait ProfileCounter {
    fn description(&self) -> &'static str;
    fn value(&self) -> String;
}

#[derive(Clone)]
pub struct IntProfileCounter {
    description: &'static str,
    value: usize,
}

impl IntProfileCounter {
    fn new(description: &'static str) -> IntProfileCounter {
        IntProfileCounter {
            description: description,
            value: 0,
        }
    }

    fn reset(&mut self) {
        self.value = 0;
    }

    #[inline(always)]
    pub fn inc(&mut self) {
        self.value += 1;
    }

    #[inline(always)]
    pub fn add(&mut self, amount: usize) {
        self.value += amount;
    }

    #[inline(always)]
    pub fn set(&mut self, amount: usize) {
        self.value = amount;
    }
}

impl ProfileCounter for IntProfileCounter {
    fn description(&self) -> &'static str {
        self.description
    }

    fn value(&self) -> String {
        format!("{}", self.value)
    }
}

#[derive(Clone)]
pub struct ResourceProfileCounter {
    description: &'static str,
    value: usize,
    size: usize,
}

impl ResourceProfileCounter {
    fn new(description: &'static str) -> ResourceProfileCounter {
        ResourceProfileCounter {
            description: description,
            value: 0,
            size: 0,
        }
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.value = 0;
        self.size = 0;
    }

    #[inline(always)]
    pub fn inc(&mut self, size: usize) {
        self.value += 1;
        self.size += size;
    }
}

impl ProfileCounter for ResourceProfileCounter {
    fn description(&self) -> &'static str {
        self.description
    }

    fn value(&self) -> String {
        let size = self.size as f32 / (1024.0 * 1024.0);
        format!("{} ({:.2} MB)", self.value, size)
    }
}

#[derive(Clone)]
pub struct TimeProfileCounter {
    description: &'static str,
    nanoseconds: u64,
    invert: bool,
}

impl TimeProfileCounter {
    pub fn new(description: &'static str, invert: bool) -> TimeProfileCounter {
        TimeProfileCounter {
            description: description,
            nanoseconds: 0,
            invert: invert,
        }
    }

    fn reset(&mut self) {
        self.nanoseconds = 0;
    }

    pub fn set(&mut self, ns: u64) {
        self.nanoseconds = ns;
    }

    pub fn profile<T, F>(&mut self, callback: F) -> T where F: FnOnce() -> T {
        let t0 = precise_time_ns();
        let val = callback();
        let t1 = precise_time_ns();
        let ns = t1 - t0;
        self.nanoseconds += ns;
        val
    }
}

impl ProfileCounter for TimeProfileCounter {
    fn description(&self) -> &'static str {
        self.description
    }

    fn value(&self) -> String {
        if self.invert {
            format!("{:.2} fps", 1000000000.0 / self.nanoseconds as f64)
        } else {
            format!("{:.2} ms", self.nanoseconds as f64 / 1000000.0)
        }
    }
}

pub struct FrameProfileCounters {
    pub total_primitives: IntProfileCounter,
    pub visible_primitives: IntProfileCounter,
    pub phases: IntProfileCounter,
    pub targets: IntProfileCounter,
}

impl FrameProfileCounters {
    pub fn new() -> FrameProfileCounters {
        FrameProfileCounters {
            total_primitives: IntProfileCounter::new("Total Primitives"),
            visible_primitives: IntProfileCounter::new("Visible Primitives"),
            phases: IntProfileCounter::new("Phases"),
            targets: IntProfileCounter::new("Target switches"),
        }
    }
}

#[derive(Clone)]
pub struct BackendProfileCounters {
    pub font_templates: ResourceProfileCounter,
    pub image_templates: ResourceProfileCounter,
    pub total_time: TimeProfileCounter,
}

impl BackendProfileCounters {
    pub fn new() -> BackendProfileCounters {
        BackendProfileCounters {
            font_templates: ResourceProfileCounter::new("Font Templates"),
            image_templates: ResourceProfileCounter::new("Image Templates"),
            total_time: TimeProfileCounter::new("Backend CPU Time", false),
        }
    }

    pub fn reset(&mut self) {
        self.total_time.reset();
    }
}

pub struct RendererProfileCounters {
    pub frame_counter: IntProfileCounter,
    pub frame_time: TimeProfileCounter,
    pub draw_calls: IntProfileCounter,
    pub vertices: IntProfileCounter,
    pub vao_count_and_size: ResourceProfileCounter,
}

pub struct RendererProfileTimers {
    pub cpu_time: TimeProfileCounter,
    pub gpu_time_paint: TimeProfileCounter,
    pub gpu_time_composite: TimeProfileCounter,
    pub gpu_time_total: TimeProfileCounter,
}

impl RendererProfileCounters {
    pub fn new() -> RendererProfileCounters {
        RendererProfileCounters {
            frame_counter: IntProfileCounter::new("Frame"),
            frame_time: TimeProfileCounter::new("FPS", true),
            draw_calls: IntProfileCounter::new("Draw Calls"),
            vertices: IntProfileCounter::new("Vertices"),
            vao_count_and_size: ResourceProfileCounter::new("VAO"),
        }
    }

    pub fn reset(&mut self) {
        self.draw_calls.reset();
        self.vertices.reset();
    }
}

impl RendererProfileTimers {
    pub fn new() -> RendererProfileTimers {
        RendererProfileTimers {
            cpu_time: TimeProfileCounter::new("Compositor CPU Time", false),
            gpu_time_paint: TimeProfileCounter::new("GPU Time (paint)", false),
            gpu_time_composite: TimeProfileCounter::new("GPU Time (composite)", false),
            gpu_time_total: TimeProfileCounter::new("GPU Time (total)", false),
        }
    }
}

struct GraphStats {
    min_value: f32,
    mean_value: f32,
    max_value: f32,
}

struct ProfileGraph {
    max_samples: usize,
    values: VecDeque<f32>,
}

impl ProfileGraph {
    fn new(max_samples: usize) -> ProfileGraph {
        ProfileGraph {
            max_samples: max_samples,
            values: VecDeque::new(),
        }
    }

    fn push(&mut self, ns: u64) {
        let ms = ns as f64 / 1000000.0;
        if self.values.len() == self.max_samples {
            self.values.pop_back();
        }
        self.values.push_front(ms as f32);
    }

    fn stats(&self) -> GraphStats {
        let mut stats = GraphStats {
            min_value: f32::MAX,
            mean_value: 0.0,
            max_value: -f32::MAX,
        };

        for value in &self.values {
            stats.min_value = stats.min_value.min(*value);
            stats.mean_value = stats.mean_value + *value;
            stats.max_value = stats.max_value.max(*value);
        }

        if !self.values.is_empty() {
            stats.mean_value = stats.mean_value / self.values.len() as f32;
        }

        stats
    }

    fn draw_graph(&self,
                  x: f32,
                  y: f32,
                  description: &'static str,
                  debug_renderer: &mut DebugRenderer) -> Rect<f32> {
        let size = Size2D::new(600.0, 120.0);
        let line_height = debug_renderer.line_height();
        let mut rect = Rect::new(Point2D::new(x, y), size);
        let stats = self.stats();

        let text_color = ColorF::new(1.0, 1.0, 0.0, 1.0);
        let text_origin = rect.origin + Point2D::new(rect.size.width, 20.0);
        debug_renderer.add_text(text_origin.x,
                                text_origin.y,
                                description,
                                &ColorF::new(0.0, 1.0, 0.0, 1.0));
        debug_renderer.add_text(text_origin.x,
                                text_origin.y + line_height,
                                &format!("Min: {:.2} ms", stats.min_value),
                                &text_color);
        debug_renderer.add_text(text_origin.x,
                                text_origin.y + line_height * 2.0,
                                &format!("Mean: {:.2} ms", stats.mean_value),
                                &text_color);
        debug_renderer.add_text(text_origin.x,
                                text_origin.y + line_height * 3.0,
                                &format!("Max: {:.2} ms", stats.max_value),
                                &text_color);

        rect.size.width += 140.0;
        debug_renderer.add_quad(rect.origin.x,
                                rect.origin.y,
                                rect.origin.x + rect.size.width + 10.0,
                                rect.origin.y + rect.size.height,
                                &ColorF::new(0.1, 0.1, 0.1, 0.8),
                                &ColorF::new(0.2, 0.2, 0.2, 0.8));

        let bx0 = x + 10.0;
        let by0 = y + 10.0;
        let bx1 = bx0 + size.width - 20.0;
        let by1 = by0 + size.height - 20.0;

        let w = (bx1 - bx0) / self.max_samples as f32;
        let h = by1 - by0;

        let color_t0 = ColorF::new(0.0, 1.0, 0.0, 1.0);
        let color_b0 = ColorF::new(0.0, 0.7, 0.0, 1.0);

        let color_t1 = ColorF::new(0.0, 1.0, 0.0, 1.0);
        let color_b1 = ColorF::new(0.0, 0.7, 0.0, 1.0);

        let color_t2 = ColorF::new(1.0, 0.0, 0.0, 1.0);
        let color_b2 = ColorF::new(0.7, 0.0, 0.0, 1.0);

        for (index, sample) in self.values.iter().enumerate() {
            let sample = *sample;
            let x1 = bx1 - index as f32 * w;
            let x0 = x1 - w;

            let y0 = by1 - (sample / stats.max_value) as f32 * h;
            let y1 = by1;

            let (color_top, color_bottom) = if sample < 1000.0 / 60.0 {
                (&color_t0, &color_b0)
            } else if sample < 1000.0 / 30.0 {
                (&color_t1, &color_b1)
            } else {
                (&color_t2, &color_b2)
            };

            debug_renderer.add_quad(x0, y0, x1, y1, color_top, color_bottom);
        }

        rect
    }
}

pub struct Profiler {
    x_left: f32,
    y_left: f32,
    x_right: f32,
    y_right: f32,
    backend_time: ProfileGraph,
    compositor_time: ProfileGraph,
    gpu_time: ProfileGraph,
}

impl Profiler {
    pub fn new() -> Profiler {
        Profiler {
            x_left: 0.0,
            y_left: 0.0,
            x_right: 0.0,
            y_right: 0.0,
            backend_time: ProfileGraph::new(600),
            compositor_time: ProfileGraph::new(600),
            gpu_time: ProfileGraph::new(600),
        }
    }

    fn draw_counters(&mut self,
                     counters: &[&ProfileCounter],
                     debug_renderer: &mut DebugRenderer,
                     left: bool) {
        let mut label_rect = Rect::zero();
        let mut value_rect = Rect::zero();
        let (mut current_x, mut current_y) = if left {
            (self.x_left, self.y_left)
        } else {
            (self.x_right, self.y_right)
        };
        let mut color_index = 0;
        let line_height = debug_renderer.line_height();

        let colors = [
            ColorF::new(1.0, 1.0, 1.0, 1.0),
            ColorF::new(1.0, 1.0, 0.0, 1.0),
        ];

        for counter in counters {
            let rect = debug_renderer.add_text(current_x,
                                               current_y,
                                               counter.description(),
                                               &colors[color_index]);
            color_index = (color_index+1) % colors.len();

            label_rect = label_rect.union(&rect);
            current_y += line_height;
        }

        color_index = 0;
        current_x = label_rect.origin.x + label_rect.size.width + 60.0;
        current_y = if left {
            self.y_left
        } else {
            self.y_right
        };

        for counter in counters {
            let rect = debug_renderer.add_text(current_x,
                                                    current_y,
                                                    &counter.value(),
                                                    &colors[color_index]);
            color_index = (color_index+1) % colors.len();

            value_rect = value_rect.union(&rect);
            current_y += line_height;
        }

        let total_rect = label_rect.union(&value_rect).inflate(10.0, 10.0);
        debug_renderer.add_quad(total_rect.origin.x,
                                total_rect.origin.y,
                                total_rect.origin.x + total_rect.size.width,
                                total_rect.origin.y + total_rect.size.height,
                                &ColorF::new(0.1, 0.1, 0.1, 0.8),
                                &ColorF::new(0.2, 0.2, 0.2, 0.8));
        let new_y = total_rect.origin.y + total_rect.size.height + 30.0;
        if left {
            self.y_left = new_y;
        } else {
            self.y_right = new_y;
        }
    }

    pub fn draw_profile(&mut self,
                        frame_profile: &FrameProfileCounters,
                        backend_profile: &BackendProfileCounters,
                        renderer_profile: &RendererProfileCounters,
                        renderer_timers: &RendererProfileTimers,
                        debug_renderer: &mut DebugRenderer) {
        self.x_left = 20.0;
        self.y_left = 40.0;
        self.x_right = 400.0;
        self.y_right = 40.0;

        self.draw_counters(&[
            &renderer_profile.frame_counter,
            &renderer_profile.frame_time,
        ], debug_renderer, true);

        self.draw_counters(&[
            &frame_profile.total_primitives,
            &frame_profile.visible_primitives,
            &frame_profile.phases,
            &frame_profile.targets,
        ], debug_renderer, true);

        self.draw_counters(&[
            &backend_profile.font_templates,
            &backend_profile.image_templates,
        ], debug_renderer, true);

        self.draw_counters(&[
            &renderer_profile.draw_calls,
            &renderer_profile.vertices,
        ], debug_renderer, true);

        self.draw_counters(&[
            &backend_profile.total_time,
            &renderer_timers.cpu_time,
            &renderer_timers.gpu_time_paint,
            &renderer_timers.gpu_time_composite,
            &renderer_timers.gpu_time_total,
        ], debug_renderer, false);

        self.backend_time.push(backend_profile.total_time.nanoseconds);
        self.compositor_time.push(renderer_timers.cpu_time.nanoseconds);
        self.gpu_time.push(renderer_timers.gpu_time_total.nanoseconds);

        let rect = self.backend_time.draw_graph(self.x_left, self.y_left, "CPU (backend)", debug_renderer);
        self.y_left += rect.size.height + 10.0;
        let rect = self.compositor_time.draw_graph(self.x_left, self.y_left, "CPU (compositor)", debug_renderer);
        self.y_left += rect.size.height + 10.0;
        let rect = self.gpu_time.draw_graph(self.x_left, self.y_left, "GPU", debug_renderer);
        self.y_left += rect.size.height + 10.0;
    }
}
