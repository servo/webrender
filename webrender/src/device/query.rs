/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::mem;
use std::rc::Rc;

use crate::device::GpuFrameId;
use crate::profiler::GpuProfileTag;

/// Backend-defined identifier of a GPU query object.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GpuQueryId(pub(super) u32);

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum GpuQueryKind {
    /// GPU time in nanoseconds spent on the commands between begin and end.
    TimeElapsed,
    /// Number of samples that passed the depth test between begin and end.
    SamplesPassed,
}

/// GPU queries and command stream annotations, as provided by a backend
/// for the GPU profiler. At most one query of each kind is active at a time.
pub trait GpuQueryBackend {
    fn create_queries(&self, count: usize) -> Vec<GpuQueryId>;
    fn delete_queries(&self, queries: &[GpuQueryId]);
    fn begin_query(&self, kind: GpuQueryKind, query: GpuQueryId);
    fn end_query(&self, kind: GpuQueryKind);
    /// Returns the result of a finished query, waiting for it if needed.
    fn query_result(&self, query: GpuQueryId) -> u64;
    /// Whether the marker methods below annotate the command stream.
    fn supports_markers(&self) -> bool;
    fn push_marker_group(&self, label: &str);
    fn pop_marker_group(&self);
    fn insert_marker(&self, label: &str);
}

#[derive(Debug, Clone)]
pub struct GpuTimer {
    pub tag: GpuProfileTag,
    pub time_ns: u64,
}

#[derive(Debug, Clone)]
pub struct GpuSampler {
    pub tag: GpuProfileTag,
    pub count: u64,
}

pub struct QuerySet<T> {
    set: Vec<GpuQueryId>,
    data: Vec<T>,
    pending: Option<GpuQueryId>,
}

impl<T> QuerySet<T> {
    fn new() -> Self {
        QuerySet {
            set: Vec::new(),
            data: Vec::new(),
            pending: None,
        }
    }

    fn reset(&mut self) {
        self.data.clear();
        self.pending = None;
    }

    fn add(&mut self, value: T) -> Option<GpuQueryId> {
        assert_eq!(self.pending, None);
        self.set.get(self.data.len()).cloned().map(|query_id| {
            self.data.push(value);
            self.pending = Some(query_id);
            query_id
        })
    }

    fn take<F: Fn(&mut T, GpuQueryId)>(&mut self, fun: F) -> Vec<T> {
        let mut data = mem::replace(&mut self.data, Vec::new());
        for (value, &query) in data.iter_mut().zip(self.set.iter()) {
            fun(value, query)
        }
        data
    }
}

pub struct GpuFrameProfile {
    queries: Rc<dyn GpuQueryBackend>,
    timers: QuerySet<GpuTimer>,
    samplers: QuerySet<GpuSampler>,
    frame_id: GpuFrameId,
    inside_frame: bool,
}

impl GpuFrameProfile {
    fn new(queries: Rc<dyn GpuQueryBackend>) -> Self {
        GpuFrameProfile {
            queries,
            timers: QuerySet::new(),
            samplers: QuerySet::new(),
            frame_id: GpuFrameId::new(0),
            inside_frame: false,
        }
    }

    fn enable_timers(&mut self, count: usize) {
        self.timers.set = self.queries.create_queries(count);
    }

    fn disable_timers(&mut self) {
        if !self.timers.set.is_empty() {
            self.queries.delete_queries(&self.timers.set);
        }
        self.timers.set = Vec::new();
    }

    fn enable_samplers(&mut self, count: usize) {
        self.samplers.set = self.queries.create_queries(count);
    }

    fn disable_samplers(&mut self) {
        if !self.samplers.set.is_empty() {
            self.queries.delete_queries(&self.samplers.set);
        }
        self.samplers.set = Vec::new();
    }

    fn begin_frame(&mut self, frame_id: GpuFrameId) {
        self.frame_id = frame_id;
        self.timers.reset();
        self.samplers.reset();
        self.inside_frame = true;
    }

    fn end_frame(&mut self) {
        self.finish_timer();
        self.finish_sampler();
        self.inside_frame = false;
    }

    fn finish_timer(&mut self) {
        debug_assert!(self.inside_frame);
        if self.timers.pending.is_some() {
            self.queries.end_query(GpuQueryKind::TimeElapsed);
            self.timers.pending = None;
        }
    }

    fn finish_sampler(&mut self) {
        debug_assert!(self.inside_frame);
        if self.samplers.pending.is_some() {
            self.queries.end_query(GpuQueryKind::SamplesPassed);
            self.samplers.pending = None;
        }
    }

    fn start_timer(&mut self, tag: GpuProfileTag) -> GpuTimeQuery {
        self.finish_timer();

        let marker = GpuMarker::new(&self.queries, tag.label);

        if let Some(query) = self.timers.add(GpuTimer { tag, time_ns: 0 }) {
            self.queries.begin_query(GpuQueryKind::TimeElapsed, query);
        }

        GpuTimeQuery(marker)
    }

    fn start_sampler(&mut self, tag: GpuProfileTag) -> GpuSampleQuery {
        self.finish_sampler();

        if let Some(query) = self.samplers.add(GpuSampler { tag, count: 0 }) {
            self.queries.begin_query(GpuQueryKind::SamplesPassed, query);
        }

        GpuSampleQuery
    }

    fn build_samples(&mut self) -> (GpuFrameId, Vec<GpuTimer>, Vec<GpuSampler>) {
        debug_assert!(!self.inside_frame);
        let queries = &self.queries;

        (
            self.frame_id,
            self.timers.take(|timer, query| {
                timer.time_ns = queries.query_result(query)
            }),
            self.samplers.take(|sampler, query| {
                sampler.count = queries.query_result(query)
            }),
        )
    }
}

impl Drop for GpuFrameProfile {
    fn drop(&mut self) {
        self.disable_timers();
        self.disable_samplers();
    }
}

const NUM_PROFILE_FRAMES: usize = 4;

pub struct GpuProfiler {
    queries: Rc<dyn GpuQueryBackend>,
    frames: [GpuFrameProfile; NUM_PROFILE_FRAMES],
    next_frame: usize,
}

impl GpuProfiler {
    pub fn new(queries: Rc<dyn GpuQueryBackend>) -> Self {
        let f = || GpuFrameProfile::new(Rc::clone(&queries));

        let frames = [f(), f(), f(), f()];
        GpuProfiler {
            queries,
            next_frame: 0,
            frames,
        }
    }

    pub fn enable_timers(&mut self) {
        const MAX_TIMERS_PER_FRAME: usize = 256;

        for frame in &mut self.frames {
            frame.enable_timers(MAX_TIMERS_PER_FRAME);
        }
    }

    pub fn disable_timers(&mut self) {
        for frame in &mut self.frames {
            frame.disable_timers();
        }
    }

    pub fn enable_samplers(&mut self) {
        const MAX_SAMPLERS_PER_FRAME: usize = 16;
        if cfg!(target_os = "macos") {
            warn!("Expect macOS driver bugs related to sample queries")
        }

        for frame in &mut self.frames {
            frame.enable_samplers(MAX_SAMPLERS_PER_FRAME);
        }
    }

    pub fn disable_samplers(&mut self) {
        for frame in &mut self.frames {
            frame.disable_samplers();
        }
    }

    pub fn build_samples(&mut self) -> (GpuFrameId, Vec<GpuTimer>, Vec<GpuSampler>) {
        self.frames[self.next_frame].build_samples()
    }

    pub fn begin_frame(&mut self, frame_id: GpuFrameId) {
        self.frames[self.next_frame].begin_frame(frame_id);
    }

    pub fn end_frame(&mut self) {
        self.frames[self.next_frame].end_frame();
        self.next_frame = (self.next_frame + 1) % self.frames.len();
    }

    pub fn start_timer(&mut self, tag: GpuProfileTag) -> GpuTimeQuery {
        self.frames[self.next_frame].start_timer(tag)
    }

    pub fn start_sampler(&mut self, tag: GpuProfileTag) -> GpuSampleQuery {
        self.frames[self.next_frame].start_sampler(tag)
    }

    pub fn finish_sampler(&mut self, _sampler: GpuSampleQuery) {
        self.frames[self.next_frame].finish_sampler()
    }

    pub fn start_marker(&mut self, label: &str) -> GpuMarker {
        GpuMarker::new(&self.queries, label)
    }

    pub fn place_marker(&mut self, label: &str) {
        GpuMarker::fire(&self.queries, label)
    }
}

/// A marker group in the command stream, closed when dropped.
#[must_use]
pub struct GpuMarker {
    queries: Option<Rc<dyn GpuQueryBackend>>,
}

impl GpuMarker {
    fn new(queries: &Rc<dyn GpuQueryBackend>, message: &str) -> Self {
        let queries = if queries.supports_markers() {
            queries.push_marker_group(message);
            Some(Rc::clone(queries))
        } else {
            None
        };
        GpuMarker { queries }
    }

    fn fire(queries: &Rc<dyn GpuQueryBackend>, message: &str) {
        if queries.supports_markers() {
            queries.insert_marker(message);
        }
    }
}

impl Drop for GpuMarker {
    fn drop(&mut self) {
        if let Some(ref queries) = self.queries {
            queries.pop_marker_group();
        }
    }
}

#[must_use]
pub struct GpuTimeQuery(#[allow(dead_code)] GpuMarker);
#[must_use]
pub struct GpuSampleQuery;
