/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use device::TextureFilter;
use frame::FrameId;
use platform::font::{FontContext, RasterizedGlyph};
use profiler::TextureCacheProfileCounters;
use rayon::ThreadPool;
use resource_cache::ResourceClassCache;
use std::sync::{Arc, Mutex, MutexGuard};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::collections::HashSet;
use std::mem;
use texture_cache::{TextureCacheItemId, TextureCache};
use internal_types::FontTemplate;
use webrender_traits::{FontKey, GlyphKey, ImageFormat};
use webrender_traits::{FontRenderMode, ImageData};
use webrender_traits::{ImageDescriptor, ColorF, LayoutPoint};
use webrender_traits::{GlyphOptions, GlyphInstance};

pub type GlyphCache = ResourceClassCache<GlyphRequest, Option<TextureCacheItemId>>;

// Contexts hold on to resources that are needed fo glyph rendering and
// are not thread safe.
// There is one context per thread in the thread pool, plus one for the
// render backend thread.
pub struct Context {
    context: FontContext,
    tx: Sender<GlyphRasterJob>,
}

unsafe impl Send for Context {}

impl Context {
    pub fn rasterize_glyph(&mut self, request: GlyphRequest) {
        let result = self.context.rasterize_glyph(
            &request.key,
            request.render_mode,
            request.glyph_options
        );

        // Sanity check.
        if let Some(ref glyph) = result {
            let bpp = 4; // We always render glyphs in 32 bits RGBA format.
            assert_eq!(glyph.bytes.len(), bpp * (glyph.width * glyph.height) as usize);
        }

        self.tx.send(
            GlyphRasterJob {
                key: request,
                result: result,
            }
        );
    }

    pub fn add_font(&mut self, font_key: FontKey, template: &FontTemplate) {
        match template {
            &FontTemplate::Raw(ref bytes, index) => {
                self.context.add_raw_font(&font_key, &**bytes, index);
            }
            &FontTemplate::Native(ref native_font_handle) => {
                self.context.add_native_font(&font_key, (*native_font_handle).clone());
            }
        }
    }

    pub fn delete_font(&mut self, font_key: FontKey) {
        self.context.delete_font(&font_key);
    }

    pub fn has_font(&self, font_key: &FontKey) -> bool {
        self.context.has_font(font_key)
    }
}

pub struct FontContexts {
    // These worker are mostly accessed from their corresponding worker threads.
    // The goal is that there should be no noticeable contention on the muteces.
    worker_contexts: Vec<Mutex<Context>>,

    // This worker should be accessed by threads that don't belong to thre thread pool
    // (in theory that's only the render backend thread so no contention expected either).
    shared_context: Mutex<Context>,

    // Stored here as a convenience to get the current thread index.
    workers: Arc<ThreadPool>,
}

impl FontContexts {
    /// Get access to the font context associated to the current thread.
    pub fn lock_current_context(&self) -> MutexGuard<Context> {
        let id = self.current_worker_id();
        self.lock_context(id)
    }

    /// Get access to any particular font context.
    ///
    /// The id is ```Some(i)``` where i is an index between 0 and num_worker_contexts
    /// for font contexts associated to the thread pool, and None for the shared
    /// global font context for use outside of the thread pool.
    pub fn lock_context(&self, id: Option<usize>) -> MutexGuard<Context> {
        match id {
            Some(index) => self.worker_contexts[index].lock().unwrap(),
            None => self.shared_context.lock().unwrap(),
        }
    }

    /// Get access to the font context usable outside of the thread pool.
    pub fn lock_shared_context(&self) -> MutexGuard<Context> {
        self.shared_context.lock().unwrap()
    }

    // number of contexts associated to workers
    pub fn num_worker_contexts(&self) -> usize {
        self.worker_contexts.len()
    }

    fn current_worker_id(&self) -> Option<usize> {
        self.workers.current_thread_index()
    }
}

pub struct FontRenderer {
    workers: Arc<ThreadPool>,
    font_contexts: Arc<FontContexts>,

    // Receives the rendered glyphs.
    glyph_rx: Receiver<GlyphRasterJob>,

    // Maintain a set of glyphs that have been requested this
    // frame. This ensures the glyph thread won't rasterize
    // the same glyph more than once in a frame. This is required
    // because the glyph cache hash table is not updated
    // until the end of the frame when we wait for glyph requests
    // to be resolved.
    pending_glyphs: HashSet<GlyphRequest>,

    // We defer removing fonts to the end of the frame so that:
    // - this work is done outside of the critical path,
    // - we don't have to worry about the ordering of events if a font is used on
    //   a frame where it is used (although it seems unlikely).
    fonts_to_remove: Vec<FontKey>,
}

impl FontRenderer {
    pub fn new(workers: Arc<ThreadPool>) -> Self {
        let (glyph_tx, glyph_rx) = channel();

        let num_workers = workers.current_num_threads();
        let mut contexts = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            contexts.push(
                Mutex::new(
                    Context {
                        context: FontContext::new(),
                        tx: glyph_tx.clone(),
                    }
                )
            );
        }

        FontRenderer {
            font_contexts: Arc::new(
                FontContexts {
                    worker_contexts: contexts,
                    shared_context: Mutex::new(
                        Context {
                            context: FontContext::new(),
                            tx: glyph_tx.clone(),
                        }
                    ),
                    workers: Arc::clone(&workers),
                }
            ),
            glyph_rx: glyph_rx,
            pending_glyphs: HashSet::new(),
            workers: workers,
            fonts_to_remove: Vec::new(),
        }
    }

    pub fn add_font(&mut self, font_key: FontKey, template: FontTemplate) {
        let font_contexts = Arc::clone(&self.font_contexts);
        // It's important to synchronously add the font for the shared context because
        // we use it to check that fonts have been properly added when requesting glyphs.
        font_contexts.lock_shared_context().add_font(font_key, &template);
        self.workers.spawn_async(move || {
            // TODO: this locks each font context while adding the font data, probably not a big deal,
            // but if there is contention on this lock we could easily have a queue of per-context
            // operations to add and delete fonts, and have these queues lazily processed by each worker
            // before rendering a glyph.
            //
            // TODO(review):
            // Perhaps this is trying to be overly smart by scheduling this work on the
            // thread pool. It is probably cheap enough to do directly on the render
            // backend thread.
            for i in 0..font_contexts.num_worker_contexts() {
                font_contexts.lock_context(Some(i)).add_font(font_key, &template);
            }
        });
    }

    pub fn delete_font(&mut self, font_key: FontKey) {
        self.fonts_to_remove.push(font_key);
    }

    pub fn request_glyphs(
        &mut self,
        font_key: FontKey,
        size: Au,
        color: ColorF,
        glyph_instances: &[GlyphInstance],
        render_mode: FontRenderMode,
        glyph_options: Option<GlyphOptions>,
    ) {
        assert!(self.font_contexts.lock_shared_context().has_font(&font_key));

        // TODO: This dispatches glyph jobs one by one using spawn_async.
        // It would most likely be more efficient to dispatch one async job
        // that uses rayon's fork-join way of dispatching tasks.

        for glyph_instance in glyph_instances {
            let glyph_key = GlyphRequest::new(
                font_key,
                size,
                color,
                glyph_instance.index,
                glyph_instance.point,
                render_mode,
                glyph_options,
            );

            if self.pending_glyphs.contains(&glyph_key) {
                // We already requested this glyph.
                continue;
            }
            self.pending_glyphs.insert(glyph_key.clone());

            // Dispatch the work.
            let font_contexts = Arc::clone(&self.font_contexts);
            self.workers.spawn_async(move || {
                profile_scope!("glyph");
                let mut font_context = font_contexts.lock_current_context();
                if !font_context.has_font(&font_key) {
                    // In the unlikely event that the font has not been added yet
                    // to this context due to tasks being processed out of order
                    // by the thread pool, just use the shared context which has
                    // its fonts added synchronously.
                    font_context = font_contexts.lock_shared_context();
                }
                font_context.rasterize_glyph(glyph_key);
            });
        }
    }

    pub fn resolve_glyphs(
        &mut self,
        current_frame_id: FrameId,
        glyph_cache: &mut GlyphCache,
        texture_cache: &mut TextureCache,
        texture_cache_profile: &mut TextureCacheProfileCounters,
    ) {
        let mut rasterized_glyphs = Vec::with_capacity(self.pending_glyphs.len());

        // Pull rasterized glyphs from the queue.
        while !self.pending_glyphs.is_empty() {
            // TODO: rather than blocking until all pending glyphs are available
            // we could try_recv and steal work from the thread pool to take advantage
            // of the fact that this thread is alive and we avoid the added latency
            // of blocking it.
            let raster_job = self.glyph_rx.recv().expect("BUG: Should be glyphs pending!");
            debug_assert!(self.pending_glyphs.contains(&raster_job.key));
            self.pending_glyphs.remove(&raster_job.key);
            if let Some(ref v) = raster_job.result {
                debug!("received {}x{} data len {}", v.width, v.height, v.bytes.len());
            }
            rasterized_glyphs.push(raster_job);
        }

        // Ensure that the glyphs are always processed in the same
        // order for a given text run (since iterating a hash set doesn't
        // guarantee order). This can show up as very small float inaccuacry
        // differences in rasterizers due to the different coordinates
        // that text runs get associated with by the texture cache allocator.
        rasterized_glyphs.sort_by(|a, b| a.key.cmp(&b.key));

        // Update the caches.
        for job in rasterized_glyphs {
            let image_id = job.result.and_then(
                |glyph| if glyph.width > 0 && glyph.height > 0 {
                    let image_id = texture_cache.new_item_id();
                    texture_cache.insert(
                        image_id,
                        ImageDescriptor {
                            width: glyph.width,
                            height: glyph.height,
                            stride: None,
                            format: ImageFormat::RGBA8,
                            is_opaque: false,
                            offset: 0,
                        },
                        TextureFilter::Linear,
                        ImageData::Raw(Arc::new(glyph.bytes)),
                        texture_cache_profile,
                    );
                    Some(image_id)
                } else {
                    None
                }
            );

            glyph_cache.insert(job.key, image_id, current_frame_id);
        }

        // Now that we are done with the critical path (rendering the glyphs),
        // we can schedule removing the fonts if needed.
        if !self.fonts_to_remove.is_empty() {
            let font_contexts = Arc::clone(&self.font_contexts);
            let mut fonts_to_remove = mem::replace(&mut self.fonts_to_remove, Vec::new());
            self.workers.spawn_async(move || {
                for &font_key in &fonts_to_remove {
                    font_contexts.lock_shared_context().delete_font(font_key);
                }
                for i in 0..font_contexts.num_worker_contexts() {
                    let mut context = font_contexts.lock_context(Some(i));
                    for &font_key in &fonts_to_remove {
                        context.delete_font(font_key);
                    }
                }
            });
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug, Ord, PartialOrd)]
pub struct GlyphRequest {
    pub key: GlyphKey,
    pub render_mode: FontRenderMode,
    pub glyph_options: Option<GlyphOptions>,
}

impl GlyphRequest {
    pub fn new(
        font_key: FontKey,
        size: Au,
        color: ColorF,
        index: u32,
        point: LayoutPoint,
        render_mode: FontRenderMode,
        glyph_options: Option<GlyphOptions>,
    ) -> GlyphRequest {
        GlyphRequest {
            key: GlyphKey::new(font_key, size, color, index, point, render_mode),
            render_mode: render_mode,
            glyph_options: glyph_options,
        }
    }
}

// TODO(nical): temporarily pub, do not merge.
pub struct GlyphRasterJob {
    pub key: GlyphRequest,
    pub result: Option<RasterizedGlyph>,
}
