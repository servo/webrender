use euclid::{Rect, Size2D};
use frame::Frame;
use internal_types::{FontTemplate, ResultMsg, DrawLayer};
use resource_cache::ResourceCache;
use scene::Scene;
use scoped_threadpool;
use std::cell::Cell;
use std::sync::Arc;
use std::sync::mpsc::{Sender, Receiver};
use texture_cache::{TextureCache, TextureCacheItemId};
use util;
use webrender_traits::{ApiMsg, IdNamespace, ResourceId, RenderApi, RenderNotifier, ScrollLayerId};

/*
    Add items from parent AABB nodes into both children.
        Only end up with items in leaf nodes.
        Need to adjust / change / remove split vs actual rect.

    Compile node:
        Clip to node boundaries.
            No need to worry about overlap!
        Only overlap issues are with layers / aabb trees
            *Should* be detectable when there is a draw list with different scroll layer
                This will work for fixed too...

 */

pub struct RenderBackend {
    api_rx: Receiver<ApiMsg>,
    api_tx: Sender<ApiMsg>,
    result_tx: Sender<ResultMsg>,

    viewport: Rect<i32>,
    device_pixel_ratio: f32,
    next_namespace_id: IdNamespace,

    thread_pool: scoped_threadpool::Pool,
    resource_cache: ResourceCache,

    scene: Scene,
    frame: Frame,
}

impl RenderBackend {
    pub fn new(api_rx: Receiver<ApiMsg>,
               api_tx: Sender<ApiMsg>,
               result_tx: Sender<ResultMsg>,
               viewport: Rect<i32>,
               device_pixel_ratio: f32,
               white_image_id: TextureCacheItemId,
               dummy_mask_image_id: TextureCacheItemId,
               texture_cache: TextureCache) -> RenderBackend {
        let mut thread_pool = scoped_threadpool::Pool::new(8);

        let resource_cache = ResourceCache::new(&mut thread_pool,
                                                texture_cache,
                                                white_image_id,
                                                dummy_mask_image_id,
                                                device_pixel_ratio);

        let backend = RenderBackend {
            thread_pool: thread_pool,
            api_rx: api_rx,
            api_tx: api_tx,
            result_tx: result_tx,
            viewport: viewport,
            device_pixel_ratio: device_pixel_ratio,
            resource_cache: resource_cache,
            scene: Scene::new(),
            frame: Frame::new(),
            next_namespace_id: IdNamespace(1),
        };

        backend
    }

    pub fn run(&mut self, notifier: Box<RenderNotifier>) {
        let mut notifier = notifier;

        loop {
            let msg = self.api_rx.recv();

            match msg {
                Ok(msg) => {
                    match msg {
                        ApiMsg::AddRawFont(id, bytes) => {
                            self.resource_cache
                                .add_font_template(id, FontTemplate::Raw(Arc::new(bytes)));
                        }
                        ApiMsg::AddNativeFont(id, native_font_handle) => {
                            self.resource_cache
                                .add_font_template(id, FontTemplate::Native(native_font_handle));
                        }
                        ApiMsg::AddImage(id, width, height, format, bytes) => {
                            self.resource_cache.add_image_template(id,
                                                                   width,
                                                                   height,
                                                                   format,
                                                                   bytes);
                        }
                        ApiMsg::UpdateImage(id, width, height, format, bytes) => {
                            self.resource_cache.update_image_template(id,
                                                                      width,
                                                                      height,
                                                                      format,
                                                                      bytes);
                        }
                        ApiMsg::AddDisplayList(id,
                                               pipeline_id,
                                               epoch,
                                               display_list_builder) => {
                            self.scene.add_display_list(id,
                                                        pipeline_id,
                                                        epoch,
                                                        display_list_builder,
                                                        &mut self.resource_cache);
                        }
                        ApiMsg::AddStackingContext(id,
                                                   pipeline_id,
                                                   epoch,
                                                   stacking_context) => {
                            self.scene.add_stacking_context(id,
                                                            pipeline_id,
                                                            epoch,
                                                            stacking_context);
                        }
                        ApiMsg::CloneApi(sender) => {
                            let new_api = RenderApi {
                                tx: self.api_tx.clone(),
                                id_namespace: self.next_namespace_id,
                                next_id: Cell::new(ResourceId(0)),
                            };

                            let IdNamespace(id_namespace) = self.next_namespace_id;
                            self.next_namespace_id = IdNamespace(id_namespace + 1);

                            sender.send(new_api).unwrap();
                        }
                        ApiMsg::SetRootStackingContext(stacking_context_id,
                                                       background_color,
                                                       epoch,
                                                       pipeline_id) => {
                            let _pf = util::ProfileScope::new("SetRootStackingContext");

                            self.scene.set_root_stacking_context(pipeline_id,
                                                                 epoch,
                                                                 stacking_context_id,
                                                                 background_color,
                                                                 &mut self.resource_cache);

                            self.build_scene();
                            self.render(&mut *notifier);
                        }
                        ApiMsg::SetRootPipeline(pipeline_id) => {
                            let _pf = util::ProfileScope::new("SetRootPipeline");

                            self.scene.set_root_pipeline_id(pipeline_id);

                            self.build_scene();
                            self.render(&mut *notifier);
                        }
                        ApiMsg::Scroll(delta) => {
                            let _pf = util::ProfileScope::new("Scroll");

                            let viewport_size = Size2D::new(self.viewport.size.width as f32,
                                                            self.viewport.size.height as f32);
                            self.frame.scroll(&delta, &viewport_size);
                            self.render(&mut *notifier);
                        }
                        ApiMsg::TranslatePointToLayerSpace(point, tx) => {
                            // TODO(pcwalton): Select other layers for mouse events.
                            let point = point / self.device_pixel_ratio;
                            match self.frame.layers.get_mut(&ScrollLayerId(0)) {
                                None => tx.send(point).unwrap(),
                                Some(layer) => tx.send(point - layer.scroll_offset).unwrap(),
                            }
                        }
                    }
                }
                Err(..) => {
                    break;
                }
            }
        }
    }

    fn build_scene(&mut self) {
        // Flatten the stacking context hierarchy
        self.frame.create(&self.scene,
                          Size2D::new(self.viewport.size.width as u32,
                                      self.viewport.size.height as u32),
                          &mut self.resource_cache);
    }

    fn render(&mut self, notifier: &mut RenderNotifier) {
        let mut frame = self.frame.build(&self.viewport,
                                         &mut self.resource_cache,
                                         &mut self.thread_pool,
                                         self.device_pixel_ratio);

        // Bit of a hack - if there was nothing visible, at least
        // add one layer to the frame so that the screen gets
        // cleared to the default UA background color. Perhaps
        // there is a better way to handle this...
        if frame.layers.len() == 0 {
            frame.layers.push(DrawLayer {
                texture_id: None,
                size: Size2D::new(self.viewport.size.width as u32,
                                   self.viewport.size.height as u32),
                commands: Vec::new(),
            });
        }

        let pending_update = self.resource_cache.pending_updates();
        if pending_update.updates.len() > 0 {
            self.result_tx.send(ResultMsg::UpdateTextureCache(pending_update)).unwrap();
        }

        let pending_update = self.frame.pending_updates();
        if pending_update.updates.len() > 0 {
            self.result_tx.send(ResultMsg::UpdateBatches(pending_update)).unwrap();
        }

        self.result_tx.send(ResultMsg::NewFrame(frame)).unwrap();
        notifier.new_frame_ready();
    }
}

