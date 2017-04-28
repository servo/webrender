/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate app_units;
extern crate euclid;
extern crate gleam;
extern crate glutin;
extern crate webrender;
extern crate webrender_traits;
extern crate threadpool;

use gleam::gl;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{channel, Sender, Receiver};
use webrender_traits::{BlobImageData, BlobImageDescriptor, BlobImageError, BlobImageRenderer, BlobImageRequest};
use webrender_traits::{BlobImageResult, TileOffset, ImageStore, ClipRegion, ColorF, ColorU, Epoch};
use webrender_traits::{DeviceUintSize, DeviceUintRect, LayoutPoint, LayoutRect, LayoutSize};
use webrender_traits::{ImageData, ImageDescriptor, ImageFormat, ImageRendering, ImageKey, TileSize};
use webrender_traits::{PipelineId, RasterizedBlobImage, TransformStyle};
use threadpool::ThreadPool;

// This example shows how to implement a very basic BlobImageRenderer that can only render
// a checkerboard pattern.

// The deserialized command list internally used by this example is just a color.
type ImageRenderingCommands = ColorU;

// Serialize/deserialze the blob.
// Ror real usecases you should probably use serde rather than doing it by hand.

fn serialize_blob(color: ColorU) -> Vec<u8> {
    vec![color.r, color.g, color.b, color.a]
}

fn deserialize_blob(blob: &[u8]) -> Result<ImageRenderingCommands, ()> {
    let mut iter = blob.iter();
    return match (iter.next(), iter.next(), iter.next(), iter.next()) {
        (Some(&r), Some(&g), Some(&b), Some(&a)) => Ok(ColorU::new(r, g, b, a)),
        (Some(&a), None, None, None) => Ok(ColorU::new(a, a, a, a)),
        _ => Err(()),
    }
}

// This is the function that applies the deserialized drawing commands and generates
// actual image data.
fn render_blob(
    commands: Arc<ImageRenderingCommands>,
    descriptor: &BlobImageDescriptor,
    tile: Option<TileOffset>,
) -> BlobImageResult {
    let color = *commands;

    // Allocate storage for the result. Right now the resource cache expects the
    // tiles to have have no stride or offset.
    let mut texels = Vec::with_capacity((descriptor.width * descriptor.height * 4) as usize);

    // Generate a per-tile pattern to see it in the demo. For a real use case it would not
    // make sense for the rendered content to depend on its tile.
    let tile_checker = match tile {
        Some(tile) => (tile.x % 2 == 0) != (tile.y % 2 == 0),
        None => true,
    };

    for y in 0..descriptor.height {
        for x in 0..descriptor.width {
            // Apply the tile's offset. This is important: all drawing commands should be
            // translated by this offset to give correct results with tiled blob images.
            let x2 = x + descriptor.offset.x as u32;
            let y2 = y + descriptor.offset.y as u32;

            // Render a simple checkerboard pattern
            let checker = if (x2 % 20 >= 10) != (y2 % 20 >= 10) { 1 } else { 0 };
            // ..nested in the per-tile cherkerboard pattern
            let tc = if tile_checker { 0 } else { (1 - checker) * 40 };

            match descriptor.format {
                ImageFormat::RGBA8 => {
                    texels.push(color.b * checker + tc);
                    texels.push(color.g * checker + tc);
                    texels.push(color.r * checker + tc);
                    texels.push(color.a * checker + tc);
                }
                ImageFormat::A8 => {
                    texels.push(color.a * checker + tc);
                }
                _ => {
                    return Err(BlobImageError::Other(format!(
                        "Usupported image format {:?}",
                        descriptor.format
                    )));
                }
            }
        }
    }

    return Ok(RasterizedBlobImage {
        data: texels,
        width: descriptor.width,
        height: descriptor.height,
    });
}

struct CheckerboardRenderer {
    // We are going to defer the rendering work to worker threads.
    // using a pre-built Arc<Mutex<ThreadPool>> rather than creating our own threads
    // makes it possible to share the same thread pool as the glyph renderer (if we
    // want to).
    workers: Arc<Mutex<ThreadPool>>,

    // the workers will use an mpsc channel to communicate the result.
    tx: Sender<(BlobImageRequest, BlobImageResult)>,
    rx: Receiver<(BlobImageRequest, BlobImageResult)>,

    // The deserialized drawing commands.
    // In this example we store them in Arcs. This isn't necessary since in this simplified
    // case the command list is a simple 32 bits value and would be cheap to clone before sending
    // to the workers. But in a more realistic scenario the commands would typically be bigger
    // and more expensive to clone, so let's pretend it is also the case here.
    image_cmds: HashMap<ImageKey, Arc<ImageRenderingCommands>>,

    // The images rendered in the current frame (not kept here between frames).
    rendered_images: HashMap<BlobImageRequest, Option<BlobImageResult>>,
}

impl CheckerboardRenderer {
    fn new(workers: Arc<Mutex<ThreadPool>>) -> Self {
        let (tx, rx) = channel();
        CheckerboardRenderer {
            image_cmds: HashMap::new(),
            rendered_images: HashMap::new(),
            workers: workers,
            tx: tx,
            rx: rx,
        }
    }
}

impl BlobImageRenderer for CheckerboardRenderer {
    fn add(&mut self, key: ImageKey, cmds: BlobImageData, _: Option<TileSize>) {
        self.image_cmds.insert(key, Arc::new(deserialize_blob(&cmds[..]).unwrap()));
    }

    fn update(&mut self, key: ImageKey, cmds: BlobImageData) {
        // Here, updating is just replacing the current version of the commands with
        // the new one (no incremental updates).
        self.image_cmds.insert(key, Arc::new(deserialize_blob(&cmds[..]).unwrap()));
    }

    fn delete(&mut self, key: ImageKey) {
        self.image_cmds.remove(&key);
    }

    fn request(&mut self,
               request: BlobImageRequest,
               descriptor: &BlobImageDescriptor,
               _dirty_rect: Option<DeviceUintRect>,
               _images: &ImageStore) {

        // Gather the input data to send to a worker thread.
        let cmds = Arc::clone(&self.image_cmds.get(&request.key).unwrap());
        let tx = self.tx.clone();
        let descriptor = descriptor.clone();

        self.workers.lock().unwrap().execute(move || {
            let result = render_blob(cmds, &descriptor, request.tile);
            tx.send((request, result)).unwrap();
        });

        // Add None in the map of rendered images. This makes it possible to differentiate
        // between commands that aren't finished yet (entry in the map is equal to None) and
        // keys that have never been requested (entry not in the map), which would cause deadlocks
        // if we were to block upon receing their result in resolve!
        self.rendered_images.insert(request, None);
    }

    fn resolve(&mut self, request: BlobImageRequest) -> BlobImageResult {
        // First look at whether we have already received the rendered image
        // that we are loooking for.
        match self.rendered_images.entry(request) {
            Entry::Vacant(_) => {
                return Err(BlobImageError::InvalidKey);
            }
            Entry::Occupied(entry) => {
                // None means we haven't yet received the result.
                if entry.get().is_some() {
                    let result = entry.remove();
                    return result.unwrap();
                }
            }
        }

        // We haven't received it yet, pull from the channel until we receive it.
        while let Ok((req, result)) = self.rx.recv() {
            if req == request {
                // There it is!
                return result
            }
            self.rendered_images.insert(req, Some(result));
        }

        // if we break out of the loop above it means the channel closed unexpectedly.
        return Err(BlobImageError::Other("Channel closed".into()));
    }
}

fn main() {
    let window = glutin::WindowBuilder::new()
                .with_title("WebRender Sample (BlobImageRenderer)")
                .with_multitouch()
                .with_gl(glutin::GlRequest::GlThenGles {
                    opengl_version: (3, 2),
                    opengles_version: (3, 0)
                })
                .build()
                .unwrap();

    unsafe {
        window.make_current().ok();
    }

    let gl = match gl::GlType::default() {
        gl::GlType::Gl => unsafe { gl::GlFns::load_with(|symbol| window.get_proc_address(symbol) as *const _) },
        gl::GlType::Gles => unsafe { gl::GlesFns::load_with(|symbol| window.get_proc_address(symbol) as *const _) },
    };

    println!("OpenGL version {}", gl.get_string(gl::VERSION));

    let (width, height) = window.get_inner_size_pixels().unwrap();

    let workers = Arc::new(Mutex::new(ThreadPool::new_with_name("Worker".to_string(), 4)));

    let opts = webrender::RendererOptions {
        debug: true,
        workers: Some(Arc::clone(&workers)),
        // Register our blob renderer, so that WebRender integrates it in the resource cache..
        // Share the same pool of worker threads between WebRender and our blob renderer.
        blob_image_renderer: Some(Box::new(CheckerboardRenderer::new(Arc::clone(&workers)))),
        device_pixel_ratio: window.hidpi_factor(),
        .. Default::default()
    };

    let size = DeviceUintSize::new(width, height);
    let (mut renderer, sender) = webrender::renderer::Renderer::new(gl, opts, size).unwrap();
    let api = sender.create_api();

    let notifier = Box::new(Notifier::new(window.create_window_proxy()));
    renderer.set_render_notifier(notifier);

    let epoch = Epoch(0);
    let root_background_color = ColorF::new(0.2, 0.2, 0.2, 1.0);

    let blob_img1 = api.generate_image_key();
    api.add_image(
        blob_img1,
        ImageDescriptor::new(500, 500, ImageFormat::RGBA8, true),
        ImageData::new_blob_image(serialize_blob(ColorU::new(50, 50, 150, 255))),
        Some(128),
    );

    let blob_img2 = api.generate_image_key();
    api.add_image(
        blob_img2,
        ImageDescriptor::new(200, 200, ImageFormat::RGBA8, true),
        ImageData::new_blob_image(serialize_blob(ColorU::new(50, 150, 50, 255))),
        None,
    );

    let pipeline_id = PipelineId(0, 0);
    let mut builder = webrender_traits::DisplayListBuilder::new(pipeline_id);

    let bounds = LayoutRect::new(LayoutPoint::zero(), LayoutSize::new(width as f32, height as f32));
    builder.push_stacking_context(webrender_traits::ScrollPolicy::Scrollable,
                                  bounds,
                                  None,
                                  TransformStyle::Flat,
                                  None,
                                  webrender_traits::MixBlendMode::Normal,
                                  Vec::new());
    builder.push_image(
        LayoutRect::new(LayoutPoint::new(30.0, 30.0), LayoutSize::new(500.0, 500.0)),
        ClipRegion::simple(&bounds),
        LayoutSize::new(500.0, 500.0),
        LayoutSize::new(0.0, 0.0),
        ImageRendering::Auto,
        blob_img1,
    );

    builder.push_image(
        LayoutRect::new(LayoutPoint::new(600.0, 60.0), LayoutSize::new(200.0, 200.0)),
        ClipRegion::simple(&bounds),
        LayoutSize::new(200.0, 200.0),
        LayoutSize::new(0.0, 0.0),
        ImageRendering::Auto,
        blob_img2,
    );

    builder.pop_stacking_context();

    api.set_display_list(
        Some(root_background_color),
        epoch,
        LayoutSize::new(width as f32, height as f32),
        builder.finalize(),
        true);
    api.set_root_pipeline(pipeline_id);
    api.generate_frame(None);

    'outer: for event in window.wait_events() {
        let mut events = Vec::new();
        events.push(event);

        for event in window.poll_events() {
            events.push(event);
        }

        for event in events {
            match event {
                glutin::Event::Closed |
                glutin::Event::KeyboardInput(_, _, Some(glutin::VirtualKeyCode::Escape)) |
                glutin::Event::KeyboardInput(_, _, Some(glutin::VirtualKeyCode::Q)) => break 'outer,
                glutin::Event::KeyboardInput(glutin::ElementState::Pressed,
                                             _, Some(glutin::VirtualKeyCode::P)) => {
                    let enable_profiler = !renderer.get_profiler_enabled();
                    renderer.set_profiler_enabled(enable_profiler);
                    api.generate_frame(None);
                }
                _ => ()
            }
        }

        renderer.update();
        renderer.render(DeviceUintSize::new(width, height));
        window.swap_buffers().ok();
    }
}

struct Notifier {
    window_proxy: glutin::WindowProxy,
}

impl Notifier {
    fn new(window_proxy: glutin::WindowProxy) -> Notifier {
        Notifier {
            window_proxy: window_proxy,
        }
    }
}

impl webrender_traits::RenderNotifier for Notifier {
    fn new_frame_ready(&mut self) {
        #[cfg(not(target_os = "android"))]
        self.window_proxy.wakeup_event_loop();
    }

    fn new_scroll_frame_ready(&mut self, _composite_needed: bool) {
        #[cfg(not(target_os = "android"))]
        self.window_proxy.wakeup_event_loop();
    }
}
