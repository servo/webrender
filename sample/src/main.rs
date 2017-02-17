/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate app_units;
extern crate euclid;
extern crate gleam;
extern crate glutin;
extern crate webrender;
extern crate webrender_traits;

use app_units::Au;
use euclid::Point2D;
use gleam::gl;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;
use webrender_traits::{BlobImageResult, BlobImageError, BlobImageDescriptor};
use webrender_traits::{ColorF, Epoch, GlyphInstance, ClipRegion, ImageRendering};
use webrender_traits::{ImageDescriptor, ImageData, ImageFormat, PipelineId};
use webrender_traits::{ImageKey, BlobImageData, BlobImageRenderer, RasterizedBlobImage};
use webrender_traits::{LayoutSize, LayoutPoint, LayoutRect, LayoutTransform, DeviceUintSize};


fn load_file(name: &str) -> Vec<u8> {
    let mut file = File::open(name).unwrap();
    let mut buffer = vec![];
    file.read_to_end(&mut buffer).unwrap();
    buffer
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

fn main() {
    let args: Vec<String> = env::args().collect();
    let res_path = if args.len() > 1 {
        Some(PathBuf::from(&args[1]))
    } else {
        None
    };

    let window = glutin::WindowBuilder::new()
                .with_title("WebRender Sample")
                .with_gl(glutin::GlRequest::GlThenGles {
                    opengl_version: (3, 2),
                    opengles_version: (3, 0)
                })
                .build()
                .unwrap();

    unsafe {
        window.make_current().ok();
    }
    // Android uses the static generator (as opposed to a global generator) at the moment
    #[cfg(not(target_os = "android"))]
    gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);

    println!("OpenGL version {}", gl::get_string(gl::VERSION));
    println!("Shader resource path: {:?}", res_path);

    let (width, height) = window.get_inner_size().unwrap();

    let opts = webrender::RendererOptions {
        resource_override_path: res_path,
        debug: true,
        precache_shaders: true,
        blob_image_renderer: Some(Box::new(FakeBlobImageRenderer::new())),
        .. Default::default()
    };

    let (mut renderer, sender) = webrender::renderer::Renderer::new(opts).unwrap();
    let api = sender.create_api();

    let notifier = Box::new(Notifier::new(window.create_window_proxy()));
    renderer.set_render_notifier(notifier);

    let epoch = Epoch(0);
    let root_background_color = ColorF::new(0.3, 0.0, 0.0, 1.0);

    let vector_img = api.generate_image_key();
    api.add_image(
        vector_img,
        ImageDescriptor::new(100, 100, ImageFormat::RGBA8).with_opaque_flag(true),
        ImageData::new_blob_image(Vec::new()),
        None,
    );

    let pipeline_id = PipelineId(0, 0);
    let mut builder = webrender_traits::DisplayListBuilder::new(pipeline_id);

    let bounds = LayoutRect::new(LayoutPoint::new(0.0, 0.0), LayoutSize::new(width as f32, height as f32));
    let clip_region = {
        let complex = webrender_traits::ComplexClipRegion::new(
            LayoutRect::new(LayoutPoint::new(50.0, 50.0), LayoutSize::new(100.0, 100.0)),
            webrender_traits::BorderRadius::uniform(20.0));

        builder.new_clip_region(&bounds, vec![complex], None)
    };

    builder.push_stacking_context(webrender_traits::ScrollPolicy::Scrollable,
                                  bounds,
                                  clip_region,
                                  0,
                                  LayoutTransform::identity().into(),
                                  LayoutTransform::identity(),
                                  webrender_traits::MixBlendMode::Normal,
                                  Vec::new());
    builder.push_image(
        LayoutRect::new(LayoutPoint::new(0.0, 0.0), LayoutSize::new(100.0, 100.0)),
        ClipRegion::simple(&bounds),
        LayoutSize::new(100.0, 100.0),
        LayoutSize::new(0.0, 0.0),
        ImageRendering::Auto,
        vector_img,
    );

    let sub_clip = {
        let mask_image = api.generate_image_key();
        api.add_image(
            mask_image,
            ImageDescriptor::new(2, 2, ImageFormat::A8).with_opaque_flag(true),
            ImageData::new(vec![0, 80, 180, 255]),
            None,
        );
        let mask = webrender_traits::ImageMask {
            image: mask_image,
            rect: LayoutRect::new(LayoutPoint::new(75.0, 75.0), LayoutSize::new(100.0, 100.0)),
            repeat: false,
        };
        let complex = webrender_traits::ComplexClipRegion::new(
            LayoutRect::new(LayoutPoint::new(50.0, 50.0), LayoutSize::new(100.0, 100.0)),
            webrender_traits::BorderRadius::uniform(20.0));

        builder.new_clip_region(&bounds, vec![complex], Some(mask))
    };

    builder.push_rect(LayoutRect::new(LayoutPoint::new(100.0, 100.0), LayoutSize::new(100.0, 100.0)),
                      sub_clip,
                      ColorF::new(0.0, 1.0, 0.0, 1.0));
    builder.push_rect(LayoutRect::new(LayoutPoint::new(250.0, 100.0), LayoutSize::new(100.0, 100.0)),
                      sub_clip,
                      ColorF::new(0.0, 1.0, 0.0, 1.0));
    let border_side = webrender_traits::BorderSide {
        color: ColorF::new(0.0, 0.0, 1.0, 1.0),
        style: webrender_traits::BorderStyle::Groove,
    };
    let border_widths = webrender_traits::BorderWidths {
        top: 10.0,
        left: 10.0,
        bottom: 10.0,
        right: 10.0,
    };
    let border_details = webrender_traits::BorderDetails::Normal(webrender_traits::NormalBorder {
        top: border_side,
        right: border_side,
        bottom: border_side,
        left: border_side,
        radius: webrender_traits::BorderRadius::uniform(20.0),
    });
    builder.push_border(LayoutRect::new(LayoutPoint::new(100.0, 100.0), LayoutSize::new(100.0, 100.0)),
                        sub_clip,
                        border_widths,
                        border_details);


    if false { // draw text?
        let font_key = api.generate_font_key();
        let font_bytes = load_file("res/FreeSans.ttf");
        api.add_raw_font(font_key, font_bytes);

        let text_bounds = LayoutRect::new(LayoutPoint::new(100.0, 200.0), LayoutSize::new(700.0, 300.0));

        let glyphs = vec![
            GlyphInstance {
                index: 48,
                point: Point2D::new(100.0, 100.0),
            },
            GlyphInstance {
                index: 68,
                point: Point2D::new(150.0, 100.0),
            },
            GlyphInstance {
                index: 80,
                point: Point2D::new(200.0, 100.0),
            },
            GlyphInstance {
                index: 82,
                point: Point2D::new(250.0, 100.0),
            },
            GlyphInstance {
                index: 81,
                point: Point2D::new(300.0, 100.0),
            },
            GlyphInstance {
                index: 3,
                point: Point2D::new(350.0, 100.0),
            },
            GlyphInstance {
                index: 86,
                point: Point2D::new(400.0, 100.0),
            },
            GlyphInstance {
                index: 79,
                point: Point2D::new(450.0, 100.0),
            },
            GlyphInstance {
                index: 72,
                point: Point2D::new(500.0, 100.0),
            },
            GlyphInstance {
                index: 83,
                point: Point2D::new(550.0, 100.0),
            },
            GlyphInstance {
                index: 87,
                point: Point2D::new(600.0, 100.0),
            },
            GlyphInstance {
                index: 17,
                point: Point2D::new(650.0, 100.0),
            },
        ];

        builder.push_text(text_bounds,
                          webrender_traits::ClipRegion::simple(&bounds),
                          glyphs,
                          font_key,
                          ColorF::new(1.0, 1.0, 0.0, 1.0),
                          Au::from_px(32),
                          Au::from_px(0),
                          None);
    }

    builder.pop_stacking_context();

    api.set_root_display_list(
        Some(root_background_color),
        epoch,
        LayoutSize::new(width as f32, height as f32),
        builder,
        true);
    api.set_root_pipeline(pipeline_id);
    api.generate_frame(None);

    for event in window.wait_events() {
        renderer.update();

        renderer.render(DeviceUintSize::new(width, height));

        window.swap_buffers().ok();

        match event {
            glutin::Event::Closed |
            glutin::Event::KeyboardInput(_, _, Some(glutin::VirtualKeyCode::Escape)) |
            glutin::Event::KeyboardInput(_, _, Some(glutin::VirtualKeyCode::Q)) => break,
            _ => ()
        }
    }
}

struct FakeBlobImageRenderer {
    images: HashMap<ImageKey, BlobImageResult>,
}

impl FakeBlobImageRenderer {
    fn new() -> Self {
        FakeBlobImageRenderer { images: HashMap::new() }
    }
}

impl BlobImageRenderer for FakeBlobImageRenderer {
    fn request_blob_image(&mut self, key: ImageKey, _: Arc<BlobImageData>, descriptor: &BlobImageDescriptor) {
        let mut texels = Vec::with_capacity((descriptor.width * descriptor.height * 4) as usize);
        for y in 0..descriptor.height {
            for x in 0..descriptor.width {
                // render a simple checkerboard pattern
                let a = if (x % 20 >= 10) != (y % 20 >= 10) { 255 } else { 0 };
                match descriptor.format {
                    ImageFormat::RGBA8 => {
                        texels.push(a);
                        texels.push(a);
                        texels.push(a);
                        texels.push(255);
                    }
                    ImageFormat::A8 => {
                        texels.push(a);
                    }
                    _ => {
                        self.images.insert(key,
                            Err(BlobImageError::Other(format!(
                                "Usupported image format {:?}",
                                descriptor.format
                            )))
                        );
                        return;
                    }
                }
            }
        }

        self.images.insert(key, Ok(RasterizedBlobImage {
            data: texels,
            width: descriptor.width,
            height: descriptor.height,
        }));
    }

    fn resolve_blob_image(&mut self, key: ImageKey) -> BlobImageResult {
        self.images.remove(&key).unwrap_or(Err(BlobImageError::InvalidKey))
    }
}
