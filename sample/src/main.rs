extern crate app_units;
extern crate webrender;
extern crate glutin;
extern crate gleam;
extern crate webrender_traits;
extern crate euclid;

use app_units::Au;
use gleam::gl;
use std::path::PathBuf;
use webrender_traits::{ColorF, Epoch, GlyphInstance};
use webrender_traits::{ImageData, ImageFormat, PipelineId, RendererKind};
use webrender_traits::{LayoutSize, LayoutPoint, LayoutRect, LayoutTransform, DeviceUintSize};
use std::fs::File;
use std::io::Read;
use std::env;

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
        self.window_proxy.wakeup_event_loop();
    }

    fn new_scroll_frame_ready(&mut self, _composite_needed: bool) {
        self.window_proxy.wakeup_event_loop();
    }

    fn pipeline_size_changed(&mut self,
                             _: PipelineId,
                             _: Option<LayoutSize>) {
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
                .with_gl(glutin::GlRequest::Specific(glutin::Api::OpenGl, (3, 2)))
                .build()
                .unwrap();

    unsafe {
        window.make_current().ok();
        gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);
    }

    println!("OpenGL version {}", gl::get_string(gl::VERSION));
    println!("Shader resource path: {:?}", res_path);

    let (width, height) = window.get_inner_size().unwrap();

    let opts = webrender::RendererOptions {
        device_pixel_ratio: 1.0,
        resource_override_path: res_path,
        enable_aa: false,
        enable_profiler: false,
        enable_recording: false,
        enable_scrollbars: false,
        debug: true,
        precache_shaders: true,
        renderer_kind: RendererKind::Native,
        enable_subpixel_aa: false,
        clear_background: true,
        clear_framebuffer: true,
        clear_color: ColorF::new(1.0, 1.0, 1.0, 1.0),
        render_target_debug: false,
    };

    let (mut renderer, sender) = webrender::renderer::Renderer::new(opts);
    let api = sender.create_api();

    let notifier = Box::new(Notifier::new(window.create_window_proxy()));
    renderer.set_render_notifier(notifier);

    let epoch = Epoch(0);
    let root_background_color = ColorF::new(0.3, 0.0, 0.0, 1.0);

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
                                  &LayoutTransform::identity(),
                                  &LayoutTransform::identity(),
                                  webrender_traits::MixBlendMode::Normal,
                                  Vec::new());

    let sub_clip = {
        let mask = webrender_traits::ImageMask {
            image: api.add_image(2, 2, None, ImageFormat::A8, ImageData::new(vec![0,80, 180, 255])),
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
        width: 10.0,
        color: ColorF::new(0.0, 0.0, 1.0, 1.0),
        style: webrender_traits::BorderStyle::Groove,
    };
    builder.push_border(LayoutRect::new(LayoutPoint::new(100.0, 100.0), LayoutSize::new(100.0, 100.0)),
                        sub_clip,
                        border_side,
                        border_side,
                        border_side,
                        border_side,
                        webrender_traits::BorderRadius::uniform(20.0));


    if false { // draw text?
        let font_bytes = load_file("res/FreeSans.ttf");
        let font_key = api.add_raw_font(font_bytes);

        let text_bounds = LayoutRect::new(LayoutPoint::new(100.0, 200.0), LayoutSize::new(700.0, 300.0));

        let glyphs = vec![
            GlyphInstance {
                index: 48,
                x: 100.0,
                y: 100.0,
            },
            GlyphInstance {
                index: 68,
                x: 150.0,
                y: 100.0,
            },
            GlyphInstance {
                index: 80,
                x: 200.0,
                y: 100.0,
            },
            GlyphInstance {
                index: 82,
                x: 250.0,
                y: 100.0,
            },
            GlyphInstance {
                index: 81,
                x: 300.0,
                y: 100.0,
            },
            GlyphInstance {
                index: 3,
                x: 350.0,
                y: 100.0,
            },
            GlyphInstance {
                index: 86,
                x: 400.0,
                y: 100.0,
            },
            GlyphInstance {
                index: 79,
                x: 450.0,
                y: 100.0,
            },
            GlyphInstance {
                index: 72,
                x: 500.0,
                y: 100.0,
            },
            GlyphInstance {
                index: 83,
                x: 550.0,
                y: 100.0,
            },
            GlyphInstance {
                index: 87,
                x: 600.0,
                y: 100.0,
            },
            GlyphInstance {
                index: 17,
                x: 650.0,
                y: 100.0,
            },
        ];

        builder.push_text(text_bounds,
                          webrender_traits::ClipRegion::simple(&bounds),
                          glyphs,
                          font_key,
                          ColorF::new(1.0, 1.0, 0.0, 1.0),
                          Au::from_px(32),
                          Au::from_px(0));
    }

    builder.pop_stacking_context();

    api.set_root_display_list(
        Some(root_background_color),
        epoch,
        LayoutSize::new(width as f32, height as f32),
        builder);
    api.set_root_pipeline(pipeline_id);

    for event in window.wait_events() {
        renderer.update();

        renderer.render(DeviceUintSize::new(width, height));

        window.swap_buffers().ok();

        match event {
            glutin::Event::Closed => break,
            glutin::Event::KeyboardInput(_element_state, scan_code, _virtual_key_code) => {
                if scan_code == 9 {
                    break;
                }
            }
            _ => ()
        }
    }
}
