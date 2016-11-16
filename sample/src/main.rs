extern crate webrender;
extern crate glutin;
extern crate gleam;
extern crate webrender_traits;
extern crate euclid;

use euclid::{Size2D, Point2D, Rect, Matrix4D};
use gleam::gl;
use std::path::PathBuf;
use std::ffi::CStr;
use webrender_traits::{AuxiliaryListsBuilder, ColorF, Epoch, GlyphInstance};
use webrender_traits::{ImageData, ImageFormat, PipelineId, RendererKind};
use std::fs::File;
use std::io::Read;
use std::env;

fn _load_file(name: &str) -> Vec<u8> {
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

pub struct WebRenderFrameBuilder {
    pub root_pipeline_id: PipelineId,
    pub next_scroll_layer_id: usize,
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
                             _: Option<Size2D<f32>>) {
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
        gl::clear_color(0.3, 0.0, 0.0, 1.0);
    }

    let version = unsafe {
        let data = CStr::from_ptr(gl::GetString(gl::VERSION) as *const _).to_bytes().to_vec();
        String::from_utf8(data).unwrap()
    };

    println!("OpenGL version {}", version);
    println!("Shader resource path: {:?}", res_path);

    let (width, height) = window.get_inner_size().unwrap();

    let opts = webrender::RendererOptions {
        device_pixel_ratio: 1.0,
        resource_override_path: res_path,
        enable_aa: false,
        enable_msaa: false,
        enable_profiler: false,
        enable_recording: false,
        enable_scrollbars: false,
        debug: true,
        precache_shaders: true,
        renderer_kind: RendererKind::Native,
        enable_subpixel_aa: false,
    };

    let (mut renderer, sender) = webrender::renderer::Renderer::new(opts);
    let api = sender.create_api();

//     let font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf";
//     let font_bytes = load_file(font_path);
//     let font_key = api.add_raw_font(font_bytes);

    let notifier = Box::new(Notifier::new(window.create_window_proxy()));
    renderer.set_render_notifier(notifier);

    let pipeline_id = PipelineId(0, 0);
    let epoch = Epoch(0);
    let root_background_color = ColorF::new(0.3, 0.0, 0.0, 1.0);

    let mut auxiliary_lists_builder = AuxiliaryListsBuilder::new();
    let mut builder = webrender_traits::DisplayListBuilder::new();

    let root_scroll_layer_id =
        webrender_traits::ScrollLayerId::new(pipeline_id, 0,
                                             webrender_traits::ServoScrollRootId(0));

    let bounds = Rect::new(Point2D::new(0.0, 0.0), Size2D::new(width as f32, height as f32));
    builder.push_stacking_context(
        webrender_traits::StackingContext::new(Some(root_scroll_layer_id),
                                               webrender_traits::ScrollPolicy::Scrollable,
                                               bounds,
                                               bounds,
                                               0,
                                               &Matrix4D::identity(),
                                               &Matrix4D::identity(),
                                               webrender_traits::MixBlendMode::Normal,
                                               Vec::new(),
                                               &mut auxiliary_lists_builder));

    let clip_region = {
        let mask = webrender_traits::ImageMask {
            image: api.add_image(2, 2, None, ImageFormat::A8, ImageData::Raw(vec![0,80, 180, 255])),
            rect: Rect::new(Point2D::new(75.0, 75.0), Size2D::new(100.0, 100.0)),
            repeat: false,
        };
        let radius = webrender_traits::BorderRadius::uniform(20.0);
        let complex = webrender_traits::ComplexClipRegion::new(
            Rect::new(Point2D::new(50.0, 50.0), Size2D::new(100.0, 100.0)),
            radius);

        webrender_traits::ClipRegion::new(&bounds,
                                          vec![complex],
                                          Some(mask),
                                          &mut auxiliary_lists_builder)
    };

    builder.push_rect(Rect::new(Point2D::new(100.0, 100.0), Size2D::new(100.0, 100.0)),
                      clip_region,
                      ColorF::new(0.0, 1.0, 0.0, 1.0));

    let _text_bounds = Rect::new(Point2D::new(100.0, 200.0), Size2D::new(700.0, 300.0));

    let _glyphs = vec![
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

//     builder.push_text(text_bounds,
//                       clip_region,
//                       glyphs,
//                       font_key,
//                       ColorF::new(1.0, 1.0, 0.0, 1.0),
//                       Au::from_px(32),
//                       Au::from_px(0),
//                       &mut frame_builder.auxiliary_lists_builder);

    builder.pop_stacking_context();

    api.set_root_display_list(
        root_background_color,
        epoch,
        pipeline_id,
        Size2D::new(width as f32, height as f32),
        builder.finalize(),
        auxiliary_lists_builder.finalize());
    api.set_root_pipeline(pipeline_id);

    for event in window.wait_events() {
        gl::clear(gl::COLOR_BUFFER_BIT);
        renderer.update();

        renderer.render(Size2D::new(width, height));

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
