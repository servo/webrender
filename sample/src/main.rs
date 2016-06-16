extern crate app_units;
extern crate webrender;
extern crate glutin;
extern crate gleam;
extern crate webrender_traits;
extern crate euclid;

use app_units::Au;
use euclid::{Size2D, Point2D, Rect, Matrix4D};
use gleam::gl;
use std::path::PathBuf;
use std::ffi::CStr;
use webrender_traits::{PipelineId, ServoStackingContextId, StackingContextId, DisplayListId};
use webrender_traits::{AuxiliaryListsBuilder, Epoch, ColorF, FragmentType, GlyphInstance};
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

pub struct WebRenderFrameBuilder {
    pub stacking_contexts: Vec<(StackingContextId, webrender_traits::StackingContext)>,
    pub display_lists: Vec<(DisplayListId, webrender_traits::BuiltDisplayList)>,
    pub auxiliary_lists_builder: AuxiliaryListsBuilder,
    pub root_pipeline_id: PipelineId,
    pub next_scroll_layer_id: usize,
}

impl WebRenderFrameBuilder {
    pub fn new(root_pipeline_id: PipelineId) -> WebRenderFrameBuilder {
        WebRenderFrameBuilder {
            stacking_contexts: vec![],
            display_lists: vec![],
            auxiliary_lists_builder: AuxiliaryListsBuilder::new(),
            root_pipeline_id: root_pipeline_id,
            next_scroll_layer_id: 0,
        }
    }

    pub fn add_stacking_context(&mut self,
                                api: &mut webrender_traits::RenderApi,
                                pipeline_id: PipelineId,
                                stacking_context: webrender_traits::StackingContext)
                                -> StackingContextId {
        assert!(pipeline_id == self.root_pipeline_id);
        let id = api.next_stacking_context_id();
        self.stacking_contexts.push((id, stacking_context));
        id
    }

    pub fn add_display_list(&mut self,
                            api: &mut webrender_traits::RenderApi,
                            display_list: webrender_traits::BuiltDisplayList,
                            stacking_context: &mut webrender_traits::StackingContext)
                            -> DisplayListId {
        let id = api.next_display_list_id();
        stacking_context.has_stacking_contexts = stacking_context.has_stacking_contexts ||
                                                 display_list.descriptor().has_stacking_contexts;
        stacking_context.display_lists.push(id);
        self.display_lists.push((id, display_list));
        id
    }

    pub fn next_scroll_layer_id(&mut self) -> webrender_traits::ScrollLayerId {
        let scroll_layer_id = self.next_scroll_layer_id;
        self.next_scroll_layer_id += 1;
        webrender_traits::ScrollLayerId::new(self.root_pipeline_id, scroll_layer_id)
    }

}

impl webrender_traits::RenderNotifier for Notifier {
    fn new_frame_ready(&mut self) {
        self.window_proxy.wakeup_event_loop();
    }

    fn pipeline_size_changed(&mut self,
                             _: PipelineId,
                             _: Option<Size2D<f32>>) {
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        println!("{} <shader path>", args[0]);
        return;
    }

    let res_path = &args[1];

    let window = glutin::WindowBuilder::new()
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
    println!("Shader resource path: {}", res_path);

    let (width, height) = window.get_inner_size().unwrap();

    let opts = webrender::RendererOptions {
        device_pixel_ratio: 1.0,
        resource_path: PathBuf::from(res_path),
        enable_aa: false,
        enable_msaa: false,
        enable_profiler: false,
    };

    let (mut renderer, sender) = webrender::renderer::Renderer::new(opts);
    let mut api = sender.create_api();

//     let font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf";
//     let font_bytes = load_file(font_path);
//     let font_key = api.add_raw_font(font_bytes);

    let notifier = Box::new(Notifier::new(window.create_window_proxy()));
    renderer.set_render_notifier(notifier);

    let pipeline_id = PipelineId(0, 0);
    let epoch = Epoch(0);
    let root_background_color = ColorF::new(0.3, 0.0, 0.0, 1.0);

    let mut frame_builder = WebRenderFrameBuilder::new(pipeline_id);
    let root_scroll_layer_id = frame_builder.next_scroll_layer_id();

    let bounds = Rect::new(Point2D::new(0.0, 0.0), Size2D::new(width as f32, height as f32));

    let servo_id = ServoStackingContextId(FragmentType::FragmentBody, 0);
    let mut sc =
        webrender_traits::StackingContext::new(servo_id,
                                               Some(root_scroll_layer_id),
                                               webrender_traits::ScrollPolicy::Scrollable,
                                               bounds,
                                               bounds,
                                               0,
                                               &Matrix4D::identity(),
                                               &Matrix4D::identity(),
                                               true,
                                               webrender_traits::MixBlendMode::Normal,
                                               Vec::new(),
                                               &mut frame_builder.auxiliary_lists_builder);

    let mut builder = webrender_traits::DisplayListBuilder::new();

    let clip_region = webrender_traits::ClipRegion::new(&bounds,
                                                        Vec::new(),
                                                        &mut frame_builder.auxiliary_lists_builder);

    builder.push_rect(Rect::new(Point2D::new(100.0, 100.0), Size2D::new(100.0, 100.0)),
                      clip_region,
                      ColorF::new(0.0, 1.0, 0.0, 1.0));

    let text_bounds = Rect::new(Point2D::new(100.0, 200.0), Size2D::new(700.0, 300.0));

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

//     builder.push_text(text_bounds,
//                       clip_region,
//                       glyphs,
//                       font_key,
//                       ColorF::new(1.0, 1.0, 0.0, 1.0),
//                       Au::from_px(32),
//                       Au::from_px(0),
//                       &mut frame_builder.auxiliary_lists_builder);

    frame_builder.add_display_list(&mut api, builder.finalize(), &mut sc);
    let sc_id = frame_builder.add_stacking_context(&mut api, pipeline_id, sc);

    api.set_root_stacking_context(sc_id,
                                  root_background_color,
                                  epoch,
                                  pipeline_id,
                                  Size2D::new(width as f32, height as f32),
                                  frame_builder.stacking_contexts,
                                  frame_builder.display_lists,
                                  frame_builder.auxiliary_lists_builder
                                               .finalize());

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
