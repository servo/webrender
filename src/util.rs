use time::precise_time_ns;
use internal_types::RenderPass;
use types::{ColorF, ImageFormat};

#[allow(dead_code)]
pub struct ProfileScope {
    name: &'static str,
    t0: u64,
}

impl ProfileScope {
    #[allow(dead_code)]
    pub fn new(name: &'static str) -> ProfileScope {
        ProfileScope {
            name: name,
            t0: precise_time_ns(),
        }
    }
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        let t1 = precise_time_ns();
        let ms = (t1 - self.t0) as f64 / 1000000f64;
        //if ms > 0.1 {
            println!("{} {}", self.name, ms);
        //}
    }
}

pub fn get_render_pass(color: &ColorF, format: ImageFormat) -> RenderPass {
    if color.a < 1.0 {
        return RenderPass::Alpha;
    }

    match format {
        ImageFormat::A8 => RenderPass::Alpha,
        ImageFormat::RGBA8 => RenderPass::Alpha,
        ImageFormat::RGB8 => RenderPass::Opaque,
        ImageFormat::Invalid => unreachable!(),
    }
}
