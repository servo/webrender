use std::{path::PathBuf, sync::Arc};
use api::{
    ColorU, FontInstanceFlags, FontInstanceKey, FontKey, FontRenderMode, GlyphDimensions,
    NativeFontHandle,
};
use memmap2::Mmap;
use skrifa::charmap::Charmap;
use skrifa::metrics::GlyphMetrics;
use skrifa::outline::{DrawSettings, Engine, HintingInstance, HintingOptions, OutlinePen, SmoothMode, Target};
use skrifa::prelude::{LocationRef, Size};
use skrifa::raw::{FileRef};
use skrifa::{GlyphId, MetadataProvider as _, Tag};
use vello_cpu::kurbo::{expand_path, Affine, BezPath, Diagonal2, Join, Shape as _};
use vello_cpu::peniko::{self, Blob};
use vello_cpu::{PaintType, PixmapMut, RenderContext, RenderSettings, Resources};

use crate::{
    FastHashMap, FontInstance, GlyphFormat, GlyphKey, GlyphRasterError, GlyphRasterResult,
    RasterizedGlyph,
};

type PenikoFont = vello_cpu::peniko::FontData;

/// Resolve the hinting settings for a font instance, following the same
/// rules as the FreeType backend: platform options select the hinting
/// target, FORCE_AUTOHINT/NO_AUTOHINT select the engine, and hinting is
/// disabled entirely under non-axis-aligned transforms or synthetic
/// italics. Returns `None` when hinting is disabled, otherwise the
/// options along with a small tag for use in the hinting-instance cache
/// key.
fn hinting_options(font: &FontInstance) -> Option<(HintingOptions, u32)> {
    // Disable hinting if there is a non-axis-aligned transform.
    if font.synthetic_italics.is_enabled() ||
        ((font.transform.scale_x != 0.0 || font.transform.scale_y != 0.0) &&
         (font.transform.skew_x != 0.0 || font.transform.skew_y != 0.0))
    {
        return None;
    }

    // Hinting platform options only exist on FreeType platforms; elsewhere
    // use the standard smooth target.
    #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "windows")))]
    let target = {
        use api::FontHinting;
        let hinting = font.platform_options.unwrap_or_default().hinting;
        match (hinting, font.render_mode) {
            (FontHinting::None, _) => return None,
            (FontHinting::Mono, _) => Target::Mono,
            (FontHinting::Light, _) => Target::from(SmoothMode::Light),
            (FontHinting::LCD, FontRenderMode::Subpixel) => {
                if font.flags.contains(FontInstanceFlags::LCD_VERTICAL) {
                    Target::from(SmoothMode::VerticalLcd)
                } else {
                    Target::from(SmoothMode::Lcd)
                }
            }
            _ => Target::from(SmoothMode::Normal),
        }
    };
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "windows"))]
    let target = Target::from(SmoothMode::Normal);
    let engine = if font.flags.contains(FontInstanceFlags::FORCE_AUTOHINT) {
        Engine::Auto(None)
    } else if font.flags.contains(FontInstanceFlags::NO_AUTOHINT) {
        Engine::Interpreter
    } else {
        Engine::AutoFallback
    };

    let target_tag = match target {
        Target::Mono => 0,
        Target::Smooth { mode: SmoothMode::Normal, .. } => 1,
        Target::Smooth { mode: SmoothMode::Light, .. } => 2,
        Target::Smooth { mode: SmoothMode::Lcd, .. } => 3,
        Target::Smooth { mode: SmoothMode::VerticalLcd, .. } => 4,
    };
    let engine_tag = match engine {
        Engine::Interpreter => 0,
        Engine::Auto(_) => 1,
        Engine::AutoFallback => 2,
    };

    Some((HintingOptions { engine, target }, target_tag | (engine_tag << 3)))
}

#[derive(Clone, Default)]
pub(crate) struct OutlinePath {
    pub(crate) path: BezPath,
}

impl OutlinePath {
    pub(crate) fn reuse(&mut self) {
        self.path.truncate(0);
    }
}

// Note that we flip the y-axis to match our coordinate system (y-down, origin
// at the baseline).
impl OutlinePen for OutlinePath {
    #[inline]
    fn move_to(&mut self, x: f32, y: f32) {
        self.path.move_to((x, -y));
    }

    #[inline]
    fn line_to(&mut self, x: f32, y: f32) {
        self.path.line_to((x, -y));
    }

    #[inline]
    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.path.curve_to((cx0, -cy0), (cx1, -cy1), (x, -y));
    }

    #[inline]
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        self.path.quad_to((cx, -cy), (x, -y));
    }

    #[inline]
    fn close(&mut self) {
        self.path.close_path();
    }
}

// struct CachedFont {
//     pub data: Arc<dyn AsRef<[u8]> + Send + Sync>,
//     pub index: u32,
//     pub settings: FontRenderSettings,
// }

// #[derive(Default)]
// struct FontRenderSettings {
//     pub font_size: f32,
// }

struct SimpleFontHandle {
    pub path: PathBuf,
    pub index: u32,
    pub data: Mmap,
}

impl SimpleFontHandle {
    pub(crate) fn new(path: PathBuf, index: u32) -> Self {
        let file = std::fs::File::open(&path).unwrap();
        let mapped = unsafe { Mmap::map(&file) }.unwrap();
        Self {
            path,
            index,
            data: mapped,
        }
    }
}

impl From<NativeFontHandle> for SimpleFontHandle {
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    fn from(native: NativeFontHandle) -> Self {
        let file = std::fs::File::open(&native.path).unwrap();
        let mapped = unsafe { Mmap::map(&file) }.unwrap();

        SimpleFontHandle {
            path: native.path,
            index: native.index,
            data: mapped,
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn from(native: NativeFontHandle) -> Self {
        let file = std::fs::File::open(&native.path).unwrap();
        let mapped = unsafe { Mmap::map(&file) }.unwrap();
        let font_file = FileRef::new(&mapped).unwrap();
        let index = ttc_index_from_postscript_name(font_file, &native.name);

        SimpleFontHandle {
            path: PathBuf::from(native.path),
            index,
            data: mapped,
        }
    }
}

trait RasterContext {
    // Static fns
    fn distribute_across_threads() -> bool;
    fn new() -> Self;
    fn begin_rasterize(font: &FontInstance);
    fn end_rasterize(font: &FontInstance);
    fn prepare_font(font: &mut FontInstance);

    // Font registration
    fn add_raw_font(&mut self, font_key: &FontKey, bytes: Arc<Vec<u8>>, index: u32);
    fn add_native_font(&mut self, font_key: &FontKey, native_font_handle: NativeFontHandle);
    fn delete_font(&mut self, font_key: &FontKey);
    fn delete_font_instance(&mut self, instance: &FontInstance);

    // Methods
    fn get_glyph_index(&mut self, font_key: FontKey, ch: char) -> Option<u32>;
    fn get_glyph_dimensions(
        &mut self,
        font: &FontInstance,
        key: &GlyphKey,
    ) -> Option<GlyphDimensions>;
    fn rasterize_glyph(&mut self, font: &FontInstance, key: &GlyphKey) -> GlyphRasterResult;
}

pub struct FontContext {
    font_cache: FastHashMap<FontKey, PenikoFont>,
    hinting_instance_cache: FastHashMap<(FontInstanceKey, u32, u32), Option<HintingInstance>>,
    // Scratch state reused between glyphs to avoid per-glyph allocations.
    render_context: RenderContext,
    resources: Resources,
    scratch_path: BezPath,
}

/// A glyph outline loaded for a particular font instance, in y-down
/// coordinates relative to the baseline origin, with the subpixel offset
/// already applied.
struct LoadedGlyph {
    path: BezPath,
    dimensions: GlyphDimensions,
}

impl FontContext {
    // Static fns
    pub fn distribute_across_threads() -> bool {
        true
    }
    pub fn new() -> FontContext {
        // Single-threaded rendering: WebRender already distributes glyph
        // rasterization across its own worker threads.
        let render_settings = RenderSettings {
            num_threads: 0,
            ..RenderSettings::default()
        };
        FontContext {
            font_cache: Default::default(),
            hinting_instance_cache: Default::default(),
            render_context: RenderContext::new_with(0, 0, render_settings),
            resources: Resources::new(),
            scratch_path: BezPath::new(),
        }
    }
    pub fn begin_rasterize(font: &FontInstance) {
        // TODO: apply instance properties
    }
    pub fn end_rasterize(font: &FontInstance) {
        // TODO: clear instance properties
    }

    pub fn prepare_font(font: &mut FontInstance) {
        // Subpixel AA (LCD) rendering is not supported yet; fall back to
        // grayscale alpha so WebRender does not expect per-channel coverage.
        font.disable_subpixel_aa();
        match font.render_mode {
            FontRenderMode::Mono => {
                // In mono mode the color of the font is irrelevant.
                font.color = ColorU::new(0xFF, 0xFF, 0xFF, 0xFF);
                // Subpixel positioning is disabled in mono mode.
                font.disable_subpixel_position();
            }
            FontRenderMode::Alpha | FontRenderMode::Subpixel => {
                // We produce coverage in all channels, so color is unused.
                font.color = ColorU::new(0xFF, 0xFF, 0xFF, 0xFF);
            }
        }
    }
    pub fn delete_font_instance(&mut self, instance: &FontInstance) {
        self.hinting_instance_cache
            .retain(|(key, ..), _| *key != instance.base.instance_key);
    }

    pub fn add_raw_font(&mut self, font_key: &FontKey, bytes: Arc<Vec<u8>>, index: u32) {
        let font = PenikoFont {
            data: Blob::new(bytes),
            index: index,
        };
        self.font_cache.insert(*font_key, font);
    }
    pub fn add_native_font(&mut self, font_key: &FontKey, native_font_handle: NativeFontHandle) {
        let handle = SimpleFontHandle::from(native_font_handle);
        let font = PenikoFont {
            data: Blob::new(Arc::new(handle.data)),
            index: handle.index,
        };
        self.font_cache.insert(*font_key, font);
    }
    pub fn delete_font(&mut self, font_key: &FontKey) {
        self.font_cache.remove(font_key);
    }

    // Methods
    pub fn get_glyph_index(&mut self, font_key: FontKey, ch: char) -> Option<u32> {
        let font = self.font_cache.get(&font_key)?;
        let data: &[u8] = font.data.as_ref().as_ref();
        let font_ref = skrifa::FontRef::from_index(data, font.index).ok()?;

        let char_map = Charmap::new(&font_ref);
        let glyph_id = char_map.map(ch)?;

        Some(glyph_id.to_u32())
    }

    /// Load the outline for a glyph, hinted if possible, with the subpixel
    /// offset applied. Returns the path (y-down, baseline origin) together
    /// with metrics and the tight device-pixel dimensions of the path.
    fn load_glyph(
        &mut self,
        font_instance: &FontInstance,
        key: &GlyphKey,
    ) -> Option<LoadedGlyph> {
        let font = self.font_cache.get(&font_instance.font_key)?;
        let data: &[u8] = font.data.as_ref().as_ref();
        let font_ref = skrifa::FontRef::from_index(data, font.index).ok()?;

        let location = font_ref.axes().location(
            font_instance
                .base
                .variations
                .iter()
                .map(|v| (Tag::from_be_bytes(v.tag.to_be_bytes()), v.value)),
        );
        let location_ref = LocationRef::from(&location);

        let req_size = font_instance.size.to_f64_px();
        let (x_scale, y_scale) = font_instance
            .transform
            .compute_scale()
            .unwrap_or((1.0, 1.0));
        // Rasterize at a uniform size scaled by the transform's y-scale; the
        // x/y aspect ratio and the residual (unit-determinant) part of the
        // transform are applied to the outline path below.
        let font_size = (req_size * y_scale) as f32;

        // The residual shape of the transform once the scale is factored out,
        // expressed directly in our y-down path coordinate space.
        let mut shape = font_instance.transform.invert_scale(x_scale, y_scale);
        if font_instance.flags.contains(FontInstanceFlags::FLIP_X) {
            shape = shape.flip_x();
        }
        if font_instance.flags.contains(FontInstanceFlags::FLIP_Y) {
            shape = shape.flip_y();
        }
        if font_instance.flags.contains(FontInstanceFlags::TRANSPOSE) {
            shape = shape.swap_xy();
        }
        let (mut tx, mut ty) = (0.0, 0.0);
        if font_instance.synthetic_italics.is_enabled() {
            let (shape_, (tx_, ty_)) =
                font_instance.synthesize_italics(shape, req_size * y_scale);
            shape = shape_;
            tx = tx_;
            ty = ty_;
        }
        let aspect = x_scale / y_scale;
        // Column-major [a, b, c, d, e, f]: x' = a*x + c*y + e, y' = b*x + d*y + f.
        let affine = Affine::new([
            shape.scale_x as f64 * aspect,
            shape.skew_y as f64 * aspect,
            shape.skew_x as f64,
            shape.scale_y as f64,
            tx,
            ty,
        ]);
        let identity_shape = affine == Affine::IDENTITY;

        let glyph_metrics = GlyphMetrics::new(&font_ref, Size::new(font_size), location_ref);
        let mut advance = glyph_metrics.advance_width(GlyphId::new(key.index()))?;
        // The advance is a horizontal vector; transform it like the outline.
        advance *= (shape.scale_x as f64 * aspect) as f32;

        let outlines = font_ref.outline_glyphs();
        let glyph_outline = outlines.get(GlyphId::new(key.index()))?;

        // Hinting instances are cached per (instance, size, options), since
        // the effective size depends on the transform scale and the options
        // on the render mode.
        let hinting_instance = match hinting_options(font_instance) {
            Some((options, options_tag)) => self
                .hinting_instance_cache
                .entry((
                    font_instance.base.instance_key,
                    font_size.to_bits(),
                    options_tag,
                ))
                .or_insert_with(|| {
                    HintingInstance::new(&outlines, Size::new(font_size), location_ref, options)
                        .ok()
                })
                .as_ref(),
            None => None,
        };

        let draw_settings = match hinting_instance {
            Some(hinting_instance) => DrawSettings::hinted(hinting_instance, false),
            None => DrawSettings::unhinted(Size::new(font_size), location_ref),
        };

        let mut outline_path = OutlinePath {
            path: std::mem::take(&mut self.scratch_path),
        };
        outline_path.reuse();
        if glyph_outline.draw(draw_settings, &mut outline_path).is_err() {
            self.scratch_path = outline_path.path;
            return None;
        }
        let mut path = outline_path.path;

        if !identity_shape {
            path.apply_affine(affine);
        }

        // Synthetic bold: fatten the outline like FreeType's
        // mozilla_glyphslot_embolden_less (half of FT_GlyphSlot_Embolden's
        // strength): the glyph grows by size/48 px in each axis, i.e. an
        // expansion radius of size/96 px per side, and the advance grows by
        // the same total amount.
        if font_instance.flags.contains(FontInstanceFlags::SYNTHETIC_BOLD) {
            let strength = req_size * y_scale / 48.0;
            let radius = strength * 0.5;
            path = expand_path(&path, Diagonal2::new(radius, radius), Join::Miter, 4.0, 0.1);
            advance += strength as f32;
        }

        // Apply the subpixel offset to the path so that both the bounding box
        // and the rasterisation account for it exactly.
        let (dx, dy) = font_instance.get_subpx_offset(key);
        if dx != 0.0 || dy != 0.0 {
            path.apply_affine(Affine::translate((dx, dy)));
        }

        // The path is in y-down coordinates with the origin at the baseline.
        // Round outward to device pixel boundaries.
        let bounds = path.bounding_box();
        let (min_x, max_x, min_y, max_y) = if bounds.is_zero_area() {
            (0, 0, 0, 0)
        } else {
            (
                bounds.x0.floor() as i32,
                bounds.x1.ceil() as i32,
                bounds.y0.floor() as i32,
                bounds.y1.ceil() as i32,
            )
        };

        let dimensions = GlyphDimensions {
            advance,
            left: min_x,
            // Distance from the baseline up to the top of the glyph.
            top: -min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        };

        Some(LoadedGlyph {
            path,
            dimensions,
        })
    }

    pub fn get_glyph_dimensions(
        &mut self,
        font_instance: &FontInstance,
        key: &GlyphKey,
    ) -> Option<GlyphDimensions> {
        let glyph = self.load_glyph(font_instance, key)?;
        let dimensions = glyph.dimensions;
        self.scratch_path = glyph.path;
        Some(dimensions)
    }
    pub fn rasterize_glyph(
        &mut self,
        font_instance: &FontInstance,
        key: &GlyphKey,
    ) -> GlyphRasterResult {
        let glyph = self
            .load_glyph(font_instance, key)
            .ok_or(GlyphRasterError::LoadFailed)?;
        let dimensions = glyph.dimensions;

        // Handle zero-sized glyphs (e.g. space chars)
        if dimensions.width == 0 || dimensions.height == 0 {
            self.scratch_path = glyph.path;
            return Err(GlyphRasterError::LoadFailed);
        }

        // When the glyph will be sampled with scaling (local raster space),
        // surround it with a transparent 1px border to avoid bleeding
        // neighbouring atlas texels.
        let padding = if font_instance.use_texture_padding() { 1 } else { 0 };
        let width = dimensions.width as u16 + 2 * padding;
        let height = dimensions.height as u16 + 2 * padding;

        let render_context = &mut self.render_context;
        render_context.reset_and_resize(width, height);

        // Render white coverage; the actual text color is applied by
        // WebRender's shaders when compositing the glyph from the atlas.
        render_context.set_paint(PaintType::Solid(peniko::Color::WHITE));

        // Position the path so that its bounding box lands exactly on the
        // pixmap: translate by (-left, top), i.e. by (-min_x, -min_y),
        // plus the padding border.
        render_context.set_transform(Affine::translate((
            (padding as i32 - dimensions.left) as f64,
            (padding as i32 + dimensions.top) as f64,
        )));
        render_context.fill_path(&glyph.path);
        render_context.flush();

        let mut buffer = vec![0; width as usize * height as usize * 4];
        render_context.render(
            PixmapMut::new(width, height, &mut buffer).unwrap(),
            &mut self.resources,
        );

        self.scratch_path = glyph.path;

        Ok(RasterizedGlyph {
            top: (dimensions.top + padding as i32) as f32,
            left: (dimensions.left - padding as i32) as f32,
            width: width as i32,
            height: height as i32,
            // The transform (including scale) is fully baked into the
            // rasterized outline.
            scale: 1.0,
            format: GlyphFormat::Alpha,
            bytes: buffer,
            is_packed_glyph: false,
        })
    }
}

/// CoreText font enumaration gives us a postscript name rather than an index.
/// This functions maps from postscript name to index
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn ttc_index_from_postscript_name(font_file: FileRef<'_>, postscript_name: &str) -> u32 {
    use skrifa::raw::{FileRef, TableProvider as _};
    use skrifa::raw::types::NameId;

    let index = match font_file {
        FileRef::Font(_) => 0,
        FileRef::Collection(collection) => 'idx: {
            for i in 0..collection.len() {
                let font = collection.get(i).unwrap();
                let name_table = font.name().unwrap();
                if name_table
                    .name_record()
                    .iter()
                    .filter(|record| record.name_id() == NameId::POSTSCRIPT_NAME)
                    .any(|record| {
                        record
                            .string(name_table.string_data())
                            .unwrap()
                            .chars()
                            .eq(postscript_name.chars())
                    })
                {
                    break 'idx i;
                }
            }

            panic!(
                "Font with postscript_name {} not found in collection",
                postscript_name
            );
        }
    };

    index
}
