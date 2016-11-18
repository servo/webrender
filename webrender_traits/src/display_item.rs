/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use display_list::AuxiliaryListsBuilder;
use euclid::{Point2D, Rect, Size2D};
use {BorderRadius, BorderDisplayItem, ClipRegion, ColorF, ComplexClipRegion};
use {FontKey, ImageKey, PipelineId, ScrollLayerId, ScrollLayerInfo, ServoScrollRootId};
use {ImageMask, ItemRange};

impl BorderDisplayItem {
    pub fn top_left_inner_radius(&self) -> Size2D<f32> {
        Size2D::new((self.radius.top_left.width - self.left.width).max(0.0),
                    (self.radius.top_left.height - self.top.width).max(0.0))
    }

    pub fn top_right_inner_radius(&self) -> Size2D<f32> {
        Size2D::new((self.radius.top_right.width - self.right.width).max(0.0),
                    (self.radius.top_right.height - self.top.width).max(0.0))
    }

    pub fn bottom_left_inner_radius(&self) -> Size2D<f32> {
        Size2D::new((self.radius.bottom_left.width - self.left.width).max(0.0),
                    (self.radius.bottom_left.height - self.bottom.width).max(0.0))
    }

    pub fn bottom_right_inner_radius(&self) -> Size2D<f32> {
        Size2D::new((self.radius.bottom_right.width - self.right.width).max(0.0),
                    (self.radius.bottom_right.height - self.bottom.width).max(0.0))
    }
}

impl BorderRadius {
    pub fn zero() -> BorderRadius {
        BorderRadius {
            top_left: Size2D::new(0.0, 0.0),
            top_right: Size2D::new(0.0, 0.0),
            bottom_left: Size2D::new(0.0, 0.0),
            bottom_right: Size2D::new(0.0, 0.0),
        }
    }

    pub fn uniform(radius: f32) -> BorderRadius {
        BorderRadius {
            top_left: Size2D::new(radius, radius),
            top_right: Size2D::new(radius, radius),
            bottom_left: Size2D::new(radius, radius),
            bottom_right: Size2D::new(radius, radius),
        }
    }
}

impl ClipRegion {
    pub fn new(rect: &Rect<f32>,
               complex: Vec<ComplexClipRegion>,
               image_mask: Option<ImageMask>,
               auxiliary_lists_builder: &mut AuxiliaryListsBuilder)
               -> ClipRegion {
        ClipRegion {
            main: *rect,
            complex: auxiliary_lists_builder.add_complex_clip_regions(&complex),
            image_mask: image_mask,
        }
    }

    pub fn simple(rect: &Rect<f32>) -> ClipRegion {
        ClipRegion {
            main: *rect,
            complex: ItemRange::empty(),
            image_mask: None,
        }
    }

    pub fn is_complex(&self) -> bool {
        self.complex.length !=0 || self.image_mask.is_some()
    }
}

impl ColorF {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> ColorF {
        ColorF {
            r: r,
            g: g,
            b: b,
            a: a,
        }
    }

    pub fn scale_rgb(&self, scale: f32) -> ColorF {
        ColorF {
            r: self.r * scale,
            g: self.g * scale,
            b: self.b * scale,
            a: self.a,
        }
    }
}

impl ComplexClipRegion {
    /// Create a new complex clip region.
    pub fn new(rect: Rect<f32>, radii: BorderRadius) -> ComplexClipRegion {
        ComplexClipRegion {
            rect: rect,
            radii: radii,
        }
    }

    //TODO: move to `util` module?
    /// Return a maximum aligned rectangle that is fully inside the clip region.
    pub fn get_inner_rect(&self) -> Option<Rect<f32>> {
        let k = 0.3; //roughly higher than `1.0 - sqrt(0.5)`
        let xl = self.rect.origin.x +
            k * self.radii.top_left.width.max(self.radii.bottom_left.width);
        let xr = self.rect.origin.x + self.rect.size.width -
            k * self.radii.top_right.width.max(self.radii.bottom_right.width);
        let yt = self.rect.origin.y +
            k * self.radii.top_left.height.max(self.radii.top_right.height);
        let yb = self.rect.origin.y + self.rect.size.height -
            k * self.radii.bottom_left.height.max(self.radii.bottom_right.height);
        if xl <= xr && yt <= yb {
            Some(Rect::new(Point2D::new(xl, yt), Size2D::new(xr-xl, yb-yt)))
        } else {
            None
        }
    }
}

impl FontKey {
    pub fn new(key0: u32, key1: u32) -> FontKey {
        FontKey(key0, key1)
    }
}

impl ImageKey {
    pub fn new(key0: u32, key1: u32) -> ImageKey {
        ImageKey(key0, key1)
    }
}

impl ScrollLayerId {
    pub fn new(pipeline_id: PipelineId,
               index: usize,
               scroll_root_id: ServoScrollRootId)
               -> ScrollLayerId {
        ScrollLayerId {
            pipeline_id: pipeline_id,
            info: ScrollLayerInfo::Scrollable(index, scroll_root_id),
        }
    }

    pub fn create_fixed(pipeline_id: PipelineId) -> ScrollLayerId {
        ScrollLayerId {
            pipeline_id: pipeline_id,
            info: ScrollLayerInfo::Fixed,
        }
    }
}
