/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Dependency tracking for tile invalidation
//!
//! This module contains types and logic for tracking dynamic content dependencies
//! (images, opacity bindings, color bindings, clip corners) used to determine
//! what needs to be redrawn each frame.

use api::{ImageKey, PropertyBindingId, ColorU};
use crate::invalidation::PrimitiveCompareResult;
use crate::internal_types::FastHashMap;
use crate::resource_cache::{ResourceCache, ImageGeneration};
use crate::invalidation::cached_surface::{PrimitiveDependencyIndex, PrimitiveDescriptor, CachedSurfaceDescriptor};
use crate::intern::ItemUid;
use crate::invalidation::vert_buffer::VertRange;
use peek_poke::{PeekPoke, peek_from_slice};


/// Information about the state of a binding.
#[derive(Debug)]
pub struct BindingInfo<T> {
    /// The current value retrieved from dynamic scene properties.
    pub value: T,
    /// True if it was changed (or is new) since the last frame build.
    pub changed: bool,
}

/// Information stored in a tile descriptor for a binding.
#[derive(Debug, PartialEq, Clone, Copy, PeekPoke)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum Binding<T> {
    Value(T),
    Binding(PropertyBindingId),
}

impl<T: Default> Default for Binding<T> {
    fn default() -> Self {
        Binding::Value(T::default())
    }
}

impl<T> From<api::PropertyBinding<T>> for Binding<T> {
    fn from(binding: api::PropertyBinding<T>) -> Binding<T> {
        match binding {
            api::PropertyBinding::Binding(key, _) => Binding::Binding(key.id),
            api::PropertyBinding::Value(value) => Binding::Value(value),
        }
    }
}

pub type OpacityBinding = Binding<f32>;
pub type OpacityBindingInfo = BindingInfo<f32>;

pub type ColorBinding = Binding<ColorU>;
pub type ColorBindingInfo = BindingInfo<ColorU>;

/// Types of dependencies that a primitive can have
#[derive(PeekPoke)]
pub enum PrimitiveDependency {
    OpacityBinding {
        binding: OpacityBinding,
    },
    ColorBinding {
        binding: ColorBinding,
    },
    Image {
        image: ImageDependency,
    },
    /// Clip dependency: prim_uid is the clip's intern uid; position is captured
    /// as quantized raster-space corners in vert_data. See PrimitiveDependencyInfo::prim_uid
    /// for why this uid is stable across scroll events.
    Clip {
        prim_uid: ItemUid,
        vert_range: VertRange,
    },
    /// Marks that a text run is being rasterized in local space because its
    /// spatial node (or an ancestor) has an animated transform (bug 2056306).
    /// Emitted only while animating, so its presence/absence flips the dep
    /// count when the animation ends, invalidating the tile. Without this the
    /// text keeps its stale local-raster (blurry) rasterization after the
    /// animation settles, since prim_uid and the raster-space corners are
    /// unchanged from the last animating frame.
    AnimatedRasterSpace {
        animating: bool,
    },
}

/// Information stored an image dependency
#[derive(Debug, Copy, Clone, PartialEq, PeekPoke, Default)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct ImageDependency {
    pub key: ImageKey,
    pub generation: ImageGeneration,
}

impl ImageDependency {
    pub const INVALID: ImageDependency = ImageDependency {
        key: ImageKey::DUMMY,
        generation: ImageGeneration::INVALID,
    };
}

/// A key for storing primitive comparison results during tile dependency tests.
#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
pub struct PrimitiveComparisonKey {
    pub prev_index: PrimitiveDependencyIndex,
    pub curr_index: PrimitiveDependencyIndex,
}

/// A helper struct to compare a primitive and all its sub-dependencies.
pub struct PrimitiveComparer<'a> {
    prev: &'a CachedSurfaceDescriptor,
    curr: &'a CachedSurfaceDescriptor,
    resource_cache: &'a ResourceCache,
    opacity_bindings: &'a FastHashMap<PropertyBindingId, OpacityBindingInfo>,
    color_bindings: &'a FastHashMap<PropertyBindingId, ColorBindingInfo>,
}

impl<'a> PrimitiveComparer<'a> {
    pub fn new(
        prev: &'a CachedSurfaceDescriptor,
        curr: &'a CachedSurfaceDescriptor,
        resource_cache: &'a ResourceCache,
        opacity_bindings: &'a FastHashMap<PropertyBindingId, OpacityBindingInfo>,
        color_bindings: &'a FastHashMap<PropertyBindingId, ColorBindingInfo>,
    ) -> Self {
        PrimitiveComparer {
            prev,
            curr,
            resource_cache,
            opacity_bindings,
            color_bindings,
        }
    }

    /// Compares two primitive descriptors using prim_uid + quantized raster-space
    /// vert corners. Transform changes are covered by vert corners; dynamic content
    /// changes (images, opacity/color bindings) are checked via the dep stream.
    pub fn compare_prim(
        &mut self,
        prev_desc: &PrimitiveDescriptor,
        curr_desc: &PrimitiveDescriptor,
    ) -> PrimitiveCompareResult {
        if prev_desc.prim_uid != curr_desc.prim_uid {
            return PrimitiveCompareResult::Descriptor;
        }

        // Compare quantized raster-space corners (all corners live in the per-tile vert_data).
        let prev_range = prev_desc.prim_corners;
        let prev_end = (prev_range.offset + prev_range.count) as usize;
        let prev_verts = &self.prev.vert_data[prev_range.offset as usize .. prev_end];

        let curr_range = curr_desc.prim_corners;
        let curr_end = (curr_range.offset + curr_range.count) as usize;
        let curr_verts = &self.curr.vert_data[curr_range.offset as usize .. curr_end];

        if prev_verts != curr_verts {
            return PrimitiveCompareResult::Descriptor;
        }

        let prev_range = prev_desc.coverage_corners;
        let prev_end = (prev_range.offset + prev_range.count) as usize;
        let prev_verts = &self.prev.vert_data[prev_range.offset as usize .. prev_end];

        let curr_range = curr_desc.coverage_corners;
        let curr_end = (curr_range.offset + curr_range.count) as usize;
        let curr_verts = &self.curr.vert_data[curr_range.offset as usize .. curr_end];

        if prev_verts != curr_verts {
            return PrimitiveCompareResult::Descriptor;
        }

        // Conservative guard: if dep counts differ fall back to invalidation
        // rather than misaligning the dep stream.
        if prev_desc.dep_count != curr_desc.dep_count {
            return PrimitiveCompareResult::Descriptor;
        }

        // Check dynamic deps (Image, Opacity, Color) that aren't captured in
        // prim_uid or vert corners.
        let mut prev_dep_data = &self.prev.dep_data[prev_desc.dep_offset as usize ..];
        let mut curr_dep_data = &self.curr.dep_data[curr_desc.dep_offset as usize ..];

        let mut prev_dep = PrimitiveDependency::Image { image: ImageDependency::INVALID };
        let mut curr_dep = PrimitiveDependency::Image { image: ImageDependency::INVALID };

        for _ in 0 .. prev_desc.dep_count {
            prev_dep_data = peek_from_slice(prev_dep_data, &mut prev_dep);
            curr_dep_data = peek_from_slice(curr_dep_data, &mut curr_dep);

            match (&prev_dep, &curr_dep) {
                (PrimitiveDependency::Clip { prim_uid: prev_uid, vert_range: prev_range }, PrimitiveDependency::Clip { prim_uid: curr_uid, vert_range: curr_range }) => {
                    if prev_uid != curr_uid {
                        return PrimitiveCompareResult::Clip;
                    }
                    let prev_end = (prev_range.offset + prev_range.count) as usize;
                    let prev_verts: &[i32] = if prev_range.is_valid() && prev_end <= self.prev.vert_data.len() {
                        &self.prev.vert_data[prev_range.offset as usize .. prev_end]
                    } else {
                        &[]
                    };
                    let curr_end = (curr_range.offset + curr_range.count) as usize;
                    let curr_verts: &[i32] = if curr_range.is_valid() && curr_end <= self.curr.vert_data.len() {
                        &self.curr.vert_data[curr_range.offset as usize .. curr_end]
                    } else {
                        &[]
                    };
                    if prev_verts != curr_verts {
                        return PrimitiveCompareResult::Clip;
                    }
                }
                (PrimitiveDependency::Image { image: prev }, PrimitiveDependency::Image { image: curr }) => {
                    if prev != curr {
                        return PrimitiveCompareResult::Image;
                    }
                    if self.resource_cache.get_image_generation(curr.key) != curr.generation {
                        return PrimitiveCompareResult::Image;
                    }
                }
                (PrimitiveDependency::OpacityBinding { binding: prev }, PrimitiveDependency::OpacityBinding { binding: curr }) => {
                    if prev != curr {
                        return PrimitiveCompareResult::OpacityBinding;
                    }
                    if let OpacityBinding::Binding(id) = curr {
                        if self.opacity_bindings.get(id).map_or(true, |info| info.changed) {
                            return PrimitiveCompareResult::OpacityBinding;
                        }
                    }
                }
                (PrimitiveDependency::ColorBinding { binding: prev }, PrimitiveDependency::ColorBinding { binding: curr }) => {
                    if prev != curr {
                        return PrimitiveCompareResult::ColorBinding;
                    }
                    if let ColorBinding::Binding(id) = curr {
                        if self.color_bindings.get(id).map_or(true, |info| info.changed) {
                            return PrimitiveCompareResult::ColorBinding;
                        }
                    }
                }
                (PrimitiveDependency::AnimatedRasterSpace { animating: prev }, PrimitiveDependency::AnimatedRasterSpace { animating: curr }) => {
                    // Both frames animating: equal, so no per-frame invalidation
                    // here (transform changes are already covered by the vert
                    // corners). The animating end/start flips the dep count,
                    // caught by the dep_count guard above.
                    if prev != curr {
                        return PrimitiveCompareResult::Descriptor;
                    }
                }
                _ => {
                    return PrimitiveCompareResult::Descriptor;
                }
            }
        }

        PrimitiveCompareResult::Equal
    }
}
