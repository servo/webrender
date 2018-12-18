/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{
    LayoutSideOffsetsAu, LayoutRect, LayoutSideOffsets, LayoutSize,
    PremultipliedColorF, Shadow
};
use api::AuHelpers;
use api::LayoutPrimitiveInfo;
use api::NormalBorder as ApiNormalBorder;
use border::create_border_segments;
use border::NormalBorderAu;
use display_list_flattener::{AsInstanceKind, CreateShadow, IsVisible};
use frame_builder::{FrameBuildingState};
use gpu_cache::GpuDataRequest;
use intern;
use prim_store::{
    BorderSegmentInfo, BrushSegment, PrimKey, PrimKeyCommonData, PrimTemplate,
    PrimTemplateCommonData, PrimitiveInstanceKind, PrimitiveOpacity,
    PrimitiveSceneData, PrimitiveStore
};
use storage;

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct NormalBorder {
    pub border: NormalBorderAu,
    pub widths: LayoutSideOffsetsAu,
}

pub type NormalBorderKey = PrimKey<NormalBorder>;

impl NormalBorderKey {
    pub fn new(
        info: &LayoutPrimitiveInfo,
        prim_relative_clip_rect: LayoutRect,
        normal_border: NormalBorder,
    ) -> Self {
        NormalBorderKey {
            common: PrimKeyCommonData::with_info(
                info,
                prim_relative_clip_rect,
            ),
            kind: normal_border,
        }
    }
}

impl intern::InternDebug for NormalBorderKey {}

impl AsInstanceKind<NormalBorderDataHandle> for NormalBorderKey {
    /// Construct a primitive instance that matches the type
    /// of primitive key.
    fn as_instance_kind(
        &self,
        data_handle: NormalBorderDataHandle,
        _: &mut PrimitiveStore,
    ) -> PrimitiveInstanceKind {
        PrimitiveInstanceKind::NormalBorder {
            data_handle,
            cache_handles: storage::Range::empty(),
        }
    }
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct NormalBorderData {
    pub brush_segments: Vec<BrushSegment>,
    pub border_segments: Vec<BorderSegmentInfo>,
    pub border: ApiNormalBorder,
    pub widths: LayoutSideOffsets,
}

impl NormalBorderData {
    /// Update the GPU cache for a given primitive template. This may be called multiple
    /// times per frame, by each primitive reference that refers to this interned
    /// template. The initial request call to the GPU cache ensures that work is only
    /// done if the cache entry is invalid (due to first use or eviction).
    pub fn update(
        &mut self,
        common: &mut PrimTemplateCommonData,
        frame_state: &mut FrameBuildingState,
    ) {
        if let Some(ref mut request) = frame_state.gpu_cache.request(&mut common.gpu_cache_handle) {
            self.write_prim_gpu_blocks(request, common.prim_size);
            self.write_segment_gpu_blocks(request);
        }

        common.opacity = PrimitiveOpacity::translucent();
    }

    fn write_prim_gpu_blocks(
        &self,
        request: &mut GpuDataRequest,
        prim_size: LayoutSize
    ) {
        // Border primitives currently used for
        // image borders, and run through the
        // normal brush_image shader.
        request.push(PremultipliedColorF::WHITE);
        request.push(PremultipliedColorF::WHITE);
        request.push([
            prim_size.width,
            prim_size.height,
            0.0,
            0.0,
        ]);
    }

    fn write_segment_gpu_blocks(
        &self,
        request: &mut GpuDataRequest,
    ) {
        for segment in &self.brush_segments {
            // has to match VECS_PER_SEGMENT
            request.write_segment(
                segment.local_rect,
                segment.extra_data,
            );
        }
    }
}

pub type NormalBorderTemplate = PrimTemplate<NormalBorderData>;

impl From<NormalBorderKey> for NormalBorderTemplate {
    fn from(key: NormalBorderKey) -> Self {
        let common = PrimTemplateCommonData::with_key_common(key.common);

        let mut border: ApiNormalBorder = key.kind.border.into();
        let widths = LayoutSideOffsets::from_au(key.kind.widths);

        // FIXME(emilio): Is this the best place to do this?
        border.normalize(&widths);

        let mut brush_segments = Vec::new();
        let mut border_segments = Vec::new();

        create_border_segments(
            common.prim_size,
            &border,
            &widths,
            &mut border_segments,
            &mut brush_segments,
        );

        NormalBorderTemplate {
            common,
            kind: NormalBorderData {
                brush_segments,
                border_segments,
                border,
                widths,
            }
        }
    }
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct NormalBorderDataMarker;

pub type NormalBorderDataStore = intern::DataStore<NormalBorderKey, NormalBorderTemplate, NormalBorderDataMarker>;
pub type NormalBorderDataHandle = intern::Handle<NormalBorderDataMarker>;
pub type NormalBorderDataUpdateList = intern::UpdateList<NormalBorderKey>;
pub type NormalBorderDataInterner = intern::Interner<NormalBorderKey, PrimitiveSceneData, NormalBorderDataMarker>;

impl intern::Internable for NormalBorder {
    type Marker = NormalBorderDataMarker;
    type Source = NormalBorderKey;
    type StoreData = NormalBorderTemplate;
    type InternData = PrimitiveSceneData;

    /// Build a new key from self with `info`.
    fn build_key(
        self,
        info: &LayoutPrimitiveInfo,
        prim_relative_clip_rect: LayoutRect,
    ) -> NormalBorderKey {
        NormalBorderKey::new(
            info,
            prim_relative_clip_rect,
            self,
        )
    }
}

impl CreateShadow for NormalBorder {
    fn create_shadow(&self, shadow: &Shadow) -> Self {
        let border = self.border.with_color(shadow.color.into());
        NormalBorder {
            border,
            widths: self.widths,
        }
    }
}

impl IsVisible for NormalBorder {
    fn is_visible(&self) -> bool {
        true
    }
}

#[test]
#[cfg(target_os = "linux")]
fn test_struct_sizes() {
    use std::mem;
    // The sizes of these structures are critical for performance on a number of
    // talos stress tests. If you get a failure here on CI, there's two possibilities:
    // (a) You made a structure smaller than it currently is. Great work! Update the
    //     test expectations and move on.
    // (b) You made a structure larger. This is not necessarily a problem, but should only
    //     be done with care, and after checking if talos performance regresses badly.
    assert_eq!(mem::size_of::<NormalBorder>(), 84, "NormalBorder size changed");
    assert_eq!(mem::size_of::<NormalBorderTemplate>(), 240, "NormalBorderTemplate size changed");
    assert_eq!(mem::size_of::<NormalBorderKey>(), 112, "NormalBorderKey size changed");
}
