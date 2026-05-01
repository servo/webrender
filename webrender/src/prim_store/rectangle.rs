/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{PropertyBinding, ColorU, ColorF, Shadow, RasterSpace};
use crate::scene_building::{CreateShadow, IsVisible};
use crate::intern;
use crate::internal_types::LayoutPrimitiveInfo;
use crate::prim_store::{
    PrimKey, InternablePrimitive, PrimitiveStore, PrimitiveKind,
    PrimTemplate, PrimTemplateCommonData, PrimitiveOpacity,
};
use crate::frame_builder::FrameBuildingState;
use crate::scene::SceneProperties;
use std::ops;

#[derive(Debug, Clone, Eq, MallocSizeOf, PartialEq, Hash)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct RectanglePrim {
    pub color: PropertyBinding<ColorU>,
}

pub type RectangleKey = PrimKey<RectanglePrim>;

pub type RectangleDataHandle = intern::Handle<RectanglePrim>;

impl RectangleKey {
    pub fn new(info: &LayoutPrimitiveInfo, kind: RectanglePrim) -> Self {
        RectangleKey { common: info.into(), kind }
    }
}

impl intern::InternDebug for RectangleKey {}

impl intern::Internable for RectanglePrim {
    type Key = RectangleKey;
    type StoreData = RectangleTemplate;
    type InternData = ();
    const PROFILE_COUNTER: usize = crate::profiler::INTERNED_PRIMITIVES;
}

impl InternablePrimitive for RectanglePrim {
    fn into_key(
        self,
        info: &LayoutPrimitiveInfo,
    ) -> RectangleKey {
        RectangleKey::new(info, self)
    }

    fn make_instance_kind(
        _key: RectangleKey,
        data_handle: RectangleDataHandle,
        _prim_store: &mut PrimitiveStore,
    ) -> PrimitiveKind {
        PrimitiveKind::Rectangle {
            data_handle,
        }
    }
}

impl IsVisible for RectanglePrim {
    fn is_visible(&self) -> bool {
        match self.color {
            PropertyBinding::Value(value) => value.a > 0,
            PropertyBinding::Binding(..) => true,
        }
    }
}

impl CreateShadow for RectanglePrim {
    fn create_shadow(
        &self,
        shadow: &Shadow,
        _: bool,
        _: RasterSpace,
    ) -> RectanglePrim {
        RectanglePrim {
            color: PropertyBinding::Value(shadow.color.into()),
        }
    }
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(MallocSizeOf)]
pub struct RectangleData {
    pub color: PropertyBinding<ColorF>,
}

pub type RectangleTemplate = PrimTemplate<RectangleData>;

impl RectangleTemplate {
    pub fn resolve(&self, scene_properties: &SceneProperties) -> ColorF {
        scene_properties.resolve_color(&self.kind.color)
    }
}

impl ops::Deref for RectangleTemplate {
    type Target = PrimTemplateCommonData;
    fn deref(&self) -> &Self::Target {
        &self.common
    }
}

impl ops::DerefMut for RectangleTemplate {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.common
    }
}

impl From<RectangleKey> for RectangleTemplate {
    fn from(item: RectangleKey) -> Self {
        RectangleTemplate {
            common: PrimTemplateCommonData::with_key_common(item.common),
            kind: RectangleData { color: item.kind.color.into() },
        }
    }
}

impl RectangleTemplate {
    pub fn update(
        &mut self,
        frame_state: &mut FrameBuildingState,
        scene_properties: &SceneProperties,
    ) {
        let mut writer = frame_state.frame_gpu_data.f32.write_blocks(1);
        writer.push_one(scene_properties.resolve_color(&self.kind.color).premultiplied());
        self.common.gpu_buffer_address = writer.finish();
        self.opacity = PrimitiveOpacity::from_alpha(
            scene_properties.resolve_color(&self.kind.color).a
        );
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
fn test_struct_sizes() {
    use std::mem;
    assert_eq!(mem::size_of::<RectanglePrim>(), 16, "RectanglePrim size changed");
    assert_eq!(mem::size_of::<RectangleTemplate>(), 48, "RectangleTemplate size changed");
    assert_eq!(mem::size_of::<RectangleKey>(), 28, "RectangleKey size changed");
}
