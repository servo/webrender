/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{PropertyBinding, ColorF};
use crate::dl_interner::{DlResolve, DlResolveContext};
use crate::prim_store::{PrimTemplate, PrimTemplateCommonData};
use crate::scene::SceneProperties;
use std::ops;

// `RectanglePrim` and its key live in `webrender_api::interned_prims`, since
// the content display list builder interns them.
pub use api::interned_prims::RectangleKey;

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(MallocSizeOf, Clone)]
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

/// Nothing to validate: a colour binding names a property, not a resource.
impl DlResolve<RectangleKey> for RectangleTemplate {
    fn resolve(key: &RectangleKey, _ctx: &DlResolveContext) -> Self {
        RectangleTemplate {
            common: PrimTemplateCommonData::with_key_common(key.common),
            kind: RectangleData { color: key.kind.color.into() },
        }
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
fn test_struct_sizes() {
    use std::mem;
    assert_eq!(mem::size_of::<api::interned_prims::RectanglePrim>(), 16, "RectanglePrim size changed");
    assert_eq!(mem::size_of::<RectangleTemplate>(), 64, "RectangleTemplate size changed");
}
