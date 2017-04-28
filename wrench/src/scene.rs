/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::collections::HashMap;
use webrender_traits::{BuiltDisplayList, ColorF, Epoch};
use webrender_traits::{LayerSize, PipelineId};

/// A representation of the layout within the display port for a given document or iframe.
#[derive(Debug)]
pub struct ScenePipeline {
    pub epoch: Epoch,
    pub viewport_size: LayerSize,
    pub background_color: Option<ColorF>,
}

/// A complete representation of the layout bundling visible pipelines together.
pub struct Scene {
    pub root_pipeline_id: Option<PipelineId>,
    pub pipeline_map: HashMap<PipelineId, ScenePipeline>,
    pub display_lists: HashMap<PipelineId, BuiltDisplayList>,
}

impl Scene {
    pub fn new() -> Scene {
        Scene {
            root_pipeline_id: None,
            pipeline_map: HashMap::default(),
            display_lists: HashMap::default(),
        }
    }

    pub fn set_root_pipeline_id(&mut self, pipeline_id: PipelineId) {
        self.root_pipeline_id = Some(pipeline_id);
    }

    pub fn begin_display_list(&mut self,
                              pipeline_id: &PipelineId,
                              epoch: &Epoch,
                              background_color: &Option<ColorF>,
                              viewport_size: &LayerSize) {
        let new_pipeline = ScenePipeline {
             epoch: epoch.clone(),
             viewport_size: viewport_size.clone(),
             background_color: background_color.clone(),
        };

        self.pipeline_map.insert(pipeline_id.clone(), new_pipeline);
    }

    pub fn finish_display_list(&mut self,
                               pipeline_id: PipelineId,
                               built_display_list: BuiltDisplayList) {
        self.display_lists.insert(pipeline_id, built_display_list);
    }
}
