/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate gleam;
extern crate glutin;
extern crate webrender;
extern crate webrender_traits;

#[macro_use]
extern crate lazy_static;

#[path="common/boilerplate.rs"]
mod boilerplate;

use boilerplate::HandyDandyRectBuilder;
use std::sync::Mutex;
use webrender_traits::*;

// This example creates a 100x100 white rect and allows the user to move it
// around by using the arrow keys. It does this by using the animation API.

fn body(builder: &mut DisplayListBuilder,
        _pipeline_id: &PipelineId,
        _layout_size: &LayoutSize)
{
    // Create a 100x100 stacking context with an animatable transform property.
    // Note the magic "42" we use as the animation key. That is used to update
    // the transform in the keyboard event handler code.
    let bounds = (0,0).to(100, 100);
    builder.push_stacking_context(webrender_traits::ScrollPolicy::Scrollable,
                                  bounds,
                                  Some(PropertyBinding::Binding(PropertyBindingKey::new(42))),
                                  TransformStyle::Flat,
                                  None,
                                  webrender_traits::MixBlendMode::Normal,
                                  Vec::new());

    // Fill it with a white rect
    let clip = builder.push_clip_region(&bounds, vec![], None);
    builder.push_rect(bounds,
                      clip,
                      ColorF::new(1.0, 1.0, 1.0, 1.0));

    builder.pop_stacking_context();
}

lazy_static! {
    static ref TRANSFORM: Mutex<LayoutTransform> = Mutex::new(LayoutTransform::identity());
}

fn event_handler(event: &glutin::Event,
                 api: &RenderApi)
{
    match *event {
        glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(key)) => {
            let offset = match key {
                 glutin::VirtualKeyCode::Down => (0.0, 10.0),
                 glutin::VirtualKeyCode::Up => (0.0, -10.0),
                 glutin::VirtualKeyCode::Right => (10.0, 0.0),
                 glutin::VirtualKeyCode::Left => (-10.0, 0.0),
                 _ => return,
            };
            // Update the transform based on the keyboard input and push it to
            // webrender using the generate_frame API. This will recomposite with
            // the updated transform.
            let new_transform = TRANSFORM.lock().unwrap().post_translated(offset.0, offset.1, 0.0);
            api.generate_frame(Some(DynamicProperties {
                transforms: vec![
                  PropertyValue {
                    key: PropertyBindingKey::new(42),
                    value: new_transform,
                  },
                ],
                floats: vec![],
            }));
            *TRANSFORM.lock().unwrap() = new_transform;
        }
        _ => ()
    }
}

fn main() {
    boilerplate::main_wrapper(body, event_handler);
}
