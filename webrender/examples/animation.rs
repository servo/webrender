/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! This example creates a 200x200 white rect and allows the user to move it
//! around by using the arrow keys and rotate with '<'/'>'.
//! It does this by using the animation API.

//! The example also features seamless opaque/transparent split of a
//! rounded cornered rectangle, which is done automatically during the
//! scene building for render optimization.

extern crate euclid;
extern crate gleam;
extern crate glutin;
extern crate webrender;
extern crate rand;

#[path = "common/boilerplate.rs"]
mod boilerplate;

use boilerplate::{Example, HandyDandyRectBuilder};
use euclid::Radians;
use euclid::vec2;
use webrender::api::*;

struct App {
    transform_key: PropertyBindingKey<LayoutTransform>,
    opacity_key: PropertyBindingKey<f32>,
    transform: LayoutTransform,
    color_keys: [PropertyBindingKey<ColorF>; 2],
    gradient_stop_keys: [PropertyBindingKey<ColorF>; 2],
    opacity: f32,
}

impl App {
    fn build_cell_type_a(&self, builder: &mut DisplayListBuilder, position: LayoutPoint) {
        let bounds = (0, 0).to(200, 200);
        let info = LayoutPrimitiveInfo::new(bounds);
        let filters = vec![
            FilterOp::Opacity(PropertyBinding::Binding(self.opacity_key), self.opacity),
        ];

        builder.push_stacking_context(
            &info,
            ScrollPolicy::Scrollable,
            Some(LayoutTransform::create_translation(position.x, position.y, 0.).into()),
            TransformStyle::Flat,
            None,
            MixBlendMode::Normal,
            filters,
        );
        builder.push_stacking_context(
            &info,
            ScrollPolicy::Scrollable,
            Some(PropertyBinding::Binding(self.transform_key)),
            TransformStyle::Flat,
            None,
            MixBlendMode::Normal,
            vec![]
        );

        builder.push_rect(&info, PropertyBinding::Binding(self.color_keys[0]));
        builder.push_box_shadow(
            &info,
            bounds,
            vec2(10., 10.),
            PropertyBinding::Binding(self.color_keys[0]),
            20.,
            0.,
            BorderRadius::uniform(0.),
            BoxShadowClipMode::Outset,
        );
        builder.pop_stacking_context();
        builder.pop_stacking_context();
    }

    fn build_cell_type_b(&self, builder: &mut DisplayListBuilder, position: LayoutPoint) {
        let bounds = (0, 0).to(200, 200);
        let complex_clip = ComplexClipRegion {
            rect: bounds,
            radii: BorderRadius::uniform(50.0),
            mode: ClipMode::Clip,
        };

        let info = LayoutPrimitiveInfo {
            local_clip: LocalClip::RoundedRect(bounds, complex_clip),
            .. LayoutPrimitiveInfo::new(bounds)
        };

        let filters = vec![
            FilterOp::Opacity(PropertyBinding::Binding(self.opacity_key), self.opacity),
        ];

        builder.push_stacking_context(
            &info,
            ScrollPolicy::Scrollable,
            Some(LayoutTransform::create_translation(position.x, position.y, 0.).into()),
            TransformStyle::Flat,
            None,
            MixBlendMode::Normal,
            filters,
        );
        builder.push_stacking_context(
            &info,
            ScrollPolicy::Scrollable,
            Some(PropertyBinding::Binding(self.transform_key)),
            TransformStyle::Flat,
            None,
            MixBlendMode::Normal,
            vec![]
        );

        // Fill it with a white rect
        let gradient = builder.create_gradient(
            LayoutPoint::new(0., 0.),
            LayoutPoint::new(200., 200.),
            vec![
                GradientStop {
                    offset: 0.5,
                    color: PropertyBinding::Binding(self.gradient_stop_keys[0])
                },
                GradientStop {
                    offset: 1.0,
                    color: PropertyBinding::Binding(self.gradient_stop_keys[1])
                }
            ],
            ExtendMode::Clamp
        );

        builder.push_gradient(
            &info,
            gradient, 
            LayoutSize::new(200., 200.), LayoutSize::zero()
        );

        builder.push_box_shadow(
            &info,
            bounds,
            vec2(10., 10.),
            PropertyBinding::Binding(self.color_keys[1]),
            20.,
            0.,
            BorderRadius::uniform(50.),
            BoxShadowClipMode::Outset,
        );

        builder.pop_stacking_context();
        builder.pop_stacking_context();
    }
}

impl Example for App {
    fn render(
        &mut self,
        _api: &RenderApi,
        builder: &mut DisplayListBuilder,
        _resources: &mut ResourceUpdates,
        _framebuffer_size: DeviceUintSize,
        _pipeline_id: PipelineId,
        _document_id: DocumentId,
    ) {
        let bounds = (0, 0).to(1024, 768);
        let info = LayoutPrimitiveInfo::new(bounds);

        builder.push_stacking_context(
            &info,
            ScrollPolicy::Scrollable,
            None,
            TransformStyle::Flat,
            None,
            MixBlendMode::Normal,
            vec![],
        );

        self.build_cell_type_a(builder, LayoutPoint::new(0., 0.));
        self.build_cell_type_b(builder, LayoutPoint::new(0., 0.));

        let mut x = 0.;
        let mut y = 0.;
        let mut idx = 0;
        while y <= 768. && x <= 1024. {
            if x + 200. >= 1024. {
                x = 0.;
                y += 200.;
            }
            
            let is_even = idx as i32 % 2 == 0;
            if is_even {
                self.build_cell_type_a(builder, LayoutPoint::new(x, y));
            } else {
                self.build_cell_type_b(builder, LayoutPoint::new(x, y ));
            }

            idx += 1;
            x += 200.;
        }
  
        builder.pop_stacking_context();
    }

    fn on_event(&mut self, event: glutin::Event, api: &RenderApi, document_id: DocumentId) -> bool {
        match event {
            glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(key)) => {
                let (offset_x, offset_y, angle, delta_opacity) = match key {
                    glutin::VirtualKeyCode::Down => (0.0, 10.0, 0.0, 0.0),
                    glutin::VirtualKeyCode::Up => (0.0, -10.0, 0.0, 0.0),
                    glutin::VirtualKeyCode::Right => (10.0, 0.0, 0.0, 0.0),
                    glutin::VirtualKeyCode::Left => (-10.0, 0.0, 0.0, 0.0),
                    glutin::VirtualKeyCode::Comma => (0.0, 0.0, 0.1, 0.0),
                    glutin::VirtualKeyCode::Period => (0.0, 0.0, -0.1, 0.0),
                    glutin::VirtualKeyCode::Z => (0.0, 0.0, 0.0, -0.1),
                    glutin::VirtualKeyCode::X => (0.0, 0.0, 0.0, 0.1),
                    glutin::VirtualKeyCode::A => (0., 0., 0., 0.),
                    _ => return false,
                };

                use rand::distributions::IndependentSample;
                let between = rand::distributions::Range::new(0., 255.);
                let mut rng = rand::thread_rng();

                self.opacity += delta_opacity;
                self.opacity = f32::min(self.opacity, 1.);
                self.opacity = f32::max(self.opacity, 0.);

                let new_transform = self.transform
                    .pre_rotate(0.0, 0.0, 1.0, Radians::new(angle))
                    .post_translate(LayoutVector3D::new(offset_x, offset_y, 0.0));
                self.transform = new_transform;

                api.generate_frame(
                    document_id,
                    Some(DynamicProperties {
                        transforms: vec![
                            PropertyValue {
                                key: self.transform_key,
                                value: self.transform
                            },
                        ],
                        floats: vec![
                            PropertyValue {
                                key: self.opacity_key,
                                value: self.opacity
                            }
                        ],
                        colors: vec![
                            PropertyValue {
                                key: self.color_keys[0],
                                value: ColorF::new(
                                    between.ind_sample(&mut rng) / 255., 
                                    between.ind_sample(&mut rng) / 255., 
                                    between.ind_sample(&mut rng) / 255., 
                                    1.
                                )
                            },
                            PropertyValue {
                                key: self.color_keys[1],
                                value: ColorF::new(
                                    between.ind_sample(&mut rng) / 255., 
                                    between.ind_sample(&mut rng) / 255., 
                                    between.ind_sample(&mut rng) / 255., 
                                    1.
                                )
                            },
                            PropertyValue {
                                key: self.gradient_stop_keys[0],
                                value: ColorF::new(
                                    between.ind_sample(&mut rng) / 255., 
                                    between.ind_sample(&mut rng) / 255., 
                                    between.ind_sample(&mut rng) / 255., 
                                    1.
                                )
                            },
                            PropertyValue {
                                key: self.gradient_stop_keys[1],
                                value: ColorF::new(
                                    between.ind_sample(&mut rng) / 255., 
                                    between.ind_sample(&mut rng) / 255., 
                                    between.ind_sample(&mut rng) / 255., 
                                    1.
                                )
                            },
                        ]
                    }),
                );
            }
            _ => (),
        }

        false
    }
}

fn main() {
    let mut app = App {
        opacity_key: PropertyBindingKey::new(30),
        transform_key: PropertyBindingKey::new(20),
        transform: LayoutTransform::identity(),
        color_keys: [PropertyBindingKey::new(40), PropertyBindingKey::new(41)],
        gradient_stop_keys: [PropertyBindingKey::new(50), PropertyBindingKey::new(51)],
        opacity: 1.,
    };
    boilerplate::main_wrapper(&mut app, None);
}