/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{FilterOpGraphPictureReference, FilterOpGraphNode, ColorF, ColorU, PropertyBinding, PropertyBindingId};
use api::SVGFE_GRAPH_MAX;
use api::units::*;
use api::FilterOpGraphPictureBufferId;
use crate::profiler::add_text_marker;
use crate::filterdata::FilterDataHandle;
use crate::intern::ItemUid;
use core::time::Duration;


#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Clone, Copy, Default, MallocSizeOf, PartialEq, Hash, Eq)]
pub enum FilterGraphPictureBufferIdKey {
    #[default]
    /// empty slot in feMerge inputs
    None,
    /// reference to another (earlier) node in filter graph
    BufferId(i16),
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Clone, Copy, Default, MallocSizeOf, PartialEq, Hash, Eq)]
pub struct FilterGraphPictureReferenceKey {
    /// Id of the picture in question in a namespace unique to this filter DAG,
    /// some are special values like
    /// FilterPrimitiveDescription::kPrimitiveIndexSourceGraphic.
    pub buffer_id: FilterGraphPictureBufferIdKey,
    /// Place the input image here in Layout space (like node.subregion)
    pub subregion: [Au; 4],
    /// Translate the subregion by this amount
    pub offset: [Au; 2],
}

impl From<FilterGraphPictureReference> for FilterGraphPictureReferenceKey {
    fn from(pic: FilterGraphPictureReference) -> Self {
        FilterGraphPictureReferenceKey{
            buffer_id: match pic.buffer_id {
                FilterOpGraphPictureBufferId::None => FilterGraphPictureBufferIdKey::None,
                FilterOpGraphPictureBufferId::BufferId(id) => FilterGraphPictureBufferIdKey::BufferId(id),
            },
            subregion: [
                Au::from_f32_px(pic.subregion.min.x),
                Au::from_f32_px(pic.subregion.min.y),
                Au::from_f32_px(pic.subregion.max.x),
                Au::from_f32_px(pic.subregion.max.y),
            ],
            offset: [
                Au::from_f32_px(pic.offset.x),
                Au::from_f32_px(pic.offset.y),
            ],
        }
    }
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Clone, MallocSizeOf, PartialEq, Hash, Eq)]
pub enum FilterGraphOpKey {
    /// combine 2 images with SVG_FEBLEND_MODE_DARKEN
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feBlendElement
    SVGFEBlendDarken,
    /// combine 2 images with SVG_FEBLEND_MODE_LIGHTEN
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feBlendElement
    SVGFEBlendLighten,
    /// combine 2 images with SVG_FEBLEND_MODE_MULTIPLY
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feBlendElement
    SVGFEBlendMultiply,
    /// combine 2 images with SVG_FEBLEND_MODE_NORMAL
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feBlendElement
    SVGFEBlendNormal,
    /// combine 2 images with SVG_FEBLEND_MODE_SCREEN
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feBlendElement
    SVGFEBlendScreen,
    /// combine 2 images with SVG_FEBLEND_MODE_OVERLAY
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendOverlay,
    /// combine 2 images with SVG_FEBLEND_MODE_COLOR_DODGE
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendColorDodge,
    /// combine 2 images with SVG_FEBLEND_MODE_COLOR_BURN
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendColorBurn,
    /// combine 2 images with SVG_FEBLEND_MODE_HARD_LIGHT
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendHardLight,
    /// combine 2 images with SVG_FEBLEND_MODE_SOFT_LIGHT
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendSoftLight,
    /// combine 2 images with SVG_FEBLEND_MODE_DIFFERENCE
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendDifference,
    /// combine 2 images with SVG_FEBLEND_MODE_EXCLUSION
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendExclusion,
    /// combine 2 images with SVG_FEBLEND_MODE_HUE
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendHue,
    /// combine 2 images with SVG_FEBLEND_MODE_SATURATION
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendSaturation,
    /// combine 2 images with SVG_FEBLEND_MODE_COLOR
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendColor,
    /// combine 2 images with SVG_FEBLEND_MODE_LUMINOSITY
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendLuminosity,
    /// transform colors of image through 5x4 color matrix (transposed for
    /// efficiency)
    /// parameters: FilterOpGraphNode, matrix[5][4]
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feColorMatrixElement
    SVGFEColorMatrix{values: [Au; 20]},
    /// transform colors of image through configurable gradients with component
    /// swizzle
    /// parameters: FilterOpGraphNode, FilterData
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feComponentTransferElement
    SVGFEComponentTransferInterned{handle: ItemUid, creates_pixels: bool},
    /// composite 2 images with chosen composite mode with parameters for that
    /// mode
    /// parameters: FilterOpGraphNode, k1, k2, k3, k4
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feCompositeElement
    SVGFECompositeArithmetic{k1: Au, k2: Au, k3: Au, k4: Au},
    /// composite 2 images with chosen composite mode with parameters for that
    /// mode
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feCompositeElement
    SVGFECompositeATop,
    /// composite 2 images with chosen composite mode with parameters for that
    /// mode
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feCompositeElement
    SVGFECompositeIn,
    /// composite 2 images with chosen composite mode with parameters for that
    /// mode
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Docs: https://developer.mozilla.org/en-US/docs/Web/SVG/Element/feComposite
    SVGFECompositeLighter,
    /// composite 2 images with chosen composite mode with parameters for that
    /// mode
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feCompositeElement
    SVGFECompositeOut,
    /// composite 2 images with chosen composite mode with parameters for that
    /// mode
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feCompositeElement
    SVGFECompositeOver,
    /// composite 2 images with chosen composite mode with parameters for that
    /// mode
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feCompositeElement
    SVGFECompositeXOR,
    /// transform image through convolution matrix of up to 25 values (spec
    /// allows more but for performance reasons we do not)
    /// parameters: FilterOpGraphNode, orderX, orderY, kernelValues[25],
    ///  divisor, bias, targetX, targetY, kernelUnitLengthX, kernelUnitLengthY,
    ///  preserveAlpha
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feConvolveMatrixElement
    SVGFEConvolveMatrixEdgeModeDuplicate{order_x: i32, order_y: i32,
        kernel: [Au; SVGFE_CONVOLVE_VALUES_LIMIT], divisor: Au, bias: Au,
        target_x: i32, target_y: i32, kernel_unit_length_x: Au,
        kernel_unit_length_y: Au, preserve_alpha: i32},
    /// transform image through convolution matrix of up to 25 values (spec
    /// allows more but for performance reasons we do not)
    /// parameters: FilterOpGraphNode, orderX, orderY, kernelValues[25],
    /// divisor, bias, targetX, targetY, kernelUnitLengthX, kernelUnitLengthY,
    /// preserveAlpha
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feConvolveMatrixElement
    SVGFEConvolveMatrixEdgeModeNone{order_x: i32, order_y: i32,
        kernel: [Au; SVGFE_CONVOLVE_VALUES_LIMIT], divisor: Au, bias: Au,
        target_x: i32, target_y: i32, kernel_unit_length_x: Au,
        kernel_unit_length_y: Au, preserve_alpha: i32},
    /// transform image through convolution matrix of up to 25 values (spec
    /// allows more but for performance reasons we do not)
    /// parameters: FilterOpGraphNode, orderX, orderY, kernelValues[25],
    ///  divisor, bias, targetX, targetY, kernelUnitLengthX, kernelUnitLengthY,
    ///  preserveAlpha
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feConvolveMatrixElement
    SVGFEConvolveMatrixEdgeModeWrap{order_x: i32, order_y: i32,
        kernel: [Au; SVGFE_CONVOLVE_VALUES_LIMIT], divisor: Au, bias: Au,
        target_x: i32, target_y: i32, kernel_unit_length_x: Au,
        kernel_unit_length_y: Au, preserve_alpha: i32},
    /// calculate lighting based on heightmap image with provided values for a
    /// distant light source with specified direction
    /// parameters: FilterOpGraphNode, surfaceScale, diffuseConstant,
    ///  kernelUnitLengthX, kernelUnitLengthY, azimuth, elevation
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEDiffuseLightingElement
    ///  https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEDistantLightElement
    SVGFEDiffuseLightingDistant{surface_scale: Au, diffuse_constant: Au,
        kernel_unit_length_x: Au, kernel_unit_length_y: Au, azimuth: Au,
        elevation: Au},
    /// calculate lighting based on heightmap image with provided values for a
    /// point light source at specified location
    /// parameters: FilterOpGraphNode, surfaceScale, diffuseConstant,
    ///  kernelUnitLengthX, kernelUnitLengthY, x, y, z
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEDiffuseLightingElement
    ///  https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEPointLightElement
    SVGFEDiffuseLightingPoint{surface_scale: Au, diffuse_constant: Au,
        kernel_unit_length_x: Au, kernel_unit_length_y: Au, x: Au, y: Au,
        z: Au},
    /// calculate lighting based on heightmap image with provided values for a
    /// spot light source at specified location pointing at specified target
    /// location with specified hotspot sharpness and cone angle
    /// parameters: FilterOpGraphNode, surfaceScale, diffuseConstant,
    ///  kernelUnitLengthX, kernelUnitLengthY, x, y, z, pointsAtX, pointsAtY,
    ///  pointsAtZ, specularExponent, limitingConeAngle
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEDiffuseLightingElement
    ///  https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFESpotLightElement
    SVGFEDiffuseLightingSpot{surface_scale: Au, diffuse_constant: Au,
        kernel_unit_length_x: Au, kernel_unit_length_y: Au, x: Au, y: Au, z: Au,
        points_at_x: Au, points_at_y: Au, points_at_z: Au, cone_exponent: Au,
        limiting_cone_angle: Au},
    /// calculate a distorted version of first input image using offset values
    /// from second input image at specified intensity
    /// parameters: FilterOpGraphNode, scale_x, scale_y, xChannelSelector, yChannelSelector
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEDisplacementMapElement
    SVGFEDisplacementMap{scale_x: Au, scale_y: Au, x_channel_selector: u32,
        y_channel_selector: u32},
    /// create and merge a dropshadow version of the specified image's alpha
    /// channel with specified offset and blur radius
    /// parameters: FilterOpGraphNode, flood_color, flood_opacity, dx, dy,
    ///  stdDeviationX, stdDeviationY
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEDropShadowElement
    SVGFEDropShadow{color: ColorU, dx: Au, dy: Au, std_deviation_x: Au,
        std_deviation_y: Au},
    /// synthesize a new image of specified size containing a solid color
    /// parameters: FilterOpGraphNode, color
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEFloodElement
    SVGFEFlood{color: ColorU},
    /// create a blurred version of the input image
    /// parameters: FilterOpGraphNode, stdDeviationX, stdDeviationY
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEGaussianBlurElement
    SVGFEGaussianBlur{std_deviation_x: Au, std_deviation_y: Au},
    /// Filter that does no transformation of the colors, needed for
    /// debug purposes, and is the default value in impl_default_for_enums.
    SVGFEIdentity,
    /// synthesize a new image based on a url (i.e. blob image source)
    /// parameters: FilterOpGraphNode, sampling_filter (see SamplingFilter in
    /// Types.h), transform
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEImageElement
    SVGFEImage{sampling_filter: u32, matrix: [Au; 6]},
    /// create a new image based on the input image with the contour stretched
    /// outward (dilate operator)
    /// parameters: FilterOpGraphNode, radiusX, radiusY
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEMorphologyElement
    SVGFEMorphologyDilate{radius_x: Au, radius_y: Au},
    /// create a new image based on the input image with the contour shrunken
    /// inward (erode operator)
    /// parameters: FilterOpGraphNode, radiusX, radiusY
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEMorphologyElement
    SVGFEMorphologyErode{radius_x: Au, radius_y: Au},
    /// represents CSS opacity property as a graph node like the rest of the
    /// SVGFE* filters
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    SVGFEOpacity{value: Au},
    /// represents CSS opacity property as a graph node like the rest of the
    /// SVGFE* filters
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    SVGFEOpacityBinding{valuebindingid: PropertyBindingId, value: Au},
    /// Filter that copies the SourceGraphic image into the specified subregion,
    /// This is intentionally the only way to get SourceGraphic into the graph,
    /// as the filter region must be applied before it is used.
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - no inputs, no linear
    SVGFESourceGraphic,
    /// Filter that copies the SourceAlpha image into the specified subregion,
    /// This is intentionally the only way to get SourceAlpha into the graph,
    /// as the filter region must be applied before it is used.
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - no inputs, no linear
    SVGFESourceAlpha,
    /// calculate lighting based on heightmap image with provided values for a
    /// distant light source with specified direction
    /// parameters: FilerData, surfaceScale, specularConstant, specularExponent,
    ///  kernelUnitLengthX, kernelUnitLengthY, azimuth, elevation
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFESpecularLightingElement
    ///  https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEDistantLightElement
    SVGFESpecularLightingDistant{surface_scale: Au, specular_constant: Au,
        specular_exponent: Au, kernel_unit_length_x: Au,
        kernel_unit_length_y: Au, azimuth: Au, elevation: Au},
    /// calculate lighting based on heightmap image with provided values for a
    /// point light source at specified location
    /// parameters: FilterOpGraphNode, surfaceScale, specularConstant,
    ///  specularExponent, kernelUnitLengthX, kernelUnitLengthY, x, y, z
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFESpecularLightingElement
    ///  https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEPointLightElement
    SVGFESpecularLightingPoint{surface_scale: Au, specular_constant: Au,
        specular_exponent: Au, kernel_unit_length_x: Au,
        kernel_unit_length_y: Au, x: Au, y: Au, z: Au},
    /// calculate lighting based on heightmap image with provided values for a
    /// spot light source at specified location pointing at specified target
    /// location with specified hotspot sharpness and cone angle
    /// parameters: FilterOpGraphNode, surfaceScale, specularConstant,
    ///  specularExponent, kernelUnitLengthX, kernelUnitLengthY, x, y, z,
    ///  pointsAtX, pointsAtY, pointsAtZ, specularExponent, limitingConeAngle
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFESpecularLightingElement
    ///  https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFESpotLightElement
    SVGFESpecularLightingSpot{surface_scale: Au, specular_constant: Au,
        specular_exponent: Au, kernel_unit_length_x: Au,
        kernel_unit_length_y: Au, x: Au, y: Au, z: Au, points_at_x: Au,
        points_at_y: Au, points_at_z: Au, cone_exponent: Au,
        limiting_cone_angle: Au},
    /// create a new image based on the input image, repeated throughout the
    /// output rectangle
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFETileElement
    SVGFETile,
    /// convert a color image to an alpha channel - internal use; generated by
    /// SVGFilterInstance::GetOrCreateSourceAlphaIndex().
    SVGFEToAlpha,
    /// synthesize a new image based on Fractal Noise (Perlin) with the chosen
    /// stitching mode
    /// parameters: FilterOpGraphNode, baseFrequencyX, baseFrequencyY,
    ///  numOctaves, seed
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFETurbulenceElement
    SVGFETurbulenceWithFractalNoiseWithNoStitching{base_frequency_x: Au,
        base_frequency_y: Au, num_octaves: u32, seed: u32},
    /// synthesize a new image based on Fractal Noise (Perlin) with the chosen
    /// stitching mode
    /// parameters: FilterOpGraphNode, baseFrequencyX, baseFrequencyY,
    ///  numOctaves, seed
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFETurbulenceElement
    SVGFETurbulenceWithFractalNoiseWithStitching{base_frequency_x: Au,
        base_frequency_y: Au, num_octaves: u32, seed: u32},
    /// synthesize a new image based on Turbulence Noise (offset vectors)
    /// parameters: FilterOpGraphNode, baseFrequencyX, baseFrequencyY,
    ///  numOctaves, seed
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFETurbulenceElement
    SVGFETurbulenceWithTurbulenceNoiseWithNoStitching{base_frequency_x: Au,
        base_frequency_y: Au, num_octaves: u32, seed: u32},
    /// synthesize a new image based on Turbulence Noise (offset vectors)
    /// parameters: FilterOpGraphNode, baseFrequencyX, baseFrequencyY,
    ///  numOctaves, seed
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFETurbulenceElement
    SVGFETurbulenceWithTurbulenceNoiseWithStitching{base_frequency_x: Au,
        base_frequency_y: Au, num_octaves: u32, seed: u32},
}

impl From<FilterGraphOp> for FilterGraphOpKey {
    fn from(op: FilterGraphOp) -> Self {
        match op {
            FilterGraphOp::SVGFEBlendDarken => FilterGraphOpKey::SVGFEBlendDarken,
            FilterGraphOp::SVGFEBlendLighten => FilterGraphOpKey::SVGFEBlendLighten,
            FilterGraphOp::SVGFEBlendMultiply => FilterGraphOpKey::SVGFEBlendMultiply,
            FilterGraphOp::SVGFEBlendNormal => FilterGraphOpKey::SVGFEBlendNormal,
            FilterGraphOp::SVGFEBlendScreen => FilterGraphOpKey::SVGFEBlendScreen,
            FilterGraphOp::SVGFEBlendOverlay => FilterGraphOpKey::SVGFEBlendOverlay,
            FilterGraphOp::SVGFEBlendColorDodge => FilterGraphOpKey::SVGFEBlendColorDodge,
            FilterGraphOp::SVGFEBlendColorBurn => FilterGraphOpKey::SVGFEBlendColorBurn,
            FilterGraphOp::SVGFEBlendHardLight => FilterGraphOpKey::SVGFEBlendHardLight,
            FilterGraphOp::SVGFEBlendSoftLight => FilterGraphOpKey::SVGFEBlendSoftLight,
            FilterGraphOp::SVGFEBlendDifference => FilterGraphOpKey::SVGFEBlendDifference,
            FilterGraphOp::SVGFEBlendExclusion => FilterGraphOpKey::SVGFEBlendExclusion,
            FilterGraphOp::SVGFEBlendHue => FilterGraphOpKey::SVGFEBlendHue,
            FilterGraphOp::SVGFEBlendSaturation => FilterGraphOpKey::SVGFEBlendSaturation,
            FilterGraphOp::SVGFEBlendColor => FilterGraphOpKey::SVGFEBlendColor,
            FilterGraphOp::SVGFEBlendLuminosity => FilterGraphOpKey::SVGFEBlendLuminosity,
            FilterGraphOp::SVGFEColorMatrix { values: color_matrix } => {
                let mut quantized_values: [Au; 20] = [Au(0); 20];
                for (value, result) in color_matrix.iter().zip(quantized_values.iter_mut()) {
                    *result = Au::from_f32_px(*value);
                }
                FilterGraphOpKey::SVGFEColorMatrix{values: quantized_values}
            }
            FilterGraphOp::SVGFEComponentTransfer => unreachable!(),
            FilterGraphOp::SVGFEComponentTransferInterned { handle, creates_pixels } => FilterGraphOpKey::SVGFEComponentTransferInterned{
                handle: handle.uid(),
                creates_pixels,
            },
            FilterGraphOp::SVGFECompositeArithmetic { k1, k2, k3, k4 } => {
                FilterGraphOpKey::SVGFECompositeArithmetic{
                    k1: Au::from_f32_px(k1),
                    k2: Au::from_f32_px(k2),
                    k3: Au::from_f32_px(k3),
                    k4: Au::from_f32_px(k4),
                }
            }
            FilterGraphOp::SVGFECompositeATop => FilterGraphOpKey::SVGFECompositeATop,
            FilterGraphOp::SVGFECompositeIn => FilterGraphOpKey::SVGFECompositeIn,
            FilterGraphOp::SVGFECompositeLighter => FilterGraphOpKey::SVGFECompositeLighter,
            FilterGraphOp::SVGFECompositeOut => FilterGraphOpKey::SVGFECompositeOut,
            FilterGraphOp::SVGFECompositeOver => FilterGraphOpKey::SVGFECompositeOver,
            FilterGraphOp::SVGFECompositeXOR => FilterGraphOpKey::SVGFECompositeXOR,
            FilterGraphOp::SVGFEConvolveMatrixEdgeModeDuplicate { order_x, order_y, kernel, divisor, bias, target_x, target_y, kernel_unit_length_x, kernel_unit_length_y, preserve_alpha } => {
                let mut values: [Au; SVGFE_CONVOLVE_VALUES_LIMIT] = [Au(0); SVGFE_CONVOLVE_VALUES_LIMIT];
                for (value, result) in kernel.iter().zip(values.iter_mut()) {
                    *result = Au::from_f32_px(*value)
                }
                FilterGraphOpKey::SVGFEConvolveMatrixEdgeModeDuplicate{
                    order_x,
                    order_y,
                    kernel: values,
                    divisor: Au::from_f32_px(divisor),
                    bias: Au::from_f32_px(bias),
                    target_x,
                    target_y,
                    kernel_unit_length_x: Au::from_f32_px(kernel_unit_length_x),
                    kernel_unit_length_y: Au::from_f32_px(kernel_unit_length_y),
                    preserve_alpha,
                }
            }
            FilterGraphOp::SVGFEConvolveMatrixEdgeModeNone { order_x, order_y, kernel, divisor, bias, target_x, target_y, kernel_unit_length_x, kernel_unit_length_y, preserve_alpha } => {
                let mut values: [Au; SVGFE_CONVOLVE_VALUES_LIMIT] = [Au(0); SVGFE_CONVOLVE_VALUES_LIMIT];
                for (value, result) in kernel.iter().zip(values.iter_mut()) {
                    *result = Au::from_f32_px(*value)
                }
                FilterGraphOpKey::SVGFEConvolveMatrixEdgeModeNone{
                    order_x,
                    order_y,
                    kernel: values,
                    divisor: Au::from_f32_px(divisor),
                    bias: Au::from_f32_px(bias),
                    target_x,
                    target_y,
                    kernel_unit_length_x: Au::from_f32_px(kernel_unit_length_x),
                    kernel_unit_length_y: Au::from_f32_px(kernel_unit_length_y),
                    preserve_alpha,
                }
            }
            FilterGraphOp::SVGFEConvolveMatrixEdgeModeWrap { order_x, order_y, kernel, divisor, bias, target_x, target_y, kernel_unit_length_x, kernel_unit_length_y, preserve_alpha } => {
                let mut values: [Au; SVGFE_CONVOLVE_VALUES_LIMIT] = [Au(0); SVGFE_CONVOLVE_VALUES_LIMIT];
                for (value, result) in kernel.iter().zip(values.iter_mut()) {
                    *result = Au::from_f32_px(*value)
                }
                FilterGraphOpKey::SVGFEConvolveMatrixEdgeModeWrap{
                    order_x,
                    order_y,
                    kernel: values,
                    divisor: Au::from_f32_px(divisor),
                    bias: Au::from_f32_px(bias),
                    target_x,
                    target_y,
                    kernel_unit_length_x: Au::from_f32_px(kernel_unit_length_x),
                    kernel_unit_length_y: Au::from_f32_px(kernel_unit_length_y),
                    preserve_alpha,
                }
            }
            FilterGraphOp::SVGFEDiffuseLightingDistant { surface_scale, diffuse_constant, kernel_unit_length_x, kernel_unit_length_y, azimuth, elevation } => {
                FilterGraphOpKey::SVGFEDiffuseLightingDistant{
                    surface_scale: Au::from_f32_px(surface_scale),
                    diffuse_constant: Au::from_f32_px(diffuse_constant),
                    kernel_unit_length_x: Au::from_f32_px(kernel_unit_length_x),
                    kernel_unit_length_y: Au::from_f32_px(kernel_unit_length_y),
                    azimuth: Au::from_f32_px(azimuth),
                    elevation: Au::from_f32_px(elevation),
                }
            }
            FilterGraphOp::SVGFEDiffuseLightingPoint { surface_scale, diffuse_constant, kernel_unit_length_x, kernel_unit_length_y, x, y, z } => {
                FilterGraphOpKey::SVGFEDiffuseLightingPoint{
                    surface_scale: Au::from_f32_px(surface_scale),
                    diffuse_constant: Au::from_f32_px(diffuse_constant),
                    kernel_unit_length_x: Au::from_f32_px(kernel_unit_length_x),
                    kernel_unit_length_y: Au::from_f32_px(kernel_unit_length_y),
                    x: Au::from_f32_px(x),
                    y: Au::from_f32_px(y),
                    z: Au::from_f32_px(z),
                }
            }
            FilterGraphOp::SVGFEDiffuseLightingSpot { surface_scale, diffuse_constant, kernel_unit_length_x, kernel_unit_length_y, x, y, z, points_at_x, points_at_y, points_at_z, cone_exponent, limiting_cone_angle } => {
                FilterGraphOpKey::SVGFEDiffuseLightingSpot{
                    surface_scale: Au::from_f32_px(surface_scale),
                    diffuse_constant: Au::from_f32_px(diffuse_constant),
                    kernel_unit_length_x: Au::from_f32_px(kernel_unit_length_x),
                    kernel_unit_length_y: Au::from_f32_px(kernel_unit_length_y),
                    x: Au::from_f32_px(x),
                    y: Au::from_f32_px(y),
                    z: Au::from_f32_px(z),
                    points_at_x: Au::from_f32_px(points_at_x),
                    points_at_y: Au::from_f32_px(points_at_y),
                    points_at_z: Au::from_f32_px(points_at_z),
                    cone_exponent: Au::from_f32_px(cone_exponent),
                    limiting_cone_angle: Au::from_f32_px(limiting_cone_angle),
                }
            }
            FilterGraphOp::SVGFEDisplacementMap { scale_x, scale_y, x_channel_selector, y_channel_selector } => {
                FilterGraphOpKey::SVGFEDisplacementMap{
                    scale_x: Au::from_f32_px(scale_x),
                    scale_y: Au::from_f32_px(scale_y),
                    x_channel_selector,
                    y_channel_selector,
                }
            }
            FilterGraphOp::SVGFEDropShadow { color, dx, dy, std_deviation_x, std_deviation_y } => {
                FilterGraphOpKey::SVGFEDropShadow{
                    color: color.into(),
                    dx: Au::from_f32_px(dx),
                    dy: Au::from_f32_px(dy),
                    std_deviation_x: Au::from_f32_px(std_deviation_x),
                    std_deviation_y: Au::from_f32_px(std_deviation_y),
                }
            }
            FilterGraphOp::SVGFEFlood { color } => FilterGraphOpKey::SVGFEFlood{color: color.into()},
            FilterGraphOp::SVGFEGaussianBlur { std_deviation_x, std_deviation_y } => {
                FilterGraphOpKey::SVGFEGaussianBlur{
                    std_deviation_x: Au::from_f32_px(std_deviation_x),
                    std_deviation_y: Au::from_f32_px(std_deviation_y),
                }
            }
            FilterGraphOp::SVGFEIdentity => FilterGraphOpKey::SVGFEIdentity,
            FilterGraphOp::SVGFEImage { sampling_filter, matrix } => {
                let mut values: [Au; 6] = [Au(0); 6];
                for (value, result) in matrix.iter().zip(values.iter_mut()) {
                    *result = Au::from_f32_px(*value)
                }
                FilterGraphOpKey::SVGFEImage{
                    sampling_filter,
                    matrix: values,
                }
            }
            FilterGraphOp::SVGFEMorphologyDilate { radius_x, radius_y } => {
                FilterGraphOpKey::SVGFEMorphologyDilate{
                    radius_x: Au::from_f32_px(radius_x),
                    radius_y: Au::from_f32_px(radius_y),
                }
            }
            FilterGraphOp::SVGFEMorphologyErode { radius_x, radius_y } => {
                FilterGraphOpKey::SVGFEMorphologyErode{
                    radius_x: Au::from_f32_px(radius_x),
                    radius_y: Au::from_f32_px(radius_y),
                }
            }
            FilterGraphOp::SVGFEOpacity{valuebinding: binding, value: _} => {
                match binding {
                    PropertyBinding::Value(value) => {
                        FilterGraphOpKey::SVGFEOpacity{value: Au::from_f32_px(value)}
                    }
                    PropertyBinding::Binding(key, default) => {
                        FilterGraphOpKey::SVGFEOpacityBinding{valuebindingid: key.id, value: Au::from_f32_px(default)}
                    }
                }
            }
            FilterGraphOp::SVGFESourceAlpha => FilterGraphOpKey::SVGFESourceAlpha,
            FilterGraphOp::SVGFESourceGraphic => FilterGraphOpKey::SVGFESourceGraphic,
            FilterGraphOp::SVGFESpecularLightingDistant { surface_scale, specular_constant, specular_exponent, kernel_unit_length_x, kernel_unit_length_y, azimuth, elevation } => {
                FilterGraphOpKey::SVGFESpecularLightingDistant{
                    surface_scale: Au::from_f32_px(surface_scale),
                    specular_constant: Au::from_f32_px(specular_constant),
                    specular_exponent: Au::from_f32_px(specular_exponent),
                    kernel_unit_length_x: Au::from_f32_px(kernel_unit_length_x),
                    kernel_unit_length_y: Au::from_f32_px(kernel_unit_length_y),
                    azimuth: Au::from_f32_px(azimuth),
                    elevation: Au::from_f32_px(elevation),
                }
            }
            FilterGraphOp::SVGFESpecularLightingPoint { surface_scale, specular_constant, specular_exponent, kernel_unit_length_x, kernel_unit_length_y, x, y, z } => {
                FilterGraphOpKey::SVGFESpecularLightingPoint{
                    surface_scale: Au::from_f32_px(surface_scale),
                    specular_constant: Au::from_f32_px(specular_constant),
                    specular_exponent: Au::from_f32_px(specular_exponent),
                    kernel_unit_length_x: Au::from_f32_px(kernel_unit_length_x),
                    kernel_unit_length_y: Au::from_f32_px(kernel_unit_length_y),
                    x: Au::from_f32_px(x),
                    y: Au::from_f32_px(y),
                    z: Au::from_f32_px(z),
                }
            }
            FilterGraphOp::SVGFESpecularLightingSpot { surface_scale, specular_constant, specular_exponent, kernel_unit_length_x, kernel_unit_length_y, x, y, z, points_at_x, points_at_y, points_at_z, cone_exponent, limiting_cone_angle } => {
                FilterGraphOpKey::SVGFESpecularLightingSpot{
                    surface_scale: Au::from_f32_px(surface_scale),
                    specular_constant: Au::from_f32_px(specular_constant),
                    specular_exponent: Au::from_f32_px(specular_exponent),
                    kernel_unit_length_x: Au::from_f32_px(kernel_unit_length_x),
                    kernel_unit_length_y: Au::from_f32_px(kernel_unit_length_y),
                    x: Au::from_f32_px(x),
                    y: Au::from_f32_px(y),
                    z: Au::from_f32_px(z),
                    points_at_x: Au::from_f32_px(points_at_x),
                    points_at_y: Au::from_f32_px(points_at_y),
                    points_at_z: Au::from_f32_px(points_at_z),
                    cone_exponent: Au::from_f32_px(cone_exponent),
                    limiting_cone_angle: Au::from_f32_px(limiting_cone_angle),
                }
            }
            FilterGraphOp::SVGFETile => FilterGraphOpKey::SVGFETile,
            FilterGraphOp::SVGFEToAlpha => FilterGraphOpKey::SVGFEToAlpha,
            FilterGraphOp::SVGFETurbulenceWithFractalNoiseWithNoStitching { base_frequency_x, base_frequency_y, num_octaves, seed } => {
                FilterGraphOpKey::SVGFETurbulenceWithFractalNoiseWithNoStitching {
                    base_frequency_x: Au::from_f32_px(base_frequency_x),
                    base_frequency_y: Au::from_f32_px(base_frequency_y),
                    num_octaves,
                    seed,
                }
            }
            FilterGraphOp::SVGFETurbulenceWithFractalNoiseWithStitching { base_frequency_x, base_frequency_y, num_octaves, seed } => {
                FilterGraphOpKey::SVGFETurbulenceWithFractalNoiseWithStitching {
                    base_frequency_x: Au::from_f32_px(base_frequency_x),
                    base_frequency_y: Au::from_f32_px(base_frequency_y),
                    num_octaves,
                    seed,
                }
            }
            FilterGraphOp::SVGFETurbulenceWithTurbulenceNoiseWithNoStitching { base_frequency_x, base_frequency_y, num_octaves, seed } => {
                FilterGraphOpKey::SVGFETurbulenceWithTurbulenceNoiseWithNoStitching {
                    base_frequency_x: Au::from_f32_px(base_frequency_x),
                    base_frequency_y: Au::from_f32_px(base_frequency_y),
                    num_octaves,
                    seed,
                }
            }
            FilterGraphOp::SVGFETurbulenceWithTurbulenceNoiseWithStitching { base_frequency_x, base_frequency_y, num_octaves, seed } => {
                FilterGraphOpKey::SVGFETurbulenceWithTurbulenceNoiseWithStitching {
                    base_frequency_x: Au::from_f32_px(base_frequency_x),
                    base_frequency_y: Au::from_f32_px(base_frequency_y),
                    num_octaves,
                    seed,
                }
            }
        }
    }
}

#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
#[derive(Debug, Clone, MallocSizeOf, PartialEq, Hash, Eq)]
pub struct FilterGraphNodeKey {
    /// Indicates this graph node was marked as unnecessary by the DAG optimizer
    /// (for example SVGFEOffset can often be folded into downstream nodes)
    pub kept_by_optimizer: bool,
    /// True if color_interpolation_filter == LinearRgb; shader will convert
    /// sRGB texture pixel colors on load and convert back on store, for correct
    /// interpolation
    pub linear: bool,
    /// virtualized picture input binding 1 (i.e. texture source), typically
    /// this is used, but certain filters do not use it
    pub inputs: Vec<FilterGraphPictureReferenceKey>,
    /// rect this node will render into, in filter space, does not account for
    /// inflate or device_pixel_scale
    pub subregion: [Au; 4],
}

impl From<FilterGraphNode> for FilterGraphNodeKey {
    fn from(node: FilterGraphNode) -> Self {
        FilterGraphNodeKey{
            kept_by_optimizer: node.kept_by_optimizer,
            linear: node.linear,
            inputs: node.inputs.into_iter().map(|node| {node.into()}).collect(),
            subregion: [
                Au::from_f32_px(node.subregion.min.x),
                Au::from_f32_px(node.subregion.min.y),
                Au::from_f32_px(node.subregion.max.x),
                Au::from_f32_px(node.subregion.max.y),
            ],
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct FilterGraphPictureReference {
    /// Id of the picture in question in a namespace unique to this filter DAG,
    /// some are special values like
    /// FilterPrimitiveDescription::kPrimitiveIndexSourceGraphic.
    pub buffer_id: FilterOpGraphPictureBufferId,
    /// Set by wrap_prim_with_filters to the subregion of the input node, may
    /// also have been offset for feDropShadow or feOffset
    pub subregion: LayoutRect,
    /// During scene build this is the offset to apply to the input subregion
    /// for feOffset, which can be optimized away by pushing its offset and
    /// subregion crop to downstream nodes.  This is always zero in render tasks
    /// where it has already been applied to subregion by that point.  Not used
    /// in get_coverage_svgfe because source_padding/target_padding represent
    /// the offset there.
    pub offset: LayoutVector2D,
    /// Equal to the inflate value of the referenced buffer, or 0
    pub inflate: i16,
    /// Padding on each side to represent how this input is read relative to the
    /// node's output subregion, this represents what the operation needs to
    /// read from ths input, which may be blurred or offset.
    pub source_padding: LayoutRect,
    /// Padding on each side to represent how this input affects the node's
    /// subregion, this can be used to calculate target subregion based on
    /// SourceGraphic subregion.  This is usually equal to source_padding except
    /// offset in the opposite direction, inflates typically do the same thing
    /// to both types of padding.
    pub target_padding: LayoutRect,
}

impl From<FilterOpGraphPictureReference> for FilterGraphPictureReference {
    fn from(pic: FilterOpGraphPictureReference) -> Self {
        FilterGraphPictureReference{
            buffer_id: pic.buffer_id,
            // All of these are set by wrap_prim_with_filters
            subregion: LayoutRect::zero(),
            offset: LayoutVector2D::zero(),
            inflate: 0,
            source_padding: LayoutRect::zero(),
            target_padding: LayoutRect::zero(),
        }
    }
}

pub const SVGFE_CONVOLVE_DIAMETER_LIMIT: usize = 5;
pub const SVGFE_CONVOLVE_VALUES_LIMIT: usize = SVGFE_CONVOLVE_DIAMETER_LIMIT *
    SVGFE_CONVOLVE_DIAMETER_LIMIT;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum FilterGraphOp {
    /// Filter that copies the SourceGraphic image into the specified subregion,
    /// This is intentionally the only way to get SourceGraphic into the graph,
    /// as the filter region must be applied before it is used.
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - no inputs, no linear
    SVGFESourceGraphic,
    /// Filter that copies the SourceAlpha image into the specified subregion,
    /// This is intentionally the only way to get SourceAlpha into the graph,
    /// as the filter region must be applied before it is used.
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - no inputs, no linear
    SVGFESourceAlpha,
    /// Filter that does no transformation of the colors, used to implement a
    /// few things like SVGFEOffset, and this is the default value in
    /// impl_default_for_enums.
    /// parameters: FilterGraphNode
    /// SVG filter semantics - selectable input with offset
    SVGFEIdentity,
    /// represents CSS opacity property as a graph node like the rest of the
    /// SVGFE* filters
    /// parameters: FilterGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    SVGFEOpacity{valuebinding: api::PropertyBinding<f32>, value: f32},
    /// convert a color image to an alpha channel - internal use; generated by
    /// SVGFilterInstance::GetOrCreateSourceAlphaIndex().
    SVGFEToAlpha,
    /// combine 2 images with SVG_FEBLEND_MODE_DARKEN
    /// parameters: FilterGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feBlendElement
    SVGFEBlendDarken,
    /// combine 2 images with SVG_FEBLEND_MODE_LIGHTEN
    /// parameters: FilterGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feBlendElement
    SVGFEBlendLighten,
    /// combine 2 images with SVG_FEBLEND_MODE_MULTIPLY
    /// parameters: FilterGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feBlendElement
    SVGFEBlendMultiply,
    /// combine 2 images with SVG_FEBLEND_MODE_NORMAL
    /// parameters: FilterGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feBlendElement
    SVGFEBlendNormal,
    /// combine 2 images with SVG_FEBLEND_MODE_SCREEN
    /// parameters: FilterGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feBlendElement
    SVGFEBlendScreen,
    /// combine 2 images with SVG_FEBLEND_MODE_OVERLAY
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendOverlay,
    /// combine 2 images with SVG_FEBLEND_MODE_COLOR_DODGE
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendColorDodge,
    /// combine 2 images with SVG_FEBLEND_MODE_COLOR_BURN
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendColorBurn,
    /// combine 2 images with SVG_FEBLEND_MODE_HARD_LIGHT
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendHardLight,
    /// combine 2 images with SVG_FEBLEND_MODE_SOFT_LIGHT
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendSoftLight,
    /// combine 2 images with SVG_FEBLEND_MODE_DIFFERENCE
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendDifference,
    /// combine 2 images with SVG_FEBLEND_MODE_EXCLUSION
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendExclusion,
    /// combine 2 images with SVG_FEBLEND_MODE_HUE
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendHue,
    /// combine 2 images with SVG_FEBLEND_MODE_SATURATION
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendSaturation,
    /// combine 2 images with SVG_FEBLEND_MODE_COLOR
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendColor,
    /// combine 2 images with SVG_FEBLEND_MODE_LUMINOSITY
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Source: https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    SVGFEBlendLuminosity,
    /// transform colors of image through 5x4 color matrix (transposed for
    /// efficiency)
    /// parameters: FilterGraphNode, matrix[5][4]
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feColorMatrixElement
    SVGFEColorMatrix{values: [f32; 20]},
    /// transform colors of image through configurable gradients with component
    /// swizzle
    /// parameters: FilterGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feComponentTransferElement
    SVGFEComponentTransfer,
    /// Processed version of SVGFEComponentTransfer with the FilterData
    /// replaced by an interned handle, this is made in wrap_prim_with_filters.
    /// Aside from the interned handle, creates_pixels indicates if the transfer
    /// parameters will probably fill the entire subregion with non-zero alpha.
    SVGFEComponentTransferInterned{handle: FilterDataHandle, creates_pixels: bool},
    /// composite 2 images with chosen composite mode with parameters for that
    /// mode
    /// parameters: FilterGraphNode, k1, k2, k3, k4
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feCompositeElement
    SVGFECompositeArithmetic{k1: f32, k2: f32, k3: f32, k4: f32},
    /// composite 2 images with chosen composite mode with parameters for that
    /// mode
    /// parameters: FilterGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feCompositeElement
    SVGFECompositeATop,
    /// composite 2 images with chosen composite mode with parameters for that
    /// mode
    /// parameters: FilterGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feCompositeElement
    SVGFECompositeIn,
    /// composite 2 images with chosen composite mode with parameters for that
    /// mode
    /// parameters: FilterOpGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Docs: https://developer.mozilla.org/en-US/docs/Web/SVG/Element/feComposite
    SVGFECompositeLighter,
    /// composite 2 images with chosen composite mode with parameters for that
    /// mode
    /// parameters: FilterGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feCompositeElement
    SVGFECompositeOut,
    /// composite 2 images with chosen composite mode with parameters for that
    /// mode
    /// parameters: FilterGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feCompositeElement
    SVGFECompositeOver,
    /// composite 2 images with chosen composite mode with parameters for that
    /// mode
    /// parameters: FilterGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feCompositeElement
    SVGFECompositeXOR,
    /// transform image through convolution matrix of up to 25 values (spec
    /// allows more but for performance reasons we do not)
    /// parameters: FilterGraphNode, orderX, orderY, kernelValues[25], divisor,
    ///  bias, targetX, targetY, kernelUnitLengthX, kernelUnitLengthY,
    ///  preserveAlpha
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feConvolveMatrixElement
    SVGFEConvolveMatrixEdgeModeDuplicate{order_x: i32, order_y: i32,
        kernel: [f32; SVGFE_CONVOLVE_VALUES_LIMIT], divisor: f32, bias: f32,
        target_x: i32, target_y: i32, kernel_unit_length_x: f32,
        kernel_unit_length_y: f32, preserve_alpha: i32},
    /// transform image through convolution matrix of up to 25 values (spec
    /// allows more but for performance reasons we do not)
    /// parameters: FilterGraphNode, orderX, orderY, kernelValues[25], divisor,
    ///  bias, targetX, targetY, kernelUnitLengthX, kernelUnitLengthY,
    ///  preserveAlpha
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feConvolveMatrixElement
    SVGFEConvolveMatrixEdgeModeNone{order_x: i32, order_y: i32,
        kernel: [f32; SVGFE_CONVOLVE_VALUES_LIMIT], divisor: f32, bias: f32,
        target_x: i32, target_y: i32, kernel_unit_length_x: f32,
        kernel_unit_length_y: f32, preserve_alpha: i32},
    /// transform image through convolution matrix of up to 25 values (spec
    /// allows more but for performance reasons we do not)
    /// parameters: FilterGraphNode, orderX, orderY, kernelValues[25], divisor,
    ///  bias, targetX, targetY, kernelUnitLengthX, kernelUnitLengthY,
    ///  preserveAlpha
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#feConvolveMatrixElement
    SVGFEConvolveMatrixEdgeModeWrap{order_x: i32, order_y: i32,
        kernel: [f32; SVGFE_CONVOLVE_VALUES_LIMIT], divisor: f32, bias: f32,
        target_x: i32, target_y: i32, kernel_unit_length_x: f32,
        kernel_unit_length_y: f32, preserve_alpha: i32},
    /// calculate lighting based on heightmap image with provided values for a
    /// distant light source with specified direction
    /// parameters: FilterGraphNode, surfaceScale, diffuseConstant,
    ///  kernelUnitLengthX, kernelUnitLengthY, azimuth, elevation
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEDiffuseLightingElement
    ///  https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEDistantLightElement
    SVGFEDiffuseLightingDistant{surface_scale: f32, diffuse_constant: f32,
        kernel_unit_length_x: f32, kernel_unit_length_y: f32, azimuth: f32,
        elevation: f32},
    /// calculate lighting based on heightmap image with provided values for a
    /// point light source at specified location
    /// parameters: FilterGraphNode, surfaceScale, diffuseConstant,
    ///  kernelUnitLengthX, kernelUnitLengthY, x, y, z
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEDiffuseLightingElement
    ///  https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEPointLightElement
    SVGFEDiffuseLightingPoint{surface_scale: f32, diffuse_constant: f32,
        kernel_unit_length_x: f32, kernel_unit_length_y: f32, x: f32, y: f32,
        z: f32},
    /// calculate lighting based on heightmap image with provided values for a
    /// spot light source at specified location pointing at specified target
    /// location with specified hotspot sharpness and cone angle
    /// parameters: FilterGraphNode, surfaceScale, diffuseConstant,
    ///  kernelUnitLengthX, kernelUnitLengthY, x, y, z, pointsAtX, pointsAtY,
    ///  pointsAtZ, specularExponent, limitingConeAngle
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEDiffuseLightingElement
    ///  https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFESpotLightElement
    SVGFEDiffuseLightingSpot{surface_scale: f32, diffuse_constant: f32,
        kernel_unit_length_x: f32, kernel_unit_length_y: f32, x: f32, y: f32,
        z: f32, points_at_x: f32, points_at_y: f32, points_at_z: f32,
        cone_exponent: f32, limiting_cone_angle: f32},
    /// calculate a distorted version of first input image using offset values
    /// from second input image at specified intensity
    /// parameters: FilterGraphNode, scale_x, scale_y, xChannelSelector, yChannelSelector
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEDisplacementMapElement
    SVGFEDisplacementMap{scale_x: f32, scale_y: f32, x_channel_selector: u32,
        y_channel_selector: u32},
    /// create and merge a dropshadow version of the specified image's alpha
    /// channel with specified offset and blur radius
    /// parameters: FilterGraphNode, flood_color, flood_opacity, dx, dy,
    ///  stdDeviationX, stdDeviationY
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEDropShadowElement
    SVGFEDropShadow{color: ColorF, dx: f32, dy: f32, std_deviation_x: f32,
        std_deviation_y: f32},
    /// synthesize a new image of specified size containing a solid color
    /// parameters: FilterGraphNode, color
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEFloodElement
    SVGFEFlood{color: ColorF},
    /// create a blurred version of the input image
    /// parameters: FilterGraphNode, stdDeviationX, stdDeviationY
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEGaussianBlurElement
    SVGFEGaussianBlur{std_deviation_x: f32, std_deviation_y: f32},
    /// synthesize a new image based on a url (i.e. blob image source)
    /// parameters: FilterGraphNode,
    ///  samplingFilter (see SamplingFilter in Types.h), transform
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEImageElement
    SVGFEImage{sampling_filter: u32, matrix: [f32; 6]},
    /// create a new image based on the input image with the contour stretched
    /// outward (dilate operator)
    /// parameters: FilterGraphNode, radiusX, radiusY
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEMorphologyElement
    SVGFEMorphologyDilate{radius_x: f32, radius_y: f32},
    /// create a new image based on the input image with the contour shrunken
    /// inward (erode operator)
    /// parameters: FilterGraphNode, radiusX, radiusY
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEMorphologyElement
    SVGFEMorphologyErode{radius_x: f32, radius_y: f32},
    /// calculate lighting based on heightmap image with provided values for a
    /// distant light source with specified direction
    /// parameters: FilerData, surfaceScale, specularConstant, specularExponent,
    ///  kernelUnitLengthX, kernelUnitLengthY, azimuth, elevation
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFESpecularLightingElement
    ///  https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEDistantLightElement
    SVGFESpecularLightingDistant{surface_scale: f32, specular_constant: f32,
        specular_exponent: f32, kernel_unit_length_x: f32,
        kernel_unit_length_y: f32, azimuth: f32, elevation: f32},
    /// calculate lighting based on heightmap image with provided values for a
    /// point light source at specified location
    /// parameters: FilterGraphNode, surfaceScale, specularConstant,
    ///  specularExponent, kernelUnitLengthX, kernelUnitLengthY, x, y, z
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFESpecularLightingElement
    ///  https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFEPointLightElement
    SVGFESpecularLightingPoint{surface_scale: f32, specular_constant: f32,
        specular_exponent: f32, kernel_unit_length_x: f32,
        kernel_unit_length_y: f32, x: f32, y: f32, z: f32},
    /// calculate lighting based on heightmap image with provided values for a
    /// spot light source at specified location pointing at specified target
    /// location with specified hotspot sharpness and cone angle
    /// parameters: FilterGraphNode, surfaceScale, specularConstant,
    ///  specularExponent, kernelUnitLengthX, kernelUnitLengthY, x, y, z,
    ///  pointsAtX, pointsAtY, pointsAtZ, specularExponent, limitingConeAngle
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFESpecularLightingElement
    ///  https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFESpotLightElement
    SVGFESpecularLightingSpot{surface_scale: f32, specular_constant: f32,
        specular_exponent: f32, kernel_unit_length_x: f32,
        kernel_unit_length_y: f32, x: f32, y: f32, z: f32, points_at_x: f32,
        points_at_y: f32, points_at_z: f32, cone_exponent: f32,
        limiting_cone_angle: f32},
    /// create a new image based on the input image, repeated throughout the
    /// output rectangle
    /// parameters: FilterGraphNode
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFETileElement
    SVGFETile,
    /// synthesize a new image based on Fractal Noise (Perlin) with the chosen
    /// stitching mode
    /// parameters: FilterGraphNode, baseFrequencyX, baseFrequencyY, numOctaves,
    ///  seed
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFETurbulenceElement
    SVGFETurbulenceWithFractalNoiseWithNoStitching{base_frequency_x: f32,
        base_frequency_y: f32, num_octaves: u32, seed: u32},
    /// synthesize a new image based on Fractal Noise (Perlin) with the chosen
    /// stitching mode
    /// parameters: FilterGraphNode, baseFrequencyX, baseFrequencyY, numOctaves,
    ///  seed
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFETurbulenceElement
    SVGFETurbulenceWithFractalNoiseWithStitching{base_frequency_x: f32,
        base_frequency_y: f32, num_octaves: u32, seed: u32},
    /// synthesize a new image based on Turbulence Noise (offset vectors)
    /// parameters: FilterGraphNode, baseFrequencyX, baseFrequencyY, numOctaves,
    ///  seed
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFETurbulenceElement
    SVGFETurbulenceWithTurbulenceNoiseWithNoStitching{base_frequency_x: f32,
        base_frequency_y: f32, num_octaves: u32, seed: u32},
    /// synthesize a new image based on Turbulence Noise (offset vectors)
    /// parameters: FilterGraphNode, baseFrequencyX, baseFrequencyY, numOctaves,
    ///  seed
    /// SVG filter semantics - selectable input(s), selectable between linear
    /// (default) and sRGB color space for calculations
    /// Spec: https://www.w3.org/TR/filter-effects-1/#InterfaceSVGFETurbulenceElement
    SVGFETurbulenceWithTurbulenceNoiseWithStitching{base_frequency_x: f32,
        base_frequency_y: f32, num_octaves: u32, seed: u32},
}

impl FilterGraphOp {
    pub fn kind(&self) -> &'static str {
        match *self {
            FilterGraphOp::SVGFEBlendColor => "SVGFEBlendColor",
            FilterGraphOp::SVGFEBlendColorBurn => "SVGFEBlendColorBurn",
            FilterGraphOp::SVGFEBlendColorDodge => "SVGFEBlendColorDodge",
            FilterGraphOp::SVGFEBlendDarken => "SVGFEBlendDarken",
            FilterGraphOp::SVGFEBlendDifference => "SVGFEBlendDifference",
            FilterGraphOp::SVGFEBlendExclusion => "SVGFEBlendExclusion",
            FilterGraphOp::SVGFEBlendHardLight => "SVGFEBlendHardLight",
            FilterGraphOp::SVGFEBlendHue => "SVGFEBlendHue",
            FilterGraphOp::SVGFEBlendLighten => "SVGFEBlendLighten",
            FilterGraphOp::SVGFEBlendLuminosity => "SVGFEBlendLuminosity",
            FilterGraphOp::SVGFEBlendMultiply => "SVGFEBlendMultiply",
            FilterGraphOp::SVGFEBlendNormal => "SVGFEBlendNormal",
            FilterGraphOp::SVGFEBlendOverlay => "SVGFEBlendOverlay",
            FilterGraphOp::SVGFEBlendSaturation => "SVGFEBlendSaturation",
            FilterGraphOp::SVGFEBlendScreen => "SVGFEBlendScreen",
            FilterGraphOp::SVGFEBlendSoftLight => "SVGFEBlendSoftLight",
            FilterGraphOp::SVGFEColorMatrix{..} => "SVGFEColorMatrix",
            FilterGraphOp::SVGFEComponentTransfer => "SVGFEComponentTransfer",
            FilterGraphOp::SVGFEComponentTransferInterned{..} => "SVGFEComponentTransferInterned",
            FilterGraphOp::SVGFECompositeArithmetic{..} => "SVGFECompositeArithmetic",
            FilterGraphOp::SVGFECompositeATop => "SVGFECompositeATop",
            FilterGraphOp::SVGFECompositeIn => "SVGFECompositeIn",
            FilterGraphOp::SVGFECompositeLighter => "SVGFECompositeLighter",
            FilterGraphOp::SVGFECompositeOut => "SVGFECompositeOut",
            FilterGraphOp::SVGFECompositeOver => "SVGFECompositeOver",
            FilterGraphOp::SVGFECompositeXOR => "SVGFECompositeXOR",
            FilterGraphOp::SVGFEConvolveMatrixEdgeModeDuplicate{..} => "SVGFEConvolveMatrixEdgeModeDuplicate",
            FilterGraphOp::SVGFEConvolveMatrixEdgeModeNone{..} => "SVGFEConvolveMatrixEdgeModeNone",
            FilterGraphOp::SVGFEConvolveMatrixEdgeModeWrap{..} => "SVGFEConvolveMatrixEdgeModeWrap",
            FilterGraphOp::SVGFEDiffuseLightingDistant{..} => "SVGFEDiffuseLightingDistant",
            FilterGraphOp::SVGFEDiffuseLightingPoint{..} => "SVGFEDiffuseLightingPoint",
            FilterGraphOp::SVGFEDiffuseLightingSpot{..} => "SVGFEDiffuseLightingSpot",
            FilterGraphOp::SVGFEDisplacementMap{..} => "SVGFEDisplacementMap",
            FilterGraphOp::SVGFEDropShadow{..} => "SVGFEDropShadow",
            FilterGraphOp::SVGFEFlood{..} => "SVGFEFlood",
            FilterGraphOp::SVGFEGaussianBlur{..} => "SVGFEGaussianBlur",
            FilterGraphOp::SVGFEIdentity => "SVGFEIdentity",
            FilterGraphOp::SVGFEImage{..} => "SVGFEImage",
            FilterGraphOp::SVGFEMorphologyDilate{..} => "SVGFEMorphologyDilate",
            FilterGraphOp::SVGFEMorphologyErode{..} => "SVGFEMorphologyErode",
            FilterGraphOp::SVGFEOpacity{..} => "SVGFEOpacity",
            FilterGraphOp::SVGFESourceAlpha => "SVGFESourceAlpha",
            FilterGraphOp::SVGFESourceGraphic => "SVGFESourceGraphic",
            FilterGraphOp::SVGFESpecularLightingDistant{..} => "SVGFESpecularLightingDistant",
            FilterGraphOp::SVGFESpecularLightingPoint{..} => "SVGFESpecularLightingPoint",
            FilterGraphOp::SVGFESpecularLightingSpot{..} => "SVGFESpecularLightingSpot",
            FilterGraphOp::SVGFETile => "SVGFETile",
            FilterGraphOp::SVGFEToAlpha => "SVGFEToAlpha",
            FilterGraphOp::SVGFETurbulenceWithFractalNoiseWithNoStitching{..} => "SVGFETurbulenceWithFractalNoiseWithNoStitching",
            FilterGraphOp::SVGFETurbulenceWithFractalNoiseWithStitching{..} => "SVGFETurbulenceWithFractalNoiseWithStitching",
            FilterGraphOp::SVGFETurbulenceWithTurbulenceNoiseWithNoStitching{..} => "SVGFETurbulenceWithTurbulenceNoiseWithNoStitching",
            FilterGraphOp::SVGFETurbulenceWithTurbulenceNoiseWithStitching{..} => "SVGFETurbulenceWithTurbulenceNoiseWithStitching",
        }
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct FilterGraphNode {
    /// Indicates this graph node was marked as necessary by the DAG optimizer
    pub kept_by_optimizer: bool,
    /// true if color_interpolation_filter == LinearRgb; shader will convert
    /// sRGB texture pixel colors on load and convert back on store, for correct
    /// interpolation
    pub linear: bool,
    /// padding for output rect if we need a border to get correct clamping, or
    /// to account for larger final subregion than source rect (see bug 1869672)
    pub inflate: i16,
    /// virtualized picture input bindings, these refer to other filter outputs
    /// by number within the graph, usually there is one element
    pub inputs: Vec<FilterGraphPictureReference>,
    /// clipping rect for filter node output
    pub subregion: LayoutRect,
}

impl From<FilterOpGraphNode> for FilterGraphNode {
    fn from(node: FilterOpGraphNode) -> Self {
        let mut inputs: Vec<FilterGraphPictureReference> = Vec::new();
        if node.input.buffer_id != FilterOpGraphPictureBufferId::None {
            inputs.push(node.input.into());
        }
        if node.input2.buffer_id != FilterOpGraphPictureBufferId::None {
            inputs.push(node.input2.into());
        }
        // If the op used by this node is a feMerge, it will add more inputs
        // after this invocation.
        FilterGraphNode{
            linear: node.linear,
            inputs,
            subregion: node.subregion,
            // These are computed later in scene_building
            kept_by_optimizer: true,
            inflate: 0,
        }
    }
}

/// Here we transform source rect to target rect for SVGFEGraph by walking
/// the whole graph and propagating subregions based on the provided
/// invalidation rect, and we want it to be a tight fit so we don't waste
/// time applying multiple filters to pixels that do not contribute to the
/// invalidated rect.
///
/// The interesting parts of the handling of SVG filters are:
/// * scene_building.rs : wrap_prim_with_filters
/// * picture.rs : get_coverage_target_svgfe (you are here)
/// * picture.rs : get_coverage_source_svgfe
/// * render_task.rs : new_svg_filter_graph
/// * render_target.rs : add_svg_filter_node_instances
pub fn get_coverage_target_svgfe(
    filters: &[(FilterGraphNode, FilterGraphOp)],
    surface_rect: LayoutRect,
) -> LayoutRect {

    // The value of BUFFER_LIMIT here must be the same as in
    // scene_building.rs, or we'll hit asserts here.
    const BUFFER_LIMIT: usize = SVGFE_GRAPH_MAX;

    // We need to evaluate the subregions based on the proposed
    // SourceGraphic rect as it isn't known at scene build time.
    let mut subregion_by_buffer_id: [LayoutRect; BUFFER_LIMIT] = [LayoutRect::zero(); BUFFER_LIMIT];
    for (id, (node, op)) in filters.iter().enumerate() {
        let full_subregion = node.subregion;
        let mut used_subregion = LayoutRect::zero();
        for input in &node.inputs {
            match input.buffer_id {
                FilterOpGraphPictureBufferId::BufferId(id) => {
                    assert!((id as usize) < BUFFER_LIMIT, "BUFFER_LIMIT must be the same in frame building and scene building");
                    // This id lookup should always succeed.
                    let input_subregion = subregion_by_buffer_id[id as usize];
                    // Now add the padding that transforms from
                    // source to target, this was determined during
                    // scene build based on the operation.
                    let input_subregion =
                        LayoutRect::new(
                            LayoutPoint::new(
                                input_subregion.min.x + input.target_padding.min.x,
                                input_subregion.min.y + input.target_padding.min.y,
                            ),
                            LayoutPoint::new(
                                input_subregion.max.x + input.target_padding.max.x,
                                input_subregion.max.y + input.target_padding.max.y,
                            ),
                        );
                    used_subregion = used_subregion
                        .union(&input_subregion);
                }
                FilterOpGraphPictureBufferId::None => {
                    panic!("Unsupported BufferId type");
                }
            }
        }
        // We can clip the used subregion to the node subregion
        used_subregion = used_subregion
            .intersection(&full_subregion)
            .unwrap_or(LayoutRect::zero());
        match op {
            FilterGraphOp::SVGFEBlendColor => {}
            FilterGraphOp::SVGFEBlendColorBurn => {}
            FilterGraphOp::SVGFEBlendColorDodge => {}
            FilterGraphOp::SVGFEBlendDarken => {}
            FilterGraphOp::SVGFEBlendDifference => {}
            FilterGraphOp::SVGFEBlendExclusion => {}
            FilterGraphOp::SVGFEBlendHardLight => {}
            FilterGraphOp::SVGFEBlendHue => {}
            FilterGraphOp::SVGFEBlendLighten => {}
            FilterGraphOp::SVGFEBlendLuminosity => {}
            FilterGraphOp::SVGFEBlendMultiply => {}
            FilterGraphOp::SVGFEBlendNormal => {}
            FilterGraphOp::SVGFEBlendOverlay => {}
            FilterGraphOp::SVGFEBlendSaturation => {}
            FilterGraphOp::SVGFEBlendScreen => {}
            FilterGraphOp::SVGFEBlendSoftLight => {}
            FilterGraphOp::SVGFEColorMatrix { values } => {
                if values[19] > 0.0 {
                    // Manipulating alpha offset can easily create new
                    // pixels outside of input subregions
                    used_subregion = full_subregion;
                    add_text_marker(
                        "SVGFEColorMatrix",
                        "SVGFEColorMatrix with non-zero alpha offset, using full subregion",
                        Duration::from_millis(1));
                }
            }
            FilterGraphOp::SVGFEComponentTransfer => unreachable!(),
            FilterGraphOp::SVGFEComponentTransferInterned{handle: _, creates_pixels} => {
                // Check if the value of alpha[0] is modified, if so
                // the whole subregion is used because it will be
                // creating new pixels outside of input subregions
                if *creates_pixels {
                    used_subregion = full_subregion;
                    add_text_marker(
                        "SVGFEComponentTransfer",
                        "SVGFEComponentTransfer with non-zero minimum alpha, using full subregion",
                        Duration::from_millis(1));
                }
            }
            FilterGraphOp::SVGFECompositeArithmetic { k1, k2, k3, k4 } => {
                // Optimization opportunity - some inputs may be
                // smaller subregions due to the way the math works,
                // k1 is the intersection of the two inputs, k2 is
                // the first input only, k3 is the second input
                // only, and k4 changes the whole subregion.
                //
                // See logic for SVG_FECOMPOSITE_OPERATOR_ARITHMETIC
                // in FilterSupport.cpp
                //
                // We can at least ignore the entire node if
                // everything is zero.
                if *k1 <= 0.0 &&
                    *k2 <= 0.0 &&
                    *k3 <= 0.0 {
                    used_subregion = LayoutRect::zero();
                }
                // Check if alpha is added to pixels as it means it
                // can fill pixels outside input subregions
                if *k4 > 0.0 {
                    used_subregion = full_subregion;
                    add_text_marker(
                        "SVGFECompositeArithmetic",
                        "SVGFECompositeArithmetic with non-zero offset, using full subregion",
                        Duration::from_millis(1));
                }
            }
            FilterGraphOp::SVGFECompositeATop => {}
            FilterGraphOp::SVGFECompositeIn => {}
            FilterGraphOp::SVGFECompositeLighter => {}
            FilterGraphOp::SVGFECompositeOut => {}
            FilterGraphOp::SVGFECompositeOver => {}
            FilterGraphOp::SVGFECompositeXOR => {}
            FilterGraphOp::SVGFEConvolveMatrixEdgeModeDuplicate{..} => {}
            FilterGraphOp::SVGFEConvolveMatrixEdgeModeNone{..} => {}
            FilterGraphOp::SVGFEConvolveMatrixEdgeModeWrap{..} => {}
            FilterGraphOp::SVGFEDiffuseLightingDistant{..} => {}
            FilterGraphOp::SVGFEDiffuseLightingPoint{..} => {}
            FilterGraphOp::SVGFEDiffuseLightingSpot{..} => {}
            FilterGraphOp::SVGFEDisplacementMap{..} => {}
            FilterGraphOp::SVGFEDropShadow{..} => {}
            FilterGraphOp::SVGFEFlood { color } => {
                // Subregion needs to be set to the full node
                // subregion for fills (unless the fill is a no-op)
                if color.a > 0.0 {
                    used_subregion = full_subregion;
                }
            }
            FilterGraphOp::SVGFEGaussianBlur{..} => {}
            FilterGraphOp::SVGFEIdentity => {}
            FilterGraphOp::SVGFEImage { sampling_filter: _sampling_filter, matrix: _matrix } => {
                // TODO: calculate the actual subregion
                used_subregion = full_subregion;
            }
            FilterGraphOp::SVGFEMorphologyDilate{..} => {}
            FilterGraphOp::SVGFEMorphologyErode{..} => {}
            FilterGraphOp::SVGFEOpacity { valuebinding: _valuebinding, value } => {
                // If fully transparent, we can ignore this node
                if *value <= 0.0 {
                    used_subregion = LayoutRect::zero();
                }
            }
            FilterGraphOp::SVGFESourceAlpha |
            FilterGraphOp::SVGFESourceGraphic => {
                used_subregion = surface_rect;
            }
            FilterGraphOp::SVGFESpecularLightingDistant{..} => {}
            FilterGraphOp::SVGFESpecularLightingPoint{..} => {}
            FilterGraphOp::SVGFESpecularLightingSpot{..} => {}
            FilterGraphOp::SVGFETile => {
                // feTile fills the entire output with
                // source pixels, so it's effectively a flood.
                used_subregion = full_subregion;
            }
            FilterGraphOp::SVGFEToAlpha => {}
            FilterGraphOp::SVGFETurbulenceWithFractalNoiseWithNoStitching{..} |
            FilterGraphOp::SVGFETurbulenceWithFractalNoiseWithStitching{..} |
            FilterGraphOp::SVGFETurbulenceWithTurbulenceNoiseWithNoStitching{..} |
            FilterGraphOp::SVGFETurbulenceWithTurbulenceNoiseWithStitching{..} => {
                // Turbulence produces pixel values throughout the
                // node subregion.
                used_subregion = full_subregion;
            }
        }
        // Store the subregion so later nodes can refer back
        // to this and propagate rects properly
        assert!((id as usize) < BUFFER_LIMIT, "BUFFER_LIMIT must be the same in frame building and scene building");
        subregion_by_buffer_id[id] = used_subregion;
    }
    subregion_by_buffer_id[filters.len() - 1]
}

/// Here we transform target rect to source rect for SVGFEGraph by walking
/// the whole graph and propagating subregions based on the provided
/// invalidation rect, and we want it to be a tight fit so we don't waste
/// time applying multiple filters to pixels that do not contribute to the
/// invalidated rect.
///
/// The interesting parts of the handling of SVG filters are:
/// * scene_building.rs : wrap_prim_with_filters
/// * picture.rs : get_coverage_target_svgfe
/// * picture.rs : get_coverage_source_svgfe (you are here)
/// * render_task.rs : new_svg_filter_graph
/// * render_target.rs : add_svg_filter_node_instances
pub fn get_coverage_source_svgfe(
    filters: &[(FilterGraphNode, FilterGraphOp)],
    surface_rect: LayoutRect,
) -> LayoutRect {

    // The value of BUFFER_LIMIT here must be the same as in
    // scene_building.rs, or we'll hit asserts here.
    const BUFFER_LIMIT: usize = SVGFE_GRAPH_MAX;

    // We're solving the source rect from target rect (e.g. due
    // to invalidation of a region, we need to know how much of
    // SourceGraphic is needed to draw that region accurately),
    // so we need to walk the DAG in reverse and accumulate the source
    // subregion for each input onto the referenced node, which can then
    // propagate that to its inputs when it is iterated.
    let mut source_subregion = LayoutRect::zero();
    let mut subregion_by_buffer_id: [LayoutRect; BUFFER_LIMIT] =
    [LayoutRect::zero(); BUFFER_LIMIT];
    let final_buffer_id = filters.len() - 1;
    assert!(final_buffer_id < BUFFER_LIMIT, "BUFFER_LIMIT must be the same in frame building and scene building");
    subregion_by_buffer_id[final_buffer_id] = surface_rect;
    for (node_buffer_id, (node, op)) in filters.iter().enumerate().rev() {
        // This is the subregion this node outputs, we can clip
        // the inputs based on source_padding relative to this,
        // and accumulate a new subregion for them.
        assert!(node_buffer_id < BUFFER_LIMIT, "BUFFER_LIMIT must be the same in frame building and scene building");
        let full_subregion = node.subregion;
        let mut used_subregion =
            subregion_by_buffer_id[node_buffer_id];
        // We can clip the propagated subregion to the node subregion before
        // we add source_padding for each input and propogate to them
        used_subregion = used_subregion
            .intersection(&full_subregion)
            .unwrap_or(LayoutRect::zero());
        if !used_subregion.is_empty() {
            for input in &node.inputs {
                let input_subregion = LayoutRect::new(
                    LayoutPoint::new(
                        used_subregion.min.x + input.source_padding.min.x,
                        used_subregion.min.y + input.source_padding.min.y,
                    ),
                    LayoutPoint::new(
                        used_subregion.max.x + input.source_padding.max.x,
                        used_subregion.max.y + input.source_padding.max.y,
                    ),
                );
                match input.buffer_id {
                    FilterOpGraphPictureBufferId::BufferId(id) => {
                        // Add the used area to the input, later when
                        // the referneced node is iterated as a node it
                        // will propagate the used bounds.
                        subregion_by_buffer_id[id as usize] =
                            subregion_by_buffer_id[id as usize]
                            .union(&input_subregion);
                    }
                    FilterOpGraphPictureBufferId::None => {}
                }
            }
        }
        // If this is the SourceGraphic or SourceAlpha, we now have the
        // source subregion we're looking for.  If both exist in the
        // same graph, we need to combine them, so don't merely replace.
        match op {
            FilterGraphOp::SVGFESourceAlpha |
            FilterGraphOp::SVGFESourceGraphic => {
                source_subregion = source_subregion.union(&used_subregion);
            }
            _ => {}
        }
    }

    // Note that this can be zero if SourceGraphic/SourceAlpha is not used
    // in this graph.
    source_subregion
}
