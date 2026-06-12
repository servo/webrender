/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::borrow::Cow;
use euclid::{
    Box2D, Point2D, Scale, Size2D, Transform3D, Vector2D, approxeq::ApproxEq as _,
    default, point2, point3,
};
use crate::units::{LayoutPixel, WorldPixel};

// Matches the definition of SK_ScalarNearlyZero in Skia.
const NEARLY_ZERO: f32 = 1.0 / 4096.0;

// Represents an optimized transform where there is only
// a scale and translation (which are guaranteed to maintain
// an axis align rectangle under transformation). The
// scaling is applied first, followed by the translation.
// TODO(gw): We should try and incorporate F <-> T units here,
//           but it's a bit tricky to do that now with the
//           way the current spatial tree works.
#[repr(C)]
#[derive(Debug, Clone, Copy, MallocSizeOf, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub struct ScaleOffset {
    pub scale: euclid::Vector2D<f32, euclid::UnknownUnit>,
    pub offset: euclid::Vector2D<f32, euclid::UnknownUnit>,
}

impl ScaleOffset {
    pub fn new(sx: f32, sy: f32, tx: f32, ty: f32) -> Self {
        ScaleOffset {
            scale: Vector2D::new(sx, sy),
            offset: Vector2D::new(tx, ty),
        }
    }

    pub fn identity() -> Self {
        ScaleOffset {
            scale: Vector2D::new(1.0, 1.0),
            offset: Vector2D::zero(),
        }
    }

    // Construct a ScaleOffset from a transform. Returns
    // None if the matrix is not a pure scale / translation.
    pub fn from_transform<F, T>(
        m: &Transform3D<f32, F, T>,
    ) -> Option<ScaleOffset> {

        // To check that we have a pure scale / translation:
        // Every field must match an identity matrix, except:
        //  - Any value present in tx,ty
        //  - Any value present in sx,sy

        if m.m12.abs() > NEARLY_ZERO ||
           m.m13.abs() > NEARLY_ZERO ||
           m.m14.abs() > NEARLY_ZERO ||
           m.m21.abs() > NEARLY_ZERO ||
           m.m23.abs() > NEARLY_ZERO ||
           m.m24.abs() > NEARLY_ZERO ||
           m.m31.abs() > NEARLY_ZERO ||
           m.m32.abs() > NEARLY_ZERO ||
           (m.m33 - 1.0).abs() > NEARLY_ZERO ||
           m.m34.abs() > NEARLY_ZERO ||
           m.m43.abs() > NEARLY_ZERO ||
           (m.m44 - 1.0).abs() > NEARLY_ZERO {
            return None;
        }

        Some(ScaleOffset {
            scale: Vector2D::new(m.m11, m.m22),
            offset: Vector2D::new(m.m41, m.m42),
        })
    }

    pub fn from_offset(offset: default::Vector2D<f32>) -> Self {
        ScaleOffset {
            scale: Vector2D::new(1.0, 1.0),
            offset,
        }
    }

    pub fn from_scale(scale: default::Vector2D<f32>) -> Self {
        ScaleOffset {
            scale,
            offset: Vector2D::new(0.0, 0.0),
        }
    }

    pub fn inverse(&self) -> Self {
        // If either of the scale factors is 0, inverse also has scale 0
        // TODO(gw): Consider making this return Option<Self> in future
        //           so that callers can detect and handle when inverse
        //           fails here.
        if self.scale.x.approx_eq(&0.0) || self.scale.y.approx_eq(&0.0) {
            return ScaleOffset::new(0.0, 0.0, 0.0, 0.0);
        }

        ScaleOffset {
            scale: Vector2D::new(
                1.0 / self.scale.x,
                1.0 / self.scale.y,
            ),
            offset: Vector2D::new(
                -self.offset.x / self.scale.x,
                -self.offset.y / self.scale.y,
            ),
        }
    }

    pub fn pre_offset(&self, offset: default::Vector2D<f32>) -> Self {
        self.pre_transform(
            &ScaleOffset {
                scale: Vector2D::new(1.0, 1.0),
                offset,
            }
        )
    }

    pub fn pre_scale(&self, scale: f32) -> Self {
        ScaleOffset {
            scale: self.scale * scale,
            offset: self.offset,
        }
    }

    pub fn then_scale(&self, scale: f32) -> Self {
        ScaleOffset {
            scale: self.scale * scale,
            offset: self.offset * scale,
        }
    }

    /// Produce a ScaleOffset that includes both self and other.
    /// The 'self' ScaleOffset is applied after `other`.
    /// This is equivalent to `Transform3D::pre_transform`.
    pub fn pre_transform(&self, other: &ScaleOffset) -> Self {
        ScaleOffset {
            scale: Vector2D::new(
                self.scale.x * other.scale.x,
                self.scale.y * other.scale.y,
            ),
            offset: Vector2D::new(
                self.offset.x + self.scale.x * other.offset.x,
                self.offset.y + self.scale.y * other.offset.y,
            ),
        }
    }

    /// Produce a ScaleOffset that includes both self and other.
    /// The 'other' ScaleOffset is applied after `self`.
    /// This is equivalent to `Transform3D::then`.
    #[allow(unused)]
    pub fn then(&self, other: &ScaleOffset) -> Self {
        ScaleOffset {
            scale: Vector2D::new(
                self.scale.x * other.scale.x,
                self.scale.y * other.scale.y,
            ),
            offset: Vector2D::new(
                other.scale.x * self.offset.x + other.offset.x,
                other.scale.y * self.offset.y + other.offset.y,
            ),
        }
    }


    pub fn map_rect<F, T>(&self, rect: &Box2D<f32, F>) -> Box2D<f32, T> {
        let x0 = rect.min.x * self.scale.x + self.offset.x;
        let y0 = rect.min.y * self.scale.y + self.offset.y;
        // TODO: If the supplied rect is invalid (has size < 0) we must ensure that the
        // returned rect has size zero else some tests fail. Using the max() of the min
        // and max points ensures that is the case. In future we could catch / assert /
        // fix these invalid rects earlier, and assert here instead.
        let x1 = rect.min.x.max(rect.max.x) * self.scale.x + self.offset.x;
        let y1 = rect.min.y.max(rect.max.y) * self.scale.y + self.offset.y;

        Box2D::new(
            Point2D::new(x0.min(x1), y0.min(y1)),
            Point2D::new(x0.max(x1), y0.max(y1)),
        )
    }

    pub fn unmap_rect<F, T>(&self, rect: &Box2D<f32, F>) -> Box2D<f32, T> {
        let x0 = (rect.min.x - self.offset.x) / self.scale.x;
        let y0 = (rect.min.y - self.offset.y) / self.scale.y;
        // TODO: If the supplied rect is invalid (has size < 0) we must ensure that the
        // returned rect has size zero else some tests fail. Using the max() of the min
        // and max points ensures that is the case. In future we could catch / assert /
        // fix these invalid rects earlier, and assert here instead.
        let x1 = (rect.min.x.max(rect.max.x) - self.offset.x) / self.scale.x;
        let y1 = (rect.min.y.max(rect.max.y) - self.offset.y) / self.scale.y;

        Box2D::new(
            Point2D::new(x0.min(x1), y0.min(y1)),
            Point2D::new(x0.max(x1), y0.max(y1)),
        )
    }

    pub fn map_vector<F, T>(&self, vector: &Vector2D<f32, F>) -> Vector2D<f32, T> {
        Vector2D::new(
            vector.x * self.scale.x,
            vector.y * self.scale.y,
        )
    }

    pub fn map_size<F, T>(&self, size: &Size2D<f32, F>) -> Size2D<f32, T> {
        Size2D::new(
            size.width * self.scale.x,
            size.height * self.scale.y,
        )
    }

    pub fn unmap_vector<F, T>(&self, vector: &Vector2D<f32, F>) -> Vector2D<f32, T> {
        Vector2D::new(
            vector.x / self.scale.x,
            vector.y / self.scale.y,
        )
    }

    pub fn map_point<F, T>(&self, point: &Point2D<f32, F>) -> Point2D<f32, T> {
        Point2D::new(
            point.x * self.scale.x + self.offset.x,
            point.y * self.scale.y + self.offset.y,
        )
    }

    pub fn unmap_point<F, T>(&self, point: &Point2D<f32, F>) -> Point2D<f32, T> {
        Point2D::new(
            (point.x - self.offset.x) / self.scale.x,
            (point.y - self.offset.y) / self.scale.y,
        )
    }

    pub fn to_transform<F, T>(&self) -> Transform3D<f32, F, T> {
        Transform3D::new(
            self.scale.x,
            0.0,
            0.0,
            0.0,

            0.0,
            self.scale.y,
            0.0,
            0.0,

            0.0,
            0.0,
            1.0,
            0.0,

            self.offset.x,
            self.offset.y,
            0.0,
            1.0,
        )
    }

    pub fn is_identity(&self) -> bool {
        self.scale.x == 1.0 &&
        self.scale.y == 1.0 &&
        self.offset.x == 0.0 &&
        self.offset.y == 0.0
    }

    pub fn is_reflection(&self) -> bool {
        self.scale.x < 0.0 ||
        self.scale.y < 0.0
    }
}

/// An enum that tries to avoid expensive transformation matrix calculations
/// when possible when dealing with non-perspective axis-aligned transformations.
#[derive(Debug, MallocSizeOf)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
pub enum FastTransform<Src, Dst> {
    /// A simple offset, which can be used without doing any matrix math.
    Offset(Vector2D<f32, Src>),

    /// A 2D transformation with an inverse.
    Transform {
        transform: Transform3D<f32, Src, Dst>,
        inverse: Option<Transform3D<f32, Dst, Src>>,
        is_2d: bool,
    },
}

impl<Src, Dst> Clone for FastTransform<Src, Dst> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Src, Dst> Copy for FastTransform<Src, Dst> { }

fn is_simple_translation<Src, Dst>(transform: &Transform3D<f32, Src, Dst>) -> bool {
    if (transform.m11 - 1.0).abs() > NEARLY_ZERO
        || (transform.m22 - 1.0).abs() > NEARLY_ZERO
        || (transform.m33 - 1.0).abs() > NEARLY_ZERO
        || (transform.m44 - 1.0).abs() > NEARLY_ZERO
    {
        return false;
    }

    transform.m12.abs() < NEARLY_ZERO
        && transform.m13.abs() < NEARLY_ZERO
        && transform.m14.abs() < NEARLY_ZERO
        && transform.m21.abs() < NEARLY_ZERO
        && transform.m23.abs() < NEARLY_ZERO
        && transform.m24.abs() < NEARLY_ZERO
        && transform.m31.abs() < NEARLY_ZERO
        && transform.m32.abs() < NEARLY_ZERO
        && transform.m34.abs() < NEARLY_ZERO
}

fn is_simple_2d_translation<Src, Dst>(transform: &Transform3D<f32, Src, Dst>) -> bool {
    if !is_simple_translation(transform) {
        return false;
    }
    transform.m43.abs() < NEARLY_ZERO
}

impl<Src, Dst> FastTransform<Src, Dst> {
    pub fn identity() -> Self {
        FastTransform::Offset(Vector2D::zero())
    }

    pub fn with_vector(offset: Vector2D<f32, Src>) -> Self {
        FastTransform::Offset(offset)
    }

    pub fn with_scale_offset(scale_offset: ScaleOffset) -> Self {
        if scale_offset.scale == Vector2D::new(1.0, 1.0) {
            FastTransform::Offset(Vector2D::from_untyped(scale_offset.offset))
        } else {
            FastTransform::Transform {
                transform: scale_offset.to_transform(),
                inverse: Some(scale_offset.inverse().to_transform()),
                is_2d: true,
            }
        }
    }

    #[inline(always)]
    pub fn with_transform(transform: Transform3D<f32, Src, Dst>) -> Self {
        if is_simple_2d_translation(&transform) {
            return FastTransform::Offset(Vector2D::new(transform.m41, transform.m42));
        }
        let inverse = transform.inverse();
        let is_2d = transform.is_2d();
        FastTransform::Transform { transform, inverse, is_2d}
    }

    pub fn to_transform(&self) -> Cow<Transform3D<f32, Src, Dst>> {
        match *self {
            FastTransform::Offset(offset) => Cow::Owned(
                Transform3D::translation(offset.x, offset.y, 0.0)
            ),
            FastTransform::Transform { ref transform, .. } => Cow::Borrowed(transform),
        }
    }

    /// Return true if this is an identity transform
    #[allow(unused)]
    pub fn is_identity(&self)-> bool {
        match *self {
            FastTransform::Offset(offset) => {
                offset == Vector2D::zero()
            }
            FastTransform::Transform { ref transform, .. } => {
                *transform == Transform3D::identity()
            }
        }
    }

    pub fn then<NewDst>(&self, other: &FastTransform<Dst, NewDst>) -> FastTransform<Src, NewDst> {
        match *self {
            FastTransform::Offset(offset) => match *other {
                FastTransform::Offset(other_offset) => {
                    FastTransform::Offset(offset + other_offset * Scale::<_, _, Src>::new(1.0))
                }
                FastTransform::Transform { transform: ref other_transform, .. } => {
                    FastTransform::with_transform(
                        other_transform
                            .with_source::<Src>()
                            .pre_translate(offset.to_3d())
                    )
                }
            }
            FastTransform::Transform { ref transform, ref inverse, is_2d } => match *other {
                FastTransform::Offset(other_offset) => {
                    FastTransform::with_transform(
                        transform
                            .then_translate(other_offset.to_3d())
                            .with_destination::<NewDst>()
                    )
                }
                FastTransform::Transform { transform: ref other_transform, inverse: ref other_inverse, is_2d: other_is_2d } => {
                    FastTransform::Transform {
                        transform: transform.then(other_transform),
                        inverse: inverse.as_ref().and_then(|self_inv|
                            other_inverse.as_ref().map(|other_inv| other_inv.then(self_inv))
                        ),
                        is_2d: is_2d & other_is_2d,
                    }
                }
            }
        }
    }

    pub fn pre_transform<NewSrc>(
        &self,
        other: &FastTransform<NewSrc, Src>
    ) -> FastTransform<NewSrc, Dst> {
        other.then(self)
    }

    pub fn pre_translate(&self, other_offset: Vector2D<f32, Src>) -> Self {
        match *self {
            FastTransform::Offset(offset) =>
                FastTransform::Offset(offset + other_offset),
            FastTransform::Transform { transform, .. } =>
                FastTransform::with_transform(transform.pre_translate(other_offset.to_3d()))
        }
    }

    pub fn then_translate(&self, other_offset: Vector2D<f32, Dst>) -> Self {
        match *self {
            FastTransform::Offset(offset) => {
                FastTransform::Offset(offset + other_offset * Scale::<_, _, Src>::new(1.0))
            }
            FastTransform::Transform { ref transform, .. } => {
                let transform = transform.then_translate(other_offset.to_3d());
                FastTransform::with_transform(transform)
            }
        }
    }

    #[inline(always)]
    pub fn is_backface_visible(&self) -> bool {
        match *self {
            FastTransform::Offset(..) => false,
            FastTransform::Transform { inverse: None, .. } => false,
            //TODO: fix this properly by taking "det|M33| * det|M34| > 0"
            // see https://www.w3.org/Bugs/Public/show_bug.cgi?id=23014
            FastTransform::Transform { inverse: Some(ref inverse), .. } => inverse.m33 < 0.0,
        }
    }

    #[inline(always)]
    pub fn transform_point2d(&self, point: Point2D<f32, Src>) -> Option<Point2D<f32, Dst>> {
        match *self {
            FastTransform::Offset(offset) => {
                let new_point = point + offset;
                Some(Point2D::from_untyped(new_point.to_untyped()))
            }
            FastTransform::Transform { ref transform, .. } => transform.transform_point2d(point),
        }
    }

    #[inline(always)]
    pub fn project_point2d(&self, point: Point2D<f32, Src>) -> Option<Point2D<f32, Dst>> {
        match* self {
            FastTransform::Offset(..) => self.transform_point2d(point),
            FastTransform::Transform{ref transform, ..} => {
                // Find a value for z that will transform to 0.

                // The transformed value of z is computed as:
                // z' = point.x * self.m13 + point.y * self.m23 + z * self.m33 + self.m43

                // Solving for z when z' = 0 gives us:
                let z = -(point.x * transform.m13 + point.y * transform.m23 + transform.m43) / transform.m33;

                transform.transform_point3d(point3(point.x, point.y, z)).map(| p3 | point2(p3.x, p3.y))
            }
        }
    }

    #[inline(always)]
    pub fn inverse(&self) -> Option<FastTransform<Dst, Src>> {
        match *self {
            FastTransform::Offset(offset) =>
                Some(FastTransform::Offset(Vector2D::new(-offset.x, -offset.y))),
            FastTransform::Transform { transform, inverse: Some(inverse), is_2d, } =>
                Some(FastTransform::Transform {
                    transform: inverse,
                    inverse: Some(transform),
                    is_2d
                }),
            FastTransform::Transform { inverse: None, .. } => None,

        }
    }
}

impl<Src, Dst> From<Transform3D<f32, Src, Dst>> for FastTransform<Src, Dst> {
    fn from(transform: Transform3D<f32, Src, Dst>) -> Self {
        FastTransform::with_transform(transform)
    }
}

impl<Src, Dst> From<Vector2D<f32, Src>> for FastTransform<Src, Dst> {
    fn from(vector: Vector2D<f32, Src>) -> Self {
        FastTransform::with_vector(vector)
    }
}

pub type LayoutFastTransform = FastTransform<LayoutPixel, LayoutPixel>;
pub type LayoutToWorldFastTransform = FastTransform<LayoutPixel, WorldPixel>;
