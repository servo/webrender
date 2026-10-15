/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::serde::{Serialize, Deserialize};
use crate::units::*;
use malloc_size_of::{MallocSizeOf, MallocSizeOfOps};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// The commands of a path. Each command consumes a number of points from the
/// path's point buffer, in order.
///
/// Every sub-path starts with `Begin` and finishes with either `Close` or
/// `End`.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
pub enum PathCommand {
    /// Consumes one point.
    LineTo,
    /// Consumes two points: the control point and the end point.
    QuadraticTo,
    /// Consumes three points: the two control points and the end point.
    CubicTo,
    /// Starts a sub-path. Consumes one point.
    Begin,
    /// Finishes a sub-path with a line back to its first point.
    Close,
    /// Finishes a sub-path without closing it.
    End,
}

/// Vector path geometry.
///
/// Paths are interned by the display list builder, so equality and hashing
/// compare the coordinates bit for bit.
#[derive(Clone, Serialize, Deserialize)]
pub struct Path {
    points: Arc<[LayoutPoint]>,
    commands: Arc<[PathCommand]>,
    /// Bounding box of the endpoints and control points. Derived from the
    /// points, so it does not take part in equality and hashing.
    aabb: LayoutRect,
}

impl PartialEq for Path {
    fn eq(&self, other: &Self) -> bool {
        if Arc::ptr_eq(&self.points, &other.points) && Arc::ptr_eq(&self.commands, &other.commands) {
            return true;
        }

        self.commands == other.commands
            && self.points.len() == other.points.len()
            && self.points.iter().zip(other.points.iter()).all(|(a, b)| {
                a.x.to_bits() == b.x.to_bits() && a.y.to_bits() == b.y.to_bits()
            })
    }
}

impl Eq for Path {}

impl Hash for Path {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.commands.hash(state);
        self.points.len().hash(state);
        for point in self.points.iter() {
            point.x.to_bits().hash(state);
            point.y.to_bits().hash(state);
        }
    }
}

impl MallocSizeOf for Path {
    fn size_of(&self, _ops: &mut MallocSizeOfOps) -> usize {
        // The buffers are shared between the content interner and every store
        // that resolved the path, so they are not attributed to any one owner.
        0
    }
}

impl Path {
    pub fn empty() -> Self {
        Path {
            points: Arc::new([]),
            commands: Arc::new([]),
            aabb: LayoutRect::zero(),
        }
    }

    /// Copies a path out of buffers in the same layout as a `Path`'s. The
    /// commands must form complete sub-paths and consume exactly `points`, and
    /// `aabb` must contain every point.
    pub fn from_slices(commands: &[PathCommand], points: &[LayoutPoint], aabb: LayoutRect) -> Self {
        #[cfg(debug_assertions)]
        {
            let mut validator = DebugValidator::new();
            let mut point_count = 0;
            for command in commands {
                match command {
                    PathCommand::Begin => {
                        validator.begin();
                        point_count += 1;
                    }
                    PathCommand::LineTo => {
                        validator.edge();
                        point_count += 1;
                    }
                    PathCommand::QuadraticTo => {
                        validator.edge();
                        point_count += 2;
                    }
                    PathCommand::CubicTo => {
                        validator.edge();
                        point_count += 3;
                    }
                    PathCommand::Close | PathCommand::End => validator.end(),
                }
            }
            validator.finish();
            assert_eq!(point_count, points.len(), "the commands do not match the points");
            assert!(points.iter().all(|p| aabb.contains_inclusive(*p)), "a point is outside of the aabb");
        }

        Path {
            points: points.into(),
            commands: commands.into(),
            aabb,
        }
    }

    /// A conservative bounding box of the path, containing its control points.
    ///
    /// Not finite if any of the path's points is not finite.
    pub fn aabb(&self) -> LayoutRect {
        self.aabb
    }

    pub fn iter(&self) -> PathIter {
        PathIter {
            points: self.points.iter(),
            commands: self.commands.iter(),
            current: LayoutPoint::zero(),
            first: LayoutPoint::zero(),
        }
    }
}

impl std::fmt::Debug for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for item in self.iter() {
            match item {
                PathEvent::Begin { at } => {
                    write!(f, "M {} {} ", at.x, at.y)?;
                }
                PathEvent::End { close, .. } => {
                    if close {
                        write!(f, "Z ")?;
                    }
                }
                PathEvent::Line { to, .. } => {
                    write!(f, "L {} {} ", to.x, to.y)?;
                }
                PathEvent::Quadratic { ctrl, to, .. } => {
                    write!(f, "Q {} {} {} {} ", ctrl.x, ctrl.y, to.x, to.y)?;
                }
                PathEvent::Cubic { ctrl1, ctrl2, to, .. } => {
                    write!(f, "C {} {} {} {} {} {} ", ctrl1.x, ctrl1.y, ctrl2.x, ctrl2.y, to.x, to.y)?;
                }
            }
        }

        Ok(())
    }
}

// See MAX_COORD in util.rs
const MAX_COORD: f32 = 1.0e9;

pub struct PathBuilder {
    points: Vec<LayoutPoint>,
    commands: Vec<PathCommand>,
    validator: DebugValidator,
    aabb: LayoutRect,
    valid: bool,
}

impl PathBuilder {
    pub fn new() -> Self {
        PathBuilder {
            points: Vec::new(),
            commands: Vec::new(),
            validator: DebugValidator::new(),
            aabb: LayoutRect {
                min: LayoutPoint::new(MAX_COORD, MAX_COORD),
                max: LayoutPoint::new(-MAX_COORD, -MAX_COORD),
            },
            valid: true,
        }
    }

    pub fn begin(&mut self, at: LayoutPoint) {
        self.validator.begin();
        self.points.push(at);
        self.commands.push(PathCommand::Begin);
        self.aabb.min = self.aabb.min.min(at);
        self.aabb.max = self.aabb.max.max(at);
        self.valid &= at.is_finite();
    }

    pub fn end(&mut self, close: bool) {
        self.validator.end();
        self.commands.push(if close { PathCommand::Close } else { PathCommand::End });
    }

    pub fn line_to(&mut self, to: LayoutPoint) {
        self.validator.edge();
        self.points.push(to);
        self.commands.push(PathCommand::LineTo);
        self.aabb.min = self.aabb.min.min(to);
        self.aabb.max = self.aabb.max.max(to);
        self.valid &= to.is_finite();
    }

    pub fn quadratic_bezier_to(&mut self, ctrl: LayoutPoint, to: LayoutPoint) {
        self.validator.edge();
        self.points.push(ctrl);
        self.points.push(to);
        self.commands.push(PathCommand::QuadraticTo);
        self.aabb.min = self.aabb.min.min(ctrl).min(to);
        self.aabb.max = self.aabb.max.max(ctrl).max(to);
        self.valid &= ctrl.is_finite() && to.is_finite();
    }

    pub fn cubic_bezier_to(&mut self, ctrl1: LayoutPoint, ctrl2: LayoutPoint, to: LayoutPoint) {
        self.validator.edge();
        self.points.push(ctrl1);
        self.points.push(ctrl2);
        self.points.push(to);
        self.commands.push(PathCommand::CubicTo);
        self.aabb.min = self.aabb.min.min(ctrl1).min(ctrl2).min(to);
        self.aabb.max = self.aabb.max.max(ctrl1).max(ctrl2).max(to);
        self.valid &= ctrl1.is_finite() && ctrl2.is_finite() && to.is_finite();
    }

    pub fn build(&mut self) -> Path {
        self.validator.finish();
        let aabb = if !self.valid {
            LayoutRect {
                min: LayoutPoint::new(f32::NAN, f32::NAN),
                max: LayoutPoint::new(f32::NAN, f32::NAN),
            }
        } else if self.points.is_empty() {
            LayoutRect::zero()
        } else {
            self.aabb
        };

        let path = Path {
            points: self.points.as_slice().into(),
            commands: self.commands.as_slice().into(),
            aabb,
        };

        self.points.clear();
        self.commands.clear();
        self.validator = DebugValidator::new();
        self.aabb = LayoutRect {
            min: LayoutPoint::new(MAX_COORD, MAX_COORD),
            max: LayoutPoint::new(-MAX_COORD, -MAX_COORD),
        };
        self.valid = true;

        path
    }
}

#[derive(Copy, Clone, Debug)]
pub enum PathEvent {
    Begin { at: LayoutPoint },
    End { last: LayoutPoint, first: LayoutPoint, close: bool },
    Line { from: LayoutPoint, to: LayoutPoint },
    Quadratic { from: LayoutPoint, ctrl: LayoutPoint, to: LayoutPoint },
    Cubic { from: LayoutPoint, ctrl1: LayoutPoint, ctrl2: LayoutPoint, to: LayoutPoint },
}

#[derive(Clone)]
pub struct PathIter<'l> {
    points: std::slice::Iter<'l, LayoutPoint>,
    commands: std::slice::Iter<'l, PathCommand>,
    current: LayoutPoint,
    first: LayoutPoint,
}

impl<'l> Iterator for PathIter<'l> {
    type Item = PathEvent;

    fn next(&mut self) -> Option<PathEvent> {
        match self.commands.next() {
            Some(&PathCommand::Begin) => {
                self.current = *self.points.next()?;
                self.first = self.current;
                Some(PathEvent::Begin { at: self.current })
            }
            Some(&PathCommand::LineTo) => {
                let from = self.current;
                self.current = *self.points.next()?;
                Some(PathEvent::Line {
                    from,
                    to: self.current,
                })
            }
            Some(&PathCommand::QuadraticTo) => {
                let from = self.current;
                let ctrl = *self.points.next()?;
                self.current = *self.points.next()?;
                Some(PathEvent::Quadratic {
                    from,
                    ctrl,
                    to: self.current,
                })
            }
            Some(&PathCommand::CubicTo) => {
                let from = self.current;
                let ctrl1 = *self.points.next()?;
                let ctrl2 = *self.points.next()?;
                self.current = *self.points.next()?;
                Some(PathEvent::Cubic {
                    from,
                    ctrl1,
                    ctrl2,
                    to: self.current,
                })
            }
            Some(&PathCommand::Close) => {
                let last = self.current;
                Some(PathEvent::End {
                    last,
                    first: self.first,
                    close: true,
                })
            }
            Some(&PathCommand::End) => {
                let last = self.current;
                self.current = self.first;
                Some(PathEvent::End {
                    last,
                    first: self.first,
                    close: false,
                })
            }
            None => None,
        }
    }
}



#[derive(Default, Copy, Clone, Debug, PartialEq)]
struct DebugValidator {
    #[cfg(debug_assertions)]
    in_subpath: bool,
}

impl DebugValidator {
    #[inline(always)]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    fn begin(&mut self) {
        #[cfg(debug_assertions)]
        {
            assert!(!self.in_subpath, "multiple begin() calls without end()");
            self.in_subpath = true;
        }
    }

    #[inline(always)]
    fn end(&mut self) {
        #[cfg(debug_assertions)]
        {
            assert!(self.in_subpath, "end() called without begin()");
            self.in_subpath = false;
        }
    }

    #[inline(always)]
    fn edge(&self) {
        #[cfg(debug_assertions)]
        assert!(self.in_subpath, "edge operation is made before begin()");
    }

    #[inline(always)]
    fn finish(&self) {
        #[cfg(debug_assertions)]
        assert!(!self.in_subpath, "build() called before end()");
    }
}
