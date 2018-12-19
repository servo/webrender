/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::marker::PhantomData;
use std::ops::{Index, IndexMut, RangeBounds};
use std::{slice, u32, vec};

/// Represents some newtyped `usize` wrapper.
pub trait Idx: Copy + Eq + 'static {
    fn new(idx: usize) -> Self;
    fn index(self) -> usize;
}

impl Idx for usize {
    fn new(idx: usize) -> Self { idx }
    fn index(self) -> usize { self }
}

impl Idx for u32 {
    fn new(idx: usize) -> Self { assert!(idx <= u32::MAX as usize); idx as u32 }
    fn index(self) -> usize { self as usize }
}

/// This custom `IndexVec` type is not only generic over the element type `T`,
/// but also over the index type, `I`, and thus allows you to use a newtype
/// wrapper around `u32`.
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct IndexVec<I: Idx, T> {
    pub raw: Vec<T>,
    _marker: PhantomData<I>
}

impl<I: Idx, T> IndexVec<I, T> {
    #[inline]
    pub fn new() -> Self {
        IndexVec { raw: Vec::new(), _marker: PhantomData }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        IndexVec { raw: Vec::with_capacity(capacity), _marker: PhantomData }
    }

    #[inline]
    pub fn from_elem_n(elem: T, n: usize) -> Self
        where T: Clone
    {
        IndexVec { raw: vec![elem; n], _marker: PhantomData }
    }

    #[inline]
    pub fn push(&mut self, d: T) -> I {
        let idx = I::new(self.len());
        self.raw.push(d);
        idx
    }

    #[inline]
    pub fn clear(&mut self) {
        self.raw.clear()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.raw.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    #[inline]
    pub fn iter(&self) -> slice::Iter<T> {
        self.raw.iter()
    }

    #[inline]
    pub fn iter_enumerated<'a>(&'a self) -> impl Iterator<Item=(I, &T)> + 'a
    {
        self.raw.iter().enumerate().map(|(i, e)| (I::new(i), e))
    }

    #[inline]
    pub fn iter_enumerated_mut<'a>(&'a mut self) -> impl Iterator<Item=(I, &mut T)> + 'a
    {
        self.raw.iter_mut().enumerate().map(|(i, e)| (I::new(i), e))
    }

    #[inline]
    pub fn drain<'a, R>(&'a mut self, range: R) -> impl Iterator<Item=T> + 'a
    where
        R: RangeBounds<usize>
    {
        self.raw.drain(range)
    }

    #[inline]
    pub fn get(&self, index: I) -> Option<&T> {
        self.raw.get(index.index())
    }

    #[inline]
    pub fn get_mut(&mut self, index: I) -> Option<&mut T> {
        self.raw.get_mut(index.index())
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.raw.as_ptr()
    }
}

impl<I: Idx, T> Default for IndexVec<I, T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<I: Idx, T> Index<I> for IndexVec<I, T> {
    type Output = T;

    #[inline]
    fn index(&self, index: I) -> &T {
        &self.raw[index.index()]
    }
}

impl<I: Idx, T> IndexMut<I> for IndexVec<I, T> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut T {
        &mut self.raw[index.index()]
    }
}

impl<I: Idx, T> IntoIterator for IndexVec<I, T> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    #[inline]
    fn into_iter(self) -> vec::IntoIter<T> {
        self.raw.into_iter()
    }

}

impl<'a, I: Idx, T> IntoIterator for &'a IndexVec<I, T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> slice::Iter<'a, T> {
        self.raw.iter()
    }
}

impl<'a, I: Idx, T> IntoIterator for &'a mut IndexVec<I, T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.raw.iter_mut()
    }
}
