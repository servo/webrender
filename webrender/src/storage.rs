/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::{iter::Extend, ops::{Index, IndexMut}, marker::PhantomData};

#[macro_export]
macro_rules! storage_index_impl {
    ($name: ident) => {
        #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
        #[cfg_attr(feature = "capture", derive(Serialize))]
        #[cfg_attr(feature = "replay", derive(Deserialize))]
        pub struct $name(u32);

        impl From<usize> for $name {
            fn from(x: usize) -> Self {
                debug_assert!(x < u32::max_value() as _);
                $name(x as u32)
            }
        }

        impl Into<usize> for $name {
            fn into(self) -> usize {
                self.0 as _
            }
        }
    };
    ($($name: ident,)*) => { $(storage_index_impl!{ $name })* };
    ($($name: ident),*) => { storage_index_impl!{ $($name,)* } };
}

#[derive(Debug, Copy, Clone)]
pub struct Range<I> {
    start: I,
    end: I,
}

impl<I: From<usize>> Default for Range<I> {
    fn default() -> Self {
        Range {
            start: 0usize.into(),
            end: 0usize.into(),
        }
    }
}

impl<I: Into<usize> + PartialOrd> Range<I> {
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

pub struct Storage<T, I> {
    data: Vec<T>,
    _marker: PhantomData<I>
}

impl<T, I: Into<usize> + From<usize>> Storage<T, I> {
    pub fn new() -> Self {
        Storage { data: vec![], _marker: PhantomData }
    }

    pub fn push(&mut self, t: T) -> I {
        let index = self.data.len();
        self.data.push(t);
        index.into()
    }

    pub fn extend<II: IntoIterator<Item=T>>(&mut self, iter: II) -> Range<I> {
        let start = self.data.len().into();
        self.data.extend(iter);
        let end = self.data.len().into();
        Range { start, end }
    }
}

impl<T, I: From<usize> + Into<usize>> Index<I> for Storage<T, I> {
    type Output = T;
    fn index(&self, index: I) -> &Self::Output {
        &self.data[index.into()]
    }
}

impl<T, I: From<usize> + Into<usize>> IndexMut<I> for Storage<T, I> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.data[index.into()]
    }
}

impl<T, I: From<usize> + Into<usize>> Index<Range<I>> for Storage<T, I> {
    type Output = [T];
    fn index(&self, index: Range<I>) -> &Self::Output {
        &self.data[index.start.into()..index.end.into()]
    }
}

impl<T, I: From<usize> + Into<usize>> IndexMut<Range<I>> for Storage<T, I> {
    fn index_mut(&mut self, index: Range<I>) -> &mut Self::Output {
        &mut self.data[index.start.into()..index.end.into()]
    }
}
