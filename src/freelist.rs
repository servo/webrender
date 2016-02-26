/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct FreeListItemId(u32);

impl FreeListItemId {
    #[inline]
    pub fn new(value: u32) -> FreeListItemId {
        FreeListItemId(value)
    }

    #[inline]
    pub fn value(&self) -> u32 {
        let FreeListItemId(value) = *self;
        value
    }
}

pub trait FreeListItem {
    fn next_free_id(&self) -> Option<FreeListItemId>;
    fn set_next_free_id(&mut self, id: Option<FreeListItemId>);
}

pub struct FreeList<T> {
    items: Vec<T>,
    first_free_index: Option<FreeListItemId>,
    alloc_count: usize,
}

impl<T: FreeListItem> FreeList<T> {
    pub fn new() -> FreeList<T> {
        FreeList {
            items: Vec::new(),
            first_free_index: None,
            alloc_count: 0,
        }
    }

    pub fn insert(&mut self, item: T) -> FreeListItemId {
        self.alloc_count += 1;
        match self.first_free_index {
            Some(free_index) => {
                let FreeListItemId(index) = free_index;
                let free_item = &mut self.items[index as usize];
                self.first_free_index = free_item.next_free_id();
                *free_item = item;
                free_index
            }
            None => {
                let item_id = FreeListItemId(self.items.len() as u32);
                self.items.push(item);
                item_id
            }
        }
    }

    #[allow(dead_code)]
    fn assert_not_in_free_list(&self, id: FreeListItemId) {
        let FreeListItemId(id) = id;
        let mut next_free_id = self.first_free_index;

        while let Some(free_id) = next_free_id {
            let FreeListItemId(index) = free_id;
            assert!(index != id);
            let free_item = &self.items[index as usize];
            next_free_id = free_item.next_free_id();
        }
    }

    pub fn get(&self, id: FreeListItemId) -> &T {
        //self.assert_not_in_free_list(id);

        let FreeListItemId(index) = id;
        &self.items[index as usize]
    }

    pub fn get_mut(&mut self, id: FreeListItemId) -> &mut T {
        //self.assert_not_in_free_list(id);

        let FreeListItemId(index) = id;
        &mut self.items[index as usize]
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.alloc_count
    }

    // TODO(gw): Actually free items from the texture cache!!
    #[allow(dead_code)]
    pub fn free(&mut self, id: FreeListItemId) {
        self.alloc_count -= 1;
        let FreeListItemId(index) = id;
        let item = &mut self.items[index as usize];
        item.set_next_free_id(self.first_free_index);
        self.first_free_index = Some(id);
    }

    /// NB: This iterates over free items too!
    pub fn iter_mut(&mut self) -> IterMut<T> {
        let first_free_index = self.first_free_index;
        let mut iterator = IterMut {
            free_list: self,
            next_index: Some(FreeListItemId(0)),
            next_free_index: first_free_index,
        };
        iterator.advance_past_free_indices();
        iterator
    }
}

pub struct IterMut<'a, T> where T: FreeListItem + 'a {
    free_list: &'a mut FreeList<T>,
    next_index: Option<FreeListItemId>,
    next_free_index: Option<FreeListItemId>,
}

impl<'a, T> IterMut<'a, T> where T: FreeListItem + 'a {
    fn advance_past_free_indices(&mut self) {
        loop {
            let next_index = match self.next_index {
                None => return,
                Some(next_index) => next_index.0 as usize,
            };
            if next_index == self.free_list.items.len() {
                self.next_index = None;
                return
            }
            let next_free_index = match self.next_free_index {
                None => return,
                Some(next_free_index) => next_free_index.0 as usize,
            };
            if next_index == next_free_index {
                self.next_index = Some(FreeListItemId(next_index as u32 + 1));
                continue
            }
            if next_free_index < next_index {
                self.next_free_index = self.free_list
                                           .items[next_free_index]
                                           .next_free_id();
                continue
            }
            debug_assert!(next_free_index > next_index);
            break
        }
    }

    pub fn free_list(&mut self) -> &mut FreeList<T> {
        self.free_list
    }
}

impl<'a, T> Iterator for IterMut<'a, T> where T: FreeListItem + 'a {
    type Item = FreeListItemId;

    fn next(&mut self) -> Option<FreeListItemId> {
        let next_index = match self.next_index {
            None => return None,
            Some(next_index) => next_index,
        };
        self.next_index = Some(FreeListItemId(next_index.0 + 1));
        self.advance_past_free_indices();
        Some(next_index)
    }
}

