use std::collections::HashMap;
use std::sync::{Arc,Mutex};
use std::cell::RefCell;

//==----------------------------------------------------==//
//      Index
//==----------------------------------------------------==//

/// Index structure that allows us to retreive objects from the log.
/// It is just a simple wrapper over whatever data structure we wish
/// to eventually use.
pub struct Index<'a> {
    table: HashMap<&'a str, usize>,
}

impl<'a> Index<'a> {

    pub fn new() -> Self {
        Index {
            table: HashMap::new(), // also ::with_capacity(N)
        }
    }

    /// Return value of object if it exists, else None.
    pub fn get(&self, key: &'a str) -> Option<usize> {
        self.table.get(key).map(|r| *r) // &usize -> usize
    }

    /// Update location of object in the index. Returns None if object
    /// was newly inserted, or the virtual address of the prior
    /// object.
    pub fn update(&mut self, key: &'a str, value: usize) -> Option<usize> {
        self.table.insert(key, value)
    }

    /// Remove an entry. If it existed, return value, else return
    /// None.
    pub fn remove(&mut self, key: &'a str) -> Option<usize> {
        self.table.remove(key)
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }
}

//==----------------------------------------------------==//
//      Unit tests
//==----------------------------------------------------==//

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::mem::size_of;
    use std::mem::transmute;
    use std::ptr::copy;
    use std::ptr::copy_nonoverlapping;
    use std::rc::Rc;
    use std::sync::Arc;

    use test::Bencher;

    #[test]
    fn index_basic() {
        let mut index = Index::new();

        match index.update("alex", 42) {
            None => {}, // expected
            Some(v) => panic!("key should not exist"),
        }
        match index.update("alex", 24) {
            None => panic!("key should exist"),
            Some(v) => assert_eq!(v, 42),
        }

        match index.get("notexist") {
            None => {}, // ok
            Some(v) => panic!("get on nonexistent key"),
        }

        match index.get("alex") {
            None => panic!("key should exist"),
            Some(vref) => {}, // ok
        }
    }
}
