use segment::EntryReference;

use std::collections::HashMap;
use std::sync::{Arc,Mutex};
use std::cell::RefCell;

//==----------------------------------------------------==//
//      Index
//==----------------------------------------------------==//

pub type IndexRef = Arc<Mutex<RefCell<Index>>>;

/// Index structure that allows us to retreive objects from the log.
/// It is just a simple wrapper over whatever data structure we wish
/// to eventually use.
/// TODO what if i store the hash of a key instead of the key itself?
/// save space? all the keys are kept in the log anyway
pub struct Index {
    table: HashMap<String, usize>,
}

impl Index {

    pub fn new() -> Self {
        Index {
            table: HashMap::new(), // also ::with_capacity(N)
        }
    }

    /// Return value of object if it exists, else None.
    pub fn get(&self, key: &str) -> Option<usize> {
        self.table.get(key).map(|r| *r) // &usize -> usize
    }

    /// Update location of object in the index. Returns None if object
    /// was newly inserted, or the virtual address of the prior
    /// object.
    pub fn update(&mut self, key: &String, value: usize) -> Option<usize> {
        self.table.insert(key.clone(), value)
    }

    /// Remove an entry. If it existed, return value, else return
    /// None.
    pub fn remove(&mut self, key: &String) -> Option<usize> {
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

    use super::super::logger;

    #[test]
    fn index_basic() {
        logger::enable();
        let mut index = Index::new();

        let key1 = String::from("alex");
        let key2 = String::from("notexist");

        match index.update(&key1, 42) {
            None => {}, // expected
            Some(v) => panic!("key should not exist"),
        }
        match index.update(&key1, 24) {
            None => panic!("key should exist"),
            Some(v) => assert_eq!(v, 42),
        }

        match index.get(key2.as_str()) {
            None => {}, // ok
            Some(v) => panic!("get on nonexistent key"),
        }

        match index.get(key1.as_str()) {
            None => panic!("key should exist"),
            Some(vref) => {}, // ok
        }
    }
}
