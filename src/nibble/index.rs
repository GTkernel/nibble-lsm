use std::collections::HashMap;
use std::sync::{Arc,RwLock,RwLockWriteGuard};

//==----------------------------------------------------==//
//      Index
//==----------------------------------------------------==//

pub type IndexRef = Arc<Index>;
pub type IndexInner = HashMap<String, usize>;

/// Index structure that allows us to retreive objects from the log.
/// It is just a simple wrapper over whatever data structure we wish
/// to eventually use.
/// TODO what if i store the hash of a key instead of the key itself?
/// save space? all the keys are kept in the log anyway
pub struct Index {
    table: RwLock<IndexInner>,
}

impl Index {

    pub fn new() -> Self {
        Index {
            table: RwLock::new(HashMap::new()), // also ::with_capacity(N)
        }
    }

    /// Return value of object if it exists, else None.
    pub fn get(&self, key: &str) -> Option<usize> {
        let table = self.table.read().unwrap();
        table.get(key).map(|r| *r) // &usize -> usize
    }

    /// Update location of object in the index. Returns None if object
    /// was newly inserted, or the virtual address of the prior
    /// object.
    pub fn update(&self, key: &String, value: usize) -> Option<usize> {
        let mut guard = self.table.write().unwrap();
        Index::update_locked(&mut guard, key, value)
    }

    /// Same as update but you pass in the held lock
    pub fn update_locked(guard: &mut RwLockWriteGuard<IndexInner>,
                         key: &String, value: usize) -> Option<usize> {
        guard.insert(key.clone(), value)
    }

    /// Remove an entry. If it existed, return value, else return
    /// None.
    pub fn remove(&self, key: &String) -> Option<usize> {
        let mut table = self.table.write().unwrap();
        table.remove(key)
    }

    pub fn len(&self) -> usize {
        let table = self.table.read().unwrap();
        table.len()
    }

    pub fn wlock(&self) -> RwLockWriteGuard<IndexInner> {
        self.table.write().unwrap()
    }
}

//==----------------------------------------------------==//
//      Unit tests
//==----------------------------------------------------==//

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::logger;

    #[test]
    fn index_basic() {
        logger::enable();
        let index = Index::new();
        let key1 = String::from("alex");
        let key2 = String::from("notexist");
        assert_eq!(index.update(&key1, 42).is_some(), false);
        let opt = index.update(&key1, 24);
        assert_eq!(opt.is_some(), true);
        assert_eq!(opt.unwrap(), 42);
        assert_eq!(index.get(key2.as_str()).is_some(), false);
        assert_eq!(index.get(key1.as_str()).is_some(), true);
    }
}
