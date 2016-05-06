use std::collections::HashMap;
use std::sync::{Arc,Mutex};

//==----------------------------------------------------==//
//      Index
//==----------------------------------------------------==//

pub type IndexRef = Arc<Mutex<Index>>;

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

    use super::super::logger;

    #[test]
    fn index_basic() {
        logger::enable();
        let mut index = Index::new();
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
