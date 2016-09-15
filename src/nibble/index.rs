use std::sync::Arc;

use hashtable::*;
use numa;
use common::{Pointer};

//==----------------------------------------------------==//
//      Index
//==----------------------------------------------------==//

pub type IndexRef = Arc<Index>;

/// Fat pointer as the value in every index entry.
/// | Allocator ID | Virtual Address |
///     16 bits     48 bits
pub type IndexEntry = u64;

/// Decompose an IndexEntry
#[inline(always)]
pub fn extract(entry: IndexEntry) -> (u16,u64) {
    ( (entry >> 48) as u16, entry & ((1u64<<48)-1) )
}

/// Create an index entry from the Socket ID and virtual address
#[inline(always)]
pub fn merge(socket: u16, va: u64) -> IndexEntry {
    ((socket as u64) << 48) | (va & ((1u64<<48)-1))
}

/// Index structure that allows us to retreive objects from the log.
/// It is just a simple wrapper over whatever data structure we wish
/// to eventually use.
pub struct Index {
    nnodes: usize,
    tables: Vec<Pointer<HashTable>>,
}

impl Index {

    pub fn new(n: usize, per: usize) -> Self {
        let mut tables: Vec<Pointer<HashTable>>;
        tables = Vec::with_capacity(n);
        let nsockets = numa::NODE_MAP.sockets();
        for i in 0..n {
            let sock = i % nsockets;
            let t = Box::new(HashTable::new(per, sock));
            let p = Pointer(Box::into_raw(t));
            tables.push(p);
        }
        Index {
            nnodes: numa::NODE_MAP.sockets(),
            tables: tables,
        }
    }

    /// Return value of object if it exists, else None.
    #[inline(always)]
    pub fn get(&self, key: u64) -> Option<IndexEntry> {
        debug_assert!(key > 0);
        let hash = HashTable::make_hash(key);
        let tidx = self.table_idx(hash);
        debug_assert!(tidx < self.tables.len());
        let ref p = self.tables[tidx];
        debug_assert!(!p.0 .is_null());
        let mut ret: Option<IndexEntry> = None;
        unsafe {
            let ht: &HashTable = &* p.0;
            let mut v: u64 = 0;
            if ht.get(key, &mut v) {
                ret = Some(v);
            }
        };
        ret
    }

    /// Update location of object in the index. Returns None if object
    /// was newly inserted, or the virtual address of the prior
    /// object.
    #[inline(always)]
    pub fn update(&self, key: u64, value: IndexEntry)
        -> (bool,Option<IndexEntry>) {

        debug_assert!(key > 0);
        let hash = HashTable::make_hash(key);
        let tidx = self.table_idx(hash);
        debug_assert!(tidx < self.tables.len());
        let ref p = self.tables[tidx];
        debug_assert!(!p.0 .is_null());
        unsafe {
            let ht: &HashTable = &* p.0;
            //ht.prefetchw(hash); // FIXME where to put?
            ht.put(key, value)
        }
    }

    /// Remove an entry. If it existed, return value, else return
    /// None.
    #[inline(always)]
    pub fn remove(&self, key: u64) -> Option<IndexEntry> {
        debug_assert!(key > 0);
        let hash = HashTable::make_hash(key);
        let tidx = self.table_idx(hash);
        debug_assert!(tidx < self.tables.len());
        let ref p = self.tables[tidx];
        debug_assert!(!p.0 .is_null());
        unsafe {
            let ht: &HashTable = &* p.0;
            let mut old: u64 = 0;
            if ht.del(key, &mut old) {
                Some(old)
            } else {
                None
            }
        }
    }

    pub fn update_lock_ifeq(&self, key: u64, new: u64, old: u64)
        -> Option<LockedBucket> {

        debug_assert!(key > 0);
        let hash = HashTable::make_hash(key);
        let tidx = self.table_idx(hash);
        debug_assert!(tidx < self.tables.len());
        let ref p = self.tables[tidx];
        debug_assert!(!p.0 .is_null());
        unsafe {
            let ht: &HashTable = &* p.0;
            ht.update_lock_ifeq(key, new, old)
        }
    }

    pub fn len(&self) -> usize {
        unimplemented!();
    }

    //
    // Priate methods
    //

    #[inline(always)]
    fn table_idx(&self, hash: u64) -> usize {
        (hash % (self.tables.len() as u64)) as usize
    }
}

//==----------------------------------------------------==//
//      Unit tests
//==----------------------------------------------------==//

mod tests {
    use super::*;
    use super::super::logger;
    use rand::{self,Rng};
    use std::sync::atomic::{AtomicUsize,Ordering};
    use crossbeam;
    use std::{thread, time};

    fn base(ntables: usize, nthreads: usize) {
        logger::enable();
        let nkeys = ntables*(1usize<<16);
        let keys_per = nkeys/nthreads;
        let cap = nkeys*10;

        let index = Index::new(ntables, cap);

        let mut guards = vec![];
        let tids = AtomicUsize::new(0);
        crossbeam::scope(|scope| {
            for _ in 0..(nthreads as u64) {
                let guard = scope.spawn(|| {

                    let mut rng = rand::thread_rng();
                    let tid = tids.fetch_add(1,
                                Ordering::Relaxed) as u64;

                    // +1 b/c zero is not a valid key
                    let start: u64 = tid * (keys_per as u64) + 1;
                    let end: u64 = start + keys_per as u64;

                    // offset each thread slightly
                    let dur = time::Duration::from_millis(tid*20);
                    thread::sleep(dur);

                    let value: u64 = rng.gen();

                    let niter = 1024/ntables;
                    for _ in 0..niter {

                        for k in start..end {
                            assert_eq!(index.get(k), None);
                        }

                        for k in start..end {
                            let (ok,opt) = index.update(k, value);
                            assert_eq!(ok, true);
                            assert_eq!(opt, None);
                            assert_eq!(index.get(k), Some(value));
                        }

                        for k in start..end {
                            let (ok,opt) = index.update(k, 0xffff);
                            assert_eq!(ok, true);
                            assert_eq!(opt, Some(value));
                        }

                        for k in start..end {
                            let (ok,opt) = index.update(k, value);
                            assert_eq!(ok, true);
                            assert_eq!(opt, Some(0xffff));
                        }

                        for k in start..end {
                            assert_eq!(index.remove(k), Some(value));
                            assert_eq!(index.get(k), None);
                            assert_eq!(index.remove(k), None);
                        }
                    }

                });
                guards.push(guard);
            }
        });

        for g in guards {
            g.join();
        }

    }

    #[test]
    fn simple() {
        base(1,1);
    }

    #[test]
    fn multiple_one_thread() {
        base(64,1);
    }

    #[test]
    fn simple_many() {
        base(1,12); 
    }

    #[test]
    fn multiple_many() {
        base(64,12);
    }
}
