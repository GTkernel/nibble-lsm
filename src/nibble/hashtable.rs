/*
 * Concurrent hash table.
 *
 * TODO to grow, we'll need a reserve memory
 */

use std::ptr;
use std::slice;
use std::fmt;
use std::hash;

use common::{fnv1a, is_even, is_odd, atomic_add, atomic_cas};
use logger::*;

// 1. use versioned lock
// 2. use ticket lock
// 3. use array/queue lock?

const VERSION_MASK: u64 = 0x1;
const ENTRIES_PER_BUCKET: usize = 4;

// We reserve this value to indicate the bucket slot is empty.
const INVALID_KEY: u64 = 0u64;

struct Bucket {
    version: u64,
    key: [u64; ENTRIES_PER_BUCKET],
    value: [u64; ENTRIES_PER_BUCKET],
    // TODO ptr to next in chain
}

impl fmt::Debug for Bucket {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "Bucket {{ ver {} key {:x} {:x} {:x} {:x} val {:x} {:x} {:x} {:x}  }}",
               self.version,
               self.key[0], self.key[1], self.key[2], self.key[3],
               self.value[0], self.value[1], self.value[2], self.value[3]
               )
    }
}

// most methods require 'volatile' operations because this is a
// concurrent hash table, and we cannot have the compiler optimizing
// with an assumption that values won't change underneath it
impl Bucket {

    #[inline]
    pub fn try_bump_version(&self, version: u64) -> bool {
        let v = &self.version as *const u64 as *mut u64;
        // volatile_add(v, 1);
        let (val,ok) = unsafe {
            atomic_cas(v, version, version+1)
        };
        ok
    }

    #[inline]
    pub fn bump_version(&self) {
        let v = &self.version as *const u64 as *mut u64;
        unsafe {
            let old = *v;
            debug_assert!(ptr::read_volatile(v) < 1_000); // XXX remove
            atomic_add(v, 1);
            debug_assert!(old+1 == ptr::read_volatile(v));
        }
        //volatile_add(v, 1);
    }

    #[inline]
    pub fn read_version(&self) -> u64 {
        let v = &self.version as *const u64;
        unsafe { ptr::read_volatile(v) }
    }

    #[inline]
    pub fn read_key(&self, idx: usize) -> u64 {
        debug_assert!(idx < ENTRIES_PER_BUCKET);
        unsafe {
            let k = self.key.get_unchecked(idx) as *const u64;
            ptr::read_volatile(k)
        }
    }

    #[inline]
    pub fn read_value(&self, idx: usize) -> u64 {
        debug_assert!(idx < ENTRIES_PER_BUCKET);
        unsafe {
            let v = self.value.get_unchecked(idx) as *const u64;
            ptr::read_volatile(v)
        }
    }

    #[inline]
    pub fn set_key(&self, idx: usize, key: u64) {
        debug_assert!(idx < ENTRIES_PER_BUCKET);
        debug_assert!(key != INVALID_KEY);
        unsafe {
            let k = self.key.get_unchecked(idx)
                        as *const u64 as *mut u64;
            ptr::write_volatile(k, key);
        }
    }

    #[inline]
    pub fn set_value(&self, idx: usize, value: u64) {
        debug_assert!(idx < ENTRIES_PER_BUCKET);
        unsafe {
            let v = self.value.get_unchecked(idx)
                        as *const u64 as *mut u64;
            ptr::write_volatile(v, value)
        }
    }
}

// TODO make this some inner type
// TODO initialize one per core, or per socket, etc.
// TODO May need to use
// the NUMA memory allocation support
pub struct HashTable {
    nbuckets: usize,
    buckets: Vec<Bucket>,
}

impl HashTable {

    pub fn new(entries: usize) -> Self {
        let nbuckets = entries / ENTRIES_PER_BUCKET;
        let mut v: Vec<Bucket> = Vec::with_capacity(nbuckets);
        unsafe {
            v.set_len(nbuckets);
        }
        for b in &mut v {
            b.version = 0;
            for i in 0..ENTRIES_PER_BUCKET {
                b.key[i] = INVALID_KEY;
                b.value[i] = 0;
            }
        }
        HashTable {
            nbuckets: nbuckets,
            buckets: v,
        }
    }

    pub fn put(&self, key: u64, value: u64) -> bool {
        let hash = self.make_hash(key);
        //let bidx = (hash & (self.nbuckets-1) as u64) as usize;
        let bidx = (hash % (self.nbuckets as u64)) as usize;
        let bucket: &Bucket = &self.buckets[bidx];

        trace!("self.nbuckets {}", self.nbuckets);
        trace!("key {:x} value {:x}", key, value);
        trace!("hash {:x} bidx {}", hash, bidx);
        trace!("version (before) {}", bucket.read_version());

        trace!("{:?}", bucket);
        debug_assert!(bucket.read_version() < 1_000);

        let mut v;
        'retry: loop {
            v = bucket.read_version();
            if is_even(v) {
                if bucket.try_bump_version(v) {
                    break;
                } else {
                    continue 'retry;
                }
            }
        }
        debug_assert!(v + 1 == bucket.read_version());
        trace!("version {}", v+1);

        let mut idx = usize::max_value();
        let mut inv = usize::max_value();

        // TODO search first, then bump version
        // to reduce critical section (behave like a reader)

        for i in 0..ENTRIES_PER_BUCKET {
            let k = bucket.read_key(i);
            if key == k {
                idx = i;
                break;
            } else if INVALID_KEY == k {
                inv = i;
            }
        }

        // if exists, overwrite
        if idx != usize::max_value() {
            trace!("found existing at {}", idx);
            debug_assert!(bucket.read_key(idx) == key);
            bucket.set_value(idx, value);
            bucket.bump_version();
            debug_assert!(v + 2 == bucket.read_version());
            return true;
        }
        // else if there is an empty slot, use that
        else if inv != usize::max_value() {
            trace!("not exist, but found empty at {}", inv);
            bucket.set_key(inv, key);
            bucket.set_value(inv, value);
            bucket.bump_version();
            debug_assert!(v + 2 == bucket.read_version());
            return true;
        }
        // else, no space to insert
        else {
            trace!("not found and no empty slots");
            bucket.bump_version();
            trace!("{:?}", bucket);
            debug_assert!(v + 2 == bucket.read_version());
            return false;
        }
    }

    pub fn get(&self, key: u64, value: &mut u64) -> bool {
        let hash = self.make_hash(key);
        //let bidx = (hash & (self.nbuckets-1) as u64) as usize;
        let bidx = (hash % (self.nbuckets as u64)) as usize;
        let bucket: &Bucket = &self.buckets[bidx];

        // TODO if we retry too many times, perhaps forcefully lock

        'retry: loop {
            let v = bucket.read_version();
            if is_odd(v) {
                continue 'retry;
            }
            for i in 0..ENTRIES_PER_BUCKET {
                if bucket.read_key(i) == key {
                    *value = bucket.read_value(i);
                    if v != bucket.read_version() {
                        continue 'retry;
                    }
                    return true;
                }
            }
            // FIXME i think we only need to check retry on a hit
            if v != bucket.read_version() {
                continue 'retry;
            }
            return false;
        }
    }

    pub fn del(&self, key: u64) -> bool {
        unimplemented!();
    }

    //
    // Private methods
    //

    #[inline]
    fn make_sip<T: hash::Hasher>(her: &mut T, value: u64) -> u64 {
        her.write_u64(value);
        her.finish()
    }

    #[inline]
    fn make_hash(&self, value: u64) -> u64 {
        //fnv1a(value)

        let mut sip = hash::SipHasher::default();
        Self::make_sip(&mut sip, value)
    }

    fn bucket_idx(&self, hash: u64) {
        unimplemented!();
    }
}

mod tests {
    use super::*;
    use super::ENTRIES_PER_BUCKET;
    use super::INVALID_KEY;

    use std::thread::{self,JoinHandle};
    use std::collections::HashSet;
    use std::ptr;

    use common::*;
    use logger;

    //#[test]
    fn init() {
        let ht = HashTable::new(1024);
        let nb = 1024 / ENTRIES_PER_BUCKET;
        assert_eq!(ht.nbuckets, nb);
        assert_eq!(ht.buckets.len(), nb);
    }

    // generate random set of keys and attempt to check existence
    #[test]
    fn get_on_empty() {
        logger::enable();

        let ht = HashTable::new(1024);
        let mut value: u64 = 0;
        for i in 0..8192 {
            let mut key;
            loop {
                key = unsafe { rdrandq() };
                if INVALID_KEY != key {
                    break;
                }
            }
            assert_eq!(ht.get(key, &mut value), false);
        }
    }

    #[test]
    fn put_some() {
        logger::enable();
        let mut key: u64;

        let ntables = 16;
        let tblsz = 1usize << 20;

        let mut tbl_fills: usize = 0;

        'outer: for _ in 0..ntables {
            let ht = HashTable::new(tblsz);
            let mut inserted = 0;
            key = unsafe { rdrandq() };
            loop {
                if ht.put(key, 0xffff) {
                    inserted += 1;
                } else {
                    tbl_fills += inserted;
                    continue 'outer;
                }
                key += 1;
            }
        }
        assert!(tbl_fills > 0);

        println!("");
        let avg = tbl_fills as f64 / ntables as f64;
        let ratio = avg / tblsz as f64;
        println!("avg fill amt {:.2} (out of {})",avg,tblsz);
        println!("fill ratio {:.2}%", ratio * 100f64);
    }

    // insert random keys, check they exist, and that others don't
    #[test]
    fn set_check_and_notexist() {
        logger::enable();

        let ht = HashTable::new(1024);
        let mut value: u64 = 0;
        let mut keys: HashSet<u64> = HashSet::with_capacity(8192);

        while keys.len() < 8192 {
            let key = unsafe { rdrandq() };
            if INVALID_KEY == key {
                continue;
            }
            keys.insert(key);
        }

        let mut drain = keys.drain();

        let mut ins: Vec<u64> = Vec::with_capacity(128);
        for _ in 0..128 {
            let key = drain.next().unwrap();
            ins.push(key);
            assert_eq!(ht.put(key, 0xffff), true);
        }

        for key in &ins {
            assert_eq!(ht.get(*key, &mut value), true);
            assert_eq!(value, 0xffff);
        }

        for key in drain {
            assert_eq!(ht.get(key, &mut value), false);
        }
    }

    // TODO many threads
    // TODO fill up
    // TODO delete some, all
}

