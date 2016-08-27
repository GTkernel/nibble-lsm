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

type find_ops = (Option<usize>, Option<usize>);

// most methods require 'volatile' operations because this is a
// concurrent hash table, and we cannot have the compiler optimizing
// with an assumption that values won't change underneath it
// TODO Use RAII for the locks
impl Bucket {

    #[inline]
    pub fn try_bump_version(&self, version: u64) -> bool {
        let v = &self.version as *const u64 as *mut u64;
        let (val,ok) = unsafe {
            atomic_cas(v, version, version+1)
        };
        ok
    }

    #[inline]
    pub fn bump_version(&self) {
        let v = &self.version as *const u64 as *mut u64;
        unsafe {
            atomic_add(v, 1);
            //volatile_add(v, 1);
        }
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

    #[inline]
    pub fn del_key(&self, idx: usize) {
        debug_assert!(idx < ENTRIES_PER_BUCKET);
        unsafe {
            let v = self.key.get_unchecked(idx)
                        as *const u64 as *mut u64;
            ptr::write_volatile(v, INVALID_KEY)
        }
    }

    #[inline]
    pub fn wait_version(&self) -> u64 {
        let mut v = self.read_version();
        loop {
            if is_even(v) { break; }
            v = self.read_version();
        }
        v
    }

    #[inline]
    pub fn wait_lock(&self) {
        let mut v;
        'retry: loop {
            v = self.read_version();
            if is_even(v) {
                if self.try_bump_version(v) {
                    return;
                } else {
                    continue 'retry;
                }
            }
        }
    }

    #[inline]
    pub fn unlock(&self) {
        self.bump_version();
    }

    #[inline]
    pub fn find_key(&self, key: u64) -> find_ops {
        let mut idx: Option<usize> = None;
        let mut inv: Option<usize> = None;
        for i in 0..ENTRIES_PER_BUCKET {
            let k = self.read_key(i);
            if key == k {
                idx = Some(i);
                break;
            } else if INVALID_KEY == k {
                inv = Some(i);
            }
        }
        (idx,inv)
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

    // does work of both insert and update
    pub fn put(&self, key: u64, value: u64) -> bool {
        let hash = self.make_hash(key);
        //let bidx = (hash & (self.nbuckets-1) as u64) as usize;
        let bidx = (hash % (self.nbuckets as u64)) as usize;
        let bucket: &Bucket = &self.buckets[bidx];

        let v = bucket.read_version();
        let mut opts = bucket.find_key(key);
        if opts.0 .is_none() && opts.1 .is_none() {
            trace!("bidx {} {:?}", bidx, bucket);
            return false;
        }

        bucket.wait_lock();
        if bucket.read_version() != v {
            opts = bucket.find_key(key);
        }

        let mut ret: bool = false;

        // if exists, overwrite
        if let Some(idx) = opts.0 {
            bucket.set_value(idx, value);
            ret = true;
        }
        // else if there is an empty slot, use that
        else if let Some(inv) = opts.1 {
            bucket.set_key(inv, key);
            bucket.set_value(inv, value);
            ret = true;
        }
        else {
            trace!("bidx {} {:?}", bidx, bucket);
        }

        bucket.unlock();
        ret
    }

    pub fn get(&self, key: u64, value: &mut u64) -> bool {
        let hash = self.make_hash(key);
        //let bidx = (hash & (self.nbuckets-1) as u64) as usize;
        let bidx = (hash % (self.nbuckets as u64)) as usize;
        let bucket: &Bucket = &self.buckets[bidx];

        // TODO if we retry too many times, perhaps forcefully lock

        'retry: loop {
            let v = bucket.wait_version();
            for i in 0..ENTRIES_PER_BUCKET {
                if bucket.read_key(i) == key {
                    *value = bucket.read_value(i);
                    if v != bucket.read_version() {
                        continue 'retry;
                    }
                    return true;
                }
            }
            return false;
        }
    }

    pub fn del(&self, key: u64) -> bool {
        let hash = self.make_hash(key);
        //let bidx = (hash & (self.nbuckets-1) as u64) as usize;
        let bidx = (hash % (self.nbuckets as u64)) as usize;
        let bucket: &Bucket = &self.buckets[bidx];

        let v = bucket.read_version();
        let mut opts = bucket.find_key(key);
        if opts.0 .is_none() {
            return false;
        }

        bucket.wait_lock();
        if bucket.read_version() != v {
            opts = bucket.find_key(key);
        }

        if opts.0 .is_none() {
            bucket.unlock();
            return false;
        }

        bucket.del_key(opts.0 .unwrap());
        bucket.unlock();
        return true;
    }

    pub fn dump(&self) {
        for i in 0..self.nbuckets {
            debug!("[{}] {:?}", i, self.buckets[i]);
        }
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
        fnv1a(value)

        //let mut sip = hash::SipHasher::default();
        //Self::make_sip(&mut sip, value)
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

    use rand::{self,Rng};

    use common::*;
    use logger;

    #[test]
    fn init() {
        logger::enable();
        println!("");
        let ht = HashTable::new(1024);
        let nb = 1024 / ENTRIES_PER_BUCKET;
        assert_eq!(ht.nbuckets, nb);
        assert_eq!(ht.buckets.len(), nb);
    }

    // generate random set of keys and attempt to check existence
    #[test]
    fn get_on_empty() {
        logger::enable();
        println!("");

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
        println!("");
        let mut key: u64;

        let ntables = 16;
        let tblsz = 1usize << 18;

        let mut tbl_fills: usize = 0;
        let mut rng = rand::thread_rng();

        'outer: for _ in 0..ntables {
            let ht = HashTable::new(tblsz);
            let mut inserted = 0;
            key = rng.gen::<u64>();
            loop {
                if ht.put(key, 0xffff) {
                    inserted += 1;
                } else {
                    tbl_fills += inserted;
                    continue 'outer;
                }
                key = rng.gen::<u64>();
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
        println!("");

        let ht = HashTable::new(1024);
        let mut value: u64 = 0;
        let mut keys: HashSet<u64> = HashSet::with_capacity(8192);

        let mut rng = rand::thread_rng();
        while keys.len() < 8192 {
            let key = rng.gen::<u64>();
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

    #[test]
    fn put_del() {
        logger::enable();
        println!("");

        let ht = HashTable::new(1024);
        let mut keys: Vec<u64> = Vec::new();

        let mut rng = rand::thread_rng();
        loop {
            let key = rng.gen::<u64>();
            if !ht.put(key, 0xffff) {
                break;
            }
            keys.push(key);
        }

        let mut value: u64 = 0;
        for key in &keys {
            assert_eq!(ht.get(*key, &mut value), true);
            assert_eq!(value, 0xffff);
            value = 0;
        }

        for key in &keys {
            assert_eq!(ht.del(*key), true);
        }

        // ht.dump();

        for key in &keys {
            assert_eq!(ht.del(*key), false);
        }
        for key in &keys {
            assert_eq!(ht.get(*key, &mut value), false);
        }
    }

    // TODO many threads
    // TODO fill up
    // TODO delete some, all
}

