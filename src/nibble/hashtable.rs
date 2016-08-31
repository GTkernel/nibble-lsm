/*
 * Concurrent hash table.
 *
 * TODO to grow, we'll need a reserve memory
 * Maybe I can use Linear Hashing
 */

use std::ptr;
use std::slice;
use std::fmt;
use std::hash;

use common::{Pointer, fnv1a, is_even, is_odd, atomic_add, atomic_cas};
use logger::*;

// 1. use versioned lock
// 2. use ticket lock
// 3. use array/queue lock?

const VERSION_MASK: u64 = 0x1;
const ENTRIES_PER_BUCKET: usize = 8;

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
    pub fn set_value(&self, idx: usize, value: u64) -> u64 {
        debug_assert!(idx < ENTRIES_PER_BUCKET);
        let ret: u64;
        unsafe {
            let v = self.value.get_unchecked(idx)
                        as *const u64 as *mut u64;
            // XXX will this ever be optimized out?
            //ptr::replace(v, value)
            ret = ptr::read_volatile(v);
            ptr::write_volatile(v, value);
        }
        ret
    }

    #[inline]
    pub fn del_key(&self, idx: usize, old: &mut u64) {
        debug_assert!(idx < ENTRIES_PER_BUCKET);
        unsafe {
            let k = self.key.get_unchecked(idx)
                        as *const u64 as *mut u64;
            let v = self.value.get_unchecked(idx)
                        as *const u64 as *mut u64;
            // XXX will this ever be optimized out?
            //ptr::replace(v, INVALID_KEY)
            *old = ptr::read_volatile(v);
            ptr::write_volatile(k, INVALID_KEY);
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

/// RAII-based lock used for clients of the API who must do
/// non-trivial amounts of work during a key update in the table.
/// Compaction, for example, is one.
pub struct LockedBucket {
    bucket: Pointer<Bucket>,
}

impl LockedBucket {

    /// Caller should lock bucket before creating
    fn new(bucket: &Bucket) -> Self {
        LockedBucket {
            bucket: Pointer(bucket as *const Bucket),
        }
    }

}

impl Drop for LockedBucket {
    fn drop(&mut self) {
        let b: &Bucket = unsafe { &*self.bucket.0 };
        b.unlock();
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

    // TODO numa init method
    // mmap the bucket array and treat as a slice

    /// does work of both insert and update
    /// (true,None) -> insertion
    /// (true,Some(x)) -> update existing, x is old value
    /// (false,None) -> cannot insert, table full
    /// (false,Some(x)) -> never returned
    pub fn put(&self, key: u64, value: u64) -> (bool,Option<u64>) {
        let hash = Self::make_hash(key);
        //let bidx = (hash & (self.nbuckets-1) as u64) as usize;
        let bidx = (hash % (self.nbuckets as u64)) as usize;
        let bucket: &Bucket = &self.buckets[bidx];

        let v = bucket.read_version();
        let mut opts = bucket.find_key(key);
        let (e,inv) = opts;
        if e.is_none() && inv.is_none() {
            return (false,None);
        }

        bucket.wait_lock();
        if bucket.read_version() != v {
            opts = bucket.find_key(key);
        }

        let (e,inv) = opts;
        let mut old: Option<u64> = None;
        let mut ok: bool = false;

        // if exists, overwrite
        if let Some(i) = e {
            old = Some(bucket.set_value(i, value));
            ok = true;
        }
        // else if there is an empty slot, use that
        else if let Some(i) = inv {
            bucket.set_key(i, key);
            bucket.set_value(i, value);
            ok = true;
        }

        bucket.unlock();
        (ok,old)
    }

    pub fn get(&self, key: u64, value: &mut u64) -> bool {
        let hash = Self::make_hash(key);
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

    pub fn del(&self, key: u64, old: &mut u64) -> bool {
        let hash = Self::make_hash(key);
        //let bidx = (hash & (self.nbuckets-1) as u64) as usize;
        let bidx = (hash % (self.nbuckets as u64)) as usize;
        let bucket: &Bucket = &self.buckets[bidx];

        let v = bucket.read_version();
        let mut opts = bucket.find_key(key);
        let (e,inv) = opts;
        if e.is_none() {
            return false;
        }

        bucket.wait_lock();
        if bucket.read_version() != v {
            opts = bucket.find_key(key);
        }

        let (e,inv) = opts;
        if e.is_none() {
            bucket.unlock();
            return false;
        }

        bucket.del_key(e.unwrap(), old);
        bucket.unlock();
        return true;
    }

    /// Grab the lock on the bucket holding the key only if the
    /// existing value matches one specified. Before returning,
    /// replace existing value with new.
    pub fn update_lock_ifeq(&self, key: u64, new: u64, old: u64)
        -> Option<LockedBucket> {

        let hash = Self::make_hash(key);
        //let bidx = (hash & (self.nbuckets-1) as u64) as usize;
        let bidx = (hash % (self.nbuckets as u64)) as usize;
        let bucket: &Bucket = &self.buckets[bidx];

        let v = bucket.read_version();
        let mut opts = bucket.find_key(key);
        let (e,inv) = opts;
        if e.is_none() {
            return None;
        }

        bucket.wait_lock();
        if bucket.read_version() != v {
            opts = bucket.find_key(key);
        }

        let (e,inv) = opts;
        if e.is_none() {
            bucket.unlock();
            return None;
        }

        let i = e.unwrap();
        if old == bucket.read_value(i) {
            bucket.set_value(i, new);
            Some(LockedBucket::new(&bucket))
        } else {
            bucket.unlock();
            None
        }
    }

    pub fn dump(&self) {
        for i in 0..self.nbuckets {
            debug!("[{}] {:?}", i, self.buckets[i]);
        }
    }

    #[inline]
    pub fn make_hash(value: u64) -> u64 {
        fnv1a(value)

        //let mut sip = hash::SipHasher::default();
        //Self::make_sip(&mut sip, value)
    }

    //
    // Private methods
    //

    #[inline]
    fn make_sip<T: hash::Hasher>(her: &mut T, value: u64) -> u64 {
        her.write_u64(value);
        her.finish()
    }

    fn bucket_idx(&self, hash: u64) {
        unimplemented!();
    }
}

impl Drop for HashTable {

    fn drop(&mut self) {
        // TODO release mmap data
    }
}

mod tests {
    use super::*;
    use super::ENTRIES_PER_BUCKET;
    use super::INVALID_KEY;

    use std::thread::{self,JoinHandle};
    use std::collections::HashSet;
    use std::ptr;
    use std::sync::atomic::{AtomicUsize,Ordering};

    use rand::{self,Rng};
    use crossbeam;

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
                let (ok,opt) = ht.put(key, 0xffff);
                if ok {
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
            let (ok,opt) = ht.put(key, 0xffff);
            assert_eq!(ok, true);
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
            let (ok,opt) = ht.put(key, 0xffff);
            if !ok {
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
            assert_eq!(ht.del(*key, &mut value), true);
        }

        // ht.dump();

        for key in &keys {
            assert_eq!(ht.del(*key, &mut value), false);
        }
        for key in &keys {
            assert_eq!(ht.get(*key, &mut value), false);
        }
    }

    fn threads_read_n(tblsz: usize, nthreads: usize) {
        logger::enable();
        println!("");

        let ht = HashTable::new(tblsz);

        let mut inserted: usize = 0;
        for key in 1..(tblsz+1) {
            let (ok,opt) = ht.put(key as u64, 0xffff);
            if ok {
                inserted += 1;
            }
        }
        //ht.dump();
        println!("keys inserted: {}", inserted);
        // only 'inserted' number of keys are valid in 0-tblsz

        let mut guards = vec![];

        crossbeam::scope(|scope| {
            for _ in 0..nthreads {
                let guard = scope.spawn(|| {
    
                    let mut keys: Vec<u64>;
    
                    keys = Vec::with_capacity((tblsz*2) as usize);
                    for k in 0..(tblsz*2) {
                        keys.push((k+1) as u64);
                    }
    
                    let mut value: u64 = 0;
                    let mut hits: usize;
                    for _ in 0..3 {
                        hits = 0;
                        shuffle(&mut keys);
                        for k in &keys {
                            if ht.get(*k, &mut value) {
                                hits += 1;
                                assert_eq!(value, 0xffff);
                                value = 0;
                            }
                        }
                        assert_eq!(hits, inserted);
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
    fn threads_read_sm() {
        threads_read_n(1024, 8);
    }

    #[test]
    fn threads_read_med() {
        threads_read_n(1_usize<<20, 8);
    }

    // each threads gets some disjoint subset of keys.
    // it will insert them, check they exist and values aren't corrupt
    // it table is full, it will remove some
    // mix put/get/del, always verifying its own set of keys
    fn threads_rw_n(tblsz: usize, nthreads: usize) {
        logger::enable();
        println!("");

        let ht = HashTable::new(tblsz);
        let total = tblsz;// >> 1; // no. keys to use
        let keys_per = total / nthreads;
        println!("keys_per {}", keys_per);

        let mut guards = vec![];

        let tids = AtomicUsize::new(0);
        crossbeam::scope(|scope| {
            for _ in 0..(nthreads as u64) {
                let guard = scope.spawn(|| {
                    let mut value: u64 = 0;
                    let tid = tids.fetch_add(1,
                                Ordering::Relaxed) as u64;
                    let mut rng = rand::thread_rng();
    
                    // keep track of which keys we think are in the
                    // table vs not. any inconsistency should indicate
                    // buggy implementation
                    let mut keys_out: Vec<u64> =
                        Vec::with_capacity(keys_per);
                    let mut keys_in: Vec<u64> =
                        Vec::with_capacity(keys_per);

                    // all keys start not inserted
                    let start: u64 = tid * (keys_per as u64);
                    for key in start..(start + (keys_per as u64)) {
                        keys_out.push(key+1); // key=0 is reserved
                    }
                    shuffle(&mut keys_out);

                    // blast insert our keys until full
                    loop {
                        let opt = keys_out.pop();
                        if opt.is_none() {
                            break;
                        }
                        let key = opt.unwrap();
                        let (ok,opt) = ht.put(key, tid ^ key);
                        if !ok {
                            break;
                        }
                        keys_in.push(key);
                    }

                    // verify keys
                    for key in &keys_in {
                        assert_eq!(ht.get(*key, &mut value), true);
                        assert_eq!(value, tid ^ *key);
                    }
                    for key in &keys_out {
                        assert_eq!(ht.get(*key, &mut value), false);
                    }

                    // del random no. of keys, check
                    // add random no. of keys, check
                    // repeat
                    for _ in 0..128 {

                        // do insertion
                        if rng.gen() {
                            let n = keys_in.len() / 2;
                            for _ in 0..n {
                                if keys_in.is_empty() {
                                    break;
                                }
                                let i = rng.next_u32() as usize
                                                % keys_in.len();
                                let key = keys_in.swap_remove(i);
                                let mut old: u64 = 0;
                                assert_eq!(ht.del(key,&mut old),true);
                                keys_out.push(key);
                            }
                        }

                        // do deletion
                        else {
                            let n = keys_out.len() / 2;
                            for _ in 0..n {
                                if keys_out.is_empty() {
                                    break;
                                }
                                let i = rng.next_u32() as usize
                                                % keys_out.len();
                                let key = keys_out.swap_remove(i);
                                assert_eq!(ht.get(key,&mut value),false);
                                let (ok,opt) = ht.put(key, tid ^ key);
                                if !ok {
                                    break;
                                }
                                keys_in.push(key);
                            }
                        }

                        // check consistency
                        for key in &keys_in {
                            assert_eq!(ht.get(*key, &mut value), true);
                            assert_eq!(value, tid ^ *key);
                        }
                        for key in &keys_out {
                            assert_eq!(ht.get(*key, &mut value), false);
                        }

                        shuffle(&mut keys_in);
                        shuffle(&mut keys_out);
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
    fn threads_rw() {
        threads_rw_n(1usize<<20, 16);
    }

    // TODO many threads
}
