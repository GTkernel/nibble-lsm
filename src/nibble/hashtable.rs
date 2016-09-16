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
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};
use std::intrinsics;

use num::Integer;

use memory::*;
use common::{Pointer, fnv1a, is_even, is_odd, atomic_add, atomic_cas};
use common::{prefetchw};
use logger::*;
use numa::{self,NodeId};
use clock;

// Methods for locking a bucket
// 1. use versioned lock
// 2. use ticket lock
// 3. use array/queue lock?

// Methods for resizing
// 1. Lock entire table, allocate new one, copy over. This is what
//    nearly all hash table implementations do.
// 2. Linear hashing (not linear probing; lock one bucket at a time,
//    relocate objects among new bucket and old)

// When we allocate a table, mmap much more than we need.
// This way, resizing will not put mmap on the critical path,
// as mmap does not scale with many threads.

/// Over-allocate virtual memory to avoid invoking mmap.
const TABLE_VLEN: usize = 1usize << 35;

const VERSION_MASK: u64 = 0x1;
const ENTRIES_PER_BUCKET: usize = 3;

/// We reserve this value to indicate the bucket slot is empty.
const INVALID_KEY: u64 = 0u64;

struct Bucket {
    version: u64,
    key:    [u64; ENTRIES_PER_BUCKET],
    value:  [u64; ENTRIES_PER_BUCKET],
}

impl fmt::Debug for Bucket {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "Bucket {{ ver {} key {:x} {:x} {:x}  }}",
               self.version,
               self.key[0], self.key[1], self.key[2])
    }
}

type find_ops = (Option<usize>, Option<usize>);

// most methods require 'volatile' operations because this is a
// concurrent hash table, and we cannot have the compiler optimizing
// with an assumption that values won't change underneath it
// TODO Use RAII for the locks
impl Bucket {

    #[inline(always)]
    pub fn try_bump_version(&self, version: u64) -> bool {
        let v = &self.version as *const u64 as *mut u64;
        let (val,ok) = unsafe {
            atomic_cas(v, version, version+1)
        };
        ok
    }

    #[inline(always)]
    pub fn bump_version(&self) {
        let v = &self.version as *const u64 as *mut u64;
        unsafe {
            atomic_add(v, 1);
            //volatile_add(v, 1);
        }
    }

    #[inline(always)]
    pub fn read_version(&self) -> u64 {
        let v = &self.version as *const u64;
        unsafe { ptr::read_volatile(v) }
    }

    #[inline(always)]
    pub fn read_key(&self, idx: usize) -> u64 {
        debug_assert!(idx < ENTRIES_PER_BUCKET);
        unsafe {
            let k = self.key.get_unchecked(idx) as *const u64;
            ptr::read_volatile(k)
        }
    }

    #[inline(always)]
    pub fn read_value(&self, idx: usize) -> u64 {
        debug_assert!(idx < ENTRIES_PER_BUCKET);
        unsafe {
            let v = self.value.get_unchecked(idx) as *const u64;
            ptr::read_volatile(v)
        }
    }

    #[inline(always)]
    pub fn set_key(&self, idx: usize, key: u64) {
        debug_assert!(idx < ENTRIES_PER_BUCKET);
        debug_assert!(key != INVALID_KEY);
        unsafe {
            let k = self.key.get_unchecked(idx)
                        as *const u64 as *mut u64;
            ptr::write_volatile(k, key);
        }
    }

    #[inline(always)]
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

    #[inline(always)]
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

    #[inline(always)]
    pub fn wait_version(&self) -> u64 {
        let mut v = self.read_version();
        loop {
            if is_even(v) { break; }
            v = self.read_version();
        }
        v
    }

    #[inline(always)]
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

    #[inline(always)]
    pub fn unlock(&self) {
        debug_assert!(self.version.is_odd());
        self.bump_version();
    }

    pub fn reset_version(&self) {
        unsafe {
            let version: *mut u64 =
                &self.version as *const u64 as *mut u64;
            ptr::write_volatile(version, 0u64);
        }
    }

    /// Locate an empty slot in the bucket. Return index.
    #[inline(always)]
    pub fn find_empty(&self) -> Option<usize> {
        for i in 0..ENTRIES_PER_BUCKET {
            if self.read_key(i) == INVALID_KEY {
                return Some(i);
            }
        }
        None
    }

    #[inline(always)]
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

pub struct HashTable {
    allow_resize: bool,
    buckets: Pointer<Bucket>,
    bucket_mmap: MemMap,
    /// Current number of buckets in the MemMap.
    nbuckets: usize,
    /// Bytes used by nbuckets (must be <= MemMap.len)
    len: usize,
}

impl HashTable {

    pub fn new(entries: usize, sock: usize) -> Self {
        let nbuckets = entries / ENTRIES_PER_BUCKET;
        let mut len = nbuckets * mem::size_of::<Bucket>();
        info!("new table on socket {} len {}", sock, len);
        let align: usize = 1usize<<21;
        // round up to next alignment
        len = (len + align - 1) & !(align-1);
        let mmap = MemMap::numa(TABLE_VLEN, NodeId(sock), align, false);
        let p = Pointer(mmap.addr() as *const Bucket);

        let sl: &mut [Bucket] = unsafe {
            slice::from_raw_parts_mut(p.0 as *mut Bucket,nbuckets)
        };
        for b in sl {
            b.version = 0;
            for i in 0..ENTRIES_PER_BUCKET {
                b.key[i] = INVALID_KEY;
                b.value[i] = 0;
            }
        }
        // TODO use threads to populate the arrays.
        HashTable {
            allow_resize: true,
            buckets: p,
            nbuckets: nbuckets,
            bucket_mmap: mmap,
            len: len,
        }
    }

    pub fn forbid_resize(&mut self) {
        self.allow_resize = false;
    }

    #[inline(always)]
    pub fn prefetchw(&self, hash: u64) {
        let bidx = (hash % (self.nbuckets as u64)) as usize;
        let buckets: &[Bucket] = self.as_slice();
        let bucket: &Bucket = &buckets[bidx];
        let addr: *const u8 = unsafe {
            //&bucket.version as *const _ as *const u8
            bucket.key.get_unchecked(2) as *const _ as *const u8
        };
        prefetchw(addr);
    }

    #[inline(always)]
    fn as_slice(&self) -> &[Bucket] {
        unsafe {
            slice::from_raw_parts(self.buckets.0, self.nbuckets)
        }
    }

    // TODO numa init method
    // mmap the bucket array and treat as a slice

    /// does work of both insert and update
    /// (true,None) -> insertion
    /// (true,Some(x)) -> update existing, x is old value
    /// (false,None) -> cannot insert, table full
    /// (false,Some(x)) -> never returned
    #[inline(always)]
    pub fn put(&self, key: u64, value: u64) -> (bool,Option<u64>) {
        let hash = Self::make_hash(key);

        let mut bidx: usize;
        let mut buckets: &[Bucket];
        let mut bucket: &Bucket;
        let mut v: u64;
        let mut opts: find_ops;

        'retry: loop {
            //let bidx = (hash & (self.nbuckets-1) as u64) as usize;
            bidx = (hash % (self.nbuckets as u64)) as usize;
            buckets = self.as_slice();
            bucket = &buckets[bidx];

            v = bucket.read_version();
            opts = bucket.find_key(key);
            let (e,inv) = opts;
            if unlikely!(e.is_none() && inv.is_none()) {
                if !self.allow_resize {
                    return (false, None);
                }
                self.resize();
                // whether we win or lose the race to resize,
                // always retry again as buckets will change
            } else {
                break;
            }
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

    #[inline(always)]
    pub fn get(&self, key: u64, value: &mut u64) -> bool {
        let hash = Self::make_hash(key);

        //let bidx = (hash & (self.nbuckets-1) as u64) as usize;
        let bidx = (hash % (self.nbuckets as u64)) as usize;
        let buckets: &[Bucket] = self.as_slice();
        let bucket: &Bucket = &buckets[bidx];

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

    #[inline(always)]
    pub fn del(&self, key: u64, old: &mut u64) -> bool {
        let hash = Self::make_hash(key);

        //let bidx = (hash & (self.nbuckets-1) as u64) as usize;
        let bidx = (hash % (self.nbuckets as u64)) as usize;
        let buckets: &[Bucket] = self.as_slice();
        let bucket: &Bucket = &buckets[bidx];

        let v = bucket.read_version();
        let mut opts = bucket.find_key(key);
        let (e,inv) = opts;
        if unlikely!(e.is_none()) {
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
    #[inline(always)]
    pub fn update_lock_ifeq(&self, key: u64, new: u64, old: u64)
        -> Option<LockedBucket> {

        let hash = Self::make_hash(key);

        //let bidx = (hash & (self.nbuckets-1) as u64) as usize;
        let bidx = (hash % (self.nbuckets as u64)) as usize;
        let buckets: &[Bucket] = self.as_slice();
        let bucket: &Bucket = &buckets[bidx];

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

    fn lock_all(&self) {
        let buckets: &[Bucket] = self.as_slice();
        for b in buckets {
            b.wait_lock();
        }
    }

    pub fn resize(&self) {
        // grow by this amount
        let factor: usize = 2;

        let nbuckets = self.nbuckets * factor;
        let len = self.len * factor;
        assert!(len <= self.bucket_mmap.len(),
            "Resizing table cannot grow beyond mmap area");

        info!("resizing table 0x{:x}", self.bucket_mmap.addr());

        // for bucket in self.as_slice() {
        //     println!("BEF {:?}", bucket);
        // }

        let begin = clock::now();

        let start = clock::now();
        unsafe {
            // this should fault pages on the socket set by mbind
            // TODO we assume zero is used to indicate INVALID_KEY and
            // that version zero is a valid starting version
            // (all true)
            self.bucket_mmap.clear_region(self.len, self.len);
        }
        let end = clock::now();
        debug!("clearing memory {} µs", clock::to_usec(end-start));

        let buckets: &[Bucket] = unsafe {
            slice::from_raw_parts(self.buckets.0, nbuckets)
        };

        let old_buckets = &buckets[0..self.nbuckets];
        let new_buckets = &buckets[self.nbuckets..nbuckets];

        // lock all buckets. prevents concurrent operations
        let start = clock::now();
        self.lock_all();
        let end = clock::now();
        debug!("locking buckets {} µs", clock::to_usec(end-start));

        let start = clock::now();
        for bidx in 0..self.nbuckets {
            let old: &Bucket = &buckets[bidx];
            // prefetch next bucket? TODO
            for kidx in 0..ENTRIES_PER_BUCKET {
                let key = old.read_key(kidx);
                if INVALID_KEY == key {
                    continue;
                }
                let hash = Self::make_hash(key);
                let bbidx = (hash % (nbuckets as u64)) as usize;
                if bidx != bbidx {
                    // prefetch new bucket? TODO
                    let new: &Bucket = &buckets[bbidx];
                    let mut value: u64 = 0;
                    old.del_key(kidx, &mut value);
                    let opt = new.find_empty();
                    // To really fix this, we'd restart resizing
                    // with even more buckets
                    assert!(opt.is_some(),
                        "While resizing table, new bucket is full!");
                    let kkidx = opt.unwrap();
                    new.set_key(kkidx, key);
                    new.set_value(kkidx, value);
                }
            }
        }
        let end = clock::now();
        debug!("resizing {} µs", clock::to_usec(end-start));

        unsafe {
            asm!("sfence" : : : "memory");

            let me: &mut Self =
                &mut *(self as *const _ as *mut _);
            me.nbuckets = nbuckets;
            me.len = len;
        }

        // at this point, the upper buckets are now accessible,
        // and threads may put/get into them. unlock the rest

        let start = clock::now();
        for bucket in old_buckets {
            bucket.unlock();
        }
        let end = clock::now();
        debug!("unlocking {} µs", clock::to_usec(end-start));

        let end = clock::now();
        debug!("resize total {} µs", clock::to_usec(end-begin));

        // for bucket in old_buckets {
        //     println!("OLD {:?}", bucket);
        // }
        // for bucket in new_buckets {
        //     println!("NEW {:?}", bucket);
        // }
    }

    #[inline(always)]
    pub fn make_hash(value: u64) -> u64 {
        fnv1a(value)

        //let mut sip = hash::SipHasher::default();
        //Self::make_sip(&mut sip, value)
    }

    //
    // Private methods
    //

    #[inline(always)]
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
    use super::Bucket;
    use super::ENTRIES_PER_BUCKET;
    use super::INVALID_KEY;

    use std::thread::{self,JoinHandle};
    use std::collections::HashSet;
    use std::ptr;
    use std::sync::atomic::{AtomicUsize,Ordering};

    use rand::{self,Rng};
    use crossbeam;
    use num::Integer;

    use common::*;
    use logger;

    #[test]
    fn init() {
        logger::enable();
        println!("");
        let tblsize = 1<<20;
        let mut ht = HashTable::new(tblsize, 0);
        ht.forbid_resize();
        let nb = tblsize / ENTRIES_PER_BUCKET;
        assert_eq!(ht.nbuckets, nb);
    }

    // generate random set of keys and attempt to check existence
    #[test]
    fn get_on_empty() {
        logger::enable();
        println!("");

        let mut ht = HashTable::new(1<<20, 0);
        ht.forbid_resize();
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
        let tblsz = 1usize << 20;

        let mut tbl_fills: usize = 0;
        let mut rng = rand::thread_rng();

        'outer: for _ in 0..ntables {
            let mut ht = HashTable::new(tblsz,0);
            ht.forbid_resize();
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

        let mut ht = HashTable::new(1<<20,0);
        ht.forbid_resize();
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

        let mut ht = HashTable::new(1<<20,0);
        ht.forbid_resize();
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

        let mut ht = HashTable::new(tblsz, 0);
        ht.forbid_resize();

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
        threads_read_n(1<<17, 8);
    }

    #[test]
    fn threads_read_med() {
        threads_read_n(1_usize<<23, 8);
    }

    // each threads gets some disjoint subset of keys.
    // it will insert them, check they exist and values aren't corrupt
    // it table is full, it will remove some
    // mix put/get/del, always verifying its own set of keys
    fn threads_rw_n(tblsz: usize, nthreads: usize) {
        logger::enable();
        println!("");

        let mut ht = HashTable::new(tblsz, 0);
        ht.forbid_resize();
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

    #[test]
    fn lock_all() {
        logger::enable();
        let entries: usize = 1usize<<20;
        let mut ht = HashTable::new(entries, 0);
        ht.lock_all();
        let buckets: &[Bucket] = ht.as_slice();
        for b in buckets {
            assert_eq!(b.read_version().is_odd(), true);
        }
    }

    #[test]
    fn resize_single_thread() {
        logger::enable();

        let entries: usize = 1usize<<20;
        let mut ht = HashTable::new(entries, 0);
        let len = ht.len;
        let nbuckets = ht.nbuckets;

        for k in 0..(1u64<<15) {
            ht.put(k+1, k+1); // 0 not a valid key..
        }

        ht.resize();
        assert_eq!(ht.len, len*2);
        assert_eq!(ht.nbuckets, nbuckets*2);

        let buckets: &[Bucket] = ht.as_slice();

        for b in buckets {
            assert_eq!(b.read_version().is_even(), true);
        }

        debug!("checking values");
        let mut value: u64 = 0;
        for k in 0..(1u64<<15) {
            assert_eq!(ht.get(k+1, &mut value), true);
            assert_eq!(value, k+1);
        }
    }

    // TODO many threads
}
