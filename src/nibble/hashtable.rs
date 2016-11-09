/*
 * Concurrent hash table.
 */

use std::ptr;
use std::slice;
use std::fmt;
use std::hash;
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering, AtomicUsize};
use std::intrinsics;

use num::Integer;

use memory::*;
use common::{Pointer, fnv1a, is_even, is_odd, atomic_add, atomic_cas};
use common::{prefetch,prefetchw};
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
const TABLE_VLEN: usize = 1usize << 34;

const VERSION_MASK: u64 = 0x1;
const ENTRIES_PER_BUCKET: usize = 15;

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
    pub fn wait_lock(&self) -> BucketGuard {
        let mut v;
        let start = clock::now();
        'retry: loop {
            v = self.read_version();
            if is_even(v) {
                if self.try_bump_version(v) {
                    return BucketGuard::new(self);
                } else {
                    if clock::to_seconds(clock::now() - start) > 1 {
                        warn!("waited too long on bucket {:?}", self);
                    }
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

/// RAII-like lock used for clients of the API who must do
/// non-trivial amounts of work during a key update in the table.
/// Compaction, for example, is one. Unlike a typical Rust guard type,
/// this one doesn't prevent aliasing.
pub struct BucketGuard {
    bucket: Pointer<Bucket>,
}

impl BucketGuard {

    /// Caller should lock bucket before creating
    fn new(bucket: &Bucket) -> Self {
        BucketGuard {
            bucket: Pointer(bucket as *const Bucket),
        }
    }

//    /// Same as hashtable::put() except the bucket is already locked
//    pub fn update(&self, key: u64, new: u64) -> (bool,Option<u64>) {
//        let mut opts: find_ops =
//            opts = bucket.find_key(key);
//        let (e,inv) = opts;
//
//        let opt =
//            if e.is_some() { e }
//            else if inv.is_some() { inv }
//            else { None };
//
//        // do update or insert
//        if opt.is_some() {
//            let i = opt.unwrap();
//            unsafe {
//                let mut bucket: &Bucket =
//                    &mut *self.bucket.0;
//                (true,Some(bucket.set_value(i, new)))
//            }
//        }
//        // XXX invoke resize instead of failing.. this is
//        // too complicated
//        // else no space in hash table
//        else {
//            (false,None)
//        }
//    }

}

impl Drop for BucketGuard {
    fn drop(&mut self) {
        let b: &Bucket = unsafe { &*self.bucket.0 };
        b.unlock();
    }
}

pub struct HashTable {
    buckets: Pointer<Bucket>,
    bucket_mmap: MemMap,

    allow_resize: bool,
    resizing: AtomicBool,
    version: u64,

    /// Current number of buckets in the MemMap.
    nbuckets: usize,
    /// Bytes used by nbuckets (must be <= MemMap.len)
    len: usize,
    /// Count of times it has resized.
    resized: AtomicUsize
}

impl HashTable {

    pub fn new(entries: usize, sock: usize) -> Self {
        let nbuckets =
            (entries / ENTRIES_PER_BUCKET).next_power_of_two();
        let len = nbuckets * mem::size_of::<Bucket>();
        let align: usize = 1usize<<21;
        let mmap = MemMap::numa(TABLE_VLEN,
                                NodeId(sock), align, false);
        let p = Pointer(mmap.addr() as *const Bucket);
        debug!("new, sock {} nbucket {} current len {}",
              sock, nbuckets, len);

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
            buckets: p,
            bucket_mmap: mmap,

            allow_resize: true,
            resizing: AtomicBool::new(false),
            version: 0u64,

            nbuckets: nbuckets,
            len: len,
            resized: AtomicUsize::new(0)
        }
    }

    pub fn default(sock: usize) -> Self {
        Self::new( 1usize << 20, sock )
    }

    /// Need this to be a volatile read, as a resize may change it.
    #[inline(always)]
    pub fn nbuckets(&self) -> usize {
        let n = &self.nbuckets as *const usize;
        unsafe {
            ptr::read_volatile(n)
        }
    }

    /// Need this to be a volatile read, as a resize may change it.
    /// If the version changes, clients must recompute the bucket
    /// for their key.
    #[inline(always)]
    fn version(&self) -> u64 {
        let n = &self.version as *const u64;
        unsafe {
            ptr::read_volatile(n)
        }
    }

    fn bump_version(&self) {
        let loc = &self.version as *const u64 as *mut u64;
        unsafe { atomic_add(loc, 1u64); }
    }

    fn wait_resizing(&self) {
        while self.resizing.load(Ordering::Relaxed) {
            ;
        }
    }

    pub fn forbid_resize(&mut self) {
        self.allow_resize = false;
    }

    #[inline(always)]
    pub fn prefetchw(&self, hash: u64) {
        let bidx = self.index(hash);
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
            slice::from_raw_parts(self.buckets.0, self.nbuckets())
        }
    }

    /// Compute the index of the bucket from the hash of the key.
    #[inline(always)]
    fn index(&self, hash: u64) -> usize {
        (hash & (self.nbuckets() as u64 - 1)) as usize
    }

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
        let mut bver: u64;
        let mut opts: find_ops;

        let mut tver = self.version();

        bidx = self.index(hash);
        buckets = self.as_slice();
        bucket = &buckets[bidx];

        'retry: loop {

            // if table version changes, recompute bucket index
            let v = self.version();
            if unlikely!(v != tver) {
                bidx = self.index(hash);
                buckets = self.as_slice();
                bucket = &buckets[bidx];
                tver = v;
            }

            bver = bucket.read_version();

            opts = bucket.find_key(key);
            let (e,inv) = opts;
            if unlikely!(e.is_none() && inv.is_none()) {
                if !self.allow_resize {
                    return (false, None);
                }
                if !self.resize() {
                    self.wait_resizing();
                }
                continue 'retry;
            }

            // this will block an in-progress resize on this bucket
            // (prevents table version from incrementing)
            let mut guard = bucket.wait_lock();

            // re-examine if table has changed.. if so, restart
            // (a table resize won't grab the bucket lock from us)
            if unlikely!(tver != self.version()) {
                continue 'retry;
            }

            // table has not changed, check if bucket has
            if bucket.read_version() != bver {
                opts = bucket.find_key(key);
            }

            let (e,inv) = opts;

            // if exists, overwrite
            if let Some(i) = e {
                let old = Some(bucket.set_value(i, value));
                return (true, old);
            }
            // else if there is an empty slot, use that
            else if let Some(i) = inv {
                bucket.set_key(i, key);
                bucket.set_value(i, value);
                return (true, None);
            }

            // hm..  again no space. we must resize
            if !self.allow_resize {
                return (false, None);
            }
            drop(guard);
            if !self.resize() {
                self.wait_resizing();
            }
            // now loop around and retry
        }
        assert!(false, "Unreachable path");
    }

    #[inline(always)]
    pub fn get(&self, key: u64, value: &mut u64) -> bool {
        let hash = Self::make_hash(key);

        let mut bidx: usize;
        let mut buckets: &[Bucket];
        let mut bucket: &Bucket;
        let mut bver: u64;
        let mut opts: find_ops;

        // TODO if we retry too many times, perhaps forcefully lock

        let mut tver = self.version();

        bidx = self.index(hash);
        buckets = self.as_slice();
        bucket = &buckets[bidx];

        'retry: loop {

            let v = self.version();
            if unlikely!(v != tver) {
                bidx = self.index(hash);
                buckets = self.as_slice();
                bucket = &buckets[bidx];
                tver = v;
            }

            let bver = bucket.wait_version();
            for i in 0..ENTRIES_PER_BUCKET {
                if bucket.read_key(i) == key {
                    *value = bucket.read_value(i);
                    let must_retry = (bucket.read_version() != bver)
                        || unlikely!(self.version() != tver);
                    if must_retry {
                        continue 'retry;
                    }
                    //prefetch(*value as *const u64 as *const u8);
                    return true;
                }
            }
            if bucket.read_version() != bver {
                trace!("bucket modified while searching, retrying");
                continue 'retry;
            }
            if unlikely!(self.version() != tver) {
                // key may have been moved to another bucket
                trace!("key not found; may have been moved, retrying");
                continue 'retry;
            }
            return false;
        } // retry loop
    }

    #[inline(always)]
    pub fn del(&self, key: u64, old: &mut u64) -> bool {
        let hash = Self::make_hash(key);

        let mut bidx: usize;
        let mut buckets: &[Bucket];
        let mut bucket: &Bucket;
        let mut bver: u64;
        let mut opts: find_ops;

        // before reading bucket, you must save table version
        let mut tver = self.version();

        bidx = self.index(hash);
        buckets = self.as_slice();
        bucket = &buckets[bidx];

        'retry: loop {

            let v = self.version();
            if unlikely!(v != tver) {
                bidx = self.index(hash);
                buckets = self.as_slice();
                bucket = &buckets[bidx];
                tver = v;
            }

            // This is code which may have a bug, in that a key cannot
            // be found for deletion (but does exist) when the table
            // undergoes resizing. We do probably want to first scan
            // the bucket before locking, in case the key doesn't
            // exist. But, would someone try to delete a key that
            // doesn't exist?
            //
            // bver = bucket.read_version();
            // let mut opts = bucket.find_key(key);
            // let (e,inv) = opts;
            // if e.is_none() {
            //     if unlikely!(self.version() != tver) ||
            //         unlikely!(bucket.read_version() != bver) {
            //             continue 'retry;
            //         }
            //     info!("key {} not found without locking bucket",
            //           key);
            //     return false;
            // }

            bucket.wait_lock();
            opts = bucket.find_key(key);

            if unlikely!(tver != self.version()) {
                continue 'retry;
            }

            // if bucket.read_version() != bver {
            //     opts = bucket.find_key(key);
            // }

            let (e,inv) = opts;
            if e.is_none() {
                info!("key {} not found after locking bucket!",
                      key);
                return false;
            }

            bucket.del_key(e.unwrap(), old);
            return true;
        }
        assert!(false, "Unreachable path");
    }

    /// Grab the lock on the bucket which would hold the key, and
    /// execute the given lambda.  Does not modify the bucket. We do
    /// not need to search for the key; only lock the bucket.
    /// Return Ok if we were able to perform the update; Err likely
    /// means we are doing an insert but there is no space left.
    /// If we are updating a value, we pass Some(old_value) to f, else
    /// None to indicate an insertion.
    /// FIXME can we just make this part of put() ? much of the code
    /// is the same
    #[inline(always)]
    pub fn update_map<F>(&self, key: u64, new: u64, f: F) -> bool
        where F: Fn(Option<u64>) {

        let hash = Self::make_hash(key);

        let mut bidx: usize;
        let mut buckets: &[Bucket];
        let mut bucket: &Bucket;
        let mut bver: u64;
        let mut opts: find_ops;

        let mut tver = self.version();

        bidx = self.index(hash);
        buckets = self.as_slice();
        bucket = &buckets[bidx];

        'retry: loop {

            // if table version changes, recompute bucket index
            let v = self.version();
            if unlikely!(v != tver) {
                bidx = self.index(hash);
                buckets = self.as_slice();
                bucket = &buckets[bidx];
                tver = v;
            }

            bver = bucket.read_version();

            opts = bucket.find_key(key);
            let (e,inv) = opts;
            if unlikely!(e.is_none() && inv.is_none()) {
                if !self.allow_resize {
                    return false;
                }
                if !self.resize() {
                    self.wait_resizing();
                }
                continue 'retry;
            }

            let mut guard = bucket.wait_lock();

            if unlikely!(tver != self.version()) {
                continue 'retry;
            }

            // table has not changed, check if bucket has
            if bucket.read_version() != bver {
                opts = bucket.find_key(key);
            }

            let (e,inv) = opts;

            // if exists, overwrite
            if let Some(i) = e {
                f(Some(bucket.set_value(i, new)));
                return true;
            }
            // else if there is an empty slot, use that
            else if let Some(i) = inv {
                bucket.set_key(i, key);
                bucket.set_value(i, new);
                f(None);
                return true;
            }

            // hm..  again no space. we must resize
            if !self.allow_resize {
                return false;
            }
            drop(guard);
            if !self.resize() {
                self.wait_resizing();
            }
            // now loop around and retry
        }
        assert!(false, "Unreachable path");
    }

    /// Grab the lock on the bucket holding the key only if the
    /// existing value matches one specified. Before returning,
    /// replace existing value with new.
    #[inline(always)]
    pub fn update_lock_ifeq(&self, key: u64, new: u64, old: u64)
        -> Option<BucketGuard> {

        let hash = Self::make_hash(key);

        let mut bidx: usize;
        let mut buckets: &[Bucket];
        let mut bucket: &Bucket;
        let mut bver: u64;
        let mut opts: find_ops;

        let mut tver = self.version();

        bidx = self.index(hash);
        buckets = self.as_slice();
        bucket = &buckets[bidx];

        'retry: loop {

            let v = self.version();
            if unlikely!(v != tver) {
                bidx = self.index(hash);
                buckets = self.as_slice();
                bucket = &buckets[bidx];
                tver = v;
            }

            // let v = bucket.read_version();
            // let mut opts = bucket.find_key(key);
            // let (e,inv) = opts;
            // if e.is_none() {
            //     return None;
            // }

            let mut guard = bucket.wait_lock();

            if unlikely!(tver != self.version()) {
                continue 'retry;
            }

            opts = bucket.find_key(key);

            // if bucket.read_version() != v {
            //     opts = bucket.find_key(key);
            // }

            let (e,inv) = opts;
            if e.is_none() {
                return None;
            }

            let i = e.unwrap();
            if old == bucket.read_value(i) {
                bucket.set_value(i, new);
                return Some(guard);
            } else {
                return None;
            }
            assert!(false, "Unreachable path");
        }
        assert!(false, "Unreachable path");
    }

    fn lock_all(&self) -> Vec<BucketGuard> {
        let mut v: Vec<BucketGuard> =
            Vec::with_capacity(self.nbuckets);
        let buckets: &[Bucket] = self.as_slice();
        for b in buckets {
            v.push(b.wait_lock());
        }
        v
    }

    /// Return true if we acquired the lock.
    /// TODO make RAII
    fn lock_for_resize(&self) -> bool {
       !self.resizing.compare_and_swap(false, true,
            Ordering::SeqCst)
    }

    fn unlock_for_resize(&self) -> bool {
        self.resizing.compare_and_swap(true, false,
            Ordering::SeqCst)
    }

    /// Return false if we raced to lock for resizing but failed.
    pub fn resize(&self) -> bool {
        // Threads which lose the race to resize will simply keep
        // trying to insert to the bucket, again trying to lock.
        // Eventually the bucket will be locked by us. After we resize
        // and unlock the bucket, the failed threads will resume.
        if !self.lock_for_resize() {
            return false;
        }

        let prior = self.resized.fetch_add(1, Ordering::Relaxed);
        if prior >= 4 {
            warn!("Table {:p} resized for {}th time",
                  self as *const Self as *const u8, prior + 1);
        }

        // grow by this amount
        let factor: usize = 2;

        let nbuckets = self.nbuckets * factor;
        let len = self.len * factor;
        assert!(len <= self.bucket_mmap.len(),
            "Resizing table cannot grow beyond mmap area");
        debug_assert!(nbuckets.is_power_of_two());

        debug!("table 0x{:x} resizing, factor {} nb {} len {}",
              self.bucket_mmap.addr(), factor,
              nbuckets, len);

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
        let all = self.lock_all();
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

        // update version before ANY thread is able to proceed
        self.bump_version();

        // ensure bucket data and version are visible
        // before upper buckets become visible
        unsafe {
            asm!("sfence" : : : "memory");
        }

        unsafe {
            let me: &mut Self = &mut *(self as *const _ as *mut _);
            // once we mutate these, the upper buckets are open
            me.nbuckets = nbuckets;
            me.len = len;
            // ensure these are visible before unlocking lower buckets
            asm!("sfence" : : : "memory");
        }

        // at this point, the upper buckets are now accessible,
        // and threads may put/get into them. unlock the rest

        let start = clock::now();
        drop(all);
        //for bucket in old_buckets { bucket.unlock(); }
        assert_eq!(self.unlock_for_resize(), true);
        let end = clock::now();
        debug!("unlocking {} µs", clock::to_usec(end-start));

        let end = clock::now();
        debug!("table 0x{:x} resized {} ms",
              self.bucket_mmap.addr(),
              clock::to_msec(end-begin));

        // for bucket in old_buckets {
        //     println!("OLD {:?}", bucket);
        // }
        // for bucket in new_buckets {
        //     println!("NEW {:?}", bucket);
        // }

        true
    }

    #[inline(always)]
    pub fn make_hash(value: u64) -> u64 {
        //fnv1a(value)

        let mut sip = hash::SipHasher::new();
        Self::make_sip(&mut sip, value)
    }

    //
    // Private methods
    //

    #[inline(always)]
    fn make_sip<T: hash::Hasher>(her: &mut T, value: u64) -> u64 {
        her.write_u64(value);
        her.finish()
    }

    fn stats(&self) {
        let buckets: &[Bucket] = self.as_slice();
        for t in buckets.iter().zip(0..) {
            let mut n = 0;
            for i in 0..ENTRIES_PER_BUCKET {
                if t.0 .key[i] != INVALID_KEY {
                    n += 1;
                }
            }
            println!("{} {}", t.1, n);
        }
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
    use std::sync::{Arc,Barrier};
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

        let entries: usize = 1usize<<12;
        let mut ht = HashTable::new(entries, 0);
        let len = ht.len;
        let nbuckets = ht.nbuckets;

        for k in 0..(1u64<<12) {
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
            assert!(ht.get(k+1, &mut value),
                "key {} not found", k+1);
            assert!(value == k+1,
                    "value for key {} is bad: {}",
                    k+1, value);
        }
    }

    #[test]
    fn resize_many_threads() {
        logger::enable();

        let nthreads = 12;
        info!("creating hash table");
        let ht = HashTable::new(1<<20, 0);
        let tids = AtomicUsize::new(1);

        let mut guards = vec![];
        crossbeam::scope(|scope| {
            for _ in 0..nthreads {

                let guard = scope.spawn(|| {

                    // add keys in phases
                    let tid = tids.fetch_add(1,
                        Ordering::Relaxed) as u64 + 1u64;
                    info!("tid {} up", tid);

                    // use large enough offset to avoid overlap
                    let start_key = (1u64<<40) * tid;
                    let first_nkeys = start_key + (1u64<<16) * tid;
                    let max_nkeys = nthreads * (1u64<<18);
                    let my_max_key = start_key + max_nkeys;
                    let mut value: u64 = 0;
                    info!("tid {} start {} firstn {} maxn {} max {}",
                        tid, start_key, first_nkeys, max_nkeys,
                        my_max_key);

                    // each thread puts a different amount, so some
                    // will finish early and begin reading the table
                    // while others are still writing.
                    for key in start_key..first_nkeys {
                        assert_eq!(ht.put(key, key), (true, None));
                        assert!(ht.get(key, &mut value),
                            "tid {} key {} not found", tid, key);
                    }
                    info!("tid {} done inserting first", tid);
                    for key in start_key..first_nkeys {
                        assert_eq!(ht.get(key, &mut value), true);
                        assert_eq!(key, value);
                    }
                    info!("tid {} done check first", tid);

                    // Overlap deletes with resizing by allowing tid 0
                    // to continue inserting while all others delete
                    // and reinsert the keys they just inserted.
                    if tid > 0 {
                        for _ in 0..4 {
                            info!("tid {} deleting first set", tid);
                            for key in start_key..first_nkeys {
                                assert_eq!(ht.del(key,&mut value), true);
                            }
                            info!("tid {} reinserting first set", tid);
                            for key in start_key..first_nkeys {
                                assert_eq!(ht.put(key,key), (true,None));
                            }
                        }
                    }
                    info!("------------------------------------");

                    // insert the remainder
                    for key in first_nkeys..my_max_key {
                        assert_eq!(ht.put(key, key), (true, None));
                        assert_eq!(ht.get(key, &mut value), true);
                    }
                    info!("tid {} done inserting all", tid);
                    // check all keys
                    for key in start_key..my_max_key {
                        assert_eq!(ht.get(key, &mut value), true);
                        assert_eq!(key, value);
                    }
                    info!("tid {} done check all", tid);
                });
                guards.push(guard);
            }
        });

        for guard in guards {
            guard.join();
        }
    }

    #[test]
    fn fill_then_stats() {
        logger::enable();

        let entries: usize = 1usize<<12;
        let mut ht = HashTable::new(entries, 0);
        ht.forbid_resize();
        let len = ht.len;
        let nbuckets = ht.nbuckets;

        let mut keys: Vec<u64> =
            Vec::with_capacity(entries);

        //for _ in 0..entries {
        //    let mut v: u64 = unsafe { rdrandq() as u64 };
        //    if v == 0 {
        //        v += 1;
        //    }
        //    keys.push(v);
        //}

        for i in 0..entries {
            let v: u64 = i as u64 + 1u64;
            keys.push(v);
        }

        for k in keys {
            match ht.put(k, k) {
                (true,None) => continue, // inserted
                (true,Some(_)) => assert!(false,"updated??"),
                (false,None) => break, // table full
                (false,Some(_)) => assert!(false), // not returned
            }
        }

        ht.stats();
    }

}
