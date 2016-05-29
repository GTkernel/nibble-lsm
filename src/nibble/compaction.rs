//! Log compaction routines.
//!
//! Deleting an object (or overwriting it) isn't straight-forward. We
//! can lock the segment, mark it as usual, then update segment state.
//! We may still need to lock the object, so if there are two
//! operations, one delete and one append, they appear in the correct
//! order? or do we not care and require external system to ensure
//! these operations are synchronized correctly? We could also just
//! check if the segment is currently being cleaned, and if not,
//! directly invalidate the object without modifying any segment
//! state (this means that the compaction state for choosing the
//! best segment to compact would need to be updated lazily, e.g.
//! another list of segrefs that were mutated and should be
//! removed/readdded to the binary heap before asking for the best
//! candidate); if yes, we could wait for the cleaning to finish or do
//! something fancier (like have a pending op buffer that the cleaner
//! will check once done?).
//!
//! We can have threads mark a segment as 'stale' wrt its compaction
//! score(s) -- need not even be atomic, just set to 1. Keep a
//! reader/writer lock on it where "reader" is any Nibbl operation,
//! such as get_object or one which invalidates an object, and
//! "writer" is the compaction thread. Stale segments would need to be
//! kept in some secondary structure to let the compactor know which
//! segments need their score(s) updated.
//!
//!
//! Epoch-based tracking of requests
//!     new request starts, reads epoch, adds itself to list
//!     request ends, removes self from list
//!     cleaner picks a segment, cleans to another segment
//!         once done, sets epoch to segment, epoch++
//!     periodically, thread checks pending segs
//!         if seg.epoch < epoch, release it
//! Where to lock? RC locks table bucket.
//! If we allow in-place edits when defuncting objects, must protect
//! against race conditions where edit cannot happen to segment behind
//! progress of cleaner.

// DONE segment manager add newly closed seg to internal list
// DONE compaction checks this list, adds to candidates set
// DONE compaction gives cleaned segments to segment manager
// DONE segment manager adds cleaned segments to reclamation list
// TODO segment manager checks epoch to release segments - give blocks
// back to block allocator and destruct segment
// TODO add statistics to segments, segment manager, nibble, etc.
// TODO segment usage table to assist compactor
// TODO implement reclamation pathway (register new ops, use epoch)

// Cleaner must ask segment manager for any newly closed segments when
// it performs cleaning. For each, it computes the score, then inserts
// into its trees.

use common::*;
use segment::*;
use index::*;
use epoch;

use std::mem;
use std::sync::{Arc,Mutex,MutexGuard,RwLockWriteGuard,RwLock};
use std::sync::atomic::{AtomicBool};
use std::thread;
use std::time::{Duration,Instant};

use crossbeam::sync::SegQueue;
use itertools;
use quicksort;

//==----------------------------------------------------==//
//      Constants
//==----------------------------------------------------==//

pub const COMPACTION_RESERVE_SEGMENTS: usize = 0;

//==----------------------------------------------------==//
//      Compactor types, macros
//==----------------------------------------------------==//

pub type CompactorRef = Arc<Mutex< Compactor >>;
pub type LiveFn = Box<Fn(&EntryReference, &RwLockWriteGuard<IndexInner>) -> bool>;
type EpochSegment = (epoch::EpochRaw, SegmentRef);
type ReclaimQueue = SegQueue<EpochSegment>;
type ReclaimQueueRef = Arc<ReclaimQueue>;

//==----------------------------------------------------==//
//      Compactor
//==----------------------------------------------------==//

// TODO Keep segments ordered by their usefulness for compaction.

type Handle = thread::JoinHandle<()>;

pub struct Compactor {
    candidates: Arc<Mutex<Vec<SegmentRef>>>,
    manager: SegmentManagerRef,
    index: IndexRef,
    seginfo: epoch::SegmentInfoTableRef,
    /// The set of worker threads doing compaction.
    workers: SegQueue<(Arc<RwLock<Worker>>,Handle)>,
    /// Global reclamation queue
    reclaim: ReclaimQueueRef,
    /// Reserve segments
    reserve: Arc<SegQueue<SegmentRef>>,
}

// TODO metrics for when compaction should begin

impl Compactor {

    pub fn new(manager: &SegmentManagerRef,
               index: &IndexRef) -> Self {
        let seginfo = match manager.lock() {
            Err(_) => panic!("lock poison"),
            Ok(guard) => guard.seginfo(),
        };
        let reserve: SegQueue<SegmentRef>;
        reserve = SegQueue::new();
        {
            let mut guard = manager.lock().unwrap();
            for _ in 0..COMPACTION_RESERVE_SEGMENTS {
                let opt = guard.alloc();
                assert!(opt.is_some(),
                    "cannot allocate reserve segment");
                let segref = opt.unwrap();
                let slot = {
                    let guard = segref.read().unwrap();
                    guard.slot()
                };
                debug!("have reserve seg {}", slot);
                reserve.push(segref);
            }
        }
        Compactor {
            candidates: Arc::new(Mutex::new(Vec::new())),
            manager: manager.clone(),
            index: index.clone(),
            seginfo: seginfo,
            workers: SegQueue::new(),
            reclaim: Arc::new(SegQueue::new()),
            reserve: Arc::new(reserve),
        }
    }

    // When we clean, we allocate new segment from segment manager and
    // move objects from one to the other. When segment is cleaned, we
    // add to a 'to be free' list that will use epochs for
    // synchronization. The newly compacted segment will need to
    // release unused blocks back to the block allocator, and the
    // segment then added back to the log.

    pub fn spawn(&mut self, role: WorkerRole) {
        let state = Arc::new(RwLock::new(Worker::new(&role, self)));
        let give = state.clone();
        let name = format!("compaction::worker::{:?}", role);
        let handle = match thread::Builder::new()
                .name(name).spawn( move || worker(give) ) {
            Ok(handle) => handle,
            Err(e) => panic!("spawning thread: {:?}: {:?}",role,e),
        };
        self.workers.push( (state, handle) );
    }

}

impl Drop for Compactor {

    fn drop(&mut self) {
        //if self.tid.is_some() {
            //let order = atomic::Ordering::Relaxed;
            //self.tstop.store(true, order);
        //}
    }
}

//==----------------------------------------------------==//
//      Worker thread functions
//==----------------------------------------------------==//

macro_rules! park {
    ( $state:expr ) => {
        if $state.must_park() { thread::park(); }
    }
}

macro_rules! park_or_sleep {
    ( $state:expr, $duration:expr ) => {
        if $state.must_park() { thread::park(); }
        else { thread::sleep($duration); }
    }
}

fn __reclaim(state: &Arc<RwLock<Worker>>) {
    let mut s = state.write().unwrap();
    s.do_reclaim();
    let park = s.reclaim.as_ref().map_or(false, |r| r.len() == 0 );
    drop(s); // release lock
    if park {
        thread::sleep(Duration::new(0, 10000000u32))
    }
}

fn __compact(state: &Arc<RwLock<Worker>>) {
    let dur = Duration::from_secs(1);
    let mut s = state.write().unwrap();
    let new = s.check_new();
    debug!("{} new candidates", new);
    s.do_compact();
    let ncand = {
        match s.candidates.lock() {
            Err(_) => panic!("lock poison"),
            Ok(guard) => guard.len(),
        }
    };
    drop(s); // unlock WorkerState
    // TODO know better when to park
    if ncand == 0 {
        // FIXME crashes if we comment this out
        thread::sleep(dur);
    }
}

fn worker(state: Arc<RwLock<Worker>>) {
    info!("thread awake");
    let role = { state.read().unwrap().role };
    loop {
        match role {
            WorkerRole::Reclaim => __reclaim(&state),
            WorkerRole::Compact => __compact(&state),
        }
    }
}

//==----------------------------------------------------==//
//      Worker structure
//==----------------------------------------------------==//

#[derive(Debug,Copy,Clone)]
pub enum WorkerRole {
    Reclaim,
    Compact,
}

#[allow(dead_code)]
struct Worker {
    role: WorkerRole,
    // FIXME Vec is not efficient as a candidates list
    // popping first element is slow; shifts N-1 left
    candidates: Arc<Mutex<Vec<SegmentRef>>>,
    manager: SegmentManagerRef,
    index: IndexRef,
    seginfo: epoch::SegmentInfoTableRef,
    park: AtomicBool,
    /// This thread's private set of to-be-reclaimed segments
    /// Used only if thread role is Reclaim
    reclaim: Option<Vec<EpochSegment>>,
    /// Reference to the global list of to-be-reclaimed segments
    /// Compaction threads push to this, Reclaim threads move SegRefs
    /// from this to their private set to manipulate
    reclaim_glob: ReclaimQueueRef,
    /// Shared access to the reserve segments.
    reserve: Arc<SegQueue<SegmentRef>>,
}

impl Worker {

    pub fn new(role: &WorkerRole, compactor: &Compactor) -> Self {
        let reclaim = match *role {
            WorkerRole::Reclaim => Some(Vec::new()),
            _ => None,
        };
        Worker {
            role: *role,
            candidates: compactor.candidates.clone(),
            manager: compactor.manager.clone(),
            index: compactor.index.clone(),
            seginfo: compactor.seginfo.clone(),
            park: AtomicBool::new(false),
            reclaim: reclaim,
            reclaim_glob: compactor.reclaim.clone(),
            reserve: compactor.reserve.clone(),
        }
    }

    pub fn add_candidate(&mut self, seg: &SegmentRef) {
        match self.candidates.lock() {
            Ok(ref mut cand) => cand.push(seg.clone()),
            Err(_) => panic!("lock poison"),
        }
    }

    #[cfg(IGNORE)]
    fn __dump_candidates(&self, guard: &MutexGuard<Vec<SegmentRef>>) {
        debug!("sorted candidates:");
        let mut i: usize = 0;
        for entry in &**guard {
            let g = entry.read().unwrap();
            let slot = g.slot();
            let live = self.seginfo.get_live(slot);
            debug!("[{}] slot {} live {}", i, slot, live);
            i += 1;
        }
    }
    //#[cfg(IGNORE)]
    fn __dump_candidates(&self, guard: &MutexGuard<Vec<SegmentRef>>) { ; }

    /// Pick a set of candidate segments and remove them from the set.
    /// Return that set of candidates with their tally of live bytes.
    /// TODO use a lazy, batch iterator to get best-N segments, like
    /// https://github.com/benashford/rust-lazysort
    /// TODO create a candidates object instead?
    pub fn next_candidates(&mut self) -> Option<(Vec<SegmentRef>,usize)> {
        let mut segs: Vec<SegmentRef> = Vec::new();
        let mut guard = self.candidates.lock().unwrap();
        // this if-stmt shouldn't actually execute if compaction
        // thread only runs when we have lots of dirty segments
        if guard.len() == 0 {
            return None;
        }
        // optimization. if all candidates fit in a seg, return all
        let total = guard.iter()
            .fold(0usize, |acc,segref| {
                  let guard = segref.read().unwrap();
                  acc + self.seginfo.get_live(guard.slot())
            });
        if total < SEGMENT_SIZE {
            mem::swap(&mut segs, &mut *guard);
            return Some( (segs,total) );
        }
        // slow path (common?)
        let pred = | a: &SegmentRef, b: &SegmentRef | {
            let (sa,sb) = {
                let guarda = a.read().unwrap();
                let guardb = b.read().unwrap();
                (guarda.slot(),guardb.slot())
            };
            // descending sort so we can use pop instead of remove(0)
            self.seginfo.get_live(sb)
                .cmp(&self.seginfo.get_live(sa))
        };
        quicksort::quicksort_by(guard.as_mut_slice(), pred);
        self.__dump_candidates(&guard);
        // grab enough candidates to fill one segment
        segs = Vec::new();
        let mut tally: usize = 0;
        while tally < SEGMENT_SIZE {
            let seg = guard.pop().unwrap();
            let slot = {
                let guard = seg.read().unwrap();
                guard.slot()
            };
            let live = self.seginfo.get_live(slot);
            if tally + live > SEGMENT_SIZE {
                guard.push(seg);
                break;
            }
            tally += live;
            segs.push(seg);
        }
        assert!(segs.len() > 0, "next candidate list is empty");
        Some( (segs,tally) )
    }

    pub fn compact(&mut self, dirty: &Vec<SegmentRef>,
               new: &SegmentRef, isLive: &LiveFn) -> Status
    {
        let status: Status = Ok(1);
        let mut new = new.write().unwrap();

        self.seginfo.set_live(new.slot(), 0usize);
        for segref in dirty {
            let dirt = segref.read().unwrap();
            debug!("performing compaction slot {} -> slot {}",
                   dirt.slot(), new.slot());

            // get_object doesn't lock a Segment to retrieve the value
            // since the virtual address is directly used

            let mut guard = self.index.wlock();
            for entry in dirt.into_iter() {
                if isLive(&entry, &guard) {
                    let key: String;
                    let va = new.append_entry(&entry);
                    unsafe { key = entry.get_key(); }
                    Index::update_locked(&mut guard, &key, va);
                }
            }
            // carry over the epoch state
            self.seginfo.incr_live(new.slot(),
                self.seginfo.swap_live(dirt.slot(), 0usize));
        }


        status
    }

    /// Iterate through the segment to ensure the epoch table reports
    /// a live size that matches what the we corroborate with the
    /// index.
    #[cfg(IGNORE)]
    fn verify(&mut self, segref: &SegmentRef,
              slot: usize, isLive: &LiveFn) {
        let guard = self.index.lock().unwrap();
        let mut seg = segref.write().unwrap();
        let mut size: usize = 0;
        // read first then iterate and measure
        let live = self.seginfo.get_live(slot);
        for entry in &*seg {
            if isLive(&entry, &guard) {
                size += entry.len;
            }
        }
        info!("verify: slot {} measured {} epoch live {}",
               slot, size, live);
        assert!(size <= live);
        assert!(live <= SEGMENT_SIZE);
    }

    /// Don't use this except when debugging epoch table
    #[inline]
    #[allow(unused_variables)]
    #[cfg(IGNORE)]
    fn verify(&mut self, segref: &SegmentRef,
              slot: usize, isLive: &LiveFn) { ; }

    /// Called by WorkerRole::Compact
    /// Select a segment, grab a clean one, compact. Remove cleaned
    /// segment from candidates and notify segment manager it must be
    /// reclaimed. 
    /// TODO FIXME we should only compact if the ratio of live bytes
    /// to used bytes in a segment exceeds some threshold, otherwise
    /// compaction will run in endless loops, moving data :)
    pub fn do_compact(&mut self) {
        // liveness checking function
        // entry: EntryReference
        // iguard: MutexGuard<Index>
        let is_live: LiveFn = Box::new( move | entry, wguard | {
            let key = unsafe { entry.get_key() };
            match wguard.get(key.as_str()) {
                Some(loc) => (*loc == entry.get_loc()),
                None => false,
            }
        });

        let (segs,livebytes) = match self.next_candidates() {
            None => { debug!("no candidates"); return; },
            Some(x) => x,
        };
        debug!("candidates: # {} livebytes {}", segs.len(), livebytes);

        //self.verify(&segref, slot, &is_live);

        // livebytes may be stale by the time we start cleaning (it's ok)
        // if significant, we may TODO free more blocks after
        // compaction

        let mut used_reserve = false;
        if livebytes > 0  {
            // allocate new segment
            let mut newseg: SegmentRef;
            let nblks = ((livebytes-1)/BLOCK_SIZE)+1;
            debug!("allocating new segment #blks {}",nblks);
            let mut retries = 0;
            let start = Instant::now();
            loop {
                let opt = {
                    let mut guard = self.manager.lock().unwrap();
                    guard.alloc_size(nblks)
                };
                match opt {
                    Some(s) => {
                        newseg = s;
                        break;
                    },
                    None => {
                        panic!("deadlock: log full, {}",
                               "cannot allocate for compaction");
                        // used_reserve = true;
                        // let r = self.reserve.try_pop();
                        // // FIXME we can spin a little if this fails,
                        // // but only if there are >1 compaction threads
                        // // since one may release a reserve soon.
                        // assert!(r.is_some(),
                        //     "no reserve segments");
                        // neweg = r.unwrap();
                    },
                }
                // yielding and retrying only works if there are other
                // compaction threads running :D FIXME
                thread::yield_now();
                retries += 1;
                if start.elapsed().as_secs() > 4 {
                    epoch::__dump();
                    panic!("waiting too long {}",
                           "for new segment: deadlock?");
                }
            }
            if retries > 0 {
                let dur = start.elapsed();
                warn!("waited {} us for seg allocation",
                      (dur.as_secs() as u32) * 1000000u32 +
                      dur.subsec_nanos() / 1000u32);
            }

            let ret = self.compact(&segs, &newseg, &is_live);
            if ret.is_err() { panic!("compact failed"); }

            // monitor the new segment, too
            let newslot = newseg.read().unwrap().slot();
            debug!("adding slot {} to candidates", newslot);
            self.add_candidate(&newseg);
        }

        // XXX how do we reclaim segments for the reserve?

        //let epoch = EPOCH.fetch_add(1, atomic::Ordering::Relaxed);
        let ep = epoch::next();
        for segref in segs {
            let slot = {
                let guard = segref.read().unwrap();
                guard.slot()
            };
            debug!("adding slot {} to reclamation", slot);
            self.reclaim_glob.push( (ep, segref) );
        }
    }

    // TODO return true if there are still segments to reclaim to
    // throttle worker thread

    /// Called by WorkerRole::Reclaim
    pub fn do_reclaim(&mut self) {
        let mut reclaim = self.reclaim.as_mut().unwrap();

        // pull any new items off the global list
        while let Some(segref) = self.reclaim_glob.try_pop() {
            // TODO stop after x pops?
            reclaim.push(segref);
        }

        // find relevant segments locally
        let min = epoch::min();
        debug_assert!(min != Some(epoch::EPOCH_QUIESCE));
        let mut release: Vec<EpochSegment> = Vec::new();
        if min.is_none() {
            // we take all waiting segments
            mem::swap(reclaim, &mut release);
        } else {
            // otherwise, check epoch
            let m = min.unwrap();
            let pred = | t: &EpochSegment | {
                let epoch = t.0;
                epoch >= m // true: NOT reclaim this segment
            };
            //   epoch >= m      epoch < m
            // [true,true,true,false,false]
            //                 ^split
            let split = itertools::partition(
                &mut reclaim.into_iter(), pred);
            // [true,true,true] [false,false]
            //  reclaim          release
            release = reclaim.split_off(split);
        }

        if release.len() > 0 {
            debug!("releasing {} segments", release.len());
            match self.manager.lock() {
                Err(_) => panic!("lock poison"),
                Ok(mut manager) =>
                    for seg in release {
                        manager.free(seg.1);
                    },
            }
        }
    }

    /// Look in segment manager for newly closed segments. If any,
    /// move to our candidates list. Returns number of segments moved.
    pub fn check_new(&mut self) -> usize {
        match self.candidates.lock() {
            Err(_) => panic!("lock poison"),
            Ok(ref mut cand) => {
                match self.manager.lock() {
                    Err(_) => panic!("lock poison"),
                    Ok(mut manager) => {
                        manager.grab_closed(cand)
                    },
                }
            },
        }
    }

    #[cfg(IGNORE)]
    pub fn must_park(&self) -> bool {
        let order = atomic::Ordering::Relaxed;
        self.park.compare_and_swap(true, false, order)
    }
}

//==----------------------------------------------------==//
//      Unit tests
//==----------------------------------------------------==//

#[cfg(IGNORE)]
mod tests {
    use super::*;

    use std::mem;
    use std::ops;
    use std::slice::from_raw_parts;
    use std::sync::{Arc,Mutex};

    use memory::*;
    use segment::*;
    use thelog::*;
    use index::*;
    use common::*;

    use rand;
    use rand::Rng;

    use super::super::logger;

    #[test]
    fn add_segments() {
        logger::enable();
        let nseg = 8;
        let index = index_ref!();
        let segmgr = segmgr_ref!(SEGMENT_SIZE, SEGMENT_SIZE*nseg);
        let mut c = Compactor::new(&segmgr, &index);
        assert_eq!(c.candidates.lock().unwrap().len(), 0);
        if let Ok(mut manager) = segmgr.lock() {
            for _ in 0..nseg {
                c.add( manager.alloc().as_ref()
                       .expect("alloc segment")
                     );
            }
        }
        assert_eq!(c.candidates.lock().unwrap().len(), nseg);
    }

    /// Big beasty compaction test. TODO break down into smaller tests
    #[test]
    fn compact_two() {
        logger::enable();
        let mut rng = rand::thread_rng();

        let index = index_ref!();
        let segmgr = segmgr_ref!(SEGMENT_SIZE, SEGMENT_SIZE<<3);
        let mut c = Compactor::new(&segmgr, &index);

        let seg_obj_ref;
        match segmgr.lock() {
            Ok(mut mgr) => {
                seg_obj_ref = mgr.alloc().unwrap();
            },
            Err(_) => panic!("mgr lock poison"),
        }

        // TODO export rand str generation
        // TODO clean this up (is copy/paste from segment tests)

        // create the key value pairs
        let alpha: Vec<char> =
            "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
            .chars().collect();

        let key_sizes: Vec<u32> = vec!(30, 89, 372); // arbitrary
        let value_sizes: Vec<u32> = vec!(433, 884, 511); // arbitrary
        let total: u32 = key_sizes.iter().fold(0, ops::Add::add)
            + value_sizes.iter().fold(0, ops::Add::add);
        let nbatches = (SEGMENT_SIZE - BLOCK_SIZE) / (total as usize);

        // create key-value pairs
        let mut keys: Vec<String> = Vec::new();
        let mut values: Vec<String> = Vec::new();
        for tuple in (&key_sizes).into_iter().zip(&value_sizes) {
            let mut s = String::with_capacity(*tuple.0 as usize);
            for _ in 0..*tuple.0 {
                let r = rng.gen::<usize>() % alpha.len();
                s.push( alpha[ r ] );
            }
            keys.push(s);
            s = String::with_capacity(*tuple.1 as usize);
            for _ in 0..*tuple.1 {
                let r = rng.gen::<usize>() % alpha.len();
                s.push( alpha[ r ] );
            }
            values.push(s);
        }

        { // hold segment mutably for limited scope
            let mut seg_obj = seg_obj_ref.borrow_mut();

            // append the objects
            for _ in 0..nbatches {
                for tuple in (&keys).into_iter().zip(&values) {
                    let key = tuple.0;
                    let value = tuple.1;
                    let loc = Some(value.as_ptr());
                    let len = value.len() as u32;
                    let obj = ObjDesc::new(key, loc, len);
                    match seg_obj.append(&obj) {
                        Err(code) => panic!("append error:: {:?}", code),
                        _ => {},
                    }
                }
            }
        }

        // we remove all objects whose value is < 500 bytes
        // using fancy closures
        let filter: LiveFn =
            Box::new( | entry, _ | { entry.datalen < 500 });

        // allocate new segment to move objects into
        let new_capacity = ((value_sizes[0] + key_sizes[0]) as usize
                            + mem::size_of::<EntryHeader>())*nbatches
                            + mem::size_of::<SegmentHeader>();
        let nblks = (new_capacity / BLOCK_SIZE) + 1;
        let seg_clean_ref;
        match segmgr.lock() {
            Ok(mut mgr) => {
                seg_clean_ref = mgr.alloc_size(nblks).unwrap();
            },
            Err(_) => panic!("manager lock poison"),
        }
        
        // move all objects whose data length < 500
        // given the above, we keep only nbatches of the first entry
        match c.compact(&seg_obj_ref, &seg_clean_ref, &filter) {
            Ok(1) => {},
            _ => panic!("compact failed"),
        }

        // Buffer to receive items into
        let total = (key_sizes[0] + value_sizes[0]) << 1;
        let buf: *mut u8 = allocate::<u8>(total as usize);

        let mut counter = 0;
        for entry in seg_clean_ref.borrow().into_iter() {
            assert_eq!(entry.keylen, key_sizes[0]);
            assert_eq!(entry.datalen, value_sizes[0]);
            unsafe {
                entry.get_data(buf);
                let nchars = values[0].len();
                let slice = from_raw_parts(buf, nchars);
                let orig = values[0].as_bytes();
                assert_eq!(slice, orig);
                let key = entry.get_key();
                assert_eq!(keys[0], key);
            }
            counter += 1;
        }

        assert_eq!(counter, nbatches);

        {
            let s = seg_clean_ref.borrow();
            let t = ((value_sizes[0] + key_sizes[0]) as usize
                     + mem::size_of::<EntryHeader>()) * nbatches
                     + mem::size_of::<SegmentHeader>();
            assert_eq!(s.nobjects(), nbatches);
            assert_eq!(s.used(), t);
            assert_eq!(s.rem(), s.len() - t);
            assert_eq!(s.head().is_some(), true);
            assert_eq!(s.curblk().is_some(), true);
            assert_eq!(s.curblk().unwrap(), s.nblks()-1);
        }

        unsafe { deallocate::<u8>(buf, total as usize); }
    }

    #[test]
    fn try_compact() {
        logger::enable();
        let index = index_ref!();
        let nseg = 32; // multiple of 4 (for this test)
        let segmgr = segmgr_ref!(SEGMENT_SIZE,
                                     nseg*SEGMENT_SIZE);
        let mut c = Compactor::new(&segmgr, &index);
        let mut log = Log::new(segmgr.clone());

        let key = String::from("laskdjflskdjflskjdflskdf");
        let value = String::from("sldfkjslkfjsldkjfksjdlfjsdfjslkd");

        // fill half the number of segments
        {
            let obj = ObjDesc::new(key.as_str(),
                            Some(value.as_ptr()),
                            value.len() as u32);
            let size = obj.len_with_header();
            let bytes_needed = (nseg/2) * SEGMENT_SIZE;
            let mut many = 0;
            loop {
                match log.append(&obj) {
                    Err(code) => match code {
                        ErrorCode::OutOfMemory => 
                            panic!("ran out of memory?"),
                        _ => panic!("log append {:?}", code),
                    },
                    _ => {},
                }
                many += size;
                if many >= bytes_needed {
                    break;
                }
            }
        }

        // check segments move properly
        match segmgr.lock() {
            Err(_) => panic!("lock poison"),
            Ok(mang) => 
                assert_eq!(mang.n_closed(), nseg/2),
        };

        // move them
        let mut n = c.check_new();
        assert_eq!(n, nseg/2);

        match c.candidates.lock() {
            Ok(cand) => n = cand.len(),
            Err(_) => panic!("poison lock"),
        }
        assert_eq!(n, nseg/2);

        // manually engage compaction
        let ncompact = nseg/4;
        for _ in 0..ncompact {
            c.do_compact();
        }

        match segmgr.lock() {
            Err(_) => panic!("lock poison"),
            Ok(mang) =>  {
                assert_eq!(mang.n_closed(), 0);
            },
        };

        // TODO verify we moved segments to reclaim list

    } // do_compact

}
