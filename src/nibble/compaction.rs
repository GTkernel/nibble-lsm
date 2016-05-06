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
use thelog::*;
use segment::*;
use index::*;

use std::collections::{VecDeque,BinaryHeap,LinkedList};
use std::sync::{Arc,Mutex,MutexGuard};
use std::thread;

use crossbeam::sync::SegQueue;

//==----------------------------------------------------==//
//      Compactor types
//==----------------------------------------------------==//

pub type CompactorRef = Arc<Mutex< Compactor >>;

pub type LiveFn = Box<Fn(&EntryReference, &MutexGuard<Index>) -> bool>;

//==----------------------------------------------------==//
//      Compactor
//==----------------------------------------------------==//

// TODO Keep segments ordered by their usefulness for compaction.

pub struct Compactor {
    // TODO approximate sorting?
    // candidates: BinaryHeap<SegmentRef>,
    candidates: Mutex<VecDeque<SegmentRef>>,
    manager: SegmentManagerRef,
    index: IndexRef,
    epochs: EpochTableRef,
    /// Cleaned segments that are waiting to be released
    reclaim: Arc<SegQueue<SegmentRef>>,
}

// TODO need a way to return clean segments back to the segment
// manager

// TODO need a thread

// TODO metrics for when compaction should begin

impl Compactor {

    pub fn new(manager: &SegmentManagerRef,
               index: &IndexRef) -> Self {
        let epochs = match manager.lock() {
            Err(_) => panic!("lock poison"),
            Ok(guard) => guard.epochs(),
        };
        let mut c = Compactor {
            candidates: Mutex::new(VecDeque::new()),
            manager: manager.clone(),
            index: index.clone(),
            epochs: epochs,
            reclaim: Arc::new(SegQueue::new()),
        };
        c
    }

    /// Add newly closed segment to the candidate set.
    /// TODO in the future, this function will be replaced with a
    /// thread that periodically scraps closed segments from LogHead
    /// instances (else this is a bottleneck).
    pub fn add(&mut self, seg: &SegmentRef) {
        match self.candidates.lock() {
            Ok(ref mut cand) => cand.push_back(seg.clone()),
            Err(pe) => panic!("Compactor lock poisoned"),
        }
    }

    // When we clean, we allocate new segment from segment manager and
    // move objects from one to the other. When segment is cleaned, we
    // add to a 'to be free' list that will use epochs for
    // synchronization. The newly compacted segment will need to
    // release unused blocks back to the block allocator, and the
    // segment then added back to the log.

    pub fn compact(&mut self, dirty: &SegmentRef,
               new_: &SegmentRef, isLive: &LiveFn) -> Status
    {
        let mut status: Status = Ok(1);

        // FIXME don't lock entire index
        match self.index.lock() {
            Ok(mut guard) => {
                let mut new = new_.borrow_mut();
                for entry in dirty.borrow_mut().into_iter() {
                    if isLive(&entry, &guard) {
                        let key: String;
                        let va = new.append_entry(&entry);
                        unsafe { key = entry.get_key(); }
                        guard.update(&key, va);
                    }
                }
            },
            Err(poison) => panic!("index lock poisoned"),
        }
        status
    }

    /// Pick a next candidate segment and remove from set.
    pub fn next_candidate(&mut self) -> Option<SegmentRef> {
        let mut segref: Option<SegmentRef> = None;
        match self.candidates.lock() {
            Err(_) => panic!("lock poison"),
            Ok(mut cand) => {
                match cand.pop_front() {
                    None => {},
                    Some(seg) => segref = Some(seg.clone()),
                }
            },
        }
        segref
    }

    /// Select a segment, grab a clean one, compact. Remove cleaned
    /// segment from candidates and notify segment manager it must be
    /// reclaimed. 
    pub fn try_compact(&mut self) {

        // liveness checking function
        let is_live: LiveFn = Box::new( move | entry, guard | {
            let key = unsafe { entry.get_key() };
            match guard.get(key.as_str()) {
                Some(loc) => (loc == entry.get_loc()),
                None => false,
            } // match
        });

        // pick a segment
        let mut segref = match self.next_candidate() {
            Some(seg) => seg,
            None => return,
        };

        // determine live amount for new segment
        let slot = segref.borrow().slot();
        let newlen = self.epochs.get_live(slot) >> 1;
        debug!("slot {} newlen {}", slot, newlen);

        // newlen may be stale by the time we start cleaning (it's ok)
        // if significant, we may TODO free more blocks after
        // compaction

        if newlen > 0  {
            // allocate new segment
            let opt: Option<SegmentRef> = 
                match self.manager.lock() {
                    Err(_) => panic!("lock poison"),
                    Ok(mut manager) => {
                        let nblks = (newlen - 1) / BLOCK_SIZE + 1;
                        debug!("nblks {}", nblks);
                        manager.alloc_size(nblks)
                    },
                };

            // unwrap it
            let mut newseg: SegmentRef =
                match opt {
                    None => panic!("OOM"),
                    Some(seg) => seg,
                };

            // do the work
            match self.compact(&segref, &newseg, &is_live) {
                Ok(1) => {},
                _ => panic!("compact failed"),
            }

            // monitor the new segment, too
            match self.candidates.lock() {
                Err(_) => panic!("lock poison"),
                Ok(ref mut cand) => cand.push_back(newseg.clone()),
            }
        }

        self.reclaim.push(segref.clone());
    }

    /// Look in segment manager for newly closed segments. If any,
    /// move to our candidates list. Returns number of segments moved.
    pub fn check_new(&mut self) -> usize {
        // TODO if only one thread accesses this list, no need for
        // lock
        let mut n: usize = 0;
        match self.candidates.lock() {
            Err(_) => panic!("lock poison"),
            Ok(ref mut cand) => {
                match self.manager.lock() {
                    Err(_) => panic!("lock poison"),
                    Ok(mut manager) => {
                        n = manager.grab_closed(cand);
                    },
                }
            },
        } // candidates lock
        n
    }

    //
    // --- Private methods ---
    //

    // TODO background thread: check_new, try_compact, etc.


    //
    // --- For unit tests ---
    //
}

//==----------------------------------------------------==//
//      Unit tests
//==----------------------------------------------------==//

#[cfg(test)]
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
        let mut index = index_ref!();
        let mut segmgr = segmgr_ref!(0, SEGMENT_SIZE, SEGMENT_SIZE*nseg);
        let mut c = Compactor::new(&segmgr, &index);
        assert_eq!(c.candidates.lock().unwrap().len(), 0);
        let mut x: usize;
        if let Ok(mut manager) = segmgr.lock() {
            for x in 0..nseg {
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

        let mut index = index_ref!();
        let mut segmgr = segmgr_ref!(0, SEGMENT_SIZE, SEGMENT_SIZE<<3);
        let mut c = Compactor::new(&segmgr, &index);

        let mut seg_obj_ref;
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
            for i in 0..*tuple.0 {
                let r = rng.gen::<usize>() % alpha.len();
                s.push( alpha[ r ] );
            }
            keys.push(s);
            s = String::with_capacity(*tuple.1 as usize);
            for i in 0..*tuple.1 {
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
            Box::new( | entry, guard | { entry.datalen < 500 });

        // allocate new segment to move objects into
        let new_capacity = ((value_sizes[0] + key_sizes[0]) as usize
                            + mem::size_of::<EntryHeader>())*nbatches
                            + mem::size_of::<SegmentHeader>();
        let nblks = (new_capacity / BLOCK_SIZE) + 1;
        let mut seg_clean_ref;
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
        let mut buf: *mut u8 = allocate::<u8>(total as usize);

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
        let mut index = index_ref!();
        let nseg = 32; // multiple of 4 (for this test)
        let mut segmgr = segmgr_ref!(0, SEGMENT_SIZE,
                                     nseg*SEGMENT_SIZE);
        let mut c = Compactor::new(&segmgr, &index);
        let mut log = Log::new(segmgr.clone());

        let mut key = String::from("laskdjflskdjflskjdflskdf");
        let mut value = String::from("sldfkjslkfjsldkjfksjdlfjsdfjslkd");

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
        for i in 0..ncompact {
            c.try_compact();
        }

        match segmgr.lock() {
            Err(_) => panic!("lock poison"),
            Ok(mang) =>  {
                assert_eq!(mang.n_closed(), 0);
            },
        };

        // TODO verify we moved segments to reclaim list

    } // try_compact

}
