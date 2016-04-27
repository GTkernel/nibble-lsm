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

// TODO segment manager add newly closed seg to internal list
// TODO compaction checks this list, adds to candidates set
// TODO compaction gives cleaned segments to segment manager
// TODO segment manager adds cleaned segments to reclamation list
// TODO segment manager checks epoch to release segments - give blocks back to block allocator and
// destruct segment

// Cleaner must ask segment manager for any newly closed segments when
// it performs cleaning. For each, it computes the score, then inserts
// into its trees.

use common::*;
use thelog::*;
use segment::*;

use std::cell::RefCell;
use std::collections::{BinaryHeap,LinkedList};
use std::sync::{Arc,Mutex};

//==----------------------------------------------------==//
//      Compactor types
//==----------------------------------------------------==//

pub type CompactorRef = Arc<Mutex<RefCell<Compactor>>>;

//==----------------------------------------------------==//
//      Compactor
//==----------------------------------------------------==//

// TODO Keep segments ordered by their usefulness for compaction.

pub struct Compactor {
    // TODO candidates: BinaryHeap<SegmentRef>,
    candidates: Mutex<LinkedList<SegmentRef>>,
}

// TODO need a way to return clean segments back to the segment
// manager

// TODO need a thread

// TODO metrics for when compaction should begin

impl Compactor {

    pub fn new() -> Self {
        let mut c = Compactor {
            candidates: Mutex::new(LinkedList::new()),
        };
        //c.spawn();
        c
    }

    /// Add newly closed segment to the candidate set.
    /// TODO in the future, this function will be replaced with a
    /// thread that periodically scraps closed segments from LogHead
    /// instances (else this is a bottleneck).
    pub fn add(&mut self, seg: &SegmentRef) {
        match self.candidates.lock() {
            Ok(ref mut list) => list.push_back(seg.clone()),
            Err(pe) => panic!("Compactor lock poisoned"),
        }
    }

    // When we clean, we allocate new segment from segment manager and
    // move objects from one to the other. When segment is cleaned, we
    // add to a 'to be free' list that will use epochs for
    // synchronization. The newly compacted segment will need to
    // release unused blocks back to the block allocator, and the
    // segment then added back to the log.

    pub fn compact<liveFn>(dirty_: &SegmentRef,
               new_: &SegmentRef, isLive: liveFn) -> Status
        where liveFn : Fn(&EntryReference) -> bool
    {
        let mut status: Status = Ok(1);

        let mut new = new_.borrow_mut();
        let mut dirty = dirty_.borrow_mut();
        for entry in dirty.into_iter() {
            if isLive(&entry) {
                let va = new.append_entry(&entry);
            }
        }

        status
    }

    //
    // --- Private methods ---
    //

    fn spawn(&mut self) -> Status {
        unimplemented!();
    }
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
    use std::cell::{RefCell};
    use std::sync::{Arc,Mutex};

    use memory::*;
    use segment::*;
    use thelog::*;
    use index::*;

    use rand;
    use rand::Rng;

    #[test]
    fn constructor() {
        let c = Compactor::new();
        assert_eq!(c.candidates.lock().unwrap().len(), 0);
    }

    #[test]
    fn add_segments() {
        let mut c = Compactor::new();
        let mut segmgr = SegmentManager::new(0, 1<<20, 1<<23);
        let mut x: usize;
        for x in 0..8 {
            c.add( segmgr.alloc().as_ref().expect("alloc segment") );
        }
        assert_eq!(c.candidates.lock().unwrap().len(), 8);
    }

    /// Big beasty compaction test. TODO break down into smaller tests
    #[test]
    fn compact() {
        let mut rng = rand::thread_rng();

        let mut segmgr = SegmentManager::new(0, 1<<20, 1<<23);
        let mut seg_obj_ref = segmgr.alloc().unwrap();

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
        let filter = |e: &EntryReference| { e.datalen < 500 };

        // allocate new segment to move objects into
        let new_capacity = ((value_sizes[0] + key_sizes[0]) as usize
                            + mem::size_of::<EntryHeader>())*nbatches
                            + mem::size_of::<SegmentHeader>();
        let nblks = (new_capacity / BLOCK_SIZE) + 1;
        let mut seg_clean_ref = segmgr.alloc_size(nblks).unwrap();
        
        // move all objects whose data length < 500
        // given the above, we keep only nbatches of the first entry
        match Compactor::compact(&seg_obj_ref,&seg_clean_ref,filter) {
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
                entry.copy_out(buf);
                let nchars = values[0].len();
                let slice = from_raw_parts(buf, nchars);
                let orig = values[0].as_bytes();
                assert_eq!(slice, orig);
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

    // TODO add objects to some segments
}
