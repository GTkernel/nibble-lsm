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


// Cleaner must ask segment manager for any newly closed segments when
// it performs cleaning. For each, it computes the score, then inserts
// into its trees.

use common::*;
use thelog::*;
use segment::*;

use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::collections::LinkedList;
use std::sync::Arc;
use std::sync::Mutex;

//==----------------------------------------------------==//
//      Compactor
//==----------------------------------------------------==//

pub type CompactorRef = Arc<Mutex<RefCell<Compactor>>>;

/// Make new CompactorRef
// #[macro_export]
// macro_rules! compref {
//     ( ) => {
//         Arc::new( Mutex::new( RefCell::new(
//                 Compactor::new()
//                 )))
//     }
// }

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

    //
    // --- Private methods ---
    //

    pub fn compact<liveFn>(seg: &SegmentRef,
               fresh: &SegmentRef, isLive: liveFn) -> Status
        where liveFn : Fn(&EntryHeader) -> bool
    {
        let mut status: Status = Ok(1);

        // for each item, check liveness, 

        // calculate size of all live objects
        // dynamically allocate sufficient blocks in new seg
        // iterate over all items and append them to new seg
        // close new seg
        // caller will release new blocks
        status
    }

    fn spawn(&mut self) -> Status {
        unimplemented!();
    }
}

// XXX need some function that can abstract how to determine if an
// entry in the log is live or not. Avoid having compaction code be
// aware of the log index itself.

//==----------------------------------------------------==//
//      Unit tests
//==----------------------------------------------------==//

#[cfg(test)]
mod tests {
    use super::*;

    use segment::*;
    use thelog::*;

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

    #[test]
    fn compact() {
        let mut segmgr = SegmentManager::new(0, 1<<20, 1<<23);
        let mut s1 = segmgr.alloc().unwrap().clone();
        let mut s2 = segmgr.allocEmpty().unwrap().clone();

        let key: &'static str = "onlyone";
        let val: &'static str = "valuevaluevalue";
        let obj = ObjDesc::new(key, Some(val.as_ptr()), val.len() as u32);

        // populate first segment

        // move all objects
        match Compactor::compact(&s1, &s2, |e| true) {
            Ok(1) => {},
            _ => panic!("compact failed"),
        }

        // move every other object TODO
        // etc.
    }

    // TODO add objects to some segments
}
