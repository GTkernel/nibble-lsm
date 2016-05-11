//==----------------------------------------------------==//
//      Cleaning and reclamation support
//==----------------------------------------------------==//

use segment::*;

use std::sync::atomic;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::mem::transmute;
use std::ptr;

use crossbeam::sync::SegQueue;

//==----------------------------------------------------==//
//      Segment usage table
//==----------------------------------------------------==//

/// Structure to track information about Segments.
/// Compactor uses 'live' to allocate new segments. Manager uses the
/// epoch to release cleaned segments.
pub struct SegmentInfo {
    /// segment timestamp
    epoch: AtomicUsize,
    /// live bytes in segment
    live:  AtomicUsize,
}

impl SegmentInfo {

    pub fn new() -> Self {
        SegmentInfo {
            epoch: AtomicUsize::new(0),
            live: AtomicUsize::new(0),
        }
    }
}

pub type SegmentInfoTableRef = Arc<SegmentInfoTable>;

pub struct SegmentInfoTable {
    /// shares index with SegmentManager::segments
    table: Vec<SegmentInfo>,
    ordering: atomic::Ordering,
}

/// Maintain information about live state of each segment. The value
/// here reflects the "live bytes" and matches what you would find
/// iterating the segment and corroborating against the index.  The
/// live bytes here may be reduced as you iterate, so be sure to first
/// read a value here before iterating; then the live bytes will only
/// be greater than or equal to the live bytes found from iterating.
impl SegmentInfoTable {

    pub fn new(slots: usize) -> Self {
        let mut v: Vec<SegmentInfo> = Vec::with_capacity(slots);
        for _ in 0..slots {
            v.push(SegmentInfo::new());
        }
        SegmentInfoTable {
            table: v,
            ordering: atomic::Ordering::Relaxed,
        }
    }

    pub fn get_live(&self, index: usize) -> usize {
        self.table[index].live.load(self.ordering)
    }

    pub fn get_epoch(&self, index: usize) -> usize {
        self.table[index].epoch.load(self.ordering)
    }

    pub fn set_live(&self, index: usize, value: usize) {
        self.table[index].live.store(value, self.ordering)
    }

    pub fn set_epoch(&self, index: usize, value: usize) {
        self.table[index].epoch.store(value, self.ordering)
    }

    pub fn incr_live(&self, index: usize, amt: usize) -> usize {
        let v = self.table[index].live.fetch_add(amt, self.ordering);
        debug_assert!(v <= SEGMENT_SIZE);
        v
    }

    pub fn incr_epoch(&self, index: usize, amt: usize) -> usize {
        self.table[index].epoch.fetch_add(amt, self.ordering)
    }

    pub fn decr_live(&self, index: usize, amt: usize) -> usize {
        debug_assert!(self.get_live(index) >= amt);
        self.table[index].live.fetch_sub(amt, self.ordering)
    }

    pub fn decr_epoch(&self, index: usize, amt: usize) -> usize {
        self.table[index].epoch.fetch_sub(amt, self.ordering)
    }

    pub fn swap_live(&self, index: usize, amt: usize) -> usize {
        self.table[index].live.swap(amt, self.ordering)
    }

    pub fn swap_epoch(&self, index: usize, amt: usize) -> usize {
        self.table[index].epoch.swap(amt, self.ordering)
    }

    //
    // --- Internal methods used for testing only ---
    //

    #[cfg(test)]
    pub fn len(&self) -> usize { self.table.len() }
}

// TODO set of pending ops
// use VecDeque

