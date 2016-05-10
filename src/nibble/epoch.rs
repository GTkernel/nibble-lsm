use std::sync::atomic;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use segment::*;

//==----------------------------------------------------==//
//      Cleaning and reclamation support
//==----------------------------------------------------==//

// TODO static AtomicUsize as global epoch

/// Structure to track information about Segments.
/// Compactor uses 'live' to allocate new segments. Manager uses the
/// epoch to release cleaned segments.
pub struct SegmentUsage {
    /// segment timestamp
    epoch: AtomicUsize,
    /// live bytes in segment
    live:  AtomicUsize,
}

impl SegmentUsage {

    pub fn new() -> Self {
        SegmentUsage {
            epoch: AtomicUsize::new(0),
            live: AtomicUsize::new(0),
        }
    }
}

pub type EpochTableRef = Arc<EpochTable>;

pub struct EpochTable {
    /// shares index with SegmentManager::segments
    table: Vec<SegmentUsage>,
    ordering: atomic::Ordering,
}

/// Maintain information about live state of each segment. The value
/// here reflects the "live bytes" and matches what you would find
/// iterating the segment and corroborating against the index.  The
/// live bytes here may be reduced as you iterate, so be sure to first
/// read a value here before iterating; then the live bytes will only
/// be greater than or equal to the live bytes found from iterating.
impl EpochTable {

    pub fn new(slots: usize) -> Self {
        let mut v: Vec<SegmentUsage> = Vec::with_capacity(slots);
        for _ in 0..slots {
            v.push(SegmentUsage::new());
        }
        EpochTable {
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

