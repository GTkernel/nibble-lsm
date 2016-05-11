//==----------------------------------------------------==//
//      Cleaning and reclamation support
//==----------------------------------------------------==//

use segment::*;
use common::*;

use std::sync::atomic;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::mem::transmute;
use std::ptr;
use std::mem;

use crossbeam::sync::SegQueue;

//==----------------------------------------------------==//
//      Segment usage table
//==----------------------------------------------------==//

/// Structure to track information about Segments.
/// Compactor uses 'live' to allocate new segments. Manager uses the
/// epoch to release cleaned segments.
struct SegmentInfo {
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

//==----------------------------------------------------==//
//      Support to read the "epoch"
//==----------------------------------------------------==//

// Different ways to implement an epoch.
// 1. We can select a single cache line to hold the 64-bit value, and
//    atomically increment this, but this doesn't scale. RAMCloud does
//    this.
// 2. Use some scalable counter, like a combining tree.
// 3. The TSC becomes the global epoch among all cores.
//    See [ParSec Eurosys'16]
//    TODO need we sync all cores' TSC?

pub type EpochRaw = u64;

#[inline(always)]
#[allow(unused_mut)]
unsafe fn rdtsc() -> u64 {
    let mut low: u32;
    let mut high: u32;
    asm!("rdtsc" : "={eax}" (low), "={edx}" (high));
    ((high as u64) << 32) | (low as u64)
}

/// Read the current value of the epoch.
#[inline(always)]
#[cfg(feature="epoch-tsc")]
pub fn read() -> EpochRaw {
    unsafe { rdtsc() }
}

//==----------------------------------------------------==//
//      Per-thread epoch tracker
//==----------------------------------------------------==//

const EPOCHTBL_MAX_THREADS: u16 = 1024;

/// Global epoch tracker for each thread.
lazy_static! {
    static ref EPOCH_TABLE: EpochTable = { EpochTable::new() };
    // can add others here
}

/// Thread-local epoch state
thread_local!(static EPOCH_SLOT: *mut EpochSlot = register());

/// Register a new thread in the epoch table.
fn register() -> *mut EpochSlot {
    let p = EPOCH_TABLE.register();
    debug!("new thread gets slot @ {:?}", p);
    p
}

/// A slot in the global table a unique thread is assigned.
#[repr(packed)]
struct EpochSlot {
    epoch: u64,
    slot: u16,
    _padding: [u8; (CACHE_LINE-10)]
}

/// Statically ensure EpochSlot is exactly one cache line.
/// thanks to @mbrubeck
#[allow(dead_code)]
fn __assert_ThreadEpoch_size() {
    type T = EpochSlot;
    type O = [u8; CACHE_LINE];
    unsafe { mem::transmute::<T,O>(unreachable!()); }
}

impl EpochSlot {

    pub fn new(slot: u16) -> Self {
        let t = EpochSlot {
            epoch: 0u64,
            slot: slot,
            _padding: [0xdB; (CACHE_LINE-10)],
        };
        t
    }
}

pub struct EpochTable {
    table: Vec<EpochSlot>,
    // FIXME release a slot when thread dies
    freeslots: SegQueue<u16>
}

impl EpochTable {

    pub fn new() -> Self {
        let freeslots: SegQueue<u16> = SegQueue::new();
        let mut table: Vec<EpochSlot> =
            Vec::with_capacity(EPOCHTBL_MAX_THREADS as usize);
        for slot in 0..EPOCHTBL_MAX_THREADS {
            freeslots.push(slot);
            let e = EpochSlot::new(slot);
            table.push(e);
        }
        debug!("epoch table base addr {:?}", table.as_ptr());
        EpochTable {
            table: table,
            freeslots: freeslots,
        }
    }

    fn register(&self) -> *mut EpochSlot {
        let slot = self.freeslots.try_pop().unwrap() as usize;
        let p: *mut EpochSlot = unsafe {
            mem::transmute(&self.table[slot])
        };
        let sl = &self.table[slot];
        debug!("new slot: epoch 0x{:x} slot {}", sl.epoch, sl.slot);
        p
    }
}

