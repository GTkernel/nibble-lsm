//==----------------------------------------------------==//
//      Cleaning and reclamation support
//==----------------------------------------------------==//

use segment::*;
use common::*;

use std::cell::UnsafeCell;
use std::sync::atomic;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::mem::transmute;
use std::mem;
use std::u64;

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

/// Special-case value of EpochRaw meaning the value is non-existent.
pub const EPOCH_QUIESCE: u64 = 0;

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
fn read() -> EpochRaw {
    unsafe { rdtsc() }
}

/// This represents option 1. above
#[cfg(IGNORE)]
static EPOCH: AtomicUsize = AtomicUsize::new(0);

//==----------------------------------------------------==//
//      Per-thread epoch tracker
//==----------------------------------------------------==//

const EPOCHTBL_MAX_THREADS: u16 = 128;

/// Global epoch tracker for each thread.
lazy_static! {
    static ref EPOCH_TABLE: EpochTable = { EpochTable::new() };
    // can add others here
}

type EpochSlotRef = UnsafeCell<*mut EpochSlot>;

/// Thread-local epoch state. Just a pointer to a slot's entry.
/// UnsafeCell gives us fast mutable access when we update (pin).
thread_local!(
    static EPOCH_SLOT: EpochSlotRef = UnsafeCell::new(register())
);

/// Register a new thread in the epoch table.
fn register() -> *mut EpochSlot {
    let p = EPOCH_TABLE.register();
    debug!("new thread gets slot @ {:?}", p);
    p
}

#[inline(always)]
pub fn next() -> EpochRaw {
    read()
}

/// Signal that a thread is entering a quiescent phase, either by
/// terminating, parking, or sleeping.
#[inline(always)]
pub fn quiesce() {
    EPOCH_SLOT.with( |slotptr| {
        let mut slot = unsafe { &mut**(slotptr.get()) };
        slot.epoch = EPOCH_QUIESCE;
    });
}

/// Store the current epoch to the thread slot.
pub fn pin() {
    EPOCH_SLOT.with( |slotptr| {
        let mut slot = unsafe { &mut**(slotptr.get()) };
        slot.epoch = read();
    });
}

pub fn slot_addr() -> usize {
    EPOCH_SLOT.with( |slotptr| {
        unsafe {
            transmute::<*mut EpochSlot,usize>(*slotptr.get())
        }
    })
}

pub fn current() -> Option<EpochRaw> {
    EPOCH_SLOT.with( |slotptr| {
        let slot = unsafe { &mut**(slotptr.get()) };
        match slot.epoch {
            EPOCH_QUIESCE => None,
            e => Some(e),
        }
    })
}

pub fn min() -> Option<EpochRaw> {
    let mut m: EpochRaw = u64::MAX;
    for slot in &EPOCH_TABLE.table {
        // avoid races by only reading once into stack
        let e = slot.epoch;
        if e == EPOCH_QUIESCE {
            continue;
        }
        debug_assert!(e != EPOCH_QUIESCE); // check race
        if e < m {
            m = e;
        }
    }
    assert!(m != EPOCH_QUIESCE); // check race
    match m {
        u64::MAX => None,
        _ => Some(m),
    }
}

pub fn __dump() {
    let mut c = 1;
    let mut out = String::new();
    for slot in &EPOCH_TABLE.table {
        out.push_str(
            format!("[{:<4}] {:<16x}",
                    slot.slot, slot.epoch).as_str()
            );
        if (c % 9) == 0 {
            out.push_str("\n");
        }
        c += 1;
    }
    print!("{}", out);
}

/// A slot in the global table a unique thread is assigned.  Holds an
/// epoch, used to coordinate memory reclamation with Segment
/// compaction.
#[repr(packed)]
pub struct EpochSlot {
    epoch: EpochRaw,
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
            epoch: EPOCH_QUIESCE,
            slot: slot,
            _padding: [0xdB; (CACHE_LINE-10)],
        };
        t
    }
}

/// Table of epoch state per-thread.
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

    /// Scan entire array for lowest epoch.
    pub fn scan_min(&self) -> EpochRaw {
        unimplemented!();
    }

    /// Register new thread, allocating one slot to it.
    fn register(&self) -> *mut EpochSlot {
        let slot = self.freeslots.try_pop().unwrap() as usize;
        let p: *mut EpochSlot = unsafe {
            mem::transmute(&self.table[slot])
        };
        let sl = &self.table[slot];
        debug!("new slot: epoch {} slot {}", sl.epoch, sl.slot);
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::read;
    use std::thread::{self,JoinHandle};
    use std::sync::{Arc};
    use std::mem;
    use crossbeam::sync::SegQueue;

    #[test]
    fn do_read() {
        let f = read();
        assert!(f > 0);
        assert!(f < read());
    }

    #[test]
    fn do_next() {
        let f = next();
        assert!(f > 0);
        assert!(f < next());
    }

    #[test]
    fn do_pin() {
        assert_eq!(current(), None);

        pin();
        let cur = current();
        assert_eq!(cur.is_some(), true);
        assert_eq!(min(), cur);

        quiesce();
        assert_eq!(current(), None);
        assert_eq!(min(), None);
    }

    #[test]
    fn compare_slot_addr() {
        let N = 4;
        let mut threads: Vec<JoinHandle<()>> = Vec::with_capacity(N);
        let queue: SegQueue<usize> = SegQueue::new();
        let arcq = Arc::new(queue);
        for _ in 0..N {
            let queue = arcq.clone();
            threads.push(
                thread::spawn( move || {
                    let addr = slot_addr();
                    queue.push(addr as usize);
                })
            );
        }
        for thread in threads {
            let _ = thread.join();
        }
        let mut v: Vec<usize> = Vec::with_capacity(N);
        while let Some(u) = arcq.try_pop() {
            v.push(u);
        }
        v.sort();
        v.dedup();
        assert_eq!(v.len(), N);
        assert_eq!( (v[1] - v[0]) % mem::size_of::<EpochSlot>(), 0);
    }

    #[test]
    fn min_scan() {
        let N = 8;
        let iter = 100000;
        let mut threads: Vec<JoinHandle<()>> = Vec::with_capacity(N);
        for _ in 0..N {
            threads.push(
                thread::spawn( move || {
                    for _ in 0..iter { pin(); }
                })
                );
        }
        for thread in threads {
            let _ = thread.join();
        }
        let first = read();
        let m = min();
        assert!(m.is_some());
        let m = m.unwrap();
        assert!(m < first);

        let mut threads: Vec<JoinHandle<()>> = Vec::with_capacity(N);
        for _ in 0..N {
            threads.push(
                thread::spawn( move || {
                    for _ in 0..iter { pin(); }
                })
                );
        }
        for thread in threads {
            let _ = thread.join();
        }
        let m = min();
        assert!(m.is_some()); // FIXME really we want it to be None
        assert!(m.unwrap() < read());

        // FIXME
        // This (ideally) should be assert!(m > first); but because
        // threads don't 'unregister' their epoch from the table
        // (release the slot, and set epoch to zero), min values are
        // forever kept in the table when they pause, or terminate.
        assert!(m.unwrap() < first);
    }
}
