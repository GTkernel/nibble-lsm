use libc;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::ptr::copy_nonoverlapping;
use std::ptr::copy;

//pub mod nibble;

const BLOCK_SIZE: usize = 1 << 16;
const SEGMENT_SIZE: usize = 1 << 20;

// -------------------------------------------------------------------
// General utilities, macros, etc.
// -------------------------------------------------------------------

/// Extract reference &T from Option<T>
macro_rules! r {
    ( $obj:expr ) => { $obj.as_ref().unwrap() }
}

/// Borrow on a reference &T from a RefCell<Option<T>>
macro_rules! rb {
    ( $obj:expr ) => { r!($obj).borrow() }
}

/// Same as rb! but mutable
macro_rules! rbm {
    ( $obj:expr ) => { r!($obj).borrow_mut() }
}

// -------------------------------------------------------------------
// Error handling
// -------------------------------------------------------------------

pub enum ErrorCode {
    SegmentFull,
    SegmentClosed,
    OutOfMemory,
}

pub fn err2str(code: ErrorCode) -> &'static str {
    match code {
        ErrorCode::SegmentFull      => { "Segment is full" },
        ErrorCode::SegmentClosed    => { "Segment is closed" },
        ErrorCode::OutOfMemory      => { "Out of memory" },
    }
}

pub type Status = Result<(usize), ErrorCode>;

// -------------------------------------------------------------------
// Block strutures
// -------------------------------------------------------------------

pub struct Block {
    addr: usize,
    len: usize,
    // TODO owning Segment
}

impl Block {

    pub fn new(addr: usize, len: usize) -> Self {
        debug!("new Block 0x{:x} {}", addr, len);
        Block { addr: addr, len: len }
    }
}

pub type BlockRef = Arc<Block>;
pub type BlockRefPool = Vec<BlockRef>;

pub struct BlockAllocator {
    block_size: usize,
    pool: BlockRefPool,
    freepool: BlockRefPool,
    mmap: MemMap,
}

impl BlockAllocator {

    pub fn new(block_size: usize, bytes: usize) -> Self {
        let mmap = MemMap::new(bytes);
        let count = bytes / block_size;
        let mut pool: Vec<Arc<Block>> = Vec::new();
        let mut freepool: Vec<Arc<Block>> = Vec::new();
        for b in 0..count {
            let addr = mmap.addr() + b*block_size;
            let b = Arc::new(Block::new(addr, block_size));
            pool.push(b.clone());
            freepool.push(b.clone());
        }
        BlockAllocator { block_size: block_size,
            pool: pool, freepool: freepool, mmap: mmap,
        }
    }

    pub fn alloc(&mut self, count: usize) -> Option<BlockRefPool> {
        // TODO lock
        match self.freepool.len() {
            0 => None,
            len =>
                if count <= len {
                    Some(self.freepool.split_off(len-count))
                } else { None },
        }
    }

    pub fn len(&self) -> usize { self.pool.len() }
    pub fn freelen(&self) -> usize { self.freepool.len() }
}

// -------------------------------------------------------------------
// Segment structures
// -------------------------------------------------------------------

pub type SegmentRef = Arc<RefCell<Segment>>;
pub type SegmentManagerRef = Arc<RefCell<SegmentManager>>;

/// Make a new SegmentRef
macro_rules! seg_ref {
    ( $id:expr, $blocks:expr ) => {
        Arc::new( RefCell::new(
                Segment::new($id, $blocks)
                ))
    }
}

/// Make a new SegmentManagerRef
macro_rules! segmgr_ref {
    ( $id:expr, $segsz:expr, $blocks:expr ) => {
        Arc::new( RefCell::new(
                SegmentManager::new( $id, $segsz, $blocks)
                ))
    }
}

/// Describe entry in the log.
///     | EntryHeader | Key | Value |
pub struct EntryHeader {
    keylen: u32,
    datalen: u32,
}

/// Description of a buffer for copying into the log.
pub struct BufDesc {
    addr: *const u8,
    len: usize,
}

pub struct Segment {
    id: usize,
    closed: bool,
    head: usize, /// Virtual address of head TODO atomic
    len: usize, /// Total capacity TODO atomic
    rem: usize, /// Remaining capacity TODO atomic
    blocks: BlockRefPool,
}

impl Segment {

    pub fn new(id: usize, blocks: BlockRefPool) -> Self {
        let mut len: usize = 0;
        for b in blocks.iter() {
            len += b.len;
        }
        let start: usize = blocks[0].addr;
        Segment {
            id: id, closed: false, head: start,
            len: len, rem: len, blocks: blocks,
        }
    }

    pub fn increment(&mut self, len: usize) {
        // TODO increment head atomically
        self.head += len; // FIXME handle block boundaries
        self.rem -= len;
    }

    // TODO append an object -- maybe make this unsafe?
    pub unsafe fn append(&mut self, buf: &BufDesc) -> Status {
        if !self.closed {
            if self.has_space_for(buf) {
                let dest = self.head as *mut u8;
                copy_nonoverlapping(buf.addr,dest,buf.len);
                self.increment(buf.len);
                Ok(buf.len)
            } else { Err(ErrorCode::SegmentFull) }
        } else { Err(ErrorCode::SegmentClosed) }
    }

    pub fn has_space_for(&self, buf: &BufDesc) -> bool {
        self.rem >= buf.len
    }

    pub fn close(&mut self) {
        self.closed = true;
    }
}

impl Drop for Segment {

    fn drop(&mut self) {
        for b in self.blocks.iter() {
            // TODO push .clone() to BlockAllocator
        }
    }
}

pub struct SegmentManager {
    id: usize,
    size: usize, /// Total memory
    next_seg_id: usize, // FIXME atomic
    segment_size: usize,
    allocator: BlockAllocator,
    segments: Vec<Option<SegmentRef>>,
    free_slots: Vec<u32>,
}

impl SegmentManager {

    pub fn new(id: usize, segsz: usize, len: usize) -> Self {
        let b = BlockAllocator::new(BLOCK_SIZE, len);
        let num = len / segsz;
        let mut v: Vec<Option<SegmentRef>> = Vec::new();
        for i in 0..num {
            v.push(None);
        }
        let mut s: Vec<u32> = vec![0; num];
        for i in 0..num {
            s[i] = i as u32;
        }
        SegmentManager {
            id: id, size: len,
            next_seg_id: 0,
            segment_size: segsz,
            allocator: b,
            segments: v,
            free_slots: s,
        }
    }

    pub fn alloc(&mut self) -> Option<SegmentRef> {
        // TODO lock, unlock
        if self.free_slots.is_empty() {
            None
        } else {
            let slot: usize;
            match self.free_slots.pop() {
                None => panic!("No free slots"),
                Some(v) => slot = v as usize,
            };
            match self.segments[slot] {
                None => {},
                _ => panic!("Alloc from non-empty slot"),
            };
            // FIXME use config obj
            let num = SEGMENT_SIZE / BLOCK_SIZE;
            let blocks = self.allocator.alloc(num);
            match blocks {
                None => panic!("Could not allocate blocks"),
                _ => {},
            };
            self.next_seg_id += 1;
            self.segments[slot] = Some(seg_ref!(self.next_seg_id,
                                                  blocks.unwrap()));
            self.segments[slot].clone()
        }
    }

    pub fn free(&self, segment: SegmentRef) {
        unimplemented!();
    }

    //
    // --- Internal methods used for testing only ---
    //

    #[cfg(test)]
    pub fn test_all_segrefs_allocated(&mut self) -> bool {
        for i in 0..self.segments.len() {
            match self.segments[i] {
                None => return false,
                _ => {},
            }
        }
        true
    }
}

// -------------------------------------------------------------------
// The log
// -------------------------------------------------------------------

// TODO i definitely need a concurrent atomic append/pop list..

pub type LogHeadRef = Arc<RefCell<LogHead>>;

pub struct LogHead {
    segment: Option<SegmentRef>,
    manager: SegmentManagerRef,
}

impl LogHead {

    pub fn new(manager: SegmentManagerRef) -> Self {
        LogHead { segment: None, manager: manager }
    }

    pub unsafe fn append(&mut self, buf: &BufDesc) -> Status {
        // allocate if head not exist
        match self.segment {
            None => { match self.roll() {
                Err(code) => return Err(code),
                Ok(ign) => {},
            }},
            _ => {},
        }
        if !rb!(self.segment).has_space_for(buf) {
            match self.roll() {
                Err(code) => return Err(code),
                Ok(ign) => {},
            }
        }
        let mut seg = rbm!(self.segment);
        // do the append
        match seg.append(buf) {
            Ok(len) => Ok(len),
            Err(code) => panic!("has space but append failed"),
        }
    }

    //
    // --- Private methods ---
    //

    /// Roll head. Close current and allocate new.
    pub fn roll(&mut self) -> Status {
        match self.segment.clone() {
            None => {
                self.segment = self.manager.borrow_mut().alloc();
                match self.segment {
                    None => Err(ErrorCode::OutOfMemory),
                    _ => Ok(1),
                }
            },
            Some(segref) => {
                {
                    let mut seg = segref.borrow_mut();
                    seg.close();
                    // TODO add segment to 'closed' list
                }
                self.segment = self.manager.borrow_mut().alloc();
                match self.segment {
                    None => Err(ErrorCode::OutOfMemory),
                    _ => Ok(1),
                }
            },
        }
    }

}

pub struct Log {
    head: LogHeadRef, // TODO make multiple
    manager: SegmentManagerRef,
}

impl Log {
    pub fn new(manager: SegmentManagerRef) -> Self {
        Log {
            head: Arc::new(RefCell::new(LogHead::new(manager.clone()))),
            manager: manager.clone(),
        }
    }

    pub fn get_head(&self) -> Option<LogHeadRef> {
        Some(self.head.clone())
    }

    pub unsafe fn append(&mut self, buf: &BufDesc) -> Status {
        // 1. determine log head to use
        let head = &self.head;
        // 2. call append on the log head
        head.borrow_mut().append(buf)
    }

    pub fn enable_cleaning(&mut self) {
        unimplemented!();
    }

    pub fn disable_cleaning(&mut self) {
        unimplemented!();
    }
}

// -------------------------------------------------------------------
// TODO Cleaning
// -------------------------------------------------------------------

// -------------------------------------------------------------------
// TODO Index structure
// -------------------------------------------------------------------

// -------------------------------------------------------------------
// TODO RPC or SHM interface
// -------------------------------------------------------------------

// -------------------------------------------------------------------
// Memory utilities
// -------------------------------------------------------------------

/// Memory mapped region in our address space.
pub struct MemMap {
    addr: usize,
    len: usize,
}

/// Create anonymous private memory mapped region.
impl MemMap {

    pub fn new(len: usize) -> Self {
        // TODO fault on local socket
        let prot: libc::c_int = libc::PROT_READ | libc::PROT_WRITE;
        let flags: libc::c_int = libc::MAP_ANON |
            libc::MAP_PRIVATE | libc::MAP_NORESERVE;
        let addr: usize = unsafe {
            let p = 0 as *mut libc::c_void;
            libc::mmap(p, len, prot, flags, 0, 0) as usize
        };
        info!("mmap 0x{:x} {} MiB", addr, len>>20);
        assert!(addr != libc::MAP_FAILED as usize);
        MemMap { addr: addr, len: len }
    }

    pub fn addr(&self) -> usize { self.addr }
    pub fn len(&self) -> usize { self.len }
}

/// Prevent dangling regions by unmapping it.
impl Drop for MemMap {

    fn drop (&mut self) {
        unsafe {
            let p = self.addr as *mut libc::c_void;
            libc::munmap(p, self.len);
        }
    }
}

// -------------------------------------------------------------------
// Test Code
// -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::cell::RefCell;

    const BLOCK_SIZE: usize = 1 << 16;
    const SEGMENT_SIZE: usize = 1 << 20;

    #[test]
    fn memory_map_init() {
        let len = 1<<26;
        let mm = MemMap::new(len);
        assert_eq!(mm.len, len);
        assert!(mm.addr != 0 as usize);
        // TODO touch the memory somehow
        // TODO verify mmap region is unmapped
    }

    #[test]
    fn block() {
        let b = Block::new(42, 37);
        assert_eq!(b.addr, 42);
        assert_eq!(b.len, 37);
    }

    #[test]
    fn block_allocator_alloc_all() {
        let num = 64;
        let bytes = num * BLOCK_SIZE;
        let mut ba = BlockAllocator::new(BLOCK_SIZE, bytes);
        assert_eq!(ba.len(), num);
        assert_eq!(ba.freelen(), num);
        assert_eq!(ba.block_size, BLOCK_SIZE);

        let mut count = 0;
        while ba.freelen() > 0 {
            match ba.alloc(2) {
                None => break,
                opt => {
                    for b in opt.unwrap() {
                        count += 1;
                    }
                },
            }
        }
        assert_eq!(count, num);
        assert_eq!(ba.len(), num);
        assert_eq!(ba.freelen(), 0);
    }

    #[test]
    fn alloc_segment() {
        let num = 64;
        let bytes = num * BLOCK_SIZE;
        let mut ba = BlockAllocator::new(BLOCK_SIZE, bytes);
        let set;
        match ba.alloc(64) {
            None => panic!("Alloc should not fail"),
            opt => set = opt.unwrap(),
        }
        let id = 42;
        let seg = Segment::new(id, set);
        assert_eq!(seg.closed, false);
        assert!(seg.head != 0);
        assert_eq!(seg.len, bytes);
        // TODO verify blocks?
    }

    #[test]
    fn segment_manager() {
        let memlen = 1<<23;
        let numseg = memlen / SEGMENT_SIZE;
        let mut mgr = SegmentManager::new(0, SEGMENT_SIZE, memlen);
        for i in 0..numseg {
            match mgr.alloc() {
                None => panic!("segment alloc failed"),
                _ => {},
            }
        }
        assert_eq!(mgr.test_all_segrefs_allocated(), true);
    }

    #[test]
    fn log() {
        let memlen = 1<<23;
        let numseg = memlen / SEGMENT_SIZE;
        let mut log;
        {
            let manager = segmgr_ref!(0, SEGMENT_SIZE, memlen);
            log = Log::new(manager.clone());
        }
        let myval: &'static str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaalex";
        let addr = myval.as_ptr();
        let len = myval.len();
        let buf = BufDesc { addr: addr, len: len };
        loop {
            let v;
            unsafe { v = log.append(&buf); }
            match v {
                Ok(ign) => {},
                Err(code) => {
                    println!("append returned {}",
                                      err2str(code));
                    break;
                }
            }
        }
    }


}
