use common::*;
use thelog::*;
use memory::*;
use compaction::*;

use std::cell::RefCell;
use std::mem::size_of;
use std::ptr;
use std::ptr::copy;
use std::ptr::copy_nonoverlapping;
use std::sync::Arc;
use std::mem::transmute;
use std::cmp;
use std::cmp::Ordering;

pub const BLOCK_SIZE: usize = 1 << 16;
pub const SEGMENT_SIZE: usize = 1 << 20;

//==----------------------------------------------------==//
//      Block structures
//==----------------------------------------------------==//


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

//==----------------------------------------------------==//
//      Block allocator
//==----------------------------------------------------==//

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

//==----------------------------------------------------==//
//      Segment utilities
//==----------------------------------------------------==//

pub type SegmentRef = Arc<RefCell<Segment>>;
pub type SegmentManagerRef = Arc<RefCell<SegmentManager>>;

/// Instantiate new Segment as a SegmentRef
macro_rules! seg_ref {
    ( $id:expr, $blocks:expr ) => {
        Arc::new( RefCell::new(
                Segment::new($id, $blocks)
                ))
    }
}

/// Instantiate new Segment with zero blocks
macro_rules! seg_ref_empty {
    ( $id:expr ) => {
        Arc::new( RefCell::new(
                Segment::empty($id)
                ))
    }
}

/// Make a new SegmentManagerRef
#[macro_export]
macro_rules! segmgr_ref {
    ( $id:expr, $segsz:expr, $blocks:expr ) => {
        Arc::new( RefCell::new(
                SegmentManager::new( $id, $segsz, $blocks)
                ))
    }
}

//==----------------------------------------------------==//
//      Object descriptor
//==----------------------------------------------------==//

/// Use this to describe an object. User is responsible for
/// transmuting their objects, and for ensuring the lifetime of the
/// originating buffers exceeds that of an instance of ObjDesc used to
/// refer to them.
pub struct ObjDesc<'a> {
    key: &'a str,
    value: Pointer,
    vlen: u32,
}

impl<'a> ObjDesc<'a> {

    pub fn new(key: &'a str, value: Pointer, vlen: u32) -> Self {
        ObjDesc { key: key, value: value, vlen: vlen }
    }

    pub fn len(&self) -> usize {
        self.key.len() + self.vlen as usize
    }

    pub fn len_with_header(&self) -> usize {
        size_of::<EntryHeader>() + self.len()
    }

    /// Releases memory associated with a .value that is allocated
    /// internally upon retreiving an object from the log.
    pub unsafe fn release_value(&mut self) {
        match self.value.take() {
            // v goes out of scope and its memory released
            Some(va) => { let v = Box::from_raw(va as *mut u8); },
            None => {},
        }
    }

    pub fn getkey(&self) -> &'a str { self.key }
    pub fn keylen(&self) -> usize { self.key.len() }
    pub fn getvalue(&self) -> Pointer { self.value }
    pub fn valuelen(&self) -> u32 { self.vlen }
}

//==----------------------------------------------------==//
//      Segment header
//==----------------------------------------------------==//

/// Structure written to offset 0 of each segment and updated when new
/// objects are appended.
#[derive(Debug)]
#[repr(packed)]
pub struct SegmentHeader {
    num_objects: u32,
}

impl SegmentHeader {

    pub fn new(n: u32) -> Self {
        SegmentHeader { num_objects: n }
    }

    pub fn nobj(&self) -> u32 { self.num_objects }
    pub fn len() -> usize { size_of::<Self>() }
}

//==----------------------------------------------------==//
//      Segment
//==----------------------------------------------------==//

// TODO need metrics for computing compaction weights
// TODO head,len,rem as atomics
pub struct Segment {
    id: usize,
    closed: bool,
    head: Option<usize>, /// Virtual address of head TODO atomic
    len: usize, /// Total capacity TODO atomic
    rem: usize, /// Remaining capacity TODO atomic
    nobj: usize, /// Objects (live or not) appended
    curblk: Option<usize>,
    front: Option<*mut SegmentHeader>,
    blocks: BlockRefPool,
}

// TODO create a wrapper type that implements sorting of a segment
// based on specific metrics (if i want to put Segments into multiple
// binary heaps)

/// A logically contiguous chunk of memory. Inside, divided into
/// virtually contiguous blocks. For now, objects cannot be broken
/// across a block boundary.
impl Segment {

    // TODO make blocks a ref
    pub fn new(id: usize, blocks: BlockRefPool) -> Self {
        let mut len: usize = 0;
        assert!(blocks.len() > 0);
        for b in blocks.iter() {
            len += b.len;
        }
        assert!(len > 0);
        let blk = 0;
        let start: usize = blocks[blk].addr;
        assert!(start != 0);
        let header = SegmentHeader::new(0);
        unsafe { ptr::write(start as *mut SegmentHeader, header); }
        Segment {
            id: id, closed: false, head: Some(start + SegmentHeader::len()),
            len: len, rem: len - SegmentHeader::len(),
            nobj: 0, curblk: Some(blk),
            front: Some(blocks[blk].addr as *mut SegmentHeader),
            blocks: blocks,
        }
    }

    /// Allocate an empty Segment. Used by the compaction code.
    pub fn empty(id: usize) -> Self {
        Segment {
            id: id, closed: true, head: None,
            len: 0, rem: 0, nobj: 0,
            curblk: None,
            front: None,
            blocks: vec!(),
        }
    }

    /// Add blocks to a segment to grow its size. Used by the
    /// compaction code, after allocating an empty segment.
    /// TODO need unit test
    pub fn extend(&mut self, blocks: &mut BlockRefPool) {
        assert!(self.blocks.len() > 0);
        match self.head {
            // segment was empty
            None => {
                self.curblk = Some(0);
                self.head = Some(blocks[0].addr);
                self.front = Some(blocks[0].addr as *mut SegmentHeader);
            },
            _ => {},
        }
        let mut len = 0 as usize;
        for b in blocks.iter() {
            len += b.len;
        }
        self.len += len;
        self.rem += len;
        self.blocks.append(blocks);
    }

    /// Append an object with header to the log. Let append_safe
    /// handle the actual work.  If successful, returns virtual
    /// address in Ok().
    /// TODO Segment shouldn't know about log-specific organization,
    /// e.g. entry headers
    pub fn append(&mut self, buf: &ObjDesc) -> Status {
        assert_eq!(self.head.is_some(), true);
        if !self.closed {
            if self.can_hold(buf) {
                let val: *const u8;
                match buf.value {
                    None => return Err(ErrorCode::EmptyObject),
                    Some(va) => { val = va; },
                }
                match buf.vlen {
                    0 => return Err(ErrorCode::EmptyObject),
                    _ => {},
                }
                let va = self.headref() as usize;
                let header = EntryHeader::new(buf);
                let hlen = size_of::<EntryHeader>();
                self.append_safe(header.as_ptr(), hlen);
                self.append_safe(buf.key.as_ptr(), buf.key.len());
                self.append_safe(val, buf.vlen as usize);
                self.nobj += 1;
                self.update_header(1);
                Ok(va)
            } else { Err(ErrorCode::SegmentFull) }
        } else { Err(ErrorCode::SegmentClosed) }
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    pub fn nobjects(&self) -> usize {
        self.nobj
    }

    /// Increment values in header by specified amount
    /// TODO can i directly update the value? like in C:
    ///         struct header *h = (struct header*)self.front;
    ///         h->num_objects++;
    fn update_header(&self, n: u32) {
        assert_eq!(self.front.is_some(), true);
        let mut header: SegmentHeader;
        unsafe { header = ptr::read(self.front.unwrap()); }
        header.num_objects += n;
        unsafe { ptr::write(self.front.unwrap(), header); }
    }

    /// Increment the head offset into the start of the next block.
    /// Return false if we cannot roll because we're already at the
    /// final block.
    fn next_block(&mut self) -> bool {
        assert_eq!(self.head.is_some(), true);
        assert_eq!(self.curblk.is_some(), true);
        let mut curblk = self.curblk.unwrap();
        let numblks: usize = self.blocks.len();
        assert!(numblks > 0);
        if curblk < (numblks - 1) {
            curblk += 1;
            self.curblk = Some(curblk);
            self.head = Some(self.blocks[curblk].addr);
            true
        } else {
            warn!("block roll asked on last block");
            false
        }
    }

    // TODO should have test code that scrapes the segment to find
    // actual entries
//    /// Count entries written to segment (ignoring index). mainly for
//    /// testing.
//    #[cfg(test)]
//    pub fn scan_entries(&self) -> usize {
//        let mut count: usize = 0;
//        let mut header = EntryHeader::empty();
//        let mut offset: usize = 0; // offset into segment (logical)
//        let mut curblk: usize = 0; // Block that offset refers to
//        // Manually cross block boundaries searching for entries.
//        while offset < self.len {
//            assert!(curblk < self.blocks.len());
//            let base: usize = self.blocks[curblk].addr;
//            let addr: usize = base + (offset % BLOCK_SIZE);
//
//            let rem = BLOCK_SIZE - (offset % BLOCK_SIZE); // remaining in block
//
//            // If header is split across block boundaries
//            if rem < size_of::<EntryHeader>() {
//                // Can't be last block if header is partial
//                assert!(curblk != self.blocks.len()-1);
//                unsafe {
//                    let mut from = addr;
//                    let mut to: usize = transmute(&header);
//                    copy(from as *const u8, to as *mut u8, rem);
//                    from = self.blocks[curblk+1].addr; // start at next block
//                    to += rem;
//                    copy(from as *const u8, to as *mut u8,
//                         size_of::<EntryHeader>() - rem);
//                }
//            }
//            // Header is fully contained in the block
//            else {
//                unsafe { header.read(addr); }
//            }
//
//            count += 1;
//
//            // Determine next location to jump to
//            offset += header.len();
//            curblk = offset / BLOCK_SIZE;
//        }
//        count
//    }

    pub fn can_hold(&self, buf: &ObjDesc) -> bool {
        self.rem >= buf.len_with_header()
    }

    //
    // --- Private methods ---
    //

    fn headref(&mut self) -> *mut u8 {
        match self.head {
            Some(va) => va as *mut u8,
            None => panic!("taking head ref but head not set"),
        }
    }

    /// Append some buffer safely across block boundaries (if needed).
    /// Caller must ensure the containing segment has sufficient
    /// capacity.
    fn append_safe(&mut self, from: *const u8, len: usize) {
        assert!(len <= self.rem);
        assert_eq!(self.head.is_some(), true);
        let mut remblk = self.rem_in_block();
        // 1. If buffer fits in remainder of block, just copy it
        if len <= remblk {
            unsafe {
                copy_nonoverlapping(from,self.headref(),len);
            }
            incr!(self.head, len);
            if len == remblk {
                self.next_block();
            }
        }
        // 2. If it spills over, perform two (or more) copies.
        else {
            let mut loc = from;
            let mut rem = len;
            // len may exceeed capacity of one block. Copy and roll in
            // pieces until the input is consumed.
            while rem > 0 {
                remblk = self.rem_in_block();
                let amt = cmp::min(remblk,rem);
                unsafe {
                    copy_nonoverlapping(loc,self.headref(), amt);
                }
                incr!(self.head, amt);
                rem -= amt;
                loc = (from as usize + amt) as *const u8;
                // If we exceeded the block, get the next one
                if remblk == amt {
                    assert_eq!(self.next_block(), true);
                }
            }
        }
        self.rem -= len;
    }

    fn rem_in_block(&self) -> usize {
        assert_eq!(self.head.is_some(), true);
        assert!(self.blocks.len() > 0);
        let curblk = self.curblk.unwrap();
        let blk = &self.blocks[curblk];
        blk.len - (self.head.unwrap() - blk.addr)
    }

    //
    // --- Test methods ---
    //

    #[cfg(test)]
    pub fn reset(&mut self) {
        let blk = 0;
        let start = self.blocks[blk].addr + SegmentHeader::len();
        self.closed = false;
        self.head = Some(self.blocks[blk].addr);
        self.rem = self.len - SegmentHeader::len();
        self.curblk = Some(blk);
        self.nobj = 0;
    }
}

impl Drop for Segment {

    fn drop(&mut self) {
        for b in self.blocks.iter() {
            // TODO push .clone() to BlockAllocator
        }
    }
}

impl<'a> IntoIterator for &'a Segment {
    type Item = EntryReference;
    type IntoIter = SegmentIter;

    fn into_iter(self) -> Self::IntoIter {
        SegmentIter::new(self)
    }
}

//==----------------------------------------------------==//
//      Segment iterator
//==----------------------------------------------------==//

/// Iterator for a Segment.  We expect a segment to be iterated over
/// by one thread at a time.  The compactor will check liveness of
/// each entry, not us.
pub struct SegmentIter {
    blocks: BlockRefPool,
    n_obj: usize,
    blk_offset: usize,
    cur_blk: usize,
    seg_offset: usize,
    next_obj: usize,
}

impl SegmentIter {

    pub fn new(seg: &Segment) -> Self {
        let offset = SegmentHeader::len();
        SegmentIter {
            blocks: seg.blocks.clone(), // refs to blocks
            n_obj: seg.nobj,
            blk_offset: offset,
            cur_blk: 0,
            seg_offset: offset,
            next_obj: 0,
        }
    }
}

impl Iterator for SegmentIter {
    type Item = EntryReference;

    fn next(&mut self) -> Option<EntryReference> {
        if self.next_obj >= self.n_obj {
            return None;
        }

        // don't advance if we're at first entry
        if self.next_obj > 0 {
            // read length of current entry
            let addr = self.blocks[self.cur_blk].addr + self.blk_offset;
            let mut entry: EntryHeader;
            unsafe {
                entry = ptr::read(addr as *const EntryHeader);
            }

            // advance to next
            self.seg_offset += entry.len();
            self.cur_blk = self.seg_offset / BLOCK_SIZE;
            self.blk_offset = self.seg_offset % BLOCK_SIZE;
        }

        // read entry info
        let addr = self.blocks[self.cur_blk].addr + self.blk_offset;
        let mut entry: EntryHeader;
        unsafe {
            entry = ptr::read(addr as *const EntryHeader);
        }
        let obj_len = entry.object_length() as usize;

        // determine which blocks belong
        let mut nblks = 1;
        let blk_tail = BLOCK_SIZE - self.seg_offset;
        if obj_len > blk_tail {
            nblks += (obj_len - blk_tail) / BLOCK_SIZE + 1;
        }

        self.next_obj += 1;

        Some( EntryReference {
            // TODO is this clone expensive?
            blocks: self.blocks.clone().into_iter()
                .skip(self.cur_blk).take(nblks).collect(),
            offset: self.blk_offset,
            len: obj_len,
        } )
    }
}

//==----------------------------------------------------==//
//      Entry reference
//==----------------------------------------------------==//

/// Reference to entry in the log. Used by Segment iterators since i)
/// items in memory don't have an associated language type (this
/// provides that function) and ii) we want to avoid copying objects
/// each time a reference is passed around; we lazily copy the object
/// from the log only when a client asks for it
pub struct EntryReference {
    blocks: BlockRefPool,
    offset: usize, // into first block
    len: usize,
}

impl EntryReference {

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn copy_out(&self, to: *mut u8) {
        unimplemented!();
    }

}

//==----------------------------------------------------==//
//      Segment manager
//==----------------------------------------------------==//

pub struct SegmentManager {
    id: usize,
    size: usize, /// Total memory
    next_seg_id: usize, // FIXME atomic
    segment_size: usize,
    allocator: BlockAllocator,
    segments: Vec<Option<SegmentRef>>,
    free_slots: Vec<u32>,
    //compactor: CompactorRef,
}

impl SegmentManager {
    // TODO write an iterator; update unit test below

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
            //compactor: compref!(),
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

    /// Grab a Segment without any blocks to use for compaction.
    /// Blocks are lazily allocated.
    /// TODO clean this up so it isn't copy/paste with alloc()
    pub fn allocEmpty(&mut self) -> Option<SegmentRef> {
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
            self.next_seg_id += 1;
            self.segments[slot] = Some(seg_ref_empty!(self.next_seg_id));
            self.segments[slot].clone()
        }
    }

    pub fn free(&self, segment: SegmentRef) {
        unimplemented!();
    }

    //    pub fn newly_closed(&mut self, seg: &SegmentRef) {
    //        match self.compactor.lock() {
    //            Ok(ref mut 
    //        }
    //        self.compactor.add(seg);
    //    }

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

    #[cfg(test)]
    pub fn test_scan_objects(&self) -> usize {
        let mut count: usize = 0;
        for opt in &self.segments {
            match *opt {
                None => {},
                Some(ref seg) => {
                    count += seg.borrow().nobjects();
                },
            }
        }
        count
    }
}

//==----------------------------------------------------==//
//      Unit tests
//==----------------------------------------------------==//

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::mem::size_of;
    use std::mem::transmute;
    use std::ptr::copy;
    use std::ptr::copy_nonoverlapping;
    use std::rc::Rc;
    use std::sync::Arc;

    use test::Bencher;

    use thelog::*;
    use common::*;

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
        assert_eq!(seg.head.is_some(), true);
        assert_eq!(seg.len, bytes);
        // TODO verify blocks?
    }

    // TODO free blocks of segment (e.g. after compaction)

    #[test]
    fn segment_manager_alloc_all() {
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

    // TODO return segments back to manager

    /// Insert one unique object repeatedly;
    /// there should be only one live object in the log.
    /// lots of copy/paste...
    #[test]
    fn segment_manager_one_obj_overwrite() {
        let memlen = 1<<23;
        let numseg = memlen / SEGMENT_SIZE;
        let manager = segmgr_ref!(0, SEGMENT_SIZE, memlen);
        let mut log = Log::new(manager.clone());

        let key: &'static str = "onlyone";
        let val: &'static str = "valuevaluevalue";
        let obj = ObjDesc::new(key, Some(val.as_ptr()), val.len() as u32);
        // fill up the log
        let mut count: usize = 0;
        loop {
            match log.append(&obj) {
                Ok(va) => count += 1,
                Err(code) => match code {
                    ErrorCode::OutOfMemory => break,
                    _ => panic!("filling log returned {:?}", code),
                },
            }
        }
        assert_eq!(manager.borrow().test_scan_objects(), count);
    }

    // TODO entry reference types

    fn iterate_segment() {
        // TODO make a macro out of these lines
        let memlen = 1<<23;
        let numseg = memlen / SEGMENT_SIZE;
        let mut mgr = SegmentManager::new(0, SEGMENT_SIZE, memlen);

        let mut segref = mgr.alloc();
        let mut seg = rbm!(segref);

        let keys = vec!("aga234sdf", "sdfn34 2309dsfa;;", "LDKJF@()#*%FS3p853D");
        let values = vec!("23487sdfl0k", "laksdfkasdjflkasjdf", "0");

        // TODO how to advance two iterators simultaneously?
        // maybe use crate itertools::ZipEq 
        for i in 0..keys.len() {
            let loc = Some(values[i].as_ptr());
            let len = values[i].len() as u32;
            let obj = ObjDesc::new(keys[i], loc, len);
            match seg.append(&obj) {
                Err(code) => panic!("appending returned {:?}", code),
                _ => {},
            }
        }

        // TODO exercise the entry reference

        let mut counter = 0;
        for entry_ref in seg.into_iter() {
            println!("len: {} blocks: {}",
                     entry_ref.len(),
                     entry_ref.num_blocks());
            counter += 1;
        }
        assert_eq!(counter, keys.len());

        // now with enough keys to fill half a segment
        seg.reset();
        // large prime values so things don't fit nicely
        let key = String::with_capacity(1489);
        let value: [u8; 2477] = [7; 2477];
        let nobj = (SEGMENT_SIZE>>1) / (key.len() + value.len());
        for i in 0..nobj {
            let loc = Some(key.as_ptr());
            let len = value.len() as u32;
            let obj = ObjDesc::new(key.as_str(), loc, len);
            match seg.append(&obj) {
                Err(code) => panic!("appending returned {:?}", code),
                _ => {},
            }
        }
    }

    #[test]
    fn iterate_segment_large_objects() {
        // TODO
    }
}
