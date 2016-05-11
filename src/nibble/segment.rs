// TODO create a block allocator for each CPU socket

use common::*;
use thelog::*;
use memory::*;
use epoch::*;

use std::cmp;
use std::mem;
use std::mem::size_of;
use std::ptr;
use std::sync::{Arc, Mutex, RwLock};
use std::collections::VecDeque;

use crossbeam::sync::SegQueue;

pub const BLOCK_SHIFT:   usize = 16;
pub const BLOCK_SIZE:    usize = 1 << BLOCK_SHIFT;
pub const SEGMENT_SHIFT: usize = 20;
pub const SEGMENT_SIZE:  usize = 1 << SEGMENT_SHIFT;

//==----------------------------------------------------==//
//      Block structures
//==----------------------------------------------------==//


/// A contiguous region of virtual memory.
/// FIXME seg may alias others across NUMA domains
#[derive(Debug)]
pub struct Block {
    addr: usize,
    len: usize,
    slot: usize,
    seg: Option<usize>,
}

impl Block {

    pub fn new(addr: usize, slot: usize, len: usize) -> Self {
        trace!("new Block 0x{:x} slot {} len {}", addr, slot, len);
        Block { addr: addr, len: len, slot: slot, seg: None }
    }

    pub unsafe fn set_segment(&self, seg: usize) {
        let p: *mut Option<usize> = mem::transmute(&self.seg);
        ptr::write(p, Some(seg));
    }

    pub unsafe fn clear_segment(&self) {
        let p: *mut Option<usize> = mem::transmute(&self.seg);
        ptr::write(p, None);
    }

    pub fn addr(&self) -> usize { self.addr }
    pub fn len(&self)  -> usize { self.len }
    pub fn slot(&self) -> usize { self.slot }
    pub fn seg(&self)  -> Option<usize> { self.seg }

}

pub type BlockRef = Arc<Block>;
pub type BlockRefPool = Vec<BlockRef>;

//==----------------------------------------------------==//
//      Block allocator
//==----------------------------------------------------==//

pub struct BlockAllocator {
    pool: BlockRefPool,
    freepool: BlockRefPool,
    mmap: MemMap,
}

impl BlockAllocator {

    pub fn new(bytes: usize) -> Self {
        let mmap = MemMap::new(bytes);
        let count = bytes / BLOCK_SIZE;
        let mut pool: Vec<Arc<Block>> = Vec::with_capacity(count);
        let mut freepool: Vec<Arc<Block>> = Vec::with_capacity(count);
        for b in 0..count {
            let addr = mmap.addr() + b*BLOCK_SIZE;
            let b = Arc::new(Block::new(addr, b, BLOCK_SIZE));
            pool.push(b.clone());
            freepool.push(b.clone());
        }
        BlockAllocator { pool: pool, freepool: freepool, mmap: mmap, }
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

    pub fn free(&mut self, pool: &mut BlockRefPool) {
        self.freepool.append(pool);
    }

    pub fn len(&self) -> usize { self.pool.len() }
    pub fn freelen(&self) -> usize { self.freepool.len() }

    /// Convert virtual address to containing block.
    pub fn block_of(&self, addr: usize) -> Option<BlockRef> {
        let idx: usize = (addr - self.mmap.addr()) >> BLOCK_SHIFT;
        if idx < self.pool.len() {
            let b = self.pool[idx].clone();
            assert!(addr >= b.addr);
            assert!(addr < (b.addr + BLOCK_SIZE));
            Some(self.pool[idx].clone())
        } else {
            None
        }
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

    /// Create ObjDesc where key is str and value is arbitrary memory.
    pub fn new(key: &'a str, value: Pointer, vlen: u32) -> Self {
        ObjDesc { key: key, value: value, vlen: vlen }
    }

    /// Create ObjDesc where key and value are String
    pub fn new2(key: &'a String, value: &'a String) -> Self {
        ObjDesc {
            key: key.as_str(),
            value: Some(value.as_ptr()),
            vlen: value.len() as u32,
        }
    }

    pub fn len(&self) -> usize {
        self.key.len() + self.vlen as usize
    }

    pub fn len_with_header(&self) -> usize {
        size_of::<EntryHeader>() + self.len()
    }

    /// Releases memory associated with a .value that is allocated
    /// internally upon retreiving an object from the log.
    #[cfg(IGNORE)]
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

pub type SegmentRef = Arc<RwLock<Segment>>;
pub type SegmentManagerRef = Arc<Mutex<SegmentManager>>;

// TODO need metrics for computing compaction weights
// TODO head,len,rem as atomics
#[allow(dead_code)]
pub struct Segment {
    id: usize,
    slot: usize, /// index into segment table TODO add unit test?
    closed: bool,
    head: Option<usize>, /// Virtual address of head TODO atomic
    len: usize, /// Total capacity TODO atomic
    rem: usize, /// Remaining capacity TODO atomic
    nobj: usize, /// Objects (live or not) appended
    curblk: Option<usize>,
    /// A *mut SegmentHeader, but kept as usize because the Rust
    /// compiler does not allow sharing of raw pointers.
    front: Option<usize>,
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
    pub fn new(id: usize, slot: usize, blocks: BlockRefPool) -> Self {
        debug!("new segment: slot {} #blks {}", slot, blocks.len());
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
            id: id, slot: slot, closed: false,
            head: Some(start + SegmentHeader::len()),
            len: len, rem: len - SegmentHeader::len(),
            nobj: 0, curblk: Some(blk),
            front: Some(blocks[blk].addr),
            blocks: blocks,
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
                self.front = Some(blocks[0].addr);
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
                if 0 == buf.vlen {
                    return Err(ErrorCode::EmptyObject);
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

    /// Copy an entry from one set of blocks to this segment. Used by
    /// the compaction logic. Returns address of relocated entry.
    /// TODO optimize to know when the blocks are contiguous
    /// TODO optimize for large objects: transfer whole blocks?
    /// TODO use Pointer or something to represent return address
    /// nearly verbatim overlap with EntryReference::copy_out
    pub fn append_entry(&mut self, entry: &EntryReference) -> usize {
        assert_eq!(self.head.is_some(), true);
        assert_eq!(self.can_hold_amt(entry.len), true);

        let new_va = self.headref() as usize;

        // append by-block since entry may not be virtually contiguous
        let mut va = entry.blocks[0].addr + entry.offset;
        let mut amt = cmp::min(BLOCK_SIZE - entry.offset, entry.len);
        trace!("append_entry len {} #blks {} amt {}",
               entry.len, entry.blocks.len(), amt);
        self.append_safe(va as *const u8, amt);

        // go for the rest
        let mut remaining = entry.len - amt;
        let mut bidx = 1;
        while remaining > 0 {
            assert!(bidx < entry.blocks.len());
            va = entry.blocks[bidx].addr;
            amt = cmp::min(BLOCK_SIZE, remaining);
            self.append_safe(va as *const u8, amt);
            remaining -= amt;
            bidx += 1;
        }

        self.nobj += 1;
        self.update_header(1);

        new_va
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    pub fn nobjects(&self) -> usize {
        self.nobj
    }

    pub fn nblocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn slot(&self) -> usize {
        self.slot
    }

    /// Increment values in header by specified amount
    /// TODO can i directly update the value? like in C:
    ///         struct header *h = (struct header*)self.front;
    ///         h->num_objects++;
    fn update_header(&self, n: u32) {
        assert_eq!(self.front.is_some(), true);
        let mut header: SegmentHeader;
        let p = self.front.unwrap() as *mut SegmentHeader;
        unsafe { header = ptr::read(p); }
        header.num_objects += n;
        unsafe { ptr::write(p, header); }
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

    pub fn can_hold_amt(&self, len: usize) -> bool {
        self.rem >= len
    }

    pub fn can_hold(&self, buf: &ObjDesc) -> bool {
        self.can_hold_amt(buf.len_with_header())
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
                ptr::copy_nonoverlapping(from,self.headref(),len);
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
                    ptr::copy_nonoverlapping(loc,self.headref(), amt);
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
    pub fn rem(&self) -> usize { self.rem }

    #[cfg(test)]
    pub fn len(&self) -> usize { self.len }

    #[cfg(test)]
    pub fn used(&self) -> usize { self.len - self.rem }

    #[cfg(test)]
    pub fn head(&self) -> Option<usize> { self.head }

    #[cfg(test)]
    pub fn curblk(&self) -> Option<usize> { self.curblk }

    #[cfg(test)]
    pub fn nblks(&self) -> usize { self.blocks.len() }

    #[cfg(test)]
    pub fn reset(&mut self) {
        let blk = 0;
        self.closed = false;
        self.head = Some(self.blocks[blk].addr);
        self.rem = self.len - SegmentHeader::len();
        self.curblk = Some(blk);
        self.nobj = 0;
    }
}

impl Drop for Segment {

    fn drop(&mut self) {
        debug!("segment slot {} dropped", self.slot);
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
            let entry: EntryHeader;
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
        let entry: EntryHeader;
        unsafe {
            entry = ptr::read(addr as *const EntryHeader);
        }
        let entry_len = entry.len_with_header();

        // determine which blocks belong
        let mut nblks = 1;
        let blk_tail = BLOCK_SIZE - (self.seg_offset % BLOCK_SIZE);
        if entry_len > blk_tail {
            nblks += ((entry_len - blk_tail) / BLOCK_SIZE) + 1;
        }

        trace!("segiter: objlen {} blktail {} segoff {} blkoff {}",
               entry_len, blk_tail, self.seg_offset,
               self.seg_offset % BLOCK_SIZE);

        self.next_obj += 1;

        Some( EntryReference {
            // TODO is this clone expensive?
            blocks: self.blocks.clone().into_iter()
                .skip(self.cur_blk).take(nblks).collect(),
            offset: self.blk_offset,
            len: entry_len,
            keylen: entry.getkeylen(),
            datalen: entry.getdatalen(),
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
    pub blocks: BlockRefPool,
    pub offset: usize, // into first block
    pub len: usize, /// header + key + data
    pub keylen: u32,
    pub datalen: u32,
}

// TODO optimize for cases where the blocks are contiguous
// copying directly, or avoid copying (provide reference to it)
impl EntryReference {

    pub fn get_loc(&self) -> usize {
        self.offset + self.blocks[0].addr
    }

    /// Copy out the key
    pub unsafe fn get_key(&self) -> String {
        let mut v: Vec<u8> = Vec::with_capacity(self.keylen as usize);
        v.set_len(self.keylen as usize);
        let mut offset = self.offset + size_of::<EntryHeader>();
        let block  = offset / BLOCK_SIZE;
        offset = offset % BLOCK_SIZE;
        let ptr = v.as_mut_ptr();
        self.copy_out(ptr, block, offset, self.keylen as usize);
        match String::from_utf8(v) {
            Ok(string) => string,
            Err(err) => panic!("{:?}", err),
        }
    }

    /// Copy out the value
    pub unsafe fn get_data(&self, out: *mut u8) {
        let mut offset = self.offset + self.len
                            - self.datalen as usize;
        let block  = offset / BLOCK_SIZE;
        offset = offset % BLOCK_SIZE;
        self.copy_out(out, block, offset, self.datalen as usize)
    }

    //
    // --- Private methods ---
    //

    /// TODO clean up code
    /// nearly verbatim overlap with Segment::append_entry
    unsafe fn copy_out(&self, out: *mut u8,
                       block: usize, offset: usize,
                       remaining: usize) {
        // reassign as mutable
        let mut block = block;
        let mut remaining = remaining;

        let mut src: *const u8;
        let mut dst: *mut u8;

        // Logical offset into new buffer
        let mut poffset: usize = 0;

        let mut va = self.blocks[block].addr + offset;
        let mut amt = cmp::min(BLOCK_SIZE - offset, remaining);
        src = va as *const u8;
        dst = ((out as usize) + poffset) as *mut u8;
        ptr::copy_nonoverlapping(src, dst, amt);
        remaining -= amt;
        poffset += amt;
        block += 1;

        while remaining > 0 {
            va = self.blocks[block].addr;
            amt = cmp::min(BLOCK_SIZE, remaining);
            src = va as *const u8;
            dst = ((out as usize) + poffset) as *mut u8;
            ptr::copy_nonoverlapping(src, dst, amt);
            remaining -= amt;
            poffset += amt;
            block += 1;
        }
    }
}

//==----------------------------------------------------==//
//      Segment manager
//==----------------------------------------------------==//

#[allow(dead_code)]
pub struct SegmentManager {
    id: usize,
    /// Total memory
    size: usize,
    next_seg_id: usize, // FIXME atomic
    allocator: BlockAllocator,
    segments: Vec<Option<SegmentRef>>,
    seginfo: SegmentInfoTableRef,
    free_slots: Arc<SegQueue<u32>>,
    // TODO lock these
    closed: VecDeque<SegmentRef>,
}

// TODO reclaim segments function and thread
impl SegmentManager {
    // TODO write an iterator; update unit test below

    pub fn new(id: usize, segsz: usize, len: usize) -> Self {
        let b = BlockAllocator::new(len);
        let num = len / segsz;
        // allocate more slots to accomodate some segments having
        // fewer blocks (not all are =SEGMENT_SIZE)
        let mut segments: Vec<Option<SegmentRef>>
            = Vec::with_capacity(num<<2);
        for _ in 0..num {
            segments.push(None);
        }
        let free_slots: SegQueue<u32> = SegQueue::new();
        for i in 0..num {
            free_slots.push(i as u32);
        }
        SegmentManager {
            id: id, size: len,
            next_seg_id: 0,
            allocator: b,
            segments: segments,
            seginfo: Arc::new(SegmentInfoTable::new(num)),
            free_slots: Arc::new(free_slots),
            closed: VecDeque::new(),
        }
    }

    /// Allocate a segment with a specific number of blocks. Used by
    /// compaction code.
    /// TODO wait/return if blocks aren't available (not panic)
    pub fn alloc_size(&mut self, nblks: usize) -> Option<SegmentRef> {
        // TODO lock, unlock
        let mut ret: Option<SegmentRef> = None;
        if let Some(s) = self.free_slots.try_pop() {
            let slot = s as usize;
            assert!(self.segments[slot].is_none());
            let blocks = match self.allocator.alloc(nblks) {
                None => panic!("Could not allocate blocks"),
                Some(b) => b,
            };
            for block in &blocks {
                unsafe { block.set_segment(slot); }
            }
            self.next_seg_id += 1;
            self.segments[slot] = Some(seg_ref!(self.next_seg_id,
                                                slot, blocks));
            ret = self.segments[slot].clone();
        }
        ret
    }

    /// Allocate a segment with default size.
    pub fn alloc(&mut self) -> Option<SegmentRef> {
        // TODO use config obj
        let nblks = (SEGMENT_SIZE - 1) / BLOCK_SIZE + 1;
        self.alloc_size(nblks)
    }

    // Consume the reference and release its blocks
    pub fn free(&mut self, segref: SegmentRef) {

        // drop only other known ref to segment
        // and extract slot#
        let slot = {
            let seg = segref.read().unwrap(); // FIXME don't lock
            self.segments[seg.slot] = None;
            seg.slot
        };

        // extract the segment (should only be one ref remaining)
        let mut seg = match Arc::try_unwrap(segref) {
            Err(_) => panic!("another ref to seg exists"),
            Ok(seglock) => { match seglock.into_inner() {
                Err(_) => panic!("lock held without arc??"),
                Ok(seg) => seg,
            }},
        };

        // TODO memset the segment? or the header?

        // release the blocks
        self.allocator.free(&mut seg.blocks);

        // this seg should have zero references at this point
        // do this last because it opens the slot above for use
        self.free_slots.push(slot as u32);
    }

    pub fn segment_of(&self, va: usize) -> Option<usize> {
        let mut ret: Option<usize> = None;
        if let Some(block) = self.allocator.block_of(va) {
            if let Some(idx) = block.seg() {
                assert!(idx < self.segments.len());
                ret = Some(idx);
            }
        }
        ret
    }

    /// Log heads pass segments here when it rolls over. Compaction
    /// threads will periodically query this queue for new segments to
    /// add to its candidate list.
    pub fn add_closed(&mut self, seg: &SegmentRef) {
        self.closed.push_back(seg.clone());
    }

    // TODO if using SegQueue, perhaps add max to pop off
    pub fn grab_closed(&mut self,
                       to: &mut VecDeque<SegmentRef>) -> usize {
        let n: usize = self.closed.len();
        to.append(&mut self.closed);
        n
    }

    pub fn seginfo(&self) -> SegmentInfoTableRef {
        self.seginfo.clone()
    }

    //
    // --- Internal methods used for testing only ---
    //

    #[cfg(test)]
    pub fn n_closed(&self) -> usize {
        self.closed.len()
    }

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
                Some(ref segref) => {
                    let seg = segref.read().unwrap();
                    count += seg.nobjects();
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

    use std::ops;
    use std::slice::from_raw_parts;
    use std::sync::{Arc,Mutex};

    use thelog::*;
    use common::*;
    use memory::*;

    use rand;
    use rand::Rng;

    use super::super::logger;

    #[test]
    fn block() {
        logger::enable();
        let b = Block::new(42, 0, 37);
        assert_eq!(b.addr, 42);
        assert_eq!(b.slot, 0);
        assert_eq!(b.len, 37);
        assert_eq!(b.seg, None);
    }

    #[test]
    fn block_allocator_alloc_all() {
        logger::enable();
        let num = 64;
        let bytes = num * BLOCK_SIZE;
        let mut ba = BlockAllocator::new(bytes);
        assert_eq!(ba.len(), num);
        assert_eq!(ba.freelen(), num);

        let mut count = 0;
        while ba.freelen() > 0 {
            match ba.alloc(2) {
                None => break,
                opt => {
                    for _ in opt.unwrap() {
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
        logger::enable();
        let num = 64;
        let bytes = num * BLOCK_SIZE;
        let mut ba = BlockAllocator::new(bytes);
        let set;
        match ba.alloc(64) {
            None => panic!("Alloc should not fail"),
            opt => set = opt.unwrap(),
        }
        let id = 42;
        let slot = 0;
        let seg = Segment::new(id, slot, set);
        assert_eq!(seg.closed, false);
        assert_eq!(seg.head.is_some(), true);
        assert_eq!(seg.len, bytes);
        // TODO verify blocks?
    }

    // TODO free blocks of segment (e.g. after compaction)

    #[test]
    fn segment_manager_alloc_all() {
        logger::enable();
        let memlen = 1<<23;
        let numseg = memlen / SEGMENT_SIZE;
        let mut mgr = SegmentManager::new(0, SEGMENT_SIZE, memlen);
        for _ in 0..numseg {
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
        logger::enable();
        let memlen = 1<<23;
        let manager = segmgr_ref!(0, SEGMENT_SIZE, memlen);
        let mut log = Log::new(manager.clone());

        let key = String::from("onlyone");
        let val = String::from("valuevaluevalue");
        let obj = ObjDesc::new(key.as_str(),
                        Some(val.as_ptr()), val.len() as u32);
        // fill up the log
        let mut count: usize = 0;
        loop {
            match log.append(&obj) {
                Ok(_) => count += 1,
                Err(code) => match code {
                    ErrorCode::OutOfMemory => break,
                    _ => panic!("filling log returned {:?}", code),
                },
            }
        }
        // FIXME rust complains when we match on the expr directly
        let l = manager.lock();
        match l {
            Ok(mgr) => assert_eq!(mgr.test_scan_objects(), count),
            Err(_) => panic!("lock poison"),
        }
    }

    // TODO entry reference types

    #[test]
    fn iterate_segment() {
        logger::enable();
        let mut rng = rand::thread_rng();

        // TODO make a macro out of these lines
        let memlen = 1<<23;
        let mut mgr = SegmentManager::new(0, SEGMENT_SIZE, memlen);

        let segref = mgr.alloc().unwrap();
        let mut seg = segref.write().unwrap();

        // TODO use crate rand::gen_ascii_chars
        // use this to generate random strings
        // split string literal into something we can index with O(1)
        // do we need a grapheme cluster iterator instead?
        let alpha: Vec<char> =
            "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
            .chars().collect();

        // TODO generate sizes randomly
        // TODO export rand str generation

        let key_sizes: Vec<u32> = vec!(30, 89, 372); // arbitrary
        let value_sizes: Vec<u32> = vec!(433, 884, 511); // arbitrary
        let total: u32 = key_sizes.iter().fold(0, ops::Add::add)
            + value_sizes.iter().fold(0, ops::Add::add);
        let nbatches = (SEGMENT_SIZE - BLOCK_SIZE) / (total as usize);

        // Buffer to receive items into
        let buf: *mut u8 = allocate::<u8>(total as usize);

        // create the key value pairs
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

        // append the objects
        for _ in 0..nbatches {
            for tuple in (&keys).into_iter().zip(&values) {
                let key = tuple.0;
                let value = tuple.1;
                let loc = Some(value.as_ptr());
                let len = value.len() as u32;
                let obj = ObjDesc::new(key.as_str(), loc, len);
                match seg.append(&obj) {
                    Err(code) => panic!("append error:: {:?}", code),
                    _ => {},
                }
            }
        }
        
        // count and verify the segment iterator
        let mut counter: usize = 0;
        for entry in seg.into_iter() {
            let idx = counter % key_sizes.len();
            assert_eq!(entry.keylen, key_sizes[idx]);
            assert_eq!(entry.datalen, value_sizes[idx]);
            assert!(total > entry.datalen);

            // compare the values
            unsafe {
                entry.get_data(buf);
                let nchars = values[idx].len();
                let slice = from_raw_parts(buf, nchars);
                let orig = values[idx].as_bytes();
                assert_eq!(slice, orig);
            }

            counter += 1;
        }
        assert_eq!(counter, nbatches * keys.len());

        unsafe { deallocate::<u8>(buf, total as usize); }
    }

    // TODO test copying out of the iterator
}
