use common::*;
use thelog::*;
use memory::*;

use std::cell::RefCell;
use std::mem::size_of;
use std::ptr;
use std::ptr::copy;
use std::ptr::copy_nonoverlapping;
use std::sync::Arc;
use std::mem::transmute;
use std::cmp;

pub const BLOCK_SIZE: usize = 1 << 16;
pub const SEGMENT_SIZE: usize = 1 << 20;

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
#[macro_export]
macro_rules! segmgr_ref {
    ( $id:expr, $segsz:expr, $blocks:expr ) => {
        Arc::new( RefCell::new(
                SegmentManager::new( $id, $segsz, $blocks)
                ))
    }
}

/// Use this to describe an object. User is responsible for
/// transmuting their objects, and for ensuring the lifetime of the
/// originating buffers exceeds that of an instance of ObjDesc used to
/// refer to them.
pub struct ObjDesc<'a> {
    key: &'a str,
    value: Option<*const u8>, // user buffer, or allocated internally
    vlen: u32,
}

impl<'a> ObjDesc<'a> {

    pub fn new(key: &'a str, value: Option<*const u8>, vlen: u32) -> Self {
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
    pub fn getvalue(&self) -> Option<*const u8> { self.value }
    pub fn valuelen(&self) -> u32 { self.vlen }
}

pub struct Segment {
    id: usize,
    closed: bool,
    head: usize, /// Virtual address of head TODO atomic
    len: usize, /// Total capacity TODO atomic
    rem: usize, /// Remaining capacity TODO atomic
    curblk: usize,
    blocks: BlockRefPool,
}

/// A logically contiguous chunk of memory. Inside, divided into
/// virtually contiguous blocks. For now, objects cannot be broken
/// across a block boundary.
impl Segment {

    pub fn new(id: usize, blocks: BlockRefPool) -> Self {
        let mut len: usize = 0;
        for b in blocks.iter() {
            len += b.len;
        }
        let blk = 0;
        let start: usize = blocks[blk].addr;
        Segment {
            id: id, closed: false, head: start,
            len: len, rem: len, curblk: blk, blocks: blocks,
        }
    }

    /// Append an object with header to the log. Let append_safe
    /// handle the actual work.  If successful, returns virtual
    /// address in Ok().
    pub fn append(&mut self, buf: &ObjDesc) -> Status {
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
                Ok(va)
            } else { Err(ErrorCode::SegmentFull) }
        } else { Err(ErrorCode::SegmentClosed) }
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    /// Increment the head offset into the start of the next block.
    /// Return false if we cannot roll because we're already at the
    /// final block.
    fn next_block(&mut self) -> bool {
        let numblks: usize = self.blocks.len();
        if self.curblk < (numblks - 1) {
            self.curblk += 1;
            self.head = self.blocks[self.curblk].addr;
            true
        } else {
            warn!("block roll asked on last block");
            false
        }
    }

    /// Scan segment for all live entries (deep scan).
    pub fn num_live(&self) -> usize {
        let mut count: usize = 0;
        let mut header = EntryHeader::empty();
        let mut offset: usize = 0; // offset into segment (logical)
        let mut curblk: usize = 0; // Block that offset refers to
        // Manually cross block boundaries searching for entries.
        while offset < self.len {
            assert!(curblk < self.blocks.len());
            let base: usize = self.blocks[curblk].addr;
            let addr: usize = base + (offset % BLOCK_SIZE);

            let rem = BLOCK_SIZE - (offset % BLOCK_SIZE); // remaining in block

            // If header is split across block boundaries
            if rem < size_of::<EntryHeader>() {
                // Can't be last block if header is partial
                assert!(curblk != self.blocks.len()-1);
                unsafe {
                    let mut from = addr;
                    let mut to: usize = transmute(&header);
                    copy(from as *const u8, to as *mut u8, rem);
                    from = self.blocks[curblk+1].addr; // start at next block
                    to += rem;
                    copy(from as *const u8, to as *mut u8,
                         size_of::<EntryHeader>() - rem);
                }
            }
            // Header is fully contained in the block
            else {
                unsafe { header.read(addr); }
            }

            // Count it or stop scanning
            match header.status() {
                EntryHeaderStatus::Live => count += 1,
                EntryHeaderStatus::Defunct => {},
                EntryHeaderStatus::Invalid => break,
            }

            // Determine next location to jump to
            offset += header.len();
            curblk = offset / BLOCK_SIZE;
        }
        count
    }

    pub fn can_hold(&self, buf: &ObjDesc) -> bool {
        self.rem >= buf.len_with_header()
    }

    //
    // --- Private methods ---
    //

    fn headref(&mut self) -> *mut u8 {
        self.head as *mut u8
    }

    /// Append some buffer safely across block boundaries (if needed).
    /// Caller must ensure the containing segment has sufficient
    /// capacity.
    fn append_safe(&mut self, from: *const u8, len: usize) {
        assert!(len <= self.rem);
        let mut remblk = self.rem_in_block();
        // 1. If buffer fits in remainder of block, just copy it
        if len <= remblk {
            unsafe {
                copy_nonoverlapping(from,self.headref(),len);
            }
            self.head += len;
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
                self.head += amt;
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
        let blk = &self.blocks[self.curblk];
        blk.len - (self.head - blk.addr)
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

    /// Returns #live objects
    #[cfg(test)]
    pub fn test_count_live_objects(&self) -> usize {
        let mut count: usize = 0;
        for opt in &self.segments {
            match *opt {
                None => {},
                Some(ref seg) => {
                    count += seg.borrow().num_live();
                },
            }
        }
        count
    }
}

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
        let mut old_va: usize = 0; // address of prior object
        // fill up the log
        let mut count: usize = 0;
        loop {
            match log.append(&obj) {
                Ok(va) => {
                    count += 1;
                    // we must manually nix the old object (the Nibble
                    // class would otherwise handle this for us)
                    if old_va > 0 {
                        unsafe { EntryHeader::invalidate(old_va); }
                    }
                    old_va = va;
                },
                Err(code) => match code {
                    ErrorCode::OutOfMemory => break,
                    _ => panic!("filling log returned {:?}", code),
                },
            }
        }
        assert_eq!(manager.borrow().test_count_live_objects(), 1);
    }
}
