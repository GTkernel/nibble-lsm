
use common::*;
use thelog::*;
use memory::*;
use meta::*;
use meta;
use numa::{self,NodeId};
use compaction;
use mcs::{McsQnode};
use sched;

use std::cmp;
use std::mem;
use std::mem::size_of;
use std::ptr;
use std::sync::{Arc};
use std::sync::atomic;
use std::sync::atomic::*;
use std::thread;
use std::time::{Instant,Duration};
use std::intrinsics;
use std::fmt;
use std::slice;
use std::collections::VecDeque;

use crossbeam::sync::SegQueue;
use parking_lot as pl;

// If block or segment size > huge page, it must be a multiple of huge page.
pub const BLOCK_SHIFT:      usize = 16;
pub const BLOCK_SIZE:       usize = 1 << BLOCK_SHIFT;
pub const BLOCK_OFF_MASK:   usize = BLOCK_SIZE - 1;
pub const SEGMENT_SHIFT:    usize = 25;
pub const SEGMENT_SIZE:     usize = 1 << SEGMENT_SHIFT;
pub const BLOCKS_PER_SEG:   usize = SEGMENT_SIZE / BLOCK_SIZE;

/// Number of blocks reserved (per BlockAllocator) for use by
/// compaction when memory is scarce.
pub const RESERVE_SEGS:     usize = 128;
pub const RESERVE_BLOCKS:   usize = RESERVE_SEGS * BLOCKS_PER_SEG;

//==----------------------------------------------------==//
//      Utility functions
//==----------------------------------------------------==//

/// Read out data from the log. The caller should check if data can be
/// directly copied out (contiguously) before invoking this.
/// blocks: the set of containing blocks
/// offset: logical from start of first block, but may refer to beyond
/// the first block (e.g. large key or object starts very near the end
/// of the first block)
/// Similar implementation as in Segment::append_entry
/// TODO benchmark this?
/// TODO merge with read_safe?
#[cold]
pub unsafe fn copy_out(blocks: &[BlockRef], offset: usize,
                       out: *mut u8, len: usize) {
    let mut remaining = len as isize;

    // which block does offset put us in?
    let mut idx = offset / BLOCK_SIZE;
    let offset = (offset % BLOCK_SIZE) as isize;

    // Logical offset into new buffer
    let mut poffset: isize = 0;

    // Copy first chunk in current block
    let mut base = blocks[idx].addr as *const u8;
    let mut amt = cmp::min(BLOCK_SIZE as isize - offset, remaining);
    let from = base.offset(offset);
    let to   = out.offset(poffset);
    copy(to, from, amt as usize);

    remaining -= amt;
    poffset += amt;
    idx += 1;

    while remaining > 0 {
        base = blocks[idx].addr as *const u8;
        amt = cmp::min(BLOCK_SIZE as isize, remaining);
        copy(out.offset(poffset), base, amt as usize);
        remaining -= amt;
        poffset += amt;
        idx += 1;
    }
}

// TODO copy_in function?

//==----------------------------------------------------==//
//      Block structures
//==----------------------------------------------------==//


/// A contiguous region of virtual memory.
#[derive(Clone)]
pub struct Block {
    /// Base address of this memory region. Immutable after creation.
    addr: usize,
    /// Length of this block in bytes. Immutable after creation.
    /// TODO get rid of this... all blocks are / should be BLOCK_SIZE
    len: usize,
    /// Index within BlockAllocator::pool. Immutable after creation.
    slot: usize,
    /// When associated with a Segment, this is the slot (index) of
    /// the containing Segment within the SegmentManager. Only load or
    /// store this.
    seg_slot: usize,
    /// When associated with a Segment, this is our index within the
    /// Segment's block set. Only load or store this.
    blk_idx: usize,
    /// Cache of the containing Segment's block list as a slice.
    /// Cannot go stale if epochs are used appropriately, and Segment
    /// data isn't mutated while an epoch is valid, and a Copy of
    /// Block does not exist outside of an epoch window.  This is an
    /// optimization to avoid locks. If this is null, then this Block
    /// does not belong to any Segment (and thus seg_slot and blk_idx
    /// are also invalid). Only load or store this.
    list: uslice<BlockRef>,
}
impl fmt::Debug for Block {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Block {{ addr 0x{:x} len {} slot {} segslot {} blk_idx {} }}",
               self.addr, self.len, self.slot, self.seg_slot, self.blk_idx
               )
    }
}

impl Block {

    pub fn new(addr: usize, slot: usize, len: usize) -> Self {
        trace!("new Block 0x{:x} slot {} len {}", addr, slot, len);
        Block {
            addr: addr, len: len, slot: slot,
            seg_slot: 0, blk_idx: 0,
            list: uslice::null(),
        }
    }

    pub unsafe fn set_segment(&self, seg_slot: usize,
                              blk_idx: usize, list: uslice<BlockRef>) {
        trace!("adding Block {} to Segment {} blk_idx {} list {:?}",
               self.slot, seg_slot, blk_idx, list);
        // XXX potential UB
        let myself = &mut *(self as *const Self as *mut Self);
        myself.seg_slot = seg_slot;
        myself.blk_idx  = blk_idx;
        myself.list     = list;
        atomic::fence(atomic::Ordering::SeqCst);
    }

    pub unsafe fn clear_segment(&self) {
        trace!("removing Block {} from Segment", self.slot);
        // XXX potential UB
        let myself = &mut *(self as *const Self as *mut Self);
        myself.list = uslice::null();
    }

    // Methods are here to prevent someone from doing an increment.
    // You can only load or store.

    pub fn addr(&self)       -> usize { self.addr }
    pub fn len(&self)        -> usize { self.len }
    pub fn slot(&self)       -> usize { self.slot }

    pub fn seg_slot(&self) -> usize {
        assert_eq!(self.list.ptr().is_null(), false,
                    " block seg ({}) asked but ptr is null: {:?}",
                    self.seg_slot, self);
        self.seg_slot
    }

    pub fn blk_idx(&self) -> usize {
        assert_eq!(self.list.ptr().is_null(), false,
                    " block idx ({}) asked but ptr is null: {:?}",
                    self.blk_idx, self);
        self.blk_idx
    }

    pub fn list(&self) -> uslice<BlockRef> {
        self.list.clone()
    }
}

pub type BlockRef = Arc<Block>;
pub type BlockRefPool = Vec<BlockRef>;

//==----------------------------------------------------==//
//      Block allocator
//==----------------------------------------------------==//

pub struct BlockAllocator {
    /// All the blocks
    pool: BlockRefPool,
    mmap: MemMap,
    /// Blocks allowed for general allocation
    freepool: pl::Mutex<BlockRefPool>,
    freepool_sz: AtomicUsize,
    freepool_mcs: AtomicPtr<McsQnode>,
    /// Set of reserve blocks used by compaction
    reserve: pl::RwLock<BlockRefPool>,
    /// Number of blocks that must be in the reserve pool before we
    /// release blocks to the general pool.
    reserve_nblks: usize,
}

impl fmt::Debug for BlockAllocator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BlockAllocator {{ #free {} #res {} }}",
               self.freepool.lock().len(),
               self.reserve.read().len())
    }
}

impl BlockAllocator {

    fn __new(bytes: usize, mmap: MemMap) -> Self {

        let count = bytes / BLOCK_SIZE;
        let mut pool:       Vec<Arc<Block>> =
            Vec::with_capacity(count);
        let mut freepool:   Vec<Arc<Block>> =
            Vec::with_capacity(count);

        let mut reserve:    Vec<Arc<Block>> =
            Vec::with_capacity(RESERVE_BLOCKS);

        let general_blks = count - RESERVE_BLOCKS;

        info!("new BlockAllocator len {} pool {} free {}",
              mmap.len(),
              mem::size_of::<Vec<Arc<Block>>>() * count,
              mem::size_of::<Vec<Arc<Block>>>() * count);

        info!("reserving {} blocks for compaction, {} for logs",
               RESERVE_BLOCKS, general_blks);

        for b in 0..count {
            let addr = mmap.addr() + b*BLOCK_SIZE;
            let blk = Arc::new(Block::new(addr, b, BLOCK_SIZE));
            pool.push(blk.clone());
            if b < general_blks {
                freepool.push(blk.clone());
            } else {
                reserve.push(blk.clone());
            }
        }
        let fpsz = freepool.len();
        BlockAllocator {
            pool: pool,
            mmap: mmap,
            freepool: pl::Mutex::new(freepool),
            freepool_sz: AtomicUsize::new(fpsz),
            freepool_mcs: AtomicPtr::new(0usize as *mut McsQnode),
            reserve:  pl::RwLock::new(reserve),
            reserve_nblks: RESERVE_BLOCKS,
        }
    }

    pub fn numa(bytes: usize, node: NodeId) -> Self {
        let align = cmp::max(numa::PAGE_SIZE_HUGE, BLOCK_SIZE);
        let mmap = MemMap::numa(bytes, node, align, true);
        mmap.exclude_corefile();
        Self::__new(bytes, mmap)
    }

    pub fn new(bytes: usize) -> Self {
        let mmap = MemMap::new(bytes);
        mmap.exclude_corefile();
        Self::__new(bytes, mmap)
    }

    /// Normal allocation path. Used by client / application threads
    /// when the log heads must roll. Each enqueues itself into an MCS
    /// lock to avoid stampeding when there is high contention.
    /// After queuing in line, the holder of the MCS lock will spin on
    /// the externalized size for the freepool, instead of competing
    /// with compaction threads for the freepool mutex.
    /// This method is blocking.
    pub fn alloc(&self, count: usize) -> BlockRefPool {
        let mut slot = McsQnode::new();
        unsafe { McsQnode::lock(&self.freepool_mcs, &mut slot); }
        let mut blks = None;
        while blks.is_none() {
            // spin on the size variable with backoff
            let mut tries = 0usize;
            while self.freepool_sz.load(Ordering::Relaxed) < count {
                tries += 1;
                if tries > 10_000usize {
                    sched::sleep_short();
                    tries = 0usize;
                }
            }
            blks = self.allocp(count);
        }
        unsafe { McsQnode::unlock(&self.freepool_mcs, &mut slot); }
        blks.unwrap()
    }

    /// Priority allocation (does not wait in MCS); directly acquire
    /// the mutex on the freepool.  Used by compaction threads.
    pub fn allocp(&self, count: usize) -> Option<BlockRefPool> {
        let mut blocks: Option<BlockRefPool> = None;
        if self.freepool_sz.load(Ordering::Relaxed) >= count {
            let mut freepool = self.freepool.lock();
            let len = freepool.len();
            if count <= len {
                let blks = freepool.split_off(len-count);
                self.freepool_sz.fetch_sub(blks.len(), Ordering::SeqCst);
                blocks = Some(blks);
            }
        }
        blocks
    }

    pub fn reserve_alloc(&self, count: usize)
        -> Option<BlockRefPool> {

        let mut blocks: Option<BlockRefPool> = None;
        if let Some(mut reserve) = self.reserve.try_write() {
            let len = reserve.len();
            if count <= len {
                debug!("alloc {} resrv blks (of {})", count, len);
                blocks = Some(reserve.split_off(len-count));
            }
        }
        blocks
    }

    /// Release a set of blocks back to the allocator.
    /// Check if the reserve pool needs filling. If so, put new blocks
    /// into that first, then the remainder into the general pool.
    /// Since this method is used by compaction and allocating blocks
    /// lies in the critical path of client threads, we directly lock
    /// the mutex instead of using MCS.
    pub fn free(&self, pool: &mut BlockRefPool) {
        loop {
            if let Some(mut reserve) = self.reserve.try_write() {
                let mut nr = cmp::min(pool.len(),
                self.reserve_nblks - reserve.len());
                if unlikely!(nr > 0) {
                    debug!("releasing {} blks back to reserve", nr);
                    for _ in 0..nr {
                        reserve.push(pool.pop().unwrap());
                    }
                }
                break;
            }
        }
        let nblks = pool.len();
        if nblks > 0 {
            debug!("releasing {} blks back to freepool", nblks);
            self.freepool.lock().append(pool);
            self.freepool_sz.fetch_add(nblks, Ordering::SeqCst);
        }
    }

    pub fn reserve_free(&self, pool: &mut BlockRefPool) {
        debug!("releasing {} blks directly back to reserve", pool.len());
        self.reserve.write().append(pool);
    }

    pub fn len(&self) -> usize {
        self.pool.len()
    }

    pub fn freelen(&self) -> usize {
        //self.freepool.lock().len()
        self.freepool_sz.load(Ordering::Relaxed)
    }

    pub fn freesz(&self) -> usize {
        self.freelen() * BLOCK_SIZE
    }

    /// Convert virtual address to containing block.
    /// We don't check if the index is valid because this function is
    /// meant only for internal use. Any addr that isn't within a
    /// block should never be passed in. Invalid index to pool will
    /// result in a runtime error and halt.
    #[inline(always)]
    pub fn block_of(&self, addr: usize) -> Block {
        let idx: usize = (addr - self.mmap.addr()) >> BLOCK_SHIFT;
        let b: &BlockRef = &self.pool[idx];
        debug_assert!(addr >= b.addr);
        debug_assert!(addr < (b.addr + BLOCK_SIZE));
        (**b).clone()
    }

    #[inline(always)]
    pub fn segment_of(&self, addr: usize) -> usize {
        let idx: usize = (addr - self.mmap.addr()) >> BLOCK_SHIFT;
        let b: &BlockRef = &self.pool[idx];
        debug_assert!(addr >= b.addr);
        debug_assert!(addr < (b.addr + BLOCK_SIZE));
        b.seg_slot()
    }
}

//==----------------------------------------------------==//
//      Object descriptor
//==----------------------------------------------------==//

/// Use this to describe an object. User is responsible for
/// transmuting their objects, and for ensuring the lifetime of the
/// originating buffers exceeds that of an instance of ObjDesc used to
/// refer to them.
#[derive(Debug)]
pub struct ObjDesc {
    pub key: u64,
    pub value: Pointer<u8>,
    pub vlen: usize,
    // false - we do everything but copy the object itself
    pub copy: bool,
}


impl ObjDesc {

    /// Create ObjDesc where key is str and value is arbitrary memory.
    pub fn new(key: u64, value: Pointer<u8>, vlen: usize) -> Self {
        ObjDesc { key: key, value: value, vlen: vlen, copy: true }
    }

    /// Create ObjDesc where key and value are String
    pub fn new2(key: u64, value: &String) -> Self {
        ObjDesc {
            key: key,
            value: Pointer(value.as_ptr()),
            vlen: value.len(),
            copy: true,
        }
    }

    /// Used to implement 'alloc' behavior (as opposed to PUT).
    pub fn null(key: u64, vlen: usize) -> Self {
        ObjDesc {
            key: key,
            value: Pointer(ptr::null::<u8>()),
            vlen: vlen,
            // nothing to copy
            copy: false,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        mem::size_of::<u64>() + self.vlen
    }

    #[inline(always)]
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

    //pub fn getkey(&self) -> &'a str { self.key }
    pub fn getkey(&self) -> u64 { self.key }
    pub fn keylen(&self) -> usize { mem::size_of::<u64>() }
    pub fn getvalue(&self) -> Pointer<u8> { self.value }
    pub fn valuelen(&self) -> usize { self.vlen }
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

pub type SegmentRef = Arc<pl::RwLock<Segment>>;
pub type SegmentManagerRef = Arc<SegmentManager>;

/// A Segment of data in the log, composed of many underlying
/// non-contiguous blocks.
/// TODO id, slot, blocks - do not change after creation.  head, len,
/// rem, nobj, curblk, front - all change while segment is a head.
/// once closed, nothing changes. As a head, we need to allow mutable
/// updates to the latter, but concurrent immutable access to the
/// former.
#[allow(dead_code)]
pub struct Segment {
    id: usize,
    /// The socket this segment is allocated on
    socket: NodeId,
    /// index into segment table TODO add unit test?
    slot: usize,
    closed: bool,
    /// Virtual address of head TODO atomic?
    head: usize,
    /// Total capacity TODO atomic
    len: usize,
    /// Remaining capacity TODO atomic
    rem: usize,
    /// Objects (live or not) appended
    nobj: usize,
    curblk: usize,
    /// A *mut SegmentHeader, but kept as usize because the Rust
    /// compiler does not allow sharing of raw pointers.
    front: usize,
    blocks: BlockRefPool,
}

impl fmt::Debug for Segment {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Seg {{ id {} sock {:?} slot {} closed {:?} len {} rem {} nobj {} blocks {:?} }}",
               self.id, self.socket, self.slot,
               self.closed, self.len, self.rem, self.nobj,
               self.blocks)
    }
}

// TODO create a wrapper type that implements sorting of a segment
// based on specific metrics (if i want to put Segments into multiple
// binary heaps)

/// A logically contiguous chunk of memory. Inside, divided into
/// virtually contiguous blocks. For now, objects cannot be broken
/// across a block boundary.
impl Segment {

    /// Print out the contents of the entire segment to STDOUT
    pub fn rawdump(&self) {
        println!("RAWDUMP {:?}", self);
        for barc in &self.blocks {
            let block = barc.as_ref();
            println!("RAWDUMP {:?}", block);
            let sl: &[u8] = unsafe {
                slice::from_raw_parts(
                    block.addr as *const usize as *const u8,
                    block.len)
            };
            let lines = block.len >> 6usize;
            for l in 0..lines {
                let b: usize = l << 6usize;
                print!("B [{:05}] ", b);
                for s in 0usize..(1usize<<4) {
                    print!("{:02x}{:02x}{:02x}{:02x} ",
                           sl[ b + s * 4 + 0 ],
                           sl[ b + s * 4 + 1 ],
                           sl[ b + s * 4 + 2 ],
                           sl[ b + s * 4 + 3 ]);
                }
                println!("");
            }
        }
    }

    // TODO make blocks a ref
    pub fn new(id: usize, sock: NodeId, slot: usize,
               mut blocks: BlockRefPool) -> Self {
        blocks.reserve(8); // avoid resizing in the future
        debug!("new segment: slot {} #blks {} socket {}",
               slot, blocks.len(), sock);
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
            id: id, socket: sock, slot: slot, closed: false,
            head: start + SegmentHeader::len(),
            len: len, rem: len - SegmentHeader::len(),
            nobj: 0, curblk: blk,
            front: blocks[blk].addr,
            blocks: blocks,
        }
    }

    /// Add blocks to a segment to grow its size. Used by the
    /// compaction code, after allocating an empty segment.
    /// TODO need unit test
    pub fn extend(&mut self, blocks: &mut BlockRefPool) {
        assert!(blocks.len() > 0);

        let addlen = blocks.len() * BLOCK_SIZE;
        self.len += addlen;
        self.rem += addlen;
        debug_assert!(addlen > 0, "addlen is zero");

        // NOTE always push blocks first. All existing blocks must
        // next have their backpointer to the block list updated,
        // because the slice as originally set to N blocks; now
        // we have >N blocks in the segment.
        self.blocks.append(blocks);

        let list = uslice::make(&self.blocks);
        for t in self.blocks.iter().zip(0..) { unsafe {
            t.0 .set_segment(self.slot, t.1, list.clone());
        } }
    }

    /// Append an object with header to the log. Let append_safe
    /// handle the actual work.  If successful, returns virtual
    /// address in Ok().
    /// TODO Segment shouldn't know about log-specific organization,
    /// e.g. entry headers
    pub fn append(&mut self, buf: &ObjDesc) -> Status {
        if unlikely!(self.closed) {
            Err(ErrorCode::SegmentClosed)
        } else if unlikely!(!self.can_hold(buf)) {
            Err(ErrorCode::SegmentFull)
        } else {
            if unlikely!(0 == buf.vlen) {
                return Err(ErrorCode::EmptyObject);
            }
            let va = self.headref() as usize;
            let header = EntryHeader::new(buf);
            let hlen = size_of::<EntryHeader>();
            self.append_safe(header.as_ptr(), hlen);
            let v = &buf.key as *const _ as *const u8;
            self.append_safe(v, mem::size_of::<u64>());
            if likely!(buf.copy) {
                self.append_safe(buf.value.0 as *const u8,
                                 buf.vlen as usize);
            } else {
                self.bump_head(buf.vlen as usize);
            }
            self.nobj += 1;
            self.update_header(1);
            Ok(va)
        }
    }

    /// Bump the head offset for this segment as if we appended
    /// the given bytes (but we do not do any copying). Used to
    /// implement LSM::alloc. append_safe performs the same work,
    /// but incrementally as it copies each piece.
    #[inline]
    fn bump_head(&mut self, len: usize) {
        let remblk = self.rem_in_block();
        if len < remblk {
            self.head += len;
            debug_assert!((self.head -
                self.blocks[self.curblk].addr) < BLOCK_SIZE);
        } else {
            let nblks: usize = 1 + (len - remblk) / BLOCK_SIZE;
            self.next_block(nblks);
            let offset = (len - remblk) % BLOCK_SIZE;
            self.head += offset;
        }
        self.rem -= len;
    }

    /// Copy an entry from one set of blocks to this segment. Used by
    /// the compaction logic. Returns address of relocated entry, or
    /// None if no space is available.  Specify whether upon
    /// expansion, the segment is allowed to allocate additional
    /// blocks.
    /// TODO optimize to know when the blocks are contiguous
    /// TODO optimize for large objects: transfer whole blocks?
    /// TODO use Pointer or something to represent return address
    /// nearly verbatim overlap with EntryReference::copy_out
    pub fn append_entry(&mut self, entry: &EntryReference)
        -> Option<usize> {

        if !self.can_hold_amt(entry.len) {
            debug!("cannot append: need {} have {}",
                   entry.len, self.rem);
            return None;
        }

        let new_va = self.headref() as usize;

        // append by-block since entry may not be virtually contiguous
        let mut va = entry.blocks[0].addr + entry.offset;
        let mut amt = cmp::min(BLOCK_SIZE - entry.offset, entry.len);
        trace!("append_entry len {} #blks {} amt {} va 0x{:x}",
               entry.len, entry.blocks.len(), amt, va);
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

        Some(new_va)
    }

    #[inline(always)]
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    #[inline(always)]
    pub fn close(&mut self) {
        atomic::fence(Ordering::SeqCst); // TODO needed?
        self.closed = true;

    }

    #[inline(always)]
    pub fn nobjects(&self) -> usize { self.nobj }
    #[inline(always)]
    pub fn nblocks(&self) -> usize { self.blocks.len() }
    #[inline(always)]
    pub fn slot(&self) -> usize { self.slot }
    #[inline(always)]
    pub fn len(&self) -> usize { self.len }
    #[inline(always)]
    pub fn socket(&self) -> NodeId { self.socket }
    #[inline(always)]
    pub fn remaining(&self) -> usize { self.rem }
    #[inline(always)]
    pub fn can_hold_amt(&self, len: usize) -> bool { self.rem >= len }
    #[inline(always)]
    pub fn can_hold(&self, buf: &ObjDesc) -> bool {
        self.can_hold_amt(buf.len_with_header())
    }

    /// Increment values in header by specified amount
    #[inline(always)]
    fn update_header(&self, n: u32) {
        let mut header: &mut SegmentHeader = unsafe {
            &mut *(self.front as *mut SegmentHeader)
        };
        header.num_objects += n;
    }

    /// Increment the head offset into the start of the next block.
    /// Return false if we cannot roll because we're already at the
    /// final block. 'n' specifies how many blocks to increment by.
    #[inline]
    fn next_block(&mut self, n: usize) -> bool {
        let mut curblk = self.curblk;
        let numblks: usize = self.blocks.len();
        debug_assert!(numblks > 0);
        if (curblk + n) < numblks {
            curblk += n;
            self.curblk = curblk;
            self.head = self.blocks[curblk].addr;
            true
        } else {
            // this can happen if the object fits _exactly_ into the
            // remainder space of a segment - it's ok, we just ignore
            // this request
            false
        }
    }

    #[inline]
    pub fn headref(&mut self) -> *mut u8 {
        self.head as *mut u8
    }

    //
    // --- Private methods ---
    //

    /// Read out a set of bytes, safely handling block boundaries.
    /// Same as append_safe but copies out, instead. Returns the VA
    /// just past the data we copied out (in case we traveled to a
    /// subsequent block, it returns its starting address).
    /// TODO merge into copy_out? or replace it?
    unsafe fn read_safe(&self, blk_idx: usize, offset: usize,
                            dst: *mut u8, len: usize) -> usize {
        // Perform 2+ copies across blocks if needed
        let mut curblk = blk_idx;
        let mut loc = (self.blocks[blk_idx].addr+offset) as *const u8;
        let mut copied = 0_usize;
        let mut amt: usize;
        loop {
            let in_blk = BLOCK_SIZE - (loc as usize & (BLOCK_SIZE-1));
            amt = cmp::min(in_blk, len - copied);
            copy(dst.offset(copied as isize), loc, amt);
            copied += amt;
            if copied >= len {
                break;
            }
            curblk += 1;
            loc = self.blocks[curblk].addr as *const u8;
        }

        let mut next = loc as usize + amt;

        // if we wrote to the end of a block, next address is actually
        // the next block itself
        if 0 == (next & (BLOCK_SIZE-1)) {
            next = self.blocks[curblk+1].addr;
        }

        next
    }

    /// Determine which block holds this address.
    /// TODO optimize with an address range to block mapping
    #[cfg(IGNORE)]
    fn block_of(&self, va: usize) -> Option<usize> {
        for tuple in self.blocks.iter().zip(0..) {
            let start = tuple.0 .addr;
            let end   = start + tuple.0 .len;
            if start <= va && va < end {
                return Some(tuple.1);
            }
        }
        None
    }

    /// Append some buffer safely across block boundaries (if needed).
    /// Caller must ensure the containing segment has sufficient
    /// capacity.
    #[inline]
    fn append_safe(&mut self, from: *const u8, len: usize) {
        assert!(len <= self.rem);
        let mut remblk = self.rem_in_block();
        // 1. If buffer fits in remainder of block, just copy it
        if len <= remblk {
            unsafe {
                copy(self.headref(),from,len);
            }
            self.head += len;
            if len == remblk {
                self.next_block(1);
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
                    copy(self.headref(),loc, amt);
                }
                self.head += amt;
                rem -= amt;
                loc = (from as usize + amt) as *const u8;
                // If we exceeded the block, get the next one
                if remblk == amt {
                    self.next_block(1);
                }
            }
        }
        self.rem -= len;
    }

    fn rem_in_block(&self) -> usize {
        debug_assert!(self.blocks.len() > 0);
        let blk = &self.blocks[self.curblk];
        blk.len - (self.head - blk.addr)
    }

    //
    // --- Test methods ---
    //

    #[cfg(test)]
    pub fn rem(&self) -> usize { self.rem }

    #[cfg(test)]
    pub fn used(&self) -> usize { self.len - self.rem }

    #[cfg(test)]
    pub fn head(&self) -> usize { self.head }

    #[cfg(test)]
    pub fn curblk(&self) -> usize { self.curblk }

    #[cfg(test)]
    pub fn nblks(&self) -> usize { self.blocks.len() }

    #[cfg(test)]
    pub fn reset(&mut self) {
        let blk = 0;
        self.closed = false;
        self.head = self.blocks[blk].addr;
        self.rem = self.len - SegmentHeader::len();
        self.curblk = blk;
        self.nobj = 0;
    }
}

impl Drop for Segment {

    fn drop(&mut self) {
        debug!("segment slot {} dropped", self.slot);
    }
}

impl<'a> IntoIterator for &'a Segment {
    type Item = EntryReference<'a>;
    type IntoIter = SegmentIter<'a>;

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
pub struct SegmentIter<'a> {
    n_obj: usize,
    blk_offset: usize,
    cur_blk: usize,
    seg_offset: usize,
    next_obj: usize,
    blocks: &'a [BlockRef],
}

impl<'a> SegmentIter<'a> {

    pub fn new(seg: &'a Segment) -> Self {
        let offset = SegmentHeader::len();
        SegmentIter {
            n_obj: seg.nobj,
            blk_offset: offset,
            cur_blk: 0,
            seg_offset: offset,
            next_obj: 0,
            blocks: seg.blocks.as_slice(),
        }
    }
}

impl<'a> Iterator for SegmentIter<'a> {
    type Item = EntryReference<'a>;

    fn next(&mut self) -> Option<EntryReference<'a>> {
        if self.next_obj >= self.n_obj {
            return None;
        }

        // don't advance if we're at first entry
        if self.next_obj > 0 {
            // read length of current entry
            let mut entry = EntryHeader::empty();;
            unsafe {
                copy_out(self.blocks,
                         self.seg_offset,
                         entry.as_mut_ptr(),
                         mem::size_of::<EntryHeader>());
            }

            debug_assert!(entry.getdatalen() > 0);
            // https://github.com/rust-lang/rust/issues/22644
            debug_assert!( (entry.getdatalen() as usize) < SEGMENT_SIZE);

            // advance to next
            self.seg_offset += entry.len();
            self.cur_blk = self.seg_offset / BLOCK_SIZE;
            self.blk_offset = self.seg_offset % BLOCK_SIZE;
        }

        // read entry info
        trace!("cur_blk {} addr 0x{:x} offset {} seg offset {}",
               self.cur_blk, self.blocks[self.cur_blk].addr,
               self.blk_offset, self.seg_offset);
        let addr = self.blocks[self.cur_blk].addr + self.blk_offset;
        trace!("reading EntryHeader from 0x{:x}", addr);
        let mut entry = EntryHeader::empty();;
        unsafe {
            copy_out(self.blocks,
                     self.seg_offset,
                     entry.as_mut_ptr(),
                     mem::size_of::<EntryHeader>());
        }

        debug_assert!(entry.getdatalen() > 0);
        // https://github.com/rust-lang/rust/issues/22644
        debug_assert!( (entry.getdatalen() as usize) < SEGMENT_SIZE);

        let entry_len = entry.len_with_header();
        trace!("read {:?}", entry);


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

        let last_blk = cmp::min(self.cur_blk + nblks + 2, self.blocks.len());
        let entry = EntryReference {
            offset: self.blk_offset,
            len: entry_len,
            datalen: entry.getdatalen(),
            blocks: &self.blocks[self.cur_blk..last_blk],
        };
        trace!("entry {:?}", entry);
        Some(entry)
    }
}

//==----------------------------------------------------==//
//      Segment manager
//==----------------------------------------------------==//

/// Per-socket manager of segments and blocks.
#[allow(dead_code)]
pub struct SegmentManager {
    /// Socket we are bound to.
    socket: Option<NodeId>,
    /// Total memory
    size: usize,
    /// Each segment gets a new ID (but slots are reused).
    next_seg_id: atomic::AtomicUsize,
    seginfo: SegmentInfoTableRef,
    /// The lower-level memory allocator we use.
    allocator: BlockAllocator,
    /// Segment slots that are unused.
    free_slots: Arc<SegQueue<u32>>,
    /// The set of Segments. Any entry which is None is available to
    /// allocate from.
    segments: pl::RwLock<Vec<Option<SegmentRef>>>,
    /// Vec to sort newly closed segments by compaction thread. Inner
    /// VecDeque is the set of segments for the associated thread.
    closed:   pl::RwLock<Vec<VecDeque<SegmentRef>>>,
    /// Used to distribute newly closed segments among the closed
    /// queues.
    next:     AtomicUsize,
    /// Blocks to free but refs not yet released... broken epochs?
    pending: pl::Mutex<VecDeque<SegmentRef>>,
}

// TODO reclaim segments function and thread
impl SegmentManager {

    #[cfg(IGNORE)]
    pub fn dump_segments(&self) {
        let inner = self.inner.read();
        println!("SEGMGR: there are {} segments",
                 inner.segments.len());
        println!("SEGMGR: {} recently closed",
                 inner.closed.len());
        for tuple in (&inner.segments).iter().zip(0..) {
            let opt = tuple.0;
            if opt.is_none() {
                println!("[{:5}] None", tuple.1);
            } else {
                // opt is Option<RwLock<SegmentRef>>
                let seg = opt.as_ref().unwrap().read();
                println!("[{:5}] {:?}", tuple.1, &*seg);
            }
        }
        println!("{:?}", self.allocator);
    }

    fn __new(sock: Option<NodeId>, segsz: usize, len: usize,
             b: BlockAllocator) -> Self {
        info!("SEGMENT_SIZE {} BLOCK_SIZE {} RESERVE_SEGS {}",
              SEGMENT_SIZE, BLOCK_SIZE, RESERVE_SEGS);
        let num = 2 * len / segsz;
        let mut segments: Vec<Option<SegmentRef>>
            = Vec::with_capacity(num);
        info!("SegmentManager {} vector len: {}",
              sock.unwrap().0,
              mem::size_of::<Vec<Option<SegmentRef>>>() *
              mem::size_of::<Option<SegmentRef>>() *
              num);
        for _ in 0..num {
            segments.push(None);
        }
        let free_slots: SegQueue<u32> = SegQueue::new();
        for i in 0..num {
            free_slots.push(i as u32);
        }
        let mut closed: Vec<VecDeque<SegmentRef>> =
            Vec::with_capacity(compaction::WTHREADS);
        for _ in 0..compaction::WTHREADS {
            closed.push(VecDeque::with_capacity(32));
        }
        info!("SegmentManager {} free_slots len {}",
              sock.unwrap().0,
              mem::size_of::<SegQueue<u32>>() *
              mem::size_of::<u32>() * num);
        SegmentManager {
            socket: sock,
            size: len,
            next_seg_id: atomic::AtomicUsize::new(0),
            allocator: b,
            seginfo: Arc::new(SegmentInfoTable::new(num)),
            free_slots: Arc::new(free_slots),
            segments: pl::RwLock::new(segments),
            closed: pl::RwLock::new(closed),
            next: AtomicUsize::new(0),
            pending: pl::Mutex::new(VecDeque::new()),
        }
    }

    pub fn numa(segsz: usize, len: usize, node: NodeId) -> Self {
        let b = BlockAllocator::numa(len, node);
        Self::__new(Some(node),segsz,len,b)
    }

    pub fn new(segsz: usize, len: usize) -> Self {
        let b = BlockAllocator::new(len);
        Self::__new(None,segsz,len,b)
    }

    pub fn get_nseg(&self) -> usize {
        self.segments.read().len()
    }

    /// Make a Copy of the Block containing this VA (not cloning the
    /// Arc holding the Block).
    #[inline(always)]
    pub fn block_of(&self, va: usize) -> Block {
        self.allocator.block_of(va)
    }

    fn make_segment(&self, mut blocks: Vec<BlockRef>)
        -> Option<SegmentRef> {

        let slot;
        match self.free_slots.try_pop() {
            None => return None,
            Some(s) => slot = s as usize,
        }
        debug_assert!(self.segments.read()[slot].is_none());

        let list = uslice::make(&blocks);
        for t in (0..).zip(&blocks) {
            unsafe {
                t.1 .set_segment(slot, t.0, list.clone());
            }
        }
        let id = self.next_seg_id.fetch_add(1,
                            atomic::Ordering::Relaxed);
        // Check we are not resizing, to avoid updating the
        // Block::list pointer when we add new elements
        debug_assert_eq!(list.ptr(), blocks.as_ptr(),
            "segment {} block list resized!", slot);

        let seg = seg_ref!(id, self.socket.unwrap(), slot, blocks);

        // indicate when we created this segment
        self.seginfo.reset_epoch(slot);

        // store it into the global array
        let mut segments = self.segments.write();
        segments[slot] = Some(seg);

        segments[slot].clone()
    }

    fn do_alloc_size(&self, nblks: usize, mcs: bool)
        -> Option<SegmentRef> {

        let mut ret: Option<SegmentRef> = None;

        let mut blocks: BlockRefPool =
            if mcs {
                self.allocator.alloc(nblks)
            } else {
                match self.allocator.allocp(nblks) {
                    None => return None,
                    Some(b) => b,
                }
            };

        // some extra in case of overflow
        blocks.reserve(8);

        let opt = self.make_segment(blocks);
        assert!(opt.is_some(), "no segment slots available?");
        opt
    }

    /// Allocate a segment with a specific number of blocks.
    /// Uses the MCS path for allocating from the underlying
    /// BlockAllocator (intended for use by client threads).
    pub fn alloc_size(&self, nblks: usize) -> Option<SegmentRef> {
        self.do_alloc_size(nblks, true)
    }

    /// Allocate a segment with a specific number of blocks.
    /// Uses the "fast path" for allocating from the underlying
    /// BlockAllocator; avoids the MCS lock. Intended for use by
    /// compaction threads.
    pub fn alloc_sizep(&self, nblks: usize) -> Option<SegmentRef> {
        self.do_alloc_size(nblks, false)
    }

    /// Allocate a segment with default size.
    #[cfg(IGNORE)]
    pub fn alloc(&self) -> Option<SegmentRef> {
        let nblks = (SEGMENT_SIZE - 1) / BLOCK_SIZE + 1;
        self.alloc_size(nblks)
    }

    pub fn reserve_alloc(&self, nblks: usize) -> Option<SegmentRef> {
        let mut ret: Option<SegmentRef> = None;

        let mut blocks = match self.allocator.reserve_alloc(nblks) {
            None => return None,
            Some(b) => b,
        };
        // some extra in case of overflow
        blocks.reserve(8);
        debug!("alloc {} resrv blks into seg", blocks.len());

        let opt = self.make_segment(blocks);
        assert!(opt.is_some(), "no segment slots available?");
        opt
    }

    fn do_free(&self, segref: SegmentRef) {
        let mut seg = match Arc::try_unwrap(segref) {
            Err(_) => panic!("another ref to seg exists"),
            Ok(seglock) => seglock.into_inner(),
        };
        for b in &seg.blocks {
            unsafe { b.clear_segment(); }
        }
        atomic::fence(atomic::Ordering::SeqCst);
        let slot = seg.slot;

        // release the blocks
        self.allocator.free(&mut seg.blocks);

        // this seg should have zero references at this point
        // do this last because it opens the slot above for use
        self.free_slots.push(slot as u32);
    }

    fn try_release_pending(&self) {
        if let Some(mut pending) = self.pending.try_lock() {
            let many = pending.len();
            for _ in 0..many {
                let s = pending.pop_front().unwrap();
                let c = Arc::strong_count(&s);
                if c == 1 {
                    info!("dropping pending seg: freeing");
                    self.do_free(s);
                } else {
                    info!("dropping pending seg: try later");
                    pending.push_back(s);
                }
            }
        }
    }

    // Consume the reference and release its blocks
    pub fn free(&self, segref: SegmentRef) {
        self.try_release_pending();

        // drop only other known ref to segment
        self.segments.write()[segref.read().slot] = None;

        // extract the segment (should only be one ref remaining)
        let c = Arc::strong_count(&segref);
        if unlikely!(c > 1) {
            {
                let s = segref.read();
                warn!("seg {} has {} other refs",
                    s.slot(), c-1);
            }
            self.pending.lock().push_back(segref);
            return;
        }

        self.do_free(segref);
    }

    /// Directly allocate raw blocks without a containing segment.
    /// Called by compaction only.
    pub fn alloc_blocks(&self, count: usize)
        -> Option<BlockRefPool> {
        debug!("allocating {} blocks out-of-band", count);
        // If, during compaction, allocation fails, we are probably in
        // a dire situation; fall back to reserve blocks. When we do
        // so, it is likely compaction threads have already drawn
        // blocks from the reserve pool by this point.
        let mut opt = self.allocator.allocp(count);
        if let None = opt {
            opt = self.allocator.reserve_alloc(count);
            if unlikely!(opt.is_none()) {
                self.try_release_pending();
                let ms = unsafe { rdrand() % 500 } as u64 + 100u64;
                debug!("Reserve pool empty sock {}. Pausing for {} ms",
                      self.socket.unwrap().0, ms);
                let dur = Duration::from_millis(ms);
                thread::sleep(dur);
            }
        }
        opt
    }

    #[inline(always)]
    pub fn segment_of(&self, va: usize) -> usize {
        self.allocator.segment_of(va)
    }

    /// Construct an EntryReference to the object pointed to by 'va'.
    /// If va is bogus, return None. Remember to pin your epoch, else
    /// your location may be updated underneath you by compaction.
    #[cfg(IGNORE)]
    pub fn get_entry_ref(&self, va: usize) -> Option<EntryReference> {
        let idx = match self.segment_of(va) {
            None => return None,
            Some(idx) => idx,
        };
        //debug_assert_eq!(self.segments[idx].is_some(), true);
        let seg = self.segments[idx].as_ref().unwrap();
        // XXX avoid locking!!
        let guard = seg.read();
        let mut entry = guard.get_entry_ref(va);
        Some(entry)
    }

    /// Log heads pass segments here when it rolls over. Compaction
    /// threads will periodically query this queue for new segments to
    /// add to its candidate list.
    pub fn add_closed(&self, seg: &SegmentRef) {
        let order = Ordering::Relaxed;
        let mut closed = self.closed.write();
        let mut n;
        loop {
            n = self.next.load(Ordering::Relaxed);
            let m = (n + 1) % compaction::WTHREADS;
            if n == self.next.compare_and_swap(n, m, order) {
                break;
            }
        }
        closed[n].push_back(seg.clone());
    }

    // TODO if using SegQueue, perhaps add max to pop off
    pub fn grab_closed(&self, i: usize, to: &mut Vec<SegmentRef>)
        -> usize {

        let mut closed = self.closed.write();
        let n = closed[i].len();
        trace!("grabbing {} candidates for {}", n, i);
        for s in closed[i].drain(..) {
            to.push(s);
        }
        n
    }

    pub fn seginfo(&self) -> SegmentInfoTableRef {
        self.seginfo.clone()
    }

    pub fn socket(&self) -> Option<NodeId> {
        self.socket
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn freesz(&self) -> usize {
        self.allocator.freesz()
    }

    // hack
    #[cfg(IGNORE)]
    pub fn dump_seg_info(&self) {
        for opt in self.segments.iter() {
            if let Some(ref segref) = *opt {
                let seg = segref.read();
                println!("socket {:?} seg {} rem {} len {}",
                         self.socket, seg.slot(),
                         seg.remaining(), seg.len());
            }
        }
    }

    //
    // --- Internal methods used for testing only ---
    //

    #[cfg(IGNORE)]
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

    #[cfg(IGNORE)]
    #[cfg(test)]
    pub fn test_scan_objects(&self) -> usize {
        let mut count: usize = 0;
        for opt in &self.segments {
            match *opt {
                None => {},
                Some(ref segref) => {
                    let seg = segref.read();
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
    use std::mem::size_of;

    use thelog::*;
    use common::*;
    use memory::*;
    use numa::NodeId;

    use rand;
    use rand::Rng;

    use super::super::logger;

    #[test]
    #[cfg(IGNORE)]
    fn block() {
        logger::enable();
        let b = Block::new(42, 0, 37);
        assert_eq!(b.addr, 42);
        assert_eq!(b.slot, 0);
        assert_eq!(b.len, 37);
        assert_eq!(b.seg, None);
    }

    #[test]
    #[cfg(IGNORE)]
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
    #[cfg(IGNORE)]
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
        assert_eq!(seg.len, bytes);
        // TODO verify blocks?
    }

    // TODO free blocks of segment (e.g. after compaction)

    #[test]
    #[cfg(IGNORE)]
    fn segment_manager_alloc_all() {
        logger::enable();
        let memlen = 1<<30;
        let numseg = memlen / SEGMENT_SIZE;
        let mut mgr = SegmentManager::new(SEGMENT_SIZE, memlen);
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
    #[cfg(IGNORE)]
    fn segment_manager_one_obj_overwrite() {
        logger::enable();
        let memlen = 1<<27;
        let manager = segmgr_ref!(SEGMENT_SIZE, memlen);
        let log = Log::new(manager.clone());

        let key = String::from("onlyone");
        let mut val = String::from("valuevaluevalue");
        for _ in 0..200 {
            val.push_str("valuevaluevaluevaluevalue");
        }
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
        let memlen = 1<<30;
        let mut mgr = SegmentManager::numa(SEGMENT_SIZE, memlen, NodeId(0));

        let segref = mgr.alloc().unwrap();
        let mut seg = segref.write();

        // TODO generate sizes randomly
        // TODO export rand str generation

        // TODO use crate rand::gen_ascii_chars
        // use this to generate random strings
        // split string literal into something we can index with O(1)
        // do we need a grapheme cluster iterator instead?
        let alpha: Vec<char> =
            "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
            .chars().collect();

        let value_sizes: Vec<u32> = vec!(4337, 511, 997, 11);
        let total: u32 = value_sizes.iter().fold(0, ops::Add::add)
            + (value_sizes.len() * size_of::<u64>()) as u32;
        let nbatches = (SEGMENT_SIZE/2) / (total as usize);

        // Buffer to receive items into
        let buf: *mut u8 = allocate::<u8>(total as usize);

        // create the values
        let mut values: Vec<String> = Vec::new();
        for size in &value_sizes {
            let mut s = String::with_capacity(*size as usize);
            for _ in 0..*size {
                let r = rng.gen::<usize>() % alpha.len();
                s.push( alpha[ r ] );
            }
            values.push(s);
        }

        let mut counter: usize = 0;

        // append the objects
        for _ in 0..nbatches {
            for value in &values {
                let key = counter as u64;
                let obj = ObjDesc::new2(key, value);
                match seg.append(&obj) {
                    Err(code) => panic!("append error:: {:?}", code),
                    _ => {},
                }
                counter += 1;
            }
        }
        
        // count and verify the segment iterator
        debug!("iterating now");
        counter = 0;
        for entry in seg.into_iter() {
            let idx = counter % value_sizes.len();
            debug!("entry {:?}", entry);
            assert_eq!(entry.keylen as usize, size_of::<u64>());
            assert_eq!(entry.datalen, value_sizes[idx]);

            // compare the keys
            unsafe {
                let k = entry.get_key();
                assert!(k == counter as u64,
                        "keys not equal, k {} counter {}",k,counter);
            }

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
        assert_eq!(counter, nbatches * values.len());

        unsafe { deallocate::<u8>(buf, total as usize); }
    }

    // TODO test copying out of the iterator
}
