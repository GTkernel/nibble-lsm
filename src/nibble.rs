use libc;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

//pub mod nibble;

const BLOCK_SIZE: usize = 1 << 16;
const SEGMENT_SIZE: usize = 1 << 20;

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

pub struct EntryHeader {
    id: usize,
    len: u32,
}

pub struct Segment {
    closed: bool,
    head: usize, // TODO atomic pointer type
    len: usize,
    blocks: BlockRefPool,
}

impl Segment {

    pub fn new(id: usize, blocks: BlockRefPool) -> Self {
        let mut len: usize = 0;
        for b in blocks.iter() {
            len += b.len;
        }
        Segment {
            closed: false, head: 0,
            len: len, blocks: blocks,
        }
    }

    // TODO increment the head atomically
    pub fn increment(&self, len: usize) {
        unimplemented!();
    }

    // TODO append an object -- maybe make this unsafe?
    pub fn append(&self) {
        unimplemented!();
    }
}

impl Drop for Segment {

    fn drop(&mut self) {
        for b in self.blocks.iter() {
            // TODO push .clone() to BlockAllocator
        }
    }
}

pub type SegmentRef = Option<Arc<RefCell<Segment>>>;

/// Create a new SegmentRef. Same signature as Segment::new
pub fn new_segmentref(id: usize, blocks: BlockRefPool) -> SegmentRef {
    Some(Arc::new(RefCell::new(Segment::new(id, blocks))))
}

pub struct SegmentManager {
    id: usize,
    next_seg_id: usize, // FIXME atomic
    segment_size: usize,
    allocator: BlockAllocator,
    segments: Vec<SegmentRef>,
    free_slots: Vec<u32>,
}

impl SegmentManager {

    pub fn new(id: usize, segsz: usize, len: usize) -> Self {
        let b = BlockAllocator::new(BLOCK_SIZE, len);
        let num = len / segsz;
        let mut v: Vec<SegmentRef> = Vec::new();
        for i in 0..num {
            v.push(None);
        }
        let mut s: Vec<u32> = vec![0; num];
        for i in 0..num {
            s[i] = i as u32;
        }
        SegmentManager {
            id: id,
            next_seg_id: 0,
            segment_size: segsz,
            allocator: b,
            segments: v,
            free_slots: s,
        }
    }

    pub fn alloc(&mut self) -> SegmentRef {
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
            self.segments[slot] = new_segmentref(self.next_seg_id, blocks.unwrap());
            self.segments[slot].clone()
        }
    }

    pub fn free(&self) {
        unimplemented!();
    }

    // Internal functions used for testing

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
        assert_eq!(seg.head, 0);
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
}
