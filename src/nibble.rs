use libc;
use std::cell::RefCell;
use std::rc::Rc;

//pub mod nibble;

// -------------------------------------------------------------------
// Block strutures
// -------------------------------------------------------------------

pub type BlockRef = Option<Rc<RefCell<Block>>>;

pub struct Block {
    addr: usize,
    len: usize,
    // TODO owning Segment
}

pub struct BlockAllocator {
    block_size: usize,
    count: usize,
    pool: Vec<Block>,
    mmap: MemMap,
}

impl Block {
    pub fn new(addr: usize, len: usize) -> Self {
        debug!("new Block 0x{:x} {}", addr, len);
        Block { addr: addr, len: len }
    }
}

impl BlockAllocator {
    pub fn new(block_size: usize, bytes: usize) -> Self {
        let mmap = MemMap::new(bytes);
        let count = bytes / block_size;
        let mut pool: Vec<Block> = Vec::new();
        for b in 0..count {
            let addr = mmap.addr() + b*block_size;
            pool.push(Block::new(addr, block_size));
        }
        BlockAllocator { block_size: block_size,
            count: count, pool: pool, mmap: mmap,
        }
    }
    // XXX when a block is free'd we must release from segment to here
    pub fn alloc(&self, count: usize) -> Vec<Block> {
        unimplemented!();
    }
}

// -------------------------------------------------------------------
// Segment structures
// -------------------------------------------------------------------

/// Identifying information for a segment
pub struct Header {
    id: usize,
}

/// Logical representation of a segment, composed of multiple smaller
/// blocks.
pub struct Segment {
    // Note: headers are written to the buffer, not necessarily kept in
    // the metadata structure here
    header: Header,
}

// TODO Segment state as enum

/// Segment manager for a set of physical memory
pub struct SegmentManager<'a> {
    manager_id: usize,
    segment_size: usize,
    allocator: BlockAllocator,
    segments: Vec<&'a mut Segment>,
}

impl Segment {
    pub fn new() -> Self {
        Segment { header: Header { id: 0 }, }
    }
}

// TODO use some config structure
impl<'a> SegmentManager<'a> {
    // TODO socket to initialize on
    pub fn new(id: usize, segsz: usize, len: usize) -> Self {
        let b = BlockAllocator::new(1<<16, len);
        SegmentManager {
            manager_id: id,
            segment_size: segsz,
            allocator: b,
            segments: Vec::new(),
        }
    }
    pub fn alloc(&self) -> Option<Segment> {
        unimplemented!();
    }
    pub fn free(&self) {
        unimplemented!();
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

    #[test]
    fn block_allocator() {
        let block_size = 1<<16;
        let alloc_size = 1<<26;
        let num_blocks = alloc_size/block_size;
        let ba = BlockAllocator::new(block_size, alloc_size);
        //ba.alloc(1);
    }

    #[test]
    fn map() {
        let len = 1<<26;
        let mm = MemMap::new(len);
        assert_eq!(mm.len, len);
        assert!(mm.addr != 0 as usize);
        // TODO touch the memory somehow
        // TODO verify mmap region is unmapped
    }
}
