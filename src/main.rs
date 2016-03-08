#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(unused_imports)]

extern crate libc;
#[macro_use]
extern crate log;

mod logger;

use std::rc::Rc;
use std::cell::RefCell;

/// Memory mapped region in our address space.
struct MemMap {
    addr: usize,
    len: usize,
}

/// Create anonymous private memory mapped region.
impl MemMap {
    fn new(len: usize) -> Self {
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
}

/// Prevent dangling regions by unmapping it.
impl Drop for MemMap {
    fn drop(&mut self) {
        unsafe {
            let p = self.addr as *mut libc::c_void;
            libc::munmap(p, self.len);
        }
    }
}

type BlockRef = Option<Rc<RefCell<Block>>>;

struct Block {
    addr: usize,
    len: usize,
}

impl Block {
    fn new(addr: usize, len: usize) -> Self {
        debug!("new Block 0x{:x} {}", addr, len);
        Block { addr: addr, len: len }
    }
}

struct BlockAllocator {
    block_size: usize,
    count: usize,
    pool: Vec<Block>,
    mmap: MemMap,
}

impl BlockAllocator {
    fn new(block_size: usize, bytes: usize) -> Self {
        let mmap = MemMap::new(bytes);
        let count = bytes / block_size;
        let mut pool: Vec<Block> = Vec::new();
        for b in 0..count {
            let addr = mmap.addr + b*block_size;
            pool.push(Block::new(addr, block_size));
        }
        BlockAllocator { block_size: block_size,
            count: count, pool: pool, mmap: mmap,
        }
    }
    fn alloc(&self, count: usize) -> Vec<Block> {
        unimplemented!();
    }
}

fn main() {
    { let _ = logger::SimpleLogger::init(); } // ignore return
    let blocksz = 1<<16;
    let len = 1<<20;
    let b = BlockAllocator::new(blocksz, len);
}

// -----------------------------------------------
// Test Code
// -----------------------------------------------

#[cfg(test)]
mod test {
    use super::BlockAllocator;
    use super::MemMap;

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
