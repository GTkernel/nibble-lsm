#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(unused_imports)]

extern crate libc;
#[macro_use]
extern crate log;

mod logger;
mod memutil;

use std::rc::Rc;
use std::cell::RefCell;

use memutil::MemMap;

type BlockRef = Option<Rc<RefCell<Block>>>;

struct Block {
    addr: usize,
    len: usize,
    // TODO owning Segment
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
            let addr = mmap.addr() + b*block_size;
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

    #[test]
    fn block_allocator() {
        let block_size = 1<<16;
        let alloc_size = 1<<26;
        let num_blocks = alloc_size/block_size;
        let ba = BlockAllocator::new(block_size, alloc_size);
        //ba.alloc(1);
    }
}
