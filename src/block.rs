use std::rc::Rc;
use std::cell::RefCell;

use memutil::MemMap;

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

pub type BlockRef = Option<Rc<RefCell<Block>>>;

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

// -----------------------------------------------
// Test Code
// -----------------------------------------------

#[test]
fn block_allocator() {
    let block_size = 1<<16;
    let alloc_size = 1<<26;
    let num_blocks = alloc_size/block_size;
    let ba = BlockAllocator::new(block_size, alloc_size);
    //ba.alloc(1);
}
