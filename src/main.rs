#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(unused_imports)]

extern crate libc;

//use std::thread;
//use std::mem;
use std::rc::Rc;
use std::cell::RefCell;

struct MmapBlock {
    addr: usize,
    len: usize,
}

// TODO fault on local socket
impl MmapBlock {
    fn new(len: usize) -> Self {
        let prot: libc::c_int = libc::PROT_READ | libc::PROT_WRITE;
        let flags: libc::c_int = libc::MAP_ANON |
            libc::MAP_PRIVATE | libc::MAP_NORESERVE;
        let addr: usize = unsafe {
            let p = 0 as *mut libc::c_void;
            libc::mmap(p, len, prot, flags, 0, 0) as usize
        };
        assert!(addr != libc::MAP_FAILED as usize);
        MmapBlock { addr: addr, len: len }
    }
}

impl Drop for MmapBlock {
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
        Block { addr: addr, len: len }
    }
}

struct BlockAllocator {
    block_size: usize,
    count: usize,
    pool: Vec<Block>,
    //mmap: MmapBlock,
}

impl BlockAllocator {
    fn new(block_size: usize, bytes: usize) -> Self {
        let count = bytes / block_size;
        let mut pool: Vec<Block> = Vec::new();
        for b in 0..count {
            pool.push(Block::new(0, block_size));
        }
        BlockAllocator {
            block_size: block_size,
            count: count,
            pool: pool,
        }
    }
    fn alloc(&self, count: usize) -> Vec<Block> {
        unimplemented!();
    }
}

fn main() {
}

// -----------------------------------------------
// Test Code
// -----------------------------------------------

#[cfg(test)]
mod test {
    use super::BlockAllocator;
    use super::MmapBlock;

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
        let mm = MmapBlock::new(1<<26);
    }

}
