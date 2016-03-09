#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(unused_imports)]

#[macro_use]
extern crate log;
extern crate libc;

mod logger;
mod block;
mod memutil;
mod segment;

// FIXME move these into one namespace, perhaps?
// Can this be done despite having separate files?
//use memutil::MemMap;
//use block::BlockAllocator;
use segment::*;
use logger::*;

fn main() {
    { let _ = SimpleLogger::init(); }
    let segmgr = SegmentManager::new(0, 1<<20, 1<<27);
}
