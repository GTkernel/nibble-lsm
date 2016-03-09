#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(unused_imports)]

#[macro_use]
extern crate log;
extern crate libc;

mod nibble;
mod logger;

// TODO create thread to hold segment manager

fn main() {
    { let _ = logger::SimpleLogger::init(); }
    let segmgr = nibble::SegmentManager::new(0, 1<<20, 1<<27);
}
