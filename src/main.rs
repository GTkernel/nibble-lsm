#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![feature(test)]

#[macro_use]
extern crate log;
extern crate libc;
extern crate test;

mod nibble;
mod logger;

// TODO create thread to hold segment manager

fn main() {
    { let _ = logger::SimpleLogger::init(log::LogLevel::Debug); }
    let segmgr = nibble::SegmentManager::new(0, 1<<20, 1<<23);
}
