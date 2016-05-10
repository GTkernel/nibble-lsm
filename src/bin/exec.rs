#[macro_use]
extern crate nibble;
#[macro_use]
extern crate log;

use log::LogLevel;
use nibble::compaction::Compactor;
use nibble::nib::Nibble;
use nibble::index::Index;
use nibble::logger::SimpleLogger;
use nibble::segment;
use nibble::segment::SegmentManager;
use std::sync::{Arc,Mutex};
use std::thread;
use std::time::Duration;

fn enable_logging() {
    let _ = SimpleLogger::init(LogLevel::Debug);
}

fn main() {
    enable_logging();

    let mut nib = Nibble::new(1<<23);
    nib.enable_compaction();

    //let manager = segmgr_ref!(0, segment::SEGMENT_SIZE, 1<<23);
    //let index = index_ref!();
    //let mut compactor = Compactor::new(&manager, &index);
    //compactor.spawn();

    // add a bunch of segments and leave the index empty
    // the compactor will think all items are dead!

    let sec = 5;
    info!("Sleeping {} seconds", sec);
    thread::sleep(Duration::from_secs(sec));
}
