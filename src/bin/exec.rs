#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(unused_imports)]

extern crate nibble;
extern crate log;

use nibble::nibble::Nibble;
use nibble::logger::SimpleLogger;
use log::LogLevel;

fn main() {
    { let _ = SimpleLogger::init(LogLevel::Debug); }
    let nib = Nibble::new(1<<26);
}
