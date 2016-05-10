#![feature(test)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]

extern crate rand; // import before nibble
#[macro_use]
extern crate log;
extern crate test;
extern crate time;

extern crate nibble;

use std::time::Duration;
use std::thread;
use rand::Rng;
use log::LogLevel;

use nibble::store::Nibble;
use nibble::segment::ObjDesc;
use nibble::logger::*;
use nibble::common::ErrorCode;

// TODO test objects larger than block, and segment
// TODO put_object which must traverse chunks
// TODO a get_object which must traverse chunks
// TODO test we can determine live vs dead entries in segment
// TODO test specific cases where header cross block boundaries

/// Allocate a bunch of objects, free some, observe compactor.
/// We assume compactor runs continuously.
fn alloc_free(pct_to_free: f32) {
    assert!(pct_to_free <= 1.0);
    assert!(pct_to_free >= 0.0);

    let _ = SimpleLogger::init(LogLevel::Debug);

    let mut nib = Nibble::new(1<<23);

    nib.enable_compaction();
    thread::yield_now();

    let mut allkeys: Vec<String> = Vec::new();

    // allocate until full
    let mut counter: usize = 0;
    let mut rng = rand::thread_rng();
    let value = rng.gen_ascii_chars().take(1000).collect();
    info!("inserting objects until full");
    loop {
        let key = counter.to_string();
        {
            let obj = ObjDesc::new2(&key, &value);
            if let Err(code) = nib.put_object(&obj) {
                match code {
                    ErrorCode::OutOfMemory => break,
                    _ => panic!("put failed"),
                }
            }
        }
        allkeys.push(key);
        counter += 1;
    }
    info!("inserted {} objects", counter);

    // free some (TODO random picking)
    let many: usize = ((counter as f32) * pct_to_free) as usize;
    for key in allkeys.iter().take(many) {
        assert!(nib.del_object(key).is_ok());
    }

    let dur = Duration::from_secs(15);
    thread::sleep(dur);
}

fn alloc_free_all() {
    alloc_free(1.0f32);
}

#[test]
fn alloc_free_half() {
    alloc_free(0.5f32);
}
