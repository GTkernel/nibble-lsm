#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]

extern crate rand; // import before kvs
#[macro_use]
extern crate log;
extern crate time;
extern crate clap;

extern crate kvs;

use clap::{Arg, App, SubCommand};
use log::LogLevel;
use kvs::common::{Pointer,ErrorCode,rdrand};
use kvs::epoch;
use kvs::logger;
use kvs::memory;
use kvs::lsm::{PutPolicy,LSM};
use kvs::numa::{self,NodeId};
use kvs::sched::*;
use kvs::segment::{ObjDesc,SEGMENT_SIZE};
use rand::Rng;
use std::mem;
use std::sync::Arc;
use std::sync::atomic::*;
use std::thread::{self,JoinHandle};
use std::collections::VecDeque;
use std::time::Duration;
use std::time::Instant;

#[derive(Clone,Copy,Debug)]
struct Config {
    size: usize,
    capacity: usize,
    nobject: usize,
}

fn run() { }

#[cfg(IGNORE)]
fn run() {
    logger::enable();

    let mut kvs = LSM::new( 1usize<<33 );
    kvs.enable_compaction(NodeId(0));
    let capacity = kvs.capacity();

    let size: usize = 1000;
    let value = memory::allocate::<u8>(size);
    let v = Pointer(value as *const u8);

    let mut counter: usize = 0;
    let mut inserted: usize = 0;
    let fill80 = ((capacity as f64) * 0.8 / 2f64) as usize;
    info!("filling socket 0 80%: {}", fill80);
    loop {
        let obj = ObjDesc::new(counter as u64, v, size as u32);
        if let Err(e) = kvs.put_where(&obj, PutPolicy::Specific(0)) {
            if let ErrorCode::OutOfMemory = e {
                info!("log filled, no more inserting");
                break;
            }
            panic!("Error {:?}", e);
        }
        counter += 1;
        inserted += size + 8 + 8; // +key +header
        if inserted >= fill80 {
            info!("reached fill capacity");
            break;
        }
    }
    println!("inserted {} objects totaling {} bytes",
             counter, (counter as usize)*size);

    println!("erasing half of objects randomly");

    let mut k: Vec<usize> = (0..counter).collect();
    let mut rng = rand::thread_rng();
    for i in 0..counter {
        let o = (rng.next_u32() as usize % (counter-i)) + i;
        k.swap(i, o);
    }

    let mut counter = 0;
    for x in &k {
        if let Err(e) = kvs.del_object(*x as u64) {
            println!("x {}", x);
            panic!("error: {:?}", e);
        }
        if counter > (k.len()>>1) {
            break;
        }
        counter += 1;
    }


    let dur = Duration::from_secs(1000000);
    std::thread::sleep(dur);
}

fn main() {
    run();
}
