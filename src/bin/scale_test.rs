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

use log::LogLevel;
use nibble::common::ErrorCode;
use nibble::epoch;
use nibble::logger;
use nibble::memory;
use nibble::nib::{PutPolicy,Nibble};
use nibble::numa::NodeId;
use nibble::segment::{ObjDesc,SEGMENT_SIZE};
use rand::Rng;
use std::thread::{self,JoinHandle};
use std::time::Duration;
use std::sync::atomic::*;
use std::sync::Arc;
use std::time::Instant;

fn run() {
    logger::enable();

    info!("creating Nibble...");
    let mut nib = Nibble::default();

    info!("inserting objects...");
    let mut keycount: usize = 0;
    let mut size: usize = 0;

    let len: usize = 512;
    let value = memory::allocate::<u8>(len);
    let v = Some(value as *const u8);

    let mut fill: usize = SEGMENT_SIZE * nib.nnodes();
    let policy = PutPolicy::Interleave;
    while fill > (len*2) {
        let key = keycount.to_string();
        let obj = ObjDesc::new(key.as_str(), v, len as u32);
        if let Err(code) = nib.put_where(&obj, policy) {
            warn!("appended {} bytes so far", fill);
            panic!("{:?}", code)
        }
        fill -= obj.len_with_header();
        keycount += 1;
    }

    info!("starting scale test -----------------------");

    // downgrade to immutable shareable alias
    let nib = Arc::new(nib);

    let nthreads: usize = 1;
    // used to block threads from starting, then again as an
    // accumulator for total ops performed among all threads
    let accum = Arc::new(AtomicUsize::new(nthreads));

    let now = Instant::now();
    let mut handles: Vec<JoinHandle<()>> = Vec::with_capacity(nthreads);
    for _ in 0..nthreads {
        let accum = accum.clone();
        let nib = nib.clone();
        handles.push( thread::spawn( move || {
            let mut ops = 0usize;
            let mut rng = rand::thread_rng();
            accum.fetch_sub(1, Ordering::Relaxed);
            while accum.load(Ordering::Relaxed) > 0 { ; }
            let now = Instant::now();
            while now.elapsed().as_secs() < 2 {
                let keyi = (rng.next_u32() as usize) % keycount;
                let key = keyi.to_string();
                let _ = nib.get_object(&key);
                ops += 1;
            }
            accum.fetch_add(ops, Ordering::Relaxed);
        }));
    }
    for handle in handles {
        let _ = handle.join();
    }

    let ops = accum.load(Ordering::Relaxed) as f64;
    let dur = now.elapsed();
    let nsec = dur.as_secs() * 1000000000u64
        + dur.subsec_nanos() as u64;
    println!("value.len {} policy {:?} threads {} mops {:.3}",
             len, policy, nthreads, (ops/1e6)/((nsec as f64)/1e9));

    unsafe { memory::deallocate(value, len); }
}

fn main() {
    run();
}
