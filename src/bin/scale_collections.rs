#![feature(test)]
#![feature(asm)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]

extern crate rand; // import before nibble
#[macro_use]
extern crate log;
extern crate test;
extern crate time;
extern crate clap;

extern crate nibble;

use std::collections::*;
use clap::{Arg, App, SubCommand};
use log::LogLevel;
use nibble::common::ErrorCode;
use nibble::epoch;
use nibble::logger;
use nibble::memory;
use nibble::nib::{PutPolicy,Nibble};
use nibble::numa::{self,NodeId};
use nibble::sched::*;
use nibble::segment::{ObjDesc,SEGMENT_SIZE};
use rand::Rng;
use std::mem;
use std::sync::Arc;
use std::sync::atomic::*;
use std::thread::{self,JoinHandle};
use std::time::Duration;
use std::time::Instant;

#[inline(always)]
#[allow(unused_mut)]
unsafe fn rdrand() -> u32 {
    let mut r: u32;
    asm!("rdrand $0" : "=r" (r));
    r
}

fn run(threads: usize) {
    logger::enable();

    let mut keycount: usize = 0;
    let value = String::from("12345678");

    let mut map: HashMap<String,String> = HashMap::new();
    for i in 0..1000000usize {
        let key = keycount.to_string();
        map.insert(key, value.clone());
        keycount += 1;
    }
    println!("starting");

    let accum = Arc::new(AtomicUsize::new(threads));

    let map = Arc::new(map);
    let mut handles: Vec<JoinHandle<()>> = Vec::with_capacity(threads);
    let now = Instant::now();
    for t in 0..threads {
        let accum = accum.clone();
        let map = map.clone();
        handles.push( thread::spawn( move || {
            let mut ops = 0usize;
            // create randomized array of keys to read
            let mut keys: Vec<String> = Vec::with_capacity(keycount);
            for i in 0..keycount {
                keys.push(i.to_string());
            }
            // knuth shuffle
            for i in 0..keycount {
                let r = unsafe { rdrand() } as usize;
                let j = ((r as usize) % (keycount-i)) + i;
                keys.swap(i,j);
            }
            accum.fetch_sub(1, Ordering::Relaxed);
            while accum.load(Ordering::Relaxed) > 0 { ; }
            let now = Instant::now();
            let mut idx = 0usize;
            while now.elapsed().as_secs() < 4 {
                for _ in 0..100usize {
                    let r = unsafe { rdrand() } as usize;
                    let key = &keys[r % keycount];
                    let _ = map.get(key);
                    ops += 1;
                }
            }
            accum.fetch_add(ops, Ordering::Relaxed);
        }));
    }
    for handle in handles {
        let _ = handle.join();
    }
    let dur = now.elapsed();

    let ops = accum.load(Ordering::Relaxed) as f64;
    let nsec = dur.as_secs() * 1000000000u64
        + dur.subsec_nanos() as u64;
    let mut fmt = String::new();
    println!("nobj.mil threads mops");
    println!("{:.3} {} {:.3}",
             (keycount as f64) / 1e6,
             threads, (ops/1e6)/((nsec as f64)/1e9));
}

fn main() {
    let matches = App::new("scale")
        // size of object
        .arg(Arg::with_name("threads")
             .long("threads")
             .takes_value(true))
        .get_matches();

    let threads = match matches.value_of("threads") {
        None => panic!("Specify no. threads"),
        Some(s) => match usize::from_str_radix(s, 10) {
            Ok(s) => s,
            Err(_) => panic!("No. threads NAN"),
        },
    };

    run(threads);
}
