#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]

// Just create an instance of Nibble and view its startup messages.

extern crate rand; // import before nibble
#[macro_use]
extern crate log;
extern crate time;
extern crate clap;
extern crate num;
extern crate crossbeam;
extern crate parking_lot as pl;

extern crate nibble;

use clap::{Arg, App, SubCommand};
use log::LogLevel;
use nibble::clock;
use nibble::common::{self,Pointer,ErrorCode,rdrand,rdrandq};
use nibble::meta;
use nibble::hashtable::{HashTable};
use nibble::logger;
use nibble::memory;
use nibble::nib::{self,Nibble};
use nibble::numa::{self,NodeId};
use nibble::sched::*;
use nibble::segment::{ObjDesc,SEGMENT_SIZE};
use rand::Rng;
use std::collections::VecDeque;
use std::mem;
use std::hash;
use std::sync::Arc;
use std::sync::atomic::*;
use std::thread::{self,JoinHandle};
use std::time::{Instant,Duration};
use std::ptr;

fn arg_as_num<T: num::Integer>(args: &clap::ArgMatches,
                               name: &str) -> T {
    match args.value_of(name) {
        None => panic!("Specify {}", name),
        Some(s) => match T::from_str_radix(s,10) {
            Err(_) => panic!("size NaN"),
            Ok(s) => s,
        },
    }
}

fn main() {
    logger::enable();

    let mut n = 304250263527209_u64;
    for i in 0..(1usize<<20) {
        n = common::fnv1a(n);
        //println!("0x{:016x}", b);
        let h = HashTable::make_hash(i as u64);
        let b = h & (1024 - 1);
        println!("{}", b);
    }
    return;

    let matches = App::new("null test to create a Nibble")
        .arg(Arg::with_name("capacity")
             .long("capacity").takes_value(true))
        .get_matches();

    let cap: usize = arg_as_num::<usize>(&matches, "capacity");

    let mut nibble = Nibble::new(cap);
    for n in 0..numa::NODE_MAP.sockets() {
        nibble.enable_compaction(NodeId(n));
    }
}

