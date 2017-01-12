#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]
#![feature(core_intrinsics)]

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
use nibble::logger;
use nibble::memory;
use nibble::nib::{self,Nibble};
use nibble::numa::{self,NodeId};
use nibble::sched::*;
use nibble::segment::{ObjDesc,SEGMENT_SIZE};
use rand::Rng;
use std::collections::VecDeque;
use std::mem;
use std::intrinsics;
use std::sync::Arc;
use std::sync::atomic::*;
use std::thread::{self,JoinHandle};
use std::time::{Instant,Duration};
use std::ptr;

fn main() {
    logger::enable();

    let capacity = 1usize << 31;
    let ht_nitems = 1usize << 20;
    let mut nib: Nibble = Nibble::new2(capacity, ht_nitems);

    let key: u64 = 1;
    let v: Vec<u8> = vec![1u8,2,3,4,5];
    let p = Pointer(v.as_ptr() as *const u8);
    let obj = ObjDesc::new(key, p, v.len()*8);
    assert!(nib.put_object(&obj).is_ok());

    let mut b = [0u8; 5];
    assert!(nib.get_object(key, &mut b).is_ok());

    println!("v: {:?}", v);
    println!("b: {:?}", b);
    assert_eq!(v, b);

    assert!(nib.del_object(key).is_ok());
}
