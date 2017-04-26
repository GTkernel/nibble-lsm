#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]
#![feature(core_intrinsics)]

extern crate rand; // import before kvs
#[macro_use]
extern crate log;
extern crate time;
extern crate clap;
extern crate num;
extern crate crossbeam;
extern crate parking_lot as pl;

extern crate kvs;

use clap::{Arg, App, SubCommand};
use log::LogLevel;
use kvs::clock;
use kvs::common::{self,Pointer,ErrorCode,rdrand,rdrandq};
use kvs::meta;
use kvs::logger;
use kvs::memory;
use kvs::lsm::{self,LSM};
use kvs::numa::{self,NodeId};
use kvs::sched::*;
use kvs::segment::{ObjDesc,SEGMENT_SIZE};
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
    let mut kvs: LSM = LSM::new2(capacity, ht_nitems);

    let key: u64 = 1;
    let v: Vec<u8> = vec![1u8,2,3,4,5];
    let p = Pointer(v.as_ptr() as *const u8);
    let obj = ObjDesc::new(key, p, v.len()*8);
    assert!(kvs.put_object(&obj).is_ok());

    let mut b = [0u8; 5];
    assert!(kvs.get_object(key, &mut b).is_ok());

    println!("v: {:?}", v);
    println!("b: {:?}", b);
    assert_eq!(v, b);

    assert!(kvs.del_object(key).is_ok());
}
