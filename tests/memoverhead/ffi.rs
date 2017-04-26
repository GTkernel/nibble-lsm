#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]
#![feature(const_fn)]

extern crate kvs;

// use kvs::clock;
use kvs::common::{Pointer,ErrorCode,rdrand,rdrandq};
// use kvs::epoch;
use kvs::logger;
// use kvs::memory;
use kvs::lsm::{self,LSM};
use kvs::numa::{self,NodeId};
// use kvs::sched::*;
use kvs::segment::{ObjDesc,SEGMENT_SIZE};
// use rand::Rng;
// use std::collections::VecDeque;
// use std::mem;
// use std::sync::Arc;
// use std::sync::atomic::*;
use std::thread::{self,JoinHandle};
use std::time::{Instant,Duration};
use std::ptr::null;

static mut KVS: Pointer<LSM> = Pointer(null::<LSM>());

#[no_mangle] pub extern
fn kvs_init(cap: usize, nitems: usize) {
	logger::enable();
	println!("# LSM allocating...");
	//let kvs: Box<LSM> = Box::new(LSM::default());
	let kvs: Box<LSM> = Box::new(LSM::new2(cap,nitems));
	println!("# LSM enabling compaction on Node 0");
    kvs.enable_compaction(NodeId(0));
	let p = Box::into_raw(kvs);
	println!("# LSM @ {:?}", p);
	unsafe {
		KVS.0 = p;
	}
}

// return 0 if ok else 1
#[no_mangle] pub extern
fn kvs_put(key: u64, len: u64) -> i32 {
	let kvs: &LSM = unsafe { &*KVS.0 };
	// println!("put {:x}", key);
    let obj = ObjDesc::null(key, len as usize);
    match kvs.put_where(&obj, lsm::PutPolicy::Specific(0)) {
        Ok(_) => 0i32,
        Err(e) => match e {
            ErrorCode::OutOfMemory => 1i32,
            _ => panic!("error put: {:?}", e),
        },
    }
}

// return 0 if ok else 1
#[no_mangle] pub extern
fn kvs_del(key: u64) -> i32 {
	let kvs: &LSM = unsafe { &*KVS.0 };
	// println!("del {:x}", key);
	match kvs.del_object(key) {
        Ok(_) => 0i32,
        Err(e) => {
            println!("ERROR: kvs_del: {:?}", e);
            1i32
        },
    }
}
