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
// use kvs::segment::{ObjDesc,SEGMENT_SIZE};
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
	println!("# LSM enabling compaction");
	for sock in 0..numa::NODE_MAP.sockets() {
		kvs.enable_compaction(NodeId(sock));
	}
	let p = Box::into_raw(kvs);
	println!("# LSM @ {:?}", p);
	unsafe {
		KVS.0 = p;
	}
}

#[no_mangle] pub extern
fn kvs_alloc(key: u64, len: u64, sock: u32) -> *const u8 {
	let kvs: &LSM = unsafe { &*KVS.0 };
	// println!("alloc {:x}", key);
	kvs.alloc(key, len, sock).0
}

#[no_mangle] pub extern
fn kvs_free(key: u64, fail: i32) {
	let kvs: &LSM = unsafe { &*KVS.0 };
	// println!("free {:x}", key);
	let ret = kvs.free(key);
    assert!((1 == fail) && ret,
        "key {} not found upon deletion", key);
}
