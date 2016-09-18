#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]
#![feature(const_fn)]

extern crate nibble;

// use nibble::clock;
use nibble::common::{Pointer,ErrorCode,rdrand,rdrandq};
// use nibble::epoch;
use nibble::logger;
// use nibble::memory;
use nibble::nib::{self,Nibble};
use nibble::numa::{self,NodeId};
// use nibble::sched::*;
// use nibble::segment::{ObjDesc,SEGMENT_SIZE};
// use rand::Rng;
// use std::collections::VecDeque;
// use std::mem;
// use std::sync::Arc;
// use std::sync::atomic::*;
use std::thread::{self,JoinHandle};
use std::time::{Instant,Duration};
use std::ptr::null;

static mut NIBBLE: Pointer<Nibble> = Pointer(null::<Nibble>());

#[no_mangle] pub extern
fn nibble_init(cap: usize, nitems: usize) {
	logger::enable();
	println!("# Nibble allocating...");
	//let nib: Box<Nibble> = Box::new(Nibble::default());
	let nib: Box<Nibble> = Box::new(Nibble::new2(cap,nitems));
	println!("# Nibble enabling compaction");
	for sock in 0..numa::NODE_MAP.sockets() {
		nib.enable_compaction(NodeId(sock));
	}
	let p = Box::into_raw(nib);
	println!("# Nibble @ {:?}", p);
	unsafe {
		NIBBLE.0 = p;
	}
}

#[no_mangle] pub extern
fn nibble_alloc(key: u64, len: u64, sock: u32) -> *const u8 {
	let nib: &Nibble = unsafe { &*NIBBLE.0 };
	// println!("alloc {:x}", key);
	nib.alloc(key, len, sock).0
}

#[no_mangle] pub extern
fn nibble_free(key: u64, fail: i32) {
	let nib: &Nibble = unsafe { &*NIBBLE.0 };
	// println!("free {:x}", key);
	let ret = nib.free(key);
    assert!((1 == fail) && ret,
        "key {} not found upon deletion", key);
}
