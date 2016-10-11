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
use nibble::segment::{ObjDesc,SEGMENT_SIZE};
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
	println!("# Nibble enabling compaction on Node 0");
    nib.enable_compaction(NodeId(0));
	let p = Box::into_raw(nib);
	println!("# Nibble @ {:?}", p);
	unsafe {
		NIBBLE.0 = p;
	}
}

// return 0 if ok else 1
#[no_mangle] pub extern
fn nibble_put(key: u64, len: u64) -> i32 {
	let nib: &Nibble = unsafe { &*NIBBLE.0 };
	// println!("put {:x}", key);
    let obj = ObjDesc::null(key, len as usize);
    match nib.put_where(&obj, nib::PutPolicy::Specific(0)) {
        Ok(_) => 0i32,
        Err(e) => match e {
            ErrorCode::OutOfMemory => 1i32,
            _ => panic!("error put: {:?}", e),
        },
    }
}

// return 0 if ok else 1
#[no_mangle] pub extern
fn nibble_del(key: u64) -> i32 {
	let nib: &Nibble = unsafe { &*NIBBLE.0 };
	// println!("del {:x}", key);
	match nib.del_object(key) {
        Ok(_) => 0i32,
        Err(e) => {
            println!("ERROR: nibble_del: {:?}", e);
            1i32
        },
    }
}
