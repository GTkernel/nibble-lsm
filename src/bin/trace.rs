// FIXME lots of copy/paste from src/bin/ycsb.rs

#![feature(core_intrinsics)]
#![allow(unused_imports)]
#![allow(dead_code)]

extern crate rand; // import before nibble
#[macro_use]
extern crate log;
extern crate time;
extern crate clap;
extern crate num;
extern crate crossbeam;
extern crate parking_lot as pl;

#[macro_use]
extern crate nibble;

use nibble::distributions::*;

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
use nibble::trace::*;

use rand::Rng;
use std::mem;
use std::intrinsics;
use std::sync::Arc;
use std::sync::atomic::*;
use std::thread::{self,JoinHandle};
use std::time::{Instant,Duration};
use std::ptr;
use std::cmp;
use std::slice;

use std::fs::File;
use std::io::{self, BufReader};
use std::io::prelude::*;
use std::collections::VecDeque;

use std::str::FromStr;

//////////////////////////////////////////////////////////////////////
// User-configurable options
//

/// XXX Make sure whatever buffer is used for receiving and inserting objects
/// is consistent across systems, e.g. stack, or heap, etc.
pub const MAX_KEYSIZE: usize = 1usize << 25;

/// How long to run the experiment before halting.
pub const RUNTIME: usize = 20;

/// Total memory to pre-allocate for MICA, Nibble, RAMCloud, etc.
/// Should match what is in the scripts/trace/run-trace script.
pub const MAX_MEMSIZE: usize = 8usize << 40;

pub const LOAD_FILE:  &'static str = "fb_etc_load.in";
pub const TRACE_FILE: &'static str = "fb_etc_trace.in";

//
// END of user-configurable options
//////////////////////////////////////////////////////////////////////

#[cfg(not(feature = "extern_ycsb"))]
static mut NIBBLE: Pointer<Nibble> = Pointer(0 as *const Nibble);

#[derive(Debug)]
struct Config {
	load_path: String,
	trace_path: String,
	/// memory size, bytes
	total: usize,
	/// max number of objects
	records: usize,
}

struct WorkloadGenerator {
	config: Config,
	sockets: usize,
}

impl WorkloadGenerator {

	pub fn new(config: Config) -> Self {
		WorkloadGenerator {
			config: config,
			sockets: numa::NODE_MAP.sockets(),
		}
	}

	/// read in the file with the load trace
	/// with two space-separated columns: key value_size
	/// both unsigned long. Then split the array among threads,
	/// and have them insert the keys
	fn __setup(&mut self, parallel: bool) -> Result<(),String> {
		let now = Instant::now();

		let mut items: Vec< (u64,u64) > = Vec::with_capacity(1usize<<34);
		info!("Allocating load trace {} seconds", now.elapsed().as_secs());
		let now = Instant::now();

		let file = match File::open(&self.config.load_path) {
			Ok(f) => f,
			Err(_) => return Err( String::from("cannot open load file") ),
		};
		let file = BufReader::new(file);

        let mut key = 1u64;
        let mut total_size = 0usize;

		for line in file.lines() {
			if line.is_err() { break; }
			let line = line.unwrap();
			if line.starts_with("#") { continue; }
			let mut iter = line.split_whitespace();
	        let size = match iter.next() {
	            None => return Err( String::from("line is missing size") ),
	            Some(s) => match s {
	                "na" => 0u64,
	                _ => match f64::from_str(s) {
	                    Ok(v) => v as u64,
	                    Err(_) => return Err(String::from("cannot parse size")),
	                },
	            },
	        };
            if size > 0 {
    	        items.push( (key,size) );
                key += 1;
                total_size += size as usize;
                if 0 == (items.len() % 50_000_000_usize) {
                    info!("Read {} mil. objects...",
                        items.len() / 1_000_000usize);
                }
            }
            // XXX remove me
            //if items.len() > 200_000_000_usize {
            //    println!("LIMITING # OBJECTS to {} total size {}",
            //            items.len(), total_size);
            //    break;
            //}
		}
		let items = items; // make immutable to share among threads
		info!("Reading file {} seconds. {} keys",
			now.elapsed().as_secs(), items.len());
		let now = Instant::now();

		self.config.records = items.len() * 2;
        info!("Running with {:?}", self.config);
		kvs_init(&self.config);

		if parallel {
			let threads_per_sock = 2;
			let threads = self.sockets * threads_per_sock;
			let per_thread = items.len() / threads;
			let ncpus = numa::NODE_MAP.ncpus();

			info!("Inserting objects with {} threads...", threads);

			// pin threads round-robin across sockets
			// use the first 'n' threads on each socket
			let mut cpus: Vec<usize> = Vec::with_capacity(ncpus);
			for i in 0..self.sockets {
				let mut ids = numa::NODE_MAP.cpus_of(NodeId(i)).get();
				ids.truncate(threads_per_sock);
				cpus.append(&mut ids);
			}
			let cpus = pl::Mutex::new(cpus);

			let offsets: Vec<usize> =
				(0usize..threads).map(|e|per_thread*e).collect();
			let offsets = pl::Mutex::new(offsets);

			let mut guards = vec![];
			crossbeam::scope( |scope| {

				for _ in 0..threads {

					let guard = scope.spawn( || {
						let cpu = match cpus.lock().pop() {
							None => panic!("No cpu for this thread?"),
							Some(c) => c,
						};
						unsafe { pin_cpu(cpu); }
						let sock = numa::NODE_MAP.sock_of(cpu);

						let value = memory::allocate::<u8>(MAX_KEYSIZE);
						let v = Pointer(value as *const u8);

						let offset = match offsets.lock().pop() {
							None => panic!("No offset for this thread?"),
							Some(o) => o,
						};

						let from = offset;
						let to   = offset + threads_per_sock;
						let s    = &items[from..to];
						for tup in s {
							let (key,size) = *tup;
							if size as usize > MAX_KEYSIZE {
								panic!("key size {} > max keysize {}",
									size, MAX_KEYSIZE);
							}
							put_object(key as u64, v, size as usize, sock.0);
						}
						unsafe { memory::deallocate(value, MAX_KEYSIZE); }
					});
					guards.push(guard);
				}

				for g in guards { g.join(); }
			}); // crossbeam
		}

		// single-threaded insertion
		// meant mainly for ramcloud (which ignores the socket parameter)
		else {
            info!("Inserting objects with 1 thread...");
			let value = memory::allocate::<u8>(MAX_KEYSIZE);
			let v = Pointer(value as *const u8);

			for (key,size) in items {
				put_object(key as u64, v, size as usize, 0);
			}
			unsafe { memory::deallocate(value, MAX_KEYSIZE); }
		}

		info!("Setup took {} seconds", now.elapsed().as_secs());

		Ok( () )
	}

    #[cfg(any(feature = "rc"))]
    #[cfg(feature = "extern_ycsb")]
	pub fn setup(&mut self) {
        if let Err(msg) = self.__setup(false:with) {
            panic!("Error in setup: {}", msg);
        }
	}

    #[cfg(any(feature = "mica", feature = "masstree", not(feature = "extern_ycsb")))]
	pub fn setup(&mut self) {
        if let Err(msg) = self.__setup(true) {
            panic!("Error in setup: {}", msg);
        }
	}

	pub fn run(&mut self) {
        let t = Instant::now();
		let trace = Trace::new2(&self.config.trace_path);
        info!("Trace loaded in {} seconds", t.elapsed().as_secs());

		let threadcount: Vec<usize>;
		let sockets = numa::NODE_MAP.sockets();
        let cpus_pernode = numa::NODE_MAP.cpus_in(NodeId(0));
		threadcount = (1usize..(sockets+1))
			.map(|e|cpus_pernode*e).collect();

		for threads in threadcount {
			info!("Running with {} threads...", threads);

			let per_thread = trace.rec.len() / threads;
			info!("trace slice has {} entries", per_thread);

			// pin threads incrementally across sockets
			let ncpus = numa::NODE_MAP.ncpus();
			let mut cpus: Vec<usize> = Vec::with_capacity(ncpus);
			for i in 0..ncpus {	cpus.push(i); }
			debug!("pinned cpus: {:?}", cpus);
			let cpus = pl::Mutex::new(cpus);

			let offsets: Vec<usize> = 
    			(0usize..threads).map(|e|per_thread*e).collect();
			debug!("offsets: {:?}", offsets);
			let offsets = pl::Mutex::new(offsets);

            let throughput = AtomicUsize::new(0);

			let mut guards = vec![];
			crossbeam::scope( |scope| {

				for _ in 0..threads {

					let guard = scope.spawn( || {
						let cpu = match cpus.lock().pop() {
							None => panic!("No cpu for this thread?"),
							Some(c) => c,
						};
						unsafe { pin_cpu(cpu); }
						let sock = numa::NODE_MAP.sock_of(cpu);

						// buffer used for PUT operations
						let value = memory::allocate::<u8>(MAX_KEYSIZE);
						let v = Pointer(value as *const u8);

						let offset = match offsets.lock().pop() {
							None => panic!("No offset for this thread?"),
							Some(o) => o,
						};

                        let mut now = Instant::now();
                        let mut nops = 0_usize;
                        let mut first = true; // iteration

						let from = offset;
						let to   = offset + per_thread;
						let s    = &trace.rec[from..to];
                        let mut iter = s.iter();
                        'outer: loop {
                            for _ in 0..600_000 {
                                let entry = match iter.next() {
                                    None => { // restart
                                        iter = s.iter();
                                        continue 'outer;
                                    },
                                    Some(e) => e,
                                };
                                let too_big = entry.size as usize > MAX_KEYSIZE;
                                if unlikely!(too_big) {
                                    panic!("key size {} > max keysize {}",
                                           entry.size, MAX_KEYSIZE);
                                }
                                // assert!(entry.key > 0, "zero key found");
                                match entry.op {
                                    Op::Get => get_object(entry.key as u64),
                                    Op::Del => del_object(entry.key as u64),
                                    Op::Set =>
                                        put_object(entry.key as u64, v,
                                                   entry.size as usize, sock.0),
                                }
                                nops += 1;
                            }
                            let t = now.elapsed().as_secs() as usize;
                            if t > RUNTIME {
                                // skip first RUNTIME seconds of measurements
                                if first {
                                    first = false;
                                    now = Instant::now();
                                    nops = 0;
                                    continue 'outer;
                                }
                                let th = (nops as f64 / t as f64) as usize;
                                info!("thread throughput {}", th);
                                throughput.fetch_add( th as usize, Ordering::SeqCst );
                                break 'outer;
                            }
						}
						unsafe { memory::deallocate(value, MAX_KEYSIZE); }
					});
					guards.push(guard);

				} // for

				for g in guards { g.join(); }

			}); // crossbeam

            let total = throughput.load(Ordering::SeqCst);
            println!("total ops/sec {}", total);

        } // for each set of threads
	}
}

fn main() {
    logger::enable();

	let config = Config {
		load_path:  String::from(LOAD_FILE),
		trace_path: String::from(TRACE_FILE),
		total: MAX_MEMSIZE,
		records: 0, // will be set after reading load_path
	};
	info!("Specified (.records will be updated) {:?}", config);
	let mut gen = WorkloadGenerator::new(config);
	gen.setup();
	gen.run();
}

//
// Refer to src/bin/ycsb.rs for comments for these hook functions.
//

#[cfg(not(feature = "extern_ycsb"))]
fn kvs_init(config: &Config) {
    let nibble =
        Box::new(Nibble::new2(config.total, config.records*2));
        //Box::new(Nibble::new2(config.total, 1usize<<30));
    info!("Enabling compaction");
    for node in 0..numa::NODE_MAP.sockets() {
        nibble.enable_compaction(NodeId(node));
    }
    unsafe {
        let p = Box::into_raw(nibble);
        NIBBLE = Pointer(p);
    }
}

#[inline(always)]
#[cfg(not(feature = "extern_ycsb"))]
fn put_object(key: u64, value: Pointer<u8>, len: usize, sock: usize) {
    let nibble: &Nibble = unsafe { &*NIBBLE.0 };
    let obj = ObjDesc::new(key, value, len);
    let nibnode = nib::PutPolicy::Specific(sock);
    loop {
        let err = nibble.put_where(&obj, nibnode);
        if err.is_err() {
            match err {
                Err(ErrorCode::OutOfMemory) => continue,
                _ => {
                    println!("Error: {:?}", err.unwrap());
                    unsafe { intrinsics::abort(); }
                },
            }
        } else {
            break;
        }
    }
}

// thread-local receive buffer 
thread_local!(
    static BUFFER: *mut u8 = memory::allocate(MAX_KEYSIZE);
);

#[inline(always)]
#[cfg(not(feature = "extern_ycsb"))]
fn get_object(key: u64) {
    BUFFER.with( |p| {
        let nibble: &Nibble = unsafe { &*NIBBLE.0 };
        // let mut buf: [u8;MAX_KEYSIZE] = unsafe { mem::uninitialized() };
        let mut buf: &mut [u8] = unsafe {
            slice::from_raw_parts_mut::<u8>(*p, MAX_KEYSIZE)
        };
        let _ = nibble.get_object(key, buf);
    });
}

#[inline(always)]
#[cfg(not(feature = "extern_ycsb"))]
fn del_object(key: u64) {
	let nibble: &Nibble = unsafe { &*NIBBLE.0 };
	let _ = nibble.del_object(key);
}

#[link(name = "micaext")]
#[cfg(feature = "mica")]
#[cfg(feature = "extern_ycsb")]
extern {
    fn extern_kvs_init();
    fn extern_kvs_put(key: u64, len: u64, buf: *const u8) -> i32;
    fn extern_kvs_del(key: u64) -> bool;
    fn extern_kvs_get(key: u64) -> bool;
}

#[link(name = "ramcloudext")]
#[cfg(feature = "rc")]
#[cfg(feature = "extern_ycsb")]
extern {
    fn extern_kvs_init();
    fn extern_kvs_put(key: u64, len: u64, buf: *const u8);
    fn extern_kvs_del(key: u64);
    fn extern_kvs_get(key: u64);
}

#[cfg(feature = "extern_ycsb")]
fn kvs_init(config: &Config) {
    unsafe {extern_kvs_init(); }
}

#[cfg(any(feature = "rc", feature = "masstree"))]
#[cfg(feature = "extern_ycsb")]
fn put_object(key: u64, value: Pointer<u8>, len: usize, sock: usize) {
    unsafe {
        trace!("PUT {:x} len {}", key, len);
        extern_kvs_put(key, len as u64, value.0);
    }
}

#[cfg(any(feature = "rc", feature = "masstree"))]
#[cfg(feature = "extern_ycsb")]
fn get_object(key: u64) {
    unsafe {
        trace!("GET {:x}", key);
        extern_kvs_get(key);
    }
}

#[cfg(any(feature = "rc", feature = "masstree"))]
#[cfg(feature = "extern_ycsb")]
fn del_object(key: u64) {
    unsafe {
        trace!("DEL {:x}", key);
        extern_kvs_del(key);
    }
}


#[cfg(feature = "mica")]
#[cfg(feature = "extern_ycsb")]
fn put_object(key: u64, value: Pointer<u8>, len: usize, sock: usize) {
    // we ignore 'sock' b/c RAMCloud is NUMA-agnostic
    // (MICA might, though...)
    unsafe {
        trace!("PUT {:x} len {}", key, len);
        let ret: i32 = extern_kvs_put(key, len as u64, value.0);
        // XXX make sure this matches with
        // mica-kvs.git/src/table.h enum put_reason
        match ret {
            50 => return,
            r => {
                match r {
                    111 => println!("MICA failed to insert in table"),
                    112 => println!("MICA failed to insert in heap"),
                    _ => println!("MICA failed with unknown: {}", ret),
                }
                //unsafe { intrinsics::abort(); }
            },
        }
    }
}

#[cfg(feature = "mica")]
#[cfg(feature = "extern_ycsb")]
fn get_object(key: u64) {
    unsafe {
        trace!("GET {:x}", key);
        if !extern_kvs_get(key) {
            println!("GET failed on key 0x{:x}", key);
            //unsafe { intrinsics::abort(); }
        }
        //assert!( extern_kvs_get(key),
            //"GET failed on key 0x{:x}", key);
    }
}

#[cfg(feature = "mica")]
#[cfg(feature = "extern_ycsb")]
fn del_object(key: u64) {
    unsafe {
        trace!("DEL {:x}", key);
        if !extern_kvs_del(key) {
            println!("DEL failed on key 0x{:x}", key);
        }
        //assert!( extern_kvs_del(key),
            //"DEL failed on key 0x{:x}", key);
    }
}


#[link(name = "masstree")]
#[cfg(feature = "masstree")]
#[cfg(feature = "extern_ycsb")]
extern {
    // must match signatures in libmasstree.cc
    fn extern_kvs_init();
    fn extern_kvs_put(key: u64, len: u64, buf: *const u8);
    fn extern_kvs_del(key: u64);
    fn extern_kvs_get(key: u64);
}
