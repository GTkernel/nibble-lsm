// FIXME lots of copy/paste from src/bin/ycsb.rs

#![feature(core_intrinsics)]
#![allow(unused_imports)]
#![allow(dead_code)]

extern crate rand; // import before kvs
#[macro_use]
extern crate log;
extern crate time;
extern crate clap;
extern crate num;
extern crate crossbeam;
extern crate parking_lot as pl;

#[macro_use]
extern crate kvs;

use kvs::distributions::*;

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
use kvs::trace::*;

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
use std::env;

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
pub const RUNTIME: usize = 30;

/// Total memory to pre-allocate for MICA, LSM, RAMCloud, etc.
/// Should match what is in the scripts/trace/run-trace script.
//pub const MAX_MEMSIZE: usize = 6_usize << 40;
//pub const MAX_MEMSIZE: usize = 2_usize << 40;
pub const MAX_MEMSIZE: usize = (1_usize << 40)
                                + 340_usize * (1_usize << 30);

pub const LOAD_FILE:  &'static str = "fb-etc-objects.dat";
pub const TRACE_FILE: &'static str = "fb-etc-trace.dat";

/// How many threads to use for the insertion/setup phase, per socket.
/// If not defined, it uses a default.
pub const ENV_SETUP_THREADS: &'static str = "TRACE_SETUP_THREADS_PER_SOCK";
pub const SETUP_THREADS_PERSOCK: usize = 12;

/// Set a limit for how many objects we insert from the trace.
pub const MAX_N_OBJECTS: Option<usize> = None;
//pub const MAX_N_OBJECTS: Option<usize> = Some(1_000_000_000_usize);

/// How many iterations to run the execution phase.
pub const EXEC_ITERS: usize = 1;

/// If running an insert-only benchmark (where we only invoke setup()),
/// run for this many seconds, then report performance.
/// You'll need to comment the gen.run() lines in main,
/// and uncomment a block of code in setup that uses this; search for
/// this variable to find it.
pub const SETUP_STOP_AFTER: usize = 60;

/// If the trace is short and we run out of operations: true to loop
/// over it until RUNTIME is reached, or false to stop at that point.
pub const CYCLE_TRACE: bool = false;

//
// END of user-configurable options
//////////////////////////////////////////////////////////////////////

#[cfg(not(feature = "extern_ycsb"))]
static mut KVS: Pointer<LSM> = Pointer(0 as *const LSM);

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

    /// don't load any initialization file. just start the KVS
    fn quick_setup(&mut self) {
        self.config.records = 1usize << 28;
        info!("Running with {:?}", self.config);
        kvs_init(&self.config);
    }

    /// read in the file with the load trace
    /// with two space-separated columns: key value_size
    /// both unsigned long. Then split the array among threads,
    /// and have them insert the keys
    #[allow(unused_variables)]
    #[allow(unused_mut)]
    fn __setup(&mut self, parallel: bool) -> Result<(),String> {
        let now = Instant::now();

        let mut items: Vec< (u64,u64) > = Vec::with_capacity(1usize<<34);
        let now = Instant::now();
        info!("Loading object snapshot...");

        let mut total_size = 0usize;

        // load in database using binary format (just an array of u32)
        let mut mapped =
            memory::MemMapFile::new(self.config.load_path.as_str());
        let nums = unsafe { mapped.as_slice::<u32>() };
        for tup in (1u64..).zip(nums) {
            let key: u64 = tup.0;
            let len: u64 = *tup.1 as u64;
            // disregard zero-byte objects (LSM chokes on them still)
            if len > 0 {
                items.push( (key,len) );
                total_size += len as usize;
                if 0 == (items.len() % 500_000_000_usize) {
                    info!("Read {} mil. objects...",
                          items.len() / 1_000_000usize);
                }
            }
            if let Some(nn) = MAX_N_OBJECTS {
                if items.len() > nn {
                    println!("LIMITING # OBJECTS to {} total size {}",
                             items.len(), total_size);
                    break;
                }
            }
        }

        let items = items; // make immutable to share among threads
        info!("Reading file {} seconds. {} keys",
            now.elapsed().as_secs(), items.len());
        let now = Instant::now();

        self.config.records = items.len();
        info!("Running with {:?}", self.config);
        kvs_init(&self.config);

        let throughput = AtomicUsize::new(0);

        if parallel {
            let threads_per_sock =
                match env::var_os(ENV_SETUP_THREADS) {
                    None => SETUP_THREADS_PERSOCK,
                    Some(osstr) => match osstr.to_str() {
                        None => panic!("{} not valid", ENV_SETUP_THREADS),
                        Some(s) => match usize::from_str_radix(s,10u32) {
                            Err(_) => panic!("{} not a number",
                                             ENV_SETUP_THREADS),
                            Ok(n) => n,
                        },
                    },
            };
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
                        let mut n = 0usize; // count total inserted

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
                        let to   = offset + per_thread;
                        let s    = &items[from..to];
                        let start = Instant::now();
                        let mut i = 0usize; // for throttling timing checks
                        for tup in s {
                            let (key,size) = *tup;
                            if unlikely!(size as usize > MAX_KEYSIZE) {
                                panic!("key size {} > max keysize {}",
                                    size, MAX_KEYSIZE);
                            }
                            put_object(key as u64, v, size as usize, sock.0);
                            n += 1;

                            //  // Comment these lines out if not doing insertion
                            //  // workload benchmarks. -->
                            //  i += 1;
                            //  // every million, check if we should stop
                            //  if unlikely!((i >> 20) > 0) {
                            //      i = 0usize;
                            //      let sec = start.elapsed().as_secs() as usize;
                            //      if sec > SETUP_STOP_AFTER {
                            //          let th = n as f64 / sec as f64;
                            //          throughput.fetch_add(th as usize,Ordering::SeqCst);
                            //          info!("Setup stopping sec {} total {} ops/sec",
                            //                SETUP_STOP_AFTER, th);
                            //          break;
                            //      }
                            //  }
                            //  // --> until here

                        }
                        let el = start.elapsed();
                        let sec: f64 = el.as_secs() as f64
                            + el.subsec_nanos() as f64 / 1e9;
                        info!("Thread insert throughput {} ops/sec. total {} ops {} sec",
                              n as f64 / sec, n as usize, sec);
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

        let th = throughput.load(Ordering::SeqCst);
        info!("Setup took {} sec, total {} ops/sec",
              now.elapsed().as_secs(), th);

        Ok( () )
    }

    #[cfg(any(feature = "rc"))]
    #[cfg(feature = "extern_ycsb")]
    pub fn setup(&mut self) {
        if let Err(msg) = self.__setup(false) {
            panic!("Error in setup: {}", msg);
        }
    }

    #[cfg(any(feature = "mica", feature="redis", feature = "masstree", not(feature = "extern_ycsb")))]
    pub fn setup(&mut self) {
        if let Err(msg) = self.__setup(true) {
            panic!("Error in setup: {}", msg);
        }
    }

    pub fn run(&mut self, warmup: bool) {
        let t = Instant::now();
        let trace = Trace::new2(&self.config.trace_path);
        info!("Trace loaded in {} seconds", t.elapsed().as_secs());

        let sockets = numa::NODE_MAP.sockets();
        let cpus_pernode = numa::NODE_MAP.cpus_in(NodeId(0));
        let threadcount: Vec<usize>;
        let runtime_: usize;
        if warmup {
            info!("Entering warmup phase...");
            threadcount = vec![240usize];
            runtime_ = 2 * 60;
        } else {
            info!("Entering test phase...");
            let mut t: Vec<usize> = (1usize..(sockets+1))
                .map(|e|cpus_pernode*e).collect();
            //let mut t: Vec<usize> = (1usize..8)
                //.map(|e|cpus_pernode*e).collect();
            t.reverse();
            threadcount = t;
            runtime_ = RUNTIME;
        };
        info!("Measurement length {} seconds", runtime_);

        for threads in threadcount {
        for _ in 0..EXEC_ITERS {
            info!("Running with {} threads...", threads);

            let per_thread = trace.rec.len() / threads;
            info!("trace slice has {} entries", per_thread);

            // pin threads incrementally across sockets
            let ncpus = numa::NODE_MAP.ncpus();
            let mut cpus: Vec<usize> = Vec::with_capacity(ncpus);
            for i in 0..ncpus { cpus.push(i); }
            debug!("pinned cpus: {:?}", cpus);
            let cpus = pl::Mutex::new(cpus);

            let offsets: Vec<usize> = 
                (0usize..threads).map(|e|per_thread*e).collect();
            debug!("offsets: {:?}", offsets);
            let offsets = pl::Mutex::new(offsets);

            let throughput = AtomicUsize::new(0);
            let mut time_overall = Instant::now();

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
                        let mut first = false; // iteration

                        let mut idx: usize = offset;
                        let s = &trace.rec[..];

                        let mut exit_now: bool = false;
                        'outer: loop {
                            for _ in 0..600_000 {
                                if unlikely!(idx >= s.len()) {
                                    info!("Restarted trace!");
                                    if !CYCLE_TRACE {
                                        exit_now = true;
                                        break;
                                    }
                                    idx = 0;
                                }
                                let entry = unsafe { s.get_unchecked(idx) };
                                idx += 1;

                                let too_big = entry.size as usize > MAX_KEYSIZE;
                                if unlikely!(too_big) {
                                    panic!("key size {} > max keysize {}",
                                           entry.size, MAX_KEYSIZE);
                                }
                                // XXX
                                let isget = match entry.op {
                                    Op::Get => true,
                                    _ => false,
                                };
                                if unlikely!(warmup && isget)
                                { continue; }
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
                            if t > runtime_ || exit_now == true {
                                if warmup { break 'outer; }
                                if first {
                                    first = false;
                                    now = Instant::now();
                                    nops = 0;
                                    continue 'outer;
                                }
                                let th = (nops as f64 / t as f64) as usize;
                                info!("thread throughput {} ops/sec", th);
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

            let s = time_overall.elapsed().as_secs();
            let total = throughput.load(Ordering::SeqCst);
            println!("total ops/sec {} time {} sec", total, s);

        } // for each set of threads

        //unsafe { extern_kvs_dump_allocator(); }

    } // for (multiple iterations)
    } // for (each set of thread counts)
}

fn main() {
    logger::enable();
    info!("Running with MAX_MEMSIZE = {}", MAX_MEMSIZE);

    let config = Config {
        load_path:  String::from(LOAD_FILE),
        trace_path: String::from(TRACE_FILE),
        total: MAX_MEMSIZE,
        records: 0, // will be set after reading load_path
    };
    info!("Specified (.records will be updated) {:?}", config);
    let mut gen = WorkloadGenerator::new(config);
    //gen.setup();
    gen.quick_setup();
    //gen.run(true); // warmup
    gen.run(false); // actual measurement
}

//
// Refer to src/bin/ycsb.rs for comments for these hook functions.
//

#[cfg(not(feature = "extern_ycsb"))]
fn kvs_init(config: &Config) {
    let kvs =
        Box::new(LSM::new2(config.total, config.records*2));
        //Box::new(LSM::new2(config.total, 1usize<<30));
    info!("Enabling compaction");
    for node in 0..numa::NODE_MAP.sockets() {
        kvs.enable_compaction(NodeId(node));
    }
    unsafe {
        let p = Box::into_raw(kvs);
        KVS = Pointer(p);
    }
}

#[inline(always)]
#[cfg(not(feature = "extern_ycsb"))]
fn put_object(key: u64, value: Pointer<u8>, len: usize, sock: usize) {
    let kvs: &LSM = unsafe { &*KVS.0 };
    let obj = ObjDesc::new(key, value, len);
    let nibnode = lsm::PutPolicy::Specific(sock);
    loop {
        let err = kvs.put_where(&obj, nibnode);
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
        let kvs: &LSM = unsafe { &*KVS.0 };
        // let mut buf: [u8;MAX_KEYSIZE] = unsafe { mem::uninitialized() };
        let mut buf: &mut [u8] = unsafe {
            slice::from_raw_parts_mut::<u8>(*p, MAX_KEYSIZE)
        };
        let _ = kvs.get_object(key, buf);
    });
}

#[inline(always)]
#[cfg(not(feature = "extern_ycsb"))]
fn del_object(key: u64) {
    let kvs: &LSM = unsafe { &*KVS.0 };
    let _ = kvs.del_object(key);
}

#[link(name = "micaext")]
#[cfg(feature = "mica")]
#[cfg(feature = "extern_ycsb")]
extern {
    fn extern_kvs_init();
    fn extern_kvs_put(key: u64, len: u64, buf: *const u8) -> i32;
    fn extern_kvs_del(key: u64) -> bool;
    fn extern_kvs_get(key: u64) -> bool;
    fn extern_kvs_dump_allocator();
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

#[cfg(any(feature = "rc", feature = "masstree", feature="redis"))]
#[cfg(feature = "extern_ycsb")]
fn put_object(key: u64, value: Pointer<u8>, len: usize, sock: usize) {
    unsafe {
        trace!("PUT {:x} len {}", key, len);
        extern_kvs_put(key, len as u64, value.0);
    }
}

#[cfg(any(feature = "rc", feature = "masstree", feature="redis"))]
#[cfg(feature = "extern_ycsb")]
fn get_object(key: u64) {
    unsafe {
        trace!("GET {:x}", key);
        extern_kvs_get(key);
    }
}

#[cfg(any(feature = "rc", feature = "masstree", feature="redis"))]
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
    unsafe {
        trace!("PUT {:x} len {}", key, len);
        let mut c: usize = 0; // retry count
        let now = Instant::now();
        'retry: loop {
            c += 1;
            if unlikely!(c > (1usize<<22)) {
                let el = now.elapsed();
                let ms: f64 = 1e3 * el.as_secs() as f64
                    + el.subsec_nanos() as f64 / 1e6;
                warn!("PUT (len {}) retries are looping: {:?} msec",
                        len, ms);
                c = 0;
                if ms > 1_000f64 {
                    warn!("PUT taking too long; skipping");
                    return;
                    //panic!("PUT looping too long; panic");
                }
            }
            let ret: i32 = extern_kvs_put(key, len as u64, value.0);
            // XXX make sure this matches with
            // mica-kvs.git/src/table.h enum put_reason
            match ret {
                50 => return,
                r => {
                    match r {
                        // table insertion failure = table too small
                        111 => println!("MICA failed to insert in table"),
                        // retry failed puts (due to fragmentation)
                        // NOTE this assumes other threads are performing DEL
                        112 => continue 'retry,
                        // Use this version to ack the failed put
                        //112 => println!("MICA failed to insert in heap"),
                        _ => println!("MICA failed with unknown: {}", ret),
                    }
                    //unsafe { intrinsics::abort(); }
                },
            }
            break;
        } // loop
    }
}

#[cfg(feature = "mica")]
#[cfg(feature = "extern_ycsb")]
fn get_object(key: u64) {
    unsafe {
        trace!("GET {:x}", key);
        if !extern_kvs_get(key) {
            //println!("GET failed on key 0x{:x}", key);
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
        extern_kvs_del(key);
        //if !extern_kvs_del(key) { println!("DEL failed on key 0x{:x}", key); }
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

#[link(name = "hiredisext")]
#[cfg(feature = "redis")]
#[cfg(feature = "extern_ycsb")]
extern {
    // must match signatures in hiredis.c
    fn extern_kvs_init();
    fn extern_kvs_put(key: u64, len: u64, buf: *const u8);
    fn extern_kvs_del(key: u64);
    fn extern_kvs_get(key: u64);
}
