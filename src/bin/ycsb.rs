#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]
#![feature(core_intrinsics)]

// NOTE XXX
// Due to rounding errors when we insert keys in parallel during the
// setup phase, some keys may not actually exist. It is best to
// double check this happens infrequently; if it is infrequent, we can
// ignore them.

extern crate rand; // import before kvs
#[macro_use]
extern crate log;
extern crate time;
extern crate clap;
extern crate num;
extern crate crossbeam;
extern crate parking_lot as pl;

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
use rand::Rng;
use std::collections::VecDeque;
use std::mem;
use std::intrinsics;
use std::sync::Arc;
use std::sync::atomic::*;
use std::thread::{self,JoinHandle};
use std::time::{Instant,Duration};
use std::ptr;
use std::cmp;

//==----------------------------------------------------------------==
//  Build-based functions
//  Compile against LSM, or exported functions.
//==----------------------------------------------------------------==

/// Used to create the stack-based buffers for holding GET output.
pub const MAX_KEYSIZE: usize = 1usize << 10;

//
// LSM redirection
//

#[cfg(not(feature = "extern_ycsb"))]
static mut KVS: Pointer<LSM> = Pointer(0 as *const LSM);

#[cfg(not(feature = "extern_ycsb"))]
fn kvs_init(config: &Config) {
    let mut kvs =
        Box::new(LSM::new2(config.total, config.records*2));
        //Box::new(LSM::new2(config.total, 1usize<<30));
    if config.comp {
        info!("Enabling compaction");
        for node in 0..numa::NODE_MAP.sockets() {
            kvs.enable_compaction(NodeId(node));
        }
    } else {
        warn!("Compaction NOT enabled");
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

#[inline(always)]
#[cfg(not(feature = "extern_ycsb"))]
fn get_object(key: u64) {
    let kvs: &LSM = unsafe { &*KVS.0 };
    let mut buf: [u8;MAX_KEYSIZE] =
        unsafe { mem::uninitialized() };
    let _ = kvs.get_object(key, &mut buf);
    //if let Err(e) = kvs.get_object(key, &mut buf) {
        //warn!("{:?} {:x}", e, key);
        //unsafe { intrinsics::abort(); }
    //}
    //unsafe {
        //let v: u64 = ptr::read_volatile(buf.as_ptr() as *const u64);
    //}
}

//
// Other KVS redirection (external library)
//

// Link against libmicaext.so which is built in
// mica-kvs.git
// and installed to /usr/local/lib/
#[link(name = "micaext")]
#[cfg(feature = "mica")]
#[cfg(feature = "extern_ycsb")]
extern {
    fn extern_kvs_init();
    fn extern_kvs_put(key: u64, len: u64, buf: *const u8) -> i32;
    fn extern_kvs_del(key: u64) -> bool;
    fn extern_kvs_get(key: u64) -> bool;
}

// Link against libramcloudext.so which is built in
// make -C ramcloud-scale-hacks.git/src/
// and installed to /usr/local/lib/
#[link(name = "ramcloudext")]
#[cfg(feature = "rc")]
#[cfg(feature = "extern_ycsb")]
extern {
    // must match signatures in ObjectManagerExported.cc
    fn extern_kvs_init();
    fn extern_kvs_put(key: u64, len: u64, buf: *const u8);
    fn extern_kvs_del(key: u64);
    fn extern_kvs_get(key: u64);
}

// Link against libhiredis.so which is built from
// https://github.gatech.edu/kernel/hiredis.git
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

// Link against libmasstree.so which is built from
// https://github.gatech.edu/kernel/masstree.git
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

#[cfg(feature = "extern_ycsb")]
fn kvs_init(config: &Config) {
    unsafe {
        extern_kvs_init();
    }
}

// method interface is the same for RAMCloud and LevelDB's hacks
// so we reuse it.
#[cfg(any(feature = "rc", feature = "redis", feature = "masstree"))]
#[cfg(feature = "extern_ycsb")]
fn put_object(key: u64, value: Pointer<u8>, len: usize, sock: usize) {
    // we ignore 'sock' b/c RAMCloud is NUMA-agnostic
    // (MICA might, though...)
    unsafe {
        trace!("PUT {:x} len {}", key, len);
        extern_kvs_put(key, len as u64, value.0);
    }
}

#[cfg(any(feature = "rc", feature = "redis", feature = "masstree"))]
#[cfg(feature = "extern_ycsb")]
fn get_object(key: u64) {
    unsafe {
        trace!("GET {:x}", key);
        extern_kvs_get(key);
    }
}

#[cfg(any(feature = "rc", feature = "redis", feature = "masstree"))]
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


//==----------------------------------------------------------------==
//  The rest of the benchmark
//==----------------------------------------------------------------==

#[derive(Debug,Clone,Copy)]
enum YCSB {
    A, B, C, WR,
}

struct WorkloadGenerator {
    config: Config,
    sockets: usize,
}

impl WorkloadGenerator {

    pub fn new(config: Config) -> Self {
        let n = config.records;
        kvs_init(&config);
        info!("WorkloadGenerator {:?}", config);
        WorkloadGenerator {
            config: config,
            sockets: numa::NODE_MAP.sockets(),
        }
    }

    fn __setup(&mut self, parallel: bool) {
        let size = self.config.size;

        let now = Instant::now();
        if parallel {
            let threads_per = 8;
            info!("Inserting {} objects of size {} with {} threads",
                  self.config.records, self.config.size,
                  threads_per * self.sockets);
            let pernode = self.config.records / self.sockets;
            let mut handles = vec![];
            for t in 0..(threads_per * self.sockets) {
                let sock = t / threads_per;
                let per_thread = pernode / threads_per;
                let start_key: u64 =
                    (sock*pernode + (t%threads_per)*per_thread) as u64;
                let end_key: u64 = start_key + (per_thread as u64);
                handles.push( thread::spawn( move || {
                    let value = memory::allocate::<u8>(size);
                    let v = Pointer(value as *const u8);
                    debug!("range [{},{}) on socket {}",
                        start_key, end_key, sock);
                    for key in start_key..end_key {
                        // Avoid inserting value zero (0)
                        put_object(key+1, v, size, sock);
                    }
                    unsafe { memory::deallocate(value, size); }
                }));
            }
            for handle in handles {
                let _ = handle.join();
            }
        }

        // use one thread to insert all keys
        else {
            let start_key: u64 = 1u64;
            let end_key: u64 = self.config.records as u64;
            let value = memory::allocate::<u8>(size);
            let v = Pointer(value as *const u8);
            info!("range [{},{})", start_key, end_key);
            for key in start_key..end_key {
                // NOTE socket param unused b/c we assume
                // RAMCloud is unaware and also has a single lock for
                // insertion (many threads inserting would be slow)
                // Avoid inserting value zero (0)
                put_object(key+1, v, size, 0 /* unused */);
            }
            unsafe { memory::deallocate(value, size); }
        }

        let sec = now.elapsed().as_secs();
        info!("insertion took {} seconds", sec);
    }

    // sequential insertion (ramcloud)
    #[cfg(any(feature = "rc"))]
    #[cfg(feature = "extern_ycsb")]
    pub fn setup(&mut self) {
        self.__setup(false);
    }

    // parallel insertion (kvs, mica, others..)
    #[cfg(any(feature = "mica", feature="redis", feature = "masstree", not(feature = "extern_ycsb")))]
    pub fn setup(&mut self) {
        self.__setup(true);
    }

    /// Run at specified op per second (ops).
    /// Zero means no throttling.
    pub fn run(&mut self) {
        let read_threshold = self.config.read_pct;
        assert!(read_threshold <= 100);

        let pernode = self.config.records / self.sockets;
        let size = self.config.size;

        // each op should have this latency (in nsec) or less
        let nspo: u64 = match self.config.ops {
            0u64 => 0u64,
            o => 1_000_000_000u64 / o,
        };
        // and the equivalent in cycles
        let cpo = clock::from_nano(nspo);
        debug!("nspo {} cpo {}", nspo, cpo);

        info!("Starting experiment");
        let mut counter = 0u64;
        let start = unsafe { clock::rdtsc() }; // for throttling

        let mut tic = unsafe { clock::rdtsc() }; // report performance
        let mut per_loop = 0u64; // ops performed per report

        let duration = self.config.dur;

        let accum = Arc::new(AtomicUsize::new(0));
        let mut threadcount: Vec<usize>;

        let cpus_pernode = numa::NODE_MAP.cpus_in(NodeId(0));
        let sockets = numa::NODE_MAP.sockets();
        let ncpus = cpus_pernode * sockets;
        info!("cpus per node: {}", cpus_pernode);
        info!("sockets:       {}", sockets);
        info!("ncpus:         {}", ncpus);

        //threadcount = vec![self.config.threads];
        // specific number of threads only
        //threadcount = vec![240];
        // max number of threads
        //threadcount = vec![cpus_pernode*sockets];
        // power of 2   1, 2, 4, 8, 16, 32, 64, 128, 256
        //threadcount = (0usize..9).map(|e|1usize<<e).collect();
        // incr of 4    1, 4, 8, 12, 16, ...
        //threadcount = (0usize..65).map(|e|if e==0 {1} else {4*e}).collect();
        // incr of 2    1, 2, 4, 6, 8, ...
        //threadcount = (0usize..130).map(|e|if e==0 {1} else {2*e}).collect();
        // incr of 1    1, 2, 3, 4, 5, ...
        //threadcount = (1usize..261).collect();
        // incr of x    where x= cpus/socket
        threadcount = (1_usize..(sockets+1))
            .map(|e|cpus_pernode*e).collect();
        info!("thread counts to use: {:?}", threadcount);
        let max_threads: usize = *threadcount.last().unwrap() as usize;

        // do setup once, to save on this cost across iterations
        // need some hackiness because Rust makes it hard to share
        // variables without synchronization. we're ok because each
        // thread only accesses one slot
        let mut gens: Vec<Box<DistGenerator>> =
            Vec::with_capacity(max_threads);
        unsafe {
            gens.set_len(max_threads);
        }
        // as long as gens doesn't resize, this should be fine
        let gens_ptr = Pointer(gens.as_ptr());

        // used for threads to pick a slot and pin to CPU for
        // forcing (hoping for) local page allocation
        let idx = Arc::new(AtomicUsize::new(0));

        let now = Instant::now();
        info!("initializing all key generators...");
        crossbeam::scope(|scope| {

            let mut guards = vec![];

            for _ in 0..max_threads {

                let guard = scope.spawn(|| {
                    let cpu = idx.fetch_add(1, Ordering::Relaxed);
                    unsafe {
                        pin_cpu(cpu);
                    }

                    // create the object
                    let item: Box<DistGenerator>;
                    item = match self.config.dist {
                        Dist::Zipfian(s) =>
                            Box::new(Zipfian::new(
                                self.config.records as u32, s)),
                        Dist::Uniform => {
                            let n = self.config.records as u32;
                            Box::new( Uniform::new(n) )
                        },
                    };

                    // add it to the global vector
                    unsafe {
                        let mut slot = gens_ptr.0 .offset(cpu as isize)
                            as *mut Box<DistGenerator>;
                        ptr::write(slot, item);
                    }
                });
                guards.push(guard);
            }

            // wait for all threads to complete
            for g in guards {
                g.join();
            }
        }); // whew! done
        let sec = now.elapsed().as_secs();
        info!("key gen took {} seconds", sec);

        // Run the experiment multiple times using different sets of
        // threads
        for nthreads in threadcount {
            for g in &mut gens {
                (**g).reset();
            }

            let mut handles: Vec<JoinHandle<()>> = Vec::with_capacity(nthreads);
            info!("running with {} threads ---------", nthreads);
            accum.store(nthreads, Ordering::Relaxed);

            // Create CPU binding strategy
            let mut cpus: VecDeque<usize> = VecDeque::with_capacity(ncpus);
            match self.config.cpu {
                CPUPolicy::Global => {
                    for i in 0..ncpus {
                        cpus.push_back(i);
                    }
                    // Knuth shuffle
                    let mut rng = rand::thread_rng();
                    for i in 0..ncpus {
                        let o = (rng.next_u32() as usize % (ncpus-i)) + i;
                        cpus.swap(i, o);
                    }
                },
                CPUPolicy::SocketRR => {
                    for i in 0..cpus_pernode {
                        for sock in 0..sockets {
                            let r = numa::NODE_MAP.cpus_of(NodeId(sock)).get();
                            cpus.push_back(r[i]);
                        }
                    }
                },
                CPUPolicy::Incremental => {
                    for i in 0..ncpus {
                        cpus.push_back(i);
                    }
                },
            }
            let cpus = pl::Mutex::new(cpus);

            // Spawn each of the threads in this set
            let mut guards = vec![];
            crossbeam::scope(|scope| {

                for _ in 0..nthreads {

                    let guard = scope.spawn( || {
                        let cpu = match cpus.lock().pop_front() {
                            None => panic!("No cpu for this thread?"),
                            Some(c) => c,
                        };
                        let config = self.config.clone();

                        let laccum = accum.clone();
                        laccum.fetch_sub(1, Ordering::Relaxed);
                        unsafe { pin_cpu(cpu); }

                        let value = memory::allocate::<u8>(size);
                        let v = Pointer(value as *const u8);

                        let sock = numa::NODE_MAP.sock_of(cpu);
                        debug!("thread on cpu {} sock {:?} INIT",
                               cpu, sock);

                        // use your own generator for key accesses
                        let keygen: &mut Box<DistGenerator> = unsafe {
                            &mut *(gens_ptr.0 .offset(cpu as isize)
                                as *mut Box<DistGenerator>)
                        };

                        // make one for determining read/writes
                        // don't want rdrand in the critical path.. slow
                        let mut rwgen: Box<DistGenerator> =
                            Box::new(Uniform::new(100));

                        // wait for all other threads to spawn
                        // after this, accum is zero
                        while accum.load(Ordering::Relaxed) > 0 { ; }
                        debug!("thread on cpu {} sock {:?} START",
                               cpu, sock);

                        // socket to apply PUT
                        let mut sockn = 0_usize;

                        // main loop (do warmup first)
                        // pick one of the loop headers (inf vs 2x iters)

                        for x in 0..2 {

                        //let x = 1;
                        //loop {

                            let mut ops = 0usize;
                            let now = Instant::now();
                            let iters = 10000usize;

                            while (now.elapsed().as_secs() as usize)
                                    < duration {
                                for _ in 0..iters {
                                    let isread = rwgen.next()
                                        < read_threshold as u32;
                                    let key = keygen.next() as u64 + 1;
                                    if isread {
                                        get_object(key);
                                    } else {
                                        let sock = match config.puts {
                                            PutPolicy::GlobalRR => {
                                                sockn = (sockn + 1)
                                                    % sockets; sockn
                                            },
                                            PutPolicy::Local => sock.0,
                                        };
                                        put_object(key,v,config.size,sock);
                                    }
                                }
                                ops += iters;

                                // throttling, if enabled
                                if cpo > 0 {
                                    let next = start + ops as u64 * cpo;
                                    while unsafe { clock::rdtsc() } < next {;}
                                }

                            } // while some time

                            if x == 1 {
                                let dur = now.elapsed();
                                let nsec = dur.as_secs() * 1000000000u64
                                    + dur.subsec_nanos() as u64;
                                let ops = (ops as f64) /
                                    ((nsec as f64)/1e9);
                                println!("# {} {} {:.3}", cpu, nthreads, ops);
                                // aggregate the performance
                                accum.fetch_add(ops as usize,
                                                Ordering::Relaxed);
                            }
                        }
                    }); // spawn
                guards.push(guard);
                } // for threads
            }); // crossbeam
            for g in guards {
                g.join();
            }
            println!("# total ops/sec {}",
                     accum.load(Ordering::Relaxed));
        }

    } // run()
}

#[derive(Clone,Copy,Debug)]
enum CPUPolicy {
    Global,
    SocketRR, // round-robin among sockets
    Incremental, // fill each socket
}

#[derive(Clone,Copy,Debug)]
enum PutPolicy {
    /// Puts append to heads on each socket in RR fashion
    GlobalRR,
    /// Puts only append to heads on local socket
    Local,
}

#[derive(Debug,Clone,Copy)]
enum Dist {
    /// Contained value is s (exponent modifier)
    Zipfian(f64),
    Uniform,
}

fn extract_dist(args: &clap::ArgMatches) -> Dist {
    match args.value_of("dist") {
        None => panic!("Specify dist"),
        Some(s) => {
            if s == "zipfian" {
                // s value from YCSB itself
                Dist::Zipfian(0.99_f64)
            } else if s == "uniform" {
                Dist::Uniform
            } else {
                panic!("unknown distribution");
            }
        },
    }
}

fn extract_cpu(args: &clap::ArgMatches) -> CPUPolicy {
    match args.value_of("cpu") {
        None => panic!("Specify CPU policy"),
        Some(s) => {
            if s == "global" {
                CPUPolicy::Global
            } else if s == "rr" {
                CPUPolicy::SocketRR
            } else if s == "incr" {
                CPUPolicy::Incremental
            } else {
                panic!("invalid CPU policy");
            }
        },
    }
}

fn extract_puts(args: &clap::ArgMatches) -> PutPolicy {
    match args.value_of("put") {
        None => panic!("Specify PUT policy"),
        Some(s) => {
            if s == "globalrr" {
                PutPolicy::GlobalRR
            } else if s == "local" {
                PutPolicy::Local
            } else {
                panic!("invalid PUT policy");
            }
        },
    }
}

// TODO: setup configuration, how to allocate objects across sockets
#[derive(Debug,Clone,Copy)]
struct Config {
    /// Amount of memory to use for kvs
    total: usize,
    /// if None, custom workload
    ycsb: Option<YCSB>,
    /// number of objects
    records: usize,
    /// size of each object
    size: usize,
    dist: Dist,
    /// 0-100 (1-read_pct = write pct)
    read_pct: usize,
    /// operations per second to sustain. 0 = unthrottled
    ops: u64,
    cpu: CPUPolicy,
    puts: PutPolicy,
    /// Number of threads to run experiment with
    threads: usize,
    /// How long to run the experiment in seconds
    dur: usize,
    /// Whether to enable compaction
    comp: bool,
}

impl Config {

    pub fn ycsb(total: usize, ops: u64, w: YCSB,
                cpu: CPUPolicy, puts: PutPolicy,
                time: usize, threads: usize,
                comp: bool) -> Self {
        let rc: usize = 1000;
        Self::ycsb_more(total, ops, w, rc, cpu,
                        puts,time,threads, comp)
    }

    // more records
    pub fn ycsb_more(total: usize, ops: u64, w: YCSB,
                     records: usize,
                     cpu: CPUPolicy, puts: PutPolicy,
                     time: usize, threads: usize,
                     comp: bool) -> Self {
        let rs: usize = 100;
        let rp: usize = match w {
            YCSB::A => 50,
            YCSB::B => 95,
            YCSB::C => 100,
            YCSB::WR => 0,
        };
        Config {
            total: total,
            ycsb: Some(w),
            records: records,
            size: rs,
            dist: Dist::Zipfian(0.99f64),
            read_pct: rp,
            ops: ops,
            cpu: cpu,
            puts: puts,
            threads: threads,
            dur: time,
            comp: comp,
        }
    }

    // directly construct it
    pub fn custom(total: usize, ops: u64, records: usize,
                  size: usize, dist: Dist,
                  read_pct: usize,
                  cpu: CPUPolicy, puts: PutPolicy,
                  time: usize, threads: usize,
                  comp: bool) -> Self {
        Config {
            total: total,
            ycsb: None,
            records: records,
            size: size,
            dist: dist,
            read_pct: read_pct,
            ops: ops,
            cpu: cpu,
            puts: puts,
            threads: threads,
            dur: time,
            comp: comp,
        }
    }
}

/// Convert an argument as number.
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

    let matches = App::new("ycsb")
        .arg(Arg::with_name("ycsb")
             .long("ycsb").takes_value(true))
        .arg(Arg::with_name("size")
             .long("size").takes_value(true))
        .arg(Arg::with_name("capacity")
             .long("capacity").takes_value(true))
        .arg(Arg::with_name("records")
             .long("records").takes_value(true))
        .arg(Arg::with_name("readpct")
             .long("readpct").takes_value(true))
        .arg(Arg::with_name("dist")
             .long("dist").takes_value(true))
        .arg(Arg::with_name("ops")
             .long("ops").takes_value(true))
        .arg(Arg::with_name("put")
             .long("put").takes_value(true))
        .arg(Arg::with_name("cpu")
             .long("cpu").takes_value(true))
        .arg(Arg::with_name("time")
             .long("time").takes_value(true))
        .arg(Arg::with_name("threads")
             .long("threads").takes_value(true))
        .arg(Arg::with_name("compaction")
             .long("compaction"))
        .get_matches();

    let config = match matches.value_of("ycsb") {

        // Custom Configuration
        None => {
            let size     = arg_as_num::<usize>(&matches, "size");
            let capacity = arg_as_num::<usize>(&matches, "capacity");
            let ops      = arg_as_num::<u64>(&matches, "ops");
            let records  = arg_as_num::<usize>(&matches, "records");
            let readpct  = arg_as_num::<usize>(&matches, "readpct");
            let threads  = arg_as_num::<usize>(&matches, "threads");
            let time     = arg_as_num::<usize>(&matches, "time");

            let puts = extract_puts(&matches);
            let cpu  = extract_cpu(&matches);
            let dist = extract_dist(&matches);

            let comp = matches.is_present("compaction");

            Config::custom(capacity, ops, records,
                           size, dist, readpct, cpu,
                           puts, time, threads, comp)
        },

        // YCSB-Specific Configuration
        Some(s) => {
            let ycsb = match s {
                "A" => YCSB::A,
                "B" => YCSB::B,
                "C" => YCSB::C,
                "WR" => YCSB::WR,
                _ => panic!("unknown YCSB configuration"),
            };

            let capacity = arg_as_num::<usize>(&matches, "capacity");
            let ops      = arg_as_num::<u64>(&matches, "ops");
            let threads  = arg_as_num::<usize>(&matches, "threads");
            let time     = arg_as_num::<usize>(&matches, "time");

            let cpu = extract_cpu(&matches);
            let puts = extract_puts(&matches);

            // optional argument
            let records = match matches.value_of("records") {
                None => None,
                Some(s) => match usize::from_str_radix(s,10) {
                    Err(_) => panic!("records NaN"),
                    Ok(s) => Some(s),
                },
            };

            let comp = matches.is_present("compaction");

            match records {
                None => Config::ycsb(capacity, ops, ycsb,
                                     cpu, puts, time,threads,comp),
                Some(r) => Config::ycsb_more(capacity,
                                             ops, ycsb, r, cpu,
                                             puts,time,threads,comp),
            }
        },
    };

    // default size 1KiB

    // A rc=1000 size=1kb 50:50 zipfian
    // B rc=1000 size=1kb 95:05 zipfian
    // C rc=1000 size=1kb  1:0  zipfian

    // W rc=1000 size=1kb  0:1  zipfian

    // or user-defined if you omit --ycsb
    // --records --size --readpct --dist --capacity --ops

    println!("NOTE -- THIS CODE VERSION IGNORES MANY CMD PARAMS");

    let mut gen = WorkloadGenerator::new(config);
    gen.setup();

    // filling the db takes a long time (20-60 minutes!)
    // so we reuse it across runs
    let dists: Vec<Dist> = vec![Dist::Uniform, Dist::Zipfian(0.99)];
    //let dists: Vec<Dist> = vec![Dist::Uniform];

    let readpct: Vec<usize> = vec![100,95,50,0];
    //let readpct: Vec<usize> = vec![100];

    for d in &dists {
        gen.config.dist = *d;
        info!("Distribution: {:?}", d);
        for r in &readpct {
            info!("Read PCT: {:?}", r);
            gen.config.read_pct = *r;
            gen.run();
        }
    }
}
