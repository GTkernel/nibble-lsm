#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]

/// Code ported from RAMCloud src/ClusterPerf.cc::ZipfianGenerator

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
use std::sync::Arc;
use std::sync::atomic::*;
use std::thread::{self,JoinHandle};
use std::time::{Instant,Duration};
use std::ptr;

//==----------------------------------------------------------------==
//  Build-based functions
//  Compile against Nibble, or exported functions.
//==----------------------------------------------------------------==

//
// Nibble redirection
//

#[cfg(not(feature = "extern_ycsb"))]
static mut NIBBLE: Pointer<Nibble> = Pointer(0 as *const Nibble);

#[cfg(not(feature = "extern_ycsb"))]
fn kvs_init(config: &Config) {
    let mut nibble =
        Box::new(Nibble::new2(config.total, config.records*2));
        //Box::new(Nibble::new2(config.total, 1usize<<30));
    if config.comp {
        info!("Enabling compaction");
        for node in 0..numa::NODE_MAP.sockets() {
            nibble.enable_compaction(NodeId(node));
        }
    } else {
        warn!("Compaction NOT enabled");
    }
    unsafe {
        let p = Box::into_raw(nibble);
        NIBBLE = Pointer(p);
    }
}

#[cfg(not(feature = "extern_ycsb"))]
fn put_object(key: u64, value: Pointer<u8>, len: usize, sock: usize) {
    let nibble: &Nibble = unsafe { &*NIBBLE.0 };
    let obj = ObjDesc::new(key, value, len);
    let nibnode = nib::PutPolicy::Specific(sock);
    while let Err(e) = nibble.put_where(&obj, nibnode) {
        match e {
            ErrorCode::OutOfMemory => {},
            _ => panic!("Error: {:?}", e),
        }
    }
}

#[cfg(not(feature = "extern_ycsb"))]
fn get_object(key: u64) {
    let nibble: &Nibble = unsafe { &*NIBBLE.0 };
    if let (Err(e),_) = nibble.get_object(key) {
        warn!("Error: {:?}", e);
    }
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
    fn extern_kvs_put(key: u64, len: u64, buf: *const u8);
    fn extern_kvs_del(key: u64);
    fn extern_kvs_get(key: u64);
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

#[cfg(feature = "extern_ycsb")]
fn kvs_init(config: &Config) {
    unsafe {
        extern_kvs_init();
    }
}

#[cfg(feature = "extern_ycsb")]
fn put_object(key: u64, value: Pointer<u8>, len: usize, sock: usize) {
    // we ignore 'sock' b/c RAMCloud is NUMA-agnostic
    // (MICA might, though...)
    unsafe {
        trace!("PUT {:x} len {}", key, len);
        extern_kvs_put(key, len as u64, value.0);
    }
}

#[cfg(feature = "extern_ycsb")]
fn get_object(key: u64) {
    unsafe {
        trace!("GET {:x}", key);
        extern_kvs_get(key);
    }
}

#[cfg(feature = "extern_ycsb")]
fn del_object(key: u64) {
    unsafe {
        trace!("DEL {:x}", key);
        extern_kvs_del(key);
    }
}


//==----------------------------------------------------------------==
//  The rest of the benchmark
//==----------------------------------------------------------------==

trait DistGenerator {
    fn next(&mut self) -> u32;
    fn reset(&mut self);
}

#[derive(Debug,Clone,Copy)]
struct Zipfian {
    n: u32,
    theta: f64,
    alpha: f64,
    zetan: f64,
    eta: f64,
}

impl Zipfian {

    pub fn new(n: u32, s: f64) -> Self {
        let theta: f64 = s;
        let zetan: f64 = Self::zeta(n as u64, theta);
        Zipfian {
            n: n, theta: theta,
            alpha: 1f64 / (1f64 - theta),
            zetan: zetan,
            eta: (1f64 - (2f64 / (n as f64)).powf(1f64-theta)) /
                (1f64 - Self::zeta(2u64, theta) / zetan),
        }
    }

    /// Compute H(N,s), the generalized Nth harmonic number
    pub fn zeta(n: u64, theta: f64) -> f64 {
        let mut sum: f64 = 0f64;
        for x in 0u64..n {
            sum += 1f64 / ((x+1) as f64).powf(theta);
        }
        sum
    }
}

impl DistGenerator for Zipfian {

    #[inline(always)]
    fn next(&mut self) -> u32 {
        let u: f64 = unsafe { rdrandq() as f64 } / 
            (std::u32::MAX as f64);
        let uz: f64 = u * self.zetan;
        if uz < 1f64 { 0u32 }
        else if uz < (1f64 + 0.5f64.powf(self.theta)) { 1u32 }
        else {
            ((self.eta*u - self.eta + 1f64).powf(self.alpha)
             * (self.n as f64)) as u32
        }
    }
    fn reset(&mut self) { }
}

struct ZipfianArray {
    n: u32,
    /// Given we execute for short periods in our experiments, we
    /// won't need to generate all data points. 'n' is the total
    /// quantity of items we would access given infinite time. 'upto'
    /// is how many operations we'll realistically perform given the
    /// duration of the experiment.
    upto: Option<u32>,
    arr: Vec<u32>,
    next: u32,
}

impl ZipfianArray {

    pub fn new(n: u32, s: f64) -> Self {
        let many = (n*4) as usize;
        let mut v: Vec<u32> = Vec::with_capacity(many);
        let mut zip = Zipfian::new(n, s);
        for _ in 0..many {
            v.push(zip.next());
        }
        // Knuth shuffle
        for i in 0..many {
            let r = unsafe { rdrand() };
            let o = (r as usize % (many-i)) + i;
            v.swap(i as usize, o as usize);
        }
        ZipfianArray { n: n, upto: None, arr: v, next: 0 }
    }

    pub fn new_capped(n: u32, s: f64, upto: u32) -> Self {
        let mut v: Vec<u32> = Vec::with_capacity(upto as usize);
        let mut zip = Zipfian::new(n, s);
        for _ in 0..upto {
            v.push(zip.next());
        }
        // Knuth shuffle
        for i in 0..upto {
            let r = unsafe { rdrand() };
            let o = (r % (upto-i)) + i;
            v.swap(i as usize, o as usize);
        }
        ZipfianArray { n: n, upto: Some(upto), arr: v, next: 0 }
    }
}

impl DistGenerator for ZipfianArray {

    #[inline(always)]
    fn next(&mut self) -> u32 {
        self.next = (self.next + 1) % self.n;
        if self.upto.is_some() {
            assert!(self.next < self.upto.unwrap(),
                    "upto exceeded. increase, or shorten expmt duration");
        }
        self.arr[self.next as usize] as u32
    }
    fn reset(&mut self) {
        self.next = 0;
    }
}

struct Uniform {
    n: u32,
    arr: Vec<u32>,
    next: u32,
}

impl Uniform {

    pub fn new(n: u32) -> Self {
        let mut v: Vec<u32> = Vec::with_capacity(n as usize);
        //info!("Generating {} keys...", n);
        for x in 0..n {
            v.push(x as u32);
        }
        //info!("Shuffling {} keys...", n);
        //let mut rng = rand::thread_rng();
        //rng.shuffle(&mut v);
        common::shuffle(&mut v);
        Uniform { n: n, arr: v, next: 0 }
    }
}

impl DistGenerator for Uniform {

    #[inline(always)]
    fn next(&mut self) -> u32 {
        self.next = (self.next + 1) % self.n;
        self.arr[self.next as usize]
    }

    fn reset(&mut self) {
        self.next = 0;
    }
}

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

        info!("Inserting {} objects of size {}",
              self.config.records, self.config.size);

        let now = Instant::now();
        if parallel {
            let pernode = self.config.records / self.sockets;
            let mut handles: Vec<JoinHandle<()>> =
                Vec::with_capacity(self.sockets);
            for sock in 0..self.sockets {
                let start_key: u64 = (sock*pernode) as u64;
                let end_key: u64   = start_key + (pernode as u64);
                handles.push( thread::spawn( move || {
                    let value = memory::allocate::<u8>(size);
                    let v = Pointer(value as *const u8);
                    info!("range [{},{}) on socket {}",
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
    #[cfg(feature = "rc")]
    #[cfg(feature = "extern_ycsb")]
    pub fn setup(&mut self) {
        self.__setup(false);
    }

    // parallel insertion (nibble, mica)
    #[cfg(any(feature = "mica", not(feature = "extern_ycsb")))]
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
        //threadcount = vec![1];
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
        // incr of 18    18,36,...
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
                            Box::new(ZipfianArray::new_capped(
                                self.config.records as u32, s,
                                (self.config.dur as f64*(2f64*4e6)) as u32
                                )),
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
                            let r = numa::NODE_MAP.cpus_of(NodeId(sock));
                            cpus.push_back(r.start + i);
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
                            let iters = 1000usize;

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
    /// Amount of memory to use for nibble
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

    let mut gen = WorkloadGenerator::new(config);
    gen.setup();
    gen.run();
}
