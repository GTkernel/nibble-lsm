/*
 * Measure the performance of LSM when threads allocate and hand
 * off objects amongst each other, e.g.,
 *
 *  Thread 1    Thread 2    Thread 3    Thread 4
 *   alloc   -->  free
 *                alloc  -->  free
 *   alloc   -->  free        alloc  -->  free
 *                alloc  -->  free
 *   alloc   -->  free        alloc  -->  free
 *                alloc  -->  free
 *                            alloc  -->  free
 *
 * or as pairs in isolation (currently implemented),
 *
 *  Thread 1a   Thread 1b   Thread 2a   Thread 2b
 *   alloc   -->  free       alloc   -->  free
 *   alloc   -->  free       alloc   -->  free
 *   alloc   -->  free       alloc   -->  free
 *
 */

#![feature(test)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]

extern crate rand; // import before kvs
#[macro_use]
extern crate log;
extern crate test;
extern crate time;
extern crate clap;
extern crate crossbeam;
extern crate num;

extern crate kvs;

use clap::{Arg, App, SubCommand};
use log::LogLevel;
use kvs::common::{Pointer,ErrorCode,rdrand};
use kvs::epoch;
use kvs::logger;
use kvs::memory;
use kvs::lsm::{PutPolicy,LSM};
use kvs::numa::{self,NodeId};
use kvs::sched::*;
use kvs::segment::{ObjDesc,SEGMENT_SIZE};
use rand::Rng;
use std::collections::VecDeque;
use std::mem;
use std::ptr;
use std::sync::{Arc,Barrier};
use std::sync::atomic::*;
use std::sync::mpsc::{channel,Sender,Receiver};
use std::thread::{self,JoinHandle};
use std::time::{Duration,Instant};

// we must use a concurrent queue to provide asynchronous passing of
// keys between threads. we import Aaron Turon's chase_lev, making two
// changes: increase default buffer size, and abort if the internal
// deque wants to resize (to avoid invoking malloc in critical path)
use kvs::chase_lev;

fn pairs(kvs: &LSM, npairs: usize, objsize: usize) {
    let mut guards = vec![];
    let mut tids: u64 = 0;
    let iters = 5000;
    println!("iters {}", iters);

    // unique keys per thread pair.  not all will be active at the
    // same time, since pairs alloc/free in tight loops
    let per: usize = 10000;
    println!("keys per thread: {}", per);

    let barrier = Arc::new(Barrier::new(npairs*2));
    let total_ops = AtomicUsize::new(0);

    let main_now = Instant::now();

    // Try to have all threads halt at same time, else missing overlap
    // will skew results.
    let mut stop_now: bool = false;

    crossbeam::scope(|scope| {
        for _ in 0..npairs {
            let tot_ops = &total_ops;
            let stop_ = Pointer(&stop_now as *const bool);

            //let (tx,rx) = channel::<u64>();
            let (mut worker, stealer) = chase_lev::deque();

            let tid = tids; tids += 1;
            let kvs_ref = &kvs;
            let b = barrier.clone();

            // Spawn sender/allocator
            let guard = scope.spawn(move || {
                let kvs = &*kvs_ref;
                let start = tid*(per as u64) + 1; // zero not a valid key

                let cpu = tid as usize;
                let sock = numa::NODE_MAP.sock_of(cpu).0;
                unsafe { pin_cpu(cpu); }
                println!("tid {} cpu {} sock {} start {}",
                         tid, cpu, sock, start);

                let mut value: Vec<u64> = Vec::with_capacity(objsize);
                unsafe { value.set_len(objsize); }
                let vptr = Pointer(value.as_ptr() as *const u8);
                let vlen = value.len() * 8;
                let mut obj = ObjDesc::new(0u64, vptr, vlen);

                b.wait();

                'all: for _ in 0..iters {
                    let mut key = start;

                    while key <= (start+ per as u64) {

                        // wait for other thread to catch up
                        if kvs.exists(key) {
                            let now_ = Instant::now();
                            loop {
                                if now_.elapsed().as_secs() > 2 {
                                    assert!(false,
                                            "sender waited too long");
                                }
                                if !kvs.exists(key) {
                                    break;
                                }
                            }
                        }

                        obj.key = key;
                        assert!(kvs.put_where(&obj,
                                   PutPolicy::Specific(sock)).is_ok());
                        key += 1;

                        worker.push(key-1);

                        if 0 == (iters % 100) { unsafe {
                            if ptr::read_volatile(stop_.0) {
                                println!("tid {} stopping", tid);
                                break 'all;
                            }
                        }}
                    }
                }

                worker.push(0u64); // =done
                unsafe {
                    ptr::write_volatile(stop_.0 as *mut _, true);
                }
            });
            guards.push(guard);

            let tid = tids; tids += 1;
            let kvs_ref = &kvs;
            let b = barrier.clone();

            // Spawn receiver/releaser
            let guard = scope.spawn(move || {
                let kvs = &*kvs_ref;
                let mut many: usize = 0;

                let cpu: usize = tid as usize;
                let sock: usize = numa::NODE_MAP.sock_of(cpu).0;
                println!("tid {} cpu {} sock {}",
                            tid, cpu, sock);
                unsafe { pin_cpu(cpu); }
                b.wait();

                let now = Instant::now();

                let mut fails = 0usize;
                loop {
                    match stealer.steal() {
                        chase_lev::Steal::Empty => {
                            fails += 1;
                            continue;
                        },
                        chase_lev::Steal::Abort => {
                            fails += 1;
                            continue;
                        },
                        chase_lev::Steal::Data(key) => {
                            if key == 0u64 {
                                break;
                            }
                            assert!(kvs.del_object(key).is_ok());
                            many += 1;
                            fails = 0usize;
                        },
                    }
                    assert!(fails < 1000,
                            "too many empty steals.. hung?");
                }
                let dur = now.elapsed();
                let sec: f64 = dur.as_secs() as f64 +
                                dur.subsec_nanos() as f64 / 1e9;
                if sec < 5f64 {
                    println!("Warning: test ran for < 5 seconds");
                }
                // *2 because each delete has an associated alloc
                let ops: usize = (2f64 * many as f64 / sec) as usize;
                println!("tid {} sec {:.2} perf: {:.2} kops/sec",
                         tid, sec, ops as f64/1e3);
                tot_ops.fetch_add(ops, Ordering::Relaxed);
            });
            guards.push(guard);
        }
    });

    for g in guards {
        g.join();
    }

    let elapsed = main_now.elapsed();
    let sec: f64 = elapsed.as_secs() as f64 +
        elapsed.subsec_nanos() as f64 / 1e9;

    let kops = total_ops.load(Ordering::Relaxed) as f64 / 1e3;
    println!("Total {:.2} kop/sec ({:.2} sec)", kops, sec);
}

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
    let matches = App::new("LSM pipeline allocation benchmark.")
        .arg(Arg::with_name("min")
             .long("min")
             .help("min thread pairs")
             .takes_value(true))
        .arg(Arg::with_name("max")
             .long("max")
             .help("max thread pairs")
             .takes_value(true))
        .arg(Arg::with_name("incr")
             .long("incr")
             .help("incr thread pairs")
             .takes_value(true))
        .arg(Arg::with_name("size")
             .long("size")
             .help("object size [bytes]")
             .takes_value(true))
        .arg(Arg::with_name("cap")
             .long("cap")
             .help("total log capacity [GiB]")
             .takes_value(true))
        .get_matches();

    let from: usize = arg_as_num(&matches, "min");
    let to: usize   = arg_as_num(&matches, "max");
    let incr: usize = arg_as_num(&matches, "incr");
    let s: usize    = arg_as_num(&matches, "size");
    let mut cap: usize  = arg_as_num(&matches, "cap");
    cap *= 1usize<<30;

    logger::enable();
    let kvs = LSM::new(cap);

    // turn on compaction for all sockets
    for sock in 0..numa::NODE_MAP.sockets() {
        kvs.enable_compaction(NodeId(sock));
    }

    let mut n: usize = from;
    while n <= to {
        println!("pairs {} obj.size {}", n, s);
        pairs(&kvs,n,s);
        n += incr;
    }
}
