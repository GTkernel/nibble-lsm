/*
 * Measure the performance of Nibble when threads allocate and hand
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

extern crate rand; // import before nibble
#[macro_use]
extern crate log;
extern crate test;
extern crate time;
extern crate clap;
extern crate crossbeam;

extern crate nibble;

use clap::{Arg, App, SubCommand};
use log::LogLevel;
use nibble::common::{Pointer,ErrorCode,rdrand};
use nibble::epoch;
use nibble::logger;
use nibble::memory;
use nibble::nib::{PutPolicy,Nibble};
use nibble::numa::{self,NodeId};
use nibble::sched::*;
use nibble::segment::{ObjDesc,SEGMENT_SIZE};
use rand::Rng;
use std::collections::VecDeque;
use std::mem;
use std::sync::{Arc,Barrier};
use std::sync::atomic::*;
use std::sync::mpsc::{channel,Sender,Receiver};
use std::thread::{self,JoinHandle};
use std::time::{Duration,Instant};

// we must use a concurrent queue to provide asynchronous passing of
// keys between threads. we import Aaron Turon's chase_lev, making two
// changes: increase default buffer size, and abort if the internal
// deque wants to resize (to avoid invoking malloc in critical path)
use nibble::chase_lev;

fn pairs(npairs: usize, objsize: usize) {
    let nib: Nibble = Nibble::new(53_687_091_200_usize);

    for sock in 0..numa::NODE_MAP.sockets() {
        nib.enable_compaction(NodeId(sock));
    }

    let mut guards = vec![];
    let mut tids: u64 = 0;
    let iters = 200;
    println!("iters {}", iters);

    // unique keys per thread pair.  not all will be active at the
    // same time, since pairs alloc/free in tight loops
    let per: usize = 10000;
    println!("keys per thread: {}", per);

    let barrier = Arc::new(Barrier::new(npairs*2));

    crossbeam::scope(|scope| {
        for _ in 0..npairs {

            //let (tx,rx) = channel::<u64>();
            let (mut worker, stealer) = chase_lev::deque();

            let tid = tids; tids += 1;
            let nib_ref = &nib;
            let b = barrier.clone();

            // Spawn sender/allocator
            let guard = scope.spawn(move || {
                let nib = &*nib_ref;
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

                for _ in 0..iters {
                    let mut key = start;

                    while key <= (start+ per as u64) {

                        // wait for other thread to catch up
                        let mut waits = 0usize;
                        while nib.exists(key) {
                            waits += 1;
                            assert!(waits < 2000,
                                    "sender waiting too long");
                        }

                        obj.key = key;
                        assert!(nib.put_where(&obj,
                                   PutPolicy::Specific(sock)).is_ok());
                        key += 1;

                        worker.push(key-1);
                    }
                }

                worker.push(0u64); // =done
            });
            guards.push(guard);

            let tid = tids; tids += 1;
            let nib_ref = &nib;
            let b = barrier.clone();

            // Spawn receiver/releaser
            let guard = scope.spawn(move || {
                println!("tid {}", tid);
                let nib = &*nib_ref;
                let mut many: usize = 0;

                unsafe { pin_cpu(tid as usize); }
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
                            assert!(nib.del_object(key).is_ok());
                            many += 1;
                            fails = 0usize;
                        },
                    }
                    assert!(fails < 1000,
                            "too many empty steals.. hung?");
                }
                let dur = now.elapsed();
                let sec = dur.as_secs();
                let ops: f64 = many as f64 / sec as f64;
                println!("tid {} perf: {:.2} kops/sec", tid, ops/1e3);
            });
            guards.push(guard);
        }
    });

    for g in guards {
        g.join();
    }
}

fn main() {
    logger::enable();
    let n: usize = 6;
    let s: usize = 100;
    println!("pairs {} obj.size {}", n, s);
    pairs(n,s);
}
