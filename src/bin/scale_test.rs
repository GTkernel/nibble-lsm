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

extern crate nibble;

use clap::{Arg, App, SubCommand};
use log::LogLevel;
use nibble::common::{ErrorCode,rdrand};
use nibble::epoch;
use nibble::logger;
use nibble::memory;
use nibble::nib::{PutPolicy,Nibble};
use nibble::numa::{self,NodeId};
use nibble::sched::*;
use nibble::segment::{ObjDesc,SEGMENT_SIZE};
use nibble::cuckoo;
use rand::Rng;
use std::mem;
use std::sync::Arc;
use std::sync::atomic::*;
use std::thread::{self,JoinHandle};
use std::collections::VecDeque;
use std::time::Duration;
use std::time::Instant;

#[derive(Clone,Copy,Debug)]
enum CPUPolicy {
    Random,
    SocketRR, // round-robin among sockets
    Incremental, // fill each socket
}

#[derive(Clone,Copy,Debug)]
enum MemPolicy {
    Random,
    Local, // access only keys local to socket
}

#[derive(Clone,Copy,Debug)]
struct Config {
    size: usize,
    capacity: usize,
    cpu: CPUPolicy,
    mem: MemPolicy,
}

fn run(config: &Config) {
    logger::enable();

    info!("config: {:?}", config);

    info!("creating Nibble...");
    let mut nib = match config.capacity {
        0 => Nibble::default(),
        _ => Nibble::new(config.capacity),
    };

    info!("inserting objects...");
    let mut fill: usize = 
            ((nib.capacity() as f64) * 0.8) as usize;
    let nobj: usize = fill/(config.size+8+8); // + header + key
    fill = nobj*(config.size+8+8);
    let pernode: usize = nobj/nib.nnodes();
    info!("cap {:.3}gb fill {:.3}gb nobj {} nobj.pernode {}",
          (nib.capacity() as f64)/((1usize<<30) as f64),
          (fill as f64)/((1usize<<30) as f64),
          nobj, pernode);
    let mut node = 0;

    // downgrade to immutable shareable alias
    let nib = Arc::new(nib);

    info!("inserting all keys -----------------");
    {
        let now = Instant::now();
        let nsock = nib.nnodes();
        let mut handles: Vec<JoinHandle<()>> = Vec::with_capacity(nsock);
        let size = config.size;
        for sock in 0..nsock {
            let start_key: u64 = (sock*pernode) as u64;
            let end_key: u64   = start_key + (pernode as u64);
            let arc = nib.clone();
            handles.push( thread::spawn( move || {
                let value = memory::allocate::<u8>(size);
                let v = Some(value as *const u8);
                info!("range [{},{}) on socket {}",
                    start_key, end_key, sock);
                for key in start_key..end_key {
                    let obj = ObjDesc::new(key, v, size as u32);
                    if let Err(code) = arc.put_where(&obj,
                                        PutPolicy::Specific(sock)) {
                        panic!("{:?}", code)
                    }
                }
                unsafe { memory::deallocate(value, size); }
            }));
        }
        for handle in handles {
            let _ = handle.join();
        }

        let sec = now.elapsed().as_secs();
        info!("insertion took {} seconds", sec);
    }

    info!("starting scale test -----------------------");

    // used to block threads from starting, then again as an
    // accumulator for total ops performed among all threads
    let accum = Arc::new(AtomicUsize::new(0));

    let mut threadcount: Vec<usize>;
    // power of 2   1, 2, 4, 8, 16, 32, 64, 128, 256
    threadcount = (0usize..9).map(|e|1usize<<e).collect();
    // incr of 4    1, 4, 8, 12, 16, ...
    //threadcount = (0usize..65).map(|e|if e==0 {1} else {4*e}).collect();
    // incr of 2    1, 2, 4, 6, 8, ...
    //threadcount = (0usize..130).map(|e|if e==0 {1} else {2*e}).collect();
    // incr of 1    1, 2, 3, 4, 5, ...
    //threadcount = (1usize..261).collect();

    println!("# tid ntid kops");

    // Run the experiment multiple times using different sets of
    // threads
    for nthreads in threadcount {

        let mut handles: Vec<JoinHandle<()>> = Vec::with_capacity(nthreads);
        info!("running with {} threads ---------", nthreads);
        accum.store(nthreads, Ordering::Relaxed);

        // Create CPU binding strategy
        let cpus_pernode = numa::NODE_MAP.cpus_in(NodeId(0));
        let nsockets = numa::NODE_MAP.sockets();
        let ncpus = cpus_pernode * nsockets;
        let mut cpus: VecDeque<usize> = VecDeque::with_capacity(ncpus);
        match config.cpu {
            CPUPolicy::Random => {
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
                    for sock in 0..nsockets {
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

        // Spawn each of the threads in this set
        for t in 0..nthreads {
            let accum = accum.clone();
            let nib = nib.clone();
            let cap = nib.capacity();
            let size = config.size;
            let cpu = cpus.pop_front().unwrap();
            let memint = config.mem;

            handles.push( thread::spawn( move || {
                accum.fetch_sub(1, Ordering::Relaxed);
                unsafe { pin_cpu(cpu); }

                let nsockets = numa::NODE_MAP.sockets();
                let sock = numa::NODE_MAP.sock_of(cpu);
                info!("thread {} on cpu {} sock {:?}", t, cpu, sock);

                // wait for all other threads to spawn
                // after this, accum is zero
                while accum.load(Ordering::Relaxed) > 0 { ; }

                // main loop (do warmup first)
                let mut n: usize = 7877 * t; // offset all threads
                for x in 0..2 {
                //loop {
                    let mut ops = 0usize;
                    let now = Instant::now();
                    while now.elapsed().as_secs() < 15 {
                        let offset = unsafe { rdrand() as usize };
                        match memint {
                            s @ MemPolicy::Local => {
                                for _ in 0..1000usize {
                                    let key = sock.0*pernode + (n % pernode);
                                    let _ = nib.get_object(key as u64);
                                    n += offset; // skip some
                                    ops += 1;
                                }
                            },
                            s @ MemPolicy::Random => {
                                for _ in 0..1000usize {
                                    let key = ((n % nsockets)*pernode)+(n % pernode);
                                    let _ = nib.get_object(key as u64);
                                    n += offset;
                                    ops += 1;
                                }
                            },
                        }
                    }
                    if x == 1 {
                        let dur = now.elapsed();
                        let nsec = dur.as_secs() * 1000000000u64
                            + dur.subsec_nanos() as u64;
                        let kops = ((ops as f64)/1e3)/((nsec as f64)/1e9);
                        println!("# {} {} {:.3}", t, nthreads, kops);
                        // aggregate the performance
                        accum.fetch_add(kops as usize, Ordering::Relaxed);
                    }
                }
            }));
        }
        for handle in handles {
            let _ = handle.join();
        }
        println!("# total kops {}",
                 accum.load(Ordering::Relaxed));
        cuckoo::print_conflicts(0usize);
    }
}

fn main() {
    let matches = App::new("scale")
        // size of object
        .arg(Arg::with_name("size")
             .long("size")
             .takes_value(true))
        .arg(Arg::with_name("capacity")
             .long("capacity")
             .takes_value(true))
        .arg(Arg::with_name("mem")
             .long("mem")
             .takes_value(true))
        .arg(Arg::with_name("cpu")
             .long("cpu")
             .takes_value(true))
        .get_matches();

    let mem = match matches.value_of("mem") {
        None => panic!("Specify memory policy"),
        Some(s) => {
            if s == "random" {
                MemPolicy::Random
            } else if s == "local" {
                MemPolicy::Local
            } else {
                panic!("invalid memory policy");
            }
        },
    };
    let cpu = match matches.value_of("cpu") {
        None => panic!("Specify CPU policy"),
        Some(s) => {
            if s == "random" {
                CPUPolicy::Random
            } else if s == "rr" {
                CPUPolicy::SocketRR
            } else if s == "incr" {
                CPUPolicy::Incremental
            } else {
                panic!("invalid CPU policy");
            }
        },
    };
    let capacity = match matches.value_of("capacity") {
        None => 0,
        Some(s) => match usize::from_str_radix(s, 10) {
            Ok(s) => s,
            Err(_) => panic!("Object size NAN"),
        },
    };
    let size = match matches.value_of("size") {
        None => panic!("Specify object size"),
        Some(s) => match usize::from_str_radix(s, 10) {
            Ok(s) => s,
            Err(_) => panic!("Object size NAN"),
        },
    };

    let mut config = Config {
        size: size,
        capacity: capacity,
        cpu: cpu,
        mem: mem,
    };

    run(&config);
}
