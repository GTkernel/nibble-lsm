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
use rand::Rng;
use std::mem;
use std::sync::Arc;
use std::sync::atomic::*;
use std::thread::{self,JoinHandle};
use std::collections::VecDeque;
use std::time::Duration;
use std::time::Instant;

struct Config {
    size: usize,
    capacity: usize,
    cpu_interleave: bool,
    mem_interleave: PutPolicy,
}

fn run(config: &Config) {
    logger::enable();

    info!("creating Nibble...");
    let mut nib = match config.capacity {
        0 => Nibble::default(),
        _ => Nibble::new(config.capacity),
    };

    info!("inserting objects...");
    let mut keycount: usize = 0;

    let value = memory::allocate::<u8>(config.size);
    let v = Some(value as *const u8);

    let fill: usize = 
            ((nib.capacity() as f64) * 0.8) as usize;
//    let fill: usize = match policy {
//        PutPolicy::Interleave =>
//            ((nib.capacity() as f64) * 0.9) as usize,
//        PutPolicy::Specific(_) => (nib.capacity()/nib.nnodes())/2,
//    };
    info!("capacity: {:.2} fill: {:.2}",
             (nib.capacity() as f64) / ((1usize<<30)as f64),
             (fill as f64) / ((1usize<<30)as f64));

    let nobj: usize = fill/config.size;
    let pernode: usize = nobj/nib.nnodes();
    info!("nobj {} pernode {}", nobj, pernode);
    let mut node = 0;
    info!("switching to node {} key {}", node, keycount);
    for c in 0..nobj {
        if node >= nib.nnodes() {
            info!("node > {} (skipping {} objects)",
                nib.nnodes(), nobj-c);
            break;
        }
        let key = keycount.to_string();
        let obj = ObjDesc::new(key.as_str(), v, config.size as u32);
        if let Err(code) = nib.put_where(&obj, PutPolicy::Specific(node)) {
            panic!("{:?}", code)
        }
        keycount += 1;
        if keycount % pernode == 0 {
            node += 1;
            info!("switching to node {} key {}", node, keycount);
        }
    }

    info!("starting scale test -----------------------");

    // downgrade to immutable shareable alias
    let nib = Arc::new(nib);

    // used to block threads from starting, then again as an
    // accumulator for total ops performed among all threads
    let accum = Arc::new(AtomicUsize::new(0));

    let mut threadcount: Vec<usize>;
    threadcount = (0usize..65).map(|e|if e==0 {1} else {4*e}).collect();

    for nthreads in threadcount {
        let mut handles: Vec<JoinHandle<()>> = Vec::with_capacity(nthreads);
        info!("running with {} threads ---------", nthreads);
        accum.store(nthreads, Ordering::Relaxed);

        // round-robin pinning
        let cpus_pernode = numa::NODE_MAP.cpus_in(NodeId(0));
        let nsockets = numa::NODE_MAP.sockets();
        let ncpus = cpus_pernode * nsockets;
        let mut cpus: VecDeque<usize> = VecDeque::with_capacity(ncpus);
        for i in 0..cpus_pernode {
            for sock in 0..nsockets {
                let r = numa::NODE_MAP.cpus_of(NodeId(sock));
                cpus.push_back(r.start + i);
            }
        }

//        // create cpu IDs threads will bind to
//        let mut cpus: VecDeque<usize> = VecDeque::new();
//        // linear
//        for sock in 0..numa::NODE_MAP.sockets() {
//            let r = numa::NODE_MAP.cpus_of(NodeId(sock));
//            for cpu in r.start..(r.end+1) {
//                cpus.push_back(cpu);
//            }
//        }
//        // Knuth shuffle
//        if cpuinterleave {
//            let mut rng = rand::thread_rng();
//            let max = cpus.len();
//            for i in 0..max {
//                let o = (rng.next_u32() as usize % (max-i)) + i;
//                cpus.swap(i, o);
//            }
//        }

        let now = Instant::now();
        for t in 0..nthreads {
            let accum = accum.clone();
            let nib = nib.clone();
            let cpu = cpus.pop_front().unwrap();
            handles.push( thread::spawn( move || {
                accum.fetch_sub(1, Ordering::Relaxed);
                unsafe { pin_cpu(cpu); }
                let sock = numa::NODE_MAP.sock_of(cpu);
                info!("thread {} cpu {} sock {:?}", t, cpu, sock);
                // pre-build the key strings, and private to each thread
                //let mut keys: Vec<String> = Vec::with_capacity(keycount);
                //for k in 0..keycount {
                    //keys.push( k.to_string() );
                //}
                let mut ops = 0usize;
                while accum.load(Ordering::Relaxed) > 0 { ; }
                info!("thread {} key range {}-{}",
                         t, sock.0*pernode, sock.0*pernode+pernode);
                let now = Instant::now();
                // XXX modify these loops to debug scalability XXX
                while now.elapsed().as_secs() < 10 {
                //loop {
                    //for _ in 0..100usize {
                    let r = unsafe { rdrand() } as usize;
                    let idx = (r % pernode) + (sock.0 * pernode);
                    //let key = &keys[idx];
                    let key = idx.to_string();
                    let _ = nib.get_object(&key);
                    ops += 1;
                    //}
                }
                accum.fetch_add(ops, Ordering::Relaxed);
            }));
        }
        for handle in handles {
            let _ = handle.join();
        }
        let dur = now.elapsed();

        let ops = accum.load(Ordering::Relaxed) as f64;
        let nsec = dur.as_secs() * 1000000000u64
            + dur.subsec_nanos() as u64;
        let mut fmt = String::new();
        println!("nobj.mil capacity.gb fill.gb size.b threads kops");
        println!("{:.3} {:.3} {:.3} {} {} {:.3}",
                 (keycount as f64) / 1e6,
                 (nib.capacity() as f64) / ((1usize<<30) as f64),
                 (fill as f64) / ((1usize<<30) as f64),
                 config.size,
                 nthreads, (ops/1e3)/((nsec as f64)/1e9));
    }
    unsafe { memory::deallocate(value, config.size); }
}

fn main() {
    let matches = App::new("scale")
        // size of object
        .arg(Arg::with_name("size")
             .long("size")
             .takes_value(true))
        .arg(Arg::with_name("threads")
             .long("threads")
             .takes_value(true))
        .arg(Arg::with_name("capacity")
             .long("capacity")
             .takes_value(true))
        .arg(Arg::with_name("mem_interleave")
             .long("mem_interleave"))
        .arg(Arg::with_name("cpu_interleave")
             .long("cpu_interleave"))
        .get_matches();

    let policy = match matches.is_present("mem_interleave") {
        true => PutPolicy::Interleave,
        false => PutPolicy::Specific(0),
    };
    let cpuinterleave = matches.is_present("cpu_interleave");
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
        cpu_interleave: cpuinterleave,
        mem_interleave: policy
    };

    run(&config);
}
