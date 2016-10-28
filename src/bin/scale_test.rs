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
use nibble::common::{Pointer,ErrorCode,rdrand};
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
    nobject: usize,
    cpu: CPUPolicy,
    mem: MemPolicy,
}

#[cfg(IGNORE)]
fn run(config: &Config) {
    logger::enable();

    info!("config: {:?}", config);

    let mut fill: usize;
    let nobj: usize = if config.nobject == 0 {
        fill = ((config.capacity as f64) * 0.8) as usize;
        fill/(config.size+8+8) // + header + key
    } else {
        fill = config.nobject * (config.size+8+8);
        config.nobject
    };
    info!("creating Nibble...");
    let mut nib = match config.capacity {
        0 => {
            if fill > Nibble::default_capacity() {
                panic!("nobjects too many for capacity");
            }
            Nibble::default()
        },
        _ => {
            if fill > config.capacity {
                panic!("nobjects too many for capacity");
            }
            Nibble::new(config.capacity)
        },
    };

    let primes: Arc<Vec<usize>> = Arc::new(vec![
        1381307, 6019001, 11193733, 14861299, 6361351, 15351997,
        2708891, 2116481, 9440021, 6157033, 8387, 4796677, 1276897,
        4200143, 7220249, 2988497, 14146999, 5322179, 14291581,
        1857197, 10220563, 13330529, 1592111, 2597939, 15020969,
        14975717, 5863679, 1614947, 9474713, 10742443, 6644069,
        4353431, 2395333, 11995661, 4167931, 12396053, 9801271,
        3887083, 5121581, 4171523, 14333611, 11998381, 9099899,
        8742317, 12136643, 7334081, 1137167, 5674547, 2865413,
        6476513, 11416421, 706033, 9013993, 12062621, 10737593,
        9398957, 11984653, 10067789, 6356041, 5073643, 8615543,
        8297459, 5118031, 12785459, 9249701, 6056731, 11694251,
        7076581, 2538353, 7597157, 15386681, 1316407, 1451759,
        12761011, 903607, 10882471, 8433619, 10823921, 1074751,
        9478793, 13141259, 10069427, 2977441, 3845579, 12657457,
        12164179, 577153, 5605477, 14701097, 3476303, 6433591,
        1887511, 495877, 8908621, 9220279, 6634447, 2438773, 5266273,
        6404117, 5436881, 2574839, 12587629, 1555231, 13881713,
        2169841, 6425707, 1236623, 6322259, 14010851, 11166613,
        2645399, 10244369, 1052237, 4396361, 4498099, 2595157,
        5999129, 5933869, 1091047, 4149023, 11117611, 11976901,
        5706919, 15093643, 9159289, 10301803, 11645737, 9598019]);

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
                let v = Pointer(value as *const u8);
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
    //threadcount = (0usize..9).map(|e|1usize<<e).collect();
    // incr of 4    1, 4, 8, 12, 16, ...
    //threadcount = (0usize..65).map(|e|if e==0 {1} else {4*e}).collect();
    // incr of 2    1, 2, 4, 6, 8, ...
    threadcount = (0usize..130).map(|e|if e==0 {1} else {2*e}).collect();
    // incr of 1    1, 2, 3, 4, 5, ...
    //threadcount = (1usize..261).collect();
    //threadcount = vec![6];

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
            let primes = primes.clone();

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
                let mut n: usize = 7877 * (t+1); // offset all threads
                for x in 0..2 {
                //let x = 1;
                //loop {
                    let mut ops = 0usize;
                    let now = Instant::now();
                    while now.elapsed().as_secs() < 5 {
                        // don't want rdrand or other generator on the
                        // critical path, so we choose random prime as
                        // offset, and read offsets in multiples of
                        // that
                        let r = unsafe { rdrand() as usize };
                        let offset = primes[r%primes.len()];
                        match memint {
                            s @ MemPolicy::Local => {
                                for _ in 0..1000usize {
                                    let key = sock.0*pernode + (n % pernode);
                                    let _ = nib.get_object(key as u64);
                                    n = n.wrapping_mul(offset); // skip some
                                    ops += 1;
                                }
                            },
                            s @ MemPolicy::Random => {
                                for _ in 0..1000usize {
                                    let key = n % nobj;
                                    //let key = ((n % nsockets)*pernode)+(n % pernode);
                                    let _ = nib.get_object(key as u64);
                                    n = n.wrapping_mul(offset); // skip some
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
        .arg(Arg::with_name("nobject")
             .long("nobject")
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
    let nobject = match matches.value_of("nobject") {
        None => 0,
        Some(s) => match usize::from_str_radix(s, 10) {
            Ok(s) => s,
            Err(_) => panic!("no. objects is NAN"),
        },
    };
    let size = match matches.value_of("size") {
        None => panic!("Specify object size"),
        Some(s) => match usize::from_str_radix(s, 10) {
            Ok(s) => s,
            Err(_) => panic!("Object size NAN"),
        },
    };

    // if nobject == 0, then we compute it based on filling 80% of
    // the capacity

    let mut config = Config {
        size: size,
        capacity: capacity,
        nobject: nobject,
        cpu: cpu,
        mem: mem,
    };

    //run(&config);
}
