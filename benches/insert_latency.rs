/*
 * Test simple put latencies, local, remote, varying object lengths.
 *
 * This should run with only one thread (cargo bench should behave
 * this way by default).
 */

#![feature(test)]

/// Specify amount of memory in GiB Nibble should allocate for the
/// logs.  The index will acquire more memory for the hash tables.
const NIBBLE_CAPACITY: usize = 30;

extern crate test;
extern crate nibble;
extern crate rand;

#[macro_use]
extern crate lazy_static;

use rand::Rng;
use test::Bencher;

use std::mem;
use std::sync::{Once, ONCE_INIT};

use nibble::common::Pointer;
use nibble::nib::{Nibble,PutPolicy};
use nibble::segment::ObjDesc;
use nibble::sched::pin_cpu;
use nibble::logger;
use nibble::numa::{self,NodeId};

// TODO Put unique objects (will not perform atomic decrement of old
// segment).

// TODO test objects larger than block, and segment
// TODO put_object which must traverse chunks
// TODO a get_object which must traverse chunks
// TODO test we can determine live vs dead entries in segment
// TODO test specific cases where header cross block boundaries

type KeyType = u64;

// Initializing this takes a while, so we do it once and reuse across
// bench iterations. Each outer function is called many times, as well
// as each invocation of Bencher::iter.
lazy_static! {
    pub static ref NIBBLE: Nibble =
        Nibble::new(NIBBLE_CAPACITY<<30);
}

static START: Once = ONCE_INIT;

// If compaction must be on, uncomment below.
fn setup() {
    START.call_once( || {
        logger::enable();
        unsafe { pin_cpu(0); }
        //let nsockets = numa::NODE_MAP.sockets();
        //for s in 0..nsockets {
        //    NIBBLE.enable_compaction(NodeId(s));
        //}
    });
}

fn insert_n(b: &mut Bencher, p: PutPolicy,
            nib: &Nibble, len: usize) {
    let mut rng = rand::thread_rng();
    let key = rng.gen::<u64>();
    let val: Vec<u8> = Vec::with_capacity(len);
    let obj = ObjDesc::new(key, Pointer(val.as_ptr()), len);
    b.iter( || { nib.put_where(&obj,p) });
}

fn local_insert_n(b: &mut Bencher, nib: &Nibble, len: usize) {
    let policy = PutPolicy::Specific(0);
    insert_n(b, policy, nib, len);
}

fn remote_insert_n(b: &mut Bencher, nib: &Nibble, len: usize) {
    let node = numa::NODE_MAP.sockets() - 1;
    let policy = PutPolicy::Specific(node);
    insert_n(b, policy, nib, len);
}

// ---- Local tests ----

#[bench]
fn local_insert_16(b: &mut Bencher) {
    setup();
    let len = 16_usize - mem::size_of::<KeyType>();
    local_insert_n(b, &NIBBLE, len);
}

#[bench]
fn local_insert_30(b: &mut Bencher) {
    setup();
    let len = 30_usize - mem::size_of::<KeyType>();
    local_insert_n(b, &NIBBLE, len);
}

#[bench]
fn local_insert_60(b: &mut Bencher) {
    setup();
    let len = 60_usize - mem::size_of::<KeyType>();
    local_insert_n(b, &NIBBLE, len);
}

#[bench]
fn local_insert_100(b: &mut Bencher) {
    setup();
    let len = 100_usize - mem::size_of::<KeyType>();
    local_insert_n(b, &NIBBLE, len);
}

#[bench]
fn local_insert_200(b: &mut Bencher) {
    setup();
    let len = 200_usize - mem::size_of::<KeyType>();
    local_insert_n(b, &NIBBLE, len);
}

#[bench]
fn local_insert_500(b: &mut Bencher) {
    setup();
    let len = 500_usize - mem::size_of::<KeyType>();
    local_insert_n(b, &NIBBLE, len);
}

#[bench]
fn local_insert_1000(b: &mut Bencher) {
    setup();
    let len = 1000_usize - mem::size_of::<KeyType>();
    local_insert_n(b, &NIBBLE, len);
}

#[bench]
fn local_insert_2000(b: &mut Bencher) {
    setup();
    let len = 3000_usize - mem::size_of::<KeyType>();
    local_insert_n(b, &NIBBLE, len);
}

#[bench]
fn local_insert_4000(b: &mut Bencher) {
    setup();
    let len = 4000_usize - mem::size_of::<KeyType>();
    local_insert_n(b, &NIBBLE, len);
}

// ---- Distant tests ----

#[bench]
fn remote_insert_16(b: &mut Bencher) {
    setup();
    let len = 16_usize - mem::size_of::<KeyType>();
    remote_insert_n(b, &NIBBLE, len);
}

#[bench]
fn remote_insert_30(b: &mut Bencher) {
    setup();
    let len = 30_usize - mem::size_of::<KeyType>();
    remote_insert_n(b, &NIBBLE, len);
}

#[bench]
fn remote_insert_60(b: &mut Bencher) {
    setup();
    let len = 60_usize - mem::size_of::<KeyType>();
    remote_insert_n(b, &NIBBLE, len);
}

#[bench]
fn remote_insert_100(b: &mut Bencher) {
    setup();
    let len = 100_usize - mem::size_of::<KeyType>();
    remote_insert_n(b, &NIBBLE, len);
}

#[bench]
fn remote_insert_200(b: &mut Bencher) {
    setup();
    let len = 200_usize - mem::size_of::<KeyType>();
    remote_insert_n(b, &NIBBLE, len);
}

#[bench]
fn remote_insert_500(b: &mut Bencher) {
    setup();
    let len = 500_usize - mem::size_of::<KeyType>();
    remote_insert_n(b, &NIBBLE, len);
}

#[bench]
fn remote_insert_1000(b: &mut Bencher) {
    setup();
    let len = 1000_usize - mem::size_of::<KeyType>();
    remote_insert_n(b, &NIBBLE, len);
}

#[bench]
fn remote_insert_2000(b: &mut Bencher) {
    setup();
    let len = 3000_usize - mem::size_of::<KeyType>();
    remote_insert_n(b, &NIBBLE, len);
}

#[bench]
fn remote_insert_4000(b: &mut Bencher) {
    setup();
    let len = 4000_usize - mem::size_of::<KeyType>();
    remote_insert_n(b, &NIBBLE, len);
}

