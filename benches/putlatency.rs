/*
 * Test simple put latencies, local, remote, varying object lengths.
 */
#![feature(test)]

extern crate test;
extern crate nibble;
extern crate rand;

use rand::Rng;
use std::mem;
use test::Bencher;

use nibble::common::Pointer;
use nibble::nib::Nibble;
use nibble::segment::ObjDesc;
use nibble::sched::pin_cpu;
use nibble::logger;

type KeyType = u64;

// TODO test objects larger than block, and segment
// TODO put_object which must traverse chunks
// TODO a get_object which must traverse chunks
// TODO test we can determine live vs dead entries in segment
// TODO test specific cases where header cross block boundaries

fn insert_n(b: &mut Bencher, nib: &Nibble, len: usize) {
    let mut rng = rand::thread_rng();
    let key = rng.gen::<u64>();
    let val: Vec<u8> = Vec::with_capacity(len);
    let obj = ObjDesc::new(key, Pointer(val.as_ptr()), len);
    b.iter( || { nib.put_object(&obj) });
}

#[bench]
fn insert_30(b: &mut Bencher) {
    logger::enable();

    unsafe { pin_cpu(0); }
    let nib = Nibble::default();
    let len: usize = 30-mem::size_of::<KeyType>();
    insert_n(b, &nib, len);
}

