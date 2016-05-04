#![feature(test)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]

extern crate test;
extern crate nibble;

use nibble::store::Nibble;
use nibble::segment::ObjDesc;

// TODO test objects larger than block, and segment
// TODO put_object which must traverse chunks
// TODO a get_object which must traverse chunks
// TODO test we can determine live vs dead entries in segment
// TODO test specific cases where header cross block boundaries

#[test]
fn alloc_90_free_all() {
    // Create log of fixed size
    // Determine how many objects to insert to achieve desired fill
    // Remove them
    let mut nib = Nibble::new(1<<30);
}
