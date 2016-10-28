#![feature(test)]
#![feature(const_fn)]
#![feature(asm)]
#![feature(repr_simd)]

// used for nibble/tlock.rs
#![feature(core_intrinsics)]

// Clippy tool
//#![feature(plugin)]
//#![plugin(clippy)]

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unused_mut)]

// Used for likely/unlikely
#![feature(core_intrinsics)]

// Remove these XXX
#![allow(dead_code)]
#![allow(unreachable_code)]

#![forbid(const_err)]

#[macro_use]
extern crate log;

#[macro_use]
extern crate lazy_static;

extern crate libc;
extern crate rand;
extern crate test;
extern crate time;
extern crate crossbeam;
extern crate itertools;
extern crate quicksort;
extern crate syscall;
extern crate parking_lot;
extern crate num;

pub mod nibble;
pub use nibble::*;
