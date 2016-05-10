#![feature(test)]
#![feature(const_fn)]
#![feature(asm)]

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

#[macro_use]
extern crate log;

extern crate libc;
extern crate rand;
extern crate test;
extern crate time;
extern crate crossbeam;
extern crate itertools;

pub mod nibble;
pub use nibble::*;
