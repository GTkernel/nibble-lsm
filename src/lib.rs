#![feature(test)]

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unused_assignments)]
#![allow(unused_imports)]

#[macro_use]
extern crate log;

extern crate libc;
extern crate rand;
extern crate test;
extern crate time;
extern crate crossbeam;

pub mod nibble;
pub use nibble::*;
