#![feature(test)]

#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unused_assignments)]
#![allow(unused_imports)]
#![allow(unused_imports)]
#![allow(unused_mut)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_variables)]

#[macro_use]
extern crate log;

extern crate libc;
extern crate rand;
extern crate test;
extern crate time;

pub mod nibble;
pub use nibble::*;
