use std::fs::File;
use std::io::{self, BufReader};
use std::io::prelude::*;
use std::collections::VecDeque;
use std::intrinsics;
use std::str::FromStr;

#[derive(Debug)]
pub enum Op {
    Get, Set, Del
}

#[derive(Debug)]
pub struct Entry {
    pub key: u64,
    pub size: u32,
    pub op: Op,
}

impl Entry {

    pub fn new(key: u64, op: Op, size: u32) -> Self {
        Entry { key: key, op: op, size: size }
    }
}

/// Object used to read in a PUT/GET/DEL trace from a file.
pub struct Trace {
    /// number of entries stored in the trace
    pub n: u32,
    /// if operation is Get or Set, value size is irrelevant
    pub rec: Vec<Entry>
}

impl Trace {

    pub fn new(path: &str) -> Self {
        let mut t = Trace { n: 0_u32, rec: Vec::with_capacity(1usize<<36) };
        if let Err(msg) = t.read_trace(path) {
            panic!("Error reading or parsing trace file {}: {}",
                path, msg);
        }
        t.n = t.rec.len() as u32;
        t
    }

    pub fn new2(path: &String) -> Self {
        Trace::new(path.as_str())
    }

    pub fn print(&self) {
        for e in &self.rec {
            println!("{:?}", e);
        }
    }

    /// Read trace from given path.
    /// We assume trace has three space-separated columns:
    ///         key op value_size
    /// where
    ///      key:    [0-9]+          unsigned long
    ///       op:    (get|del|set)   string
    /// val size:    (na|[0-9]+)     'na' or unsigned long
    ///
    /// Lines starting exactly with '#' will be skipped.
    fn read_trace(&mut self, path: &str) -> Result<(),&str> {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) => return Err( "Cannot open file" ),
        };
        let file = BufReader::new(file);
        for line in file.lines() {
            if line.is_err() { break; }
            let line = line.unwrap();
            if line.starts_with("#") { continue; }
            let mut iter = line.split_whitespace();
            let key = match iter.next() {
                None => return Err( "Line has no key" ),
                Some(k) => match u64::from_str_radix(k, 10) {
                    // +1 to ensure no key is zero
                    Ok(v) => v + 1,
                    Err(e) => return Err( "Key is non-numeric" ),
                },
            };

            assert!(key > 0, "keys cannot be zero");

            let op = match iter.next() {
                None => return Err( "Line missing 2nd column" ),
                Some(o) => match o {
                    "get" => Op::Get,
                    "set" => Op::Set,
                    "del" => Op::Del,
                    _ => return Err( "Unknown operation" ),
                },
            };
            let size = match iter.next() {
                None => return Err( "Line missing 3rd column" ),
                Some(s) => match s {
                    "na" => 0u32,
                    _ => match f64::from_str(s) {
                        Ok(v) => v as u32,
                        Err(e) => return Err( "Size is non-numeric" ),
                    },
                },
            };
            if size > 0 {
                self.rec.push( Entry::new(key,op,size) );
            }
        }
        Ok( () )
    }
}
